"""源码包条件发布的离线测试，模拟 artifact 传递和 HTTP PUT，不连接外部服务。"""

from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import quote

import pytest
import yaml


CI_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTORY_URL = "https://download.t-head.cn/artifactory/pypi_generic/"
SDIST_NAME = "demo_package-1.2.3+oe.tar.gz"

SHELL_SHIM = r'''
python3() {
  "$TEST_PYTHON" "$@"
}
curl() {
  "$TEST_PYTHON" -c '
import json
import os
import sys

with open(os.environ["TEST_CURL_CALLS"], "a", encoding="utf-8") as output:
    output.write(json.dumps(sys.argv[1:]) + "\n")
print(os.environ["TEST_HTTP_STATUS"], end="")
sys.exit(int(os.environ["TEST_CURL_EXIT"]))
' "$@"
}
'''


@pytest.fixture
def workflow():
    return yaml.load((CI_ROOT / "workflows/matrix-release.yml").read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)


@pytest.fixture
def publish_job(workflow):
    return workflow["jobs"]["publish-sdist"]


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory(prefix=".publish-sdist-test-", dir=CI_ROOT) as directory:
        yield Path(directory)


@pytest.fixture
def run_step(workspace):
    def run(step, **overrides):
        env = dict(os.environ, TEST_PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1",
                   TEST_CURL_CALLS=str(workspace / "curl-calls.jsonl"), TEST_HTTP_STATUS="201",
                   TEST_CURL_EXIT="0", GITHUB_OUTPUT=str(workspace / "github-output"),
                   ARTIFACTORY_URL=ARTIFACTORY_URL, PACKAGE_NAME="demo_package",
                   RELEASE_USERNAME="test-release-user", RELEASE_TOKEN="test-release-token")
        env.update(overrides)
        return subprocess.run(["bash"], input=SHELL_SHIM + step["run"], cwd=workspace,
                              env=env, text=True, capture_output=True, timeout=30)

    return run


@pytest.fixture
def run_upload(publish_job, workspace, run_step):
    def run(filenames=(SDIST_NAME,), **overrides):
        directory = workspace / "sdist-dist"
        directory.mkdir(exist_ok=True)
        for filename in filenames:
            (directory / filename).write_bytes(b"source archive test data")
        result = run_step(publish_job["steps"][-1], **overrides)
        calls_file = workspace / "curl-calls.jsonl"
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
        return result, calls

    return run


def render_artifact_expression(expression, index="0", run="100", attempt="2"):
    values = {"inputs.package_name": "demo_package", "github.run_id": run,
              "github.run_attempt": attempt, "strategy.job-index": index}
    for key, value in values.items():
        expression = expression.replace("${{ " + key + " }}", value)
    return expression


def test_publish_waits_for_all_matrix_jobs_and_handles_aggregate_failure(publish_job):
    assert set(publish_job["needs"]) == {"build-sdist", "matrix-job"}
    assert publish_job["if"] == (
        "${{ !cancelled() && needs.build-sdist.result == 'success' && "
        "(needs.matrix-job.result == 'success' || needs.matrix-job.result == 'failure') }}"
    )
    assert "strategy" not in publish_job
    assert "continue-on-error" not in publish_job


def test_markers_only_follow_successful_wheel_upload(workflow, workspace, run_step):
    steps = workflow["jobs"]["matrix-job"]["steps"]
    upload = next(step for step in steps if step.get("name") == "上传 wheel 到 Artifactory")
    create, save = steps[-2:]
    assert steps.index(upload) < steps.index(create)
    assert create["name"] == "记录矩阵任务成功"
    assert save["uses"] == "actions/upload-artifact@v4"
    for step in (create, save):
        assert "if" not in step
        assert "continue-on-error" not in step
    result = run_step(create, RUNNER_TEMP=str(workspace))
    assert result.returncode == 0, result.stderr
    assert (workspace / "matrix-success.txt").read_text() == "success\n"
    assert save["with"]["path"] == "${{ runner.temp }}/matrix-success.txt"
    assert save["with"]["if-no-files-found"] == "error"
    assert "outputs" not in workflow["jobs"]["matrix-job"]
    assert workflow["jobs"]["matrix-job"]["strategy"]["fail-fast"] == "false"


def test_marker_names_are_unique_and_run_attempt_scoped(workflow, publish_job):
    name = workflow["jobs"]["matrix-job"]["steps"][-1]["with"]["name"]
    download = publish_job["steps"][0]
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"]["merge-multiple"] == "false"
    pattern = render_artifact_expression(download["with"]["pattern"])
    names = [render_artifact_expression(name, index=str(i)) for i in range(144)]
    assert len(set(names)) == 144
    assert all(fnmatchcase(value, pattern) for value in names)
    assert not fnmatchcase(render_artifact_expression(name, attempt="1"), pattern)
    assert not fnmatchcase(render_artifact_expression(name, run="99"), pattern)
    assert not fnmatchcase(names[0].replace("demo_package", "other_package"), pattern)


def test_source_download_and_upload_use_gate_and_matching_artifact(workflow, publish_job):
    gate = publish_job["steps"][1]
    assert gate["id"] == "matrix-success"
    download, upload = publish_job["steps"][2:]
    condition = "${{ steps.matrix-success.outputs.any_success == 'true' }}"
    assert download["if"] == upload["if"] == condition
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"]["name"] == workflow["jobs"]["build-sdist"]["steps"][-1]["with"]["name"]
    assert download["with"]["path"] == "sdist-dist"
    assert "env" not in publish_job
    assert all("RELEASE_TOKEN" not in str(step) for step in publish_job["steps"][:-1])
    assert upload["env"] == {
        "ARTIFACTORY_URL": ARTIFACTORY_URL,
        "PACKAGE_NAME": "${{ inputs.package_name }}",
        "RELEASE_USERNAME": "${{ secrets.release_username }}",
        "RELEASE_TOKEN": "${{ secrets.release_token }}",
    }
    assert all(not step.get("uses", "").startswith("actions/checkout") for step in publish_job["steps"])


@pytest.mark.parametrize("outcomes,expected", [
    (["success", "failure"], True),
    (["failure", "success"], True),
    (["success", "success"], True),
    (["success"], True),
    (["failure", "failure"], False),
    (["failure", "skipped"], False),
    (["cancelled"], False),
    ([], False),
])
def test_any_success_gate_and_conditional_upload(workflow, publish_job, workspace, run_step,
                                               run_upload, outcomes, expected):
    save_name = workflow["jobs"]["matrix-job"]["steps"][-1]["with"]["name"]
    pattern = render_artifact_expression(publish_job["steps"][0]["with"]["pattern"])
    # 即使旧尝试有成功记录，本次全部失败也不能误触发发布。
    artifacts = [(render_artifact_expression(save_name, attempt="1"), "success\n")]
    for index, outcome in enumerate(outcomes):
        if outcome == "success":
            artifacts.append((render_artifact_expression(save_name, index=str(index)), "success\n"))
    for name, contents in artifacts:
        if fnmatchcase(name, pattern):
            directory = workspace / "matrix-results" / name
            directory.mkdir(parents=True)
            (directory / "matrix-success.txt").write_text(contents)
    result = run_step(publish_job["steps"][1])
    assert result.returncode == 0, result.stderr
    output = (workspace / "github-output").read_text()
    assert output == f"any_success={str(expected).lower()}\n"
    if output == "any_success=true\n":
        upload_result, calls = run_upload()
        assert upload_result.returncode == 0, upload_result.stderr
        assert len(calls) == 1
    else:
        assert "跳过源码包发布" in result.stdout
        assert not (workspace / "curl-calls.jsonl").exists()


@pytest.mark.parametrize("contents", ["", "failure\n", "false\n", "success-invalid\n"])
def test_invalid_marker_is_not_counted(publish_job, workspace, run_step, contents):
    directory = workspace / "matrix-results/marker"
    directory.mkdir(parents=True)
    (directory / "matrix-success.txt").write_text(contents)
    result = run_step(publish_job["steps"][1])
    assert result.returncode == 0
    assert (workspace / "github-output").read_text() == "any_success=false\n"


def test_publish_shell_syntax(publish_job):
    for step in publish_job["steps"]:
        if "run" in step:
            result = subprocess.run(["bash", "-n"], input=step["run"], text=True, capture_output=True)
            assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("status", ["200", "201", "204"])
def test_sdist_put_preserves_layout_and_uses_existing_credentials(run_upload, status):
    result, calls = run_upload(TEST_HTTP_STATUS=status)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    args = calls[0]
    assert args[-1] == ARTIFACTORY_URL + "demo_package/1.2.3%2Boe/" + quote(SDIST_NAME, safe="")
    assert args[args.index("--upload-file") + 1] == "sdist-dist/" + SDIST_NAME
    assert args[args.index("--user") + 1] == "test-release-user:test-release-token"
    assert args[args.index("--proto") + 1] == "=https"
    assert args[args.index("--retry") + 1] == "3"
    assert "--fail" in args
    assert "--location" not in args
    assert "--insecure" not in args
    assert "test-release-token" not in result.stdout + result.stderr
    assert "test-release-user" not in result.stdout + result.stderr


def test_prerelease_and_hyphenated_package_directory(run_upload):
    filename = "demo-package-2.0.0rc1.dev5+oe.tar.gz"
    result, calls = run_upload((filename,), PACKAGE_NAME="Demo-Package")
    assert result.returncode == 0, result.stderr
    assert calls[0][-1] == ARTIFACTORY_URL + "Demo-Package/2.0.0rc1.dev5%2Boe/" + quote(filename, safe="")


@pytest.mark.parametrize("filenames", [
    (),
    (SDIST_NAME, "demo_package-1.2.4+oe.tar.gz"),
    ("other_package-1.2.3+oe.tar.gz",),
    ("demo_package-1.2.3.tar.gz",),
    ("demo_package-1.2.3+other.tar.gz",),
    ("demo_package-1.2.3+oe\n.tar.gz",),
    ("demo_package-1.2.3%2Boe.tar.gz",),
])
def test_missing_or_invalid_sdist_never_uploads(run_upload, filenames):
    result, calls = run_upload(filenames)
    assert result.returncode != 0
    assert "::error::" in result.stderr
    assert calls == []


@pytest.mark.parametrize("overrides", [
    {"RELEASE_USERNAME": ""}, {"RELEASE_TOKEN": ""},
    {"PACKAGE_NAME": "../outside"}, {"PACKAGE_NAME": "other"},
])
def test_missing_credentials_and_wrong_package_block_upload(run_upload, overrides):
    result, calls = run_upload(**overrides)
    assert result.returncode != 0
    assert calls == []


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_non_regular_source_archive_is_rejected(run_upload, workspace, kind):
    directory = workspace / "sdist-dist"
    directory.mkdir()
    path = directory / SDIST_NAME
    if kind == "symlink":
        target = workspace / "unrelated"
        target.write_bytes(b"not an artifact")
        path.symlink_to(target)
    else:
        path.mkdir()
    result, calls = run_upload(())
    assert result.returncode != 0
    assert "源码包必须是普通文件" in result.stderr
    assert calls == []


@pytest.mark.parametrize("status", ["302", "307", "401", "403", "409", "500"])
def test_unsuccessful_http_status_fails_publication(run_upload, status):
    result, calls = run_upload(TEST_HTTP_STATUS=status)
    assert result.returncode != 0
    assert f"HTTP 状态码：{status}" in result.stderr
    assert len(calls) == 1
    assert "已上传源码包" not in result.stdout


@pytest.mark.parametrize("exit_code", ["7", "22", "28"])
def test_upload_failure_propagates(run_upload, exit_code):
    result, calls = run_upload(TEST_CURL_EXIT=exit_code, TEST_HTTP_STATUS="000")
    assert result.returncode == int(exit_code)
    assert len(calls) == 1
    assert "已上传源码包" not in result.stdout
