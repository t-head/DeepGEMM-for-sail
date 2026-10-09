"""wheel 上传 step 的离线测试：运行真实路径处理代码，模拟 curl，不发送网络请求。"""

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
WHEEL_NAME = "demo_package-1.2.3+dev5-cp310-cp310-linux_x86_64.whl"

CURL_SHIM = r'''
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
print(os.environ.get("TEST_HTTP_STATUS", "201"), end="")
sys.exit(int(os.environ.get("TEST_CURL_EXIT", "0")))
' "$@"
}
'''


@pytest.fixture
def workflow():
    return yaml.load((CI_ROOT / "workflows/matrix-release.yml").read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)


@pytest.fixture
def upload_step(workflow):
    return next(step for step in workflow["jobs"]["matrix-job"]["steps"]
                if step.get("name") == "上传 wheel 到 Artifactory")


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory(prefix=".upload-test-", dir=CI_ROOT) as directory:
        root = Path(directory)
        (root / "dist").mkdir()
        yield root


@pytest.fixture
def run_upload(upload_step, workspace):
    def run(filenames=(WHEEL_NAME,), **overrides):
        for filename in filenames:
            (workspace / "dist" / filename).write_bytes(b"test wheel data")
        calls_file = workspace / "curl-calls.jsonl"
        env = dict(os.environ, TEST_PYTHON=sys.executable, TEST_CURL_CALLS=str(calls_file),
                   TEST_HTTP_STATUS="201", TEST_CURL_EXIT="0", PYTHONDONTWRITEBYTECODE="1",
                   ARTIFACTORY_URL=ARTIFACTORY_URL, PACKAGE_NAME="demo_package",
                   RELEASE_USERNAME="test-release-user", RELEASE_TOKEN="test-release-token")
        env.update(overrides)
        result = subprocess.run(["bash"], input=CURL_SHIM + upload_step["run"], cwd=workspace,
                                env=env, text=True, capture_output=True, timeout=30)
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
        return result, calls

    return run


def test_upload_follows_build_and_credentials_are_step_scoped(workflow, upload_step):
    job = workflow["jobs"]["matrix-job"]
    assert job["steps"][job["steps"].index(upload_step) - 1]["name"] == "编译"
    assert upload_step["name"] == "上传 wheel 到 Artifactory"
    assert upload_step["shell"] == "bash"
    assert upload_step["if"] == "${{ inputs.publish }}"
    assert "continue-on-error" not in upload_step
    assert upload_step["env"] == {
        "ARTIFACTORY_URL": ARTIFACTORY_URL,
        "PACKAGE_NAME": "${{ inputs.package_name }}",
        "RELEASE_USERNAME": "${{ secrets.release_username }}",
        "RELEASE_TOKEN": "${{ secrets.release_token }}",
    }
    for name in ("RELEASE_USERNAME", "RELEASE_TOKEN"):
        assert name not in job.get("env", {})
        assert name not in workflow.get("env", {})
        assert all(name not in step.get("env", {}) for step in job["steps"] if step is not upload_step)


def test_upload_shell_syntax(upload_step):
    result = subprocess.run(["bash", "-n"], input=upload_step["run"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("status", ["200", "201", "204"])
def test_authenticated_upload_preserves_layout_and_encodes_local_version(run_upload, status):
    result, calls = run_upload(TEST_HTTP_STATUS=status)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    args = calls[0]
    assert args[-1] == ARTIFACTORY_URL + "demo_package/1.2.3%2Bdev5/" + quote(WHEEL_NAME, safe="")
    assert args[args.index("--upload-file") + 1] == "dist/" + WHEEL_NAME
    assert args[args.index("--user") + 1] == "test-release-user:test-release-token"
    assert args[args.index("--proto") + 1] == "=https"
    assert args[args.index("--retry") + 1] == "3"
    assert args[args.index("--output") + 1] == "/dev/null"
    assert "--fail" in args
    assert "--location" not in args
    assert "--location-trusted" not in args
    assert "--insecure" not in args
    assert "--verbose" not in args
    assert "test-release-user" not in result.stdout + result.stderr
    assert "test-release-token" not in result.stdout + result.stderr
    assert "已上传 " + WHEEL_NAME in result.stdout


@pytest.mark.parametrize("version", [
    "2.4.0", "2.4.1.dev3+gabcdef123", "2.4.1.dev3+gabcdef123.d20260101",
])
def test_scm_wheel_versions_preserve_upload_path(run_upload, version):
    filename = f"demo_package-{version}-cp310-cp310-linux_x86_64.whl"
    result, calls = run_upload((filename,))
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert calls[0][-1] == (
        ARTIFACTORY_URL + "demo_package/" + quote(version, safe="") + "/" + quote(filename, safe=""))


def test_multiple_wheels_and_optional_build_tag(run_upload):
    filenames = ("demo_package-1.2.3-2abc-py3-none-any.whl",
                 "demo_package-1.2.3-cp312-cp312-linux_x86_64.whl")
    result, calls = run_upload(filenames)
    assert result.returncode == 0, result.stderr
    assert {args[-1] for args in calls} == {
        ARTIFACTORY_URL + "demo_package/1.2.3/" + filename for filename in filenames
    }


def test_distribution_name_normalization_preserves_requested_directory(run_upload):
    result, calls = run_upload(PACKAGE_NAME="Demo-Package")
    assert result.returncode == 0, result.stderr
    assert calls[0][-1].startswith(ARTIFACTORY_URL + "Demo-Package/1.2.3%2Bdev5/")


def test_no_wheel_fails_without_uploading_other_artifacts(workspace, run_upload):
    (workspace / "dist/source.tar.gz").write_bytes(b"not a wheel")
    result, calls = run_upload(())
    assert result.returncode != 0
    assert "::error::dist 目录中未找到 wheel 文件" in result.stderr
    assert calls == []


@pytest.mark.parametrize("credential", ["RELEASE_USERNAME", "RELEASE_TOKEN"])
def test_missing_credentials_fail_before_network(run_upload, credential):
    result, calls = run_upload(**{credential: ""})
    assert result.returncode != 0
    assert calls == []


@pytest.mark.parametrize("filename", [
    "invalid.whl",
    "demo_package-notversion-py3-none-any.whl",
    "demo_package-1.2.3-py3-none-any%2F.whl",
    "demo_package-1.2.3-py3-none-any\n.whl",
    "other_package-1.2.3-py3-none-any.whl",
])
def test_invalid_wheel_names_fail_before_network(run_upload, filename):
    result, calls = run_upload((filename,))
    assert result.returncode != 0
    assert "::error::" in result.stderr
    assert calls == []


def test_all_paths_are_validated_before_first_upload(run_upload):
    result, calls = run_upload((WHEEL_NAME, "zzz_other-1.2.3-py3-none-any.whl"))
    assert result.returncode != 0
    assert "包名与 package_name 不一致" in result.stderr
    assert calls == []


@pytest.mark.parametrize("package_name", ["", "../other", "name|other"])
def test_invalid_package_path_is_rejected(run_upload, package_name):
    result, calls = run_upload(PACKAGE_NAME=package_name)
    assert result.returncode != 0
    assert calls == []


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_non_regular_wheels_are_rejected(workspace, run_upload, kind):
    path = workspace / "dist" / WHEEL_NAME
    if kind == "symlink":
        target = workspace / "unrelated-file"
        target.write_bytes(b"not an artifact")
        path.symlink_to(target)
    else:
        path.mkdir()
    result, calls = run_upload(())
    assert result.returncode != 0
    assert "上传产物必须是普通 wheel 文件" in result.stderr
    assert calls == []


@pytest.mark.parametrize("status", ["301", "302", "307", "401", "403", "409", "500"])
def test_non_success_status_stops_uploads(run_upload, status):
    filenames = (WHEEL_NAME, "demo_package-1.2.3-cp312-cp312-linux_x86_64.whl")
    result, calls = run_upload(filenames, TEST_HTTP_STATUS=status)
    assert result.returncode != 0
    assert "HTTP 状态码：" + status in result.stderr
    assert len(calls) == 1
    assert "已上传" not in result.stdout


@pytest.mark.parametrize("exit_code", ["7", "22", "28"])
def test_curl_failure_propagates(run_upload, exit_code):
    result, calls = run_upload(TEST_CURL_EXIT=exit_code, TEST_HTTP_STATUS="000")
    assert result.returncode == int(exit_code)
    assert len(calls) == 1
    assert "已上传" not in result.stdout
