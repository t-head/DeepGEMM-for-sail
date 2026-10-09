"""GitHub Release 离线测试：执行 workflow 中的 JavaScript，仅模拟 GitHub API。"""

from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest
import yaml


CI_ROOT = Path(__file__).resolve().parents[1]
WHEEL = "demo_package-1.2.3+oe-cp310-cp310-linux_x86_64.whl"
OTHER_WHEEL = "demo_package-1.2.3+oe-cp312-cp312-linux_x86_64.whl"
CONTENT = "test wheel data"
DIGEST = "sha256:" + hashlib.sha256(CONTENT.encode()).hexdigest()
SHA = "a" * 40

NODE_HARNESS = r'''
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const calls = [];
const logs = [];
const repos = {};
for (const method of ['getCommit', 'getReleaseByTag', 'createRelease', 'listReleaseAssets', 'uploadReleaseAsset']) {
  repos[method] = async params => {
    const logged = {...params};
    if (Buffer.isBuffer(logged.data)) logged.data = logged.data.toString('utf8');
    calls.push({method, params: logged});
    if (input.errors && input.errors[method]) {
      throw Object.assign(new Error('模拟 API 失败'), {status: input.errors[method]});
    }
    switch (method) {
      case 'getCommit': return {data: {sha: input.remoteSha || input.context.sha}};
      case 'getReleaseByTag':
        if (input.release) return {data: input.release};
        throw Object.assign(new Error('不存在'), {status: 404});
      case 'createRelease': return {data: {id: 42, tag_name: params.tag_name}};
      case 'listReleaseAssets': return {data: input.assets || []};
      case 'uploadReleaseAsset': return {data: {id: 100, name: params.name}};
    }
  };
}
const github = {
  rest: {repos},
  paginate: async (method, params) => {
    if (method !== repos.listReleaseAssets) throw new Error('意外的分页 API');
    return (await method(params)).data;
  }
};
const core = {info: text => logs.push(text)};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  try {
    await new AsyncFunction('github', 'context', 'core', 'require', input.script)(github, input.context, core, require);
    process.stdout.write(JSON.stringify({calls, logs, error: null}));
  } catch (error) {
    process.stdout.write(JSON.stringify({calls, logs, error: error.message}));
  }
})();
'''


@pytest.fixture
def workflow():
    return yaml.load((CI_ROOT / "workflows/matrix-release.yml").read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)


@pytest.fixture
def release_job(workflow):
    return workflow["jobs"]["publish-github-release"]


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory(prefix=".github-release-test-", dir=CI_ROOT) as directory:
        yield Path(directory)


@pytest.fixture
def run_release(release_job, workspace):
    def run(files=None, package_name="demo_package", **overrides):
        if files is None:
            files = {"matrix-0/" + WHEEL: CONTENT}
        for relative, content in files.items():
            path = workspace / "release-wheels" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        payload = {
            "script": release_job["steps"][-1]["with"]["script"],
            "context": {"repo": {"owner": "t-head", "repo": "caller-project"},
                        "ref": "refs/tags/v1.2.3", "sha": SHA},
        }
        payload.update(overrides)
        result = subprocess.run(["node", "-e", NODE_HARNESS], input=json.dumps(payload),
                                cwd=workspace, env=dict(os.environ, PACKAGE_NAME=package_name),
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return run


def method_names(result):
    return [call["method"] for call in result["calls"]]


def test_tag_only_gate_and_partial_matrix_failure(workflow, release_job):
    assert release_job["if"] == (
        "${{ !cancelled() && github.ref_type == 'tag' && needs.build-sdist.result == 'success' && "
        "(needs.matrix-job.result == 'success' || needs.matrix-job.result == 'failure') }}"
    )
    assert set(release_job["needs"]) == {"build-sdist", "matrix-job"}
    assert "strategy" not in release_job
    assert "continue-on-error" not in release_job
    steps = workflow["jobs"]["matrix-job"]["steps"]
    save = next(step for step in steps if step.get("name") == "保存 GitHub Release wheel 产物")
    assert steps[steps.index(save) - 1]["name"] == "上传 wheel 到 Artifactory"
    assert save["if"] == "${{ github.ref_type == 'tag' }}"
    assert save["uses"] == "actions/upload-artifact@v4"
    assert save["with"]["path"] == "dist/*.whl"
    assert save["with"]["if-no-files-found"] == "error"
    assert "continue-on-error" not in save


def test_artifacts_are_isolated_by_run_attempt_and_matrix(workflow, release_job):
    save = next(step for step in workflow["jobs"]["matrix-job"]["steps"]
                if step.get("name") == "保存 GitHub Release wheel 产物")
    download = release_job["steps"][0]
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"]["path"] == "release-wheels"
    assert download["with"]["merge-multiple"] == "false"
    name = save["with"]["name"]
    pattern = download["with"]["pattern"]
    values = {"inputs.package_name": "demo_package", "github.run_id": "100", "github.run_attempt": "2"}
    for key, value in values.items():
        name = name.replace("${{ " + key + " }}", value)
        pattern = pattern.replace("${{ " + key + " }}", value)
    names = [name.replace("${{ strategy.job-index }}", str(index)) for index in range(144)]
    assert len(set(names)) == 144
    assert all(fnmatchcase(name, pattern) for name in names)
    assert not fnmatchcase(names[0].replace("-100-2-", "-100-1-"), pattern)
    assert not fnmatchcase(names[0].replace("-100-2-", "-99-2-"), pattern)


def test_write_permission_is_limited_to_release_job(workflow, release_job):
    assert workflow["permissions"] == {"contents": "read"}
    assert release_job["permissions"] == {"contents": "write"}
    for name, job in workflow["jobs"].items():
        if name != "publish-github-release":
            assert job.get("permissions", workflow["permissions"])["contents"] == "read"
    caller = yaml.load((CI_ROOT / "workflows/build.yml").read_text(), Loader=yaml.BaseLoader)
    assert caller["jobs"]["build"]["permissions"] == {"contents": "write"}
    assert caller["jobs"]["build"]["uses"] == "./.github/workflows/matrix-release.yml"
    publish = release_job["steps"][-1]
    assert publish["uses"] == "actions/github-script@v7"
    assert publish["with"]["github-token"] == "${{ github.token }}"
    assert "release_token" not in str(release_job)
    assert "actions/checkout" not in str(release_job)
    assert release_job["concurrency"] == {
        "group": "github-release-${{ github.repository }}-${{ github.ref }}",
        "cancel-in-progress": "false",
    }


def test_creates_release_on_calling_repository_and_uploads_exact_bytes(run_release):
    result = run_release()
    assert result["error"] is None
    assert method_names(result) == ["getCommit", "getReleaseByTag", "createRelease", "listReleaseAssets", "uploadReleaseAsset"]
    for call in result["calls"]:
        assert call["params"]["owner"] == "t-head"
        assert call["params"]["repo"] == "caller-project"
    create = result["calls"][2]["params"]
    assert create["tag_name"] == create["name"] == "v1.2.3"
    assert create["target_commitish"] == SHA
    upload = result["calls"][-1]["params"]
    assert upload["release_id"] == 42
    assert upload["name"] == WHEEL
    assert upload["data"] == CONTENT
    assert upload["headers"]["content-length"] == len(CONTENT)


def test_existing_release_metadata_is_not_changed(run_release):
    result = run_release(release={"id": 88, "draft": True, "prerelease": True, "name": "用户标题"})
    assert result["error"] is None
    assert "createRelease" not in method_names(result)
    assert result["calls"][-1]["params"]["release_id"] == 88


def test_no_wheels_skips_without_any_remote_calls(run_release):
    result = run_release(files={})
    assert result["error"] is None
    assert result["calls"] == []
    assert "跳过 GitHub Release" in result["logs"][0]


def test_branch_ref_is_rejected_without_remote_calls(run_release):
    result = run_release(context={"repo": {"owner": "t-head", "repo": "caller"},
                                  "ref": "refs/heads/main", "sha": SHA})
    assert "仅允许 tag" in result["error"]
    assert result["calls"] == []


def test_moved_tag_blocks_release(run_release):
    result = run_release(remoteSha="b" * 40)
    assert "当前提交与构建提交不一致" in result["error"]
    assert method_names(result) == ["getCommit"]


def test_partial_matrix_output_and_multiple_files(run_release):
    result = run_release(files={"matrix-1/" + WHEEL: CONTENT, "matrix-8/" + OTHER_WHEEL: "other wheel"})
    assert result["error"] is None
    assert {call["params"]["name"] for call in result["calls"] if call["method"] == "uploadReleaseAsset"} == {WHEEL, OTHER_WHEEL}
    assert method_names(result).count("createRelease") == 1


def test_same_name_identical_matrix_artifacts_are_deduplicated(run_release):
    result = run_release(files={"matrix-0/" + WHEEL: CONTENT, "matrix-1/" + WHEEL: CONTENT})
    assert result["error"] is None
    assert method_names(result).count("uploadReleaseAsset") == 1


def test_same_name_different_matrix_artifacts_fail_before_remote_calls(run_release):
    result = run_release(files={"matrix-0/" + WHEEL: CONTENT, "matrix-1/" + WHEEL: "different"})
    assert "同名但内容不同" in result["error"]
    assert result["calls"] == []


def test_existing_identical_asset_is_skipped(run_release):
    result = run_release(release={"id": 88}, assets=[{"name": WHEEL, "state": "uploaded", "digest": DIGEST}])
    assert result["error"] is None
    assert "uploadReleaseAsset" not in method_names(result)
    assert "附件内容相同" in result["logs"][0]


@pytest.mark.parametrize("asset", [
    {"name": WHEEL, "state": "uploaded", "digest": "sha256:different"},
    {"name": WHEEL, "state": "uploaded"},
    {"name": WHEEL, "state": "starter", "digest": DIGEST},
])
def test_existing_conflicting_asset_blocks_all_uploads(run_release, asset):
    result = run_release(files={"matrix-0/" + WHEEL: CONTENT, "matrix-1/" + OTHER_WHEEL: CONTENT},
                         release={"id": 88}, assets=[asset])
    assert "无法确认内容一致" in result["error"]
    assert "uploadReleaseAsset" not in method_names(result)


@pytest.mark.parametrize("method,status", [
    ("getCommit", 404), ("getReleaseByTag", 403), ("getReleaseByTag", 500),
    ("createRelease", 422), ("listReleaseAssets", 403), ("uploadReleaseAsset", 500),
])
def test_api_failure_propagates_without_overwriting_assets(run_release, method, status):
    result = run_release(errors={method: status})
    assert result["error"] == "模拟 API 失败"
    assert method_names(result)[-1] == method
    if method in ("getCommit", "getReleaseByTag"):
        assert "createRelease" not in method_names(result)


@pytest.mark.parametrize("filename", ["invalid.whl", "other_package-1.2.3-py3-none-any.whl"])
def test_invalid_wheel_blocks_network(run_release, filename):
    result = run_release(files={"matrix-0/" + filename: CONTENT})
    assert "文件名不合法或包名" in result["error"]
    assert result["calls"] == []


def test_symlink_wheel_is_rejected(workspace, run_release):
    target = workspace / "outside"
    target.write_text(CONTENT)
    directory = workspace / "release-wheels/matrix-0"
    directory.mkdir(parents=True)
    (directory / WHEEL).symlink_to(target)
    result = run_release(files={})
    assert "必须是普通文件" in result["error"]
    assert result["calls"] == []
