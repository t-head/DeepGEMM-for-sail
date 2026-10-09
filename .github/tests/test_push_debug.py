"""push 调试入口的离线验证，不运行构建容器或访问发布服务。"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import pytest
import yaml


CI_ROOT = Path(__file__).resolve().parents[1]
DEBUG_INPUTS = {
    "package_name": "deep_gemm",
    "framework": "torch",
    "python_versions": '["312"]',
    "os_versions": '["ubuntu2404"]',
    "hggcrt_versions": '["3"]',
    "sdk_versions": '["2.2.0"]',
    "torch_versions": '["2.13.0"]',
}


@pytest.fixture
def caller():
    # BaseLoader 只构造基础字符串/容器，保留 Actions 的 on 键及布尔配置原文。
    return yaml.load((CI_ROOT / "workflows/build.yml").read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)


@pytest.fixture
def template():
    return yaml.load((CI_ROOT / "workflows/matrix-release.yml").read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)


def test_push_only_targets_debug_branch_and_keeps_manual_entry(caller):
    assert caller["on"]["push"] == {"branches": ["feat/add-release-workflow"]}
    assert "workflow_dispatch" in caller["on"]
    assert caller["jobs"]["build"]["uses"] == "./.github/workflows/matrix-release.yml"
    assert caller["jobs"]["build"]["with"]["template_ref"] == "${{ github.sha }}"
    assert caller["jobs"]["build"]["with"]["template_repository"] == "${{ github.repository }}"


@pytest.mark.parametrize("key,value", DEBUG_INPUTS.items())
def test_push_parameters_are_fixed_and_dispatch_inputs_preserved(caller, key, value):
    expected = "${{ github.event_name == 'push' && '" + value + "' || inputs." + key + " }}"
    assert caller["jobs"]["build"]["with"][key] == expected
    assert key in caller["on"]["workflow_dispatch"]["inputs"]
    if key.endswith("_versions"):
        assert isinstance(json.loads(value), list)


def test_debug_disables_publishing_and_does_not_forward_release_secrets(caller, template):
    job = caller["jobs"]["build"]
    assert job["with"]["publish"] == "${{ github.event_name != 'push' }}"
    interface = template["on"]["workflow_call"]
    assert interface["inputs"]["publish"]["type"] == "boolean"
    assert interface["inputs"]["publish"]["default"] == "true"
    for name in ("release_username", "release_token"):
        assert job["secrets"][name] == (
            "${{ github.event_name != 'push' && secrets." + name.upper() + " || '' }}"
        )
        assert interface["secrets"][name]["required"] == "false"


def test_all_publish_paths_are_gated(template):
    jobs = template["jobs"]
    steps = {step["name"]: step for step in jobs["matrix-job"]["steps"]}
    for name in ("上传 wheel 到 Artifactory", "记录矩阵任务成功", "保存矩阵任务成功标记"):
        assert steps[name]["if"] == "${{ inputs.publish }}"
    assert steps["保存 GitHub Release wheel 产物"]["if"] == "${{ inputs.publish && github.ref_type == 'tag' }}"
    for name in ("publish-sdist", "publish-github-release"):
        assert jobs[name]["if"].startswith("${{ !cancelled() && inputs.publish && ")
    assert "if" not in jobs["build-sdist"]
    assert "if" not in jobs["matrix-job"]


def test_debug_keeps_downloadable_artifacts_without_release_credentials(template):
    steps = template["jobs"]["matrix-job"]["steps"]
    save = next(step for step in steps if step["name"] == "保存调试 wheel 产物")
    build = next(step for step in steps if step["name"] == "编译")
    assert steps.index(build) < steps.index(save)
    assert save["if"] == "${{ !inputs.publish }}"
    assert save["uses"] == "actions/upload-artifact@v4"
    assert save["with"]["path"] == "dist/*.whl"
    assert save["with"]["if-no-files-found"] == "error"
    assert save["with"]["name"] == (
        "${{ inputs.package_name }}-debug-wheels-${{ github.run_id }}-"
        "${{ github.run_attempt }}-${{ strategy.job-index }}"
    )
    assert "env" not in save
    assert template["jobs"]["build-sdist"]["steps"][-1]["uses"] == "actions/upload-artifact@v4"
    assert template["jobs"]["matrix-job"].get("permissions", template["permissions"]) == {"contents": "read"}


def test_push_inputs_generate_one_real_matrix_entry(caller):
    # 从实际入口提取 push 分支字面值，运行真实矩阵脚本，检查当前镜像映射是否匹配。
    inputs = {}
    for key in DEBUG_INPUTS:
        match = re.fullmatch(
            r"\$\{\{ github\.event_name == 'push' && '([^']+)' \|\| inputs\." + key + r" \}\}",
            caller["jobs"]["build"]["with"][key],
        )
        assert match is not None
        inputs[key] = match[1]
    inputs["publish"] = False
    with tempfile.TemporaryDirectory(prefix=".push-debug-test-", dir=CI_ROOT) as directory:
        output = Path(directory) / "github-output"
        result = subprocess.run(
            [sys.executable, "-B", str(CI_ROOT / "scripts/prepare_matrix.py")],
            env=dict(os.environ, INPUTS_JSON=json.dumps(inputs), GITHUB_OUTPUT=str(output)),
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        matrix = json.loads(output.read_text().removeprefix("matrix="))
    assert len(matrix["include"]) == 1
    entry = matrix["include"][0]
    for key, value in DEBUG_INPUTS.items():
        if key.endswith("_versions"):
            assert entry[key.removesuffix("s")] == json.loads(value)[0]
    assert entry["image"] == "reg.docker.alibaba-inc.com/aisw/thead-ppu:2.2.0-hggcrt3-ubuntu24.04-py312-20261005"
