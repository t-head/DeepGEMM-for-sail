"""打包 CPU 测试：验证源码包流程及 wheel 的真实 SCM 版本推导，不编译 C++ 扩展。"""

import ast
from email.parser import Parser
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
import pytest
from setuptools_scm import get_version
import yaml


CI_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = CI_ROOT / "workflows/matrix-release.yml"
SETUP_TEMPLATE = CI_ROOT / "scripts/setup.py"

# 仅拦截 SCM 模块，版本规范化和源码包构建仍使用真实 Python 代码。
# 在执行 setup.py 前断言 FAKE_WHEEL，防止回归时进入下载分支。
PYTHON_SHIM = r'''
python() {
  if [[ "${1:-}" == "-c" ]]; then
    "$TEST_PYTHON" -c '
import os
from pathlib import Path
import sys
import types

def get_version(root, local_scheme):
    assert root == "source"
    assert local_scheme == "no-local-version"
    assert (Path(root) / "setup.py").is_file()
    if os.environ.get("TEST_SCM_FAIL") == "1":
        raise LookupError("SCM version unavailable")
    return os.environ["TEST_SCM_VERSION"]

sys.modules["setuptools_scm"] = types.SimpleNamespace(get_version=get_version)
exec(sys.argv[1])
' "$2"
  else
    [[ "${__FAKE_BUILD:-}" == "FAKE_WHEEL" ]] || return 42
    [[ "${TEST_SDIST_FAIL:-}" != "1" ]] || return 23
    "$TEST_PYTHON" "$@"
  fi
}
'''


@pytest.fixture
def workflow():
    # BaseLoader 保留 GitHub Actions 的 on 键，不按 YAML 1.1 转换为布尔值。
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


@pytest.fixture
def build_step(workflow):
    return next(step for step in workflow["jobs"]["build-sdist"]["steps"]
                if step.get("name") == "填充模板并构建源码包")


@pytest.fixture
def workspace():
    # 临时构建只写入工作区，测试结束后自动清理。
    with tempfile.TemporaryDirectory(prefix=".sdist-test-", dir=CI_ROOT) as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        (source / "setup.py").write_text("# 调用方原始构建脚本\n", encoding="utf-8")
        template = root / "ci-template/.github/scripts/setup.py"
        template.parent.mkdir(parents=True)
        template.write_bytes(SETUP_TEMPLATE.read_bytes())
        yield root


@pytest.fixture
def run_build(build_step, workspace):
    def run(version="1.2.3", package_name="demo_package", framework="torch", **overrides):
        env = dict(os.environ, TEST_PYTHON=sys.executable, TEST_SCM_VERSION=version,
                   pkg_name=package_name, framework=framework, PYTHONDONTWRITEBYTECODE="1")
        env.pop("__FAKE_BUILD", None)
        env.pop("TEST_SCM_FAIL", None)
        env.pop("TEST_SDIST_FAIL", None)
        env.update(overrides)
        return subprocess.run(["bash"], input=PYTHON_SHIM + build_step["run"],
                              cwd=workspace, env=env, text=True, capture_output=True,
                              timeout=60)

    return run


def test_sdist_precedes_matrix_and_follows_authorization(workflow):
    jobs = workflow["jobs"]
    assert jobs["prepare-matrix"]["steps"][0]["name"] == "校验调用方组织"
    assert jobs["build-sdist"]["needs"] == "prepare-matrix"
    assert set(jobs["matrix-job"]["needs"]) == {"prepare-matrix", "build-sdist"}
    assert jobs["matrix-job"]["strategy"]["matrix"] == "${{ fromJSON(needs.prepare-matrix.outputs.matrix) }}"
    assert "if" not in jobs["build-sdist"]
    assert "continue-on-error" not in jobs["build-sdist"]


def test_source_and_template_checkouts_are_separate(workflow):
    source, template = workflow["jobs"]["build-sdist"]["steps"][:2]
    assert source["uses"] == template["uses"] == "actions/checkout@v4"
    assert source["with"]["path"] == "source"
    assert source["with"]["fetch-depth"] == "0"
    assert "repository" not in source["with"]
    assert "ref" not in source["with"]
    assert template["with"]["path"] == "ci-template"
    assert template["with"]["repository"] == "${{ inputs.template_repository }}"
    assert template["with"]["ref"] == "${{ inputs.template_ref }}"
    assert template["with"]["token"] == "${{ secrets.template_token || github.token }}"
    assert source["with"]["persist-credentials"] == template["with"]["persist-credentials"] == "false"


def test_inputs_and_artifact_configuration(workflow, build_step):
    job = workflow["jobs"]["build-sdist"]
    assert "container" not in job
    assert build_step["env"] == {
        "pkg_name": "${{ inputs.package_name }}",
        "framework": "${{ inputs.framework }}",
    }
    assert "release_token" not in str(job)
    upload = job["steps"][-1]
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["name"] == "${{ inputs.package_name }}-sdist"
    assert upload["with"]["path"] == "sdist/dist/*.tar.gz"
    assert upload["with"]["if-no-files-found"] == "error"


def test_sdist_shell_syntax(workflow):
    for step in workflow["jobs"]["build-sdist"]["steps"]:
        if "run" in step:
            result = subprocess.run(["bash", "-n"], input=step["run"],
                                    text=True, capture_output=True)
            assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("version,framework,expected_version", [
    ("1.2.3", "torch", "1.2.3+oe"),
    ("1.2.4.dev5+gabcdef", "torch", "1.2.4.dev5+oe"),
    ("2.0.0rc1+old", "", "2.0.0rc1+oe"),
])
def test_builds_sdist_with_rendered_metadata(workspace, run_build, version, framework, expected_version):
    result = run_build(version=version, framework=framework)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (workspace / "source/setup.py").read_text() == "# 调用方原始构建脚本\n"
    assert (workspace / "ci-template/.github/scripts/setup.py").read_bytes() == SETUP_TEMPLATE.read_bytes()

    rendered = (workspace / "sdist/setup.py").read_text(encoding="utf-8")
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(rendered).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"PACKAGE_NAME", "framework", "build_version"}
    }
    assert assignments == {
        "PACKAGE_NAME": "demo_package",
        "framework": framework,
        "build_version": expected_version,
    }
    archives = list((workspace / "sdist/dist").glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        prefix = archive.getnames()[0].split("/")[0]
        metadata = Parser().parsestr(archive.extractfile(f"{prefix}/PKG-INFO").read().decode("utf-8"))
        assert canonicalize_name(metadata["Name"]) == "demo-package"
        assert metadata["Version"] == expected_version
        assert metadata["Requires-Python"] == ">=3.8"
        assert archive.extractfile(f"{prefix}/setup.py").read().decode("utf-8") == rendered
        assert not any("ci-template" in name or "/source/" in name for name in archive.getnames())


@pytest.mark.parametrize("field,value", [
    ("package_name", ""),
    ("package_name", "../outside"),
    ("package_name", "bad|name"),
    ("package_name", 'bad"name'),
    ("package_name", "$(printf unsafe)"),
    ("package_name", "name\nother"),
    ("framework", "torch|other"),
    ("framework", 'torch"'),
    ("framework", "$(printf unsafe)"),
    ("framework", "torch\nother"),
])
def test_rejects_unsafe_metadata_before_rendering(workspace, run_build, field, value):
    result = run_build(**{field: value})
    assert result.returncode != 0
    assert f"::error::{field}" in result.stderr
    assert not (workspace / "sdist").exists()


@pytest.mark.parametrize("field", ["PACKAGE_NAME", "framework", "build_version"])
def test_missing_placeholder_fails(workspace, run_build, field):
    template = workspace / "ci-template/.github/scripts/setup.py"
    template.write_text(template.read_text().replace(f'{field} = ""', f'{field} = "changed"'), encoding="utf-8")
    result = run_build()
    assert result.returncode != 0
    assert "::error::源码包模板缺少预期占位字段" in result.stderr
    assert not (workspace / "sdist").exists()


@pytest.mark.parametrize("overrides", [{"TEST_SCM_FAIL": "1"}, {"version": "not-a-version"}])
def test_invalid_scm_version_stops_build(workspace, run_build, overrides):
    result = run_build(**overrides)
    assert result.returncode != 0
    assert not (workspace / "sdist").exists()


def test_sdist_failure_propagates(workspace, run_build):
    result = run_build(TEST_SDIST_FAIL="1")
    assert result.returncode == 23
    assert not list((workspace / "sdist").glob("dist/*.tar.gz"))


# 运行仓库实际 setup.py，仅隔离 C++ 编译和头文件复制；版本推导及 wheel 打包使用真实实现。
WHEEL_BUILD_RUNNER = r'''
import runpy
import sys
import types
import setuptools

for name in ("torch", "torch.utils", "torch.utils.cpp_extension"):
    sys.modules[name] = types.ModuleType(name)
cpp_extension = sys.modules["torch.utils.cpp_extension"]
cpp_extension.CppExtension = lambda **kwargs: None
cpp_extension.BuildExtension = object
original_setup = setuptools.setup

def cpu_setup(**kwargs):
    kwargs.update(ext_modules=[], cmdclass={}, packages=[], package_data={})
    return original_setup(**kwargs)

setuptools.setup = cpu_setup
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''


@pytest.fixture
def wheel_build(workspace, monkeypatch):
    source = workspace / "source"
    (source / "setup.py").write_bytes((CI_ROOT.parent / "setup.py").read_bytes())
    # 避免调用者的 SCM 覆盖值或 Git 配置影响临时仓库，且不修改用户 Git 配置。
    for name in tuple(os.environ):
        if name.startswith(("SETUPTOOLS_SCM_", "GIT_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "SCM Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "scm-test@example.invalid")

    def run(*args, cwd=None):
        return subprocess.run(
            [sys.executable, "-c", WHEEL_BUILD_RUNNER, str(source / "setup.py"), *args],
            cwd=workspace if cwd is None else cwd,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            text=True, capture_output=True, timeout=60,
        )

    return run


@pytest.fixture
def scm_repository(workspace, wheel_build):
    source = workspace / "source"

    def git(*args):
        return subprocess.run(["git", *args], cwd=source, check=True,
                              text=True, capture_output=True)

    git("init", "--quiet")
    git("add", "setup.py")
    git("commit", "--quiet", "-m", "初始化测试源码")
    return source, git


@pytest.mark.parametrize("tag,state,expected_public", [
    ("2.4.0", "exact", "2.4.0"),
    ("v2.4.0", "exact", "2.4.0"),
    ("v2.4.0rc1", "exact", "2.4.0rc1"),
    ("1.0.0+v0.1.0", "exact", "1.0.0"),
    ("1.0.0+v0.1.0", "commit", "1.0.1.dev1"),
    ("v2.4.0", "commit", "2.4.1.dev1"),
    ("v2.4.0", "dirty", "2.4.1.dev0"),
    (None, "exact", "0.1.dev1"),
])
def test_wheel_and_sdist_share_scm_public_version(
    scm_repository, wheel_build, tag, state, expected_public,
):
    source, git = scm_repository
    if tag:
        git("tag", tag)
    if state != "exact":
        setup_path = source / "setup.py"
        setup_path.write_text(setup_path.read_text() + "\n# 测试后续变更\n", encoding="utf-8")
        if state == "commit":
            git("add", "setup.py")
            git("commit", "--quiet", "-m", "测试后续提交")

    # 从仓库外执行，验证 SCM 根目录绑定到 setup.py，而不是当前工作目录。
    result = wheel_build("--version")
    assert result.returncode == 0, result.stdout + result.stderr
    version = Version(result.stdout.strip().splitlines()[-1])
    assert version.public == expected_public
    assert version.public == Version(get_version(root=source, local_scheme="no-local-version")).public
    if tag and state == "exact":
        assert version == Version(tag)
    else:
        assert any(part.startswith("g") for part in version.local.split("."))
        if state == "dirty":
            date = version.local.split(".")[-1]
            assert date.startswith("d") and len(date) == 9 and date[1:].isdigit()


@pytest.mark.parametrize("variable", [
    "SETUPTOOLS_SCM_PRETEND_VERSION",
    "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DEEP_GEMM",
])
def test_wheel_version_override_without_git(wheel_build, monkeypatch, variable):
    monkeypatch.setenv(variable, "3.2.1+cpu")
    result = wheel_build("--version")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "3.2.1+cpu"


def test_wheel_without_scm_metadata_fails(wheel_build):
    result = wheel_build("--version")
    assert result.returncode != 0
    assert "setuptools-scm was unable to detect version" in result.stderr


def test_wheel_reads_sdist_metadata_without_git(workspace, wheel_build):
    (workspace / "source/PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: deep_gemm\nVersion: 3.2.1\n", encoding="utf-8")
    result = wheel_build("--version")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "3.2.1"


def test_wheel_filename_and_metadata_use_scm_version(scm_repository, wheel_build):
    source, git = scm_repository
    git("tag", "v2.4.0")
    result = wheel_build("bdist_wheel", cwd=source)
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list((source / "dist").glob("*.whl"))
    assert len(wheels) == 1
    name, version, _, _ = parse_wheel_filename(wheels[0].name)
    assert name == "deep-gemm"
    assert version == Version("2.4.0")
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata_path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
    assert metadata["Version"] == str(version)


def test_wheel_ci_has_scm_dependency_and_full_history(workflow):
    checkout = next(step for step in workflow["jobs"]["matrix-job"]["steps"]
                    if step.get("name") == "检出待编译仓库")
    assert checkout["with"]["fetch-depth"] == "0"
    caller = yaml.load((CI_ROOT / "workflows/build.yml").read_text(encoding="utf-8"),
                       Loader=yaml.BaseLoader)
    command = caller["jobs"]["build"]["with"]["build_command"]
    assert "setuptools-scm==9.2.2" in command
    assert command.index("setuptools-scm==9.2.2") < command.index("setup.py bdist_wheel")
