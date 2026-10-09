"""矩阵准备脚本的 CPU 单元测试，不依赖 Torch、PPU 或镜像仓库。"""

import importlib.util
import itertools
import json
from pathlib import Path
import runpy
from unittest.mock import call, create_autospec, mock_open

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_matrix.py"
DEFAULT_INPUTS = {
    "python_versions": '["310"]',
    "os_versions": '["ubuntu2204"]',
    "hggcrt_versions": '["3"]',
    "torch_versions": '["2.8.0"]',
    "sdk_versions": '["2.1.0"]',
}
ALL_VERSIONS = {
    "python_versions": ["38", "310", "312"],
    "os_versions": ["ubuntu2004", "ubuntu2204", "ubuntu2404", "alios7u2"],
    "hggcrt_versions": ["2", "3"],
    "torch_versions": ["2.8.0", "2.10.0", "2.13.0"],
    "sdk_versions": ["2.1.0", "2.2.0"],
}
CONFIGURED_PAIRS = [
    ("ubuntu2004", "38"),
    ("ubuntu2204", "310"),
    ("ubuntu2404", "312"),
    ("alios7u2", "38"),
    ("alios7u2", "310"),
    ("alios7u2", "312"),
]


@pytest.fixture
def script():
    # 按文件路径加载，避免导入 deep_gemm 及其硬件依赖。
    spec = importlib.util.spec_from_file_location("prepare_matrix_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def output_file(script, monkeypatch):
    output = mock_open()
    # 仅替换目标模块的文件接口，避免影响 pytest 自身的文件读取。
    monkeypatch.setattr(script, "open", output, raising=False)
    return output


@pytest.fixture
def run_matrix(script, output_file, monkeypatch):
    monkeypatch.delenv("BUILD_CONTAINER_IMAGES", raising=False)
    monkeypatch.setenv("GITHUB_OUTPUT", "github-output")

    def run(inputs=None, raw_inputs=None):
        raw = json.dumps(DEFAULT_INPUTS if inputs is None else inputs)
        monkeypatch.setenv("INPUTS_JSON", raw if raw_inputs is None else raw_inputs)
        script.main()
        output_file.assert_called_once_with("github-output", "a", encoding="utf-8")
        written = "".join(call.args[0] for call in output_file().write.call_args_list)
        assert written.startswith("matrix=")
        assert written.endswith("\n")
        assert written.count("\n") == 1
        matrix = json.loads(written[len("matrix="):])
        assert set(matrix) == {"include"}
        return matrix["include"]

    return run


def test_fail_formats_github_annotation(script):
    with pytest.raises(SystemExit) as error:
        script.fail("测试错误")
    assert str(error.value) == "::error::测试错误"


@pytest.mark.parametrize("value", [["38", "310"], {"key": "value"}, None, 3, "text"])
def test_parse_json_accepts_valid_json(script, value):
    assert script.parse_json(json.dumps(value), "参数") == value


@pytest.mark.parametrize("raw", ["", "[", "[1,]", "not-json", None, []])
def test_parse_json_rejects_invalid_json(script, raw):
    with pytest.raises(SystemExit, match="::error::参数 必须是合法 JSON"):
        script.parse_json(raw, "参数")


@pytest.mark.parametrize("name,versions", [
    ("get_torch_url", ("310", "ubuntu2204", "3", "2.8.0", "2.1.0")),
    ("get_sdk_url", ("ubuntu2204", "3", "2.1.0")),
])
def test_url_stubs_return_empty_string(script, name, versions):
    assert getattr(script, name)(*versions) == ""


@pytest.mark.parametrize("os_version,python_version", CONFIGURED_PAIRS)
def test_each_configured_image(script, run_matrix, capsys, os_version, python_version):
    inputs = dict(DEFAULT_INPUTS, os_versions=json.dumps([os_version]),
                  python_versions=json.dumps([python_version]))
    assert run_matrix(inputs) == [{
        "python_version": python_version,
        "os_version": os_version,
        "hggcrt_version": "3",
        "torch_version": "2.8.0",
        "sdk_version": "2.1.0",
        "image": script.IMAGES_MAP[f"{os_version}-py{python_version}"],
        "torch_url": "",
        "sdk_url": "",
    }]
    assert capsys.readouterr().out == "已生成 1 个矩阵任务\n"


def test_package_metadata_does_not_change_matrix(script, run_matrix):
    inputs = dict(DEFAULT_INPUTS, package_name="deep_gemm", framework="torch")
    expected_versions = {
        name.removesuffix("s"): json.loads(value)[0]
        for name, value in DEFAULT_INPUTS.items()
    }
    assert run_matrix(inputs) == [{
        **expected_versions,
        "image": script.IMAGES_MAP["ubuntu2204-py310"],
        "torch_url": "",
        "sdk_url": "",
    }]


def test_full_five_axis_cartesian_product(script, run_matrix, monkeypatch, capsys):
    # 当前映射没有覆盖所有 Python × OS 组合；本测试补齐虚拟映射以隔离展开逻辑。
    images = {
        f"{os_version}-py{python_version}": f"test.invalid/matrix:{os_version}-py{python_version}"
        for os_version, python_version in itertools.product(
            ALL_VERSIONS["os_versions"], ALL_VERSIONS["python_versions"]
        )
    }
    monkeypatch.setattr(script, "IMAGES_MAP", images)

    def make_torch_url(python_version, os_version, hggcrt_version, torch_version, sdk_version):
        return (
            f"https://test.invalid/torch/{os_version}/py{python_version}/"
            f"hggcrt{hggcrt_version}/torch{torch_version}/sdk{sdk_version}.whl"
        )

    def make_sdk_url(os_version, hggcrt_version, sdk_version):
        return f"https://test.invalid/sdk/{os_version}/hggcrt{hggcrt_version}/sdk{sdk_version}.tar.gz"

    torch_resolver = create_autospec(script.get_torch_url, side_effect=make_torch_url)
    sdk_resolver = create_autospec(script.get_sdk_url, side_effect=make_sdk_url)
    monkeypatch.setattr(script, "get_torch_url", torch_resolver)
    monkeypatch.setattr(script, "get_sdk_url", sdk_resolver)
    result = run_matrix({key: json.dumps(values) for key, values in ALL_VERSIONS.items()})
    fields = [key.removesuffix("s") for key in ALL_VERSIONS]
    assert [tuple(entry[key] for key in fields) for entry in result] == list(
        itertools.product(*ALL_VERSIONS.values())
    )
    assert len(result) == 144
    sdk_fields = ["os_version", "hggcrt_version", "sdk_version"]
    for entry in result:
        assert set(entry) == set(fields) | {"image", "torch_url", "sdk_url"}
        assert entry["image"] == images[f"{entry['os_version']}-py{entry['python_version']}"]
        assert entry["torch_url"] == make_torch_url(**{key: entry[key] for key in fields})
        assert entry["sdk_url"] == make_sdk_url(**{key: entry[key] for key in sdk_fields})
    assert torch_resolver.call_args_list == [
        call(**{key: entry[key] for key in fields}) for entry in result
    ]
    assert sdk_resolver.call_args_list == [
        call(**{key: entry[key] for key in sdk_fields}) for entry in result
    ]
    assert capsys.readouterr().out == "已生成 144 个矩阵任务\n"


def test_normalizes_numeric_python_and_deduplicates_all_axes(run_matrix):
    inputs = {
        "python_versions": '[312,"312",38,"38",310]',
        "os_versions": '["alios7u2","alios7u2"]',
        "hggcrt_versions": '["3","2","3"]',
        "torch_versions": '["2.13.0","2.8.0","2.13.0"]',
        "sdk_versions": '["2.2.0","2.1.0","2.2.0"]',
    }
    result = run_matrix(inputs)
    fields = [key.removesuffix("s") for key in inputs]
    expected = itertools.product(
        ["312", "38", "310"], ["alios7u2"], ["3", "2"],
        ["2.13.0", "2.8.0"], ["2.2.0", "2.1.0"],
    )
    assert [tuple(entry[key] for key in fields) for entry in result] == list(expected)
    assert len(result) == 24


def test_url_resolver_receives_normalized_python(script, run_matrix, monkeypatch):
    resolver = create_autospec(script.get_torch_url, return_value="https://test.invalid/torch.whl")
    monkeypatch.setattr(script, "get_torch_url", resolver)
    result = run_matrix(dict(DEFAULT_INPUTS, python_versions='[310,"310"]'))
    resolver.assert_called_once_with(
        python_version="310", os_version="ubuntu2204", hggcrt_version="3",
        torch_version="2.8.0", sdk_version="2.1.0",
    )
    assert result[0]["torch_url"] == "https://test.invalid/torch.whl"


def test_only_selected_image_mapping_is_required(script, run_matrix, monkeypatch):
    monkeypatch.setattr(script, "IMAGES_MAP", {
        "ubuntu2204-py310": "test.invalid/matrix:custom",
        "unused-py38": None,
    })
    assert run_matrix()[0]["image"] == "test.invalid/matrix:custom"


def test_rejects_malformed_outer_json(run_matrix, output_file):
    with pytest.raises(SystemExit, match="触发参数 必须是合法 JSON"):
        run_matrix(raw_inputs="{")
    output_file.assert_not_called()


@pytest.mark.parametrize("name", DEFAULT_INPUTS)
def test_rejects_missing_input(run_matrix, output_file, name):
    inputs = dict(DEFAULT_INPUTS)
    del inputs[name]
    with pytest.raises(SystemExit, match=f"{name} 必须是合法 JSON"):
        run_matrix(inputs)
    output_file.assert_not_called()


@pytest.mark.parametrize("name", DEFAULT_INPUTS)
def test_rejects_malformed_version_array(run_matrix, output_file, name):
    with pytest.raises(SystemExit, match=f"{name} 必须是合法 JSON"):
        run_matrix(dict(DEFAULT_INPUTS, **{name: "["}))
    output_file.assert_not_called()


@pytest.mark.parametrize("name", DEFAULT_INPUTS)
@pytest.mark.parametrize("raw", ["[]", "null", "{}", '"310"', "310"])
def test_requires_nonempty_array(run_matrix, output_file, name, raw):
    with pytest.raises(SystemExit, match=f"{name} 必须是非空 JSON 数组"):
        run_matrix(dict(DEFAULT_INPUTS, **{name: raw}))
    output_file.assert_not_called()


@pytest.mark.parametrize("name", DEFAULT_INPUTS)
@pytest.mark.parametrize("value", [None, True, 3.1, {}, [], "", " ", "3\n", "$(id)"])
def test_rejects_invalid_version_elements(run_matrix, output_file, name, value):
    with pytest.raises(SystemExit, match=f"{name} 的元素必须是非空版本字符串"):
        run_matrix(dict(DEFAULT_INPUTS, **{name: json.dumps([value])}))
    output_file.assert_not_called()


@pytest.mark.parametrize("name,value", [
    ("python_versions", "39"),
    ("os_versions", "ubuntu1804"),
    ("hggcrt_versions", "4"),
    ("torch_versions", "2.7.0"),
    ("sdk_versions", "2.3.0"),
])
def test_rejects_unsupported_candidates(run_matrix, output_file, name, value):
    with pytest.raises(SystemExit, match=f"{name} 包含不支持的值 {value}"):
        run_matrix(dict(DEFAULT_INPUTS, **{name: json.dumps([value])}))
    output_file.assert_not_called()


@pytest.mark.parametrize("name", [name for name in DEFAULT_INPUTS if name != "python_versions"])
def test_numeric_elements_are_only_supported_for_python(run_matrix, output_file, name):
    with pytest.raises(SystemExit, match=f"{name} 的元素必须是非空版本字符串"):
        run_matrix(dict(DEFAULT_INPUTS, **{name: "[2]"}))
    output_file.assert_not_called()


def test_rejects_unconfigured_python_os_pair(run_matrix, output_file):
    with pytest.raises(SystemExit, match="ubuntu2204-py38"):
        run_matrix(dict(DEFAULT_INPUTS, python_versions='["38"]'))
    output_file.assert_not_called()


@pytest.mark.parametrize("image", [None, "", " ", "test image", "test\nimage", 123, [], {}])
def test_rejects_invalid_selected_image(script, run_matrix, output_file, monkeypatch, image):
    monkeypatch.setattr(script, "IMAGES_MAP", {"ubuntu2204-py310": image})
    with pytest.raises(SystemExit, match="ubuntu2204-py310"):
        run_matrix()
    output_file.assert_not_called()


def test_checks_every_selected_image_before_writing_output(run_matrix, output_file):
    inputs = dict(DEFAULT_INPUTS, python_versions='["38","310"]',
                  os_versions='["ubuntu2004","alios7u2"]')
    with pytest.raises(SystemExit, match="ubuntu2004-py310"):
        run_matrix(inputs)
    output_file.assert_not_called()


def test_rejects_matrix_over_github_limit(script, run_matrix, output_file, monkeypatch):
    # 当前候选值最多生成 144 个任务，模拟乘积以覆盖超限保护分支。
    monkeypatch.setattr(script.math, "prod", lambda values: 257)
    with pytest.raises(SystemExit, match="257.*256"):
        run_matrix()
    output_file.assert_not_called()


def test_script_entrypoint(monkeypatch):
    monkeypatch.setenv("INPUTS_JSON", json.dumps(DEFAULT_INPUTS))
    monkeypatch.setenv("GITHUB_OUTPUT", "github-output")
    monkeypatch.delenv("BUILD_CONTAINER_IMAGES", raising=False)
    output = mock_open()
    runpy.run_path(str(SCRIPT_PATH), init_globals={"open": output}, run_name="__main__")
    output.assert_called_once_with("github-output", "a", encoding="utf-8")
    written = "".join(call.args[0] for call in output().write.call_args_list)
    assert json.loads(written[len("matrix="):])["include"][0]["hggcrt_version"] == "3"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
