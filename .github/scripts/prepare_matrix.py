import itertools
import json
import math
import os
import re

IMAGES_MAP = {
    "ubuntu2004-py38": "registry.cn-hangzhou.aliyuncs.com/aliyun-ai/pytorch:2.8.0-py38-cu117",
    "ubuntu2204-py310": "registry.cn-hangzhou.aliyuncs.com/aliyun-ai/pytorch:2.8.0-py310-cu117",
    "ubuntu2404-py312-sdk2.2.0-hggcrt3-torch2.13.0": "reg.docker.alibaba-inc.com/aisw/thead-ppu:2.2.0-hggcrt3-ubuntu24.04-py312-20261005",
    "alios7u2-py38": "registry.cn-hangzhou.aliyuncs.com/aliyun-ai/pytorch:2.8.0-py38-cu117",
    "alios7u2-py310": "registry.cn-hangzhou.aliyuncs.com/aliyun-ai/pytorch:2.8.0-py310-cu117",
    "alios7u2-py312": "registry.cn-hangzhou.aliyuncs.com/aliyun-ai/pytorch:2.8.0-py312-cu117",
}

def fail(message):
    raise SystemExit(f"::error::{message}")


def parse_json(raw, label):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        fail(f"{label} 必须是合法 JSON")


def get_torch_url(
    python_version: str,
    os_version: str,
    hggcrt_version: str,
    torch_version: str,
    sdk_version: str,
) -> str:
    """根据五个版本获取 Torch 下载地址，暂返回空字符串作为占位。"""
    # TODO: 实现 Torch 下载地址查询逻辑。
    return ""


def get_sdk_url(os_version: str, hggcrt_version: str, sdk_version: str) -> str:
    """根据 OS、HGGCRT 和 SDK 版本获取 SDK 下载地址，暂返回空字符串作为占位。"""
    # TODO: 实现 SDK 下载地址查询逻辑。
    return ""


def main():
    inputs = parse_json(os.environ["INPUTS_JSON"], "触发参数")
    candidates = {
        "python_versions": {"38", "310", "312"},
        "os_versions": {"ubuntu2004", "ubuntu2204", "ubuntu2404", "alios7u2"},
        "hggcrt_versions": {"2", "3"},
        "torch_versions": {"2.8.0", "2.10.0", "2.13.0"},
        "sdk_versions": {"2.1.0", "2.2.0"},
    }
    axes = {}
    for name, allowed in candidates.items():
        values = parse_json(inputs.get(name, ""), name)
        if not isinstance(values, list) or not values:
            fail(f"{name} 必须是非空 JSON 数组")
        normalized = []
        for value in values:
            # Python 版本同时支持 [38,310] 和 ["38","310"]。
            if name == "python_versions" and type(value) is int:
                value = str(value)
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+\-]*", value):
                fail(f"{name} 的元素必须是非空版本字符串，仅允许字母、数字及 . _ + -")
            if allowed is not None and value not in allowed:
                fail(f"{name} 包含不支持的值 {value}；候选值：{', '.join(sorted(allowed))}")
            if value not in normalized:
                normalized.append(value)
        axes[name.removesuffix("s")] = normalized

    count = math.prod(len(values) for values in axes.values())
    if count > 256:
        fail(f"矩阵包含 {count} 个任务，超过 GitHub Actions 的 256 个任务限制，请缩小版本范围")

    for os_version, python_version, sdk_version, hggcrt_version, torch_version in itertools.product(axes["os_version"], axes["python_version"], axes["sdk_version"], axes["hggcrt_version"], axes["torch_version"]):
        key = f"{os_version}-py{python_version}-sdk{sdk_version}-hggcrt{hggcrt_version}-torch{torch_version}"
        image = IMAGES_MAP.get(key)
        if not isinstance(image, str) or not image or re.search(r"\s", image):
            fail(f"BUILD_CONTAINER_IMAGES 缺少 {key} 或镜像地址为空/包含空白字符")

    combinations = []
    for values in itertools.product(*axes.values()):
        entry = dict(zip(axes, values))
        key = f"{entry['os_version']}-py{entry['python_version']}-sdk{entry['sdk_version']}-hggcrt{entry['hggcrt_version']}-torch{entry['torch_version']}"
        entry["image"] = IMAGES_MAP.get(key)
        # entry["torch_url"] = get_torch_url(
        #     python_version=entry["python_version"],
        #     os_version=entry["os_version"],
        #     hggcrt_version=entry["hggcrt_version"],
        #     torch_version=entry["torch_version"],
        #     sdk_version=entry["sdk_version"],
        # )
        # entry["sdk_url"] = get_sdk_url(
        #     os_version=entry["os_version"],
        #     hggcrt_version=entry["hggcrt_version"],
        #     sdk_version=entry["sdk_version"],
        # )
        combinations.append(entry)

    matrix = json.dumps({"include": combinations}, separators=(",", ":"))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"matrix={matrix}\n")
    print(f"已生成 {count} 个矩阵任务")


if __name__ == "__main__":
    main()
