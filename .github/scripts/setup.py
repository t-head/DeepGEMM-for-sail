import os
import platform
import setuptools
import subprocess
import sys
import zipfile
import re
import json
import tempfile
import shutil
import importlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from packaging.version import parse
from packaging.version import Version
from packaging.requirements import Requirement, InvalidRequirement
from setuptools.command.install import install as _install
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

PACKAGE_NAME = ""
build_version = ""
edition_type = "oe"
USE_MANYLINUX = False
RUNTIME_MARK = "hggcrt"
framework = ""
BASE_WHEEL_URL = "https://download.t-head.cn/artifactory/pypi_generic/"
JSON_URL = "https://download.t-head.cn/artifactory/pypi_generic/"
STRATEGY_URL = "https://download.t-head.cn/artifactory/pypi_generic/strategy_hggc.py"
STRATEGY_SIG_URL = "https://download.t-head.cn/artifactory/pypi_generic/strategy.sig"
extra_info = {}
RED = "\033[91m"
RESET = "\033[0m"


@dataclass
class WheelResult:
    """
    用于封装 wheel 包查找结果的数据类。

    在整个构建流程中, WheelResult 作为统一的返回结构，承载了从远端仓库或本地策略
    匹配到的 wheel 包的关键信息，包括下载地址、文件名以及最终确定的构建版本号。

    Attributes:
        url: wheel 包的完整下载 URL。若未找到匹配的包则为 None。
        filename: wheel 包的文件名（如 "package-1.0.0+{RUNTIME_MARK}121ubuntu2204ce-cp310-cp310-linux_x86_64.whl"）。
                  若未找到匹配的包则为 None。
        build_version: 最终匹配到的构建版本号字符串（如 "1.0.0+v0.1.0.ppu2.0.0"）。
                       若未找到匹配的包则为 None。
    """
    url: Optional[str]
    filename: Optional[str]
    build_version: Optional[str]

    def __iter__(self):
        return iter((self.url, self.filename, self.build_version))


g_wheel_result: Optional[WheelResult] = None


def run_cmd(cmd: str, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    ret = subprocess.run(args=cmd, shell=True, stdout=stdout, stderr=stderr, encoding="utf-8")
    return ret


def get_os_name():
    """
    检测当前操作系统类型并返回标准化的操作系统名称字符串。

    通过读取 /etc/os-release 文件来判断当前系统类型。支持以下操作系统：
      - Ubuntu 20.04 -> "ubuntu2004"
      - Ubuntu 22.04 -> "ubuntu2204"
      - Ubuntu 24.04 -> "ubuntu2404"
      - AliOS -> "alios7u2"
    如果启用了 USE_MANYLINUX 全局标志，则在不支持的系统上回退为 "alios7u2"。
    其他不支持的系统返回 "any"。

    该名称用于拼接 wheel 文件名，确保安装包与目标操作系统匹配。

    Returns:
        str: 标准化的操作系统名称，如 "ubuntu2204"、"alios7u2" 或 "any"。
    """
    cmd = "cat /etc/os-release"
    ret = run_cmd(cmd)
    if ret.returncode != 0:
        print("Fail to get OS info, use alios7u2")
        return "alios7u2"
    else:
        output = ret.stdout
        if USE_MANYLINUX:
            print("This package is build in manylinux, use os alios7u2.")
            return "alios7u2"
        if "Ubuntu" in output and "20.04" in output:
            return "ubuntu2004"
        if "Ubuntu" in output and "22.04" in output:
            return "ubuntu2204"
        if "Ubuntu" in output and "24.04" in output:
            return "ubuntu2404"
        if "alios" in output:
            return "alios7u2"
        print("No exact match for the OS, use alios7u2")
        return "alios7u2"


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_framework_version():
    """
    获取当前环境中安装的深度学习框架的版本号。

    根据全局变量 framework 的值来确定需要检测的框架类型：
      - 若 framework 为空或 None, 返回空字符串(表示不依赖任何框架)。
      - 若 framework 为 "torch"，则导入 PyTorch 并提取其版本号。版本号会去除
        本地版本标识符（如 "+{RUNTIME_MARK}121"），只保留主版本.次版本.微版本格式。
      - 若 framework 为其他值，则尝试使用 importlib 动态导入该模块并读取其 __version__ 属性。

    框架版本号用于拼接 wheel 文件名，确保安装包与当前框架版本兼容。

    Returns:
        str: 框架版本号字符串（如 "2.1.0"），或在未指定框架时返回空字符串。

    Raises:
        SystemExit: 当 framework 为 "torch" 但导入 PyTorch 失败时，程序退出。
    """
    if framework == '' or framework is None:
        return ""
    if framework == 'torch':
        try:
            import torch
            torch_version_raw = parse(torch.__version__)
            try:
                torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}.{torch_version_raw.micro}"
            except AttributeError:
                torch_version = str(torch_version_raw).split("+")[0]
            return torch_version
        except Exception as e:
            print(f"Import torch failed: {e}")
            exit(1)
    else:
        try:
           import importlib
           __sail_framework = importlib.import_module(framework)
           return __sail_framework.__version__
        except Exception as e:
           print(f"Import framework {framework} failed: {e}")
           return ""


def get_runtime_version():
    """
    获取当前环境的 runtime 版本号。

    优先读取 HGGCRT_VERSION，未设置或格式非法时通过 hgcc 检测。

    Returns:
        str or None: runtime 版本号字符串，若无法获取则返回 None。
    """
    runtime_version = get_runtime_version_by_env()
    if runtime_version is not None:
        return runtime_version
    return get_runtime_version_by_hgcc()


def get_runtime_version_by_hgcc():
    """
    通过执行 hgcc --version 命令获取当前系统安装的 runtime 版本号。

    解析 hgcc 的输出信息，使用正则表达式匹配 "Runtime API version v2" 格式的版本号。
    例如输出 "Runtime API version v2" 时, 提取的版本号为 "2"。

    当环境变量未提供合法版本时，使用该方法检测；仅接受整数 API 版本。

    Returns:
        str or None: runtime 版本号字符串（如 "2"），若无法获取则返回 None。
    """
    try:
        result = subprocess.run(['hgcc', '--version'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True)
        output = os.linesep.join((result.stdout or "", result.stderr or ""))
        # 例如 Runtime API version v2，仅返回数字部分 2。
        for line in output.splitlines():
            match = re.fullmatch(r'\s*Runtime\s+API\s+version\s+v([0-9]+)\s*', line)
            if match:
                version = match.group(1)
                print(f"Get runtime version by hgcc --version: {output}")
                return version
        print(f"Miss match runtime version by hgcc --version: {output}")
        return None
    except Exception as e:
        print(f"Unexpected error for hgcc --version: {e}")
        return None


def get_runtime_version_by_env():
    """
    通过读取环境变量 HGGCRT_VERSION 获取 hggcrt runtime 版本号。
    环境变量 HGGCRT_VERSION 必须是 hggcrt 前缀加整数 API 版本，例如 "hggcrt3"。
    Returns:
        str or None: hggcrt 版本号字符串（如 "3"），若环境变量未设置或格式不符则返回 None。
    """
    try:
        # HGGCRT_VERSION=hggcrt3
        output = os.getenv('HGGCRT_VERSION')
        match = re.fullmatch(r'hggcrt([0-9]+)', output or "")
        if match:
            print(f"Get hggcrt version by env 'HGGCRT_VERSION': {output}")
            return match.group(1)
        else:
            print(f"Can not get env 'HGGCRT_VERSION': {output}. Example: HGGCRT_VERSION=hggcrt3")
            return None
    except Exception as e:
        print(f"Unexpected error for get env HGGCRT_VERSION: {e}")
        return None


def get_sdk_version_by_env():
    """
    通过读取环境变量 SAIL_PYPI_SDK_VESION 获取 SDK 版本号。

    该环境变量用于在无法通过其他途径自动检测 SDK 版本时，由用户手动指定。预期格式为纯版本号字符串，如 "1.7.0"。

    Returns:
        str or None: SDK 版本号字符串，若环境变量未设置则返回 None。
    """
    try:
        # SDK_VESION_FOR_PYPI=1.7.0
        output = os.getenv('SAIL_PYPI_SDK_VESION')
        if output is not None:
            print(f"Get sdk version by env 'SAIL_PYPI_SDK_VESION': {output}")
            return output
        else:
            print(f"Can not get env 'SAIL_PYPI_SDK_VESION': {output}. Example: SAIL_PYPI_SDK_VESION=1.7.0")
            return None
    except Exception as e:
        print(f"Unexpected error for get env SAIL_PYPI_SDK_VESION: {e}")
        return None


def load_plugin(file_path):
    """
    从指定文件路径动态加载一个 Python 模块作为插件。

    使用 importlib 的底层 API,将任意路径的 .py 文件作为名为 "dynamic_plugin"
    的模块加载到内存中。该机制用于在运行时从远端下载的策略文件 (strategy.py)
    中加载匹配逻辑，实现策略的动态更新而无需修改本脚本。

    Args:
        file_path: 要加载的 Python 文件的绝对路径。

    Returns:
        module or None: 成功加载后返回模块对象，可通过 module.run() 等方式调用
                        其中的函数。加载失败时返回 None。
    """
    try:
        spec = importlib.util.spec_from_file_location("dynamic_plugin", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Failed to load plugin from {file_path}: {e}")
        return None
    return module


def get_wheel_url():
    """
    根据当前环境信息(Python 版本、平台、操作系统、RUNTIME 版本、框架版本等)
    确定目标 wheel 包的下载 URL 和文件名。

    该函数是整个构建流程的核心入口，负责协调环境检测和版本匹配。处理流程如下：

    1. 收集当前环境信息: Python 版本 (cpXY)、平台标识、操作系统名称、
       包版本号、框架版本和 RUNTIME 版本。
    2. 检查环境变量 SAIL_SDK_MATCH_STRATEGY:
       - 若为 "off"，使用旧版本地配置匹配逻辑 (get_wheel_url_old),
         直接根据环境参数拼接 wheel URL。
       - 否则（默认），从远端下载 strategy.py 策略脚本，动态加载并执行
         其 run() 方法来进行智能版本匹配。策略脚本会下载 index.json
         版本索引文件，综合考虑 SDK 版本兼容性选择最佳匹配。
    3. 将匹配结果存储到全局变量 g_wheel_result 中。
    4. 若未找到匹配的包，输出详细的环境信息错误提示并退出程序。

    Returns:
        tuple: (wheel_url, wheel_filename) 元组, 分别为 wheel 包的完整下载
               URL 和文件名。

    Raises:
        SystemExit: 当无法找到与当前环境兼容的安装包时, 程序退出。

    Side Effects:
        修改全局变量 g_wheel_result, 存储完整的匹配结果。
    """
    python_version = None
    platform_name = None
    os_name = None
    package_version = None
    framework_version = None
    runtime_version = None
    strategy = None
    global g_wheel_result
    try:
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_name = get_platform()
        os_name = get_os_name()
        package_version = build_version.split("+")[0]
        framework_version = get_framework_version()
        runtime_version = get_cuda_version()
        strategy = os.getenv('SAIL_SDK_MATCH_STRATEGY')
        if strategy and strategy.lower() == "off":
            wheel_url, wheel_filename = get_wheel_url_old(
                python_version=python_version,
                platform_name=platform_name,
                os_name=os_name,
                package_version=package_version,
                framework_version=framework_version,
                runtime_version=runtime_version
                )
            g_wheel_result = WheelResult(url=wheel_url, filename=wheel_filename, build_version=build_version)
        else:
            with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
                strategy_file_path = os.path.join(temp_dir, "strategy.py")
                download_file(STRATEGY_URL, "strategy.py", temp_file_path=temp_dir)
                strategy_module = load_plugin(strategy_file_path)
                data = {
                    "base_wheel_url": BASE_WHEEL_URL,
                    "base_json_url": JSON_URL,
                    "build_version": build_version,
                    "edition_type": edition_type,
                    "framework": framework,
                    "package_name": PACKAGE_NAME,
                    "runtime_mark": RUNTIME_MARK,
                    "runtime_version": runtime_version
                }
                if strategy_module is None or not hasattr(strategy_module, "run"):
                    print(f"Can not access strategy.py")
                    g_wheel_result = WheelResult(url=None, filename=None, build_version=None)
                else:
                    wheel_url, wheel_filename, best_matching_build_version = strategy_module.run(data, globals())
                    g_wheel_result = WheelResult(url=wheel_url, filename=wheel_filename, build_version=best_matching_build_version)
                    print(f"Strategy return: {wheel_url} {wheel_filename} {best_matching_build_version}")
    except Exception as e:
        g_wheel_result = WheelResult(url=None, filename=None, build_version=None)
        print(f"{e}")
        print(f"{RED}Strategy:{strategy}. No installation package compatible with the current environment{RESET}")
        print(f"{RED}python:{python_version}, platform:{platform_name}, os:{os_name}, runtime_version:{runtime_version}{RESET}")
        print(f"{RED}version:{package_version}, framework:{framework}{framework_version}, build version:{build_version}{RESET}")
        print(f"{RED}Contact PTG to release the current configuration package if required.{RESET}")
        exit(1)
    if g_wheel_result.url is None or g_wheel_result.filename is None or g_wheel_result.url == "" or g_wheel_result.filename == "":
        print(f"{RED}No installation package compatible with the current environment{RESET}")
        print(f"{RED}python:{python_version}, platform:{platform_name}, os:{os_name}, runtime_version:{runtime_version}{RESET}")
        print(f"{RED}version:{package_version}, framework:{framework}{framework_version}, build version:{build_version}{RESET}")
        print(f"{RED}Contact PTG to release the current configuration package if required.{RESET}")
        exit(1)
    return g_wheel_result.url, g_wheel_result.filename


def get_wheel_url_old(python_version, platform_name, os_name, package_version, framework_version, runtime_version):
    """
    使用本地配置方式拼接 wheel 包的下载 URL 和文件名（不依赖远端策略）。

    当环境变量 SAIL_SDK_MATCH_STRATEGY 设置为 "off" 时，跳过远端策略匹配,
    直接根据传入的环境参数按固定规则拼接 wheel 文件名和 URL。

    文件名格式遵循以下规则：
      - 有框架依赖时：{包名}-{版本}+{RUNTIME_MARK}{RUNTIME}{框架}{框架版本}{OS}{版本类型}-{Python}-{Python}-{平台}.whl
      - 无框架依赖时：{包名}-{版本}+{RUNTIME_MARK}{RUNTIME}{OS}{版本类型}-{Python}-{Python}-{平台}.whl

    Args:
        python_version: Python 版本标识（如 "cp310"）。
        platform_name: 平台标识（如 "linux_x86_64"）。
        os_name: 操作系统名称（如 "ubuntu2204"）。
        package_version: 包版本号（不含 build 后缀，如 "1.0.0"）。
        framework_version: 框架版本号（如 "2.1.0"），为空表示无框架依赖。
        runtime_version: RUNTIME 版本号（如 CUDA 的 "121" 或 HGGCRT 的 "3"）。

    Returns:
        tuple: (wheel_url, wheel_filename) 元组。

    Side Effects:
        修改全局变量 g_wheel_result。
    """
    global g_wheel_result
    print("Get whl url with local config.")
    if framework != "" and framework_version is not None:
        wheel_filename = f"{PACKAGE_NAME}-{package_version}+{RUNTIME_MARK}{runtime_version}{framework}{framework_version}{os_name}{edition_type}-{python_version}-{python_version}-{platform_name}.whl"
        wheel_url = f"{BASE_WHEEL_URL}/{PACKAGE_NAME}/{build_version}/{wheel_filename}"
        print(f"Whl with framework is wheel_url: {wheel_url}")
        g_wheel_result = WheelResult(url=wheel_url, filename=wheel_filename, build_version=build_version)
        return wheel_url, wheel_filename
    else:
        wheel_filename = f"{PACKAGE_NAME}-{package_version}+{RUNTIME_MARK}{runtime_version}{os_name}{edition_type}-{python_version}-{python_version}-{platform_name}.whl"
        wheel_url = f"{BASE_WHEEL_URL}/{PACKAGE_NAME}/{build_version}/{wheel_filename}"
        print(f"Whl is wheel_url: {wheel_url}")
        g_wheel_result = WheelResult(url=wheel_url, filename=wheel_filename, build_version=build_version)
        return wheel_url, wheel_filename


def download_file(file_url, filename, temp_file_path=None):
    """
    该函数使用 wget 命令下载文件。主要用于从制品仓库(Artifactory)下载 wheel 包或其他配置文件。

    Args:
        file_url: 要下载的文件的完整 URL。
        filename: 下载后保存的文件名。
        temp_file_path: 可选的临时目录路径。若提供，文件将保存到该目录下。

    Returns:
        str: 下载文件的本地绝对路径。

    Raises:
        SystemExit: 当下载失败且不是临时文件下载时，程序退出。
    """
    file_path = os.path.abspath(filename)
    if temp_file_path is not None:
        file_path = os.path.join(temp_file_path, filename)
    else:
        print(f"Guessing wheel URL: {file_url}")
    output = []
    try:
        # Attempt to download the wheel package.
        cmd = f"wget -O {file_path} {file_url}"
        ret = run_cmd(cmd)
        output = ret.stdout.splitlines() + ret.stderr.splitlines()
        if ret.returncode == 0:
            if os.path.exists(file_path):
                return file_path
            else:
                print("File not found: " + str(file_path))
                print("Unknown Error! Please check environment or contact PTG IT Team.")
                for line in output:
                    print(line)
                exit(1)
        else:
            print(f"Attempt to download the file: {file_url} failed.")
            for line in output:
                print(line)
            if temp_file_path is not None:
                return file_path
            exit(1)
    except Exception as e:
        print(f"An error occurred in download_file: {e}")
        exit(-1)


def extract_metadata_from_wheel(wheel_file):
    """
    从 wheel 包中提取 METADATA 文件的内容。

    Wheel 包本质上是一个 ZIP 压缩文件，其中包含一个 METADATA 文件，
    记录了包的元数据信息，包括名称、版本、依赖关系等。该函数打开 wheel
    包，定位并读取 METADATA 文件的内容。

    Args:
        wheel_file: wheel 包文件的本地路径。

    Returns:
        str: METADATA 文件的完整内容。若未找到 METADATA 文件或发生错误，
             返回空字符串。
    """
    try:
        with zipfile.ZipFile(wheel_file, 'r') as zip_ref:
            metadata_filename = next((name for name in zip_ref.namelist() if name.endswith('METADATA')), None)
            if not metadata_filename:
                print("METADATA file not found in the wheel package.")
                return ""
            with zip_ref.open(metadata_filename) as metadata_file:
                metadata_content = metadata_file.read().decode('utf-8')
        return metadata_content
    except Exception as e:
        print(f"Error for extract wheel metadata: {e}")
        return ""


def marker_operand_to_str(node):
    """
    将 Marker 三元组中的单个操作数（Variable/Op/Value）还原为 PEP 508 片段。

    是否需要加引号由节点类型决定, 不能按位置判断: PEP 508 允许字面量出现在
    左侧（如 "arm" not in platform_machine）, 此时三元组为 (Value, Op, Variable)。
    packaging 的节点自带 serialize()，Variable 输出变量名、Value 输出带引号的
    字符串字面量，直接复用可避免还原出非法 marker。

    Args:
        node: Marker._markers 三元组中的 Variable、Op 或 Value 节点。

    Returns:
        str: 还原后的 PEP 508 片段。
    """
    serialize = getattr(node, "serialize", None)
    if callable(serialize):
        return serialize()
    return str(node)


def marker_node_to_str(node):
    """
    将 Marker 表达式树的单个节点还原为 PEP 508 条件字符串。

    节点可能是 (Variable, Op, Value) 三元组（如 python_version < "3.10"）,
    或括号分组对应的嵌套列表，嵌套列表会递归还原并加括号。

    Args:
        node: Marker._markers 中的单个节点。

    Returns:
        str: 还原后的条件字符串。
    """
    if isinstance(node, tuple) and len(node) == 3:
        return " ".join(marker_operand_to_str(item) for item in node)
    if isinstance(node, list):
        return "(" + " ".join(
            item if isinstance(item, str) else marker_node_to_str(item)
            for item in node
        ) + ")"
    return str(node)


def split_extra_marker(marker):
    """
    从 PEP 508 环境标记 (Marker) 中分离 extra 名称与剩余环境条件。

    Marker 内部的 _markers 是一棵表达式树, 由 (Variable, Op, Value) 三元组、
    "and"/"or" 连接符及括号分组的嵌套列表组成, 该结构自 packaging 16.1
    以来保持稳定。

    标准打包工具生成的 METADATA 中, extra 子句总是位于顶层且由 and
    连接（如 python_version < "3.11" and extra == 'dev'），此时可安全拆分；
    顶层含 or 的复杂组合保守地整体视为非 extra 条件，不做拆分。

    Args:
        marker: packaging.markers.Marker 对象。

    Returns:
        tuple: (extra_name, residual_condition) 元组。
               extra_name 为 extra 名称（如 "dev"）, 不含 extra 子句时为 None;
               residual_condition 为剩余条件字符串, 无剩余条件时为 None。
    """
    nodes = marker._markers
    if any(node == "or" for node in nodes):
        return None, str(marker)
    extra_name = None
    residual_parts = []
    for node in nodes:
        if (isinstance(node, tuple) and len(node) == 3
                and str(node[0]) == "extra" and str(node[1]) == "=="):
            extra_name = str(node[2])
        elif node == "and":
            continue
        else:
            residual_parts.append(marker_node_to_str(node))
    residual_condition = " and ".join(residual_parts) if residual_parts else None
    return extra_name, residual_condition


def parse_requires_dist(metadata_content):
    """
    解析 METADATA 文件中的 Requires-Dist 字段，提取依赖关系。

    METADATA 文件中的 Requires-Dist 字段列出了包的所有依赖项。
    每行先用 packaging.requirements.Requirement 做 PEP 508 合法性校验
    （非法行告警并跳过，避免脏数据传入 setup() 导致构建失败）,
    再根据环境标记 (marker) 中是否含 extra 子句对依赖项分类：
      - 无标记或仅含普通环境标记（如 platform_machine、python_version）：
        添加到 install_requires 列表，条件原样保留，交由 pip 安装时求值。
      - 含 extra 子句：添加到 extras_require 字典按 extra 名称分组，
        与 extra 组合的其余条件（如 python_version < "3.11"）会保留在依赖项中。

    依赖文本保留 METADATA 原样（不用 Requirement 重建），避免 SpecifierSet
    规范化时重排版本约束顺序。

    METADATA 为 RFC 822 格式: 首个空行之前是头部字段, 之后是包描述正文
    (long_description)。只解析头部字段区域且要求字段名从行首开始, 避免把正文里
    出现的 "Requires-Dist: xxx" 字样（如 README 中的示例代码）误当成真实依赖。

    Args:
        metadata_content: METADATA 文件的完整内容字符串。

    Returns:
        tuple: (install_requires, extras_require) 元组。
               install_requires 是依赖列表（可能带环境标记）。
               extras_require 是按 extra 名称分组的条件依赖字典。
    """
    try:
        install_requires = []
        extras_require = {}
        field_prefix = "Requires-Dist:"
        for raw_line in metadata_content.splitlines():
            # 首个空行之后是描述正文, 遇空行即停止解析字段。
            # 旧式 METADATA 折叠在 Description 字段里的正文, 其空行会被编码为
            # 8 个空格而非真正的空行, 因此不会在此提前截断头部字段。
            if raw_line == "":
                break
            # 必须从行首开始匹配: 行内出现的字样不是字段, 缩进行在 RFC 822
            # 中属于上一字段的续行, 同样不是新的 Requires-Dist 字段。
            if not raw_line.startswith(field_prefix):
                continue
            line = raw_line[len(field_prefix):].strip()
            if not line:
                continue
            # 用 packaging.requirements 对整行做 PEP 508 校验与 marker 解析
            try:
                requirement = Requirement(line)
            except InvalidRequirement as req_err:
                print(f"Skip invalid Requires-Dist line '{line}': {req_err}")
                continue
            dependency = line.split(';', 1)[0].strip()
            if requirement.marker is None:
                install_requires.append(dependency)
                continue
            extra_name, residual_condition = split_extra_marker(requirement.marker)
            if extra_name is not None:
                if extra_name not in extras_require:
                    extras_require[extra_name] = []
                # extras_require 的值同样支持环境标记, 保留与 extra
                # 组合的剩余条件（如 python_version < "3.11"）
                if residual_condition:
                    extras_require[extra_name].append(f"{dependency}; {residual_condition}")
                else:
                    extras_require[extra_name].append(dependency)
            else:
                # 非 extra 的环境标记（如 platform_machine、python_version 等），
                # 原样保留条件，交由 pip 在安装时根据当前环境求值。
                condition = line.split(';', 1)[1].strip()
                install_requires.append(f"{dependency}; {condition}")
        return install_requires, extras_require
    except Exception as e:
        print(f"Error for parse requires dist: {e}")
        return [], {}


class DisableInstallCommand(_install):
    """
    自定义的 install 命令类，用于禁用从 PyPI 源的直接安装。

    该类继承自 setuptools 的 install 命令，重写了 run 方法。
    当用户尝试通过 pip install 直接安装此包时，会触发该命令，
    输出提示信息并退出。这种设计是为了防止用户从 PyPI 安装不兼容的预编译包，
    确保用户使用正确的安装方式。
    """
    def run(self):
        print("Custom installation from this pypi source is disabled.")
        print("PLEASE try cloning from github and run python setup.py install.")
        exit(-1)


class CloneInstallCommand(_install):
    """
    自定义的 install 命令类，用于禁用克隆安装方式。

    该类继承自 setuptools 的 install 命令，重写了 run 方法。
    当前实现直接退出，表示克隆安装功能已被禁用。
    """
    def run(self):
        print("Clone from github is disabled.")
        exit(-1)


class CachedWheelsCommand(_bdist_wheel):
    """
    自定义的 bdist_wheel 命令类，用于使用预编译的 wheel 包。

    该类继承自 wheel 的 bdist_wheel 命令，重写了 run 方法。
    当 pip 无法找到现有的 wheel 包时，会触发该命令。
    本实现不执行实际的编译构建流程，而是将预先下载的 wheel 包重命名为 pip 期望的文件名，从而绕过构建过程。

    实现使用预编译的二进制包，避免在用户机器上进行耗时的编译，同时确保包与当前环境完全兼容。
    """
    def run(self):
        """
        执行 wheel 构建命令时被调用。

        将全局变量 g_wheel_result 中存储的预下载 wheel 包重命名为 pip 期望的标准文件名，并保存到 dist 目录。

        Raises:
            SystemExit: 当重命名操作失败时，程序退出。
        """
        try:
            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            if g_wheel_result and g_wheel_result.filename:
                os.rename(g_wheel_result.filename, wheel_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            exit(-1)


if __name__ == "__main__":
    install_requires, extras_require = [], {}
    extra_args = {}
    if os.getenv("__FAKE_BUILD") != "FAKE_WHEEL":
        wheel_url, wheel_filename = get_wheel_url()
        best_matching_build_version = g_wheel_result.build_version if g_wheel_result else None
        if best_matching_build_version == build_version:
            file_path = download_file(wheel_url, wheel_filename)
            metadata = extract_metadata_from_wheel(file_path)
            install_requires, extras_require = parse_requires_dist(metadata)
        else:
            if best_matching_build_version != "" and best_matching_build_version is not None:
                extra_args["python_requires"] = ">9.9"
    else:
        extra_args["python_requires"] = ">=3.8"
    install_requires += ['numpy', 'packaging']

    setuptools.setup(
        name=PACKAGE_NAME,
        description=PACKAGE_NAME,
        version=build_version,
        install_requires=install_requires,
        extras_require=extras_require,
        **extra_args,
        packages=setuptools.find_packages(exclude=("tests*", "benchmarks*")),
        cmdclass={
            "bdist_wheel": CachedWheelsCommand,
            "install": DisableInstallCommand if "PTG_Internal" not in os.environ.keys() else CloneInstallCommand
        },
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        zip_safe=False,
    )
