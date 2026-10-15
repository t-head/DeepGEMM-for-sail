import fcntl
import hashlib
import functools
import os
import re
import subprocess
import uuid
from torch.utils.cpp_extension import CUDA_HOME
from typing import Tuple

from .runtime import Runtime, RuntimeCache
from .template import typename_map

runtime_cache = RuntimeCache()

_jit_include_dir = f'{os.path.dirname(os.path.abspath(__file__))}/../include'


def compute_text_hash(text: str) -> str:
    md5 = hashlib.md5()
    md5.update(text.encode('utf-8'))
    return md5.hexdigest()[0:12]


def get_jit_include_dir() -> str:
    """The one include root the JIT resolves megamoe headers from."""
    return _jit_include_dir


_include_re = re.compile(r'^\s*#\s*include\s*[<"](?:\.\./)*(megamoe/[^">]+)[">]', re.MULTILINE)


def _resolve_header_path(rel_path: str) -> str:
    return f'{_jit_include_dir}/{rel_path}'


def _collect_included_megamoe_headers(code: str) -> tuple:
    seen = set()
    pending = list(_include_re.findall(code))
    while pending:
        rel_path = pending.pop()
        if rel_path in seen:
            continue
        seen.add(rel_path)
        with open(_resolve_header_path(rel_path), 'r') as f:
            pending.extend(_include_re.findall(f.read()))
    return tuple(sorted(seen))


@functools.lru_cache(maxsize=None)
def get_megamoe_headers_digest(code: str) -> str:
    md5 = hashlib.md5()
    for rel_path in _collect_included_megamoe_headers(code):
        md5.update(rel_path.encode('utf-8'))
        with open(_resolve_header_path(rel_path), 'rb') as f:
            md5.update(f.read())

    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def get_nvcc_path_and_version() -> Tuple[str, str]:
    paths = []
    if os.getenv('DG_NVCC_COMPILER'):
        paths.append(os.getenv('DG_NVCC_COMPILER'))
    paths.append(f'{CUDA_HOME}/bin/nvcc')

    minimum_version = '11.6'
    version_pattern = re.compile(r'release (\d+\.\d+)')
    for path in paths:
        if os.path.exists(path):
            match = version_pattern.search(os.popen(f'{path} --version').read())
            assert match, f'Cannot get the version of NVCC compiler {path}'
            version = match.group(1)
            version_tuple = tuple(int(x) for x in version.split('.'))
            required_version_tuple = tuple(int(x) for x in minimum_version.split('.'))
            assert version_tuple >= required_version_tuple, f'NVCC {path} version {version} is lower than {minimum_version}'
            return path, version
    raise RuntimeError('Cannot find any available NVCC compiler')


@functools.lru_cache(maxsize=None)
def get_jit_root_dir():
    if 'DG_JIT_CACHE_DIR' in os.environ:
        path = os.getenv('DG_JIT_CACHE_DIR')
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.expanduser('~') + '/.megamoe'


@functools.lru_cache(maxsize=None)
def get_tmp_dir():
    return f'{get_jit_root_dir()}/tmp'


@functools.lru_cache(maxsize=None)
def get_cache_dir():
    return f'{get_jit_root_dir()}/cache'


def ensure_tmp_dir():
    tmp_dir = get_tmp_dir()
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def write_file(path, data, is_binary=False):
    tmp_file_path = f'{ensure_tmp_dir()}/file.tmp.{str(uuid.uuid4())}.{compute_text_hash(path)}'
    with open(tmp_file_path, 'wb' if is_binary else 'w') as f:
        f.write(data)
    os.replace(tmp_file_path, path)


def build_runtime(name: str, arg_defs: tuple, code: str) -> Runtime:
    cpp_standard = int(os.getenv('DG_JIT_CPP_STANDARD') or 17)
    nvcc_flags = [
        f'-std=c++{cpp_standard}',
        '-shared',
        '-O3',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-gencode=arch=compute_89,code=sm_89',
        f'--ptxas-options=--register-usage-level={os.environ.get("DG_REGISTER_LEVEL", "10")}',
        '--diag-suppress=39,174,177,940',
        '-Xptxas',
        '-v',
        # PPU codegen switches must go through nvcc's -mllvm passthrough; the bare -ppu-* spelling is
        # only accepted by an hgcc-fronted nvcc and is rejected by the CUDA-compatible one.
        '-mllvm',
        '-ppu-simt-branch=false',
        '-mllvm',
        '-ppu-patch-fence-ppu=false',
        '-mllvm',
        '-wno-loop-miss-transform',
        '-mllvm',
        '-ppu-cg-to-kp1=true',
        '-mllvm',
        '-ppu-fix-uninit=true'
    ]

    # -fconcepts is deliberately absent: with the CUDA-compatible toolchain's libstdc++ it sends the
    # host parse down a C++20 path that the clang-13-based front end cannot read.
    cxx_flags = ['-fPIC', '-O3', '-Wno-deprecated-declarations', '-Wno-abi']
    # Dev-only escape hatch: extra nvcc/backend flags from the environment, so scheduling knobs
    # (DG_REGISTER_LEVEL, the -ppu-* switches, -O levels, -mllvm -align-all-*) can be swept
    # without editing this file. ';' separates flags (the run harness passes one argv token).
    # Empty by default, so the flag list -- and the JIT cache signature -- is unchanged.
    extra_nvcc_flags = os.environ.get('DG_NVCC_EXTRA_FLAGS', '').replace(';', ' ').split()
    flags = [*nvcc_flags, *extra_nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
    include_dirs = [get_jit_include_dir()]
    # The include-directory paths are deliberately absent from the signature: the header
    # contents already enter through get_megamoe_headers_digest, so the same tree compiles to the
    # same cache entry regardless of the checkout path it lives in.
    signature = f'{name}$${get_megamoe_headers_digest(code)}$${code}$${get_nvcc_path_and_version()}$${flags}'
    runtime_name = f'kernel.{name}.{compute_text_hash(signature)}'
    path = f'{get_cache_dir()}/{runtime_name}'

    disable_cache_env = os.environ.get("DG_JIT_DISABLE_CACHE")
    if (runtime_cache[path] is not None) and (disable_cache_env is None or disable_cache_env == '0'):
        print(f'Using cached JIT runtime {path} {runtime_name} during build')
        return runtime_cache[path]

    os.makedirs(path, exist_ok=True)
    args_path = f'{path}/kernel.args'
    src_path = f'{path}/kernel.cu'
    write_file(args_path, repr([(arg_def[0], typename_map[arg_def[1]]) for arg_def in arg_defs]))
    write_file(src_path, code)

    so_path = f'{path}/kernel.so'
    tmp_so_path = f'{ensure_tmp_dir()}/nvcc.tmp.{str(uuid.uuid4())}.{compute_text_hash(so_path)}.so'

    nvcc_command = [get_nvcc_path_and_version()[0], src_path, '-o', tmp_so_path, *flags, *[f'-I{d}' for d in include_dirs]]

    if os.getenv('DG_JIT_DEBUG', None):
        print(f'Compiling JIT runtime {runtime_name} with command {nvcc_command}')

    with open(f'{path}.lock', 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if (disable_cache_env is None or disable_cache_env == '0') and runtime_cache[path] is not None:
            print(f'Using cached JIT runtime {path} {runtime_name} during build (built concurrently by another process)')
            return runtime_cache[path]
        return_code = subprocess.check_call(nvcc_command)
        assert return_code == 0, f'Failed to compile {src_path}'

        os.replace(tmp_so_path, so_path)
        write_file(f'{path}/.committed', '')

    runtime_cache[path] = Runtime(path)
    return runtime_cache[path]
