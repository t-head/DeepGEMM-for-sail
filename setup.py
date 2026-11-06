import os
import setuptools
import shutil
import subprocess
import torch
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension


def get_ppu_sdk_version():
    # Query version macros using the compiler belonging to the selected SDK.
    try:
        compiler = os.path.join(ppu_sdk, 'bin', 'hgcc')
        output = subprocess.check_output(
            [compiler, '-E', '-x', 'cu', '/dev/stdin', '-o', '/dev/stdout'],
            input=(
                'PPU_VERSION __HGGCCC_VER_MAJOR__ '
                '__HGGCCC_VER_MINOR__ __HGGCCC_VER_BUILD__\n'
            ),
            text=True,
            timeout=10,
        )
        major, minor, build = map(int, output.rsplit('PPU_VERSION', 1)[1].split())
        return (major, minor, build)
    except Exception:
        pass
    return (0, 0, 0)


current_dir = os.path.dirname(os.path.realpath(__file__))

# SDK root: env PPU_SDK > env PPU_HOME > default, then the compiler on PATH.
ppu_sdk = os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK'
if not os.path.isfile(os.path.join(ppu_sdk, 'bin', 'hgcc')):
    compiler = shutil.which('hgcc')
    if compiler:
        ppu_sdk = os.path.dirname(os.path.dirname(os.path.realpath(compiler)))
ppu_include = os.path.join(ppu_sdk, 'targets', 'x86_64-linux', 'include')

sources = ['csrc/python_api.cpp']
build_include_dirs = [path for path in (
    ppu_include,
    os.path.join(ppu_sdk, 'include'),
    os.path.join(ppu_sdk, 'include', 'cccl'),
) if os.path.isdir(path)] + [
    current_dir + '/deep_gemm/include',
    current_dir + '/third-party/actlize_v1.0.0/',
    current_dir + '/third-party/actlize_v1.0.0/include',
    current_dir + '/third-party/actlize_v1.0.0/tools',
    current_dir + '/third-party/fmt/include',
    current_dir + '/third-party/actlize_v1.0.0/include/cute',
]

build_libraries = ['hggc', 'hggcrt1', 'hgrtc', 'acblasLt']
# Search real libraries before the driver stubs used only at link time.
build_library_dirs = [path for path in (
    os.path.join(ppu_sdk, 'lib'),
    os.path.join(ppu_sdk, 'lib64'),
    os.path.join(ppu_sdk, 'lib64', 'stubs'),
) if os.path.isdir(path)]

third_party_include_dirs = [
    'third-party/actlize_v1.0.0/include/cute',
    'third-party/actlize_v1.0.0/include/cutlass',
    'third-party/actlize_v1.0.0/include/ppu_include.hpp',
    'third-party/actlize_v1.0.0/include/accutlass.hpp',
    'third-party/actlize_v1.0.0/tools'
]

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self.make_jit_include_symlinks()

    @staticmethod
    def make_jit_include_symlinks():
        # Make symbolic links of third-party include directories, one subdirectory per actlize
        # library so that their colliding headers do not shadow each other
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            actlize_dirname = d.split('/')[1]
            src_dir = f'{current_dir}/{d}'
            dst_parent = f'{current_dir}/deep_gemm/include/{actlize_dirname}'
            dst_dir = f'{dst_parent}/{dirname}'
            assert os.path.exists(src_dir)
            os.makedirs(dst_parent, exist_ok=True)

            if os.path.islink(dst_dir) or os.path.exists(dst_dir):
                assert os.path.islink(dst_dir)
                os.unlink(dst_dir)

            os.symlink(src_dir, dst_dir, target_is_directory=True)


class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Second, make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Finally, run the regular build
        build_py.run(self)


    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in ('DG_JIT_CACHE_DIR', 'DG_JIT_PRINT_COMPILER_COMMAND', 'DG_JIT_DISABLE_SHORTCUT_CACHE'):
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        with open(os.path.join(self.build_lib, 'deep_gemm', 'envs.py'), 'w') as f:
            f.write(code)


    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(self.build_lib, 'deep_gemm/include')
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy third-party includes to the build directory
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            actlize_dirname = d.split('/')[1]
            src_dir = os.path.join(current_dir, d)
            include_dir = os.path.join(build_include_dir, actlize_dirname)
            os.makedirs(include_dir, exist_ok=True)
            dst_dir = os.path.join(include_dir, dirname)

            # Remove existing directory if it exists

            # Copy the directory
            if os.path.isfile(src_dir):
                if os.path.exists(dst_dir):
                    os.remove(dst_dir)
                shutil.copy(src_dir, dst_dir)
            else:
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)

DEEPGEMM_VERSION = '1.1.0'  # DeepGEMM version
RELEASE_VERSION = 'v1.0.0'  # Release version


def get_package_version(ppu_sdk_version):
    try:
        from setuptools_scm import Configuration
        from setuptools_scm.git import parse
    except ImportError as error:
        raise RuntimeError('Missing build dependency: install setuptools-scm first') from error

    release_tag = f'{DEEPGEMM_VERSION}+{RELEASE_VERSION}'
    # Match only this release tag; ignore uncommitted changes.
    scm = parse(current_dir, Configuration(), describe_command=[
        'git', 'describe', '--tags', '--long', '--match', release_tag,
    ])
    if scm is None or scm.node is None:
        raise RuntimeError('Git commit information unavailable; build from a Git repository with commit history')
    if ppu_sdk_version == (0, 0, 0):
        raise RuntimeError('PPU SDK version unavailable; check hgcc --version for the selected SDK')

    ppu_version = '.'.join(map(str, ppu_sdk_version))
    local_version = f'{RELEASE_VERSION}.ppu{ppu_version}'
    if scm.distance in (None, 0):
        return f'{DEEPGEMM_VERSION}+{local_version}'
    return f'{DEEPGEMM_VERSION}.dev{scm.distance}+{local_version}.{scm.node[:9]}'


if __name__ == '__main__':
    extra_cxx_args = ["-O3", "-std=c++17", "-DUSE_HGGC"]
    extra_hgcc_args = ["-O3", "-std=c++17", "--use_fast_math", "-DUSE_HGGC"]

    ppu_sdk_version = get_ppu_sdk_version()
    package_version = get_package_version(ppu_sdk_version)
    if ppu_sdk_version >= (2, 0, 0):
        extra_cxx_args.append("-DDG_HGGC_SUPPORT_PCH=1")
        extra_hgcc_args.append("-DDG_HGGC_SUPPORT_PCH=1")

    ext_modules = []
    ext_modules.append(
        CppExtension(name='deep_gemm._C',
                        sources=sources,
                        include_dirs=build_include_dirs,
                        libraries=build_libraries,
                        library_dirs=build_library_dirs,
                        extra_compile_args={
            "cxx": extra_cxx_args,
            "hgcc": extra_hgcc_args,
        }))
    setuptools.setup(
        name='deep_gemm',
        version=package_version,
        setup_requires=['setuptools-scm'],
        packages=find_packages('.'),
        package_data={
            'deep_gemm': [
                'include/deep_gemm/**/*',
                'include/actlize_v1.0.0/**/*',
                'deep_gemm_tuner/configs/*',
            ]
        },
        ext_modules=ext_modules,
        zip_safe=False,
        cmdclass={
            'develop': PostDevelopCommand,
            'build_py': CustomBuildPy,
            'build_ext': BuildExtension,
        },
    )
