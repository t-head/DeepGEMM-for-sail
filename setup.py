import os
import setuptools
import shutil
import subprocess
import torch
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension, BuildExtension

current_dir = os.path.dirname(os.path.realpath(__file__))
# cxx_flags = ['-std=c++20', '-O3', '-fPIC', '-Wno-psabi', '-g']

sources = ['csrc/python_api.cu']
build_include_dirs = [
    f'{CUDA_HOME}/include',
    current_dir + '/deep_gemm/include',
    current_dir + '/third-party/cutlass3/',
    current_dir + '/third-party/cutlass3/include',
    current_dir + '/third-party/cutlass3/tools',
    current_dir + '/third-party/fmt/include',
    current_dir + '/third-party/cutlass3/include/cute',
]
build_libraries = ['cuda', 'cudart', 'nvrtc']
build_library_dirs = [
    f'{CUDA_HOME}/lib64',
    f'{CUDA_HOME}/lib64/stub'
]

third_party_include_dirs = [
    'third-party/cutlass/include/accutlass.h',
    'third-party/cutlass/include/cutlass',
    'third-party/cutlass/include/aiu',
    'third-party/cutlass3/include/cute',
    'third-party/cutlass3/include/cutlass',
    'third-party/cutlass3/include/ppu',
    'third-party/cutlass3/tools'
]



class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self.make_jit_include_symlinks()

    @staticmethod
    def make_jit_include_symlinks():
        # Make symbolic links of third-party include directories
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            cutlass_dirname = d.split('/')[1]
            src_dir = f'{current_dir}/{d}'
            dst_dir = f'{current_dir}/deep_gemm/include/{dirname}' if cutlass_dirname == 'cutlass' else  f'{current_dir}/deep_gemm/include/cutlass3/{dirname}'
            assert os.path.exists(src_dir)

            if os.path.exists(dst_dir):
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
            cutlass_dirname = d.split('/')[1]
            src_dir = os.path.join(current_dir, d)
            include_dir = build_include_dir
            if cutlass_dirname == 'cutlass3':
                include_dir = include_dir + '/cutlass3'
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

def custom_local_scheme(version):
    return 'dev%03d.%s' % (version.distance, version.short_node)
    
def custom_version_scheme(version):
    return '1.0.0'

if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = ''

    cuda_version = torch.version.cuda

    def parse_cuda_version(version_str):
        if version_str is None:
            return (0, 0)
        major, minor = map(int, version_str.split('.')[:2])
        return (major, minor)

    cuda_ver = parse_cuda_version(torch.version.cuda)
    should_build_ext = False
    if cuda_ver <= (12, 6):
        should_build_ext = True
    
    ext_modules = []
    if should_build_ext:
        ext_modules.append(
            CUDAExtension(name='deep_gemm.deep_gemm_cpp',
                         sources=sources,
                         include_dirs=build_include_dirs,
                         libraries=build_libraries,
                         library_dirs=build_library_dirs,
                         extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "--use_fast_math",
                    # 可选：指定 nvcc 使用的主机编译器（如 clang）
                    # "-ccbin", "/usr/bin/clang++-14",
                    # 指定架构（关键！）
                    "-gencode=arch=compute_80,code=sm_80",
                ],   # cxx_flags = ['-std=c++20', '-O3', '-fPIC', '-Wno-psabi', '-g']
            }))
    setuptools.setup(
        name='deep_gemm',
        use_scm_version={
            "local_scheme": custom_local_scheme,
            "version_scheme": custom_version_scheme,
        },
        setup_requires=["setuptools-scm==9.2.2"],
        packages=find_packages('.'), # old version: packages=['deep_gemm', 'deep_gemm/jit', 'deep_gemm/jit_kernels', 'deep_gemm/deep_gemm_tuner'],, 
        package_data={
            'deep_gemm': [
                'include/deep_gemm/**/*',
                'include/cutlass/**/*',
                'include/cutlass3/**/*',
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
