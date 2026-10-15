import ast
import ctypes
import os
import torch
from typing import Optional

from .template import to_ctype_arg, type_by_name


def read_kernel_args(path: str):
    """Resolve the kernel signature stored next to `kernel.so`.

    The file holds `[(name, type_name), ...]` with `type_name` a string from
    `typename_map` (e.g. `'torch.int'`), so it contains only string
    literals and is decoded without `eval`.
    """
    with open(os.path.join(path, 'kernel.args'), 'r') as f:
        return [(name, type_by_name[type_name]) for name, type_name in ast.literal_eval(f.read())]


class Runtime:

    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = None
        self.arg_specs = None

        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        required_files = ['kernel.cu', 'kernel.args', 'kernel.so', '.committed']
        if not all(os.path.exists(os.path.join(path, file)) for file in required_files):
            return False

        try:
            read_kernel_args(path)
        except (SyntaxError, ValueError, KeyError, TypeError):
            return False
        return True

    def __call__(self, *args) -> int:
        if self.lib is None or self.arg_specs is None:
            self.lib = ctypes.CDLL(os.path.join(self.path, 'kernel.so'))
            self.arg_specs = read_kernel_args(self.path)

        assert len(args) == len(self.arg_specs), f'Expected {len(self.arg_specs)} arguments, got {len(args)}'
        ctype_args = []
        for arg, (name, dtype) in zip(args, self.arg_specs):
            if isinstance(arg, torch.Tensor):
                assert arg.dtype == dtype, f'Expected tensor dtype `{dtype}` for `{name}`, got `{arg.dtype}`'
            else:
                assert isinstance(arg, dtype), f'Expected built-in type `{dtype}` for `{name}`, got `{type(arg)}`'
            ctype_args.append(to_ctype_arg(arg))

        return_code = ctypes.c_int(0)
        self.lib.launch(*ctype_args, ctypes.byref(return_code))
        return return_code.value


class RuntimeCache:

    def __init__(self) -> None:
        self.cache = {}

    def __getitem__(self, path: str) -> Optional[Runtime]:
        if path in self.cache:
            return self.cache[path]

        if os.path.exists(path) and Runtime.is_path_valid(path):
            runtime = Runtime(path)
            self.cache[path] = runtime
            return runtime
        return None

    def __setitem__(self, path, runtime) -> None:
        self.cache[path] = runtime
