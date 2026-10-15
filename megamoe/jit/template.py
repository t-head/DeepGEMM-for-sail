import ctypes
import os
import torch
from typing import Any, Dict, Iterable, Tuple

typename_map: Dict[Any, str] = {
    **{
        t: t.__name__
        for t in (int, float)
    },
    torch.int: 'torch.int',
    torch.int8: 'torch.int8',
    torch.uint8: 'torch.uint8',
    torch.uint16: 'torch.uint16',
    torch.int64: 'torch.int64',
    torch.bfloat16: 'torch.bfloat16',
    torch.cuda.Stream: 'torch.cuda.Stream',
}

type_by_name: Dict[str, Any] = {name: t for t, name in typename_map.items()}

# Per torch argument type: (the extern "C" launch parameter type, the type the kernel body casts it to).
kernel_arg_types = {
    int: ('int', 'int'),
    float: ('float', 'float'),
    torch.int: ('void*', 'int*'),
    torch.int8: ('void*', 'int8_t*'),
    torch.uint8: ('void*', 'uint8_t*'),
    torch.uint16: ('void*', 'uint16_t*'),
    torch.int64: ('void*', 'int64_t*'),
    torch.bfloat16: ('void*', '__nv_bfloat16*'),
    torch.cuda.Stream: ('void*', 'cudaStream_t'),
}


def to_ctype_arg(value: Any) -> Any:
    if hasattr(value, 'data_ptr'):
        return ctypes.c_void_p(value.data_ptr())

    if hasattr(value, 'cuda_stream'):
        return ctypes.c_void_p(value.cuda_stream)

    if isinstance(value, bool):
        return ctypes.c_bool(value)
    elif isinstance(value, int):
        return ctypes.c_int(value)
    elif isinstance(value, float):
        return ctypes.c_float(value)

    raise TypeError(f'unsupported kernel argument type: {type(value)!r}')


def render_cpp_template(template: str, substitutions: Dict[str, Any]) -> str:
    new_template = template
    for key, value in substitutions.items():
        value_str = str(value)
        if isinstance(value, bool):
            value_str = value_str.lower()
        new_template = new_template.replace(f'{{{key}}}', f'{value_str}')
    return new_template


def generate_source(includes: Iterable[str], arg_defs: Iterable[Tuple], body: str, preamble: str = '') -> str:
    code = '// MegaMoE auto-generated JIT CUDA source file\n\n'

    if preamble:
        code += preamble.rstrip() + '\n\n'

    default_sys_includes = ['<cuda.h>', '<cuda_fp8.h>', '<cuda_runtime.h>', '<iostream>']

    assert isinstance(includes, (list, tuple))
    sys_includes = sorted(list(set(default_sys_includes + [include for include in includes if include.startswith('<')])))
    package_includes = sorted(list(set([include for include in includes if include.startswith('"')])))
    code += '\n'.join(f'#include {include}' for include in sys_includes) + '\n\n'
    code += '\n'.join(f'#include {include}' for include in package_includes) + '\n\n'

    raw_name_prefix = '__raw_'

    def render_arg_decl(name, torch_type):
        decl_type, cast_type = kernel_arg_types[torch_type]
        return f'{decl_type} ' + (raw_name_prefix if decl_type != cast_type else '') + name

    code += 'extern "C" void launch('
    code += ', '.join([render_arg_decl(*arg_def) for arg_def in arg_defs] + [
        'int& __return_code',
    ])
    code += ') {\n'

    code += '    // Cast raw types (if needed)\n'
    for arg_name, arg_type in arg_defs:
        decl_type, cast_type = kernel_arg_types[arg_type]
        if decl_type != cast_type:
            code += f'    auto {arg_name} = reinterpret_cast<{cast_type}>({raw_name_prefix}{arg_name});\n'

    code += '\n'.join([(('    ' if line else '') + line) for line in body.split('\n')])

    code += '}\n\n'

    if os.getenv('DG_JIT_DEBUG', None):
        print(f'Generated code:\n{code}')

    return code
