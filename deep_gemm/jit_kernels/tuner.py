import copy
import os
import torch
from typing import Any, Dict

from ..jit import build, cpp_format, generate, Runtime, set_jit_include_dir


class JITTuner:
    def __init__(self) -> None:
        self.tuned = {}

    def key_format(self, key):
        if key:
            return 'block{0}x{1}x{2}xwarp{3}x{4}xstage{5}'.format(
                key['BLOCK_M'], key['BLOCK_N'], key['BLOCK_K'], key['WARP_M'], key['WARP_N'], key['NUM_STAGES'])
        else:
            return None

    def key_compare(self, key1, key2):
        keys = ['BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'WARP_M', 'WARP_N', 'NUM_STAGES']
        rtn = all(key1[k] == key2[k] for k in keys)
        return rtn

    def compile_and_tune(self, name: str, keys: Dict[str, Any], space: tuple,
                         includes: tuple, arg_defs: tuple, template: str, args: tuple, jit_include_dir: str = None) -> Runtime:
        # NOTES: we always assume the space and template will not change
        # We also assume the GPU device will not be changed
        # NOTES: the function must have no accumulated side effects
        keys = {k: keys[k] for k in sorted(keys.keys())}
        signature = (name, f'{keys}')
        if signature in self.tuned:
            if os.getenv('DG_JIT_DEBUG', None):
                print(f'Using cached JIT kernel {name} with keys {keys}')
            return self.tuned[signature]

        if os.getenv('DG_JIT_DEBUG', None):
            print(f'Auto-tuning JIT kernel {name} with keys {keys}')

        assert signature not in self.tuned
        assert args is not None
        space = (dict(), ) if len(space) == 0 else space

        kernels = []
        for tuned_keys in space:
            assert isinstance(tuned_keys, dict)
            full_keys = copy.deepcopy(keys)
            full_keys.update(tuned_keys)
            code = generate(includes, arg_defs, cpp_format(template, full_keys))

            # Illegal build must raise errors
            set_jit_include_dir(jit_include_dir)
            kernels.append((build(name, arg_defs, code), tuned_keys))

        best_runtime, best_time, best_keys = None, None, None
        default_tile_time = None
        for runtime, tuned_keys in kernels:
            if len(space) > 1:
                # Check kernel validity
                return_code = runtime(*args)
                if return_code != 0:
                    # Pass illegal kernels, e.g. insufficient shared memory capacity
                    if os.getenv('DG_JIT_DEBUG', None):
                        print(f'Illegal JIT kernel {name} with keys {keys} and tuned keys {tuned_keys}: error code {return_code}')
                    continue

                # Measure performance with L2 flush and a large GEMM kernel before to reduce overhead between kernels
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()
                torch.randn((8192, 8192), dtype=torch.float, device='cuda') @ torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                start_event.record()
                for i in range(20):
                    assert runtime(*args) == 0
                end_event.record()
                end_event.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
            else:
                elapsed_time = 0

            if tuned_keys and self.key_compare(keys, tuned_keys):
                default_tile_time = elapsed_time

            # Compare if better
            if best_time is None or elapsed_time < best_time:
                best_runtime, best_time, best_keys = runtime, elapsed_time, tuned_keys
            if os.getenv('DG_JIT_DEBUG', None):
                print(f'Tuned JIT kernel {name} with keys {keys} and tuned keys {tuned_keys} has time {elapsed_time}')
        assert best_runtime is not None, f'Failed to tune JIT kernel {name} with keys {keys}'

        # Cache the best runtime and return
        if os.getenv('DG_JIT_DEBUG', None) or os.getenv('DG_PRINT_AUTOTUNE', None):
            print(f'Best JIT kernel {name} with keys {self.key_format(keys)} and time {default_tile_time} has tuned keys {self.key_format(best_keys)} and time {best_time}')
            if best_keys:
                print('Best JIT kernel CSV,{0},{1:.4f},{2},{3:.4f},{4:.4f}'.format(self.key_format(keys), default_tile_time, self.key_format(best_keys), best_time, default_tile_time/best_time))
                print('Best JIT kernel CFG,({},{},{}):({},{},{},{},{},{})'.format(
                    args[5], keys['N'], keys['K'],
                    best_keys['BLOCK_M'], best_keys['BLOCK_N'], best_keys['BLOCK_K'],
                    best_keys['WARP_M'], best_keys['WARP_N'], best_keys['NUM_STAGES']
                ))
        self.tuned[signature] = best_runtime
        return best_runtime


jit_tuner = JITTuner()
