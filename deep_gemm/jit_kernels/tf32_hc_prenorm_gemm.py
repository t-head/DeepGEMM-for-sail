import math
import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, is_ppu1v5_device

includes_cutlass3 = ('<deep_gemm/impls/tf32_hc_prenorm_gemm.cuh>', )

template_cutlass3 = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N};
constexpr auto K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto NUM_SPLITS = {NUM_SPLITS};
constexpr bool FAST_BF16_TO_TF32 = {FAST_BF16_TO_TF32};

using gemm_t = HcPrenormGemm<
    N, K,
    BLOCK_M, BLOCK_N, BLOCK_K,
    NUM_SPLITS,
    FAST_BF16_TO_TF32
>;

// Launch kernel
gemm_t::run(d, sqr_sum, m, a, b, stream, num_sms, smem_size, ws, ws_s, counter);
"""

def _align(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment

_ws_cache = {}
_counter_cache = {}
_retired = []


def _round_up_pow2(x: int) -> int:
    return 1 << max(0, x - 1).bit_length()


def _get_workspace(m: int, n: int, num_splits: int, grid_m: int, device, stream):
    key = (device.type, -1 if device.index is None else device.index, stream.cuda_stream)
    need_d, need_s = num_splits * m * n, num_splits * m

    ws, ws_s = _ws_cache.get(key, (None, None))
    if ws is None or ws.numel() < need_d:
        if ws is not None:
            _retired.append(ws)
        ws = torch.empty(_round_up_pow2(need_d), device=device, dtype=torch.float32)
    if ws_s is None or ws_s.numel() < need_s:
        if ws_s is not None:
            _retired.append(ws_s)
        ws_s = torch.empty(_round_up_pow2(need_s), device=device, dtype=torch.float32)
    _ws_cache[key] = (ws, ws_s)

    counter = _counter_cache.get(key + (num_splits,))
    if counter is None or counter.numel() < grid_m:
        if counter is not None:
            _retired.append(counter)
        counter = torch.zeros(_round_up_pow2(grid_m), device=device, dtype=torch.int)
        _counter_cache[key + (num_splits,)] = counter
    return ws, ws_s, counter

def tf32_hc_prenorm_gemm(a: torch.Tensor,
                       b: torch.Tensor,
                       d: torch.Tensor,
                       sqr_sum: torch.Tensor,
                       num_splits: int = None,
                       configs = None) -> None:
    m, k = a.shape
    n, k_ = b.shape

    assert k == k_
    assert n > 0 and k > 0
    assert a.dtype == torch.bfloat16
    assert b.dtype == torch.float32
    assert a.is_contiguous() and b.is_contiguous()

    if is_ppu1v5_device():
        block_n = min(_align(n, 8), 32)
    else:
        block_n = min(_align(n, 16), 32)

    assert n <= block_n
    assert n <= 32 and n % 8 == 0

    block_m = 64
    block_k = 64
    assert k % block_k == 0

    fast_bf16_to_tf32 = True

    k_blocks = k // block_k
    grid_m = (m + block_m - 1) // block_m
    cap = 32 if grid_m <= 2 else 16
    target = min(cap, max(1, 256 // grid_m))
    num_splits = 1 << (target.bit_length() - 1)
    while num_splits > 1 and (k_blocks % num_splits or k_blocks // num_splits < 2):
        num_splits //= 2

    assert k_blocks >= 2, f'K={k} is too small, need at least two blocks of {block_k}'
    assert k_blocks % num_splits == 0 and k_blocks // num_splits >= 2, (
        f'num_splits={num_splits} must divide k_blocks={k_blocks} evenly and leave '
        f'at least two K blocks per split')

    d_shape = (m, n)
    s_shape = (m,)

    if d is None:
        d = torch.empty(d_shape, device=a.device, dtype=torch.float32)
    else:
        assert d.dtype == torch.float32
        assert d.shape in (d_shape, (1, *d_shape)), (
            f"expected d.shape=={d_shape} or {(1, *d_shape)}, got {tuple(d.shape)}")
        assert d.is_contiguous()

    if sqr_sum is None:
        sqr_sum = torch.empty(s_shape, device=a.device, dtype=torch.float32)
    else:
        assert sqr_sum.dtype == torch.float32
        assert sqr_sum.shape in (s_shape, (1, *s_shape)), (
            f"expected sqr_sum.shape=={s_shape} or {(1, *s_shape)}, got {tuple(sqr_sum.shape)}")
        assert sqr_sum.is_contiguous()

    smem_size = 0

    num_sms = get_num_sms()
    stream = torch.cuda.current_stream()

    ws, ws_s, counter = _get_workspace(m, n, num_splits, grid_m, a.device, stream)

    args = (a, b, d, sqr_sum, m, stream, num_sms, smem_size, ws, ws_s, counter)

    runtime = jit_tuner.compile_and_tune(
        name='tf32_hc_prenorm_gemm',
        keys={
            'N': n,
            'K': k,
            'BLOCK_M': block_m,
            'BLOCK_N': block_n,
            'BLOCK_K': block_k,
            'NUM_SPLITS': num_splits,
            'FAST_BF16_TO_TF32': 'true' if fast_bf16_to_tf32 else 'false',
        },
        space=(),
        includes=includes_cutlass3,
        arg_defs=(
            ('a', torch.bfloat16),
            ('b', torch.float32),
            ('d', torch.float32),
            ('sqr_sum', torch.float32),
            ('m', int),
            ('stream', torch.cuda.Stream),
            ('num_sms', int),
            ('smem_size', int),
            ('ws', torch.float32),
            ('ws_s', torch.float32),
            ('counter', torch.int),
        ),
        template=template_cutlass3,
        jit_include_dir='actlize_v1.0.0',
        args=args,
    )

    runtime(*args)
