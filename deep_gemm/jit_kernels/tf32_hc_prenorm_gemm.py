import math
import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, is_ppu1v5_device

includes_cutlass3 = ('"../deep_gemm/tf32_hc_prenorm_gemm.cuh"', )

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
    FAST_BF16_TO_TF32,
    true
>;

// Launch kernel
gemm_t::run(d, sqr_sum, m, a, b, stream, num_sms, smem_size);
"""

def _align(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment

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

    if d is None:
        d = torch.empty((1, m, n), device=a.device, dtype=torch.float32)
    else:
        assert d.dtype == torch.float32
        if num_splits is None:
            assert d.shape == (m, n)
        else:
            assert d.shape == (1, m, n)
        assert d.is_contiguous()

    if sqr_sum is None:
        sqr_sum = torch.empty((m,), device=a.device, dtype=torch.float32)
    else:
        assert sqr_sum.dtype == torch.float32
        if num_splits is None:
            assert sqr_sum.shape == (m, )
        else:
            assert sqr_sum.shape == (1, m)
        assert sqr_sum.is_contiguous()

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
    if k_blocks >= 448:  # K >= 28672
        num_splits = 16
    else:
        num_splits = min(32, k_blocks)
    num_splits = max(num_splits, 1)

    smem_size = 0  

    num_sms = get_num_sms()
    stream = torch.cuda.current_stream()

    args = (a, b, d, sqr_sum, m, stream, num_sms, smem_size)

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
        ),
        template=template_cutlass3,
        jit_include_dir='actlize_v1.0.0',
        args=args,
    )

    runtime(*args)
