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

    block_m = 64 if (m <= 256 or (m <= 4096 and k <= 8192)) else 128
    block_k = 128 if m <= 256 else 64
    assert k % block_k == 0

    fast_bf16_to_tf32 = (m <= 256 or k <= 8192 or m >= 8192)

    if m <= 256:
        num_splits = min(64, k // block_k)
    elif m <= 512:
        num_splits = min(32, k // block_k)
    elif m >= 8192:
        num_splits = min(32, k // block_k)
    else:
        num_splits = min(16, k // block_k)
    num_splits = max(num_splits, 1)

    smem_size = 2 * (
        block_m * (block_k + 8) * 2 +
        block_n * (block_k + 4) * 4
    )

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
        jit_include_dir='cutlass3',
        args=args,
    )

    runtime(*args)
