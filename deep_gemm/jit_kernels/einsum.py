import math
import torch
from functools import lru_cache
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout,get_col_major_tma_aligned_tensor,get_extra_info, GemmType
from .gemm_fp8 import get_best_configs
from .gemm_int8 import get_best_configs as get_best_int8_configs
# C++ code templates
includes = ('"../deep_gemm/fp8_gemm.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto BLOCK_N_PADDING = {BLOCK_N_PADDING};
constexpr auto kSwizzleDMode = {SWIZZLE_D_MODE};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};

// Make a templated GEMM
using gemm_t = Fp8Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, GemmType::BatchGemm>;

// Launch kernel
gemm_t::run(d, a, b, sfa,
            sfb, nullptr, nullptr, m, 0,
            stream, num_sms, smem_size);
"""

includes_cutlass3 = ('"../deep_gemm/int8_gemm_cutlass3.cuh"', )
template_cutlass3 = """
using namespace deep_gemm;

// Templated args from Python JIT call
using ElementAB = {ElementAB};
using ElementAcc = {ElementAcc};
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};

// Make a templated grouped GEMM
using gemm_t = Gemm<ElementAB, ElementAcc, N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::BatchGemm>;

// Launch kernel
gemm_t::run(d, nullptr, nullptr,
            m, 0, (ElementAB*)a, sfa, (ElementAB*)b, sfb,
            stream, num_sms, smem_size);
"""

def fp8_einsum(expr: str,
               a: Tuple[torch.Tensor, torch.Tensor],
               b: Tuple[torch.Tensor, torch.Tensor],
               d: torch.Tensor, c: torch.Tensor = None, recipe: list = [1, 1, 128]) -> None:
    # recipe is not used for PPU, only support [1, 128, 128]
    if (expr == "bhr,hdr->bhd"):
        # Permute dims to satisfy the order of (batch_size, m, n, k)
        # (batch_size, m, n, k): (h, b, d, r)
        lhs, lhs_scales = a
        rhs, rhs_scales = b
        perm_a = lhs.permute(1, 0, 2).contiguous()
        perm_sfa = lhs_scales.permute(1, 0, 2).contiguous()
        fp8_bmm(perm_a, perm_sfa, rhs, rhs_scales, d, c)
    elif (expr == "bhd,hdr->bhr"):
        raise NotImplementedError(
                "bhd,hdr->bhr is not yet supported in PPU fp8_einsum."
            )
    elif (expr == "bhd,bhr->hdr"):
        raise NotImplementedError(
                "bhd,bhr->hdr is not yet supported in PPU fp8_einsum."
            )
    else:
        raise ValueError(f"unsupported expr expression: {expr}!")

def fp8_bmm(a: torch.Tensor, sfa: torch.Tensor,
            b: torch.Tensor, sfb: torch.Tensor,
            d: torch.Tensor, c: torch.Tensor = None, recipe: list = [1, 1, 128], compiled_dims: str = None,
            expr: str = None, configs = None) -> None:
    # recipe & compiled_dims are not used for PPU
    groups, m, k = a.shape
    groups_, n, k_ = b.shape
    # groups__, m_, n_ = d.shape
    if sfa.shape == (groups, m, 1) and sfb.shape == (groups_, n, 1):
        return int8_bmm(a, sfa, b, sfb, d, c)

    assert k % 128 == 0, f"K={k} must be a multiple of 128"

    # Type and shape checks
    # assert groups == groups_ and groups_ == groups__
    # assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert a.dtype == torch.float8_e4m3fn and sfa.dtype == torch.float32
    assert b.dtype == torch.float8_e4m3fn and sfb.dtype == torch.float32
    assert d.dtype == torch.bfloat16
    assert a.is_contiguous() and b.is_contiguous() and d.is_contiguous()

    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    sfa = get_col_major_tma_aligned_tensor(sfa)
    assert sfb.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    if configs is not None:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, groups, num_sms)

    args = (a, sfa, b, sfb, d, m, torch.cuda.current_stream(), num_sms, smem_config[0])
    runtime = jit_tuner.compile_and_tune(
        name='batch_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'BLOCK_K' : block_k, 'WARP_M' : warp_m, 'WARP_N' : warp_n,
              'NUM_GROUPS': groups,
              'SWIZZLE_D_MODE': smem_config[1],
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_STAGES': num_stages},
        space=(),
        includes=includes,
        arg_defs=(('a', torch.float8_e4m3fn), ('sfa', torch.float),
                  ('b', torch.float8_e4m3fn), ('sfb', torch.float),
                  ('d', torch.bfloat16), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args,
        jit_include_dir='cutlass3'
    )
    # Run the kernel
    runtime(*args)

def int8_bmm(a: torch.Tensor, sfa: torch.Tensor,
             b: torch.Tensor, sfb: torch.Tensor,
             d: torch.Tensor, c: torch.Tensor = None, recipe: list = [1, 1, 128], compiled_dims: str = None,
             expr: str = None, configs = None) -> None:
    # recipe & compiled_dims are not used for PPU
    groups, m, k = a.shape
    groups_, n, k_ = b.shape
    # groups__, m_, n_ = d.shape

    # Type and shape checks
    # assert groups == groups_ and groups_ == groups__
    # assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert a.dtype == torch.int8 or a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.int8 or b.dtype == torch.float8_e4m3fn
    assert a.dtype == b.dtype
    assert sfa.dtype == torch.float32 and sfb.dtype == torch.float32
    assert d.dtype == torch.bfloat16
    assert a.is_contiguous() and b.is_contiguous() and d.is_contiguous()
    assert sfb.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes_cutlass3, template_cutlass3

    extra_info = get_extra_info()
    num_sms = get_num_sms()
    if configs is not None:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_int8_configs(m, n, k, groups, num_sms, gemm_type=GemmType.BatchGemm)

    args = (a, sfa, b, sfb, d, m, torch.cuda.current_stream(), num_sms, smem_config[0])
    ElementAB = "cutlass::float_e4m3_t" if a.dtype == torch.float8_e4m3fn else "int8_t"
    ElementAcc = "float" if a.dtype == torch.float8_e4m3fn else "int32_t"
    runtime = jit_tuner.compile_and_tune(
        name='batch_gemm_' + ElementAB + '_bf16_nt',
        keys={'ElementAB' : ElementAB, "ElementAcc" : ElementAcc,
              'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'BLOCK_K' : block_k, 'WARP_M' : warp_m, 'WARP_N' : warp_n,
              'NUM_GROUPS': groups,
              'SWIZZLE_D_MODE': smem_config[1],
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_STAGES': num_stages},
        space=(),
        includes=includes_cutlass3,
        arg_defs=(('a', a.dtype), ('sfa', torch.float),
                  ('b', b.dtype), ('sfb', torch.float),
                  ('d', torch.bfloat16), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template_cutlass3,
        jit_include_dir='cutlass3',
        args=args,
    )
    # Run the kernel
    runtime(*args)