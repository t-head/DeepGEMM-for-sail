import torch
from typing import Tuple

from .gemm_fp4 import get_best_configs, get_smem_config
from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div

# C++ code templates
includes = ('"../deep_gemm/fp4_gemm_cutlass3.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};

// Make a templated grouped GEMM
using gemm_t = Fp4Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}>;

gemm_t::run(lhs, lhs_scales, rhs, rhs_scales,
            bias, out, m, grouped_layout, block_m_info, expected_m,
            stream, num_sms, smem_size, signal);
"""

def m_grouped_gemm_fp4_fp4_fp32_nt_nopad(lhs_: Tuple[torch.Tensor, torch.Tensor],
                                         rhs_: Tuple[torch.Tensor, torch.Tensor],
                                         bias: torch.Tensor, out: torch.Tensor,
                                         m_indices: torch.Tensor, m_rows: torch.Tensor = None,
                                         configs = None) -> None:
    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs.dtype == torch.uint8 and rhs.dtype == torch.uint8
    assert lhs_scales.dtype == torch.uint8 and rhs_scales.dtype == torch.uint8
    assert bias.dtype == torch.float32
    assert out.dtype == torch.float32
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()
    assert rhs_scales.is_contiguous() and lhs_scales.is_contiguous() and m_indices.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    expected_m = ceil_div(m, num_groups)

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()

    # TODO: enable fp4 get_best_configs
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms)
        # num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages = (num_sms, 256, 256, 128, 64, 64, 3)
        # smem_config = get_smem_config(num_stages, k, block_m, block_n, block_k)

    if m_rows is None:
        counts = torch.bincount(m_indices)
        min_n = min(counts.size(0), num_groups)
        experts_for_rows = torch.zeros(num_groups, dtype=torch.int32, device='cuda')
        if min_n > 0:
            experts_for_rows[:min_n] = counts[:min_n]
        m_rows = experts_for_rows

    ## the largest blockM_num is, num_groups - 1 only has 1 token, the last group has (m-1) tokens, blockM_num = num_group -1  + ceil_div(m + 1 - num_group, block_m)
    ## total line num: blockM_num + 1, line0 is used to store the real blockM_num
    ## total_size = (blockM_num + 1) * 4 * sizeof(int) Byte
    block_m_info = torch.empty((num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4, dtype=torch.int32, device=m_rows.device)

    args = (lhs, lhs_scales, rhs, rhs_scales, bias, out, m, m_rows, block_m_info, expected_m, torch.cuda.current_stream(), num_sms, smem_config[0], torch.empty(0).int())
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp4_fp4_fp32_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages,'GEMM_TYPE': 'GroupedNoPad'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.uint8), ('lhs_scales', torch.uint8),
                  ('rhs', torch.uint8), ('rhs_scales', torch.uint8),
                  ('bias', torch.float32), ('out', torch.float32),
                  ('m', int), ('grouped_layout', torch.int32), ('block_m_info', torch.int32), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('signal', torch.int32)),
        template=template,
        args=args,
        jit_include_dir='cutlass3'
    )

    # Run the kernel
    runtime(*args)

    return out