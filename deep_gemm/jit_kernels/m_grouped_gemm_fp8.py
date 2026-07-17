import torch
from typing import Tuple

from .gemm_fp8 import get_best_configs
from .tuner import jit_tuner
from .utils import get_col_major_tma_aligned_tensor, get_num_sms, ceil_div, GemmType
from .m_grouped_gemm_int8 import m_grouped_gemm_a8w8_per_channel_nt_contiguous
from .m_grouped_gemm_int8 import m_grouped_gemm_a8w8_per_channel_nt_masked
from .m_grouped_gemm_int8 import m_grouped_gemm_a8w8_per_channel_nt_nopad
from .gemm import get_gemv_best_configs

# C++ code templates
includes = ('"../deep_gemm/fp8_gemm.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
// constexpr auto BLOCK_K = 128;
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto BLOCK_N_PADDING = {BLOCK_N_PADDING};
constexpr auto kSwizzleDMode = {SWIZZLE_D_MODE};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kEnableSboOverlap = {ENABLE_SBO_OVERLAP};

// Make a templated grouped GEMM
using gemm_t = Fp8Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kEnableSboOverlap>;

gemm_t::run(out, lhs, rhs,
            lhs_scales, rhs_scales, grouped_layout, block_m_info,
            m, expected_m, stream, num_sms, smem_size, signal);
"""

includes_gemv = ('"deep_gemm/blockwise_gemvt.cuh"', )
template_gemv = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto ThreadPerN = {ThreadPerN};
constexpr auto NPerThread = {NPerThread};
constexpr auto NUM_UNROLL = {NUM_UNROLL};
constexpr auto SWZL_SIZE_M = {SWZL_SIZE_M};
constexpr auto USE_SMALL_K = {USE_SMALL_K};
constexpr auto BlockSize = {BlockSize};

// Make a templated grouped GEMM
using gemm_v = BlockWiseGemvt<__hg_fp8_e4m3, __ppu_bfloat16, {acc_type}, N, K, kNumGroups, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M, BlockSize, USE_SMALL_K>;

// Launch kernel
gemm_v::run(out, grouped_layout,
            m, lhs, rhs,
            stream,
            lhs_scales, rhs_scales);
"""

def m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs_: Tuple[torch.Tensor, torch.Tensor],
                                              rhs_: Tuple[torch.Tensor, torch.Tensor],
                                              out: torch.Tensor, m_indices: torch.Tensor, configs = None) -> None:
    """
    Do a grouped GEMM (contiguous format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.
    On the M axis, inputs are grouped into several batches, of which batch sizes aligned to
        `get_m_alignment_for_contiguous_layout()` (128).

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m_sum, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m_sum, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m_sum, n]`, representing the result.
        m_indices: a tensor of shape `[m_sum]` with type `torch.int`.
            `m_indices[i]` records the group which the i-th row of the LHS belong to,
            which means that the i-th row of the LHS matrix will be multiplied with `rhs[m_indices[i]]`.
            Values of `m_indices` in every-m-alignment-block must also be the same.
    """
    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    if lhs_scales.shape == (m, 1) and rhs_scales.shape == (num_groups, n, 1):
        return m_grouped_gemm_a8w8_per_channel_nt_contiguous(lhs_, rhs_, out, m_indices, configs)

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)

    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, num_groups, num_sms, gemm_type=GemmType.GroupedContiguous)
    expected_m = ceil_div(m, num_groups)
    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            m_indices, m_indices, m, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0], torch.empty(0).int())
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'BLOCK_K' : block_k, 'WARP_M' : warp_m, 'WARP_N' : warp_n,
              'SWIZZLE_D_MODE': smem_config[1],
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages,
              'ENABLE_SBO_OVERLAP': False,
              'GEMM_TYPE': 'GroupedContiguous'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16), ('grouped_layout', torch.int32), ('block_m_info', torch.int32),
                  ('m', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('signal', torch.int32)),
        template=template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )
    # Run the kernel
    runtime(*args)


def m_grouped_gemm_fp8_fp8_bf16_nt_masked(lhs_: Tuple[torch.Tensor, torch.Tensor],
                                          rhs_: Tuple[torch.Tensor, torch.Tensor],
                                          out: torch.Tensor, masked_m: torch.Tensor, expected_m: int, configs = None,
                                          max_block_n: int = 256, enable_sbo_overlap: bool = False,
                                          signal: torch.Tensor = torch.empty(0).int()) -> None:
    """
    Do a grouped GEMM (masked format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.
    Moreover, this alignment requirement is different with the contiguous-format kernel, as we require that each batch
        should be separately transposed.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, m_max, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[num_groups, m_max, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[num_groups, m_max, n]`, representing the result.
        masked_m: a tensor of shape `[num_groups]`, `masked_m[i]` records actual rows of the `lhs[i]` matrix to compute
            in the i-th group.
        expected_m: a value hint (which is a value on CPU) for the M expectation of each batch,
            correctly setting this value may lead to better performance.
    """
    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    num_groups, m, k = lhs.shape
    num_groups_, n, k_ = rhs.shape
    num_groups__, m_, n_ = out.shape
    num_groups___ = masked_m.numel()

    if lhs_scales.shape == (num_groups, m, 1) and rhs_scales.shape == (num_groups, n, 1):
        return m_grouped_gemm_a8w8_per_channel_nt_masked(lhs_, rhs_, out, masked_m, expected_m, configs)

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert lhs_scales.shape == (num_groups, m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and masked_m.is_contiguous()

    if enable_sbo_overlap:
        assert signal is not None
        assert signal.is_contiguous()
        assert signal.dtype == torch.int32

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)

    assert rhs_scales.is_contiguous()

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, gemm_type=GemmType.GroupedMasked, max_block_n=max_block_n)

    # Extra checks for TMA store
    # if num_groups > 1 and m > block_m:
    #     assert m % block_m == 0, f'For masked grouped GEMM, shape M should be multiple of the block M (current block M: {block_m})'

    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            masked_m, masked_m, m, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0], signal)
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'BLOCK_K' : block_k,  'WARP_M' : warp_m, 'WARP_N' : warp_n,
              'SWIZZLE_D_MODE': smem_config[1],
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages,
              'ENABLE_SBO_OVERLAP': enable_sbo_overlap,
              'GEMM_TYPE': 'GroupedMasked'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('block_m_info', torch.int32), ('m', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('signal', torch.int32)),
        template=template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )
    # Run the kernel
    runtime(*args)

    return (block_m, ceil_div(n, block_n))

def m_grouped_gemm_fp8_fp8_bf16_nt_nopad(lhs_: Tuple[torch.Tensor, torch.Tensor],
                                         rhs_: Tuple[torch.Tensor, torch.Tensor],
                                         out: torch.Tensor, m_indices: torch.Tensor,
                                         m_rows: torch.Tensor = None, configs = None) -> None:
    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    if lhs_scales.shape == (m, 1) and rhs_scales.shape == (num_groups, n, 1):
        return m_grouped_gemm_a8w8_per_channel_nt_nopad(lhs_, rhs_, out, m_indices, m_rows, configs)

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)

    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    expected_m = ceil_div(m, num_groups)

    # Auto-tuning with compilation
    global includes, template, includes_gemv, template_gemv
    num_sms = get_num_sms()
    '''
    # use gemv for small k
    if k == 128 and m <= num_groups:
        BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread, USE_SMALL_K = get_gemv_best_configs(m, n, k, num_groups, num_sms, torch.int8)
        if ThreadPerN != -1:
            args = (lhs, rhs, out, m_indices, m, torch.cuda.current_stream(), lhs_scales, rhs_scales)

            runtime = jit_tuner.compile_and_tune(
                name='m_grouped_gemv_fp8_fp8_bf16_nt',
                keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                    'ThreadPerN':ThreadPerN, 'NUM_UNROLL':NUM_UNROLL,
                    'SWZL_SIZE_M':SWZL_SIZE_M, 'NPerThread':NPerThread,
                    'BlockSize':BlockSize, 'USE_SMALL_K':USE_SMALL_K,
                    'acc_type':'float'},
                space=(),
                includes=includes_gemv,
                arg_defs=(('lhs', torch.float8_e4m3fn),
                        ('rhs', torch.float8_e4m3fn),
                        ('out', torch.bfloat16),
                        ('grouped_layout', torch.int32), ('m', int),
                        ('stream', torch.cuda.Stream),
                        ('lhs_scales', torch.float),
                        ('rhs_scales', torch.float)),
                template=template_gemv,
                args=args
            )
            runtime(*args)
            return
    '''

    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, gemm_type=GemmType.GroupedNoPad)

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

    args = (lhs, lhs_scales, rhs, rhs_scales, out,
        m_rows, block_m_info, m, expected_m,
        torch.cuda.current_stream(), num_sms, smem_config[0], torch.empty(0).int())

    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k,
            'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
            'WARP_M': warp_m, 'WARP_N': warp_n,
            'SWIZZLE_D_MODE': smem_config[1],
            'BLOCK_N_PADDING': smem_config[2],
            'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
            'ENABLE_SBO_OVERLAP': False,
            'GEMM_TYPE': 'GroupedNoPad'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                    ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                    ('out', torch.bfloat16),
                    ('grouped_layout', torch.int32), ('block_m_info', torch.int32),
                    ('m', int), ('expected_m', int),
                    ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                    ('signal', torch.int32)),
        template=template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )
    # Run the kernel
    runtime(*args)
