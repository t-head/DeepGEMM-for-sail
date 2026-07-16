import torch
from typing import Tuple

from .tuner import jit_tuner
from .gemm import get_best_configs as bf16_get_best_configs
from .gemm_int8 import get_best_configs as perchannel_get_best_configs
from .gemm_fp8 import get_best_configs as fp8_blkwise_get_best_configs
from .gemm_fp4 import get_best_configs as fp4_get_best_configs
from .utils import get_num_sms, ceil_div, GemmType, get_col_major_tma_aligned_tensor
import os

# C++ code templates
includes_fusedmoe_gemm = ('"../deep_gemm/fused_moe_gemm.cuh"', )
template_fusedmoe_gemm = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kEnableSboOverlap = {ENABLE_SBO_OVERLAP};

// Make a templated grouped GEMM
using fused_moe_gemm = FusedMoeGemm<N, K, kNumGroups,
            BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumStages,
            GemmType::{GEMM_TYPE}, kEnableSboOverlap, KernelType::{KERNEL_TYPE}>;

// Launch kernel
fused_moe_gemm::run(out, lhs, rhs, m_rows, expert_ids_and_cumsum, sorted_token_ids,
            aligned_num_m_blocks, m, stream, num_sms);
"""

includes_fusedmoe_gemm_with_blkwise_quant = (
    '"../deep_gemm/fused_moe_gemm_with_blkwise_quant.cuh"',
)
template_fusedmoe_gemm_with_blkwise_quant = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kEnableSboOverlap = {ENABLE_SBO_OVERLAP};

// Make a templated grouped GEMM
using fused_moe_gemm_with_blkwise_quant = FusedMoeGemmWithBlkwiseQuant<
            N, K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumStages,
            GemmType::{GEMM_TYPE}, kEnableSboOverlap, KernelType::{KERNEL_TYPE}>;

// Launch kernel
fused_moe_gemm_with_blkwise_quant::run(out, lhs, rhs, lhs_scales, rhs_scales,
            m_rows, expert_ids_and_cumsum, sorted_token_ids,
            aligned_num_m_blocks, m, topk, stream, num_sms);
"""

includes_fusedmoe_gemm_with_perchannel_quant = (
    '"../deep_gemm/fused_moe_gemm_with_perchannel_quant.cuh"',
)
template_fusedmoe_gemm_with_perchannel_quant = """
using namespace deep_gemm;

// Templated args from Python JIT call
using SrcT = {SrcT};
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kEnableSboOverlap = {ENABLE_SBO_OVERLAP};

// Make a templated grouped GEMM
using fused_moe_gemm_with_perchannel_quant = FusedMoeGemmWithPerChannelQuant<
            SrcT, N, K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumStages,
            GemmType::{GEMM_TYPE}, kEnableSboOverlap, KernelType::{KERNEL_TYPE}>;

// Launch kernel
fused_moe_gemm_with_perchannel_quant::run(out, lhs, rhs, lhs_scales, rhs_scales,
            m_rows, expert_ids_and_cumsum, sorted_token_ids,
            aligned_num_m_blocks, m, stream, num_sms);
"""

includes_fusedgemm_util_kernel = ('"../deep_gemm/fused_gemm_util.cuh"', )
template_fusedgemm_util_kernel = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto kTopK = {TOPK};

// Launch kernel
moe_align_block_size_kernel_launcher<BLOCK_M, kNumGroups, kTopK>(m_rows,
            expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
            inv_perm, m_indices, topk_ids, numel, max_num_m_blocks,
            intermediate_buffer, stream);
"""

includes_fp4_fusedmoe_gemm = ('"../deep_gemm/fused_moe_fp4_gemm.cuh"', )
template_fp4_fusedmoe_gemm = """
using namespace deep_gemm;
// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kEnableSboOverlap = {ENABLE_SBO_OVERLAP};

// Make a templated grouped GEMM
using fused_moe_gemm_fp4 = Fp4FusedMoeGemm<N, K, kNumGroups,
            BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumStages,
            GemmType::{GEMM_TYPE}, kEnableSboOverlap, KernelType::{KERNEL_TYPE}>;

// Launch kernel
fused_moe_gemm_fp4::run(out, lhs, rhs, lhs_scales, rhs_scales,
            m_rows, expert_ids_and_cumsum, sorted_token_ids,
            aligned_num_m_blocks, m, topk, stream, num_sms);
"""

def moe_align_block_size(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    topk_ids: torch.Tensor,
    perchannel_quant: bool = False,
    config=None,
):
    """
    Align token assignments to blocks for MoE Grouped GEMM computation.

    Arguments:
        lhs: Left-hand side input tensor of shape `[num_token, k]`.
        rhs: Right-hand side input tensor of shape `[num_groups, n, k]`.
        topk_ids: A tensor of shape `[num_token, top_k]` with type `torch.int32`,
                indicating which expert each token selects.
        perchannel_quant: Whether per-channel quantization is used (affects auto-config selection).
        config: Optional GEMM config tuple. If None, auto-selected via get_best_configs.

    Returns:
        config: The GEMM config tuple used for block_m selection.
        m_rows: A tensor of shape `[num_groups]` with type `torch.int32`,
                indicating the number of tokens per expert.
        expert_ids_and_cumsum: A combined tensor of shape `[max_num_m_blocks, 4]` containing:
            - expert_ids: which expert each block processes
            - group_num: per-expert token count
            - cumsum_aligned: block-aligned cumsum
            - cumsum_compact: compact cumsum
        sorted_token_ids: A tensor of shape `[max_num_m_blocks, block_m]` with type `torch.int32`,
                indicating token indices each block processes, padded for incomplete blocks.
        aligned_num_m_blocks: The total number of M blocks after alignment.
        inv_perm: A tensor of shape `[numel]` with type `torch.int32`,
                inverse permutation: inv_perm[token_idx * topk + topk_slot] = sorted position.
        m_indices: A tensor of shape `[numel]` with type `torch.int32`,
                per-row expert ID for nopad GEMM2.
    """
    assert topk_ids.dtype == torch.int32

    num_token, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    dtype = lhs.dtype
    numel = topk_ids.numel()
    topk = topk_ids.shape[1]

    # Auto-select config if not provided
    if config is None:
        expected_m = ceil_div(numel, num_groups)
        num_sms = get_num_sms()
        if dtype == torch.bfloat16:
            config = bf16_get_best_configs(expected_m, n, k, num_groups, num_sms, GemmType.GroupedFused)
        elif perchannel_quant:
            config = perchannel_get_best_configs(expected_m, n, k, num_groups, num_sms, GemmType.GroupedFused)
        elif dtype == torch.float8_e4m3fn:
            config = fp8_blkwise_get_best_configs(expected_m, n, k, num_groups, num_sms, GemmType.GroupedFused)
        elif dtype == torch.uint8:
            config = fp4_get_best_configs(numel, expected_m, n, k, num_groups, num_sms, GemmType.GroupedFused)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    block_m = config[1]
    assert num_groups > 0, "num_groups must be positive"
    assert block_m > 0, "block_m must be positive"

    # the largest blockM_num is:
    #  * num_groups - 1 only has 1 token
    #  * the last group has (numel - (num_groups - 1)) tokens
    max_num_m_blocks = num_groups - 1 + ceil_div(numel + 1 - num_groups, block_m)

    block_size = 1 << (num_groups - 1).bit_length()  # next_pow2
    s_total_ub = max_num_m_blocks * block_m
    num_blocks_pad = ceil_div(s_total_ub, block_size)

    expert_ids_and_cumsum = torch.empty((max_num_m_blocks, 4), dtype=torch.int32, device=topk_ids.device)
    sorted_token_ids = torch.empty((max_num_m_blocks, block_m), dtype=torch.int32, device=topk_ids.device)
    aligned_num_m_blocks = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    inv_perm = torch.empty((numel,), dtype=torch.int32, device=topk_ids.device)
    m_rows = torch.empty((num_groups,), dtype=torch.int32, device=topk_ids.device)
    m_indices = torch.empty((numel,), dtype=torch.int32, device=topk_ids.device)

    # Intermediate buffer (host allocates, kernel fills)
    #   block_counts:  (num_blocks_pad, num_groups)
    #   local_offsets: (num_blocks_pad, num_groups)
    #   cumsum:        (num_groups,)
    intermediate_buffer = torch.empty(
        (2 * num_blocks_pad + 2, num_groups),
        dtype=torch.int32, device=topk_ids.device)

    global includes_fusedgemm_util_kernel, template_fusedgemm_util_kernel

    args = (m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
            inv_perm, m_indices, topk_ids, numel, max_num_m_blocks, intermediate_buffer,
            torch.cuda.current_stream())

    runtime = jit_tuner.compile_and_tune(
        name='moe_align_block_size',
        keys={'NUM_GROUPS': num_groups,
                'BLOCK_M': block_m, 'BLOCK_N': 1, 'BLOCK_K': 1,
                'WARP_M': 1, 'WARP_N': 1, 'NUM_STAGES': 1,
                'TOPK': topk,
            },
        space=(),
        includes=includes_fusedgemm_util_kernel,
        arg_defs=(('m_rows', torch.int32),
                ('expert_ids_and_cumsum', torch.int32),
                ('sorted_token_ids', torch.int32),
                ('aligned_num_m_blocks', torch.int32),
                ('inv_perm', torch.int32),
                ('m_indices', torch.int32),
                ('topk_ids', torch.int32),
                ('numel', int),
                ('max_num_m_blocks', int),
                ('intermediate_buffer', torch.int32),
                ('stream', torch.cuda.Stream)),
        template=template_fusedgemm_util_kernel,
        jit_include_dir='actlize_v1.0.0',
        args=args
    )
    runtime(*args)
    # Drop smem_config (last item): only the first 7 tuning params are needed
    return config[:7], m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, inv_perm, m_indices

def m_grouped_gemm_bf16_bf16_bf16_nt_fused(lhs: torch.Tensor,
                                     rhs: torch.Tensor,
                                     out: torch.Tensor,
                                     m_rows: torch.Tensor,
                                     expert_ids_and_cumsum: torch.Tensor,
                                     sorted_token_ids: torch.Tensor,
                                     aligned_num_m_blocks: torch.Tensor,
                                     configs) -> None:
    """
    Perform a grouped GEMM (contiguous format) with BF16 inputs and BF16 output,.

    Requirements:
        LHS, RHS, and output tensors must be in contiguous format.
        RHS are required to be transposed.

    Arguments:
        lhs: The BF16 input tensor (typed `torch.bfloat16`) of shape `[num_token, k]`.
        rhs: The BF16 input tensor (typed `torch.bfloat16`) of shape `[num_groups, n, k]`.
        out: The BF16 output tensor of shape `[m_sum, n]`, representing the computation result.

        expert_ids_and_cumsum: Combined tensor of shape `[max_num_m_blocks, 4]` containing:
            - expert_ids: which expert each block processes (int32)
            - group_num: per-expert token count
            - cumsum_aligned: block-aligned cumsum
            - cumsum_compact: compact cumsum
        sorted_token_ids: A tensor of shape `[max_num_m_blocks, block_m]` with type `torch.int32`,
                         indicating token indices each block processes, padded for incomplete blocks.
        aligned_num_m_blocks: The total number of M blocks after alignment.
        configs: Configuration parameters for kernel tuning.

    Where:
        num_token: the number of input tokens.
        top_k: the number of experts selected per token.
        num_groups: the number of expert groups (equal to num_experts).
        num_experts: the total number of experts.
        k: the hidden dimension of input features.
        n: the hidden dimension of output features.
        block_m: the maximum number of tokens processed per block in m dimension.
        max_num_m_blocks: the max number of blocks in m dimension.
        m_sum: the total number of token-expert pairs, equal to `num_token * top_k`.
    """
    num_token, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_sum, n_ = out.shape

    # Type and shape checks
    assert k == k_ and n == n_
    assert lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous()
    assert k % 8 == 0, (
        "K must be a multiple of 8, "
        "so that 8 bfloat16 elements can be loaded with 128b aligned vectorized memory access."
    )

    # Do nothing if `m_sum` is zero
    if m_sum == 0:
        return
     # Auto-tuning with compilation
    global includes_fusedmoe_gemm, template_fusedmoe_gemm
    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages = configs
    # torch.set_printoptions(threshold=10000000, linewidth=10000, precision=2, sci_mode=False)
    # print(expert_ids_and_cumsum.shape, expert_ids_and_cumsum)
    # print(sorted_token_ids, sorted_token_ids.shape)
    # print(aligned_num_m_blocks)

    # print("out_shape:", out.shape)
    # print("lhs_shape:", lhs.shape)
    args = (lhs, rhs, out, m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
            num_token, torch.cuda.current_stream(), int(num_sms))

    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='fusedmoe_gemm_bf16_bf16_bf16_nt',
        keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_STAGES': num_stages,
                'ENABLE_SBO_OVERLAP': False,
                'GEMM_TYPE': 'GroupedFused',
                'KERNEL_TYPE': kernel_type},
        space=(),
        includes=includes_fusedmoe_gemm,
        arg_defs=(('lhs', torch.bfloat16),
                ('rhs', torch.bfloat16),
                ('out', torch.bfloat16),
                ('m_rows', torch.int32),
                ('expert_ids_and_cumsum', torch.int32),
                ('sorted_token_ids', torch.int32),
                ('aligned_num_m_blocks', torch.int32),
                ('m', int),
                ('stream', torch.cuda.Stream),
                ('num_sms', int)),
        template=template_fusedmoe_gemm,
        jit_include_dir='actlize_v1.0.0',
        args=args
    )
    runtime(*args)

def m_grouped_gemm_perchannel_nt_fused(lhs_: Tuple[torch.Tensor],
                                     rhs_: Tuple[torch.Tensor],
                                     out: torch.Tensor,
                                     m_rows: torch.Tensor,
                                     expert_ids_and_cumsum: torch.Tensor,
                                     sorted_token_ids: torch.Tensor,
                                     aligned_num_m_blocks: torch.Tensor,
                                     configs) -> None:

    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    num_token, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_sum, n_ = out.shape

    # Type and shape checks
    assert k == k_ and n == n_
    assert lhs.dtype == torch.int8 or lhs.dtype == torch.float8_e4m3fn
    assert rhs.dtype == torch.int8 or rhs.dtype == torch.float8_e4m3fn
    assert out.dtype == torch.bfloat16
    assert lhs_scales.shape == (num_token, 1)
    assert rhs_scales.shape == (num_groups, n, 1)
    assert lhs_scales.dtype == torch.float32 and rhs_scales.dtype == torch.float32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous()
    assert k % 16 == 0, (
        "K must be a multiple of 16, "
        "so that 16 8-bit elements can be loaded with 128b aligned vectorized memory access."
    )

    # Do nothing if `m_sum` is zero
    if m_sum == 0:
        return
     # Auto-tuning with compilation
    global includes_fusedmoe_gemm_with_perchannel_quant, template_fusedmoe_gemm_with_perchannel_quant

    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages = configs

    # print(expert_ids_and_cumsum.shape, expert_ids_and_cumsum)
    # print(sorted_token_ids, sorted_token_ids.shape)

    # print("out_shape:", out.shape)
    # print("lhs:", lhs.shape, lhs.dtype)
    # print("lhs_scales:", lhs_scales.shape, lhs_scales.dtype)

    # print("rhs:", rhs.shape, rhs.dtype)
    # print("rhs_scales:", rhs_scales.shape, rhs_scales.dtype)
    # print("num_sms:", num_sms)

    args = (lhs, lhs_scales, rhs, rhs_scales, out, m_rows,
            expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
            num_token, torch.cuda.current_stream(), int(num_sms))

    SrcT = "__hg_fp8_e4m3" if lhs.dtype == torch.float8_e4m3fn else "int8_t"
    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='fusedmoe_gemm_a8w8_nt_' + SrcT,
        keys={'SrcT' : SrcT, 'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_STAGES': num_stages,
                'ENABLE_SBO_OVERLAP': False,
                'GEMM_TYPE': 'GroupedFused',
                'KERNEL_TYPE': kernel_type},
        space=(),
        includes=includes_fusedmoe_gemm_with_perchannel_quant,
        arg_defs=(('lhs', lhs.dtype), ('lhs_scales', torch.float),
                ('rhs', lhs.dtype), ('rhs_scales', torch.float),
                ('out', torch.bfloat16),
                ('m_rows', torch.int32),
                ('expert_ids_and_cumsum', torch.int32),
                ('sorted_token_ids', torch.int32),
                ('aligned_num_m_blocks', torch.int32),
                ('m', int),
                ('stream', torch.cuda.Stream),
                ('num_sms', int)),
        template=template_fusedmoe_gemm_with_perchannel_quant,
        jit_include_dir='actlize_v1.0.0',
        args=args
    )
    runtime(*args)

def m_grouped_gemm_fp8_fp8_bf16_nt_fused(lhs_: Tuple[torch.Tensor],
                                     rhs_: Tuple[torch.Tensor],
                                     out: torch.Tensor,
                                     m_rows: torch.Tensor,
                                     expert_ids_and_cumsum: torch.Tensor,
                                     sorted_token_ids: torch.Tensor,
                                     aligned_num_m_blocks: torch.Tensor,
                                     configs) -> None:

    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    num_token, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_sum, n_ = out.shape

    # Type and shape checks
    assert k == k_ and n == n_
    assert lhs.dtype == torch.float8_e4m3fn and rhs.dtype == torch.float8_e4m3fn
    assert lhs_scales.dtype == torch.float32 and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous()
    assert k % 16 == 0, (
        "K must be a multiple of 16, "
        "so that 16 8-bit elements can be loaded with 128b aligned vectorized memory access."
    )

    # per-channel quant
    if lhs_scales.shape == (num_token, 1) and rhs_scales.shape == (num_groups, n, 1):
        m_grouped_gemm_perchannel_nt_fused(lhs_, rhs_, out,
            m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, configs)
        return

    # blockwise quant
    topk = int(m_sum / num_token)
    # Do nothing if `m_sum` is zero
    if m_sum == 0:
        return

    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
     # Auto-tuning with compilation
    global includes_fusedmoe_gemm_with_blkwise_quant, template_fusedmoe_gemm_with_blkwise_quant
    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages = configs
    # block_m, block_n, block_k, warp_m, warp_n, num_stages = 64, 128, 128, 32, 32, 2
    assert n % 128 == 0, f"n ({n}) must be divisible by 128)"
    assert k % 128 == 0, f"k ({k}) must be divisible by 128)"
    assert block_k == 128, "currently only support block_k = 128."

    # print(expert_ids_and_cumsum.shape, expert_ids_and_cumsum)
    # print(sorted_token_ids, sorted_token_ids.shape)

    # print("out_shape:", out.shape)
    # print("lhs:", lhs.shape, lhs.dtype)
    # print("lhs_scales:", lhs_scales.shape, lhs_scales.dtype)

    # print("rhs:", rhs.shape, rhs.dtype)
    # print("rhs_scales:", rhs_scales.shape, rhs_scales.dtype)
    # print("num_sms:", num_sms)

    args = (lhs, lhs_scales, rhs, rhs_scales, out, m_rows,
            expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
            num_token, topk, torch.cuda.current_stream(), int(num_sms))

    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='fusedmoe_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_STAGES': num_stages,
                'ENABLE_SBO_OVERLAP': False,
                'GEMM_TYPE': 'GroupedFused',
                'KERNEL_TYPE': kernel_type},
        space=(),
        includes=includes_fusedmoe_gemm_with_blkwise_quant,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                ('out', torch.bfloat16),
                ('m_rows', torch.int32),
                ('expert_ids_and_cumsum', torch.int32),
                ('sorted_token_ids', torch.int32),
                ('aligned_num_m_blocks', torch.int32),
                ('m', int),
                ('topk', int),
                ('stream', torch.cuda.Stream),
                ('num_sms', int)),
        template=template_fusedmoe_gemm_with_blkwise_quant,
        jit_include_dir='actlize_v1.0.0',
        args=args
    )
    runtime(*args)

def m_grouped_gemm_int8_int8_bf16_nt_fused(lhs_: Tuple[torch.Tensor],
                                     rhs_: Tuple[torch.Tensor],
                                     out: torch.Tensor,
                                     m_rows: torch.Tensor,
                                     expert_ids_and_cumsum: torch.Tensor,
                                     sorted_token_ids: torch.Tensor,
                                     aligned_num_m_blocks: torch.Tensor,
                                     configs) -> None:
    m_grouped_gemm_perchannel_nt_fused(lhs_, rhs_, out,
        m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, configs)

def m_grouped_gemm_fp4_fp4_bf16_nt_fused(lhs_: Tuple[torch.Tensor],
                                     rhs_: Tuple[torch.Tensor],
                                     out: torch.Tensor,
                                     m_rows: torch.Tensor,
                                     expert_ids_and_cumsum: torch.Tensor,
                                     sorted_token_ids: torch.Tensor,
                                     aligned_num_m_blocks: torch.Tensor,
                                     configs) -> None:
    from .gemm_fp4 import check_mxfp4_scales_layout

    lhs, lhs_scales = lhs_
    rhs, rhs_scales = rhs_
    num_token, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_sum, n_ = out.shape

    # Type and shape checks
    assert k == k_ and n == n_
    assert lhs.dtype == torch.uint8 and rhs.dtype == torch.uint8
    assert lhs_scales.dtype == torch.uint16 and rhs_scales.dtype == torch.uint16
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous()
    ### lhs_scales is k major for fp4 fused moe, which has better sfa acp efficiency.
    assert lhs_scales.is_contiguous() and check_mxfp4_scales_layout(scale=rhs_scales, is_sfa=False)
    assert k % 16 == 0, (
        "K must be a multiple of 16, "
        "so that 16 8-bit elements can be loaded with 128b aligned vectorized memory access."
    )
    assert n % 2 == 0, f"n ({n}) must be divisible by 2)"

    topk = int(m_sum / num_token)
    # Do nothing if `m_sum` is zero
    if m_sum == 0:
        return
    
    ### parse tile config
    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages = configs

    global includes_fp4_fusedmoe_gemm, template_fp4_fusedmoe_gemm
    args = (lhs, lhs_scales, rhs, rhs_scales, out, m_rows,
            expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
            num_token, topk, torch.cuda.current_stream(), int(num_sms))
    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='fusedmoe_gemm_fp4_fp4_bf16_nt',
        keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_STAGES': num_stages,
                'ENABLE_SBO_OVERLAP': False,
                'GEMM_TYPE': 'GroupedFused',
                'KERNEL_TYPE': kernel_type},
        space=(),
        includes=includes_fp4_fusedmoe_gemm,
        arg_defs=(('lhs', lhs.dtype), ('lhs_scales', torch.uint16),
                ('rhs', lhs.dtype), ('rhs_scales', torch.uint16),
                ('out', torch.bfloat16),
                ('m_rows', torch.int32),
                ('expert_ids_and_cumsum', torch.int32),
                ('sorted_token_ids', torch.int32),
                ('aligned_num_m_blocks', torch.int32),
                ('m', int),
                ('topk', int),
                ('stream', torch.cuda.Stream),
                ('num_sms', int)),
        template=template_fp4_fusedmoe_gemm,
        jit_include_dir='actlize_v1.0.0',
        args=args
    )
    runtime(*args)