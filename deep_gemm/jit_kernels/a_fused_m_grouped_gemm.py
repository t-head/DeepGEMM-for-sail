import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div
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
            aligned_num_m_blocks, m, topk, stream, num_sms);
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
            aligned_num_m_blocks, m, topk, stream, num_sms);
"""

includes_fusedgemm_util_kernel = ('"../deep_gemm/fused_gemm_util.cuh"', )
template_fusedgemm_util_kernel = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto BLOCK_M = {BLOCK_M};

// Launch kernel
moe_align_block_size_kernel_launcher<BLOCK_M, kNumGroups>(m_rows,
            expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
            topk_ids, numel, max_num_m_blocks, stream);
"""

def moe_align_block_size(topk_ids: torch.Tensor, num_groups:int, block_m:int):
    """
    Align token assignments to blocks for MoE Grouped GEMM computation.

    Arguments:
        topk_ids: A tensor of shape `[num_token, top_k]` with type `torch.int32`,
                indicating which expert each token selects.
        num_groups: Number of expert.
        block_m: Block size for M dimension (number of tokens per block).

    Returns:
        m_rows: A tensor indicating the number of tokens assigned to each expert.
        expert_ids_and_cumsum: A combined tensor of shape `[max_num_m_blocks, 2]` containing:
            - expert_ids: which expert each block processes
            - cumsum_m:
        sorted_token_ids: A tensor of shape `[max_num_m_blocks, block_m]` with type `torch.int32`,
                indicating token indices each block processes, padded for incomplete blocks.
        aligned_num_m_blocks: The total number of M blocks after alignment.
    """
    assert topk_ids.dtype == torch.int32
    numel = topk_ids.numel()

    # the largest blockM_num is:
    #  * num_groups - 1 only has 1 token
    #  * the last group has (numel - (num_groups - 1)) tokens
    max_num_m_blocks = num_groups - 1 + ceil_div(numel + 1 - num_groups, block_m)

    m_rows = torch.empty((num_groups), dtype=torch.int32, device=topk_ids.device)
    expert_ids_and_cumsum = torch.empty((max_num_m_blocks, 2), dtype=torch.int32, device=topk_ids.device)
    sorted_token_ids = torch.empty((max_num_m_blocks,block_m), dtype=torch.int32, device=topk_ids.device)
    aligned_num_m_blocks = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    global includes_fusedgemm_util_kernel, template_fusedgemm_util_kernel

    args = (m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, topk_ids,
            numel, max_num_m_blocks, torch.cuda.current_stream())

    runtime = jit_tuner.compile_and_tune(
        name='moe_align_block_size',
        keys={'NUM_GROUPS': num_groups,
                'BLOCK_M': block_m, 'BLOCK_N': 1, 'BLOCK_K': 1,
                'WARP_M': 1, 'WARP_N': 1, 'NUM_STAGES': 1,
            },
        space=(),
        includes=includes_fusedgemm_util_kernel,
        arg_defs=(('m_rows', torch.int32),
                ('expert_ids_and_cumsum', torch.int32),
                ('sorted_token_ids', torch.int32),
                ('aligned_num_m_blocks', torch.int32),
                ('topk_ids', torch.int32),
                ('numel', int),
                ('max_num_m_blocks', int),
                ('stream', torch.cuda.Stream)),
        template=template_fusedgemm_util_kernel,
        jit_include_dir='cutlass3',
        args=args
    )
    runtime(*args)
    return m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks

def m_grouped_gemm_bf16_bf16_bf16_nt_fused(lhs: Tuple[torch.Tensor],
                                     rhs: Tuple[torch.Tensor],
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

        m_rows: A tensor indicating the number of tokens assigned to each expert.
        expert_ids_and_cumsum: Combined tensor of shape `[max_num_m_blocks, 2]` containing:
            - expert_ids: which expert each block processes (int32)
            - cumsum_m
        sorted_token_ids: A tensor of shape `[max_num_m_blocks, block_m]` with type `torch.int32`,
                         indicating token indices each block processes, padded for incomplete blocks.
        aligned_num_m_blocks: The total number of M blocks after alignment.
        configs: Optional configuration parameters for kernel tuning.

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

    topk = int(m_sum / num_token)

    # Do nothing if `m_sum` is zero
    if m_sum == 0:
        return
     # Auto-tuning with compilation
    global includes_fusedmoe_gemm, template_fusedmoe_gemm
    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    # torch.set_printoptions(threshold=10000000, linewidth=10000, precision=2, sci_mode=False)
    # print(m_rows)
    # print(expert_ids_and_cumsum.shape, expert_ids_and_cumsum)
    # print(sorted_token_ids, sorted_token_ids.shape)
    # print(aligned_num_m_blocks)

    # print("out_shape:", out.shape)
    # print("lhs_shape:", lhs.shape)
    args = (lhs, rhs, out, m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks,
            num_token, topk, torch.cuda.current_stream(), int(num_sms))

    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='fusedMoeGemm_bf16_bf16_bf16_nt',
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
                ('topk', int),
                ('stream', torch.cuda.Stream),
                ('num_sms', int)),
        template=template_fusedmoe_gemm,
        jit_include_dir='cutlass3',
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
        "K must be a multiple of 8, "
        "so that 16 8-bit elements can be loaded with 128b aligned vectorized memory access."
    )

    topk = int(m_sum / num_token)
    # Do nothing if `m_sum` is zero
    if m_sum == 0:
        return
     # Auto-tuning with compilation
    global includes_fusedmoe_gemm_with_perchannel_quant, template_fusedmoe_gemm_with_perchannel_quant

    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs

    # print("m_rows:", m_rows.shape, m_rows)
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

    SrcT = "__nv_fp8_e4m3" if lhs.dtype == torch.float8_e4m3fn else "int8_t"
    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='fusedMoeGemm_a8w8_nt_' + SrcT,
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
                ('topk', int),
                ('stream', torch.cuda.Stream),
                ('num_sms', int)),
        template=template_fusedmoe_gemm_with_perchannel_quant,
        jit_include_dir='cutlass3',
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
        "K must be a multiple of 8, "
        "so that 16 8-bit elements can be loaded with 128b aligned vectorized memory access."
    )

    # per-channel quant
    if lhs_scales.shape == (num_token, 1) and rhs_scales.shape == (num_groups, n, 1):
        m_grouped_gemm_perchannel_nt_fused(lhs_, rhs_, out, m_rows,
            expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, configs)
        return

    # blockwise quant
    topk = int(m_sum / num_token)
    # Do nothing if `m_sum` is zero
    if m_sum == 0:
        return
     # Auto-tuning with compilation
    global includes_fusedmoe_gemm_with_blkwise_quant, template_fusedmoe_gemm_with_blkwise_quant
    num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs

    # print("m_rows:", m_rows.shape, m_rows)
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
        name='fusedMoeGemm_fp8_fp8_bf16_nt',
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
        jit_include_dir='cutlass3',
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
    m_grouped_gemm_perchannel_nt_fused(lhs_, rhs_, out, m_rows,
        expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, configs)