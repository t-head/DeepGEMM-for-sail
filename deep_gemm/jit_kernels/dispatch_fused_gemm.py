import os
import torch
from typing import Tuple

from .gemm_fp4 import (
    get_best_configs, get_smem_config_fp4,
    check_mxfp4_scales_layout, preprocess_mxfp4_scales,
    _post_preprocess_mxfp4_scales
)
from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, GemmType

# ==============================================================
# JIT Templates
# ==============================================================

# --- MXFP4 Quantization: BF16 → FP4 in symmetric buffer ---
includes_quant = ('"../deep_gemm/mxfp4_quant.cuh"', )
template_quant = """
using namespace deep_gemm;
launch_mxfp4_quantize<{HIDDEN}>(
    input, topk_ids, sym_buf_base,
    num_tokens, topk,
    {NUM_LOCAL_EXPERTS}, {NUM_TOTAL_EXPERTS}, {MAX_TOKENS_PER_EXPERT},
    generation,
    stream);
"""

# --- Dispatch Preprocess: read routing info → generate M-block metadata ---
includes_preprocess = ('"../deep_gemm/dispatch_preprocess.cuh"', )
template_preprocess = """
using namespace deep_gemm;

launch_dispatch_preprocess<{BLOCK_M}>(
    sym_buf_addrs,
    {RANK_IDX}, {NUM_RANKS},
    {NUM_LOCAL_EXPERTS}, {NUM_TOTAL_EXPERTS}, {MAX_TOKENS_PER_EXPERT}, {HIDDEN},
    {LOCAL_EXPERT_START},
    generation,
    grouped_layout,
    reinterpret_cast<uint64_t*>(remote_addr_a),
    reinterpret_cast<uint64_t*>(remote_addr_sfa),
    reinterpret_cast<uint32_t*>(out_total_m_blocks),
    reinterpret_cast<uint32_t*>(out_shape_m),
    stream);
"""

# --- Expert-level Preprocess: merge expert-rank pairs into expert groups ---
template_expert_preprocess = """
using namespace deep_gemm;

launch_dispatch_expert_preprocess<{BLOCK_M}>(
    sym_buf_addrs,
    {RANK_IDX}, {NUM_RANKS},
    {NUM_LOCAL_EXPERTS}, {NUM_TOTAL_EXPERTS}, {MAX_TOKENS_PER_EXPERT}, {HIDDEN},
    {LOCAL_EXPERT_START},
    generation,
    grouped_layout,
    reinterpret_cast<uint64_t*>(rank_addr_a),
    reinterpret_cast<uint64_t*>(rank_addr_sfa),
    reinterpret_cast<uint32_t*>(rank_split_m),
    reinterpret_cast<uint32_t*>(rank_counts),
    reinterpret_cast<uint32_t*>(out_total_m_blocks),
    reinterpret_cast<uint32_t*>(out_shape_m),
    stream);
"""

# --- Fused Dispatch GEMM1 (block-copy mode) ---
includes_gemm = ('"../deep_gemm/fp4_gemm_cutlass3.cuh"', )

template_gemm_block_copy = """
using namespace deep_gemm;

constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};

using gemm_t = Fp4Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N,
                        kNumGroups, kNumStages, GemmType::FusedDispatch>;
gemm_t::template run_fused_dispatch<{NUM_RANKS}, {NUM_COPY_BLOCKS}>(
    rhs, rhs_scales, bias, out, shape_m,
    grouped_layout,
    max_tokens_per_expert,
    stream, num_sms, smem_size,
    reinterpret_cast<const uint64_t*>(rank_addr_a),
    reinterpret_cast<const uint64_t*>(rank_addr_sfa),
    reinterpret_cast<const uint32_t*>(rank_split_m),
    reinterpret_cast<const uint32_t*>(rank_counts),
    reinterpret_cast<const uint64_t*>(merged_sfa_addrs),
    reinterpret_cast<uint8_t*>(local_fp4_buf),
    reinterpret_cast<volatile uint32_t*>(copy_ready_flags));
"""

# ==============================================================
# Python API
# ==============================================================

def get_sym_buffer_size(
    num_local_experts: int,
    num_total_experts: int,
    max_tokens_per_expert: int,
    hidden_dim: int,
) -> int:
    """Total symmetric buffer size in bytes for one rank."""
    metadata = ((num_total_experts * 4 + 15) // 16) * 16
    fp4_data = num_total_experts * max_tokens_per_expert * (hidden_dim // 2)
    k_blocks = (hidden_dim + 31) // 32
    k_scale_blocks = (k_blocks + 1) // 2
    scales = num_total_experts * k_scale_blocks * max_tokens_per_expert * 2
    base = metadata + fp4_data + scales
    flag_offset = ((base + 15) // 16) * 16
    return flag_offset + 16


def create_preprocess_workspace(
    num_local_experts: int,
    num_ranks: int,
    max_tokens_per_expert: int,
    block_m: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_m_blocks_per_pair = ceil_div(max_tokens_per_expert, block_m)
    max_total_m_blocks = num_local_experts * num_ranks * max_m_blocks_per_pair
    return (
        torch.zeros((1 + max_total_m_blocks) * 4, dtype=torch.int32, device=device),
        torch.zeros(max_total_m_blocks, dtype=torch.int64, device=device),
        torch.zeros(max_total_m_blocks, dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )


def mxfp4_quantize_to_sym_buffer(
    input_bf16: torch.Tensor,
    topk_ids: torch.Tensor,
    sym_buf: torch.Tensor,
    num_local_experts: int,
    num_total_experts: int,
    max_tokens_per_expert: int,
    generation: int = 0,
) -> None:
    """Quantize BF16 activations to MXFP4 and write into symmetric buffer."""
    num_tokens, hidden_dim = input_bf16.shape
    topk = topk_ids.shape[1]

    assert input_bf16.dtype == torch.bfloat16 and input_bf16.is_contiguous()
    assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
    assert topk_ids.shape[0] == num_tokens
    assert hidden_dim % 32 == 0

    if num_tokens == 0:
        return

    global includes_quant, template_quant

    args = (input_bf16, topk_ids, sym_buf,
            num_tokens, topk,
            generation,
            torch.cuda.current_stream())
    arg_defs = (
        ('input', torch.bfloat16),
        ('topk_ids', torch.int32),
        ('sym_buf_base', torch.uint8),
        ('num_tokens', int),
        ('topk', int),
        ('generation', int),
        ('stream', torch.cuda.Stream),
    )

    runtime = jit_tuner.compile_and_tune(
        name='mxfp4_quantize_to_sym_buffer',
        keys={
            'HIDDEN': hidden_dim,
            'NUM_LOCAL_EXPERTS': num_local_experts,
            'NUM_TOTAL_EXPERTS': num_total_experts,
            'MAX_TOKENS_PER_EXPERT': max_tokens_per_expert,
            'BLOCK_M': 1, 'BLOCK_N': 1, 'BLOCK_K': 1,
            'WARP_M': 1, 'WARP_N': 1, 'NUM_STAGES': 1,
        },
        space=(),
        includes=includes_quant,
        arg_defs=arg_defs,
        template=template_quant,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)


def dispatch_preprocess(
    sym_buf_addrs: torch.Tensor,
    rank_idx: int,
    num_ranks: int,
    num_local_experts: int,
    num_total_experts: int,
    max_tokens_per_expert: int,
    hidden_dim: int,
    local_expert_start: int,
    block_m: int,
    generation: int = 0,
    sync: bool = True,
    _workspace=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Read routing metadata and generate M-block scheduling for fused GEMM."""
    assert sym_buf_addrs.dtype == torch.int64
    assert sym_buf_addrs.numel() >= num_ranks

    device = sym_buf_addrs.device

    max_m_blocks_per_pair = ceil_div(max_tokens_per_expert, block_m)
    max_total_m_blocks = num_local_experts * num_ranks * max_m_blocks_per_pair

    if _workspace is not None:
        grouped_layout, remote_addr_a, remote_addr_sfa, out_total_m_blocks, out_shape_m = _workspace
    else:
        grouped_layout = torch.zeros((1 + max_total_m_blocks) * 4, dtype=torch.int32, device=device)
        remote_addr_a = torch.zeros(max_total_m_blocks, dtype=torch.int64, device=device)
        remote_addr_sfa = torch.zeros(max_total_m_blocks, dtype=torch.int64, device=device)
        out_total_m_blocks = torch.zeros(1, dtype=torch.int32, device=device)
        out_shape_m = torch.zeros(1, dtype=torch.int32, device=device)

    global includes_preprocess, template_preprocess

    args = (sym_buf_addrs, grouped_layout, remote_addr_a, remote_addr_sfa,
            out_total_m_blocks, out_shape_m,
            generation,
            torch.cuda.current_stream())

    runtime = jit_tuner.compile_and_tune(
        name='dispatch_preprocess',
        keys={
            'RANK_IDX': rank_idx,
            'NUM_RANKS': num_ranks,
            'NUM_LOCAL_EXPERTS': num_local_experts,
            'NUM_TOTAL_EXPERTS': num_total_experts,
            'MAX_TOKENS_PER_EXPERT': max_tokens_per_expert,
            'HIDDEN': hidden_dim,
            'LOCAL_EXPERT_START': local_expert_start,
            'BLOCK_M': block_m,
            'BLOCK_N': 1, 'BLOCK_K': 1,
            'WARP_M': 1, 'WARP_N': 1, 'NUM_STAGES': 1,
        },
        space=(),
        includes=includes_preprocess,
        arg_defs=(
            ('sym_buf_addrs', torch.int64),
            ('grouped_layout', torch.int32),
            ('remote_addr_a', torch.int64),
            ('remote_addr_sfa', torch.int64),
            ('out_total_m_blocks', torch.int32),
            ('out_shape_m', torch.int32),
            ('generation', int),
            ('stream', torch.cuda.Stream),
        ),
        template=template_preprocess,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)

    if sync:
        torch.cuda.current_stream().synchronize()
        total_m_blocks = out_total_m_blocks.item()
        shape_m = out_shape_m.item()
    else:
        max_m_blocks_per_pair = ceil_div(max_tokens_per_expert, block_m)
        total_m_blocks = num_local_experts * num_ranks * max_m_blocks_per_pair
        shape_m = num_local_experts * num_ranks * max_tokens_per_expert

    return grouped_layout, remote_addr_a, remote_addr_sfa, total_m_blocks, shape_m


def create_expert_preprocess_workspace(
    num_local_experts: int,
    num_ranks: int,
    max_tokens_per_expert: int,
    block_m: int,
    device,
):
    """Allocate workspace for expert-level preprocess."""
    max_expert_tokens = num_ranks * max_tokens_per_expert
    max_m_blocks_per_expert = ceil_div(max_expert_tokens, block_m)
    max_total_m_blocks = num_local_experts * max_m_blocks_per_expert
    max_rank_entries = max_total_m_blocks * num_ranks
    return (
        torch.zeros((1 + max_total_m_blocks) * 4, dtype=torch.int32, device=device),
        torch.zeros(max_rank_entries, dtype=torch.int64, device=device),
        torch.zeros(max_rank_entries, dtype=torch.int64, device=device),
        torch.zeros(max_rank_entries, dtype=torch.int32, device=device),
        torch.zeros(max_rank_entries, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )


def dispatch_expert_preprocess(
    sym_buf_addrs: torch.Tensor,
    rank_idx: int,
    num_ranks: int,
    num_local_experts: int,
    num_total_experts: int,
    max_tokens_per_expert: int,
    hidden_dim: int,
    local_expert_start: int,
    block_m: int,
    generation: int = 0,
    sync: bool = True,
    _workspace=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Expert-level preprocess: groups by expert, producing per-rank split metadata."""
    assert sym_buf_addrs.dtype == torch.int64
    device = sym_buf_addrs.device

    if _workspace is not None:
        grouped_layout, rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts, \
            out_total_m_blocks, out_shape_m = _workspace
    else:
        grouped_layout, rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts, \
            out_total_m_blocks, out_shape_m = create_expert_preprocess_workspace(
                num_local_experts, num_ranks, max_tokens_per_expert, block_m, device)

    global includes_preprocess, template_expert_preprocess

    args = (sym_buf_addrs, grouped_layout, rank_addr_a, rank_addr_sfa,
            rank_split_m, rank_counts,
            out_total_m_blocks, out_shape_m,
            generation,
            torch.cuda.current_stream())

    runtime = jit_tuner.compile_and_tune(
        name='dispatch_expert_preprocess',
        keys={
            'RANK_IDX': rank_idx,
            'NUM_RANKS': num_ranks,
            'NUM_LOCAL_EXPERTS': num_local_experts,
            'NUM_TOTAL_EXPERTS': num_total_experts,
            'MAX_TOKENS_PER_EXPERT': max_tokens_per_expert,
            'HIDDEN': hidden_dim,
            'LOCAL_EXPERT_START': local_expert_start,
            'BLOCK_M': block_m,
            'BLOCK_N': 1, 'BLOCK_K': 1,
            'WARP_M': 1, 'WARP_N': 1, 'NUM_STAGES': 1,
        },
        space=(),
        includes=includes_preprocess,
        arg_defs=(
            ('sym_buf_addrs', torch.int64),
            ('grouped_layout', torch.int32),
            ('rank_addr_a', torch.int64),
            ('rank_addr_sfa', torch.int64),
            ('rank_split_m', torch.int32),
            ('rank_counts', torch.int32),
            ('out_total_m_blocks', torch.int32),
            ('out_shape_m', torch.int32),
            ('generation', int),
            ('stream', torch.cuda.Stream),
        ),
        template=template_expert_preprocess,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)

    if sync:
        torch.cuda.current_stream().synchronize()
        total_m_blocks = out_total_m_blocks.item()
        shape_m = out_shape_m.item()
    else:
        max_expert_tokens = num_ranks * max_tokens_per_expert
        max_m_blocks_per_expert = ceil_div(max_expert_tokens, block_m)
        total_m_blocks = num_local_experts * max_m_blocks_per_expert
        shape_m = num_local_experts * num_ranks * max_tokens_per_expert

    return grouped_layout, rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts, \
           total_m_blocks, shape_m


def create_block_copy_buffers(num_local_experts, num_ranks, max_tokens_per_expert, hidden_dim, block_m, device):
    """Allocate buffers for block-copy mode: local FP4 + per-M-block completion flags.
    Last element of copy_ready_flags is reserved as atomic tile counter for work-stealing."""
    k_half = hidden_dim // 2
    local_fp4 = torch.zeros(num_local_experts * max_tokens_per_expert * k_half,
                            dtype=torch.uint8, device=device)
    max_expert_tokens = num_ranks * max_tokens_per_expert
    max_m_blocks_per_expert = ceil_div(max_expert_tokens, block_m)
    max_total_m_blocks = num_local_experts * max_m_blocks_per_expert
    copy_ready_flags = torch.zeros(max_total_m_blocks + 1, dtype=torch.int32, device=device)
    return local_fp4, copy_ready_flags


def fused_dispatch_block_copy_gemm1_fp4(
    rhs_: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    grouped_layout: torch.Tensor,
    rank_addr_a: torch.Tensor,
    rank_addr_sfa: torch.Tensor,
    rank_split_m: torch.Tensor,
    rank_counts: torch.Tensor,
    shape_m: int,
    max_tokens_per_expert: int,
    num_ranks: int,
    local_fp4_buf: torch.Tensor,
    copy_ready_flags: torch.Tensor,
    num_copy_blocks: int = 1,
    configs=None,
    merged_sfa_addrs: torch.Tensor = None,
) -> None:
    """Block-copy fused dispatch GEMM1: dedicated copy blocks + GEMM from local HBM."""
    rhs, rhs_scales = rhs_
    num_groups, n, k = rhs.shape

    if not check_mxfp4_scales_layout(scale=rhs_scales, is_sfa=False):
        rhs_scales = _post_preprocess_mxfp4_scales(scale=rhs_scales)
        if not check_mxfp4_scales_layout(scale=rhs_scales, is_sfa=False):
            rhs_scales = preprocess_mxfp4_scales(scale=rhs_scales)

    if shape_m == 0:
        return

    expected_m = ceil_div(shape_m, num_groups)

    global includes_gemm, template_gemm_block_copy
    num_sms = get_num_sms()

    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = \
            get_best_configs(shape_m, expected_m, n, k, num_groups, num_sms,
                             gemm_type=GemmType.GroupedNoPad)

    bias = torch.empty(0, dtype=torch.float32, device=rhs.device)
    if merged_sfa_addrs is None:
        merged_sfa_addrs = torch.zeros(num_groups, dtype=torch.int64, device=rhs.device)

    copy_ready_flags.zero_()

    args = (rhs, rhs_scales, bias, out, shape_m,
            grouped_layout,
            max_tokens_per_expert,
            torch.cuda.current_stream(), num_sms, smem_config[0],
            rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts,
            merged_sfa_addrs,
            local_fp4_buf, copy_ready_flags)

    runtime = jit_tuner.compile_and_tune(
        name='fused_dispatch_block_copy_gemm1_fp4',
        keys={
            'N': n, 'K': k,
            'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
            'WARP_M': warp_m, 'WARP_N': warp_n,
            'NUM_GROUPS': num_groups,
            'NUM_STAGES': num_stages,
            'NUM_RANKS': num_ranks,
            'NUM_COPY_BLOCKS': num_copy_blocks,
        },
        space=(),
        includes=includes_gemm,
        arg_defs=(
            ('rhs', torch.uint8),
            ('rhs_scales', torch.uint16),
            ('bias', torch.float32),
            ('out', torch.bfloat16),
            ('shape_m', int),
            ('grouped_layout', torch.int32),
            ('max_tokens_per_expert', int),
            ('stream', torch.cuda.Stream),
            ('num_sms', int),
            ('smem_size', int),
            ('rank_addr_a', torch.int64),
            ('rank_addr_sfa', torch.int64),
            ('rank_split_m', torch.int32),
            ('rank_counts', torch.int32),
            ('merged_sfa_addrs', torch.int64),
            ('local_fp4_buf', torch.uint8),
            ('copy_ready_flags', torch.int32),
        ),
        template=template_gemm_block_copy,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)
