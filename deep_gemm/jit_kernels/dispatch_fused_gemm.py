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
    sym_buf_addrs, rank_idx, num_ranks,
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
    reinterpret_cast<uint32_t*>(masked_m),
    reinterpret_cast<uint32_t*>(out_total_m_blocks),
    reinterpret_cast<uint32_t*>(out_shape_m),
    reinterpret_cast<uint64_t*>(dbg_cyc),
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
                        kNumGroups, kNumStages, GemmType::{GEMM_TYPE}>;
gemm_t::template run_fused_dispatch<{NUM_RANKS}, {NUM_COPY_BLOCKS}, {K_TILES_PER_FLAG}>(
    rhs, rhs_scales, bias, out, shape_m,
    grouped_layout, masked_m,
    max_tokens_per_expert,
    stream, num_sms, smem_size,
    reinterpret_cast<const uint64_t*>(rank_addr_a),
    reinterpret_cast<const uint64_t*>(rank_addr_sfa),
    reinterpret_cast<const uint32_t*>(rank_split_m),
    reinterpret_cast<const uint32_t*>(rank_counts),
    reinterpret_cast<const uint64_t*>(merged_sfa_addrs),
    reinterpret_cast<uint8_t*>(local_fp4_buf),
    reinterpret_cast<volatile uint32_t*>(copy_ready_flags),
    reinterpret_cast<uint64_t*>(kstripe_profile_buf),
    kstripe_profile_max_mb,
    (bool)copy_only,
    (uint32_t)copy_mode);
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
    data_region = metadata + fp4_data + scales
    # Double-buffered data region (parity = generation & 1) so a 1-iteration-ahead producer
    # cannot overwrite the buffer a slower consumer is still reading. Must match
    # DispatchBufferLayout::ready_flag_offset() (== align16(2 * total_bytes())).
    flag_offset = ((2 * data_region + 15) // 16) * 16
    # Arrival slots: one uint32 per rank (atomic-arrival barrier). 128B = up to 32 ranks.
    return flag_offset + 128


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
    sym_buf_addrs: torch.Tensor = None,
    rank_idx: int = 0,
    num_ranks: int = 1,
) -> None:
    """Quantize BF16 activations to MXFP4 and write into symmetric buffer.

    When generation > 0, an atomic-arrival kernel pushes `generation` into every
    consumer's local slot[rank_idx]; sym_buf_addrs (peer base ptrs, int64) is required
    in that case. Consumers (dispatch_preprocess) poll their local slot instead of
    spinning on a remote NVLink flag.
    """
    num_tokens, hidden_dim = input_bf16.shape
    topk = topk_ids.shape[1]

    assert input_bf16.dtype == torch.bfloat16 and input_bf16.is_contiguous()
    assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
    assert topk_ids.shape[0] == num_tokens
    assert hidden_dim % 32 == 0
    if sym_buf_addrs is None:
        # No peers to signal (generation must be 0 in this case); dummy keeps arg_defs stable.
        assert generation == 0, "sym_buf_addrs required when generation > 0"
        sym_buf_addrs = torch.zeros(1, dtype=torch.int64, device=input_bf16.device)
    assert sym_buf_addrs.dtype == torch.int64

    if num_tokens == 0:
        return

    global includes_quant, template_quant

    args = (input_bf16, topk_ids, sym_buf,
            num_tokens, topk,
            generation,
            sym_buf_addrs, rank_idx, num_ranks,
            torch.cuda.current_stream())
    arg_defs = (
        ('input', torch.bfloat16),
        ('topk_ids', torch.int32),
        ('sym_buf_base', torch.uint8),
        ('num_tokens', int),
        ('topk', int),
        ('generation', int),
        ('sym_buf_addrs', torch.int64),
        ('rank_idx', int),
        ('num_ranks', int),
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
    # Keep masked metadata in the tail of the same allocation so the fused
    # dispatch API can recover it from grouped_layout without changing every
    # legacy call site.
    grouped_layout = torch.zeros(
        (1 + max_total_m_blocks) * 4 + 2 * num_local_experts,
        dtype=torch.int32, device=device)
    masked_m = grouped_layout[-2 * num_local_experts:]
    return (
        grouped_layout,
        torch.zeros(max_rank_entries, dtype=torch.int64, device=device),
        torch.zeros(max_rank_entries, dtype=torch.int64, device=device),
        torch.zeros(max_rank_entries, dtype=torch.int32, device=device),
        torch.zeros(max_rank_entries, dtype=torch.int32, device=device),
        # [expert token counts, expert base M-block offsets]. The first half is
        # the production GroupedMasked scheduler input; the second maps its
        # expert-local tiles to the compact copy-ready flag array.
        masked_m,
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
    return_masked_m: bool = False,
    dbg_cyc: torch.Tensor = None,
    _workspace=None,
) -> Tuple:
    """Expert-level preprocess: groups by expert, producing per-rank split metadata.

    dbg_cyc: optional int64[>=3] buffer. If given, kernel writes per-phase cycle
    counts [setup+barrier, remote-count-reads, compute(Phase4+5)] (thread 0).
    """
    assert sym_buf_addrs.dtype == torch.int64
    device = sym_buf_addrs.device
    if dbg_cyc is None:
        dbg_cyc = torch.zeros(4, dtype=torch.int64, device=device)  # >=3: kernel writes [0..2]
    assert dbg_cyc.dtype == torch.int64 and dbg_cyc.numel() >= 3

    if _workspace is not None:
        grouped_layout, rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts, masked_m, \
            out_total_m_blocks, out_shape_m = _workspace
    else:
        grouped_layout, rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts, masked_m, \
            out_total_m_blocks, out_shape_m = create_expert_preprocess_workspace(
                num_local_experts, num_ranks, max_tokens_per_expert, block_m, device)

    global includes_preprocess, template_expert_preprocess

    args = (sym_buf_addrs, grouped_layout, rank_addr_a, rank_addr_sfa,
            rank_split_m, rank_counts, masked_m,
            out_total_m_blocks, out_shape_m, dbg_cyc,
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
            ('masked_m', torch.int32),
            ('out_total_m_blocks', torch.int32),
            ('out_shape_m', torch.int32),
            ('dbg_cyc', torch.int64),
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

    result = (grouped_layout, rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts,
              total_m_blocks, shape_m)
    return result + (masked_m,) if return_masked_m else result


def create_block_copy_buffers(num_local_experts, num_ranks, max_tokens_per_expert, hidden_dim, block_m, device):
    """Allocate buffers for block-copy mode: local FP4 + per-M-block completion flags."""
    k_half = hidden_dim // 2
    local_fp4 = torch.zeros(num_local_experts * max_tokens_per_expert * k_half,
                            dtype=torch.uint8, device=device)
    max_expert_tokens = num_ranks * max_tokens_per_expert
    max_m_blocks_per_expert = ceil_div(max_expert_tokens, block_m)
    max_total_m_blocks = num_local_experts * max_m_blocks_per_expert
    # 2x size: [0..total-1] = per-M-block ready flags (read by GEMM),
    # [total..2*total-1] = per-M-block done-counters for cooperative copy
    # (copy_mode=1); the last of ncb blocks to finish an M-block sets its flag.
    copy_ready_flags = torch.zeros(2 * max_total_m_blocks, dtype=torch.int32, device=device)
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
    k_tiles_per_flag: int = 0,
    configs=None,
    merged_sfa_addrs: torch.Tensor = None,
    kstripe_profile_buf: torch.Tensor = None,
    copy_only: bool = False,
    copy_mode: int = 0,
    masked_m: torch.Tensor = None,
) -> None:
    """Block-copy fused GEMM1 with selectable NoPad or Masked scheduling.

    Set FUSED_GEMM_GROUPING=nopad|masked (default: nopad). Masked mode expects
    ``out`` shaped [num_groups, padded_m, n]. The standard expert preprocess
    workspace stores ``masked_m`` in the grouped-layout tail automatically.
    NoPad retains the existing compact [shape_m, n] output contract.
    """
    rhs, rhs_scales = rhs_
    num_groups, n, k = rhs.shape

    if not check_mxfp4_scales_layout(scale=rhs_scales, is_sfa=False):
        rhs_scales = _post_preprocess_mxfp4_scales(scale=rhs_scales)
        if not check_mxfp4_scales_layout(scale=rhs_scales, is_sfa=False):
            rhs_scales = preprocess_mxfp4_scales(scale=rhs_scales)

    if shape_m == 0:
        return

    grouping = os.getenv('FUSED_GEMM_GROUPING', 'nopad').strip().lower()
    if grouping not in ('nopad', 'masked'):
        raise ValueError(f"FUSED_GEMM_GROUPING must be 'nopad' or 'masked', got {grouping!r}")
    use_masked = grouping == 'masked'

    if use_masked:
        if masked_m is None:
            masked_m = grouped_layout[-2 * num_groups:]
        assert masked_m.dtype == torch.int32
        assert masked_m.is_contiguous() and masked_m.numel() >= 2 * num_groups
        assert out.dim() == 3 and out.shape[0] == num_groups and out.shape[2] == n
        gemm_shape_m = out.shape[1]
        # Keep the padded storage/output stride independent from the tuning
        # workload. The production masked path tunes on active rows, otherwise
        # a capacity of 256 incorrectly selects block_m=256 for ~128-row experts.
        expected_m = ceil_div(shape_m, num_groups)
        tuning_m = shape_m
        config_gemm_type = GemmType.GroupedMasked
        jit_gemm_type = 'FusedDispatchMasked'
    else:
        assert out.dim() == 2 and out.shape == (shape_m, n)
        # The masked pointer is compile-time dead on the NoPad path. Reuse an
        # existing int32 tensor to keep the legacy call contract allocation-free.
        if masked_m is None:
            masked_m = grouped_layout
        gemm_shape_m = shape_m
        expected_m = ceil_div(shape_m, num_groups)
        tuning_m = shape_m
        config_gemm_type = GemmType.GroupedNoPad
        jit_gemm_type = 'FusedDispatch'

    # NOTE: block_m here MUST match the block_m used to build grouped_layout /
    # rank_split_m in dispatch_expert_preprocess. The copy kernel places each
    # M-block at m_block_in_expert * BLOCK_M rows; a mismatch writes 2nd+ blocks
    # to the wrong expert region (silently zeroing their output). The preprocess
    # side (get_gemm_block_m) uses plain expected_m with no <=128 bump, so we
    # must not bump here either.
    if not use_masked and expected_m <= 128:  # legacy NoPad tuning clamp
        expected_m = 129
    _force = int(os.getenv('FORCE_EXPECTED_M', '0'))  # A/B: 强制 block_m(须与 test 侧 get_gemm_block_m 一致)
    if _force > 0:
        expected_m = _force

    global includes_gemm, template_gemm_block_copy
    num_sms = get_num_sms()

    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = \
            get_best_configs(tuning_m, expected_m, n, k, num_groups, num_sms,
                             gemm_type=config_gemm_type)

    bias = torch.empty(0, dtype=torch.float32, device=rhs.device)
    if merged_sfa_addrs is None:
        merged_sfa_addrs = torch.zeros(num_groups, dtype=torch.int64, device=rhs.device)

    copy_ready_flags.zero_()

    # Per-tile k-stripe wait profiling. Pass a zeroed int64 tensor of length
    # 4 * (>= num tiles = num_m_blocks * num_n_blocks). Record per tile_idx:
    #   [tile*4+0] = P2P stripe-wait cycles      [tile*4+1] = mainloop cycles
    #   [tile*4+2] = wave number (current_iter)  [tile*4+3] = GEMM CTA id
    # Group rows (compute>0) by wave to see first-wave vs later-wave P2P exposure.
    if kstripe_profile_buf is None:
        kstripe_profile_buf = torch.empty(0, dtype=torch.int64, device=rhs.device)
        kstripe_profile_max_mb = 0
    else:
        kstripe_profile_buf.zero_()
        kstripe_profile_max_mb = kstripe_profile_buf.numel() // 4  # capacity in tiles

    args = (rhs, rhs_scales, bias, out, gemm_shape_m,
            grouped_layout, masked_m,
            max_tokens_per_expert,
            torch.cuda.current_stream(), num_sms, smem_config[0],
            rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts,
            merged_sfa_addrs,
            local_fp4_buf, copy_ready_flags,
            kstripe_profile_buf, kstripe_profile_max_mb,
            int(copy_only), int(copy_mode))

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
            'K_TILES_PER_FLAG': k_tiles_per_flag,
            'GEMM_TYPE': jit_gemm_type,
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
            ('masked_m', torch.int32),
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
            ('kstripe_profile_buf', torch.int64),
            ('kstripe_profile_max_mb', int),
            ('copy_only', int),
            ('copy_mode', int),
        ),
        template=template_gemm_block_copy,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)
