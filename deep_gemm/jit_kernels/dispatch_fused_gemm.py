import os
import torch
from dataclasses import dataclass
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

# --- Expert preprocess: read routing info → generate M-block metadata ---
includes_preprocess = ('"../deep_gemm/expert_preprocess.cuh"', )
template_expert_prepare = """
using namespace deep_gemm;

launch_dispatch_expert_prepare(
    sym_buf_addrs,
    {RANK_IDX}, {NUM_RANKS},
    {NUM_LOCAL_EXPERTS}, {NUM_TOTAL_EXPERTS}, {MAX_TOKENS_PER_EXPERT}, {HIDDEN},
    {LOCAL_EXPERT_START},
    generation,
    reinterpret_cast<uint32_t*>(pair_counts),
    reinterpret_cast<uint32_t*>(masked_m),
    reinterpret_cast<uint32_t*>(out_shape_m),
    reinterpret_cast<uint32_t*>(out_expected_m),
    reinterpret_cast<uint64_t*>(dbg_cyc),
    stream);
"""

template_expert_finalize = """
using namespace deep_gemm;

launch_dispatch_expert_finalize<{BLOCK_M}>(
    sym_buf_addrs,
    {RANK_IDX}, {NUM_RANKS},
    {NUM_LOCAL_EXPERTS}, {NUM_TOTAL_EXPERTS}, {MAX_TOKENS_PER_EXPERT}, {HIDDEN},
    {LOCAL_EXPERT_START},
    generation,
    reinterpret_cast<const uint32_t*>(pair_counts),
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


template_expert_preprocess_merged = """
using namespace deep_gemm;

launch_dispatch_expert_preprocess_merged<{BLOCK_M}>(
    sym_buf_addrs,
    {RANK_IDX}, {NUM_RANKS},
    {NUM_LOCAL_EXPERTS}, {NUM_TOTAL_EXPERTS}, {MAX_TOKENS_PER_EXPERT}, {HIDDEN},
    {LOCAL_EXPERT_START},
    generation,
    reinterpret_cast<uint32_t*>(pair_counts),
    grouped_layout,
    reinterpret_cast<uint64_t*>(rank_addr_a),
    reinterpret_cast<uint64_t*>(rank_addr_sfa),
    reinterpret_cast<uint32_t*>(rank_split_m),
    reinterpret_cast<uint32_t*>(rank_counts),
    reinterpret_cast<uint32_t*>(masked_m),
    reinterpret_cast<uint32_t*>(out_total_m_blocks),
    reinterpret_cast<uint32_t*>(out_shape_m),
    reinterpret_cast<uint32_t*>(out_expected_m),
    reinterpret_cast<uint64_t*>(dbg_cyc),
    stream);
"""

# --- Standalone SFA (A-scale) gather/repack preprocess (DG_SFA_PREPROCESS) ---
# Does the SAME per-M-block SFA gather+repack as the fused copy blocks' inline
# copy_mblock_sfa, but for ALL M-blocks up front in a dedicated pre-GEMM launch,
# so the fused GEMM can run skip_sfa_copy=True (copy blocks no longer touch SFA).
# Lives in fp4_gemm_cutlass3.cuh (single source of truth for the SFA layout).
template_sfa_preprocess = """
using namespace deep_gemm;

launch_dispatch_sfa_preprocess<{BLOCK_M}, {NUM_RANKS}>(
    grouped_layout,
    reinterpret_cast<const uint64_t*>(rank_addr_sfa),
    reinterpret_cast<const uint32_t*>(rank_split_m),
    reinterpret_cast<const uint32_t*>(rank_counts),
    reinterpret_cast<uint16_t*>(local_sfa_buf),
    k_scale_blocks, max_tokens_per_expert,
    num_sms, num_threads,
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
    reinterpret_cast<uint16_t*>(local_sfa_buf),
    reinterpret_cast<volatile uint32_t*>(copy_ready_flags),
    reinterpret_cast<uint64_t*>(kstripe_profile_buf),
    kstripe_profile_max_mb,
    (bool)copy_only,
    (bool)skip_sfa_copy,
    (bool)sfa_source_host,
    {RANK_IDX});
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


def _mxfp4_quantize_to_sym_buffer(
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
    in that case. Consumers (expert prepare) poll their local slot instead of
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
        name='_mxfp4_quantize_to_sym_buffer',
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


@dataclass(frozen=True)
class ExpertPreprocessWorkspace:
    grouped_layout: torch.Tensor
    rank_addr_a: torch.Tensor
    rank_addr_sfa: torch.Tensor
    rank_split_m: torch.Tensor
    rank_counts: torch.Tensor
    masked_m: torch.Tensor
    pair_counts: torch.Tensor
    out_total_m_blocks: torch.Tensor
    out_shape_m: torch.Tensor
    out_expected_m: torch.Tensor


def create_expert_preprocess_workspace(
    num_local_experts: int,
    num_ranks: int,
    max_tokens_per_expert: int,
    device,
) -> ExpertPreprocessWorkspace:
    """Allocate BLOCK_M-independent storage for expert prepare + finalize.

    The destination staging contract caps each aggregated owner expert at
    ``max_tokens_per_expert``. Reserving one metadata entry per possible token
    is therefore sufficient for every positive BLOCK_M without guessing the
    configuration that prepare will select.
    """
    max_total_m_blocks = num_local_experts * max_tokens_per_expert
    max_rank_entries = max_total_m_blocks * num_ranks
    # Keep masked metadata in the tail of the same allocation so the fused
    # dispatch API can recover it from grouped_layout without changing every
    # legacy call site.
    grouped_layout = torch.zeros(
        (1 + max_total_m_blocks) * 4 + 2 * num_local_experts,
        dtype=torch.int32, device=device)
    masked_m = grouped_layout[-2 * num_local_experts:]
    return ExpertPreprocessWorkspace(
        grouped_layout=grouped_layout,
        rank_addr_a=torch.zeros(max_rank_entries, dtype=torch.int64, device=device),
        rank_addr_sfa=torch.zeros(max_rank_entries, dtype=torch.int64, device=device),
        rank_split_m=torch.zeros(max_rank_entries, dtype=torch.int32, device=device),
        rank_counts=torch.zeros(max_rank_entries, dtype=torch.int32, device=device),
        # [expert token counts, expert base M-block offsets]. The first half is
        # the production GroupedMasked scheduler input; the second maps its
        # expert-local tiles to the compact copy-ready flag array.
        masked_m=masked_m,
        # BLOCK_M-independent count cache, rank-major [rank, local_expert].
        pair_counts=torch.zeros(num_ranks * num_local_experts, dtype=torch.int32, device=device),
        out_total_m_blocks=torch.zeros(1, dtype=torch.int32, device=device),
        out_shape_m=torch.zeros(1, dtype=torch.int32, device=device),
        out_expected_m=torch.zeros(1, dtype=torch.int32, device=device),
    )


def dispatch_expert_prepare(
    sym_buf_addrs: torch.Tensor,
    rank_idx: int,
    num_ranks: int,
    num_local_experts: int,
    num_total_experts: int,
    max_tokens_per_expert: int,
    hidden_dim: int,
    local_expert_start: int,
    generation: int = 0,
    sync: bool = True,
    dbg_cyc: torch.Tensor = None,
    _workspace=None,
) -> Tuple:
    """BLOCK_M-independent expert count phase.

    Waits for the generation, caches per-(rank, expert) counts, writes the first
    half of masked_m, and returns total M plus max expert M.
    """
    assert sym_buf_addrs.dtype == torch.int64
    device = sym_buf_addrs.device
    if dbg_cyc is None:
        dbg_cyc = torch.zeros(4, dtype=torch.int64, device=device)  # >=3: kernel writes [0..2]
    assert dbg_cyc.dtype == torch.int64 and dbg_cyc.numel() >= 3

    if _workspace is None:
        _workspace = create_expert_preprocess_workspace(
            num_local_experts, num_ranks, max_tokens_per_expert, device)
    masked_m = _workspace.masked_m
    pair_counts = _workspace.pair_counts
    out_shape_m = _workspace.out_shape_m
    out_expected_m = _workspace.out_expected_m

    args = (sym_buf_addrs, pair_counts, masked_m, out_shape_m, out_expected_m,
            dbg_cyc, generation, torch.cuda.current_stream())

    runtime = jit_tuner.compile_and_tune(
        name='dispatch_expert_prepare',
        keys={
            'RANK_IDX': rank_idx,
            'NUM_RANKS': num_ranks,
            'NUM_LOCAL_EXPERTS': num_local_experts,
            'NUM_TOTAL_EXPERTS': num_total_experts,
            'MAX_TOKENS_PER_EXPERT': max_tokens_per_expert,
            'HIDDEN': hidden_dim,
            'LOCAL_EXPERT_START': local_expert_start,
            'BLOCK_M': 1,
            'BLOCK_N': 1, 'BLOCK_K': 1,
            'WARP_M': 1, 'WARP_N': 1, 'NUM_STAGES': 1,
        },
        space=(),
        includes=includes_preprocess,
        arg_defs=(
            ('sym_buf_addrs', torch.int64),
            ('pair_counts', torch.int32),
            ('masked_m', torch.int32),
            ('out_shape_m', torch.int32),
            ('out_expected_m', torch.int32),
            ('dbg_cyc', torch.int64),
            ('generation', int),
            ('stream', torch.cuda.Stream),
        ),
        template=template_expert_prepare,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)

    if sync:
        torch.cuda.current_stream().synchronize()
        shape_m = out_shape_m.item()
        expected_m = out_expected_m.item()
        if expected_m > max_tokens_per_expert:
            raise RuntimeError(
                f"expert token count {expected_m} exceeds staging capacity "
                f"{max_tokens_per_expert}")
    else:
        shape_m = None
        expected_m = None

    return shape_m, expected_m, masked_m


def dispatch_expert_finalize(
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
    """BLOCK_M-dependent expert layout phase using cached pair counts."""
    assert sym_buf_addrs.dtype == torch.int64
    device = sym_buf_addrs.device
    if dbg_cyc is None:
        dbg_cyc = torch.zeros(4, dtype=torch.int64, device=device)
    assert dbg_cyc.dtype == torch.int64 and dbg_cyc.numel() >= 3

    if _workspace is None:
        raise ValueError("dispatch_expert_prepare workspace is required")
    grouped_layout = _workspace.grouped_layout
    rank_addr_a = _workspace.rank_addr_a
    rank_addr_sfa = _workspace.rank_addr_sfa
    rank_split_m = _workspace.rank_split_m
    rank_counts = _workspace.rank_counts
    masked_m = _workspace.masked_m
    pair_counts = _workspace.pair_counts
    out_total_m_blocks = _workspace.out_total_m_blocks
    out_shape_m = _workspace.out_shape_m

    args = (sym_buf_addrs, pair_counts, grouped_layout, rank_addr_a, rank_addr_sfa,
            rank_split_m, rank_counts, masked_m, out_total_m_blocks, out_shape_m,
            dbg_cyc, generation, torch.cuda.current_stream())

    runtime = jit_tuner.compile_and_tune(
        name='dispatch_expert_finalize',
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
            ('pair_counts', torch.int32),
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
        template=template_expert_finalize,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)

    if sync:
        torch.cuda.current_stream().synchronize()
        total_m_blocks = out_total_m_blocks.item()
        shape_m = out_shape_m.item()
    else:
        max_m_blocks_per_expert = ceil_div(max_tokens_per_expert, block_m)
        total_m_blocks = num_local_experts * max_m_blocks_per_expert
        shape_m = num_local_experts * max_tokens_per_expert

    result = (grouped_layout, rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts,
              total_m_blocks, shape_m)
    return result + (masked_m,) if return_masked_m else result


def dispatch_expert_preprocess_merged(
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
    sync: bool = False,
    return_masked_m: bool = False,
    dbg_cyc: torch.Tensor = None,
    _workspace=None,
) -> Tuple:
    """Single-launch prepare+finalize for the fixed-BLOCK_M (masked) path.

    Equivalent to dispatch_expert_prepare() followed by dispatch_expert_finalize()
    but in one kernel launch (saves the finalize launch + kernel-boundary gap).
    Requires block_m known on host; does not support the sync=True host-readback
    contract of the split prepare (there is no intermediate stage to read back).
    """
    assert sym_buf_addrs.dtype == torch.int64
    device = sym_buf_addrs.device
    if dbg_cyc is None:
        dbg_cyc = torch.zeros(4, dtype=torch.int64, device=device)
    assert dbg_cyc.dtype == torch.int64 and dbg_cyc.numel() >= 3

    if _workspace is None:
        raise ValueError("dispatch_expert_preprocess_merged workspace is required")
    grouped_layout = _workspace.grouped_layout
    rank_addr_a = _workspace.rank_addr_a
    rank_addr_sfa = _workspace.rank_addr_sfa
    rank_split_m = _workspace.rank_split_m
    rank_counts = _workspace.rank_counts
    masked_m = _workspace.masked_m
    pair_counts = _workspace.pair_counts
    out_total_m_blocks = _workspace.out_total_m_blocks
    out_shape_m = _workspace.out_shape_m
    out_expected_m = _workspace.out_expected_m

    args = (sym_buf_addrs, pair_counts, grouped_layout, rank_addr_a, rank_addr_sfa,
            rank_split_m, rank_counts, masked_m, out_total_m_blocks, out_shape_m,
            out_expected_m, dbg_cyc, generation, torch.cuda.current_stream())

    runtime = jit_tuner.compile_and_tune(
        name='dispatch_expert_preprocess_merged',
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
            ('pair_counts', torch.int32),
            ('grouped_layout', torch.int32),
            ('rank_addr_a', torch.int64),
            ('rank_addr_sfa', torch.int64),
            ('rank_split_m', torch.int32),
            ('rank_counts', torch.int32),
            ('masked_m', torch.int32),
            ('out_total_m_blocks', torch.int32),
            ('out_shape_m', torch.int32),
            ('out_expected_m', torch.int32),
            ('dbg_cyc', torch.int64),
            ('generation', int),
            ('stream', torch.cuda.Stream),
        ),
        template=template_expert_preprocess_merged,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)

    if sync:
        torch.cuda.current_stream().synchronize()
        total_m_blocks = out_total_m_blocks.item()
        shape_m = out_shape_m.item()
    else:
        max_m_blocks_per_expert = ceil_div(max_tokens_per_expert, block_m)
        total_m_blocks = num_local_experts * max_m_blocks_per_expert
        shape_m = num_local_experts * max_tokens_per_expert

    result = (grouped_layout, rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts,
              total_m_blocks, shape_m)
    return result + (masked_m,) if return_masked_m else result


def dispatch_sfa_preprocess(
    grouped_layout: torch.Tensor,
    rank_addr_sfa: torch.Tensor,
    rank_split_m: torch.Tensor,
    rank_counts: torch.Tensor,
    local_sfa_buf: torch.Tensor,
    num_ranks: int,
    block_m: int,
    k_scale_blocks: int,
    max_tokens_per_expert: int,
    num_sms: int,
    num_threads: int = 256,
) -> None:
    """Standalone SFA (A-scale) gather/repack for ALL M-blocks (DG_SFA_PREPROCESS).

    Fills ``local_sfa_buf`` exactly as the fused copy blocks' inline
    ``copy_mblock_sfa`` would (same layout / addressing — one source of truth in
    fp4_gemm_cutlass3.cuh), but in a single pre-GEMM launch. Lets the fused GEMM
    run with ``skip_sfa_copy=True`` so its copy blocks no longer do SFA while the
    GEMM still reads the already-published ``local_sfa_buf`` (default gpu source).

    MUST run on the same stream as — and before — the fused GEMM: kernel-boundary
    completion makes the ``local_sfa_buf`` writes visible to the GEMM's SFA TMA
    (L2 domain) with no explicit fence. ``block_m`` MUST equal the fused GEMM's
    BLOCK_M (rows are placed at m_block_in_expert*BLOCK_M).
    """
    assert grouped_layout.dtype == torch.int32
    assert local_sfa_buf.dtype == torch.uint16
    global includes_gemm, template_sfa_preprocess
    args = (grouped_layout, rank_addr_sfa, rank_split_m, rank_counts, local_sfa_buf,
            int(k_scale_blocks), int(max_tokens_per_expert),
            int(num_sms), int(num_threads), torch.cuda.current_stream())

    runtime = jit_tuner.compile_and_tune(
        name='dispatch_sfa_preprocess',
        keys={
            'BLOCK_M': block_m,
            'NUM_RANKS': num_ranks,
            'BLOCK_N': 1, 'BLOCK_K': 1,
            'WARP_M': 1, 'WARP_N': 1, 'NUM_STAGES': 1,
        },
        space=(),
        includes=includes_gemm,
        arg_defs=(
            ('grouped_layout', torch.int32),
            ('rank_addr_sfa', torch.int64),
            ('rank_split_m', torch.int32),
            ('rank_counts', torch.int32),
            ('local_sfa_buf', torch.uint16),
            ('k_scale_blocks', int),
            ('max_tokens_per_expert', int),
            ('num_sms', int),
            ('num_threads', int),
            ('stream', torch.cuda.Stream),
        ),
        template=template_sfa_preprocess,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)


def create_block_copy_buffers(num_local_experts, num_ranks, max_tokens_per_expert, hidden_dim, block_m, device):
    """Allocate buffers for block-copy mode: local FP4 + local SFA + per-M-block flags."""
    k_half = hidden_dim // 2
    local_fp4 = torch.zeros(num_local_experts * max_tokens_per_expert * k_half,
                            dtype=torch.uint8, device=device)
    # GPU-side SFA staging: per-expert column-major [k_scale_blocks, max_tokens]
    # (K-stride = max_tokens). k_scale_blocks = ceil(K_packed / 32) matches the
    # GEMM's SFK and run_fused_dispatch's k_scale_blocks.
    k_scale_blocks = ceil_div(k_half, 32)
    local_sfa = torch.zeros(num_local_experts * k_scale_blocks * max_tokens_per_expert,
                            dtype=torch.uint16, device=device)
    max_m_blocks_per_expert = ceil_div(max_tokens_per_expert, block_m)
    max_total_m_blocks = num_local_experts * max_m_blocks_per_expert
    # [0..total-1] = per-M-block ready flags (read by GEMM); a copy block sets
    # copy_ready_flags[mb]=1 once it has published M-block mb's data.
    copy_ready_flags = torch.zeros(max_total_m_blocks, dtype=torch.int32, device=device)
    return local_fp4, local_sfa, copy_ready_flags


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
    local_sfa_buf: torch.Tensor,
    copy_ready_flags: torch.Tensor,
    *,
    expected_m: int,
    num_copy_blocks: int = 1,
    k_tiles_per_flag: int = 0,
    configs=None,
    merged_sfa_addrs: torch.Tensor = None,
    kstripe_profile_buf: torch.Tensor = None,
    copy_only: bool = False,
    skip_sfa_copy: bool = None,
    sfa_source_host: bool = None,
    masked_m: torch.Tensor = None,
    rank_idx: int = 0,
) -> None:
    """Block-copy fused GEMM1 with masked grouped scheduling.

    ``out`` is shaped [num_groups, padded_m, n]. The standard expert preprocess
    workspace stores ``masked_m`` (per-expert counts + base M-block) in the
    grouped-layout tail automatically.
    """
    rhs, rhs_scales = rhs_
    num_groups, n, k = rhs.shape

    if not check_mxfp4_scales_layout(scale=rhs_scales, is_sfa=False):
        rhs_scales = _post_preprocess_mxfp4_scales(scale=rhs_scales)
        if not check_mxfp4_scales_layout(scale=rhs_scales, is_sfa=False):
            rhs_scales = preprocess_mxfp4_scales(scale=rhs_scales)

    if shape_m == 0:
        return
    if expected_m <= 0 or expected_m > max_tokens_per_expert:
        raise ValueError(
            f"expected_m must be in [1, {max_tokens_per_expert}], got {expected_m}")

    # Masked is the only fused scheduling path. ``out`` is [num_groups, padded_m, n];
    # masked_m (per-expert counts + base M-block) lives in the grouped-layout tail.
    if masked_m is None:
        masked_m = grouped_layout[-2 * num_groups:]
    assert masked_m.dtype == torch.int32
    assert masked_m.is_contiguous() and masked_m.numel() >= 2 * num_groups
    assert out.dim() == 3 and out.shape[0] == num_groups and out.shape[2] == n
    gemm_shape_m = out.shape[1]
    # Tune on active rows, not padded capacity: a capacity of e.g. 256 would
    # incorrectly select block_m=256 for ~128-row experts (the old NoPad clamp bug).
    tuning_m = shape_m
    config_gemm_type = GemmType.GroupedMasked
    jit_gemm_type = 'FusedDispatchMasked'

    # NOTE: block_m here MUST match the block_m used to build grouped_layout /
    # rank_split_m in dispatch_expert_finalize. The copy kernel places each
    # M-block at m_block_in_expert * BLOCK_M rows; a mismatch writes 2nd+ blocks
    # to the wrong expert region (silently zeroing their output).
    tuning_expected_m = expected_m
    _force = int(os.getenv('FORCE_EXPECTED_M', '0'))  # diagnostic: force block_m
    if _force > 0:
        tuning_expected_m = _force

    global includes_gemm, template_gemm_block_copy
    num_sms = get_num_sms()

    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = \
            get_best_configs(tuning_m, tuning_expected_m, n, k, num_groups, num_sms,
                             gemm_type=config_gemm_type)

    # --- Optional config override (opt-in via env; defaults unchanged) ---
    # block_n 256->512 halves N_blocks -> halves wave count (288->144 tiles /
    # 36 SM = 8->4 waves; compute unchanged, bit-exact). Enable with
    # FUSED_BLOCK_N=512. Only valid at block_m==128 + warp_n=64: smem = 262016 <
    # ppu cap 262144, and warp_iter_num_sfb <= 4 (fp4_gemm_cutlass3.cuh:2124).
    # Measured 8-GPU prod (ncb=3): pipeline 0.292->0.264ms (-9.6%, 3-run stable).
    # Set FUSED_BLOCK_N=512 FUSED_WARP_N=64 together; copy path (A only) is untouched.
    _bn_ovr = int(os.getenv('FUSED_BLOCK_N', '0'))
    _bm_ovr = int(os.getenv('FUSED_BLOCK_M', '0'))
    _st_ovr = int(os.getenv('FUSED_STAGES', '0'))
    _wn_ovr = int(os.getenv('FUSED_WARP_N', '0'))
    if _bn_ovr > 0 or _bm_ovr > 0 or _st_ovr > 0 or _wn_ovr > 0:
        if _bn_ovr > 0:
            block_n = _bn_ovr
        if _bm_ovr > 0:
            block_m = _bm_ovr
        if _st_ovr > 0:
            num_stages = _st_ovr
        if _wn_ovr > 0:
            warp_n = _wn_ovr
        from .gemm_fp4 import get_smem_config_fp4
        smem_config = get_smem_config_fp4(num_stages=num_stages, block_m=block_m,
                                          block_n=block_n, warp_m=warp_m, warp_n=warp_n,
                                          block_k=block_k)
        if int(os.getenv('FUSED_CFG_VERBOSE', '0')):
            print(f"[FUSED_CFG_OVERRIDE] block_m={block_m} block_n={block_n} "
                  f"block_k={block_k} warp_m={warp_m} warp_n={warp_n} "
                  f"stages={num_stages} smem={smem_config[0]}", flush=True)

    bias = torch.empty(0, dtype=torch.float32, device=rhs.device)

    # Diagnostic A/B: DG_SFA_SOURCE=host makes the GEMM read ptr_scale_A from the
    # host-built merged_sfa (pre-0f40bad read path, K-stride = M) instead of the
    # GPU-side local_sfa_buf (K-stride = max_tokens). Explicit arg wins; otherwise
    # env — but env only flips host mode ON where the caller actually supplied
    # merged_sfa_addrs, so uninstrumented call sites stay on the GPU path (no crash
    # reading the zeroed placeholder) when the whole run is launched with the env.
    if sfa_source_host is None:
        sfa_source_host = (os.getenv('DG_SFA_SOURCE', 'gpu').lower() == 'host'
                           and merged_sfa_addrs is not None)
    if sfa_source_host and merged_sfa_addrs is None:
        raise ValueError(
            "sfa_source_host=True requires merged_sfa_addrs (host-built merged_sfa "
            "per-expert base pointers). Build it (e.g. tests' build_merged_sfa) and "
            "pass merged_sfa_addrs=.")
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

    # Diagnostic A/B: skip the GPU-side SFA copy in the copy blocks to measure its
    # exposed cost behind the FP4 copy. Output is INCORRECT when set — timing only.
    # Explicit arg wins; otherwise fall back to env SKIP_SFA_COPY (default off).
    if skip_sfa_copy is None:
        skip_sfa_copy = int(os.getenv('SKIP_SFA_COPY', '0')) != 0

    # DG_SFA_PREPROCESS: hoist the GPU-side SFA gather/repack out of the fused
    # copy blocks into a dedicated pre-GEMM kernel. Fill local_sfa_buf for all
    # M-blocks up front (same stream -> visible to the GEMM's SFA TMA), then run
    # the fused GEMM with skip_sfa_copy=True so its copy blocks no longer touch
    # SFA. The GEMM still reads local_sfa_buf (default gpu source), so output
    # stays valid. Motivation: on high-latency machines SFA's short strided
    # remote reads are exposed on the fused critical path as a serial tail after
    # the FP4 copy; a separate kernel lets them complete/overlap before the GEMM.
    # Orthogonal, host-side only (no GEMM recompile). Mutually exclusive with the
    # host SFA source (which reads remote_addr_sfa, not local_sfa_buf).
    if os.getenv('DG_SFA_PREPROCESS', '0') != '0' and not sfa_source_host:
        dispatch_sfa_preprocess(
            grouped_layout, rank_addr_sfa, rank_split_m, rank_counts, local_sfa_buf,
            num_ranks=num_ranks, block_m=block_m,
            k_scale_blocks=ceil_div(k, 32),
            max_tokens_per_expert=max_tokens_per_expert,
            num_sms=num_sms)
        skip_sfa_copy = True

    args = (rhs, rhs_scales, bias, out, gemm_shape_m,
            grouped_layout, masked_m,
            max_tokens_per_expert,
            torch.cuda.current_stream(), num_sms, smem_config[0],
            rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts,
            merged_sfa_addrs,
            local_fp4_buf, local_sfa_buf, copy_ready_flags,
            kstripe_profile_buf, kstripe_profile_max_mb,
            int(copy_only), int(skip_sfa_copy), int(sfa_source_host))

    runtime = jit_tuner.compile_and_tune(
        name='fused_dispatch_block_copy_gemm1_fp4',
        keys={
            'N': n, 'K': k,
            'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
            'WARP_M': warp_m, 'WARP_N': warp_n,
            'NUM_GROUPS': num_groups,
            'NUM_STAGES': num_stages,
            'NUM_RANKS': num_ranks,
            'RANK_IDX': rank_idx,
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
            ('local_sfa_buf', torch.uint16),
            ('copy_ready_flags', torch.int32),
            ('kstripe_profile_buf', torch.int64),
            ('kstripe_profile_max_mb', int),
            ('copy_only', int),
            ('skip_sfa_copy', int),
            ('sfa_source_host', int),
        ),
        template=template_gemm_block_copy,
        args=args,
        jit_include_dir='cutlass3',
    )
    runtime(*args)


@dataclass(frozen=True)
class BlockCopyRoundResult:
    """Output plus read-only diagnostics for one context-managed round."""
    out: torch.Tensor
    shape_m: int
    expected_m: int
    block_m: int
    generation: int
    parity: int


class BlockCopyDispatchContext:
    """Own a block-copy buffer set and its monotonic generation sequence.

    Production callers submit complete rounds through :meth:`run`; generation
    and parity are derived internally and cannot be supplied by the caller.
    The context is stream-affine so quantize, prepare, finalize, and GEMM remain
    ordered on the same double-buffer lifetime.
    """

    def __init__(
        self,
        sym_buf: torch.Tensor,
        sym_buf_addrs: torch.Tensor,
        rank_idx: int,
        num_ranks: int,
        num_local_experts: int,
        num_total_experts: int,
        max_tokens_per_expert: int,
        local_expert_start: int,
        group=None,
    ):
        if sym_buf.dtype != torch.uint8 or not sym_buf.is_contiguous():
            raise ValueError("sym_buf must be a contiguous uint8 tensor")
        if sym_buf_addrs.dtype != torch.int64 or sym_buf_addrs.numel() < num_ranks:
            raise ValueError("sym_buf_addrs must contain every rank base address")
        self._sym_buf = sym_buf
        self._sym_buf_addrs = sym_buf_addrs
        self._rank_idx = rank_idx
        self._num_ranks = num_ranks
        self._num_local_experts = num_local_experts
        self._num_total_experts = num_total_experts
        self._max_tokens = max_tokens_per_expert
        self._local_expert_start = local_expert_start
        self._workspace = create_expert_preprocess_workspace(
            num_local_experts, num_ranks, max_tokens_per_expert, sym_buf.device)
        self._buffer_cache = {}
        self._generation = 0
        self._stream = torch.cuda.current_stream(sym_buf.device)

        # Symmetric memory is uninitialized. Zero local arrival slots before any
        # peer can publish generation 1, then collectively finish initialization.
        self._sym_buf[-128:].zero_()
        self._stream.synchronize()
        if group is not None:
            import torch.distributed as dist
            dist.barrier(group=group)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def parity(self) -> int:
        return self._generation & 1

    def _check_stream(self):
        current = torch.cuda.current_stream(self._sym_buf.device)
        if current.cuda_stream != self._stream.cuda_stream:
            raise RuntimeError(
                "BlockCopyDispatchContext is stream-affine; use a separate "
                "context/buffer set for another stream")

    def _select_configs(self, shape_m, expected_m, n, k):
        # Masked-only fused path: tune on active rows, no NoPad clamp.
        tuning_expected_m = expected_m
        forced = int(os.getenv('FORCE_EXPECTED_M', '0'))
        if forced > 0:
            tuning_expected_m = forced
        return get_best_configs(
            shape_m, tuning_expected_m, n, k, self._num_local_experts,
            get_num_sms(), gemm_type=GemmType.GroupedMasked)

    def run(
        self,
        input_bf16: torch.Tensor,
        topk_ids: torch.Tensor,
        rhs_: Tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor = None,
        num_copy_blocks: int = 1,
        k_tiles_per_flag: int = 0,
    ) -> BlockCopyRoundResult:
        self._check_stream()
        rhs, _ = rhs_
        num_groups, n, k = rhs.shape
        if num_groups != self._num_local_experts:
            raise ValueError(
                f"rhs groups {num_groups} != local experts {self._num_local_experts}")
        if input_bf16.shape[1] != k * 2:
            raise ValueError(
                f"input hidden {input_bf16.shape[1]} != packed rhs K {k} * 2")

        # Allocate before launch. Advancing immediately prevents a failed round
        # from ever reusing a generation that may already have been published.
        self._generation += 1
        generation = self._generation
        _mxfp4_quantize_to_sym_buffer(
            input_bf16, topk_ids, self._sym_buf,
            num_local_experts=self._num_total_experts,
            num_total_experts=self._num_total_experts,
            max_tokens_per_expert=self._max_tokens,
            generation=generation,
            sym_buf_addrs=self._sym_buf_addrs,
            rank_idx=self._rank_idx,
            num_ranks=self._num_ranks)
        shape_m, expected_m, masked_m = dispatch_expert_prepare(
            self._sym_buf_addrs, self._rank_idx, self._num_ranks,
            self._num_local_experts, self._num_total_experts,
            self._max_tokens, input_bf16.shape[1], self._local_expert_start,
            generation=generation, _workspace=self._workspace)
        configs = self._select_configs(shape_m, expected_m, n, k)
        block_m = configs[1]
        grouped_layout, rank_addr_a, rank_addr_sfa, rank_split_m, rank_counts, \
            _, finalized_shape_m = dispatch_expert_finalize(
                self._sym_buf_addrs, self._rank_idx, self._num_ranks,
                self._num_local_experts, self._num_total_experts,
                self._max_tokens, input_bf16.shape[1], self._local_expert_start,
                block_m, generation=generation, _workspace=self._workspace)
        if finalized_shape_m != shape_m:
            raise RuntimeError(
                f"prepare/finalize shape mismatch: {shape_m} != {finalized_shape_m}")

        if out is None:
            out = torch.empty((num_groups, self._max_tokens, n),
                              dtype=torch.bfloat16, device=rhs.device)
        cache_key = (block_m, input_bf16.shape[1])
        if cache_key not in self._buffer_cache:
            self._buffer_cache[cache_key] = create_block_copy_buffers(
                self._num_local_experts, self._num_ranks, self._max_tokens,
                input_bf16.shape[1], block_m, rhs.device)
        local_fp4, local_sfa, copy_ready_flags = self._buffer_cache[cache_key]
        fused_dispatch_block_copy_gemm1_fp4(
            rhs_, out, grouped_layout, rank_addr_a, rank_addr_sfa,
            rank_split_m, rank_counts, shape_m, self._max_tokens,
            self._num_ranks, local_fp4, local_sfa, copy_ready_flags,
            expected_m=expected_m, num_copy_blocks=num_copy_blocks,
            k_tiles_per_flag=k_tiles_per_flag, configs=configs,
            masked_m=masked_m, rank_idx=self._rank_idx)
        return BlockCopyRoundResult(
            out=out, shape_m=shape_m, expected_m=expected_m,
            block_m=block_m, generation=generation, parity=generation & 1)
