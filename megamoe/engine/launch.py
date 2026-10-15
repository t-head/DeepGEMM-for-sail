import math
from typing import Optional, Tuple

import torch

from ..jit import build_runtime, render_cpp_template, generate_source

from .config import (_DISPATCH_MAX_TOPK, _W4A4_BLOCK_K, _W4A4_BLOCK_N, _read_env_int, _is_env_set, _get_kernel_config,
                     _PREFILL_TOKEN_THRESHOLD, _uses_tiled_gemm, _WARPS_PER_BLOCK, _GEMM_WARP_N, _get_topk_weight_site)
from .workspace import (MegamoeWorkspace, MegamoeWorkspaceLayout, align_up, _get_reserved_pool_tokens_per_expert, _uses_dynamic_pool,
                        get_megamoe_workspace_layout)

_SMEM_CAP_BYTES = 262144


def _get_gemm_stage_bytes(block_m: int, block_n: int, block_k: int, hidden: int, num_experts: int) -> Tuple[int, int, int, int, int]:
    expert_count_bytes = align_up(num_experts * 4, 1024)
    asf_bytes = (hidden // 128) * block_m * 4
    a_bytes = block_m * (block_k // 2)
    b_bytes = block_n * (block_k // 2)
    sfb_bytes = block_n * (block_k // 64) * 2
    return expert_count_bytes, asf_bytes, a_bytes, b_bytes, sfb_bytes


def _plan_pipeline_stages(block_m: int, block_n: int, block_k: int, hidden: int, num_experts: int,
                          stage_cap: int) -> Optional[Tuple[int, int]]:
    expert_count_bytes, asf_bytes, a_bytes, b_bytes, sfb_bytes = _get_gemm_stage_bytes(block_m, block_n, block_k, hidden, num_experts)
    for s in range(stage_cap, 1, -1):

        for a in range(s, max(1, s - 2), -1):
            if expert_count_bytes + s * b_bytes + a * (a_bytes + sfb_bytes) + asf_bytes <= _SMEM_CAP_BYTES:
                return s, a
    return None


def _get_compile_defines(gemm_stages: int,
                         m_tiles_per_warp: int = 1,
                         skip_empty_experts: int = 0,
                         experts_per_rank: int = 0,
                         k_loop_unroll: int = 1,
                         uses_dynamic_pool: bool = False,
                         is_prefill: bool = False,
                         asfb_stages: int = 0,
                         prefill_column_tiles: bool = False,
                         uses_tiled_gemm: bool = False,
                         num_ranks: int = 0) -> str:
    defines = []

    defines.append(f"#define DG_GEMM_ASFB_STAGES {asfb_stages if asfb_stages else gemm_stages}")

    if is_prefill:
        defines.append("#define DG_PREFILL 1")
    if uses_tiled_gemm and not is_prefill:
        defines.append("#define DG_TILED_GEMM 1")
    if prefill_column_tiles:
        defines.append("#define DG_PREFILL_COLUMN_TILES 1")

    if uses_dynamic_pool:
        defines.append("#define DG_DYNAMIC_POOL 1")
    defines.append(f"#define DG_TOPK_WEIGHT_SITE {_get_topk_weight_site()}")
    if gemm_stages != 2:
        defines.append(f"#define DG_GEMM_STAGES {gemm_stages}")
    if m_tiles_per_warp != 1:
        defines.append(f"#define DG_M_TILES_PER_WARP {m_tiles_per_warp}")
    if k_loop_unroll != 1:
        defines.append(f'#define DG_GEMM_K_LOOP_UNROLL _Pragma("unroll {k_loop_unroll}")')
    icn_timeout_sec = _read_env_int("DG_ICN_BARRIER_TIMEOUT_SEC", 0)
    if _is_env_set("DG_ICN_BARRIER_TIMEOUT_SEC") and icn_timeout_sec <= 0:
        raise ValueError(
            "DG_ICN_BARRIER_TIMEOUT_SEC must be positive; use DG_ICN_BARRIER_DISABLE_TIMEOUT=1 to disable the timeout")
    if icn_timeout_sec > 0:
        defines.append(f"#define DG_ICN_BARRIER_TIMEOUT_SEC {icn_timeout_sec}")
    if _read_env_int("DG_ICN_BARRIER_DISABLE_TIMEOUT", 0) != 0:
        defines.append("#define DG_ICN_BARRIER_DISABLE_TIMEOUT 1")
    if skip_empty_experts:
        defines.append("#define DG_SCHED_SKIP_EMPTY 1")
    if (m_tiles_per_warp <= 1 or (m_tiles_per_warp == 2 and 0 < experts_per_rank <= 32) or (m_tiles_per_warp == 3 and num_ranks == 16)):
        defines.append("#define DG_SCHED_LANE_MBLOCK_SUM 1")
    if m_tiles_per_warp >= 4:
        defines.append("#define DG_SCHED_L2_COST_MODEL 1")

        defines.append("#define DG_SCHED_SERIAL_MBLOCK_SUM_NO_UNROLL 1")
    return "\n".join(defines)


def _check_cuda_tensor(tensor: torch.Tensor, name: str, dtype: torch.dtype, shape: Tuple[int, ...], device: torch.device) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


_includes_fp4_fp4 = ('"megamoe/impls/ppu890_fp4_fp4_megamoe.cuh"', )

_arg_defs_fp4_fp4 = (
    ('y', torch.bfloat16),
    ('cumulative_stats', torch.int),
    ('use_cumulative_stats', int),
    ('num_tokens', int),
    ('activation_clamp', float),
    ('peer_ptrs', torch.int64),
    ('rank', int),
    ('l1_weights', torch.int8),
    ('l1_weights_sf', torch.uint16),
    ('l2_weights', torch.int8),
    ('l2_weights_sf', torch.uint16),
    ('x_bf16', torch.bfloat16),
    ('physical_push_mask_lo', int),
    ('physical_push_mask_hi', int),
    ('physical_quant_mask_lo', int),
    ('physical_quant_mask_hi', int),
    ('stream', torch.cuda.Stream),
)

_template_fp4_fp4 = r"""
using namespace megamoe;

auto kernel = ppu890_fp4_fp4_megamoe<
    {ALIGNED_MAX_TOKENS_PER_RANK},
    {DECLARED_MAX_TOKENS_PER_RANK},
    {HIDDEN}, {INTERMEDIATE_HIDDEN},
    {NUM_EXPERTS}, {NUM_TOPK},
    {NUM_EXPERTS_PER_GROUP},
    {BLOCK_M}, {BLOCK_N}, {BLOCK_K},
    {NUM_MAX_POOL_TOKENS},
    {NUM_SF_POOL_TOKENS},
    {NUM_THREADS},
    {NUM_SMS}, {NUM_RANKS},
    {HAS_ACTIVATION_CLAMP},
    {MAX_TOKENS_PER_SOURCE}, {RESERVED_M_BLOCKS_PER_EXPERT},
    {SMEM_BYTES}>;

layout::SymBuffer<{NUM_RANKS}> sym_buffer;
sym_buffer.rank_idx = static_cast<uint32_t>(rank);
sym_buffer.base = peer_ptrs[rank];
for (int i = 0; i < {NUM_RANKS}; ++i)
    sym_buffer.rank_addr_deltas[i] = peer_ptrs[i] - sym_buffer.base;

cudaError_t status = cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    {SMEM_BYTES});
if (status != cudaSuccess) {
    __return_code = static_cast<int>(status);
    return;
}

kernel<<<{NUM_SMS}, {NUM_THREADS}, {SMEM_BYTES}, stream>>>(
    reinterpret_cast<void*>(y),
    use_cumulative_stats ? cumulative_stats : nullptr,
    static_cast<uint32_t>(num_tokens),
    activation_clamp,
    sym_buffer,
    reinterpret_cast<const uint8_t*>(l1_weights),
    reinterpret_cast<const uint8_t*>(l1_weights_sf),
    reinterpret_cast<const uint8_t*>(l2_weights),
    reinterpret_cast<const uint8_t*>(l2_weights_sf),
    reinterpret_cast<const __nv_bfloat16*>(x_bf16),
    static_cast<uint32_t>(physical_push_mask_lo),
    static_cast<uint32_t>(physical_push_mask_hi),
    static_cast<uint32_t>(physical_quant_mask_lo),
    static_cast<uint32_t>(physical_quant_mask_hi));
__return_code = static_cast<int>(cudaGetLastError());
"""

_runtime_cache_fp4_fp4 = {}


def _choose_m_tiles_per_warp(expected_tokens_per_expert: float, warps_per_block: int = _WARPS_PER_BLOCK, warp_n: int = _GEMM_WARP_N) -> int:
    row_unit = max(1, warps_per_block // warp_n) * 16
    if expected_tokens_per_expert <= 2 * row_unit:
        return 2 if expected_tokens_per_expert > row_unit else 1
    if expected_tokens_per_expert <= 200:
        return 3
    return 4


def _get_experts_per_rank(num_experts: int, num_ranks: int) -> int:
    return max(1, num_experts // num_ranks)


def _check_declared_token_bound(num_tokens: int, declared_max_tokens_per_rank: int) -> None:
    if num_tokens > declared_max_tokens_per_rank:
        raise ValueError(f"num_tokens ({num_tokens}) exceeds the declared max tokens per rank "
                         f"({declared_max_tokens_per_rank})")


def _plan_tile_ladder(num_ranks: int, num_experts: int, num_topk: int, num_tokens: int):
    experts_per_rank = _get_experts_per_rank(num_experts, num_ranks)
    expected_tokens_per_expert = num_tokens * num_topk / experts_per_rank

    m_tiles_per_warp = _read_env_int("DG_M_TILES_PER_WARP", _choose_m_tiles_per_warp(expected_tokens_per_expert))
    return m_tiles_per_warp, _get_kernel_config(m_tiles_per_warp=m_tiles_per_warp)


def _plan_gemm_shape(num_ranks: int, num_experts: int, num_topk: int, hidden: int, gemm_stages: int, num_tokens: int):
    default_m_tiles_per_warp, kernel_config = _plan_tile_ladder(num_ranks, num_experts, num_topk, num_tokens)
    k_loop_unroll = 2 if (kernel_config.block_m == 96 and num_ranks <= 16) else 1

    stage_plan = _plan_pipeline_stages(kernel_config.block_m, _W4A4_BLOCK_N, _W4A4_BLOCK_K, hidden, num_experts, gemm_stages)
    if stage_plan is None:
        raise ValueError(f"no GEMM pipeline fits the {_SMEM_CAP_BYTES} B smem cap for BLOCK_M={kernel_config.block_m} "
                         f"BLOCK_N={_W4A4_BLOCK_N} BLOCK_K={_W4A4_BLOCK_K}")
    gemm_stages, asfb_stages = stage_plan
    expert_count_bytes, asf_bytes, a_bytes, b_bytes, sfb_bytes = _get_gemm_stage_bytes(kernel_config.block_m, _W4A4_BLOCK_N, _W4A4_BLOCK_K,
                                                                                       hidden, num_experts)
    gemm_smem_bytes = gemm_stages * b_bytes + asfb_stages * (a_bytes + sfb_bytes) + asf_bytes
    return default_m_tiles_per_warp, kernel_config, k_loop_unroll, gemm_stages, asfb_stages, expert_count_bytes + gemm_smem_bytes


def _make_specialization_keys(layout: MegamoeWorkspaceLayout, kernel_config, num_ranks: int, num_experts: int, num_topk: int,
                              experts_per_rank: int, hidden: int, intermediate_hidden: int, declared_max_tokens_per_rank: int, num_sms: int,
                              has_activation_clamp: bool, smem_bytes: int) -> dict:
    max_tokens_per_source = layout.max_tokens_per_source
    return {
        'ALIGNED_MAX_TOKENS_PER_RANK': layout.aligned_max_tokens_per_rank,
        'DECLARED_MAX_TOKENS_PER_RANK': declared_max_tokens_per_rank,
        'HIDDEN': hidden,
        'INTERMEDIATE_HIDDEN': intermediate_hidden,
        'NUM_EXPERTS': num_experts,
        'NUM_TOPK': num_topk,
        'NUM_EXPERTS_PER_GROUP': experts_per_rank,
        'BLOCK_M': kernel_config.block_m,
        'BLOCK_N': _W4A4_BLOCK_N,
        'BLOCK_K': _W4A4_BLOCK_K,
        'NUM_MAX_POOL_TOKENS': layout.num_max_pool_tokens,
        'NUM_SF_POOL_TOKENS': layout.num_sf_pool_tokens,
        'NUM_THREADS': kernel_config.num_threads,
        'NUM_SMS': num_sms,
        'NUM_RANKS': num_ranks,
        'HAS_ACTIVATION_CLAMP': str(has_activation_clamp).lower(),
        'MAX_TOKENS_PER_SOURCE': max(1, max_tokens_per_source),
        'RESERVED_M_BLOCKS_PER_EXPERT': _get_reserved_pool_tokens_per_expert(num_ranks, max_tokens_per_source) // kernel_config.block_m,
        'SMEM_BYTES': smem_bytes,
    }


def _build_runtime_fp4_fp4(layout: MegamoeWorkspaceLayout, num_ranks: int, num_experts: int, num_topk: int, hidden: int,
                           intermediate_hidden: int, num_tokens: int, has_activation_clamp: bool, num_sms: int, gemm_stages: int,
                           declared_max_tokens_per_rank: int):
    from .sm_topology import _choose_push_pool_size
    experts_per_rank = _get_experts_per_rank(num_experts, num_ranks)
    default_m_tiles_per_warp, kernel_config, k_loop_unroll, gemm_stages, asfb_stages, smem_bytes = _plan_gemm_shape(
        num_ranks, num_experts, num_topk, hidden, gemm_stages, num_tokens)

    _choose_push_pool_size(num_sms, num_ranks)
    if hidden != 8192:
        raise ValueError("the dispatch is laid out for hidden=8192 (32 x lines + 2 sf lines)")
    if num_topk > _DISPATCH_MAX_TOPK:
        raise ValueError(f"the dispatch needs topk <= {_DISPATCH_MAX_TOPK}: the slot header must carry the "
                         "topk pairs, the token id, the sf displaced-bit word and the slot gate within the line")
    expected_tokens_per_expert = num_tokens * num_topk / experts_per_rank

    is_prefill = declared_max_tokens_per_rank > _PREFILL_TOKEN_THRESHOLD
    prefill_column_token_limit = 2048 if num_ranks <= 8 else 4096
    uses_tiled_gemm = _uses_tiled_gemm(declared_max_tokens_per_rank, intermediate_hidden, num_ranks, num_tokens)
    if uses_tiled_gemm:
        smem_bytes = max(smem_bytes, align_up(2 * experts_per_rank * 4, 1024) + 211968)
    defines = _get_compile_defines(gemm_stages,
                                   m_tiles_per_warp=default_m_tiles_per_warp,
                                   experts_per_rank=experts_per_rank,
                                   k_loop_unroll=k_loop_unroll,
                                   uses_dynamic_pool=_uses_dynamic_pool(declared_max_tokens_per_rank),
                                   is_prefill=is_prefill,
                                   prefill_column_tiles=uses_tiled_gemm and num_tokens <= prefill_column_token_limit,
                                   uses_tiled_gemm=uses_tiled_gemm,
                                   num_ranks=num_ranks,
                                   skip_empty_experts=_read_env_int("DG_SCHED_SKIP_EMPTY",
                                                                    1 if expected_tokens_per_expert <= 0.35 else 0),
                                   asfb_stages=asfb_stages)
    keys = _make_specialization_keys(layout, kernel_config, num_ranks, num_experts, num_topk, experts_per_rank, hidden, intermediate_hidden,
                                     declared_max_tokens_per_rank, num_sms, has_activation_clamp, smem_bytes)
    cache_key = tuple(sorted(keys.items())) + (('DEFINES', defines), )
    runtime = _runtime_cache_fp4_fp4.get(cache_key)
    if runtime is None:
        code = generate_source(_includes_fp4_fp4, _arg_defs_fp4_fp4, render_cpp_template(_template_fp4_fp4, keys), preamble=defines)
        runtime = build_runtime('ppu890_fp4_fp4_megamoe', _arg_defs_fp4_fp4, code)
        _runtime_cache_fp4_fp4[cache_key] = runtime
    return runtime


def _check_launch_request(y: torch.Tensor, x_bf16: torch.Tensor, workspace: MegamoeWorkspace, recipe: Tuple[int, int, int],
                          activation: str) -> int:
    if recipe != (1, 1, 32):
        raise ValueError("PPU890 W4A4 Megamoe requires recipe=(1, 1, 32)")
    if activation != 'swiglu':
        raise ValueError("PPU890 W4A4 Megamoe currently supports only SwiGLU")
    if y.dtype != torch.bfloat16 or not y.is_cuda or not y.is_contiguous():
        raise ValueError("y must be a contiguous CUDA bfloat16 tensor")
    num_tokens = y.shape[0]
    if tuple(y.shape) != (num_tokens, workspace.hidden):
        raise ValueError(f"y must have shape [num_tokens, {workspace.hidden}]")
    if num_tokens > workspace.aligned_max_tokens_per_rank:
        raise ValueError("num_tokens exceeds workspace capacity")
    _check_declared_token_bound(num_tokens, workspace.declared_max_tokens_per_rank)
    if x_bf16 is None:
        raise ValueError("x_bf16 is required")
    if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_cuda or not x_bf16.is_contiguous():
        raise ValueError("x_bf16 must be a contiguous CUDA bfloat16 tensor")
    if tuple(x_bf16.shape) != (num_tokens, workspace.hidden):
        raise ValueError(f"x_bf16 must have shape [{num_tokens}, {workspace.hidden}]")
    return num_tokens


def _check_weight_tensors(l1_weights: Tuple[torch.Tensor, torch.Tensor], l2_weights: Tuple[torch.Tensor, torch.Tensor],
                          workspace: MegamoeWorkspace, device: torch.device) -> int:
    l1_weight, l1_weight_sf = l1_weights
    l2_weight, l2_weight_sf = l2_weights
    num_experts_per_rank = workspace.num_experts // workspace.num_ranks
    _check_cuda_tensor(l1_weight, "l1_weight", torch.int8, (num_experts_per_rank, 2 * workspace.intermediate_hidden, workspace.hidden // 2),
                       device)
    _check_cuda_tensor(l1_weight_sf, "l1_weight_sf", torch.uint16,
                       (num_experts_per_rank, workspace.hidden // 64, 2 * workspace.intermediate_hidden), device)
    _check_cuda_tensor(l2_weight, "l2_weight", torch.int8, (num_experts_per_rank, workspace.hidden, workspace.intermediate_hidden // 2),
                       device)
    _check_cuda_tensor(l2_weight_sf, "l2_weight_sf", torch.uint16,
                       (num_experts_per_rank, workspace.intermediate_hidden // 64, workspace.hidden), device)
    return num_experts_per_rank


def _resolve_cumulative_stats(cumulative_local_expert_recv_counts: Optional[torch.Tensor], num_experts_per_rank: int,
                              device: torch.device) -> Tuple[torch.Tensor, int]:
    if cumulative_local_expert_recv_counts is None:
        return torch.zeros(num_experts_per_rank, dtype=torch.int32, device=device), 0
    return cumulative_local_expert_recv_counts, 1


def _build_role_masks(device: torch.device, num_sms: int, num_ranks: int) -> Tuple[int, int]:
    from .sm_topology import (_detect_dead_sm, _select_push_smids, _choose_push_pool_size, _get_live_smids, _choose_idle_sm)
    dead_smid = _detect_dead_sm(device)
    num_push_sms = _choose_push_pool_size(num_sms, num_ranks)
    if _read_env_int("DG_MOE_PUSH_DIE_BALANCED", 1):
        push_smids = _select_push_smids(device, num_push_sms)
    else:
        from .sm_topology import _TOTAL_PHYSICAL_SMS
        push_smids = [s for s in range(_TOTAL_PHYSICAL_SMS) if s != dead_smid][:num_push_sms]

    live_smids = _get_live_smids(dead_smid)
    idle_smid = _choose_idle_sm(dead_smid)
    quant_smids = [s for s in live_smids if s != idle_smid]
    assert idle_smid not in push_smids, (f"the idle SM {idle_smid} is also a push SM: it belongs to neither role set, so it skips the "
                                         "dispatch handoff and must not own a push share")

    physical_push_mask = 0
    physical_quant_mask = 0
    for s in push_smids:
        physical_push_mask |= 1 << s
    for s in quant_smids:
        physical_quant_mask |= 1 << s
    assert all(s < 64 for s in push_smids + quant_smids), "role masks are two 32-bit words per set, so smids must be < 64"
    return physical_push_mask, physical_quant_mask


def _make_launch_args(y: torch.Tensor, cumulative_stats: torch.Tensor, use_cumulative_stats: int, num_tokens: int,
                      activation_clamp: Optional[float], workspace: MegamoeWorkspace, l1_weights: Tuple[torch.Tensor, torch.Tensor],
                      l2_weights: Tuple[torch.Tensor,
                                        torch.Tensor], x_bf16: torch.Tensor, physical_push_mask: int, physical_quant_mask: int) -> Tuple:
    l1_weight, l1_weight_sf = l1_weights
    l2_weight, l2_weight_sf = l2_weights
    clamp = float('inf') if activation_clamp is None else float(activation_clamp)
    return (
        y,
        cumulative_stats,
        use_cumulative_stats,
        num_tokens,
        clamp,
        workspace.peer_ptrs,
        workspace.rank,
        l1_weight,
        l1_weight_sf,
        l2_weight,
        l2_weight_sf,
        x_bf16,
        physical_push_mask & 0xFFFFFFFF,
        physical_push_mask >> 32,
        physical_quant_mask & 0xFFFFFFFF,
        physical_quant_mask >> 32,
        torch.cuda.current_stream(y.device),
    )


def compile_fp4_fp4_megamoe(num_ranks: int,
                            num_experts: int,
                            aligned_max_tokens_per_rank: int,
                            num_topk: int,
                            hidden: int,
                            intermediate_hidden: int,
                            num_tokens: int,
                            activation_clamp: Optional[float] = None,
                            num_sms: Optional[int] = None,
                            gemm_stages: Optional[int] = None,
                            declared_max_tokens_per_rank: Optional[int] = None):
    if declared_max_tokens_per_rank is None:
        declared_max_tokens_per_rank = aligned_max_tokens_per_rank

    layout = get_megamoe_workspace_layout(num_ranks,
                                          num_experts,
                                          aligned_max_tokens_per_rank,
                                          num_topk,
                                          hidden,
                                          intermediate_hidden,
                                          declared_max_tokens_per_rank=declared_max_tokens_per_rank)

    if not 0 <= num_tokens <= min(layout.aligned_max_tokens_per_rank, layout.max_tokens_per_source):
        raise ValueError("num_tokens exceeds the workspace's per-rank token capacity")
    _check_declared_token_bound(num_tokens, declared_max_tokens_per_rank)

    from .sm_topology import _get_num_sms
    num_sms = _get_num_sms(num_sms)

    _, _predicted_tile = _plan_tile_ladder(num_ranks, num_experts, num_topk, num_tokens)
    _default_stage_count = 3 if _plan_pipeline_stages(_predicted_tile.block_m, _W4A4_BLOCK_N, _W4A4_BLOCK_K, hidden, num_experts, 3) else 2
    gemm_stages = (_read_env_int("DG_GEMM_STAGES", _default_stage_count) if gemm_stages is None else gemm_stages)
    if gemm_stages < 2:
        raise ValueError("gemm_stages must be at least 2")
    return _build_runtime_fp4_fp4(layout, num_ranks, num_experts, num_topk, hidden, intermediate_hidden, num_tokens,
                                  activation_clamp is not None and math.isfinite(activation_clamp), num_sms, gemm_stages,
                                  declared_max_tokens_per_rank)


def launch_fp4_fp4_megamoe(y: torch.Tensor,
                           l1_weights: Tuple[torch.Tensor, torch.Tensor],
                           l2_weights: Tuple[torch.Tensor, torch.Tensor],
                           workspace: MegamoeWorkspace,
                           cumulative_local_expert_recv_counts: Optional[torch.Tensor] = None,
                           recipe: Tuple[int, int, int] = (1, 1, 32),
                           activation: str = 'swiglu',
                           activation_clamp: Optional[float] = None,
                           num_sms: Optional[int] = None,
                           gemm_stages: Optional[int] = None,
                           x_bf16: torch.Tensor = None) -> None:
    num_tokens = _check_launch_request(y, x_bf16, workspace, recipe, activation)
    num_experts_per_rank = _check_weight_tensors(l1_weights, l2_weights, workspace, y.device)
    cumulative_stats, use_cumulative_stats = _resolve_cumulative_stats(cumulative_local_expert_recv_counts, num_experts_per_rank, y.device)
    from .sm_topology import _get_num_sms
    num_sms = _get_num_sms(num_sms)
    runtime = compile_fp4_fp4_megamoe(workspace.num_ranks,
                                      workspace.num_experts,
                                      workspace.aligned_max_tokens_per_rank,
                                      workspace.num_topk,
                                      workspace.hidden,
                                      workspace.intermediate_hidden,
                                      num_tokens,
                                      declared_max_tokens_per_rank=workspace.declared_max_tokens_per_rank,
                                      activation_clamp=activation_clamp,
                                      num_sms=num_sms,
                                      gemm_stages=gemm_stages)
    uses_tiled_gemm = _uses_tiled_gemm(workspace.declared_max_tokens_per_rank, workspace.intermediate_hidden, workspace.num_ranks,
                                       num_tokens)
    workspace._select_gemm_scale_views(uses_tiled_gemm)
    if uses_tiled_gemm:
        from .weights import prepare_gemm_weights
        l1_weights, l2_weights = prepare_gemm_weights(l1_weights, l2_weights, workspace)
    physical_push_mask, physical_quant_mask = _build_role_masks(y.device, num_sms, workspace.num_ranks)
    args = _make_launch_args(y, cumulative_stats, use_cumulative_stats, num_tokens, activation_clamp, workspace, l1_weights, l2_weights,
                             x_bf16, physical_push_mask, physical_quant_mask)
    return_code = runtime(*args)
    if return_code != 0:
        raise RuntimeError(f"PPU890 W4A4 Megamoe launch failed with CUDA error code {return_code}")
