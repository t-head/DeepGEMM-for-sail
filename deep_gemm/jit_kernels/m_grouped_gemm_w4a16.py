import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, is_ppu1v5_device

# C++ code templates
includes = ('"../deep_gemm/w4a16_gemm_cutlass3.cuh"', )
w4_a16_template = """
using namespace deep_gemm;
using ElementA = cutlass::bfloat16_t;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto WARP_K = {WARP_K};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kGroupSize = {GROUP_SIZE};

// Make a templated grouped GEMM
using gemm_t = W4A16Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kGroupSize>;
gemm_t::run((const ElementA*) lhs, rhs, (const ElementA*) rhs_scales,
            (ElementA*) out, m, grouped_layout, block_m_info, sorted_token_ids,
            expected_m, stream, num_sms);
"""

def get_w4a16_config(gemm_type, expected_m, n, k, group_size):
    num_sms = get_num_sms()
    for block_m in [16, 32, 64, 128]:
        if expected_m / block_m < 0.9: break
    warp_m = block_m if block_m <= 64 else block_m // 2
    # Now warps_on_k is only tested on group_size == 32, and it is not compatible with fused async_copy loadA (too many threads)
    warps_on_k = (block_m == warp_m and group_size == 32 and gemm_type != "GroupedFused")
    if warps_on_k and k >= 2048:
        if is_ppu1v5_device(): # warps_on_n = 4, warps_on_k = 4
            block_n, warp_n, block_k, warp_k = 256, 64, 128, 32
        else: # warps_on_n = 2, warps_on_k = 8
            block_n, warp_n, block_k, warp_k = 128, 64, 256, 32
    else:
        block_n, warp_n, block_k, warp_k = 256, 64, 64, 64
    # small tile for debug
    # block_m, warp_m = 16, 16
    # block_n, warp_n = 128, 64
    # block_k, warp_k = 64, 32
    num_stages = 3
    configs = (num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages)
    return configs


def m_grouped_gemm_w4a16_common(gemm_type: str, expected_m: int,
                                lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                group_size: int, configs,
                                m_indices: torch.Tensor, m_rows: torch.Tensor,
                                sorted_token_ids: torch.Tensor = None):
    """
    W4A16 grouped GEMM without padding.

    Args:
        lhs: Activation tensor in BF16, shape nopad: (m, k) or fused: (num_token, k)
        rhs_: Tuple of (weight, scale)
            - weight: 4-bit weight stored in int32, shape (num_groups, k // 16, n * 2)
            - scale: Per-channel scale in BF16, shape (num_groups, k // group_size), n)
        out: Output tensor in BF16, shape (m, n)
        m_indices: Group indices for each row of lhs, shape nopad: (m,) or fused: (num_token, topk)
        m_rows: Number of rows per group, shape (num_groups,)
        group_size: Group size for quantization (default 32)
        configs: Optional pre-configured kernel parameters
    """
    rhs, rhs_scales = rhs_
    m, k = lhs.shape
    num_groups, _, n2 = rhs.shape
    n = n2 // 2
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    if m == 0: return
    assert k % group_size == 0, f"K must be a multiple of group_size, got k={k}, group_size={group_size}"
    assert m_ == m__ and n == n_
    assert rhs.shape == (num_groups, k // 16, n * 2), f"Weights shape {rhs.shape} != ({num_groups}, {k // 16}, {n * 2})"
    assert rhs_scales.shape == (num_groups, k // group_size, n), f"Scale shape {rhs_scales.shape} != ({num_groups}, {k // group_size}, {n})"
    assert n > 0 and k > 0
    assert lhs.dtype == torch.bfloat16
    assert rhs.dtype == torch.int32
    assert rhs_scales.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()
    assert rhs_scales.is_contiguous() and m_indices.is_contiguous()

    num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages = configs
    assert warp_n == 64
    block_m_info = torch.empty((num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4, dtype=torch.int32, device=m_rows.device)

    if sorted_token_ids is None:
        sorted_token_ids = torch.empty(0, dtype=torch.int32, device=m_rows.device)

    args = (lhs, rhs, rhs_scales, out, m, m_rows, block_m_info, sorted_token_ids, expected_m, torch.cuda.current_stream(), num_sms)
    global includes, w4_a16_template
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_w4a16',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n, 'WARP_K': warp_k, 'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages, 'GROUP_SIZE': group_size, 'GEMM_TYPE': gemm_type},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.bfloat16), ('rhs', torch.int32), ('rhs_scales', torch.bfloat16),
                  ('out', torch.bfloat16),
                  ('m', int), ('grouped_layout', torch.int32), ('block_m_info', torch.int32), ('sorted_token_ids', torch.int32),
                  ('expected_m', int), ('stream', torch.cuda.Stream), ('num_sms', int)),
        template=w4_a16_template,
        args=args,
        jit_include_dir='cutlass3'
    )

    runtime(*args)
    return

def m_grouped_gemm_w4a16_fused(lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                m_indices: torch.Tensor, m_rows: torch.Tensor = None,
                                sorted_token_ids: torch.Tensor = None,
                                configs = None):
    num_token, topk = m_indices.shape
    num_groups = rhs_[0].shape[0]
    assert num_groups >= topk
    m = m_indices.numel()
    expected_m = ceil_div(m, num_groups)
    n = rhs_[1].shape[2]
    k = lhs.shape[1]
    group_size = lhs.shape[1] // rhs_[1].shape[1]
    if configs is None:
        configs = get_w4a16_config("GroupedFused", expected_m, n, k, group_size)
    block_m = configs[1]

    # TODO: use cuda kernel to calculate sorted_token_ids
    if m_rows is None and sorted_token_ids is None:
        # m_rows
        m_rows = torch.zeros(num_groups, dtype=torch.int32).to('cpu')
        for eid in m_indices.to('cpu'):
            m_rows[eid] += 1

        # expert_ids
        expert_ids_v = []
        # for m in m_rows:
        for eid, m_per_expert in enumerate(m_rows):
            num_blocks_m_per_expert = (m_per_expert + block_m - 1) // block_m
            expert_ids_v.extend([eid] * num_blocks_m_per_expert)
        expert_ids = torch.tensor(expert_ids_v, dtype=torch.int32).to('cuda')

        # sorted_token_ids
        num_blocks_m = expert_ids.shape[0]
        # 初始化 sorted_token_ids，用 num_token 填充
        sorted_token_ids = torch.full((num_blocks_m, block_m), num_token,
                                    dtype=torch.int32, device='cpu')
        # 预先计算每个 expert 的 token 列表
        expert_tokens = {}
        for expert_id in range(num_groups):
            # 找出哪些 token 选择了这个 expert
            mask = (m_indices == expert_id)  # [num_token, topk]
            selected = mask.any(dim=1)  # [num_token]
            expert_tokens[expert_id] = torch.where(selected)[0]  # [num_selected]
        # 记录每个 expert 已经处理了多少个 token（用于 block 切分）
        expert_token_pos = torch.zeros(num_groups, dtype=torch.int32, device='cpu')
        for block_idx in range(num_blocks_m):
            expert_id = expert_ids[block_idx].item()
            # 获取该 expert 的所有 token
            tokens = expert_tokens[expert_id]
            # 计算该 block 的起始位置
            start_pos = expert_token_pos[expert_id].item()
            # 计算该 block 处理的 token 范围
            end_pos = min(start_pos + block_m, len(tokens))
            # 复制 token 索引
            count = end_pos - start_pos
            sorted_token_ids[block_idx, :count] = tokens[start_pos:end_pos]
            # 更新该 expert 的 token 位置
            expert_token_pos[expert_id] += count
        sorted_token_ids = sorted_token_ids.to('cuda')
        m_rows = m_rows.to('cuda')

    return m_grouped_gemm_w4a16_common("GroupedFused", expected_m, lhs, rhs_, out, group_size, configs, m_indices, m_rows, sorted_token_ids)


def m_grouped_gemm_w4a16_nopad(lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                m_indices: torch.Tensor, m_rows: torch.Tensor = None,
                                configs = None):
    num_groups = rhs_[0].shape[0]
    m = lhs.shape[0]
    group_size = lhs.shape[1] // rhs_[1].shape[1]
    expected_m = ceil_div(m, num_groups)
    n = rhs_[1].shape[2]
    k = lhs.shape[1]
    if configs is None:
        configs = get_w4a16_config("GroupedNoPad", expected_m, n, k, group_size)
    if m_rows is None:
        counts = torch.bincount(m_indices)
        min_n = min(counts.size(0), num_groups)
        experts_for_rows = torch.zeros(num_groups, dtype=torch.int32, device='cuda')
        if min_n > 0:
            experts_for_rows[:min_n] = counts[:min_n]
        m_rows = experts_for_rows
    return m_grouped_gemm_w4a16_common("GroupedNoPad", expected_m, lhs, rhs_, out, group_size, configs, m_indices, m_rows, None)


def m_grouped_gemm_w4a16_masked(lhs: torch.Tensor,
                                rhs_: Tuple[torch.Tensor, torch.Tensor],
                                out: torch.Tensor,
                                masked_m: torch.Tensor, expected_m: int, group_size: int = 32, configs=None):
    pass
