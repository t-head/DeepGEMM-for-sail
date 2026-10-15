"""High-throughput expert-layout permutation helpers for Megamoe tests."""

from typing import Tuple

import torch
import triton
import triton.language as tl

from megamoe.engine.workspace import align_up

# The grouped GEMM's contiguous-layout M alignment: ht_permute pads every expert's row count up to it,
# which is what the deep_gemm grouped GEMM expects of the m_rows it is handed.
CONTIGUOUS_M_ALIGNMENT = 128


@triton.jit
def _ep_scatter_layout_kernel(
    num_recv_tokens_per_expert,
    expert_start_offset,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + expert_offsets,
        mask=expert_offsets < num_experts,
        other=0,
    )
    tokens_per_expert = (tokens_per_expert + BLOCK_E - 1) // BLOCK_E * BLOCK_E
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(
        expert_start_offset + expert_offsets,
        cumsum,
        mask=expert_offsets < num_experts,
    )

    cur_expert_start = tl.load(expert_start_offset + cur_expert)
    cur_expert_token_count = tl.load(num_recv_tokens_per_expert + cur_expert)
    m_indices_start_ptr = m_indices + cur_expert_start
    offset_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_count, BLOCK_E, num_stages=4):
        tl.store(m_indices_start_ptr + start_m + offset_expert, cur_expert)


@triton.jit
def _ep_scatter_tokens_kernel(
    total_token_count,
    expert_start_offset,
    recv_x,
    recv_x_stride0,
    recv_x_sf,
    recv_x_sf_stride0,
    recv_topk_idx,
    recv_topk_idx_stride0,
    output_tensor,
    output_tensor_stride0,
    output_tensor_sf,
    output_tensor_sf_stride0,
    output_index,
    output_index_stride0,
    num_topk: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SF_HIDDEN_SIZE: tl.constexpr,
    SF_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    num_grid_programs = tl.num_programs(0)
    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE
    offset_in_sf = tl.arange(0, SF_HIDDEN_SIZE_PAD)
    mask_sf = offset_in_sf < SF_HIDDEN_SIZE

    for token_id in range(start_token_id, total_token_count, num_grid_programs):
        x_values = tl.load(
            recv_x + token_id * recv_x_stride0 + offset_in,
            mask=mask,
        )
        sf_values = tl.load(
            recv_x_sf + token_id * recv_x_sf_stride0 + offset_in_sf,
            mask=mask_sf,
        )

        for topk_index in tl.range(0, num_topk, 1, num_stages=4):
            expert_id = tl.load(recv_topk_idx + token_id * recv_topk_idx_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_offset + expert_id, 1)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index,
                )
                tl.store(
                    output_tensor + dest_token_index * output_tensor_stride0 + offset_in,
                    x_values,
                    mask=mask,
                )
                tl.store(
                    output_tensor_sf + dest_token_index * output_tensor_sf_stride0 + offset_in_sf,
                    sf_values,
                    mask=mask_sf,
                )


@triton.jit
def _ep_gather_kernel(
    total_token_count,
    input_tensor,
    input_tensor_stride0,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_weight,
    recv_topk_weight_stride0,
    input_index,
    input_index_stride0,
    output_tensor,
    output_tensor_stride0,
    num_topk: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    cur_token_start = tl.program_id(1)
    num_grid_programs = tl.num_programs(1)

    for cur_token in range(cur_token_start, total_token_count, num_grid_programs):
        offset_d = tl.arange(0, BLOCK_D)
        mask = cur_block * BLOCK_D + offset_d < HIDDEN_SIZE
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, num_topk):
            expert_id = tl.load(recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index)
            if expert_id >= 0:
                source_token_index = tl.load(input_index + cur_token * input_index_stride0 + topk_index)
                topk_weight = tl.load(recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index)
                x_row = tl.load(input_tensor + source_token_index * input_tensor_stride0 + cur_block * BLOCK_D + offset_d, mask=mask)
                accumulator += x_row.to(tl.float32) * topk_weight
        tl.store(
            output_tensor + cur_token * output_tensor_stride0 + cur_block * BLOCK_D + offset_d,
            accumulator.to(output_tensor.dtype.element_ty),
            mask=mask,
        )


def ht_permute(
    recv_x: torch.Tensor,
    recv_x_sf: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    num_recv_tokens_per_expert_list,
    alignment: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand compact dispatch rows into flat expert-aligned GEMM rows.

    The element-wise scatter copies the W4A4 packed-FP4 activations and their
    packed-UE8M0 scales.
    """
    num_experts_per_rank = len(num_recv_tokens_per_expert_list)
    hidden = recv_x.size(1)
    sf_hidden = recv_x_sf.size(1)
    device = recv_x.device
    aligned_counts = [align_up(count, alignment) for count in num_recv_tokens_per_expert_list]
    num_padded_rows = sum(aligned_counts)
    num_recv_tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list, dtype=torch.int32, device=device)
    m_rows = torch.tensor(aligned_counts, dtype=torch.int32, device=device)

    expert_start_offset = torch.zeros((num_experts_per_rank, ), dtype=torch.int32, device=device)
    output = torch.empty((num_padded_rows, hidden), dtype=recv_x.dtype, device=device)
    output_sf = torch.empty((num_padded_rows, sf_hidden), dtype=recv_x_sf.dtype, device=device)
    expert_ids = torch.empty((num_padded_rows, ), dtype=torch.int32, device=device)
    inv_perm = torch.empty(recv_topk_idx.shape, dtype=torch.int32, device=device)

    _ep_scatter_layout_kernel[(num_experts_per_rank, )](
        num_recv_tokens_per_expert,
        expert_start_offset,
        expert_ids,
        num_experts=num_experts_per_rank,
        BLOCK_E=alignment,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts_per_rank),
        num_warps=8,
    )
    grid = (min(recv_topk_idx.shape[0], 1024 * 8), )
    _ep_scatter_tokens_kernel[grid](
        recv_topk_idx.shape[0],
        expert_start_offset,
        recv_x,
        recv_x.stride(0),
        recv_x_sf,
        recv_x_sf.stride(0),
        recv_topk_idx,
        recv_topk_idx.stride(0),
        output,
        output.stride(0),
        output_sf,
        output_sf.stride(0),
        inv_perm,
        inv_perm.stride(0),
        num_topk=recv_topk_idx.shape[1],
        HIDDEN_SIZE=hidden,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden),
        SF_HIDDEN_SIZE=sf_hidden,
        SF_HIDDEN_SIZE_PAD=triton.next_power_of_2(sf_hidden),
        num_warps=8,
    )
    return output, output_sf, expert_ids, inv_perm, m_rows


def ht_unpermute_reduce(
    gemm_output: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Gather expert rows and reduce routed outputs back to compact token rows."""
    num_tokens = output.shape[0]
    hidden = gemm_output.shape[1]
    block_d = min(hidden, 1024)
    grid = (triton.cdiv(hidden, block_d), min(num_tokens, 1024))
    _ep_gather_kernel[grid](
        num_tokens,
        gemm_output,
        gemm_output.stride(0),
        recv_topk_idx,
        recv_topk_idx.stride(0),
        recv_topk_weights,
        recv_topk_weights.stride(0),
        inv_perm,
        inv_perm.stride(0),
        output,
        output.stride(0),
        num_topk=recv_topk_idx.shape[1],
        HIDDEN_SIZE=hidden,
        BLOCK_D=block_d,
        num_warps=2,
    )
