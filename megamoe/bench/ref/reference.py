"""Quantized weight helpers and torch reference for Megamoe tests."""

import math
from typing import List, Tuple

import torch
import torch.distributed as dist

from .dist import uneven_all_gather
from .fp4 import (
    cast_back_from_fp4,
    per_token_cast_to_fp4,
    dequantize_from_fp4_e2m1,
    quantize_to_fp4_e2m1,
)


def _preprocess_fp4_sfb_per_n_k32(sf: torch.Tensor) -> torch.Tensor:
    """FP32 pow2 SF [G, N, K/32] -> uint16 physical [G, K/64, N]."""
    assert sf.dtype == torch.float32
    assert sf.is_contiguous()
    *prefix, n, k_groups = sf.shape
    assert k_groups % 2 == 0
    assert (sf.view(torch.int32) & ((1 << 23) - 1) == 0).all()
    ue8m0 = (sf.view(torch.int32) >> 23).to(torch.uint8).contiguous()
    return ue8m0.view(*prefix, n, k_groups).view(torch.uint16).permute(0, 2, 1).contiguous()


def _unpack_fp4_sfb_per_n_k32(sf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """uint16 physical [*, K/64, N] -> FP32 pow2 SF [*, N, K/32]."""
    assert sf.dtype == torch.uint16
    assert k % 64 == 0
    ue8m0_exponents = sf.transpose(-1, -2).contiguous().view(torch.uint8).view(*sf.shape[:-2], n, k // 32)
    return torch.pow(2.0, ue8m0_exponents.to(torch.float32) - 127.0)


def quantize_grouped_fp4_weights(
    weight: torch.Tensor,
    pow2: bool = False,
    k_granularity: int = 32,
    n_granularity: int = 1,
    use_preprocessed_sfb: bool = False,
    store_n_granularity: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(G, N, K) bf16 -> packed FP4 [G, N, K//2] int8 + block-(n_granularity, k_granularity) FP32 SF
    [G, N//n_granularity, K//k_granularity].
    pow2=True rounds each SF up to a power of two (UE8M0), required when the kernel
    feeds the SF into the native fp4 MMA's hardware mx SF registers (W4A4).
    store_n_granularity (only with use_preprocessed_sfb): compute SF with n_granularity-wide groups, then
    broadcast those values to a store_n_granularity-wide physical stride before preprocessing. The
    W4A4 call site uses k_granularity=32, n_granularity=1 and store_n_granularity=1: one SF slot per N column, and
    the physical uint16 SFB tensor is [G, K/64, N].
    """
    g, n, k = weight.shape
    assert n % n_granularity == 0 and k % k_granularity == 0
    w_view = weight.view(g, n // n_granularity, n_granularity, k // k_granularity, k_granularity).float()
    amax = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4)
    sf = amax / 6.0
    if pow2:
        sf = torch.pow(2.0, torch.ceil(torch.log2(sf.clamp_min(1e-30))))
    codes = quantize_to_fp4_e2m1(w_view / sf.unsqueeze(-1).unsqueeze(-3)).view(g, n, k)
    code_pairs = codes.view(g, n, k // 2, 2)
    packed = (code_pairs[..., 0] & 0x0F) | ((code_pairs[..., 1] & 0x0F) << 4)
    if use_preprocessed_sfb:
        assert pow2 and k_granularity == 32, "preprocessed SFB requires pow2 SF with k_granularity=32"
        assert n_granularity % store_n_granularity == 0, "n_granularity must be a multiple of store_n_granularity"
        sf_full = sf.repeat_interleave(n_granularity // store_n_granularity, dim=-2)
        sf = _preprocess_fp4_sfb_per_n_k32(sf_full.contiguous())
    return packed.to(torch.int8).contiguous(), sf.contiguous()


def _dequant_grouped_fp4(packed: torch.Tensor, sf: torch.Tensor, k_granularity: int = 32, n_granularity: int = 1) -> torch.Tensor:
    """Inverse of quantize_grouped_fp4_weights with block-(n_granularity, k_granularity) SF. A uint16 SF is the
    preprocessed [*, K/64, N] physical form, which pins the granularity to one per N column."""
    *prefix, n, packed_cols = packed.shape
    k = packed_cols * 2
    if sf.dtype == torch.uint16:
        sf = _unpack_fp4_sfb_per_n_k32(sf, n, k)
        n_granularity = 1
        k_granularity = 32
    unpacked = torch.zeros((*prefix, n, k), dtype=torch.int8, device=packed.device)
    unpacked[..., ::2] = packed & 0x0F
    unpacked[..., 1::2] = (packed >> 4) & 0x0F
    dequantized = dequantize_from_fp4_e2m1(unpacked).view(*prefix, n // n_granularity, n_granularity, k // k_granularity, k_granularity)
    return (dequantized * sf.unsqueeze(-1).unsqueeze(-3)).view(*prefix, n, k)


def compute_swiglu_fp32(gate_up: torch.Tensor, activation_clamp: float) -> torch.Tensor:
    """SwiGLU matching the fused PPU890 path's clamp semantics."""
    two_cols = gate_up.size(-1)
    half_cols = two_cols // 2
    gate, up = gate_up[..., :half_cols], gate_up[..., half_cols:]
    if math.isfinite(activation_clamp):
        gate = gate.clamp(max=activation_clamp)
        up = up.clamp(min=-activation_clamp, max=activation_clamp)
    return torch.nn.functional.silu(gate) * up


def _reduce_reference_chunk(y_chunk: torch.Tensor, x_fp4_chunk: torch.Tensor, x_sf_chunk: torch.Tensor, topk_idx_chunk: torch.Tensor,
                            topk_weights_chunk: torch.Tensor, l1_weights_gathered: List[torch.Tensor], l1_sf_gathered: List[torch.Tensor],
                            l2_weights_gathered: List[torch.Tensor], l2_sf_gathered: List[torch.Tensor], num_experts_per_rank: int,
                            num_topk: int, hidden: int, intermediate_hidden: int, activation_clamp: float, topk_weight_site: int) -> None:
    """Fill one row chunk of the reference output.

    Dequantize the chunk, walk it in stable expert order (L1 GEMM, SwiGLU, L2 GEMM, topk reduction) and
    write the per-row result into ``y_chunk``. Every chunk runs the identical per-row math, so the
    chunked result matches a full-width one.
    """
    num_chunk_rows = x_fp4_chunk.size(0)
    x_fp32 = cast_back_from_fp4(x_fp4_chunk, x_sf_chunk, k_granularity=32, use_packed_ue8m0=True)
    combine_buffer = torch.zeros(num_chunk_rows, num_topk, hidden, dtype=torch.float32, device="cuda")
    valid_slots = topk_idx_chunk >= 0
    token_idx, slot_idx = valid_slots.nonzero(as_tuple=True)
    if token_idx.numel():
        expert_ids_flat = topk_idx_chunk[token_idx, slot_idx]
        sorted_expert_ids, sort_order = torch.sort(expert_ids_flat, stable=True)
        token_idx, slot_idx = token_idx[sort_order], slot_idx[sort_order]
        weights_flat = topk_weights_chunk[token_idx, slot_idx]
        unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
        cursor = 0
        for expert_id_value, count in zip(unique_experts.tolist(), counts.tolist()):
            selected_rows = token_idx[cursor:cursor + count]
            topk_slots = slot_idx[cursor:cursor + count]
            slot_weights = weights_flat[cursor:cursor + count]
            cursor += count
            x_selected = x_fp32[selected_rows]
            expert_id = int(expert_id_value)
            dst_rank = expert_id // num_experts_per_rank
            dst_local = expert_id % num_experts_per_rank
            l1_weight = _dequant_slot_weights(l1_weights_gathered, l1_sf_gathered, dst_rank, dst_local)
            l1_y = torch.einsum("sk,nk->sn", x_selected, l1_weight)
            del l1_weight

            l1_y = compute_swiglu_fp32(l1_y, activation_clamp)
            num_rows, l1_hidden = l1_y.shape
            assert l1_hidden == intermediate_hidden
            l2_packed, l2_sf = per_token_cast_to_fp4(l1_y, use_ue8m0=True, k_granularity=32, use_packed_ue8m0=False)
            l2_in = cast_back_from_fp4(l2_packed, l2_sf, k_granularity=32, use_packed_ue8m0=False).view(num_rows, l1_hidden)

            l2_weight = _dequant_slot_weights(l2_weights_gathered, l2_sf_gathered, dst_rank, dst_local)
            l2_y = torch.einsum("sn,mn->sm", l2_in, l2_weight)
            del l2_weight

            if topk_weight_site == 1:
                l2_y = l2_y * slot_weights.unsqueeze(-1)
            combine_buffer[selected_rows, topk_slots] = l2_y.to(torch.bfloat16).float()
    if topk_weight_site == 2:
        y_chunk[:] = (combine_buffer.to(torch.bfloat16).float() * topk_weights_chunk.float().unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)
    else:
        y_chunk[:] = combine_buffer.to(torch.bfloat16).sum(dim=1).to(torch.bfloat16)


def _dequant_slot_weights(weights_gathered: List[torch.Tensor], sf_gathered: List[torch.Tensor], dst_rank: int,
                          dst_local: int) -> torch.Tensor:
    """Dequantize one (rank, local expert) slot's packed FP4 weight at the reference's k-granularity."""
    return _dequant_grouped_fp4(weights_gathered[dst_rank][dst_local], sf_gathered[dst_rank][dst_local], k_granularity=32)


def _gather_rank_weights(l1_weights_fp4: torch.Tensor, l1_weights_sf: torch.Tensor, l2_weights_fp4: torch.Tensor,
                         l2_weights_sf: torch.Tensor, num_ranks: int,
                         group: dist.ProcessGroup) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Every rank's packed weights and scales, in rank order."""
    l1_weights_gathered = [torch.empty_like(l1_weights_fp4) for _ in range(num_ranks)]
    l2_weights_gathered = [torch.empty_like(l2_weights_fp4) for _ in range(num_ranks)]
    dist.all_gather(l1_weights_gathered, l1_weights_fp4, group=group)
    dist.all_gather(l2_weights_gathered, l2_weights_fp4, group=group)

    def gather_sf(sf: torch.Tensor) -> List[torch.Tensor]:
        if sf.dtype == torch.uint16:
            sf_i32 = sf.to(torch.int32)
            gathered = [torch.empty_like(sf_i32) for _ in range(num_ranks)]
            dist.all_gather(gathered, sf_i32, group=group)
            return [x.to(torch.uint16) for x in gathered]
        gathered = [torch.empty_like(sf) for _ in range(num_ranks)]
        dist.all_gather(gathered, sf, group=group)
        return gathered

    return l1_weights_gathered, gather_sf(l1_weights_sf), l2_weights_gathered, gather_sf(l2_weights_sf)


def compute_reference_output(
    topk_idx_local: torch.Tensor,
    topk_weights_local: torch.Tensor,
    l1_weights_fp4: torch.Tensor,
    l1_weights_sf: torch.Tensor,
    l2_weights_fp4: torch.Tensor,
    l2_weights_sf: torch.Tensor,
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation_clamp: float,
    x_bf16: torch.Tensor = None,
) -> torch.Tensor:
    """PyTorch W4A4 reference for this rank's fused output. x_bf16 is the same input the fused kernel
    sees; the reference quantizes it here so both sides read identical data."""
    from ..bf16_to_mxfp4 import bf16_to_mxfp4_cuda
    x_fp4_local, x_sf_u8 = bf16_to_mxfp4_cuda(x_bf16)
    x_sf_local = x_sf_u8.contiguous().view(torch.int32)

    num_experts_per_rank = num_experts // num_ranks
    from megamoe.engine.config import _get_topk_weight_site
    topk_weight_site = _get_topk_weight_site()

    x_fp4_gathered = uneven_all_gather(x_fp4_local, group=group)
    x_sf_gathered = uneven_all_gather(x_sf_local, group=group)
    topk_idx_gathered = uneven_all_gather(topk_idx_local, group=group)
    topk_weights_gathered = uneven_all_gather(topk_weights_local, group=group)
    num_gathered_rows = x_fp4_gathered.size(0)

    local_size = torch.tensor([x_fp4_local.size(0)], device="cuda", dtype=torch.long)
    sizes_tensor = torch.empty(num_ranks, dtype=torch.long, device="cuda")
    dist.all_gather_into_tensor(sizes_tensor, local_size, group=group)
    sizes_list = sizes_tensor.tolist()
    assert sum(sizes_list) == num_gathered_rows

    l1_weights_gathered, l1_sf_gathered, l2_weights_gathered, l2_sf_gathered = _gather_rank_weights(
        l1_weights_fp4, l1_weights_sf, l2_weights_fp4, l2_weights_sf, num_ranks, group)
    # The full-width reference intermediates do not fit alongside the fused workspace at the
    # prefill sizes: the [rows, topk, hidden] fp32 combine buffer and the fp32 dequantization of
    # the gathered activations both scale with the gathered row count. Reduce both in row chunks
    # instead, writing each chunk's result straight into the preallocated output; every chunk runs
    # the identical per-row math (dequantize, stable expert sort, the same topk reduction), so the
    # result is unchanged.
    chunk_rows = max(1, (2 << 30) // (num_topk * hidden * 4))
    y_full_bf16 = torch.empty(num_gathered_rows, hidden, dtype=torch.bfloat16, device="cuda")
    for row_lo in range(0, num_gathered_rows, chunk_rows):
        row_hi = min(row_lo + chunk_rows, num_gathered_rows)
        _reduce_reference_chunk(y_full_bf16[row_lo:row_hi], x_fp4_gathered[row_lo:row_hi], x_sf_gathered[row_lo:row_hi],
                                topk_idx_gathered[row_lo:row_hi], topk_weights_gathered[row_lo:row_hi], l1_weights_gathered, l1_sf_gathered,
                                l2_weights_gathered, l2_sf_gathered, num_experts_per_rank, num_topk, hidden, intermediate_hidden,
                                activation_clamp, topk_weight_site)
    start = sum(sizes_list[:rank_idx])
    end = start + sizes_list[rank_idx]
    return y_full_bf16[start:end].contiguous()
