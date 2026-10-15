from typing import Tuple

import torch

from .config import _GEMM_WARP_N, _W4A4_BLOCK_N


def _interleave_l1_weight(weight: torch.Tensor, granularity: int = 8) -> torch.Tensor:
    if weight.ndim < 2 or weight.shape[1] % (2 * granularity) != 0:
        raise ValueError("L1 weight N dimension must be divisible by 16")
    num_groups, num_cols, *rest = weight.shape
    half_cols = num_cols // 2
    gate = weight[:, :half_cols].reshape(num_groups, half_cols // granularity, granularity, *rest)
    up = weight[:, half_cols:].reshape(num_groups, half_cols // granularity, granularity, *rest)
    interleaved = torch.stack((gate, up), dim=2).reshape(num_groups, num_cols, *rest)
    return torch.empty_like(weight).copy_(interleaved)


def _interleave_l1_sfb(sfb: torch.Tensor, granularity: int = 8) -> torch.Tensor:
    if sfb.ndim < 3 or sfb.shape[-1] % (2 * granularity) != 0:
        raise ValueError("L1 SFB N dimension must be divisible by 16")
    *prefix, num_cols = sfb.shape
    half_cols = num_cols // 2
    gate = sfb[..., :half_cols].reshape(*prefix, half_cols // granularity, granularity)
    up = sfb[..., half_cols:].reshape(*prefix, half_cols // granularity, granularity)
    interleaved = torch.stack((gate, up), dim=-2).reshape(*prefix, num_cols)
    return torch.empty_like(sfb).copy_(interleaved)


def _pair_interleave_sfb(sfb: torch.Tensor) -> torch.Tensor:
    *prefix, num_cols = sfb.shape
    if num_cols % 16 != 0:
        raise ValueError("SFB N dimension must be divisible by 16")
    regrouped = sfb.reshape(*prefix, num_cols // 16, 2, 8)
    interleaved = regrouped.transpose(-2, -1).reshape(*prefix, num_cols)
    return torch.empty_like(sfb).copy_(interleaved)


def _vectorize_sfb_n_tiles(sfb: torch.Tensor, warp_n_cols: int) -> torch.Tensor:
    *prefix, num_cols = sfb.shape
    if warp_n_cols != 64:
        raise ValueError(f"per-warp column count {warp_n_cols} must be 64 (4 n16 tiles, 8 part0 threads)")
    if num_cols % warp_n_cols != 0:
        raise ValueError(f"SFB N ({num_cols}) must be a multiple of the per-warp column count ({warp_n_cols})")
    int16s_per_tile = warp_n_cols // 2
    num_n_tiles = warp_n_cols // 16
    num_threads = warp_n_cols // 8
    inverse_order_list = [0] * (4 * num_threads)
    for g in range(num_threads):
        for nt in range(num_n_tiles):
            inverse_order_list[4 * g + nt] = 8 * nt + g
    inverse_order = torch.tensor(inverse_order_list, dtype=torch.long)
    grouped = sfb.view(torch.int16).reshape(*prefix, num_cols // warp_n_cols, int16s_per_tile, 2)
    gathered = grouped[..., inverse_order.to(sfb.device), :].reshape(*prefix, num_cols)
    out = torch.empty(gathered.shape, dtype=torch.int16, device=sfb.device)
    return out.copy_(gathered).view(sfb.dtype)


def _get_w4a4_output_cols_per_warp() -> int:
    return _W4A4_BLOCK_N // _GEMM_WARP_N


def transform_weights_for_megamoe(
        l1_weights: Tuple[torch.Tensor, torch.Tensor],
        l2_weights: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    warp_n_cols = _get_w4a4_output_cols_per_warp()
    l1_weight, l1_sf = l1_weights
    if l1_sf.dtype == torch.uint16:
        l1_sf = _vectorize_sfb_n_tiles(_pair_interleave_sfb(_interleave_l1_sfb(l1_sf)), warp_n_cols)
    l2_weight, l2_sf = l2_weights
    if l2_sf.dtype == torch.uint16:
        l2_sf = _vectorize_sfb_n_tiles(_pair_interleave_sfb(l2_sf), warp_n_cols)
        num_groups, num_cols, k_half_bytes = l2_weight.shape
        assert k_half_bytes % 16 == 0, "W4A4 L2 weight K/2 bytes must tile into 16-byte groups"
        l2_weight = (l2_weight.view(num_groups, num_cols, k_half_bytes // 16, 4,
                                    4).transpose(-1, -2).reshape(num_groups, num_cols, k_half_bytes).contiguous())
    return (
        (_interleave_l1_weight(l1_weight), l1_sf),
        (l2_weight, l2_sf),
    )


def prepare_gemm_weights(l1_weights, l2_weights, workspace):
    tensors = (*l1_weights, *l2_weights)
    identity = tuple((id(tensor), None if torch.is_inference(tensor) else tensor._version) for tensor in tensors)
    cached = getattr(workspace, '_gemm_weight_cache', None)
    if cached is not None and cached[0] == identity:
        return cached[2]

    def restore_scale_columns(scales):
        num_columns = scales.shape[-1]
        num_threads = 8
        inverse_order = [0] * 32
        for g in range(num_threads):
            for n in range(4):
                inverse_order[4 * g + n] = 8 * n + g
        inverse = torch.tensor(inverse_order, dtype=torch.long, device=scales.device).argsort()
        grouped = scales.view(torch.int16).reshape(*scales.shape[:-1], num_columns // 64, 32, 2)
        restored = grouped[..., inverse, :].reshape(*scales.shape)
        restored = restored.reshape(*scales.shape[:-1], num_columns // 16, 8, 2).transpose(-2, -1)
        return restored.reshape(*scales.shape).contiguous().view(torch.uint16)

    first_weight, first_scales = l1_weights
    second_weight, second_scales = l2_weights
    experts, columns, packed_reduction = second_weight.shape
    second_weight = second_weight.reshape(experts, columns, packed_reduction // 16, 4,
                                          4).transpose(-1, -2).reshape(experts, columns, packed_reduction).contiguous()
    prepared = ((first_weight, restore_scale_columns(first_scales)), (second_weight, restore_scale_columns(second_scales)))
    workspace._gemm_weight_cache = (identity, tensors, prepared)
    return prepared
