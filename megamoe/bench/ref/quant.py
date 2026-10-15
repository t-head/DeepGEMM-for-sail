"""SwiGLU activation-quantization kernels used by Megamoe baselines."""

from typing import Tuple

import torch
import triton
import triton.language as tl

from .fp4 import preprocess_mxfp4_scales


@triton.jit
def _swiglu_to_mxfp4_kernel(
    x_ptr,
    stride_x_e,
    stride_x_m,
    stride_x_n,
    y_ptr,
    stride_y_e,
    stride_y_m,
    stride_y_n,
    y_sf_ptr,
    stride_sf_e,
    stride_sf_h,
    stride_sf_m,
    masked_m_ptr,
    num_tokens,
    HIDDEN_SIZE: tl.constexpr,
    activation_clamp: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    group_size: tl.constexpr = 32
    quant_max: tl.constexpr = 6.0
    num_groups: tl.constexpr = BLOCK_K // group_size
    num_scale_pairs: tl.constexpr = num_groups // 2
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)
    pid_e_i64 = pid_e.to(tl.int64)
    num_token_stripes = tl.num_programs(1)
    num_valid_tokens = tl.load(masked_m_ptr + pid_e).to(tl.int64) if HAS_MASK else num_tokens.to(tl.int64)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_out_k = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    offs_scale_pairs = pid_k * num_scale_pairs + tl.arange(0, num_scale_pairs)
    x_base = x_ptr + pid_e_i64 * stride_x_e + offs_k * stride_x_n
    y_base = y_ptr + pid_e_i64 * stride_y_e + offs_out_k * stride_y_n
    sf_base = y_sf_ptr + pid_e_i64 * stride_sf_e + offs_scale_pairs * stride_sf_h

    for token in tl.range(pid_m, num_valid_tokens, num_token_stripes, num_stages=NUM_STAGES):
        gate = tl.load(x_base + token * stride_x_m, mask=offs_k < HIDDEN_SIZE, other=0.0).to(tl.float32)
        up = tl.load(
            x_base + token * stride_x_m + HIDDEN_SIZE * stride_x_n,
            mask=offs_k < HIDDEN_SIZE,
            other=0.0,
        ).to(tl.float32)
        if HAS_CLAMP:
            gate = tl.minimum(gate, activation_clamp)
            up = tl.minimum(tl.maximum(up, -activation_clamp), activation_clamp)
        y = gate * tl.sigmoid(gate) * up
        y_grouped = tl.reshape(y, [num_groups, group_size])
        dequant_scale = tl.maximum(tl.max(tl.abs(y_grouped), axis=1), 1.0e-10)
        dequant_scale /= quant_max
        scale_bits = dequant_scale.to(tl.uint32, bitcast=True)
        ue8m0_bits = (scale_bits + 0x007FFFFF) & 0x7F800000
        rounded_scale = ue8m0_bits.to(tl.float32, bitcast=True)
        y_scaled = tl.reshape(
            y_grouped * tl.reshape(1.0 / rounded_scale, [num_groups, 1]),
            [BLOCK_K],
        )
        pairs = tl.reshape(y_scaled, [BLOCK_K // 2, 2])
        lo_f, hi_f = tl.split(pairs)
        packed = tl.inline_asm_elementwise(
            """
            {
                .reg .b8 r;
                cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
                mov.b32 $0, {r, r, r, r};
            }
            """,
            constraints="=r,f,f",
            args=[hi_f.to(tl.float32), lo_f.to(tl.float32)],
            dtype=tl.uint8,
            is_pure=True,
            pack=1,
        )
        scale_pairs = tl.reshape(ue8m0_bits, [num_scale_pairs, 2])
        lo_u32, hi_u32 = tl.split(scale_pairs)
        scale_u16 = ((lo_u32 >> 23) | ((hi_u32 >> 23) << 8)).to(tl.uint16)
        tl.store(
            y_base + token * stride_y_m,
            packed,
            mask=offs_out_k < (HIDDEN_SIZE // 2),
        )
        tl.store(
            sf_base + token * stride_sf_m,
            scale_u16,
            mask=offs_scale_pairs < (HIDDEN_SIZE // group_size // 2),
        )


def _plan_swiglu_launch(hidden: int, activation_clamp, has_mask: bool) -> Tuple[int, dict]:
    """The swiglu quantizer's k-tile width and its launch kwargs, shared by both entry points.

    The two entry points differ only in the mask they pass, so the recipe (k-tile width, stage count,
    warps and the clamp convention) has to come from one place or the two can drift apart.
    """
    block_k = 256 if hidden > 256 else (128 if hidden > 128 else 64)
    return block_k, dict(HIDDEN_SIZE=hidden,
                         activation_clamp=float(activation_clamp) if activation_clamp is not None else 0.0,
                         HAS_CLAMP=activation_clamp is not None,
                         HAS_MASK=has_mask,
                         BLOCK_K=block_k,
                         NUM_STAGES=4,
                         num_warps=4)


def swiglu_masked_to_mxfp4(
    x: torch.Tensor,
    masked_m: torch.Tensor,
    activation_clamp: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Produce masked packed-FP4 values and column-major uint16 scales."""
    num_experts, m_max, two_cols = x.shape
    hidden = two_cols // 2
    assert hidden % 64 == 0
    y = torch.empty((num_experts, m_max, hidden // 2), dtype=torch.uint8, device=x.device)
    y_sf_physical = torch.empty((num_experts, hidden // 64, m_max), dtype=torch.uint16, device=x.device)
    block_k, launch_kwargs = _plan_swiglu_launch(hidden, activation_clamp, has_mask=True)
    grid = (
        triton.cdiv(hidden, block_k),
        64 if num_experts < 4 else 32,
        num_experts,
    )
    _swiglu_to_mxfp4_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y_sf_physical,
        y_sf_physical.stride(0),
        y_sf_physical.stride(1),
        y_sf_physical.stride(2),
        masked_m,
        0,
        **launch_kwargs,
    )
    return y, y_sf_physical.permute(0, 2, 1)


def swiglu_flat_to_mxfp4(
    x: torch.Tensor,
    activation_clamp: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Produce flat expert-contiguous packed-FP4 values and uint16 scales."""
    m, two_cols = x.shape
    hidden = two_cols // 2
    assert hidden % 64 == 0
    y = torch.empty((m, hidden // 2), dtype=torch.uint8, device=x.device)
    y_sf_physical = torch.empty((hidden // 64, m), dtype=torch.uint16, device=x.device)
    y_sf = y_sf_physical.permute(1, 0)
    if m == 0:
        return y, y_sf
    block_k, launch_kwargs = _plan_swiglu_launch(hidden, activation_clamp, has_mask=False)
    grid = (triton.cdiv(hidden, block_k), min(m, 1024), 1)
    _swiglu_to_mxfp4_kernel[grid](
        x,
        0,
        x.stride(0),
        x.stride(1),
        y,
        0,
        y.stride(0),
        y.stride(1),
        y_sf_physical,
        0,
        y_sf_physical.stride(0),
        y_sf_physical.stride(1),
        y,
        m,
        **launch_kwargs,
    )
    return y, y_sf


def normalize_mxfp4_dispatch_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert high-throughput DeepEP packed scales to MegaMoE SFA layout."""
    if scale.dtype == torch.uint16:
        return scale
    if scale.dtype == torch.uint8:
        return preprocess_mxfp4_scales(scale.contiguous())
    if scale.dtype in (torch.int32, torch.int):
        return preprocess_mxfp4_scales(scale.contiguous().view(torch.uint8))
    raise RuntimeError("Unsupported DeepEP MXFP4 dispatch scale dtype: "
                       f"{scale.dtype}; expected torch.uint16, torch.uint8, or torch.int32")
