"""Pure-torch FP4/UE8M0 quantization primitives for the reference and benchmarks."""

from typing import Tuple

import torch


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exponent = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exponent.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    assert (x.view(torch.int) & ((1 << 23) - 1) == 0).all()
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def unpack_ue8m0_from_int(packed_sf: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_ue8m0_to_int: uint8 exponent -> FP32 power-of-two."""
    return (packed_sf.view(torch.uint8).to(torch.int) << 23).view(torch.float)


def quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Quantize (already SF-scaled) values to E2M1 nibble codes (int8, 0..15)."""
    abs_x = x.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=abs_x.dtype)
    bucket_idx = torch.bucketize(abs_x, boundaries)
    code = bucket_idx.to(torch.uint8)
    sign = (x < 0) & (bucket_idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_to_fp4_e2m1 on nibble codes (int8, 0..15) -> FP32."""
    fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=torch.float)
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def per_token_cast_to_fp4(x: torch.Tensor,
                          use_ue8m0: bool,
                          k_granularity: int = 32,
                          use_packed_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP4 (E2M1) activation quantization: packed 2-per-byte codes + per-``k_granularity`` UE8M0 scales."""
    assert x.dim() == 2
    m, n = x.shape
    assert n % 2 == 0
    assert not use_packed_ue8m0 or use_ue8m0
    # Rounded here rather than through the kernel's workspace helper: the reference must not make
    # the pool layout a dependency of its own numbers.
    padded_n = (n + k_granularity - 1) // k_granularity * k_granularity
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // k_granularity, k_granularity)
    # 1e-10 is the kernel's activation floor (common/fp4.cuh and the x-quantizer it shares code with),
    # not 1e-4: both sides compute this scale independently, so a group whose amax falls between two
    # such floors would be scaled differently on each side.
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-10)
    sf = x_amax / 6.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    code_pairs = codes.view(m, padded_n // 2, 2)
    packed = (code_pairs[:, :, 0] & 0x0F) | ((code_pairs[:, :, 1] & 0x0F) << 4)
    packed = packed[:, :n // 2].contiguous()
    return packed, (pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf)


def cast_back_from_fp4(packed: torch.Tensor, sf: torch.Tensor, k_granularity: int = 32, use_packed_ue8m0: bool = False) -> torch.Tensor:
    """Dequantize packed FP4 (E2M1) + per-``k_granularity`` UE8M0 scales to FP32. Inverse of per_token_cast_to_fp4."""
    m, packed_cols = packed.shape
    n = packed_cols * 2
    if use_packed_ue8m0:
        sf = unpack_ue8m0_from_int(sf)
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x_dequantized = dequantize_from_fp4_e2m1(unpacked)
    group_idx = torch.arange(n, device=packed.device) // k_granularity
    x_restored = x_dequantized * sf[:, group_idx]
    return x_restored


def preprocess_mxfp4_scales(scale: torch.Tensor) -> torch.Tensor:
    """Reorder MXFP4 uint8 scales into the SFA/SFB layout the PPU MMA consumes."""
    assert scale.dtype == torch.uint8, f"scale dtype must be torch.uint8, got {scale.dtype}"
    assert scale.is_contiguous(), "MXFP4 scales must be contiguous"
    assert scale.dim() in (2, 3), "MXFP4 scale rank must be 2 (dense/nopad) or 3 (masked)"
    if scale.shape[-1] % 2:
        scale = torch.nn.functional.pad(scale, (0, 1))
    if scale.dim() == 2:
        return scale.view(torch.uint16).t().contiguous().t()
    return scale.view(torch.uint16).permute(0, 2, 1).contiguous().permute(0, 2, 1)
