"""Test quantization and reference decoding in natural [E, N, K] order.

Kernel-specific permutations belong to deep_gemm.preprocess_weights_and_scales.
"""
import torch

from deep_gemm.utils.math import (
    ceil_to_ue8m0, per_token_cast_to_fp4, cast_back_from_fp4,
    _quantize_to_fp4_e2m1, _dequantize_from_fp4_e2m1,
)


def quant_w4(y, groupsize=32, d='w4a16'):
    """Return reference [E,N,K], packed uint8 [E,N,K/2], raw scales [E,N,K/32].

    INT4 uses offset-binary codes (zero is 8) and BF16 scales. MXFP4 uses
    E2M1 codes and raw E8M0 bytes. Even K occupies the low nibble.
    Preserve the existing non-MMA and MMA quantizers' rounding behavior.
    """
    if d not in ('w4a16', 'w4fa16', 'w4fa16_mma', 'w4fa8_mma'):
        raise ValueError(f'Unsupported W4 quantization type: {d}')
    assert groupsize == 32
    e, n, k = y.shape
    assert k % groupsize == 0 and y.dtype in (torch.float16, torch.bfloat16)
    refs, weights, raw_scales = [], [], []
    for expert in range(e):
        if d in ('w4fa16_mma', 'w4fa8_mma'):
            packed, scales = per_token_cast_to_fp4(
                y[expert], use_ue8m0=True, gran_k=groupsize, use_packed_ue8m0=False)
            ref = cast_back_from_fp4(packed, scales, gran_k=groupsize).bfloat16()
            scales = ((scales.float().view(torch.int32) >> 23) & 255).byte().reshape(n, k // groupsize)
            packed = packed.view(torch.uint8)
        else:
            w = y[expert].T.contiguous().reshape(-1, groupsize, n).permute(1, 0, 2)
            w = w.reshape(groupsize, -1)
            if d == 'w4a16':
                scales = w.abs().amax(0, keepdim=True)
                scales *= 2 / 15
                codes = (torch.round(w / scales).int() + 8).clamp(0, 15)
                ref = (codes - 8) * scales
            else:
                scales = ceil_to_ue8m0((w.abs().amax(0, keepdim=True) / 6).clamp_min(1e-4))
                codes = _quantize_to_fp4_e2m1(w / scales)
                ref = _dequantize_from_fp4_e2m1(codes) * scales

            def natural(t):
                return t.reshape(groupsize, -1, n).permute(1, 0, 2).reshape(k, n).T.contiguous()

            ref = natural(ref)
            if d != 'w4a16':
                ref = ref.bfloat16()
            codes = natural(codes).byte()
            packed = codes[:, ::2] | (codes[:, 1::2] << 4)
            scales = scales.reshape(k // groupsize, n).T.contiguous()
            if d == 'w4fa16':
                scales = ((scales.float().view(torch.int32) >> 23) & 255).byte()
            else:
                scales = scales.bfloat16()
        refs.append(ref)
        weights.append(packed)
        raw_scales.append(scales)
    return torch.stack(refs), torch.stack(weights), torch.stack(raw_scales)


def decode_weight(packed, scales):
    """Decode natural packed MXFP4 and raw E8M0 scales for a reference GEMM."""
    values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                          device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    return values[codes] * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def quant_activation(x, weight_inv_scale=1.0):
    """Quantize E4M3, then fold the weight inverse into the FP32 output row scale."""
    scale = (x.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-12)
    quantized = (x.float() / scale).to(torch.float8_e4m3fn)
    if weight_inv_scale != 1.0:
        scale *= weight_inv_scale
    return quantized, scale
