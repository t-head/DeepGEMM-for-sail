import torch
import numpy as np
from typing import Tuple, Iterable

from deep_gemm.utils.math import (
    ceil_div,
    align,
    ceil_to_ue8m0,
    pack_ue8m0_to_int,
    per_token_cast_to_fp8,
    per_channel_cast_to_fp8 as per_channel_cast_to_fp8_128align,
    per_block_cast_to_fp8,
    per_custom_dims_cast_to_fp8,
    per_token_cast_to_int8,
    _quantize_to_fp4_e2m1,
    per_token_cast_to_fp4,
    transpose_packed_fp4,
    _dequantize_from_fp4_e2m1,
    unpack_ue8m0_from_int,
    cast_back_from_fp4,
)
from deep_gemm.testing.numeric import calc_diff, count_bytes

round_up = align


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_e8m0 = []
    for i in range(16):  # 64 / 4
        for sw in (0, 2, 1, 3):
            scale_perm_e8m0.append(scale_perm[4 * i + sw])
    scale_perm_chennel = []
    for i in range(4):
        scale_perm_chennel.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    scale_perm_fp4_mma = []
    for i in range(8):
        for j in range(8):
            scale_perm_fp4_mma.extend([2 * i + j * 16, 2 * i + j * 16 + 1])
    print(scale_perm_fp4_mma)
    return perm, scale_perm, scale_perm_chennel, scale_perm_e8m0, scale_perm_fp4_mma

_perm, _scale_perm, _scale_perm_chennel, _scale_perm_e8m0, _scale_perm_fp4_mma = _get_perms()


def quant_w4a16(y, groupsize=32, d='w4a16'):
    """
    Weight quantization for W4A16 (INT4), W4FA16 (FP4 with E8M0 scale),
    W4FA16_S16 (FP4 with BF16 scale), or W4FA16_MMA (unpermuted FP4 with
    E8M0 scale for the FP4-MMA dequant path).

    Args:
        y: weight tensor, shape (e, n, k), dtype=torch.bfloat16
        groupsize: quantization group size, default 32
        d: 'w4a16' (default, INT4), 'w4fa16' (FP4 + E8M0 scale),
           'w4fa16_s16' (FP4 + BF16 scale), or 'w4fa16_mma' (FP4 + E8M0
           scale without weight permutation).

    Returns:
        refs: (e, k, n), dequantized reference for verification
        For d != 'w4fa16_mma':
            qs: (e, k // 16, n * 2), int32, packed and permuted weights
            scales: (e, k // groupsize, n), per-group scale
        For d == 'w4fa16_mma':
            qs: (e, n, k // 2), uint8, unpermuted packed FP4 weights;
                even K is in the low nibble and odd K is in the high nibble
            scales: (e, n // 64, k // 64, 128), uint8 E8M0 exponents;
                the last dimension is [n_inner=0..63, k_group_inner=0..1]
    """
    assert d in ('w4a16', 'w4fa16', 'w4fa16_s16', 'w4fa16_mma'), \
        f"d must be 'w4a16'/'w4fa16'/'w4fa16_s16'/'w4fa16_mma', got {d}"
    quant_format = 'int4' if d == 'w4a16' else 'fp4'
    use_e8m0 = d in ('w4fa16', 'w4fa16_mma')
    e, n, k = y.shape
    tile = 16
    maxq = 2 ** 4 - 1
    assert k % groupsize == 0
    assert k % tile == 0 and n % tile == 0
    assert y.dtype in [torch.half, torch.bfloat16]

    if d == 'w4fa16_mma':
        assert groupsize == 32, 'w4fa16_mma requires groupsize=32'
        assert n % 64 == 0 and k % 64 == 0, \
            'w4fa16_mma requires n and k to be multiples of 64'

        # Keep the logical [N, K] order. Reuse the generic per-token FP4
        # quantizer with group size 32, and do not apply MMA/Marlin
        # permutation. per_token_cast_to_fp4 packs even K into the low nibble
        # and odd K into the high nibble.
        refs = torch.empty((e, n, k), dtype=torch.bfloat16, device=y.device)
        qs = torch.empty((e, n, k // 2), dtype=torch.uint8, device=y.device)
        scales_out = torch.empty((e, n // 64, k * 2), dtype=torch.uint8, device=y.device)

        for i in range(e):
            packed, scales = per_token_cast_to_fp4(
                y[i],
                use_ue8m0=True,
                gran_k=groupsize,
                use_packed_ue8m0=False,
            )
            refs[i].copy_(cast_back_from_fp4(packed, scales, gran_k=groupsize).to(torch.bfloat16))
            qs[i].copy_(packed.view(torch.uint8))

            # Convert numerical powers of two into raw E8M0 exponent bytes.
            scales = scales.reshape(n, k // groupsize)
            scales = ((scales.to(torch.float32).view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
            # [N,K/32] -> [N/64,64,K/64,2] -> [N/64,K/64,64,2] -> perm -> [N/64,K*2]
            scales = scales.reshape(n // 64, 64, k // 64, 2).permute(0, 2, 1, 3).reshape(n // 64, k * 2)
            scales = scales.reshape((-1, len(_scale_perm_fp4_mma)))[:, _scale_perm_fp4_mma].reshape(scales.shape).contiguous()
            scales_out[i].copy_(scales)

        return refs, qs, scales_out

    all_refs, all_qs, all_scales = [], [], []

    for i in range(e):
        w = y[i].T.contiguous()
        wk, wn = w.shape

        w = w.reshape((-1, groupsize, wn))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

        if quant_format == 'int4':
            s = torch.max(torch.abs(w), 0, keepdim=True)[0]
            s *= 2 / maxq
            w_q = torch.round(w / s).int()
            w_q += (maxq + 1) // 2
            w_q = torch.clamp(w_q, 0, maxq)
            ref = (w_q - (maxq + 1) // 2) * s
        else:  # 'fp4'
            s = torch.max(torch.abs(w), 0, keepdim=True)[0]
            s = s / 6.0
            s = s.clamp_min(1e-4)
            if use_e8m0:
                s = ceil_to_ue8m0(s)
            w_q = _quantize_to_fp4_e2m1(w / s)
            ref = _dequantize_from_fp4_e2m1(w_q) * s

        def reshape(t):
            t = t.reshape((groupsize, -1, wn))
            t = t.permute(1, 0, 2)
            t = t.reshape((wk, wn)).contiguous()
            return t

        ref = reshape(ref)
        w = reshape(w_q)
        s = s.reshape((-1, wn)).contiguous()

        if use_e8m0:
            # E8M0 perm includes the extra [0, 2, 1, 3] swap
            s = s.reshape((-1, len(_scale_perm_e8m0)))[:, _scale_perm_e8m0]
            s = s.reshape((-1, wn)).contiguous()
            # ceil_to_ue8m0 keeps the input dtype (bf16), so widen to fp32 before extracting the exponent bits.
            s = ((s.to(torch.float32).view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        else:
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
            s = s.reshape((-1, wn)).contiguous()

        w = w.reshape((wk // tile, tile, wn // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((wk // tile, wn * tile))
        w = w.reshape((-1, _perm.numel()))[:, _perm].reshape(w.shape)

        w = w.to(torch.int32)
        q = torch.zeros((w.shape[0], w.shape[1] // 8), dtype=torch.int32, device=w.device)
        for j in range(8):
            q |= w[:, j::8] << (4 * j)

        all_refs.append(ref.T)
        all_qs.append(q)
        all_scales.append(s)

    refs = torch.stack(all_refs)
    if quant_format == 'fp4':
        refs = refs.to(torch.bfloat16)
    qs = torch.stack(all_qs)
    scales = torch.stack(all_scales)
    return refs, qs, scales
