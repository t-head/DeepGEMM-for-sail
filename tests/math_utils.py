import torch
import numpy as np
from typing import Tuple, Iterable


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    return ((x + y - 1) // y) * y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def pack_ue8m0_to_int(x: torch.Tensor):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    assert (x.view(torch.int) & ((1 << 23) - 1) == 0).all()
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0=False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, 128)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous(), sf


def per_channel_cast_to_fp8_128align(x: torch.Tensor, use_ue8m0=False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(0) % 128 == 0
    m, n = x.shape
    x_view = x.view(-1, 128, n)
    x_amax = x_view.abs().float().amax(dim=1).view(-1, n).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(1))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0=False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(x_view.size(0), x_view.size(2))


def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims: Tuple, use_ue8m0=False, keep_scale_dim=False) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    if keep_scale_dim:
        return x_scaled, sf
    else:
        return x_scaled, sf.squeeze()


def per_token_cast_to_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape

    x_view = x.view(m, -1, n)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)

    scale = 127.0 / x_amax.unsqueeze(2)
    x_normalized = x_view * scale
    x_int8 = x_normalized.round().clamp(-128, 127).to(torch.int8)
    x_int8 = x_int8.view(m, -1)
    return x_int8, (x_amax / 127.0).view(m, -1)


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def per_token_cast_to_fp4(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128,
                          use_packed_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    assert n % 2 == 0
    assert not use_packed_ue8m0 or use_ue8m0
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)  # int8, (m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)  # int8
    return packed[:, :n // 2].contiguous(), pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


def transpose_packed_fp4(a: torch.Tensor) -> torch.Tensor:
    assert a.dtype == torch.int8
    assert a.dim() == 2
    m, n2 = a.shape
    n = n2 * 2
    assert (m % 2) == 0
    lo = a & 0x0F
    hi = (a >> 4) & 0x0F
    codes = torch.empty((m, n), device=a.device, dtype=torch.int8)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    codes_t = codes.transpose(0, 1).contiguous()
    codes2 = codes_t.view(n, m // 2, 2)
    out = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return out.contiguous()


def _dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=torch.float)
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def unpack_ue8m0_from_int(packed_sf: torch.Tensor) -> torch.Tensor:
    return (packed_sf.view(torch.uint8).to(torch.int) << 23).view(torch.float)


def cast_back_from_fp4(packed: torch.Tensor, sf: torch.Tensor, gran_k: int = 128,
                       use_packed_ue8m0: bool = False) -> torch.Tensor:
    m, n2 = packed.shape
    n = n2 * 2
    if use_packed_ue8m0:
        sf = unpack_ue8m0_from_int(sf)
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x_dequantized = _dequantize_from_fp4_e2m1(unpacked)
    group_idx = torch.arange(n, device=packed.device) // gran_k
    x_restored = x_dequantized * sf[:, group_idx]
    return x_restored


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
    return perm, scale_perm, scale_perm_chennel, scale_perm_e8m0

_perm, _scale_perm, _scale_perm_chennel, _scale_perm_e8m0 = _get_perms()


def quant_w4a16(y, groupsize=32, d='w4a16'):
    """
    Weight quantization for W4A16 (INT4), W4FA16 (FP4 with E8M0 scale) or
    W4FA16_S16 (FP4 with BF16 scale).

    Args:
        y: weight tensor, shape (e, n, k), dtype=torch.bfloat16
        groupsize: quantization group size, default 32
        d: 'w4a16' (default, INT4), 'w4fa16' (FP4 + E8M0 scale) or
           'w4fa16_s16' (FP4 + BF16 scale)

    Returns:
        refs: (e, k, n), dequantized reference for verification
        qs: (e, k // 16, n * 2), int32, packed weights (8 nibbles per int32)
        scales: (e, k // groupsize, n), per-group scale:
            bfloat16 numerical scale for 'w4a16'/'w4fa16_s16', or raw E8M0
            exponent bytes as uint8 for 'w4fa16' (same layout as SGLang's
            MXFP4 Marlin path)
    """
    assert d in ('w4a16', 'w4fa16', 'w4fa16_s16'), \
        f"d must be 'w4a16'/'w4fa16'/'w4fa16_s16', got {d}"
    quant_format = 'int4' if d == 'w4a16' else 'fp4'
    use_e8m0 = (d == 'w4fa16')
    e, n, k = y.shape
    tile = 16
    maxq = 2 ** 4 - 1
    assert k % groupsize == 0
    assert k % tile == 0 and n % tile == 0
    assert y.dtype in [torch.half, torch.bfloat16]

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
