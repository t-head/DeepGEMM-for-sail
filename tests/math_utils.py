import torch
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