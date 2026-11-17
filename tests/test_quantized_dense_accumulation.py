"""Quantized dense GEMMs reject unsupported C before writing the output."""
import pytest
import torch

import deep_gemm as dg


def quantized_inputs(kind, m):
    n, k = 128, 256
    dtype = torch.int8 if kind == 'int8' else torch.float8_e4m3fn
    a = torch.ones((m, k), device='cuda').to(dtype)
    b = torch.ones((n, k), device='cuda').to(dtype)
    if kind == 'fp8-block':
        sa = torch.ones((m, k // 128), device='cuda')
        sb = torch.ones((n // 128, k // 128), device='cuda')
    else:
        sa = torch.ones((m, 1), device='cuda')
        sb = torch.ones((n, 1), device='cuda')
    out = torch.full((m, n), -7, dtype=torch.bfloat16, device='cuda')
    api = dg.int8_gemm_nt if kind == 'int8' else dg.fp8_gemm_nt
    return api, (a, sa), (b, sb), out


@pytest.mark.parametrize('kind', ['int8', 'fp8-channel', 'fp8-block'])
@pytest.mark.parametrize('m', [0, 16])
@pytest.mark.parametrize('alias', [False, True])
def test_quantized_dense_rejects_c(kind, m, alias):
    api, a, b, out = quantized_inputs(kind, m)
    c = out if alias else torch.full_like(out, 128)
    before = out.clone()
    with pytest.raises(RuntimeError, match='does not support C accumulation; pass c=None'):
        api(a, b, out, c=c)
    torch.testing.assert_close(out, before, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['int8', 'fp8-channel', 'fp8-block'])
@pytest.mark.parametrize('m', [0, 16])
def test_quantized_dense_without_c(kind, m):
    api, a, b, out = quantized_inputs(kind, m)
    api(a, b, out, c=None)
    torch.testing.assert_close(out, torch.full_like(out, 256), rtol=0, atol=0)
