"""Unsupported inputs must fail on the host before modifying output tensors."""
import pytest
import torch
import deep_gemm as dg


def dense_quantized(kind, m=16, n=128, k=256):
    dtype = torch.int8 if kind == 'int8' else torch.float8_e4m3fn
    a = torch.ones((m, k), device='cuda').to(dtype)
    b = torch.ones((n, k), device='cuda').to(dtype)
    block = kind == 'fp8-block'
    sa = torch.ones((m, (k + 127) // 128 if block else 1), device='cuda')
    sb = torch.ones(((n + 127) // 128, (k + 127) // 128) if block else (n, 1), device='cuda')
    out = torch.full((m, n), -123, dtype=torch.bfloat16, device='cuda')
    return (dg.int8_gemm_nt if kind == 'int8' else dg.fp8_gemm_nt), (a, sa), (b, sb), out


def reject(call, message, out):
    before = out.clone()
    with pytest.raises(RuntimeError, match=message):
        call()
    torch.testing.assert_close(out, before, rtol=0, atol=0)


def test_bug07_grouped_bf16_gemv_stride():
    a = torch.ones((1, 70), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((8, 128, 70), dtype=torch.bfloat16, device='cuda')
    out = torch.full((1, 128), -123, dtype=torch.bfloat16, device='cuda')
    ids = torch.zeros(1, dtype=torch.int32, device='cuda')
    reject(lambda: dg.m_grouped_bf16_gemm_nt_nopad(a, b, out, ids),
           'grouped BF16 GEMV requires', out)


@pytest.mark.parametrize('kind', ['int8', 'fp8-channel', 'fp8-block'])
def test_bug08_quantized_dense_odd_n(kind):
    api, a, b, out = dense_quantized(kind, n=65)
    reject(lambda: api(a, b, out), 'epilogue requires even N', out)


def test_bug08_fused_bf16_odd_n():
    a = torch.ones((1, 128), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((2, 65, 128), dtype=torch.bfloat16, device='cuda')
    out = torch.full((1, 65), -123, dtype=torch.bfloat16, device='cuda')
    meta = torch.zeros(4, dtype=torch.int32, device='cuda')
    reject(lambda: dg.m_grouped_bf16_gemm_nt_fused(a, b, out, meta, meta, meta, meta,
                                                  (1, 16, 64, 128, 16, 32, 2)),
           'epilogue requires even N', out)


@pytest.mark.parametrize('k', [80, 96])
def test_bug10_fused_bf16_k_tail(k):
    a = torch.ones((1, k), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((2, 64, k), dtype=torch.bfloat16, device='cuda')
    out = torch.full((1, 64), -123, dtype=torch.bfloat16, device='cuda')
    meta = torch.zeros(4, dtype=torch.int32, device='cuda')
    reject(lambda: dg.m_grouped_bf16_gemm_nt_fused(a, b, out, meta, meta, meta, meta,
                                                  (1, 16, 64, 64, 16, 16, 2)),
           'A-copy kernel does not mask K tails', out)
