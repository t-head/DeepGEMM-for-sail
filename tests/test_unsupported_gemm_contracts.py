"""Unsupported inputs must fail on the host before modifying output tensors."""
import pytest
import torch
import deep_gemm as dg


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
