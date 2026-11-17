"""Grouped BF16 GEMV must multiply in FP32 before accumulating."""
import pytest
import torch

import deep_gemm as dg


@pytest.mark.parametrize('k', [128, 512])
def test_grouped_bf16_gemv_products(k):
    # (1 + 2^-7)^2 - (1 + 2^-6) = 2^-14, exactly in FP32/BF16.
    # Rounding the first product to BF16 instead produces zero.
    n, groups = 64, 8
    b = torch.zeros((groups, n, k), dtype=torch.bfloat16, device='cuda')
    b[:, :, 0] = 1 + 2**-7
    b[:, :, 1] = 1 + 2**-6
    for m in (1, 8):
        a = torch.zeros((m, k), dtype=torch.bfloat16, device='cuda')
        a[:, 0] = 1 + 2**-7
        a[:, 1] = -1
        indices = torch.arange(m, dtype=torch.int32, device='cuda') % groups
        out = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        dg.m_grouped_bf16_gemm_nt_nopad(a, b, out, indices)
        torch.testing.assert_close(out, torch.full_like(out, 2**-14), rtol=0, atol=0)
