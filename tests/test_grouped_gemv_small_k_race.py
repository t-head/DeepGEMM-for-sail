"""Stress shared-slot reuse in the 810E BF16/INT8 small-K GEMV kernel."""
import os

import pytest
import torch

import deep_gemm as dg


@pytest.mark.parametrize('kind', ['bf16', 'int8'])
@pytest.mark.parametrize('k', [80, 128, 160, 256])
def test_grouped_gemv_small_k_slot_reuse(kind, k):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip('requires the 810E (SM80) small-K GEMV dispatch')

    iterations = int(os.getenv('DG_GEMV_RACE_ITERS', '32768'))
    assert iterations > 0, 'DG_GEMV_RACE_ITERS must be positive'
    groups, m, n = 512, 4 * 10, 2560
    dtype = torch.bfloat16 if kind == 'bf16' else torch.int8
    a = torch.ones((m, k), dtype=dtype, device='cuda')
    b = torch.zeros((groups, n, k), dtype=dtype, device='cuda')
    # The first column of each 64-column output tile is K; subsequent
    # columns are zero. An early refill can turn a live marker into zero.
    b[:, ::64, :] = 1
    ids = (torch.arange(m, dtype=torch.int32, device='cuda') * 13) % groups
    expected = torch.zeros((m, n), dtype=torch.bfloat16, device='cuda')
    expected[:, ::64] = k
    if kind == 'bf16':
        api = dg.m_grouped_bf16_gemm_nt_nopad
    else:
        api = dg.m_grouped_int8_gemm_nt_nopad
        a = a, torch.ones((m, 1), device='cuda')
        b = b, torch.ones((groups, n, 1), device='cuda')

    # Retain every call's output until checked, so a later correct call
    # cannot overwrite evidence of a transient failure.
    chunk_size = min(iterations, 32)
    outputs = torch.empty((chunk_size, m, n), dtype=torch.bfloat16, device='cuda')
    api(a, b, outputs[0], ids)
    if os.getenv('HGGC_WARM_UP') or dg.get_compile_mode() == 1:
        return
    torch.cuda.synchronize()

    wrong_calls = 0
    first_error = None
    for start in range(0, iterations, chunk_size):
        count = min(chunk_size, iterations - start)
        outputs[:count].fill_(-123)
        for i in range(count):
            api(a, b, outputs[i], ids)
        different = outputs[:count] != expected.unsqueeze(0)
        bad_calls = different.flatten(1).any(1)
        wrong_calls += int(bad_calls.sum().item())
        if first_error is None and bad_calls.any().item():
            call, row, col = different.nonzero()[0].tolist()
            first_error = (start + call, row, col, outputs[call, row, col].item(),
                           expected[row, col].item())
    print(f'{kind} K={k}: wrong_calls={wrong_calls}/{iterations}, first_error={first_error}', flush=True)
    assert wrong_calls == 0, f'shared-slot corruption: {wrong_calls} calls, first={first_error}'
