"""Stress shared-slot reuse in the 810E BF16/INT8 small-K GEMV kernel."""
import os

import pytest
import torch

import deep_gemm as dg


def _require_810e():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip('requires the 810E (SM80) small-K GEMV dispatch')


def _check_calls(api, a, b, ids, expected, iterations, label):
    assert iterations > 0, 'stress iterations must be positive'
    # Retain each output until checked: a later call must not hide corruption.
    chunk_size = min(iterations, 32)
    outputs = torch.empty((chunk_size, *expected.shape), dtype=torch.bfloat16, device='cuda')
    api(a, b, outputs[0], ids)
    if os.getenv('HGGC_WARM_UP') or dg.get_compile_mode() == 1:
        return
    torch.cuda.synchronize()
    graph = None
    if os.getenv('DG_GEMV_RACE_GRAPH') == '1':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(chunk_size):
                api(a, b, outputs[i], ids)
        torch.cuda.synchronize()

    wrong_calls = 0
    first_error = None
    for start in range(0, iterations, chunk_size):
        count = min(chunk_size, iterations - start)
        outputs[:count].fill_(-123)
        if graph is None:
            for i in range(count):
                api(a, b, outputs[i], ids)
        else:
            graph.replay()
        different = outputs[:count] != expected.unsqueeze(0)
        bad_calls = different.flatten(1).any(1)
        wrong_calls += int(bad_calls.sum().item())
        if first_error is None and bad_calls.any().item():
            call, row, col = different.nonzero()[0].tolist()
            first_error = (start + call, row, col, outputs[call, row, col].item(),
                           expected[row, col].item())
    mode = 'graph' if graph is not None else 'eager'
    print(f'{label} {mode}: wrong_calls={wrong_calls}/{iterations}, first_error={first_error}', flush=True)
    assert wrong_calls == 0, f'shared-slot corruption: {wrong_calls} calls, first={first_error}'


@pytest.mark.parametrize('kind', ['bf16', 'int8'])
@pytest.mark.parametrize('k', [80, 128, 160, 256])
def test_grouped_gemv_small_k_slot_reuse(kind, k):
    _require_810e()

    iterations = int(os.getenv('DG_GEMV_RACE_ITERS', '32768'))
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

    _check_calls(api, a, b, ids, expected, iterations, f'{kind} K={k}')


@pytest.mark.parametrize('k', [144, 160, 192, 240, 256])
def test_grouped_bf16_small_k_dyadic(k):
    _require_810e()
    # Every product and FP32 sum is exact; this checks all columns with signed,
    # nonuniform values without depending on a particular reduction order.
    rng = torch.Generator().manual_seed(89481340 + k)
    groups, m, n = 64, 17, 128
    a_cpu = torch.randint(-4, 5, (m, k), generator=rng).bfloat16() / 8
    b_cpu = torch.randint(-4, 5, (groups, n, k), generator=rng).bfloat16() / 8
    ids_cpu = (torch.arange(m, dtype=torch.int32) * 13 + 7) % groups
    expected = torch.stack([a_cpu[i].float() @ b_cpu[int(ids_cpu[i])].float().T
                            for i in range(m)]).bfloat16().cuda()
    iterations = int(os.getenv('DG_GEMV_RACE_RANDOM_ITERS', '1024'))
    _check_calls(dg.m_grouped_bf16_gemm_nt_nopad, a_cpu.cuda(), b_cpu.cuda(),
                 ids_cpu.cuda(), expected, iterations, f'bf16 dyadic K={k}')
