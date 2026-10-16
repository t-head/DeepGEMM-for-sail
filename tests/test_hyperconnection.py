import torch
import random

import deep_gemm
from deep_gemm.testing import (
    test_filter,
    bench_kineto,
    calc_diff, count_bytes
)
from deep_gemm.testing import get_arch_major


@test_filter(lambda: get_arch_major() == 8)
def test_hc_prenorm_gemm() -> None:
    # Needs TF32 precision for PyTorch GEMMs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cases = [(m, n, k) for m in (13, 137, 4096, 8192)
             for n, k in ((24, 28672), (24, 7680), (24, 7168))]
    cases += [
        (m, 24, k)
        for m in (
            *range(2, 9), 16, 20, 24, 32, 60, 65, 128, 129,
            1248, 4096, 8192, 9984, 19968,
        )
        for k in (16384, 20480, 28672)
    ]
    cases.append((1024, 32, 64))
    cases = list(dict.fromkeys(cases))

    print('Testing hyperconnection prenorm GEMM:')
    for m, n, k in cases:
        a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
        b = torch.randn((n, k), dtype=torch.float, device='cuda')
        d = torch.empty((m, n), dtype=torch.float, device='cuda')
        s = torch.empty((m,), dtype=torch.float, device='cuda')
        deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=None)

        ref_d = a.float() @ b.T
        ref_s = a.float().square().sum(-1)
        diff = max(calc_diff(d, ref_d), calc_diff(s, ref_s))
        assert diff < 1e-8, f'{m=}, {n=}, {k=}, {diff:.10f}'

        times = bench_kineto(
            lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=None),
            ('tf32_hc_prenorm_gemm', 'tf32_hc_round_b'), suppress_kineto_output=True)
        t = sum(times)
        print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, num_splits= 0): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d, s) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    test_hc_prenorm_gemm()
