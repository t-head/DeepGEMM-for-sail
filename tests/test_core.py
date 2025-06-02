import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_m_alignment_for_contiguous_layout

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

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def construct(m: int, k: int, n: int, d: torch.dtype) -> \
        Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    if d == torch.bfloat16:
        return x, y, out, ref_out
    else:
        x_int8, y_int8 = per_token_cast_to_int8(x), per_token_cast_to_int8(y)
        return x_int8, y_int8, out, ref_out

def test_gemm(d: torch.dtype) -> None:
    print('Testing GEMM:')
    for m in (64, 128, 4096):
        for k, n in [(576, 7168), (7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
    
    # m = 32
    # n = 32
    # k = 256

            x, y, out, ref_out = construct(m, k, n, d)
            if d == torch.bfloat16:
                deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out)
            else:
                deep_gemm.gemm_int8_int8_bf16_nt(x, y, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # Construct new tensors only once to avoid L2 cache acceleration (creating them puts them in L2)
            x, y, out, ref_out = construct(m, k, n, d)

            # noinspection PyShadowingNames
            def test_func():
                if d == torch.bfloat16:
                    deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out)
                else:
                    deep_gemm.gemm_int8_int8_bf16_nt(x, y, out)

            t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
            print(f' > Performance (dtype={str(d)}, m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print("Passed\n")


def construct_contiguous_grouped(num_groups: int, expected_m_per_group: int, k: int, n: int, d: torch.dtype) -> \
        Tuple[int, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    m_aligned = get_m_alignment_for_contiguous_layout()
    group_m_list = []

    m = 0
    for i in range(num_groups):
        group_m = m_aligned * random.randint(int(expected_m_per_group * 0.7) // m_aligned, int(expected_m_per_group * 1.3) // m_aligned)
        m += group_m
        group_m_list.append(group_m)

    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    # x = torch.ones((m, k), device='cuda', dtype=torch.bfloat16)
    # y = torch.ones((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    
    for i, group_m in enumerate(group_m_list):
        end = start + group_m
        m_indices[start:end] = i
        ref_out[start:end] = x[start:end] @ y[i].t()
        start = end
    
    if d == torch.bfloat16:
        return m, x, y, m_indices, out, ref_out
    else:
        x_int8 = per_token_cast_to_int8(x)
        y_int8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((num_groups, n, 1), device='cuda', dtype=torch.float))
        for i in range(num_groups):
            y_int8[0][i], y_int8[1][i] = per_token_cast_to_int8(y[i])

        return m, x_int8, y_int8, m_indices, out, ref_out

def test_m_grouped_gemm_contiguous(d: torch.dtype) -> None:
    print('Testing grouped contiguous GEMM:')

    for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):
        # TODO: make a stronger test
            # num_groups = 16
            # expected_m_per_group = 128
            # k = 16
            # n = 32

            m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n, d)
            if (d == torch.bfloat16):
                deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
            else:
                deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)
            diff = calc_diff(out, ref_out)

            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # Construct new tensors only once to avoid L2 cache acceleration (creating them puts them in L2)
            m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n, d)

            # noinspection PyShadowingNames
            def test_func():
                if (d == torch.bfloat16):
                    deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
                else:
                    deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)

            t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
            print(f' > Perf ((contiguous dtype={str(d)}, {num_groups=:2}, {expected_m_per_group=:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(m * k + num_groups * k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print("Passed\n")


def construct_grouped_masked(num_groups: int, m: int, k: int, n: int, d: torch.dtype):
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    # x = torch.ones((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    # y = torch.ones((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    if d == torch.bfloat16:
        return x, y, out, ref_out
    else:
        x_int8 = (torch.empty_like(x, dtype=torch.int8), torch.empty((num_groups, m, 1), device='cuda', dtype=torch.float))
        y_int8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((num_groups, n, 1), device='cuda', dtype=torch.float))
        for i in range(num_groups):
            x_int8[0][i], x_int8[1][i] = per_token_cast_to_int8(x[i])
            y_int8[0][i], y_int8[1][i] = per_token_cast_to_int8(y[i])

        return x_int8, y_int8, out, ref_out

def test_m_grouped_gemm_masked(d: torch.dtype) -> None:
    print('Testing grouped masked GEMM:')

    for num_groups, m in ((1, 1024), (2, 512), (4, 256), (32, 128), (64, 64), (256, 16)):
        for k, n in ((7168, 4096), (2048, 7168), ):
        # Test correctness

    # num_groups = 2
    # m = 32
    # k = 128
    # n = 32

            masked_m_candidates = list(filter(lambda candidate: candidate <= m, (8, 10, 20, 35, 50, 64, 128, 192, 256, 320, 384)))
            for i in range(10):
                x, y, out, ref_out = construct_grouped_masked(num_groups, m, k, n, d)

                masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)

                for j in range(num_groups):
                    masked_m[j] = random.choice(masked_m_candidates)
                expected_m = min(int(masked_m.float().mean()) + 1, m)

                if (d == torch.bfloat16):
                    deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m)
                else:
                    deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m)


                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{m=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
                
                # noinspection PyShadowingNames
                def test_func():
                    if (d == torch.bfloat16):
                        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m)
                    else:
                        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m)

                    # Test performance with fixed shapes
                    # noinspection PyUnboundLocalVariable
                    valid_m = masked_m.sum().item()
                    t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
                    print(f' > Perf ((masked dtype={str(d)}, {num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                        f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
                        f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
    print("Passed\n")

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm(torch.int8)
    test_m_grouped_gemm_contiguous(torch.int8)
    test_m_grouped_gemm_masked(torch.int8)

    test_gemm(torch.bfloat16)
    test_m_grouped_gemm_contiguous(torch.bfloat16)
    test_m_grouped_gemm_masked(torch.bfloat16)


