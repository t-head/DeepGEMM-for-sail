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
    alignment = get_m_alignment_for_contiguous_layout()
    group_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])

    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)    
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)
    
    start = 0
    
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + ceil_div(group_m, alignment) * alignment
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        ref_out[start:aligned_end] = x[start:aligned_end] @ y[i].t()
        start = aligned_end
    ref_out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_out), ref_out)
    
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

    for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168),
                                                   (8, 4096, 7168, 4096), (8, 4096, 2048, 7168),
                                                   (32, 256, 7168, 4096), (32, 256, 2048, 7168)):
        # num_groups, expected_m_per_group, k, n = 4, 8192, 2048, 7168
        # NOTES: we should mask the unfilled part before calculating difference
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n, d)
        if (d == torch.bfloat16):
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
        else:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)
        out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
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


def construct_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int, d: torch.dtype):
    x = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    # x = torch.ones((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    # y = torch.ones((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    out = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    # Construct mask
    masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m

    if d == torch.bfloat16:
        return x, y, masked_m, out, ref_out
    else:
        x_int8 = (torch.empty_like(x, dtype=torch.int8), torch.empty((num_groups, max_m, 1), device='cuda', dtype=torch.float))
        y_int8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((num_groups, n, 1), device='cuda', dtype=torch.float))
        for i in range(num_groups):
            x_int8[0][i], x_int8[1][i] = per_token_cast_to_int8(x[i])
            y_int8[0][i], y_int8[1][i] = per_token_cast_to_int8(y[i])

        return x_int8, y_int8, masked_m, out, ref_out

def test_m_grouped_gemm_masked(d: torch.dtype) -> None:
    print('Testing grouped masked GEMM:')

    for num_groups, expected_m_per_group in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168), ):
            # Test correctness
            for i in range(10):
                x, y, masked_m, out, ref_out = construct_grouped_masked(num_groups, 4096, expected_m_per_group, k, n, d)

                if (d == torch.bfloat16):
                    deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)
                else:
                    deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)

                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{m=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
                
            # noinspection PyShadowingNames
            def test_func():
                if (d == torch.bfloat16):
                    deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)
                else:
                    deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)

            # Test performance with fixed shapes
            # noinspection PyUnboundLocalVariable
            valid_m = masked_m.sum().item()
            t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
            print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
                f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')

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


