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

def test_gemm(d: torch.dtype, file = None) -> None:
    print('Testing GEMM:')

    def test_func(m, n, k, d):
        print("test_gemm->test_func: ", m, n, k, d)
        x, y, out, ref_out = construct(m, k, n, d)
        if d == torch.bfloat16:
            deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out)
        else:
            deep_gemm.gemm_int8_int8_bf16_nt(x, y, out)
        if not cycle:
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    if file is not None:
        num_groups, m, n, k, expected_m_per_group = parse_dump_file(file)
        test_func(m, n, k, d)
        return

    for m in (64, 128, 4096):
        for k, n in [(576, 7168), (7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x, y, out, ref_out = construct(m, k, n, d)
            if d == torch.bfloat16:
                deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out)
            else:
                deep_gemm.gemm_int8_int8_bf16_nt(x, y, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            if benchmark:
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

def read_numbers_from_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                try:
                    number = int(stripped_line)
                    numbers.append(number)
                except ValueError:
                    print(f"Warning: skip invalid: {stripped_line}")
    return numbers

def parse_dump_file(file):
    import re, math
    if ("GroupedMasked" in file or "Contiguous" in file or "GroupedNoPad" in file):
        pattern = r'groups(\d+)_m(\d+)_n(\d+)_k(\d+)_em(\d+)'
        match = re.search(pattern, file)

        if match:
            num_groups = int(match.group(1))
            m = int(match.group(2))
            n = int(match.group(3))
            k = int(match.group(4))
            expected_m_per_group = int(match.group(5))

            print(f"m: {m}")
            print(f"n: {n}")
            print(f"k: {k}")
            print(f"expected_m_per_group: {expected_m_per_group}")
        else:
            print("Pattern not found.")
    elif "DenseGemm" in file:
        # print("DenseGemm found int file, ", file)
        pattern = r'm(\d+)_n(\d+)_k(\d+)'
        match = re.search(pattern, file)
        if match:
            num_groups = 1
            expected_m_per_group = 1
            m = int(match.group(1))
            n = int(match.group(2))
            k = int(match.group(3))
            print(f"m: {m}")
            print(f"n: {n}")
            print(f"k: {k}")
        else:
            print("Pattern not found.")
    else:
        print("GemmType not supported.")
    return num_groups, m, n, k, expected_m_per_group


def construct_contiguous_grouped(num_groups: int, expected_m_per_group: int, k: int, n: int, d: torch.dtype, file: str, alignment: int) -> \
        Tuple[int, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:

    if file is not None:
        index = read_numbers_from_file(file)
        m_indices = torch.tensor(index, device='cuda', dtype=torch.int32)
        m = expected_m_per_group
    else:
        group_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
        m = sum([ceil_div(x, alignment) * alignment for x in group_ms])
        m_indices = torch.empty(m, device='cuda', dtype=torch.int32)

    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    if file is not None:
        for i, group in enumerate(m_indices):
            if group != -1:
                ref_out[i] = x[i] @ y[group].t()
    else:
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

def construct_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int, d: torch.dtype, file: str):
    x = torch.randn((num_groups, max_m, k), device='cpu', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cpu', dtype=torch.bfloat16)

    out = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    if not cycle:
        ref_out = torch.einsum('gmk,gnk->gmn', x, y)
    else:
        ref_out = torch.empty_like(out)

    # Construct mask
    if file is not None:
        list_m = read_numbers_from_file(file)
        masked_m = torch.tensor(list_m, device='cuda', dtype=torch.int)
    else:
        masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
        for j in range(num_groups):
            masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m

    if d == torch.bfloat16:
        return x.to('cuda'), y.to('cuda'), masked_m, out, ref_out.to('cuda')
    else:
        x_int8 = (torch.empty_like(x, dtype=torch.int8), torch.empty((num_groups, max_m, 1), device='cpu', dtype=torch.float))
        y_int8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((num_groups, n, 1), device='cpu', dtype=torch.float))
        for i in range(num_groups):
            x_int8[0][i], x_int8[1][i] = per_token_cast_to_int8(x[i])
            y_int8[0][i], y_int8[1][i] = per_token_cast_to_int8(y[i])

        return (x_int8[0].to("cuda"), x_int8[1].to("cuda")), (y_int8[0].to("cuda"), y_int8[1].to("cuda")), masked_m, out, ref_out.to('cuda')

def test_m_grouped_gemm_contiguous(d: torch.dtype, file=None) -> None:
    print('Testing grouped contiguous GEMM:')

    def test_func():
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n, d, file, get_m_alignment_for_contiguous_layout())
        if (d == torch.bfloat16):
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
        else:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)

        if not cycle:
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    if file is not None:
        num_groups, expected_m_per_group, n, k, m = parse_dump_file(file)
        test_func()
    else:
        for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168),
                                                       (8, 4096, 7168, 4096), (8, 4096, 2048, 7168),
                                                       (32, 256, 7168, 4096), (32, 256, 2048, 7168)):

        # num_groups, expected_m_per_group, k, n = 2, 2, 256, 32
            test_func()

    if benchmark:
        # noinspection PyShadowingNames
        def test_func():
            if (d == torch.bfloat16):
                deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
            else:
                deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)

        t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
        valid_m = (m_indices != -1).sum().item()
        print(f' > Perf ((contiguous dtype={str(d)}, {num_groups=:2}, {expected_m_per_group=:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
        f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
        f'{(valid_m * k + num_groups * k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')

    print("Passed\n")


def test_m_grouped_gemm_masked(d: torch.dtype, file: str) -> None:
    print('Testing grouped masked GEMM:')

    def test_func():
        x, y, masked_m, out, ref_out = construct_grouped_masked(num_groups, max_m, expected_m_per_group, k, n, d, file)

        if (d == torch.bfloat16):
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)
        else:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)

        if not cycle:
            for j in range(num_groups):
                diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                if (masked_m[j] != 0):
                    assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

    if file is not None:
        num_groups, max_m, n, k, expected_m_per_group = parse_dump_file(file)
        test_func()
    else:
        # Test correctness
        # num_groups, expected_m_per_group, k, n = 4, 2, 128, 64
        for num_groups, expected_m_per_group in ((1, 1024), (2, 512), (4, 256)):
            for k, n in ((7168, 4096), (2048, 7168), ):
                for i in range(10):
                    max_m = 2048
                    test_func()

        if benchmark:
            # noinspection PyShadowingNames
            def test_func():
                if (d == torch.bfloat16):
                    deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)
                else:
                    deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)

            x, y, masked_m, out, ref_out = construct_grouped_masked(num_groups, max_m, expected_m_per_group, k, n, d, file)

            # Test performance with fixed shapes
            # noinspection PyUnboundLocalVariable
            valid_m = masked_m.sum().item()
            t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)

            print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
                f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')                
    print('passed\n')


def test_m_grouped_gemm_nopad(d: torch.dtype, file: str) -> None:
    print('Testing grouped unpad GEMM:')

    def test_func():
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n, d, file, 1)
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y, out, m_indices)

        if not cycle:
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    if file is not None:
        num_groups, expected_m_per_group, n, k, m = parse_dump_file(file)
        test_func()
    else:
        for num_groups, expected_m_per_group in ((256, 1), (256, 4), (256, 16), (256, 32), (128, 8), (128, 64), (128, 1024)):
            for k, n in ((7168, 4096), (2048, 7168), (256, 768), (512, 128)):
        # num_groups, expected_m_per_group, k, n = 4, 2, 128, 8
                test_func()

    print("Passed\n")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    global benchmark
    benchmark = 0

    import argparse

    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--file',  type=str, default=None, help="File path to be processed (optional).")
    parser.add_argument("--cycle", action="store_true", help="measure cycles instead of duration")

    args = parser.parse_args()
    global cycle
    cycle = 0
    if (args.cycle):
        cycle = 1

    if args.file is not None:
        if "int8" in args.file and "GroupedContiguous" in args.file:
            test_m_grouped_gemm_contiguous(torch.int8, args.file)
        elif "int8" in args.file and "GroupedMasked" in args.file:
            test_m_grouped_gemm_masked(torch.int8, args.file)
        elif "GroupedContiguous" in args.file:
            test_m_grouped_gemm_contiguous(torch.bfloat16, args.file)
        elif "GroupedMasked" in args.file:
            test_m_grouped_gemm_masked(torch.bfloat16, args.file)
        elif "GroupedNoPad" in args.file:
            test_m_grouped_gemm_nopad(torch.bfloat16, args.file)
        elif "DenseGemm" in args.file:
            test_gemm(torch.int8, args.file)
        else:
            "invalid dump file\n"
    else:
        test_gemm(torch.int8)
        test_m_grouped_gemm_contiguous(torch.int8, args.file)
        test_m_grouped_gemm_masked(torch.int8, args.file)

        test_gemm(torch.bfloat16)
        test_m_grouped_gemm_contiguous(torch.bfloat16, args.file)
        test_m_grouped_gemm_masked(torch.bfloat16, args.file)
        test_m_grouped_gemm_nopad(torch.bfloat16, args.file)


