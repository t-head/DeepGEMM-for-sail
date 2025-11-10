import random
import torch
from typing import Tuple
import os
import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor, get_col_major_tensor, get_m_alignment_for_contiguous_layout
from utils import read_numbers_from_file, parse_dump_file
from utils import judge_device_type
use_ppu = judge_device_type()
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)

    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)

    # Transpose earlier so that the testing will not trigger transposing kernels
    if use_ppu:
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    else:
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))

    return x_fp8, y_fp8, out, ref_out

def get_m_indices_from_index(index: torch.Tensor, expected_m_per_group, alignment : int) -> torch.Tensor:
    if alignment == 1: # for nopad
        m_row = torch.tensor(index, device='cuda', dtype=torch.int32)
        start = 0
        m_indices = torch.empty(expected_m_per_group, device='cuda', dtype=torch.int32)
        for i, group in enumerate(m_row):
            actual_end = start + group
            m_indices[start:actual_end] = i
            aligned_end = start + ceil_div(group, alignment) * alignment
            m_indices[actual_end:aligned_end] = -1
            start = aligned_end
    else: # for contiguous
        m_indices = torch.tensor(index, device='cuda', dtype=torch.int32)
    return m_indices

def construct_contiguous_grouped(num_groups: int, expected_m_per_group: int, k: int, n: int, file: str, alignment: int) -> \
        Tuple[int, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    if file is not None:
        index = read_numbers_from_file(file)
        m_indices = get_m_indices_from_index(index, expected_m_per_group, alignment)
        m = expected_m_per_group
    else :
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

    x_fp8 = per_token_cast_to_fp8(x)
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])
    if use_ppu:
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    else:
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))

    return m, x_fp8, y_fp8, m_indices, out, ref_out

def construct_masked_grouped(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int, file: str) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.zeros((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    # Construct mask
    if file is not None:
        list_m = read_numbers_from_file(file)
        masked_m = torch.tensor(list_m, device='cuda', dtype=torch.int)
    else:
        masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
        for j in range(num_groups):
            masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))

    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, max_m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # Transpose earlier so that the testing will not trigger transposing kernels
    if use_ppu:
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    else:
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, masked_m, out, ref_out


def test_gemm(file: str) -> None:
    print('Testing GEMM:')
    def test_func(m, n, k):
        print("test_gemm->test_func: ", m, n, k)
        x_fp8, y_fp8, out, ref_out = construct(m, k, n)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        if not cycle:
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    if file is not None:
        num_groups, m, n, k, expected_m_per_group = parse_dump_file(file)
        test_func(m, n, k)
        print("Passed\n")
        return

    for m in (64, 128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=2e-1, atol=1)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
            if benchmark:
                # Construct new tensors only once to avoid L2 cache acceleration (creating them puts them in L2)
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)

                # noinspection PyShadowingNames
                def test_func():
                    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

                t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
                print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                    f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                    f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print("Passed\n")

def test_m_grouped_gemm_contiguous(file: str) -> None:
    print('Testing grouped contiguous GEMM:')
    def test_func():
        m, x_fp8, y_fp8, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n, file, get_m_alignment_for_contiguous_layout())
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        if not cycle:
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=2e-1, atol=1)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    if file is not None:
        num_groups, expected_m_per_group, n, k, m = parse_dump_file(file)
        test_func()
    else:
        for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168),
                                                       (8, 4096, 7168, 4096), (8, 4096, 2048, 7168),
                                                       (32, 256, 7168, 4096), (32, 256, 2048, 7168)):
            m, x_fp8, y_fp8, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n, file, get_m_alignment_for_contiguous_layout())
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=2e-1, atol=1)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            if benchmark:
                # NOTES: we should mask the unfilled part before calculating difference
                m, x_fp8, y_fp8, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n, file, get_m_alignment_for_contiguous_layout())

                # noinspection PyShadowingNames
                def test_func():
                    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

                t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)
                valid_m = (m_indices != -1).sum().item()
                print(f' > Perf ({num_groups=:2}, {expected_m_per_group=:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                    f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
                    f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
    print("Passed\n")


def test_m_grouped_gemm_masked(file: str) -> None:
    print('Testing grouped masked GEMM:')
    def test_func():
        x_fp8, y_fp8, masked_m, out, ref_out = construct_masked_grouped(num_groups, max_m, expected_m_per_group, k, n, file)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m_per_group)

        if not cycle:
            for j in range(num_groups):
                diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                if (masked_m[j] != 0):
                    if diff >= 0.001:
                    # if True:
                        print(f"ref_out[{j}]:", ref_out[j, :masked_m[j].item()])
                        print(f"out[{j}]:", out[j, :masked_m[j].item()])
                        torch.testing.assert_close(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()], rtol=5e-1, atol=2)
                    assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

    if file is not None:
        num_groups, max_m, n, k, expected_m_per_group = parse_dump_file(file)
        test_func()
    else:

        for num_groups, expected_m_per_group in ((1, 1024), (2, 512), (4, 256)):
            for k, n in ((7168, 4096), (2048, 7168), ):
            # Test correctness

                for i in range(10):
                    x_fp8, y_fp8, masked_m, out, ref_out = construct_masked_grouped(num_groups, 4096, expected_m_per_group, k, n, file)

                    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m_per_group)
                    for j in range(num_groups):
                        diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                        if diff >= 0.001:
                            print(f"ref_out[{j}]:", ref_out[j, :masked_m[j].item()])
                            print(f"out[{j}]:", out[j, :masked_m[j].item()])
                            torch.testing.assert_close(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()], rtol=5e-1, atol=2)
                        assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
                if benchmark:
                    # Construct new tensors only once to avoid L2 cache acceleration (creating them puts them in L2)
                    x_fp8, y_fp8, masked_m, out, ref_out = construct_masked_grouped(num_groups, 4096, expected_m_per_group, k, n, file)

                    # noinspection PyShadowingNames
                    def test_func():
                        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m_per_group)

                    valid_m = masked_m.sum().item()
                    # Test performance with fixed shapes
                    t = bench_kineto(test_func, 'gemm', suppress_kineto_output=True)

                    print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                        f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
                        f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
    print("Passed\n")

def test_m_grouped_gemm_nopad(file: str) -> None:
    print('Testing grouped unpad GEMM:')

    def test_func():
        m, x_fp8, y_fp8, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n, file, 1)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(x_fp8, y_fp8, out, m_indices)

        if not cycle:
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.0015:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=5e-1, atol=2)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    if file is not None:
        num_groups, expected_m_per_group, n, k, m = parse_dump_file(file)
        test_func()
    else:
        for num_groups, expected_m_per_group in ((256, 1), (256, 4), (256, 16), (256, 32), (128, 8), (128, 64), (128, 1024)):
            for k, n in ((7168, 4096), (2048, 7168), (256, 768), (512, 128)):
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
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the folder of DG cases')
    parser.add_argument("--cycle", action="store_true", help="measure cycles instead of duration")
    parser.add_argument("--dtype",  default="fp8", type=str, required=False, help='data type of the cases, e.g. fp8, e4m3, e5m2')
    parser.add_argument('--func', default=None, type=str, choices=["DenseGemm","GroupedContiguous", "GroupedMasked", "GroupedNoPad"], required=False, help='target test func')

    args = parser.parse_args()
    global cycle
    cycle = 0
    if (args.cycle):
        cycle = 1
    dg_cases = list()
    if args.file is not None or args.caselist is not None:
        if args.file:
            dg_cases = [args.file]
        elif args.caselist:
            if ".dump" in args.caselist:
                dg_cases = [args.caselist]
            elif not os.path.isdir(args.caselist):
                print("args.caselist is a file!")
                with open(args.caselist, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line == "" or line.startswith("#"):
                            continue
                        dg_cases.append(line)
            else:
                print("args.caselist is a folder!")
                for root, dirs, files in os.walk(args.caselist):
                    for file in files:
                        full_path = os.path.join(root, file)
                        dg_cases.append(full_path)
        total = len(dg_cases)
        for idx, file in enumerate(dg_cases):
            print(f'Profiling {idx + 1}/{total}')
            print(f'case name:{file}')
            if "GroupedContiguous" in file:
                test_m_grouped_gemm_contiguous(file)
            elif "GroupedMasked" in file:
                test_m_grouped_gemm_masked(file)
            elif "GroupedNoPad" in file:
                test_m_grouped_gemm_nopad(file)
            elif "DenseGemm" in file:
                test_gemm(file)
            else:
                "invalid dump file\n"
    else:
        if args.func is not None:
            if "GroupedContiguous" in args.func:
                test_m_grouped_gemm_contiguous(args.file)
            elif "GroupedMasked" in args.func:
                test_m_grouped_gemm_masked(args.file)
            elif "GroupedNoPad" in args.func:
                test_m_grouped_gemm_nopad(args.file)
            elif "DenseGemm" in args.func:
                test_gemm(args.file)
            else:
                print("invalid func type\n")
        else:
            test_gemm(args.file)
            # test_m_grouped_gemm_contiguous(args.file)
            # test_m_grouped_gemm_masked(args.file)
            # test_m_grouped_gemm_nopad(args.file)
