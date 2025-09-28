import random
import torch
from typing import Tuple
import os

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_m_alignment_for_contiguous_layout
from utils import read_numbers_from_file, parse_dump_file, per_token_cast_to_int8
from utils import calc_diff, construct, construct_contiguous_grouped, construct_grouped_masked
from utils import set_cycle, cycle
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
                torch.testing.assert_close(out, ref_out, rtol=2e-1, atol=1)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    if file is not None:
        num_groups, m, n, k, expected_m_per_group = parse_dump_file(file)
        test_func(m, n, k, d)
        print("Passed\n")
        return

    for m in (64, 128, 4096):
        for k, n in [(576, 7168), (7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x, y, out, ref_out = construct(m, k, n, d)
            if d == torch.bfloat16:
                deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out)
            else:
                deep_gemm.gemm_int8_int8_bf16_nt(x, y, out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=2e-1, atol=1)
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

def test_m_grouped_gemm_contiguous(d: torch.dtype, file=None) -> None:
    print('Testing grouped contiguous GEMM:')

    def test_func(num_groups, m, expected_m_per_group, n, k):
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, expected_m_per_group, k, n, d, file, get_m_alignment_for_contiguous_layout())
        if (d == torch.bfloat16):
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
        else:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)

        if not cycle:
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=2e-1, atol=1)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    if file is not None:
        num_groups, m, n, k, expected_m_per_group = parse_dump_file(file)
        test_func(num_groups, m, expected_m_per_group, n, k)
    else:
        for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168),
                                                       (8, 4096, 7168, 4096), (8, 4096, 2048, 7168),
                                                       (32, 256, 7168, 4096), (32, 256, 2048, 7168)):

        # num_groups, expected_m_per_group, k, n = 2, 2, 256, 32
            test_func(num_groups, num_groups*expected_m_per_group, expected_m_per_group, k, n)

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

    def test_func(num_groups, m, n, k, expected_m_per_group):
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, expected_m_per_group, k, n, d, file, 1)
        if (d == torch.bfloat16):
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y, out, m_indices)
        else:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(x, y, out, m_indices)

        if not cycle:
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.0015:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=5e-1, atol=2)

            assert diff < 0.0015, f'{m=}, {k=}, {n=}, {diff:.5f}'

    if file is not None:
        num_groups, m, n, k, expected_m_per_group = parse_dump_file(file)
        test_func(num_groups, m, n, k, expected_m_per_group)
    else:
        for num_groups, expected_m_per_group in ((256, 1), (256, 4), (256, 16), (256, 32), (128, 8), (128, 64), (128, 1024)):
            for k, n in ((7168, 4096), (2048, 7168), (256, 768), (512, 128)):
        # num_groups, expected_m_per_group, k, n = 256, 1, 7168, 4096
                
                test_func(num_groups, num_groups*expected_m_per_group, n, k, expected_m_per_group)

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
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the folder of DG cases')
    parser.add_argument('--dtype', default="bf16", type=str, choices=["int8","bf16", "all"], required=False, help='data type of the cases')
    parser.add_argument('--func', default=None, type=str, choices=["DenseGemm","GroupedContiguous", "GroupedMasked", "GroupedNoPad"], required=False, help='target test func')

    args = parser.parse_args()
    if (args.cycle):
        set_cycle(1)
    else:
        set_cycle(0)
    if args.dtype == "all":
        args.dtype = "int8,bf16"
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
            dtype = torch.int8 if args.dtype == "int8" else torch.bfloat16
            if "GroupedContiguous" in file:
                test_m_grouped_gemm_contiguous(dtype, file)
            elif "GroupedMasked" in file:
                test_m_grouped_gemm_masked(dtype, file)
            elif "GroupedNoPad" in file:
                test_m_grouped_gemm_nopad(dtype, file)
            elif "DenseGemm" in file:
                test_gemm(dtype, file)
            else:
                print("invalid dump file\n")
    else:
        if args.func is not None:
            if "GroupedContiguous" in args.func:
                test_m_grouped_gemm_contiguous(torch.int8, args.file)
                test_m_grouped_gemm_contiguous(torch.bfloat16, args.file)
            elif "GroupedMasked" in args.func:
                test_m_grouped_gemm_masked(torch.int8, args.file)
                test_m_grouped_gemm_masked(torch.bfloat16, args.file)
            elif "GroupedNoPad" in args.func:
                test_m_grouped_gemm_nopad(torch.int8, args.file)
                test_m_grouped_gemm_nopad(torch.bfloat16, args.file)
            elif "DenseGemm" in args.func:
                test_gemm(torch.int8, args.file)
                test_gemm(torch.bfloat16, args.file)
            else:
                print("invalid test function\n")
        else:
            test_gemm(torch.int8, args.file)
            test_m_grouped_gemm_contiguous(torch.int8, args.file)
            test_m_grouped_gemm_masked(torch.int8, args.file)
            test_m_grouped_gemm_nopad(torch.int8, args.file)

            test_gemm(torch.bfloat16, args.file)
            test_m_grouped_gemm_contiguous(torch.bfloat16, args.file)
            test_m_grouped_gemm_masked(torch.bfloat16, args.file)
            test_m_grouped_gemm_nopad(torch.bfloat16, args.file)


