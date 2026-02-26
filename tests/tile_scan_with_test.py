import random
import torch
from typing import Tuple
import os
import argparse
import torch.multiprocessing as mp
import copy

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_m_alignment_for_contiguous_layout, get_col_major_tensor
from deep_gemm.jit_kernels.gemm_int8 import get_smem_config
from deep_gemm.jit_kernels.utils import get_search_space, get_num_sms
from utils import read_numbers_from_file, parse_dump_file, parse_deepgemm_string_re, judge_device_type
from utils import construct, construct_contiguous_grouped, construct_grouped_masked, split_list_into_groups
from deepgemm_tools import get_supported_configs, get_best_configs

def call_test_func(gemm_type, func_args):
    supported_call_funcs = {
        "GroupedContiguous": test_m_grouped_gemm_contiguous,
        "GroupedNoPad": test_m_grouped_gemm_nopad,
        "GroupedMasked": test_m_grouped_gemm_masked,
        "DenseGemm": test_gemm,
        "Normal": test_gemm,
    }
    if gemm_type in supported_call_funcs.keys():
        test_func = supported_call_funcs[gemm_type]
        data_type = func_args['data_type'] if func_args else torch.int8
        return test_func(data_type, func_args)
    else:
        print("currently not support search tile for gemm_type: ", gemm_type)
        exit(1)

def test_func_dense(cycle, tid, m, n, k, d, tile_list, x, y, out, ref_out):
    print("tid = ", tid, ' tile_list = ', tile_list)
    for tile_config in tile_list:
        print("scan_tile = ", tile_config)
        print("test_gemm->test_func: ", m, n, k, d)
        if d == torch.bfloat16:
            deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out, tile_config)
        elif d == torch.float8_e4m3fn:
            deep_gemm.gemm_fp8_fp8_bf16_nt(x, y, out, tile_config)
        elif d == torch.uint8:
            m, n = out.shape
            bias = torch.zeros((m, n), device='cuda', dtype=torch.float)
            deep_gemm.gemm_fp4_fp4_bf16_nt(x, y, bias, out, tile_config)
        else:
            deep_gemm.gemm_int8_int8_bf16_nt(x, y, out, tile_config)
        if not cycle and not os.environ.get('HGGC_WARM_UP', False):
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
            else:
                print("Passed")
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

def test_gemm(d: torch.dtype, args = None) -> None:
    print('Testing GEMM:')
    m, n, k, num_group = 64, 2304, 4096, 1
    if args:
        m, n, k = args['m'], args['n'], args['k']
    else:
        print('use default testcase')

    x, y, out, ref_out = construct(m, k, n, d)
    tile_list = get_tile_list(d, m, n, k, num_group, 'dense', True)

    enable_multithread = not bool(cycle)
    if enable_multithread:
        tile_idx = [i for i in range(len(tile_list))]
        tile_group = split_list_into_groups(tile_idx, thread_count)
        processes = []

        print("tile_group = ", tile_group)
        mp.set_start_method('spawn', force=True)
        for tid in range(thread_count):
            p = mp.Process(target=test_func_dense, args=(cycle, tid, m, n, k, d, [tile_list[x] for x in tile_group[tid]], x, y, out, ref_out))
            p.start()
            processes.append(p)

        # wait for all sub-process done
        for p in processes:
            p.join()
    else:
        test_func_dense(cycle, 0, m, n, k, d, tile_list, x, y, out, ref_out)

def test_func_contiguous(cycle, tid, m, n, k, d, tile_list, x, y, out, m_indices, distribute, ref_out):
    print("tid = ", tid, ' tile_list = ', tile_list)
    for tile_config in tile_list:
        print("scan_tile = ", tile_config)
        print("test_func_contiguous: ", m, n, k, d)
        if d == torch.bfloat16:
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices, tile_config)
        elif d == torch.int8:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices, tile_config)
        else:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x, y, out, m_indices, tile_config)

        if not cycle and not os.environ.get('HGGC_WARM_UP', False):
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=2e-1, atol=1)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

def test_m_grouped_gemm_contiguous(d: torch.dtype, args=None) -> None:
    print('Testing grouped contiguous GEMM:')
    num_groups, expected_m_per_group, m, n, k, distribution = 128, 4, 512, 4096, 384, None
    if args:
        num_groups, expected_m_per_group, m, n, k, distribution = args['groups'], args['em'], args['m'], args['n'], args['k'], args['distribution']
    else:
        print('use default testcase')
    m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, k, n, d, distribution, get_m_alignment_for_contiguous_layout())
    tile_list = get_tile_list(d, m, n, k, num_groups, 'contiguous', True)

    enable_multithread = not bool(cycle)
    if enable_multithread:
        tile_idx = [i for i in range(len(tile_list))]
        tile_group = split_list_into_groups(tile_idx, thread_count)
        processes = []

        print("tile_group = ", tile_group)
        mp.set_start_method('spawn', force=True)
        for tid in range(thread_count):
            p = mp.Process(target=test_func_contiguous, args=(cycle, tid, m, n, k, d, [tile_list[x] for x in tile_group[tid]], x, y, out, m_indices, distribution, ref_out))
            p.start()
            processes.append(p)

        # wait for all sub-process done
        for p in processes:
            p.join()
    else:
        test_func_contiguous(cycle, 0, m, n, k, d, tile_list, x, y, out, m_indices, distribution, ref_out)
    return

    print("Passed\n")

def test_func_masked(cycle, tid, m, n, k, d, tile_list, x, y, out, masked_m, em, num_groups, distribute, ref_out):
    print("tid = ", tid, ' tile_list = ', tile_list)
    for tile_config in tile_list:
        print("scan_tile = ", tile_config)
        print("test_func_masked: ", m, n, k, d)

        if (d == torch.bfloat16):
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, em, tile_config)
        elif d == torch.float8_e4m3fn:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x, y, out, masked_m, em, tile_config)
        else:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, em, tile_config)

        if not cycle and not os.environ.get('HGGC_WARM_UP', False):
            for j in range(num_groups):
                diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                if (masked_m[j] != 0):
                    assert diff < 0.001, f'{em=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

def test_m_grouped_gemm_masked(d: torch.dtype, args) -> None:
    print('Testing grouped masked GEMM:')
    num_groups, expected_m_per_group, max_m, n, k, distribution = 128, 4, 512, 4096, 384, "uniform"
    if args:
        num_groups, expected_m_per_group, max_m, n, k, distribution = args['groups'], args['em'], args['m'], args['n'], args['k'], args['distribution']
    else:
        print('use default testcase')

    if distribution:
        distribute = None
    else:
        distribute = torch.tensor(distribution, dtype=torch.int32, device='cuda')

    x, y, masked_m, out, ref_out, signal, max_m = construct_grouped_masked(num_groups, max_m, expected_m_per_group, k, n, d, distribution)

    tile_list = get_tile_list(d, max_m, n, k, num_groups, 'masked', True)

    enable_multithread = not bool(cycle)
    if enable_multithread:
        tile_idx = [i for i in range(len(tile_list))]
        tile_group = split_list_into_groups(tile_idx, thread_count)
        processes = []

        print("tile_group = ", tile_group)
        mp.set_start_method('spawn', force=True)
        for tid in range(thread_count):
            p = mp.Process(target=test_func_masked, args=(cycle, tid, max_m, n, k, d,
                [tile_list[x] for x in tile_group[tid]], x, y, out, masked_m, expected_m_per_group, num_groups, distribute, ref_out))
            p.start()
            processes.append(p)

        # wait for all sub-process done
        for p in processes:
            p.join()
    else:
        test_func_masked(cycle, 0, max_m, n, k, d, tile_list, x, y, out, masked_m, expected_m_per_group, num_groups, distribution, ref_out)
    print('Passed\n')


def test_func_nopad(cycle, tid, m, n, k, d, tile_list, x, y, out, m_indices, distribute, ref_out):
    print("tid = ", tid, ' tile_list = ', tile_list)
    for tile_config in tile_list:
        print("scan_tile = ", tile_config)
        print("test_func_nopad: ", m, n, k, d)

        if (d == torch.bfloat16):
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y, out, m_indices, distribute, tile_config)
        elif d == torch.float8_e4m3fn:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(x, y, out, m_indices, distribute, tile_config)
        elif d == torch.uint8:
            y_value, _ = y
            num_groups, n, k = y_value.shape
            bias = torch.zeros((num_groups, n), device='cuda', dtype=torch.float)
            deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(x, y, bias, out, m_indices, distribute, tile_config)
        else:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(x, y, out, m_indices, distribute, tile_config)

        if not cycle and not os.environ.get('HGGC_WARM_UP', False):
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)

            if diff >= 0.0015:
                print("ref_out:", ref_out)
                print("out:", out)
            else:
                print("Passed")
            assert diff < 0.0015, f'{m=}, {k=}, {n=}, {diff:.5f}'

def test_m_grouped_gemm_nopad(d: torch.dtype, args = None) -> None:
    print('Testing grouped unpad GEMM:')
    num_groups, m, n, k, distribution = 128, 512, 4096, 384, "uniform"
    if args:
        num_groups, m, n, k, distribution = args['groups'], args['m'], args['n'], args['k'], args['distribution']
    else:
        print('use default testcase')

    if args['distribution'] == "uniform":
        distribute = None
    else:
        distribute = torch.tensor(distribution, dtype=torch.int32, device='cuda')

    if d == torch.uint8:
        from test_fp4_core import construct_grouped
        x, y, m_indices, bias, out, ref_out = construct_grouped(num_groups, m, k, n, distribution, 1)
        ref_out = ref_out - bias
    else:
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, k, n, d, distribution, 1)
    tile_list = get_tile_list(d, m, n, k, num_groups, 'nopad', True)
    '''
    block_m, block_n, block_k, warp_m, warp_n, num_stages = 64, 256, 128, 32, 32, 3
    smem_config = get_smem_config(num_stages, k, block_m, block_n, block_k, 1)
    sm = get_num_sms()
    tile_list = [(sm, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config)]
    '''

    enable_multithread = not bool(cycle)
    if enable_multithread:
        tile_idx = [i for i in range(len(tile_list))]
        tile_group = split_list_into_groups(tile_idx, thread_count)
        processes = []

        print("tile_group = ", tile_group)
        mp.set_start_method('spawn', force=True)
        for tid in range(thread_count):
            p = mp.Process(target=test_func_nopad, args=(cycle, tid, m, n, k, d, [tile_list[x] for x in tile_group[tid]], x, y, out, m_indices, distribute, ref_out))
            p.start()
            processes.append(p)

        # wait for all sub-process done
        for p in processes:
            p.join()
    else:
        test_func_nopad(cycle, 0, m, n, k, d, tile_list, x, y, out, m_indices, distribute, ref_out)
    return

def get_tile_list(d: torch.dtype, m: int, n: int, k: int, num_groups: int, gemm_type: str=None, from_deepgemm: bool=True):
    """
    Returns search space according input gemm type

    Arguments:
        gemm_type: nopad, masked, dense

    Returns:
        The a tuple like get_best_configs()
    """
    if from_deepgemm:
        search_space = get_search_space(d, gemm_type, m, n, k)
        config_list = []
        for tile in search_space:
            block_m, block_n, warp_m, warp_n, block_k, num_stages = tile
            smem_config = get_smem_config(num_stages, k, block_m, block_n, block_k, 1)
            sm = get_num_sms()
            config_list.append((sm, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config))
        return config_list
    else:
        return get_supported_configs(m, n, k, num_groups, 39)

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--format',  type=str, default=None, help="Case cmd to describe problem size.")
    parser.add_argument('--file',  type=str, default=None, help="File path to be processed (optional).")
    parser.add_argument("--cycle", action="store_true", help="measure cycles instead of duration")
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the folder of DG cases')
    parser.add_argument('--thread_count', default=16, type=int, required=False, help='the thread_count when run multi thread prebuild')

    args = parser.parse_args()

    global cycle, thread_count
    cycle = 0
    if (args.cycle):
        cycle = 1
    thread_count = args.thread_count
    print(f"cycle: {cycle}, thread_count: {thread_count}")
    if os.environ.get('HGGC_WARM_UP', False):
        print(f'WARM UP FOR COMPILING ...')

    judge_device_type()
    dg_cases = []
    if args.format:
        one_case = parse_deepgemm_string_re(args.format)
        dg_cases.append(one_case)
    elif args.caselist:
        if not os.path.isfile(args.caselist):
            print("[Warning] caselist shoulde be a file. {} is not exist.", args.caselist)
        with open(args.caselist, "r") as f:
            for line in f:
                one_case = parse_deepgemm_string_re(line)
                dg_cases.append(one_case)
    total = len(dg_cases)
    if len(dg_cases) == 0:
        print("[Warning] Please provide problem cases for tile scan, use --format or --caselist.")
    for idx, one_case in enumerate(dg_cases):
        print(f'Profiling {idx + 1}/{total}')
        print(f'case info:{one_case}')
        call_test_func(one_case['gemm_type'], one_case)
