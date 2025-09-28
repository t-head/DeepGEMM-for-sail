import os
import torch
import random
import deep_gemm
from utils import parse_deepgemm_string_re, read_cmds_from_file
from utils import test_gemm, test_m_grouped_gemm_contiguous, test_m_grouped_gemm_masked, test_m_grouped_gemm_nopad
from utils import set_cycle
from utils import judge_device_type
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
        judge_device_type()
        return test_func(func_args)
    else:
        print("invalid test function\n")
        exit(1)

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
    parser.add_argument('--format',  type=str, default=None, help="Case cmd to describe problem size.")
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the folder of DG cases')
    parser.add_argument("--acc_check", action="store_true", help="compare with original torch impl")
#     parser.add_argument("--backend", default=None, type=str, required=False, help="different implementation")

    args = parser.parse_args()
    if (args.acc_check):
        set_cycle(0)
    dg_cases = list()
    if args.format:
        one_case = parse_deepgemm_string_re(args.format)
        dg_cases.append(one_case)
    elif args.caselist:
        dg_cases = read_cmds_from_file(os.path.abspath(args.caselist))
    else:
        print("ERROR: must give --caselist or --format")
        exit(1)
    total = len(dg_cases)
    for idx, one_case in enumerate(dg_cases):
        print(f'Profiling {idx + 1}/{total}')
        print(f'case info:{one_case}')
        call_test_func(one_case['gemm_type'], one_case)
