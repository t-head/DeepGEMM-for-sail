import os
import gc
import sys
import torch
import random
import deep_gemm
from utils import parse_deepgemm_string_re, read_cmds_from_file, get_arch_version
from utils import test_gemm, test_m_grouped_gemm_contiguous, test_m_grouped_gemm_masked, test_m_grouped_gemm_nopad, test_m_grouped_gemm_fused
from utils import test_mqa_logits, test_paged_mqa_logits
from utils import set_acc_check, set_benchmark
from utils import judge_device_type, set_ref_backend
import atexit

def device_version_supported(one_case):
    """Per-case device version gate.

    Returns (supported, reason). Cases that rely on hardware capabilities
    absent on the current device are skipped instead of failing, so the same
    caselist can run on both PPU1.0 and PPU1.5.
    """
    data_type = one_case.get('data_type')
    # fp4 (mxfp4, ue8m0 scale) GEMM/attention kernels require PPU1.5 (sm_89) or later,
    # see the device assert in deep_gemm/jit_kernels/gemm_fp4.py
    if data_type == torch.uint8 and get_arch_version() < 89:
        return False, "fp4(mxfp4) requires PPU1.5 (ZW890, sm_89) or later"
    # fp8 GEMM kernels hang on PPU1.0 (ZW810E), require PPU1.5 (sm_89) or later
    if data_type == torch.float8_e4m3fn and get_arch_version() < 89:
        return False, "fp8 requires PPU1.5 (ZW890, sm_89) or later"
    return True, ""
def device_sync_at_exit():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("Device synchronized on exit.")
atexit.register(device_sync_at_exit)
def call_test_func(gemm_type, func_args):
    supported_call_funcs = {
        "GroupedContiguous": test_m_grouped_gemm_contiguous,
        "GroupedNoPad": test_m_grouped_gemm_nopad,
        "GroupedFused" : test_m_grouped_gemm_fused,
        "GroupedMasked": test_m_grouped_gemm_masked,
        "DenseGemm": test_gemm,
        "Normal": test_gemm,
        "MqaLogits": test_mqa_logits,
        "PagedMqaLogits": test_paged_mqa_logits,
        # Avg variants reuse the same test functions, which branch on gemm_type
        "MqaAvgLogits": test_mqa_logits,
        "PagedMqaAvgLogits": test_paged_mqa_logits,
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

    import argparse

    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--format',  type=str, default=None, help="Case cmd to describe problem size.")
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the folder of DG cases')
    parser.add_argument('--case_idx', default=None, type=int, required=False, help='the line index(1~line) of case in caselist file')
    parser.add_argument("--disable_acc", action="store_true", help="if or not open accuracy check")
    parser.add_argument('--benchmark', default=False, action="store_true", required=False, help='specify if run benchmark')
    parser.add_argument('--ref_backend', default="device", type=str, required=False, choices=["host", "device"], help='specify the backend used to compute ref output')

    args = parser.parse_args()
    set_benchmark(args.benchmark)
    set_ref_backend(args.ref_backend)
    if args.disable_acc:
        set_acc_check(0)
    dg_cases = list()
    if args.format:
        one_case = parse_deepgemm_string_re(args.format)
        dg_cases.append(one_case)
    elif args.caselist:
        dg_cases = read_cmds_from_file(os.path.abspath(args.caselist))
        if len(dg_cases) == 0:
            print("no dg_cases found")
            exit(-1)
        if args.case_idx is not None:
            if not 1 <= args.case_idx <= len(dg_cases):
                print(f"ERROR: --case_idx must be between 1 and {len(dg_cases)}")
                sys.exit(2)
            dg_cases = [dg_cases[args.case_idx-1]]
    else:
        print("ERROR: must give --caselist or --format")
        exit(1)
    total = len(dg_cases)
    failures = []
    skipped = []
    for idx, one_case in enumerate(dg_cases):
        print(f'Profiling {idx + 1}/{total}')
        print(f'case info:{one_case}')
        supported, reason = device_version_supported(one_case)
        if not supported:
            skipped.append((idx + 1, reason))
            print(f"> Skipped (device version gate): {reason}\n", flush=True)
            continue
        try:
            call_test_func(one_case['gemm_type'], one_case)
        except Exception as e:
            failures.append((idx + 1, str(e)))
            print(f"❌ Test {idx + 1} failed with error: {e}")
            del e  # release traceback-held tensors immediately
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
