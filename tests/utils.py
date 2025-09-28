import os
import csv
import subprocess
import re
import traceback
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor, get_m_alignment_for_contiguous_layout

import deep_gemm
import random
import torch
from typing import Tuple
from enum import Enum
import ast
global cycle
cycle = 1
global use_ppu

class DGStatus(Enum):
    Pass = 0
    Fail = 1
    Skip = 2
def judge_device_type():
    device_name = torch.cuda.get_device_name()
    use_ppu_ = (device_name.lower().find("ppu") != -1) or (device_name.lower().find("zw") != -1)
    if not any(k in device_name.lower() for k in ['ppu', 'zw','nvidia']):
        raise ValueError("Unrecognized device name: {}!".format(device_name))
    global use_ppu
    use_ppu = use_ppu_
    return use_ppu_

def set_cycle(value):
    global cycle
    cycle = value

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

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

def construct(m: int, k: int, n: int, d: torch.dtype) -> \
        Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    if not cycle:
        ref_out = x @ y.t()
    else:
        ref_out = torch.empty_like(out)

    if d == torch.bfloat16:
        return x, y, out, ref_out
    elif d == torch.int8:
        x_int8, y_int8 = per_token_cast_to_int8(x), per_token_cast_to_int8(y)
        return x_int8, y_int8, out, ref_out
    elif d == torch.float8_e4m3fn:
        x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
        # Transpose earlier so that the testing will not trigger transposing kernels
        if use_ppu:
            from deep_gemm import  get_col_major_tensor
            x_fp8 = (x_fp8[0], get_col_major_tensor(x_fp8[1]))
        else:
            x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
        return x_fp8, y_fp8, out, ref_out
    else:
        print("ERROR: Unsupported dtype, please check!")
        exit(1)

def unbincount(counts):
    return torch.repeat_interleave(torch.arange(len(counts)),torch.tensor(counts))

def bincount(counts, min_lenth):
    return torch.bincount(torch.tensor(counts), weights=None, minlength=min_lenth)

def construct_group_m_list(distribution, num_groups, expected_m_per_group):
    group_m_list = list()
    if distribution == None: # default value
        distribution = "uniform"
    if type(distribution) is list:
        return distribution
    elif distribution.endswith("dump") or distribution.endswith("bin"):
        print(f"Read dump from {distribution}")
        group_m_list = read_numbers_from_file(distribution)
        if len(group_m_list) < num_groups:
            for i in range(len(group_m_list), num_groups):
                group_m_list.append(0)
    elif distribution == "uniform":
        group_m_list = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    elif distribution == "normal" or distribution == "gaussian":
        # avg is expected_m_per_group, sigma is expected_m_per_group * 0.5
        group_m_list = [max(0, int(random.gauss(expected_m_per_group, expected_m_per_group * 0.5))) for _ in range(num_groups)]
    elif distribution == "zipf":
        import numpy as np
        np.random.seed(0)
        # 为zip分布加入扰动，避免大量相同的值
        dist = np.random.zipf(2.0, num_groups) +  [random.gauss(0, 1) for _ in range(num_groups)]
        scale = expected_m_per_group * num_groups / dist.sum()
        group_m_list = np.round(dist * scale).astype(int)
    else:
        print("ERROR: Unsupported distribution type, please check!")
        exit(1)
    return group_m_list


def construct_contiguous_grouped(num_groups: int, m: int, expected_m_per_group: int, k: int, n: int, d: torch.dtype, distribution: str, alignment: int) -> \
        Tuple[int, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    group_ms = construct_group_m_list(distribution, num_groups, expected_m_per_group)
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    x = x.to('cpu')
    y = y.to('cpu')

    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + ceil_div(group_m, alignment) * alignment
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        if not cycle:
            ref_out[start:aligned_end] = x[start:aligned_end] @ y[i].t()
        start = aligned_end

    if not cycle:
        ref_out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_out), ref_out)

    if d == torch.bfloat16:
        return m, x.to('cuda'), y.to('cuda'), m_indices, out, ref_out.to('cuda')
    elif d == torch.int8:
        x_int8 = per_token_cast_to_int8(x)
        y_int8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((num_groups, n, 1), device='cpu', dtype=torch.float))
        for i in range(num_groups):
            y_int8[0][i], y_int8[1][i] = per_token_cast_to_int8(y[i])
        return m, (x_int8[0].to("cuda"), x_int8[1].to("cuda")), (y_int8[0].to("cuda"), y_int8[1].to("cuda")), m_indices, out, ref_out.to('cuda')
    elif d == torch.float8_e4m3fn:
        assert m % 4 == 0, f'TMA alignment error: {m}'
        x_fp8 = per_token_cast_to_fp8(x)
        y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), k // 128), device='cpu', dtype=torch.float))
        for i in range(num_groups):
            y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])
        if use_ppu:
            from deep_gemm import  get_col_major_tensor
            x_fp8 = (x_fp8[0], get_col_major_tensor(x_fp8[1]))
        else:
            x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
        return m, (x_fp8[0].to('cuda'),x_fp8[1].to('cuda')), (y_fp8[0].to('cuda'), y_fp8[1].to('cuda')), m_indices, out, ref_out.to('cuda')
    else:
        print("ERROR: Unsupported dtype, please check!")
        exit(1)

def construct_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int, d: torch.dtype, distribution: str):
    x = torch.randn((num_groups, max_m, k), device='cpu', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cpu', dtype=torch.bfloat16)
    out = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)

    if not cycle:
        ref_out = torch.einsum('gmk,gnk->gmn', x, y)
    else:
        ref_out = torch.empty_like(out)

    # Construct mask
    list_m =  construct_group_m_list(distribution, num_groups, expected_m_per_group)
    masked_m = torch.tensor(list_m, device='cuda', dtype=torch.int)
    assert masked_m.amax().item() <= max_m

    if d == torch.bfloat16:
        return x.to('cuda'), y.to('cuda'), masked_m, out, ref_out.to('cuda')
    elif d == torch.int8:
        x_int8 = (torch.empty_like(x, dtype=torch.int8), torch.empty((num_groups, max_m, 1), device='cpu', dtype=torch.float))
        y_int8 = (torch.empty_like(y, dtype=torch.int8), torch.empty((num_groups, n, 1), device='cpu', dtype=torch.float))
        for i in range(num_groups):
            x_int8[0][i], x_int8[1][i] = per_token_cast_to_int8(x[i])
            y_int8[0][i], y_int8[1][i] = per_token_cast_to_int8(y[i])
        return (x_int8[0].to("cuda"), x_int8[1].to("cuda")), (y_int8[0].to("cuda"), y_int8[1].to("cuda")), masked_m, out, ref_out.to('cuda')
    elif d == torch.float8_e4m3fn:
        x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, max_m, k // 128), device='cpu', dtype=torch.float))
        y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cpu', dtype=torch.float))
        for i in range(num_groups):
            x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
            y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

        # Transpose earlier so that the testing will not trigger transposing kernels
        if use_ppu:
            from deep_gemm import  get_col_major_tensor
            x_fp8 = (x_fp8[0], get_col_major_tensor(x_fp8[1]))
        else:
            x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
        return (x_fp8[0].to('cuda'),x_fp8[1].to('cuda')), (y_fp8[0].to('cuda'), y_fp8[1].to('cuda')), masked_m, out, ref_out.to('cuda')
    else:
        print("ERROR: Unsupported dtype, please check!")
        exit(1)

def run_cmd(cmd: str, timeout=3600, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    print(f"Run command: {cmd}, timeout: {timeout}")
    try:
        ret = subprocess.run(args=cmd, timeout=timeout, shell=True, stdout=stdout, stderr=stderr, encoding="utf-8")
        if stdout:
            for line in ret.stdout.splitlines() + ret.stderr.splitlines():
                print(line)
        if ret.returncode != 0:
            print(f"Run command failed!")
        else:
            print(f"Run command succeed!")
        return ret
    except Exception as e:
        print(traceback.format_exc())
        return None

def str_to_list(s, type_func=int):
    """Convert a comma-separated string to a list of a specified type."""
    return [type_func(i.strip()) for i in s.split(',')]

def split_list_into_groups(lst, num):
    group_size = len(lst) // num
    remainder = len(lst) % num
    start = 0
    groups = []
    for i in range(num):
        groups.append([])
    for i in range(len(lst)):
        group_idx = i % num
        groups[group_idx].append(lst[i])
    return groups

def worker(gpu_id, cases, output, device, mode, acc_check):
    # 设置当前进程可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process {os.getpid()} is running on GPU {gpu_id}")
    run_cycle_on_device(cases, output, device, mode, acc_check, gpu_id)

def read_detail_from_nculog(filename):
    detail_info = {'m': 0, 'n': 0, 'k': 0}
    keyword_pattern = r"(GemmGrouped-BF16|GemmGrouped-FP8|GemmGrouped-INT8|GemV-BF16|GemV-Small-BF16)"
    pattern_dict = {"group": r"group:(\d+)", "problem": r"problem:\[(\d+), (\d+), (\d+)\]", "expected_m":r"expected_m:(\d+)", "gemm_type": r"gemm_type:(\w+)",
    "ThreadblockShape": r"ThreadblockShape\[(\d+), (\d+), (\d+)\]" , "WarpShape": r"WarpShape\[(\d+), (\d+), (\d+)\]", "kNumStages":r"kNumStages:(\d+)",
    "num_sms": r"num_sms:(\d+)", "max_active_tb_num": r"max_active_tb_num:(\d+)", "threadblock_count": r"threadblock_count:(\d+)",
    "smem_size":r"smem_size:(\d+)", "vreg": r"vreg:(\d+)", "stack": r"stack:(\d+)",
    "BlockSize":r"BlockSize:(\d+)", "NPerThread":r"NPerThread:(\d+)", "ThreadPerN":r"ThreadPerN:(\d+)", "NPerBlock":r"NPerBlock:(\d+)", "SWZL_SIZE_M":r"SWZL_SIZE_M:(\d+)"}
    with open(filename, newline='') as log_file:
        lines = log_file.readlines()
        for idx, line in enumerate(lines):
            keyword_match = re.search(keyword_pattern, line)
            if keyword_match:
                target_lines = "".join(lines[idx+1:idx+10])
                for _info, _pattern in pattern_dict.items():
                    _info_match = re.search(_pattern, target_lines)
                    if _info_match:
                        if len(_info_match.groups()) == 1 :
                            detail_info[_info] = _info_match.groups()[0]
                        else:
                            detail_info[_info] = _info_match.groups()
                    else:
                        detail_info[_info] = " "
                break
    detail_info['m'] = detail_info['problem'][0]
    detail_info['n'] = detail_info['problem'][1]
    detail_info['k'] = detail_info['problem'][2]
    # print(detail_info)
    return detail_info


# devices = {
#     "name": ["cycle", "tensor core efficiency", "waves"],
#      hopper ppu tc: sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.avg.pct_of_peak_sustained_elapsed
#     "gpu":  ["sm__cycles_active.max", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "launch__waves_per_multiprocessor"],
#     "ppu":  ["ce__cycles_active.max", "cu__inst_executed_pipe_tensor_fp16.avg.pct_of_peak_sustained_active", "launch__waves_per_cu"],
# }

def read_cycle_from_nculog(filename):
    # kernel_pattern = r"(.*)deep_gemm(.*)"
    # kernel_pattern = r"(.*)kernel(.*)Device(.*)"
    kernel_pattern = r"(.*)Device\s+\d+"
    cycles_pattern = r"__cycles_active.max"
    global tc_pattern
    # tc_pattern = "we_pipe_tensor_cycles_active"
    tc_pattern = "_tensor_(.*)avg.pct_of_peak_sustained"
    hbm_pattern = "dram"
    kernel_list = []
    cycles_list = []
    tc_list = []
    hbm_list = []

    with open(filename, newline='') as log_file:
        for line in log_file.read().split("\n"):
            if re.search(kernel_pattern, line):
                kernel_list.append(line.strip())
            if re.search(cycles_pattern, line):
                cycles_list.append(int(line.strip().split()[-1]))
            if re.search(tc_pattern, line):
                tc_list.append(float(line.strip().split()[-1]))
            if re.search(hbm_pattern, line):
                hbm_list.append(float(line.strip().split()[-1]))

    if (len(kernel_list) != len(cycles_list)) or (len(kernel_list) != len(tc_list)) or (len(kernel_list) != len(hbm_list)):
        print(f"assert len(kernel_list){len(kernel_list)} == len(cycles_list){len(cycles_list)} == len(tc_list){len(tc_list)} == len(hbm_list){len(hbm_list)} failed!!")
        return 0, 0, [], 0
    # assert(len(kernel_list) == len(cycles_list))
    # assert(len(kernel_list) == len(tc_list))
    # assert(len(kernel_list) == len(hbm_list))

    op_cycles = dict()
    fwd_cycle_sum = 0
    fwd_tc_sum = 0
    fwd_hbm_sum = 0

    for i in range(len(kernel_list)):
        op = kernel_list[i]
        cycle = cycles_list[i]
        op_cycles[op] = cycle
        if "deep_gemm" in op.lower():
            fwd_cycle_sum += cycle
            fwd_tc_sum += tc_list[i]
            fwd_hbm_sum += hbm_list[i]
    # calculate statistics data
    if fwd_cycle_sum != 0:
        # fwd unit case
        return fwd_cycle_sum, fwd_tc_sum, op_cycles, fwd_hbm_sum
    else:
        print("Not valid CSV file!")
        return 0, 0, [], 0
        #exit(-1)

def clean_casename(name):
    _need_replace = ['--', '=', 'format', 'Formatted', '[', ']', ":", "*", " ", ","]
    # for item in _need_replace:
    #     name = name.replace(item, "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    while "__" in name:
        name = name.replace("__", "_")
    return name

def run_cycle_on_device(cases, output_file, dev="gpu", mode="metrics", acc_check=False, gpu_id="0"):
    output_lines = list()
    headers = ["casename","cycle","tc efficiency", "hbm efficiency", "dtype", "result", "cmd", "detail"]
    # new_row=["casename"]  metrics.get("name", [])  ["detail"]
    # output_lines.append(new_row)
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    total = len(cases)
    for idx, case in enumerate(cases):
        print(f'Profiling {idx + 1}/{total} on device{gpu_id}')
        print(f'case name:{case}')
        log_file = f"./logs/gpu{gpu_id}_{case.replace(" ","").replace(",", "_").replace(":", "_").replace('/','_').replace('.','_')}.log"
        cmd = "rm -f "+ log_file
        run_cmd(cmd)
        # gpu
        # metrics = devices.get(dev, [])
        # metrics_string = ', '.join(metrics) if metrics else ""
        current_file_path = os.path.abspath(__file__)
        pattern = r"data_type:(bf16|int8|fp8)"
        dtype = re.search(pattern, case).groups()[0]
        script = f"{os.path.dirname(current_file_path)}/run_deep_gemm.py"
        if mode == "full":
            output_name = clean_casename(case)
            cmd = '{} --set full -o {} python {} --format {} \
                2>&1 | tee {}'.format("ncu" if dev == "gpu" else "acu", output_name, script, case, log_file)
        else:
            if mode == "show_log":
                os.environ["show_log"] = "1"
            # Ampere:"sm__cycles_active.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed"
            metrics_string = "sm__cycles_active.max,sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed" if dev=="gpu" else \
                            "ce__cycles_active.max,cu__inst_executed_pipe_tensor_{}.avg.pct_of_peak_sustained_active,dram__llc_bytes_read.sum.pct_of_peak_sustained_elapsed".format(dtype)
                            # "ce__cycles_active.max,cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__llc_bytes_read.sum.pct_of_peak_sustained_elapsed"

            cmd = '{} --clock-control none {} --metrics="{}"  \
                --page=details python {} --format {} \
                2>&1 | tee {}'.format("ncu" if dev == "gpu" else "acu", "" if dev == "gpu" else '--kernel-name "regex:Kernel|device_kernel"', metrics_string, script, case, log_file)

        ret = run_cmd(cmd)
        result = "Fail"
        cycle, tc, detail, hbm = 0, 0, "", 0
        if mode != "full" and ret != None:
            if ret.returncode == 0:
                cycle, tc, detail, hbm = read_cycle_from_nculog(log_file)
                if mode == "show_log":
                    other_metrics = read_detail_from_nculog(log_file)
                    print(other_metrics)

        if acc_check:
            # check accuracy
            cmd = "python {} --format {} --acc_check".format(script, case)
            ret = run_cmd(cmd)
            if ret.returncode == 0:
                result = "Pass"
            else:
                result = "Fail"
                print("ERROR: failed to pass accuracy test, please check!!")
        else:
            # only perf, check cycle found
            if cycle != 0:
                result = "Pass"
            else:
                result = "Fail"
                print("ERROR: failed to find cycle info, please check!!")
        row =  [f"'{case.replace(',','_')}_{dtype}'", str(cycle), str(tc), str(hbm), dtype, result, str(cmd), str(detail)]
        if mode == "show_log":
            if "problem" not in headers:
                headers.extend(other_metrics.keys())
            row.extend(other_metrics.values())
        output_lines.append(row)
        if not os.path.exists(f"{output_file}.csv"):
            with open(f"{output_file}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        with open(f"{output_file}.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            print("write result succeed")

        if len(cases) == 1 and result == "Fail":
            exit(-1) # only one case, fail and exit

    output_file = output_file + '.csv'
    if len(cases) == 1:
        with open("local.log", "w") as f:
            writer = csv.writer(f)
            for row in output_lines:
                writer.writerow(row)
            print("write result to local.log succeed")

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

def parse_deepgemm_string_re(s):
    # give default value
    result = {"distribution": "uniform"}
    supported_keys = ["data_type", "groups", "m", "n", "k", "em", "distribution"]
    supported_gemm_type = ["GroupedContiguous", "GroupedNoPad", "GroupedMasked", "Normal", "DenseGemm"]
    import re
    dg_params = r"(?:\[DeepGemm\] --format=|\[DeepGemm with Distribution\] --format=)?(GroupedContiguous|GroupedNoPad|GroupedMasked|DenseGemm|Normal),(.+)"
    pattern = re.compile(dg_params)
    m = pattern.match(s.strip("."))
    if not m:
        print("Invalid input format string did not match deepgemm params:", dg_params)
        exit(1)
    grps = m.groups()
    if grps[0] in supported_gemm_type:
        result['gemm_type'] = grps[0]
    param_pattern = r"(\w+):(\[.*?\]|[^,]+)"
    match_string = re.findall(param_pattern, grps[1])
    if len(match_string) == 0:
        print("ERROR: wrong deepgemm format input, please check!!")
    for key, value in match_string:
        if value.startswith('[') and value.endswith(']'):
            # 解析为列表
            result[key] = ast.literal_eval(value)
        elif key == "data_type":
            result[key] = convert_data_type_to_dtype(value)
        else:
            # 尝试转换为整数、浮点数等
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
                    pass
    return result

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

def convert_data_type_to_dtype(data_type):
    if data_type == "bf16":
        return torch.bfloat16
    elif data_type == "int8":
        return torch.int8
    elif data_type == "fp8":
        return torch.float8_e4m3fn
    else:
        print("ERROR: Unsupported dtype, please check!")
        exit(1)

def test_gemm(args) -> None:
    print('Testing GEMM:')
    def test_func(m, n, k, d):
        print("test_gemm->test_func: ", m, n, k, d)
        x, y, out, ref_out = construct(m, k, n, d)
        if d == torch.bfloat16:
            deep_gemm.gemm_bf16_bf16_bf16_nt(x, y, out)
        elif d == torch.int8:
            deep_gemm.gemm_int8_int8_bf16_nt(x, y, out)
        elif d == torch.float8_e4m3fn:
            deep_gemm.gemm_fp8_fp8_bf16_nt(x, y, out)
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)
        if not cycle:
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    test_func(args['m'], args['n'], args['k'], args["data_type"])
    print("Passed\n")
    return

def test_m_grouped_gemm_contiguous(args) -> None:
    print('Testing grouped contiguous GEMM:')

    def test_func(num_groups, m, n, k, expected_m_per_group, d, distribution):
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, expected_m_per_group, k, n, d, distribution, get_m_alignment_for_contiguous_layout())
        print("test_m_grouped_gemm_contiguous->test_func: ", num_groups, m, n, k, expected_m_per_group, d)
        if d == torch.bfloat16:
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(x, y, out, m_indices)
        elif d == torch.int8:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(x, y, out, m_indices)
        else:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x, y, out, m_indices)

        if not cycle:
            out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.001:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=2e-1, atol=1)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

    test_func(args['groups'], args['m'], args['n'], args['k'], args['em'], args['data_type'], args['distribution'])
    print("Passed\n")
    return

def test_m_grouped_gemm_masked(args) -> None:
    print('Testing grouped masked GEMM:')

    def test_func(num_groups, max_m, n, k, expected_m_per_group, d, distribution):
        print("test_m_grouped_gemm_masked->test_func: ", num_groups, max_m, n, k, expected_m_per_group, d)
        x, y, masked_m, out, ref_out = construct_grouped_masked(num_groups, max_m, expected_m_per_group, k, n, d, distribution)

        if d == torch.bfloat16:
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)
        elif d == torch.int8:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)
        elif d == torch.float8_e4m3fn:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x, y, out, masked_m, expected_m_per_group)
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)
        if not cycle:
            for j in range(num_groups):
                diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                if (masked_m[j] != 0):
                    if diff >= 0.001:
                        print(f"ref_out[{j}]:", ref_out[j, :masked_m[j].item()])
                        print(f"out[{j}]:", out[j, :masked_m[j].item()])
                        # torch.testing.assert_close(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()], rtol=5e-1, atol=2)
                    assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

    test_func(args["groups"], args['m'], args['n'], args['k'], args['em'], args['data_type'], args['distribution'])
    print("Passed\n")
    return

def test_m_grouped_gemm_nopad(args) -> None:
    print('Testing grouped unpad GEMM:')

    def test_func(num_groups, m, n, k, expected_m_per_group, d, distribution):
        print("test_m_grouped_gemm_nopad->test_func: ", num_groups, m, n, k, expected_m_per_group, d, distribution)
        m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, expected_m_per_group, k, n, d, distribution, 1)
        if d == torch.bfloat16:
            deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y, out, m_indices)
        elif d == torch.int8:
            deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(x, y, out, m_indices)
        elif d == torch.float8_e4m3fn:
            print("ERROR: fp8 + grouped_nopad not supported yet, please check!")
            exit(1)
            # deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(x, y, out, m_indices)
        else:
            print("ERROR: Unsupported dtype, please check!")
            exit(1)

        if not cycle:
            # out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
            diff = calc_diff(out, ref_out)
            if diff >= 0.0015:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=5e-1, atol=2)

            assert diff < 0.0015, f'{m=}, {k=}, {n=}, {diff:.5f}'

    test_func(args['groups'], args['m'], args['n'], args['k'], args['em'], args['data_type'], args['distribution'])
    print("Passed\n")
    return


def read_cmds_from_file(casefile):
    dg_cases = list()
    with open(casefile, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.endswith("list"):
                _case = read_cmds_from_file(line)
                dg_cases.extend(_case)
            else:
                _case = parse_deepgemm_string_re(line)
                dg_cases.append(_case)
    return dg_cases