import os
import csv
import subprocess
import re
import traceback

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

def worker(gpu_id, cases, output, device, dtype, mode, acc_check):
    # 设置当前进程可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process {os.getpid()} is running on GPU {gpu_id}")
    dtypes = str_to_list(dtype, str)
    for _d in dtypes:
        run_cycle_on_device(cases, output, device, _d, mode, acc_check, gpu_id)

def read_detail_from_nculog(filename):
    detail_info = {}
    keyword_pattern = r"(GemmGrouped-BF16|GemmGrouped-FP8|GemmGrouped-INT8|GemV-BF16|GemV-Small-BF16)"
    pattern_dict = {"group": r"group:(\d+)", "problem": r"problem:\[(\d+), (\d+), (\d+)\]", "expected_m":r"expected_m:(\d+)", 
    "ThreadblockShape": r"ThreadblockShape\[(\d+), (\d+), (\d+)\]" , "WarpShape": r"WarpShape\[(\d+), (\d+), (\d+)\]", "kNumStages":r"kNumStages:(\d+)",
    "num_sms": r"num_sms:(\d+)", "max_active_tb_num": r"max_active_tb_num:(\d+)", "threadblock_count": r"threadblock_count:(\d+)",
    "smem_size":r"smem_size:(\d+)", "vreg": r"vreg:(\d+)", "stack": r"stack:(\d+)"}
    with open(filename, newline='') as log_file:
        lines = log_file.readlines()
        for idx, line in enumerate(lines):
            print(idx)



# devices = {
#     "name": ["cycle", "tensor core efficiency", "waves"],
#     "gpu":  ["sm__cycles_active.max", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "launch__waves_per_multiprocessor"],
#     "ppu":  ["ce__cycles_active.max", "cu__inst_executed_pipe_tensor_fp16.avg.pct_of_peak_sustained_active", "launch__waves_per_cu"],
# }

def read_cycle_from_nculog(filename):
    # kernel_pattern = r"(.*)deep_gemm(.*)"
    # kernel_pattern = r"(.*)kernel(.*)Device(.*)"
    kernel_pattern = r"(.*)Device\s+\d+"
    cycles_pattern = "__cycles_active.max"
    global tc_pattern
    tc_pattern = "we_pipe_tensor_cycles_active"
    #tc_pattern = "pct_of_peak_sustained_active"
    hbm_pattern = "bytes_read"
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

def run_cycle_on_device(cases, output_file, dev="gpu", dtype="bf16", mode="metrics", acc_check=False, gpu_id="0"):
    output_lines = list()
    headers = ["casename","cycle","tc efficiency", "hbm efficiency", "dtype", "result", "cmd", "detail"]
    if not os.path.exists(f"{output_file}.csv"):
        with open(f"{output_file}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    # new_row=["casename"]  metrics.get("name", [])  ["detail"] 
    # output_lines.append(new_row)
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    total = len(cases)
    for idx, case in enumerate(cases):
        print(f'Profiling {idx + 1}/{total} on device{gpu_id}')
        print(f'case name:{case}')
        log_file = f"./logs/gpu{gpu_id}_{case.replace('/','_').replace('.','_')}_{dtype}.log"
        cmd = "rm -f "+ log_file
        run_cmd(cmd)
        # gpu
        # metrics = devices.get(dev, [])
        # metrics_string = ', '.join(metrics) if metrics else ""
        current_file_path = os.path.abspath(__file__)
        if dev == "gpu":
            script = "test_core_gpu.py"
        elif dtype == "fp8":
            script = "test_fp8_core.py"
        else:
            script = "test_core.py"
        script = f"{os.path.dirname(current_file_path)}/{script}"
        if mode == "full":
            output_name = clean_casename(case)
            cmd = '{} --set full -o {} python {}  --cycle --file {} --dtype {} \
                2>&1 | tee {}'.format("ncu" if dev == "gpu" else "acu", output_name, script, case, dtype, log_file)
        else:
            if mode == "show_log":
                os.environ["show_log"] = "1"
            metrics_string = "sm__cycles_active.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,dram__bytes.read.sum.pct_of_peak_sustained_elapsed" if dev=="gpu" else \
                            "ce__cycles_active.max,cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__llc_bytes_read.sum.pct_of_peak_sustained_elapsed"
            #                 "ce__cycles_active.max,cu__inst_executed_pipe_tensor_{}.avg.pct_of_peak_sustained_active,dram__llc_bytes_read.sum.pct_of_peak_sustained_elapsed".format(dtype)
                           
            
            cmd = '{} --clock-control none --metrics="{}"  \
                --page=details python {} --cycle --file {} --dtype {} \
                2>&1 | tee {}'.format("ncu" if dev == "gpu" else "acu", metrics_string, script, case, dtype, log_file)

        ret = run_cmd(cmd)
        result = "Fail"
        cycle, tc, detail, hbm = 0, 0, "", 0
        if mode != "full" and ret != None:
            if ret.returncode == 0:
                cycle, tc, detail, hbm = read_cycle_from_nculog(log_file)
                if mode == "show_log":
                    other_metrics = read_detail_from_nculog(log_file)
        if cycle != 0:
            result = "Pass"
        else:
            print("ERROR: failed to run cmd, please check!!")
            if len(cases) == 1:
                exit(-1) # only one case, fail and exit
        row =  [f"'{case.replace(',','_')}_{dtype}'", str(cycle), str(tc), str(hbm), dtype, result, str(cmd), str(detail)]
        output_lines.append(row)
        with open(f"{output_file}.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            print("write result succeed")

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
