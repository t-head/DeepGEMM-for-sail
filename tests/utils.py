import os
import csv
import subprocess
import re

def run_cmd(cmd: str, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    print(f"Run command: {cmd}, timeout: {timeout}")
    ret = subprocess.run(args=cmd, timeout=timeout, shell=True, stdout=stdout, stderr=stderr, encoding="utf-8")
    if stdout:
        for line in ret.stdout.splitlines() + ret.stderr.splitlines():
            print(line)
    if ret.returncode != 0:
        print(f"Run command failed!")
    else:
        print(f"Run command succeed!")
    return ret


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
    tc_pattern = "pct_of_peak_sustained_active"
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

    assert(len(kernel_list) == len(cycles_list))
    assert(len(kernel_list) == len(tc_list))
    assert(len(kernel_list) == len(hbm_list))

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
        return 0, 0, [],[]
        #exit(-1)


def run_cycle_on_device(cases, output_file, dev="gpu", force_int8=False):
    output_lines = list()
    headers = ["casename","cycle","tc efficiency", "hbm efficiency", "cmd","detail"]
    # new_row=["casename"]  metrics.get("name", [])  ["detail"] 
    # output_lines.append(new_row)

    for case in cases:
        log_file = "./gpu_cycles_single_case.log"
        cmd = "rm -f "+ log_file
        run_cmd(cmd)
        # gpu
        # metrics = devices.get(dev, [])
        # metrics_string = ', '.join(metrics) if metrics else ""

        metrics_string = "sm__cycles_active.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,dram__bytes.read.sum.pct_of_peak_sustained_elapsed" if dev=="gpu" else \
                         "ce__cycles_active.max,cu__inst_executed_pipe_tensor_{}.avg.pct_of_peak_sustained_active,dram__llc_bytes_read.sum.pct_of_peak_sustained_elapsed".format("int8" if ("int8" in os.path.basename(case) or force_int8) else "bf16")
        cmd = '{} --clock-control none --metrics="{}"  \
              --page=details python ./{} --file {} --cycle {}\
              2>&1 | tee -a {}'.format("ncu" if dev == "gpu" else "acu", metrics_string, "test_core.py" if dev == "ppu" else "test_core_gpu.py", case, "  --force_int8" if force_int8 else "", log_file)

        run_cmd(cmd)

        cycle, tc, detail, hbm = read_cycle_from_nculog(log_file)
        output_lines.append([case.replace(",","_"), str(cycle), str(tc), str(hbm), str(cmd), str(detail)])

    output_file = output_file + '.csv'
    if len(cases) == 1:
        with open("local.log", "w") as f:
            writer = csv.writer(f)
            for row in output_lines:
                writer.writerow(row)
            print("write result to local.log succeed")
    
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    with open(output_file, "a+") as f:
        writer = csv.writer(f)
        for row in output_lines:
            writer.writerow(row)
        print("write result succeed")

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
