import sys
import re
import pandas as pd
import os
import argparse
from utils import run_cmd, parse_deepgemm_string_re

def parse(filename, outfile):
    tile_data = []
    tile_cycle = []
    tile_hbm = []
    tile_tc = []
    # ThreadblockShape[16, 64, 128], WarpShape[16, 16, 128], kNumStages:2
    pattern_tile = r'ThreadblockShape\[(\d+),\s*(\d+),\s*(\d+)\].*?WarpShape\[(\d+),\s*(\d+),\s*(\d+)\],\s*kNumStages:(\d+)'
    pattern_cycle = r"ce__cycles_active.*cycle\s+(\d+)$"
    pattern_hbm = r"ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed.*?(\d+\.\d+)$"
    pattern_tc = r"cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed.*?(\d+\.\d+)$"
    with open(filename, "r") as f:
        for line in f:
            out = re.findall(pattern_tile, line.strip())
            if out:
                tile_data.append("x".join(out[0]))
                continue
            cycle_out = re.findall(pattern_cycle, line.strip())
            if cycle_out:
                tile_cycle.append(int(cycle_out[0]))
                continue
            hbm_out = re.findall(pattern_hbm, line.strip())
            if hbm_out:
                tile_hbm.append(float(hbm_out[0]))
                continue
            tc_out = re.findall(pattern_tc, line.strip())
            if tc_out:
                tile_tc.append(float(tc_out[0]))
                continue
    try:
        df = pd.DataFrame({
            "tile" : tile_data,
            "cycle" : tile_cycle,
            "hbm" : tile_hbm,
            "tc" : tile_tc
        })
    except Exception as e:
        print(f"Unexpected error for parsing {filename}: {e}")
        return None
#    print(df)
    df_sorted = df.sort_values(by='cycle', ascending=True).reset_index(drop=True)
#    print(df_sorted)
    df_sorted.to_csv("%s.csv" % outfile, index=False)

    best_tile = df_sorted.loc[0, 'tile']
    cta_m, cta_n, cta_k, warp_m, warp_n, warp_k, stage = [int(v) for v in best_tile.split('x')]
    return cta_m, cta_n, cta_k, warp_m, warp_n, stage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--format',  type=str, default=None, help="Case cmd to describe problem size.")
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the folder of DG cases')
    args = parser.parse_args()
    dg_cases = []
    if args.format:
        dg_cases.append(args.format)
    elif args.caselist:
        if not os.path.isfile(args.caselist):
            print("[Warning] caselist shoulde be a file. {} is not exist.", args.caselist)
        with open(args.caselist, "r") as f:
            for line in f:
                case = line.strip()[line.find("--format=") + len("--format="):]
                dg_cases.append(case)
    total = len(dg_cases)
    if len(dg_cases) == 0:
        print("[Warning] Please provide problem cases for tile scan, use --format or --caselist.")

    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    lut_file = "./logs/lut_case"
    with open(lut_file, "w") as f:
        pass
    for idx, case in enumerate(dg_cases):
        print(f'Profiling {idx + 1}/{total}. case:', case)
        one_case = parse_deepgemm_string_re(case)
        m, n, k = int(one_case['m']), int(one_case['n']), int(one_case['k'])
        group = one_case['groups'] if 'groups' in one_case else 1
        fname = "m{}_n{}_k{}_group{}".format(m, n, k, group)
        print(f'case name:{one_case}')
        log_file = "./logs/tile_scan.log.{}".format(fname)
        cmd = "rm -f " + log_file
        run_cmd(cmd)
        # gpu
        # metrics = devices.get(dev, [])
        # metrics_string = ', '.join(metrics) if metrics else ""
        current_file_path = os.path.abspath(__file__)
        script = f"{os.path.dirname(current_file_path)}/tile_scan_with_test.py"
        os.environ["show_log"] = "1"
        metrics_string = "ce__cycles_active.max,cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
        cmd = '{} --clock-control none {} --metrics="{}"  --page=details python {} {} --cycle \
                2>&1 | tee {}'.format("acu", '--kernel-name "regex:Kernel|device_kernel"', metrics_string, script, "--format=" + case, log_file)
        print(cmd)
        ret = run_cmd(cmd)
        if ret != None and ret.returncode == 0:
            best_config = parse(log_file, "./logs/{}".format(fname))
            if best_config != None:
                #(  4,  768,  4096, ( 32,  64, 128, 16, 32,  3))
                with open(lut_file, "a") as f:
                    print("({:5}, {:5}, {:5}, ({:3}, {:3}, {:3}, {:2}, {:2}, {})),"
                            .format(m, n, k, best_config[0], best_config[1], best_config[2], best_config[3], best_config[4], best_config[5]), file=f)

