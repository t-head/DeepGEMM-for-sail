import argparse
import os
import torch
from utils import run_cycle_on_device, str_to_list, worker, split_list_into_groups
from utils import set_acc_check, set_ref_backend
import multiprocessing as mp
import atexit
def device_sync_at_exit():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("Device synchronized on exit.")
atexit.register(device_sync_at_exit)
device_name = torch.cuda.get_device_name()
USE_PPU = (device_name.lower().find("ppu") != -1) or (device_name.lower().find("zw") != -1)
if not any(k in device_name.lower() for k in ['ppu','zw','nvidia']):
    print("Warning: Unrecognized device name: "+ device_name)

if USE_PPU:
    # os.environ['HGGC_PROFILE_MODE'] = '4'
    os.environ['HGGC_RESET_CACHE'] = '1'
    os.environ['ALIPPU_RESET_CE_MASK'] = '1'

def read_lines_from_file(casefile):
    dg_lines = list()
    with open(args.caselist, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.endswith("list"):
                _dg_lines = read_lines_from_file(line)
                dg_lines.extend(_dg_lines)
            else:
                dg_lines.append(line)
    return dg_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Performance Testing for DeepGemm with format or list.')
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the folder of DG cases')
    parser.add_argument('--case_idx', default=None, type=int, required=False, help='the line index(1~line) of case in caselist file')
    parser.add_argument('--format',  type=str, default=None, help="Case cmd to describe problem size.")
    parser.add_argument('--output', default="output", type=str, required=False, help='the output storing cycles of DG cases')
    parser.add_argument('--mode', default="metrics", type=str, choices=["metrics","full","show_log","umd_perf"], required=False, help='run perf mode')
    parser.add_argument('--device', default=None, type=str, required=False, help='devices index to run cases, 0 means gpu0. 0,3 means gpu0,1,2,3')
    parser.add_argument('--disable_acc', action="store_true", required=False, help='if or not open accuracy check')
    parser.add_argument('--ref_backend', default="device", type=str, required=False, choices=["host", "device"], help='specify the backend used to compute ref output')

    args = parser.parse_args()
    set_ref_backend(args.ref_backend)

    dg_cases = list()
    if args.format:
        dg_cases = [args.format]
    elif args.caselist:
        dg_cases = read_lines_from_file(os.path.abspath(args.caselist))
        if len(dg_cases) == 0:
            print("no dg_cases found")
            exit(-1)
        if args.case_idx:
            dg_cases = [dg_cases[args.case_idx-1]]
    else:
        print("Must give must give --caselist or --format")
        exit(-1)

    if args.disable_acc:
        set_acc_check(0)

    if args.device == None:
        run_cycle_on_device(dg_cases, args.output, "ppu" if USE_PPU else "gpu", args.mode)
    else:
        devices = str_to_list(args.device)
        if len(devices) == 1 or len(devices) > 2:
            num_gpus = devices
        elif len(devices) == 2:
            num_gpus = [i for i in range(devices[0], devices[1] + 1)]
        else:
            num_gpus = [0]
        processes = []
        cases_groups = split_list_into_groups(dg_cases, len(num_gpus))
        for i in range(len(num_gpus)):
            p = mp.Process(target=worker, args=(num_gpus[i], cases_groups[i], args.output, "ppu" if USE_PPU else "gpu", args.mode))
            p.start()
            processes.append(p)
