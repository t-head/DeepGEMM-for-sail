import argparse
import os
import torch
from utils import run_cycle_on_device, str_to_list, worker, split_list_into_groups
import multiprocessing as mp

device_name = torch.cuda.get_device_name()
USE_PPU = (device_name.lower().find("ppu") != -1) or (device_name.lower().find("zw") != -1)
if not any(k in device_name.lower() for k in ['ppu','zw','nvidia']):
    print("Warning: Unrecognized device name: "+ device_name)

if USE_PPU:
    # os.environ['HGGC_PROFILE_MODE'] = '4'
    os.environ['HGGC_RESET_CACHE'] = '1'
    os.environ['ALIPPU_RESET_CE_MASK'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Performance Testing for DeepGemm with format or list.')
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the folder of DG cases')
    parser.add_argument('--file', default=None, type=str, required=False, help='the string of DG cases')
    parser.add_argument('--output', default="output", type=str, required=False, help='the output storing cycles of DG cases')
    parser.add_argument('--dtype', default="bf16", type=str, choices=["int8","fp8","bf16","int8,fp8","int8,bf16","fp8,bf16","int8,fp8,bf16"], required=False, help='data type of the cases')
    parser.add_argument('--mode', default="metrics", type=str, choices=["metrics","full"], required=False, help='acu mode')
    parser.add_argument('--device', default=None, type=str, required=False, help='devices index to run cases, 0 means gpu0. 0,3 means gpu0,1,2,3')

    args = parser.parse_args()
    dg_cases = list()
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
                    dg_cases.append(line.strip())
        else:
            print("args.caselist is a folder!")
            for root, dirs, files in os.walk(args.caselist):
                for file in files:
                    full_path = os.path.join(root, file)
                    dg_cases.append(full_path)
    else:
        print("Must give a file path or a caselist directory!")
        exit(-1)
    if args.device == None:
        # print(dg_cases)
        dtypes = str_to_list(args.dtype, str)
        for _d in dtypes:
            run_cycle_on_device(dg_cases, args.output, "ppu" if USE_PPU else "gpu", _d, args.mode)
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
            # 创建子进程并传递 GPU ID, 在worker中循环 backend的取值
            p = mp.Process(target=worker, args=(num_gpus[i], cases_groups[i], args.output, "ppu" if USE_PPU else "gpu", args.dtype, args.mode))
            p.start()
            processes.append(p)
