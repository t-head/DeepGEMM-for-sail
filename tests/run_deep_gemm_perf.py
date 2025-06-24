import argparse
import os
import torch
from utils import run_cycle_on_device

device_name = torch.cuda.get_device_name()
USE_PPU = (device_name.lower().find("ppu") != -1)
if not any(k in device_name.lower() for k in ['ppu','nvidia']):
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
    parser.add_argument('--force_int8', action="store_true", help="force use int8 data type")

    args = parser.parse_args()
    dg_cases = list()
    if args.file:
        dg_cases = [args.file]
    elif args.caselist:
        for root, dirs, files in os.walk(args.caselist):
            for file in files:
                full_path = os.path.join(root, file)
                dg_cases.append(full_path)
    else:
        print("Must give a string a format or a caselist file!")
        exit(-1)

    # print(dg_cases)
    run_cycle_on_device(dg_cases, args.output, "ppu" if USE_PPU else "gpu", True if args.force_int8 else False)
