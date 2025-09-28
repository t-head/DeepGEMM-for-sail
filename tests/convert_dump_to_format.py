import os
import re
from utils import parse_deepgemm_string_re, parse_dump_file, read_numbers_from_file

def get_data_type(s):
    supported_data_type = ["bf16", "int8", "fp8"]
    for item in supported_data_type:
        if item in s:
            return item
    return None

def get_gemm_type(s):
    supported_gemm_type = ["GroupedContiguous", "GroupedNoPad", "GroupedMasked", "DenseGemm"]
    for item in supported_gemm_type:
        if item in s:
            return item
    return None

def convert_cmd_to_caselist(casefile):
    dg_cases = list()
    lines = list()
    if not os.path.isdir(casefile):
        print("args.caselist is a file!")
        with open(casefile, "r") as f:
            lines = f.readlines()
    else:
        print("args.caselist is a folder!")
        for root, dirs, files in os.walk(args.caselist):
            for file in files:
                full_path = os.path.join(root, file)
                lines.append(full_path)

    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        if line.endswith("list"):
            _temp_dg_cases = convert_cmd_to_caselist(os.path.abspath(line))
            dg_cases.extend(_temp_dg_cases)
        if line.endswith("dump"):
            num_groups, m, n, k, expected_m_per_group = parse_dump_file(line)
            gemm_type = get_gemm_type(line)
            data_type = get_data_type(line)
            group_m_list = read_numbers_from_file(os.path.abspath(line))
            group_m_list = ",".join([str(x) for x in group_m_list])
            if gemm_type == "DenseGemm":
                _case_format = f"[DeepGemm] --format={gemm_type},data_type:{data_type},groups:{num_groups},m:{m},n:{n},k:{k},em:{expected_m_per_group},distribution:normal"
            else:
                _case_format = f"[DeepGemm] --format={gemm_type},data_type:{data_type},groups:{num_groups},m:{m},n:{n},k:{k},em:{expected_m_per_group},distribution:[{group_m_list}]"
            dg_cases.append(_case_format)
    return dg_cases


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the folder or caselist of DG dump cases')
    parser.add_argument('--output', default="converted.caselist", type=str, required=False, help='the output caselist filename')
    args = parser.parse_args()
    dg_cases = convert_cmd_to_caselist(os.path.abspath(args.caselist))
    with open(args.output, "w") as f:
        f.writelines("\n".join(dg_cases))