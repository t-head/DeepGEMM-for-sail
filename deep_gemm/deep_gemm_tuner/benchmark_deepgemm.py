#!/usr/bin/env python3
import argparse
import os, json, re
from sglang.srt.utils import logger
from collections import defaultdict
from deep_gemm import deep_gemm_tuner as tuner
from .autotune_deepgemm import dispatch_tune_method
from .utils import get_deep_gemm_best_configs, search_for_suitable_config

try:
    from prettytable import PrettyTable
except ImportError as e:
    logger.warning(
        f"{e}, please run 'pip install prettytable'."
    )
    from .deepgemm_tools import PrettyTable

try:
    from termcolor import colored
    def colorize_score_termcolor(value):
        if value >= 1.0:
            return colored(f"{value:.2f}", "green")
        else:
            return colored(f"{value:.2f}", "red")
except ImportError as e:
    logger.warning(
        f"{e}, please run 'pip install prettytable'."
    )
    def colorize_score_termcolor(value):
        return f"{value:.2f}"

colorize_score = colorize_score_termcolor


def get_tuned_configs(config_file_path: str):
    config_dict = dict()

    try:
        with open(config_file_path,'r') as f:
            config_dict = json.load(f)
        config_dict = dict(
            [((x["M"], x["N"], x["K"], x["num_groups"],
                x['nopad'] if 'nopad' in x else False,
                x['dtype'] if 'dtype' in x else "int8"), x) for x in config_dict]
        )
    except :
        logger.warning(
            f"Empty config found in {config_file_path}."
        )

    def gen_NKG_M_map(best_config):
        nkg_m_map = defaultdict(list)
        for key in best_config:
            m, n, k, g, nopad, dtype = key
            nkg_m_map[(n, k, g, nopad, dtype)].append(m)
        return nkg_m_map
    return config_dict, gen_NKG_M_map(config_dict)

def benckmark_normal_atom(M, N, K, num_groups, nopad, dtype, config, baseline_config=None):
    num_sms = config["num_min_sms"]
    block_m = config["best_block_m"]
    block_n = config["best_block_n"]
    block_k = config["block_k"]
    warp_m = config["warp_m"]
    warp_n = config["warp_n"]
    num_stages = config["best_num_stages"]
    smem_config = config["best_smem_config"]
    benchmark_func = dispatch_tune_method(num_groups, nopad, dtype)
    if baseline_config:
        b_num_sms = baseline_config["num_min_sms"]
        b_block_m = baseline_config["best_block_m"]
        b_block_n = baseline_config["best_block_n"]
        b_block_k = baseline_config["block_k"]
        b_warp_m = baseline_config["warp_m"]
        b_warp_n = baseline_config["warp_n"]
        b_num_stages = baseline_config["best_num_stages"]
        b_smem_config = baseline_config["best_smem_config"]
        baseline_time = benchmark_func(M, K, N, config=(b_num_sms, b_block_m, b_block_n, b_block_k, b_warp_m, b_warp_n, b_num_stages, b_smem_config), num_groups=num_groups)
    else:
        baseline_time = benchmark_func(M, K, N, config=None, num_groups=num_groups)

    config_time = benchmark_func(M, K, N, config=(num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config), num_groups=num_groups)
    return baseline_time, config_time

def benchmark_tuned_configs(config_file_path=None):
    table = PrettyTable()
    table.float_format = "0.2f"
    table.field_names = ["(M(expect M), N, K, num_groups)", "default time", "optimized time", "speedup"]
    if config_file_path:
        best_configs, _ = get_tuned_configs(config_file_path)
    else:
        best_configs, _ = get_deep_gemm_best_configs()
    for k, config in best_configs.items():
        M, N, K, num_group, nopad, dtype = k
        if config_file_path is None and num_group > 1:
            continue
        baseline_time, config_time = benckmark_normal_atom(M, N, K, config=eval(config['config']), num_groups=num_group, nopad=nopad, dtype=dtype)
        table.add_row([f"({M}, {N}, {K}, {num_group}, {nopad}, {dtype})", f"{baseline_time:.2f}", f"{config_time:.2f}", colorize_score(baseline_time/config_time)])

    print(table)

def benchmark_attach_case(config_file_path=None, concurrency_list=[1], MTP=1, topk=8, experts_num=256):
    if config_file_path:
        best_configs, nkg_m_map = get_tuned_configs(config_file_path)
    else:
        best_configs, nkg_m_map = get_deep_gemm_best_configs()
    table = PrettyTable()
    table.float_format = "0.2f"
    table.field_names = ["(M(expect M), N, K, num_groups)", "M attached", "default time", "optimized time", "speedup"]

    for nkg in nkg_m_map:
        for concurrency in concurrency_list:
            concurrency=int(concurrency)
            N, K, num_groups, nopad, dtype = nkg
            if num_groups > 1:
                ## masked expect_m estimate
                M = concurrency*MTP*topk//experts_num+1
                # continue
            else:
                M = concurrency
            config = None
            M_candidate = M
            if best_configs is not None:
                # find neariest config
                config, M_candidate = search_for_suitable_config(M, N, K, num_groups, nopad, dtype, best_configs, nkg_m_map)

            if config:
                baseline_time, config_time = benckmark_normal_atom(M, N, K, config=config, num_groups=num_groups, nopad=nopad, dtype=dtype)
                table.add_row([f"({M}, {N}, {K}, {num_groups}, {nopad}, {dtype})", M_candidate, f"{baseline_time:.2f}", f"{config_time:.2f}", colorize_score(baseline_time/config_time)])
            else:
                table.add_row([f"({M}, {N}, {K}, {num_groups}, {nopad}, {dtype})", M_candidate, -1, -1, 1])

    print(table)

def benchmark_increment(config_file_path):
    best_configs, nkg_m_map = get_tuned_configs(config_file_path)
    baseline_configs, base_nkg_m_map = get_deep_gemm_best_configs()

    table = PrettyTable()
    table.float_format = "0.2f"
    table.field_names = ["(M(expect M), N, K, num_groups)", "M baseline attached", "baseline time", "M attached", "optimized time", "speedup"]

    benchmark_result = []
    for nkg in nkg_m_map:
        N, K, num_groups, nopad, dtype = nkg
        for M in nkg_m_map[nkg]:
            config = None
            M_candidate = M
            if best_configs is not None:
                # find neariest config
                config, M_candidate = search_for_suitable_config(M, N, K,  num_groups, nopad, dtype, best_configs, nkg_m_map)

            baseline_config = None
            M_baseline_candidate = M
            if baseline_configs is not None:
                # find neariest config
                baseline_config, M_baseline_candidate = search_for_suitable_config(M, N, K, num_groups, nopad, dtype, baseline_configs, base_nkg_m_map)

            if config:
                baseline_time, config_time = benckmark_normal_atom(M, N, K, config=config, num_groups=num_groups, baseline_config=baseline_config, nopad=nopad, dtype=dtype)
                table.add_row([f"({M}, {N}, {K}, {num_groups}, {nopad}, {dtype})", M_baseline_candidate, f"{baseline_time:.2f}", M_candidate, f"{config_time:.2f}", colorize_score(baseline_time/config_time)])
                benchmark_result.append((M, N, K, num_groups, baseline_time/config_time))
            else:
                table.add_row([f"({M}, {N}, {K}, {num_groups}, {nopad}, {dtype})", M_baseline_candidate, -1,  M_candidate, -1, 1])

    print(table)
    return benchmark_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    parser_checkin = subparsers.add_parser('checkin', help='Run comparison benchmark')
    parser_checkin.add_argument(
        "--tuned-config", type=str, default=None
    )

    parser_benchmark = subparsers.add_parser('benchmark', help='Run benchmark vs default')
    parser_benchmark.add_argument(
        "--tuned-config", type=str, default=None
    )

    parser_checkin = subparsers.add_parser('all', help='Run comparison benchmark')

    parser_attach = subparsers.add_parser('attach', help='Run benchmark of a specific case')
    parser_attach.add_argument(
        "--tuned-config", type=str, default=None
    )
    parser_attach.add_argument(
        "--concurrency", nargs='+', required=True, help='list of test concurrency'
    )
    parser_attach.add_argument(
        "--MTP", type=int, default=1
    )
    parser_attach.add_argument(
        "--topk", type=int, default=8
    )
    parser_attach.add_argument(
        "--experts-num", type=int, default=256
    )
    args = parser.parse_args()

    if args.command == 'checkin':
        benchmark_increment(args.tuned_config)
    elif args.command == 'attach':
        # for concurracy in (8, 25, 48, 117, 156, 207, 256):
        #   benchmark_attach_case(args.tuned_config, concurracy)
        benchmark_attach_case(args.tuned_config, args.concurrency, args.MTP, args.topk, args.experts_num)
    elif args.command == 'benchmark':
        benchmark_tuned_configs(args.tuned_config)
    elif args.command == 'all':
        logger.info("Full test on dense gemm")
        benchmark_tuned_configs()
