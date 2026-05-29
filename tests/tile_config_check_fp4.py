import torch
import multiprocessing as mp
import json
from typing import List, Dict, Tuple
import deep_gemm
import argparse
import copy
from test_fp4_core import quantize_fp4_torch, dequantize_fp4_torch, construct_grouped
from deep_gemm.jit_kernels.utils import get_num_sms
from deep_gemm import preprocess_mxfp4_scales, calc_diff
from utils import construct_group_m_list, split_list_into_groups
from deep_gemm.jit_kernels.gemm_fp4 import get_smem_config_fp4
from deep_gemm import ceil_div
from deep_gemm.jit_kernels.utils import get_search_space

# This tile config check file only support MXFP4
def test_kernel_config(configs: Tuple, m, n, k, num_groups, gemm_type) -> Tuple[bool, str]:
    if 'dense' in gemm_type.lower():
        try:
            A = torch.randn(m, k, dtype=torch.bfloat16, device='cuda').contiguous()
            B = torch.randn(n, k, dtype=torch.bfloat16, device='cuda').contiguous()
            x = quantize_fp4_torch(A)
            y = quantize_fp4_torch(B)
            a_dequant = dequantize_fp4_torch(x[0], x[1]).cuda().float()
            b_dequant = dequantize_fp4_torch(y[0], y[1]).cuda().float()
            bias = torch.randn(1, n, dtype=torch.float32, device='cuda')
            out = torch.zeros(m, n, dtype=torch.bfloat16, device='cuda')
            ref_out = torch.mm(a_dequant, b_dequant.T)
            ref_out = ref_out + bias
            x_scale = preprocess_mxfp4_scales(scale=x[1])
            y_scale = preprocess_mxfp4_scales(scale=y[1])
            x = x[0], x_scale
            y = y[0], y_scale

            deep_gemm.gemm_fp4_fp4_bf16_nt(x, y, bias, out, configs)
            torch.cuda.synchronize()

            diff = calc_diff(out, ref_out.to('cuda').to(torch.bfloat16))
            if diff < 0.001:
                return True, "Success"
            else:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.allclose(out, ref_out, rtol=1e-3, atol=1e-4)
                return False, "Gemm compute result is wrong!!!"

        except RuntimeError as e:
            error_msg = str(e)
            return False, error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, error_msg
    elif 'nopad' in gemm_type.lower():
        try:
            x, y, m_indices, bias, out, ref_out = construct_grouped(num_groups, m, k, n, 'uniform', 1)

            deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(x, y, bias, out, m_indices, configs=configs)
            torch.cuda.synchronize()
            diff = calc_diff(out, ref_out.to('cuda').to(torch.bfloat16))
            if diff < 0.001:
                return True, "Success"
            else:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.allclose(out, ref_out, rtol=1e-3, atol=1e-4)
                return False, "Gemm compute result is wrong!!!"

        except RuntimeError as e:
            error_msg = str(e)
            return False, error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, error_msg

def test_all_configs(config_list: List[Tuple],
                     gemm_type: str,
                     output_file: str = "kernel_config_results.txt",
                     json_file: str = "kernel_config_results.json",
                     m: int = 1024,
                     n: int = 1024,
                     k: int = 1024,
                     num_groups: int = 1,
                     success_rate_threshold = 0.95) -> Dict:
    valid_configs = []
    invalid_configs = []
    all_results_and_configs = {}

    print(f"Start testing {len(config_list)} configs...")
    print(f"Test matrix dimensions: M={m}, N={n}, K={k}\n")
    
    enable_multithread = not bool(cycle)
    if enable_multithread:
        mp.set_start_method('spawn', force=True)
        global thread_count
        thread_count = min(thread_count, len(config_list))
        with mp.Pool(processes=thread_count) as pool:
            tasks = []
            for idx in range(len(config_list)):
                num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = config_list[idx]
                task_args = (config_list[idx], m, n, k, num_groups, gemm_type)
                tasks.append(task_args)
                config_dict = {
                    'num_sms': num_sms,
                    'block_m': block_m,
                    'block_n': block_n,
                    'block_k': block_k,
                    'warp_m': warp_m,
                    'warp_n': warp_n,
                    'num_stages': num_stages,
                    'error': None,
                    'result': None
                }
                all_results_and_configs[idx] = config_dict

            results = pool.starmap(test_kernel_config, tasks)
        # print(results)
        for idx in range(len(results)):
            is_valid, error_msg = results[idx]
            all_results_and_configs[idx]['error'] = error_msg
            all_results_and_configs[idx]['result'] = is_valid

    else:
        for i, configs in enumerate(config_list):
            num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs

            print(f"[{i+1}/{len(config_list)}] test config:")
            print(f"  num_sms={num_sms}, block_m={block_m}, block_n={block_n}, block_k={block_k}")
            print(f"  warp_m={warp_m}, warp_n={warp_n}, num_stages={num_stages}")
            is_valid, error_msg = test_kernel_config(configs, m, n, k, num_groups, gemm_type)

            config_dict = {
                'num_sms': num_sms,
                'block_m': block_m,
                'block_n': block_n,
                'block_k': block_k,
                'warp_m': warp_m,
                'warp_n': warp_n,
                'num_stages': num_stages,
                'error': error_msg,
                'result': is_valid
            }
            all_results_and_configs[i] = config_dict

    for i, config in all_results_and_configs.items():
        is_valid, error_msg = config["result"], config["error"]
        if is_valid:
            print(f"[{i+1}/{len(all_results_and_configs)}] Test success!{config}\n")
            valid_configs.append(config_dict)
        else:
            print(f"[{i+1}/{len(all_results_and_configs)}] Test failure: {config}, {error_msg}\n")
            invalid_configs.append(config_dict)

    results = {
        'test_params': {'m': m, 'n': n, 'k': k},
        'total': len(config_list),
        'valid': len(valid_configs),
        'invalid': len(invalid_configs),
        'valid_configs': valid_configs,
        'invalid_configs': invalid_configs
    }
    success_rate = round(results['valid']/results['total'], 4)
    assert success_rate >= success_rate_threshold, f'expect {success_rate_threshold=}, actual {success_rate=}'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"{gemm_type} Kernel Configuration Test Results\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Test Matrix Size: M={m}, N={n}, K={k}\n")
        f.write(f"Total Configs: {results['total']}\n")
        f.write(f"Valid Configs: {results['valid']}\n")
        f.write(f"Invalid Configs: {results['invalid']}\n")
        f.write(f"Success Rate: {success_rate*100:.2f}%\n\n")

        f.write("=" * 100 + "\n")
        f.write("VALID CONFIGURATIONS\n")
        f.write("=" * 100 + "\n\n")
        for i, config in enumerate(valid_configs, 1):
            f.write(f"{i}. configs = ({config['num_sms']}, {config['block_m']}, {config['block_n']}, "
                   f"{config['block_k']}, {config['warp_m']}, {config['warp_n']}, {config['num_stages']})\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("INVALID CONFIGURATIONS\n")
        f.write("=" * 100 + "\n\n")
        for i, config in enumerate(invalid_configs, 1):
            f.write(f"{i}. configs = ({config['num_sms']}, {config['block_m']}, {config['block_n']}, "
                   f"{config['block_k']}, {config['warp_m']}, {config['warp_n']}, {config['num_stages']})\n")
            f.write(f"   Error: {config['error']}\n\n")

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"{'='*100}")
    print(f"{gemm_type} test finished!")
    print(f"Total configurations: {results['total']}")
    print(f"Valid configurations: {results['valid']}")
    print(f"Invalid configurations: {results['invalid']}")
    print(f"Success rate: {results['valid']/results['total']*100:.2f}%")
    print(f"\nResults saved to:")
    print(f"  - {output_file}")
    print(f"  - {json_file}")
    print(f"{'='*100}")

    return results

def get_search_space_doublecheck(d: torch.dtype, gemm_type : str, m:int=0, n:int=0, k:int=0) -> list:
    """
    Returns search space according input gemm type

    Arguments:
        gemm_type: nopad, dense

    Returns:
        The tile list:{block_m, block_n, warp_m, warp_n, stage}
    """
    assert gemm_type in ('nopad', 'dense')

    block_k = 64 if d == torch.bfloat16 else 128
    tile_list = [
        # blockM = 16
        [16, 64, 16, 16, block_k, 2],
        [16, 64, 16, 16, block_k, 3],
        [16, 64, 16, 16, block_k, 4],
        [16, 64, 16, 16, block_k, 5],
        [16, 64, 16, 16, block_k * 2, 2],
        [16, 64, 16, 16, block_k * 2, 3],
        [16, 64, 16, 16, block_k * 2, 4],
        [16, 64, 16, 16, block_k * 2, 5],

        [16, 128, 16, 32, block_k, 2],
        [16, 128, 16, 32, block_k, 3],
        [16, 128, 16, 32, block_k, 4],
        [16, 128, 16, 32, block_k * 2, 2],
        [16, 128, 16, 32, block_k * 2, 3],
        [16, 128, 16, 32, block_k * 2, 4],
        [16, 128, 16, 16, block_k, 2],
        [16, 128, 16, 16, block_k, 3],
        [16, 128, 16, 16, block_k, 4],
        [16, 128, 16, 16, block_k * 2, 2],
        [16, 128, 16, 16, block_k * 2, 3],
        [16, 128, 16, 16, block_k * 2, 4],

        [16, 256, 16, 64, block_k, 2],
        [16, 256, 16, 64, block_k, 3],
        [16, 256, 16, 64, block_k * 2, 2],
        [16, 256, 16, 64, block_k * 2, 3],

        # blockM = 32
        [32, 64, 16, 32, block_k, 2],
        [32, 64, 16, 32, block_k, 3],
        [32, 64, 16, 32, block_k, 4],
        [32, 64, 16, 32, block_k, 5],
        [32, 64, 16, 32, block_k * 2, 2],
        [32, 64, 16, 32, block_k * 2, 3],
        [32, 64, 16, 32, block_k * 2, 4],
        [32, 64, 16, 16, block_k, 2],
        [32, 64, 16, 16, block_k, 3],
        [32, 64, 16, 16, block_k * 2, 2],
        [32, 64, 16, 16, block_k * 2, 3],

        [32, 128, 16, 64, block_k, 2],
        [32, 128, 16, 64, block_k, 3],
        [32, 128, 16, 64, block_k, 4],
        [32, 128, 16, 64, block_k * 2, 2],
        [32, 128, 16, 64, block_k * 2, 3],
        [32, 128, 16, 32, block_k, 2],
        [32, 128, 16, 32, block_k, 3],
        [32, 128, 16, 32, block_k * 2, 2],
        [32, 128, 16, 32, block_k * 2, 3],

        [32, 256, 16, 64, block_k, 2],
        [32, 256, 16, 64, block_k, 3],
        [32, 256, 16, 64, block_k * 2, 2],
        [32, 256, 16, 64, block_k * 2, 3],

        # blockM = 64
        [64, 64, 16, 16, block_k, 2],
        [64, 64, 16, 16, block_k, 3],
        [64, 64, 16, 16, block_k, 4],
        [64, 64, 16, 16, block_k * 2, 2],
        [64, 64, 16, 16, block_k * 2, 3],
        [64, 64, 16, 16, block_k * 2, 4],
        [64, 64, 16, 16, int(block_k / 2), 2],
        [64, 64, 16, 16, int(block_k / 2), 3],
        [64, 64, 16, 16, int(block_k / 2), 4],

        [64, 64, 32, 32, block_k, 2],
        [64, 64, 32, 32, block_k, 3],
        [64, 64, 32, 32, block_k, 4],
        [64, 64, 32, 32, block_k * 2, 2],
        [64, 64, 32, 32, block_k * 2, 3],
        [64, 64, 32, 32, block_k * 2, 4],
        [64, 64, 32, 32, int(block_k / 2), 2],
        [64, 64, 32, 32, int(block_k / 2), 3],
        [64, 64, 32, 32, int(block_k / 2), 4],

        [64, 128, 32, 64, block_k, 2],
        [64, 128, 32, 64, block_k, 3],
        [64, 128, 32, 32, block_k, 2],
        [64, 128, 32, 32, block_k, 3],
        [64, 128, 32, 64, block_k * 2, 2],
        [64, 128, 32, 64, block_k * 2, 3],
        [64, 128, 32, 32, block_k * 2, 2],
        [64, 128, 32, 32, block_k * 2, 3],
        [64, 128, 32, 64, int(block_k / 2), 2],
        [64, 128, 32, 64, int(block_k / 2), 3],
        [64, 128, 32, 32, int(block_k / 2), 2],
        [64, 128, 32, 32, int(block_k / 2), 3],

        [64, 256, 32, 64, block_k, 2],
        [64, 256, 32, 64, block_k, 3],
        [64, 256, 32, 64, block_k * 2, 2],
        [64, 256, 32, 64, block_k * 2, 3],
        [64, 256, 32, 64, int(block_k / 2), 2],
        [64, 256, 32, 64, int(block_k / 2), 3],
        [64, 256, 32, 64, int(block_k / 2), 2],
        [64, 256, 32, 64, int(block_k / 2), 4],

        # blockM = 128
        [128, 128, 64, 64, block_k    , 2],
        [128, 128, 64, 64, block_k    , 3],
        [128, 128, 64, 64, block_k    , 4],
        [128, 128, 64, 64, block_k * 2, 2],
        [128, 128, 64, 64, block_k * 2, 3],
        [128, 128, 64, 64, block_k * 2, 4],
        [128, 128, 64, 64, int(block_k / 2), 2],
        [128, 128, 64, 64, int(block_k / 2), 3],
        [128, 128, 64, 64, int(block_k / 2), 4],
        [128, 256, 64, 64, block_k    , 2],
        [128, 256, 64, 64, block_k    , 3],
        [128, 256, 64, 64, block_k    , 4],
        [128, 256, 64, 64, block_k * 2, 2],
        [128, 256, 64, 64, block_k * 2, 3],
        [128, 256, 64, 64, block_k * 2, 4],
        [128, 256, 64, 64, int(block_k / 2), 2],
        [128, 256, 64, 64, int(block_k / 2), 3],
        [128, 256, 64, 64, int(block_k / 2), 4],
        # [128, 320, 64, 80, block_k    , 2],
        # [128, 320, 64, 80, block_k    , 3],
        # [128, 320, 64, 80, block_k    , 4],
        # [128, 320, 64, 80, block_k * 2, 2],
        # [128, 320, 64, 80, block_k * 2, 3],
        # [128, 320, 64, 80, block_k * 2, 4],
        # [128, 320, 64, 80, int(block_k / 2), 2],
        # [128, 320, 64, 80, int(block_k / 2), 3],
        # [128, 320, 64, 80, int(block_k / 2), 4],

        # blockM = 256
        [256, 64, 32, 64, block_k,      2],
        [256, 64, 32, 64, block_k,      3],
        [256, 64, 32, 64, block_k * 2,  2],
        [256, 64, 32, 64, block_k * 2,  3],
        [256, 128, 64, 64, block_k    , 2],
        [256, 128, 64, 64, block_k    , 3],
        [256, 128, 64, 64, block_k    , 4],
        [256, 128, 64, 64, block_k * 2, 2],
        [256, 128, 64, 64, block_k * 2, 3],
        [256, 128, 64, 64, block_k * 2, 4],
        [256, 128, 64, 64, int(block_k / 2), 2],
        [256, 128, 64, 64, int(block_k / 2), 3],
        [256, 128, 64, 64, int(block_k / 2), 4],
        [256, 256, 64, 64, block_k,     2],
        [256, 256, 64, 64, block_k,     3],
        [256, 256, 64, 64, block_k,     4],
        [256, 256, 64, 64, block_k * 2, 2],
        [256, 256, 64, 64, block_k * 2, 3],
        [256, 256, 64, 64, block_k * 2, 4],
        [256, 256, 64, 64, int(block_k / 2), 2],
        [256, 256, 64, 64, int(block_k / 2), 3],
        [256, 256, 64, 64, int(block_k / 2), 4],
        # [256, 320, 64, 80, block_k,     2],
        # [256, 320, 64, 80, block_k,     3],
        # [256, 320, 64, 80, block_k,     4],
        # [256, 320, 64, 80, block_k * 2, 2],
        # [256, 320, 64, 80, block_k * 2, 3],
        # [256, 320, 64, 80, block_k * 2, 4],
        # [256, 320, 64, 80, int(block_k / 2), 2],
        # [256, 320, 64, 80, int(block_k / 2), 3],
        # [256, 320, 64, 80, int(block_k / 2), 4],

        # blockM = 320
        # [320, 256, 80, 64, block_k    , 2],
        # [320, 256, 80, 64, block_k    , 3],
        # [320, 256, 80, 64, block_k    , 4],
        # [320, 256, 80, 64, block_k * 2, 2],
        # [320, 256, 80, 64, block_k * 2, 3],
        # [320, 256, 80, 64, block_k * 2, 4],
        # [320, 256, 80, 64, int(block_k / 2), 2],
        # [320, 256, 80, 64, int(block_k / 2), 3],
        # [320, 256, 80, 64, int(block_k / 2), 4],
        ]

    if 'dense' in gemm_type:
        tile_list.extend([
            # blockM = 128
            [128, 128, 64, 64, block_k    , 2],
            [128, 128, 64, 64, block_k    , 3],
            [128, 128, 64, 64, block_k    , 4],
            [128, 128, 64, 64, block_k * 2, 3],
            [128, 256, 64, 64, block_k    , 2],
            [128, 256, 64, 64, block_k    , 3],

            # blockM = 256
            [256, 64, 32, 64, block_k,      2],
            [256, 64, 32, 64, block_k,      3],
            [256, 64, 32, 64, block_k * 2,  2],
            [256, 64, 32, 64, block_k * 2,  3],
            [256, 128, 64, 64, block_k    , 2],
            [256, 128, 64, 64, block_k    , 3],
            [256, 128, 64, 64, block_k    , 4],
            [256, 256, 64, 64, block_k,     4],

            # blockK = 64B
            [128, 128, 64, 64, int(block_k / 2), 2],
            [128, 256, 64, 64, int(block_k / 2), 2],
            [256, 128, 64, 64, int(block_k / 2), 2],
        ])

    tile_list_rtn = set()
    if d == torch.float8_e4m3fn:
        for tile in tile_list:
            if tile[0] != 48 and tile[0] != 160 and tile[0] < 256 and not (tile[1] == 256 and tile[4] == 256):
                tile_list_rtn.add(tuple(tile))
                if tile[4] == block_k and tile[0] != 192:
                    tile_copy = copy.deepcopy(tile)
                    tile_copy[4] = int(block_k / 2)
                    tile_list_rtn.add(tuple(tile_copy))
        return tile_list_rtn

    if d == torch.uint8:
        for tile in tile_list:
            if tile[2] % 16 == 0 and tile[3] % 16 == 0 and tile[0] % 16 == 0 and tile[1] % 16 == 0 and tile[4] % 32 == 0:
                tile_list_rtn.add(tuple(tile))
                if tile[4] == block_k:
                    tile_copy = copy.deepcopy(tile)
                    tile_copy[4] = int(block_k / 2)
                    if tile[4] != 0 and tile_copy[4] % 32 == 0:
                        tile_list_rtn.add(tuple(tile_copy))

    return tile_list_rtn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GEMM tile check')
    parser.add_argument('--type', type=str, required=True, # now use grouped as nopad
                       choices=['grouped','contiguous', 'dense', 'nopad', 'masked'],
                       help='Choose the gemm configure')
    parser.add_argument('--groups', default=8, type=int, help='Number of groups for Grouped GEMM')
    parser.add_argument('--target_stage', default=3, type=int, choices=[-1,2,3,4,5], help='specify which num_stages config to run, -1 means all')
    parser.add_argument('--m', default=8192, type=int, help='M dimension')
    parser.add_argument('--n', default=8192, type=int, help='N dimension')
    parser.add_argument('--k', default=8192, type=int, help='K dimension')
    parser.add_argument("--cycle", action="store_true", help="measure cycles instead of duration")
    parser.add_argument('--thread_count', default=12, type=int, required=False, help='the thread_count when run multi thread prebuild')
    args = parser.parse_args()

    num_groups = args.groups
    m, n, k = args.m, args.n, args.k
    gemm_type = args.type
    global cycle, thread_count
    cycle = 0
    if (args.cycle):
        cycle = 1
    thread_count = args.thread_count
    print(f"cycle: {cycle}, thread_count: {thread_count}")

    d = torch.uint8 ### dtype for mxfp4
    search_space = []
    if gemm_type in ['dense']:
        search_space = get_search_space_doublecheck(d, 'dense', m, n, k)
    elif gemm_type in ['grouped', 'contiguous', 'nopad', 'masked',"GroupedContiguous", "GroupedNoPad", "GroupedMasked"]:
        # actually get_search_space_doublecheck only add special config for 'dense'
        search_space = get_search_space_doublecheck(d, gemm_type, m, n, k)
    else:
        print("Must give a gemm type configure ")

    config_list = []
    print(f"only run config_list that satisfy num_stages=={args.target_stage}")
    for tile in search_space:
        block_m, block_n, warp_m, warp_n, block_k, num_stages = tile
        if args.target_stage > 0 and num_stages != args.target_stage:
            # only run num_stage == target_stage, because run all configs take too much time (>7200s)
            continue
        sm = get_num_sms()
        smem_config = get_smem_config_fp4(num_stages, block_m, block_n, warp_m, warp_n, block_k)
        config_list.append((sm, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config))

    results = test_all_configs(config_list, gemm_type, f'{gemm_type}_kernel_config_results_{m}{n}{k}_double.txt', f'{gemm_type}_kernel_config_results_{m}{n}{k}_double.json', m, n, k, num_groups)