import torch
import json
from typing import List, Dict, Tuple
import deep_gemm
import argparse
from test_fp4_core import quantize_fp4_torch, uint8_padding, dequantize_fp4_torch
from deep_gemm.jit_kernels.utils import get_num_sms
from utils import construct_group_m_list
from deep_gemm.jit_kernels.gemm_fp4 import get_smem_config
from deep_gemm import ceil_div
from deep_gemm.jit_kernels.utils import get_search_space

# This tile config check file only support MXFP4
def test_kernel_config(configs: Tuple, m, n, k, num_groups, gemm_type) -> Tuple[bool, str]:
    if 'dense' in gemm_type:
        try:
            A = torch.randn(m, k, dtype=torch.bfloat16, device='cuda').contiguous()
            B = torch.randn(n, k, dtype=torch.bfloat16, device='cuda').contiguous()
            x = quantize_fp4_torch(A.to(torch.bfloat16))
            y = quantize_fp4_torch(B.to(torch.bfloat16))
            a_dequant = dequantize_fp4_torch(x[0], x[1]).cuda()
            b_dequant = dequantize_fp4_torch(y[0], y[1]).cuda()
            bias = torch.randn(1, n, dtype=torch.float32, device='cuda')
            out = torch.zeros(m, n, dtype=torch.float32, device='cuda')
            ref_out = torch.mm(a_dequant, b_dequant.T)
            x_scale = uint8_padding(x[1])
            y_scale = uint8_padding(y[1])
            x = x[0], x_scale
            y = y[0], y_scale

            deep_gemm.gemm_fp4_fp4_fp32_nt(x, y, bias, out, configs)
            torch.cuda.synchronize()

            if torch.allclose(out, ref_out.to('cuda').to(torch.float), rtol=1e-2, atol=1e-3):
                return True, "Success"
            else:
                return False, "Gemm compute result is wrong!!!"

        except RuntimeError as e:
            error_msg = str(e)
            return False, error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, error_msg
    elif 'grouped' in gemm_type:
        try:
            x, y, m_indices, bias, out, ref_out = construct_grouped(num_groups, m, k, n, 'uniform', 1)

            deep_gemm.m_grouped_gemm_fp4_fp4_fp32_nt_nopad(x, y, bias, out, m_indices, configs=configs)
            torch.cuda.synchronize()
            if torch.allclose(out, ref_out, rtol=1e-2, atol=1e-3):
                return True, "Success"
            else:
                return False, "Gemm compute result is wrong!!!"

        except RuntimeError as e:
            error_msg = str(e)
            return False, error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, error_msg

def construct_grouped(num_groups: int, m: int, k: int, n: int, distribution: str, alignment: int):
    group_ms = construct_group_m_list(distribution, num_groups, m)
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    bias = torch.zeros((num_groups, n), device='cuda', dtype=torch.float)

    out = torch.empty((m, n), device='cuda', dtype=torch.float)
    ref_out = torch.empty((m, n), device='cuda', dtype=torch.float)

    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + ceil_div(group_m, alignment) * alignment
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        a, a_scale = quantize_fp4_torch(x[start:aligned_end].to(torch.bfloat16).cuda())
        b, b_scale = quantize_fp4_torch(y[i].to(torch.bfloat16).cuda())
        a_dequant = dequantize_fp4_torch(a, a_scale).to(torch.float)
        b_dequant = dequantize_fp4_torch(b, b_scale).to(torch.float)
        ref_out[start:aligned_end] = a_dequant @ b_dequant.t()
        ref_out[start:aligned_end] = ref_out[start:aligned_end] + bias[i]
        start = aligned_end

    x_fp4 = quantize_fp4_torch(x.to(torch.bfloat16).to('cuda'))
    x_fp4_scale = uint8_padding(x_fp4[1])
    y_fp4 = (torch.empty((num_groups, n, int(k / 2)), device='cuda', dtype=torch.uint8), torch.empty((num_groups, n, int(k / 32)), device='cuda', dtype=torch.uint8))
    y_scale = []
    for i in range(num_groups):
        y_fp4[0][i], y_fp4[1][i] = quantize_fp4_torch(y[i].to(torch.bfloat16))
        y_scale.append(uint8_padding(y_fp4[1][i]))
    y_fp4_scale = torch.stack(y_scale, dim=0)
    return (x_fp4[0].to("cuda"), x_fp4_scale.to("cuda")), (y_fp4[0].to("cuda"), y_fp4_scale.to("cuda")), m_indices, bias, out, ref_out.to('cuda').to(torch.float)

def test_all_configs(config_list: List[Tuple],
                     gemm_type: str,
                     output_file: str = "kernel_config_results.txt",
                     json_file: str = "kernel_config_results.json",
                     m: int = 1024,
                     n: int = 1024,
                     k: int = 1024,
                     num_groups: int = 1) -> Dict:
    valid_configs = []
    invalid_configs = []

    print(f"Start testing {len(config_list)} configs...")
    print(f"Test matrix dimensions: M={m}, N={n}, K={k}\n")

    for i, configs in enumerate(config_list):
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs

        print(f"[{i+1}/{len(config_list)}] test config:")
        print(f"  num_sms={num_sms}, block_m={block_m}, block_n={block_n}, block_k={block_k}")
        print(f"  warp_m={warp_m}, warp_n={warp_n}, num_stages={num_stages}")
        if 'dense' in gemm_type:
            is_valid, error_msg = test_kernel_config(configs, m, n, k, num_groups, gemm_type)
        elif 'grouped' in gemm_type:
            is_valid, error_msg = test_kernel_config(configs, m, n, k, num_groups, gemm_type)

        config_dict = {
            'num_sms': num_sms,
            'block_m': block_m,
            'block_n': block_n,
            'block_k': block_k,
            'warp_m': warp_m,
            'warp_n': warp_n,
            'num_stages': num_stages,
            'error': error_msg
        }

        if is_valid:
            print(f"Test success!\n")
            valid_configs.append(config_dict)
        else:
            print(f"Test failure: {error_msg}\n")
            invalid_configs.append(config_dict)

    results = {
        'test_params': {'m': m, 'n': n, 'k': k},
        'total': len(config_list),
        'valid': len(valid_configs),
        'invalid': len(invalid_configs),
        'valid_configs': valid_configs,
        'invalid_configs': invalid_configs
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"{gemm_type} Kernel Configuration Test Results\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Test Matrix Size: M={m}, N={n}, K={k}\n")
        f.write(f"Total Configs: {results['total']}\n")
        f.write(f"Valid Configs: {results['valid']}\n")
        f.write(f"Invalid Configs: {results['invalid']}\n")
        f.write(f"Success Rate: {results['valid']/results['total']*100:.2f}%\n\n")

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
        [128, 320, 64, 80, block_k    , 2],
        [128, 320, 64, 80, block_k    , 3],
        [128, 320, 64, 80, block_k    , 4],
        [128, 320, 64, 80, block_k * 2, 2],
        [128, 320, 64, 80, block_k * 2, 3],
        [128, 320, 64, 80, block_k * 2, 4],
        [128, 320, 64, 80, int(block_k / 2), 2],
        [128, 320, 64, 80, int(block_k / 2), 3],
        [128, 320, 64, 80, int(block_k / 2), 4],

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
        [256, 320, 64, 80, block_k,     2],
        [256, 320, 64, 80, block_k,     3],
        [256, 320, 64, 80, block_k,     4],
        [256, 320, 64, 80, block_k * 2, 2],
        [256, 320, 64, 80, block_k * 2, 3],
        [256, 320, 64, 80, block_k * 2, 4],
        [256, 320, 64, 80, int(block_k / 2), 2],
        [256, 320, 64, 80, int(block_k / 2), 3],
        [256, 320, 64, 80, int(block_k / 2), 4],

        # blockM = 320
        [320, 256, 80, 64, block_k    , 2],
        [320, 256, 80, 64, block_k    , 3],
        [320, 256, 80, 64, block_k    , 4],
        [320, 256, 80, 64, block_k * 2, 2],
        [320, 256, 80, 64, block_k * 2, 3],
        [320, 256, 80, 64, block_k * 2, 4],
        [320, 256, 80, 64, int(block_k / 2), 2],
        [320, 256, 80, 64, int(block_k / 2), 3],
        [320, 256, 80, 64, int(block_k / 2), 4],
        ]

    tile_list_rtn = []
    if d == torch.float8_e4m3fn:
        for tile in tile_list:
            if tile[0] != 48 and tile[0] != 160 and tile[0] < 256 and not (tile[1] == 256 and tile[4] == 256):
                tile_list_rtn.append(tile)
                if tile[4] == block_k and tile[0] != 192:
                    tile_copy = copy.deepcopy(tile)
                    tile_copy[4] = int(block_k / 2)
                    tile_list_rtn.append(tile_copy)
        return tile_list_rtn

    # add block_k / 2 tile for k <256
    if k != 0 and k <= 512:
        for tile in tile_list:
            tile_list_rtn.append(tile)
            if tile[4] == block_k:
                tile_copy = copy.deepcopy(tile)
                tile_copy[4] = int(block_k / 2)
                tile_list_rtn.append(tile_copy)
    else:
        tile_list_rtn = tile_list

    return tile_list_rtn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GEMM tile check')
    parser.add_argument('--type', type=str,
                       choices=['grouped', 'dense'],
                       help='Choose the gemm configure')
    parser.add_argument('--groups', default=1, type=int, help='Number of groups for Grouped GEMM')
    parser.add_argument('--m', default=8192, type=int, help='M dimension')
    parser.add_argument('--n', default=8192, type=int, help='N dimension')
    parser.add_argument('--k', default=8192, type=int, help='K dimension')
    args = parser.parse_args()

    num_groups = args.groups
    m, n, k = args.m, args.n, args.k
    gemm_type = args.type

    out = torch.zeros(m, n, dtype=torch.float32, device='cuda')
    if gemm_type in ['dense']:
        search_space = get_search_space_doublecheck(out, 'dense', m, n, k)
    elif gemm_type in ['grouped']:
        search_space = get_search_space_doublecheck(out, 'dense', m, n, k)

    config_list = []
    for tile in search_space:
        block_m, block_n, warp_m, warp_n, block_k, num_stages = tile
        sm = get_num_sms()
        smem_config = get_smem_config(num_stages, k, block_m, block_n, block_k)
        config_list.append((sm, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config))

    results = test_all_configs(config_list, gemm_type, f'{gemm_type}_kernel_config_results_{m}{n}{k}_double.txt', f'{gemm_type}_kernel_config_results_{m}{n}{k}_double.json', m, n, k, num_groups)