import random
import torch
import deep_gemm
from deep_gemm.jit_kernels.utils import is_ppu1v5_device
from utils import construct_contiguous_grouped, test_m_grouped_gemm_masked, test_m_grouped_gemm_nopad, test_m_grouped_gemm_fused
from utils import set_acc_check, set_benchmark, parse_deepgemm_string_re
from deep_gemm.testing.numeric import calc_diff


def test_tile_loop(d='w4a16'):
    print("Testing GroupedNoPad GEMM (W4A16) with tile configs...")
    # kimi k2.5 tp8
    num_groups, m, k, n, group_size = 384, 32, 7168, 512, 32
    m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, k, n, d, 'uniform', 1, quant_type='group', group_size=group_size)
    num_sms = deep_gemm.get_num_sms()
    if d == 'w4fa16_mma':
        configs_list = [
            (num_sms, 16, 128, 128, 16, 64, 128, 3, 1),
            (num_sms, 128, 256, 128, 64, 64, 128, 2, 1),
            (num_sms, 16, 256, 256, 16, 64, 64, 3, 1),
            (num_sms, 32, 256, 128, 32, 64, 64, 3, 1),
            (num_sms, 16, 256, 128, 16, 64, 128, 2, 2),
        ]
    else:
        configs_list = [
            (num_sms, 16, 64, 32, 16, 64, 32, 4, 1),
            (num_sms, 16, 128, 128, 16, 64, 32, 4, 1),
            (num_sms, 16, 256, 64, 16, 64, 32, 4, 1),
            (num_sms, 16, 256, 128, 16, 64, 32, 4, 1),
            (num_sms, 32, 256, 128, 32, 64, 32, 4, 1),
            (num_sms, 64, 256, 128, 64, 64, 32, 4, 1),
            (num_sms, 128, 128, 64, 64, 64, 64, 4, 1),
        ]
    for configs in configs_list:
        deep_gemm.m_grouped_gemm_w4a16_nopad(x, y, out, m_indices, configs=configs)
        diff = calc_diff(out, ref_out)
        if diff >= 0.0015:
            print("ref_out:", ref_out)
            print("out:", out)
            torch.testing.assert_close(out, ref_out, rtol=5e-1, atol=2)

        assert diff < 0.0015, f'{m=}, {k=}, {n=}, {diff:.5f}'
        print(f"Passed acc_check with tile:{configs}\n")
    print("Passed\n")


def test_m_grouped_gemm_nopad_loop(d='w4a16'):
    print("Testing GroupedNoPad GEMM (W4A16)...")
    for num_groups, expected_m_per_group in ((256, 1), (256, 16), (384, 64)):
        for k, n in ((256, 768), (2048, 7168)):
            case = f'GroupedNoPad,data_type:{d},groups:{num_groups},m:{num_groups * expected_m_per_group},n:{n},k:{k},distribution:uniform,quant_type:group,group_size:32'
            print(case)
            test_m_grouped_gemm_nopad(parse_deepgemm_string_re(case))
    print("Passed\n")


def test_m_grouped_gemm_masked_loop(d='w4a16'):
    print("Testing GroupedMasked GEMM (W4A16)...")
    for num_groups, expected_m_per_group in ((64, 1), (16, 16), (128, 48)):
        for k, n in ((256, 768), (2048, 7168)):
            case = f'GroupedMasked,data_type:{d},groups:{num_groups},m:{num_groups * expected_m_per_group},n:{n},k:{k},distribution:zipf,quant_type:group,group_size:32'
            print(case)
            test_m_grouped_gemm_masked(parse_deepgemm_string_re(case))
    print("Passed\n")


def test_m_grouped_gemm_fused_loop(d='w4a16'):
    print("Testing GroupedFused GEMM (W4A16)...")
    for num_groups, num_token, topk in ((256, 1, 8), (256, 16, 8), (384, 32, 16)):
        for k, n in ((256, 768), (2048, 7168)):
            case = f'GroupedFused,data_type:{d},groups:{num_groups},num_token:{num_token},topk:{topk},n:{n},k:{k},quant_type:group,group_size:32'
            print(case)
            test_m_grouped_gemm_fused(parse_deepgemm_string_re(case))
    print("Passed\n")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    set_acc_check(True)
    set_benchmark(False)

    methods = ['w4a16', 'w4fa16']
    if is_ppu1v5_device():
        methods.append('w4fa16_mma')
    for d in methods:
        test_m_grouped_gemm_fused_loop(d)
        test_m_grouped_gemm_masked_loop(d)
        test_m_grouped_gemm_nopad_loop(d)
        test_tile_loop(d)
