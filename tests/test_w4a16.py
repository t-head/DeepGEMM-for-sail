import random
import torch
import torch.nn.functional as F
import deep_gemm
from deep_gemm.jit_kernels.utils import is_ppu1v5_device
from utils import construct_group_m_list, construct_contiguous_grouped, test_m_grouped_gemm_masked, test_m_grouped_gemm_nopad, test_m_grouped_gemm_fused
from utils import set_acc_check, get_acc_check, set_benchmark, get_benchmark
from math_utils import quant_w4a16, calc_diff


def debug_test():
    torch.set_printoptions(precision=2, sci_mode=False)
    acc_check = True
    num_groups, m, k, n, group_size = 1, 32, 64, 128, 32

    group_ms = construct_group_m_list('uniform', num_groups, m)
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        m_indices[start:actual_end] = i
        m_indices[actual_end:start + group_m] = -1
        start = actual_end

    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    y_ref, y_quant, y_scale = quant_w4a16(y, group_size)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    def print_tid_reg(mode, t, tid):
        group        = tid // 4
        tid_in_group = tid % 4
        row0 = group
        row1 = group + 8
        c0   = tid_in_group * 2
        c1   = tid_in_group * 2 + 8
        ppu15 = is_ppu1v5_device()
        if ppu15 and (mode == 'A' or mode == 'C'):
            coords = [
                (row0, c0), (row0, c0+1),
                (row1, c0), (row1, c0+1),
                (row0, c1), (row0, c1+1),
                (row1, c1), (row1, c1+1),
            ]
        elif (not ppu15 and mode == 'A') or mode == 'B':
            coords = [
                (row0, c0), (row0, c0+1),
                (row0, c1), (row0, c1+1),
                (row1, c0), (row1, c0+1),
                (row1, c1), (row1, c1+1),
            ]
        elif not ppu15 and mode == 'C':
            c0, c1 = tid_in_group, tid_in_group + 8
            coords = [
                (row0, c0), (row0, c0+4),
                (row0, c1), (row0, c1+4),
                (row1, c0), (row1, c0+4),
                (row1, c1), (row1, c1+4),
            ]
        elems = [t[r, c].item() for r, c in coords]
        elem_str  = " ".join(f"{v:.2f}" for v in elems)
        print(elem_str)

    def printt(t, is_int=False):
        if is_int:
            print(' '.join(f'{v}' for v in t.flatten().tolist()))
        else:
            print(' '.join(f'{v:.2f}' for v in t.flatten().tolist()))

    def debug(kid, tid):
        # print(x.shape, y_quant.shape, y_scale.shape, y_ref.shape, out.shape)
        eid, mid, nid = 0, 0, 0
        print("workidx:", eid, mid, nid, "kid:", kid, "tid:", tid)
        print("tCrA:", end=' ')
        xt = x[mid*16:(mid+1)*16, kid*16:(kid+1)*16]
        print_tid_reg('A', xt, tid)
        print("tCrB_i32:", end=' ')
        printt(y_quant[eid, kid, tid*4:(tid+1)*4], is_int=True)
        print("tCrS:", end=' ')
        printt(y_scale[eid, kid // 2, tid//4*8:(tid//4+1)*8])
        print("tCrB:", end=' ')
        y_reft = y_ref[eid, nid*16:(nid+1)*16, kid*16:(kid+1)*16]
        print_tid_reg('B', y_reft, tid)
        if True:
            tmpct = xt @ y_reft.T
            print("accum:", end=' ')
            print_tid_reg('C', tmpct, tid)
        if acc_check:
            print("out:", end=' ')
            outt = out[mid*16:(mid+1)*16, nid*16:(nid+1)*16]
            print_tid_reg('C', outt, tid)
            print("refout:", end=' ')
            ref_outt = ref_out[mid*16:(mid+1)*16, nid*16:(nid+1)*16]
            print_tid_reg('C', ref_outt, tid)

    def debug_diff():
        mask = (out - ref_out).abs() > 1e-4
        idx = torch.nonzero(mask)
        print(idx)
        for i in idx:
            i = tuple(i.tolist())
            print(i, out[i].item(), ref_out[i].item(), abs(out[i].item() - ref_out[i].item()))

    # debug(0, 0)
    deep_gemm.m_grouped_gemm_w4a16_nopad(x, (y_quant, y_scale), out, m_indices)
    if not acc_check: return
    deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(x, y_ref, ref_out, m_indices)
    diff = calc_diff(out, ref_out)
    # debug_diff()
    if diff >= 0.001:
        print("ref_out:", ref_out)
        print("out:", out)
        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-4)
    assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
    print(f"Passed acc_check\n")
    return


def test_tile_loop():
    print("Testing GroupedNoPad GEMM (W4A16) with tile configs...")
    # kimi k2.5 tp8
    num_groups, m, k, n, group_size = 384, 32, 7168, 512, 32
    m, x, y, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, m, k, n, 'w4a16', 'uniform', 1, quant_type='group', group_size=group_size)
    num_sms = deep_gemm.get_num_sms()
    configs_list = [
        (num_sms, 16, 64, 32, 16, 64, 32, 4),
        (num_sms, 16, 128, 128, 16, 64, 32, 4),
        (num_sms, 16, 256, 64, 16, 64, 32, 4),
        (num_sms, 16, 256, 128, 16, 64, 32, 4),
        (num_sms, 32, 256, 128, 32, 64, 32, 4),
        (num_sms, 64, 256, 128, 64, 64, 32, 4),
        (num_sms, 128, 128, 64, 64, 64, 64, 4),
    ]
    for configs in configs_list:
        deep_gemm.m_grouped_gemm_w4a16_nopad(x, y, out, m_indices, configs=configs)
        if True:
            diff = calc_diff(out, ref_out)
            if diff >= 0.0015:
                print("ref_out:", ref_out)
                print("out:", out)
                torch.testing.assert_close(out, ref_out, rtol=5e-1, atol=2)

            assert diff < 0.0015, f'{m=}, {k=}, {n=}, {diff:.5f}'
            print(f"Passed acc_check with tile:{configs}\n")
    print("Passed\n")


def test_m_grouped_gemm_nopad_loop():
    print("Testing GroupedNoPad GEMM (W4A16)...")
    for num_groups, expected_m_per_group in ((256, 1), (256, 16), (384, 64)):
        for k, n in ((256, 768), (2048, 7168)):
            group_size = 32
            print(f"Testing with num_groups={num_groups}, m={num_groups * expected_m_per_group}, n={n}, k={k}, group_size={group_size}")
            args = {
                "data_type": "w4a16",
                "groups": num_groups,
                "m": num_groups * expected_m_per_group,
                "n": n,
                "k": k,
                "distribution": "uniform",
                "quant_type": "group",
                "group_size": group_size
            }
            test_m_grouped_gemm_nopad(args)
    print("Passed\n")


def test_m_grouped_gemm_masked_loop():
    print("Testing GroupedMasked GEMM (W4A16)...")
    for num_groups, expected_m_per_group in ((64, 1), (16, 16), (128, 48)):
        for k, n in ((256, 768), (2048, 7168)):
            group_size = 32
            print(f"Testing with num_groups={num_groups}, m={num_groups * expected_m_per_group}, n={n}, k={k}, group_size={group_size}")
            args = {
                "data_type": "w4a16",
                "groups": num_groups,
                "m": num_groups * expected_m_per_group,
                "n": n,
                "k": k,
                "distribution": "zipf",
                "quant_type": "group",
                "group_size": group_size
            }
            test_m_grouped_gemm_masked(args)
    print("Passed\n")


def test_m_grouped_gemm_fused_loop():
    print("Testing GroupedFused GEMM (W4A16)...")
    for num_groups, num_token, topk in ((256, 1, 8), (256, 16, 8), (384, 32, 16)):
        for k, n in ((256, 768), (2048, 7168)):
            group_size = 32
            print(f"Testing with num_groups={num_groups}, num_token={num_token}, topk={topk}, n={n}, k={k}, group_size={group_size}")
            args = {
                "data_type": "w4a16",
                "groups": num_groups,
                "num_token": num_token,
                "topk": topk,
                "n": n,
                "k": k,
                "quant_type": "group",
                "group_size": group_size
            }
            test_m_grouped_gemm_fused(args)
    print("Passed\n")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    set_acc_check(1)

    debug_test()
    # test_m_grouped_gemm_fused_loop()
    # test_m_grouped_gemm_masked_loop()
    # test_m_grouped_gemm_nopad_loop()
    # test_tile_loop()
