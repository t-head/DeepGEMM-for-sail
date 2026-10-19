import torch
import torch.nn.functional as F
import time
from typing import Tuple
import deep_gemm
from deep_gemm import preprocess_mxfp4_scales, moe_align_block_size
from math_utils import ceil_div, calc_diff
from utils import construct, check_signal, construct_group_m_list, get_ref_backend, find_next_power_of_2
from utils import construct_contiguous_grouped as construct_nopad_base
from utils import construct_grouped_masked as construct_masked_base
from utils import quantize_fp4_torch, dequantize_fp4_torch
import argparse

# torch.cuda.manual_seed(42)


def test_gemm(args):
    m, n, k = args['m'], args['n'], args['k']
    x, y, out, ref_out = construct(m, k, n, torch.uint8)
    bias = torch.randn(1, n, dtype=torch.float32, device='cuda') if args.get("with_bias", True) else None

    deep_gemm.gemm_fp4_fp4_bf16_nt(x, y, bias, out)

    ref_out = (ref_out.float() + bias).to(torch.bfloat16) if bias is not None else ref_out
    # import ipdb; ipdb.set_trace()
    diff = calc_diff(out, ref_out)
    if diff >= 0.001:
        print("ref_out:", ref_out)
        print("out:", out)
        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-4)
    assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
    print(f"Passed with acc_check. diff: {diff}\n")

def test_m_grouped_gemm_nopad(args) -> None:
    num_groups, m, n, k, distribution = args['groups'], args['m'], args['n'], args['k'], args['distribution']
    with_bias = args.get("with_bias", True)
    m, x, y, m_indices, out, ref_out = construct_nopad_base(num_groups, m, k, n, torch.uint8, distribution, 1)
    bias = torch.randn((num_groups, n), device='cuda', dtype=torch.float) if with_bias else None
    if bias is not None:
        ref_out = (ref_out.float() + torch.where((m_indices != -1).unsqueeze(1), bias[m_indices.clamp(min=0)], torch.zeros_like(bias[0]))).to(torch.bfloat16)

    deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(x, y, bias, out, m_indices)

    diff = calc_diff(out, ref_out)
    if diff >= 0.001:
        print("ref_out:", ref_out)
        print("out:", out)
        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-4)
    assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
    print(f"Passed with acc_check. diff: {diff}\n")
    return

def test_m_grouped_gemm_masked(args) -> None:
    num_groups, m, n, k, distribution = args['groups'], args['m'], args['n'], args['k'], args['distribution']
    enable_sbo_overlap = args['enable_sbo_overlap'] if 'enable_sbo_overlap' in args else False
    with_bias = args.get("with_bias", True)
    expected_m_per_group = ceil_div(m, num_groups)
    x, y, masked_m, out, ref_out, signal, max_m = construct_masked_base(num_groups, m, expected_m_per_group, k, n, torch.uint8, distribution, enable_sbo_overlap)
    bias = torch.randn((num_groups, n), device='cuda', dtype=torch.float) if with_bias else None
    if bias is not None:
        ref_out = (ref_out.float() + bias.unsqueeze(1)).to(torch.bfloat16)
    result = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(x, y, bias, out, masked_m, expected_m_per_group, enable_sbo_overlap=enable_sbo_overlap, signal=signal)

    if enable_sbo_overlap:
        block_m, threshold = result
        check_signal(num_groups, max_m, block_m, threshold, signal, masked_m)

    for j in range(num_groups):
        diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
        if (masked_m[j] != 0):
            if diff >= 0.001:
                print(f"ref_out[{j}]:", ref_out[j, :masked_m[j].item()])
                print(f"out[{j}]:", out[j, :masked_m[j].item()])
            assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
    print(f"Passed with acc_check. diff: {diff}\n")
    return


def construct_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int, distribution: str,
                             enable_sbo_overlap: bool = False, with_bias: bool = True, enable_act_and_quant_fusing: bool = False):
    tensor_device = 'cuda' if get_ref_backend() == "device" else 'cpu'
    # Construct mask
    list_m =  construct_group_m_list(distribution, num_groups, max_m, is_mask=True, em=expected_m_per_group)
    masked_m = torch.tensor(list_m, device=tensor_device, dtype=torch.int)
    max_m = max(128, find_next_power_of_2(list_m))
    assert masked_m.amax().item() <= max_m, f"max masked_m={masked_m.amax().item()}, allowed max_m={max_m}"
    
    x = torch.randn((num_groups, max_m, k), device=tensor_device, dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device=tensor_device, dtype=torch.bfloat16)
    if enable_act_and_quant_fusing:
        out = torch.empty((num_groups, max_m, (n // 4)), dtype=torch.uint8, device='cuda')
        out_scale = torch.empty((num_groups, max_m, ceil_div((n // 4), 32)), dtype=torch.uint16, device='cuda')
    else:
        out = torch.empty((num_groups, max_m, n), device=tensor_device, dtype=torch.bfloat16)
    bias = torch.randn((num_groups, n), device='cuda', dtype=torch.float) if with_bias else None

    x_fp4 = []
    x_fp4_scale = []
    y_fp4 = []
    y_fp4_scale = []
    x_ref = []
    y_ref = []
    for i in range(num_groups):
        a, a_scale = quantize_fp4_torch(x[i].cuda())
        b, b_scale = quantize_fp4_torch(y[i].cuda())
        a_dequant = dequantize_fp4_torch(a, a_scale).to(torch.float)
        b_dequant = dequantize_fp4_torch(b, b_scale).to(torch.float)
        x_fp4.append(a)
        x_fp4_scale.append(a_scale)
        y_fp4.append(b)
        y_fp4_scale.append(b_scale)
        x_ref.append(a_dequant)
        y_ref.append(b_dequant)

    x_fp4 = torch.stack(x_fp4, dim=0)
    x_fp4_scale = torch.stack(x_fp4_scale, dim=0)
    x_fp4_scale = preprocess_mxfp4_scales(scale=x_fp4_scale)
    y_fp4 = torch.stack(y_fp4, dim=0)
    y_fp4_scale = torch.stack(y_fp4_scale, dim=0)
    y_fp4_scale = preprocess_mxfp4_scales(scale=y_fp4_scale)
    x_ref = torch.stack(x_ref, dim=0)
    y_ref = torch.stack(y_ref, dim=0)

    ref_out = torch.einsum('gmk,gnk->gmn', x_ref, y_ref)
    ref_out = ref_out + bias.unsqueeze(1) if bias is not None else ref_out

    max_signal_size = num_groups * ceil_div(max_m, 64)
    signal = torch.zeros(max_signal_size, dtype=torch.int32, device=tensor_device) if enable_sbo_overlap else torch.empty(0).int()

    if enable_act_and_quant_fusing:
        return (x_fp4.to('cuda'), x_fp4_scale.to('cuda')), (y_fp4.to('cuda'), y_fp4_scale.to('cuda')), bias, masked_m.to('cuda'), out, out_scale, ref_out.to('cuda'), signal.to('cuda'), max_m
    else:
        return (x_fp4.to('cuda'), x_fp4_scale.to('cuda')), (y_fp4.to('cuda'), y_fp4_scale.to('cuda')), bias, masked_m.to('cuda'), out, ref_out.to('cuda'), signal.to('cuda'), max_m

def test_m_grouped_gemm_nopad_loop(num_groups: int = None, m: int = None, n: int = None, k: int = None) -> None:
    print("Running GroupedNoPad GEMM test...")
    if num_groups is not None and m is not None and n is not None and k is not None:
        print(f"Testing with num_groups={num_groups}, m={m * num_groups}, n={n}, k={k}")
        args = {"groups": num_groups, "m": m * num_groups, "n": n, "k": k, "distribution": "uniform"}
        test_m_grouped_gemm_nopad(args)
    else:
        print("Running default test suite...")
        for with_bias in [True, False]:
            for num_groups, expected_m_per_group in ((256, 1), (256, 4), (256, 16), (256, 32), (128, 8), (128, 64), (128, 1024)):
                for k, n in ((256, 768), (512, 128), (2048, 7168), (7168, 4096)):
                    print(f"Testing with num_groups={num_groups}, m={num_groups * expected_m_per_group}, n={n}, k={k}")
                    args = {"groups": num_groups, "m": num_groups * expected_m_per_group, "n": n, "k": k, "distribution": "uniform", "with_bias": with_bias}
                    test_m_grouped_gemm_nopad(args)
    print("Passed\n")

def test_m_grouped_gemm_masked_loop(num_groups: int = None, m: int = None, n: int = None, k: int = None) -> None:
    print("Running GroupedMasked GEMM test...")
    if num_groups is not None and m is not None and n is not None and k is not None:
        print(f"Testing with num_groups={num_groups}, m={m * num_groups}, n={n}, k={k}")
        args = {"groups": num_groups, "m": m * num_groups, "n": n, "k": k, "distribution": "uniform"}
        test_m_grouped_gemm_masked(args)
    else:
        print("Running default test suite...")
        for with_bias in [True, False]:
            for num_groups, expected_m_per_group in ((256, 1), (256, 4), (256, 16), (256, 32), (128, 8), (128, 64), (128, 1024)):
                for k, n in ((256, 768), (512, 128), (2048, 7168), (7168, 4096)):
                    print(f"Testing with num_groups={num_groups}, m={num_groups * expected_m_per_group}, n={n}, k={k}")
                    args = {"groups": num_groups, "m": num_groups * expected_m_per_group, "n": n, "k": k, "distribution": "uniform", "with_bias": with_bias}
                    test_m_grouped_gemm_masked(args)
    print("Passed\n")

def test_gemm_loop(m: int = None, n: int = None, k: int = None) -> None:
    print("Running GEMM test...")
    if m is not None and n is not None and k is not None:
        print(f"Testing with m={m}, n={n}, k={k}")
        args = {"m": m, "n": n, "k": k}
        test_gemm(args)
    else:
        print("Running default test suite...")
        for with_bias in [True, False]:
            for m in (64, 128, 4096):
                for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
                    print(f"Testing with m={m}, n={n}, k={k}")
                    args = {"m": m, "n": n, "k": k, "with_bias": with_bias}
                    test_gemm(args)
    print("Passed\n")

def test_m_grouped_gemm_masked_silu_and_mul_post_quant(args) -> None:
    num_groups, m, n, k, distribution, swiglu_limit = args['groups'], args['m'], args['n'], args['k'], args['distribution'], args['swiglu_limit']
    enable_sbo_overlap = args['enable_sbo_overlap'] if 'enable_sbo_overlap' in args else False

    expected_m_per_group = ceil_div(m, num_groups)
    x, y, bias, masked_m, out, out_scale, ref_out, signal, max_m = construct_grouped_masked(num_groups, m, expected_m_per_group, k, n, distribution, enable_sbo_overlap, with_bias=False, enable_act_and_quant_fusing=True)

    from deep_gemm import preprocess_mxfp4_weight_for_act_and_quant_fusing
    weight_scale = y[1].contiguous().view(torch.uint8)
    y_interleave = preprocess_mxfp4_weight_for_act_and_quant_fusing(weight=y[0], weight_scale=weight_scale)
    result = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(x, y_interleave, bias, out, masked_m, expected_m_per_group, enable_sbo_overlap=enable_sbo_overlap, signal=signal, out_scale=out_scale, swiglu_limit=swiglu_limit)

    ### Reference
    if swiglu_limit > 0.0:
        gate_clamped = torch.clamp(ref_out[:, :, :(ref_out.shape[-1]//2)], max=swiglu_limit)
        up_clamped = torch.clamp(ref_out[:, :, (ref_out.shape[-1]//2):], min=-swiglu_limit, max=swiglu_limit)
        silu = gate_clamped * torch.sigmoid(gate_clamped) * up_clamped
    else:
        silu = ref_out[:, :, :(ref_out.shape[-1]//2)] * torch.sigmoid(ref_out[:, :, :(ref_out.shape[-1]//2)]) * ref_out[:, :, (ref_out.shape[-1]//2):]
    ref_quant_torch, ref_quant_scale_torch_ = quantize_fp4_torch(src_tensor=silu)
    ref_quant_scale_torch = preprocess_mxfp4_scales(scale=ref_quant_scale_torch_)

    ### Check
    max_diff, max_diff_scale = 0, 0
    for j in range(num_groups):
        diff = calc_diff(out[j, :masked_m[j].item()], ref_quant_torch[j, :masked_m[j].item()])
        if (masked_m[j] != 0):
            if diff >= 0.0001:
                print(f"ref_out[{j}]:", ref_quant_torch[j, :masked_m[j].item()])
                print(f"out[{j}]:", out[j, :masked_m[j].item()])
            assert diff < 0.0001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
        diff_scale = calc_diff(out_scale[j, :masked_m[j].item()], ref_quant_scale_torch[j, :masked_m[j].item()])
        if (masked_m[j] != 0):
            if diff_scale >= 0.0001:
                print(f"ref_scale_out[{j}]:", ref_quant_scale_torch[j, :masked_m[j].item()])
                print(f"out_scale[{j}]:", out_scale[j, :masked_m[j].item()])
            assert diff_scale < 0.0001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
        max_diff = max(diff, max_diff)
        max_diff_scale = max(diff_scale, max_diff_scale)
    print(f"Passed with acc_check. max_diff: {max_diff}, max_diff_scale: {max_diff_scale}\n")

def test_m_grouped_gemm_masked_silu_and_mul_post_quant_loop(num_groups: int = None, m: int = None, n: int = None, k: int = None) -> None:
    print("Running GroupedMasked GEMM with enable_act_and_quant_fusing test...")
    print("Running default test suite...")
    for num_groups, expected_m_per_group in ((8, 31), (8, 256),):
        for k, n in ((4096, 4096), (8192, 4096), (1024, 4096), (4096, 4032),):
            for swiglu_limit in (0.0, 10.0):
                print(f"Testing with num_groups={num_groups}, m={num_groups * expected_m_per_group}, n={n}, k={k}")
                args = {"groups": num_groups, "m": num_groups * expected_m_per_group, "n": n, "k": k, "distribution": "uniform", "swiglu_limit": swiglu_limit}
                test_m_grouped_gemm_masked_silu_and_mul_post_quant(args)
    print("Passed\n")
        
def construct_grouped_fused(num_groups: int, m: int, k: int, n: int, topk: int, distribution: str, alignment: int,
                           enable_act_and_quant_fusing: bool = False):
    ### 0. prepare input
    score = torch.randn((m, num_groups), device='cuda', dtype=torch.float32)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    ### the shape of fusedMoe is same with fusedNoPad
    if enable_act_and_quant_fusing:
        out = torch.empty((m * topk, (n // 4)), device='cuda', dtype=torch.uint8)
        out_scale = torch.empty((m * topk, ceil_div((n // 4), 32)), device='cuda', dtype=torch.uint16)
    else:
        out = torch.empty((m * topk, n), device='cuda', dtype=torch.bfloat16)
    ref_out = []

    ### 1. do quantization
    x_fp4_tupple = quantize_fp4_torch(x)
    x_fp4 = x_fp4_tupple[0]
    ### no preprocess_mxfp4_scales, padding to even
    x_scale_fp4 = x_fp4_tupple[1]
    if (x_scale_fp4.shape[-1] % 2):
        x_scale_fp4 = torch.nn.functional.pad(x_scale_fp4, (0, 1))
    x_scale_fp4 = x_scale_fp4.view(torch.uint16) 

    y_fp4_tuple = (torch.empty((num_groups, n, x_fp4.shape[-1]), device='cuda', dtype=torch.uint8), torch.empty((num_groups, n, x_fp4_tupple[1].shape[-1]), device='cuda', dtype=torch.uint8))
    y_scale = []
    for i in range(num_groups):
        y_fp4_tuple[0][i], y_fp4_tuple[1][i] = quantize_fp4_torch(y[i])
        y_scale.append(y_fp4_tuple[1][i])
    y_fp4 = y_fp4_tuple[0]
    y_scale_fp4_ = torch.stack(y_scale, dim=0)
    y_scale_fp4 = preprocess_mxfp4_scales(scale=y_scale_fp4_)

    ### 2. calculate reference output
    ### the fused epilogue reference needs the pre-activation gemm result in float32
    ref_dtype = torch.float32 if enable_act_and_quant_fusing else torch.bfloat16
    k_ = x_fp4.shape[-1]
    sfk_ = x_fp4_tupple[1].shape[-1]
    x_fp4_ = x_fp4.view(m, -1, k_).repeat(1, topk, 1).reshape(-1, k_)
    x_scale_fp4_ = x_fp4_tupple[1].view(m, -1, sfk_).repeat(1, topk, 1).reshape(-1, sfk_)
    topk_ids_ = topk_ids.view(-1)
    for i in range(y_fp4.shape[0]):
        mask = (topk_ids_ == i)
        if mask.sum():
            ref_out_group = dequantize_fp4_torch(x_fp4_[mask], x_scale_fp4_[mask]).to(ref_dtype) @ dequantize_fp4_torch(y_fp4[i], y_scale_fp4_[i]).to(ref_dtype).transpose(0, 1)
            ref_out.append(ref_out_group)
    ref_out = torch.concat(ref_out, dim=0)

    ### 3. return
    if enable_act_and_quant_fusing:
        return (x_fp4, x_scale_fp4), (y_fp4, y_scale_fp4), topk_ids, out, out_scale, ref_out
    return (x_fp4, x_scale_fp4), (y_fp4, y_scale_fp4), topk_ids, out, ref_out

def test_m_grouped_gemm_fused(args) -> None:
    num_groups, m, n, k, topk, distribution = args['groups'], args['m'], args['n'], args['k'], args['topk'], args['distribution']
    x, y, topk_ids, out, ref_out = construct_grouped_fused(num_groups, m, k, n, topk, distribution, 1)

    config, m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, inv_perm, expert_ids = (
        moe_align_block_size(x[0], y[0], topk_ids, False)
    )
    deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_fused(x, y, out, m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, config)

    diff = calc_diff(out, ref_out)
    if diff >= 0.001:
        print("ref_out:", ref_out)
        print("out:", out)
        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-4)
    assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
    print(f"Passed with acc_check. diff: {diff}\n")
    return

def test_m_grouped_gemm_fused_loop(num_groups: int = None, m: int = None, n: int = None, k: int = None, topk: int = None) -> None:
    print("Running GroupedFused GEMM test...")
    if num_groups is not None and m is not None and n is not None and k is not None and topk is not None:
        print(f"Testing with num_groups={num_groups}, m={m}, n={n}, k={k}, topk={topk}")
        args = {"groups": num_groups, "m": m, "n": n, "k": k, "topk": topk, "distribution": "uniform"}
        test_m_grouped_gemm_fused(args)
    else:
        print("Running default test suite...")
        ### derived from glm5 tp16
        for num_groups, topk, m in ((256, 8, 16), (256, 1, 16 * 8), (256, 8, 64), (256, 1, 64 * 8), (256, 8, 256), (256, 1, 256 * 8), (256, 8, 2048), (256, 1, 2048 * 8), ):
            for k, n in ((6144, 256), ):
                print(f"Testing with num_groups={num_groups}, m={m}, n={n}, k={k}, topk={topk}")
                args = {"groups": num_groups, "m": m, "n": n, "k": k, "topk": topk, "distribution": "uniform"}
                test_m_grouped_gemm_fused(args)
    print("Passed\n")

def test_m_grouped_gemm_fused_silu_and_mul_post_quant(args) -> None:
    num_groups, m, n, k, topk, distribution, swiglu_limit = args['groups'], args['m'], args['n'], args['k'], args['topk'], args['distribution'], args['swiglu_limit']
    x, y, topk_ids, out, out_scale, ref_out = construct_grouped_fused(
        num_groups, m, k, n, topk, distribution, 1, enable_act_and_quant_fusing=True)

    from deep_gemm import preprocess_mxfp4_weight_for_act_and_quant_fusing
    weight_scale = y[1].contiguous().view(torch.uint8)
    y_interleave = preprocess_mxfp4_weight_for_act_and_quant_fusing(weight=y[0], weight_scale=weight_scale)

    config, m_rows, expert_ids_and_cumsum, sorted_token_ids, aligned_num_m_blocks, inv_perm, expert_ids = (
        moe_align_block_size(x[0], y[0], topk_ids, False, enable_act_and_quant_fusing=True)
    )
    deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_fused(x, y_interleave, out, m_rows, expert_ids_and_cumsum,
                                                   sorted_token_ids, aligned_num_m_blocks, config,
                                                   out_scale=out_scale, swiglu_limit=swiglu_limit)

    ### Reference
    if swiglu_limit > 0.0:
        gate_clamped = torch.clamp(ref_out[:, :(ref_out.shape[-1]//2)], max=swiglu_limit)
        up_clamped = torch.clamp(ref_out[:, (ref_out.shape[-1]//2):], min=-swiglu_limit, max=swiglu_limit)
        silu = gate_clamped * torch.sigmoid(gate_clamped) * up_clamped
    else:
        silu = ref_out[:, :(ref_out.shape[-1]//2)] * torch.sigmoid(ref_out[:, :(ref_out.shape[-1]//2)]) * ref_out[:, (ref_out.shape[-1]//2):]
    ref_quant_torch, ref_quant_scale_torch_ = quantize_fp4_torch(src_tensor=silu)
    ref_quant_scale_torch = preprocess_mxfp4_scales(scale=ref_quant_scale_torch_)

    ### Check
    diff = calc_diff(out, ref_quant_torch).item()
    diff_scale = calc_diff(out_scale, ref_quant_scale_torch).item()
    if diff >= 0.00001 or diff_scale >= 0.00001:
        print("ref_out:", ref_quant_torch)
        print("out:", out)
        print("ref_out_scale:", ref_quant_scale_torch)
        print("out_scale:", out_scale)
        torch.testing.assert_close(out, ref_quant_torch, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(out_scale, ref_quant_scale_torch, rtol=1e-3, atol=1e-4)
    assert diff < 0.00001, f'{m=}, {n=}, {k=}, {topk=}, {diff:.5f}'
    assert diff_scale < 0.00001, f'{m=}, {n=}, {k=}, {topk=}, {diff_scale:.5f}'
    print(f"Passed with acc_check. {diff=}, {diff_scale=}\n")

def test_m_grouped_gemm_fused_silu_and_mul_post_quant_loop(num_groups: int = None, m: int = None, n: int = None, k: int = None, topk: int = None) -> None:
    print("Running GroupedFused GEMM with enable_act_and_quant_fusing test...")
    if num_groups is not None and m is not None and n is not None and k is not None and topk is not None:
        print(f"Testing with num_groups={num_groups}, m={m}, n={n}, k={k}, topk={topk}")
        args = {"groups": num_groups, "m": m, "n": n, "k": k, "topk": topk, "distribution": "uniform", "swiglu_limit": 0.0}
        test_m_grouped_gemm_fused_silu_and_mul_post_quant(args)
    else:
        print("Running default test suite...")
        ### gemm1 shapes: n is 2 * intermediate_size (gate & up)
        for num_groups, topk, m in ((8, 8, 16), (8, 8, 256), (32, 4, 512)):
            for k, n in ((4096, 4096), (4096, 1024)):
                for swiglu_limit in (0.0, 10.0):
                    print(f"Testing with num_groups={num_groups}, m={m}, n={n}, k={k}, topk={topk}, {swiglu_limit=}")
                    args = {"groups": num_groups, "m": m, "n": n, "k": k, "topk": topk, "distribution": "uniform", "swiglu_limit": swiglu_limit}
                    test_m_grouped_gemm_fused_silu_and_mul_post_quant(args)
    print("Passed\n")

def test_m_grouped_gemm_nopad_silu_and_mul_post_quant(args) -> None:
    num_groups, m, n, k, distribution, swiglu_limit = args['groups'], args['m'], args['n'], args['k'], args['distribution'], args['swiglu_limit']
    bias = None

    m, x, y, m_indices, _, _ = construct_nopad_base(num_groups, m, k, n, torch.uint8, distribution, 1)
    out = torch.empty((m, (n // 4)), dtype=torch.uint8, device='cuda')
    out_scale = torch.empty((m, ceil_div((n // 4), 32)), dtype=torch.uint16, device='cuda')
    ### calculate ref_out in float32 dtype
    ref_out = torch.empty((m, n), device='cuda', dtype=torch.float32)
    for group_idx in range(num_groups):
        mask = (m_indices == group_idx)
        if mask.sum() == 0: continue

        a_q, a_s = x[0][mask], x[1].contiguous().view(torch.uint8)[mask]
        b_q, b_s = y[0][group_idx], y[1].contiguous().view(torch.uint8)[group_idx]

        a_dq = dequantize_fp4_torch(a_q, a_s).to(torch.float)
        b_dq = dequantize_fp4_torch(b_q, b_s).to(torch.float)

        ref_out[mask] = a_dq @ b_dq.t()
    ref_out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_out), ref_out)

    from deep_gemm import preprocess_mxfp4_weight_for_act_and_quant_fusing
    weight_scale = y[1].contiguous().view(torch.uint8)
    y_interleave = preprocess_mxfp4_weight_for_act_and_quant_fusing(weight=y[0], weight_scale=weight_scale)

    deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(x, y_interleave, bias, out, m_indices, out_scale=out_scale, swiglu_limit=swiglu_limit)

    ### Reference
    if swiglu_limit > 0.0:
        gate_clamped = torch.clamp(ref_out[:, :(ref_out.shape[-1]//2)], max=swiglu_limit)
        up_clamped = torch.clamp(ref_out[:, (ref_out.shape[-1]//2):], min=-swiglu_limit, max=swiglu_limit)
        silu = gate_clamped * torch.sigmoid(gate_clamped) * up_clamped
    else:
        silu = ref_out[:, :(ref_out.shape[-1]//2)] * torch.sigmoid(ref_out[:, :(ref_out.shape[-1]//2)]) * ref_out[:, (ref_out.shape[-1]//2):]
    ref_quant_torch, ref_quant_scale_torch_ = quantize_fp4_torch(src_tensor=silu)
    ref_quant_scale_torch = preprocess_mxfp4_scales(scale=ref_quant_scale_torch_)

    ### Check
    diff = calc_diff(out, ref_quant_torch).item()
    diff_scale = calc_diff(out_scale, ref_quant_scale_torch).item()
    if diff >= 0.00001 or diff_scale >= 0.00001:
        print("ref_out:", ref_quant_torch)
        print("out:", out)
        print("ref_out_scale:", ref_quant_scale_torch)
        print("out_scale:", out_scale)
        torch.testing.assert_close(out, ref_quant_torch, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(out_scale, ref_quant_scale_torch, rtol=1e-3, atol=1e-4)
    assert diff < 0.00001, f'{m=}, {k=}, {n=}, {diff:.5f}'
    assert diff_scale < 0.00001, f'{m=}, {k=}, {n=}, {diff_scale:.5f}'
    print(f"Passed with acc_check. {diff=}, {diff_scale=}\n")

def test_m_grouped_gemm_nopad_silu_and_mul_post_quant_loop(num_groups: int = None, m: int = None, n: int = None, k: int = None) -> None:
    print("Running GroupedNoPad GEMM with enable_act_and_quant_fusing test...")
    print("Running default test suite...")
    # DPSKV4FLASH(N4096K4096E256TOPK6) DPSKV4PRO(N6144K7168E384TOPK6)
    for num_groups, expected_m_per_group in ((4, 256), ):
        for k, n in ((4096, 4096), ):
            for swiglu_limit in (0.0, 10.0):
                print(f"Testing with num_groups={num_groups}, m={num_groups * expected_m_per_group}, n={n}, k={k}")
                args = {"groups": num_groups, "m": num_groups * expected_m_per_group, "n": n, "k": k, "distribution": "uniform", "swiglu_limit": swiglu_limit}
                test_m_grouped_gemm_nopad_silu_and_mul_post_quant(args)
    print("Passed\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process target function api test.")
    parser.add_argument('--func', default=None, choices=["DenseGemm", "GroupedNoPad", "GroupedMasked", "GroupedFused", "GroupedMaskedSiluAndMulPostQuant", "GroupedNoPadSiluAndMulPostQuant", "GroupedFusedSiluAndMulPostQuant"], required=False, help='target test func')
    parser.add_argument('--num_groups', type=int, help='Number of groups for Grouped GEMM')
    parser.add_argument('--m', type=int, help='M dimension')
    parser.add_argument('--n', type=int, help='N dimension')
    parser.add_argument('--k', type=int, help='K dimension')
    args = parser.parse_args()

    num_groups = args.num_groups
    m, n, k = args.m, args.n, args.k

    if args.func is not None:
        if args.func in ['GroupedNoPad']:
            test_m_grouped_gemm_nopad_loop(num_groups, m, n, k)

        if args.func in ['GroupedMasked']:
            test_m_grouped_gemm_masked_loop(num_groups, m, n, k)

        if args.func in ['DenseGemm']:
            test_gemm_loop(m, n, k)

        if args.func in ['GroupedFused']:
            test_m_grouped_gemm_fused_loop(m, n, k)

        if args.func in ['GroupedMaskedSiluAndMulPostQuant']:
            test_m_grouped_gemm_masked_silu_and_mul_post_quant_loop(num_groups, m, n, k)

        if args.func in ['GroupedNoPadSiluAndMulPostQuant']:
            test_m_grouped_gemm_nopad_silu_and_mul_post_quant_loop(num_groups, m, n, k)

        if args.func in ['GroupedFusedSiluAndMulPostQuant']:
            test_m_grouped_gemm_fused_silu_and_mul_post_quant_loop(num_groups, m, n, k)
    else:
        test_m_grouped_gemm_nopad_loop(num_groups, m, n, k)
        test_m_grouped_gemm_masked_loop(num_groups, m, n, k)
        test_gemm_loop(m, n, k)
        test_m_grouped_gemm_fused_loop(m, n, k)
