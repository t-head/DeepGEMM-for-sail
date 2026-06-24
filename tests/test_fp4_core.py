import torch
import torch.nn.functional as F
import time
from typing import Tuple
import deep_gemm
from deep_gemm import calc_diff, ceil_div, preprocess_mxfp4_scales, moe_align_block_size
from utils import construct_group_m_list, get_ref_backend, find_next_power_of_2, check_signal
import argparse

# torch.cuda.manual_seed(42)

def right_shift_unsigned(x, shift):
    return (x >> shift) & ((1 << (32 - shift)) - 1)

def quantize_fp4_torch(src_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert src_tensor.dtype in {torch.float32, torch.bfloat16, torch.float16}
    device = src_tensor.device
    *batch_dims, N, K = src_tensor.shape
    assert K % 2 == 0

    src = src_tensor.to(torch.float32)
    padded_K = ((K + 31) // 32) * 32
    pad_amount = padded_K - K
    padded_src = F.pad(src, (0, pad_amount))
    valid_mask = F.pad(torch.ones_like(src, dtype=torch.bool), (0, pad_amount))

    abs_src = torch.abs(padded_src)
    abs_src = torch.where(valid_mask, abs_src, torch.tensor(-1.0, device=device))
    abs_groups = abs_src.view(*batch_dims, N, padded_K // 32, 32)
    max_val, _ = abs_groups.max(dim=-1, keepdim=True)

    dequant_scale = max_val / 6.0
    ds_int = dequant_scale.view(torch.int32)
    ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
    dequant_scale_rounded = ds_int_rounded.view(torch.float32)
    quant_scale = torch.where(dequant_scale_rounded == 0, torch.tensor(0.0, device=device), 1.0 / dequant_scale_rounded)

    padded_src_groups = padded_src.view(*batch_dims, N, padded_K // 32, 32)
    quant_tensor = padded_src_groups * quant_scale
    quant_tensor = quant_tensor.view(*batch_dims, N, padded_K)[..., :K]

    q_int = quant_tensor.contiguous().view(torch.int32)
    signs = q_int & 0x80000000
    exponents = right_shift_unsigned(q_int, 23) & 0xFF
    mantissas = q_int & 0x7FFFFF

    E8_BIAS, E2_BIAS = 127, 1
    mantissas = torch.where(exponents < E8_BIAS, (0x400000 | right_shift_unsigned(mantissas, 1)) >> (E8_BIAS - exponents - 1), mantissas)
    exponents = torch.maximum(exponents, torch.tensor(E8_BIAS - E2_BIAS, device=device)) - (E8_BIAS - E2_BIAS)

    e2m1_tmp = right_shift_unsigned(((exponents << 2) | right_shift_unsigned(mantissas, 21)) + 1, 1)
    e2m1_tmp = torch.minimum(e2m1_tmp, torch.tensor(0x7, device=device))
    e2m1_value = (right_shift_unsigned(signs, 28) | e2m1_tmp).to(torch.uint8)

    e2m1_value = e2m1_value.view(*batch_dims, N, K // 2, 2)
    packed_tensor = e2m1_value[..., 0] | (e2m1_value[..., 1] << 4)
    scale_uint8 = (ds_int_rounded.squeeze(-1) >> 23).to(torch.uint8)

    return packed_tensor, scale_uint8

def dequantize_fp4_torch(quant_tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    assert quant_tensor.dtype == torch.uint8 and scale.dtype == torch.uint8
    assert target_dtype in {torch.float32, torch.bfloat16, torch.float16}

    device = quant_tensor.device
    *batch_dims, N, packed_K = quant_tensor.shape
    K = packed_K * 2

    quant_int = quant_tensor.to(torch.int32)
    evens = quant_int & 0xF
    odds = (quant_int >> 4) & 0xF

    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    lookup_table = torch.tensor(vals + [-v for v in vals], dtype=torch.float32, device=device)
    fp32_tensor = torch.stack([lookup_table[evens], lookup_table[odds]], dim=-1).view(*batch_dims, N, K)

    dq_scale = (scale.to(torch.int32) << 23).view(torch.float32)
    padded_K = dq_scale.shape[-1] * 32
    padded_tensor = F.pad(fp32_tensor, (0, padded_K - K))
    padded_tensor = padded_tensor.view(*batch_dims, N, dq_scale.shape[-1], 32)
    out_padded = (padded_tensor * dq_scale.unsqueeze(-1)).view(*batch_dims, N, padded_K)

    return out_padded[..., :K].to(target_dtype)

def test_gemm(args):
    m, n, k = args['m'], args['n'], args['k']
    A = torch.randn(m, k, dtype=torch.bfloat16, device='cuda').contiguous()
    B = torch.randn(n, k, dtype=torch.bfloat16, device='cuda').contiguous()
    x = quantize_fp4_torch(A)
    y = quantize_fp4_torch(B)
    a_dequant = dequantize_fp4_torch(x[0], x[1]).cuda()
    b_dequant = dequantize_fp4_torch(y[0], y[1]).cuda()
    bias = torch.randn(1, n, dtype=torch.float32, device='cuda') if args.get("with_bias", True) else None
    out = torch.zeros(m, n, dtype=torch.bfloat16, device='cuda')
    x_scale = preprocess_mxfp4_scales(scale=x[1])
    y_scale = preprocess_mxfp4_scales(scale=y[1])
    x = x[0], x_scale
    y = y[0], y_scale

    deep_gemm.gemm_fp4_fp4_bf16_nt(x, y, bias, out)

    ref_out = torch.mm(a_dequant, b_dequant.T)
    ref_out = (ref_out + bias).to(torch.bfloat16) if bias is not None else ref_out.to(torch.bfloat16)
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
    x, y, m_indices, bias, out, ref_out = construct_grouped(num_groups, m, k, n, distribution, 1, with_bias)

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
    x, y, bias, masked_m, out, ref_out, signal, max_m = construct_grouped_masked(num_groups, m, expected_m_per_group, k, n, distribution, enable_sbo_overlap, with_bias=with_bias)
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

def construct_grouped(num_groups: int, m: int, k: int, n: int, distribution: str, alignment: int, with_bias: bool = True):
    group_ms = construct_group_m_list(distribution, num_groups, m)
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    bias = torch.randn((num_groups, n), device='cuda', dtype=torch.float) if with_bias else None

    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
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
        ref_out[start:aligned_end] = ref_out[start:aligned_end] + bias[i] if bias is not None else ref_out[start:aligned_end]
        start = aligned_end

    x_fp4 = quantize_fp4_torch(x.to(torch.bfloat16).to('cuda'))
    x_fp4_scale = preprocess_mxfp4_scales(scale=x_fp4[1])
    y_fp4 = (torch.empty((num_groups, n, int(k / 2)), device='cuda', dtype=torch.uint8), torch.empty((num_groups, n, int(k / 32)), device='cuda', dtype=torch.uint8))
    y_scale = []
    for i in range(num_groups):
        y_fp4[0][i], y_fp4[1][i] = quantize_fp4_torch(y[i].to(torch.bfloat16))
        # y_scale.append(uint8_padding(y_fp4[1][i]))
        y_scale.append(y_fp4[1][i])
    y_fp4_scale = torch.stack(y_scale, dim=0)
    y_fp4_scale = preprocess_mxfp4_scales(scale=y_fp4_scale)
    return (x_fp4[0].to("cuda"), x_fp4_scale.to("cuda")), (y_fp4[0].to("cuda"), y_fp4_scale.to("cuda")), m_indices, bias, out, ref_out.to('cuda').to(torch.bfloat16)


def construct_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int, distribution: str,
                             enable_sbo_overlap: bool = False, with_bias: bool = True, enable_silu_and_mul_quant_fusing: bool = False):
    tensor_device = 'cuda' if get_ref_backend() == "device" else 'cpu'
    # Construct mask
    list_m =  construct_group_m_list(distribution, num_groups, max_m, is_mask=True, em=expected_m_per_group)
    masked_m = torch.tensor(list_m, device=tensor_device, dtype=torch.int)
    max_m = max(128, find_next_power_of_2(list_m))
    assert masked_m.amax().item() <= max_m, f"max masked_m={masked_m.amax().item()}, allowed max_m={max_m}"
    
    x = torch.randn((num_groups, max_m, k), device=tensor_device, dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device=tensor_device, dtype=torch.bfloat16)
    if enable_silu_and_mul_quant_fusing:
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

    if enable_silu_and_mul_quant_fusing:
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
    x, y, bias, masked_m, out, out_scale, ref_out, signal, max_m = construct_grouped_masked(num_groups, m, expected_m_per_group, k, n, distribution, enable_sbo_overlap, with_bias=False, enable_silu_and_mul_quant_fusing=True)

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
    print("Running GroupedMasked GEMM with enable_silu_and_mul_quant_fusing test...")
    print("Running default test suite...")
    for num_groups, expected_m_per_group in ((8, 256), (8, 16)):
        for k, n in ((4096, 4096), (4096, 4032)):
            for swiglu_limit in (0.0, 10.0):
                print(f"Testing with num_groups={num_groups}, m={num_groups * expected_m_per_group}, n={n}, k={k}")
                args = {"groups": num_groups, "m": num_groups * expected_m_per_group, "n": n, "k": k, "distribution": "uniform", "swiglu_limit": swiglu_limit}
                test_m_grouped_gemm_masked_silu_and_mul_post_quant(args)
    print("Passed\n")
        
def construct_grouped_fused(num_groups: int, m: int, k: int, n: int, topk: int, distribution: str, alignment: int):
    ### 0. prepare input
    score = torch.randn((m, num_groups), device='cuda', dtype=torch.float32)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)

    ### the shape of fusedMoe is same with fusedNoPad
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
    k_ = x_fp4.shape[-1]
    sfk_ = x_fp4_tupple[1].shape[-1]
    x_fp4_ = x_fp4.view(m, -1, k_).repeat(1, topk, 1).reshape(-1, k_)
    x_scale_fp4_ = x_fp4_tupple[1].view(m, -1, sfk_).repeat(1, topk, 1).reshape(-1, sfk_)
    topk_ids_ = topk_ids.view(-1)
    for i in range(y_fp4.shape[0]):
        mask = (topk_ids_ == i)
        if mask.sum():
            ref_out_group = dequantize_fp4_torch(x_fp4_[mask], x_scale_fp4_[mask]) @ dequantize_fp4_torch(y_fp4[i], y_scale_fp4_[i]).transpose(0, 1)
            ref_out.append(ref_out_group)
    ref_out = torch.concat(ref_out, dim=0)

    ### 3. return
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process target function api test.")
    parser.add_argument('--func', default=None, choices=["DenseGemm", "GroupedNoPad", "GroupedMasked", "GroupedFused", "GroupedMaskedSiluAndMulPostQuant"], required=False, help='target test func')
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
    else:
        test_m_grouped_gemm_nopad_loop(num_groups, m, n, k)
        test_m_grouped_gemm_masked_loop(num_groups, m, n, k)
        test_gemm_loop(m, n, k)
        test_m_grouped_gemm_fused_loop(m, n, k)