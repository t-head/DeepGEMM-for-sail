import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm.testing.bench import *
from utils import test_mqa_logits, test_paged_mqa_logits, set_acc_check
from utils import test_mqa_logits, test_paged_mqa_logits, test_sparse_mqa_logits, set_acc_check, parse_deepgemm_string_re
from deep_gemm.jit_kernels.utils import is_ppu1v5_device


def test_ks_ke():
    # Single INT8 case: seq_len_q=1 with ks != 0, manually constructed (not via test_mqa_logits).
    # Verifies the kernel correctly offsets into KV when the valid window does not start at 0.
    # Only q[0, 0, 0]=1 and k[4, 0]=5 are non-zero, so logits[0, 4] = relu(1*5) * 1 * 1 = 5.
    print('Testing INT8 MQA Logits (seq_len_q=1, ks!=0):')
    sq, nh, hd = 1, 32, 128
    skv = 6
    q = torch.zeros((sq, nh, hd), device='cuda', dtype=torch.int8)
    k = torch.zeros((skv, hd), device='cuda', dtype=torch.int8)
    k_scale = torch.ones(skv, device='cuda', dtype=torch.float32)
    weights = torch.ones((sq, nh), device='cuda', dtype=torch.float32)
    q[0, 0, 0] = 1
    k[4, 0] = 5
    # ks != 0: only KV token 4 is within [ks, ke)
    ks = torch.tensor([4], device='cuda', dtype=torch.int32)
    ke = torch.tensor([5], device='cuda', dtype=torch.int32)
    # Reference: logits = (relu(q·k) * weights).sum(heads) * k_scale
    scores = torch.matmul(q.float(), k.float().T)            # [1, 32, 6]
    ref = (scores.relu() * weights[:, :, None]).sum(dim=1)  # [1, 6]
    ref = ref * k_scale[None, :]
    actual = deep_gemm.int8_mqa_logits(q, (k, k_scale), weights, ks, ke, clean_logits=False)
    torch.testing.assert_close(actual[0, 4].float(), ref[0, 4].float(), rtol=1e-3, atol=1e-3)
    print(f'  INT8 ks!=0 case: expected[0,4]={ref[0, 4].item():.4f}, actual[0,4]={actual[0, 4].item():.4f}')
    print("Passed\n")


def test_per_layer_cache_view():
    num_kv_blocks, block_kv, num_heads, head_dim = 8, 64, 32, 128
    num_layers = 4
    max_context_len = num_kv_blocks * block_kv

    v = (torch.arange(max_context_len) % 7 + 1).float()
    s = 0.001 * (1 + torch.arange(max_context_len).float())
    k = torch.full((max_context_len, head_dim), 0).to(torch.int8)
    k[:] = v.to(torch.int8).unsqueeze(1)
    page = torch.cat([k.view(torch.uint8).reshape(num_kv_blocks, block_kv * head_dim),
                      torch.from_numpy(s.numpy().astype('<f4')).view(torch.uint8).reshape(num_kv_blocks, block_kv * 4)],
                     dim=1).cuda()

    q = torch.ones(1, 1, num_heads, head_dim, device='cuda').to(torch.int8)
    weights = torch.full((1, num_heads), 1.0 / num_heads, device='cuda')
    block_table = torch.arange(num_kv_blocks, dtype=torch.int32, device='cuda').unsqueeze(0)
    context_lens = torch.full((1, 1), max_context_len, dtype=torch.int32, device='cuda')
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_kv, deep_gemm.get_num_sms(),
        metadata_extra=(1, num_heads, head_dim, 1))

    def check(fused_kv_cache):
        logits = deep_gemm.int8_paged_mqa_logits(
            q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
            max_context_len, clean_logits=False, logits_dtype=torch.float32)
        return int((((logits.float().cpu()[0] / (head_dim * v * s)) - 1).abs() < 0.005).sum())

    fused_kv_cache = torch.zeros(num_kv_blocks, block_kv * (head_dim + 4),
                                 dtype=torch.uint8, device='cuda')
    fused_kv_cache.copy_(page)
    assert check(fused_kv_cache.view(num_kv_blocks, block_kv, 1, head_dim + 4)) == max_context_len

    fused_kv_cache = torch.zeros(num_kv_blocks, num_layers, block_kv * (head_dim + 4),
                                 dtype=torch.uint8, device='cuda')
    fused_kv_cache[:, 3, :] = page
    fused_kv_cache = fused_kv_cache[:, 3].view(num_kv_blocks, block_kv, 1, head_dim + 4)
    assert fused_kv_cache.view(num_kv_blocks, block_kv, head_dim + 4).storage_offset() \
        == 3 * block_kv * (head_dim + 4)
    assert check(fused_kv_cache) == max_context_len



def test_mqa_logits_loop():
    print('Testing MQA Logits:')
    data_types = ['int8']
    if is_ppu1v5_device():
        data_types.extend(['fp8', 'fp4'])
    for data_type in data_types:
        for seq_len in (2048, 4096):
            for num_heads, head_dim in [(32, 128), (64, 128)]:
                if data_type == 'fp4' and num_heads != 64:
                    continue
                for seq_len_kv in (4096, 8192, 16384, 32768, 65536, 131072):
                    do_check = seq_len_kv < 32768
                    case = f'MqaLogits,data_type:{data_type},seq_len_q:{seq_len},seq_len_kv:{seq_len_kv},num_heads:{num_heads},head_dim:{head_dim}'
                    set_acc_check(do_check)
                    test_mqa_logits(parse_deepgemm_string_re(case))
                    if do_check:
                        test_mqa_logits(parse_deepgemm_string_re(case + ',logits_dtype:bf16'))
    cases = [
        # BF16, compressed logits, and BF16 weights/logits.
        'MqaLogits,data_type:bf16,seq_len_q:8191,seq_len_kv:8191,num_heads:64,head_dim:128',
        'MqaLogits,data_type:int8,seq_len_q:4096,seq_len_kv:8192,num_heads:64,head_dim:128,compressed_logits:1',
        'MqaLogits,data_type:int8,seq_len_q:1024,seq_len_kv:4096,num_heads:32,head_dim:128,logits_dtype:bf16,weights_dtype:bf16',
    ]
    for case in cases:
        set_acc_check(True)
        test_mqa_logits(parse_deepgemm_string_re(case))
    print("Passed\n")


def test_paged_mqa_logits_loop():
    print('Testing Paged MQA Logits:')
    data_types = ['int8']
    if is_ppu1v5_device():
        data_types.extend(['fp8', 'fp4'])
    for data_type in data_types:
        for batch_size, next_n in [(1, 1), (64, 1), (64, 2), (128, 1)]:
            for num_heads, head_dim in [(32, 128), (64, 128)]:
                if next_n == 2 and num_heads == 32:
                    continue
                if data_type == 'fp4' and num_heads != 64:
                    continue
                for avg_kv in (8192, 32768):
                    do_check = avg_kv < 32768
                    case = f'PagedMqaLogits,data_type:{data_type},batch_size:{batch_size},next_n:{next_n},avg_context_len:{avg_kv},num_heads:{num_heads},head_dim:{head_dim}'
                    set_acc_check(do_check)
                    test_paged_mqa_logits(parse_deepgemm_string_re(case))
                    if do_check:
                        test_paged_mqa_logits(parse_deepgemm_string_re(case + ',logits_dtype:bf16'))
    cases = [
        # Empty contexts, a large batch, BF16 weights/logits, and multiple next tokens.
        'PagedMqaLogits,data_type:int8,batch_size:4,next_n:1,num_heads:64,head_dim:128,distribution:[20,10,0,0]',
        'PagedMqaLogits,data_type:int8,batch_size:16,next_n:1,num_heads:32,head_dim:128,distribution:[4090,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1]',
        'PagedMqaLogits,data_type:bf16,batch_size:1119,next_n:1,num_heads:64,head_dim:128,avg_context_len:1087',
        'PagedMqaLogits,data_type:int8,batch_size:64,next_n:1,num_heads:64,head_dim:128,avg_context_len:8192,logits_dtype:bf16,weights_dtype:bf16',
        'PagedMqaLogits,data_type:int8,batch_size:64,next_n:4,num_heads:64,head_dim:128,avg_context_len:8192',
    ]
    if is_ppu1v5_device():
        cases.append('PagedMqaLogits,data_type:fp4,batch_size:64,next_n:6,num_heads:64,head_dim:128,avg_context_len:8192,logits_dtype:bf16')
    for case in cases:
        set_acc_check(True)
        test_paged_mqa_logits(parse_deepgemm_string_re(case))
    # CUDA graph replay with changing context lengths.
    case = 'PagedMqaLogits,data_type:int8,batch_size:8,next_n:1,num_heads:32,head_dim:128,pre_distribution:[4090,0,1,0,1],distribution:[4090,0,1,0,1,100,200,300]'
    set_acc_check(False)
    test_paged_mqa_logits(parse_deepgemm_string_re(case))
    print("Passed\n")


def test_mqa_avg_logits_loop():
    if not is_ppu1v5_device():
        print('Skipping avg MQA logits: requires PPU1.5')
        return
    print('Testing Avg MQA Logits:')
    # Prefill covers all q_scale modes, both output dtypes, and an odd Q tail.
    prefill_cases = [
        'MqaAvgLogits,data_type:fp8,seq_len_q:129,seq_len_kv:1024,num_heads:4,head_dim:128,q_scale:0,logits_dtype:fp32',
        'MqaAvgLogits,data_type:fp8,seq_len_q:256,seq_len_kv:1024,num_heads:4,head_dim:128,q_scale:1,logits_dtype:bf16',
        'MqaAvgLogits,data_type:fp8,seq_len_q:256,seq_len_kv:1024,num_heads:4,head_dim:128,q_scale:2,logits_dtype:fp32',
    ]
    # Paged decode covers the four-head tile, empty contexts, and multiple next tokens.
    paged_cases = [
        'PagedMqaAvgLogits,data_type:fp8,batch_size:4,next_n:1,num_heads:4,head_dim:128,distribution:[0,1,127,1024],logits_dtype:bf16',
        'PagedMqaAvgLogits,data_type:fp8,batch_size:8,next_n:2,num_heads:4,head_dim:128,avg_context_len:1024,logits_dtype:fp32',
        'PagedMqaAvgLogits,data_type:fp8,batch_size:8,next_n:5,num_heads:4,head_dim:128,avg_context_len:1024,logits_dtype:bf16',
    ]
    set_acc_check(True)
    for case in prefill_cases:
        print(case)
        test_mqa_logits(parse_deepgemm_string_re(case))
    for case in paged_cases:
        print(case)
        test_paged_mqa_logits(parse_deepgemm_string_re(case))
    print("Passed\n")


def test_sparse_mqa_logits_loop():
    if not is_ppu1v5_device():
        print('Skipping sparse MQA logits: requires PPU1.5')
        return
    print('Testing Sparse MQA Logits:')
    cases = [
        # Prefill: both block sizes, aligned/unaligned starts, odd Q tails, and empty selections.
        'SparseMqaLogits,data_type:fp4,seq_len_q:512,seq_len_kv:1024,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:256,use_unaligned_ks:0,check_metadata:1',
        'SparseMqaLogits,data_type:fp4,seq_len_q:512,seq_len_kv:1024,num_heads:32,head_dim:128,sparse_block_kv:16,num_max_sparse_blocks:256,use_unaligned_ks:0,check_metadata:1',
        'SparseMqaLogits,data_type:fp4,seq_len_q:9,seq_len_kv:639,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:128,use_unaligned_ks:1,check_metadata:1',
        'SparseMqaLogits,data_type:fp4,seq_len_q:9,seq_len_kv:639,num_heads:32,head_dim:128,sparse_block_kv:16,num_max_sparse_blocks:128,use_unaligned_ks:1,check_metadata:1',
        'SparseMqaLogits,data_type:fp4,seq_len_q:2,seq_len_kv:0,num_heads:32,head_dim:128,sparse_block_kv:16,num_max_sparse_blocks:4,use_unaligned_ks:0,check_metadata:1',
        # Paged decode: request boundaries, multi-entry schedules, and partial splits.
        'PagedSparseMqaLogits,data_type:fp4,seq_len_q:313,seq_len_kv:1023,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:128,check_metadata:1',
        'PagedSparseMqaLogits,data_type:fp4,seq_len_q:512,seq_len_kv:65536,num_heads:32,head_dim:128,sparse_block_kv:16,num_max_sparse_blocks:128,check_metadata:1',
        'PagedSparseMqaLogits,data_type:fp4,seq_len_q:1,seq_len_kv:1025,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:256,check_metadata:1',
        'PagedSparseMqaLogits,data_type:fp4,seq_len_q:2,seq_len_kv:0,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:4,check_metadata:1',
    ]
    set_acc_check(True)
    for case in cases:
        print(case)
        test_sparse_mqa_logits(parse_deepgemm_string_re(case))
    print("Passed\n")


def test_nvtx():
    print('Testing nvtx dump:')
    import torch.cuda.nvtx as nvtx
    nvtx.range_push("paged_mqa_logits")
    for batch_size in (1, 64):
        for num_heads in (32, 64):
            case = f'PagedMqaLogits,data_type:int8,batch_size:{batch_size},next_n:1,avg_context_len:8192,num_heads:{num_heads},head_dim:128'
            set_acc_check(False)
            test_paged_mqa_logits(parse_deepgemm_string_re(case))
    nvtx.range_pop()
    print("Passed\n")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    # test_nvtx()

    test_ks_ke()
    test_per_layer_cache_view()
    test_mqa_logits_loop()
    test_paged_mqa_logits_loop()
    test_mqa_avg_logits_loop()
    test_sparse_mqa_logits_loop()
