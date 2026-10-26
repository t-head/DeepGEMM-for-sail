import random
import torch

import deep_gemm
from utils import test_mqa_logits, test_paged_mqa_logits, test_sparse_mqa_logits, set_acc_check, parse_deepgemm_string_re
from utils import is_ppu1v5_device

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
        'PagedMqaLogits,data_type:int8,batch_size:4,next_n:1,num_heads:64,head_dim:128,distribution:[20,10,0,0],page_kv:32',
        'PagedMqaLogits,data_type:int8,batch_size:16,next_n:1,num_heads:32,head_dim:128,distribution:[4090,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1]',
        'PagedMqaLogits,data_type:bf16,batch_size:1119,next_n:1,num_heads:64,head_dim:128,avg_context_len:1087,page_kv:128',
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
    # FP4 prefill: cross both output dtypes with all three Q-scale modes.
    # Q=129 leaves a partial Q tile. KV=258 covers short KV, including
    # the third KV128 block, for both dense and compressed output.
    # KV=1025 exercises compressed output;
    # KV=8193 enables default KV balancing, with every window > 3/4 of KV.
    prefill_cases += [
        f'MqaAvgLogits,data_type:fp4,seq_len_q:129,seq_len_kv:{seq_len_kv},'
        f'num_heads:4,head_dim:128,q_scale:{q_scale},logits_dtype:{logits_dtype},'
        f'compressed_logits:{compressed_logits}'
        for logits_dtype in ('fp32', 'bf16')
        for q_scale in (0, 1, 2)
        for seq_len_kv, compressed_logits in ((258, 0), (258, 1), (513, 0), (1025, 1), (8193, 0))
    ]
    # On 39 CUs, Q4096/KV4096 selects four MMA M blocks with G=2 for
    # both output dtypes. BF16 also enables warp interleaving, exercising
    # the synchronization between groups rather than just a single group.
    prefill_cases += [
        f'MqaAvgLogits,data_type:fp4,seq_len_q:4096,seq_len_kv:4096,'
        f'num_heads:4,head_dim:128,q_scale:{q_scale},logits_dtype:{logits_dtype}'
        for logits_dtype in ('fp32', 'bf16')
        for q_scale in (0, 1, 2)
    ]
    # Paged decode covers the four-head tile, empty contexts, and multiple next tokens.
    paged_cases = [
        'PagedMqaAvgLogits,data_type:fp8,batch_size:4,next_n:1,num_heads:4,head_dim:128,distribution:[0,1,127,1024],logits_dtype:bf16,page_kv:32',
        'PagedMqaAvgLogits,data_type:fp8,batch_size:8,next_n:2,num_heads:4,head_dim:128,avg_context_len:1024,logits_dtype:fp32',
        'PagedMqaAvgLogits,data_type:fp8,batch_size:8,next_n:5,num_heads:4,head_dim:128,avg_context_len:1024,logits_dtype:bf16,page_kv:128',
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


def test_sparse_workspace():
    """Check sparse workspace lifetime across graph replay and larger eager calls.

    Packed FP4 inputs use an exact analytic logits reference. Metadata is also
    checked against the existing sparse schedule reference on two streams.
    """
    if not is_ppu1v5_device():
        print('Skipping sparse workspace: requires PPU1.5')
        return
    from utils import decode_sparse_metadata, ref_paged_sparse_metadata, verify_paged_sparse_metadata

    PAGE_KV = 128
    BLOCK_KV = 8
    TOPK_BLOCKS = 32
    NUM_PAGES = 32

    class PagedCase:
        def __init__(self, rows, phase=0):
            self.rows = rows
            self.q = (
                torch.full((rows, 1, 32, 64), 0x22, dtype=torch.int8, device="cuda"),
                torch.full((rows, 1, 32), 0x7F7F7F7F, dtype=torch.int32, device="cuda"),
            )
            self.weights = torch.full((rows, 32), 1 / 32, dtype=torch.bfloat16, device="cuda")
            # Fused page: all packed K payloads followed by all scale words.
            # E2M1 code 2 = 1, code 4 = 2; E8M0 byte 127 = scale 1.
            storage = torch.empty((NUM_PAGES, PAGE_KV * 68), dtype=torch.uint8, device="cuda")
            for page in range(NUM_PAGES):
                storage[page, : PAGE_KV * 64].fill_(0x22 if page % 2 == 0 else 0x44)
            storage[:, PAGE_KV * 64 :].fill_(0x7F)
            self.kv = storage.view(NUM_PAGES, PAGE_KV, 1, 68)
            self.lens = torch.empty(rows, dtype=torch.int32, device="cuda")
            self.table = torch.empty((rows, 9), dtype=torch.int32, device="cuda")
            self.blocks = torch.empty((rows, TOPK_BLOCKS), dtype=torch.int32, device="cuda")
            self.requests = torch.tensor([i // 3 for i in range(rows)], dtype=torch.int32, device="cuda")
            self.update(phase)

        def update(self, phase):
            boundaries = [127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025]
            lens = [boundaries[(i + phase) % len(boundaries)] for i in range(self.rows)]
            # Same request => identical table rows, including a pair crossing a page.
            tables = [[(j * 7 + (i // 3) * 3 + phase) % NUM_PAGES for j in range(9)]
                      for i in range(self.rows)]
            blocks, counts, expected, valid = [], [], [], []
            for length, table in zip(lens, tables):
                available = (length + BLOCK_KV - 1) // BLOCK_KV
                count = min(available, TOPK_BLOCKS)
                selected = list(range(count))
                if available > count:
                    selected[-1] = available - 1
                blocks.append(selected + [0] * (TOPK_BLOCKS - count))
                counts.append(count)
                row_expected, row_valid = [], []
                for slot in range(TOPK_BLOCKS):
                    block = blocks[-1][slot]
                    physical_page = table[block // (PAGE_KV // BLOCK_KV)]
                    for token in range(BLOCK_KV):
                        row_expected.append(128 * (1 + physical_page % 2))
                        row_valid.append(slot < count and block * BLOCK_KV + token < length)
                expected.append(row_expected)
                valid.append(row_valid)
            self.lens.copy_(torch.tensor(lens, dtype=torch.int32))
            self.table.copy_(torch.tensor(tables, dtype=torch.int32))
            self.blocks.copy_(torch.tensor(blocks, dtype=torch.int32))
            self.counts = counts
            self.expected = torch.tensor(expected, dtype=torch.bfloat16)
            self.valid = torch.tensor(valid, dtype=torch.bool)

        def run(self):
            metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
                self.lens, self.table, self.requests, PAGE_KV,
                self.blocks, torch.int8, BLOCK_KV,
            )
            logits = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
                self.q, self.kv, self.weights, metadata, TOPK_BLOCKS, BLOCK_KV,
            )
            return metadata, logits

        def check(self, outputs):
            metadata, logits = outputs
            ref = ref_paged_sparse_metadata(
                self.blocks, self.counts, self.lens, self.table, self.requests,
                PAGE_KV, BLOCK_KV, TOPK_BLOCKS,
            )
            verify_paged_sparse_metadata(
                decode_sparse_metadata(metadata, BLOCK_KV), ref, self.rows, self.requests,
            )
            actual = logits[:, : TOPK_BLOCKS * BLOCK_KV].cpu()
            torch.testing.assert_close(actual[self.valid], self.expected[self.valid], rtol=0, atol=0)


    def capture(case, stream):
        with torch.cuda.stream(stream):
            for _ in range(3):
                case.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = case.run()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        case.check(outputs)
        return graph, outputs

    print("device:", torch.cuda.get_device_name(), "DeepGemm:", deep_gemm.__file__, flush=True)
    torch.cuda.init()
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(first):
        small_a = PagedCase(9)
    with torch.cuda.stream(second):
        small_b = PagedCase(13, phase=3)
    first.synchronize()
    second.synchronize()
    graph_a, output_a = capture(small_a, first)
    graph_b, output_b = capture(small_b, second)
    print("PASS: capture + exact logits/metadata reference on two streams", flush=True)

    # 2048 query rows exceed either captured case's old workspace allocation.
    # Release the eager intermediates and keep allocator churn alive through replay.
    with torch.cuda.stream(first):
        large = PagedCase(2048, phase=5)
        large_outputs = large.run()
    first.synchronize()
    large.check(large_outputs)
    del large_outputs, large
    torch.cuda.synchronize()
    churn = [torch.full((384 + rows * 8,), 0xA5, dtype=torch.uint8, device="cuda")
             for rows in (9, 13, 512, 2048)]
    torch.cuda.synchronize()
    print("PASS: 2048-row eager call after small captures", flush=True)

    for phase in range(20):
        with torch.cuda.stream(first):
            small_a.update(phase)
            graph_a.replay()
        with torch.cuda.stream(second):
            small_b.update(phase + 3)
            graph_b.replay()
        first.synchronize()
        second.synchronize()
        small_a.check(output_a)
        small_b.check(output_b)
    assert len(churn) == 4
    print("PASS: 40 replays, changing lengths/page mappings, exact metadata + logits", flush=True)

    cold_stream = torch.cuda.Stream()
    with torch.cuda.stream(cold_stream):
        cold_case = PagedCase(3)
    cold_stream.synchronize()
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=cold_stream):
            cold_case.run()
    except RuntimeError as error:
        assert "Warm up sparse MQA metadata" in str(error), str(error)
    else:
        raise AssertionError("First workspace allocation during capture must be rejected")
    graph_c, output_c = capture(cold_case, cold_stream)
    cold_case.check(output_c)
    print("PASS: cold capture rejected; warmup then capture/replay succeeds", flush=True)



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
    test_sparse_workspace()
