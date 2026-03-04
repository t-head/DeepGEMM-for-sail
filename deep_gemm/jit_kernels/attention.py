import math
import torch
from functools import lru_cache
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout, get_extra_info, get_paged_mqa_logits_tb_per_sm

def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment

# C++ code templates
includes = ('"../deep_gemm/ppu_mqa_logits.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
using ElementQK = {ElementQK}; //cutlass::bfloat16_t or cutlass::float_e4m3_t
using ElementAcc = {ElementAcc};
constexpr auto kNumHeads = {kNumHeads};
constexpr auto kHeadDim = {kHeadDim};
constexpr auto BLOCK_QH = {BLOCK_QH};
constexpr auto BLOCK_KV = {BLOCK_KV};
constexpr auto WARP_QH = {WARP_QH};
constexpr auto WARP_KV = {WARP_KV};
constexpr auto kNumQStages = {kNumQStages};
constexpr auto kNumKVStages = {kNumKVStages};

// Make a templated GEMM
using atten_t = Attention<ElementQK, ElementAcc, kNumHeads, kHeadDim, BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages>;

// Launch kernel
atten_t::run((const ElementQK*)q, (const ElementQK*)k, k_scales, weights, (uint32_t*)cu_seq_len_k_start, (uint32_t*)cu_seq_len_k_end, logits,
             seq_len_q, seq_len_k, aligned_seq_len_kv, stream, num_sms);
"""

def mqa_logits_common(q: torch.Tensor,
                      k: torch.Tensor, k_scales: torch.Tensor,
                      weights: torch.Tensor,
                      cu_seq_len_k_start: torch.Tensor,
                      cu_seq_len_k_end: torch.Tensor,
                      clean_logits: bool = True):
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_k, head_dim_ = k.shape
    seq_len_, num_heads_ = weights.shape

    assert(seq_len_q == seq_len_)
    assert(num_heads == num_heads_ and head_dim == head_dim_)
    assert(cu_seq_len_k_start.size(0) == seq_len_q)
    assert(cu_seq_len_k_end.size(0) == seq_len_q)

    assert(q.is_contiguous() and k.is_contiguous())
    assert(weights.is_contiguous())
    assert(cu_seq_len_k_start.is_contiguous())
    assert(cu_seq_len_k_end.is_contiguous())

    assert(q.dtype == torch.float8_e4m3fn or q.dtype == torch.bfloat16 or q.dtype == torch.int8)
    assert(k.dtype == torch.float8_e4m3fn or k.dtype == torch.bfloat16 or k.dtype == torch.int8)
    assert(q.dtype == k.dtype)

    assert(weights.dtype == torch.float32)
    assert(cu_seq_len_k_start.dtype == torch.int32)
    assert(cu_seq_len_k_end.dtype == torch.int32)

    if q.dtype != torch.bfloat16:
        seq_len_kv_ = k_scales.shape[0]
        assert(seq_len_k == seq_len_kv_)
        assert(k_scales.is_contiguous())
        assert(k_scales.dtype == torch.float32)

    debug = False
    if debug:
        print("cu_seq_len_k_start = ", cu_seq_len_k_start)
        print("cu_seq_len_k_end = ", cu_seq_len_k_end)
        print("k_scales = ", k_scales)
        q_fp32 = q.to(torch.float32)
        k_fp32 = k.to(torch.float32)
        torch.set_printoptions(linewidth=200)
        score_with_fp32 = torch.einsum('mhd,nd->hmn', q_fp32, k_fp32)
        # print("score_with_fp32 = ", score_with_fp32[0, :, 0])
        print("score_with_fp32 size stride = ", score_with_fp32.size(), score_with_fp32.stride())
        print("score_with_fp32[0, 1, 0] = ", score_with_fp32[0, 1, 0])
        print("score_with_fp32[:, 1, 0] = ", score_with_fp32[:, 1, 0])
        temp = score_with_fp32[:, 1, 0]
        for row in range(8):
            print(temp[(row*8):(row*8 +8)])
        print("score_with_fp32[:, 1, 0].relu() = ", score_with_fp32[:, 1, 0].relu())
        temp = score_with_fp32[:, 1, 0].relu()
        for row in range(8):
            print(temp[(row*8):(row*8 +7)])
        print("q_fp32 size = ", q_fp32.size(), q_fp32.stride())
        print("q_fp32[1, 0, 0] = ", q_fp32[1, 0, 0])

        print("k_fp32 size = ", k_fp32.size(), k_fp32.stride())
        print("k_fp32[0, 0] = ", k_fp32[0, 0])
        q_mul_k_0 = sum(q_fp32[1, 0, :] * k_fp32[0, :])
        q_mul_k_1 = sum(q_fp32[1, 1, :] * k_fp32[0, :])
        # print("q vector = ", q_fp32[1, 0, :])
        # print("k vector = ", k_fp32[0, :])
        print("q_mul_k_0 = ", q_mul_k_0)
        print("q_mul_k_1 = ", q_mul_k_1)
        print("weights[1,0] = ", weights[1,0])
        print("k_scales[0] = ", k_scales[0])

        score_mul_weight_reduce64 = sum(score_with_fp32[:, 1, 0].relu() * weights[1, :])
        print("score_mul_weight_reduce64 = ", score_mul_weight_reduce64)

        score_mul_weight_reduce64_mul_kscale = score_mul_weight_reduce64 * k_scales[0]
        print("score_mul_weight_reduce64_mul_kscale = ", score_mul_weight_reduce64_mul_kscale)
        # import pdb;pdb.set_trace()



    # defalut tile config for fp8 and int8
    block_qh, block_kv, warp_qh, warp_kv, num_q_stages, num_kv_stages = [256, 256, 64, 64, 3, 3] if num_heads == 64 else [128, 256, 32, 64, 3, 3]
    block_q = block_qh / num_heads
    assert(block_qh % num_heads == 0)

    seq_len_alignment = 4
    assert(seq_len_alignment % block_q == 0)
    aligned_seq_len = align(seq_len_q, seq_len_alignment)
    aligned_seq_len_kv = align(seq_len_k + block_kv, 4)
    logits = torch.empty(aligned_seq_len, aligned_seq_len_kv, dtype=torch.float, device=q.device)
    logits = logits[0:seq_len_q, 0:seq_len_k]

    # Auto-tuning with compilation
    global includes, template
    ElementQK = "cutlass::float_e4m3_t"
    ElementAcc = "float"
    if q.dtype == torch.bfloat16:
        ElementQK = 'cutlass::bfloat16_t'
        block_qh = 128
        block_kv = 128
        warp_kv = 32
    elif q.dtype == torch.int8:
        ElementQK = 'int8_t'
        ElementAcc = "int32_t"

    num_sms = get_num_sms()
    stream = torch.cuda.current_stream()
    # num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, 1, num_sms)
    args = (q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits,
            seq_len_q, seq_len_k, aligned_seq_len_kv, stream, num_sms)
    runtime = jit_tuner.compile_and_tune(
        name='attention_mqa_logits_' + ElementQK,
        keys={'ElementQK': ElementQK, 'ElementAcc' : ElementAcc,
              'kNumHeads': num_heads, 'kHeadDim': head_dim,
              'BLOCK_QH': block_qh, 'BLOCK_KV': block_kv,
              'WARP_QH' : warp_qh, 'WARP_KV' : warp_kv,
              'kNumQStages': num_q_stages, 'kNumKVStages': num_kv_stages},
        space=(),
        includes=includes,
        arg_defs=(('q', q.dtype), ('k', k.dtype), ('k_scales', torch.float), ('weights', torch.float),
                  ('cu_seq_len_k_start', torch.int32), ('cu_seq_len_k_end', torch.int32),('logits', torch.float),
                  ('seq_len_q', int), ('seq_len_k', int), ('aligned_seq_len_kv', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int)),
        template=template,
        args=args,
        jit_include_dir='cutlass3'
    )

    runtime(*args)

    # print("ppu logits before mask:", logits)
    if clean_logits:
        mask_lo = torch.arange(0, seq_len_k, device='cuda')[None, :] >= cu_seq_len_k_start[:, None]
        mask_hi = torch.arange(0, seq_len_k, device='cuda')[None, :] < cu_seq_len_k_end[:, None]
        mask = mask_lo & mask_hi
        logits = logits.masked_fill(~mask, float('-inf'))
    return logits

def bf16_mqa_logits(q: torch.Tensor,
                    kv: torch.Tensor,
                    weights: torch.Tensor,
                    cu_seq_len_k_start: torch.Tensor,
                    cu_seq_len_k_end: torch.Tensor,
                    clean_logits: bool = True):
    k_scales = torch.empty(0)
    return mqa_logits_common(q, kv, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits)

def fp8_mqa_logits(q: torch.Tensor,
                   kv_s: Tuple[torch.Tensor],
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True):
    k, k_scales = kv_s
    return mqa_logits_common(q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits)

def int8_mqa_logits(q: torch.Tensor,
                   kv_s: Tuple[torch.Tensor],
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True):
    k, k_scales = kv_s
    return mqa_logits_common(q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits)


includes_paged = ('"../deep_gemm/ppu_paged_mqa_logits.cuh"', )
template_paged_metadata = """
using namespace deep_gemm;
constexpr uint32_t SPLIT_KV = {SPLIT_KV};
constexpr uint32_t kNumSMs = {kNumSMs};
launch_paged_mqa_logits_metadata<SPLIT_KV, kNumSMs>(
    batch_size, (uint32_t*)context_lens, (uint32_t*)schedule_metadata, stream);
"""

template_paged = """
using namespace deep_gemm;
// Templated args from Python JIT call
using ElementQK = {ElementQK}; //cutlass::bfloat16_t or cutlass::float_e4m3_t
using ElementAcc = {ElementAcc};
constexpr uint32_t kNextN = {kNextN};
constexpr uint32_t kNumHeads = {kNumHeads};
constexpr uint32_t kHeadDim = {kHeadDim};
constexpr uint32_t BLOCK_KV = {BLOCK_KV};
constexpr uint32_t kNumQStages = {kNumQStages};
constexpr uint32_t kNumKVStages = {kNumKVStages};
constexpr uint32_t SPLIT_KV = {SPLIT_KV};

// Make a templated GEMM
using atten_t = PagedAttention<ElementQK, ElementAcc, kNextN, kNumHeads, kHeadDim, BLOCK_KV, kNumQStages, kNumKVStages, SPLIT_KV>;

// Launch kernel
atten_t::run((const ElementQK*)q, (const ElementQK*)k, k_scales, weights, batch_size, logits_stride, block_table_stride,
             (uint32_t*)context_lens, logits, (uint32_t*)block_table, (uint32_t*)schedule_meta, stream, num_sms);
"""


def get_paged_mqa_logits_metadata(context_lens: torch.Tensor,
                                  block_kv: int,
                                  num_sms: int,
                                  q: torch.Tensor = None):
    batch_size = context_lens.shape[0]
    assert(context_lens.dtype == torch.int32)
    assert(context_lens.is_contiguous())
    # shared memory limit
    assert(batch_size <= 65536)

    num_math_warpgroups = 1 # sm80, no warpgroup
    split_kv = block_kv * num_math_warpgroups

    tb_per_cu = 1
    if q is not None:
        batch_size, next_n, num_heads, head_dim = q.shape
        tb_per_cu = get_paged_mqa_logits_tb_per_sm(next_n, split_kv, num_heads, head_dim, q.element_size())
    num_blocks = num_sms * tb_per_cu
    schedule_metadata = torch.empty((num_blocks + 1, 2), dtype=context_lens.dtype, device=context_lens.device)

    stream = torch.cuda.current_stream()
    args = (batch_size, context_lens, schedule_metadata, stream)
    runtime = jit_tuner.compile_and_tune(
        name='attention_paged_mqa_logits_metadata',
        keys={'SPLIT_KV': split_kv,
              'kNumSMs': num_blocks},
        space=(),
        includes=includes_paged,
        arg_defs=(('batch_size', int),
                  ('context_lens', torch.int32),
                  ('schedule_metadata', torch.int32),
                  ('stream', torch.cuda.Stream)),
        template=template_paged_metadata,
        args=args,
        jit_include_dir='cutlass3'
    )

    runtime(*args)

    return schedule_metadata

def paged_mqa_logits_common(q: torch.Tensor,
                            fused_kv_cache: torch.Tensor,
                            weights: torch.Tensor,
                            context_lens: torch.Tensor,
                            block_table: torch.Tensor,
                            schedule_meta: torch.Tensor,
                            max_context_len: int,
                            clean_logits: bool = True):

    batch_size, next_n, num_heads, head_dim = q.shape
    num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf = fused_kv_cache.shape
    batch_size_ = context_lens.shape[0]
    batch_size_next_n, num_heads_ = weights.shape
    batch_size__, max_block_len = block_table.shape
    schedule_meta_size, meta_info_size = schedule_meta.shape
    kv_cache_stride_bytes = fused_kv_cache.stride(0)
    block_table_stride = block_table.stride(0)

    size_of_scale_float = 0 if q.dtype == torch.bfloat16 else 4
    num_sms = get_num_sms()
    assert(batch_size == batch_size_ and batch_size == batch_size__)
    assert(batch_size_next_n == batch_size * next_n)
    assert(num_heads == num_heads_ and num_heads_kv == 1)
    assert(head_dim_with_sf == head_dim + size_of_scale_float)
    assert((schedule_meta_size - 1) % num_sms == 0 and meta_info_size == 2)

    assert(next_n == 1 or next_n == 2)
    assert(block_kv == 64)

    assert(q.is_contiguous())
    if q.dtype != torch.bfloat16:
        assert(kv_cache_stride_bytes % size_of_scale_float == 0)
    assert(fused_kv_cache.stride(1) == head_dim_with_sf)
    assert(fused_kv_cache.stride(2) == head_dim_with_sf)
    assert(fused_kv_cache.stride(3) == 1)
    assert(weights.is_contiguous())
    assert(context_lens.is_contiguous())
    assert(block_table.stride(1) == 1)
    assert(schedule_meta.is_contiguous())

    if q.dtype != torch.bfloat16:
        assert(fused_kv_cache.dtype == torch.uint8)
    assert(weights.dtype == torch.float)
    assert(context_lens.dtype == torch.int32)
    assert(block_table.dtype == torch.int32)
    assert(schedule_meta.dtype == torch.int32)


    # Derive FP8 values and SF tensor from KV cache
    k = torch.as_strided(
        input=fused_kv_cache,
        size=(num_kv_blocks, block_kv, head_dim),
        stride=(kv_cache_stride_bytes, head_dim, 1),
    ).view(dtype=q.dtype)

    # import pdb;pdb.set_trace()
    if q.dtype == torch.bfloat16:
        k_scales = torch.empty(0)
    else:
        k_scales = torch.as_strided(
            input=fused_kv_cache,
            size=(num_kv_blocks, block_kv * 4),
            stride=(kv_cache_stride_bytes, 1),
            storage_offset = block_kv * head_dim,
        ).view(dtype=torch.float)

    debug = False
    if debug:
        print("kv size = ", k.size(), " stride = ", k.stride())
        print("context_lens[0] = ", context_lens[0] )
        # sum0 = q[0,0,0,:] * k[]
        # print(q[])
        # import pdb;pdb.set_trace()

    num_math_warp_groups = 1
    aligned_max_context_len = align(max_context_len, num_math_warp_groups * block_kv)
    logits = torch.zeros((batch_size * next_n, aligned_max_context_len), dtype=torch.float, device=q.device)
    logits = logits[..., :max_context_len]

    num_q_stages = 3
    num_kv_stages = 3
    split_kv = num_math_warp_groups * block_kv
    logits_stride = aligned_max_context_len

    # Construct TMAs
    assert(head_dim == 32 or head_dim == 64 or head_dim == 128)

    global includes_paged, template_paged
    ElementQK = "cutlass::float_e4m3_t"
    ElementAcc = "float"
    if q.dtype == torch.bfloat16:
        ElementQK = 'cutlass::bfloat16_t'
    elif q.dtype == torch.int8:
        ElementQK = 'int8_t'
        ElementAcc = "int32_t"

    stream = torch.cuda.current_stream()
    args = (q, k, k_scales, weights, batch_size, logits_stride, block_table_stride, context_lens, logits,
            block_table, schedule_meta, stream, schedule_meta_size - 1)
    runtime = jit_tuner.compile_and_tune(
        name='attention_paged_mqa_logits_' + ElementQK,
        keys={'ElementQK': ElementQK, 'ElementAcc' : ElementAcc,
              'kNextN' : next_n, 'kNumHeads': num_heads,
              'kHeadDim': head_dim, 'BLOCK_KV': block_kv,
              'kNumQStages': num_q_stages, 'kNumKVStages': num_kv_stages,
              'SPLIT_KV': split_kv},
        space=(),
        includes=includes_paged,
        arg_defs=(('q', q.dtype), ('k', k.dtype), ('k_scales', torch.float), ('weights', torch.float),
                  ('batch_size', int), ('logits_stride', int), ('block_table_stride', int), ('context_lens', torch.int32),
                  ('logits', torch.float), ('block_table', torch.int32), ('schedule_meta', torch.int32),
                  ('stream', torch.cuda.Stream), ('num_sms', int)),
        template=template_paged,
        args=args,
        jit_include_dir='cutlass3'
    )

    runtime(*args)

    return logits

def bf16_paged_mqa_logits(q: torch.Tensor,
                          fused_kv_cache: torch.Tensor,
                          weights: torch.Tensor,
                          context_lens: torch.Tensor,
                          block_table: torch.Tensor,
                          schedule_meta: torch.Tensor,
                          max_context_len: int,
                          clean_logits: bool = True):
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits)

def fp8_paged_mqa_logits(q: torch.Tensor,
                         fused_kv_cache: torch.Tensor,
                         weights: torch.Tensor,
                         context_lens: torch.Tensor,
                         block_table: torch.Tensor,
                         schedule_meta: torch.Tensor,
                         max_context_len: int,
                         clean_logits: bool = True):
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits)

def int8_paged_mqa_logits(q: torch.Tensor,
                         fused_kv_cache: torch.Tensor,
                         weights: torch.Tensor,
                         context_lens: torch.Tensor,
                         block_table: torch.Tensor,
                         schedule_meta: torch.Tensor,
                         max_context_len: int,
                         clean_logits: bool = True):
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits)
