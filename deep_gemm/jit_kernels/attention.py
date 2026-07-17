import math
import torch
from functools import lru_cache
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout, get_extra_info, get_paged_mqa_logits_tile

def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment

# C++ code templates
includes = ('"../deep_gemm/ppu_mqa_logits.cuh"', )
includes_fp4_mqa = ('"../deep_gemm/fp4_mqa_logits.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
using ElementQK = {ElementQK}; //cutlass::bfloat16_t or cutlass::float_e4m3_t
using ElementAcc = {ElementAcc};
using ElementLogits = {ElementLogits};  // float or __ppu_bfloat16
constexpr auto kNumHeads = {kNumHeads};
constexpr auto kHeadDim = {kHeadDim};
constexpr auto BLOCK_QH = {BLOCK_QH};
constexpr auto BLOCK_KV = {BLOCK_KV};
constexpr auto WARP_QH = {WARP_QH};
constexpr auto WARP_KV = {WARP_KV};
constexpr auto kNumQStages = {kNumQStages};
constexpr auto kNumKVStages = {kNumKVStages};
using StrideKType = {StrideKType};

// Make a templated GEMM
using atten_t = Attention<ElementQK, ElementAcc, ElementLogits, kNumHeads, kHeadDim, BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages, StrideKType>;

// Launch kernel
atten_t::run((const ElementQK*)q, (const ElementQK*)k, k_scales, weights, (uint32_t*)cu_seq_len_k_start, (uint32_t*)cu_seq_len_k_end, logits,
             seq_len_q, seq_len_k, static_cast<StrideKType>(aligned_seq_len_kv), stream, num_sms);
"""

template_fp4_mqa = """
using namespace deep_gemm;
// Templated args from Python JIT call
using ElementQK = {ElementQK};          // uint8_t (packed FP4)
using ElementAcc = {ElementAcc};
using ElementLogits = {ElementLogits};  // float or __ppu_bfloat16
constexpr int kNumHeads = {kNumHeads};
constexpr int kHeadDim = {kHeadDim};       // packed head_dim (64)
constexpr int BLOCK_QH = {BLOCK_QH};
constexpr int BLOCK_KV = {BLOCK_KV};
constexpr int WARP_QH = {WARP_QH};
constexpr int WARP_KV = {WARP_KV};
constexpr int kNumQStages = {kNumQStages};
constexpr int kNumKVStages = {kNumKVStages};
using StrideKType = {StrideKType};

// Make a templated GEMM
using atten_t = AttentionFP4<ElementQK, ElementAcc, ElementLogits, kNumHeads, kHeadDim, BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages, StrideKType>;

// Launch kernel
atten_t::run((const ElementQK*)q, (const uint32_t*)q_sf, (const ElementQK*)k, (const uint32_t*)k_sf, weights,
             cu_seq_len_k_start, cu_seq_len_k_end, logits,
             seq_len_q, seq_len_k, static_cast<StrideKType>(aligned_seq_len_kv), stream, num_sms);
"""


def _select_stride_k_type(aligned_seq_len: int, aligned_seq_len_kv: int) -> str:
    """Pick StrideKType for kernel: uint32_t when q_idx*stride_k fits in 32-bit,
    otherwise fall back to uint64_t to avoid overflow."""
    # aligned_seq_len_kv is passed via JIT as ctypes.c_int (signed 32-bit),
    # so it itself must fit in int32 positive range.
    assert aligned_seq_len_kv < (1 << 31), \
        f"aligned_seq_len_kv({aligned_seq_len_kv}) exceeds int32 positive range; " \
        f"JIT passes it via ctypes.c_int."
    if aligned_seq_len * aligned_seq_len_kv < (1 << 32):
        return "uint32_t"
    return "uint64_t"


def mqa_logits_common(q: torch.Tensor,
                      k: torch.Tensor, k_scales: torch.Tensor,
                      weights: torch.Tensor,
                      cu_seq_len_k_start: torch.Tensor,
                      cu_seq_len_k_end: torch.Tensor,
                      clean_logits: bool = True,
                      logits_dtype: torch.dtype = torch.float32):
    assert(logits_dtype in (torch.float32, torch.bfloat16))
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

    # defalut tile config for fp8 and int8
    block_qh, block_kv, warp_qh, warp_kv, num_q_stages, num_kv_stages = [256, 256, 64, 64, 3, 3] if num_heads == 64 else [128, 256, 32, 64, 3, 3]
    block_q = block_qh / num_heads
    assert(block_qh % num_heads == 0)

    seq_len_alignment = 4
    assert(seq_len_alignment % block_q == 0)
    aligned_seq_len = align(seq_len_q, seq_len_alignment)
    aligned_seq_len_kv = align(seq_len_k + block_kv, 4)
    # stride_k may be uint32_t or uint64_t depending on whether q_idx * stride_k
    # overflows 32-bit. Decided dynamically here and forwarded as a JIT template key.
    stride_k_type = _select_stride_k_type(aligned_seq_len, aligned_seq_len_kv)
    logits = torch.empty(aligned_seq_len, aligned_seq_len_kv, dtype=logits_dtype, device=q.device)
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

    ElementLogits = "float" if logits_dtype == torch.float32 else "__ppu_bfloat16"
    num_sms = get_num_sms()
    stream = torch.cuda.current_stream()
    # num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, 1, num_sms)
    args = (q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits,
            seq_len_q, seq_len_k, aligned_seq_len_kv, stream, num_sms)
    runtime = jit_tuner.compile_and_tune(
        name='attention_mqa_logits_' + ElementQK,
        keys={'ElementQK': ElementQK, 'ElementAcc' : ElementAcc,
              'ElementLogits': ElementLogits,
              'kNumHeads': num_heads, 'kHeadDim': head_dim,
              'BLOCK_QH': block_qh, 'BLOCK_KV': block_kv,
              'WARP_QH' : warp_qh, 'WARP_KV' : warp_kv,
              'kNumQStages': num_q_stages, 'kNumKVStages': num_kv_stages,
              'StrideKType': stride_k_type},
        space=(),
        includes=includes,
        arg_defs=(('q', q.dtype), ('k', k.dtype), ('k_scales', torch.float), ('weights', torch.float),
                  ('cu_seq_len_k_start', torch.int32), ('cu_seq_len_k_end', torch.int32),('logits', logits_dtype),
                  ('seq_len_q', int), ('seq_len_k', int), ('aligned_seq_len_kv', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int)),
        template=template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
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
                    clean_logits: bool = True,
                    logits_dtype: torch.dtype = torch.float32):
    k_scales = torch.empty(0)
    return mqa_logits_common(q, kv, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, logits_dtype=logits_dtype)

def fp8_mqa_logits(q: torch.Tensor,
                   kv_s: Tuple[torch.Tensor],
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True,
                   logits_dtype: torch.dtype = torch.float32):
    k, k_scales = kv_s
    return mqa_logits_common(q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, logits_dtype=logits_dtype)

def int8_mqa_logits(q: torch.Tensor,
                   kv_s: Tuple[torch.Tensor],
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True,
                   logits_dtype: torch.dtype = torch.float32):
    k, k_scales = kv_s
    return mqa_logits_common(q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, logits_dtype=logits_dtype)


def fp4_mqa_logits(q: torch.Tensor,
                   q_sf: torch.Tensor,
                   k: torch.Tensor,
                   k_sf: torch.Tensor,
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True,
                   logits_dtype: torch.dtype = torch.float32):
    """FP4 MQA logits (non-paged).

    Args:
        q:     packed FP4 Q, int8 [seq_len_q, num_heads, head_dim_packed]
               where head_dim_packed = original_head_dim // 2 = 64
        q_sf:  UE8M0 scale for Q, int32 [seq_len_q, num_heads]
               Packed from uint8 e8m0 (4\u00d7uint8 per int32)
        k:     packed FP4 K, uint8 [seq_len_k, head_dim_packed]
        k_sf:  UE8M0 scale for K, int32 [seq_len_k]
        weights:      float32 [seq_len_q, num_heads]
        cu_seq_len_k_start: int32 [seq_len_q]
        cu_seq_len_k_end:   int32 [seq_len_q]
        clean_logits: whether to mask out-of-range logits with -inf
        logits_dtype: output logits dtype, torch.float32 or torch.bfloat16
    """
    assert(logits_dtype in (torch.float32, torch.bfloat16))
    seq_len_q, num_heads, head_dim_packed = q.shape
    seq_len_k, head_dim_packed_ = k.shape
    seq_len_k_ = k_sf.shape[0]
    seq_len_q__, num_heads_ = weights.shape

    assert(head_dim_packed == head_dim_packed_ and head_dim_packed == 64)
    assert(num_heads == 32 or num_heads == 64)
    assert(num_heads == num_heads_)
    assert(seq_len_q == seq_len_q__)
    assert(seq_len_k == seq_len_k_)
    assert(cu_seq_len_k_start.size(0) == seq_len_q)
    assert(cu_seq_len_k_end.size(0) == seq_len_q)

    assert(q.is_contiguous() and k.is_contiguous())
    assert(q_sf.is_contiguous() and k_sf.is_contiguous())
    assert(weights.is_contiguous())
    assert(cu_seq_len_k_start.is_contiguous() and cu_seq_len_k_end.is_contiguous())

    assert(q.dtype == torch.int8)
    assert(k.dtype == torch.int8)
    assert(q_sf.dtype == torch.int32)
    assert(k_sf.dtype == torch.int32)
    assert(weights.dtype == torch.float32)
    assert(cu_seq_len_k_start.dtype == torch.int32)
    assert(cu_seq_len_k_end.dtype == torch.int32)

    # Default tile config for FP4 non-paged MQA logits
    num_sms = get_num_sms()
    stream = torch.cuda.current_stream()
    if num_heads == 64 and logits_dtype == torch.bfloat16:
        if seq_len_k >= 8192:
            block_q, warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = 4, num_heads, 256, 64, 1, 4
        else:
            block_q, warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = 4, num_heads, 64, 16, 1, 4
    elif num_heads == 64 and logits_dtype == torch.float32:
        block_q, warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = 4, num_heads, 256, 64, 1, 3
    elif num_heads == 32 and logits_dtype == torch.bfloat16:
        block_q, warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = 4, num_heads, 64, 16, 1, 3
    elif num_heads == 32 and logits_dtype == torch.float32:
        block_q, warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = 4, num_heads, 64, 16, 1, 4
    block_qh = num_heads * block_q

    seq_len_alignment = 4
    assert(seq_len_alignment % block_q == 0)
    aligned_seq_len = align(seq_len_q, seq_len_alignment)
    aligned_seq_len_kv = align(seq_len_k + block_kv, 4)
    # stride_k may be uint32_t or uint64_t depending on whether q_idx * stride_k
    # overflows 32-bit. Decided dynamically here and forwarded as a JIT template key.
    stride_k_type = _select_stride_k_type(aligned_seq_len, aligned_seq_len_kv)
    logits = torch.empty(aligned_seq_len, aligned_seq_len_kv, dtype=logits_dtype, device=q.device)
    logits = logits[0:seq_len_q, 0:seq_len_k]

    global includes_fp4_mqa, template_fp4_mqa
    ElementQK = "uint8_t"
    ElementAcc = "float"
    ElementLogits = "float" if logits_dtype == torch.float32 else "__ppu_bfloat16"

    args = (q.view(torch.uint8), q_sf, k.view(torch.uint8), k_sf, weights,
            cu_seq_len_k_start, cu_seq_len_k_end, logits,
            seq_len_q, seq_len_k, aligned_seq_len_kv,
            stream, num_sms)
    runtime = jit_tuner.compile_and_tune(
        name='attention_mqa_logits_fp4',
        keys={'ElementQK': ElementQK, 'ElementAcc': ElementAcc,
              'ElementLogits': ElementLogits,
              'kNumHeads': num_heads, 'kHeadDim': head_dim_packed,
              'BLOCK_QH': block_qh, 'BLOCK_KV': block_kv,
              'WARP_QH': warp_qh, 'WARP_KV': warp_kv,
              'kNumQStages': num_q_stages, 'kNumKVStages': num_kv_stages,
              'StrideKType': stride_k_type},
        space=(),
        includes=includes_fp4_mqa,
        arg_defs=(('q', torch.uint8), ('q_sf', torch.int32), ('k', torch.uint8), ('k_sf', torch.int32), ('weights', torch.float),
                  ('cu_seq_len_k_start', torch.int32), ('cu_seq_len_k_end', torch.int32),
                  ('logits', logits_dtype),
                  ('seq_len_q', int), ('seq_len_k', int), ('aligned_seq_len_kv', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int)),
        template=template_fp4_mqa,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )

    runtime(*args)

    if clean_logits:
        mask_lo = torch.arange(0, seq_len_k, device='cuda')[None, :] >= cu_seq_len_k_start[:, None]
        mask_hi = torch.arange(0, seq_len_k, device='cuda')[None, :] < cu_seq_len_k_end[:, None]
        mask = mask_lo & mask_hi
        logits = logits.masked_fill(~mask, float('-inf'))
    return logits


includes_paged = ('"../deep_gemm/ppu_paged_mqa_logits.cuh"', )
includes_paged_fp4 = ('"../deep_gemm/fp4_paged_mqa_logits.cuh"', )
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
using ElementLogits = {ElementLogits};  // float or __ppu_bfloat16
constexpr uint32_t kNextN = {kNextN};
constexpr uint32_t kNumHeads = {kNumHeads};
constexpr uint32_t kHeadDim = {kHeadDim};
constexpr uint32_t BLOCK_KV = {BLOCK_KV};
constexpr uint32_t kNumQStages = {kNumQStages};
constexpr uint32_t kNumKVStages = {kNumKVStages};
constexpr uint32_t SPLIT_KV = {SPLIT_KV};

// Make a templated GEMM
using atten_t = PagedAttention<ElementQK, ElementAcc, ElementLogits, kNextN, kNumHeads, kHeadDim, BLOCK_KV, kNumQStages, kNumKVStages, SPLIT_KV>;

// Launch kernel
atten_t::run((const ElementQK*)q, (const ElementQK*)k, k_scales, weights, batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride,
             (uint32_t*)context_lens, logits, (uint32_t*)block_table, (uint32_t*)schedule_meta, stream, num_sms, num_blocks);
"""

template_paged_fp4 = """
using namespace deep_gemm;
// Templated args from Python JIT call
using ElementQK = {ElementQK};     // uint8_t (packed FP4)
using ElementAcc = {ElementAcc};
using ElementLogits = {ElementLogits}; // float or cutlass::bfloat16_t
constexpr uint32_t kNextN = {kNextN};
constexpr uint32_t kNumHeads = {kNumHeads};
constexpr uint32_t kHeadDim = {kHeadDim};   // packed head_dim (64)
constexpr uint32_t BLOCK_KV = {BLOCK_KV};
constexpr uint32_t kNumQStages = {kNumQStages};
constexpr uint32_t kNumKVStages = {kNumKVStages};
constexpr uint32_t SPLIT_KV = {SPLIT_KV};

// Make a templated GEMM
using atten_t = PagedAttentionFP4<ElementQK, ElementAcc, ElementLogits, kNextN, kNumHeads, kHeadDim, BLOCK_KV, kNumQStages, kNumKVStages, SPLIT_KV>;

// Launch kernel
atten_t::run((const ElementQK*)q, (const uint32_t*)q_sf, (const ElementQK*)k, (const uint32_t*)k_scales, weights, batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride,
             (uint32_t*)context_lens, logits, (uint32_t*)block_table, (uint32_t*)schedule_meta, stream, num_sms, num_blocks);
"""


def get_paged_mqa_logits_metadata(context_lens: torch.Tensor,
                                  block_kv: int,
                                  num_sms: int,
                                  metadata_extra: tuple = None):
    batch_size = context_lens.shape[0]
    assert(context_lens.dtype == torch.int32)
    assert(context_lens.is_contiguous())
    # shared memory limit
    assert(batch_size <= 65536)

    num_math_warpgroups = 1
    split_kv = block_kv * num_math_warpgroups

    tb_per_cu = 1
    if metadata_extra is not None:
        next_n, num_heads, head_dim, element_size = metadata_extra
        # if element_size == 1 and head_dim == 64: # fp4 warp-interleave
        #     num_math_warpgroups = 4
        #     split_kv = block_kv * num_math_warpgroups
        _, _, tb_per_cu = get_paged_mqa_logits_tile(next_n, split_kv, num_heads, head_dim, element_size)
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
        jit_include_dir='actlize_v1.0.0'
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
                            clean_logits: bool = True,
                            logits_dtype: torch.dtype = torch.float32):
    assert(logits_dtype in (torch.float32, torch.bfloat16))

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

    if q.dtype == torch.bfloat16:
        k_scales = torch.empty(0)
    else:
        k_scales = torch.as_strided(
            input=fused_kv_cache,
            size=(num_kv_blocks, block_kv * 4),
            stride=(kv_cache_stride_bytes, 1),
            storage_offset = block_kv * head_dim,
        ).view(dtype=torch.float)

    num_math_warp_groups = 1
    aligned_max_context_len = align(max_context_len, num_math_warp_groups * block_kv)
    logits = torch.empty((batch_size * next_n, aligned_max_context_len), dtype=logits_dtype, device=q.device)
    logits = logits[..., :max_context_len]

    split_kv = num_math_warp_groups * block_kv
    num_q_stages, num_kv_stages, _ = get_paged_mqa_logits_tile(next_n, split_kv, num_heads, head_dim, q.element_size())
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

    ElementLogits = "float" if logits_dtype == torch.float32 else "__ppu_bfloat16"
    stream = torch.cuda.current_stream()
    args = (q, k, k_scales, weights, batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride, context_lens, logits,
            block_table, schedule_meta, stream, num_sms, schedule_meta_size - 1)
    runtime = jit_tuner.compile_and_tune(
        name='attention_paged_mqa_logits_' + ElementQK,
        keys={'ElementQK': ElementQK, 'ElementAcc' : ElementAcc,
              'ElementLogits': ElementLogits,
              'kNextN' : next_n, 'kNumHeads': num_heads,
              'kHeadDim': head_dim, 'BLOCK_KV': block_kv,
              'kNumQStages': num_q_stages, 'kNumKVStages': num_kv_stages,
              'SPLIT_KV': split_kv},
        space=(),
        includes=includes_paged,
        arg_defs=(('q', q.dtype), ('k', k.dtype), ('k_scales', torch.float), ('weights', torch.float),
                  ('batch_size', int), ('logits_stride', int), ('kv_cache_stride_bytes', int), ('block_table_stride', int), ('context_lens', torch.int32),
                  ('logits', logits_dtype), ('block_table', torch.int32), ('schedule_meta', torch.int32),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('num_blocks', int)),
        template=template_paged,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )

    runtime(*args)

    if clean_logits:
        offsets = torch.arange(next_n, device=context_lens.device)
        context_lens_expanded = (context_lens[:, None] - next_n + offsets[None, :]).reshape(-1)
        positions = torch.arange(max_context_len, device=logits.device)  # [max_context_len]
        mask = positions[None, :] <= context_lens_expanded[:, None]  # [batch_size * next_n, max_context_len]
        logits = logits.masked_fill(~mask, float('-inf'))
    return logits

def bf16_paged_mqa_logits(q: torch.Tensor,
                          fused_kv_cache: torch.Tensor,
                          weights: torch.Tensor,
                          context_lens: torch.Tensor,
                          block_table: torch.Tensor,
                          schedule_meta: torch.Tensor,
                          max_context_len: int,
                          clean_logits: bool = True,
                          logits_dtype: torch.dtype = torch.float32):
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits, logits_dtype=logits_dtype)

def fp8_paged_mqa_logits(q: torch.Tensor,
                         fused_kv_cache: torch.Tensor,
                         weights: torch.Tensor,
                         context_lens: torch.Tensor,
                         block_table: torch.Tensor,
                         schedule_meta: torch.Tensor,
                         max_context_len: int,
                         clean_logits: bool = True,
                         logits_dtype: torch.dtype = torch.float32):
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits, logits_dtype=logits_dtype)

def int8_paged_mqa_logits(q: torch.Tensor,
                         fused_kv_cache: torch.Tensor,
                         weights: torch.Tensor,
                         context_lens: torch.Tensor,
                         block_table: torch.Tensor,
                         schedule_meta: torch.Tensor,
                         max_context_len: int,
                         clean_logits: bool = True,
                         logits_dtype: torch.dtype = torch.float32):
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits, logits_dtype=logits_dtype)


def fp4_paged_mqa_logits(q: torch.Tensor,
                         q_sf: torch.Tensor,
                         fused_kv_cache: torch.Tensor,
                         weights: torch.Tensor,
                         context_lens: torch.Tensor,
                         block_table: torch.Tensor,
                         schedule_meta: torch.Tensor,
                         max_context_len: int,
                         clean_logits: bool = True,
                         logits_dtype: torch.dtype = torch.float32):
    """FP4 paged MQA logits with dedicated indexer for FP4 fused KV cache layout.

    Args:
        q:  packed FP4 Q, int8 [batch, next_n, num_heads, head_dim_packed]
               where head_dim_packed = original_head_dim // 2
        q_sf:  UE8M0 scale for Q, int32 [batch, next_n, num_heads]
               Packed from uint8 e8m0 (4×uint8 per int32)
        fused_kv_cache: fused FP4 KV cache, uint8
               [num_kv_blocks, block_kv, 1, head_dim_packed + scale_bytes]
               Per-row layout: [fp4_packed (head_dim_packed bytes), scale (scale_bytes bytes)]
        weights:       float32 [batch * next_n, num_heads]
        context_lens:  int32   [batch]
        block_table:   int32   [batch, max_block_len]
        schedule_meta: int32   [num_blocks+1, 2]
        max_context_len: maximum context length
        clean_logits:  whether to mask out-of-range logits with -inf
        logits_dtype:  output logits dtype, torch.float32 or torch.bfloat16.
                       The kernel produces logits in this dtype directly,
                       no separate conversion kernel is needed.
    """
    assert(logits_dtype in (torch.float32, torch.bfloat16))
    batch_size, next_n, num_heads, head_dim_packed = q.shape
    num_kv_blocks, block_kv, num_heads_kv, fused_row_bytes = fused_kv_cache.shape
    batch_size_ = context_lens.shape[0]
    batch_size_next_n, num_heads_ = weights.shape
    batch_size__, max_block_len = block_table.shape
    schedule_meta_size, meta_info_size = schedule_meta.shape
    kv_cache_stride_bytes = fused_kv_cache.stride(0)
    block_table_stride = block_table.stride(0)

    # FP4 packed Q: original head_dim=128, packed head_dim=64, scale_bytes=4
    assert(head_dim_packed == 64)
    scale_bytes = fused_row_bytes - head_dim_packed
    assert(scale_bytes == 4)
    num_sms = get_num_sms()
    assert(batch_size == batch_size_ and batch_size == batch_size__)
    assert(batch_size_next_n == batch_size * next_n)
    assert(num_heads == num_heads_ and num_heads_kv == 1)
    assert(fused_row_bytes == head_dim_packed + scale_bytes)
    assert((schedule_meta_size - 1) % num_sms == 0 and meta_info_size == 2)

    assert(next_n == 1 or next_n == 2)
    assert(block_kv == 64)

    assert(q.is_contiguous())
    assert(q.dtype == torch.int8)
    assert(q_sf.is_contiguous())
    assert(q_sf.dtype == torch.int32)
    assert(q_sf.shape == (batch_size, next_n, num_heads))
    assert(fused_kv_cache.dtype == torch.uint8)
    assert(kv_cache_stride_bytes % scale_bytes == 0)
    assert(fused_kv_cache.stride(1) == fused_row_bytes)
    assert(fused_kv_cache.stride(2) == fused_row_bytes)
    assert(fused_kv_cache.stride(3) == 1)
    assert(weights.is_contiguous())
    assert(weights.dtype == torch.float)
    assert(context_lens.is_contiguous())
    assert(context_lens.dtype == torch.int32)
    assert(block_table.stride(1) == 1)
    assert(block_table.dtype == torch.int32)
    assert(schedule_meta.is_contiguous())
    assert(schedule_meta.dtype == torch.int32)

    k = torch.as_strided(
        input=fused_kv_cache,
        size=(num_kv_blocks, block_kv, head_dim_packed),
        stride=(kv_cache_stride_bytes, head_dim_packed, 1),
    )
    # k_scales: view the scale region as int32 (4×uint8 per row packed into one uint32)
    k_scales = torch.as_strided(
        input=fused_kv_cache,
        size=(num_kv_blocks, block_kv * 4),
        stride=(kv_cache_stride_bytes, 1),
        storage_offset=block_kv*head_dim_packed,
    ).view(torch.int32)

    num_math_warp_groups = 1
    # num_math_warp_groups = 4 # 4 warpgroups, each warpgroup 4 warps, 16 warps for warp-interleave
    aligned_max_context_len = align(max_context_len, num_math_warp_groups * block_kv)
    logits = torch.empty((batch_size * next_n, aligned_max_context_len), dtype=logits_dtype, device=q.device)
    logits = logits[..., :max_context_len]

    split_kv = num_math_warp_groups * block_kv
    num_q_stages, num_kv_stages, _ = get_paged_mqa_logits_tile(
        next_n, split_kv, num_heads, head_dim_packed, q.element_size())
    logits_stride = aligned_max_context_len

    global includes_paged_fp4, template_paged_fp4
    ElementQK = "uint8_t"
    ElementAcc = "float"
    ElementLogits = "float" if logits_dtype == torch.float32 else "__ppu_bfloat16"

    stream = torch.cuda.current_stream()
    args = (q.view(torch.uint8), q_sf, k, k_scales, weights, batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride, context_lens, logits,
            block_table, schedule_meta, stream, num_sms, schedule_meta_size - 1)
    runtime = jit_tuner.compile_and_tune(
        name='attention_paged_mqa_logits_fp4',
        keys={'ElementQK': ElementQK, 'ElementAcc': ElementAcc,
              'ElementLogits': ElementLogits,
              'kNextN': next_n, 'kNumHeads': num_heads,
              'kHeadDim': head_dim_packed, 'BLOCK_KV': block_kv,
              'kNumQStages': num_q_stages, 'kNumKVStages': num_kv_stages,
              'SPLIT_KV': split_kv},
        space=(),
        includes=includes_paged_fp4,
        arg_defs=(('q', torch.uint8), ('q_sf', torch.int32), ('k', torch.uint8), ('k_scales', torch.int32), ('weights', torch.float),
                  ('batch_size', int), ('logits_stride', int), ('kv_cache_stride_bytes', int), ('block_table_stride', int), ('context_lens', torch.int32),
                  ('logits', logits_dtype), ('block_table', torch.int32), ('schedule_meta', torch.int32),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('num_blocks', int)),
        template=template_paged_fp4,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )

    runtime(*args)

    if clean_logits:
        offsets = torch.arange(next_n, device=context_lens.device)
        context_lens_expanded = (context_lens[:, None] - next_n + offsets[None, :]).reshape(-1)
        positions = torch.arange(max_context_len, device=logits.device)  # [max_context_len]
        mask = positions[None, :] <= context_lens_expanded[:, None]  # [batch_size * next_n, max_context_len]
        logits = logits.masked_fill(~mask, float('-inf'))
    return logits


# New unified FP8/FP4 APIs
# q  : tuple(q_fp, Optional[q_sf])
#        FP8 mode : q_fp is float8_e4m3fn, q_sf is None
#        FP4 mode : q_fp is packed FP4 (int8), q_sf is UE8M0 scale factor (int32)
# kv : tuple(kv_fp, kv_sf)
#        FP8 mode : kv_fp is float8_e4m3fn, kv_sf is per-token float32 scale
#        FP4 mode : kv_fp is packed FP4 (int8), kv_sf is UE8M0 scale factor (int32)
# logits_dtype : output dtype, torch.float32 or torch.bfloat16

def fp8_fp4_mqa_logits(q: Tuple,
                      kv: Tuple,
                      weights: torch.Tensor,
                      cu_seq_len_k_start: torch.Tensor,
                      cu_seq_len_k_end: torch.Tensor,
                      clean_logits: bool = True,
                      max_seqlen_k: int = 0,
                      logits_dtype: torch.dtype = torch.float32):
    q_fp, q_sf = q
    kv_fp, kv_sf = kv
    is_fp4 = q_sf is not None
    # max_seqlen_k (compressed logits) is not supported in the current PPU
    if max_seqlen_k != 0:
        raise NotImplementedError(
            "compressed_logits (max_seqlen_k != 0) is not yet supported in PPU fp8_fp4_mqa_logits."
        )

    if is_fp4:
        logits = fp4_mqa_logits(q_fp, q_sf, kv_fp, kv_sf, weights,
                                cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                                logits_dtype=logits_dtype)
        return logits
    else:
        logits = mqa_logits_common(q_fp, kv_fp, kv_sf, weights,
                                   cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                                   logits_dtype=logits_dtype)
        return logits


def fp8_fp4_paged_mqa_logits(q: Tuple,
                             fused_kv_cache: torch.Tensor,
                             weights: torch.Tensor,
                             context_lens: torch.Tensor,
                             block_table: torch.Tensor,
                             schedule_meta: torch.Tensor,
                             max_context_len: int,
                             clean_logits: bool = False,
                             logits_dtype: torch.dtype = torch.float32):
    q_fp, q_sf = q
    is_fp4 = q_sf is not None

    if is_fp4:
        logits = fp4_paged_mqa_logits(q_fp, q_sf, fused_kv_cache, weights,
                                     context_lens, block_table, schedule_meta,
                                     max_context_len, clean_logits,
                                     logits_dtype=logits_dtype)
        return logits
    else:
        logits = paged_mqa_logits_common(q_fp, fused_kv_cache, weights, context_lens,
                                        block_table, schedule_meta, max_context_len, clean_logits,
                                        logits_dtype=logits_dtype)
        return logits
