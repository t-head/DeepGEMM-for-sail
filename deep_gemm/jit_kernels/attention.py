import math
import torch
from functools import lru_cache
from typing import Tuple, Optional

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout, get_extra_info, get_paged_mqa_logits_tile

def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment

@lru_cache(maxsize=None)
def print_once(msg):
    """Print a warning message only once, in yellow."""
    print(f"\033[33m{msg}\033[0m")

# C++ code templates
includes = ('"../deep_gemm/ppu_mqa_logits.cuh"', )
includes_fp4_mqa = ('"../deep_gemm/fp4_mqa_logits.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
using ElementQK = {ElementQK}; //cutlass::bfloat16_t or cutlass::float_e4m3_t
using ElementAcc = {ElementAcc};
using ElementLogits = {ElementLogits};  // float or __ppu_bfloat16
using ElementWeights = {ElementWeights};  // float or __ppu_bfloat16
constexpr auto kNumHeads = {kNumHeads};
constexpr auto kHeadDim = {kHeadDim};
constexpr auto BLOCK_QH = {BLOCK_QH};
constexpr auto BLOCK_KV = {BLOCK_KV};
constexpr auto WARP_QH = {WARP_QH};
constexpr auto WARP_KV = {WARP_KV};
constexpr auto kNumQStages = {kNumQStages};
constexpr auto kNumKVStages = {kNumKVStages};
using StrideKType = {StrideKType};
constexpr bool kIsCompressedLogits = {kIsCompressedLogits};

// Make a templated GEMM
using atten_t = Attention<ElementQK, ElementAcc, ElementLogits, ElementWeights, kNumHeads, kHeadDim, BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages, StrideKType, kIsCompressedLogits>;

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
using ElementWeights = {ElementWeights};  // float or __ppu_bfloat16
constexpr int kNumHeads = {kNumHeads};
constexpr int kHeadDim = {kHeadDim};       // packed head_dim (64)
constexpr int BLOCK_QH = {BLOCK_QH};
constexpr int BLOCK_KV = {BLOCK_KV};
constexpr int WARP_QH = {WARP_QH};
constexpr int WARP_KV = {WARP_KV};
constexpr int kNumQStages = {kNumQStages};
constexpr int kNumKVStages = {kNumKVStages};
using StrideKType = {StrideKType};
constexpr bool kIsCompressedLogits = {kIsCompressedLogits};

// Make a templated GEMM
using atten_t = AttentionFP4<ElementQK, ElementAcc, ElementLogits, ElementWeights, kNumHeads, kHeadDim, BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages, StrideKType, kIsCompressedLogits>;

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
                      max_seqlen_k: int = 0,
                      logits_dtype: torch.dtype = torch.float32,
                      q_sf: Optional[torch.Tensor] = None,
                      k_sf: Optional[torch.Tensor] = None):
    """MQA logits (non-paged) for FP8/BF16/INT8/FP4.

    Args:
        q:     Q tensor.
               FP8/BF16/INT8: [seq_len_q, num_heads, head_dim], dtype = float8_e4m3fn/bfloat16/int8
               FP4:           [seq_len_q, num_heads, head_dim_packed], dtype = int8
                              where head_dim_packed = original_head_dim // 2 = 64
        k:     K tensor.
               FP8/BF16/INT8: [seq_len_k, head_dim], dtype = same as q
               FP4:           [seq_len_k, head_dim_packed], dtype = int8
        k_scales: Per-token KV scale factor.
               FP8/INT8: float32 [seq_len_k]
               BF16:     empty tensor (unused)
               FP4:      empty tensor (unused, k_sf is used instead)
        weights:           [seq_len_q, num_heads], float32 or bfloat16
        cu_seq_len_k_start: int32 [seq_len_q]
        cu_seq_len_k_end:   int32 [seq_len_q]
        clean_logits: whether to mask out-of-range logits with -inf
        max_seqlen_k: 0 for non-compressed, >0 for compressed mode
        logits_dtype: output logits dtype, torch.float32 or torch.bfloat16
        q_sf:  UE8M0 scale for Q (FP4 only), int32 [seq_len_q, num_heads]
               Packed from uint8 e8m0 (4×uint8 per int32). None for FP8/BF16/INT8.
        k_sf:  UE8M0 scale for K (FP4 only), int32 [seq_len_k]
               Packed from uint8 e8m0 (4×uint8 per int32). None for FP8/BF16/INT8.
    """
    is_fp4 = q_sf is not None
    num_sms = get_num_sms()
    stream = torch.cuda.current_stream()

    # --- Shape extraction & validation ---
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_k, head_dim_ = k.shape
    seq_len_, num_heads_ = weights.shape
    assert(seq_len_q == seq_len_)
    assert(num_heads == num_heads_)
    assert(cu_seq_len_k_start.size(0) == seq_len_q)
    assert(cu_seq_len_k_end.size(0) == seq_len_q)
    assert(q.is_contiguous() and k.is_contiguous())
    assert(weights.is_contiguous())
    assert(cu_seq_len_k_start.is_contiguous())
    assert(cu_seq_len_k_end.is_contiguous())
    assert(weights.dtype == torch.float32 or (logits_dtype == torch.bfloat16 and weights.dtype == torch.bfloat16))
    assert(logits_dtype in (torch.float32, torch.bfloat16))
    if logits_dtype == torch.float32:
        assert weights.dtype == torch.float32, "fp32 logits requires fp32 weights"
    assert(cu_seq_len_k_start.dtype == torch.int32)
    assert(cu_seq_len_k_end.dtype == torch.int32)

    if is_fp4:
        assert k_sf is not None
        assert(head_dim == head_dim_ and head_dim == 64)
        assert(num_heads == 32 or num_heads == 64)
        assert(k_sf.shape[0] == seq_len_k)
        assert(q_sf.is_contiguous() and k_sf.is_contiguous())
        assert(q.dtype == torch.int8 and k.dtype == torch.int8)
        assert(q_sf.dtype == torch.int32 and k_sf.dtype == torch.int32)
    else:
        assert(head_dim == head_dim_)
        assert(q.dtype == torch.float8_e4m3fn or q.dtype == torch.bfloat16 or q.dtype == torch.int8)
        assert(k.dtype == torch.float8_e4m3fn or k.dtype == torch.bfloat16 or k.dtype == torch.int8)
        assert(q.dtype == k.dtype)
        if q.dtype != torch.bfloat16:
            seq_len_kv_ = k_scales.shape[0]
            assert(seq_len_k == seq_len_kv_)
            assert(k_scales.is_contiguous())
            assert(k_scales.dtype == torch.float32)

    # --- Tile config ---
    block_q = 4
    block_qh = block_q * num_heads
    if is_fp4:
        if num_heads == 64 and logits_dtype == torch.bfloat16:
            if seq_len_k >= 8192:
                warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = num_heads, 256, 64, 1, 4
            else:
                warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = num_heads, 64, 16, 1, 4
        elif num_heads == 64 and logits_dtype == torch.float32:
            warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = num_heads, 256, 64, 1, 3
        elif num_heads == 32 and logits_dtype == torch.bfloat16:
            warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = num_heads, 64, 16, 1, 3
        elif num_heads == 32 and logits_dtype == torch.float32:
            warp_qh, block_kv, warp_kv, num_q_stages, num_kv_stages = num_heads, 64, 16, 1, 4
        ElementQK = "uint8_t"
        ElementAcc = "float"
    else:
        num_q_stages, num_kv_stages = 1, 3
        warp_qh = num_heads
        if q.dtype == torch.bfloat16:
            ElementQK, ElementAcc = 'cutlass::bfloat16_t', 'float'
            block_kv, warp_kv = 128, 32
        elif q.dtype == torch.float8_e4m3fn:
            ElementQK, ElementAcc = 'cutlass::float_e4m3_t', 'float'
            warp_kv = 32 if num_heads == 32 else 64
            block_kv = warp_kv * 4
        elif q.dtype == torch.int8:
            ElementQK, ElementAcc = 'int8_t', "int32_t"
            warp_kv = 32 if num_heads == 32 else 64
            block_kv = warp_kv * 4

    # --- Alignment & logits allocation (common) ---
    is_compressed = max_seqlen_k > 0
    if is_compressed:
        assert not clean_logits, "clean_logits must be False when compressed (max_seqlen_k > 0)"

    aligned_seq_len = align(seq_len_q, block_q)
    # Logits row stride must be 1024-byte aligned
    logits_stride_alignment = 1024 // (4 if logits_dtype == torch.float32 else 2)
    assert logits_stride_alignment % block_kv == 0
    if is_compressed:
        aligned_seq_len_kv = align(max_seqlen_k, logits_stride_alignment)
    else:
        aligned_seq_len_kv = align(seq_len_k + block_kv, logits_stride_alignment)
    # stride_k may be uint32_t or uint64_t depending on whether q_idx * stride_k
    # overflows 32-bit. Decided dynamically here and forwarded as a JIT template key.
    stride_k_type = _select_stride_k_type(aligned_seq_len, aligned_seq_len_kv)
    logits = torch.empty(aligned_seq_len, aligned_seq_len_kv, dtype=logits_dtype, device=q.device)
    if is_compressed:
        logits = logits[0:seq_len_q, 0:max_seqlen_k]
    else:
        logits = logits[0:seq_len_q, 0:seq_len_k]
    kWeightsBF16 = (logits_dtype == torch.bfloat16 and weights.dtype == torch.bfloat16)
    ElementWeights = "__ppu_bfloat16" if kWeightsBF16 else "float"
    ElementLogits = "float" if logits_dtype == torch.float32 else "__ppu_bfloat16"

    # --- JIT compilation ---
    jit_keys = {'ElementQK': ElementQK, 'ElementAcc': ElementAcc,
                'ElementLogits': ElementLogits, 'ElementWeights': ElementWeights,
                'kNumHeads': num_heads, 'kHeadDim': head_dim,
                'BLOCK_QH': block_qh, 'BLOCK_KV': block_kv,
                'WARP_QH': warp_qh, 'WARP_KV': warp_kv,
                'kNumQStages': num_q_stages, 'kNumKVStages': num_kv_stages,
                'StrideKType': stride_k_type, 'kIsCompressedLogits': is_compressed}

    if is_fp4:
        global includes_fp4_mqa, template_fp4_mqa
        args = (q.view(torch.uint8), q_sf, k.view(torch.uint8), k_sf, weights,
                cu_seq_len_k_start, cu_seq_len_k_end, logits,
                seq_len_q, seq_len_k, aligned_seq_len_kv, stream, num_sms)
        runtime = jit_tuner.compile_and_tune(
            name='attention_mqa_logits_fp4',
            keys=jit_keys, space=(),
            includes=includes_fp4_mqa,
            arg_defs=(('q', torch.uint8), ('q_sf', torch.int32), ('k', torch.uint8), ('k_sf', torch.int32), ('weights', weights.dtype),
                      ('cu_seq_len_k_start', torch.int32), ('cu_seq_len_k_end', torch.int32), ('logits', logits_dtype),
                      ('seq_len_q', int), ('seq_len_k', int), ('aligned_seq_len_kv', int),
                      ('stream', torch.cuda.Stream), ('num_sms', int)),
            template=template_fp4_mqa, args=args, jit_include_dir='actlize_v1.0.0')
    else:
        global includes, template
        args = (q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits,
                seq_len_q, seq_len_k, aligned_seq_len_kv, stream, num_sms)
        runtime = jit_tuner.compile_and_tune(
            name='attention_mqa_logits_' + ElementQK,
            keys=jit_keys, space=(),
            includes=includes,
            arg_defs=(('q', q.dtype), ('k', k.dtype), ('k_scales', torch.float), ('weights', weights.dtype),
                      ('cu_seq_len_k_start', torch.int32), ('cu_seq_len_k_end', torch.int32), ('logits', logits_dtype),
                      ('seq_len_q', int), ('seq_len_k', int), ('aligned_seq_len_kv', int),
                      ('stream', torch.cuda.Stream), ('num_sms', int)),
            template=template, args=args, jit_include_dir='actlize_v1.0.0')

    runtime(*args)

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
                    max_seqlen_k: int = 0,
                    logits_dtype: torch.dtype = torch.float32):
    k_scales = torch.empty(0)
    return mqa_logits_common(q, kv, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, max_seqlen_k, logits_dtype=logits_dtype)

def fp8_mqa_logits(q: torch.Tensor,
                   kv_s: Tuple[torch.Tensor],
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True,
                   max_seqlen_k: int = 0,
                   logits_dtype: torch.dtype = torch.float32):
    k, k_scales = kv_s
    return mqa_logits_common(q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, max_seqlen_k, logits_dtype=logits_dtype)

def int8_mqa_logits(q: torch.Tensor,
                   kv_s: Tuple[torch.Tensor],
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True,
                   max_seqlen_k: int = 0,
                   logits_dtype: torch.dtype = torch.float32):
    k, k_scales = kv_s
    return mqa_logits_common(q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, max_seqlen_k, logits_dtype=logits_dtype)


includes_metadata = ('"../deep_gemm/paged_mqa_logits_scheduler.cuh"', )
includes_paged = ('"../deep_gemm/ppu_paged_mqa_logits.cuh"', )
includes_paged_fp4 = ('"../deep_gemm/fp4_paged_mqa_logits.cuh"', )
template_paged_metadata = """
using namespace deep_gemm;
constexpr uint32_t SPLIT_KV = {SPLIT_KV};
constexpr uint32_t kNumSMs = {kNumSMs};
launch_paged_mqa_logits_metadata<SPLIT_KV, kNumSMs>(
    batch_size, next_n, (uint32_t*)context_lens, (uint32_t*)schedule_metadata, stream);
"""

template_paged = """
using namespace deep_gemm;
// Templated args from Python JIT call
using ElementQK = {ElementQK}; //cutlass::bfloat16_t or cutlass::float_e4m3_t
using ElementAcc = {ElementAcc};
using ElementLogits = {ElementLogits};  // float or __ppu_bfloat16
using ElementWeights = {ElementWeights};  // float or __ppu_bfloat16
constexpr uint32_t kNextN = {kNextN};
constexpr uint32_t kNumHeads = {kNumHeads};
constexpr uint32_t kHeadDim = {kHeadDim};
constexpr uint32_t BLOCK_KV = {BLOCK_KV};
constexpr uint32_t WARP_KV = {WARP_KV};
constexpr uint32_t kNumQStages = {kNumQStages};
constexpr uint32_t kNumKVStages = {kNumKVStages};
constexpr uint32_t SPLIT_KV = {SPLIT_KV};

// Make a templated GEMM
using atten_t = PagedAttention<ElementQK, ElementAcc, ElementLogits, ElementWeights, kNextN, kNumHeads, kHeadDim, BLOCK_KV, WARP_KV, kNumQStages, kNumKVStages, SPLIT_KV>;

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
using ElementWeights = {ElementWeights}; // float or __ppu_bfloat16
constexpr uint32_t kNextN = {kNextN};
constexpr uint32_t kNumHeads = {kNumHeads};
constexpr uint32_t kHeadDim = {kHeadDim};   // packed head_dim (64)
constexpr uint32_t BLOCK_KV = {BLOCK_KV};
constexpr uint32_t WARP_KV = {WARP_KV};
constexpr uint32_t kNumQStages = {kNumQStages};
constexpr uint32_t kNumKVStages = {kNumKVStages};
constexpr uint32_t SPLIT_KV = {SPLIT_KV};
constexpr bool SPLIT_MBLOCK = {SPLIT_MBLOCK};

// Make a templated GEMM
using atten_t = PagedAttentionFP4<ElementQK, ElementAcc, ElementLogits, ElementWeights, kNextN, kNumHeads, kHeadDim, BLOCK_KV, WARP_KV, kNumQStages, kNumKVStages, SPLIT_KV, SPLIT_MBLOCK>;

// Launch kernel
atten_t::run((const ElementQK*)q, (const uint32_t*)q_sf, (const ElementQK*)k, (const uint32_t*)k_scales, weights, batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride,
             (uint32_t*)context_lens, logits, (uint32_t*)block_table, (uint32_t*)schedule_meta, stream, num_sms, num_blocks);
"""

def get_paged_mqa_logits_metadata(context_lens: torch.Tensor,
                                  block_kv: int,
                                  num_sms: int,
                                  indices: Optional[torch.Tensor] = None,
                                  metadata_extra: Optional[tuple] = None):
    if isinstance(indices, tuple):
        assert False, "indices received a tuple — did you pass metadata_extra as a positional arg? Use metadata_extra=metadata_extra keyword arg instead."
    if indices is not None:
        print_once("Warning: indices (varlen) is not supported on PPU, falling back to non-varlen mode (performance may be affected)")
    assert context_lens.dim() == 2, "context_lens must be 2D [batch_size, next_n]"
    batch_size = context_lens.size(0)
    next_n = context_lens.size(1)
    assert(context_lens.dtype == torch.int32)
    assert(context_lens.is_contiguous())
    # shared memory limit
    assert(batch_size <= 65536)

    tb_per_cu = 1
    split_kv = block_kv * 4  # fallback: assume max num_math_warpgroups=4 to avoid crash
    if metadata_extra is not None:
        next_n, num_heads, head_dim, element_size = metadata_extra
        _, _, split_kv, _, _, tb_per_cu = get_paged_mqa_logits_tile(next_n, block_kv, num_heads, head_dim, element_size)
    else:
        print_once("Warning: metadata_extra is None on ppu, falling back to compatibility mode (performance may be affected)")
    num_blocks = num_sms * tb_per_cu
    schedule_metadata = torch.empty((num_blocks + 1, 2), dtype=context_lens.dtype, device=context_lens.device)

    stream = torch.cuda.current_stream()
    args = (batch_size, next_n, context_lens, schedule_metadata, stream)
    runtime = jit_tuner.compile_and_tune(
        name='attention_paged_mqa_logits_metadata',
        keys={'SPLIT_KV': split_kv,
              'kNumSMs': num_blocks},
        space=(),
        includes=includes_metadata,
        arg_defs=(('batch_size', int),
                  ('next_n', int),
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
                            logits_dtype: torch.dtype = torch.float32,
                            q_sf: Optional[torch.Tensor] = None):
    """Paged MQA logits for FP8/BF16/INT8/FP4.

    Args:
        q:     Q tensor.
               FP8/BF16/INT8: [batch, next_n, num_heads, head_dim], dtype = float8_e4m3fn/bfloat16/int8
               FP4:           [batch, next_n, num_heads, head_dim_packed], dtype = int8
                              where head_dim_packed = original_head_dim // 2 = 64
        fused_kv_cache: fused KV cache, uint8
               FP8/BF16/INT8: [num_kv_blocks, block_kv, 1, head_dim + scale_bytes]
               FP4:           [num_kv_blocks, block_kv, 1, head_dim_packed + scale_bytes]
               Per-row layout: [values (head_dim or head_dim_packed bytes), scale (scale_bytes bytes)]
        weights:       [batch * next_n, num_heads], float32 or bfloat16
        context_lens:  int32 [batch_size, next_n]
        block_table:   int32 [batch_size, max_block_len]
        schedule_meta: int32 [num_blocks+1, 2]
        max_context_len: maximum context length
        clean_logits:  whether to mask out-of-range logits with -inf
        logits_dtype:  output logits dtype, torch.float32 or torch.bfloat16
        q_sf:  UE8M0 scale for Q (FP4 only), int32 [batch, next_n, num_heads]
               Packed from uint8 e8m0 (4×uint8 per int32). None for FP8/BF16/INT8.
    """
    is_fp4 = q_sf is not None

    batch_size, next_n, num_heads, head_dim = q.shape
    num_kv_blocks, block_kv, num_heads_kv, kv_last_dim = fused_kv_cache.shape
    # Only 2D context_lens [batch_size, next_n] is supported
    assert context_lens.dim() == 2, "context_lens must be 2D [batch_size, next_n]"
    assert context_lens.size(1) == next_n, f"context_lens next_n={context_lens.size(1)} != q next_n={next_n}"
    batch_size_ = context_lens.size(0)
    batch_size_next_n, num_heads_ = weights.shape
    batch_size__, max_block_len = block_table.shape
    schedule_meta_size, meta_info_size = schedule_meta.shape
    kv_cache_stride_bytes = fused_kv_cache.stride(0)
    block_table_stride = block_table.stride(0)

    num_sms = get_num_sms()
    assert(batch_size == batch_size_ and batch_size == batch_size__)
    assert(batch_size_next_n == batch_size * next_n)
    assert(num_heads == num_heads_ and num_heads_kv == 1)
    assert((schedule_meta_size - 1) % num_sms == 0 and meta_info_size == 2)
    assert(1 <= next_n <= 6)
    assert(block_kv == 64)

    assert(q.is_contiguous())
    assert(fused_kv_cache.stride(1) == kv_last_dim)
    assert(fused_kv_cache.stride(2) == kv_last_dim)
    assert(fused_kv_cache.stride(3) == 1)
    assert(weights.is_contiguous())
    assert(weights.dtype == torch.float32 or (logits_dtype == torch.bfloat16 and weights.dtype == torch.bfloat16))
    assert(logits_dtype in (torch.float32, torch.bfloat16))
    if logits_dtype == torch.float32:
        assert weights.dtype == torch.float32, "fp32 logits requires fp32 weights"
    assert(context_lens.is_contiguous())
    assert(context_lens.dtype == torch.int32)
    assert(block_table.stride(1) == 1)
    assert(block_table.dtype == torch.int32)
    assert(schedule_meta.is_contiguous())
    assert(schedule_meta.dtype == torch.int32)

    if is_fp4:
        assert(head_dim == 64)
        scale_bytes = kv_last_dim - head_dim
        assert(scale_bytes == 4)
        assert(q.dtype == torch.int8)
        assert(q_sf.is_contiguous())
        assert(q_sf.dtype == torch.int32)
        assert(q_sf.shape == (batch_size, next_n, num_heads))
        assert(fused_kv_cache.dtype == torch.uint8)
        assert(kv_cache_stride_bytes % scale_bytes == 0)
    else:
        size_of_scale_float = 0 if q.dtype == torch.bfloat16 else 4
        assert(head_dim == 32 or head_dim == 64 or head_dim == 128)
        assert(kv_last_dim == head_dim + size_of_scale_float)
        assert(q.dtype == torch.float8_e4m3fn or q.dtype == torch.bfloat16 or q.dtype == torch.int8)
        if q.dtype == torch.bfloat16:
            assert(1 <= next_n <= 4)
        if q.dtype != torch.bfloat16:
            assert(kv_cache_stride_bytes % size_of_scale_float == 0)
            assert(fused_kv_cache.dtype == torch.uint8)
            if next_n > 4:
                print_once(f"Warning: fp8/int8 paged_mqa_logits with next_n = {next_n} > 4 on PPU may affect performance")

    # Derive values and SF tensor from KV cache
    k = torch.as_strided(
        input=fused_kv_cache,
        size=(num_kv_blocks, block_kv, head_dim),
        stride=(kv_cache_stride_bytes, head_dim, 1),
    )
    if not is_fp4:
        k = k.view(dtype=q.dtype)

    if is_fp4:
        k_scales = torch.as_strided(
            input=fused_kv_cache,
            size=(num_kv_blocks, block_kv * 4),
            stride=(kv_cache_stride_bytes, 1),
            storage_offset=block_kv * head_dim,
        ).view(torch.int32)
    elif q.dtype == torch.bfloat16:
        k_scales = torch.empty(0)
    else:
        k_scales = torch.as_strided(
            input=fused_kv_cache,
            size=(num_kv_blocks, block_kv * 4),
            stride=(kv_cache_stride_bytes, 1),
            storage_offset=block_kv * head_dim,
        ).view(dtype=torch.float)

    num_q_stages, num_kv_stages, split_kv, warp_kv, split_mblock, _ = get_paged_mqa_logits_tile(next_n, block_kv, num_heads, head_dim, q.element_size())
    # Logits row stride must be 1024-byte aligned
    logits_stride_alignment = 1024 // (4 if logits_dtype == torch.float32 else 2)
    assert logits_stride_alignment % split_kv == 0
    aligned_max_context_len = align(max_context_len, logits_stride_alignment)
    logits = torch.empty((batch_size * next_n, aligned_max_context_len), dtype=logits_dtype, device=q.device)
    logits = logits[..., :max_context_len]

    logits_stride = aligned_max_context_len

    ElementLogits = "float" if logits_dtype == torch.float32 else "__ppu_bfloat16"
    kWeightsBF16 = (logits_dtype == torch.bfloat16 and weights.dtype == torch.bfloat16)
    ElementWeights = "__ppu_bfloat16" if kWeightsBF16 else "float"
    stream = torch.cuda.current_stream()

    jit_keys = {'ElementLogits': ElementLogits, 'ElementWeights': ElementWeights,
                'kNextN': next_n, 'kNumHeads': num_heads,
                'kHeadDim': head_dim, 'BLOCK_KV': block_kv, 'WARP_KV': warp_kv,
                'kNumQStages': num_q_stages, 'kNumKVStages': num_kv_stages,
                'SPLIT_KV': split_kv}

    if not is_fp4:
        assert not split_mblock, "SPLIT_MBLOCK is only supported for FP4 paged kernel"

    if is_fp4:
        global includes_paged_fp4, template_paged_fp4
        ElementQK = "uint8_t"
        ElementAcc = "float"
        args = (q.view(torch.uint8), q_sf, k, k_scales, weights, batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride, context_lens, logits,
                block_table, schedule_meta, stream, num_sms, schedule_meta_size - 1)
        runtime = jit_tuner.compile_and_tune(
            name='attention_paged_mqa_logits_fp4',
            keys={**jit_keys, 'ElementQK': ElementQK, 'ElementAcc': ElementAcc, 'SPLIT_MBLOCK': split_mblock},
            space=(),
            includes=includes_paged_fp4,
            arg_defs=(('q', torch.uint8), ('q_sf', torch.int32), ('k', torch.uint8), ('k_scales', torch.int32), ('weights', weights.dtype),
                      ('batch_size', int), ('logits_stride', int), ('kv_cache_stride_bytes', int), ('block_table_stride', int), ('context_lens', torch.int32),
                      ('logits', logits_dtype), ('block_table', torch.int32), ('schedule_meta', torch.int32),
                      ('stream', torch.cuda.Stream), ('num_sms', int), ('num_blocks', int)),
            template=template_paged_fp4, args=args, jit_include_dir='actlize_v1.0.0')
    else:
        global includes_paged, template_paged
        ElementQK = "cutlass::float_e4m3_t"
        ElementAcc = "float"
        if q.dtype == torch.bfloat16:
            ElementQK = 'cutlass::bfloat16_t'
        elif q.dtype == torch.int8:
            ElementQK = 'int8_t'
            ElementAcc = "int32_t"
        args = (q, k, k_scales, weights, batch_size, logits_stride, kv_cache_stride_bytes, block_table_stride, context_lens, logits,
                block_table, schedule_meta, stream, num_sms, schedule_meta_size - 1)
        runtime = jit_tuner.compile_and_tune(
            name='attention_paged_mqa_logits_' + ElementQK,
            keys={**jit_keys, 'ElementQK': ElementQK, 'ElementAcc': ElementAcc},
            space=(),
            includes=includes_paged,
            arg_defs=(('q', q.dtype), ('k', k.dtype), ('k_scales', torch.float), ('weights', weights.dtype),
                      ('batch_size', int), ('logits_stride', int), ('kv_cache_stride_bytes', int), ('block_table_stride', int), ('context_lens', torch.int32),
                      ('logits', logits_dtype), ('block_table', torch.int32), ('schedule_meta', torch.int32),
                      ('stream', torch.cuda.Stream), ('num_sms', int), ('num_blocks', int)),
            template=template_paged, args=args, jit_include_dir='actlize_v1.0.0')

    runtime(*args)

    if clean_logits:
        assert False, "clean_logits not supported with 2D context_lens, use external masking"
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

    if is_fp4:
        return mqa_logits_common(q_fp, kv_fp, torch.empty(0), weights,
                                cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, max_seqlen_k,
                                logits_dtype=logits_dtype, q_sf=q_sf, k_sf=kv_sf)
    else:
        return mqa_logits_common(q_fp, kv_fp, kv_sf, weights,
                                cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, max_seqlen_k,
                                logits_dtype=logits_dtype)


def fp8_fp4_paged_mqa_logits(q: Tuple,
                             fused_kv_cache: torch.Tensor,
                             weights: torch.Tensor,
                             context_lens: torch.Tensor,
                             block_table: torch.Tensor,
                             schedule_meta: torch.Tensor,
                             max_context_len: int,
                             clean_logits: bool = False,
                             logits_dtype: torch.dtype = torch.float32,
                             indices: Optional[torch.Tensor] = None):
    q_fp, q_sf = q
    if indices is not None:
        print_once("Warning: indices (varlen) is not supported on PPU, falling back to non-varlen mode (performance may be affected)")

    return paged_mqa_logits_common(q_fp, fused_kv_cache, weights, context_lens,
                                   block_table, schedule_meta, max_context_len, clean_logits,
                                   logits_dtype=logits_dtype, q_sf=q_sf)
