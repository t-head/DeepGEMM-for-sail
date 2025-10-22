import math
import torch
from functools import lru_cache
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout, get_extra_info

# C++ code templates
includes = ('"../deep_gemm/fp8_mqa_logits.cuh"', )
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

def mqa_logits_common(q: torch.Tensor, q_scales: torch.Tensor,
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

    weights_update = weights * q_scales if q.dtype == torch.int8 else weights

    if q.dtype == torch.int8:
        assert(q_scales != None)
        assert(q_scales.is_contiguous())
        assert(q_scales.size(0) == weights.size(0))

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

    def align(value, alignment):
        return (value + alignment - 1) // alignment * alignment

    # defalut tile config for fp8 and int8
    block_qh = 256
    block_kv = 256
    warp_qh = 64
    warp_kv = 64
    num_q_stages = 3
    num_kv_stages = 3
    block_q = block_qh / num_heads
    assert(block_qh % num_heads == 0)

    seq_len_alignment = 4
    assert(seq_len_alignment % block_q == 0)
    aligned_seq_len = align(seq_len_q, seq_len_alignment)
    aligned_seq_len_kv = align(seq_len_k + block_kv, 4)
    logits = torch.zeros(aligned_seq_len, aligned_seq_len_kv, dtype=torch.float, device=q.device)
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
    args = (q, k, k_scales, weights_update, cu_seq_len_k_start, cu_seq_len_k_end, logits,
            seq_len_q, seq_len_k, aligned_seq_len_kv, stream, num_sms)
    runtime = jit_tuner.compile_and_tune(
        name='attention_mqa_logits_fp8',
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
    q_scales = None
    k_scales = torch.empty(0)
    return mqa_logits_common(q, q_scales, kv, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits)

def fp8_mqa_logits(q: torch.Tensor,
                   kv_s: Tuple[torch.Tensor],
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True):
    k, k_scales = kv_s
    q_scales = None
    return mqa_logits_common(q, q_scales, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits)

def int8_mqa_logits(q_s: Tuple[torch.Tensor],
                   kv_s: Tuple[torch.Tensor],
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True):
    q, q_scales = q_s
    k, k_scales = kv_s
    return mqa_logits_common(q, q_scales, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits)

def get_paged_mqa_logits_metadata():
    return

def fp8_paged_mqa_logits():
    return