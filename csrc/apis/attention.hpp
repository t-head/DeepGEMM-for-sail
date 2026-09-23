#pragma once

#include "../utils/compatibility.hpp"
#include <torch/extension.h>

#include <mutex>
#include <set>
#include <string>

#include "../jit_kernels/impls/mqa_logits.hpp"
#include "../jit_kernels/impls/paged_mqa_logits.hpp"
#include "../jit_kernels/impls/sparse_mqa_logits.hpp"

namespace deep_gemm::attention {

// Prints each distinct message only once, in yellow -- mirrors `print_once` in
// `deep_gemm/jit_kernels/attention.py` (which uses `@lru_cache` for the same effect)
static void print_once(const std::string& msg) {
    static std::mutex mutex;
    static std::set<std::string> printed;
    const std::lock_guard<std::mutex> lock(mutex);
    if (printed.insert(msg).second)
        printf("\033[33m%s\033[0m\n", msg.c_str());
}

// MQA logits (non-paged) for FP8/BF16/INT8/FP4. Equivalent of `mqa_logits_common` in
// `deep_gemm/jit_kernels/attention.py`.
//
// Allocates the logits tensor (its row stride must be 1024-byte aligned), launches the kernel and
// optionally masks out-of-range entries. Returns the (sliced) logits.
//
// Args:
//   q:        Q tensor.
//             FP8/BF16/INT8: [seq_len_q, num_heads, head_dim], dtype = float8_e4m3fn/bfloat16/int8
//             FP4:           [seq_len_q, num_heads, head_dim_packed], dtype = int8,
//                            where head_dim_packed = original_head_dim / 2 = 64
//   k:        K tensor.
//             FP8/BF16/INT8: [seq_len_k, head_dim], dtype = same as q
//             FP4:           [seq_len_k, head_dim_packed], dtype = int8
//   k_scales: per-token KV scale factor.
//             FP8/INT8: float32 [seq_len_k]
//             BF16:     empty tensor (unused)
//             FP4:      empty tensor (unused, `k_sf` is used instead)
//   weights:            [seq_len_q, num_heads], float32 or bfloat16
//   cu_seq_len_k_start: int32 [seq_len_q]
//   cu_seq_len_k_end:   int32 [seq_len_q]
//   clean_logits: whether to mask out-of-range logits with -inf
//   max_seqlen_k: 0 for non-compressed, >0 for compressed mode
//   logits_dtype: output logits dtype, float32 or bfloat16
//   q_sf:     UE8M0 scale for Q (FP4 only), int32 [seq_len_q, num_heads].
//             Packed from uint8 e8m0 (4x uint8 per int32). `nullopt` for FP8/BF16/INT8.
//   k_sf:     UE8M0 scale for K (FP4 only), int32 [seq_len_k]. Same packing as `q_sf`.
static torch::Tensor mqa_logits_common(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& k_scales,
                                       const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                                       const torch::Tensor& cu_seq_len_k_end, bool clean_logits, int max_seqlen_k,
                                       torch::ScalarType logits_dtype,
                                       const std::optional<torch::Tensor>& q_sf = std::nullopt,
                                       const std::optional<torch::Tensor>& k_sf = std::nullopt,
                                       const std::optional<torch::Tensor>& q_scale = std::nullopt) {
    const bool is_fp4 = q_sf.has_value();
    // Shape extraction & validation
    const auto& [seq_len_q, num_heads, head_dim] = get_shape<3>(q);
    const auto& [seq_len_k, head_dim_] = get_shape<2>(k);

    // Avg variant: an empty weights tensor selects it (mirrors the Python entry);
    // the optional q_scale rides in the kernel's weights slot
    const bool is_avg = weights.numel() == 0;
    if (is_avg) {
        TORCH_CHECK(not is_fp4, "avg variant supports fp8 only");
        TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "avg variant supports fp8 only");
        TORCH_CHECK(logits_dtype == torch::kFloat32 || logits_dtype == torch::kBFloat16,
                    "avg variant supports float32 or bfloat16 logits");
        if (q_scale.has_value()) {
            TORCH_CHECK(q_scale->scalar_type() == torch::kFloat32, "q_scale must be float32");
            TORCH_CHECK(q_scale->is_contiguous(), "q_scale must be contiguous");
            TORCH_CHECK(q_scale->numel() == 1 or (q_scale->dim() == 1 and q_scale->numel() == seq_len_q),
                        "q_scale must be a scalar or a [seq_len_q] vector");
        }
    } else {
        const auto& [seq_len_, num_heads_] = get_shape<2>(weights);
        DG_HOST_ASSERT(seq_len_q == seq_len_);
        DG_HOST_ASSERT(num_heads == num_heads_);
        TORCH_CHECK(num_heads != 4, "num_heads == 4 supports the avg variant only (pass empty weights)");
        TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
        DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat32 or
                       (logits_dtype == torch::kBFloat16 and weights.scalar_type() == torch::kBFloat16));
        if (logits_dtype == torch::kFloat32)
            TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "fp32 logits requires fp32 weights");
    }
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len_q);
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len_q);
    TORCH_CHECK(q.is_contiguous() and k.is_contiguous(), "q and k must be contiguous");
    TORCH_CHECK(cu_seq_len_k_start.is_contiguous(), "cu_seq_len_k_start must be contiguous");
    TORCH_CHECK(cu_seq_len_k_end.is_contiguous(), "cu_seq_len_k_end must be contiguous");
    DG_HOST_ASSERT(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16);
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt32);

    const auto& qk_dtype = q.scalar_type();
    if (is_fp4) {
        DG_HOST_ASSERT(k_sf.has_value());
        DG_HOST_ASSERT(head_dim == head_dim_ and head_dim == 64);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(k_sf->size(0) == seq_len_k);
        TORCH_CHECK(q_sf->is_contiguous() and k_sf->is_contiguous(), "q_sf and k_sf must be contiguous");
        DG_HOST_ASSERT(qk_dtype == torch::kInt8 and k.scalar_type() == torch::kInt8);
        DG_HOST_ASSERT(q_sf->scalar_type() == torch::kInt32 and k_sf->scalar_type() == torch::kInt32);
    } else {
        DG_HOST_ASSERT(head_dim == head_dim_);
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == torch::kBFloat16 or qk_dtype == torch::kInt8);
        DG_HOST_ASSERT(qk_dtype == k.scalar_type());
        if (qk_dtype != torch::kBFloat16) {
            DG_HOST_ASSERT(k_scales.size(0) == seq_len_k);
            TORCH_CHECK(k_scales.is_contiguous(), "k_scales must be contiguous");
            DG_HOST_ASSERT(k_scales.scalar_type() == torch::kFloat32);
        }
    }

    const bool is_compressed = max_seqlen_k > 0;
    if (is_compressed)
        TORCH_CHECK(not clean_logits, "clean_logits must be False when compressed (max_seqlen_k > 0)");

    return mqa_logits(q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, seq_len_q, seq_len_k, num_heads,
                      head_dim, clean_logits, max_seqlen_k, logits_dtype, q_sf, k_sf, q_scale);
}

// Paged MQA logits for FP8/BF16/INT8/FP4. Equivalent of `paged_mqa_logits_common` in
// `deep_gemm/jit_kernels/attention.py`.
//
// Args:
//   q:              Q tensor.
//                   FP8/BF16/INT8: [batch, next_n, num_heads, head_dim],
//                                  dtype = float8_e4m3fn/bfloat16/int8
//                   FP4:           [batch, next_n, num_heads, head_dim_packed], dtype = int8,
//                                  where head_dim_packed = original_head_dim / 2 = 64
//   fused_kv_cache: fused KV cache, uint8.
//                   FP8/BF16/INT8: [num_kv_blocks, block_kv, 1, head_dim + scale_bytes]
//                   FP4:           [num_kv_blocks, block_kv, 1, head_dim_packed + scale_bytes]
//                   Per-row layout: [values (head_dim or head_dim_packed bytes),
//                                    scale (scale_bytes bytes)]
//                   The value and scale views are recovered here via `as_strided`.
//   weights:        [batch * next_n, num_heads], float32 or bfloat16
//   context_lens:   int32 [batch_size, next_n]
//   block_table:    int32 [batch_size, max_block_len]
//   schedule_meta:  int32 [num_blocks+1, 2], produced by `get_paged_mqa_logits_metadata`
//   max_context_len: maximum context length
//   clean_logits:   whether to mask out-of-range logits with -inf (unsupported here, see below)
//   logits_dtype:   output logits dtype, float32 or bfloat16
//   q_sf:           UE8M0 scale for Q (FP4 only), int32 [batch, next_n, num_heads].
//                   Packed from uint8 e8m0 (4x uint8 per int32). `nullopt` for FP8/BF16/INT8.
static torch::Tensor paged_mqa_logits_common(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                            const torch::Tensor& weights, const torch::Tensor& context_lens,
                                            const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                            int max_context_len, bool clean_logits, torch::ScalarType logits_dtype,
                                            const std::optional<torch::Tensor>& q_sf = std::nullopt) {
    const bool is_fp4 = q_sf.has_value();

    const auto& [batch_size, next_n, num_heads, head_dim] = get_shape<4>(q);
    const auto& [num_kv_blocks, block_kv, num_heads_kv, kv_last_dim] = get_shape<4>(fused_kv_cache);
    DG_HOST_ASSERT(context_lens.dim() == 2);
    DG_HOST_ASSERT(context_lens.size(1) == next_n);
    const auto& [schedule_meta_size, meta_info_size] = get_shape<2>(schedule_meta);
    const int64_t kv_cache_stride_bytes = fused_kv_cache.stride(0);

    // Avg variant: an empty weights tensor selects it (mirrors the Python entry);
    // 4-head tiles are padded to 16 inside the kernel and are avg-only
    const bool is_avg = weights.numel() == 0;
    if (is_avg) {
        TORCH_CHECK(not is_fp4, "avg variant supports fp8 only");
        TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "avg variant supports fp8 only");
        TORCH_CHECK(logits_dtype == torch::kFloat32 || logits_dtype == torch::kBFloat16,
                    "avg variant supports float32 or bfloat16 logits");
    } else if (num_heads == 4) {
        TORCH_CHECK(false, "num_heads == 4 supports the avg variant only (pass empty weights)");
    }

    const int num_sms = get_num_sms();
    DG_HOST_ASSERT(batch_size == context_lens.size(0) and batch_size == block_table.size(0));
    if (not is_avg) {
        const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
        DG_HOST_ASSERT(batch_size_next_n == batch_size * next_n);
        DG_HOST_ASSERT(num_heads == num_heads_);
        TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
        DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat32 or
                       (logits_dtype == torch::kBFloat16 and weights.scalar_type() == torch::kBFloat16));
        if (logits_dtype == torch::kFloat32)
            TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "fp32 logits requires fp32 weights");
    }
    DG_HOST_ASSERT(num_heads_kv == 1);
    DG_HOST_ASSERT((schedule_meta_size - 1) % num_sms == 0 and meta_info_size == 2);
    DG_HOST_ASSERT(1 <= next_n and next_n <= 6);
    DG_HOST_ASSERT(block_kv == 64);

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == kv_last_dim);
    DG_HOST_ASSERT(fused_kv_cache.stride(2) == kv_last_dim);
    DG_HOST_ASSERT(fused_kv_cache.stride(3) == 1);
    DG_HOST_ASSERT(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16);
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt32);
    TORCH_CHECK(schedule_meta.is_contiguous(), "schedule_meta must be contiguous");
    DG_HOST_ASSERT(schedule_meta.scalar_type() == torch::kInt32);

    const auto& qk_dtype = q.scalar_type();
    if (is_fp4) {
        DG_HOST_ASSERT(head_dim == 64);
        DG_HOST_ASSERT(kv_last_dim - head_dim == 4);
        DG_HOST_ASSERT(qk_dtype == torch::kInt8);
        TORCH_CHECK(q_sf->is_contiguous(), "q_sf must be contiguous");
        DG_HOST_ASSERT(q_sf->scalar_type() == torch::kInt32);
        DG_HOST_ASSERT((q_sf->sizes() == std::vector<int64_t>{batch_size, next_n, num_heads}));
        DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kUInt8);
        DG_HOST_ASSERT(kv_cache_stride_bytes % 4 == 0);
    } else {
        const int size_of_scale_float = qk_dtype == torch::kBFloat16 ? 0 : 4;
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        DG_HOST_ASSERT(kv_last_dim == head_dim + size_of_scale_float);
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == torch::kBFloat16 or qk_dtype == torch::kInt8);
        if (qk_dtype == torch::kBFloat16) {
            DG_HOST_ASSERT(1 <= next_n and next_n <= 4);
        } else {
            DG_HOST_ASSERT(kv_cache_stride_bytes % size_of_scale_float == 0);
            DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kUInt8);
            if (next_n > 4)
                print_once(fmt::format("Warning: fp8/int8 paged_mqa_logits with next_n = {} > 4 on PPU may affect "
                                       "performance",
                                       next_n));
        }
    }

    TORCH_CHECK(not clean_logits, "clean_logits not supported with 2D context_lens, use external masking");

    return paged_mqa_logits(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, batch_size, next_n,
                            num_heads, head_dim, num_kv_blocks, block_kv, schedule_meta_size, max_context_len,
                            logits_dtype, q_sf);
}

extern "C" {

torch::Tensor bf16_mqa_logits(const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& weights,
                              const torch::Tensor& cu_seq_len_k_start, const torch::Tensor& cu_seq_len_k_end,
                              bool clean_logits = true, int max_seqlen_k = 0,
                              torch::ScalarType logits_dtype = torch::kFloat32) {
    DG_HOST_ASSERT(q.scalar_type() == torch::kBFloat16);
    // BF16 carries no per-token KV scale
    const auto& k_scales = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    return mqa_logits_common(q, kv, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                             max_seqlen_k, logits_dtype);
}

torch::Tensor fp8_mqa_logits(const torch::Tensor& q, const std::pair<torch::Tensor, torch::Tensor>& kv_s,
                             const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                             const torch::Tensor& cu_seq_len_k_end, bool clean_logits = true, int max_seqlen_k = 0,
                             torch::ScalarType logits_dtype = torch::kFloat32) {
    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    return mqa_logits_common(q, kv_s.first, kv_s.second, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                             max_seqlen_k, logits_dtype);
}

torch::Tensor int8_mqa_logits(const torch::Tensor& q, const std::pair<torch::Tensor, torch::Tensor>& kv_s,
                              const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                              const torch::Tensor& cu_seq_len_k_end, bool clean_logits = true, int max_seqlen_k = 0,
                              torch::ScalarType logits_dtype = torch::kFloat32) {
    DG_HOST_ASSERT(q.scalar_type() == torch::kInt8);
    return mqa_logits_common(q, kv_s.first, kv_s.second, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                             max_seqlen_k, logits_dtype);
}

// Unified FP8/FP4 entry point (non-paged).
//   q  = (q_fp, optional q_sf)
//          FP8 mode: `q_fp` is float8_e4m3fn, `q_sf` is absent
//          FP4 mode: `q_fp` is packed FP4 (int8), `q_sf` is the UE8M0 scale factor (int32)
//   kv = (kv_fp, kv_sf)
//          FP8 mode: `kv_fp` is float8_e4m3fn, `kv_sf` is a per-token float32 scale
//          FP4 mode: `kv_fp` is packed FP4 (int8), `kv_sf` is the UE8M0 scale factor (int32)
//   logits_dtype: output dtype, float32 or bfloat16
torch::Tensor fp8_fp4_mqa_logits(const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
                                 const std::pair<torch::Tensor, torch::Tensor>& kv, const torch::Tensor& weights,
                                 const torch::Tensor& cu_seq_len_k_start, const torch::Tensor& cu_seq_len_k_end,
                                 bool clean_logits = true, int max_seqlen_k = 0,
                                 torch::ScalarType logits_dtype = torch::kFloat32) {
    if (q.second.has_value()) {
        DG_HOST_ASSERT(q.first.scalar_type() == torch::kInt8);
        const auto& empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(q.first.device()));
        return mqa_logits_common(q.first, kv.first, empty, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                                 max_seqlen_k, logits_dtype, q.second, kv.second);
    }
    DG_HOST_ASSERT(q.first.scalar_type() == torch::kFloat8_e4m3fn);
    return mqa_logits_common(q.first, kv.first, kv.second, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                             max_seqlen_k, logits_dtype);
}

// sum_h(ReLU(dot(q_h, k))) * k_scale / sqrt(head_dim), with an optional q_scale
// (scalar broadcast or per-Q-row vector) folded in after the head reduction
torch::Tensor fp8_mqa_avg_logits(const torch::Tensor& q, const std::pair<torch::Tensor, torch::Tensor>& kv_s,
                                 const torch::Tensor& cu_seq_len_k_start, const torch::Tensor& cu_seq_len_k_end,
                                 bool clean_logits = true, int max_seqlen_k = 0,
                                 const std::optional<torch::Tensor>& q_scale = std::nullopt,
                                 torch::ScalarType logits_dtype = torch::kFloat32) {
    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    const auto& empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    return mqa_logits_common(q, kv_s.first, kv_s.second, empty, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                             max_seqlen_k, logits_dtype, std::nullopt, std::nullopt, q_scale);
}

// Paged counterpart of `fp8_mqa_avg_logits` (unity mode -- no q_scale, matching PAI)
torch::Tensor fp8_paged_mqa_avg_logits(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                       const torch::Tensor& context_lens, const torch::Tensor& block_table,
                                       const torch::Tensor& schedule_meta, int max_context_len,
                                       bool clean_logits = false,
                                       torch::ScalarType logits_dtype = torch::kFloat32,
                                       const std::optional<torch::Tensor>& indices = std::nullopt) {
    if (indices.has_value())
        print_once("Warning: indices (varlen) is not supported on PPU, falling back to non-varlen mode "
                   "(performance may be affected)");
    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    const auto& empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    return paged_mqa_logits_common(q, fused_kv_cache, empty, context_lens, block_table, schedule_meta,
                                   max_context_len, clean_logits, logits_dtype);
}

// Builds the `schedule_metadata` table. `metadata_extra` = (next_n, num_heads, head_dim, element_size);
// when absent we fall back to the compatibility tile, mirroring the Python path.
//
// NOTES: the Python version also asserts that `indices` is not a tuple, to catch `metadata_extra`
// being passed positionally. That check is unnecessary here -- pybind rejects a tuple for
// `std::optional<torch::Tensor>` with a TypeError before the body runs.
torch::Tensor get_paged_mqa_logits_metadata(
    const torch::Tensor& context_lens, int block_kv, int num_sms, std::optional<torch::Tensor> indices = std::nullopt,
    std::optional<std::tuple<int, int, int, int>> metadata_extra = std::nullopt) {
    if (indices.has_value())
        print_once("Warning: indices (varlen) is not supported on PPU, falling back to non-varlen mode "
                   "(performance may be affected)");
    if (not metadata_extra.has_value())
        print_once("Warning: metadata_extra is None on ppu, falling back to compatibility mode "
                   "(performance may be affected)");
    DG_HOST_ASSERT(context_lens.dim() == 2);
    const int batch_size = context_lens.size(0);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");
    // shared memory limit
    DG_HOST_ASSERT(batch_size <= 65536);

    return paged_mqa_logits_metadata(context_lens, batch_size, block_kv, num_sms, metadata_extra);
}

torch::Tensor bf16_paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                    const torch::Tensor& weights, const torch::Tensor& context_lens,
                                    const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                    int max_context_len, bool clean_logits = true,
                                    torch::ScalarType logits_dtype = torch::kFloat32) {
    DG_HOST_ASSERT(q.scalar_type() == torch::kBFloat16);
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                   max_context_len, clean_logits, logits_dtype);
}

torch::Tensor fp8_paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                   const torch::Tensor& weights, const torch::Tensor& context_lens,
                                   const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                   int max_context_len, bool clean_logits = true,
                                   torch::ScalarType logits_dtype = torch::kFloat32) {
    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                   max_context_len, clean_logits, logits_dtype);
}

torch::Tensor int8_paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                    const torch::Tensor& weights, const torch::Tensor& context_lens,
                                    const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                    int max_context_len, bool clean_logits = true,
                                    torch::ScalarType logits_dtype = torch::kFloat32) {
    DG_HOST_ASSERT(q.scalar_type() == torch::kInt8);
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                   max_context_len, clean_logits, logits_dtype);
}

// Unified FP8/FP4 entry point (paged). `q = (q_fp, optional q_sf)`: presence of `q_sf` selects the
// FP4 kernel, otherwise the FP8 one. See `paged_mqa_logits_common` for tensor layouts.
// `indices` (varlen) is accepted for API compatibility but unsupported on PPU.
torch::Tensor fp8_fp4_paged_mqa_logits(const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
                                       const torch::Tensor& fused_kv_cache, const torch::Tensor& weights,
                                       const torch::Tensor& context_lens, const torch::Tensor& block_table,
                                       const torch::Tensor& schedule_meta, int max_context_len,
                                       bool clean_logits = false,
                                       torch::ScalarType logits_dtype = torch::kFloat32,
                                       std::optional<torch::Tensor> indices = std::nullopt) {
    if (indices.has_value())
        print_once("Warning: indices (varlen) is not supported on PPU, falling back to non-varlen mode "
                   "(performance may be affected)");
    if (q.second.has_value())
        DG_HOST_ASSERT(q.first.scalar_type() == torch::kInt8);
    else
        DG_HOST_ASSERT(q.first.scalar_type() == torch::kFloat8_e4m3fn);
    return paged_mqa_logits_common(q.first, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                   max_context_len, clean_logits, logits_dtype, q.second);
}

// ---- Sparse MQA logits (DeepSeek V4.1 DSA indexer; ported from open-source 26/09) ----
// MXFP4 only.
//
// `sparse_kv_block_indices[num_q_tokens, num_max_sparse_blocks]` holds, per token, a
// strictly-increasing prefix of its selected absolute KV block ids (each block covers
// `sparse_block_kv` tokens anchored to the global token grid; with `use_unaligned_ks` the anchor
// carries a `ks % sparse_block_kv` offset). The valid count is inferred from the KV length:
// min(num_max_sparse_blocks, ceil_div(kv_end - kv_start, sparse_block_kv)); the tail padding is
// never read. Output logits are slot-compacted bf16: column = slot * sparse_block_kv + token_in_block.
torch::Tensor get_sparse_mqa_logits_metadata(const torch::Tensor& cu_seq_len_k_start,
                                             const torch::Tensor& cu_seq_len_k_end,
                                             int num_kv_tokens,
                                             const torch::Tensor& sparse_kv_block_indices,
                                             torch::ScalarType qk_dtype,
                                             int sparse_block_kv, bool use_unaligned_ks = false) {
    const int num_q_tokens = static_cast<int>(cu_seq_len_k_start.size(0));
    DG_HOST_ASSERT(num_q_tokens > 0 and cu_seq_len_k_end.size(0) == num_q_tokens);
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt32 and cu_seq_len_k_end.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous() and cu_seq_len_k_end.is_contiguous());
    DG_HOST_ASSERT(sparse_kv_block_indices.dim() == 2 and sparse_kv_block_indices.size(0) == num_q_tokens);
    DG_HOST_ASSERT(sparse_kv_block_indices.scalar_type() == torch::kInt32 and sparse_kv_block_indices.is_contiguous());
    DG_HOST_ASSERT(num_kv_tokens > 0);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    sparse_mqa_logits::get_sparse_split_kv(qk_dtype);  // MXFP4 only.
    const int num_max_sparse_blocks = static_cast<int>(sparse_kv_block_indices.size(1));
    const int64_t num_metadata_bytes = sparse_mqa_logits::get_num_sparse_metadata_bytes(
        num_q_tokens, num_max_sparse_blocks, sparse_block_kv, sparse_mqa_logits::get_num_sparse_slots());
    auto metadata = torch::empty({num_metadata_bytes},
                                 sparse_kv_block_indices.options().dtype(torch::kUInt8));
    // Cross-CTA coordination: WorkspaceState counters + one QBlockInfo per token. Process-wide
    // cache, self-cleaning (the kernel resets the counters at the end).
    const int64_t num_workspace_bytes =
        static_cast<int64_t>(sizeof(sparse_mqa_logits::WorkspaceState)) +
        static_cast<int64_t>(num_q_tokens) * sizeof(sparse_mqa_logits::QBlockInfo);
    auto& workspace = sparse_mqa_logits::get_sparse_workspace(metadata.options(), num_workspace_bytes);
    sparse_mqa_logits::launch_sparse_mqa_logits_metadata(
        metadata, workspace, static_cast<uint32_t>(num_q_tokens), static_cast<uint32_t>(num_kv_tokens),
        static_cast<uint32_t>(num_max_sparse_blocks), static_cast<uint32_t>(sparse_block_kv),
        reinterpret_cast<const uint32_t*>(cu_seq_len_k_start.data_ptr<int>()),
        reinterpret_cast<const uint32_t*>(cu_seq_len_k_end.data_ptr<int>()),
        reinterpret_cast<const uint32_t*>(sparse_kv_block_indices.data_ptr<int>()),
        use_unaligned_ks);
    return metadata;
}

// Paged counterpart: KV windows are [0, context_lens[q]) per token; the physical location of a
// selected logical block resolves through `block_table` (one row per Q token). `indices[q]` is
// the request id of token q — Q blocks (kBlockQ = 2) only pair tokens of the same request. No
// unaligned-ks support (pages are whole sparse blocks by construction).
torch::Tensor get_paged_sparse_mqa_logits_metadata(const torch::Tensor& context_lens,
                                                   const torch::Tensor& block_table,
                                                   const torch::Tensor& indices, int page_kv,
                                                   const torch::Tensor& sparse_kv_block_indices,
                                                   torch::ScalarType qk_dtype, int sparse_block_kv) {
    const int num_q_tokens = static_cast<int>(context_lens.size(0));
    DG_HOST_ASSERT(num_q_tokens > 0 and block_table.size(0) == num_q_tokens and indices.size(0) == num_q_tokens);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32 and context_lens.is_contiguous());
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt32 and block_table.is_contiguous() and
                   block_table.dim() == 2);
    DG_HOST_ASSERT(indices.scalar_type() == torch::kInt32 and indices.is_contiguous());
    DG_HOST_ASSERT(sparse_kv_block_indices.dim() == 2 and sparse_kv_block_indices.size(0) == num_q_tokens);
    DG_HOST_ASSERT(sparse_kv_block_indices.scalar_type() == torch::kInt32 and sparse_kv_block_indices.is_contiguous());
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    DG_HOST_ASSERT(page_kv % sparse_block_kv == 0);
    sparse_mqa_logits::get_sparse_split_kv(qk_dtype);  // MXFP4 only.
    const int num_max_sparse_blocks = static_cast<int>(sparse_kv_block_indices.size(1));
    const int64_t num_metadata_bytes = sparse_mqa_logits::get_num_sparse_metadata_bytes(
        num_q_tokens, num_max_sparse_blocks, sparse_block_kv, sparse_mqa_logits::get_num_sparse_slots(),
        /*is_paged=*/true);
    auto metadata = torch::empty({num_metadata_bytes},
                                 sparse_kv_block_indices.options().dtype(torch::kUInt8));
    const int64_t num_workspace_bytes =
        static_cast<int64_t>(sizeof(sparse_mqa_logits::WorkspaceState)) +
        static_cast<int64_t>(num_q_tokens) * sizeof(sparse_mqa_logits::QBlockInfo);
    auto& workspace = sparse_mqa_logits::get_sparse_workspace(metadata.options(), num_workspace_bytes);
    sparse_mqa_logits::launch_sparse_mqa_logits_metadata(
        metadata, workspace, static_cast<uint32_t>(num_q_tokens), /*num_kv_tokens=*/1,
        static_cast<uint32_t>(num_max_sparse_blocks), static_cast<uint32_t>(sparse_block_kv),
        /*cu_seq_len_k_start=*/nullptr, /*cu_seq_len_k_end=*/nullptr,
        reinterpret_cast<const uint32_t*>(sparse_kv_block_indices.data_ptr<int>()),
        /*use_unaligned_ks=*/false,
        /*is_paged=*/true, static_cast<uint32_t>(page_kv),
        reinterpret_cast<const uint32_t*>(context_lens.data_ptr<int>()),
        reinterpret_cast<const uint32_t*>(block_table.data_ptr<int>()),
        static_cast<uint32_t>(block_table.stride(0)),
        reinterpret_cast<const uint32_t*>(indices.data_ptr<int>()));
    return metadata;
}

// Computes logits only on the per-token selected KV blocks prebuilt in `metadata` (by
// `get_sparse_mqa_logits_metadata`). Layout constraints mirror the open-source 26/09 API:
// heads=32 / head_dim=128 are layout constants; weights must be bf16.
torch::Tensor fp8_fp4_sparse_mqa_logits(const std::pair<torch::Tensor, torch::Tensor>& q,
                                        const std::pair<torch::Tensor, torch::Tensor>& kv,
                                        const torch::Tensor& weights, const torch::Tensor& metadata,
                                        int num_max_sparse_blocks, int sparse_block_kv,
                                        bool use_unaligned_ks = false) {
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const auto& q_fp = q.first;
    const auto& q_sf = q.second;
    const auto& kv_fp = kv.first;
    const auto& kv_sf = kv.second;
    DG_HOST_ASSERT(q_fp.scalar_type() == torch::kInt8 and q_fp.is_contiguous());
    DG_HOST_ASSERT(kv_fp.scalar_type() == torch::kInt8 and kv_fp.is_contiguous());
    const auto q_shape = q_fp.sizes();  // [num_q_tokens, kNumHeads, kHeadDim / 2]
    const int num_q_tokens = static_cast<int>(q_shape[0]);
    DG_HOST_ASSERT(q_shape.size() == 3 and
                   q_shape[1] == static_cast<int64_t>(sparse_mqa_logits::kNumHeads) and
                   q_shape[2] == static_cast<int64_t>(sparse_mqa_logits::kHeadDim / 2));
    DG_HOST_ASSERT(q_sf.scalar_type() == torch::kInt32 and q_sf.is_contiguous() and
                   q_sf.dim() == 2 and q_sf.size(0) == num_q_tokens and
                   q_sf.size(1) == static_cast<int64_t>(sparse_mqa_logits::kNumHeads));
    DG_HOST_ASSERT(kv_fp.dim() == 2 and kv_fp.size(1) == sparse_mqa_logits::kHeadDim / 2);
    DG_HOST_ASSERT(kv_sf.scalar_type() == torch::kInt32 and kv_sf.is_contiguous() and kv_sf.numel() == kv_fp.size(0));
    DG_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16 and weights.is_contiguous() and
                   weights.dim() == 2 and weights.size(0) == num_q_tokens and
                   weights.size(1) == static_cast<int64_t>(sparse_mqa_logits::kNumHeads));
    DG_HOST_ASSERT(metadata.scalar_type() == torch::kUInt8 and metadata.is_contiguous() and metadata.dim() == 1 and
                   // Skip full header validation to avoid synchronizing the stream
                   metadata.numel() >= static_cast<int64_t>(sizeof(sparse_mqa_logits::MetadataHeader)));
    const int num_output_tokens = num_max_sparse_blocks * sparse_block_kv;
    const int logits_stride = align(num_output_tokens, 512);  // 1024B row alignment for bf16
    auto logits = torch::empty({align(num_q_tokens, static_cast<int>(sparse_mqa_logits::kBlockQ)), logits_stride},
                               q_fp.options().dtype(torch::kBFloat16));
    logits = logits.slice(0, 0, num_q_tokens);
    sparse_mqa_logits::launch_fp4_sparse_mqa_logits(
        q_fp, q_sf, kv_fp, kv_sf, weights, metadata, logits,
        static_cast<uint32_t>(logits_stride), static_cast<uint32_t>(sparse_block_kv),
        static_cast<uint32_t>(num_q_tokens), static_cast<uint32_t>(kv_fp.size(0)),
        static_cast<uint32_t>(num_max_sparse_blocks), use_unaligned_ks);
    return logits;
}

// Paged counterpart of `fp8_fp4_sparse_mqa_logits`: `kv_cache` is the fused fp4 cache viewed as
// `[num_pages, page_kv, 1, head_dim/2 + 4]` uint8 (each packed 64B token row trails a 4B SF
// u32), with 512B-aligned page stride. Consumes the metadata built by
// `get_paged_sparse_mqa_logits_metadata`.
torch::Tensor fp8_fp4_paged_sparse_mqa_logits(const std::pair<torch::Tensor, torch::Tensor>& q,
                                              const torch::Tensor& kv_cache, const torch::Tensor& weights,
                                              const torch::Tensor& metadata,
                                              int num_max_sparse_blocks, int sparse_block_kv) {
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const auto& q_fp = q.first;
    const auto& q_sf = q.second;
    const auto q_shape = q_fp.sizes();  // [num_q_tokens, kNumHeads, kHeadDim / 2]
    const int num_q_tokens = static_cast<int>(q_shape[0]);
    DG_HOST_ASSERT(q_shape.size() == 3 and
                   q_shape[1] == static_cast<int64_t>(sparse_mqa_logits::kNumHeads) and
                   q_shape[2] == static_cast<int64_t>(sparse_mqa_logits::kHeadDim / 2));
    DG_HOST_ASSERT(q_fp.scalar_type() == torch::kInt8 and q_fp.is_contiguous());
    DG_HOST_ASSERT(q_sf.scalar_type() == torch::kInt32 and q_sf.is_contiguous() and
                   q_sf.dim() == 2 and q_sf.size(0) == num_q_tokens and
                   q_sf.size(1) == static_cast<int64_t>(sparse_mqa_logits::kNumHeads));
    const int64_t head_dim_with_sf = sparse_mqa_logits::kHeadDim / 2 + sizeof(uint32_t);
    DG_HOST_ASSERT(kv_cache.scalar_type() == torch::kUInt8 and kv_cache.dim() == 4 and
                   kv_cache.size(2) == 1 and kv_cache.size(3) == head_dim_with_sf and
                   kv_cache.stride(1) == head_dim_with_sf and kv_cache.stride(3) == 1 and
                   kv_cache.stride(0) % 512 == 0);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16 and weights.is_contiguous() and
                   weights.dim() == 2 and weights.size(0) == num_q_tokens and
                   weights.size(1) == static_cast<int64_t>(sparse_mqa_logits::kNumHeads));
    DG_HOST_ASSERT(metadata.scalar_type() == torch::kUInt8 and metadata.is_contiguous() and metadata.dim() == 1 and
                   metadata.numel() >= static_cast<int64_t>(sizeof(sparse_mqa_logits::MetadataHeader)));
    const int page_kv = static_cast<int>(kv_cache.size(1));
    DG_HOST_ASSERT(page_kv % sparse_block_kv == 0);
    const int num_output_tokens = num_max_sparse_blocks * sparse_block_kv;
    const int logits_stride = align(num_output_tokens, 512);  // 1024B row alignment for bf16
    auto logits = torch::empty({align(num_q_tokens, static_cast<int>(sparse_mqa_logits::kBlockQ)), logits_stride},
                               q_fp.options().dtype(torch::kBFloat16));
    logits = logits.slice(0, 0, num_q_tokens);
    sparse_mqa_logits::launch_fp4_paged_sparse_mqa_logits(
        q_fp, q_sf, kv_cache, weights, metadata, logits,
        static_cast<uint32_t>(logits_stride), static_cast<uint32_t>(sparse_block_kv),
        static_cast<uint32_t>(num_q_tokens), /*seq_len_kv=*/static_cast<uint32_t>(kv_cache.size(0) * page_kv),
        static_cast<uint32_t>(num_max_sparse_blocks));
    return logits;
}

}

static void register_apis(pybind11::module_& m) {
    // Non-paged MQA logits
    m.def("bf16_mqa_logits", &bf16_mqa_logits, py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0, py::arg("logits_dtype") = torch::kFloat32);
    m.def("fp8_mqa_logits", &fp8_mqa_logits, py::arg("q"), py::arg("kv_s"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0, py::arg("logits_dtype") = torch::kFloat32);
    m.def("int8_mqa_logits", &int8_mqa_logits, py::arg("q"), py::arg("kv_s"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0, py::arg("logits_dtype") = torch::kFloat32);
    m.def("fp8_fp4_mqa_logits", &fp8_fp4_mqa_logits, py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0, py::arg("logits_dtype") = torch::kFloat32);
    // Sparse MQA logits (contiguous + paged)
    m.def("get_sparse_mqa_logits_metadata", &get_sparse_mqa_logits_metadata,
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("num_kv_tokens"),
          py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"), py::arg("sparse_block_kv"),
          py::arg("use_unaligned_ks") = false);
    m.def("fp8_fp4_sparse_mqa_logits", &fp8_fp4_sparse_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"), py::arg("metadata"),
          py::arg("num_max_sparse_blocks"), py::arg("sparse_block_kv"),
          py::arg("use_unaligned_ks") = false);
    m.def("get_paged_sparse_mqa_logits_metadata", &get_paged_sparse_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("block_table"), py::arg("indices"), py::arg("page_kv"),
          py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"), py::arg("sparse_block_kv"));
    m.def("fp8_fp4_paged_sparse_mqa_logits", &fp8_fp4_paged_sparse_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"), py::arg("metadata"),
          py::arg("num_max_sparse_blocks"), py::arg("sparse_block_kv"));
    m.def("fp8_mqa_avg_logits", &fp8_mqa_avg_logits, py::arg("q"), py::arg("kv_s"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0, py::arg("q_scale") = std::nullopt,
          py::arg("logits_dtype") = torch::kFloat32);
    // Paged MQA logits
    m.def("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata, py::arg("context_lens"),
          py::arg("block_kv"), py::arg("num_sms"), py::arg("indices") = std::nullopt,
          py::arg("metadata_extra") = std::nullopt);
    m.def("bf16_paged_mqa_logits", &bf16_paged_mqa_logits, py::arg("q"), py::arg("fused_kv_cache"),
          py::arg("weights"), py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("clean_logits") = true, py::arg("logits_dtype") = torch::kFloat32);
    m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits, py::arg("q"), py::arg("fused_kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"), py::arg("max_context_len"),
          py::arg("clean_logits") = true, py::arg("logits_dtype") = torch::kFloat32);
    m.def("int8_paged_mqa_logits", &int8_paged_mqa_logits, py::arg("q"), py::arg("fused_kv_cache"),
          py::arg("weights"), py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("clean_logits") = true, py::arg("logits_dtype") = torch::kFloat32);
    m.def("fp8_fp4_paged_mqa_logits", &fp8_fp4_paged_mqa_logits, py::arg("q"), py::arg("fused_kv_cache"),
          py::arg("weights"), py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("clean_logits") = false,
          py::arg("logits_dtype") = torch::kFloat32, py::arg("indices") = std::nullopt);
    m.def("fp8_paged_mqa_avg_logits", &fp8_paged_mqa_avg_logits, py::arg("q"), py::arg("fused_kv_cache"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"), py::arg("max_context_len"),
          py::arg("clean_logits") = false, py::arg("logits_dtype") = torch::kFloat32,
          py::arg("indices") = std::nullopt);
}

} // namespace deep_gemm::attention
