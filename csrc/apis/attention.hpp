#pragma once

#include "../utils/compatibility.hpp"
#include <torch/extension.h>

#include <string>

#include "../jit_kernels/impls/mqa_logits.hpp"
#include "../jit_kernels/impls/paged_mqa_logits.hpp"
#include "../jit_kernels/impls/sparse_mqa_logits.hpp"

namespace deep_gemm::attention {

// actlize 810E TSM loads do not normalize whole-tile row coordinates
// across the 32-row cubes assembled from page32. Reject this layout until
// actlize provides a validated cross-cube copy implementation.
static void check_paged_mqa_page_size(int page_kv) {
    TORCH_CHECK(page_kv != 32 or get_ppu_arch() != AC_PPU0010,
                "page_kv=32 is unsupported on 810E due to an actlize TSM cross-cube addressing issue; use 64 or 128");
}

// Format support is device-specific: an int8 container may hold INT8 or MXFP4.
static void check_indexer_arch(torch::ScalarType qk_dtype, bool is_fp4) {
    const auto arch = get_ppu_arch();
    if (is_fp4) {
        TORCH_CHECK(arch == AC_PPU0015, "MXFP4 indexer is supported only on M890 (PPU0015)");
    } else if (qk_dtype == torch::kFloat8_e4m3fn) {
        TORCH_CHECK(arch == AC_PPU0015, "FP8 indexer is supported only on M890 (PPU0015)");
    } else {
        TORCH_CHECK(qk_dtype == torch::kInt8, "indexer supports only FP8, INT8 or packed MXFP4 Q/K");
        TORCH_CHECK(arch == AC_PPU0010, "INT8 indexer is supported only on 810E (PPU0010)");
    }
}

// Q/K tensors use the physical storage dimension; FP4 packs two values per byte.
static void check_indexer_head_dim(int storage_head_dim, bool is_fp4) {
    if (is_fp4) {
        TORCH_CHECK(storage_head_dim == 64,
                    "MXFP4 indexer requires logical head_dim=128 (packed tensor dimension=64)");
    } else {
        TORCH_CHECK(storage_head_dim == 64 or storage_head_dim == 128,
                    "FP8/INT8 indexer requires logical head_dim=64 or 128");
    }
}

// Keep public dtype/layout policies separate from the shared launch path.
static torch::Tensor checked_weights(const torch::Tensor& weights, const torch::Tensor& q,
                                      torch::ScalarType dtype) {
    TORCH_CHECK(weights.dim() == 2 and weights.numel() > 0 and weights.scalar_type() == dtype,
                "weights must be a 2D tensor with dtype ", dtype);
    TORCH_CHECK(weights.device() == q.device() and weights.stride(1) == 1,
                "weights must share Q's device and have unit column stride");
    TORCH_CHECK((weights.stride(0) * weights.element_size()) % 16 == 0,
                "weights row stride must be 16-byte aligned");
    // Existing PPU kernels address packed weights. Accept padded public views
    // by packing here, rather than ignoring their row stride in the kernel.
    return weights.contiguous();
}

static void check_row_indices(const std::optional<torch::Tensor>& indices,
                              const torch::Tensor& context_lens) {
    if (not indices.has_value()) return;
    TORCH_CHECK(context_lens.dim() == 2 and context_lens.size(1) == 1,
                "request indices require next_n == 1");
    TORCH_CHECK(indices->dim() == 1 and indices->size(0) == context_lens.size(0) and
                indices->scalar_type() == torch::kInt32 and indices->is_contiguous() and
                indices->device() == context_lens.device(),
                "indices must be contiguous int32 [Q] on context_lens.device");
    // PPU schedules each expanded query row independently. Request ids do not
    // alter addressing: block_table and context_lens are already per-query.
    // Unlike the SM100 scheduler, this backend does not pair rows for KV reuse.
}

// Compatibility stub: dense PPU logits do not use scheduling metadata.
static pybind11::none get_mqa_logits_metadata(const torch::Tensor&, const torch::Tensor&,
                                              int, int) {
    return pybind11::none();
}

// MQA logits (non-paged) for FP8/INT8/FP4.
//
// Allocates the logits tensor (its row stride must be 1024-byte aligned), launches the kernel and
// leaves out-of-range entries unspecified. Returns the (sliced) logits.
//
// Args:
//   q:        Q tensor.
//             FP8/INT8: [seq_len_q, num_heads, head_dim], dtype = float8_e4m3fn/int8
//             FP4:           [seq_len_q, num_heads, head_dim_packed], dtype = int8,
//                            where head_dim_packed = original_head_dim / 2 = 64
//   k:        K tensor.
//             FP8/INT8: [seq_len_k, head_dim], dtype = same as q
//             FP4:           [seq_len_k, head_dim_packed], dtype = int8
//   k_scales: per-token KV scale factor.
//             FP8/INT8: float32 [seq_len_k]
//             FP4:      empty tensor (unused, `k_sf` is used instead)
//   weights:            [seq_len_q, num_heads], float32 or bfloat16; empty selects avg
//   cu_seq_len_k_start: int32 [seq_len_q]
//   cu_seq_len_k_end:   int32 [seq_len_q]
//   clean_logits: compatibility parameter; True is rejected
//   max_seqlen_k: 0 for non-compressed, >0 for compressed mode (required for FP4 AVG)
//   logits_dtype: output logits dtype, float32 or bfloat16
//   q_sf:     UE8M0 scale for Q (FP4 only), int32 [seq_len_q, num_heads].
//             Packed from uint8 e8m0 (4x uint8 per int32). `nullopt` for FP8/INT8.
//   k_sf:     UE8M0 scale for K (FP4 only), int32 [seq_len_k]. Same packing as `q_sf`.
//   q_scale:  Optional float32 scalar or [seq_len_q] multiplier for avg, separate from FP4 `q_sf`.
static torch::Tensor mqa_logits_common(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& k_scales,
                                       const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                                       const torch::Tensor& cu_seq_len_k_end, bool clean_logits, int max_seqlen_k,
                                       torch::ScalarType logits_dtype,
                                       const std::optional<torch::Tensor>& q_sf = std::nullopt,
                                       const std::optional<torch::Tensor>& k_sf = std::nullopt,
                                       const std::optional<torch::Tensor>& q_scale = std::nullopt) {
    TORCH_CHECK(not clean_logits, "clean_logits=True is not supported on PPU");
    TORCH_CHECK(max_seqlen_k >= 0, "max_seqlen_k must be nonnegative");
    const bool is_fp4 = q_sf.has_value();
    check_indexer_arch(q.scalar_type(), is_fp4);
    // Shape extraction & validation
    const auto& [seq_len_q, num_heads, head_dim] = get_shape<3>(q);
    const auto& [seq_len_k, head_dim_] = get_shape<2>(k);
    TORCH_CHECK(head_dim == head_dim_, "Q and K head_dim must match");
    check_indexer_head_dim(head_dim, is_fp4);

    // Avg variant: an empty weights tensor selects it;
    // the optional q_scale rides in the kernel's weights slot
    const bool is_avg = weights.numel() == 0;
    TORCH_CHECK(is_avg ? num_heads == 4 : (num_heads == 32 or num_heads == 64),
                is_avg ? "AVG indexer requires num_heads=4" : "DSA indexer requires num_heads=32 or 64");
    if (is_avg) {
        TORCH_CHECK(is_fp4 or q.scalar_type() == torch::kFloat8_e4m3fn,
                    "avg variant supports fp8 or packed fp4 only");
        TORCH_CHECK(logits_dtype == torch::kFloat32 || logits_dtype == torch::kBFloat16,
                    "avg variant supports float32 or bfloat16 logits");
        if (is_fp4)
            TORCH_CHECK(max_seqlen_k > 0, "FP4 AVG requires compressed logits (max_seqlen_k > 0)");
        if (q_scale.has_value()) {
            TORCH_CHECK(q_scale->scalar_type() == torch::kFloat32, "q_scale must be float32");
            TORCH_CHECK(q_scale->is_contiguous(), "q_scale must be contiguous");
            TORCH_CHECK(q_scale->device() == q.device(), "q_scale must be on the same device as q");
            TORCH_CHECK(q_scale->numel() == 1 or (q_scale->dim() == 1 and q_scale->numel() == seq_len_q),
                        "q_scale must be a scalar or a [seq_len_q] vector");
        }
    } else {
        const auto& [seq_len_, num_heads_] = get_shape<2>(weights);
        DG_HOST_ASSERT(seq_len_q == seq_len_);
        DG_HOST_ASSERT(num_heads == num_heads_);
        TORCH_CHECK(weights.is_contiguous(), "internal weights must be contiguous");
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
        TORCH_CHECK(q_sf->dim() == 2 and q_sf->size(0) == seq_len_q and q_sf->size(1) == num_heads,
                    "q_sf must have shape [seq_len_q, num_heads]");
        TORCH_CHECK(k_sf->dim() == 1 and k_sf->size(0) == seq_len_k,
                    "k_sf must have shape [seq_len_k]");
        TORCH_CHECK(q_sf->device() == q.device() and k_sf->device() == q.device(),
                    "q_sf and k_sf must be on the same device as q");
        TORCH_CHECK(q_sf->is_contiguous() and k_sf->is_contiguous(), "q_sf and k_sf must be contiguous");
        DG_HOST_ASSERT(qk_dtype == torch::kInt8 and k.scalar_type() == torch::kInt8);
        DG_HOST_ASSERT(q_sf->scalar_type() == torch::kInt32 and k_sf->scalar_type() == torch::kInt32);
    } else {
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == torch::kInt8);
        DG_HOST_ASSERT(qk_dtype == k.scalar_type());
        DG_HOST_ASSERT(k_scales.size(0) == seq_len_k);
        TORCH_CHECK(k_scales.is_contiguous(), "k_scales must be contiguous");
        DG_HOST_ASSERT(k_scales.scalar_type() == torch::kFloat32);
    }

    const bool is_compressed = max_seqlen_k > 0;
    if (is_compressed)
        TORCH_CHECK(not clean_logits, "clean_logits must be False when compressed (max_seqlen_k > 0)");

    return mqa_logits(q, k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, seq_len_q, seq_len_k, num_heads,
                      head_dim, clean_logits, max_seqlen_k, logits_dtype, q_sf, k_sf, q_scale);
}

// Paged MQA logits for FP8/INT8/FP4.
//
// Args:
//   q:              Q tensor.
//                   FP8/INT8: [batch, next_n, num_heads, head_dim],
//                                  dtype = float8_e4m3fn/int8
//                   FP4:           [batch, next_n, num_heads, head_dim_packed], dtype = int8,
//                                  where head_dim_packed = original_head_dim / 2 = 64
//   fused_kv_cache: FP8/INT8: uint8 [num_kv_blocks, page_kv, 1, head_dim + 4]
//                   FP4:      uint8 [num_kv_blocks, page_kv, 1, head_dim_packed + 4]
//                   Each quantized page stores all rows' values first, then all rows' scales:
//                     [page_kv * value_bytes][page_kv * 4 scale bytes].
//                   FP8/INT8 scales are float32; FP4 scales are packed UE8M0 (int32).
//                   The shape above is a storage view, not an interleaved per-row layout.
//                   Page strides may include padding; value/scale views use `as_strided`.
//   weights:        [batch * next_n, num_heads], float32 or bfloat16; empty selects avg
//   context_lens:   int32 [batch_size, next_n]
//   block_table:    int32 [batch_size, max_block_len]
//   schedule_meta:  int32 [num_blocks+1, 2], produced by `get_paged_mqa_logits_metadata`
//   max_context_len: maximum context length
//   clean_logits:   whether to mask out-of-range logits with -inf (unsupported here, see below)
//   logits_dtype:   output logits dtype, float32 or bfloat16
//   q_sf:           UE8M0 scale for Q (FP4 only), int32 [batch, next_n, num_heads].
//                   Packed from uint8 e8m0 (4x uint8 per int32). `nullopt` for FP8/INT8.
static torch::Tensor paged_mqa_logits_common(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                            const torch::Tensor& weights, const torch::Tensor& context_lens,
                                            const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                            int max_context_len, bool clean_logits, torch::ScalarType logits_dtype,
                                            const std::optional<torch::Tensor>& q_sf = std::nullopt) {
    TORCH_CHECK(not clean_logits, "clean_logits=True is not supported on PPU");
    const bool is_fp4 = q_sf.has_value();
    check_indexer_arch(q.scalar_type(), is_fp4);
    const bool is_avg = weights.numel() == 0;
    const auto qk_dtype = q.scalar_type();
    const auto& [batch_size, next_n, num_heads, head_dim] = get_shape<4>(q);
    TORCH_CHECK(is_avg ? num_heads == 4 : (num_heads == 32 or num_heads == 64),
                is_avg ? "AVG indexer requires num_heads=4" : "DSA indexer requires num_heads=32 or 64");
    check_indexer_head_dim(head_dim, is_fp4);
    const auto& [num_kv_blocks, page_kv, num_heads_kv, kv_last_dim] = get_shape<4>(fused_kv_cache);
    check_paged_mqa_page_size(page_kv);
    const auto& [schedule_meta_size, meta_info_size] = get_shape<2>(schedule_meta);

    // Shared request/page geometry and scheduler metadata.
    DG_HOST_ASSERT(context_lens.dim() == 2);
    DG_HOST_ASSERT(context_lens.size(1) == next_n);
    DG_HOST_ASSERT(batch_size == context_lens.size(0) and batch_size == block_table.size(0));
    DG_HOST_ASSERT(num_heads_kv == 1);
    DG_HOST_ASSERT((schedule_meta_size - 1) % get_num_sms() == 0 and meta_info_size == 2);
    TORCH_CHECK(is_avg ? (1 <= next_n and next_n <= 6) : next_n == 1,
                "weighted paged indexer requires next_n == 1; AVG supports next_n in [1, 6]");
    DG_HOST_ASSERT(page_kv == 32 or page_kv == 64 or page_kv == 128);
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt32);
    TORCH_CHECK(schedule_meta.is_contiguous(), "schedule_meta must be contiguous");
    DG_HOST_ASSERT(schedule_meta.scalar_type() == torch::kInt32);

    // Q and cache storage. A present q_sf selects packed FP4, not INT8 arithmetic.
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == kv_last_dim);
    DG_HOST_ASSERT(fused_kv_cache.stride(2) == kv_last_dim);
    DG_HOST_ASSERT(fused_kv_cache.stride(3) == 1);
    if (is_fp4) {
        DG_HOST_ASSERT(head_dim == 64);
        DG_HOST_ASSERT(kv_last_dim - head_dim == 4);
        DG_HOST_ASSERT(qk_dtype == torch::kInt8);
        TORCH_CHECK(q_sf->is_contiguous(), "q_sf must be contiguous");
        TORCH_CHECK(q_sf->device() == q.device(), "q_sf must be on the same device as q");
        DG_HOST_ASSERT(q_sf->scalar_type() == torch::kInt32);
        DG_HOST_ASSERT((q_sf->sizes() == std::vector<int64_t>{batch_size, next_n, num_heads}));
        DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kUInt8);
        DG_HOST_ASSERT(fused_kv_cache.stride(0) % 4 == 0);
    } else {
        DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
        DG_HOST_ASSERT(kv_last_dim == head_dim + 4);
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == torch::kInt8);
        DG_HOST_ASSERT(fused_kv_cache.stride(0) % 4 == 0);
        DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kUInt8);
    }

    // Output and reduction mode. Four-head tiles are reserved for AVG.
    DG_HOST_ASSERT(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16);
    if (is_avg) {
        TORCH_CHECK(num_heads == 4, "avg variant supports num_heads == 4 only");
        TORCH_CHECK(is_fp4 or qk_dtype == torch::kFloat8_e4m3fn,
                    "avg variant supports fp8 or packed fp4 with q_sf only");
    } else {
        const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
        DG_HOST_ASSERT(batch_size_next_n == batch_size * next_n);
        DG_HOST_ASSERT(num_heads == num_heads_);
        TORCH_CHECK(weights.is_contiguous(), "internal weights must be contiguous");
        DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat32 or
                       (logits_dtype == torch::kBFloat16 and weights.scalar_type() == torch::kBFloat16));
        if (logits_dtype == torch::kFloat32)
            TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "fp32 logits requires fp32 weights");
    }
    TORCH_CHECK(not clean_logits, "clean_logits not supported with 2D context_lens, use external masking");

    return paged_mqa_logits(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, batch_size, next_n,
                            num_heads, head_dim, num_kv_blocks, page_kv, schedule_meta_size, max_context_len,
                            logits_dtype, q_sf);
}

extern "C" {

// Legacy FP8: ordinary per-token KV scale, FP32 gates and output.
torch::Tensor fp8_mqa_logits(const torch::Tensor& q, const std::pair<torch::Tensor, torch::Tensor>& kv,
                             const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                             const torch::Tensor& cu_seq_len_k_end, bool clean_logits = false, int max_seqlen_k = 0) {
    TORCH_CHECK(not clean_logits, "clean_logits=True is not supported on PPU");
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "legacy FP8 requires e4m3 Q");
    return mqa_logits_common(q, kv.first, kv.second, checked_weights(weights, q, torch::kFloat32),
                             cu_seq_len_k_start, cu_seq_len_k_end, false, max_seqlen_k, torch::kFloat32);
}

// 810E legacy INT8 retains both output dtypes for its prefill top-k paths.
torch::Tensor int8_mqa_logits(const torch::Tensor& q, const std::pair<torch::Tensor, torch::Tensor>& kv,
                              const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                              const torch::Tensor& cu_seq_len_k_end, bool clean_logits = false, int max_seqlen_k = 0) {
    TORCH_CHECK(not clean_logits, "clean_logits=True is not supported on PPU");
    TORCH_CHECK(q.scalar_type() == torch::kInt8, "legacy INT8 requires int8 Q");
    const auto logits_dtype = weights.scalar_type();
    TORCH_CHECK(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16,
                "INT8 weights must be float32 or bfloat16");
    return mqa_logits_common(q, kv.first, kv.second, checked_weights(weights, q, logits_dtype),
                             cu_seq_len_k_start, cu_seq_len_k_end, false, max_seqlen_k, logits_dtype);
}

// Upstream-compatible signature, MXFP4-only on PPU1.5. AVG is separate.
torch::Tensor fp8_fp4_mqa_logits(const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
                                 const std::pair<torch::Tensor, torch::Tensor>& kv, const torch::Tensor& weights,
                                 const torch::Tensor& cu_seq_len_k_start, const torch::Tensor& cu_seq_len_k_end,
                                 int max_seqlen_k, const std::optional<torch::Tensor>& schedule_meta = std::nullopt) {
    TORCH_CHECK(q.second.has_value() and q.first.scalar_type() == torch::kInt8,
                "PPU fp8_fp4_mqa_logits supports MXFP4 only; use fp8_mqa_logits for ordinary FP8");
    TORCH_CHECK(max_seqlen_k > 0, "MXFP4 requires max_seqlen_k > 0");
    // schedule_meta is accepted for API compatibility and intentionally ignored.
    const auto empty = torch::empty({0}, q.first.options().dtype(torch::kFloat32));
    return mqa_logits_common(q.first, kv.first, empty, checked_weights(weights, q.first, torch::kBFloat16),
                             cu_seq_len_k_start, cu_seq_len_k_end, false, max_seqlen_k, torch::kBFloat16, q.second, kv.second);
}

// Unified FP8/FP4 avg entry point (non-paged, 4 heads), with the same q/kv tuples
// as the quantized inputs and no weights input. FP4 q_sf/k_sf are consumed by MMA;
// q_scale is an independent optional scalar or per-Q-row multiplier after reduction.
// Normalization uses the logical head dimension (128 for packed FP4), with no /num_heads.
// Output defaults to BF16; FP32 is also supported. All inputs require max_seqlen_k > 0; no cleaning parameter.
torch::Tensor fp8_fp4_mqa_avg_logits(const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
                                     const std::pair<torch::Tensor, torch::Tensor>& kv,
                                     const torch::Tensor& cu_seq_len_k_start, const torch::Tensor& cu_seq_len_k_end,
                                     int max_seqlen_k,
                                     const std::optional<torch::Tensor>& q_scale = std::nullopt,
                                     torch::ScalarType logits_dtype = torch::kBFloat16) {
    TORCH_CHECK(max_seqlen_k > 0, "AVG requires max_seqlen_k > 0");
    const auto& empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(q.first.device()));
    if (q.second.has_value()) {
        DG_HOST_ASSERT(q.first.scalar_type() == torch::kInt8);
        return mqa_logits_common(q.first, kv.first, empty, empty, cu_seq_len_k_start, cu_seq_len_k_end, false,
                                 max_seqlen_k, logits_dtype, q.second, kv.second, q_scale);
    }
    DG_HOST_ASSERT(q.first.scalar_type() == torch::kFloat8_e4m3fn);
    return mqa_logits_common(q.first, kv.first, kv.second, empty, cu_seq_len_k_start, cu_seq_len_k_end, false,
                             max_seqlen_k, logits_dtype, std::nullopt, std::nullopt, q_scale);
}

// Backward-compatible FP8-only entry: sum_h(ReLU(dot(q_h, k))) * k_scale / sqrt(head_dim),
// with an optional q_scale (scalar broadcast or per-Q-row vector).
torch::Tensor fp8_mqa_avg_logits(const torch::Tensor& q, const std::pair<torch::Tensor, torch::Tensor>& kv_s,
                                 const torch::Tensor& cu_seq_len_k_start, const torch::Tensor& cu_seq_len_k_end,
                                 int max_seqlen_k,
                                 const std::optional<torch::Tensor>& q_scale = std::nullopt,
                                 torch::ScalarType logits_dtype = torch::kBFloat16) {
    return fp8_fp4_mqa_avg_logits({q, std::nullopt}, kv_s, cu_seq_len_k_start, cu_seq_len_k_end,
                                 max_seqlen_k, q_scale, logits_dtype);
}

// MXFP4-only paged AVG (4 heads, unity mode: no additional q_scale).
// FP4 uses packed head_dim = 64 (logical head_dim = 128) and mandatory q_sf;
// Ordinary FP8 uses the separate fp8_paged_mqa_avg_logits entry point.
// Output defaults to BF16; FP32 is also supported.
// Build FP4 AVG metadata with metadata_extra=(next_n, 4, 128, torch::kInt8).
torch::Tensor fp8_fp4_paged_mqa_avg_logits(const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
                                           const torch::Tensor& fused_kv_cache,
                                           const torch::Tensor& context_lens, const torch::Tensor& block_table,
                                           const torch::Tensor& schedule_meta, int max_context_len,
                                           const std::optional<torch::Tensor>& indices = std::nullopt,
                                           torch::ScalarType logits_dtype = torch::kBFloat16) {
    TORCH_CHECK(q.second.has_value() and q.first.scalar_type() == torch::kInt8,
                "PPU fp8_fp4_paged_mqa_avg_logits supports MXFP4 only; use fp8_paged_mqa_avg_logits for ordinary FP8");
    TORCH_CHECK(not indices.has_value(), "AVG does not support request indices");
    const auto& empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(q.first.device()));
    return paged_mqa_logits_common(q.first, fused_kv_cache, empty, context_lens, block_table, schedule_meta,
                                   max_context_len, false, logits_dtype, q.second);
}

// Backward-compatible FP8-only paged AVG entry.
torch::Tensor fp8_paged_mqa_avg_logits(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                       const torch::Tensor& context_lens, const torch::Tensor& block_table,
                                       const torch::Tensor& schedule_meta, int max_context_len,
                                       const std::optional<torch::Tensor>& indices = std::nullopt,
                                       torch::ScalarType logits_dtype = torch::kBFloat16) {
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "FP8 AVG requires e4m3 Q");
    TORCH_CHECK(not indices.has_value(), "AVG does not support request indices");
    const auto empty = torch::empty({0}, q.options().dtype(torch::kFloat32));
    return paged_mqa_logits_common(q, fused_kv_cache, empty, context_lens, block_table,
                                   schedule_meta, max_context_len, false, logits_dtype);
}

// metadata_extra = (next_n, num_heads, logical_head_dim, qk_dtype).
// torch::kInt8 means packed MXFP4 on M890, but true INT8 on 810E.
// FP8 uses torch::kFloat8_e4m3fn on M890. Derive packed storage internally.
// metadata_extra is mandatory on PPU0010/PPU0015; None is rejected.
// Pass metadata_extra by keyword; the fourth positional argument is indices.
torch::Tensor get_paged_mqa_logits_metadata(
    const torch::Tensor& context_lens, int page_kv, int num_sms, std::optional<torch::Tensor> indices = std::nullopt,
    std::optional<std::tuple<int, int, int, torch::ScalarType>> metadata_extra = std::nullopt) {
    check_paged_mqa_page_size(page_kv);
    check_row_indices(indices, context_lens);
    TORCH_CHECK(metadata_extra.has_value(),
                "metadata_extra=(next_n, num_heads, logical_head_dim, qk_dtype) is required on PPU0010/PPU0015; "
                "omitting it can cause severe scheduling performance degradation");
    DG_HOST_ASSERT(context_lens.dim() == 2);
    const int batch_size = context_lens.size(0);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");
    TORCH_CHECK(context_lens.size(1) >= 1 and context_lens.size(1) <= 6, "next_n must be in [1, 6]");
    TORCH_CHECK(num_sms > 0, "num_sms must be positive");
    const auto& [n, h, d, qk_dtype] = *metadata_extra;
    TORCH_CHECK(n == context_lens.size(1), "metadata_extra next_n must match context_lens");
    TORCH_CHECK(h == 4 or h == 32 or h == 64,
                "metadata_extra num_heads must be 4 (AVG) or 32/64 (DSA)");
    TORCH_CHECK(h == 4 or n == 1, "weighted metadata requires next_n == 1");
    const auto arch = get_ppu_arch();
    TORCH_CHECK(h != 4 or arch == AC_PPU0015, "AVG metadata requires M890 and num_heads=4");
    TORCH_CHECK((arch == AC_PPU0010 and qk_dtype == torch::kInt8) or
                (arch == AC_PPU0015 and (qk_dtype == torch::kInt8 or qk_dtype == torch::kFloat8_e4m3fn)),
                "metadata_extra qk_dtype: 810E supports torch.int8 (INT8); "
                "M890 supports torch.int8 (MXFP4) or torch.float8_e4m3fn (FP8)");
    TORCH_CHECK(d == 64 or d == 128,
                "metadata_extra requires logical head_dim=64 or 128");
    TORCH_CHECK(not (arch == AC_PPU0015 and qk_dtype == torch::kInt8) or d == 128,
                "M890 MXFP4 metadata_extra requires logical head_dim=128, not packed head_dim=64");
    // shared memory limit
    TORCH_CHECK(batch_size > 0 and batch_size <= 65536, "batch_size must be in [1, 65536]");

    return paged_mqa_logits_metadata(context_lens, batch_size, page_kv, num_sms, *metadata_extra);
}

torch::Tensor fp8_paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& kv_cache,
                                   const torch::Tensor& weights, const torch::Tensor& context_lens,
                                   const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                   int max_context_len, bool clean_logits = false,
                                   const std::optional<torch::Tensor>& indices = std::nullopt) {
    TORCH_CHECK(not clean_logits, "clean_logits=True is not supported on PPU");
    TORCH_CHECK(not indices.has_value(), "legacy FP8 does not support request indices");
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "legacy FP8 requires e4m3 Q");
    return paged_mqa_logits_common(q, kv_cache, checked_weights(weights, q, torch::kFloat32),
                                   context_lens, block_table, schedule_meta, max_context_len, false, torch::kFloat32);
}

torch::Tensor int8_paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& kv_cache,
                                    const torch::Tensor& weights, const torch::Tensor& context_lens,
                                    const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                    int max_context_len, bool clean_logits = false,
                                    const std::optional<torch::Tensor>& indices = std::nullopt) {
    TORCH_CHECK(not clean_logits, "clean_logits=True is not supported on PPU");
    TORCH_CHECK(not indices.has_value(), "legacy INT8 does not support request indices");
    TORCH_CHECK(q.scalar_type() == torch::kInt8, "legacy INT8 requires int8 Q");
    const auto logits_dtype = weights.scalar_type();
    TORCH_CHECK(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16,
                "INT8 weights must be float32 or bfloat16");
    return paged_mqa_logits_common(q, kv_cache, checked_weights(weights, q, logits_dtype),
                                   context_lens, block_table, schedule_meta, max_context_len, false, logits_dtype);
}

torch::Tensor fp8_fp4_paged_mqa_logits(const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
                                       const torch::Tensor& kv_cache, const torch::Tensor& weights,
                                       const torch::Tensor& context_lens, const torch::Tensor& block_table,
                                       const torch::Tensor& schedule_meta, int max_context_len,
                                       const std::optional<torch::Tensor>& indices = std::nullopt) {
    TORCH_CHECK(q.second.has_value() and q.first.scalar_type() == torch::kInt8,
                "PPU fp8_fp4_paged_mqa_logits supports MXFP4 only");
    TORCH_CHECK(indices.has_value(), "MXFP4 paged logits require per-query request indices");
    check_row_indices(indices, context_lens);
    return paged_mqa_logits_common(q.first, kv_cache, checked_weights(weights, q.first, torch::kBFloat16),
                                   context_lens, block_table, schedule_meta, max_context_len, false,
                                   torch::kBFloat16, q.second);
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
    check_indexer_arch(torch::kInt8, true);
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
    auto& workspace = sparse_mqa_logits::get_sparse_workspace(metadata.options(), num_q_tokens);
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
    check_indexer_arch(torch::kInt8, true);
    const int num_q_tokens = static_cast<int>(context_lens.size(0));
    DG_HOST_ASSERT(num_q_tokens > 0 and block_table.size(0) == num_q_tokens and indices.size(0) == num_q_tokens);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32 and context_lens.is_contiguous());
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt32 and block_table.is_contiguous() and
                   block_table.dim() == 2);
    DG_HOST_ASSERT(indices.scalar_type() == torch::kInt32 and indices.is_contiguous());
    DG_HOST_ASSERT(sparse_kv_block_indices.dim() == 2 and sparse_kv_block_indices.size(0) == num_q_tokens);
    DG_HOST_ASSERT(sparse_kv_block_indices.scalar_type() == torch::kInt32 and sparse_kv_block_indices.is_contiguous());
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    DG_HOST_ASSERT(page_kv > 0 and page_kv % sparse_block_kv == 0);
    sparse_mqa_logits::get_sparse_split_kv(qk_dtype);  // MXFP4 only.
    const int num_max_sparse_blocks = static_cast<int>(sparse_kv_block_indices.size(1));
    const int64_t num_metadata_bytes = sparse_mqa_logits::get_num_sparse_metadata_bytes(
        num_q_tokens, num_max_sparse_blocks, sparse_block_kv, sparse_mqa_logits::get_num_sparse_slots(),
        /*is_paged=*/true);
    auto metadata = torch::empty({num_metadata_bytes},
                                 sparse_kv_block_indices.options().dtype(torch::kUInt8));
    auto& workspace = sparse_mqa_logits::get_sparse_workspace(metadata.options(), num_q_tokens);
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
                                        const torch::Tensor& weights_input, const torch::Tensor& metadata,
                                        int num_max_sparse_blocks, int sparse_block_kv,
                                        bool use_unaligned_ks = false) {
    check_indexer_arch(torch::kInt8, true);
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const auto& q_fp = q.first;
    const auto weights = checked_weights(weights_input, q_fp, torch::kBFloat16);
    const auto& q_sf = q.second;
    const auto& kv_fp = kv.first;
    const auto& kv_sf = kv.second;
    TORCH_CHECK(q_fp.dim() == 3 and kv_fp.dim() == 2, "sparse Q/K must have rank 3/2");
    TORCH_CHECK(q_fp.size(1) == sparse_mqa_logits::kNumHeads, "sparse indexer requires num_heads=32");
    check_indexer_head_dim(q_fp.size(2), true);
    check_indexer_head_dim(kv_fp.size(1), true);
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
    return logits.slice(1, 0, num_output_tokens);
}

// Paged counterpart of `fp8_fp4_sparse_mqa_logits`: `kv_cache` is the fused fp4 cache viewed as
// `[num_pages, page_kv, 1, head_dim/2 + 4]` uint8 (each packed 64B token row trails a 4B SF
// u32), with 512B-aligned page stride. Consumes the metadata built by
// `get_paged_sparse_mqa_logits_metadata`.
torch::Tensor fp8_fp4_paged_sparse_mqa_logits(const std::pair<torch::Tensor, torch::Tensor>& q,
                                              const torch::Tensor& kv_cache, const torch::Tensor& weights_input,
                                              const torch::Tensor& metadata,
                                              int num_max_sparse_blocks, int sparse_block_kv) {
    check_indexer_arch(torch::kInt8, true);
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const auto& q_fp = q.first;
    const auto weights = checked_weights(weights_input, q_fp, torch::kBFloat16);
    const auto& q_sf = q.second;
    DG_HOST_ASSERT(q_fp.dim() == 4);
    const auto [num_q_tokens, next_n, num_heads, head_dim_packed] = get_shape<4>(q_fp);
    TORCH_CHECK(num_heads == sparse_mqa_logits::kNumHeads, "sparse indexer requires num_heads=32");
    check_indexer_head_dim(head_dim_packed, true);
    DG_HOST_ASSERT(num_q_tokens > 0 and next_n == 1 and
                   num_heads == sparse_mqa_logits::kNumHeads and
                   head_dim_packed == sparse_mqa_logits::kHeadDim / 2);
    DG_HOST_ASSERT(q_fp.scalar_type() == torch::kInt8 and q_fp.is_contiguous());
    DG_HOST_ASSERT(q_sf.scalar_type() == torch::kInt32 and q_sf.is_contiguous() and
                   q_sf.dim() == 3 and q_sf.size(0) == num_q_tokens and
                   q_sf.size(1) == 1 and q_sf.size(2) == num_heads);
    const int64_t head_dim_with_sf = sparse_mqa_logits::kHeadDim / 2 + sizeof(uint32_t);
    DG_HOST_ASSERT(kv_cache.scalar_type() == torch::kUInt8 and kv_cache.dim() == 4 and
                   kv_cache.size(0) > 0 and kv_cache.size(2) == 1 and kv_cache.size(3) == head_dim_with_sf and
                   kv_cache.stride(1) == head_dim_with_sf and kv_cache.stride(3) == 1 and
                   kv_cache.stride(0) > 0 and kv_cache.stride(0) <= std::numeric_limits<int>::max() and
                   kv_cache.stride(0) % 512 == 0);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16 and weights.is_contiguous() and
                   weights.dim() == 2 and weights.size(0) == num_q_tokens and
                   weights.size(1) == static_cast<int64_t>(sparse_mqa_logits::kNumHeads));
    DG_HOST_ASSERT(metadata.scalar_type() == torch::kUInt8 and metadata.is_contiguous() and metadata.dim() == 1 and
                   metadata.numel() >= static_cast<int64_t>(sizeof(sparse_mqa_logits::MetadataHeader)));
    const int page_kv = static_cast<int>(kv_cache.size(1));
    DG_HOST_ASSERT(page_kv > 0 and page_kv % sparse_block_kv == 0);
    const int num_output_tokens = num_max_sparse_blocks * sparse_block_kv;
    const int logits_stride = align(num_output_tokens, 512);  // 1024B row alignment for bf16
    auto logits = torch::empty({align<int64_t>(num_q_tokens, sparse_mqa_logits::kBlockQ), logits_stride},
                               q_fp.options().dtype(torch::kBFloat16));
    logits = logits.slice(0, 0, num_q_tokens);
    sparse_mqa_logits::launch_fp4_paged_sparse_mqa_logits(
        q_fp, q_sf, kv_cache, weights, metadata, logits,
        static_cast<uint32_t>(logits_stride), static_cast<uint32_t>(sparse_block_kv),
        static_cast<uint32_t>(num_q_tokens), /*seq_len_kv=*/static_cast<uint32_t>(kv_cache.size(0) * page_kv),
        static_cast<uint32_t>(num_max_sparse_blocks));
    return logits.slice(1, 0, num_output_tokens);
}

}

static void register_apis(pybind11::module_& m) {
    m.def("get_mqa_logits_metadata", &get_mqa_logits_metadata,
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("num_kv_tokens"), py::arg("num_heads"));
    m.def("fp8_mqa_logits", &fp8_mqa_logits, py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("clean_logits") = false,
          py::arg("max_seqlen_k") = 0);
    m.def("int8_mqa_logits", &int8_mqa_logits, py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("clean_logits") = false,
          py::arg("max_seqlen_k") = 0);
    m.def("fp8_fp4_mqa_logits", &fp8_fp4_mqa_logits, py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("max_seqlen_k"),
          py::arg("schedule_meta") = std::nullopt);
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
    m.def("fp8_fp4_mqa_avg_logits", &fp8_fp4_mqa_avg_logits, py::arg("q"), py::arg("kv"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("max_seqlen_k"), py::arg("q_scale") = std::nullopt,
          py::arg("logits_dtype") = torch::kBFloat16);
    m.def("fp8_mqa_avg_logits", &fp8_mqa_avg_logits, py::arg("q"), py::arg("kv_s"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("max_seqlen_k"), py::arg("q_scale") = std::nullopt,
          py::arg("logits_dtype") = torch::kBFloat16);
    // Paged MQA logits
    m.def("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata, py::arg("context_lens"),
          py::arg("block_kv"), py::arg("num_sms"), py::arg("indices") = std::nullopt,
          py::arg("metadata_extra") = std::nullopt);
    m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits, py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"), py::arg("max_context_len"),
          py::arg("clean_logits") = false, py::arg("indices") = std::nullopt);
    m.def("int8_paged_mqa_logits", &int8_paged_mqa_logits, py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"), py::arg("max_context_len"),
          py::arg("clean_logits") = false, py::arg("indices") = std::nullopt);
    m.def("fp8_fp4_paged_mqa_logits", &fp8_fp4_paged_mqa_logits, py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"), py::arg("max_context_len"),
          py::arg("indices") = std::nullopt);
    m.def("fp8_paged_mqa_avg_logits", &fp8_paged_mqa_avg_logits, py::arg("q"), py::arg("fused_kv_cache"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"), py::arg("max_context_len"),
          py::arg("indices") = std::nullopt, py::arg("logits_dtype") = torch::kBFloat16);
    m.def("fp8_fp4_paged_mqa_avg_logits", &fp8_fp4_paged_mqa_avg_logits, py::arg("q"), py::arg("fused_kv_cache"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"), py::arg("max_context_len"),
          py::arg("indices") = std::nullopt, py::arg("logits_dtype") = torch::kBFloat16);
}

} // namespace deep_gemm::attention
