#pragma once

#include "../utils/compatibility.hpp"
#include <torch/extension.h>

#include <mutex>
#include <set>
#include <string>

#include "../jit_kernels/impls/mqa_logits.hpp"
#include "../jit_kernels/impls/paged_mqa_logits.hpp"

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
                                       const std::optional<torch::Tensor>& k_sf = std::nullopt) {
    const bool is_fp4 = q_sf.has_value();
    // Shape extraction & validation
    const auto& [seq_len_q, num_heads, head_dim] = get_shape<3>(q);
    const auto& [seq_len_k, head_dim_] = get_shape<2>(k);
    const auto& [seq_len_, num_heads_] = get_shape<2>(weights);

    DG_HOST_ASSERT(seq_len_q == seq_len_);
    DG_HOST_ASSERT(num_heads == num_heads_);
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len_q);
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len_q);
    TORCH_CHECK(q.is_contiguous() and k.is_contiguous(), "q and k must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(cu_seq_len_k_start.is_contiguous(), "cu_seq_len_k_start must be contiguous");
    TORCH_CHECK(cu_seq_len_k_end.is_contiguous(), "cu_seq_len_k_end must be contiguous");
    DG_HOST_ASSERT(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat32 or
                   (logits_dtype == torch::kBFloat16 and weights.scalar_type() == torch::kBFloat16));
    if (logits_dtype == torch::kFloat32)
        TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "fp32 logits requires fp32 weights");
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
                      head_dim, clean_logits, max_seqlen_k, logits_dtype, q_sf, k_sf);
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
    const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
    const auto& [schedule_meta_size, meta_info_size] = get_shape<2>(schedule_meta);
    const int64_t kv_cache_stride_bytes = fused_kv_cache.stride(0);

    const int num_sms = get_num_sms();
    DG_HOST_ASSERT(batch_size == context_lens.size(0) and batch_size == block_table.size(0));
    DG_HOST_ASSERT(batch_size_next_n == batch_size * next_n);
    DG_HOST_ASSERT(num_heads == num_heads_ and num_heads_kv == 1);
    DG_HOST_ASSERT((schedule_meta_size - 1) % num_sms == 0 and meta_info_size == 2);
    DG_HOST_ASSERT(1 <= next_n and next_n <= 6);
    DG_HOST_ASSERT(block_kv == 64);

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == kv_last_dim);
    DG_HOST_ASSERT(fused_kv_cache.stride(2) == kv_last_dim);
    DG_HOST_ASSERT(fused_kv_cache.stride(3) == 1);
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat32 or
                   (logits_dtype == torch::kBFloat16 and weights.scalar_type() == torch::kBFloat16));
    DG_HOST_ASSERT(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16);
    if (logits_dtype == torch::kFloat32)
        TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "fp32 logits requires fp32 weights");
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
    // BF16 carries no per-token KV scale
    const auto& k_scales = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    return mqa_logits_common(q, kv, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                             max_seqlen_k, logits_dtype);
}

torch::Tensor fp8_mqa_logits(const torch::Tensor& q, const std::pair<torch::Tensor, torch::Tensor>& kv_s,
                             const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                             const torch::Tensor& cu_seq_len_k_end, bool clean_logits = true, int max_seqlen_k = 0,
                             torch::ScalarType logits_dtype = torch::kFloat32) {
    return mqa_logits_common(q, kv_s.first, kv_s.second, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                             max_seqlen_k, logits_dtype);
}

torch::Tensor int8_mqa_logits(const torch::Tensor& q, const std::pair<torch::Tensor, torch::Tensor>& kv_s,
                              const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                              const torch::Tensor& cu_seq_len_k_end, bool clean_logits = true, int max_seqlen_k = 0,
                              torch::ScalarType logits_dtype = torch::kFloat32) {
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
        const auto& empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(q.first.device()));
        return mqa_logits_common(q.first, kv.first, empty, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                                 max_seqlen_k, logits_dtype, q.second, kv.second);
    }
    return mqa_logits_common(q.first, kv.first, kv.second, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits,
                             max_seqlen_k, logits_dtype);
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
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                   max_context_len, clean_logits, logits_dtype);
}

torch::Tensor fp8_paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                   const torch::Tensor& weights, const torch::Tensor& context_lens,
                                   const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                   int max_context_len, bool clean_logits = true,
                                   torch::ScalarType logits_dtype = torch::kFloat32) {
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                   max_context_len, clean_logits, logits_dtype);
}

torch::Tensor int8_paged_mqa_logits(const torch::Tensor& q, const torch::Tensor& fused_kv_cache,
                                    const torch::Tensor& weights, const torch::Tensor& context_lens,
                                    const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                    int max_context_len, bool clean_logits = true,
                                    torch::ScalarType logits_dtype = torch::kFloat32) {
    return paged_mqa_logits_common(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                   max_context_len, clean_logits, logits_dtype);
}

// Unified FP8/FP4 entry point (paged). `q = (q_fp, optional q_sf)`: presence of `q_sf` selects the
// FP4 kernel, otherwise the FP8/INT8/BF16 one. See `paged_mqa_logits_common` for tensor layouts.
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
    return paged_mqa_logits_common(q.first, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                   max_context_len, clean_logits, logits_dtype, q.second);
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
}

} // namespace deep_gemm::attention
