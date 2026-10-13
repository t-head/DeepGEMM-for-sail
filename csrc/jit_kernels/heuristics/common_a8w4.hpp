#pragma once
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>
#include <torch/torch.h>

namespace deep_gemm_a8w4_common {

// ============================================================================
// A8W4 INT8 Weight Pack & Preprocess Utilities
// ============================================================================
// Weights are prepared in two steps, mirroring the usual quant -> pack -> layout-preprocess flow:
//   1. `pack_a8w4_weight`: int8 values in [-8, 7] -> standard packed INT4 bytes (uint8).
//   2. `preprocess_a8w4_weight`: reorder those bytes into the layout the A8W4 kernels expect.

/// Pack signed INT4 values (stored one per int8) into bytes along the last dimension.
/// `weight` is contiguous int8 with shape [..., K] and values in [-8, 7]; the result is uint8 with shape
/// [..., K / 2] where byte i holds k = 2 * i in the low nibble and k = 2 * i + 1 in the high nibble.
// NOTE: uint8 is only a raw byte container; each nibble keeps its own two's-complement sign bit.
inline torch::Tensor pack_a8w4_weight(const torch::Tensor& weight) {
    TORCH_CHECK(weight.dtype() == torch::kInt8, "pack_a8w4_weight: weight must be int8");
    TORCH_CHECK(weight.is_contiguous(), "pack_a8w4_weight: weight must be contiguous");
    TORCH_CHECK(weight.dim() >= 1, "pack_a8w4_weight: weight must have at least 1 dimension");
    const int64_t k = weight.size(-1);
    TORCH_CHECK(k % 2 == 0, "pack_a8w4_weight: the last dimension must be even, got ", k);
    if (weight.numel() > 0) {
        // A single device-to-host copy of (min, max); the packing below stays on `weight`'s device
        const auto [min_val, max_val] = weight.aminmax();
        const auto range = torch::stack({min_val, max_val}).to(torch::kCPU);
        const int lo = range[0].item<int>(), hi = range[1].item<int>();
        TORCH_CHECK(lo >= -8 && hi <= 7, "pack_a8w4_weight: values must be in [-8, 7], got [", lo, ", ", hi, "]");
    }

    // Reinterpret the two's-complement bytes, so masking / shifting are plain logical bit operations
    const auto bytes = weight.view(torch::kUInt8);
    const auto low = bytes.slice(-1, 0, k, 2).bitwise_and(0x0F);
    const auto high = bytes.slice(-1, 1, k, 2).__lshift__(4);
    return low.bitwise_or(high).contiguous();
}

/// Reorder packed INT4 MoE weights into the layout of the A8W4 INT8 kernels.
/// `weight` is the contiguous uint8 output of `pack_a8w4_weight` with shape [..., N, K / 2]; the result
/// has the same shape and dtype.
// NOTE: the reorder is a byte gather inside every 16-byte (32-element K) group; nibbles never cross
// bytes, so each byte keeps its (even k, odd k) pair. Output byte j takes packed byte src[j]:
//   j      :  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15
//   src[j] :  0  1  8  9 |  2  3 10 11 |  4  5 12 13 |  6  7 14 15
// i.e. a 2x4 -> 4x2 transpose of 2-byte units, so one TSM_LD hands each lane exactly the K values of
// its two K32 INT8-MMA atoms without any cross-lane shuffle.
inline torch::Tensor preprocess_a8w4_weight(const torch::Tensor& weight) {
    TORCH_CHECK(weight.dtype() == torch::kUInt8,
                "preprocess_a8w4_weight: weight must be the uint8 output of pack_a8w4_weight");
    TORCH_CHECK(weight.is_contiguous(), "preprocess_a8w4_weight: weight must be contiguous");
    TORCH_CHECK(weight.dim() >= 2, "preprocess_a8w4_weight: weight must have shape [..., N, K / 2]");
    const int64_t k_bytes = weight.size(-1);
    TORCH_CHECK(k_bytes % 16 == 0,
                "preprocess_a8w4_weight: K must be a multiple of 32 (K / 2 = ", k_bytes, ")");

    // [..., K / 32, j1 = 2, j2 = 4] 2-byte units -> [..., K / 32, j2 = 4, j1 = 2]
    std::vector<int64_t> grouped_shape = weight.sizes().vec();
    grouped_shape.back() = k_bytes / 16;
    grouped_shape.push_back(2);
    grouped_shape.push_back(4);
    return weight.view(torch::kInt16).view(grouped_shape).transpose(-1, -2).contiguous()
        .flatten(-3).view(torch::kUInt8);
}

// ============================================================================
// A8W4 INT8 Shared Memory Config
// ============================================================================

/// A8W4 reuses the INT8 tile and stage selection; only the shared memory size differs.
// Same as deep_gemm_int8::get_smem_config except that B packs two int4 per byte along K.
// The epilogue reuses the mainloop shared memory, so the buffer must cover both.
inline std::tuple<int, int, int> get_smem_config_a8w4(int num_stages, int block_m, int block_n, int block_k) {
    const int mainloop_size = num_stages * (block_m * block_k + block_n * block_k / 2);
    const int epilogue_size = block_m * block_n * static_cast<int>(sizeof(c10::BFloat16));
    return std::make_tuple(std::max(mainloop_size, epilogue_size), 128, 0);
}

// ============================================================================

} // namespace deep_gemm_a8w4_common
