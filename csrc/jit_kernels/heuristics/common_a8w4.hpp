#pragma once
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include <deep_gemm/common/utils_rtc.cuh>

namespace deep_gemm_a8w4_common {

// ============================================================================
// A8W4 INT8 Device Weight Preprocess
// ============================================================================
// The model stores weights in the portable packed signed-INT4 format. This device-aware transform
// only reorders packed bytes into the layout expected by the PPU A8W4 kernels.

/// Reorder portable packed signed-INT4 MoE weights into the PPU A8W4 kernel layout.
/// `weight` is contiguous uint8 with shape [..., N, K / 2]. Byte i stores logical k=2*i in the low
/// nibble and k=2*i+1 in the high nibble, both in two's-complement signed-INT4 representation.
// NOTE: the reorder is a byte gather inside every 16-byte (32-element K) group; nibbles never cross
// bytes, so each byte keeps its (even k, odd k) pair. Output byte j takes packed byte src[j]:
//   j      :  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15
//   src[j] :  0  1  8  9 |  2  3 10 11 |  4  5 12 13 |  6  7 14 15
// i.e. a 2x4 -> 4x2 transpose of 2-byte units, so one TSM_LD hands each lane exactly the K values of
// its two K32 INT8-MMA atoms without any cross-lane shuffle.
inline torch::Tensor preprocess_a8w4_weight(const torch::Tensor& weight) {
    TORCH_CHECK(weight.dtype() == torch::kUInt8,
                "preprocess_a8w4_weight: weight must use portable packed signed-INT4 uint8 storage");
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
// A8W4 INT8 Heuristic Finalization
// ============================================================================

struct A8W4HeuristicConfig {
    int block_m;
    int block_n;
    int block_k;
    int warp_m;
    int warp_n;
    int num_stages;
};

// Run the shared INT8 search to completion first, then replace its final result only for measured
// A8W4 workload classes. Unmatched shapes retain the complete shared-search configuration.
inline A8W4HeuristicConfig finalize_heuristic_config(
        int m, int n, int k, int num_groups, GemmType gemm_type,
        int block_m, int block_n, int block_k, int warp_m, int warp_n, int num_stages) {
    A8W4HeuristicConfig config{block_m, block_n, block_k, warp_m, warp_n, num_stages};

    // Wide short-K, e.g. GLM gemm2 N=6144/K=256. Validated by TP and EP sweeps.
    const bool is_wide_short_k =
        gemm_type != GemmType::GroupedContiguous && n > 512 && k <= 256;
    if (is_wide_short_k) {
        return {32, 128, 128, 32, 32, 2};
    }

    // Large prefill: BM256/BN256 or BN512 collapses to one CTA/CU. BM128/BN256 preserves
    // two resident CTAs on the measured PPU0010 workloads.
    if (gemm_type != GemmType::GroupedContiguous && k >= 4096 && m > 48 &&
        (block_m > 128 || block_n > 256)) {
        return {128, 256, 128, 64, 64, 2};
    }

    // Long-K gemm1 transition region. Keep measured configurations explicit until broader shape
    // coverage justifies replacing them with a cost model.
    if (gemm_type != GemmType::GroupedContiguous && n <= 512 && k >= 4096) {
        if (m <= 16) {
            return {16, 64, 256, 16, 16, 2};
        }
        if (m <= 48) {
            return {64, 128, 128, 32, 32, 3};
        }
    }

    // Preserve the original packed-B BM16 small-N normalization for unmatched A8W4 shapes.
    if (config.block_m == 16 && n <= 512) {
        config.block_n = 64;
        config.warp_m = 16;
        config.warp_n = 16;
    }

    const int m_per_group = gemm_type == GemmType::GroupedContiguous
        ? (m + num_groups - 1) / num_groups : m;
    if (m_per_group < 6 && (config.block_m == 16 || config.block_m == 32)) {
        config.num_stages = 2;
    }
    return config;
}

// ============================================================================

} // namespace deep_gemm_a8w4_common
