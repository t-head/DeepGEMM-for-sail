#pragma once
#include <cstdint>
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
// First transpose 2-byte units inside every 16-byte (32-element K) group so one TSM_LD hands each
// lane the K values of its two K32 INT8-MMA atoms. Then transpose each eight-nibble converter group
// from {a,b,c,d,e,f,g,h} to packed bytes {a,e}, {b,f}, {c,g}, {d,h}. Finally toggle every nibble's
// sign bit, changing two's-complement INT4 into the unsigned, zero-point-8 encoding consumed by the
// seven-instruction register converter.
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
    auto reordered = weight.view(torch::kInt16).view(grouped_shape).transpose(-1, -2).contiguous()
        .flatten(-3).view(torch::kUInt8);

    std::vector<int64_t> converter_shape = reordered.sizes().vec();
    converter_shape.back() = k_bytes / 4;
    converter_shape.push_back(4);
    auto blocks = reordered.view(converter_shape);
    auto low = blocks.bitwise_and(0x0f);
    auto high = blocks.bitwise_right_shift(4);
    auto pack_nibbles = [](const torch::Tensor& lower, const torch::Tensor& upper) {
        return lower.bitwise_or(upper.bitwise_left_shift(4));
    };
    auto transposed = torch::stack({
        pack_nibbles(low.select(-1, 0), low.select(-1, 2)),
        pack_nibbles(high.select(-1, 0), high.select(-1, 2)),
        pack_nibbles(low.select(-1, 1), low.select(-1, 3)),
        pack_nibbles(high.select(-1, 1), high.select(-1, 3)),
    }, -1);
    return transposed.bitwise_xor(0x88).flatten(-2);
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

// Run the shared INT8 search to completion first, then replace its final result
// for grouped (Masked / NoPad) A8W4 workloads.  The M-based tile selection below
// is derived from cross-EP ACExt A8W4 LUT analysis (ep1/ep8/ep16, four (N,K)
// combinations).  GroupedContiguous retains the original shared-search result.
inline A8W4HeuristicConfig finalize_heuristic_config(
        int m, int n, int k, int num_groups, GemmType gemm_type,
        int block_m, int block_n, int block_k, int warp_m, int warp_n, int num_stages) {
    A8W4HeuristicConfig config{block_m, block_n, block_k, warp_m, warp_n, num_stages};

    // ---- Rule 1: Wide short-K (unchanged) ----
    // GLM gemm2 N=6144/K=256.  Validated by TP NoPad 13-point sweep.
    if (n > 512 && k <= 256) {
        return {32, 128, 128, 32, 32, 2};
    }

    // ---- Rule 2: Long-K small-N (GLM gemm1 N=512/K=6144) ----
    // The shared INT8 search already produces good results for this shape class.
    // Only override M<=16 (verified BM16 decode tile) and very small M_per_group
    // stage adjustment; all other M values retain the shared-search configuration.
    if (n <= 512 && k >= 4096) {
        if (m <= 16) {
            return {16, 64, 256, 16, 16, 2};
        }
        if (m <= 48) {
            return {64, 128, 128, 32, 32, 3};
        }
        // M>48: preserve the shared-search tile but ensure stage=2 for very small M.
        const int m_per_group = (m + num_groups - 1) / num_groups;
        if (m_per_group < 6 && (config.block_m == 16 || config.block_m == 32)) {
            config.num_stages = 2;
        }
        // Normalize BM16 small-N warp shape for packed-B.
        if (config.block_m == 16) {
            config.block_n = 64;
            config.warp_m = 16;
            config.warp_n = 16;
        }
        return config;
    }

    // ---- Rule 3: M-based tile selection for remaining grouped shapes ----
    // Applies to N>512 (e.g. N=4096/K=7168, N=7168/K=2048).
    //
    // A8W8 uses dynamic tile for these shapes (BM256/BN256 when expected_m>73).
    // A8W4 cannot use dynamic tile, so we replicate the large-tile selection here.
    // With BM16/BN64 on N=4096, tile count per expert explodes (384 tiles at m=86)
    // and scheduling overhead dominates.  Large tiles cut this by 20-30x.

    // Large expected_m: match A8W8 dynamic tile's large-EM path.
    if (m > 73) {
        return {128, 256, 128, 64, 64, 2};
    }
    // Medium expected_m: BM64 keeps tile count compact on large-N shapes.
    // Without this, BM16 at m=30-73 produces excessive tile count
    // (e.g. 96 tiles/expert at m=43/N=4096) causing 30-40% regression.
    if (m > 32) {
        return {64, 128, 128, 32, 32, 2};
    }
    // Small-medium expected_m: BM16 with larger BN for reasonable tile count.
    if (m > 16) {
        return {16, 128, 128, 16, 32, 2};
    }
    // Small expected_m (decode).
    return {16, 64, 256, 16, 16, 2};
}

// ============================================================================

} // namespace deep_gemm_a8w4_common
