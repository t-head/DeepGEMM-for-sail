#pragma once
#include <cstdint>
#include <vector>
#include <torch/torch.h>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm_a8w4_common {

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
// combinations).  `m` is the per-group expected_m.  A8W4 only exposes Masked and
// NoPad, so the rules currently apply to both; `gemm_type` is kept for future
// layout-specific rules.
inline A8W4HeuristicConfig finalize_heuristic_config(
        int m, int n, int k, int num_groups, deep_gemm::GemmType gemm_type,
        int block_m, int block_n, int block_k, int warp_m, int warp_n, int num_stages) {
    A8W4HeuristicConfig config{block_m, block_n, block_k, warp_m, warp_n, num_stages};

    // ---- Rule 1: Wide short-K (unchanged) ----
    // GLM gemm2 N=6144/K=256.  Validated by TP NoPad 13-point sweep.
    if (n > 512 && k <= 256) {
        return {32, 128, 128, 32, 32, 2};
    }

    // ---- Rule 2: Long-K small-N (GLM gemm1 N=512/K=6144) ----
    // M<=16 uses the verified BM16 decode tile and M<=48 uses BM64.  Larger M keeps
    // the shared-search tile, only forcing stage=2 for very small M_per_group and
    // normalizing the BM16 warp shape for packed-B.
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
