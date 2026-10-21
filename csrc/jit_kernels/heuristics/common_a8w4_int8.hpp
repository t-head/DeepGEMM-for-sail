#pragma once
#include <cstdint>
#include <vector>
#include <torch/torch.h>
#include <deep_gemm/common/types.cuh>
#include "common_int8.hpp"
#include "../../utils/utils.hpp"

namespace deep_gemm_a8w4_common {

// ============================================================================
// A8W4 INT8 Heuristic Tile Selection
// ============================================================================

struct A8W4HeuristicConfig {
    int block_m;
    int block_n;
    int block_k;
    int warp_m;
    int warp_n;
    int num_stages;
};

// A tile dimension that does not divide the problem dimension pays for every padded lane: the
// MMA and INT4-dequant work grows with the padded width, and a packed-INT4 B tile is re-read once
// per M tile.  Return the largest legal power-of-two BN that tiles N exactly; 64 is the floor
// because it is the smallest BN the A8W4 mainloop tiles well.
inline int pick_block_n(int n, int max_block_n) {
    for (int block_n = max_block_n; block_n >= 64; block_n /= 2) {
        if (n % block_n == 0) {
            return block_n;
        }
    }
    return 64;
}

// The A8W4 runtime LUT kind, mirroring `deep_gemm_int8::get_a8w8_lut_kind` so each path resolves
// its own LUT through a getter in its own namespace.
inline deep_gemm_lut::RuntimeLutKind get_a8w4_lut_kind() {
    return deep_gemm_lut::RuntimeLutKind::A8W4INT8;
}

// Self-contained A8W4 tile selection for grouped (Masked / NoPad) workloads.  Mirrors the
// `deep_gemm_int8::get_best_configs` interface: same 8-tuple return, same `runtime_lut_kind` /
// `from_runtime_lut` plumbing, and a runtime-LUT hit takes priority over the heuristic.  `m` is the
// per-group expected_m.  A8W4 only exposes Masked and NoPad, so the rules apply to both;
// `gemm_type` is kept for future layout-specific rules.  There is no built-in LUT for A8W4.
//
// Rules 2 and 3 below were re-fitted on OperatorForge tile sweeps of 12 grouped
// shapes (NoPad + Masked, N=256..7168, K=3072..7168, expected_m 16..3072, real
// zipf routing), timing every legal tile against the A8W8 reference at each
// expected_m.  Every tile returned bit-identical output (INT32 accumulation is
// exact, so tile choice cannot change the result on unpadded rows).
static std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>
get_best_configs(int m, int n, int k, int num_groups, int num_sms,
                 deep_gemm::GemmType gemm_type,
                 std::optional<deep_gemm_lut::RuntimeLutKind> runtime_lut_kind = std::nullopt,
                 bool* from_runtime_lut = nullptr) {
    // A8W4 is only implemented for grouped GEMMs on PPU1.0.
    DG_HOST_ASSERT(get_ppu_arch() == AC_PPU0010 && gemm_type != deep_gemm::GemmType::Normal);
    if (from_runtime_lut) *from_runtime_lut = false;

    int block_m, block_n, block_k, warp_m, warp_n, num_stages;
    std::optional<deep_gemm_int8_lut::Tile> lut_result;
    if (runtime_lut_kind.has_value()) {
        lut_result = deep_gemm_lut::get_runtime_config(m, n, k, num_groups, gemm_type, *runtime_lut_kind);
        if (from_runtime_lut) *from_runtime_lut = lut_result.has_value();
    }
    if (lut_result.has_value()) {
        std::tie(block_m, block_n, block_k, warp_m, warp_n, num_stages) = lut_result.value();
    } else {
        A8W4HeuristicConfig config;
        // ---- Rule 1: Wide short-K ----
        // GLM gemm2 N=6144/K=128, KIMI gemm2 N=3584/K=192.
        // While the per-expert rows fit in one M tile, BM16 beats BM32 by 7-45% (KIMI em=0.9..18,
        // GLM em=1.4..20): the B tile is read once per expert either way, and BM16 does less MMA and
        // dequant work on padded rows.  From em~20 BM16 spills into a second M tile and re-reads B,
        // so BM32 takes over (GLM em=23: 204 vs 218 us, em=27: 222 vs 242).
        // K=192 keeps BK=128 even though K/BK=1.5 leaves a half-K tail: forcing BK=64 measured
        // 1-5% slower, so the deeper K pipeline is worth more than the tail.
        if (n > 512 && k <= 256) {
            config = m <= 20 ? A8W4HeuristicConfig{16, 128, 128, 16, 32, 2}
                             : A8W4HeuristicConfig{32, 128, 128, 32, 32, 2};
        } else if (m <= 16) {
            // ---- Rule 2: Small-M rungs ----
            // Decode-shaped workloads are latency-bound, not tile-count-bound, so they want
            // the narrowest M tile; BK=256 keeps the K pipeline deep enough to cover it.
            config = A8W4HeuristicConfig{16, 64, 256, 16, 16, 2};
        } else if (n <= 256 && k >= 4096 && m <= 48) {
            // N<=256 leaves only two BN=128 column tiles, so there is not enough CTA
            // parallelism to hide a K>=4096 loop: a third pipeline stage beats two by 15%
            // at em=32 (GLM-5 gemm1 N=256/K=6144) and 1.9% at em=43 (KIMI-K2.5 gemm1
            // N=256/K=7168).  Past em=48 the extra M tiles supply the parallelism and
            // stage 2 earns the smem back.
            config = A8W4HeuristicConfig{32, 128, 128, 32, 32, 3};
        } else {
            // ---- Rule 3: shared M ladder for every remaining grouped shape ----
            // Reached by N>512 at any K, by N<=512 when K<4096 (KIMI gemm1 N=384/K=3584),
            // and by long-K small-N above the Rule 2 rungs.
            //
            // Block shape and warp split have to be chosen as a pair.  The same 64x128
            // block is 10-40% slower split 32x32 (8 warps) than split 64x64 (2 warps) at
            // every expected_m, which is what made the old BM64 rung lose to BM32 and made
            // BM128 look necessary above em~200.  With the splits paired correctly BM64
            // wins from em~80 to em~1024 and BM128 only edges ahead above that (0.2-3.8%
            // at em=1500..3072), while BM32 wins below em~80.
            //
            // The BM32/BM64 crossover lands anywhere in em=71..113 depending on shape;
            // 80 is the midpoint, and across the band the two are a tie within run-to-run
            // noise (medians differ by up to 6% but best-of-3 minima agree to 1%).
            //
            // BN=256 never pays off on A8W4 -- it costs 20% at em=128 and 21% at em=512 on
            // Masked N=4096 -- and BN must divide N, so cap it at 128 and reduce by
            // divisibility.  pick_block_n(n, 128) only ever returns 128 or 64.
            const int block_n_tuned = pick_block_n(n, 128);
            if (m > 1024 && block_n_tuned == 128) {
                config = A8W4HeuristicConfig{128, 128, 128, 64, 32, 2};
            } else if (m > 80) {
                config = block_n_tuned == 128 ? A8W4HeuristicConfig{64, 128, 128, 64, 64, 2}
                                              : A8W4HeuristicConfig{64, 64, 128, 32, 32, 2};
            } else {
                config = A8W4HeuristicConfig{32, block_n_tuned, 128, 32, 32, 2};
            }
        }
        block_m = config.block_m;
        block_n = config.block_n;
        block_k = config.block_k;
        warp_m = config.warp_m;
        warp_n = config.warp_n;
        num_stages = config.num_stages;
    }

    const auto smem_config = deep_gemm_int8::get_smem_config(
        num_stages, k, block_m, block_n, block_k, /*bpp=*/1, /*element_b_bits=*/4);
    DG_HOST_ASSERT(std::get<0>(smem_config) <= 262144);
    return std::make_tuple(num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config);
}

// ============================================================================
// A8W4 INT8 GroupedNoPad GEMV
// ============================================================================

struct A8W4GemvConfig {
    int block_size;
    int thread_per_n;
    int n_per_thread;
    int num_unroll;
    int swizzle_m;
};

// Decide whether a GroupedNoPad call with `m` routed rows over `num_groups` experts should use the
// GEMV kernel, and pick its launch shape. For long K, an A8W4 launch-shape scan shows the crossover
// at 0.5 rows/expert (GLM K=6144: +8.5% at 0.5, but -5.1% by 0.75; EP8 switches to GEMM before 0.67).
// Every thread consumes one K32 group (16 packed rhs bytes) per step.
inline bool get_gemv_config(int m, int n, int k, int num_groups, A8W4GemvConfig& config) {
    // Short K: GEMV reads the B tile once per routed row, the tile GEMM once per expert M tile.
    // GEMV therefore only wins while there is at most ~1 row per expert: above that the GEMM
    // reuses B across rows and avoids the per-row launch overhead.  Measured crossovers agree
    // within 1-2% of a 1.25 rows/expert gate on both shapes (GEMV/GEMM, us, against the
    // BM16/BN128/BK128 tile):
    //   KIMI gemm2 (N=3584,K=192,E=896)  135/141 @em=0.68, 158/186 @0.80, 183/178 @0.95,
    //                                    210/183 @1.11, 246/211 @1.30
    //   GLM  gemm2 (N=6144,K=128,E=256)   72/79  @em=1.19,  84/85  @1.41,  98/95  @1.66,
    //                                    111/97  @1.94, 173/111 @3.19
    // The previous 3.25 rows/expert gate was calibrated against the BM32 tile and kept GEMV
    // 2-3x too long (KIMI M=86..167 paid 7-84% on gemm2).
    const bool short_k = k <= 256;
    const bool use_gemv = (short_k && m * 4 <= 5 * num_groups) ||
                          (k > 256 && m <= 0.5 * num_groups);
    if (!use_gemv)
        return false;

    const int num_k32 = k / 32;
    if (num_k32 <= 32) {
        // Short K: a single K pass per thread; the scan's robust shape across GLM K=256 is
        // 64 threads, 8 K threads/output and 8 outputs/thread.
        int thread_per_n = 1;
        while (thread_per_n * 2 <= num_k32 && thread_per_n < 8)
            thread_per_n *= 2;
        config = {/*block_size=*/64, thread_per_n, /*n_per_thread=*/8, /*num_unroll=*/1, /*swizzle_m=*/1};
    } else {
        // Long K: 16 K threads leave 16 independent N lanes in a 256-thread block; four outputs per
        // lane amortize lhs loads. Unroll two K32 groups and predicate the odd tail in the kernel.
        config = {/*block_size=*/256, /*thread_per_n=*/16, /*n_per_thread=*/4, /*num_unroll=*/2,
                  /*swizzle_m=*/1};
    }

    // The kernel has no N tail: shrink the per-thread N work until full blocks tile N.
    const int n_loop_step = config.block_size / config.thread_per_n;
    while (config.n_per_thread > 1 && n % (config.n_per_thread * n_loop_step) != 0)
        config.n_per_thread /= 2;
    return n % (config.n_per_thread * n_loop_step) == 0;
}

// ============================================================================

} // namespace deep_gemm_a8w4_common
