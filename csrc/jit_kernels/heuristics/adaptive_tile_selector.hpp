#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>
#include <optional>
#include <utility>
#include <cassert>
#include <climits>
#include <cstdio>
#include <string>

#include "../../utils/math.hpp"

namespace deep_gemm_adaptive {

using ::ceil_div;

inline bool bf16_adaptive_enabled(int m, int n, int k) {
    // 0: force disable, 1: force enable, other values: use default
    const char* e = std::getenv("DG_BF16_ADAPTIVE");
    if (e != nullptr) {
        if (std::string(e) == "0") return false;
        if (std::string(e) == "1") return true;
    }
    // Default adaptive shape gate
    return true;
}

inline bool int8_adaptive_enabled(int m, int n, int k) {
    // 0: force disable, 1: force enable, other values: use default
    const char* e = std::getenv("DG_INT8_ADAPTIVE");
    if (e != nullptr) {
        if (std::string(e) == "0") return false;
        if (std::string(e) == "1") return true;
    }
    // Default adaptive shape gate
    return (m <= 160 && (
        (n >= 10240 && k >= 1024) ||
        (n == 8192 && k == 16384)
    ));
}

// ============================================================
// Constants (Python lines 28-43)
// ============================================================
static constexpr int SMEM_SIZE           = 256 * 1024;
static constexpr int BASE_BLOCK_K        = 64;
static constexpr int MAX_BLOCK_K         = 512;

static constexpr int WE_PER_CU           = 8;
// Hardware warp limit: 131072 regs / (128 regs/thread * 32 threads/warp) = 32 warps
static constexpr int MAX_WARPS_PER_BLOCK = 32;

// ============================================================
// Env var caching (thread-safe function-local statics)
// ============================================================
// Compute-bound candidate source (env DG_TILE_CANDIDATES):
//   - Not set / "hardcoded": fixed tile list below (the 36 unique tiles the
//     dynamic selector picked across the 550-case compute-bound benchmark)
//   - "dynamic":             enumerate all legal warp grids on the fly
inline bool use_hardcoded_tile_candidates() {
    static const bool val = []() {
        const char* e = std::getenv("DG_TILE_CANDIDATES");
        if (e == nullptr) return true;
        return std::string(e) != "dynamic";
    }();
    return val;
}

// ============================================================
// LUT: measured WarpOnN caps (Python lines 396-427)
// ============================================================
// Each entry caps WarpOnN (number of warps laid out along the block-N
// direction, in warps) for one (BM, WM, WN) warp-grid combo, as measured on
// hardware. Two thresholds per entry:
//   sf (spill-free)    — max WarpOnN at which the kernel runs with NO
//                        register spill. Mirrors the basic_vreg theory tier
//                        (ACC + double-buffered inputs live simultaneously).
//                        This is the cap actually used by selection.
//   ok (spills-but-ok) — max WarpOnN at which the kernel does spill, but the
//                        spill stays within budget (launches and runs with
//                        acceptable performance). Mirrors the extreme_vreg
//                        tier (ACC + single-buffered inputs). Kept here for
//                        reference only; not used by selection.
// Only combos where measurement DIVERGES from the analytic VREG estimate are
// listed (agreement cases were pruned); get_max_warp_on_n falls back to the
// estimate for any (BM, WM, WN) not found here.
struct WarpOnNCap {
    int bm, wm, wn, sf, ok;
};

static constexpr std::array<WarpOnNCap, 25> REGISTER_WARPONN_CAP = {{
    // BM <= 160: memory-bound path (9 entries)
    { 80,  80, 32, 29, 29},
    { 96,  48, 64, 14, 14},
    { 96,  96, 16, 31, 32},
    { 96,  96, 32, 16, 16},
    { 96,  96, 64, 14, 14},
    {112, 112, 16, 22, 22},
    {128,  64, 48, 12, 12},
    {144, 144, 16, 31, 32},
    {160,  80, 16, 15, 16},
    // BM > 160: compute-bound path (16 entries)
    {192,  96, 16, 13, 16},
    {192,  96, 32, 12, 12},
    {192,  96, 48,  8,  8},
    {192,  96, 64,  8,  8},
    {256,  64, 32,  7,  8},
    {256,  64, 48,  6,  6},
    {256, 128, 16, 15, 16},
    {256, 128, 32,  8,  8},
    {256, 128, 48,  1,  8},
    {384,  96, 16,  6,  8},
    {384,  96, 32,  6,  6},
    {384,  96, 64,  4,  4},
    {512,  64, 48,  3,  3},
    {512, 128, 16,  7,  8},
    {512, 128, 32,  4,  4},
    {512, 128, 48,  1,  4},
}};

// Max WarpOnN for a (BM, WM, WN) warp grid: measured spill-free LUT cap
// (sf) first, analytic estimate for uncovered combos.
inline int get_max_warp_on_n(int block_m, int warp_m, int warp_n) {
    for (const auto& e : REGISTER_WARPONN_CAP) {
        if (e.bm == block_m && e.wm == warp_m && e.wn == warp_n) {
            return e.sf;
        }
    }

    // Analytic estimate
    int warp_on_m = block_m / warp_m;

    // Constraint 1: warp budget
    int warp_cap = 32 / warp_on_m;

    // Constraint 2: SMEM capacity
    int smem_cap = (1024 - block_m) / warp_n;

    int hw_max = std::min(warp_cap, smem_cap);

    // Constraint 3: VREG pressure (acc_size=4B, input_size=2B)
    int acc_vreg = warp_m * warp_n * 4 / (32 * 4);
    int input_vreg = (warp_m + warp_n) * 16 * 2 / (32 * 4);
    int basic_vreg = acc_vreg + input_vreg * 2;

    static constexpr int VREG_PER_WARP[3] = {128, 168, 256};
    static constexpr int TOTAL_WARPS[3]   = {32, 24, 16};

    // sf tier: determined by basic_vreg (ACC + double-buffered inputs).
    // (The ok tier would instead use extreme_vreg = acc + single-buffered
    //  inputs; it is not computed here — see LUT comment above.)
    int sf;
    if (basic_vreg <= VREG_PER_WARP[0])      sf = TOTAL_WARPS[0] / warp_on_m;
    else if (basic_vreg <= VREG_PER_WARP[1]) sf = TOTAL_WARPS[1] / warp_on_m;
    else if (basic_vreg <= VREG_PER_WARP[2]) sf = TOTAL_WARPS[2] / warp_on_m;
    else                                     sf = 0;

    return std::min(sf, hw_max);
}

// Spill check for a warp tile: estimated warp-tile minimum regs vs the
// compiler reg limit at the actual warp count.
inline bool tile_spill_ok(int wm, int wn, int total_warps) {
    static constexpr int MISC_REGS = 20;
    int min_regs = (wm * wn) / 32 + (wm + wn) / 2 + MISC_REGS;
    int warps_per_we = ceil_div(total_warps, 8);
    int limit = (warps_per_we == 0) ? 256 : std::min(512 / warps_per_we, 256);
    if (warps_per_we == 3) limit = 168;
    return min_regs <= limit;
}

inline int max_bn_smem(int block_m, int stage) {
    return SMEM_SIZE / (BASE_BLOCK_K * 2 * stage) - block_m;
}

// Return the WARP_K tile size (= block_k / WarpOnK), NOT the WarpOnK factor itself.
// WarpOnK=1 -> warp_k = block_k (no K-split)
// WarpOnK=2 -> warp_k = block_k / 2
inline int get_warp_k(int block_m, int block_n, int block_k, int warp_m, int warp_n, int num_stages) {
    int warp_on_m = std::max(1, block_m / warp_m);
    int warp_on_n = std::max(1, block_n / warp_n);
    int base_warps = warp_on_m * warp_on_n;
    int warp_on_k_max = std::max(1, 32 / base_warps);
    int warp_on_k = block_k / 128;

    if ((block_k == 256 || block_k == 512) && warp_on_k <= warp_on_k_max) {
        return 128;
    } else {
        return block_k;
    }
}

// ============================================================
// Memory-bound path (Python lines 158-512)
// ============================================================
struct MemBoundResult {
    int block_m;
    int warp_m;
    int block_n;
    int warp_n;
    int block_k;
    int num_stages;
    int warp_k;
};

inline MemBoundResult select_tile_memory_bound(int cutlass_n, int cutlass_m, int cutlass_k,
                                                int num_sms) {
    static constexpr std::array<int,4> WN_CANDIDATES_SMALLBM = {16, 32, 48, 64};
    // BM = m ceil-aligned to a multiple of 16 (MMA tile granularity)
    int block_m = ceil_div(cutlass_m, 16) * 16;
    // BM=144 -> 160 correction
    if (block_m == 144) block_m = 160;
    // Warp tile M: identity for BM<=112, half for BM>=128
    int warp_m = (block_m <= 112) ? block_m : block_m / 2;

    auto achievable_bn = [&](int wn) -> int {
        return get_max_warp_on_n(block_m, warp_m, wn) * wn;
    };

    // For BM>=128, WN=16 over-shrinks the warp tile. Drop WN=16.
    std::vector<int> wn_candidates;
    if (block_m >= 128) {
        for (int wn : WN_CANDIDATES_SMALLBM) {
            if (wn >= 32) wn_candidates.push_back(wn);
        }
    } else {
        for (int wn : WN_CANDIDATES_SMALLBM) {
            wn_candidates.push_back(wn);
        }
    }

    // Step 1: plan_max_bn
    int plan_max_bn = 0;
    for (int wn : wn_candidates) {
        plan_max_bn = std::max(plan_max_bn, achievable_bn(wn));
    }
    if (plan_max_bn == 0) plan_max_bn = 1;  // safety

    // Step 2: wave count
    int m_tiles = ceil_div(cutlass_m, block_m);
    int tile_max = m_tiles * ceil_div(cutlass_n, plan_max_bn);
    int wave_upper = ceil_div(tile_max, num_sms);
    if (wave_upper == 0) wave_upper = 1;

    // Step 3: per-wave BN target
    int bn_wave_even = ceil_div(m_tiles * cutlass_n, num_sms * wave_upper);

    // Step 4: pick largest WN whose capacity covers bn_wave_even
    int warp_on_m = block_m / warp_m;
    int warp_n_result = 32;  // default
    for (int wn : wn_candidates) {
        if (achievable_bn(wn) >= bn_wave_even) {
            warp_n_result = wn;
            break;
        }
    }

    // Step 5: block_n = ceil-align bn_wave_even to warp_n
    int block_n = ceil_div(bn_wave_even, warp_n_result) * warp_n_result;

    // Step 6: BK, stages, warp_k — memory-bound policy.
    // Start BK=64, double BK while stages > 3, up to BK=512.
    // Memory-bound keeps stages in {2, 3}.
    int block_k = BASE_BLOCK_K;
    int max_stages = 0;
    while (true) {
        max_stages = SMEM_SIZE / ((block_m + block_n) * block_k * 2);
        if (max_stages <= 3 || block_k >= MAX_BLOCK_K) {
            break;
        }
        block_k *= 2;
    }

    // Cap stages at 3; also bounded by K-iterations.
    int num_stages = std::min({max_stages, 3, ceil_div(cutlass_k, block_k)});
    num_stages = std::max(num_stages, 2);
    int warp_k = get_warp_k(block_m, block_n, block_k, warp_m, warp_n_result, num_stages);

    return {block_m, warp_m, block_n, warp_n_result, block_k, num_stages, warp_k};
}

// ============================================================
// Compute-bound path (Python lines 290-663)
// ============================================================

// Cost model: block makespan ~ K * max(compute, memory) / throughput.
//   compute ~ BM*BN * ceil(tw/8)*8/tw  — WE imbalance is priced (~6.7% per
//                                        empty WE slot) instead of hard-filtered
//   memory  ~ BM + BN                  — per-block A/B operand refetch traffic
// K cancels in comparisons, so the model is K-agnostic; the flops/bytes ratio
// R prices compute against memory (skinny tiles with BM*BN/(BM+BN) < R are
// memory-priced, fat tiles compute-priced). M/N padding is implicitly priced:
// padded rows/cols still pay BM*BN compute and BM+BN traffic.
struct ComputeBoundTile {
    int bn, wm, wn;
    double cost;     // waves * block_time (makespan estimate, for cross-BM compare)
    double traffic;  // blocks * (BM + BN): total per-block operand refetch volume
                     // (tie-breaker for near-equal makespan, m>512 region)
    // Hardcoded-candidate mode only: baked BK/stages/warp_k of the fixed tile
    // (dynamic mode leaves these 0 and derives them via select_adaptive_smem).
    int block_k = 0, num_stages = 0, warp_k = 0;
};

// Shape-independent warp-grid candidate for one block_m: everything here is
// determined solely by block_m (+ WarpOnN LUT/register model). The
// shape-dependent metrics (waves/last_util/valid_util/cost) are added by
// compute_bound_tile when scoring against (m, n, num_sms).
struct ComputeCandidate {
    int wm, wn, bn, total_warps, we_r, warp_imb;
};

// Pass 1 (shape-independent): enumerate all legal (WM, WN, WarpOnN) warp
// grids for a given block_m, i.e. the full candidate tile set for the
// compute-bound path:
//   WM   = BM/2^k divisors that are multiples of 16 (MMA warp granularity),
//          capped at warp_on_m <= 8
//   WN   = 16-step from 32 to 128. WN=16 is excluded: skinny warp tiles
//          under-utilize the tensor pipeline and lose B-operand reuse
//   WoN  = 1..get_max_warp_on_n(BM, WM, WN), with tw >= 8 warps (one full WE
//          round — below it the CU front-end cannot hide latency; a cliff,
//          not a priceable penalty) and tile_spill_ok (candidate-level spill
//          gate at the actual warp count). BN = WoN x WN.
// BM legality: BM=128 is allowed (wins in the underfilled m>512 sub-region,
// e.g. 128x256 at (1024,1024) in the run-152 design sweep). BM=144 has no
// power-of-2-halving WM that is a multiple of 16; BM=160 is excluded
// conservatively (unproven in the compute-bound path). Illegal BMs yield an
// empty set.
inline std::vector<ComputeCandidate> compute_bound_candidates(int block_m) {
    static constexpr std::array<int, 7> WN_LIST = {32, 48, 64, 80, 96, 112, 128};
    std::vector<ComputeCandidate> cands;
    if (block_m < 128 || block_m == 144 || block_m == 160) return cands;

    for (int wm = block_m / 2; wm >= 16; wm /= 2) {
        if (block_m % wm != 0) continue;
        if (wm % 16 != 0) continue;
        int warp_on_m = block_m / wm;
        if (warp_on_m > 8) continue;

        for (int wn : WN_LIST) {
            int max_won = get_max_warp_on_n(block_m, wm, wn);
            if (max_won < 1) continue;

            for (int won = 1; won <= max_won; won++) {
                int total_warps = warp_on_m * won;
                // No separate warp-count filter needed: both get_max_warp_on_n
                // paths stay within the 32-warp HW bound (MAX_WARPS_PER_BLOCK):
                // LUT sf/ok are measured up to it (e.g. {256,64,32}: sf=7 ->
                // 28 warps verified on HW), and the analytic warp_cap =
                // 32/warp_on_m is exactly a 32-warp budget.
                if (total_warps < WE_PER_CU) continue;
                // Candidate-level spill gate: get_max_warp_on_n is a shape-level
                // loop bound (won-blind); this checks the actual warp count
                // against the compiler regs/WARP limit (e.g. BM=192 w48x112:
                // estimate allows won<=4, but min_regs 268 regs > 256 regs limit).
                if (!tile_spill_ok(wm, wn, total_warps)) continue;

                // WE-balance rank: balanced grids ordered 16 > 24 > 8 warps;
                // any non-multiple-of-8 grid ranks after all balanced ones
                int we_r = (total_warps % WE_PER_CU != 0) ? 100 + (MAX_WARPS_PER_BLOCK - total_warps)
                         : (total_warps == 16) ? 0
                         : (total_warps == 24) ? 1
                         : (total_warps == 8)  ? 2 : 3;
                int warp_imb = std::abs(warp_on_m - won);

                cands.push_back({wm, wn, won * wn, total_warps, we_r, warp_imb});
            }
        }
    }
    return cands;
}

// Hardcoded compute-bound tile list: the 36 unique tiles the dynamic selector
// picked across the 550-case compute-bound benchmark (run-149 selector,
// num_sms=39 CUs). Sorted by BM ascending. Fields:
// (BM, BN, BK, WM, WN, warp_k, stages).
struct HardcodedTile {
    int bm, bn, bk, wm, wn, warp_k, stages;
};

static constexpr std::array<HardcodedTile, 37> HARDCODED_COMPUTE_TILES = {{
    {192,  64, 128, 48, 32, 128, 4},
    {192,  96, 128, 48, 48, 128, 3},
    {192, 128, 128, 48, 32, 128, 3},
    {192, 144, 128, 48, 48, 128, 3},
    {192, 160,  64, 48, 80,  64, 4},
    {192, 192,  64, 48, 48,  64, 4},
    {192, 224,  64, 48, 32,  64, 4},
    {192, 240,  64, 48, 48,  64, 4},
    {192, 256,  64, 48, 64,  64, 4},
    {192, 288,  64, 48, 48,  64, 4},
    {192, 320,  64, 48, 80,  64, 4},
    {192, 336,  64, 96, 48,  64, 3},
    {192, 384,  64, 48, 96,  64, 3},
    {256,  64, 128, 32, 32, 128, 3},
    {256,  80, 128, 32, 80, 128, 3},
    {256,  96,  64, 32, 48,  64, 4},
    {256, 112,  64, 32, 112, 64, 4},
    {256, 128,  64, 64, 32,  64, 4},
    {256, 144,  64, 32, 48,  64, 4},
    {256, 160,  64, 32, 80,  64, 4},
    {256, 192,  64, 64, 48,  64, 4},
    {256, 224,  64, 32, 112, 64, 4},
    {256, 240,  64, 32, 80,  64, 4},
    {256, 256,  64, 64, 64,  64, 4},
    {256, 320,  64, 64, 80,  64, 3},
    {320, 128,  64, 80, 32,  64, 4},
    {320, 160,  64, 80, 32,  64, 4},
    {320, 192,  64, 80, 48,  64, 4},
    {320, 256,  64, 80, 64,  64, 3},
    {384, 128,  64, 96, 32,  64, 4},
    {384, 144,  64, 48, 48,  64, 3},
    {384, 160,  64, 48, 80,  64, 3},
    {384, 192,  64, 96, 48,  64, 3},
    {448, 128,  64, 112, 32, 64, 3},
    {512, 128,  64, 128, 32, 64, 3},
    {512, 128,  64, 64, 64,  64, 3},
    {512, 160,  64, 64, 80,  64, 3},
}};

// Hardcoded-mode candidate source: the fixed tiles of HARDCODED_COMPUTE_TILES
// whose BM matches block_m, converted to warp-grid candidates (BN/WM/WN/warp
// count derived from the tile fields; BK/stages/warp_k are baked in and
// returned via ComputeBoundTile instead of select_adaptive_smem).
inline std::vector<ComputeCandidate> hardcoded_candidates(int block_m) {
    std::vector<ComputeCandidate> cands;
    for (const auto& t : HARDCODED_COMPUTE_TILES) {
        if (t.bm != block_m) continue;
        int warp_on_m = t.bm / t.wm;
        int won = t.bn / t.wn;
        int total_warps = warp_on_m * won;
        int we_r = (total_warps % WE_PER_CU != 0) ? 100 + (MAX_WARPS_PER_BLOCK - total_warps)
                 : (total_warps == 16) ? 0
                 : (total_warps == 24) ? 1
                 : (total_warps == 8)  ? 2 : 3;
        cands.push_back({t.wm, t.wn, t.bn, total_warps, we_r,
                         std::abs(warp_on_m - won)});
    }
    return cands;
}

inline std::optional<ComputeBoundTile> compute_bound_tile(int block_m, int m, int n, int /*k*/, int num_sms) {
    // Flops/bytes ratio R of the makespan cost model (env DG_TILE_COST_RATIO)
    static const double cost_ratio = []() {
        const char* e = std::getenv("DG_TILE_COST_RATIO");
        if (e == nullptr) return 96.0;
        double v = std::atof(e);
        return v > 0.0 ? v : 96.0;
    }();

    // Pass 2: score each candidate against (m, n, num_sms). Candidate source:
    // hardcoded fixed tile list (default) or dynamic enumeration.
    const bool hardcoded = use_hardcoded_tile_candidates();
    const auto base_cands = hardcoded ? hardcoded_candidates(block_m)
                                      : compute_bound_candidates(block_m);

    struct Candidate {
        int wm, wn, bn, waves, total_warps, we_r, warp_imb;
        double last_util, valid_util, cost;
    };
    std::vector<Candidate> candidates;

    int m_tiles = ceil_div(m, block_m);
    for (const auto& c : base_cands) {
        int blocks = m_tiles * ceil_div(n, c.bn);
        int waves = ceil_div(blocks, num_sms);
        double last_util = static_cast<double>(blocks - (waves - 1) * num_sms) / num_sms;
        double valid_util = static_cast<double>(n) / (ceil_div(n, c.bn) * c.bn);

        // Makespan estimate: waves * max(compute, memory) block time.
        double we_pen = static_cast<double>(ceil_div(c.total_warps, WE_PER_CU) * WE_PER_CU)
                      / c.total_warps;
        double block_time = std::max(
            static_cast<double>(block_m) * c.bn * we_pen / cost_ratio,
            static_cast<double>(block_m + c.bn));
        double cost = waves * block_time;

        candidates.push_back({c.wm, c.wn, c.bn, waves, c.total_warps, c.we_r, c.warp_imb,
                              last_util, valid_util, cost});
    }

    if (candidates.empty()) return std::nullopt;

    // Within-BM scoring (total order, deterministic):
    //   1. waves      — wave quantization dominates makespan
    //   2. last_util  — SM fill of the last wave (≈ minimizes BN within a wave count)
    //   3. valid_util — N-padding waste
    //   4. we_r       — WE-balanced grids: 16 > 24 > 8 warps
    //   5. warp_imb   — square-ish warp grid (|WoM-WoN|) for L2 reuse
    //   6. wm*wn      — larger warp tile (register-level reuse)
    //   7. wn         — final tie-break: fatter warp-N (unique per candidate)
    // Run-148 lesson: do NOT use the makespan cost for the within-BM choice —
    // its compute term assumes constant per-CU efficiency, but small warp tiles
    // (wm*wn) run slower per FLOP, so the cost model overvalued wave-count cuts
    // from small-warp-tile configs (e.g. 256x144_w32x48 over 256x320_w64x80,
    // ~15% slower in practice). The cost model is only used by the caller for
    // the cross-BM comparison, where the wave/fill metrics cannot compare.
    auto best = std::min_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return std::make_tuple(a.waves, -a.last_util, -a.valid_util, a.we_r,
                                   a.warp_imb, -(a.wm * a.wn), -a.wn)
                 < std::make_tuple(b.waves, -b.last_util, -b.valid_util, b.we_r,
                                   b.warp_imb, -(b.wm * b.wn), -b.wn);
        });

    int best_blocks = m_tiles * ceil_div(n, best->bn);
    double best_traffic = static_cast<double>(best_blocks) * (block_m + best->bn);
    if (hardcoded) {
        // Bake the fixed tile's BK/stages/warp_k into the result
        for (const auto& t : HARDCODED_COMPUTE_TILES) {
            if (t.bm == block_m && t.bn == best->bn && t.wm == best->wm && t.wn == best->wn) {
                return ComputeBoundTile{best->bn, best->wm, best->wn, best->cost, best_traffic,
                                        t.bk, t.stages, t.warp_k};
            }
        }
        assert(false && "hardcoded candidate not found in HARDCODED_COMPUTE_TILES");
    }
    return ComputeBoundTile{best->bn, best->wm, best->wn, best->cost, best_traffic};
}

// Select (block_k, num_stages) to maximize SMEM fill: each (BK, S) pair occupies
// S * (BK/64) slabs of the (BM+BN)*64*2B budget; the fullest feasible pair wins.
// K-aware:
//   - stages deeper than the K-iteration count never fill, so S is capped at
//     max(2, ceil(k/bk)) (mirrors the memory-bound clamp);
//   - among equal-fill pairs, prefer the least K-tail waste (k_iters*bk - k),
//     then deeper pipeline (larger S), then smaller BK.
inline std::pair<int,int> select_adaptive_smem(int block_m, int block_n, int k) {
    static constexpr std::array<int,3> STAGE_OPTIONS = {2, 3, 4};
    double bound = static_cast<double>(SMEM_SIZE) / (static_cast<double>(block_m + block_n) * BASE_BLOCK_K * 2.0);
    int best_bk = 0, best_s = 0;
    double best_gap = 1e18;
    int best_tail = INT_MAX;

    for (int bk = BASE_BLOCK_K; bk <= MAX_BLOCK_K; bk *= 2) {
        int k_iters = ceil_div(k, bk);
        int s_cap = std::max(2, std::min((int)STAGE_OPTIONS.back(), k_iters));
        int tail = k_iters * bk - k;
        for (int s : STAGE_OPTIONS) {
            if (s > s_cap) continue;
            double val = static_cast<double>(s) * (bk / BASE_BLOCK_K);
            if (val > bound) continue;
            double gap = bound - val;
            bool update = false;
            if (best_bk == 0) {
                update = true;
            } else if (gap < best_gap) {
                update = true;
            } else if (gap == best_gap) {
                if (tail != best_tail)  update = (tail < best_tail);
                else if (s != best_s)   update = (s > best_s);
                else                    update = (bk < best_bk);
            }
            if (update) {
                best_bk = bk;
                best_s = s;
                best_gap = gap;
                best_tail = tail;
            }
        }
    }
    if (best_bk == 0) {
        // Unreachable from the compute path (the BN cap guarantees bound >= 2,
        // so (BK=64, S=2) is always feasible); defensive default.
        return {BASE_BLOCK_K, 2};
    }
    return {best_bk, best_s};
}

// ============================================================
// M>512 wave-fit tile selection (replaces the cross-BM ladder there).
// ============================================================
// History of this region (all A/B'd on the 462-case M>512 subset vs
// acblas run_id=150 / old dev run_id=151, 890P 39CU, BF16 K=1024):
//   - cross-BM makespan ladder: geomean 0.63 vs acblas (calibrated for
//     m<=512; picks high-M-padding BM=192 tiles here).
//   - free cost-model choice:   geomean ~1.00, but bimodal — 8-warp
//     128x256 mispicks in multi-wave regions (M=640..1280 x N>=4608:
//     -8..-11% vs old 256x256) and deep-wave BM=128 BN>=384 (doubled
//     A-refetch: 128x512 family geomean 0.91 vs acblas).
// Current policy: override the legacy 256x256 pin only when the geometry
// case is unambiguous:
//   - waves==1: fewer/less-padded tiles reliably win (underfilled grid);
//   - waves>=2: require >=10% better M-padding utilization than BM=256
//     (full-700 A/B: margin-1.111 bucket (M=1152 family) wins 37/43 at
//     gm 0.935 vs old; margin-1.091 bucket (M=1408 family) loses at
//     gm 1.026 — 1.08 mis-gated it, 1.10 rejects only that bucket).
// BM=128 is capped at BN=256 (traffic bound, see above).
struct Mgt512BmConfig {
    int bm, wm;
    std::array<std::pair<int,int>,3> bn_list;  // (BN, WN)
    int bn_count;
};
static constexpr std::array<Mgt512BmConfig,3> MGT512_BM_CONFIGS = {{
    {128, 64, {{{256, 64}, {0, 0}, {0, 0}}}, 1},
    {192, 48, {{{128, 64}, {256, 64}, {0, 0}}}, 2},
    {256, 64, {{{128, 64}, {256, 64}, {0, 0}}}, 2},
}};

// Per-tile overhead in k-step equivalents: pipeline fill/drain + epilogue
// store, charged once per wave round (env DG_M512_OVH_K).
inline int mgt512_ovh_k() {
    static const int val = []() {
        const char* e = std::getenv("DG_M512_OVH_K");
        int v = (e != nullptr) ? std::atoi(e) : 64;
        return v > 0 ? v : 64;
    }();
    return val;
}

struct Mgt512Tile { int bm, bn, wm, wn; };

// Returns nullopt when the legacy 256x256x64/w64x64/S4 pin should hold
// (small N, swapped small-N shapes, or no clearly-better geometry).
inline std::optional<Mgt512Tile> select_tile_m_gt_512(int m, int n, int /*k*/,
                                                      int num_sms) {
    // Small-N (incl. swapped) shapes: legacy pin is validated; ladder/model
    // picks there regress 5-18% vs the old .so (run-151). N>=2048 only.
    if (n < 2048) return std::nullopt;

    struct Cand {
        int bm, bn, wm, wn, blocks, waves;
        double m_util, cost;
    };
    std::vector<Cand> cands;
    for (const auto& cfg : MGT512_BM_CONFIGS) {
        int mt = ceil_div(m, cfg.bm);
        int wom = cfg.bm / cfg.wm;
        for (int i = 0; i < cfg.bn_count; i++) {
            int bn = cfg.bn_list[i].first, wn = cfg.bn_list[i].second;
            int tw = wom * (bn / wn);
            if (tw < 8 || tw > MAX_WARPS_PER_BLOCK) continue;
            if (!tile_spill_ok(cfg.wm, wn, tw)) continue;
            int nt = ceil_div(n, bn);
            int blocks = mt * nt;
            int waves = ceil_div(blocks, num_sms);
            double we = double(ceil_div(tw, WE_PER_CU) * WE_PER_CU) / tw;
            double cost = waves * (cfg.bm * bn * we
                                   + (cfg.bm + bn) * mgt512_ovh_k());
            cands.push_back({cfg.bm, bn, cfg.wm, wn, blocks, waves,
                             double(m) / (mt * cfg.bm), cost});
        }
    }
    if (cands.empty()) return std::nullopt;

    // Baseline = BM=256 candidate with the lowest cost (256x256 preferred
    // over 256x128 via the -bn tie-break below).
    const Cand* base256 = nullptr;
    for (const auto& c : cands) {
        if (c.bm != 256) continue;
        if (base256 == nullptr || c.cost < base256->cost
            || (c.cost == base256->cost && c.bn > base256->bn))
            base256 = &c;
    }
    const Cand* best = nullptr;
    for (const auto& c : cands) {
        if (best == nullptr || c.cost < best->cost
            || (c.cost == best->cost
                && (c.waves < best->waves
                    || (c.waves == best->waves && c.bn > best->bn))))
            best = &c;
    }
    if (best->bm == 256) return std::nullopt;  // legacy pin already optimal

    bool accept;
    if (best->waves == 1) {
        accept = true;  // wave-fit wins are reliable at 1 wave
    } else {
        double base_util = base256 != nullptr ? base256->m_util : 1.0;
        accept = best->m_util > base_util * 1.10;  // need real padding win
    }
    if (!accept) return std::nullopt;
    return Mgt512Tile{best->bm, best->bn, best->wm, best->wn};
}

// ============================================================
// Main entry (Python lines 666-744)
// ============================================================

using AdaptiveResult = std::tuple<int, int, int, int, int, int, int, int>;
// (num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages)

inline AdaptiveResult get_adaptive_configs_impl(int m, int n, int k, int num_sms);

inline AdaptiveResult get_adaptive_configs(int m, int n, int k, int num_sms) {
    if (n < m) {
        auto [ns, bm, bn, bk, wm, wn, wk, stages] = get_adaptive_configs_impl(n, m, k, num_sms);
        // Swap BM<->BN and WM<->WN
        int new_bm = bn, new_bn = bm;
        int new_wm = wn, new_wn = wm;
        int num_tiles = ceil_div(m, new_bm) * ceil_div(n, new_bn);
        return {std::min(num_tiles, num_sms), new_bm, new_bn, bk, new_wm, new_wn, wk, stages};
    }
    return get_adaptive_configs_impl(m, n, k, num_sms);
}

inline AdaptiveResult get_adaptive_configs_impl(int m, int n, int k, int num_sms) {
    // NOTE: the old m > 512 && n > 512 hard-code (256x256x64, w64x64, S4) is
    // removed. That region now flows through the same compute-bound cross-BM
    // machinery below, with two region-scoped extensions (see the loop):
    //   1. bm_floor = 128 — the underfilled sub-region (blocks ~< 2*num_sms)
    //      is often won by BM=128 tiles: sweep (1024,1024): 128x256 15.63 us
    //      vs 256x256 22.05 us; (544,4096): 192x384 24.75 us vs 37.43 us.
    //   2. near-tie traffic tie-break — in the deep-wave sub-region the
    //      makespan model degenerates to ties among all WE-balanced
    //      compute-priced tiles (cost ~= m*n/(num_sms*R)); sweeping showed
    //      the empirical winner is the min-traffic one ((1024,40960):
    //      256x256 309.45 us < 512x128 311.03 us < 128x512 341.09 us;
    //      (1024,1024): 128x256 < 192x192 17.80 us).
    // The fixed 256x256 remains the model's pick for deep-wave shapes, so
    // removing the hard-code only changes underfilled/misaligned shapes.

    int block_m = 0, block_n = 0, block_k = 0, warp_m = 0, warp_n = 0, warp_k = 0, num_stages = 0;

    if (m <= 160) {
        // Memory-bound path: BM/WM derivation + BN/WN/BK/stages/warp_k
        // co-selection all happen inside select_tile_memory_bound
        auto res = select_tile_memory_bound(n, m, k, num_sms);
        int num_tiles = ceil_div(m, res.block_m) * ceil_div(n, res.block_n);
        return {std::min(num_tiles, num_sms), res.block_m, res.block_n, res.block_k,
                res.warp_m, res.warp_n, res.warp_k, res.num_stages};
    } else if (m > 512) {
        // M>512: wave-fit override on top of the legacy 256x256 pin (see
        // select_tile_m_gt_512). Nullopt keeps the legacy tile.
        auto t = select_tile_m_gt_512(m, n, k, num_sms);
        if (t.has_value()) {
            block_m = t->bm;
            block_n = t->bn;
            warp_m = t->wm;
            warp_n = t->wn;
            auto [bk, s] = select_adaptive_smem(block_m, block_n, k);
            block_k = bk;
            num_stages = s;
            warp_k = get_warp_k(block_m, block_n, block_k, warp_m, warp_n,
                                num_stages);
        } else {
            block_m = 256;
            block_n = 256;
            warp_m = 64;
            warp_n = 64;
            block_k = 64;
            warp_k = 64;
            num_stages = 4;
        }
    } else {
        // Compute-bound path: global cross-BM comparison. Every ladder BM from
        // bm_ceil down to the floor contributes its best (BN,WM,WN)
        // candidate. For m <= 512 the floor is 192 and the makespan-priced
        // global minimum wins (ties keep the larger BM) — this replaces the
        // old BM-first + sm_fill_floor policy, which accepted the first BM
        // reaching 30 blocks without ever comparing it against smaller BMs
        // (run-147 regression clusters: m=416/448 took a 2-wave skinny-BN
        // BM=448 tile over the 1-wave fat-BN BM=256 tile; m=480/512 small-n
        // took BM=512 skinny-BN with 2x A-refetch traffic; the WE%8 hard
        // filter plus the block floor also killed faster WE-imbalanced tiles
        // at m=288/320).
        // For m > 512 the floor extends to 128 and near-ties (within 2%) in
        // makespan are re-ranked by total refetch traffic (see above).
        // Compute-bound BM ladder (m > 160): the BMs proven in the compute path.
        // BM=128 only enters as bm_floor when m > 512.
        static constexpr std::array<int,7> BLOCKM_COMPUTE_CANDIDATES = {128, 192, 256, 320, 384, 448, 512};
        const int bm_floor = (m > 512) ? 128 : 192;
        // Upper BM bound: smallest compute-ladder BM covering m (stays 512 for
        // m > 512: full ladder scan, M-padding is priced by the makespan model).
        int bm_ceil = BLOCKM_COMPUTE_CANDIDATES.back();
        for (int c : BLOCKM_COMPUTE_CANDIDATES) {
            if (c >= m) { bm_ceil = c; break; }
        }
        block_m = bm_ceil;  // fallback BM if no candidate wins below
        bool found = false;
        double best_cost = 1e300;
        double best_traffic = 1e300;

        for (int i = (int)BLOCKM_COMPUTE_CANDIDATES.size() - 1; i >= 0; i--) {
            int trial_bm = BLOCKM_COMPUTE_CANDIDATES[i];
            if (trial_bm > bm_ceil) continue;
            if (trial_bm < bm_floor) break;  // ladder is scanned descending
            auto ct = compute_bound_tile(trial_bm, m, n, k, num_sms);
            if (!ct.has_value()) continue;
            bool better;
            if (m > 512) {
                better = (ct->cost < best_cost * 0.98)
                      || (ct->cost <= best_cost * 1.02 && ct->traffic < best_traffic);
            } else {
                better = ct->cost < best_cost;
            }
            if (better) {
                best_cost = ct->cost;
                best_traffic = ct->traffic;
                block_m = trial_bm;
                block_n = ct->bn;
                warp_m = ct->wm;
                warp_n = ct->wn;
                if (ct->block_k > 0) {  // hardcoded mode: baked BK/stages/warp_k
                    block_k = ct->block_k;
                    num_stages = ct->num_stages;
                    warp_k = ct->warp_k;
                }
                found = true;
            }
        }

        if (!found) {
            block_m = 256;
            block_n = 256;
            warp_m = 64;
            warp_n = 64;
            block_k = 64;
            warp_k = 64;
            num_stages = 4;
        }

        if (block_k == 0) {
            // Dynamic mode (or the nearly-unreachable fallback above):
            // derive BK/stages/warp_k from SMEM fill.
            auto [bk, s] = select_adaptive_smem(block_m, block_n, k);
            block_k = bk;
            num_stages = s;
            warp_k = get_warp_k(block_m, block_n, block_k, warp_m, warp_n, num_stages);
        }
    }

    int num_tiles = ceil_div(m, block_m) * ceil_div(n, block_n);
    return {std::min(num_tiles, num_sms), block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages};
}

} // namespace deep_gemm_adaptive
