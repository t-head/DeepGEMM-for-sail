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
#include <string>

#include "../../utils/math.hpp"

namespace deep_gemm_adaptive {

using ::ceil_div;

inline bool is_int8_adaptive_shape(int m, int n, int k) {
    return (m <= 160 && (
        (n >= 10240 && k >= 1024) ||
        (n == 8192 && k == 16384)
    ));
}

inline bool is_bf16_adaptive_shape(int m, int n, int k) {
    return (m <= 160 && n >= 5120 && k >= 1024);
}

inline bool bf16_adaptive_enabled(int m, int n, int k) {
    // 0: force disable, 1: force enable, other values: use default
    const char* e = std::getenv("DG_BF16_ADAPTIVE");
    if (e != nullptr) {
        if (std::string(e) == "0") return false;
        if (std::string(e) == "1") return true;
    }
    return is_bf16_adaptive_shape(m, n, k);
}

inline bool int8_adaptive_enabled(int m, int n, int k) {
    // 0: force disable, 1: force enable, other values: use default
    const char* e = std::getenv("DG_INT8_ADAPTIVE");
    if (e != nullptr) {
        if (std::string(e) == "0") return false;
        if (std::string(e) == "1") return true;
    }
    return is_int8_adaptive_shape(m, n, k);
}

// ============================================================
// Constants (Python lines 28-43)
// ============================================================
static constexpr int SMEM_SIZE           = 256 * 1024;
static constexpr int BASE_BLOCK_K        = 64;
static constexpr int MAX_BLOCK_K         = 512;
static constexpr int BLOCK_N_MAX         = 992;
static constexpr std::array<int,3> STAGE_OPTIONS     = {2, 3, 4};
static constexpr std::array<int,14> BLOCKM_CANDIDATES = {16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 256, 320, 384};
static constexpr std::array<int,6> WM_CANDIDATES     = {16, 32, 48, 64, 96, 192};
static constexpr std::array<int,4> WN_CANDIDATES_SMALLBM = {16, 32, 48, 64};

static constexpr int WE_PER_CU                = 8;
static constexpr int REG_FILE_SIZE             = 131072;
static constexpr int NATURAL_REGS_PER_THREAD   = 168;
static constexpr int THREADS_PER_WARP          = 32;
static constexpr int MAX_WARPS_PER_BLOCK       = REG_FILE_SIZE / (NATURAL_REGS_PER_THREAD * THREADS_PER_WARP);  // = 24
static constexpr int MISC_REGS                 = 20;

// ============================================================
// Env var caching (thread-safe function-local statics)
// ============================================================
// Whether to use register-pressure model only (skip compute-tile path).
// Controlled by env DG_TILE_REG_MODEL:
//   - Not set (nullptr): enabled by default (conservative: avoids compute-tile overhead)
//   - "1":               explicitly enabled
//   - Any other value:   disabled (allows compute-tile path)
inline bool use_reg_model_only() {
    static const bool val = []() {
        const char* e = std::getenv("DG_TILE_REG_MODEL");
        if (e == nullptr) return true;   // not set → default enabled
        return std::string(e) == "1";    // explicit "1" → enabled; else disabled
    }();
    return val;
}

inline bool compute_tile_enabled() {
    static const bool val = []() {
        const char* e = std::getenv("DG_COMPUTE_TILE");
        return e == nullptr || std::string(e) != "0";
    }();
    return val;
}

// ============================================================
// LUT: _SPILL_FREE_WN_CAP (7 entries, Python lines 114-122)
// ============================================================
inline int spill_free_wn_cap(int wm, bool& found) {
    // Returns the empirical spill-free WN cap for the given WM.
    // Sets found=true if an entry exists, false otherwise.
    switch (wm) {
        case 16:  found = true; return 224;
        case 32:  found = true; return 128;
        case 48:  found = true; return 128;
        case 64:  found = true; return 96;
        case 80:  found = true; return 64;
        case 96:  found = true; return 32;
        case 112: found = true; return 32;
        default:  found = false; return 0;
    }
}

// ============================================================
// LUT: _REGISTER_WARPONN_CAP (27 entries, Python lines 396-427)
// ============================================================
struct WarpOnNCap {
    int bm, wm, wn, sf, ok;
};

static constexpr std::array<WarpOnNCap, 24> REGISTER_WARPONN_CAP = {{
    // BM <= 160: memory-bound path (8 entries)
    { 80,  80, 32, 29, 29},
    { 96,  48, 64, 14, 14},
    { 96,  96, 16, 31, 32},
    { 96,  96, 32, 24, 24},
    { 96,  96, 64, 14, 14},
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

inline bool lookup_register_warponn_cap(int bm, int wm, int wn, int& sf, int& ok) {
    for (const auto& e : REGISTER_WARPONN_CAP) {
        if (e.bm == bm && e.wm == wm && e.wn == wn) {
            sf = e.sf;
            ok = e.ok;
            return true;
        }
    }
    return false;
}

// ============================================================
// Register model functions (Python lines 45-155)
// ============================================================
inline int compute_compiler_limit(int warps_per_block) {
    int warps_per_WE = ceil_div(warps_per_block, 8);
    if (warps_per_WE == 0) return 256;
    int raw = 512 / warps_per_WE;
    if (warps_per_WE == 3) raw = 168;
    return std::min(raw, 256);
}

inline int estimate_minimum_regs(int wm, int wn) {
    return (wm * wn) / 32 + (wm + wn) / 2 + MISC_REGS;
}

inline bool is_model_spill_free(int wm, int wn, int total_warps) {
    return estimate_minimum_regs(wm, wn) <= compute_compiler_limit(total_warps);
}

inline bool tile_spill_ok(int wm, int wn, int total_warps) {
    if (use_reg_model_only()) {
        return is_model_spill_free(wm, wn, total_warps);
    }
    bool found = false;
    int empirical = spill_free_wn_cap(wm, found);
    if (found) {
        return wn <= empirical;
    }
    return is_model_spill_free(wm, wn, total_warps);
}

inline int model_wn_cap(int wm, int total_warps = 0) {
    if (total_warps == 0) total_warps = MAX_WARPS_PER_BLOCK;
    int limit = compute_compiler_limit(total_warps);
    double denom = wm / 32.0 + 0.5;
    if (denom <= 0) return 256;
    double max_wn = (limit - wm / 2.0 - MISC_REGS) / denom;
    return std::max(16, (static_cast<int>(max_wn) / 16) * 16);
}

inline int wn_per_warp_cap(int warp_m) {
    if (use_reg_model_only()) {
        return model_wn_cap(warp_m);
    }
    bool found = false;
    int empirical = spill_free_wn_cap(warp_m, found);
    if (found) {
        return empirical;
    }
    return model_wn_cap(warp_m);
}

// ============================================================
// Forward declarations for mutual recursion
// ============================================================
inline std::pair<int,int> pick_wm_wn(int block_m, int block_n);
inline int get_warp_n(int block_m, int block_n);
inline int valid_warp_n(int block_m, int block_n);
inline int warp_grid_total(int block_m, int block_n);

// ============================================================
// Memory-bound path (Python lines 158-512)
// ============================================================
inline int select_blockm(int cutlass_m) {
    for (int c : BLOCKM_CANDIDATES) {
        if (cutlass_m <= c) return c;
    }
    return 256;
}

inline int get_warp_m(int block_m, int block_n = -1) {
    if (block_m > 160) {
        if (block_n > 0) {
            auto [wm, wn] = pick_wm_wn(block_m, block_n);
            if (wm != 0) return wm;
        }
        return 64;
    }
    // BM <= 160: identity for BM<=112, half for BM>=128
    return (block_m <= 112) ? block_m : block_m / 2;
}

inline int max_warps_on_n_for(int block_m) {
    int warp_m = get_warp_m(block_m);
    int warp_on_m = std::max(1, block_m / warp_m);
    int budget = (block_m > 160) ? MAX_WARPS_PER_BLOCK : 32;
    return std::max(1, budget / warp_on_m);
}

inline std::vector<std::pair<int,int>> valid_wn_candidates(int block_m, int block_n) {
    std::vector<std::pair<int,int>> results;
    if (block_m > 160 || block_n < 16 || block_n % 16 != 0) return results;
    int warp_m = get_warp_m(block_m);
    int warp_on_m = std::max(1, block_m / warp_m);
    int max_warp_on_n = max_warps_on_n_for(block_m);
    int wn_max_regs = wn_per_warp_cap(warp_m);
    int wn_cap = (std::min(wn_max_regs, block_n) / 16) * 16;
    int min_warp_on_n = 1;
    if (warp_on_m == 1) {
        for (int w = 16; w < block_n; w += 16) {
            if (block_n % w == 0 && block_n / w >= 2) {
                min_warp_on_n = 2;
                break;
            }
        }
    }
    for (int wn = wn_cap; wn > 0; wn -= 16) {
        if (block_n % wn == 0) {
            int won = block_n / wn;
            if (won >= min_warp_on_n && won <= max_warp_on_n) {
                int tw = warp_on_m * won;
                if (tw <= MAX_WARPS_PER_BLOCK) {
                    results.push_back({wn, tw});
                }
            }
        }
    }
    return results;
}

inline int valid_warp_n(int block_m, int block_n) {
    if (block_m > 160) {
        auto [wm, wn] = pick_wm_wn(block_m, block_n);
        return wn;  // 0 if None
    }
    int warp_m = get_warp_m(block_m);
    int warp_on_m = std::max(1, block_m / warp_m);
    int mf = model_wn_cap(warp_m);
    auto cands = valid_wn_candidates(block_m, block_n);
    for (auto& [wn, tw] : cands) {
        if (wn > mf && warp_on_m * (block_n / wn) < 5) continue;
        return wn;
    }
    return 0;  // None
}

inline int get_warp_n(int block_m, int block_n) {
    int wn = valid_warp_n(block_m, block_n);
    assert(wn != 0 && "_get_warp_n: no TSM-safe WN for BM/BN. BN must be in the set produced by _snap_bn_to_valid.");
    return wn;
}

inline int snap_bn_to_valid(int block_m, int target_bn, int max_bn) {
    if (target_bn >= 16 && target_bn <= max_bn && valid_warp_n(block_m, target_bn) != 0) {
        return target_bn;
    }
    int radius = 16;
    int max_radius = std::max(target_bn, max_bn);
    while (radius <= max_radius) {
        int cands[2] = {target_bn + radius, target_bn - radius};
        for (int cand : cands) {
            if (cand >= 16 && cand <= max_bn && valid_warp_n(block_m, cand) != 0) {
                return cand;
            }
        }
        radius += 16;
    }
    return 16;
}

inline int max_bn_smem(int block_m, int stage) {
    return SMEM_SIZE / (BASE_BLOCK_K * 2 * stage) - block_m;
}

// Forward declarations needed for memory-bound
inline std::pair<int,int> estimate_warpOnN(int block_m, int warp_m, int warp_n, int acc_size = 4, int input_size = 2);
inline int get_max_warp_on_n(int block_m, int warp_m, int warp_n);
inline int get_warp_k(int block_m, int block_n, int block_k, int warp_m, int warp_n, int num_stages);

struct MemBoundResult {
    int block_n;
    int warp_n;
    int block_k;
    int num_stages;
    int warp_k;
};

inline MemBoundResult select_tile_memory_bound(int cutlass_n, int cutlass_m, int cutlass_k,
                                                int num_sms, int block_m, int warp_m) {
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

    return {block_n, warp_n_result, block_k, num_stages, warp_k};
}

// ============================================================
// Compute-bound path (Python lines 290-663)
// ============================================================
inline int max_bn_for_bm(int block_m) {
    int max_bn = 0;
    for (int wm : WM_CANDIDATES) {
        if (wm > block_m || block_m % wm != 0) continue;
        for (int wn = 16; wn <= 256; wn += 16) {
            int won = get_max_warp_on_n(block_m, wm, wn);
            if (won >= 1) {
                max_bn = std::max(max_bn, won * wn);
            }
        }
    }
    return max_bn;
}

inline int warp_grid_total(int block_m, int block_n) {
    if (block_m > 160) {
        auto [wm, wn] = pick_wm_wn(block_m, block_n);
        if (wm == 0) return MAX_WARPS_PER_BLOCK + 1;
        return (block_m / wm) * (block_n / wn);
    }
    int warp_m = get_warp_m(block_m);
    int warp_on_m = std::max(1, block_m / warp_m);
    int wn = get_warp_n(block_m, block_n);
    int wxn = std::max(1, ceil_div(block_n, wn));
    return warp_on_m * wxn;
}

inline int we_balance_rank(int total_warps) {
    if (total_warps % WE_PER_CU != 0) {
        return 100 + (MAX_WARPS_PER_BLOCK - total_warps);
    }
    if (total_warps == 16) return 0;
    if (total_warps == 24) return 1;
    if (total_warps == 8)  return 2;
    return 3;
}

inline std::pair<int,int> pick_wm_wn(int block_m, int block_n) {
    if (block_n < 16 || block_n % 16 != 0) return {0, 0};

    struct Candidate { int wm, wn, total; };
    std::vector<Candidate> candidates;

    for (int wm : WM_CANDIDATES) {
        if (wm > block_m || block_m % wm != 0) continue;
        int warp_on_m = block_m / wm;
        for (int wn = 16; wn <= block_n; wn += 16) {
            if (block_n % wn != 0) continue;
            int wxn = block_n / wn;
            int total = warp_on_m * wxn;
            if (!tile_spill_ok(wm, wn, total)) continue;
            if (total > MAX_WARPS_PER_BLOCK) continue;
            candidates.push_back({wm, wn, total});
        }
    }
    if (candidates.empty()) return {0, 0};

    // Filter: prefer wn >= 32
    std::vector<Candidate> wide;
    for (auto& c : candidates) {
        if (c.wn >= 32) wide.push_back(c);
    }
    auto& pool = wide.empty() ? candidates : wide;

    // tier1: total%8==0 and warp_on_m<=4
    std::vector<Candidate> tier1;
    for (auto& c : pool) {
        if (c.total % 8 == 0 && (block_m / c.wm) <= 4) {
            tier1.push_back(c);
        }
    }

    if (!tier1.empty()) {
        auto best = std::min_element(tier1.begin(), tier1.end(),
            [&](const Candidate& a, const Candidate& b) {
                auto ka = std::make_tuple(we_balance_rank(a.total),
                    std::abs((block_m / a.wm) - (block_n / a.wn)), -a.wn, -a.wm);
                auto kb = std::make_tuple(we_balance_rank(b.total),
                    std::abs((block_m / b.wm) - (block_n / b.wn)), -b.wn, -b.wm);
                return ka < kb;
            });
        return {best->wm, best->wn};
    } else {
        auto best = std::min_element(pool.begin(), pool.end(),
            [&](const Candidate& a, const Candidate& b) {
                auto ka = std::make_tuple(std::abs(a.wm - 64), we_balance_rank(a.total),
                    std::abs((block_m / a.wm) - (block_n / a.wn)), -a.wn, -a.wm);
                auto kb = std::make_tuple(std::abs(b.wm - 64), we_balance_rank(b.total),
                    std::abs((block_m / b.wm) - (block_n / b.wn)), -b.wn, -b.wm);
                return ka < kb;
            });
        return {best->wm, best->wn};
    }
}

inline std::pair<int,int> estimate_warpOnN(int block_m, int warp_m, int warp_n, int acc_size, int input_size) {
    int warp_on_m = block_m / warp_m;

    // Constraint 1: warp budget
    int warp_cap = 32 / warp_on_m;

    // Constraint 2: SMEM capacity
    int smem_cap = (1024 - block_m) / warp_n;

    int hw_max = std::min(warp_cap, smem_cap);

    // Constraint 3: VREG pressure
    int acc_vreg = warp_m * warp_n * acc_size / (32 * 4);
    int input_vreg = (warp_m + warp_n) * 16 * input_size / (32 * 4);
    int basic_vreg = acc_vreg + input_vreg * 2;
    int extreme_vreg = acc_vreg + input_vreg;

    static constexpr int VREG_PER_WARP[3] = {128, 168, 256};
    static constexpr int TOTAL_WARPS[3]   = {32, 24, 16};

    int sf, ok;

    // sf: determined by basic_vreg
    if (basic_vreg <= VREG_PER_WARP[0])      sf = TOTAL_WARPS[0] / warp_on_m;
    else if (basic_vreg <= VREG_PER_WARP[1]) sf = TOTAL_WARPS[1] / warp_on_m;
    else if (basic_vreg <= VREG_PER_WARP[2]) sf = TOTAL_WARPS[2] / warp_on_m;
    else                                     sf = 0;

    // ok: determined by extreme_vreg
    if (extreme_vreg <= VREG_PER_WARP[0])      ok = TOTAL_WARPS[0] / warp_on_m;
    else if (extreme_vreg <= VREG_PER_WARP[1]) ok = TOTAL_WARPS[1] / warp_on_m;
    else if (extreme_vreg <= VREG_PER_WARP[2]) ok = TOTAL_WARPS[2] / warp_on_m;
    else                                       ok = 0;

    return {std::min(sf, hw_max), std::min(ok, hw_max)};
}

inline int get_max_warp_on_n(int block_m, int warp_m, int warp_n) {
    int sf_val, ok_val;
    bool found = lookup_register_warponn_cap(block_m, warp_m, warp_n, sf_val, ok_val);
    if (!found) {
        auto [sf, ok] = estimate_warpOnN(block_m, warp_m, warp_n);
        sf_val = sf;
        ok_val = ok;
    }
    return use_reg_model_only() ? sf_val : ok_val;
}

inline int compute_adaptive_blockn(int cutlass_n, int cu_num, int block_m, int cutlass_m) {
    int max_warp_on_n_val = max_warps_on_n_for(block_m);
    int max_bn_smem_val = max_bn_smem(block_m, 2);
    int max_bn_regs = max_bn_for_bm(block_m);
    int max_bn = std::min({max_bn_smem_val, BLOCK_N_MAX, max_bn_regs});
    int plan_max_bn = max_bn;

    int m_tiles = ceil_div(cutlass_m, block_m);
    int target_wave = ceil_div(m_tiles * cutlass_n, cu_num * plan_max_bn);
    double ideal_bn = static_cast<double>(m_tiles) * cutlass_n / (static_cast<double>(target_wave) * cu_num);
    int warp_n_val = std::max(16, static_cast<int>(std::ceil(ideal_bn / max_warp_on_n_val / 16.0)) * 16);
    int bn_baseline = std::min(plan_max_bn, static_cast<int>(std::ceil(ideal_bn / warp_n_val)) * warp_n_val);
    bn_baseline = snap_bn_to_valid(block_m, bn_baseline, max_bn);

    auto grid_healthy = [&](int bn) -> bool {
        auto [wm, wn] = pick_wm_wn(block_m, bn);
        if (wm == 0) return false;
        int total_warps = (block_m / wm) * (bn / wn);
        return (block_m / wm) <= 4 && wn >= 32 && total_warps >= 8;
    };

    auto realized_waves = [&](int bn) -> int {
        return ceil_div(m_tiles * ceil_div(cutlass_n, bn), cu_num);
    };

    int base_waves = realized_waves(bn_baseline);

    // Check if a larger BN gives fewer waves with a healthy grid
    std::vector<int> fewer_healthy;
    for (int bn = bn_baseline + 16; bn <= max_bn; bn += 16) {
        if (grid_healthy(bn) && realized_waves(bn) < base_waves) {
            fewer_healthy.push_back(bn);
        }
    }
    if (!fewer_healthy.empty()) {
        int min_w = INT_MAX;
        for (int bn : fewer_healthy) min_w = std::min(min_w, realized_waves(bn));
        for (int bn : fewer_healthy) {
            if (realized_waves(bn) == min_w) {
                bn_baseline = bn;
                break;
            }
        }
    }

    if (grid_healthy(bn_baseline)) return bn_baseline;

    // Search nearby for a healthy grid
    std::vector<int> cands;
    int lo = std::max(16, bn_baseline - 64);
    int hi = std::min(max_bn, bn_baseline + 64);
    for (int bn = lo; bn <= hi; bn += 16) {
        if (bn == bn_baseline || !grid_healthy(bn)) continue;
        cands.push_back(bn);
    }
    if (cands.empty()) return bn_baseline;

    auto key_fn = [&](int bn) {
        return std::make_tuple(std::abs(bn - bn_baseline),
                               we_balance_rank(warp_grid_total(block_m, bn)), -bn);
    };
    int snapped = *std::min_element(cands.begin(), cands.end(),
        [&](int a, int b) { return key_fn(a) < key_fn(b); });

    auto waves_fn = [&](int bn) -> int {
        return ceil_div(ceil_div(cutlass_m, block_m) * ceil_div(cutlass_n, bn), cu_num);
    };
    if (waves_fn(bn_baseline) == 1 && waves_fn(snapped) >= 2) {
        return bn_baseline;
    }
    return snapped;
}

inline std::optional<std::tuple<int,int,int>> compute_bound_tile(int block_m, int m, int n, int k, int num_sms) {
    if (block_m != 192 && block_m != 256 && block_m != 320 && block_m != 384) return std::nullopt;
    if (k < 2048 || n < 12288) return std::nullopt;

    int baseline_bn = compute_adaptive_blockn(n, num_sms, block_m, m);
    int m_tiles = ceil_div(m, block_m);
    int baseline_wave = ceil_div(m_tiles * ceil_div(n, baseline_bn), num_sms);
    if (baseline_wave == 1) return std::nullopt;

    int max_wave = baseline_wave;
    if (baseline_wave >= 3 && block_m >= 256) max_wave = baseline_wave + 1;

    struct CBCandidate { int wm, wn, bn, warp_on_m, warp_on_n, prop_wave; };
    std::vector<CBCandidate> candidates;

    // WM_CANDIDATES + [80]
    std::vector<int> wm_list(WM_CANDIDATES.begin(), WM_CANDIDATES.end());
    wm_list.push_back(80);

    for (int wm : wm_list) {
        if (wm > block_m || block_m % wm != 0) continue;
        int warp_on_m = block_m / wm;
        if (warp_on_m > 4 || 16 % warp_on_m != 0) continue;
        int warp_on_n = 16 / warp_on_m;
        for (int wn : {32, 48, 64}) {
            if (!tile_spill_ok(wm, wn, warp_on_m * warp_on_n)) continue;
            int bn = warp_on_n * wn;
            if (bn < 16 || bn > BLOCK_N_MAX || bn > max_bn_smem(block_m, 2)) continue;
            int prop_wave = ceil_div(m_tiles * ceil_div(n, bn), num_sms);
            bool above_formula = (wn > model_wn_cap(wm));
            if (above_formula && (prop_wave > baseline_wave || baseline_bn < bn - 32)) continue;
            if (!above_formula && prop_wave > max_wave) continue;
            candidates.push_back({wm, wn, bn, warp_on_m, warp_on_n, prop_wave});
        }
    }
    if (candidates.empty()) return std::nullopt;

    auto best = std::min_element(candidates.begin(), candidates.end(),
        [](const CBCandidate& a, const CBCandidate& b) {
            auto ka = std::make_tuple(a.prop_wave, std::abs(a.warp_on_m - a.warp_on_n), -a.bn, a.wm * a.wn);
            auto kb = std::make_tuple(b.prop_wave, std::abs(b.warp_on_m - b.warp_on_n), -b.bn, b.wm * b.wn);
            return ka < kb;
        });
    return std::make_tuple(best->bn, best->wm, best->wn);
}

inline std::pair<int,int> select_adaptive_smem(int block_m, int block_n) {
    double bound = static_cast<double>(SMEM_SIZE) / (static_cast<double>(block_m + block_n) * BASE_BLOCK_K * 2.0);
    int best_bk = 0, best_s = 0;
    double best_gap = 1e18;

    for (int s : STAGE_OPTIONS) {
        int bk = BASE_BLOCK_K;
        while (bk <= MAX_BLOCK_K) {
            double val = static_cast<double>(s) * (bk / BASE_BLOCK_K);
            if (val <= bound) {
                double gap = bound - val;
                bool update = false;
                if (best_bk == 0) {
                    update = true;
                } else if (gap < best_gap) {
                    update = true;
                } else if (gap == best_gap) {
                    update = (s > best_s) || (s == best_s && bk < best_bk);
                }
                if (update) {
                    best_bk = bk;
                    best_s = s;
                    best_gap = gap;
                }
            } else {
                break;
            }
            bk *= 2;
        }
    }
    return {best_bk, best_s};
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
    // m > 512 && n > 512: hard-coded large tile
    if (m > 512 && n > 512) {
        int block_m = 256, block_n = 256, block_k = 64;
        int warp_m = 64, warp_n = 64;
        int num_stages = 4;
        int warp_k = block_k;
        int num_tiles = ceil_div(m, block_m) * ceil_div(n, block_n);
        return {std::min(num_tiles, num_sms), block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages};
    }

    int block_m = select_blockm(m);
    int block_n, block_k, warp_m, warp_n, warp_k, num_stages;

    if (block_m <= 160) {
        // BM=144 -> 160 correction
        if (block_m == 144) block_m = 160;

        // Memory-bound path
        warp_m = get_warp_m(block_m);
        auto res = select_tile_memory_bound(n, m, k, num_sms, block_m, warp_m);
        block_n = res.block_n;
        warp_n = res.warp_n;
        block_k = res.block_k;
        num_stages = res.num_stages;
        warp_k = res.warp_k;
    } else {
        // Compute-bound override (gated): try 16-warp WE-ideal first
        auto ct = compute_tile_enabled() ? compute_bound_tile(block_m, m, n, k, num_sms) : std::nullopt;
        if (ct.has_value()) {
            auto [bn, cwm, cwn] = ct.value();
            block_n = bn;
            warp_m = cwm;
            warp_n = cwn;
        } else {
            block_n = compute_adaptive_blockn(n, num_sms, block_m, m);
            warp_m = get_warp_m(block_m, block_n);
            warp_n = get_warp_n(block_m, block_n);
        }
        auto [bk, s] = select_adaptive_smem(block_m, block_n);
        block_k = bk;
        num_stages = s;
        warp_k = get_warp_k(block_m, block_n, block_k, warp_m, warp_n, num_stages);
    }

    int num_tiles = ceil_div(m, block_m) * ceil_div(n, block_n);
    return {std::min(num_tiles, num_sms), block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages};
}

} // namespace deep_gemm_adaptive
