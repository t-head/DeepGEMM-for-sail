#pragma once
#include <algorithm>
#include <cstdint>
#include <hggc_ad.h>
#include <optional>
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"
#include "../../utils/utils.hpp"
#include "runtime.hpp"
#include "gemm_search_space.hpp"
#include "adaptive_tile_selector.hpp"
#include "runtime_lut.hpp"
#include <deep_gemm/common/types.cuh>

using namespace deep_gemm;
namespace deep_gemm_bf16_common {
std::tuple<int, int, int> get_smem_config(int num_stages, int k, int block_m, int block_n, int block_k = 128,
                                          int bpp = 2) {
    // Try swizzle first, as it does not waste shared memory
    int swizzle_mode = 128;
    // int block_n_padding = get_block_n_padding_for_smem_d(block_n) if swizzle_mode == 0 else 0
    int block_n_padding = 0;
    int smem_d = block_m * (block_n + block_n_padding);
    int smem_a_per_stage = block_m * block_k;
    // int smem_scales_a_per_stage = block_m * 4;
    int smem_b_per_stage = block_n * block_k;
    // int smem_scales_b = ceil_div(k, block_k) * 4;
    // int smem_barrier = num_stages * 8 * 2;
    // int smem_size = 0;
    int smem_size_d = smem_d * 2;
    int smem_size_a = num_stages * smem_a_per_stage * bpp;
    int smem_size_b = num_stages * smem_b_per_stage * bpp;
    int smem_size = std::max(smem_size_d, smem_size_a + smem_size_b);
    // smem_size += num_stages * smem_a_per_stage;
    // smem_size += num_stages * smem_scales_a_per_stage;
    // smem_size += num_stages * smem_b_per_stage;
    // smem_size += ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8;
    // smem_size += smem_barrier;

    // Swizzle and padding are not compatible
    // assert(int(swizzle_mode > 0) + int(block_n_padding > 0) <= 1);
    if ((swizzle_mode > 0) + (block_n_padding > 0) > 1) {
        throw std::runtime_error("Swizzle and padding are not compatible");
    }

    return std::make_tuple(smem_size, swizzle_mode, block_n_padding);
}

int get_smem_occ(int block_m, int block_n) {
    if (block_m <= 0) {
        return 0;
    }

    const int ppu_capacity = 262144;
    const int block_k = 64;
    const int bpp = 2;
    const int num_stages = 2;

    int smem_d = block_m * block_n;
    int smem_a_per_stage = block_m * block_k;
    int smem_b_per_stage = block_n * block_k;
    int smem_size_d = smem_d * 2;
    int smem_size_a = num_stages * smem_a_per_stage * bpp;
    int smem_size_b = num_stages * smem_b_per_stage * bpp;
    int smem_size = std::max(smem_size_d, smem_size_a + smem_size_b);

    return ppu_capacity / smem_size;
}

std::tuple<int, int, int, int, int, bool> get_gemv_best_configs(int m, int n, int k, int num_groups, int num_sms,
                                                                torch::ScalarType dtype) {
    int Alignment;
    if (dtype == torch::kInt8) { // Assuming 1 represents torch.int8, 0 represents other dtypes
        Alignment = 16;
    } else {
        Alignment = 8;
    }

    int small_k_algo_limit = 32 * Alignment;
    bool SmallK = false;
    int BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M, Stages;

    if (k <= small_k_algo_limit) {
        if (k <= 8 * Alignment) {
            BlockSize = 64;
            ThreadPerN = 8;
            NPerThread = 16;
            NUM_UNROLL = 1;
            SWZL_SIZE_M = 1;
        } else if (k <= 16 * Alignment) {
            BlockSize = 64;
            ThreadPerN = 16;
            NPerThread = 16;
            NUM_UNROLL = 1;
            SWZL_SIZE_M = 1;
        } else {
            BlockSize = 64;
            ThreadPerN = 32;
            NPerThread = 16;
            NUM_UNROLL = 1;
            SWZL_SIZE_M = 1;
        }
        SmallK = true;
        Stages = 5;
    } else {
        BlockSize = 256;
        if (k % (32 * 2 * Alignment) == 0) {
            ThreadPerN = 32;
            NUM_UNROLL = 2;
            if (m >= 16 * 8) {
                NPerThread = 4;
                SWZL_SIZE_M = 4;
            } else if (m >= 4 * 8) {
                NPerThread = 2;
                SWZL_SIZE_M = 2;
            } else {
                NPerThread = 1;
                SWZL_SIZE_M = 1;
            }
        } else if (dtype == torch::kInt8 && k % (32 * Alignment) == 0 && n % 32 == 0) {
            // int8 K aligns to 512 but not 1024 (K=1536/2560/3584...).
            ThreadPerN = 32;
            NUM_UNROLL = 1;
            SWZL_SIZE_M = 1;
            if (m >= 16 * 8) {
                NPerThread = 4;
            } else if (m >= 4 * 8) {
                NPerThread = 2;
            } else {
                NPerThread = 1;
            }
        } else if (k % (8 * Alignment) == 0) {
            ThreadPerN = 8;
            NUM_UNROLL = 1;
            SWZL_SIZE_M = 1;
            if (m >= 8 * 8) {
                NPerThread = 4;
            } else {
                NPerThread = 1;
            }
        } else {
            std::cout << "DeepGemm: gemv not support m:" << m << ", n:" << n << ", k:" << k << ", groups:" << num_groups
                      << ", num_sms:" << num_sms << "\n";
            ThreadPerN = -1;
            NUM_UNROLL = -1;
            SWZL_SIZE_M = -1;
            NPerThread = -1;
        }
    }

    return std::make_tuple(BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread, SmallK);
}

// For one or two short dot products, two BF16 elements per load distribute
// the reduction across 16 lanes and reduce serial multiply/add work per lane.
static bool dense_gemv_pair_load(const int rows, const int k, const bool is810e) {
    return is810e && rows <= 2 && k >= 32 && k <= 96 && k % 32 == 0;
}

// N=1 uses the same one-row GEMV runtime with transposed logical coordinates.
static void dense_gemv_select_configs_810e(int m, int k, bool is810e,
                                           int& bx, int& by, int& unroll, bool& pairLoad) {
    pairLoad = dense_gemv_pair_load(m, k, is810e);
    const bool shortReduction = k <= 256 && is810e;
    const bool manyRows = is810e && m >= 4096 && k <= 8192;
    // One warp avoids a block-wide reduction for medium K. Deeper reductions
    // retain more K lanes, while long streams favor smaller resident blocks.
    const int rowLanes = manyRows ? (k > 1024 && k <= 2048 ? 32 :
                                     k > 2048 && k <= 3072 ? 128 : 64) : 128;
    bx = pairLoad ? 16 : !shortReduction ? rowLanes : k <= 64 ? 8 : k <= 128 ? 16 : 32;
    by = shortReduction ? 32 / bx : 1;
    unroll = pairLoad ? k / 32 : shortReduction ? 1 : 2;
}

// Dense GEMV (m == 1 or 2) launch-config selection: BlockX keys on k/8, BlockY =
// 128/BlockX. Returns false when uncovered or the tile path is faster (fall back).
bool dense_gemv_select_configs(int m, int n, int k, const void* w, const void* x,
                               int& block_x, int& block_y, int& k_per_thread, int& npt, bool transposed_n1 = false) {
    if (get_env<int>("DG_DISABLE_DENSE_GEMV", 0))
        return false;
    // m == 2 tile gate: above this n the tile path wins despite ~8x wasted math
    // (A/B: +9% at n=1536, -13% at n=4608). Env-tunable; lower than m == 1's 3072.
    if (m == 2 && n > get_env<int>("DG_DENSE_GEMV_M2_MAX_N", 2048))
        return false;
    // Below this launch overhead dominates; tile pins BN=16 and loses on tiny n
    // (n=12 k=7168: 9.60 vs GEMV 3.80 us, 2.5x). Env-tunable for A/B.
    const bool is810e = std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos;
    // Scalar GEMV stores cover even sub-vector outputs. Sending short K / N<8
    // to the vector epilogue would write beyond the logical output row.
    if (n < get_env<int>("DG_DENSE_GEMV_MIN_N", is810e ? 1 : 8) || k < 8)
        return false;
    // bf16 + int4 (16B) vector loads: k must be a multiple of 8 elements and both
    // base pointers 16B aligned (rhs rows then stay aligned too, stride == k).
    constexpr int kElemAlign = 16 / 2;
    if (k % kElemAlign != 0)
        return false;
    if ((reinterpret_cast<uintptr_t>(w) & 0xF) != 0 || (reinterpret_cast<uintptr_t>(x) & 0xF) != 0)
        return false;

    // m == 2 rows-per-thread: npt=2 shares one (x0,x1) pair over two W rows, cutting
    // x traffic to m == 1 level. n>512 && k>=2048 -> npt=2 else 1; m == 1 untouched.
    constexpr int kM2NptGateN = 512;
    constexpr int kM2NptGateK = 2048;
    npt = (m == 2 && n > kM2NptGateN && k >= kM2NptGateK) ? 2 : 1;
    // DG_DENSE_GEMV_M2_NPT: force npt to 1 or 2 for A/B sweeps (ignored at m==1).
    if (m == 2) {
        const int npt_override = get_env<int>("DG_DENSE_GEMV_M2_NPT", 0);
        if (npt_override == 1 || npt_override == 2)
            npt = npt_override;
    }

    // A single warp is sufficient for short reductions. Extra warps add a
    // shared-memory reduction and do not supply useful K parallelism.
    if (is810e && n <= 128 && k <= 128) {
        block_x = k <= 64 ? 8 : 16;
        block_y = 32 / block_x;
        k_per_thread = 1;
        return true;
    }

    if (is810e && m == 1 && k <= 1024) {
        // Complete short reductions in one unrolled pass. A single warp can
        // hold two small-N rows; larger outputs retain four rows per CTA.
        if (n >= 64 && n <= 128 && k >= 256 && k % 128 == 0) {
            block_x = 16; block_y = 2; k_per_thread = k / 128;
            return true;
        }
        if (n >= 512 && k >= 512 && k % 256 == 0) {
            block_x = 32; block_y = 4; k_per_thread = k / 256;
            return true;
        }
    }

    // m == 1 tile gate: in the large-n x large-k corner SIMT FMA loses to tile
    // (~1.81 vs 1.87 TB/s). 4900-case scans put the crossover at n>=2944 && k>=3584.
    constexpr int kGemvTileGateN = 3072;
    constexpr int kGemvTileGateK = 3584;
    if (!is810e && !transposed_n1 && m == 1 && n >= kGemvTileGateN && k >= kGemvTileGateK)
        return false;

    k_per_thread = 2;
    const int add_times = k / kElemAlign;
    constexpr int block_size = 128;
    if (n <= 100 || (is810e && ((n >= kGemvTileGateN && k >= kGemvTileGateK) ||
                                (m == 1 && k >= 4096)))) {
        // One fat block per row, full-block K reduction (BlockX == 128
        // exercises the cross-warp SMEM reduce). Deep single-row products
        // also need this K parallelism when N is below the wide-output gate.
        block_x = 128;
    } else if (add_times <= 2) {
        block_x = 2;
    } else if (add_times <= 4) {
        block_x = 4;
    } else if (add_times <= 8) {
        block_x = 8;
    } else if (add_times <= 64) {
        block_x = 16;
    } else if (add_times <= 512) {
        block_x = 32;
    } else if (add_times <= 1024) {
        block_x = 64;
    } else {
        block_x = 128;
    }
    block_y = block_size / block_x;
    return true;
}

// Returns 8-element tuple: (num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config)
// NOTE: baseline 8-element contract. Adaptive warp_k/dense_s2_opt injection is done in the dense impl
// (bf16_gemm.hpp), so MoE grouped callers unpack this tuple directly without narrowing.
using ConfigResult = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

std::tuple<int, int, int, int, int, int, int, int, std::tuple<int, int, int>>
get_gemm_best_configs_v2(const std::vector<int>& shape, int dtype, int num_sms);

// Keep measured dense BF16 tiles in the dtype-specific selector. The shared
// adaptive selector also serves INT8, so these entries must not live there.
inline deep_gemm_adaptive::AdaptiveResult select_dense_bf16_tile(int m, int n, int k, int num_sms) {
    struct TileEntry {
        int m, n, k, bm, bn, bk, wm, wn, stages;
    };
    static constexpr std::array<TileEntry, 4> measured_tiles{{
        {32, 7168, 12288, 32, 192, 64, 16, 32, 6},
        {32, 7168, 8192,  32, 192, 64, 32, 16, 6},
        {64, 7168, 8192,  64, 192, 64, 32, 32, 7},
        {64, 12288, 1536, 64, 320, 64, 64, 32, 4},
    }};
    // The measurements use this SKU with its full CU budget. Reduced budgets
    // and other devices retain the general selector's resource-aware choices.
    if (num_sms == 39) {
        const auto prop = device_runtime->get_prop();
        if (prop->major == 8 && prop->minor == 9 && prop->multiProcessorCount == 39 &&
            std::strcmp(prop->name, "ZW-M890P") == 0) {
            for (const auto& t : measured_tiles) {
                if (m == t.m && n == t.n && k == t.k)
                    return {num_sms, t.bm, t.bn, t.bk, t.wm, t.wn, t.bk, t.stages};
            }
        }
    }
    return deep_gemm_adaptive::get_adaptive_configs(m, n, k, num_sms);
}

// Preserve the 8-tuple contract while returning the adaptive warp_k to dense
// callers, so they can use the same selection without running it a second time.
ConfigResult get_adaptive_configs_bf16(int m, int n, int k, int num_sms, int* selected_warp_k = nullptr) {
    auto [sms, bm, bn, bk, wm, wn, wk, stages] =
        select_dense_bf16_tile(m, n, k, num_sms);
    if (selected_warp_k) *selected_warp_k = wk;
    auto smem = get_smem_config(stages, k, bm, bn, bk);
    return std::make_tuple(sms, bm, bn, bk, wm, wn, stages, smem);
}

ConfigResult get_best_configs(int m, int n, int k, int num_groups, int num_sms,
                              GemmType gemm_type = GemmType::Normal, int max_block_n = 256,
                              bool* from_runtime_lut = nullptr, int* selected_warp_k = nullptr) {
    const bool is_k_grouped = gemm_type == GemmType::KGroupedContiguous;
    if (from_runtime_lut) *from_runtime_lut = false;
    if (selected_warp_k) *selected_warp_k = 0; // Non-adaptive callers use block_k.
    if (auto tile = deep_gemm_lut::get_runtime_config(
            m, n, k, num_groups, gemm_type, deep_gemm_lut::RuntimeLutKind::BF16)) {
        auto [bm, bn, bk, wm, wn, stages] = *tile;
        if (from_runtime_lut) *from_runtime_lut = true;
        return ConfigResult{num_sms, bm, bn, bk, wm, wn, stages,
                            get_smem_config(stages, k, bm, bn, bk, 2)};
    }
    if (gemm_type == GemmType::Normal && m >= 4096 && n >= 4096 && k >= 4096 &&
        m % 64 == 0 && n % 64 == 0 && k % 64 == 0) {
        const std::string device_name(device_runtime->get_prop()->name);
        if (device_name.find("ZW810") != std::string::npos) {
            auto [ns, bm, bn, bk, wm, wn, wk, stages, smem] =
                get_gemm_best_configs_v2({m, n, k}, 2, num_sms);
            (void)wk;
            return ConfigResult{ns, bm, bn, bk, wm, wn, stages, smem};
        }
    }
    // Keep the 8-tuple result and return adaptive warp_k through the optional output.
    // The dense implementation still derives dense_s2_opt when preparing the launch.
    if (gemm_type == GemmType::Normal && is_ppu1v5_device() &&
        deep_gemm_adaptive::bf16_adaptive_enabled(m, n, k)) {
        return get_adaptive_configs_bf16(m, n, k, num_sms, selected_warp_k);
    }
    // Generate block_ms
    std::vector<int> block_ms;
    if (is_k_grouped) {
        // TN AIU loads require at least 64 contiguous bytes (32 BF16 elements).
        block_ms = k >= 384 ? std::vector{256, 128, 64, 32} : std::vector{64, 32};
    } else if (gemm_type != GemmType::MGroupedContiguous) {
        if (k >= 384) {
            block_ms = {256, 128, 64, 32, 16};
        } else {
            block_ms = {64, 32, 16};
        }
    } else {
        block_ms = {heuristics_runtime->get_mk_alignment_for_contiguous_layout()};
    }

    // Assert max_block_n is power of 2
    assert(max_block_n > 0 && (max_block_n & (max_block_n - 1)) == 0);

    // Generate block_ns
    std::vector<int> block_ns;
    int bit_length = 0;
    int temp = max_block_n;
    while (temp > 0) {
        temp >>= 1;
        bit_length++;
    }

    // TODO check
    int start_bit = (k >= 384) ? (bit_length - 1) : (bit_length - 2);
    for (int i = start_bit; i >= 5; i--) {
        block_ns.push_back(1 << i);
    }

    // Lambda functions
    auto fix_wave_saturate = [num_sms](int x) -> int {
        return x == 0 ? num_sms : x;
    };

    auto get_num_waves = [m, n, num_groups, num_sms](int bm, int bn) -> int {
        if (bm == 0)
            return 0;
        return ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms);
    };

    auto get_last_wave_util = [m, n, num_groups, num_sms, &fix_wave_saturate](int bm, int bn) -> int {
        return fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms);
    };

    auto get_block_utils = [](int m, int bm) -> double {
        if (bm == 0)
            return 0.0;
        if (m % bm != 0) {
            return ((m / double(bm)) / ((m + bm - 1) / bm));
        } else {
            return 1.0;
        }
    };

    // Decide block sizes by waves
    int best_block_m = 0, best_block_n = 0;
    int min_n_threshold = gemm_type == GemmType::Normal ? 1 : 32;
    for (int block_m : block_ms) {
        // Filter block_ns
        std::vector<int> block_ns_after_filter;
        for (int block_n : block_ns) {
            bool condition;
            if (is_k_grouped) {
                // All candidates meet the TN alignment; FP32 output uses no TSM.
                condition = true;
            } else if (is_ppu1v5_device() && ((m >= 128 && k > 2048) || m >= 256)) {
                condition = (block_n != n && min_n_threshold) && !(block_m == 16 && block_n <= 32);
            } else {
                condition = (block_m <= 128 || block_n <= 128) && (block_n != n && min_n_threshold) &&
                            !(block_m == 16 && block_n <= 32);
            }

            if (condition) {
                block_ns_after_filter.push_back(block_n);
            }
        }

        for (int block_n : block_ns_after_filter) {
            bool success = false;
            int num_waves = get_num_waves(block_m, block_n);
            int best_num_waves = get_num_waves(best_block_m, best_block_n);
            double num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n);
            double best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n);
            int num_occ = get_smem_occ(block_m, block_n);
            int best_num_occ = get_smem_occ(best_block_m, best_block_n);

            if (best_block_m == 0 || best_block_n == 0) {
                success = true;
            } else if (m < 512 || n < 512) {
                // if single group block is small, balance wave, utils and occ
                double occ_wave = double(num_waves) / num_occ;
                double best_occ_wave = double(best_num_waves) / best_num_occ;
                double ai_util = double(block_m * block_n) / (block_m + block_n);
                double best_ai_util = double(best_block_m * best_block_n) / (best_block_m + best_block_n);
                bool valid_occ = (double(num_occ) / best_num_occ) >= 1;
                bool valid_wave = (occ_wave / best_occ_wave) <= 1;
                bool valid_util = (num_utils / best_num_utils) >= 1;
                bool valid_ai = (ai_util / best_ai_util) >= 1;

                int valid_count =
                    (valid_wave ? 1 : 0) + (valid_util ? 1 : 0) + (valid_occ ? 1 : 0) + (valid_ai ? 1 : 0);
                success = valid_count >= 3;
            } else if (num_waves < best_num_waves) {
                success = true;
            } else if (num_waves == best_num_waves) {
                // Check last wave utilization
                int util = get_last_wave_util(block_m, block_n);
                int best_util = get_last_wave_util(best_block_m, best_block_n);
                success = util > best_util;
                if (util == best_util) {
                    // Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= (block_m == best_block_m && block_n < best_block_n);
                    // Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= (block_n == best_block_n && block_m < best_block_m);
                    // Case 3: different for both `block_m` and `block_n`, `block_n` larger is better
                    success |= (block_m != best_block_m && block_n > best_block_n);
                }
            }
            // std::cout << "/n block_n is " << block_n << "success is " << success << std::endl;
            if (success) {
                best_block_m = block_m;
                best_block_n = block_n;
            }
        }
    }

    if (!is_ppu1v5_device() && (m > 64 && n == 128 && k == 4096)) {
        best_block_m = 128;
        best_block_n = 128;
    }

    // small m hbm bound or latency bound, wave is not useful, for better occ for 810e hbm bound, use smallest blockN
    // for m16
    if (not is_k_grouped && m < 20 && n < 512) {
        best_block_m = 16;
        best_block_n = 64;
    }

    assert(best_block_m != 0 && best_block_n != 0);

    // Always pick the longest one
    // NOTES: for double B scales, the best number of stages may be reduced
    int best_num_stages = 0;
    std::tuple<int, int, int> best_smem_config;
    int ppu_capacity = 262144;
    int block_k = 64;

    if (k <= 64) {
        block_k = 32;
    }

    if (k >= 4096 && (best_block_m <= 32 && best_block_n <= 64)) {
        block_k = 128;
    }

    // Generate stage candidates
    std::vector<int> stage_candidates;
    for (int s : {8, 7, 6, 5, 4, 3, 2}) {
        if (s <= k / block_k) {
            stage_candidates.push_back(s);
        }
    }

    if (stage_candidates.empty() || (128 % best_block_n != 0 && 128 / gcd(128, best_block_n) <= 4) ||
        best_block_m == 16 || best_block_m == 32) {
        stage_candidates = {3, 2};
    }

    if (best_block_m == 256 && best_block_n == 256) {
        stage_candidates = {4};
    }

    int best_occ = 0;
    for (int num_stages : stage_candidates) {
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k);
        if (std::get<0>(best_smem_config) <= ppu_capacity) {
            int occ = ppu_capacity / std::get<0>(best_smem_config);
            if (k < 512 || (best_block_m > 32 && best_block_n >= 64) && occ >= best_occ) {
                // compute block and too small-k use higher occ rather than large stage
                best_num_stages = num_stages;
                best_occ = occ;
            } else {
                best_num_stages = num_stages;
                break;
            }
        }
    }

    assert(std::get<0>(best_smem_config) != 0);
    assert(best_num_stages != 0);

    // Recompute the minimal number of SMs required
    // NOTES: less L2 cache usage and less GPU frequency drop
    int num_waves = get_num_waves(best_block_m, best_block_n);
    int num_min_sms = ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups;

    int warp_m = best_block_m / 2;
    int warp_n = best_block_n / 2;

    if (best_block_m == 256 && best_block_n == 256) {
        warp_m = best_block_m / 4;
        warp_n = best_block_n / 4;
    } else if (best_block_m == 32 && best_block_n >= 64) {
        warp_m = 32;
        warp_n = best_block_n / 4;
    } else if (best_block_n == 32 && n <= 128 && best_block_m >= 64) {
        warp_m = best_block_m / 4;
        warp_n = 32;
    } else if (best_block_m == 128 || (best_block_m == 256 && best_block_n >= 32)) {
        warp_m = 64;
        warp_n = best_block_n <= 128 ? best_block_n / 2 : best_block_n / 4;
    } else if (best_block_m == 16) {
        warp_m = 16;
        warp_n = (best_block_n <= 128) ? best_block_n / 4 : best_block_n / 8;
    } else if (best_block_n == 128 || best_block_n == 256) {
        warp_m = (best_block_m != 32) ? best_block_m / 2 : best_block_m;
        warp_n = best_block_n / 4;
    }

    return ConfigResult{std::min(num_min_sms, num_sms), best_block_m, best_block_n,
                        block_k, warp_m, warp_n, best_num_stages, best_smem_config};
}

// Pre-configured optimal tiling greater than or equal to 4096
const std::vector<std::vector<int>> CONFIG_TILE_GREATER_4096 = {{128, 128, 64, 64, 32, 64, 2},
                                                                {256, 128, 64, 64, 64, 64, 2},
                                                                {512, 128, 64, 64, 64, 64, 3},
                                                                {128, 256, 64, 32, 128, 64, 2}};
std::tuple<int, int, int, int, int, int, int, int, std::tuple<int, int, int>>
get_gemm_best_configs_v2(const std::vector<int>& shape, int dtype, int num_sms) {
    MatmulHeuristicsTile candidate_tile(shape, dtype, CONFIG_TILE_GREATER_4096);

    auto tile_list = candidate_tile.get_candidate_tile(1);

    assert(!tile_list.empty() && "tile_list must contain at least one candidate");
    const auto& tile_item = tile_list[0];
    assert(tile_item.size() >= 11 && "tile_item must have at least 11 elements");

    int bm = tile_item[3];
    int bn = tile_item[4];
    int bk = tile_item[5];
    int wm = tile_item[6];
    int wn = tile_item[7];
    int stages = tile_item[9];
    int num_min_sms = tile_item[10];

    assert(shape.size() == 3 && "shape must be [m, n, k]");
    int k = shape[2];
    std::tuple<int, int, int> best_smem_config = get_smem_config(stages, k, bm, bn, bk);

    return {std::min(num_min_sms, num_sms), bm, bn, bk, wm, wn, bk, stages, best_smem_config};
}
// Keep the 810E resource distinction local to BF16 dispatch. The public
// CU getter and other operators retain their existing budget semantics.
static int get_tensor_cu_count_810e() {
    static const int tensorCUs = [] {
        const auto prop = device_runtime->get_prop();
        const std::string name(prop->name);
        int count = prop->multiProcessorCount;
        if (name.find("ZW810E") != std::string::npos || name.find("ZW610E") != std::string::npos) {
            int ordinal;
            HGdevice device;
            DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&ordinal));
            DG_HOST_ASSERT(hgDeviceGet(&device, ordinal) == HGGC_SUCCESS);
            DG_HOST_ASSERT(hgDeviceGetAttributeAD(&count, HGAD_DEVICE_ATTRIBUTE_TENSOR_MULTIPROCESSOR_COUNT,
                                                  device) == HGGC_SUCCESS);
        }
        DG_HOST_ASSERT(count > 0 && count <= prop->multiProcessorCount);
        return count;
    }();
    return tensorCUs;
}

static int get_num_tensor_sms_810e() {
    return std::min(deep_gemm::get_num_sms(), get_tensor_cu_count_810e());
}

// 810E dispatch plans depend on geometry, K alignment and output precision.
// Tile fields are BM, BN, BK, WM, WN, stages, split-K, then scheduling flags.
enum class BF16GemmKernelType810E { Direct, TSM, Standard, Hybrid };

// Offline wave-cost model for the measured 810E Tensor topology. This is not
// a caller budget or dispatch gate; those use the queried Tensor CU count.
inline constexpr int kTensorCUs810E = 20;

struct BF16GemmConfig810E {
    int bm, bn, bk, wm, wn, stages, splits;
    // 0: N first; 1: M first; >=2: groups of this many M tiles.
    int tileOrder = 0;
    int peelFirstK = 0;
    int unrollK = 0;
    bool narrowNTail = false;
};

// Shared-memory residency bound for the measured 810E tile families. Register
// allocation can impose a tighter limit; this is not a general occupancy model.
static int64_t get_smem_waves_810e(int64_t m, int64_t n,
                                   const BF16GemmConfig810E& t) {
    const int64_t mainBytes = int64_t(t.bm + t.bn) * t.bk * 2 * t.stages;
    const int64_t epiBytes = int64_t(t.bm / t.wm) * 16 * t.bn * 4 + 128;
    const int64_t resident = std::max<int64_t>(1, (256LL << 10) / std::max(mainBytes, epiBytes));
    const int64_t blocks = ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn)) * t.splits;
    return ceil_div(blocks, resident * kTensorCUs810E);
}

struct ConfigResult810E {
    int64_t m, n, k;
    torch::ScalarType output;
    BF16GemmKernelType810E kind;
    BF16GemmConfig810E tile;
    // Hybrid divides only the M dimension; both halves cover the complete K.
    int splitM = 0;
    BF16GemmConfig810E tailTile{};
};

// The shape is retained for the standard launcher and hybrid M split;
// no shape-keyed table is searched.
static ConfigResult810E get_adaptive_configs_bf16_810e(
    int64_t m, int64_t n, int64_t k, torch::ScalarType output,
    bool perElementOutput = false, bool lowBudget = false) {
    ConfigResult810E plan{m, n, k, output, BF16GemmKernelType810E::TSM,
        {128, 128, k >= 128 && k % 64 == 0 ? 64 : 32, 64, 32, 2, 1}};
    auto& t = plan.tile;
    t.stages = std::min<int64_t>(3, k / t.bk);
    if (n < 64) {
        t.bm = n >= 8 ? 128 : 64;
        t.bn = n >= 24 ? 32 : 16;
        t.wm = t.bm == 128 ? 32 : 16;
        t.wn = t.bn == 16 ? 16 : 32;
        if (n % 8 != 0) plan.kind = BF16GemmKernelType810E::Direct;
    } else if (m <= 512 && n <= 2048 && k >= 768) {
        const bool shortRows = m <= 192 && n < 1024 && k < 4096;
        plan.kind = m >= 128 && !shortRows ? BF16GemmKernelType810E::TSM : BF16GemmKernelType810E::Direct;
        t.bm = m <= 16 ? 16 : m < 128 || shortRows ? 32 : 64;
        t.bn = (shortRows && m >= 128 && n >= 384 && n <= 512) ||
               (output == torch::kFloat && m >= 128 &&
                ((m <= 256 && n >= 1024) || (n >= 384 && m * n <= 131072))) ? 128 : 64;
        t.wm = t.bm <= 32 ? 16 : 32;
        t.wn = t.bm == 16 ? 16 : 32;
    } else if (m <= 32) {
        t.bm = 32; t.bn = 128; t.wm = 16; t.wn = 32;
        if (k >= 2048 && k % 128 == 0) t.bk = 128;
        t.stages = std::min<int64_t>(3, k / t.bk);
    } else if (m <= 256) {
        // At least seven eighths of a BM128 tile is useful here. More M rows
        // per CTA reduce rereads of a large B matrix compared with BM64/BN256.
        const bool largeB = n * k * 2 > (64LL << 20);
        t.bm = m >= 224 || (m > 64 && m <= 128 && largeB) ? 128 : 64;
        t.bn = n >= 12288 && t.bm == 64 && !(m <= 64 && k >= 4096) ? 256 : 128;
        t.wm = 32; t.wn = t.bn == 256 || t.bm == 128 ? 64 : 32;
        if (m > 64 && largeB && t.bn == 128) t.stages = 2;
    } else if (m <= 1024 && m % 128 != 0 &&
               (n >= 65536 || (n >= 16384 && k >= 768 && ceil_div(m, int64_t(128)) * 128 * 4 > m * 5))) {
        // Use a large tile for the complete M rows and a smaller tile for the
        // remainder; both launches retain the complete K reduction.
        plan.kind = BF16GemmKernelType810E::Hybrid;
        plan.splitM = int(m / 128 * 128);
        const int remainder = int(m - plan.splitM);
        const int tailM = remainder <= 16 ? 16 : remainder <= 32 ? 32 : remainder <= 64 ? 64 : 128;
        plan.tailTile = {tailM, 128, t.bk, tailM <= 32 ? 16 : 32, 32, t.stages, 1};
        if (m > 512 && tailM <= 32 && k >= 2048 && k % 128 == 0)
            plan.tailTile = {tailM, 128, 128, 16, 32, 3, 1};
        t.bm = 128; t.bn = 128; t.wm = 32; t.wn = 64; t.stages = 2;
    } else if (n <= 64) {
        t.bm = 256; t.bn = 64; t.wm = 64; t.wn = 32;
    } else if (n <= 512) {
        t.bm = 128; t.bn = n <= 288 && n != 128 ? 64 : 128;
        t.wm = 32; t.wn = t.bn == 64 ? 32 : 64;
        if (k >= 10240 && n % 64 == 0) {
            t.bm = 256; t.bn = 64; t.wm = 64; t.wn = 32; t.stages = 5;
        }
    } else {
        t.stages = 2;
        // Only transpose traversal when B exceeds the 64 MiB LLC. If the
        // complete A does not fit either, retain an 8 MiB A panel while walking N.
        constexpr int64_t cacheBytes = 64LL << 20;
        if (m >= 1024 && n * k * 2 > cacheBytes) {
            t.tileOrder = 1;
            if (m * k * 2 > cacheBytes) {
                const int64_t panelTiles = (8LL << 20) / (t.bm * k * 2);
                t.tileOrder = 2;
                while (t.tileOrder * 2 <= std::min<int64_t>(16, panelTiles))
                    t.tileOrder *= 2;
            }
            t.peelFirstK = 1;
        }
    }
    if (k <= 512) {
        t.stages = 2;
        if (m <= 64) { t.bn = 64; t.wn = 32; }
    }
    if (m > 64 && m <= 128 && n >= 4096 && k <= 128) {
        // A nearly full BM128 tile avoids reading B for two M tiles. With
        // shorter K, BM64/BN128 keeps enough CTAs and a smaller epilogue.
        t.bm = m * 8 >= 128 * 7 && k >= 128 ? 128 : 64;
        t.bn = 128; t.wm = 32; t.wn = t.bm == 128 ? 64 : 32;
    }
    // For medium reductions with one useful M tile, a BM128 tile reuses B
    // across both halves. Keep the larger pipeline for deep reductions.
    if (m > 64 && m <= 128 && n >= 4096 && k >= 768 && k <= 4096 &&
        n * k * 2 <= (64LL << 20)) {
        t = {128, 128, t.bk, 32, 64, 2, 1};
    }
    // With many N tiles and a large B working set, BM64/BN256 can schedule
    // adjacent M tiles together while keeping enough CTAs for all tensor CUs.
    if (plan.kind == BF16GemmKernelType810E::TSM && m > 64 && m <= 512 &&
        (m <= 192 || ceil_div(m, int64_t(128)) * 128 * 4 > m * 5) && n >= 4096 && k >= 768 &&
        n * k * 2 >= (64LL << 20) && (m > 128 || n * k * 2 <= (128LL << 20)) &&
        ceil_div(m, int64_t(64)) * ceil_div(n, int64_t(256)) >= 32) {
        t = {64, 256, t.bk, 32, 64, 3, 1};
    }
    // A short K ending in BK32 has enough work for a third stage on large
    // outputs; small grids below retain their two-stage latency configuration.
    if (m >= 1024 && m * n > 524288 && k <= 512 && k % 64 != 0)
        t.stages = 3;
    // Narrow outputs need more independent CTAs than one wave of tensor CUs.
    if (m >= 512 && n >= 8 && n <= 32 && ceil_div(m, int64_t(64)) < 128) {
        t.bm = 64; t.wm = 16;
    }
    // Persistent scheduling amortizes narrow-N tile setup on tall matrices.
    // Three stages hide HBM latency; the larger-M tile reuses each B panel.
    if (m >= 1024 && n > 64 && n <= 1024 &&
        k >= 1024 && k < 8192 && k % 64 == 0) {
        plan.kind = BF16GemmKernelType810E::Standard;
        t = {128, 128, 64, 32, 64, 3, 1};
        if (n <= 128 && k >= 4096 && ceil_div(m, int64_t(128)) >= 64 &&
            ceil_div(m, int64_t(128)) < 128)
            t.stages = 2;
        if (output == torch::kFloat && m >= 16384)
            t.stages = 2;
        if (output == torch::kBFloat16 && m >= 16384 && n >= 768)
            t = {64, 128, 64, 32, 32, 2, 1};
        // Halve BN when that removes an otherwise mostly empty final N tile.
        if (n > 128 && n % 128 > 0 && n % 128 <= 64)
            t = {128, 64, 64, 32, 32, 2, 1};
    }
    // Short reductions do not amortize a custom epilogue's setup. The standard
    // two-stage pipeline covers these wide outputs with fewer instructions.
    if (output == torch::kBFloat16 && m > 32 && m <= 256 &&
        n >= 4096 && k >= 256 && k <= 512 && k % 64 == 0) {
        plan.kind = BF16GemmKernelType810E::Standard;
        t = {64, m <= 64 ? 64 : 128, 64, 32, 32, 2, 1};
    }
    const int64_t tiles = ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn));
    const bool narrowGrid = m >= 512 && n >= 8 && n <= 32 && ceil_div(m, int64_t(64)) < 128;
    const bool underfilled = tiles < kTensorCUs810E || narrowGrid;
    const bool narrowOutput = m <= 512 && tiles < 128 && !(m >= 128 && n >= 1024);
    if ((underfilled || narrowOutput) && k >= 768 && n <= 2048 &&
        plan.kind != BF16GemmKernelType810E::Standard) {
        // More independent K partitions help under-filled grids. Bound the
        // workspace and retain enough complete BK tiles for every prologue.
        const int64_t targetTiles = m > 512 && !narrowGrid ? 64 : 128;
        const int limit = int(std::min<int64_t>({64, targetTiles / tiles, (k / t.bk) / t.stages}));
        while (t.splits * 2 <= limit) t.splits *= 2;
    }
    t.stages = std::min<int64_t>(t.stages, (k / t.bk) / t.splits);
    if (output == torch::kBFloat16 && m > 32 && m <= 64 &&
        n >= 64 && n <= 128 && k >= 2048 && k % 128 == 0) {
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {64, 32, 128, 16, 16, 3, 1};
        t.splits = int(std::min<int64_t>({64, 128 / ceil_div(n, int64_t(32)), k / 128 / 3}));
    }
    if (output == torch::kBFloat16 && m > 32 && m <= 64 &&
        n >= 256 && n <= 512 && k >= 8192 && k % 64 == 0) {
        plan.kind = BF16GemmKernelType810E::Direct;
        t = {64, 128, 64, 32, 32, 3, 1};
        const int limit = int(std::min<int64_t>(64 / ceil_div(n, int64_t(128)), k / 64 / 3));
        while (t.splits * 2 <= limit) t.splits *= 2;
    }
    if (output == torch::kBFloat16 && m >= 512 && m < 1024 &&
        n >= 128 && n <= 512 && k >= 4096 && k % 64 == 0) {
        // Two K partitions hide latency without a large partial workspace.
        plan.kind = BF16GemmKernelType810E::Direct;
        t = {128, 64, 64, 32, 32, 3, 1};
        const int64_t blocks = ceil_div(m, int64_t(128)) * ceil_div(n, int64_t(64));
        const int limit = int(std::min<int64_t>(64 / blocks, k / 64 / 3));
        while (t.splits * 2 <= limit) t.splits *= 2;
    }
    if (output == torch::kBFloat16 && m > 96 && m <= 128 &&
        n >= 1024 && n <= 2048 && k >= 2048 && k % 64 == 0) {
        // BM64 has the same padded M extent as BM32 here, with fewer loads.
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {64, 64, 64, 32, 32, 3, 1};
        // A fourth stage overlaps long-K loads while this narrow tile retains
        // enough resident CTAs for a nearly full pair of M tiles.
        if (k >= 4096 && k < 8192) t.stages = 4;
    }
    if (m > 64 && m <= 128 && n > 2048 && n <= 4096 &&
        k >= 4096 && k < 8192 && k % 64 == 0) {
        // Four stages cover long-K load latency for this medium tile grid.
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {64, 128, 64, 32, 32, 4, 1};
    }
    if (m > 32 && m <= 64 && n >= 1280 && n <= 2048 &&
        k >= 768 && k <= 2048 && k % 128 == 0) {
        // Twenty or more N tiles fill the tensor CUs without split-K. BK128
        // amortizes descriptor setup while retaining three complete stages.
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {64, 64, 128, 32, 32, 3, 1};
    }
    if (m > 128 && m <= 512 && n >= 2048 && n <= 4096 &&
        k >= 768 && k <= 2048 && k % 64 == 0 && plan.kind == BF16GemmKernelType810E::TSM) {
        // Smaller register tiles retain parallelism. Deeper reductions in
        // smaller grids need one more stage to hide load latency.
        t = {64, 128, 64, 32, 32, 2, 1};
        if (k >= 1536 && ceil_div(m, int64_t(64)) * ceil_div(n, int64_t(128)) < 64)
            t.stages = 3;
    }
    if (output == torch::kBFloat16 && m > 128 && m <= 256 &&
        n >= 2048 && n <= 8192 && k > 2048 && k < 4096 && k % 64 == 0) {
        t = {64, 256, 64, 32, 64, 3, 1};
    }
    if (m >= 1024 && n >= 1024 && k >= 256 && k <= 512 && k % 64 == 0 &&
        ceil_div(m, int64_t(128)) * ceil_div(n, int64_t(128)) <= 512) {
        t = {64, 128, 64, 32, 32, 2, 1};
    }
    // A narrow M and enough N tiles already fill the tensor CUs. A deeper K
    // tile then amortizes AIU descriptors without a split-K reduction launch.
    if (m <= 16 && n >= 1536 && n <= 2048 && k >= 2048 && k % 256 == 0) {
        t.bk = 256;
        t.stages = 3;
        t.splits = 1;
    }
    if (m <= 32 && k >= 2048 && (m > 16 || k >= 4096) && t.splits == 1 && t.bn >= 128 &&
        ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn)) < 64) {
        // Long reductions need enough independent CTAs to hide memory latency.
        t.bn = 64;
        t.wn = 32;
    }
    if (m <= 16 && n >= 4096 && k >= 2048 && k <= 4096 && k % 128 == 0) {
        // Two warps per CTA cover the useful rows of this bandwidth-bound
        // product without carrying a mostly empty BM32 accumulator.
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {16, 64, 128, 16, 32, 3, 1};
    }
    if (m <= 16 && n >= 2048 && n <= 8192 && k >= 8192 && k % 128 == 0) {
        plan.kind = BF16GemmKernelType810E::TSM;
        // Once N supplies four CTAs per tensor CU, a fourth pipeline stage
        // sustains the complete reduction without a partial buffer/reducer.
        if (ceil_div(n, int64_t(64)) >= 4 * kTensorCUs810E)
            t = {16, 64, 128, 16, 16, 4, 1};
        else {
            t = {16, 64, 128, 16, 16, 3, 2};
            t.peelFirstK = 1;
        }
    }
    if (output == torch::kBFloat16 && m > 32 && m <= 64 &&
        n >= 16384 && k >= 6144 && k % 64 == 0) {
        t = {64, 256, 64, 32, 64, 3, 1};
        t.peelFirstK = 1;
    }
    if (k <= 512 && n >= 64 && n <= 2048 && m * n <= 524288) {
        // Small outputs and short K do not amortize large register tiles.
        // Keep enough small CTAs while avoiding an extra reduction launch.
        plan.kind = BF16GemmKernelType810E::TSM;
        const int bm = k <= 192 || m <= 32 ? 32 : 64;
        t = {bm, 64, k >= 128 && k % 64 == 0 ? 64 : 32, bm / 2, 32, 2, 1};
    }
    // Larger K tiles amortize short-grid pipeline overhead. A single complete
    // K product uses AIU padding for a final BK tail; split-K remains aligned.
    const int alignment = output == torch::kFloat ? 4 : 8;
    if (m <= 64 && n >= 2048 && n <= 8192 && n % alignment == 0 && k >= 768 && k <= 2048 &&
        (k % 64 != 0 || (m <= 32 && k >= 1536))) {
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {m <= 16 ? 16 : m <= 32 ? 32 : 64, 64,
             128, m <= 32 ? 16 : 32, m <= 16 ? 16 : 32, 3, 1};
        if (k % 64 == 0) t.bn = 128;
    }
    if (m >= 128 && n >= 2048 && n <= 8192 && n % alignment == 0 &&
        k >= 768 && k <= 2048 && k % 64 != 0) {
        // Retain BK64 throughput for a final BK32 remainder. The complete-K
        // mainloop pads that remainder without shortening the reduction.
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {128, 128, 64, 32, 64, 2, 1};
        if (ceil_div(m, int64_t(128)) * 128 * 4 > m * 5)
            t = {64, 128, 64, 32, 32, 2, 1};
        if (ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn)) < kTensorCUs810E)
            t = {64, 128, 64, 32, 32, 3, 1};
    }
    if (m >= 1024 && n >= 64 && n <= 128 && n % alignment == 0 &&
        k >= 768 && k % 64 != 0) {
        // The complete-K mainloop can pad one BK32 remainder. BK64 halves
        // descriptor and pipeline overhead on these tall, narrow products.
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {128, 128, 64, 32, 64, 3, 1};
    }
    // A small A panel fits in LLC. Visit its M tiles consecutively so wide B
    // panels are reused before advancing N; keep narrow grids in N-first order.
    if (plan.kind == BF16GemmKernelType810E::TSM && m <= 512 && m > t.bm &&
        n >= 8192 && n * k * 2 >= (48LL << 20) &&
        ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn)) >= 64) {
        t.tileOrder = 1;
        t.peelFirstK = 1;
    }
    if (plan.kind == BF16GemmKernelType810E::TSM && m <= 512 && n >= 64 && n <= 2048 &&
        k <= 192 && k % t.bk == 0 && t.splits == 1) {
        t.peelFirstK = 1;
        t.unrollK = 1;
    }
    if (m >= 1024 && n >= 64 && n <= 128 && k >= 256 && k <= 512 && k % 64 == 0) {
        t = {64, 64, 64, 32, 32, 2, 1};
    }
    if (n < 8 && m * k * 2 >= (64LL << 20) && k >= 2048 && k % t.bk == 0) {
        // Tiny N cannot supply enough independent memory requests per CTA.
        // More K partitions expose the large A stream to additional CTAs.
        const int limit = int(std::min<int64_t>(8, k / t.bk / t.stages));
        while (t.splits * 2 <= limit) t.splits *= 2;
    }
    if (plan.kind == BF16GemmKernelType810E::TSM && m >= 1024 && n >= 4096 &&
        k >= 128 && k <= 512 && k % 64 == 0 && t.bm == 128 && t.bn == 128)
        t.peelFirstK = 1;
    if (plan.kind == BF16GemmKernelType810E::TSM && m >= 1024 && n >= 2048 &&
        k >= 768 && k <= 2048 && k % 64 != 0 &&
        t.bm == 128 && t.bn == 128 && t.bk == 64 && t.stages == 2) {
        t.peelFirstK = 1;
        // Walk M first to reuse each cached B panel across the many A tiles.
        if (n * k * 2 <= (64LL << 20)) t.tileOrder = 1;
        if (n <= 4096 && n % 128 > 0 && n % 128 <= 16 && n % alignment == 0) {
            // Keep full tiles wide while a 16-column CTA handles the narrow
            // remainder in the same launch. N-first traversal balances work.
            t.narrowNTail = true;
            t.tileOrder = 0;
        }
    }
    const bool shortWideRows = output == torch::kBFloat16 ?
        ceil_div(m, int64_t(128)) * 128 * 2 <= m * 3 :
        (n % 128 == 0 && (m <= 128 || ceil_div(m, int64_t(128)) * 128 * 7 <= m * 8));
    if (m > 64 && m <= 256 && n >= 8192 &&
        k >= 256 && k <= 512 && k % 64 == 0 &&
        shortWideRows) {
        // Reuse each B panel across more M rows when the larger tile is well
        // occupied. FP32 stores favor one M tile, or at least seven eighths
        // occupancy when two tiles are needed. Peel the first K iteration.
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {128, 128, 64, 64, 32, 2, 1};
        t.peelFirstK = 1;
    }
    if (plan.kind == BF16GemmKernelType810E::TSM && m > 32 && m <= 64 && n >= 8192 &&
        k >= 2048 && k < 6144 && n * k * 2 < (256LL << 20) &&
        t.bm == 64 && t.bn == 128 && t.bk == 64 && t.splits == 1) {
        // Moderate B streams favor the occupancy of two stages. Larger streams
        // keep the deeper pipeline needed to sustain their steady-state loads.
        t.stages = 2;
    }
    if (plan.kind == BF16GemmKernelType810E::TSM && m > 64 && m <= 128 && k >= 4096 &&
        n * k * 2 > (128LL << 20) &&
        t.bm == 128 && t.bn == 128 && t.bk == 64 && t.stages == 2 &&
        ceil_div(n, int64_t(t.bn)) <= 128 && t.splits == 1) {
        // A modest grid cannot hide repeated loads of a B stream larger than
        // twice LLC with occupancy alone; one more stage overlaps those loads.
        t.stages = 3;
        t.peelFirstK = 1;
    }
    if (plan.kind == BF16GemmKernelType810E::TSM && m > 32 && m <= 64 && n >= 16384 &&
        k >= 6144 && k % 64 == 0 && t.bm == 64 && t.bn == 128 && t.bk == 64 && t.splits == 1) {
        // A fourth stage hides long-K input latency when this wide grid still
        // supplies enough CTAs despite the larger shared-memory allocation.
        t.stages = 4;
        t.peelFirstK = 1;
    }
    if (plan.kind == BF16GemmKernelType810E::TSM && m >= 1024 && n > 1024 &&
        k >= 4096 && k % 64 == 0 && n * k * 2 <= (64LL << 20)) {
        // For a cached B matrix, a half-width N remainder favors narrower
        // CTAs; no CTA then spends half of its deep reduction on padding.
        if (n % 128 == 64)
            t = {128, 64, 64, 32, 32, 2, 1};
        // Exchange M for N coverage when the last BM128 tile is mostly empty.
        // Both operands fitting LLC lets the wider N tile reuse its A rows.
        else if (output == torch::kBFloat16 && m >= 2048 && n >= 2048 && n % 256 == 0 &&
                 m % 128 > 0 && m % 128 <= 64 && m * k * 2 <= (64LL << 20))
            t = {64, 256, 64, 32, 64, 3, 1};
    }
    if (output == torch::kBFloat16 && plan.kind == BF16GemmKernelType810E::TSM &&
        m > 32 && m <= 64 && n >= 3072 && n < 16384 && k >= 4096 && k % 64 == 0 &&
        n * k * 2 <= (128LL << 20) && t.bm == 64 && t.bn == 128 && t.bk == 64 && t.splits == 1) {
        // Four stages retain two CTAs per tensor CU, versus three with three
        // stages. Deepen only when this does not add a resident wave. Wider
        // grids can instead reuse A across BN256 and halve their CTA count.
        const int64_t nTiles = ceil_div(n, int64_t(128));
        if (ceil_div(n, int64_t(256)) >= 32 && n * k * 2 >= (64LL << 20)) {
            t = {64, 256, 64, 32, 64, 3, 1};
            t.peelFirstK = 1;
        } else if (ceil_div(nTiles, int64_t(2 * kTensorCUs810E)) <=
                   ceil_div(nTiles, int64_t(3 * kTensorCUs810E))) {
            t.stages = 4;
            t.peelFirstK = 1;
        }
    }
    if (output == torch::kBFloat16 && plan.kind == BF16GemmKernelType810E::TSM &&
        m > 96 && m <= 128 && n >= 3072 && n <= 8192 && k >= 4096 && k <= 8192 && k % 128 == 0 &&
        (ceil_div(n, int64_t(128)) <= 2 * kTensorCUs810E || t.bn == 256)) {
        // One BM128 tile avoids rereading B. BK128 amortizes descriptor work;
        // keep grids needing an extra resident wave on their existing tile.
        t = {128, 128, 128, 32, 64, 2, 1};
        t.peelFirstK = 1;
    }
    if (output == torch::kBFloat16 && plan.kind == BF16GemmKernelType810E::TSM &&
        m > 64 && m <= 512 && n >= 2560 && n <= 8192 && k >= 768 && k % 64 == 0 &&
        (n >= 3072 || k <= 8192) &&
        ceil_div(m, int64_t(96)) * 96 * 4 <= ceil_div(m, int64_t(t.bm)) * t.bm * 3) {
        // BM96 is worthwhile when it removes at least a quarter of padded M
        // work. Three stages overlap medium and deep reductions without split-K.
        t = {96, 128, 64, 48, 32, 3, 1};
        t.peelFirstK = 1;
    }
    if (m > 64 && m <= 96 && n >= 8192 && n % alignment == 0 && k >= 768 && k % 64 == 0) {
        // BM96 removes the padding of a second BM64 tile. WM48 retains a
        // power-of-two warp grid and lowers both registers and shared storage.
        // A cached B fits the two-stage pipeline; streams need a third stage.
        plan.kind = BF16GemmKernelType810E::TSM;
        plan.splitM = 0;
        t = {96, 128, 64, 48, 32, n * k * 2 <= (64LL << 20) ? 2 : 3, 1};
        t.peelFirstK = 1;
    }
    if (output == torch::kBFloat16 && m > 2 && m <= 64 && n > 2048 &&
        k >= 2048 && k % 128 == 0 && n * k * 2 <= (128LL << 20)) {
        const int64_t tiles64 = ceil_div(n, int64_t(64));
        const int64_t tiles128 = ceil_div(n, int64_t(128));
        if (tiles128 < kTensorCUs810E) {
            // Just beyond the split-K region, BN128 leaves tensor CUs idle.
            // BN64 fills them; BK128/four stages hide long-reduction latency
            // while the complete grid still fits one shared-memory wave.
            const int bm = m <= 16 ? 16 : m <= 32 ? 32 : 64;
            t = {bm, 64, 128, bm <= 32 ? 16 : 32, 32, 4, 1};
            t.peelFirstK = 1;
        } else if (m <= 32 && tiles64 > 3 * kTensorCUs810E &&
                   tiles64 <= 4 * kTensorCUs810E) {
            // BM32/BN64/BK128/S3 holds three CTAs per tensor CU. BM16
            // holds four; for more rows, BN128 holds two and halves the grid.
            // Either change removes a second wave in this interval. Outside
            // it, the narrower tile can provide better memory parallelism.
            t = {m <= 16 ? 16 : 32, m <= 16 ? 64 : 128, 128, 16, 32, 3, 1};
            t.peelFirstK = 1;
        }
    }
    if (output == torch::kBFloat16 && m > 2 && m <= 64 && n >= 8192 &&
        k >= 128 && k <= 512 && k % 64 == 0) {
        BF16GemmConfig810E narrow{m <= 16 ? 16 : m <= 32 ? 32 : 64,
                                     128, 64, m <= 32 ? 16 : 32, 32, 2, 1};
        BF16GemmConfig810E wide{64, 256, 64, 32, 64, 2, 1};
        const auto oldWaves = get_smem_waves_810e(m, n, t);
        const auto narrowWaves = get_smem_waves_810e(m, n, narrow);
        const auto wideWaves = get_smem_waves_810e(m, n, wide);
        // Small-M short reductions amortize loop/epilogue setup across BN128.
        // For more rows, widen only when it removes a shared-memory wave.
        const bool unroll = m <= 32 ? ceil_div(n, int64_t(64)) > 10 * kTensorCUs810E :
            narrowWaves < oldWaves && narrowWaves <= wideWaves;
        if (unroll || (m > 32 && wideWaves < oldWaves && wideWaves < narrowWaves)) {
            plan.kind = BF16GemmKernelType810E::TSM;
            t = unroll ? narrow : wide;
            t.peelFirstK = 1;
            t.unrollK = unroll;
        }
    }
    if (output == torch::kBFloat16 && plan.kind == BF16GemmKernelType810E::TSM &&
        m > 2 && m <= 64 && n >= 8192 && k >= 768 && k <= 2048 && k % 64 == 0 &&
        t.bk == 64 && t.stages == 3 && t.splits == 1) {
        BF16GemmConfig810E next{m <= 16 ? 16 : m <= 32 ? 32 : 64,
                                   128, 64, m <= 32 ? 16 : 32, 32, m <= 32 ? 4 : 3, 1};
        if (m > 32) {
            auto narrow = next;
            narrow.bn = 64;
            if (get_smem_waves_810e(m, n, narrow) <= get_smem_waves_810e(m, n, next))
                next = narrow;
        }
        // Deepen a small register tile, or replace an oversized N tile with
        // more independent CTAs, without adding a shared-memory wave.
        if (get_smem_waves_810e(m, n, next) <= get_smem_waves_810e(m, n, t)) {
            t = next;
            t.peelFirstK = 1;
        }
    }
    if (output == torch::kBFloat16 && plan.kind == BF16GemmKernelType810E::TSM &&
        m > 32 && m <= 64 && k > 2048 && k <= 8192 && k % 64 == 0 && t.splits == 1) {
        const int64_t tiles = ceil_div(n, int64_t(256));
        if (tiles >= 32 && tiles <= 8 * kTensorCUs810E) {
            // BN256 amortizes each A panel across more output columns. BK128
            // halves the K-loop count, but also halves shared-memory residency;
            // use it only when fewer K iterations outweigh the extra waves.
            const bool deepK = k % 128 == 0 &&
                ceil_div(tiles, int64_t(kTensorCUs810E)) <
                    2 * ceil_div(tiles, int64_t(2 * kTensorCUs810E));
            t = {64, 256, deepK ? 128 : 64, 32, 64, 3, 1};
            t.peelFirstK = 1;
        }
    }
    if (output == torch::kBFloat16 && plan.kind == BF16GemmKernelType810E::TSM &&
        m >= 192 && m <= 256 && n >= 2048 && n <= 4096 &&
        k >= 4096 && k <= 8192 && k % 64 == 0 && t.splits == 1) {
        BF16GemmConfig810E next{128, 128, 64, 32, 64, 3, 1};
        if (get_smem_waves_810e(m, n, next) < get_smem_waves_810e(m, n, t)) {
            t = next;
            t.peelFirstK = 1;
        }
    }
    if (output == torch::kBFloat16 && plan.kind == BF16GemmKernelType810E::TSM &&
        m > 256 && m <= 512 && n >= 2048 && n <= 4096 &&
        k >= 2048 && k <= 8192 && k % 64 == 0 && t.splits == 1 &&
        get_smem_waves_810e(m, n, t) > 1) {
        BF16GemmConfig810E best = t;
        int64_t bestPadding = ceil_div(m, int64_t(t.bm)) * t.bm;
        bool found = false;
        for (int bm : {64, 96}) {
            BF16GemmConfig810E next{bm, 128, 64, bm == 96 ? 48 : 32, 32, 2, 1};
            const int64_t blocks = ceil_div(m, int64_t(bm)) * ceil_div(n, int64_t(128));
            const int64_t padding = ceil_div(m, int64_t(bm)) * bm;
            // A full grid supplies the overlap lost by shortening the pipeline.
            // Keep it in one shared-memory wave without increasing padded rows.
            if (blocks >= 3 * kTensorCUs810E && get_smem_waves_810e(m, n, next) == 1 &&
                padding <= bestPadding && (!found || padding < bestPadding)) {
                best = next; bestPadding = padding; found = true;
            }
        }
        if (found) { t = best; t.peelFirstK = 1; }
    }
    if (output == torch::kBFloat16 && m > 96 && m <= 224 && n >= 2048 &&
        k >= 768 && k % 64 == 0 && n % 128 == 0 && t.splits == 1) {
        const int bm = m <= 112 ? 112 : m <= 128 ? 128 : m <= 160 ? 160 : m <= 192 ? 192 : 224;
        const int wm = bm == 112 ? 112 : bm == 128 ? 64 : bm / 2;
        const int64_t nTiles = ceil_div(n, int64_t(128));
        const bool compactWorthwhile = bm != 112 || t.bk == 128 || t.bn == 256 ||
            (nTiles <= 40 && k >= 3072) || n * k * 2 > (96LL << 20);
        const bool narrowReuse = bm != 160 || t.bm >= 128 || t.bn >= 256 ||
            nTiles <= kTensorCUs810E || nTiles >= 2 * kTensorCUs810E;
        if (compactWorthwhile && narrowReuse && bm != 128 && bm != t.bm && bm <= ceil_div(m, int64_t(t.bm)) * t.bm &&
            nTiles >= (bm == 112 ? 32 : 16) &&
            (bm != 224 || nTiles <= kTensorCUs810E || nTiles >= 32)) {
            const int bk = nTiles <= kTensorCUs810E && bm >= 160 && bm <= 192 && k % 128 == 0 ? 128 : 64;
            int stages = nTiles <= 2 * kTensorCUs810E ? 3 : 2;
            if (bm == 112 && n * k * 2 > (96LL << 20)) stages = 3;
            if (bm == 160 || bm == 192) {
                BF16GemmConfig810E two{bm, 128, bk, wm, 32, 2, 1};
                auto three = two; three.stages = 3;
                if (get_smem_waves_810e(m, n, three) <= get_smem_waves_810e(m, n, two)) stages = 3;
            }
            if (bm == 224) stages = nTiles < 24 || (k >= 2048 && nTiles > 40) ? 3 : 2;
            if ((n * k * 2 > (128LL << 20) || (nTiles > 64 && n * k * 2 > (64LL << 20))) && bm >= 160) stages = 3;
            BF16GemmConfig810E next{bm, 128, bk, wm, 32, stages, 1};
            // An equally padded, register-heavy tile needs to remove at least
            // two resident waves to offset its lower warp-level parallelism.
            const auto oldWaves = get_smem_waves_810e(m, n, t);
            const bool preservesSingleWave = oldWaves > 1 || bk > t.bk ||
                bm * 4 <= ceil_div(m, int64_t(t.bm)) * t.bm * 3;
            const bool lessPadding = bm < ceil_div(m, int64_t(t.bm)) * t.bm;
            const bool removesWaves = n * k * 2 <= (128LL << 20) &&
                get_smem_waves_810e(m, n, next) + 1 < oldWaves;
            if (preservesSingleWave && (lessPadding || removesWaves)) {
                plan.kind = BF16GemmKernelType810E::TSM;
                plan.splitM = 0;
                t = next;
                t.peelFirstK = 1;
            }
        }
    }
    if (output == torch::kBFloat16 && m > 2 && m <= 16 && n >= 4096 &&
        k >= 1024 && k <= 8192 && k % 128 == 0) {
        BF16GemmConfig810E next{16, 64, 128, 16, 16, 3, 1};
        auto deep = next; deep.stages = 4;
        if (get_smem_waves_810e(m, n, deep) <= get_smem_waves_810e(m, n, next)) next = deep;
        plan.kind = BF16GemmKernelType810E::TSM;
        t = next; t.peelFirstK = 1;
    }
    if (output == torch::kBFloat16 && m > 16 && m <= 32 && n >= 16384 &&
        k >= 768 && k <= 2048 && k % 128 == 0) {
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {32, 128, 128, 16, 32, 3, 1}; t.peelFirstK = 1;
    }
    if (output == torch::kBFloat16 && m > 16 && m <= 32 && n > 5120 && n <= 8192 &&
        k >= 2048 && k <= 4096 && k % 64 == 0) {
        plan.kind = BF16GemmKernelType810E::TSM;
        t = {32, 64, 64, 16, 16, 3, 1}; t.peelFirstK = 1;
    }
    // Use the established peeled K loop for long reductions on short M
    // grids. This avoids the unpeeled variant exercised by the rejected
    // persistent experiment, without changing K coverage or accumulation.
    if (plan.kind == BF16GemmKernelType810E::TSM && m > 64 && m <= 512 &&
        k >= 4096 && k % t.bk == 0 && t.splits == 1)
        t.peelFirstK = 1;
    if (perElementOutput && (plan.tile.bm & (plan.tile.bm - 1)) != 0) {
        // The scalar Direct fallback retains its supported power-of-two tiles.
        plan.tile.bm = plan.tile.bm < 128 ? 64 : 128;
        plan.tile.wm = 32;
    }
    if (lowBudget) {
        plan.kind = BF16GemmKernelType810E::Direct;
        plan.splitM = 0;
        // One persistent product avoids a separate reduction launch outside
        // the requested budget. Keep the scalar epilogue for partial vectors.
        plan.tile.splits = 1;
    }
    if (perElementOutput) {
        // Per-element stores also support unaligned output storage. Such rows
        // cannot use an EVT tile or its complete-K padding specialization.
        if (plan.kind == BF16GemmKernelType810E::Standard) plan.kind = BF16GemmKernelType810E::Direct;
        if (k % plan.tile.bk != 0) {
            plan.tile.bk = 32;
            plan.tile.stages = std::min<int64_t>(3, k / 32 / plan.tile.splits);
        }
    }
    return plan;
}

static ConfigResult get_best_configs_810e(const BF16GemmConfig810E& tile) {
    // The standard launcher recomputes shared storage from the first seven
    // fields. No cached smem estimate leaks
    // across output dtypes or implementations.
    return {get_num_tensor_sms_810e(), tile.bm, tile.bn, tile.bk, tile.wm, tile.wn, tile.stages, {0, 0, 0}};
}


// Keep explicit > runtime LUT > automatic precedence in the selector layer.
static std::optional<ConfigResult> get_runtime_configs_bf16_810e(
    int m, int n, int k, std::optional<ConfigResult> configs) {
    if (configs || !m || !n || !k) return configs;
    if (std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos) {
        if (auto tile = deep_gemm_lut::get_runtime_config(
                m, n, k, 1, GemmType::Normal, deep_gemm_lut::RuntimeLutKind::BF16)) {
            auto [bm, bn, bk, wm, wn, stages] = *tile;
            return ConfigResult{get_num_tensor_sms_810e(), bm, bn, bk, wm, wn, stages, {0, 0, 0}};
        }
    }
    return configs;
}

static ConfigResult get_best_configs_810e(int m, int n, int k, int numSms) {
    auto configs = get_best_configs(m, n, k, 1, numSms);
    // PPU 1.0 requires WN>=16 and complete prologue tiles for short K.
    std::get<5>(configs) = std::max(16, std::get<5>(configs));
    auto& bk = std::get<3>(configs);
    if (k / bk < 2) bk = 32;
    std::get<6>(configs) = std::min(std::get<6>(configs), k / bk);
    return configs;
}

// Validate user tiles before packing, accumulation or JIT. In particular, zero
// warp dimensions must not reach divisions in the lower-level launchers.
static void check_configs_810e(const std::optional<ConfigResult>& configs, int64_t k,
                               int outputBytes) {
    if (!configs || is_ppu1v5_device()) return;
    const auto [sms, bm, bn, bk, wm, wn, stages, unused] = *configs;
    const auto pow2 = [](int x) { return x > 0 && (x & (x - 1)) == 0; };
    TORCH_CHECK(sms > 0 && sms <= get_num_tensor_sms_810e(), "BF16 config CU count exceeds the configured budget");
    TORCH_CHECK(bm >= 16 && bm <= 512 && pow2(bm) && bn >= 16 && bn <= 512 && pow2(bn) &&
                bk >= 32 && bk <= 256 && pow2(bk) && wm >= 16 && pow2(wm) &&
                wn >= 16 && pow2(wn) && bm % wm == 0 && bn % wn == 0 &&
                stages >= 2 && stages <= 8, "invalid BF16 config tile or warp dimensions");
    TORCH_CHECK(k >= 64 && k % 32 == 0 && k / bk >= stages,
                "BF16 config requires complete prologue K tiles (K>=64, K%32=0)");
    const int64_t threads = int64_t(bm / wm) * (bn / wn) * 32;
    const int64_t smem = std::max(int64_t(bm + bn) * bk * 2 * stages, int64_t(bm) * bn * outputBytes);
    const auto prop = device_runtime->get_prop();
    TORCH_CHECK(threads <= prop->maxThreadsPerBlock && smem <= prop->sharedMemPerBlock,
                "BF16 config exceeds device thread or shared-memory limits");
}

} // namespace deep_gemm_bf16_common
