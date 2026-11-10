#pragma once
#include <array>
#include <cstdint>
#include <cstring>
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
    const bool is_810e = std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos;
    // Scalar GEMV stores cover even sub-vector outputs. Sending short K / N<8
    // to the vector epilogue would write beyond the logical output row.
    if (n < get_env<int>("DG_DENSE_GEMV_MIN_N", is_810e ? 1 : 8) || k < 8)
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
    if (is_810e && n <= 128 && k <= 128) {
        block_x = k <= 64 ? 8 : 16;
        block_y = 32 / block_x;
        k_per_thread = 1;
        return true;
    }

    if (is_810e && m == 1 && k <= 1024) {
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
    if (!is_810e && !transposed_n1 && m == 1 && n >= kGemvTileGateN && k >= kGemvTileGateK)
        return false;

    k_per_thread = 2;
    const int add_times = k / kElemAlign;
    constexpr int block_size = 128;
    if (n <= 100 || (is_810e && ((n >= kGemvTileGateN && k >= kGemvTileGateK) ||
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
// 810E dispatch plans depend on geometry, K alignment and output precision.
// Tile fields are BM, BN, BK, WM, WN, stages, split-K, then scheduling flags.
enum class Model810eDispatchKind { Direct, TSM, Standard, Hybrid };
enum class Model810eOutputType { BF16, FP32 };

// ZW810E exposes 64 general CUs, of which 20 support tensor instructions.
// This matches the default tensor budget in DeviceRuntime::get_num_sms().
inline constexpr int kModel810eTensorCUs = 20;

struct Model810eDispatchTile {
    int bm, bn, bk, wm, wn, stages, splits;
    // 0: N first; 1: M first; >=2: groups of this many M tiles.
    int tile_order = 0;
    int peel_first_k = 0;
    int specialize_shape = 0;
    int unroll_k = 0;
    bool narrow_n_tail = false;
};

// Shared-memory residency bound for the measured 810E tile families. Register
// allocation can impose a tighter limit; this is not a general occupancy model.
static int64_t model_810e_smem_waves(int64_t batch, int64_t m, int64_t n,
                                   const Model810eDispatchTile& t) {
    const int64_t main_bytes = int64_t(t.bm + t.bn) * t.bk * 2 * t.stages;
    const int64_t epi_bytes = int64_t(t.bm / t.wm) * 16 * t.bn * 4 + 128;
    const int64_t resident = std::max<int64_t>(1, (256LL << 10) / std::max(main_bytes, epi_bytes));
    const int64_t blocks = batch * ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn)) * t.splits;
    return ceil_div(blocks, resident * kModel810eTensorCUs);
}

// Compact M tiles retain power-of-two warp grids while reducing padded rows.
// Batches must still supply two tensor-CU waves; larger tiles break that bound
// in small batches, where a pair of smaller M tiles can be preferable.
static Model810eDispatchTile model_810e_compact_batch_tile(
    int64_t batch, int64_t m, int64_t n, int64_t k, Model810eDispatchTile t) {
    if (m <= 64 || m > 224 || n < 128 || n > 1024 || n % 128 != 0 ||
        k < 128 || k > 512 || k % 64 != 0 || t.bn != 128) return t;
    int64_t padding = ceil_div(m, int64_t(t.bm)) * t.bm;
    int best = t.bm;
    for (int bm : {80, 96, 112, 128, 160, 192, 224}) {
        const int64_t padded = ceil_div(m, int64_t(bm)) * bm;
        const int64_t blocks = batch * ceil_div(m, int64_t(bm)) * (n / 128);
        if (blocks >= 2 * kModel810eTensorCUs &&
            (padded < padding || (padded == padding && bm > best))) {
            padding = padded;
            best = bm;
        }
    }
    if (best != t.bm) {
        const int wm = best == 80 || best == 112 ? best : best / 2;
        t = {best, 128, 64, wm, 32, 2, 1};
    }
    return t;
}

struct Model810eDenseDispatchEntry {
    int64_t m, n, k;
    Model810eOutputType output;
    Model810eDispatchKind kind;
    Model810eDispatchTile tile;
    // Hybrid divides only the M dimension; both halves cover the complete K.
    int split_m = 0;
    Model810eDispatchTile tail_tile{};
};

// The shape is retained for the standard launcher and hybrid M split;
// no shape-keyed table is searched.
static Model810eDenseDispatchEntry get_model_810e_dense_dispatch(
    int64_t m, int64_t n, int64_t k, Model810eOutputType output) {
    Model810eDenseDispatchEntry plan{m, n, k, output, Model810eDispatchKind::TSM,
        {128, 128, k >= 128 && k % 64 == 0 ? 64 : 32, 64, 32, 2, 1}};
    auto& t = plan.tile;
    t.stages = std::min<int64_t>(3, k / t.bk);
    if (n < 64) {
        t.bm = n >= 8 ? 128 : 64;
        t.bn = n >= 24 ? 32 : 16;
        t.wm = t.bm == 128 ? 32 : 16;
        t.wn = t.bn == 16 ? 16 : 32;
        if (n % 8 != 0) plan.kind = Model810eDispatchKind::Direct;
    } else if (m <= 512 && n <= 2048 && k >= 768) {
        const bool short_rows = m <= 192 && n < 1024 && k < 4096;
        plan.kind = m >= 128 && !short_rows ? Model810eDispatchKind::TSM : Model810eDispatchKind::Direct;
        t.bm = m <= 16 ? 16 : m < 128 || short_rows ? 32 : 64;
        t.bn = (short_rows && m >= 128 && n >= 384 && n <= 512) ||
               (output == Model810eOutputType::FP32 && m >= 128 &&
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
        const bool large_b = n * k * 2 > (64LL << 20);
        t.bm = m >= 224 || (m > 64 && m <= 128 && large_b) ? 128 : 64;
        t.bn = n >= 12288 && t.bm == 64 && !(m <= 64 && k >= 4096) ? 256 : 128;
        t.wm = 32; t.wn = t.bn == 256 || t.bm == 128 ? 64 : 32;
        if (m > 64 && large_b && t.bn == 128) t.stages = 2;
    } else if (m <= 1024 && m % 128 != 0 &&
               (n >= 65536 || (n >= 16384 && k >= 768 && ceil_div(m, int64_t(128)) * 128 * 4 > m * 5))) {
        // Use a large tile for the complete M rows and a smaller tile for the
        // remainder; both launches retain the complete K reduction.
        plan.kind = Model810eDispatchKind::Hybrid;
        plan.split_m = int(m / 128 * 128);
        const int remainder = int(m - plan.split_m);
        const int tail_m = remainder <= 16 ? 16 : remainder <= 32 ? 32 : remainder <= 64 ? 64 : 128;
        plan.tail_tile = {tail_m, 128, t.bk, tail_m <= 32 ? 16 : 32, 32, t.stages, 1};
        if (m > 512 && tail_m <= 32 && k >= 2048 && k % 128 == 0)
            plan.tail_tile = {tail_m, 128, 128, 16, 32, 3, 1};
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
        constexpr int64_t cache_bytes = 64LL << 20;
        if (m >= 1024 && n * k * 2 > cache_bytes) {
            t.tile_order = 1;
            if (m * k * 2 > cache_bytes) {
                const int64_t panel_tiles = (8LL << 20) / (t.bm * k * 2);
                t.tile_order = 2;
                while (t.tile_order * 2 <= std::min<int64_t>(16, panel_tiles))
                    t.tile_order *= 2;
            }
            t.peel_first_k = 1;
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
    if (plan.kind == Model810eDispatchKind::TSM && m > 64 && m <= 512 &&
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
        plan.kind = Model810eDispatchKind::Standard;
        t = {128, 128, 64, 32, 64, 3, 1};
        if (n <= 128 && k >= 4096 && ceil_div(m, int64_t(128)) >= 64 &&
            ceil_div(m, int64_t(128)) < 128)
            t.stages = 2;
        if (output == Model810eOutputType::FP32 && m >= 16384)
            t.stages = 2;
        if (output == Model810eOutputType::BF16 && m >= 16384 && n >= 768)
            t = {64, 128, 64, 32, 32, 2, 1};
        // Halve BN when that removes an otherwise mostly empty final N tile.
        if (n > 128 && n % 128 > 0 && n % 128 <= 64)
            t = {128, 64, 64, 32, 32, 2, 1};
    }
    // Short reductions do not amortize a custom epilogue's setup. The standard
    // two-stage pipeline covers these wide outputs with fewer instructions.
    if (output == Model810eOutputType::BF16 && m > 32 && m <= 256 &&
        n >= 4096 && k >= 256 && k <= 512 && k % 64 == 0) {
        plan.kind = Model810eDispatchKind::Standard;
        t = {64, m <= 64 ? 64 : 128, 64, 32, 32, 2, 1};
    }
    const int64_t tiles = ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn));
    const bool narrow_grid = m >= 512 && n >= 8 && n <= 32 && ceil_div(m, int64_t(64)) < 128;
    const bool underfilled = tiles < kModel810eTensorCUs || narrow_grid;
    const bool narrow_output = m <= 512 && tiles < 128 && !(m >= 128 && n >= 1024);
    if ((underfilled || narrow_output) && k >= 768 && n <= 2048 &&
        plan.kind != Model810eDispatchKind::Standard) {
        // More independent K partitions help under-filled grids. Bound the
        // workspace and retain enough complete BK tiles for every prologue.
        const int64_t target_tiles = m > 512 && !narrow_grid ? 64 : 128;
        const int limit = int(std::min<int64_t>({64, target_tiles / tiles, (k / t.bk) / t.stages}));
        while (t.splits * 2 <= limit) t.splits *= 2;
    }
    t.stages = std::min<int64_t>(t.stages, (k / t.bk) / t.splits);
    if (output == Model810eOutputType::BF16 && m > 32 && m <= 64 &&
        n >= 64 && n <= 128 && k >= 2048 && k % 128 == 0) {
        plan.kind = Model810eDispatchKind::TSM;
        t = {64, 32, 128, 16, 16, 3, 1};
        t.splits = int(std::min<int64_t>({64, 128 / ceil_div(n, int64_t(32)), k / 128 / 3}));
    }
    if (output == Model810eOutputType::BF16 && m > 32 && m <= 64 &&
        n >= 256 && n <= 512 && k >= 8192 && k % 64 == 0) {
        plan.kind = Model810eDispatchKind::Direct;
        t = {64, 128, 64, 32, 32, 3, 1};
        const int limit = int(std::min<int64_t>(64 / ceil_div(n, int64_t(128)), k / 64 / 3));
        while (t.splits * 2 <= limit) t.splits *= 2;
    }
    if (output == Model810eOutputType::BF16 && m >= 512 && m < 1024 &&
        n >= 128 && n <= 512 && k >= 4096 && k % 64 == 0) {
        // Two K partitions hide latency without a large partial workspace.
        plan.kind = Model810eDispatchKind::Direct;
        t = {128, 64, 64, 32, 32, 3, 1};
        const int64_t blocks = ceil_div(m, int64_t(128)) * ceil_div(n, int64_t(64));
        const int limit = int(std::min<int64_t>(64 / blocks, k / 64 / 3));
        while (t.splits * 2 <= limit) t.splits *= 2;
    }
    if (output == Model810eOutputType::BF16 && m > 96 && m <= 128 &&
        n >= 1024 && n <= 2048 && k >= 2048 && k % 64 == 0) {
        // BM64 has the same padded M extent as BM32 here, with fewer loads.
        plan.kind = Model810eDispatchKind::TSM;
        t = {64, 64, 64, 32, 32, 3, 1};
        // A fourth stage overlaps long-K loads while this narrow tile retains
        // enough resident CTAs for a nearly full pair of M tiles.
        if (k >= 4096 && k < 8192) t.stages = 4;
    }
    if (m > 64 && m <= 128 && n > 2048 && n <= 4096 &&
        k >= 4096 && k < 8192 && k % 64 == 0) {
        // Four stages cover long-K load latency for this medium tile grid.
        plan.kind = Model810eDispatchKind::TSM;
        t = {64, 128, 64, 32, 32, 4, 1};
    }
    if (m > 32 && m <= 64 && n >= 1280 && n <= 2048 &&
        k >= 768 && k <= 2048 && k % 128 == 0) {
        // Twenty or more N tiles fill the tensor CUs without split-K. BK128
        // amortizes descriptor setup while retaining three complete stages.
        plan.kind = Model810eDispatchKind::TSM;
        t = {64, 64, 128, 32, 32, 3, 1};
    }
    if (m > 128 && m <= 512 && n >= 2048 && n <= 4096 &&
        k >= 768 && k <= 2048 && k % 64 == 0 && plan.kind == Model810eDispatchKind::TSM) {
        // Smaller register tiles retain parallelism. Deeper reductions in
        // smaller grids need one more stage to hide load latency.
        t = {64, 128, 64, 32, 32, 2, 1};
        if (k >= 1536 && ceil_div(m, int64_t(64)) * ceil_div(n, int64_t(128)) < 64)
            t.stages = 3;
    }
    if (output == Model810eOutputType::BF16 && m > 128 && m <= 256 &&
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
        plan.kind = Model810eDispatchKind::TSM;
        t = {16, 64, 128, 16, 32, 3, 1};
    }
    if (m <= 16 && n >= 2048 && n <= 8192 && k >= 8192 && k % 128 == 0) {
        plan.kind = Model810eDispatchKind::TSM;
        // Once N supplies four CTAs per tensor CU, a fourth pipeline stage
        // sustains the complete reduction without a partial buffer/reducer.
        if (ceil_div(n, int64_t(64)) >= 4 * kModel810eTensorCUs)
            t = {16, 64, 128, 16, 16, 4, 1};
        else {
            t = {16, 64, 128, 16, 16, 3, 2};
            t.peel_first_k = 1;
        }
    }
    if (output == Model810eOutputType::BF16 && m > 32 && m <= 64 &&
        n >= 16384 && k >= 6144 && k % 64 == 0) {
        t = {64, 256, 64, 32, 64, 3, 1};
        t.peel_first_k = 1;
    }
    if (k <= 512 && n >= 64 && n <= 2048 && m * n <= 524288) {
        // Small outputs and short K do not amortize large register tiles.
        // Keep enough small CTAs while avoiding an extra reduction launch.
        plan.kind = Model810eDispatchKind::TSM;
        const int bm = k <= 192 || m <= 32 ? 32 : 64;
        t = {bm, 64, k >= 128 && k % 64 == 0 ? 64 : 32, bm / 2, 32, 2, 1};
    }
    // Larger K tiles amortize short-grid pipeline overhead. A single complete
    // K product uses AIU padding for a final BK tail; split-K remains aligned.
    const int alignment = output == Model810eOutputType::FP32 ? 4 : 8;
    if (m <= 64 && n >= 2048 && n <= 8192 && n % alignment == 0 && k >= 768 && k <= 2048 &&
        (k % 64 != 0 || (m <= 32 && k >= 1536))) {
        plan.kind = Model810eDispatchKind::TSM;
        t = {m <= 16 ? 16 : m <= 32 ? 32 : 64, 64,
             128, m <= 32 ? 16 : 32, m <= 16 ? 16 : 32, 3, 1};
        if (k % 64 == 0) t.bn = 128;
    }
    if (m >= 128 && n >= 2048 && n <= 8192 && n % alignment == 0 &&
        k >= 768 && k <= 2048 && k % 64 != 0) {
        // Retain BK64 throughput for a final BK32 remainder. The complete-K
        // mainloop pads that remainder without shortening the reduction.
        plan.kind = Model810eDispatchKind::TSM;
        t = {128, 128, 64, 32, 64, 2, 1};
        if (ceil_div(m, int64_t(128)) * 128 * 4 > m * 5)
            t = {64, 128, 64, 32, 32, 2, 1};
        if (ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn)) < kModel810eTensorCUs)
            t = {64, 128, 64, 32, 32, 3, 1};
    }
    if (m >= 1024 && n >= 64 && n <= 128 && n % alignment == 0 &&
        k >= 768 && k % 64 != 0) {
        // The complete-K mainloop can pad one BK32 remainder. BK64 halves
        // descriptor and pipeline overhead on these tall, narrow products.
        plan.kind = Model810eDispatchKind::TSM;
        t = {128, 128, 64, 32, 64, 3, 1};
    }
    // A small A panel fits in LLC. Visit its M tiles consecutively so wide B
    // panels are reused before advancing N; keep narrow grids in N-first order.
    if (plan.kind == Model810eDispatchKind::TSM && m <= 512 && m > t.bm &&
        n >= 8192 && n * k * 2 >= (48LL << 20) &&
        ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn)) >= 64) {
        t.tile_order = 1;
        t.peel_first_k = 1;
    }
    if (plan.kind == Model810eDispatchKind::TSM && m <= 512 && n >= 64 && n <= 2048 &&
        k <= 192 && k % t.bk == 0 && t.splits == 1) {
        t.peel_first_k = 1;
        t.unroll_k = 1;
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
    if (plan.kind == Model810eDispatchKind::TSM && m >= 1024 && n >= 4096 &&
        k >= 128 && k <= 512 && k % 64 == 0 && t.bm == 128 && t.bn == 128)
        t.peel_first_k = 1;
    if (plan.kind == Model810eDispatchKind::TSM && m >= 1024 && n >= 2048 &&
        k >= 768 && k <= 2048 && k % 64 != 0 &&
        t.bm == 128 && t.bn == 128 && t.bk == 64 && t.stages == 2) {
        t.peel_first_k = 1;
        // Walk M first to reuse each cached B panel across the many A tiles.
        if (n * k * 2 <= (64LL << 20)) t.tile_order = 1;
        if (n <= 4096 && n % 128 > 0 && n % 128 <= 16 && n % alignment == 0) {
            // Keep full tiles wide while a 16-column CTA handles the narrow
            // remainder in the same launch. N-first traversal balances work.
            t.narrow_n_tail = true;
            t.tile_order = 0;
        }
    }
    const bool short_wide_rows = output == Model810eOutputType::BF16 ?
        ceil_div(m, int64_t(128)) * 128 * 2 <= m * 3 :
        (n % 128 == 0 && (m <= 128 || ceil_div(m, int64_t(128)) * 128 * 7 <= m * 8));
    if (m > 64 && m <= 256 && n >= 8192 &&
        k >= 256 && k <= 512 && k % 64 == 0 &&
        short_wide_rows) {
        // Reuse each B panel across more M rows when the larger tile is well
        // occupied. FP32 stores favor one M tile, or at least seven eighths
        // occupancy when two tiles are needed. Peel the first K iteration.
        plan.kind = Model810eDispatchKind::TSM;
        t = {128, 128, 64, 64, 32, 2, 1};
        t.peel_first_k = 1;
    }
    if (plan.kind == Model810eDispatchKind::TSM && m > 32 && m <= 64 && n >= 8192 &&
        k >= 2048 && k < 6144 && n * k * 2 < (256LL << 20) &&
        t.bm == 64 && t.bn == 128 && t.bk == 64 && t.splits == 1) {
        // Moderate B streams favor the occupancy of two stages. Larger streams
        // keep the deeper pipeline needed to sustain their steady-state loads.
        t.stages = 2;
    }
    if (plan.kind == Model810eDispatchKind::TSM && m > 64 && m <= 128 && k >= 4096 &&
        n * k * 2 > (128LL << 20) &&
        t.bm == 128 && t.bn == 128 && t.bk == 64 && t.stages == 2 &&
        ceil_div(n, int64_t(t.bn)) <= 128 && t.splits == 1) {
        // A modest grid cannot hide repeated loads of a B stream larger than
        // twice LLC with occupancy alone; one more stage overlaps those loads.
        t.stages = 3;
        t.peel_first_k = 1;
    }
    if (plan.kind == Model810eDispatchKind::TSM && m > 32 && m <= 64 && n >= 16384 &&
        k >= 6144 && k % 64 == 0 && t.bm == 64 && t.bn == 128 && t.bk == 64 && t.splits == 1) {
        // A fourth stage hides long-K input latency when this wide grid still
        // supplies enough CTAs despite the larger shared-memory allocation.
        t.stages = 4;
        t.peel_first_k = 1;
    }
    if (plan.kind == Model810eDispatchKind::TSM && m >= 1024 && n > 1024 &&
        k >= 4096 && k % 64 == 0 && n * k * 2 <= (64LL << 20)) {
        // For a cached B matrix, a half-width N remainder favors narrower
        // CTAs; no CTA then spends half of its deep reduction on padding.
        if (n % 128 == 64)
            t = {128, 64, 64, 32, 32, 2, 1};
        // Exchange M for N coverage when the last BM128 tile is mostly empty.
        // Both operands fitting LLC lets the wider N tile reuse its A rows.
        else if (output == Model810eOutputType::BF16 && m >= 2048 && n >= 2048 && n % 256 == 0 &&
                 m % 128 > 0 && m % 128 <= 64 && m * k * 2 <= (64LL << 20))
            t = {64, 256, 64, 32, 64, 3, 1};
    }
    if (output == Model810eOutputType::BF16 && plan.kind == Model810eDispatchKind::TSM &&
        m > 32 && m <= 64 && n >= 3072 && n < 16384 && k >= 4096 && k % 64 == 0 &&
        n * k * 2 <= (128LL << 20) && t.bm == 64 && t.bn == 128 && t.bk == 64 && t.splits == 1) {
        // Four stages retain two CTAs per tensor CU, versus three with three
        // stages. Deepen only when this does not add a resident wave. Wider
        // grids can instead reuse A across BN256 and halve their CTA count.
        const int64_t n_tiles = ceil_div(n, int64_t(128));
        if (ceil_div(n, int64_t(256)) >= 32 && n * k * 2 >= (64LL << 20)) {
            t = {64, 256, 64, 32, 64, 3, 1};
            t.peel_first_k = 1;
        } else if (ceil_div(n_tiles, int64_t(2 * kModel810eTensorCUs)) <=
                   ceil_div(n_tiles, int64_t(3 * kModel810eTensorCUs))) {
            t.stages = 4;
            t.peel_first_k = 1;
        }
    }
    if (output == Model810eOutputType::BF16 && plan.kind == Model810eDispatchKind::TSM &&
        m > 96 && m <= 128 && n >= 3072 && n <= 8192 && k >= 4096 && k <= 8192 && k % 128 == 0 &&
        (ceil_div(n, int64_t(128)) <= 2 * kModel810eTensorCUs || t.bn == 256)) {
        // One BM128 tile avoids rereading B. BK128 amortizes descriptor work;
        // keep grids needing an extra resident wave on their existing tile.
        t = {128, 128, 128, 32, 64, 2, 1};
        t.peel_first_k = 1;
    }
    if (output == Model810eOutputType::BF16 && plan.kind == Model810eDispatchKind::TSM &&
        m > 64 && m <= 512 && n >= 2560 && n <= 8192 && k >= 768 && k % 64 == 0 &&
        (n >= 3072 || k <= 8192) &&
        ceil_div(m, int64_t(96)) * 96 * 4 <= ceil_div(m, int64_t(t.bm)) * t.bm * 3) {
        // BM96 is worthwhile when it removes at least a quarter of padded M
        // work. Three stages overlap medium and deep reductions without split-K.
        t = {96, 128, 64, 48, 32, 3, 1};
        t.peel_first_k = 1;
    }
    if (m > 64 && m <= 96 && n >= 8192 && n % alignment == 0 && k >= 768 && k % 64 == 0) {
        // BM96 removes the padding of a second BM64 tile. WM48 retains a
        // power-of-two warp grid and lowers both registers and shared storage.
        // A cached B fits the two-stage pipeline; streams need a third stage.
        plan.kind = Model810eDispatchKind::TSM;
        plan.split_m = 0;
        t = {96, 128, 64, 48, 32, n * k * 2 <= (64LL << 20) ? 2 : 3, 1};
        t.peel_first_k = 1;
    }
    if (output == Model810eOutputType::BF16 && m > 2 && m <= 64 && n > 2048 &&
        k >= 2048 && k % 128 == 0 && n * k * 2 <= (128LL << 20)) {
        const int64_t tiles64 = ceil_div(n, int64_t(64));
        const int64_t tiles128 = ceil_div(n, int64_t(128));
        if (tiles128 < kModel810eTensorCUs) {
            // Just beyond the split-K region, BN128 leaves tensor CUs idle.
            // BN64 fills them; BK128/four stages hide long-reduction latency
            // while the complete grid still fits one shared-memory wave.
            const int bm = m <= 16 ? 16 : m <= 32 ? 32 : 64;
            t = {bm, 64, 128, bm <= 32 ? 16 : 32, 32, 4, 1};
            t.peel_first_k = 1;
        } else if (m <= 32 && tiles64 > 3 * kModel810eTensorCUs &&
                   tiles64 <= 4 * kModel810eTensorCUs) {
            // BM32/BN64/BK128/S3 holds three CTAs per tensor CU. BM16
            // holds four; for more rows, BN128 holds two and halves the grid.
            // Either change removes a second wave in this interval. Outside
            // it, the narrower tile can provide better memory parallelism.
            t = {m <= 16 ? 16 : 32, m <= 16 ? 64 : 128, 128, 16, 32, 3, 1};
            t.peel_first_k = 1;
        }
    }
    if (output == Model810eOutputType::BF16 && m > 2 && m <= 64 && n >= 8192 &&
        k >= 128 && k <= 512 && k % 64 == 0) {
        Model810eDispatchTile narrow{m <= 16 ? 16 : m <= 32 ? 32 : 64,
                                     128, 64, m <= 32 ? 16 : 32, 32, 2, 1};
        Model810eDispatchTile wide{64, 256, 64, 32, 64, 2, 1};
        const auto old_waves = model_810e_smem_waves(1, m, n, t);
        const auto narrow_waves = model_810e_smem_waves(1, m, n, narrow);
        const auto wide_waves = model_810e_smem_waves(1, m, n, wide);
        // Small-M short reductions amortize loop/epilogue setup across BN128.
        // For more rows, widen only when it removes a shared-memory wave.
        const bool unroll = m <= 32 ? ceil_div(n, int64_t(64)) > 10 * kModel810eTensorCUs :
            narrow_waves < old_waves && narrow_waves <= wide_waves;
        if (unroll || (m > 32 && wide_waves < old_waves && wide_waves < narrow_waves)) {
            plan.kind = Model810eDispatchKind::TSM;
            t = unroll ? narrow : wide;
            t.peel_first_k = 1;
            t.unroll_k = unroll;
        }
    }
    if (output == Model810eOutputType::BF16 && plan.kind == Model810eDispatchKind::TSM &&
        m > 2 && m <= 64 && n >= 8192 && k >= 768 && k <= 2048 && k % 64 == 0 &&
        t.bk == 64 && t.stages == 3 && t.splits == 1) {
        Model810eDispatchTile next{m <= 16 ? 16 : m <= 32 ? 32 : 64,
                                   128, 64, m <= 32 ? 16 : 32, 32, m <= 32 ? 4 : 3, 1};
        if (m > 32) {
            auto narrow = next;
            narrow.bn = 64;
            if (model_810e_smem_waves(1, m, n, narrow) <= model_810e_smem_waves(1, m, n, next))
                next = narrow;
        }
        // Deepen a small register tile, or replace an oversized N tile with
        // more independent CTAs, without adding a shared-memory wave.
        if (model_810e_smem_waves(1, m, n, next) <= model_810e_smem_waves(1, m, n, t)) {
            t = next;
            t.peel_first_k = 1;
        }
    }
    if (output == Model810eOutputType::BF16 && plan.kind == Model810eDispatchKind::TSM &&
        m > 32 && m <= 64 && k > 2048 && k <= 8192 && k % 64 == 0 && t.splits == 1) {
        const int64_t tiles = ceil_div(n, int64_t(256));
        if (tiles >= 32 && tiles <= 8 * kModel810eTensorCUs) {
            // BN256 amortizes each A panel across more output columns. BK128
            // halves the K-loop count, but also halves shared-memory residency;
            // use it only when fewer K iterations outweigh the extra waves.
            const bool deep_k = k % 128 == 0 &&
                ceil_div(tiles, int64_t(kModel810eTensorCUs)) <
                    2 * ceil_div(tiles, int64_t(2 * kModel810eTensorCUs));
            t = {64, 256, deep_k ? 128 : 64, 32, 64, 3, 1};
            t.peel_first_k = 1;
        }
    }
    if (output == Model810eOutputType::BF16 && plan.kind == Model810eDispatchKind::TSM &&
        m >= 192 && m <= 256 && n >= 2048 && n <= 4096 &&
        k >= 4096 && k <= 8192 && k % 64 == 0 && t.splits == 1) {
        Model810eDispatchTile next{128, 128, 64, 32, 64, 3, 1};
        if (model_810e_smem_waves(1, m, n, next) < model_810e_smem_waves(1, m, n, t)) {
            t = next;
            t.peel_first_k = 1;
        }
    }
    if (output == Model810eOutputType::BF16 && plan.kind == Model810eDispatchKind::TSM &&
        m > 256 && m <= 512 && n >= 2048 && n <= 4096 &&
        k >= 2048 && k <= 8192 && k % 64 == 0 && t.splits == 1 &&
        model_810e_smem_waves(1, m, n, t) > 1) {
        Model810eDispatchTile best = t;
        int64_t best_padding = ceil_div(m, int64_t(t.bm)) * t.bm;
        bool found = false;
        for (int bm : {64, 96}) {
            Model810eDispatchTile next{bm, 128, 64, bm == 96 ? 48 : 32, 32, 2, 1};
            const int64_t blocks = ceil_div(m, int64_t(bm)) * ceil_div(n, int64_t(128));
            const int64_t padding = ceil_div(m, int64_t(bm)) * bm;
            // A full grid supplies the overlap lost by shortening the pipeline.
            // Keep it in one shared-memory wave without increasing padded rows.
            if (blocks >= 3 * kModel810eTensorCUs && model_810e_smem_waves(1, m, n, next) == 1 &&
                padding <= best_padding && (!found || padding < best_padding)) {
                best = next; best_padding = padding; found = true;
            }
        }
        if (found) { t = best; t.peel_first_k = 1; }
    }
    if (output == Model810eOutputType::BF16 && m > 96 && m <= 224 && n >= 2048 &&
        k >= 768 && k % 64 == 0 && n % 128 == 0 && t.splits == 1) {
        const int bm = m <= 112 ? 112 : m <= 128 ? 128 : m <= 160 ? 160 : m <= 192 ? 192 : 224;
        const int wm = bm == 112 ? 112 : bm == 128 ? 64 : bm / 2;
        const int64_t n_tiles = ceil_div(n, int64_t(128));
        const bool compact_worthwhile = bm != 112 || t.bk == 128 || t.bn == 256 ||
            (n_tiles <= 40 && k >= 3072) || n * k * 2 > (96LL << 20);
        const bool narrow_reuse = bm != 160 || t.bm >= 128 || t.bn >= 256 ||
            n_tiles <= kModel810eTensorCUs || n_tiles >= 2 * kModel810eTensorCUs;
        if (compact_worthwhile && narrow_reuse && bm != 128 && bm != t.bm && bm <= ceil_div(m, int64_t(t.bm)) * t.bm &&
            n_tiles >= (bm == 112 ? 32 : 16) &&
            (bm != 224 || n_tiles <= kModel810eTensorCUs || n_tiles >= 32)) {
            const int bk = n_tiles <= kModel810eTensorCUs && bm >= 160 && bm <= 192 && k % 128 == 0 ? 128 : 64;
            int stages = n_tiles <= 2 * kModel810eTensorCUs ? 3 : 2;
            if (bm == 112 && n * k * 2 > (96LL << 20)) stages = 3;
            if (bm == 160 || bm == 192) {
                Model810eDispatchTile two{bm, 128, bk, wm, 32, 2, 1};
                auto three = two; three.stages = 3;
                if (model_810e_smem_waves(1, m, n, three) <= model_810e_smem_waves(1, m, n, two)) stages = 3;
            }
            if (bm == 224) stages = n_tiles < 24 || (k >= 2048 && n_tiles > 40) ? 3 : 2;
            if ((n * k * 2 > (128LL << 20) || (n_tiles > 64 && n * k * 2 > (64LL << 20))) && bm >= 160) stages = 3;
            Model810eDispatchTile next{bm, 128, bk, wm, 32, stages, 1};
            // An equally padded, register-heavy tile needs to remove at least
            // two resident waves to offset its lower warp-level parallelism.
            const auto old_waves = model_810e_smem_waves(1, m, n, t);
            const bool preserves_single_wave = old_waves > 1 || bk > t.bk ||
                bm * 4 <= ceil_div(m, int64_t(t.bm)) * t.bm * 3;
            const bool less_padding = bm < ceil_div(m, int64_t(t.bm)) * t.bm;
            const bool removes_waves = n * k * 2 <= (128LL << 20) &&
                model_810e_smem_waves(1, m, n, next) + 1 < old_waves;
            if (preserves_single_wave && (less_padding || removes_waves)) {
                plan.kind = Model810eDispatchKind::TSM;
                plan.split_m = 0;
                t = next;
                t.peel_first_k = 1;
            }
        }
    }
    if (output == Model810eOutputType::BF16 && m > 2 && m <= 16 && n >= 4096 &&
        k >= 1024 && k <= 8192 && k % 128 == 0) {
        Model810eDispatchTile next{16, 64, 128, 16, 16, 3, 1};
        auto deep = next; deep.stages = 4;
        if (model_810e_smem_waves(1, m, n, deep) <= model_810e_smem_waves(1, m, n, next)) next = deep;
        plan.kind = Model810eDispatchKind::TSM;
        t = next; t.peel_first_k = 1;
    }
    if (output == Model810eOutputType::BF16 && m > 16 && m <= 32 && n >= 16384 &&
        k >= 768 && k <= 2048 && k % 128 == 0) {
        plan.kind = Model810eDispatchKind::TSM;
        t = {32, 128, 128, 16, 32, 3, 1}; t.peel_first_k = 1;
    }
    if (output == Model810eOutputType::BF16 && m > 16 && m <= 32 && n > 5120 && n <= 8192 &&
        k >= 2048 && k <= 4096 && k % 64 == 0) {
        plan.kind = Model810eDispatchKind::TSM;
        t = {32, 64, 64, 16, 16, 3, 1}; t.peel_first_k = 1;
    }
    // Use the established peeled K loop for long reductions on short M
    // grids. This avoids the unpeeled variant exercised by the rejected
    // persistent experiment, without changing K coverage or accumulation.
    if (plan.kind == Model810eDispatchKind::TSM && m > 64 && m <= 512 &&
        k >= 4096 && k % t.bk == 0 && t.splits == 1)
        t.peel_first_k = 1;
    return plan;
}

// In-CTA K reduction replaces a two-launch split-K product where output
// parallelism is modest and K is too short to amortize global partials.
// Keep deep reductions and large output grids on the established scheduler.
static int get_model_810e_cute_free_config(int m, int n, int k,
    Model810eOutputType output, Model810eDispatchTile& tile) {
    if (m < 16 || m > 256 || n < 128 || n > 2048 || k < 768 || k > 1536 ||
        k % 32 != 0 || (k > 1024 && n < 256)) return 0;
    if (get_model_810e_dense_dispatch(m, n, k, output).tile.splits <= 1) return 0;
    tile = {m <= 32 ? 16 : 32, 64, 256, 16, 32, 2, 1};
    const int tiles = ceil_div(m, tile.bm) * ceil_div(n, tile.bn);
    // BM32/BK256 uses 96 KiB. Conservatively budget two CTAs per tensor CU
    // for both tile sizes to avoid additional waves as the output grid grows.
    if (tiles > 2 * kModel810eTensorCUs) return 0;
    // Extra K warps help the smallest grids; larger grids already supply
    // independent work and prefer fewer reduction participants per CTA.
    return tiles < kModel810eTensorCUs ? 4 : 2;
}

static Model810eDispatchTile get_model_810e_batch_dispatch(
    int64_t batch, int64_t m, int64_t n, int64_t k) {
    const int bk = k >= 128 && k % 64 == 0 ? 64 : 32;
    const int stages = int(std::min<int64_t>(k <= 256 || m > 64 ? 2 : 3, k / bk));
    Model810eDispatchTile t = m <= 64 ?
        Model810eDispatchTile{64, n >= 512 ? 256 : 128, bk, 32, n >= 512 ? 64 : 32, stages, 1} :
        Model810eDispatchTile{128, 128, bk, k >= 512 ? 64 : 32, k >= 512 ? 32 : 64, stages, 1};
    const auto blocks = [&]() { return batch * ceil_div(m, int64_t(t.bm)) * ceil_div(n, int64_t(t.bn)); };
    if (m <= 64 && blocks() < 2 * kModel810eTensorCUs) {
        // A batch contributes independent CTAs. Reduce the tile only when the
        // combined grid is small, retaining larger tiles for large batches.
        t.bm = 16; t.bn = 64; t.wm = 16; t.wn = 32;
        if (blocks() < kModel810eTensorCUs) { t.bn = 32; t.wn = 16; }
    } else if (m > 64 && ceil_div(m, int64_t(t.bm)) * t.bm * 4 > m * 5) {
        // Avoid spending more than one quarter of the M work on padding.
        while (t.bm > 32 && ceil_div(m, int64_t(t.bm)) * t.bm * 4 > m * 5) t.bm /= 2;
        t.bn = 64; t.wm = t.bm <= 32 ? 16 : 32; t.wn = 32;
    }
    if (t.bm == 64 && m <= 32 && n >= 256 && n <= 1024 &&
        k >= 128 && k <= 256 && k % 64 == 0) {
        // Batch already supplies parallelism. Cover only the useful M rows;
        // BN128 keeps enough resident CTAs while reducing accumulator work.
        t = {m <= 16 ? 16 : 32, 128, 64, 16, 32, 2, 1};
    }
    return t;
}

} // namespace deep_gemm_bf16_common
