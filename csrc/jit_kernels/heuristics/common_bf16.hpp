#pragma once
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"
#include "../../utils/utils.hpp"
#include "gemm_search_space.hpp"
#include "adaptive_tile_selector.hpp"

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

// Returns 8-element tuple: (num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config)
// NOTE: baseline 8-element contract. Adaptive warp_k/dense_s2_opt injection is done in the dense impl
// (bf16_gemm.hpp), so MoE grouped callers unpack this tuple directly without narrowing.
using ConfigResult = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

// Adaptive tile selection for DenseGemm (BF16). warp_k/dense_s2_opt are injected in the dense
// impl; here we only apply the adaptive tile choice and return the 8-tuple.
ConfigResult get_adaptive_configs_bf16(int m, int n, int k, int num_sms) {
    auto [sms, bm, bn, bk, wm, wn, wk, stages] =
        deep_gemm_adaptive::get_adaptive_configs(m, n, k, num_sms);
    (void)wk;
    auto smem = get_smem_config(stages, k, bm, bn, bk);
    return std::make_tuple(sms, bm, bn, bk, wm, wn, stages, smem);
}

ConfigResult get_best_configs(int m, int n, int k, int num_groups, int num_sms, bool is_grouped_contiguous = false,
                              bool is_grouped_masked = false, int max_block_n = 256) {
    // Adaptive tile selection for DenseGemm (BF16). warp_k/dense_s2_opt are injected in the dense
    // impl; here we only apply the adaptive tile choice and return the 8-tuple.
    if (num_groups == 1 && !is_grouped_contiguous && !is_grouped_masked && is_ppu1v5_device() &&
        (deep_gemm_adaptive::is_bf16_adaptive_shape(m, n, k) ||
         deep_gemm_adaptive::bf16_adaptive_enabled())) {
        return get_adaptive_configs_bf16(m, n, k, num_sms);
    }
    // Generate block_ms
    std::vector<int> block_ms;
    if (!is_grouped_contiguous) {
        if (k >= 384) {
            block_ms = {256, 128, 64, 32, 16};
        } else {
            block_ms = {64, 32, 16};
        }
    } else {
        block_ms = {get_m_alignment_for_contiguous_layout()};
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
    int min_n_threshold = (num_groups == 1 && !is_grouped_contiguous && !is_grouped_masked) ? 1 : 32;
    for (int block_m : block_ms) {
        // Filter block_ns
        std::vector<int> block_ns_after_filter;
        for (int block_n : block_ns) {
            bool condition;
            if (is_ppu1v5_device() && ((m >= 128 && k > 2048) || m >= 256)) {
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
    if (m < 20 && n < 512) {
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

    return std::make_tuple(std::min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n,
                           best_num_stages, best_smem_config);
}

// Pre-configured optimal tiling greater than or equal to 4096
const std::vector<std::vector<int>> CONFIG_TILE_GREATER_4096 = {{128, 128, 64, 64, 32, 64, 2},
                                                                {256, 128, 64, 64, 64, 64, 2},
                                                                {512, 128, 64, 64, 64, 64, 3},
                                                                {128, 256, 64, 32, 128, 64, 2}};
std::tuple<int, int, int, int, int, int, int, int, bool, std::tuple<int, int, int>>
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

    return {num_min_sms, bm, bn, bk, wm, wn, bk, stages, false, best_smem_config};
}
} // namespace deep_gemm_bf16_common
