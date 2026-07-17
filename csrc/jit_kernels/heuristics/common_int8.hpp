#pragma once
#include <tuple>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"
#include "../../utils/utils.hpp"
#include "gemm_int8_lut.hpp"
using namespace deep_gemm;
namespace deep_gemm_int8 {

std::tuple<int, int, int> get_smem_config(int num_stages, int k, int block_m, int block_n, int block_k = 128,
                                          int bpp = 1) {
    // Try swizzle first
    int swizzle_mode = 128;
    // int block_n_padding = (swizzle_mode == 0) ? get_block_n_padding_for_smem_d(block_n) : 0;
    int block_n_padding = 0;

    int smem_d = block_m * (block_n + block_n_padding);
    int smem_a_per_stage = block_m * block_k;
    int smem_b_per_stage = block_n * block_k;

    int smem_size_d = smem_d * 2;
    int smem_size_a = num_stages * smem_a_per_stage * bpp;
    int smem_size_b = num_stages * smem_b_per_stage * bpp;

    int smem_size = std::max(smem_size_d, smem_size_a + smem_size_b);

    // Swizzle and padding are mutually exclusive
    assert((swizzle_mode > 0) + (block_n_padding > 0) <= 1);

    return std::make_tuple(smem_size, swizzle_mode, block_n_padding);
}

std::tuple<int> get_num_occ(int block_m, int block_n, int block_k, int num_stages) {
    const int ppu_capacity = 262144;
    const int bpp = 1;

    int smem_d = block_m * block_n;
    int smem_a_per_stage = block_m * block_k;
    int smem_b_per_stage = block_n * block_k;

    int smem_size_d = smem_d * 2;
    int smem_size_a = num_stages * smem_a_per_stage * bpp;
    int smem_size_b = num_stages * smem_b_per_stage * bpp;

    int smem_size = std::max(smem_size_d, smem_size_a + smem_size_b);
    int smem_occ = ppu_capacity / smem_size;

    // Create lookup table
    std::unordered_map<long long, int> vreg_occ_lut = {
        {16LL * 100000 + 256, 3}, {32LL * 100000 + 128, 3}, {32LL * 100000 + 256, 2}, {64LL * 100000 + 64, 2},
        {64LL * 100000 + 128, 2}, {64LL * 100000 + 256, 2}, {128LL * 100000 + 128, 2}};

    long long key = static_cast<long long>(block_m) * 100000 + block_n;
    int vreg_occ = 1;

    // Lookup
    auto it = vreg_occ_lut.find(key);
    if (it != vreg_occ_lut.end()) {
        vreg_occ = it->second;
    }

    return std::make_tuple(std::min(smem_occ, vreg_occ));
}

int get_smem_occ(int block_m, int block_n, int block_k, int num_stages) {
    if (block_m == 0) {
        return 0;
    }

    // use static suppose.
    const int ppu_capacity = 262144;
    const int bpp = 1;

    int smem_d = block_m * block_n;
    int smem_a_per_stage = block_m * block_k;
    int smem_b_per_stage = block_n * block_k;

    int smem_size_d = smem_d * 2;
    int smem_size_a = num_stages * smem_a_per_stage * bpp;
    int smem_size_b = num_stages * smem_b_per_stage * bpp;

    int smem_size = std::max(smem_size_d, smem_size_a + smem_size_b);

    return ppu_capacity / smem_size;
}

std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>
get_best_configs_ppu1v5(int m, int n, int k, int num_groups, int num_sms, bool is_grouped_contiguous = false,
                        bool is_grouped_masked = false) {
    // todo: add more tiles for ppu1.5
    int best_block_m = 256;
    int best_block_n = 256;
    int best_block_k = 128;
    int best_warp_m = 64;
    int best_warp_n = 64;
    int best_stages = 4;

    auto best_smem_config = get_smem_config(best_stages, k, best_block_m, best_block_n, best_block_k, 1);
    int num_min_sms = get_sm_count();

    return std::make_tuple(num_min_sms, best_block_m, best_block_n, best_block_k, best_warp_m, best_warp_n, best_stages,
                           best_smem_config);
}

std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>
get_best_configs_dense_ppu1v5(int m, int n, int k, int num_groups, int num_sms) {
    // FIXME: block m can add 16, and blockM/N could be 512, and 48, 96 blockM.
    std::vector<int> block_ms = {256, 192, 128, 64, 32, 16};
    std::vector<int> block_ns = {256, 128, 64, 32};

    auto fix_wave_saturate = [num_sms](int x) -> int {
        return (x == 0) ? num_sms : x;
    };

    auto get_num_waves = [m, n, num_groups, num_sms](int bm, int bn) -> int {
        if (bm == 0)
            return 0;
        return ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms);
    };

    auto get_last_wave_util = [m, n, num_groups, num_sms, fix_wave_saturate](int bm, int bn) -> int {
        int waves = ceil_div(m, bm) * ceil_div(n, bn) * num_groups;
        return fix_wave_saturate(waves % num_sms);
    };

    auto get_block_utils = [](int dim, int block_dim) -> double {
        if (block_dim == 0)
            return 0.0;
        if (dim % block_dim != 0) {
            int num_blocks = (dim + block_dim - 1) / block_dim;
            return (static_cast<double>(dim) / block_dim) / num_blocks;
        }
        return 1.0;
    };

    auto get_block_ai = [](int block_m, int block_n) -> double {
        return static_cast<double>(block_m * block_n) / (block_m + block_n);
    };

    // Decide block sizes by waves
    int best_block_m = 0, best_block_n = 0;

    for (int block_m : block_ms) {
        // NOTES:
        // for PPU1.0: the block sizes can not be too large, so at least one dim less than 128
        // for PPU1.5: the tile 256x256 is good for many compute bound case

        std::vector<int> block_ns_after_filter;
        if ((m >= 128 && k >= 2048) || m >= 256) {
            // block_ns_after_filter = filter(lambda bn: (bn != n and n >= 1), block_ns)
            for (int bn : block_ns) {
                if (bn != n && n >= 1) {
                    block_ns_after_filter.push_back(bn);
                }
            }
        } else {
            // block_ns_after_filter = filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 1) and not
            // (block_m == 16 and bn == 32)), block_ns)
            for (int bn : block_ns) {
                if ((block_m <= 128 || bn <= 128) && (bn != n && n >= 1) && !(block_m == 16 && bn == 32)) {
                    block_ns_after_filter.push_back(bn);
                }
            }
        }

        for (int block_n : block_ns_after_filter) {
            bool success = false;
            int num_waves = get_num_waves(block_m, block_n);
            int best_num_waves = (best_block_m == 0) ? 0 : get_num_waves(best_block_m, best_block_n);

            double num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n);
            double best_num_utils =
                (best_block_m == 0) ? 0.0 : get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n);

            int num_occ = get_smem_occ(block_m, block_n, 128, 2);
            int best_num_occ = (best_block_m == 0) ? 0 : get_smem_occ(best_block_m, best_block_n, 128, 2);

            if (best_block_m == 0 || best_block_n == 0) {
                success = true;
            } else if (m <= 256 || n < 512) {
                // if single group block is small, balance wave, utils and occ
                double occ_wave = static_cast<double>(num_waves) / num_occ;
                double best_occ_wave = static_cast<double>(best_num_waves) / best_num_occ;
                double ai_util = get_block_ai(block_m, block_n);
                double best_ai_util = get_block_ai(best_block_m, best_block_n);

                bool valid_occ = (static_cast<double>(num_occ) / best_num_occ) >= 1;
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

            if (success) {
                best_block_m = block_m;
                best_block_n = block_n;
            }
        }
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
    const int ppu_capacity = 262144;

    int block_k = 128;
    if (k <= 256) {
        block_k = 64;
    }
    if (k >= 4096 && ((best_block_m <= 32 && best_block_n <= 64) || (best_block_m == 64 && best_block_n == 128))) {
        block_k = 256;
    }

    std::vector<int> stage_candidates;
    int max_stages = k / block_k;
    if (5 <= max_stages)
        stage_candidates.push_back(5);
    if (4 <= max_stages)
        stage_candidates.push_back(4);
    if (3 <= max_stages)
        stage_candidates.push_back(3);
    if (2 <= max_stages)
        stage_candidates.push_back(2);

    // if not stage_candidates or (128 % best_block_n != 0 and 128 // gcd(128, best_block_n) <= 4):
    if (stage_candidates.empty() || (128 % best_block_n != 0 && 128 / gcd(128, best_block_n) <= 4)) {
        stage_candidates = {3, 2};
    }

    if (best_block_m >= 128 && best_block_n >= 128) {
        stage_candidates = {4};
    }

    int best_occ = 0;
    for (int num_stages : stage_candidates) {
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k, 1);

        if (std::get<0>(best_smem_config) <= ppu_capacity) {
            int occ = ppu_capacity / std::get<0>(best_smem_config);
            if (k < 512 || (best_block_m > 64 && best_block_n >= 64) && occ >= best_occ) {
                // compute block use higher occ rather than large stage
                best_num_stages = num_stages;
                best_occ = occ;
            } else {
                best_num_stages = num_stages;
                break;
            }
        }
    }

    assert(best_num_stages != 0);

    // Recompute the minimal number of SMs required
    // NOTES: less L2 cache usage and less GPU frequency drop
    int num_waves = get_num_waves(best_block_m, best_block_n);
    int num_min_sms;
    if (is_ppu1v5_device()) {
        num_min_sms = num_sms;
    } else {
        num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves);
    }

    assert(num_min_sms <= num_sms);

    int warp_m = best_block_m / 2;
    int warp_n = best_block_n / 2;

    if (best_block_m >= 128 && best_block_n == 256) {
        warp_m = best_block_m / 4;
        warp_n = best_block_n / 4;
    } else if (best_block_m == 32 && best_block_n >= 64) {
        warp_m = 32;
        warp_n = best_block_n / 4;
    } else if (best_block_n == 32 && n <= 128 && best_block_m >= 64) {
        warp_m = best_block_m / 4;
        warp_n = 32;
    } else if ((best_block_m == 128 || best_block_m == 256 || best_block_m == 192) && best_block_n >= 32) {
        warp_m = best_block_m / 4;
        warp_n = (best_block_n != 32) ? best_block_n / 2 : best_block_n;
    } else if (best_block_m == 16) {
        warp_m = 16;
        best_block_n = (n < 512) ? 64 : best_block_n;
        warp_n = (best_block_n <= 128) ? best_block_n / 4 : best_block_n / 8;
    } else if (best_block_n == 128 || best_block_n == 256) {
        warp_m = (best_block_m != 32) ? best_block_m / 2 : best_block_m;
        warp_n = best_block_n / 4;
    }

    return std::make_tuple(std::min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n,
                           best_num_stages, best_smem_config);
}

std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>
get_best_configs_ppu1v5(int m, int n, int k, int num_groups, int num_sms, bool is_grouped_contiguous = false,
                        bool is_grouped_masked = false, int max_block_n = 256) {
    if (num_groups == 1 && is_grouped_contiguous == false && is_grouped_masked == false) {
        return get_best_configs_dense_ppu1v5(m, n, k, num_groups, num_sms);
    }

    // FIXME: block m can add 16, and blockM/N could be 512, and 48, 96 blockM.
    std::vector<int> block_ms;
    if (!is_grouped_contiguous) {
        if (k >= 384) {
            block_ms = {256, 128, 64, 32, 16};
        } else {
            block_ms = {128, 64, 32, 16};
        }
    } else {
        block_ms = {get_m_alignment_for_contiguous_layout()};
    }

    // block_ns = (256, 128, 64, 32)
    assert(max_block_n > 0 && (max_block_n & (max_block_n - 1)) == 0);

    std::vector<int> block_ns;
    int bit_length = 0;
    int temp_n = max_block_n;
    while (temp_n > 0) {
        bit_length++;
        temp_n >>= 1;
    }

    int start_bit = (k >= 384) ? bit_length - 1 : bit_length - 2;
    for (int x = start_bit; x > 4; x--) {
        block_ns.push_back(1 << x);
    }

    auto fix_wave_saturate = [num_sms](int x) -> int {
        return (x == 0) ? num_sms : x;
    };

    auto get_num_waves = [m, n, num_groups, num_sms](int bm, int bn) -> int {
        if (bm == 0)
            return 0;
        return ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms);
    };

    auto get_last_wave_util = [m, n, num_groups, num_sms, fix_wave_saturate](int bm, int bn) -> int {
        int waves = ceil_div(m, bm) * ceil_div(n, bn) * num_groups;
        return fix_wave_saturate(waves % num_sms);
    };

    auto get_block_utils = [](int dim, int block_dim) -> double {
        if (block_dim == 0)
            return 0.0;
        if (dim % block_dim != 0) {
            int num_blocks = (dim + block_dim - 1) / block_dim;
            return (static_cast<double>(dim) / block_dim) / num_blocks;
        }
        return 1.0;
    };

    auto get_block_ai = [](int block_m, int block_n) -> double {
        return static_cast<double>(block_m * block_n) / (block_m + block_n);
    };

    // Decide block sizes by waves
    int best_block_m = 0, best_block_n = 0;

    for (int block_m : block_ms) {
        // NOTES:
        // for PPU1.0: the block sizes can not be too large, so at least one dim less than 128
        // for PPU1.5: the tile 256x256 is good for many compute bound case

        std::vector<int> block_ns_after_filter;
        if (is_ppu1v5_device() && ((m >= 128 && k > 2048) || (m >= 256 && k >= 512))) {
            // block_ns_after_filter = filter(lambda bn: (bn != n and n >= 32) and not (block_m == 16 and bn <= 32),
            // block_ns)
            for (int bn : block_ns) {
                if ((bn != n && n >= 32) && !(block_m == 16 && bn <= 32)) {
                    block_ns_after_filter.push_back(bn);
                }
            }
        } else {
            // block_ns_after_filter = filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 32) and
            // not (block_m == 16 and bn <= 32)), block_ns)
            for (int bn : block_ns) {
                if ((block_m <= 128 || bn <= 128) && (bn != n && n >= 32) && !(block_m == 16 && bn <= 32)) {
                    block_ns_after_filter.push_back(bn);
                }
            }
        }

        for (int block_n : block_ns_after_filter) {
            bool success = false;
            int num_waves = get_num_waves(block_m, block_n);
            int best_num_waves = (best_block_m == 0) ? 0 : get_num_waves(best_block_m, best_block_n);

            double num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n);
            double best_num_utils =
                (best_block_m == 0) ? 0.0 : get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n);

            int num_occ = get_smem_occ(block_m, block_n, 128, 2);
            int best_num_occ = (best_block_m == 0) ? 0 : get_smem_occ(best_block_m, best_block_n, 128, 2);

            if (best_block_m == 0 || best_block_n == 0) {
                success = true;
            } else if (m < 512 || n < 512) {
                // if single group block is small, balance wave, utils and occ
                double occ_wave = static_cast<double>(num_waves) / num_occ;
                double best_occ_wave = static_cast<double>(best_num_waves) / best_num_occ;
                double ai_util = get_block_ai(block_m, block_n);
                double best_ai_util = get_block_ai(best_block_m, best_block_n);

                bool valid_occ = (static_cast<double>(num_occ) / best_num_occ) >= 1;
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

            if (success) {
                best_block_m = block_m;
                best_block_n = block_n;
            }
        }
    }

    if (is_ppu1v5_device() && (m >= 96 && m < 128 && n > 2048 && k > 2048 && num_groups >= 8)) {
        best_block_m = 192;
        best_block_n = 256;
    }

    if ((best_block_m - 10 <= m && m <= best_block_m) && best_block_m == 16) {
        best_block_m = best_block_m * 2;
    }

    // qwen3-next & deepseek gemm2 need to fix some issues
    if ((best_block_m - 10 <= m && m <= best_block_m) && (best_block_m == 32 || best_block_m == 64) && k > 256) {
        best_block_m = best_block_m * 2;
    }

    // small m hbm bound or latency bound, wave is not useful, for better occ for 810e hbm bound, use smallest blockN
    // for m16
    if (m < 10 && n < 512) {
        best_block_m = 16;
        best_block_n = 64;
    }

    assert(best_block_m != 0 && best_block_n != 0);

    // Always pick the longest one
    // NOTES: for double B scales, the best number of stages may be reduced
    int best_num_stages = 0;
    std::tuple<int, int, int> best_smem_config;
    const int ppu_capacity = 262144;

    int block_k = 128;
    if (k == 128) {
        block_k = 64;
    }
    if (k >= 4096 && (best_block_m <= 32 && best_block_n <= 64)) {
        block_k = 256;
    }

    std::vector<int> stage_candidates;
    int max_stages = k / block_k;
    if (8 <= max_stages)
        stage_candidates.push_back(8);
    if (7 <= max_stages)
        stage_candidates.push_back(7);
    if (6 <= max_stages)
        stage_candidates.push_back(6);
    if (5 <= max_stages)
        stage_candidates.push_back(5);
    if (4 <= max_stages)
        stage_candidates.push_back(4);
    if (3 <= max_stages)
        stage_candidates.push_back(3);
    if (2 <= max_stages)
        stage_candidates.push_back(2);

    // if not stage_candidates or (128 % best_block_n != 0 and 128 // gcd(128, best_block_n) <= 4) or best_block_m == 16
    // or best_block_m == 32:
    if (stage_candidates.empty() || (128 % best_block_n != 0 && 128 / gcd(128, best_block_n) <= 4) ||
        best_block_m == 16 || best_block_m == 32) {
        stage_candidates = {3, 2};
    }

    if (best_block_m > 128 && best_block_n == 256) {
        stage_candidates = {4};
    }

    int best_occ = 0;
    for (int num_stages : stage_candidates) {
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k, 1);

        if (std::get<0>(best_smem_config) <= ppu_capacity) {
            int occ = ppu_capacity / std::get<0>(best_smem_config);
            if (k < 512 || (best_block_m > 32 && best_block_n >= 64) && occ >= best_occ) {
                // compute block use higher occ rather than large stage
                best_num_stages = num_stages;
                best_occ = occ;
            } else {
                best_num_stages = num_stages;
                break;
            }
        }
    }

    assert(best_num_stages != 0);

    // Recompute the minimal number of SMs required
    // NOTES: less L2 cache usage and less GPU frequency drop
    int num_waves = get_num_waves(best_block_m, best_block_n);
    int num_min_sms;
    if (is_ppu1v5_device()) {
        num_min_sms = num_sms;
    } else {
        num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves);
    }

    assert(num_min_sms <= num_sms);

    int warp_m = best_block_m / 2;
    int warp_n = best_block_n / 2;

    if (best_block_m > 128 && best_block_n == 256) {
        warp_m = best_block_m / 4;
        warp_n = best_block_n / 4;
    } else if (best_block_m == 64 && best_block_n >= 128) {
        warp_m = (k < 256) ? 64 : 32;
        warp_n = 64;
    } else if (best_block_m == 32 && best_block_n >= 64) {
        warp_m = 32;
        warp_n = (best_block_n <= 128) ? best_block_n / 2 : best_block_n / 4;
    } else if (best_block_n == 32 && n <= 128 && best_block_m >= 64) {
        warp_m = best_block_m / 4;
        warp_n = 32;
    } else if ((best_block_m == 128 || best_block_m == 256) && best_block_n >= 32) {
        warp_m = 64;
        warp_n = (best_block_n <= 128) ? best_block_n / 2 : best_block_n / 4;
    } else if (best_block_m == 16) {
        warp_m = 16;
        best_block_n = (n < 512) ? 64 : best_block_n;
        warp_n = (best_block_n < 128) ? best_block_n / 2 : best_block_n / 4;
    } else if (best_block_n == 128 || best_block_n == 256) {
        warp_m = (best_block_m != 32) ? best_block_m / 2 : best_block_m;
        warp_n = 64;
    }

    return std::make_tuple(std::min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n,
                           best_num_stages, best_smem_config);
}

std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>
get_best_configs(int m, int n, int k, int num_groups, int num_sms, bool is_grouped_contiguous = false,
                 bool is_grouped_masked = false, int max_block_n = 256) {
    auto lut_result = deep_gemm_int8_lut::get_best_configs_from_lut(m, n, k);
    if (num_groups == 1 && lut_result.has_value()) {
        int best_block_m, best_block_n, best_block_k, best_warp_m, best_warp_n, best_stages;
        std::tie(best_block_m, best_block_n, best_block_k,
                 best_warp_m, best_warp_n, best_stages) = lut_result.value();
        auto best_smem_config = get_smem_config(best_stages, k,
                                                best_block_m, best_block_n,
                                                best_block_k, 1);
        return std::make_tuple(num_sms, best_block_m, best_block_n, best_block_k,
                               best_warp_m, best_warp_n, best_stages, best_smem_config);
    }
    if (is_ppu1v5_device()) {
        return get_best_configs_ppu1v5(m, n, k, num_groups, num_sms, is_grouped_contiguous, is_grouped_masked,
                                       max_block_n);
    }
    // FIXME: block m can add 16, and blockM/N could be 512, and 48, 96 blockM.
    std::vector<int> block_ms;
    if (!is_grouped_contiguous) {
        if (k > 384) {
            block_ms = {256, 128, 64, 32, 16};
        } else {
            block_ms = {64, 32, 16};
        }
    } else {
        block_ms = {get_m_alignment_for_contiguous_layout()};
    }

    // block_ns = (256, 128, 64, 32)
    assert(max_block_n > 0 && (max_block_n & (max_block_n - 1)) == 0);

    std::vector<int> block_ns;
    int bit_length = 0;
    int temp_n = max_block_n;
    while (temp_n > 0) {
        bit_length++;
        temp_n >>= 1;
    }

    int start = (k > 384) ? (bit_length - 1) : (bit_length - 2);
    for (int x = start; x > 4; --x) {
        block_ns.push_back(1 << x);
    }

    auto fix_wave_saturate = [num_sms](int x) -> int {
        return (x == 0) ? num_sms : x;
    };

    auto get_num_waves = [m, n, num_groups, num_sms](int bm, int bn) -> int {
        if (bm == 0)
            return 0;
        return ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms);
    };

    auto get_last_wave_util = [m, n, num_groups, num_sms, fix_wave_saturate](int bm, int bn) -> int {
        int waves = ceil_div(m, bm) * ceil_div(n, bn) * num_groups;
        return fix_wave_saturate(waves % num_sms);
    };

    auto get_block_utils = [](int dim, int block_dim) -> double {
        if (block_dim == 0)
            return 0.0;
        if (dim % block_dim != 0) {
            int num_blocks = (dim + block_dim - 1) / block_dim;
            return (static_cast<double>(dim) / block_dim) / num_blocks;
        }
        return 1.0;
    };

    auto get_block_ai = [](int block_m, int block_n) -> double {
        return static_cast<double>(block_m * block_n) / (block_m + block_n);
    };

    // Decide block sizes by waves
    int best_block_m = 0, best_block_n = 0;
    int min_n_threshold = (num_groups == 1 && is_grouped_contiguous == false && is_grouped_masked == false) ? 1 : 32;

    for (int block_m : block_ms) {
        // NOTES:
        // for PPU1.0: the block sizes can not be too large, so at least one dim less than 128
        // for PPU1.5: the tile 256x256 is good for many compute bound case

        std::vector<int> block_ns_after_filter;
        if (is_ppu1v5_device() && ((m >= 128 && k > 2048) || m >= 256)) {
            // block_ns_after_filter = filter(lambda bn: (bn != n and n >= min_n_threshold) and not (block_m == 16 and
            // bn <= 32), block_ns)
            for (int bn : block_ns) {
                if ((bn != n && n >= min_n_threshold) && !(block_m == 16 && bn <= 32)) {
                    block_ns_after_filter.push_back(bn);
                }
            }
        } else {
            // block_ns_after_filter = filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >=
            // min_n_threshold) and not (block_m == 16 and bn <= 32)), block_ns)
            for (int bn : block_ns) {
                if ((block_m <= 128 || bn <= 128) && (bn != n && n >= min_n_threshold) &&
                    !(block_m == 16 && bn <= 32)) {
                    block_ns_after_filter.push_back(bn);
                }
            }
        }

        for (int block_n : block_ns_after_filter) {
            bool success = false;
            int num_waves = get_num_waves(block_m, block_n);
            int best_num_waves = (best_block_m == 0) ? 0 : get_num_waves(best_block_m, best_block_n);

            double num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n);
            double best_num_utils =
                (best_block_m == 0) ? 0.0 : get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n);

            int num_occ = get_smem_occ(block_m, block_n, 128, 2);
            int best_num_occ = (best_block_m == 0) ? 0 : get_smem_occ(best_block_m, best_block_n, 128, 2);

            if (best_block_m == 0 || best_block_n == 0) {
                success = true;
            } else if (m < 512 || n < 512) {
                // if single group block is small, balance wave, utils and occ
                double occ_wave = static_cast<double>(num_waves) / num_occ;
                double best_occ_wave = static_cast<double>(best_num_waves) / best_num_occ;
                double ai_util = get_block_ai(block_m, block_n);
                double best_ai_util = get_block_ai(best_block_m, best_block_n);

                bool valid_occ = (static_cast<double>(num_occ) / best_num_occ) >= 1;
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

            if (success) {
                best_block_m = block_m;
                best_block_n = block_n;
            }
        }
    }

    if (is_ppu1v5_device() && (m >= 96 && m < 128 && n > 2048 && k > 2048 && num_groups >= 8)) {
        best_block_m = 192;
        best_block_n = 256;
    }

    // small m hbm bound or latency bound, wave is not useful, for better occ for 810e hbm bound, use smallest blockN
    // for m16
    if (m < 6 && n < 512) {
        best_block_m = 16;
        best_block_n = 64;
    }

    if ((best_block_m - 10 <= m && m <= best_block_m) && best_block_m == 16 && k > 384) {
        best_block_m = best_block_m * 2;
    }

    assert(best_block_m != 0 && best_block_n != 0);

    // Always pick the longest one
    // NOTES: for double B scales, the best number of stages may be reduced
    int best_num_stages = 0;
    std::tuple<int, int, int> best_smem_config;
    const int ppu_capacity = 262144;

    if (!is_ppu1v5_device() && 64 < m && m < 128) {
        best_block_m = 64;
    }

    int block_k = 128;
    if (k <= 256) {
        block_k = 64;
    }
    if (k >= 4096 && (best_block_m <= 32 && best_block_n <= 128)) {
        block_k = 256;
    }

    std::vector<int> stage_candidates;
    int max_stages = k / block_k;
    if (8 <= max_stages)
        stage_candidates.push_back(8);
    if (7 <= max_stages)
        stage_candidates.push_back(7);
    if (6 <= max_stages)
        stage_candidates.push_back(6);
    if (5 <= max_stages)
        stage_candidates.push_back(5);
    if (4 <= max_stages)
        stage_candidates.push_back(4);
    if (3 <= max_stages)
        stage_candidates.push_back(3);
    if (2 <= max_stages)
        stage_candidates.push_back(2);

    // Recompute the minimal number of SMs required
    // NOTES: less L2 cache usage and less GPU frequency drop
    int num_waves = get_num_waves(best_block_m, best_block_n);
    int num_min_sms;
    num_min_sms = num_sms;

    assert(num_min_sms <= num_sms);

    int warp_m = best_block_m / 2;
    int warp_n = best_block_n / 2;

    if (best_block_m > 128 && best_block_n == 256) {
        warp_m = best_block_m / 4;
        warp_n = best_block_n / 4;
    } else if (best_block_m == 32 && best_block_n >= 64) {
        warp_m = (k > 256) ? best_block_m / 2 : 32;
        warp_n = (best_block_n <= 128) ? best_block_n / 4 : best_block_n / 8;
    } else if (best_block_n == 32 && n <= 128 && best_block_m >= 64) {
        warp_m = best_block_m / 4;
        warp_n = 32;
    } else if ((best_block_m == 128) || ((best_block_m == 256) && best_block_n >= 32)) {
        warp_m = 64;
        warp_n = (best_block_n <= 128) ? best_block_n / 2 : best_block_n / 4;
    } else if (best_block_m == 16) {
        warp_m = 16;
        best_block_n = (n <= 512) ? 64 : best_block_n;
        warp_n = (best_block_n <= 128) ? best_block_n / 4 : best_block_n / 8;
    } else if (best_block_n == 128 || best_block_n == 256) {
        warp_m = (best_block_m != 32) ? best_block_m / 2 : best_block_m;
        warp_n = best_block_n / 4;
    }

    if (stage_candidates.empty() || (128 % best_block_n != 0 && 128 / gcd(128, best_block_n) <= 4) ||
        best_block_m == 16 || best_block_m == 32) {
        stage_candidates = {3, 2};
    }

    if (best_block_m == 64 && best_block_n == 128) {
        stage_candidates = {3, 2};
    }

    if (best_block_m > 128 && best_block_n == 256) {
        stage_candidates = {4};
    }

    int best_occ = 0;
    for (int num_stages : stage_candidates) {
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k, 1);
        if (std::get<0>(best_smem_config) <= ppu_capacity) {
            int occ = ppu_capacity / std::get<0>(best_smem_config);
            if (k < 512 || ((best_block_m > 64 && best_block_n >= 64) && occ >= best_occ)) {
                best_num_stages = num_stages;
                best_occ = occ;
            } else {
                best_num_stages = num_stages;
                break;
            }
        }
    }

    assert(best_num_stages != 0);
    return std::make_tuple(std::min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n,
                           best_num_stages, best_smem_config);
}

} // namespace deep_gemm_int8
