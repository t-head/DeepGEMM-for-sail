#pragma once
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"
#include "../../utils/utils.hpp"
#include "gemm_fp8_lut.hpp"

using namespace deep_gemm;
namespace deep_gemm_fp8_common {

struct MulticastConfig {
    int num_multicast;
    bool is_multicast_on_a;

    MulticastConfig(const int& num_multicast, const bool& is_multicast_on_a)
        : num_multicast(num_multicast), is_multicast_on_a(is_multicast_on_a) {
        DG_HOST_ASSERT(1 <= num_multicast and num_multicast <= 2);
    }
};

struct SharedMemoryConfig {
    int smem_size;
    int swizzle_a_mode;
    int swizzle_b_mode;
    int swizzle_cd_mode;
};

struct ThreadConfig {
    int num_threads;

    int num_tma_threads;
    int num_math_threads;

    int num_non_epilogue_threads;
    int num_epilogue_threads;

    static ThreadConfig sm90(const int& num_tma_threads, const int& num_math_threads) {
        auto config = ThreadConfig();
        config.num_threads = num_tma_threads + num_math_threads;
        config.num_tma_threads = num_tma_threads;
        config.num_math_threads = num_math_threads;
        return config;
    }

    static ThreadConfig sm100(const int& num_non_epilogue_threads, const int& num_epilogue_threads) {
        auto config = ThreadConfig();
        config.num_threads = num_non_epilogue_threads + num_epilogue_threads;
        config.num_non_epilogue_threads = num_non_epilogue_threads;
        config.num_epilogue_threads = num_epilogue_threads;
        return config;
    }
};

static bool is_multicast_legal(const int& shape_dim, const int& block_dim, const int& num_multicast, const int& num_sms,
                               const bool& require_divisible) {
    const bool& divisible = ceil_div(shape_dim, block_dim) % num_multicast == 0 or not require_divisible;
    return divisible and num_sms % num_multicast == 0;
}

template <typename size_type_t>
static int get_swizzle_mode(const int& block_size, const size_type_t& elem_size) {
    // `> 0` means interleaving
    // 16B actually means non-swizzling (but interleaving)
    for (const int& mode : {128, 64, 32, 16}) {
        if ((block_size * static_cast<int>(elem_size)) % mode == 0)
            return mode;
    }
    DG_HOST_UNREACHABLE("Unreachable");
}

std::tuple<int, int, int> get_smem_config(int num_stages, int k, int block_m, int block_n, int block_k = 128) {
    // Try swizzle first, as it does not waste shared memory
    int swizzle_mode = 128;
    int block_n_padding = 0;
    int smem_d = block_m * (block_n + block_n_padding);
    int smem_a_per_stage = block_m * block_k;
    int smem_b_per_stage = block_n * block_k;
    int smem_size_d = smem_d * 2;
    int smem_size_a = num_stages * smem_a_per_stage;
    int smem_size_b = num_stages * smem_b_per_stage;
    int smem_scales_a_per_stage = block_m * ceil_div(block_k, 128) * 4;
    int smem_scales_b_per_stage = ceil_div(block_n, 128) * std::max(ceil_div(block_k, 128), 8) * 4;
    // int smem_scales_a_per_stage = block_m * 4;
    // int smem_scales_b = ceil_div(k, block_k) * 4;
    int smem_size = std::max(smem_size_d, smem_size_a + smem_size_b);
    smem_size += ceil_div(num_stages * smem_scales_a_per_stage, 128) * 128;
    smem_size += ceil_div(num_stages * smem_scales_b_per_stage, 128) * 128;
    // Swizzle and padding are not compatible
    assert((swizzle_mode > 0) + (block_n_padding > 0) <= 1);
    return std::make_tuple(smem_size, swizzle_mode, block_n_padding);
}

int get_smem_occ(int block_m, int block_n, int block_k, int num_stages) {
    if (block_m == 0) { // Assuming 0 represents None in this context
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

using ConfigResult = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
ConfigResult get_best_configs_dense(int m, int n, int k, int num_groups, int num_sms) {
    std::vector<int> block_ms = {256, 192, 128, 64};
    std::vector<int> block_ns = {256, 128, 64};

    auto fix_wave_saturate = [num_sms](int x) -> int {
        return x == 0 ? num_sms : x;
    };

    auto get_num_waves = [m, n, num_groups, num_sms](int bm, int bn) -> int {
        if (bm == 0)
            return 0;
        return ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms);
    };

    auto get_last_wave_util = [m, n, num_groups, num_sms, fix_wave_saturate](int bm, int bn) -> int {
        return fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms);
    };

    auto get_block_utils = [](int dim, int bm) -> double {
        if (bm == 0)
            return 0.0;
        if (dim % bm != 0) {
            return ((dim / double(bm)) / ((dim + bm - 1) / bm));
        } else {
            return 1.0;
        }
    };

    int best_block_m = 0, best_block_n = 0;

    for (int block_m : block_ms) {
        for (int block_n : block_ns) {
            // Filter condition: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 1) and not
            // (block_m == 128 and bn == 128))
            if (!((block_m <= 128 || block_n <= 128) && (block_n != n && n >= 1) &&
                  !(block_m == 128 && block_n == 128))) {
                continue;
            }

            bool success = false;
            int num_waves = get_num_waves(block_m, block_n);
            int best_num_waves = get_num_waves(best_block_m, best_block_n);
            // double num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n);
            // double best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n,
            // best_block_n); int num_occ = get_smem_occ(block_m, block_n, 128, 2); int best_num_occ
            // = get_smem_occ(best_block_m, best_block_n, 128, 2);

            if (best_block_m == 0 || best_block_n == 0) {
                success = true;
            } else if (num_waves < best_num_waves) {
                success = true;
            } else if (num_waves == best_num_waves) {
                int util = get_last_wave_util(block_m, block_n);
                int best_util = get_last_wave_util(best_block_m, best_block_n);
                success = util > best_util;
                if (util == best_util) {
                    success |= (block_m == best_block_m && block_n < best_block_n);
                    success |= (block_n == best_block_n && block_m < best_block_m);
                    success |= (block_m != best_block_m && block_n > best_block_n);
                }
            }

            if (success) {
                best_block_m = block_m;
                best_block_n = block_n;
            }
        }
    }

    if (m >= 2048 && n >= 2048 && k >= 2048) {
        best_block_m = 192;
        best_block_n = 256;
    }

    int best_num_stages = 0;
    std::tuple<int, int, int> best_smem_config;
    int ppu_capacity = 262144;
    int block_k = 128;

    std::vector<int> stage_candidates;
    for (int s : {4, 3, 2}) {
        if (s <= k / block_k) {
            stage_candidates.push_back(s);
        }
    }

    if (stage_candidates.empty() || (128 % best_block_n != 0 && 128 / gcd(128, best_block_n) <= 4)) {
        stage_candidates = {4, 3, 2};
    }

    int best_occ = 0;
    for (int num_stages : stage_candidates) {
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k);
        if (std::get<0>(best_smem_config) < ppu_capacity) {
            // int occ = ppu_capacity / std::get<0>(best_smem_config);
            best_num_stages = num_stages;
            break;
        }
    }

    // assert best_num_stages is not None
    assert(std::get<0>(best_smem_config) != 0); // Check that best_smem_config is not null
    assert(best_num_stages != 0);
    int num_waves_final = get_num_waves(best_block_m, best_block_n);
    int num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves_final);
    assert(num_min_sms <= num_sms);
    int warp_m = best_block_m / 4;
    int warp_n = best_block_n / 4;

    if (best_block_m == 64 && best_block_n == 256) {
        warp_m = 32;
        warp_n = 32;
    }

    ConfigResult result = std::make_tuple(num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n,
                                          best_num_stages, best_smem_config);
    return result;
}

static ConfigResult get_best_configs(int m, int n, int k, int num_groups, int num_sms,
                                     bool is_grouped_contiguous = false, bool is_grouped_masked = false,
                                     int max_block_n = 256) {
    auto lut_result = deep_gemm_fp8_lut::get_best_configs_from_lut(
        m, n, k, num_groups, is_grouped_contiguous, is_grouped_masked);
    if (lut_result.has_value()) {
        int best_block_m, best_block_n, best_block_k, best_warp_m, best_warp_n, best_stages;
        std::tie(best_block_m, best_block_n, best_block_k,
                 best_warp_m, best_warp_n, best_stages) = lut_result.value();
        auto best_smem_config = get_smem_config(best_stages, k, best_block_m, best_block_n, best_block_k);
        return std::make_tuple(num_sms, best_block_m, best_block_n, best_block_k,
                               best_warp_m, best_warp_n, best_stages, best_smem_config);
    }

    if (num_groups == 1 && is_grouped_contiguous == false && is_grouped_masked == false) {
        auto result = get_best_configs_dense(m, n, k, num_groups, num_sms);
        return result;
    }
    std::vector<int> block_ms;
    if (!is_grouped_contiguous) {
        block_ms = (k > 384) ? std::vector<int>{256, 192, 128, 64, 32, 16} : std::vector<int>{64, 32, 16};
    } else {
        block_ms = {get_m_alignment_for_contiguous_layout()};
    }
    assert(max_block_n > 0 && (max_block_n & (max_block_n - 1)) == 0);
    int bit_length = 32 - __builtin_clz(static_cast<unsigned>(max_block_n));
    std::vector<int> block_ns;
    if (k > 384) {
        for (int exp = bit_length - 1; exp > 4; --exp) {
            block_ns.push_back(1 << exp);
        }
    } else {
        for (int exp = bit_length - 2; exp > 4; --exp) {
            block_ns.push_back(1 << exp);
        }
    }
    auto fix_wave_saturate = [num_sms](int x) -> int {
        return x == 0 ? num_sms : x;
    };

    auto get_num_waves = [m, n, num_groups, num_sms](int bm, int bn) -> int {
        if (bm == 0)
            return 0;
        return ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms);
    };

    auto get_last_wave_util = [m, n, num_groups, num_sms, fix_wave_saturate](int bm, int bn) -> int {
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

    int best_block_m = 0, best_block_n = 0;
    // int best_num_occ = 1;

    for (int block_m : block_ms) {
        for (int block_n : block_ns) {
            // Filter condition: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 32))
            if (!((block_m <= 128 || block_n <= 128) && (block_n != n && n >= 32)) ||
                (block_m == 16 and block_n <= 32)) {
                continue;
            }

            bool success = false;
            int num_waves = get_num_waves(block_m, block_n);
            int best_num_waves = get_num_waves(best_block_m, best_block_n);
            double num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n);
            double best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n);
            int num_occ = get_smem_occ(block_m, block_n, 128, 2);
            int best_num_occ_current = get_smem_occ(best_block_m, best_block_n, 128, 2);

            if (best_block_m == 0 || best_block_n == 0) {
                success = true;
            } else if (m < 64 || n < 512) {
                double occ_wave = (double)num_waves / num_occ;
                double best_occ_wave = (double)best_num_waves / best_num_occ_current;
                double ai_util = (double)(block_m * block_n) / (block_m + block_n);
                double best_ai_util = (double)(best_block_m * best_block_n) / (best_block_m + best_block_n);
                bool valid_occ = ((double)num_occ / best_num_occ_current) >= 1;
                bool valid_wave = (occ_wave / best_occ_wave) <= 1;
                bool valid_util = (num_utils / best_num_utils) >= 1;
                bool valid_ai = (ai_util / best_ai_util) >= 1;
                int score = (valid_wave ? 1 : 0) + (valid_util ? 1 : 0) + (valid_occ ? 1 : 0) + (valid_ai ? 1 : 0);
                success = score >= 3;
            } else if (num_waves < best_num_waves) {
                success = true;
            } else if (num_waves == best_num_waves) {
                int util = get_last_wave_util(block_m, block_n);
                int best_util = get_last_wave_util(best_block_m, best_block_n);
                success = util > best_util;
                if (util == best_util) {
                    success |= (block_m == best_block_m && block_n < best_block_n);
                    success |= (block_n == best_block_n && block_m < best_block_m);
                    success |= (block_m != best_block_m && block_n > best_block_n);
                }
            }

            if (success) {
                best_block_m = block_m;
                best_block_n = block_n;
            }
        }
    }

    if (!is_grouped_contiguous) {
        if (m > 256 && n >= 256) {
            best_block_m = 192;
            best_block_n = 256;
        }
        if (m >= 128 && k <= 512) {
            best_block_m = 64;
            best_block_n = 128;
        }
        if ((best_block_m - 10 <= m && m <= best_block_m) && best_block_m == 16) {
            best_block_m = best_block_m * 2;
        }
        if ((best_block_m - 10 <= m && m <= best_block_m) && (best_block_m == 32) && k > 384) {
            best_block_m = best_block_m * 2;
        }
    }

    int num_waves_final = get_num_waves(best_block_m, best_block_n);
    int num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves_final);

    int warp_m = best_block_m / 2;
    int warp_n = best_block_n / 2;

    if (best_block_m == 32 && best_block_n >= 64) {
        warp_m = best_block_m / 2;
        // best_block_n = best_block_n == 128 ? 64 : best_block_n;
        warp_n = best_block_n <= 128 ? best_block_n / 2 : best_block_n / 4;
    } else if (best_block_n == 32 && n <= 128 && best_block_m >= 64) {
        warp_m = best_block_m / 4;
        warp_n = 32;
    } else if (best_block_m == 192 && best_block_n == 128) {
        best_block_n = 256;
        warp_m = 48;
        warp_n = 64;
    } else if ((best_block_m == 128 || best_block_m == 192 || best_block_m == 256) && best_block_n >= 32) {
        warp_m = best_block_m / 4;
        warp_n = best_block_n <= 128 ? best_block_n / 2 : best_block_n / 4;
    } else if (best_block_m == 16) {
        warp_m = 16;
        best_block_n = n < 512 ? 64 : best_block_n;
        warp_n = best_block_n <= 128 ? best_block_n / 4 : best_block_n / 8;
    } else if (best_block_n == 128 || best_block_n == 256) {
        warp_m = best_block_m != 32 ? best_block_m / 2 : best_block_m;
        warp_n = best_block_n / 4;
    }

    int best_num_stages = 0;
    std::tuple<int, int, int> best_smem_config;
    int ppu_capacity = 262144;
    int block_k = 128;

    std::vector<int> stage_candidates;
    for (int s : {7, 6, 5, 4, 3, 2}) {
        if (s <= k / block_k) {
            stage_candidates.push_back(s);
        }
    }

    if (stage_candidates.empty() || (128 % best_block_n != 0 && 128 / gcd(128, best_block_n) <= 4) ||
        best_block_m == 16 || best_block_m == 32) {
        stage_candidates = {3, 2};
    }
    if (best_block_m >= 128 && best_block_n >= 128) {
        stage_candidates = {4};
    }
    if ((best_block_m == 128 || best_block_m == 64) && best_block_n == 128 && k > 384) {
        stage_candidates = {3};
    }

    int best_occ = 0;
    for (int num_stages : stage_candidates) {
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k);
        if (std::get<0>(best_smem_config) < ppu_capacity) {
            int occ = ppu_capacity / std::get<0>(best_smem_config);
            if (k < 512 || (best_block_m > 64 && best_block_n >= 64) && occ >= best_occ) {
                // compute block and too small-k use higer occ rather than large stage
                best_num_stages = num_stages;
                best_occ = occ;
            } else {
                best_num_stages = num_stages;
                break;
            }
        }
    }

    ConfigResult result = std::make_tuple(num_min_sms, best_block_m, best_block_n, block_k, warp_m, warp_n,
                                          best_num_stages, best_smem_config);
    return result;
}
} // namespace deep_gemm_fp8_common
