#pragma once
#include <algorithm>
#include <cassert>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"
#include "../../utils/utils.hpp"
#include <deep_gemm/common/utils_rtc.cuh>
using namespace deep_gemm;
namespace deep_gemm_fp4_common {

// ============================================================================
// MXFP4 Scales Layout Check & Preprocess Utilities
// ============================================================================

/// Check whether the MXFP4 scale tensor has the correct layout (uint16, M/N-major)
// keep in sync with deep_gemm/jit_kernels/gemm_fp4.py
// NOTE: Only the canonical packed layouts are accepted: an M/N-major scale as produced by
// `preprocess_mxfp4_scales`, or its degenerate form when a dim has extent 1. Tensors
// whose strides carry padding (for example a row slice of a larger buffer) are
// rejected on purpose, even though they may be M/N-major.
inline bool check_mxfp4_scales_layout(const torch::Tensor& scale) {
    if (scale.dtype() != torch::kUInt16) return false;

    // The canonical layout is always M/N-major: stride_MN == 1, stride_K == MN, stride_G == MN * K.
    int64_t target_stride[3] = {};
    if (scale.dim() == 2) {
        const auto mn = scale.size(0);
        target_stride[0] = 1;
        target_stride[1] = mn;
    } else if (scale.dim() == 3) {
        const auto mn = scale.size(1);
        const auto k = scale.size(2);
        target_stride[0] = mn * k;
        target_stride[1] = 1;
        target_stride[2] = mn;
    } else {
        return false;
    }

    // The stride of an extent-1 dim never participates in the address computation, so it is
    // **unobservable** and must NOT be compared: for example a (G, 1, K) scale is byte-identical
    // whether its MN stride is K or 1, which means that its stride might be (K, 1, 1) or (K, K, 1).
    for (int64_t i = 0; i < scale.dim(); ++i) {
        if (scale.size(i) > 1 && scale.stride(i) != target_stride[i]) return false;
    }
    return true;
}

/// Preprocess mxfp4 scales: uint8 -> pad if odd -> view as uint16 -> transpose to M/N-major
inline torch::Tensor preprocess_mxfp4_scales(const torch::Tensor& scale) {
    torch::Tensor s = scale;
    // make scale contiguous for SGLang warmup.
    if (!s.is_contiguous()) s = s.contiguous();
    if (s.dtype() == torch::kUInt16) s = s.view(torch::kUInt8);
    DG_HOST_ASSERT(s.dtype() == torch::kUInt8);
    DG_HOST_ASSERT(s.dim() == 2 || s.dim() == 3);

    if (s.size(-1) % 2 != 0) {
        s = at::constant_pad_nd(s, {0, 1}, 0);
    }
    DG_HOST_ASSERT(s.size(-1) % 2 == 0);
    s = s.view(torch::kUInt16);
    if (s.dim() == 2) {
        return s.t().contiguous().t();
    } else {
        return s.permute({0, 2, 1}).contiguous().permute({0, 2, 1});
    }
}

/// Post-preprocess for forward compatibility (grouped tensor already uint16 but not transposed)
inline torch::Tensor post_preprocess_mxfp4_scales(const torch::Tensor& scale) {
    if (scale.dtype() == torch::kUInt16 && scale.is_contiguous() && scale.dim() == 3) {
        return scale.permute({0, 2, 1}).contiguous().permute({0, 2, 1});
    }
    return scale;
}

/// Interleave a fused-MoE gemm1 weight so that the epilogue reads gate/up (W1/W3) pairs from
/// adjacent N positions, and return the matching preprocessed scale.
// keep in sync with deep_gemm/jit_kernels/gemm_fp4.py
// NOTE: must be called *before* `preprocess_mxfp4_scales`, never after. Both inputs are required to
// still be raw contiguous uint8 here, which is exactly what `preprocess_mxfp4_scales` takes away:
// it packs the scale into uint16 and re-strides it to N-major.
inline std::pair<torch::Tensor, torch::Tensor> preprocess_mxfp4_weight_for_act_and_quant_fusing(
        const torch::Tensor& weight, const torch::Tensor& weight_scale) {
    DG_HOST_ASSERT(weight.dim() == 3);
    DG_HOST_ASSERT(weight_scale.dim() == 3);
    DG_HOST_ASSERT(weight.dtype() == torch::kUInt8);
    DG_HOST_ASSERT(weight_scale.dtype() == torch::kUInt8);
    DG_HOST_ASSERT(weight.is_contiguous());
    DG_HOST_ASSERT(weight_scale.is_contiguous());

    // do interleaving to make up and gate be adjacent.
    const int64_t num_groups = weight.size(0), n = weight.size(1), k = weight.size(2);
    DG_HOST_ASSERT(n % 2 == 0);  // silu_and_mul splits N in half
    const int64_t half_n = n / 2;
    const auto gate = weight.slice(1, 0, half_n);
    const auto up = weight.slice(1, half_n, n);
    // `at::stack` writes a fresh contiguous tensor, so the following `view` is always legal.
    const auto weight_out = at::stack({gate, up}, 2).view({num_groups, n, k});

    const int64_t sfn = weight_scale.size(1), sfk = weight_scale.size(2);
    // `weight` is uint8, so one byte holds two fp4 values: sfk == ceil_div(2 * k, 32).
    DG_HOST_ASSERT(weight_scale.size(0) == num_groups && sfn == n && sfk == ceil_div<int64_t>(k, 16));
    const auto gate_scale = weight_scale.slice(1, 0, half_n);
    const auto up_scale = weight_scale.slice(1, half_n, n);
    const auto weight_scale_out = at::stack({gate_scale, up_scale}, 2).view({num_groups, sfn, sfk});
    return {weight_out, preprocess_mxfp4_scales(weight_scale_out)};
}

// ============================================================================

using TileConfig = std::map<std::tuple<int, int, int>, std::tuple<int, int, int>>;

inline const TileConfig& get_tile_config_normal() {
    static const TileConfig config = {
        {{16, 64, 128}, {16, 32, 3}},
        {{16, 64, 256}, {16, 16, 4}},
        {{32, 128, 128}, {32, 64, 2}},
        {{32, 64, 128}, {32, 32, 2}},
        {{64, 64, 128}, {32, 64, 3}},
        {{64, 128, 128}, {32, 64, 2}},
        {{128, 128, 128}, {32, 64, 2}},
        {{128, 256, 64}, {64, 64, 3}},
    };
    return config;
}

inline const TileConfig& get_tile_config_smallK() {
    static const TileConfig config = {
        {{16, 64, 128}, {16, 16, 2}},
        {{32, 64, 128}, {16, 32, 2}},
        {{32, 128, 128}, {16, 64, 2}},
        {{64, 128, 128}, {32, 64, 2}},
        {{64, 256, 128}, {32, 64, 2}},
    };
    return config;
}

std::pair<int, int> get_sf_per_stage_size(int block_mn, int block_k) {
    int bpp = 2;                           // bpp=2 means sizeof(uint16)
    int smem_sf_k = ceil_div(block_k, 32); // uint16
    int base_smem_sf_size = block_mn * smem_sf_k * bpp;

    if (block_mn <= 64) {
        return {base_smem_sf_size, 0};
    } else {
        int aiu_num_on_m = ceil_div(block_mn, 64);
        int sf_padding_size = 16 * aiu_num_on_m * bpp; // 16 means padding 16 rows for bank conflicts
        int invalid_sf_element_size = 16 * bpp;        // -16 means cute::cosize will only reserve spaces until the last valid element
        return {base_smem_sf_size + sf_padding_size, invalid_sf_element_size};
    }
}

std::tuple<int, int, int> get_smem_config_fp4(int num_stages, int block_m, int block_n, int warp_m, int warp_n,
                                              int block_k, int shape_n, bool has_bias,
                                              bool enable_act_and_quant_fusing, int bpp = 1) {
    // Try swizzle first, as it does not waste shared memory
    int swizzle_mode = 128;
    int block_n_padding = 0;
    int warp_on_m = block_m / warp_m;

    int smem_a_per_stage = block_m * block_k;
    int smem_b_per_stage = block_n * block_k;

    auto [smem_sfa_per_stage, invalid_sfa_element_size] = get_sf_per_stage_size(block_m, block_k);
    auto [smem_sfb_per_stage, invalid_sfb_element_size] = get_sf_per_stage_size(block_n, block_k);

    int smem_size_d = 0;  // default: using CollectiveEpilogueNoTsm
    int smem_size_a = num_stages * smem_a_per_stage * bpp;
    int smem_size_b = num_stages * smem_b_per_stage * bpp;
    int smem_size_sfa = num_stages * smem_sfa_per_stage * bpp - invalid_sfa_element_size;
    int smem_size_sfb = num_stages * smem_sfb_per_stage * bpp - invalid_sfb_element_size;

    // re-calculate epilogue smem size to fit into different epilogue type
    if (shape_n % 2 != 0 || has_bias) {
        // using CollectiveEpilogueWithTsm
        smem_size_d = warp_on_m * 16 * block_n * static_cast<int>(sizeof(float));
    } else if (enable_act_and_quant_fusing) {
        // using CollectiveEpilogueSiluAndMulPostQuant
        smem_size_d = (block_m * ((block_n / 2) + 4)) * static_cast<int>(sizeof(float)); // 4 means padding
    }

    int smem_size = std::max(smem_size_d, smem_size_a + smem_size_b + smem_size_sfa + smem_size_sfb);

    // Swizzle and padding are not compatible
    DG_HOST_ASSERT((swizzle_mode > 0) + (block_n_padding > 0) <= 1);
    return std::make_tuple(smem_size, swizzle_mode, block_n_padding);
}

int get_fused_smem_size_fp4(int num_stages, int block_m, int block_n, int block_k,
                            bool enable_silu_and_mul_quant_fusing) {
    const auto round_up_128 = [](int value) { return ceil_div(value, 128) * 128; };
    const auto [sfa_per_stage, invalid_sfa_size] = get_sf_per_stage_size(block_m, block_k);
    const auto [sfb_per_stage, invalid_sfb_size] = get_sf_per_stage_size(block_n, block_k);
    // fp4 is stored as uint8, so the element size of A/B is 1 byte
    const int smem_a_size = round_up_128(num_stages * block_m * block_k);
    const int smem_b_size = num_stages * block_n * block_k;
    const int smem_sfa_size = num_stages * sfa_per_stage - invalid_sfa_size;
    const int smem_sfb_size = num_stages * sfb_per_stage - invalid_sfb_size;
    // `Fp4FusedMoeEpilogue::kSmemSize`, which is 0 for the default (bf16) epilogue
    const int epilogue_smem_size = enable_silu_and_mul_quant_fusing
        ? round_up_128(block_m * (block_n / 2 + 4) * static_cast<int>(sizeof(float)))
        : 0;
    return std::max(smem_a_size + smem_b_size, epilogue_smem_size) + smem_sfa_size + smem_sfb_size;
}

int get_smem_occ(int block_m, int block_n, int block_k, int num_stages, int warp_m, int warp_n, int shape_n, bool has_bias, bool enable_act_and_quant_fusing) {
    if (block_m == 0) {
        return 0;
    }

    // use static suppose.
    const int ppu_capacity = 262144;
    int smem_size = std::get<0>(get_smem_config_fp4(num_stages, block_m, block_n, warp_m, warp_n, block_k, shape_n, has_bias, enable_act_and_quant_fusing));
    return ppu_capacity / smem_size;
}

using ConfigResult = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

ConfigResult get_best_configs_dense_ppu1v5(int m, int n, int k, int num_groups, int num_sms, bool has_bias, bool enable_act_and_quant_fusing) {
    std::vector<int> block_ms = {256, 128, 64, 32, 16};
    std::vector<int> block_ns = {256, 128, 64, 32, 16};

    auto fix_wave_saturate = [num_sms](int x) -> int { return x == 0 ? num_sms : x; };

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

    auto get_block_ai = [](int block_m, int block_n) -> double {
        return double(block_m * block_n) / (block_m + block_n);
    };

    auto get_warp_mn_dense = [n, k](int block_m_, int block_n_) -> std::tuple<int, int, int, int> {
        if (block_m_ == 0 || block_n_ == 0) {
            return {0, 0, 0, 0};
        }

        int warp_m_ = block_m_ / 2;
        int warp_n_ = block_n_ / 2;

        if (block_m_ >= 128 && block_n_ == 256) {
            warp_m_ = block_m_ / 4;
            warp_n_ = block_n_ / 4;
        } else if (block_m_ == 32 && block_n_ >= 64) {
            warp_m_ = 32;
            warp_n_ = block_n_ / 4;
        } else if (block_n_ == 32 && n <= 128 && block_m_ >= 64) {
            warp_m_ = block_m_ / 4;
            warp_n_ = 32;
        } else if (block_m_ == 128 || block_m_ == 256 || (block_m_ == 192 && block_n_ >= 32)) {
            warp_m_ = block_m_ / 4;
            warp_n_ = (block_n_ != 32) ? block_n_ / 2 : block_n_;
        } else if (block_m_ == 16) {
            warp_m_ = 16;
            block_n_ = (n < 512) ? 64 : block_n_;
            warp_n_ = 16;
        } else if (block_n_ == 128 || block_n_ == 256) {
            warp_m_ = (block_m_ != 32) ? block_m_ / 2 : block_m_;
            warp_n_ = block_n_ / 4;
        } else if (block_n_ == 16) {
            warp_n_ = 16;
        }

        return {block_m_, block_n_, warp_m_, warp_n_};
    };

    int best_block_m = 0, best_block_n = 0;
    for (int block_m : block_ms) {
        std::vector<int> block_ns_after_filter;
        bool use_loose_filter = (m >= 128 && k >= 2048) || m >= 256;
        for (int bn : block_ns) {
            if (use_loose_filter) {
                if (bn != n && n >= 1) {
                    block_ns_after_filter.push_back(bn);
                }
            } else {
                if ((block_m <= 128 || bn <= 128) && (bn != n && n >= 1) && !(block_m == 16 && bn == 32)) {
                    block_ns_after_filter.push_back(bn);
                }
            }
        }

        for (int block_n : block_ns_after_filter) {
            bool success = false;
            int num_waves = get_num_waves(block_m, block_n);
            int best_num_waves = get_num_waves(best_block_m, best_block_n);
            double num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n);
            double best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n);

            auto [tmp_block_m, tmp_block_n, tmp_warp_m, tmp_warp_n] = get_warp_mn_dense(block_m, block_n);
            auto [tmp_best_block_m, tmp_best_block_n, tmp_best_warp_m, tmp_best_warp_n] = get_warp_mn_dense(best_block_m, best_block_n);
            int num_occ = get_smem_occ(tmp_block_m, tmp_block_n, 128, 2, tmp_warp_m, tmp_warp_n, n, has_bias, enable_act_and_quant_fusing);
            int best_num_occ = get_smem_occ(tmp_best_block_m, tmp_best_block_n, 128, 2, tmp_best_warp_m, tmp_best_warp_n, n, has_bias, enable_act_and_quant_fusing);

            if (best_block_m == 0 || best_block_n == 0) {
                success = true;
            } else if (m <= 256 || n < 512) {
                double occ_wave = double(num_waves) / num_occ;
                double best_occ_wave = double(best_num_waves) / best_num_occ;
                double ai_util = get_block_ai(block_m, block_n);
                double best_ai_util = get_block_ai(best_block_m, best_block_n);

                bool valid_occ = (double(num_occ) / best_num_occ) >= 1.0;
                bool valid_wave = (occ_wave / best_occ_wave) <= 1.0;
                bool valid_util = (num_utils / best_num_utils) >= 1.0;
                bool valid_ai = (ai_util / best_ai_util) >= 1.0;

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

    // small m hbm bound or latency bound, wave is not useful, for better occ for 810e hbm bound, use smallest blockN for m16
    if (m < 20 && n < 512) {
        best_block_m = 16;
        best_block_n = 64;
    }

    DG_HOST_ASSERT(best_block_m != 0 && best_block_n != 0);

    // Always pick the longest one
    int block_k = 128;
    if (k <= 256) {
        block_k = 64;
    }
    if (k >= 4096 && ((best_block_m <= 32 && best_block_n <= 64) || (best_block_m == 64 && best_block_n == 128))) {
        block_k = 256;
    }

    int num_waves = get_num_waves(best_block_m, best_block_n);
    int num_min_sms = num_sms;
    DG_HOST_ASSERT(num_min_sms <= num_sms);

    auto [bm_out, bn_out, warp_m, warp_n] = get_warp_mn_dense(best_block_m, best_block_n);

    std::vector<int> stage_candidates;
    for (int s : {8, 7, 6, 5, 4, 3, 2}) {
        if (s <= k / block_k) {
            stage_candidates.push_back(s);
        }
    }
    if (stage_candidates.empty()) {
        stage_candidates = {2};
    }

    int best_num_stages = 0;
    std::tuple<int, int, int> best_smem_config;
    const int ppu_capacity = 262144;

    for (int num_stages : stage_candidates) {
        best_smem_config = get_smem_config_fp4(num_stages, bm_out, bn_out, warp_m, warp_n, block_k, n, has_bias, enable_act_and_quant_fusing);
        if (std::get<0>(best_smem_config) <= ppu_capacity) {
            best_num_stages = num_stages;
            break;
        }
    }

    DG_HOST_ASSERT(std::get<0>(best_smem_config) != 0);
    DG_HOST_ASSERT(best_num_stages != 0);

    return std::make_tuple(std::min(num_min_sms, num_sms), bm_out, bn_out, block_k, warp_m, warp_n, best_num_stages,
                           best_smem_config);
}

ConfigResult get_best_configs(int total_m, int m, int n, int k, int num_groups, int num_sms,
                              bool has_bias,
                              bool enable_act_and_quant_fusing,
                              GemmType gemm_type = GemmType::DenseGemm,
                              int max_block_n = 256, int min_block_n = 32) {
    // C++ layer does not perform device checking; is_ppu1v5_device() assert skipped
    (void)total_m;

    if (gemm_type == GemmType::DenseGemm) {
        return get_best_configs_dense_ppu1v5(m, n, k, num_groups, num_sms, has_bias, enable_act_and_quant_fusing);
    }

    std::vector<int> block_ms = (k > 768) ? std::vector<int>{256, 128, 64, 32, 16} : std::vector<int>{128, 64, 32, 16};

    DG_HOST_ASSERT(max_block_n > 0 && (max_block_n & (max_block_n - 1)) == 0);
    DG_HOST_ASSERT(min_block_n > 0 && (min_block_n & (min_block_n - 1)) == 0);
    // SiluAndMulPostQuant fusing only supports block_n >= 64
    min_block_n = (enable_act_and_quant_fusing && min_block_n < 64) ? 64 : min_block_n;
    int bit_length = 32 - __builtin_clz(static_cast<unsigned>(max_block_n));
    int bit_length_min = 32 - __builtin_clz(static_cast<unsigned>(min_block_n)) - 1; // exponent of min_block_n
    // `exp > bit_length_min - 1` mirrors python's right-open range stop, so min_block_n is included
    std::vector<int> block_ns;
    if (k >= 384) {
        for (int exp = bit_length - 1; exp > (bit_length_min - 1); --exp) {
            block_ns.push_back(1 << exp);
        }
    } else {
        for (int exp = bit_length - 2; exp > (bit_length_min - 1); --exp) {
            block_ns.push_back(1 << exp);
        }
    }

    auto fix_wave_saturate = [num_sms](int x) -> int { return x == 0 ? num_sms : x; };

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

    auto get_block_ai = [](int block_m, int block_n) -> double {
        return double(block_m * block_n) / (block_m + block_n);
    };

    auto get_warp_mn_grouped = [n, k](int block_m_, int block_n_) -> std::tuple<int, int, int, int> {
        if (block_m_ == 0 || block_n_ == 0) {
            return {0, 0, 0, 0};
        }

        int warp_m_ = block_m_ / 2;
        int warp_n_ = block_n_ / 2;

        if (block_m_ > 128 && block_n_ == 256) {
            warp_m_ = block_m_ / 4;
            warp_n_ = block_n_ / 4;
        } else if (block_m_ == 64 && block_n_ >= 128) {
            warp_m_ = (k < 256) ? 64 : 32;
            warp_n_ = 64;
        } else if (block_m_ == 32 && block_n_ >= 64) {
            warp_m_ = 32;
            warp_n_ = (block_n_ <= 128) ? block_n_ / 2 : block_n_ / 4;
        } else if (block_n_ == 32 && n <= 128 && block_m_ >= 64) {
            warp_m_ = block_m_ / 4;
            warp_n_ = 32;
        } else if (block_m_ == 128 || (block_m_ == 256 && block_n_ >= 32)) {
            warp_m_ = 64;
            warp_n_ = (block_n_ <= 128) ? block_n_ / 2 : block_n_ / 4;
        } else if (block_m_ == 16) {
            warp_m_ = 16;
            block_n_ = (n < 512) ? 64 : block_n_;
            warp_n_ = 16;
        } else if (block_n_ == 128 || block_n_ == 256) {
            warp_m_ = (block_m_ != 32) ? block_m_ / 2 : block_m_;
            warp_n_ = 64;
        } else if (block_n_ == 16) {
            warp_n_ = 16;
        }

        return {block_m_, block_n_, warp_m_, warp_n_};
    };

    int best_block_m = 0, best_block_n = 0;
    for (int block_m : block_ms) {
        std::vector<int> block_ns_after_filter;
        bool use_loose_filter = (m >= 96 && k > 1024) || (m >= 256 && k >= 512);
        for (int bn : block_ns) {
            if (use_loose_filter) {
                if ((bn != n && n >= 32) && !(block_m == 16 && bn <= 32)) {
                    block_ns_after_filter.push_back(bn);
                }
            } else {
                if ((block_m <= 128 || bn <= 128) && (bn != n && n >= 32) && !(block_m == 16 && bn <= 32)) {
                    block_ns_after_filter.push_back(bn);
                }
            }
        }

        for (int block_n : block_ns_after_filter) {
            bool success = false;
            int num_waves = get_num_waves(block_m, block_n);
            int best_num_waves = get_num_waves(best_block_m, best_block_n);
            double num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n);
            double best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n);

            auto [tmp_block_m, tmp_block_n, tmp_warp_m, tmp_warp_n] = get_warp_mn_grouped(block_m, block_n);
            auto [tmp_best_block_m, tmp_best_block_n, tmp_best_warp_m, tmp_best_warp_n] = get_warp_mn_grouped(best_block_m, best_block_n);
            int num_occ = get_smem_occ(tmp_block_m, tmp_block_n, 128, 2, tmp_warp_m, tmp_warp_n, n, has_bias, enable_act_and_quant_fusing);
            int best_num_occ = get_smem_occ(tmp_best_block_m, tmp_best_block_n, 128, 2, tmp_best_warp_m, tmp_best_warp_n, n, has_bias, enable_act_and_quant_fusing);

            if (best_block_m == 0 || best_block_n == 0) {
                success = true;
            } else if (m < 512 || n < 512) {
                double occ_wave = num_waves;
                double best_occ_wave = best_num_waves;
                double ai_util = get_block_ai(block_m, block_n);
                double best_ai_util = get_block_ai(best_block_m, best_block_n);

                bool valid_occ = (double(num_occ) / best_num_occ) >= 1.0;
                bool valid_wave = (occ_wave / best_occ_wave) <= 1.0;
                bool valid_util = (num_utils / best_num_utils) >= 1.0;
                bool valid_ai = (ai_util / best_ai_util) >= 1.0;

                int score = (valid_wave ? 1 : 0) + (valid_util ? 1 : 0) + (valid_ai ? 1 : 0);
                success = score >= 2;
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

    if ((best_block_m - 10 <= m && m <= best_block_m) && best_block_m == 16) {
        best_block_m = best_block_m * 2;
    }

    // qwen3-next & deepseek gemm2 need to fix some issues
    if ((best_block_m - 10 <= m && m <= best_block_m) && (best_block_m == 32 || best_block_m == 64) && k >= 192) {
        best_block_m = best_block_m * 2;
    }

    DG_HOST_ASSERT(best_block_m != 0 && best_block_n != 0);

    // Always pick the longest one
    int block_k = 64;
    if (best_block_m <= 32 && best_block_n <= 128) {
        if (best_block_n <= 64 || k <= 128) {
            block_k = 128;
        }
    }
    if (best_block_m == 256 && best_block_n == 256) {
        block_k = 128;
    }
    if (k <= 256) {
        block_k = 128;
        // for deepseek-pro tp8 gemm2
        if (k == 192 && best_block_m == 128 && best_block_n == 128) {
            block_k = 64;
        }
    }

    if (m < 6 && n >= 512) {
        if (m < 2) {
            best_block_m = 16;
        } else {
            best_block_m = 32;
        }
        best_block_n = 64;
        block_k = 128;
    }
    // for deepseek-v4 pro tp16 gemm1 smallN
    if (m < 2 && n < 512) {
        best_block_m = 16;
        best_block_n = 64;
        block_k = 256;
    }
    // for DeepSeek-V4 Pro EP
    if ((n == 6144 && k == 3584) || (n == 7168 && k == 1536)) {
        if (m < 6) {
            if (best_block_m == 32 && best_block_n == 64) {
                best_block_m = 64;
                best_block_n = 64;
                block_k = 128;
                if (k == 1536) {
                    best_block_n = 128;
                }
            }
        } else if ((best_block_m == 32 || best_block_m == 64) && best_block_n == 256) {
            best_block_m = 128;
            best_block_n = 256;
            block_k = 64;
        }
    }

    const TileConfig& tile_config =
        (k < 128 && !(best_block_n >= 128 && best_block_m >= 128)) ? get_tile_config_smallK() : get_tile_config_normal();

    auto it = tile_config.find({best_block_m, best_block_n, block_k});
    if (it != tile_config.end()) {
        auto [warp_m, warp_n, num_stages] = it->second;
        auto best_smem_config = get_smem_config_fp4(num_stages, best_block_m, best_block_n, warp_m, warp_n, block_k, n, has_bias, enable_act_and_quant_fusing);
        return std::make_tuple(num_sms, best_block_m, best_block_n, block_k, warp_m, warp_n, num_stages, best_smem_config);
    }

    // todo: opt this logic
    if (m >= 520 && k <= 768) {
        if (best_block_m == 128 && best_block_n == 256) {
            std::swap(best_block_m, best_block_n);
        }
    }

    int num_waves = get_num_waves(best_block_m, best_block_n);
    int num_min_sms = num_sms;
    DG_HOST_ASSERT(num_min_sms <= num_sms);

    // NOTE: Python rebinds best_block_m/best_block_n from get_warp_mn_grouped() before deciding
    // stage_candidates, so the conditions below must use the post-call values (bm_out/bn_out).
    auto [bm_out, bn_out, warp_m, warp_n] = get_warp_mn_grouped(best_block_m, best_block_n);

    std::vector<int> stage_candidates;
    for (int s : {8, 7, 6, 5, 4, 3, 2}) {
        if (s <= k / block_k) {
            stage_candidates.push_back(s);
        }
    }
    if (stage_candidates.empty()) {
        stage_candidates = {2};
    }
    if (bm_out <= 32) {
        stage_candidates = {3, 2};
    } else {
        stage_candidates = {3};
    }
    if (bm_out >= 128 && bn_out >= 128) {
        stage_candidates = {3, 4};
    }
    if (bm_out == 128 && bn_out == 256) {
        if (k >= 768) {
            stage_candidates = {4};
        } else {
            stage_candidates = {3};
        }
    }

    int best_num_stages = 0;
    std::tuple<int, int, int> best_smem_config;
    const int ppu_capacity = 262144;

    for (int num_stages : stage_candidates) {
        best_smem_config = get_smem_config_fp4(num_stages, bm_out, bn_out, warp_m, warp_n, block_k, n, has_bias, enable_act_and_quant_fusing);
        if (std::get<0>(best_smem_config) <= ppu_capacity) {
            best_num_stages = num_stages;
            break;
        }
    }

    DG_HOST_ASSERT(std::get<0>(best_smem_config) != 0);
    DG_HOST_ASSERT(best_num_stages != 0);

    return std::make_tuple(std::min(num_min_sms, num_sms), bm_out, bn_out, block_k, warp_m, warp_n, best_num_stages,
                           best_smem_config);
}

/// Decide whether to enable the MoE dynamic-tile kernel and which variant to use.
/// Returns (enable_moe_dynamic_tile, dynamic_tile_id), where dynamic_tile_id is the
/// FP4DynamicTileId enumerator name -- "LargeEM", "LargeK", "LargeK_G2", "SmallEM", or
/// "Disabled" when not enabled. Mirrors select_moe_dynamic_tile() in m_grouped_gemm_fp4.py.
///
/// NOTE: the shape lists compare against `k * 2` because `k` here counts fp4 elements while the
/// model shapes are quoted in bf16 elements.
inline std::pair<bool, std::string> select_moe_dynamic_tile(int n, int k, int expected_m, bool has_bias) {
    const auto extra_info = get_extra_info();
    const bool env_use_moe_dynamic_tile = extra_info.at("use_moe_dynamic_tile") != 0;
    if (has_bias || !env_use_moe_dynamic_tile) {
        return {false, "Disabled"};
    }

    // qwen3.8
    static const std::vector<std::pair<int, int>> model_gemm1_shape_list = {{4096, 8192}};
    static const std::vector<std::pair<int, int>> model_gemm2_shape_list = {{8192, 2048}};
    const std::pair<int, int> shape{n, k * 2};
    const auto in_list = [&shape](const std::vector<std::pair<int, int>>& list) {
        return std::find(list.begin(), list.end(), shape) != list.end();
    };

    if (in_list(model_gemm1_shape_list)) {
        // TODO: fix the stack issue for the large-EM case
        if (expected_m < 6 || expected_m > 76) {
            return {false, "Disabled"};
        }
    } else if (in_list(model_gemm2_shape_list)) {
        if (expected_m < 6 || expected_m > 51) {
            return {false, "Disabled"};
        }
    } else {
        if (expected_m < 23) {
            return {false, "Disabled"};
        }
    }

    // select dynamic-tile variant: LargeEM, LargeK, LargeK_G2, SmallEM
    std::string dynamic_tile_id;
    if (k > 2048) {
        dynamic_tile_id = (expected_m > 32) ? "LargeEM" : "LargeK";
    } else if (expected_m > 117) {
        dynamic_tile_id = "LargeEM";
    } else {
        dynamic_tile_id = (k <= 1024) ? "SmallEM" : "LargeK_G2";
    }
    return {true, dynamic_tile_id};
}

/// Device-side launch constants of Fp4DeepGemmDynamicTile that the host cannot derive: the kernel
/// picks its tile shape from a table indexed by kDynamicTileId, so neither the shared-storage size
/// nor the block thread count follow from get_smem_config_fp4() or the usual
/// (block_m/warp_m)*(block_n/warp_n)*32 formula. Values were read back from hgcc by instantiating
/// each variant; the generated kernel pins shared_storage_size with a static_assert so a device-side
/// change fails to compile rather than corrupting memory / launching with the wrong block shape.
struct DynamicTileLaunchConst {
    int shared_storage_size;  // == Fp4DeepGemmDynamicTile::SharedStorageSize
    int block_threads;        // == Fp4DeepGemmDynamicTile::MaxThreadsPerBlock (get_block_shape().x)
};

inline DynamicTileLaunchConst dynamic_tile_launch_const(const std::string& dynamic_tile_id) {
    if (dynamic_tile_id == "LargeEM")   return {209600, 512};
    if (dynamic_tile_id == "LargeK")    return {130976, 256};
    if (dynamic_tile_id == "LargeK_G2") return {130912, 256};
    if (dynamic_tile_id == "SmallEM")   return { 87264, 128};
    DG_HOST_ASSERT(false && "unknown FP4DynamicTileId");
    return {0, 0};
}

} // namespace deep_gemm_fp4_common
