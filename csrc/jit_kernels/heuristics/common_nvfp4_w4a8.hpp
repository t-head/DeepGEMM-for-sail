#pragma once
#include <algorithm>
#include <array>
#include <optional>
#include <tuple>
#include <utility>
#include <torch/torch.h>
#include <deep_gemm/common/nvfp4_gemm_traits.cuh>
#include <deep_gemm/common/types.cuh>
#include "../../utils/utils.hpp"

namespace deep_gemm::nvfp4::detail {
struct Config {
    int num_sms;
    int block_m, block_n, block_k;
    int warp_m, warp_n, warp_k;
    int num_stages, n_expand;
};
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, int>;

static bool is_shared_decode_tile(const Config& c) {
    return is_shared_decode_tile(
        c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k);
}

static const char* cpp_gemm_type_name(GemmType type) {
    switch (type) {
        case GemmType::Normal: return "Normal";
        case GemmType::MGroupedMasked: return "MGroupedMasked";
        case GemmType::MGroupedNoPad: return "MGroupedNoPad";
        case GemmType::MGroupedFused: return "MGroupedFused";
        default: TORCH_CHECK(false, "Unsupported W4A8 GEMM type");
    }
}

// Empirical PPU1.5 schedules for nopad GEMMs with K128 activation scales and
// N128xK128 outer weight scales. The N/K >= 4096 bounds limit the tuning policy;
// they are not input alignment requirements. Other cases use the fallback below.
static std::optional<Config> get_tuned_nopad_config(
        GemmType type, int expected_m, int n, int k, int num_sms, bool aligned_scales) {
    if (type != GemmType::MGroupedNoPad || !aligned_scales || n < 4096 || k < 4096)
        return std::nullopt;

    // Tile: block M/N/K, warp M/N/K, stages. Upper bounds are exclusive and
    // apply to expected rows per expert, not input tokens per rank.
    using Tile = std::tuple<int, int, int, int, int, int, int>;
    static const std::array<std::pair<int, Tile>, 7> table{{
        {12,  {16, 64, 256, 16, 64, 64, 2}},  // Narrow N for small expert batches.
        {16,  {16, 256, 128, 16, 64, 128, 2}},
        {32,  {32, 128, 128, 32, 32, 128, 3}},
        {65,  {64, 128, 128, 64, 32, 128, 3}},
        {129, {128, 128, 128, 64, 32, 128, 3}}, // One shared decode covers the expert.
        {257, {256, 64, 128, 64, 32, 128, 3}},
        {640, {128, 128, 128, 64, 32, 128, 3}}, // Two M warp groups share B decode.
    }};
    Tile tile{256, 64, 128, 64, 32, 128, 3}; // Four M groups; retain a narrow N tile.
    for (const auto& [upper_m, candidate] : table) {
        if (expected_m < upper_m) {
            tile = candidate;
            break;
        }
    }
    Config c{};
    c.num_sms = num_sms;
    std::tie(c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, c.num_stages) = tile;
    if (expected_m < 12 && k % 256 == 0) {
        // Two K128 warps reuse each outer scale across a full K128 partial sum.
        c.warp_n = 32;
        c.warp_k = 128;
    }
    return c;
}

static Config get_best_configs(GemmType type, int expected_m, int n, int k,
                               int a_scale_group, int b_scale_block) {
    const int num_sms = get_num_sms();
    const bool aligned_scales = b_scale_block == 128 && a_scale_group == 128 &&
        k > 512 && k % 128 == 0;
    if (const auto tuned = get_tuned_nopad_config(type, expected_m, n, k, num_sms, aligned_scales))
        return *tuned;

    // Scaled partial sums keep an extra FP32 accumulator. Start with M16;
    // wider aligned-scale tiles can amortize B decode without spilling it.
    Config c{};
    c.num_sms = num_sms;
    c.block_m = c.warp_m = 16;
    const bool wide_k = k >= 1024;
    if (wide_k && expected_m < 12) {
        c.block_n = c.warp_n = c.warp_k = 64;
        c.block_k = 256;
    } else {
        c.block_n = 256;
        c.warp_n = 64;
        c.block_k = c.warp_k = wide_k ? 128 : 64;
    }
    if (b_scale_block > 0 && (expected_m >= 96 ||
            (type == GemmType::MGroupedNoPad && aligned_scales && expected_m >= 32))) {
        c.block_m = c.warp_m = aligned_scales ? 64 : 32;
        c.block_n = 128;
        c.warp_n = 32;
    }
    if (n < c.block_n) c.block_n = n >= 128 ? 128 : 64;
    c.num_stages = k <= 512 || wide_k ? 2 : 3;
    // Stage K128 scales with M32/M64 operands when the kernel supports it.
    if (aligned_scales && (c.block_m == 32 || c.block_m == 64) && c.block_n % 128 == 0 &&
            type != GemmType::MGroupedFused) c.num_stages = 3;
    return c;
}

static Config config(GemmType type, int expected_m, int n, int k,
                     const std::optional<ConfigTuple>& requested = std::nullopt,
                     int a_scale_group = 0, int b_scale_block = 0) {
    TORCH_CHECK(b_scale_block == 0 || b_scale_block == 128,
                "rhs_scale_block_size must be 0 (per channel) or 128 (128x128 blocks)");
    Config c{};
    if (requested) {
        std::tie(c.num_sms, c.block_m, c.block_n, c.block_k,
                 c.warp_m, c.warp_n, c.warp_k, c.num_stages) = *requested;
    } else {
        c = get_best_configs(type, expected_m, n, k, a_scale_group, b_scale_block);
    }
    const bool shared_decode = a_scale_group == 128 && b_scale_block == 128 &&
        type != GemmType::MGroupedFused && k % 128 == 0 &&
        is_shared_decode_tile(c);
    TORCH_CHECK(c.num_sms > 0 && c.num_sms <= get_num_sms() &&
                (c.block_m == 16 || c.block_m == 32 || c.block_m == 64 || c.block_m == 128 ||
                 (c.block_m == 256 && shared_decode)) &&
                (c.warp_m == c.block_m || shared_decode) && (c.warp_n == 32 || c.warp_n == 64) &&
                c.block_n >= 64 && c.block_n % 64 == 0 && c.block_k >= 64 && c.block_k % 32 == 0 &&
                c.warp_k >= 32 && c.warp_k % 32 == 0 && c.block_n % c.warp_n == 0 &&
                c.block_k % c.warp_k == 0 && c.num_stages >= 2 && c.num_stages <= 4,
                "Invalid W4A8 tile configuration");
    const int warps_on_k = c.block_k / c.warp_k;
    TORCH_CHECK((warps_on_k & (warps_on_k - 1)) == 0,
                "W4A8 K-warp reduction requires a power of two number of warp groups");
    // N expansion must fill every pipeline stage before advancing to the next N tile.
    c.n_expand = !shared_decode && k <= 512 && n % (c.block_n * 4) == 0 && k % c.block_k == 0 &&
                 k / c.block_k >= c.num_stages && c.num_stages == 2 && c.block_k == c.warp_k ? 4 : 1;
    return c;
}

static int smem_size(const Config& c, int group_size, int b_scale_block = 0,
                     int a_scale_group = 0, int k = 0, bool fused = false) {
    const int a = c.block_m * c.block_k * c.num_stages;
    const int b = c.block_k * c.block_n / 2 * c.num_stages;
    const int s = c.block_k / group_size * c.block_n * c.num_stages;
    const int reduce = 16 * 64 * (c.block_n / c.warp_n) * (c.block_k / c.warp_k) * 4;
    const int epilogue = 16 * c.block_n * 2;
    const int lut = group_size == 16 && b_scale_block > 0 ? 1024 : 256;
    const bool staged_scales = group_size == 16 && !fused &&
        use_fast_block_scales(k, c.block_k, c.warp_k, c.n_expand,
                              a_scale_group, b_scale_block) &&
        can_stage_block_scales(c.block_m, c.block_n, c.block_k, c.warp_m);
    const int outer_scales = staged_scales ? (c.block_m + 32) * c.num_stages * sizeof(float) : 0;
    const int decoded = staged_scales && is_shared_decode_tile(c) ? c.block_n * c.block_k : 0;
    return lut + outer_scales + decoded + std::max({a + b + s, reduce, epilogue});
}

} // namespace deep_gemm::nvfp4::detail
