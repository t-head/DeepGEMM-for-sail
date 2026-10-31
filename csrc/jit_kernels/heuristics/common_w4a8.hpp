#pragma once
#include <algorithm>
#include <string>
#include "../../jit/device_runtime.hpp"
#include <deep_gemm/common/w4a8_arguments.cuh>
#include <tuple>
#include <vector>
#include <utility>
#include "../../utils/exception.hpp"
#include "../../utils/utils.hpp"

namespace deep_gemm::w4fa8 {
struct W4A8Config {
    int num_sms;
    int block_m, block_n, block_k;
    int warp_m, warp_n, warp_k;
    int num_stages, n_expand;
};
using W4A8ConfigTuple = std::tuple<int, int, int, int, int, int, int, int, int>;

static W4A8Config unpack_config(const W4A8ConfigTuple& tuple) {
    const auto& [sms, bm, bn, bk, wm, wn, wk, stages, expand] = tuple;
    return {sms, bm, bn, bk, wm, wn, wk, stages, expand};
}
static W4A8ConfigTuple pack_config(const W4A8Config& c) {
    return {c.num_sms, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, c.num_stages, c.n_expand};
}
static int get_num_threads(const W4A8Config& c) {
    return (c.block_m / c.warp_m) * (c.block_n / c.warp_n) * (c.block_k / c.warp_k) * 32;
}

static void check_config(const W4A8Config& c, int n, int k, GemmType kind) {
    DG_HOST_ASSERT(c.num_sms > 0 and c.num_sms <= get_num_sms());
    DG_HOST_ASSERT(c.block_m == 16 or c.block_m == 32 or c.block_m == 64 or c.block_m == 128);
    DG_HOST_ASSERT(c.warp_m == 16 or c.warp_m == 32 or c.warp_m == 64 or c.warp_m == 128);
    DG_HOST_ASSERT(c.block_m % c.warp_m == 0 and (c.warp_n == 64 or c.warp_n == 32));
    DG_HOST_ASSERT(c.block_n >= 64 and c.block_n % 64 == 0);
    if (c.warp_n == 32 or c.warp_m == 128) {
        DG_HOST_ASSERT(kind == GemmType::MGroupedNoPad and c.warp_m == 128 and c.warp_n == 32);
        DG_HOST_ASSERT(c.block_m == 128 and c.block_k == c.warp_k and c.n_expand == 1);
    }
    // PPU1.5 AIU minimum 64B swizzle requires 128 packed-FP4 K elements.
    DG_HOST_ASSERT(c.block_k == 128 and (c.warp_k == 64 or c.warp_k == 128));
    DG_HOST_ASSERT(k % c.block_k == 0 and c.num_stages >= 2 and c.num_stages <= 4);
    DG_HOST_ASSERT(c.n_expand >= 1 and c.n_expand <= 4);
    if (c.n_expand > 1) {
        DG_HOST_ASSERT(kind == GemmType::MGroupedNoPad and c.num_stages == 2);
        DG_HOST_ASSERT(c.block_k == c.warp_k);
        DG_HOST_ASSERT(n % (c.block_n * c.n_expand) == 0);
    }
    DG_HOST_ASSERT(c.block_m == c.warp_m or c.block_k == c.warp_k);
    DG_HOST_ASSERT(get_num_threads(c) <= 1024);
}

static int get_smem_size(const W4A8Config& c, int k) {
    const bool resident_a = c.n_expand > 1 && c.warp_m == 64 && k <= 384;
    const int a_slots = resident_a ? std::max(c.num_stages, k / c.block_k) : c.num_stages;
    const int mainloop =
        c.block_m * c.block_k * a_slots + (c.block_n * c.block_k / 2 + c.block_n * c.block_k / 32) * c.num_stages;
    const int reduction = 16 * 64 * (c.block_n / 64) * (c.block_k / c.warp_k) * 4;
    return std::max(mainloop, reduction);
}

// Bound M before rounding so even a large tuning hint cannot overflow.
static W4A8Config get_best_configs(int expected_m, int n, int k, int groups, GemmType kind) {
    DG_HOST_ASSERT(expected_m >= 0 && groups > 0 && n > 0 && n % 64 == 0 && k > 0);
    DG_HOST_ASSERT(kind == GemmType::MGroupedFused || kind == GemmType::MGroupedNoPad ||
                   kind == GemmType::MGroupedMasked);
    // Wider M reuses each native FP4 decode over eight M16 fragments.
    // Keep the small-M, short-K and other grouped paths on their existing tiles.
    if (kind == GemmType::MGroupedNoPad and expected_m >= 128 and n >= 1024 and n % 128 == 0 and k >= 1024) {
        W4A8Config c{get_num_sms(), 128, 128, 128, 128, 32, 128, 2, 1};
        check_config(c, n, k, kind);
        return c;
    }
    int bm = 16;
    while (bm < expected_m && bm < 64)
        bm *= 2;
    const int bn = n >= 256 ? 256 : n >= 128 ? 128 : 64;
    // M32/N256 has enough shared memory for three stages at four CTAs/SM
    // (64,512 bytes per CTA). M16 and M64 lose residency with a third stage.
    const int stages = kind == GemmType::MGroupedMasked && bm == 32 && bn == 256 ? 3 : 2;
    W4A8Config c{get_num_sms(), bm, bn, 128, bm, 64, 128, stages, 1};
    if (kind == GemmType::MGroupedNoPad and k <= 512 and k % c.block_k == 0 and c.num_stages == 2 and
        c.block_k == c.warp_k) {
        for (const int candidate : {4, 3, 2}) {
            if (n % (bn * candidate) == 0) {
                c.n_expand = candidate;
                break;
            }
        }
    }
    check_config(c, n, k, kind);
    return c;
}
static const char* get_gemm_type_name(GemmType kind) {
    switch (kind) {
    case GemmType::MGroupedFused:
        return "MGroupedFused";
    case GemmType::MGroupedNoPad:
        return "MGroupedNoPad";
    case GemmType::MGroupedMasked:
        return "MGroupedMasked";
    default:
        DG_HOST_ASSERT(false);
    }
    return "";
}

// Strategy domains follow the scheduler capacity, alignment and minimum work
// needed to amortize prefix/metadata construction. They contain no model shapes.
static bool supports_native_strategy() { return is_ppu1v5_device() && get_num_sms() > 0; }

static W4A8Config small_m_config(int expected_m, int n = 256, GemmType kind = GemmType::MGroupedNoPad, int groups = 1) {
    const int bm = expected_m <= 16 ? 16 : expected_m <= 32 ? 32 : 64;
    const bool fused = kind == GemmType::MGroupedFused;
    const int base_n = n >= 256 ? 256 : 128;
    const bool underfilled = int64_t(groups) * ceil_div(n, base_n) < get_num_sms();
    const int bn = fused && bm == 32 ? 128 : base_n;
    const int wn = bm == 32 || (fused && bm == 16 && underfilled) ? 32 : 64;
    return {get_num_sms(), bm, bn, 128, bm, wn, 128, 2, 1};
}

static int specialized_smem_size(const W4A8Config& c, int k, bool routed, bool prefix = true) {
    const int base = get_smem_size(c, k);
    if (routed)
        return base + ((2 * c.block_m * int(sizeof(float)) + 127) / 128) * 128;
    return base + (prefix && c.block_m == 32 ? 2 * 64 * int(sizeof(uint32_t)) : 0);
}

static bool fits_strategy(const W4A8Config& c, int k, bool routed, bool prefix = true) {
    const auto prop = device_runtime->get_prop();
    return get_num_threads(c) <= prop->maxThreadsPerBlock &&
           specialized_smem_size(c, k, routed, prefix) <= int(prop->sharedMemPerBlockOptin);
}

static bool fits_compiled_strategy(int local_bytes) {
    // Allow a small compiler frame, but reject spilled operand fragments.
    return local_bytes <= 64;
}

static bool small_m_eligible(int groups, int m, int n, int k, GemmType kind = GemmType::MGroupedNoPad) {
    const bool prefix = kind != GemmType::MGroupedFused && groups <= 64;
    if (groups < 1 || groups > 1024 || ceil_div(m, groups) >= 512 || n < 128 || n % 64 || k < 512 ||
        k % 128)
        return false;
    return fits_strategy(small_m_config(ceil_div(m, groups), n, kind, groups), k, false, prefix);
}

static NativeStrategy small_m_strategy(const W4A8Config& c, GemmType kind) {
    return kind == GemmType::MGroupedFused && c.block_m == 16 && c.warp_n == 64 ? NativeStrategy::Default
                                                                                : NativeStrategy::Compact;
}

static W4A8Config routed_config(bool small) {
    const int bm = small ? 160 : 192;
    return {get_num_sms(), bm, 256, 128, bm, 32, 128, small ? 2 : 3, 1};
}

static bool routed_eligible(int groups, int m, int n, int k, GemmType kind, const W4A8Config& original) {
    if (groups < 1 || groups > 1024 || (groups <= 64 && kind != GemmType::MGroupedFused) || n < 256 || n % 256 ||
        k < 512 || k % 128)
        return false;
    const int expected_m = ceil_div(m, groups);
    if (expected_m < 96 || expected_m > 256 || int64_t(ceil_div(m, 192)) * (n / 256) < get_num_sms())
        return false;
    // Small expert sets need several waves to repay the device metadata pass.
    if (groups <= 64 && int64_t(ceil_div(m, 192)) * (n / 256) < int64_t(get_num_sms()) * 4)
        return false;
    // Avoid wide tiles when extra padding outweighs their decode reuse.
    if (kind == GemmType::MGroupedFused &&
        ceil_div(expected_m, 160) * 160 * 4 > ceil_div(expected_m, original.block_m) * original.block_m * 5)
        return false;
    // The token layout belongs to this public config, not to the internal tile.
    // Metadata records preserve its padded expert offsets explicitly.
    if (kind == GemmType::MGroupedFused &&
        pack_config(original) != pack_config(get_best_configs(expected_m, n, k, groups, kind)))
        return false;
    return fits_strategy(routed_config(true), k, true) &&
           (kind == GemmType::MGroupedFused || fits_strategy(routed_config(false), k, true));
}

// Limit the per-call selection tax relative to the calibrated native kernel.
inline constexpr float kMaxAdaptiveSelectionOverhead = 0.05f;

static bool adaptive_rows_eligible(int groups, int m, int n, int k) {
    if (!supports_native_strategy() || groups < 1 || groups > 1024 || n < 256 || n % 256 || k < 512 || k % 128)
        return false;
    const int rows = ceil_div(m, groups);
    if (rows <= 256 || rows >= 512)
        return false;
    const auto c = routed_config(true);
    const int64_t work = int64_t(groups) * ceil_div(rows, c.block_m) * (n / c.block_n);
    return work >= int64_t(get_num_sms()) * (groups <= 64 ? 4 : 1) && fits_strategy(c, k, true);
}

inline constexpr int64_t kStreamedWeightBudget = 384LL * 1024 * 1024;

// Keep at least one complete N256 core tile in each partial buffer. Expert
// chunks can shrink before N is split; a single oversized expert uses N slices.
static int streamed_tile_n(int n, int k, int groups) {
    const auto prop = device_runtime->get_prop();
    if (prop->maxThreadsPerBlock < 512 || prop->sharedMemPerBlockOptin < 256 * 1024 || groups < 8 || groups > 1024 ||
        n < 1024 || n % 64 || k < 1024 || k % 128)
        return 0;
    const int64_t capacity = (kStreamedWeightBudget - 8) / k / 256 * 256;
    return capacity >= 256 ? int(std::min(int64_t(n), capacity)) : 0;
}
static int streamed_chunk_limit(int tile_n, int k, int groups) {
    // One warp of expert counts stays below the stock scheduler's 128-expert
    // preprocessing boundary, and bounds decoder specialization/code size.
    return std::min({groups / 2, 32, int((kStreamedWeightBudget - 8) / (int64_t(tile_n) * k))});
}

// Estimate complete waves for a bounded expert chunk, including its tail.
static int64_t streamed_waves(int m, int n, int groups, int num_sms, int chunk) {
    const int64_t row_tiles = (int64_t(m) + int64_t(groups) * 256 - 1) / (int64_t(groups) * 256);
    const int64_t n_tiles = (int64_t(n) + 255) / 256;
    auto waves = [&](int experts) {
        const int64_t tiles = int64_t(experts) * row_tiles * n_tiles;
        return ceil_div(tiles, int64_t(std::max(1, num_sms)));
    };
    return int64_t(groups / chunk) * waves(chunk) + waves(groups % chunk);
}

// Short calls need enough N/K tile work to repay additional decode/core
// launches. Use tile counts rather than a list of projection dimensions.
static bool short_streamed_eligible(int m, int n, int k, int groups, int num_sms) {
    const int64_t nk_tiles = int64_t(ceil_div(n, 256)) * (k / 128);
    const int tile_n = streamed_tile_n(n, k, groups);
    if (!tile_n || nk_tiles < 128 || k / 128 < 16 || int64_t(m) < int64_t(groups) * 512 ||
        int64_t(m) >= int64_t(groups) * 1024)
        return false;
    const int maximum = streamed_chunk_limit(tile_n, k, groups);
    return streamed_waves(m, tile_n, groups, num_sms, std::min(4, maximum)) <=
           streamed_waves(m, tile_n, groups, num_sms, maximum);
}

static int streamed_min_rows(int n, int k, int groups, int m, int num_sms) {
    if (!streamed_tile_n(n, k, groups))
        return 2048;
    return short_streamed_eligible(m, n, k, groups, num_sms) ? 512 : 1024;
}

static bool default_fused_config(const W4A8Config& c, int m, int n, int k, int groups) {
    return pack_config(c) == pack_config(get_best_configs(ceil_div(m, groups), n, k, groups, GemmType::MGroupedFused));
}

static bool streamed_fused_eligible(int m, int n, int k, int groups, int num_sms) {
    // Bound the additional activation workspace independently of weight chunks.
    constexpr int64_t activation_budget = 1024LL * 1024 * 1024;
    return supports_native_strategy() && streamed_tile_n(n, k, groups) &&
           int64_t(m) * (k + int(sizeof(float))) <= activation_budget &&
           int64_t(m) >= int64_t(groups) * streamed_min_rows(n, k, groups, m, num_sms);
}

struct StreamedPlan {
    std::vector<std::pair<int, int>> groups;
    int max_groups;
    bool ramp, short_call;
};
static StreamedPlan make_streamed_plan(int m, int n, int k, int groups, int num_sms, bool fused = false) {
    const int tile_n = streamed_tile_n(n, k, groups);
    const int maximum = streamed_chunk_limit(tile_n, k, groups);
    const bool short_call = short_streamed_eligible(m, n, k, groups, num_sms);
    int selected = maximum;
    int64_t best = INT64_MAX;
    for (int candidate = 1; candidate <= maximum; ++candidate) {
        const int64_t waves = streamed_waves(m, tile_n, groups, num_sms, candidate);
        if (short_call) {
            // Minimize decode startup without increasing complete compute waves.
            if (waves <= streamed_waves(m, tile_n, groups, num_sms, maximum)) {
                selected = candidate;
                break;
            }
        } else {
            const int64_t score = waves + ceil_div(groups, candidate);
            if (score <= best) {
                best = score;
                selected = candidate;
            }
        }
    }
    const bool ramp = int64_t(tile_n) * k >= 8LL * 1024 * 1024 && int64_t(m) >= int64_t(groups) * 1024 &&
                      (fused || int64_t(m) < int64_t(groups) * 4096);
    std::vector<std::pair<int, int>> plan;
    int chunk = 0;
    for (int start = 0, target = ramp ? std::min(2, selected) : selected; start < groups;) {
        const int count = std::min({target, selected, groups - start});
        plan.emplace_back(start, count);
        chunk = std::max(chunk, count);
        start += count;
        if (ramp)
            target = std::min(selected, target * 2);
    }
    return {std::move(plan), chunk, ramp, short_call};
}

} // namespace deep_gemm::w4fa8
