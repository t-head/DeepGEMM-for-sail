#pragma once
#include <algorithm>
#include <string>
#include "../../jit/device_runtime.hpp"
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

// Measured specialization domains. Shape checks live here; GPU kernels receive
// dimensions through template parameters and per-call arguments.
inline constexpr int kTunedSms = 39;

static bool has_small_m_profile(int groups, int n, int k) {
    return (groups == 28 || groups == 56) && ((n == 6144 && k == 3584) || (n == 3584 && k == 3072));
}

static bool small_m_eligible(int groups, int m, int n, int k) {
    return ceil_div(m, groups) <= 64 && has_small_m_profile(groups, n, k);
}

static bool nonempty_streamed_eligible(int groups, int m, int n, int k, int num_sms) {
    return has_small_m_profile(groups, n, k) && int64_t(m) == int64_t(groups) * 512 && num_sms == kTunedSms;
}

struct RoutedProblem {
    int groups, rows, source_rows, n, k;
};
inline constexpr RoutedProblem kRoutedProblem{896, 131072, 8192, 768, 3584};

static bool has_routed_profile(int groups, int m, int n, int k, GemmType kind) {
    const auto& p = kRoutedProblem;
    return groups == p.groups && n == p.n && k == p.k &&
           m == (kind == GemmType::MGroupedFused ? p.source_rows : p.rows);
}

static bool use_tuned_device() {
    const auto prop = device_runtime->get_prop();
    return std::string(prop->name) == "ZW-M890P" && prop->multiProcessorCount == kTunedSms &&
           get_num_sms() == kTunedSms;
}

struct SmallMPolicy {
    int k_batch;
    bool schedule_bound;
};
// The accepted compact-fragment schedules consume one K32 slice at a time.
inline constexpr SmallMPolicy kSmallMPolicy{1, false};

static W4A8Config small_m_config(int expected_m) {
    const int bm = expected_m <= 16 ? 16 : expected_m <= 32 ? 32 : 64;
    return {get_num_sms(), bm, 256, 128, bm, bm == 32 ? 32 : 64, 128, 2, 1};
}

static W4A8Config routed_config(bool small, GemmType kind) {
    if (kind == GemmType::MGroupedFused && !small)
        return {get_num_sms(), 192, 384, 256, 96, 48, 256, 2, 1};
    const int bm = small ? 160 : 192;
    return {get_num_sms(), bm, 256, 128, bm, 32, 128, small ? 2 : 3, 1};
}

static bool is_default_routed_config(const W4A8Config& c) {
    const W4A8Config expected{get_num_sms(), 64, 256, 128, 64, 64, 128, 2, 1};
    return pack_config(c) == pack_config(expected);
}

static int specialized_smem_size(const W4A8Config& c, int k, bool routed) {
    const int base = get_smem_size(c, k);
    if (routed)
        return base + ((2 * c.block_m * int(sizeof(float)) + 127) / 128) * 128;
    // M32 shares the prefix table between warps; M16/M64 keep it in registers.
    return base + (c.block_m == 32 ? 2 * 64 * int(sizeof(uint32_t)) : 0);
}

// Larger width removes repeated core segmentation when the bounded buffer fits.
static bool full_width_streamed(int n, int k, int groups) {
    constexpr int64_t budget = 384LL * 1024 * 1024;
    return groups >= 28 && groups < 128 && n >= 1024 && n % 512 == 0 && k >= 1024 &&
           int64_t(n) * k <= (budget - 8) / 14;
}

// Short calls need long K and enough tiles to avoid paying for additional waves.
// Keep the established native route when four-expert chunks fragment those waves.
static bool short_streamed_eligible(int m, int n, int k, int groups, int num_sms) {
    if (!full_width_streamed(n, k, groups) || n < 3584 || k < 3072 || int64_t(m) < int64_t(groups) * 512 ||
        int64_t(m) >= int64_t(groups) * 1024)
        return false;
    const int64_t row_tiles = (int64_t(m) + int64_t(groups) * 256 - 1) / (int64_t(groups) * 256);
    const int64_t n_tiles = (int64_t(n) + 255) / 256;
    const int sm_count = std::max(1, num_sms);
    auto estimated_waves = [&](int size) {
        auto waves = [&](int count) {
            const int64_t tiles = int64_t(count) * row_tiles * n_tiles;
            return (tiles + sm_count - 1) / sm_count;
        };
        return int64_t(groups / size) * waves(size) + waves(groups % size);
    };
    return estimated_waves(4) <= estimated_waves(14);
}

static int streamed_min_rows(int n, int k, int groups, int m, int num_sms) {
    if (!full_width_streamed(n, k, groups))
        return 2048;
    return short_streamed_eligible(m, n, k, groups, num_sms) ? 512 : 1024;
}

// Each call overwrites a bounded, partial FP8 tile; no decoded values survive as a weight cache.
static int streamed_tile_n(int n, int k, int groups) {
    if (groups < 14 || groups >= 128 || n < 1024 || n % 512 || k < 1024)
        return 0;
    if (full_width_streamed(n, k, groups))
        return n;
    constexpr int64_t max_workspace_bytes = 128LL * 1024 * 1024;
    int tile = 0;
    for (int candidate = 256; candidate <= std::min(2048, n / 2); candidate += 256) {
        if (n % candidate == 0 && int64_t(14) * candidate * k + 8 <= max_workspace_bytes)
            tile = candidate;
    }
    return tile;
}

struct StreamedPlan {
    std::vector<std::pair<int, int>> groups;
    int max_groups;
    bool ramp, short_call;
};
static StreamedPlan make_streamed_plan(int m, int n, int k, int groups, int num_sms) {
    // Small chunks repay their extra kernel/event startup only when the
    // full-width weight slice is large and the call has a short compute tail.
    const int64_t row_tiles = (int64_t(m) + int64_t(groups) * 256 - 1) / (int64_t(groups) * 256);
    const int64_t n_tiles = (int64_t(n) + 255) / 256;
    const int sm_count = std::max(1, num_sms);
    auto estimated_waves = [&](int size) {
        auto waves = [&](int count) {
            const int64_t tiles = int64_t(count) * row_tiles * n_tiles;
            return (tiles + sm_count - 1) / sm_count;
        };
        return int64_t(groups / size) * waves(size) + waves(groups % size);
    };
    const bool short_call = short_streamed_eligible(m, n, k, groups, num_sms);
    const bool smaller_chunk =
        short_call || (int64_t(m) >= int64_t(groups) * 1024 && int64_t(m) < int64_t(groups) * 2048 &&
                       int64_t(14) * n * k > 256LL * 1024 * 1024 && estimated_waves(4) <= estimated_waves(14));
    // Shorten initial decode at R1024/R2048 while retaining long later
    // collectives. Uniform four-expert chunks remain the measured R512 choice.
    // R4096 keeps the existing fourteen-expert plan until separately measured.
    const bool ramp =
        n >= 3584 && k >= 3072 && int64_t(m) >= int64_t(groups) * 1024 && int64_t(m) < int64_t(groups) * 4096;
    std::vector<std::pair<int, int>> plan;
    int chunk = 0;
    for (int start = 0, target = ramp ? 2 : (smaller_chunk ? 4 : 14); start < groups;) {
        const int count = std::min(target, groups - start);
        plan.emplace_back(start, count);
        chunk = std::max(chunk, count);
        start += count;
        if (ramp)
            target = std::min(14, target * 2);
    }
    return {std::move(plan), chunk, ramp, short_call};
}

} // namespace deep_gemm::w4fa8
