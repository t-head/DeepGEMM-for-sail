#pragma once
#include <algorithm>
#include <string>
#include "../../jit/device_runtime.hpp"
#include <deep_gemm/common/w4a8_arguments.cuh>
#include <tuple>
#include <vector>
#include <variant>
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
// Execution choices are independent; only supported combinations are admitted.
enum class ChunkSchedule { Fixed, Ramp };
enum class GatherPolicy { None, Full, Chunk };
enum class EmptyExpertPolicy { Dense, Nonempty };
enum class DecodeFencePolicy { Fenced, FenceFree };

struct NativeKernelConfig {
    W4A8Config tile;
    FragmentPolicy fragment;

    ALoadPolicy a_load() const { return uses_ring_a(tile.warp_m) ? ALoadPolicy::Ring2 : ALoadPolicy::Bulk; }
};
struct StreamedConfig {
    W4A8Config core;
    int decode_n, expert_chunk;
    ChunkSchedule schedule;
    GatherPolicy gather;
    EmptyExpertPolicy empty_experts;
    DecodeFencePolicy decode_fence;
    std::vector<std::pair<int, int>> group_chunks;
};
using ExecutionPlan = std::variant<NativeKernelConfig, StreamedConfig>;

using W4A8ConfigTuple = std::tuple<int, int, int, int, int, int, int, int, int>;

static W4A8Config unpack_config(const W4A8ConfigTuple& tuple) {
    const auto& [sms, bm, bn, bk, wm, wn, wk, stages, expand] = tuple;
    return {sms, bm, bn, bk, wm, wn, wk, stages, expand};
}
static W4A8ConfigTuple pack_config(const W4A8Config& c) {
    return {c.num_sms, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, c.num_stages, c.n_expand};
}
static int get_num_threads(const W4A8Config& c) {
    const int64_t threads = int64_t(c.block_m / c.warp_m) * (c.block_n / c.warp_n) * 32;
    DG_HOST_ASSERT(threads > 0 && threads <= 1024);
    return int(threads);
}

static void check_config(const W4A8Config& c, int n, int k, GemmType kind) {
    DG_HOST_ASSERT(c.num_sms > 0 and c.num_sms <= get_num_sms());
    DG_HOST_ASSERT(c.block_m == 16 or c.block_m == 32 or c.block_m == 64);
    DG_HOST_ASSERT(c.warp_m == 16 or c.warp_m == 32 or c.warp_m == 64);
    DG_HOST_ASSERT(c.block_m == c.warp_m and c.warp_n == 64);
    DG_HOST_ASSERT(c.block_n >= 64 and c.block_n % 64 == 0);
    // PPU1.5 AIU minimum 64B swizzle requires 128 packed-FP4 K elements.
    DG_HOST_ASSERT(c.block_k == 128 and c.warp_k == c.block_k);
    DG_HOST_ASSERT(k % c.block_k == 0 and c.num_stages >= 2 and c.num_stages <= 4);
    DG_HOST_ASSERT(c.n_expand >= 1 and c.n_expand <= 4);
    if (c.n_expand > 1) {
        DG_HOST_ASSERT(kind == GemmType::MGroupedNoPad and c.num_stages == 2);
        DG_HOST_ASSERT(c.block_k == c.warp_k);
        DG_HOST_ASSERT(n % (int64_t(c.block_n) * c.n_expand) == 0);
    }
    DG_HOST_ASSERT(get_num_threads(c) <= 1024);
}

static int get_smem_size(const W4A8Config& c, int k) {
    const int threads = get_num_threads(c);
    const int copy_rows = threads / std::min(threads, std::min(c.block_k, 128) / 16);
    const int a_rows = uses_ring_a(c.warp_m) ? ceil_div(c.block_m, copy_rows) * copy_rows : c.block_m;
    const bool resident_a = c.n_expand > 1 && c.warp_m == 64 && k <= 384;
    const int a_slots = resident_a ? std::max(c.num_stages, k / c.block_k) : c.num_stages;
    const int64_t mainloop = int64_t(a_rows) * c.block_k * a_slots +
                            int64_t(c.block_n) * (c.block_k / 2 + c.block_k / 32) * c.num_stages;
    int64_t bytes = mainloop;
    if (uses_ring_a(c.warp_m))
        bytes = ceil_div(bytes, int64_t(128)) * 128 + ceil_div(int64_t(2 * c.block_m * sizeof(float)), int64_t(128)) * 128;
    DG_HOST_ASSERT(bytes <= INT32_MAX);
    return int(bytes);
}

// Configuration selection uses only host-known shapes and device resources.
static bool supports_native_strategy() { return is_ppu1v5_device() && get_num_sms() > 0; }

static bool fits_strategy(const W4A8Config& c, int k) {
    const auto prop = device_runtime->get_prop();
    return get_num_threads(c) <= prop->maxThreadsPerBlock &&
           get_smem_size(c, k) <= int(prop->sharedMemPerBlockOptin);
}

// Bound M before rounding so even a large tuning hint cannot overflow.
static W4A8Config get_default_tile(int expected_m, int n, int k, int groups, GemmType kind) {
    DG_HOST_ASSERT(expected_m >= 0 && groups > 0 && n > 0 && n % 64 == 0 && k > 0);
    DG_HOST_ASSERT(kind == GemmType::MGroupedFused || kind == GemmType::MGroupedNoPad ||
                   kind == GemmType::MGroupedMasked);
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
    // Refine N partitioning for small standard tiles; M and routing layout stay unchanged.
    if (supports_native_strategy() && expected_m <= 256 && groups <= 1024 &&
        n >= 128 && k >= 512 && k % 128 == 0) {
        auto candidate = c;
        const bool underfilled = int64_t(groups) * ceil_div(n, bn) < c.num_sms;
        candidate.block_n = bm == 32 ? 128 : bn;
        candidate.warp_n = bm == 32 || (bm == 16 && underfilled) ? 32 : 64;
        candidate.num_stages = 2;
        candidate.n_expand = 1;
        if (fits_strategy(candidate, k))
            c = candidate;
    }
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

static bool fits_compiled_strategy(int local_bytes) {
    // Allow a small compiler frame, but reject spilled operand fragments.
    return local_bytes <= 64;
}
// Plan one tile per call; the scheduler uses each expert's actual row count.
static W4A8Config default_wide_config(int expected_m, int k, GemmType kind) {
    const bool fused = kind == GemmType::MGroupedFused;
    const int max_m = fused ? 176 : 192;
    int bm;
    if (int64_t(expected_m) * 10 <= max_m * 9) {
        const int required = int(ceil_div(int64_t(expected_m) * 10, int64_t(9)));
        bm = std::max(80, ceil_div(required, 16) * 16);
        // Long-K Fused loses decode/compute overlap at M176; retain M160.
        if (fused && k > 2048 && bm == 176)
            bm = 160;
    } else if (!fused && int64_t(expected_m) + 16 <= max_m) {
        // Keep one M16 of headroom before splitting the NoPad transition range.
        bm = max_m;
    } else {
        const int64_t blocks = std::max(int64_t(2), ceil_div(int64_t(expected_m) * 10, int64_t(160 * 9)));
        const int required = int(ceil_div(int64_t(expected_m) * 10, blocks * 9));
        bm = std::clamp(ceil_div(required, 16) * 16, 80, 160);
    }
    const int stages = fused && bm <= 144 ? 3 : 2;
    return {get_num_sms(), bm, 256, 128, bm, 32, 128, stages, 1};
}

static bool wide_tile_eligible(int groups, int m, int n, int k, GemmType kind) {
    if (groups < 1 || groups > 1024 || n < 256 || n % 256 ||
        k < 512 || k % 128)
        return false;
    const int expected_m = ceil_div(m, groups);
    if (expected_m < 65)
        return false;
    const auto tile = default_wide_config(expected_m, k, kind);
    const int64_t tasks = int64_t(ceil_div(m, tile.block_m)) * (n / 256);
    if (tasks < get_num_sms())
        return false;
    return fits_strategy(tile, k);
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
static bool is_short_streamed_workload(int m, int n, int k, int groups, int num_sms) {
    const int64_t nk_tiles = int64_t(ceil_div(n, 256)) * (k / 128);
    const int tile_n = streamed_tile_n(n, k, groups);
    if (!tile_n || nk_tiles < 128 || k / 128 < 16 || int64_t(m) < int64_t(groups) * 512 ||
        int64_t(m) >= int64_t(groups) * 1024)
        return false;
    const int maximum = streamed_chunk_limit(tile_n, k, groups);
    return streamed_waves(m, tile_n, groups, num_sms, std::min(4, maximum)) <=
           streamed_waves(m, tile_n, groups, num_sms, maximum);
}

static int streamed_min_rows(int n, int k) {
    // Admission depends on workload; short-call chunk scheduling stays independent.
    const int64_t nk_tiles = int64_t(ceil_div(n, 256)) * (k / 128);
    return k >= 2048 && nk_tiles >= 128 ? 512 : 1024;
}

static bool streamed_fused_eligible(int m, int n, int k, int groups) {
    // Bound the additional activation workspace independently of weight chunks.
    constexpr int64_t activation_budget = 1024LL * 1024 * 1024;
    return supports_native_strategy() && streamed_tile_n(n, k, groups) &&
           int64_t(m) * (k + int(sizeof(float))) <= activation_budget &&
           int64_t(m) >= int64_t(groups) * streamed_min_rows(n, k);
}

struct StreamedPlan {
    std::vector<std::pair<int, int>> groups;
    int max_groups;
};
static StreamedPlan make_streamed_plan(int groups, int expert_chunk, ChunkSchedule schedule) {
    std::vector<std::pair<int, int>> plan;
    const bool ramp = schedule == ChunkSchedule::Ramp;
    int maximum = 0;
    for (int start = 0, target = ramp ? std::min(2, expert_chunk) : expert_chunk; start < groups;) {
        const int count = std::min({target, expert_chunk, groups - start});
        plan.emplace_back(start, count);
        maximum = std::max(maximum, count);
        start += count;
        if (ramp) target = std::min(expert_chunk, target * 2);
    }
    return {std::move(plan), maximum};
}

static W4A8Config streamed_core_config(int num_sms) {
    return {num_sms, w4fa8_streamed::kBlockM, 256, 128, 64, 64, 128, 4, 1};
}

static int64_t streamed_plan_score(int m, int n, int groups, int num_sms, const StreamedPlan& plan) {
    const int64_t tiles_per_expert = ceil_div(int64_t(m), int64_t(groups) * 256) * ceil_div(int64_t(n), int64_t(256));
    int64_t score = plan.groups.size();
    for (const auto& group : plan.groups)
        score += ceil_div(int64_t(group.second) * tiles_per_expert, int64_t(std::max(1, num_sms)));
    return score;
}

static int select_streamed_chunk(int m, int tile_n, int k, int groups, int num_sms, bool minimize_startup,
                                 ChunkSchedule schedule) {
    const int maximum = streamed_chunk_limit(tile_n, k, groups);
    const int64_t max_waves = minimize_startup ? streamed_waves(m, tile_n, groups, num_sms, maximum) : 0;
    int selected = maximum;
    int64_t best = INT64_MAX;
    for (int candidate = 1; candidate <= maximum; ++candidate) {
        if (minimize_startup) {
            // Minimize decode startup without increasing complete compute waves.
            if (streamed_waves(m, tile_n, groups, num_sms, candidate) <= max_waves)
                return candidate;
        } else {
            const int64_t score = schedule == ChunkSchedule::Ramp
                                      ? streamed_plan_score(m, tile_n, groups, num_sms, make_streamed_plan(groups, candidate, schedule))
                                      : streamed_waves(m, tile_n, groups, num_sms, candidate) + ceil_div(groups, candidate);
            if (score <= best) {
                best = score;
                selected = candidate;
            }
        }
    }
    return selected;
}

static void check_streamed_config(const StreamedConfig& c, int m, int n, int k, int groups, GemmType kind) {
    const auto& t = c.core;
    DG_HOST_ASSERT(kind == GemmType::MGroupedNoPad || kind == GemmType::MGroupedFused);
    DG_HOST_ASSERT(groups >= 8 && groups <= 1024 && n >= 1024 && n % 64 == 0 && k >= 1024 && k % 128 == 0);
    DG_HOST_ASSERT(t.num_sms > 0 && t.num_sms <= get_num_sms());
    DG_HOST_ASSERT(pack_config(t) == W4A8ConfigTuple(t.num_sms, w4fa8_streamed::kBlockM, 256, 128, 64, 64, 128, 4, 1));
    const auto prop = device_runtime->get_prop();
    DG_HOST_ASSERT(prop->maxThreadsPerBlock >= 512 && prop->sharedMemPerBlockOptin >= 256 * 1024);
    DG_HOST_ASSERT(c.decode_n > 0 && c.decode_n <= n && (c.decode_n == n || c.decode_n % 256 == 0));
    DG_HOST_ASSERT(c.expert_chunk >= 1 && c.expert_chunk <= std::min(32, groups / 2));
    DG_HOST_ASSERT(!c.group_chunks.empty());
    DG_HOST_ASSERT(int64_t(c.expert_chunk) * c.decode_n * k + 8 <= kStreamedWeightBudget);
    DG_HOST_ASSERT(c.empty_experts != EmptyExpertPolicy::Nonempty || c.decode_n == n);
    if (kind == GemmType::MGroupedFused) {
        DG_HOST_ASSERT(c.gather != GatherPolicy::None);
        DG_HOST_ASSERT(c.gather != GatherPolicy::Chunk || c.decode_n == n);
        DG_HOST_ASSERT(int64_t(m) * (k + sizeof(float)) <= 1024LL * 1024 * 1024);
    } else {
        DG_HOST_ASSERT(c.gather == GatherPolicy::None);
    }
}

static StreamedConfig default_streamed_config(int m, int n, int k, int groups, int num_sms,
                                               bool fused, bool allow_sparse) {
    const int tile_n = streamed_tile_n(n, k, groups);
    const bool short_workload = is_short_streamed_workload(m, n, k, groups, num_sms);
    const bool ramp = int64_t(tile_n) * k >= 8LL * 1024 * 1024 && int64_t(m) >= int64_t(groups) * 1024 &&
                      (fused || int64_t(m) < int64_t(groups) * 4096);
    const auto schedule = ramp ? ChunkSchedule::Ramp : ChunkSchedule::Fixed;
    const int chunk = select_streamed_chunk(m, tile_n, k, groups, num_sms, short_workload, schedule);
    auto plan = make_streamed_plan(groups, chunk, schedule);
    // Preserve the existing defaults; runtime consumes each policy independently.
    const auto gather = !fused ? GatherPolicy::None : !short_workload && tile_n == n ? GatherPolicy::Chunk : GatherPolicy::Full;
    const auto decode_fence = short_workload && k % 1024 != 0 ? DecodeFencePolicy::FenceFree : DecodeFencePolicy::Fenced;
    return {streamed_core_config(num_sms), tile_n, plan.max_groups, schedule, gather,
            allow_sparse && tile_n == n ? EmptyExpertPolicy::Nonempty : EmptyExpertPolicy::Dense, decode_fence,
            std::move(plan.groups)};
}

static void check_native_kernel(const NativeKernelConfig& spec, int n, int k, int groups, GemmType kind) {
    const auto& c = spec.tile;
    TORCH_CHECK(c.block_m == c.warp_m, "W4FA8 native tiles require block_m == warp_m");
    TORCH_CHECK(c.block_k == c.warp_k, "W4FA8 native tiles require block_k == warp_k");
    DG_HOST_ASSERT(groups > 0 && groups <= 1024 && c.num_sms > 0 && c.num_sms <= get_num_sms());
    if (uses_ring_a(c.warp_m)) {
        DG_HOST_ASSERT(kind != GemmType::MGroupedMasked);
        DG_HOST_ASSERT(c.block_m >= 80 && c.block_m <= 192 && c.block_m % 16 == 0);
        DG_HOST_ASSERT(c.warp_m == c.block_m && c.block_n == 256 && c.warp_n == 32);
        DG_HOST_ASSERT(c.block_k == 128 && c.warp_k == 128 && c.n_expand == 1);
        DG_HOST_ASSERT(c.num_stages >= 2 && c.num_stages <= 4 && n % 256 == 0 && k >= 512 && k % 128 == 0);
        DG_HOST_ASSERT(spec.fragment == FragmentPolicy::RollingK32);
    } else if (spec.fragment == FragmentPolicy::FullK) {
        check_config(c, n, k, kind);
    } else {
        DG_HOST_ASSERT(n >= 128);
        DG_HOST_ASSERT(c.warp_n == 32);
        DG_HOST_ASSERT(c.block_m == 16 || c.block_m == 32);
        DG_HOST_ASSERT(c.block_m == c.warp_m && (c.block_n == 128 || c.block_n == 256));
        DG_HOST_ASSERT(c.block_k == 128 && c.warp_k == 128 && c.n_expand == 1);
        DG_HOST_ASSERT(k >= 512 && k % 128 == 0 && c.num_stages == 2);
        DG_HOST_ASSERT(spec.fragment == FragmentPolicy::RollingK32);
    }
    const auto prop = device_runtime->get_prop();
    DG_HOST_ASSERT(get_num_threads(c) <= prop->maxThreadsPerBlock &&
                   get_smem_size(c, k) <= int(prop->sharedMemPerBlockOptin));
}
static void check_plan(const ExecutionPlan& plan, int m, int n, int k, int groups, GemmType kind) {
    if (const auto* native = std::get_if<NativeKernelConfig>(&plan))
        check_native_kernel(*native, n, k, groups, kind);
    else
        check_streamed_config(std::get<StreamedConfig>(plan), m, n, k, groups, kind);
}

// Derive the single native implementation from the tile and host-known shape.
static NativeKernelConfig resolve_native_config(const W4A8Config& c, int n, int k) {
    TORCH_CHECK(c.block_m == c.warp_m, "W4FA8 native tiles require block_m == warp_m");
    TORCH_CHECK(c.block_k == c.warp_k, "W4FA8 native tiles require block_k == warp_k");
    if (uses_ring_a(c.warp_m))
        return {c, FragmentPolicy::RollingK32};
    const bool rolling = (c.block_m == 16 || c.block_m == 32) && c.block_m == c.warp_m && c.warp_n == 32 &&
                       (c.block_n == 128 || c.block_n == 256) && c.block_k == 128 && c.warp_k == 128 &&
                       c.num_stages == 2 && c.n_expand == 1 && n >= 128 && k >= 512 && k % 128 == 0;
    const bool full = c.warp_n == 64;
    TORCH_CHECK(full || rolling, "W4FA8 tile has no supported native implementation");
    return {c, rolling ? FragmentPolicy::RollingK32 : FragmentPolicy::FullK};
}

static W4A8Config get_best_configs(int expected_m, int n, int k, int groups, GemmType kind) {
    const int em = expected_m;
    const int m = expected_m * groups;
    const auto base = get_default_tile(em, n, k, groups, kind);
    if (kind != GemmType::MGroupedMasked && supports_native_strategy()) {
        const bool fused = kind == GemmType::MGroupedFused;
        if (streamed_tile_n(n, k, groups) && int64_t(m) >= int64_t(groups) * streamed_min_rows(n, k) &&
            (!fused || streamed_fused_eligible(m, n, k, groups)))
            return streamed_core_config(base.num_sms);
        if (wide_tile_eligible(groups, m, n, k, kind))
            return default_wide_config(em, k, kind);
    }
    return base;
}

// Public configs contain only a tile; execution details stay internal.
static ExecutionPlan resolve_config(const W4A8Config& tile, int m, int n, int k, int groups, GemmType kind) {
    if (tile.block_m == w4fa8_streamed::kBlockM) {
        TORCH_CHECK(pack_config(tile) == W4A8ConfigTuple(tile.num_sms, w4fa8_streamed::kBlockM, 256, 128, 64, 64, 128, 4, 1),
                    "unsupported W4FA8 streamed tile");
        TORCH_CHECK(kind != GemmType::MGroupedMasked && supports_native_strategy() && streamed_tile_n(n, k, groups),
                    "W4FA8 streamed tile is unavailable for this interface, shape or device");
        auto plan = default_streamed_config(m, n, k, groups, tile.num_sms,
                                           kind == GemmType::MGroupedFused, true);
        plan.core = tile;
        check_streamed_config(plan, m, n, k, groups, kind);
        return plan;
    }
    return resolve_native_config(tile, n, k);
}

} // namespace deep_gemm::w4fa8
