#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include "common_fp4.hpp"
#include <deep_gemm/layout/dispatch_layout.cuh>

#include <deep_gemm/layout/routed_moe.cuh>
#include "common_fp8.hpp"
#include "common_int8.hpp"
#include "common_w4a16.hpp"
#include "common_w4a8.hpp"
#include "mega_moe_tuning.hpp"

namespace deep_gemm::mega {

// Rank-local token counts describe a uniform-routing estimate, not measured
// expert loads. Cap it by the owning workspace's per-expert row capacity.
inline int expected_expert_rows(int tokens, int topk, int groups, int ranks, int capacity) {
    DG_HOST_ASSERT(tokens >= 0 && topk > 0 && groups > 0 && ranks > 0 && capacity >= 0);
    const int64_t owner_capacity = int64_t(ranks) * capacity;
    return std::max<int64_t>(1, std::min(owner_capacity, int64_t(tokens) * topk / groups + 1));
}

enum class MmaKind { BF16, MXFP8FP4, MXFP4 };

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    if (mma_type_str == "fp8xfp4" or mma_type_str == "fp8xfp8")
        return MmaKind::MXFP8FP4;
    DG_HOST_ASSERT(mma_type_str == "mxf4xmxf4");
    return MmaKind::MXFP4;
}

struct MegaMoEPhaseConfig {
    int num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages;
    int smem_size;
};

struct MegaMoEL1Config : MegaMoEPhaseConfig {
    int copy_blocks, k_tiles_per_flag;
};

struct MegaMoEL2Config : MegaMoEPhaseConfig {};

struct MegaMoEConfig {
    MegaMoEL1Config l1;
    MegaMoEL2Config l2;
};

static MegaMoEPhaseConfig make_phase_config(const deep_gemm_fp4_common::ConfigResult& config) {
    const auto [num_sms, bm, bn, bk, wm, wn, stages, smem] = config;
    return {num_sms, bm, bn, bk, wm, wn, stages, std::get<0>(smem)};
}

struct CopyTile {
    int bm, bn, bk, wm, wn, stages;
};

// K and BK count packed FP4 bytes. Select a physical stripe before converting
// to K tiles; the launch contract also limits one M block to seven publications.
static int aligned_copy_ktpf(int k, int bk) {
    DG_HOST_ASSERT(k > 0 && bk > 0 && k % bk == 0 && k % 256 == 0);
    const int initial_stripe = 512 * next_power_of_two(ceil_div(k, 1024));
    const int stripe_bytes = std::max({16 * bk, initial_stripe, std::min(k, 2048)});
    const int alignment = 512 / std::gcd(bk, 512);
    const int minimum = std::max(ceil_div(stripe_bytes, bk), ceil_div(k / bk, 7));
    return ceil_div(minimum, alignment) * alignment;
}

static bool fused_tile_fits(const MegaMoEPhaseConfig& config, int num_ranks, int num_experts_per_rank) {
    const int bm = config.block_m, bn = config.block_n;
    const int threads = (bm / config.warp_m) * (bn / config.warp_n) * 32;
    return threads >= 64 && bn >= 64 &&
        num_experts_per_rank * (num_ranks + kExpertMetadataAuxFields) *
            static_cast<int>(sizeof(uint32_t)) <= config.smem_size;
}

static std::optional<CopyTile> select_copy_tile(
    int n, int k, int num_experts_per_rank, int num_ranks, int expected_m, const MegaMoEPhaseConfig& base) {
    // DPSK V4 Pro geometry (H7168 / I3072); expert-count scopes are matched below.
    if (n == 6144 && k == 3584) {
        const bool shared_tile = num_ranks == 16 && num_experts_per_rank == 24 && expected_m >= 115 && expected_m <= 117;
        const bool family = base.block_m == 128 && base.block_n == 256 && base.block_k == 64;
        if (expected_m >= 16 && family && !shared_tile) {
            constexpr CopyTile wide{128, 512, 128, 64, 64, 3};
            constexpr CopyTile large{256, 256, 128, 64, 64, 3};
            return expected_m <= 117 ? wide : large;
        }
    }
    return std::nullopt;
}

static int select_l1_copy_blocks(
    int expected_m, int n, int k, int num_experts_per_rank, const MegaMoEPhaseConfig& config, int k_tiles_per_flag) {
    const int warps = (config.block_m / config.warp_m) * (config.block_n / config.warp_n);
    const int cta_slots = config.num_sms * std::min(4, 16 / warps);
    const int m_tiles = ceil_div(expected_m, config.block_m), warp_rows = m_tiles * config.warp_m;
    const int64_t copy_tasks = int64_t(num_experts_per_rank) * m_tiles;
    const int64_t stripe_cost = int64_t(8) * (config.block_m / warps)
        * (warp_rows + std::min(warp_rows, expected_m - 1)) / warp_rows + 12;
    const int64_t copy_cost = ceil_div<int64_t>(int64_t(expected_m - 1) * ceil_div(k, 512), m_tiles * warps)
        + stripe_cost * ceil_div(k, config.block_k * k_tiles_per_flag);
    const int64_t gemm_cost = int64_t(config.warp_m) * config.warp_n * k / (16 * 1024);

    int best_ncb = 3;
    int64_t best_cost = INT64_MAX;
    // Visit the existing candidates first; NCB2 must strictly reduce cost.
    for (const int ncb : {3, 4, 6, 2}) {
        if (ncb >= config.num_sms) continue;
        const int64_t copy = ceil_div<int64_t>(copy_tasks, ncb) * copy_cost;
        const int64_t compute = ceil_div<int64_t>(
            copy_tasks * ceil_div(n, config.block_n), cta_slots - ncb) * gemm_cost;
        const bool low_headroom = copy * 3 > compute * 2;
        // For Qwen3.8 (H8192/I2048), keep NCB6 when NCB3 has insufficient supply headroom.
        if (n == 4096 && k == 4096 && config.num_sms > 6 && ncb == 3 && low_headroom) return 6;
        // Two copy CTAs need headroom for unmodeled peer-load and publication latency.
        if (ncb == 2 && low_headroom) continue;

        const int64_t cost = std::max(copy, compute);
        if (cost >= best_cost) continue;
        best_ncb = ncb;
        best_cost = cost;
    }
    return best_ncb;
}

static MegaMoEL1Config get_l1_config(
    int shape_m, int n, int k, int num_experts_per_rank, int num_sms, int num_ranks, int expected_m) {
    auto config = make_phase_config(deep_gemm_fp4_common::get_best_configs(
        shape_m, expected_m, n, k, num_experts_per_rank, num_sms, false, false,
        GemmType::MegaMoE, 256, 32));
    // Metadata preprocessing reuses the GEMM shared allocation before compute.
    config.smem_size = std::max(config.smem_size,
        num_experts_per_rank * (num_ranks + kExpertMetadataAuxFields) * static_cast<int>(sizeof(uint32_t)));
    DG_HOST_ASSERT(fused_tile_fits(config, num_ranks, num_experts_per_rank));

    const auto record = select_copy_tile(n, k, num_experts_per_rank, num_ranks, expected_m, config);
    if (record) {
        const auto& t = *record;
        const MegaMoEPhaseConfig candidate{config.num_sms, t.bm, t.bn, t.bk,
            t.wm, t.wn, t.stages, std::get<0>(deep_gemm_fp4_common::get_smem_config_fp4(
                t.stages, t.bm, t.bn, t.wm, t.wn, t.bk, n, false, false))};
        if (fused_tile_fits(candidate, num_ranks, num_experts_per_rank)) {
            config = candidate;
        }
    }
    const int k_tiles_per_flag = aligned_copy_ktpf(k, config.block_k);
    return {config, select_l1_copy_blocks(expected_m, n, k, num_experts_per_rank, config, k_tiles_per_flag), k_tiles_per_flag};
}

static MegaMoEL2Config get_l2_config(
    int shape_m, int expected_m, int n, int k, int num_experts_per_rank, int num_sms) {
    auto config = make_phase_config(deep_gemm_fp4_common::get_best_configs(
        shape_m, expected_m, n, k, num_experts_per_rank, num_sms, false, false,
        GemmType::MegaMoE, 256, 128));
    // BN128 uses ordinary scatter; BN256 uses remote bulk scatter.
    // The owner protocol lets each reader match its writer's layout.
    const int old_wn = config.warp_n;
    config.warp_n = std::max(config.warp_n, 32);
    if (config.warp_n != old_wn)
        config.smem_size = std::get<0>(deep_gemm_fp4_common::get_smem_config_fp4(
            config.num_stages, config.block_m, config.block_n, config.warp_m, config.warp_n,
            config.block_k, n, false, false));
    DG_HOST_ASSERT(config.block_n == 128 || config.block_n == 256);
    // Scatter needs two accumulator tiles in addition to the mainloop resource contract.
    config.smem_size = std::max(config.smem_size, 2 * (config.block_m / config.warp_m) *
                                16 * config.block_n * static_cast<int>(sizeof(float)));
    return {config};
}

static MegaMoEConfig get_mega_moe_config(
    int num_ranks, int num_experts_per_rank, int num_max_tokens_per_rank,
    int hidden, int intermediate_hidden, int expected_m, int num_sms) {
    return {
        get_l1_config(expected_m * num_experts_per_rank,
            intermediate_hidden * 2, hidden / 2, num_experts_per_rank,
            num_sms, num_ranks, expected_m),
        get_l2_config(num_ranks * num_max_tokens_per_rank, expected_m,
            hidden, intermediate_hidden / 2, num_experts_per_rank,
            num_sms)
    };
}

inline int combine_num_threads(int tokens, int hidden) {
    if (tokens >= 512) return 256;
    if (tokens <= 64 && hidden > 4096) return 1024;
    return 512;
}

using PPUW4ConfigTuple = std::tuple<int, int, int, int, int, int, int, int>;
inline deep_gemm_w4a16_common::W4A16Config ppu_w4_config(int em, int n, int k,
                                                         std::optional<PPUW4ConfigTuple> requested) {
    using namespace deep_gemm_w4a16_common;
    if (requested) {
        const auto& [ns, bm, bn, bk, wm, wn, wk, s] = *requested;
        return {ns, bm, bn, bk, wm, wn, wk, s, 1};
    }
    return get_best_configs(em, n, k, 1, get_num_sms(), GemmType::MGroupedNoPad,
                           W4A16Type::mxfp4_e8m0_mma, /*single_k_warp=*/true, /*full_n_tiles=*/true);
}

// Transport geometry is independent of the GEMM1 compute tile.
struct DispatchConfig {
    int block_m = 0;
    int num_ranks = 0, num_experts = 0, num_experts_per_rank = 0;
    int num_topk = 0, nmtpr = 0, token_bytes = 0, sf_bytes = 0;
    int pull_ctas = 0, pull_group_size = 1, pull_slots = 0;
};

struct Gemm2MetadataConfig {
    int block_m = 0;
    bool valid_rows = true;
};

struct Gemm1Config {
    DispatchConfig dispatch;
    MegaMoEPhaseConfig tile{};
    int n = 0, k = 0, warp_k = 0;
    int num_ctas = 0, num_threads = 0, master_experts = 0;
    Gemm2MetadataConfig metadata;
    bool fast_math = true, swiglu_oai = false;
    w4fa8::FragmentPolicy fragment = w4fa8::FragmentPolicy::FullK;
};

struct PreDispatchConfig {
    int block_threads = 0;
    int hidden = 0;
    int num_ctas = 0;
    int num_topk = 0;
    bool int32_indices = false;
};

struct ScatterConfig {
    int num_ranks = 1;
    int block_k = 0;
    int block_m = 0;
    int block_n = 0;
    int scratch_rows = 0;
    int smem_size = 0;
    bool check_packed_routes = false;
    int k = 0;
    int n = 0;
    int num_groups = 0;
    int num_master_groups = 0;
    int num_stages = 0;
    int num_threads = 0;
    bool use_padded_valid_rows = false;
    int warp_m = 0;
    int warp_n = 0;
    int num_sms = 0, warp_k = 0;
    w4fa8::FragmentPolicy fragment = w4fa8::FragmentPolicy::FullK;
};

inline int fp8_gemm1_smem_size(const Gemm1Config& c, bool blockwise) {
    if (blockwise) {
        // This fixed BK128/four-stage collective includes aligned block scales.
        const int compute = std::get<0>(deep_gemm_fp8_common::get_smem_config(
            c.tile.num_stages, c.dispatch.token_bytes, c.tile.block_m, c.tile.block_n, c.tile.block_k));
        return std::max({compute, c.num_threads / 32 * c.dispatch.token_bytes, c.dispatch.num_experts * 4});
    }
    // GEMM1 uses the mainloop storage, without ordinary GEMM's BF16 output tile.
    const int compute = c.tile.num_stages * (c.tile.block_m + c.tile.block_n) * c.tile.block_k;
    const int dispatch = ceil_div(c.dispatch.num_experts * 4, 16) * 16 + c.num_threads / 32 * c.dispatch.token_bytes;
    return std::max(compute, dispatch);
}

inline int64_t w4a16_scatter_smem_size(const deep_gemm_w4a16_common::W4A16Config& c) {
    const int64_t scratch = int64_t(c.block_m) * (c.block_n / 2 + 4) * sizeof(uint32_t);
    return std::max<int64_t>(deep_gemm_w4a16_common::get_smem_config(c, true, 1, 32), scratch);
}

inline int64_t w4a8_scatter_smem_size(const w4fa8::W4A8Config& c, int k) {
    const int64_t scratch = int64_t(c.block_m) * (c.block_n / 2 + 4) * sizeof(uint32_t);
    return std::max<int64_t>(w4fa8::get_smem_size(c, k), scratch);
}

inline int channel_pull_group_size(const Gemm1Config& c, int expected_rows) {
    constexpr int default_teams = 3;
    const bool one_batch = std::min(expected_rows, c.tile.block_m) <= c.num_threads / 32;
    const bool needs_more_blocks = default_teams * ceil_div(c.n, c.tile.block_n) <
                                   c.num_ctas - c.dispatch.pull_ctas;
    // Avoid splitting a single warp batch when three ready M blocks cannot
    // feed the compute grid. Larger blocks retain cooperative row partitioning.
    return one_batch && needs_more_blocks ? 1 : c.dispatch.pull_ctas / default_teams;
}

using Config = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
inline TuningConfig tuning_config(const TuningWorkload& workload) {
    return select_tuning(workload);
}

inline int threads(int bm, int bn, int wm, int wn) {
    TORCH_CHECK(wm > 0 && wn > 0 && bm % wm == 0 && bn % wn == 0, "Invalid MegaMoE warp tile");
    return bm / wm * (bn / wn) * 32;
}
inline PreDispatchConfig w4a16_pre_dispatch_config(int hidden, int topk) {
    return {256, hidden, get_num_sms(), topk};
}
inline int dispatch_pull_slots(int num_launch_threads, int token_bytes, int scale_bytes) {
    // Match the device pull-staging limit; pull and GEMM reuse this storage.
    constexpr int max_staging_bytes = 256 * 1024;
    TORCH_CHECK(num_launch_threads > 0 && num_launch_threads % 32 == 0);
    TORCH_CHECK(token_bytes > 0 && scale_bytes >= 0 && token_bytes <= max_staging_bytes - scale_bytes,
                "MegaMoE pull staging must fit at least one token and its scales in 256 KiB");
    return std::min({16, num_launch_threads / 32, max_staging_bytes / (token_bytes + scale_bytes)});
}
// Shared transport dimensions; payload sizes are selected by the precision backend.
inline Gemm1Config make_gemm1_config(int ranks, int experts, int topk, int capacity, int token_bytes,
                                     int scale_bytes) {
    TORCH_CHECK(ranks > 0 && experts > 0 && experts % ranks == 0 && topk > 0 && capacity >= 0);
    TORCH_CHECK(token_bytes > 0 && scale_bytes >= 0);
    Gemm1Config c;
    c.dispatch.num_ranks = ranks;
    c.tile.num_sms = get_num_sms();
    c.num_ctas = c.tile.num_sms;
    c.dispatch.num_experts = experts;
    c.dispatch.num_experts_per_rank = experts / ranks;
    c.dispatch.num_topk = topk;
    c.dispatch.nmtpr = capacity;
    c.dispatch.token_bytes = token_bytes;
    c.dispatch.sf_bytes = scale_bytes;
    return c;
}

inline int fp8_dispatch_block_m(std::optional<int> requested, bool blockwise) {
    const int bm = requested.value_or(blockwise ? 128 : 256);
    if (blockwise) {
        TORCH_CHECK(bm == 128, "FP8 blockwise dispatch requires block_m=128; "
                               "the GEMM1 compute tile is selected separately");
    } else {
        TORCH_CHECK(bm == 128 || bm == 256, "FP8 channelwise dispatch/GEMM1 requires block_m=128 or 256");
    }
    return bm;
}

inline Gemm1Config make_fp8_gemm1_config(int ranks, int experts, int topk, int hidden, int capacity,
                                          std::optional<int> requested_block_m, bool blockwise) {
    TORCH_CHECK(hidden > 0 && hidden % (blockwise ? 128 : 16) == 0);
    const int bm = fp8_dispatch_block_m(requested_block_m, blockwise);
    auto c = make_gemm1_config(ranks, experts, topk, capacity, hidden, blockwise ? 4 * hidden / 128 : 4);
    c.num_threads = blockwise ? 256 : 512;
    c.dispatch.block_m = bm;
    return c;
}
inline void fp8_gemm1_config(Gemm1Config& c, int tokens, const Gemm1Tuning& tuning,
                             std::optional<int> block_m = std::nullopt) {
    // Pull reads unpadded source rows; only this rank's landing pool and GEMM
    // metadata use BM. Workspace offsets and peer handshakes are independent
    // of BM, so the same allocation supports per-call and per-rank selection.
    const auto requested = block_m ? block_m : tuning.block_m;
    const std::vector<int> allowed = requested ? std::vector<int>{fp8_dispatch_block_m(requested, false)}
                                              : std::vector<int>{256, 128, 64, 32, 16};
    const int expected = mega::expected_expert_rows(tokens, c.dispatch.num_topk, c.dispatch.num_experts_per_rank,
                                                      c.dispatch.num_ranks, c.dispatch.nmtpr);
    const auto supported = [&](const Tile& tile) {
        const auto [bm, bn, bk, wm, wn, stages] = tile;
        return (bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256) && (bn == 128 || bn == 256) &&
               c.n % bn == 0 && (bk == 64 || bk == 128 || bk == 256) &&
               c.dispatch.token_bytes % bk == 0 && wm >= 16 && wm <= 64 && wm % 16 == 0 && bm % wm == 0 &&
               (wn == 32 || wn == 64) && stages >= 2 && stages <= 8 && threads(bm, bn, wm, wn) <= 1024 &&
               std::get<0>(deep_gemm_int8::get_smem_config(stages, c.dispatch.token_bytes, bm, bn, bk)) <= 256 * 1024;
    };
    const auto tile_of = [](const Config& config) {
        const auto [sms, bm, bn, bk, wm, wn, stages, smem] = config;
        return Tile{bm, bn, bk, wm, wn, stages};
    };
    std::optional<Tile> selected = tuning.tile;
    if (!selected) {
        // Validate external LUTs before resource arithmetic or JIT: ordinary
        // GEMM entries need not satisfy this fused epilogue's constraints.
        if (const auto lut = deep_gemm_lut::get_runtime_config(expected, c.n, c.dispatch.token_bytes,
                c.dispatch.num_experts_per_rank, GemmType::MGroupedNoPad,
                deep_gemm_lut::RuntimeLutKind::FP8Channelwise)) {
            const auto [bm, bn, bk, wm, wn, stages] = *lut;
            const Tile tile{bm, bn, bk, wm, wn, stages};
            if (std::find(allowed.begin(), allowed.end(), bm) != allowed.end() && supported(tile))
                selected = tile;
        }
    }
    if (!selected) {
        auto generic = deep_gemm_int8::get_best_configs(expected, c.n, c.dispatch.token_bytes,
            c.dispatch.num_experts_per_rank, c.tile.num_sms, GemmType::MGroupedNoPad, 256,
            std::nullopt, nullptr, 256, &allowed);
        if (!supported(tile_of(generic)))
            generic = deep_gemm_int8::get_best_configs_ppu1v5(expected, c.n, c.dispatch.token_bytes,
                c.dispatch.num_experts_per_rank, c.tile.num_sms, GemmType::MGroupedNoPad, 256, 256, &allowed);
        selected = tile_of(generic);
    }
    const auto& tile = *selected;
    TORCH_CHECK(supported(tile), "Unsupported FP8 channelwise GEMM1 tile");
    const auto [tm, tn, tk, twm, twn, ts] = tile;
    c.dispatch.block_m = tm;
    c.tile.block_m = c.dispatch.block_m;
    c.tile.block_n = tn;
    c.tile.block_k = tk;
    c.tile.warp_m = twm;
    c.tile.warp_n = twn;
    c.tile.num_stages = ts;
    c.num_threads = threads(c.tile.block_m, c.tile.block_n, c.tile.warp_m, c.tile.warp_n);
    c.dispatch.pull_ctas = tuning.pull_ctas.value_or(6);
    TORCH_CHECK(c.dispatch.pull_ctas > 0 && c.dispatch.pull_ctas < c.tile.num_sms && c.dispatch.pull_ctas % 3 == 0,
                "FP8 channelwise pull CTAs must be a positive multiple of 3 below num_sms");
}

inline int blockwise_compute_m(std::optional<int> expected_m, int groups, int n, int sms) {
    TORCH_CHECK(groups > 0 && n > 0 && sms > 0 && (!expected_m || *expected_m >= 0));
    if (!expected_m || *expected_m <= 128)
        return 128;
    const int m128 = ceil_div(*expected_m, 128), m192 = ceil_div(*expected_m, 192);
    if (groups * m192 * ceil_div(n, 256) < sms && m128 > m192)
        return 128;
    // Empirical fallback: BM192 may trade up to 12.5% more padded rows for
    // lower per-row overhead. This is a tuning budget, not a kernel constraint.
    constexpr int padding_budget_numerator = 9, padding_budget_denominator = 8;
    return int64_t(m192) * 192 * padding_budget_denominator <= int64_t(m128) * 128 * padding_budget_numerator ? 192
                                                                                                              : 128;
}

inline void fp8_blockwise_gemm1_config(Gemm1Config& c, int tokens, int n, const Gemm1Tuning& tuning) {
    c.dispatch.block_m = fp8_dispatch_block_m(c.dispatch.block_m, true);
    const int groups = c.dispatch.num_experts_per_rank;
    const int em = mega::expected_expert_rows(tokens, c.dispatch.num_topk, groups, c.dispatch.num_ranks, c.dispatch.nmtpr);
    c.tile.block_m = tuning.block_m.value_or(blockwise_compute_m(em, groups, n, c.tile.num_sms));
    TORCH_CHECK(c.tile.block_m == 128 || c.tile.block_m == 32 || c.tile.block_m == 192,
                "FP8 blockwise GEMM1 requires compute block_m=32, 128 or 192");
    TORCH_CHECK(!tuning.tile, "blockwise GEMM1 accepts block_m, not a full tile override");
    c.tile.block_n = 256;
    c.tile.block_k = 128;
    c.tile.warp_m = std::max(16, c.tile.block_m / 4);
    c.tile.warp_n = c.tile.block_n / 4;
    c.tile.num_stages = 4;
    c.num_threads = threads(c.tile.block_m, c.tile.block_n, c.tile.warp_m, c.tile.warp_n);
    // Bound dedicated pull CTAs to a quarter of the grid. Wider GEMMs leave
    // more computation per pulled row, so they need a smaller pull fraction.
    constexpr int pull_grid_divisor = 4, pull_balance_columns = 512;
    const int limit = std::max(1, c.tile.num_sms / pull_grid_divisor);
    int pulls = std::min(limit, std::max(1, ceil_div(c.tile.num_sms * pull_balance_columns, n + pull_balance_columns)));
    pulls = std::min(limit, pulls * std::max(ceil_div(c.dispatch.num_ranks, 32), 2));
    c.dispatch.pull_ctas = tuning.pull_ctas.value_or(pulls);
    TORCH_CHECK(c.dispatch.pull_ctas > 0 && c.dispatch.pull_ctas < c.tile.num_sms);
}

inline ScatterConfig fp8_blockwise_scatter_config(int n, int k, int groups, int masters, std::optional<int> expected,
                                                  const Gemm2Tuning& tuning) {
    ScatterConfig c;
    c.n = n;
    c.k = k;
    c.num_groups = groups;
    c.num_master_groups = masters;
    c.block_m = blockwise_compute_m(expected, groups, n, get_num_sms());
    c.num_stages = std::min(4, std::max(2, k / 128));
    if (tuning.tile) {
        const auto& t = *tuning.tile;
        TORCH_CHECK(t[1] == 256 && t[2] == 128 && t[3] == std::max(16, t[0] / 4) && t[4] == 64,
                    "unsupported blockwise GEMM2 tile");
        c.block_m = t[0];
        c.num_stages = t[5];
    }
    TORCH_CHECK((c.block_m == 32 || c.block_m == 128 || c.block_m == 192) && c.num_stages >= 2 && c.num_stages <= 4);
    c.block_n = 256;
    c.block_k = 128;
    c.warp_m = std::max(16, c.block_m / 4);
    c.warp_n = 64;
    c.num_threads = threads(c.block_m, c.block_n, c.warp_m, c.warp_n);
    // Mainloop and native epilogue share a prefix; scatter adds a disjoint,
    // 128-byte-aligned BF16 tile. Prefer a full tile when the arena fits.
    const int prefix = std::get<0>(deep_gemm_fp8_common::get_smem_config(
        c.num_stages, k, c.block_m, c.block_n, c.block_k));
    const auto smem_size = [&](int rows) {
        return ceil_div(prefix, 128) * 128 + ceil_div(rows * c.block_n * 2, 128) * 128;
    };
    constexpr int smem_capacity = 256 * 1024;
    c.scratch_rows = smem_size(c.block_m) <= smem_capacity ? 0 : 32;
    c.smem_size = smem_size(c.scratch_rows ? c.scratch_rows : c.block_m);
    TORCH_CHECK(c.smem_size <= smem_capacity, "FP8 blockwise scatter exceeds shared-memory capacity");
    return c;
}

inline Gemm1Config w4a16_gemm1_config(Gemm1Config c, int n, int expected,
                                        const Gemm1Tuning& tuning) {
    c.n = n;
    // Transport padding and the MMA tile have independent purposes. Preserve
    // the selected compute tile when configuring transport.
    const auto generic = ppu_w4_config(expected, n, c.dispatch.token_bytes / 2, std::nullopt);
    const auto t = tuning.tile.value_or(Tile{generic.block_m, generic.block_n, generic.block_k,
                                             generic.warp_m, generic.warp_n, generic.num_stages});
    c.tile.block_m = t[0];
    c.tile.block_n = t[1];
    c.tile.block_k = t[2];
    c.tile.warp_m = t[3];
    c.tile.warp_n = t[4];
    c.warp_k = t[2];
    c.tile.num_stages = t[5];
    c.num_threads = threads(c.tile.block_m, c.tile.block_n, c.tile.warp_m, c.tile.warp_n);
    c.dispatch.pull_slots = dispatch_pull_slots(c.num_threads, c.dispatch.token_bytes, c.dispatch.sf_bytes);
    const deep_gemm_w4a16_common::W4A16Config mma{
        c.tile.num_sms, t[0], t[1], t[2], t[3], t[4], t[2], t[5], 1};
    const int smem = std::max(deep_gemm_w4a16_common::get_smem_config(mma, true, 1, 32),
                              c.dispatch.pull_slots * (c.dispatch.token_bytes + c.dispatch.sf_bytes));
    c.tile.smem_size = std::max(smem, c.dispatch.num_experts * 4);
    const auto prop = device_runtime->get_prop();
    // Match ordinary GEMM's eight-CTA occupancy cap. These resource bounds
    // avoid under-launching small tiles; compiled register use is checked later.
    const int resident = std::min({8, prop->maxThreadsPerMultiProcessor / c.num_threads,
                                   int(prop->sharedMemPerMultiprocessor / smem)});
    TORCH_CHECK(resident > 0, "W4A16 GEMM1 cannot keep a CTA resident");
    c.num_ctas = c.tile.num_sms * resident;
    TORCH_CHECK(c.num_ctas > 1, "W4A16 GEMM1 requires pull and compute CTAs");
    const int pull_groups = std::max(1, c.tile.num_sms / 4);
    c.dispatch.pull_ctas = std::clamp(tuning.pull_ctas.value_or(pull_groups), 1, c.num_ctas - 1);
    // A larger measured pull budget cooperates on each block rather than
    // increasing the number of independent transport streams.
    c.dispatch.pull_group_size = int(ceil_div(c.dispatch.pull_ctas, pull_groups));
    return c;
}

inline std::pair<Config, ScatterConfig>
channel_scatter_policy(int expected_m, int n, int k, int groups, int masters, int capacity, bool padded,
                       std::optional<Config> requested, std::optional<std::string> kernel_override,
                       std::optional<int> sms_override,
                       const Gemm2Tuning& tuning = {}) {
    TORCH_CHECK(expected_m >= 0 && groups > 0);
    const int em = std::max(1, expected_m), physical_sms = get_num_sms();
    const int sms = sms_override.value_or(physical_sms);
    TORCH_CHECK(sms > 0 && sms <= physical_sms);
    std::string kernel = kernel_override.value_or("Default");
    TORCH_CHECK(kernel == "Default",
                "FP8 channelwise MegaMoE supports Default scatter only; "
                "OverlapPrologue does not implement replica-aware kernel parameters");
    if (!requested && tuning.tile && (!tuning.capacity || *tuning.capacity == capacity) &&
        (!tuning.requires_no_replicas || masters == groups)) {
        const auto& t = *tuning.tile;
        requested = Config{sms,  t[0], t[1], t[2],
                           t[3], t[4], t[5], deep_gemm_int8::get_smem_config(t[5], k, t[0], t[1], t[2], 1)};
    }
    if (!requested)
        requested = deep_gemm_int8::get_best_configs(em, n, k, groups, sms, GemmType::MGroupedNoPad, 256,
                                                     std::nullopt, nullptr, /*fixed_block_n=*/256);
    const auto [ns, bm, bn, bk, wm, wn, stages, smem] = *requested;
    TORCH_CHECK(bn == 256 && n % bn == 0, "channelwise scatter requires complete 256-column tiles");
    ScatterConfig c;
    c.n = n;
    c.k = k;
    c.block_m = bm;
    c.block_n = bn;
    c.block_k = bk;
    c.warp_m = wm;
    c.warp_n = wn;
    c.num_groups = groups;
    c.num_master_groups = masters;
    c.num_stages = stages;
    // Default scatter reuses a union of the mainloop and BF16 output tile.
    c.smem_size = std::get<0>(smem);
    c.use_padded_valid_rows = padded;
    return {*requested, c};
}

inline auto w4a16_scatter_config(int em, int n, int k, std::optional<PPUW4ConfigTuple> configs,
                                 const Gemm2Tuning& tuning = {}) {
    TORCH_CHECK(n > 0 && n % 256 == 0, "W4A16 scatter requires N divisible by 256");
    if (!configs && tuning.tile) {
        const auto& t = *tuning.tile;
        configs = PPUW4ConfigTuple{get_num_sms(), t[0], t[1], t[2], t[3], t[4], t[2], t[5]};
    }
    auto c = ppu_w4_config(em, n, k, configs);
    TORCH_CHECK(c.block_n > 0 && c.block_n % 256 == 0 && n % c.block_n == 0,
                "W4A16 scatter requires complete 256-column tiles");
    TORCH_CHECK(w4a16_scatter_smem_size(c) <= 256 * 1024,
                "W4A16 scatter tile exceeds shared memory capacity");
    return c;
}

inline int fp8_gemm1_ctas(const Gemm1Config& c, int tokens, int pulls) {
    const int expected = mega::expected_expert_rows(tokens, c.dispatch.num_topk, c.dispatch.num_experts_per_rank,
                                                      c.dispatch.num_ranks, c.dispatch.nmtpr);
    const int64_t jobs = int64_t(c.dispatch.num_experts_per_rank) * ceil_div(expected, c.tile.block_m) *
                         ceil_div(c.n, c.tile.block_n);
    const int work = std::min<int64_t>(4, ceil_div(jobs + pulls, int64_t(c.tile.num_sms)));
    // Match the G1 thread/work budget used by W4A4. Compiled occupancy is the
    // final cap because every CTA participates in the dispatch grid barriers.
    return c.tile.num_sms * std::clamp(std::min(512 / c.num_threads, work), 1, 4);
}

inline w4fa8::NativeKernelConfig w4a8_gemm_config(int expected, int n, int k, int groups,
                                                 const std::optional<Tile>& tile) {
    // Direct activation/scatter epilogues consume complete N256 tiles.
    // Keep these constraints inside selection, including shared-memory fitting.
    auto c = w4fa8::get_default_tile(expected, n, k, groups, GemmType::MGroupedNoPad, 256, false);
    if (!tile && expected <= std::numeric_limits<int>::max() / groups && w4fa8::supports_native_strategy() &&
        w4fa8::wide_tile_eligible(groups, expected * groups, n, k, GemmType::MGroupedNoPad))
        c = w4fa8::default_wide_config(expected, k, GemmType::MGroupedNoPad);
    if (tile) {
        const auto& [bm, bn, bk, wm, wn, stages] = *tile;
        c.block_m = bm; c.block_n = bn; c.block_k = bk;
        c.warp_m = wm; c.warp_n = wn; c.warp_k = bk; c.num_stages = stages;
    }
    c.n_expand = 1;
    const auto spec = w4fa8::resolve_native_config(c, n, k);
    w4fa8::check_native_kernel(spec, n, k, groups, GemmType::MGroupedNoPad);
    TORCH_CHECK(c.block_n % 256 == 0 && n % c.block_n == 0 && c.warp_k == c.block_k,
                "W4A8 MegaMoE requires complete 256-column tiles and one K warp");
    TORCH_CHECK(w4a8_scatter_smem_size(c, k) <= 256 * 1024,
                "W4A8 MegaMoE tile exceeds shared memory capacity");
    return spec;
}

struct MegaMoEPrecisionConfigs {
    Gemm1Config l1;
    ScatterConfig l2;
    int expected_m = 0;
};

inline const MegaMoEPrecisionConfigs& get_mega_moe_precision_configs(
    Precision precision, int ranks, int experts, int topk, int hidden, int intermediate,
    int capacity, int tokens, int masters, const std::string& activation, int device) {
    using Key = std::tuple<Precision, int, int, int, int, int, int, int, int, std::string, int, int>;
    static thread_local std::map<Key, MegaMoEPrecisionConfigs> cache;
    const Key key{precision, ranks, experts, topk, hidden, intermediate,
                  capacity, tokens, masters, activation, device, get_num_sms()};
    if (const auto found = cache.find(key); found != cache.end())
        return found->second;
    const auto tuning = tuning_config({precision, ranks, experts, topk, hidden, intermediate, tokens, activation});
    const int groups = experts / ranks;
    const int expected = mega::expected_expert_rows(tokens, topk, groups, ranks, capacity);
    MegaMoEPrecisionConfigs result;
    result.expected_m = expected;
    if (precision == Precision::W4A16) {
        auto c = make_gemm1_config(ranks, experts, topk, capacity, hidden * 2, 0);
        c.dispatch.block_m = tuning.buffer.block_m.value_or(64);
        result.l1 = w4a16_gemm1_config(c, 2 * intermediate, expected, tuning.gemm1);
        const auto tile = w4a16_scatter_config(expected, hidden, intermediate, std::nullopt, tuning.gemm2);
        auto& g2 = result.l2;
        g2.num_sms = tile.num_sms;
        g2.block_m = tile.block_m; g2.block_n = tile.block_n; g2.block_k = tile.block_k;
        g2.warp_m = tile.warp_m; g2.warp_n = tile.warp_n; g2.warp_k = tile.warp_k;
        g2.num_stages = tile.num_stages;
        g2.num_threads = deep_gemm_w4a16_common::get_num_threads(tile, true);
        g2.smem_size = w4a16_scatter_smem_size(tile);
    } else if (precision == Precision::W4A8) {
        const auto g1 = w4a8_gemm_config(expected, 2 * intermediate, hidden, groups, tuning.gemm1.tile);
        auto c = make_gemm1_config(ranks, experts, topk, capacity, hidden, sizeof(float));
        c.dispatch.block_m = g1.tile.block_m;
        c.tile.block_m = g1.tile.block_m; c.tile.block_n = g1.tile.block_n; c.tile.block_k = g1.tile.block_k;
        c.tile.warp_m = g1.tile.warp_m; c.tile.warp_n = g1.tile.warp_n; c.warp_k = g1.tile.warp_k;
        c.tile.num_stages = g1.tile.num_stages;
        c.fragment = g1.fragment;
        c.num_threads = w4fa8::get_num_threads(g1.tile);
        const int64_t pull_smem = int64_t(c.num_threads / 32) * hidden + ((int64_t(experts) * 4 + 15) / 16 * 16);
        TORCH_CHECK(pull_smem <= 256 * 1024, "W4A8 pull staging exceeds shared memory capacity");
        c.dispatch.pull_ctas = tuning.gemm1.pull_ctas.value_or(6);
        TORCH_CHECK(c.dispatch.pull_ctas > 0 && c.dispatch.pull_ctas < c.tile.num_sms && c.dispatch.pull_ctas % 3 == 0,
                    "W4A8 pull CTAs must be a positive multiple of 3 below num_sms");
        c.tile.smem_size = std::max<int64_t>(w4fa8::get_smem_size(g1.tile, hidden), pull_smem);
        const int resident = int(device_runtime->get_prop()->sharedMemPerMultiprocessor / c.tile.smem_size);
        c.num_ctas = c.tile.num_sms * std::clamp(resident, 1, 8);
        result.l1 = c;
        const auto g2 = w4a8_gemm_config(expected, hidden, intermediate, groups, tuning.gemm2.tile);
        auto& scatter = result.l2;
        scatter.num_sms = g2.tile.num_sms;
        scatter.block_m = g2.tile.block_m; scatter.block_n = g2.tile.block_n; scatter.block_k = g2.tile.block_k;
        scatter.warp_m = g2.tile.warp_m; scatter.warp_n = g2.tile.warp_n; scatter.warp_k = g2.tile.warp_k;
        scatter.num_stages = g2.tile.num_stages;
        scatter.num_threads = w4fa8::get_num_threads(g2.tile);
        scatter.smem_size = w4a8_scatter_smem_size(g2.tile, intermediate);
        scatter.fragment = g2.fragment;
    } else {
        const bool blockwise = precision == Precision::FP8Blockwise;
        auto c = make_fp8_gemm1_config(ranks, experts, topk, hidden, capacity, std::nullopt, blockwise);
        c.n = 2 * intermediate;
        c.master_experts = masters;
        if (blockwise) {
            fp8_blockwise_gemm1_config(c, tokens, 2 * intermediate, tuning.gemm1);
            result.l2 = fp8_blockwise_scatter_config(hidden, intermediate, groups, masters, expected, tuning.gemm2);
        } else {
            fp8_gemm1_config(c, tokens, tuning.gemm1);
            result.l2 = channel_scatter_policy(expected, hidden, intermediate, groups, masters, capacity, true,
                std::nullopt, std::nullopt, std::nullopt, tuning.gemm2).second;
        }
        c.tile.smem_size = fp8_gemm1_smem_size(c, blockwise);
        c.num_ctas = fp8_gemm1_ctas(c, tokens, c.dispatch.pull_ctas);
        result.l1 = c;
        result.l2.num_sms = c.tile.num_sms;
    }
    auto& g1 = result.l1;
    g1.n = 2 * intermediate;
    g1.k = hidden;
    auto& g2 = result.l2;
    g2.n = hidden; g2.k = intermediate;
    g2.num_ranks = ranks; g2.num_groups = groups; g2.num_master_groups = masters;
    if (precision == Precision::FP8Channelwise || precision == Precision::W4A8)
        g1.dispatch.pull_group_size = channel_pull_group_size(g1, expected);
    if (groups >= 128) {
        g1.metadata.block_m = g2.block_m;
        g1.metadata.valid_rows = precision == Precision::FP8Blockwise ||
            (precision == Precision::FP8Channelwise && g2.use_padded_valid_rows);
    }
    return cache.emplace(key, std::move(result)).first->second;
}

} // namespace deep_gemm::mega
