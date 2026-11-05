#pragma once
#include <deep_gemm/layout/routed_moe.cuh>

#include "common_fp4.hpp"
#include "common_fp8.hpp"
#include "common_int8.hpp"
#include "mega_moe_fp8_w4a16_tuning.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <optional>

namespace deep_gemm::mega_moe_config::fp8_w4a16 {
inline int combine_num_threads(int tokens, int hidden) {
    if (tokens >= 512) return 256;
    if (tokens <= 64 && hidden > 4096) return 1024;
    return 512;
}

struct DispatchConfig {
    bool fast_math = true;
    // Dispatch/padding granularity. FP8 blockwise chooses its GEMM1 tile separately.
    int block_m = 0;
    float activation_clamp = -1;
    int gemm1_block_k = 0;
    int gemm1_block_n = 0;
    int gemm1_n = 0;
    int gemm1_compute_m = 128; // FP8 GEMM1 tile; channelwise shares the dispatch granularity.
    int gemm1_stages = 0;
    int gemm1_warp_m = 0;
    int gemm1_warp_n = 0;
    int gemm2_block_m = 0; // Nonzero when GEMM1 publishes GEMM2's valid-tile list.
    int master_experts = 0;
    int nmtpr = 0;
    int num_experts = 0;
    int num_experts_per_rank = 0;
    int pull_ctas = 0;
    int pull_group_size = 1;
    int num_ranks = 0;
    int num_sms = 0;
    int num_ctas = 0;
    int num_threads = 0;
    int num_topk = 0;
    int pull_slots = 0;
    int sf_bytes = 0;
    float swiglu_alpha = 0;
    bool swiglu_oai = false;
    int token_bytes = 0;
    int w4_block_k = 0;
    int w4_block_m = 0;
    int w4_block_n = 0;
    int w4_stages = 0;
    int w4_warp_k = 0;
    int w4_warp_m = 0;
    int w4_warp_n = 0;
};

struct PreDispatchConfig {
    int block_threads = 0;
    int hidden = 0;
    int num_ctas = 0;
    int num_topk = 0;
};

struct ScatterConfig {
    int block_k = 0;
    int block_m = 0;
    int block_n = 0;
    int scratch_rows = 0;
    int smem_size = 0;
    bool check_packed_routes = false;
    int k = 0;
    std::string kernel_type = "Default";
    int n = 0;
    int num_groups = 0;
    int num_master_groups = 0;
    int num_stages = 0;
    int num_threads = 0;
    bool use_padded_valid_rows = false;
    int warp_m = 0;
    int warp_n = 0;
};

inline bool channel_scatter_uses_worklist(const ScatterConfig& c) {
    return c.num_groups >= 128 && c.use_padded_valid_rows && c.kernel_type == "Default";
}

inline int channel_pull_group_size(const DispatchConfig& c, int expected_rows) {
    constexpr int default_teams = 3;
    const bool one_batch = std::min(expected_rows, c.gemm1_compute_m) <= c.num_threads / 32;
    const bool needs_more_blocks = default_teams * ceil_div(c.gemm1_n, c.gemm1_block_n) <
                                   c.num_ctas - c.pull_ctas;
    // Avoid splitting a single warp batch when three ready M blocks cannot
    // feed the compute grid. Larger blocks retain cooperative row partitioning.
    return one_batch && needs_more_blocks ? 1 : c.pull_ctas / default_teams;
}

using Config = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
inline TuningConfig tuning_config(const TuningWorkload& workload) {
    return select_tuning(workload);
}

inline int threads(int bm, int bn, int wm, int wn) {
    TORCH_CHECK(wm > 0 && wn > 0 && bm % wm == 0 && bn % wn == 0, "Invalid MegaMoE warp tile");
    return bm / wm * (bn / wn) * 32;
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
inline DispatchConfig dispatch_config(int ranks, int experts, int topk, int capacity, int token_bytes,
                                      int scale_bytes) {
    TORCH_CHECK(ranks > 0 && experts > 0 && experts % ranks == 0 && topk > 0 && capacity >= 0);
    TORCH_CHECK(token_bytes > 0 && scale_bytes >= 0);
    DispatchConfig c;
    c.num_ranks = ranks;
    c.num_sms = get_num_sms();
    c.num_ctas = c.num_sms;
    c.num_experts = experts;
    c.num_experts_per_rank = experts / ranks;
    c.num_topk = topk;
    c.nmtpr = capacity;
    c.token_bytes = token_bytes;
    c.sf_bytes = scale_bytes;
    return c;
}

inline int fp8_dispatch_block_m(std::optional<int> requested, bool blockwise) {
    const int bm = requested.value_or(blockwise ? 128 : 256);
    TORCH_CHECK(bm == 128 || bm == 256, "FP8 channelwise dispatch/GEMM1 requires block_m=128 or 256");
    return bm;
}

inline DispatchConfig fp8_dispatch_config(int ranks, int experts, int topk, int hidden, int capacity,
                                          std::optional<int> requested_block_m, bool blockwise) {
    TORCH_CHECK(hidden > 0 && hidden % (blockwise ? 128 : 16) == 0);
    const int bm = fp8_dispatch_block_m(requested_block_m, blockwise);
    auto c = dispatch_config(ranks, experts, topk, capacity, hidden, blockwise ? 4 * hidden / 128 : 4);
    c.num_threads = blockwise ? 256 : 512;
    c.block_m = bm;
    return c;
}
// GEMM1 metadata and GEMM2 must agree on row padding. Output quantization
// changes the scatter epilogue, not this shared default tile selection.

inline void fp8_gemm1_config(DispatchConfig& c, int tokens, const Gemm1Tuning& tuning,
                             std::optional<int> block_m = std::nullopt) {
    // Pull reads unpadded source rows; only this rank's landing pool and GEMM
    // metadata use BM. Workspace offsets and peer handshakes are independent
    // of BM, so the same allocation supports per-call and per-rank selection.
    const auto requested = block_m ? block_m : tuning.block_m;
    const std::vector<int> allowed = requested ? std::vector<int>{fp8_dispatch_block_m(requested, false)}
                                              : std::vector<int>{256, 128};
    const int expected = std::max(1, int(ceil_div(int64_t(tokens) * c.num_topk, int64_t(c.num_experts_per_rank))));
    const auto supported = [&](const Tile& tile) {
        const auto [bm, bn, bk, wm, wn, stages] = tile;
        return (bm == 16 || bm == 32 || bm == 128 || bm == 256) && (bn == 128 || bn == 256) &&
               c.gemm1_n % bn == 0 && (bk == 64 || bk == 128 || bk == 256) &&
               c.token_bytes % bk == 0 && wm >= 16 && wm <= 64 && wm % 16 == 0 && bm % wm == 0 &&
               (wn == 32 || wn == 64) && stages >= 2 && stages <= 8 && threads(bm, bn, wm, wn) <= 1024 &&
               std::get<0>(deep_gemm_int8::get_smem_config(stages, c.token_bytes, bm, bn, bk)) <= 256 * 1024;
    };
    const auto tile_of = [](const Config& config) {
        const auto [sms, bm, bn, bk, wm, wn, stages, smem] = config;
        return Tile{bm, bn, bk, wm, wn, stages};
    };
    std::optional<Tile> selected = tuning.tile;
    if (!selected) {
        // Validate external LUTs before resource arithmetic or JIT: ordinary
        // GEMM entries need not satisfy this fused epilogue's constraints.
        if (const auto lut = deep_gemm_lut::get_runtime_config(expected, c.gemm1_n, c.token_bytes,
                c.num_experts_per_rank, GemmType::MGroupedNoPad,
                deep_gemm_lut::RuntimeLutKind::FP8Channelwise)) {
            const auto [bm, bn, bk, wm, wn, stages] = *lut;
            const Tile tile{bm, bn, bk, wm, wn, stages};
            if (std::find(allowed.begin(), allowed.end(), bm) != allowed.end() && supported(tile))
                selected = tile;
        }
    }
    if (!selected) {
        auto generic = deep_gemm_int8::get_best_configs(expected, c.gemm1_n, c.token_bytes,
            c.num_experts_per_rank, c.num_sms, GemmType::MGroupedNoPad, 256,
            std::nullopt, nullptr, 256, &allowed);
        if (!supported(tile_of(generic)))
            generic = deep_gemm_int8::get_best_configs_ppu1v5(expected, c.gemm1_n, c.token_bytes,
                c.num_experts_per_rank, c.num_sms, GemmType::MGroupedNoPad, 256, 256, &allowed);
        selected = tile_of(generic);
    }
    const auto& tile = *selected;
    TORCH_CHECK(supported(tile), "Unsupported FP8 channelwise GEMM1 tile");
    const auto [tm, tn, tk, twm, twn, ts] = tile;
    c.block_m = tm;
    c.gemm1_compute_m = c.block_m;
    c.gemm1_block_n = tn;
    c.gemm1_block_k = tk;
    c.gemm1_warp_m = twm;
    c.gemm1_warp_n = twn;
    c.gemm1_stages = ts;
    c.num_threads = threads(c.gemm1_compute_m, c.gemm1_block_n, c.gemm1_warp_m, c.gemm1_warp_n);
    c.pull_ctas = tuning.pull_ctas.value_or(6);
    TORCH_CHECK(c.pull_ctas > 0 && c.pull_ctas < c.num_sms && c.pull_ctas % 3 == 0,
                "FP8 channelwise pull CTAs must be a positive multiple of 3 below num_sms");
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
    TORCH_CHECK(kernel == "Default" || kernel == "OverlapPrologue");
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
    c.kernel_type = kernel;
    c.use_padded_valid_rows = padded;
    return {*requested, c};
}

inline int fp8_gemm1_ctas(const DispatchConfig& c, int tokens, int pulls) {
    const int expected = std::max(1, int(ceil_div(int64_t(tokens) * c.num_topk, int64_t(c.num_experts_per_rank))));
    const int64_t jobs = int64_t(c.num_experts_per_rank) * ceil_div(expected, c.gemm1_compute_m) *
                         ceil_div(c.gemm1_n, c.gemm1_block_n);
    const int work = std::min<int64_t>(4, ceil_div(jobs + pulls, int64_t(c.num_sms)));
    // Match the G1 thread/work budget used by W4A4. Compiled occupancy is the
    // final cap because every CTA participates in the dispatch grid barriers.
    return c.num_sms * std::clamp(std::min(512 / c.num_threads, work), 1, 4);
}

struct RoutedMoEConfigs {
    DispatchConfig dispatch;
    ScatterConfig scatter;
};

inline const RoutedMoEConfigs& get_routed_moe_configs(Precision precision, int ranks, int experts, int topk,
                                                      int hidden, int intermediate, int capacity, int tokens,
                                                      int masters, const std::string& activation, int device) {
    using Key = std::tuple<Precision, int, int, int, int, int, int, int, int, std::string, int, int>;
    static thread_local std::map<Key, RoutedMoEConfigs> cache;
    const Key key{precision, ranks,  experts, topk,       hidden, intermediate,
                  capacity,  tokens, masters, activation, device, get_num_sms()};
    const auto found = cache.find(key);
    if (found != cache.end())
        return found->second;
    const auto tuning = tuning_config({precision, ranks, experts, topk, hidden, intermediate, tokens, activation});
    const int groups = experts / ranks;
    const int expected = std::max(1, int(ceil_div(int64_t(tokens) * topk, int64_t(groups))));
    RoutedMoEConfigs result;
    const bool blockwise = false;
    auto c = fp8_dispatch_config(ranks, experts, topk, hidden, capacity, std::nullopt, blockwise);
    c.gemm1_n = 2 * intermediate;
    c.master_experts = masters;
    fp8_gemm1_config(c, tokens, tuning.gemm1);
    result.scatter = channel_scatter_policy(expected, hidden, intermediate, groups,
                                            masters, capacity, true, std::nullopt, std::nullopt,
                                            std::nullopt, tuning.gemm2)
                         .second;
    c.num_ctas = fp8_gemm1_ctas(c, tokens, c.pull_ctas);
    result.dispatch = c;
    result.dispatch.pull_group_size = channel_pull_group_size(result.dispatch, expected);
    if (channel_scatter_uses_worklist(result.scatter))
        result.dispatch.gemm2_block_m = result.scatter.block_m;
    return cache.emplace(key, std::move(result)).first->second;
}

} // namespace deep_gemm::mega_moe_config::fp8_w4a16
