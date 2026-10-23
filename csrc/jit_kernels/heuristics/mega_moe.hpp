#pragma once
#include <deep_gemm/layout/mega_moe_types.cuh>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include "common_fp4.hpp"
#include "common_w4a16.hpp"
#include "mega_moe_tuning.hpp"

namespace deep_gemm::mega_moe_config {
struct DispatchConfig {
    // Dispatch/padding granularity. FP8 blockwise chooses its GEMM1 tile separately.
    int block_m = 0;
    int chunk_blocks = 0;
    int clamp = 0;
    float clamp_value = 0;
    bool enable_gemm1 = false;
    bool gemm1_dynamic_queue = false;
    bool enable_remote_pull = false;
    bool enable_remote_push = false;
    bool quantize_output = false;
    bool fused_situ = false;
    bool fused_swiglu = false;
    int gemm2_block_m = 0;
    int gemm1_block_k = 0;
    int gemm1_block_n = 0;
    int gemm1_n = 0;
    int gemm1_compute_m = 128;  // FP8 GEMM1 tile; channelwise shares the dispatch granularity.
    int gemm1_stages = 0;
    int gemm1_warp_m = 0;
    int gemm1_warp_n = 0;
    int gemm_sms = 0;
    int master_experts = 0;
    int nmtpr = 0;
    int num_experts = 0;
    int num_experts_per_rank = 0;
    int num_pull_sms = 0;
    int num_ranks = 0;
    int num_sms = 0;
    int num_threads = 0;
    int num_topk = 0;
    int paired_block_pull = 0;
    int producer_warps = 0;
    int pull_slots = 0;
    int pull_sms = 0;
    mega_moe::Gemm1Epilogue gemm1_epilogue = mega_moe::Gemm1Epilogue::IdentityBf16;
    int sfb_layout_block_n = 0;
    int sf_bytes = 0;
    bool silu_then_clamp = false;
    float situ_beta = 0;
    float situ_linear_beta = 0;
    bool situ_tanh_bf16x2 = false;
    bool sm_specialization = false;
    float swiglu_alpha = 0;
    bool swiglu_oai = false;
    int token_bytes = 0;
    int triple_block_pull = 0;
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
    int num_sms = 0;
    int topk = 0;
};

struct ScatterConfig {
    int block_k = 0;
    int block_m = 0;
    int block_n = 0;
    int scratch_rows = 0;
    bool bm128_overlap_pair_staging = false;
    bool check_packed_routes = false;
    int k = 0;
    std::string kernel_type = "Default";
    bool fp8_output = false;
    int fp8_exp_shift = 0;
    mega_moe::Fp8ScaleMode fp8_scale_mode = mega_moe::Fp8ScaleMode::Fixed;
    int n = 0;
    int num_groups = 0;
    int num_master_groups = 0;
    int m_tiles_per_group = 2;
    int num_stages = 0;
    int num_threads = 0;
    bool rolling_pair_scatter = false;
    bool use_padded_valid_rows = false;
    int warp_m = 0;
    int warp_n = 0;
    int world_size = 0;
};

inline constexpr std::pair<const char*, int DispatchConfig::*> dispatch_config_fields[] = {
    {"NUM_RANKS", &DispatchConfig::num_ranks},
    {"NUM_SMS", &DispatchConfig::num_sms},
    {"NUM_THREADS", &DispatchConfig::num_threads},
    {"NUM_EXPERTS", &DispatchConfig::num_experts},
    {"NUM_EXPERTS_PER_RANK", &DispatchConfig::num_experts_per_rank},
    {"NUM_TOPK", &DispatchConfig::num_topk},
    {"BLOCK_M", &DispatchConfig::block_m},
    {"CHUNK_BLOCKS", &DispatchConfig::chunk_blocks},
    {"NMTPR", &DispatchConfig::nmtpr},
    {"TOKEN_BYTES", &DispatchConfig::token_bytes},
    {"SF_BYTES", &DispatchConfig::sf_bytes},
    {"GEMM1_N", &DispatchConfig::gemm1_n},
    {"GEMM1_BLOCK_N", &DispatchConfig::gemm1_block_n},
    {"GEMM1_BLOCK_K", &DispatchConfig::gemm1_block_k},
    {"GEMM1_WARP_M", &DispatchConfig::gemm1_warp_m},
    {"GEMM1_WARP_N", &DispatchConfig::gemm1_warp_n},
    {"GEMM1_STAGES", &DispatchConfig::gemm1_stages},
};

using Config = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
using Overrides = std::map<std::string, int>;

inline bool flag(const char* name, bool fallback = false) {
    return get_env<std::string>(name, fallback ? "1" : "0") != "0";
}
inline int integer(const char* name, int fallback) {
    return get_env<int>(name, fallback);
}

inline TuningConfig tuning_config(const TuningWorkload& workload) {
    return select_tuning(workload, device_runtime->get_prop()->name, get_num_sms(),
                         flag("DG_MOE_USE_TUNED_CONFIGS", true));
}

inline std::tuple<int, int, int> fp4_smem(int stages, int bm, int bn, int wm, int wn, int bk = 128,
                                          int bpp = 1) {
    TORCH_CHECK(stages > 0 && bm > 0 && bn > 0 && bk > 0 && bpp > 0);
    const auto [sfa, invalid_a] = deep_gemm_fp4_common::get_sf_per_stage_size(bm, bk);
    const auto [sfb, invalid_b] = deep_gemm_fp4_common::get_sf_per_stage_size(bn, bk);
    return {std::max(bm * bn * 2,
                     stages * (bm + bn) * bk * bpp + stages * (sfa + sfb) * bpp - invalid_a - invalid_b),
            128, 0};
}
inline Config fp4_config(int total_m, int m, int n, int k, int groups, int sms,
                         GemmType type = GemmType::MGroupedNoPad, int max_bn = 256) {
    auto config =
        deep_gemm_fp4_common::get_best_configs(total_m, m, n, k, groups, sms, false, false, type, max_bn);
    const auto [ns, bm, bn, bk, wm, wn, stages, smem] = config;
    std::get<7>(config) = fp4_smem(stages, bm, bn, wm, wn, bk);
    return config;
}
inline int threads(int bm, int bn, int wm, int wn) {
    TORCH_CHECK(wm > 0 && wn > 0 && bm % wm == 0 && bn % wn == 0, "Invalid MegaMoE warp tile");
    return bm / wm * (bn / wn) * 32;
}
inline int pre_dispatch_threads(int hidden) {
    for (int t = std::min(hidden / 8, 512) / 32 * 32; t >= 32; t -= 32)
        if (hidden / 8 % t == 0)
            return t;
    return 32;
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
inline void apply_gemm1_tuning(DispatchConfig& c, const Gemm1Tuning& tuning,
                               const std::optional<Overrides>& overrides, int max_bn = 0, int max_bk = 0) {
    const auto set = [&](const char* key, int DispatchConfig::* field, int value) {
        if (!overrides || !overrides->count(key))
            c.*field = value;
    };
    // A workspace may impose a smaller tile/packed-scale layout than the
    // measured profile. Use the workspace's tile selector in that case.
    const bool use_tile = tuning.tile && (!max_bn || (*tuning.tile)[1] <= max_bn) &&
                          (!max_bk || (*tuning.tile)[2] <= max_bk);
    if (use_tile) {
        const auto& t = *tuning.tile;
        set("BLOCK_M", &DispatchConfig::block_m, t[0]);
        set("GEMM1_BLOCK_N", &DispatchConfig::gemm1_block_n, t[1]);
        set("GEMM1_BLOCK_K", &DispatchConfig::gemm1_block_k, t[2]);
        set("GEMM1_WARP_M", &DispatchConfig::gemm1_warp_m, t[3]);
        set("GEMM1_WARP_N", &DispatchConfig::gemm1_warp_n, t[4]);
        set("GEMM1_STAGES", &DispatchConfig::gemm1_stages, t[5]);
    }
    if (tuning.block_m)
        set("BLOCK_M", &DispatchConfig::block_m, *tuning.block_m);
    if ((use_tile || tuning.block_m) && (!overrides || !overrides->count("NUM_THREADS")))
        c.num_threads = threads(c.block_m, c.gemm1_block_n, c.gemm1_warp_m, c.gemm1_warp_n);
}
// Shared transport dimensions; payload sizes are selected by the precision backend.
inline DispatchConfig dispatch_config(int ranks, int experts, int topk, int capacity,
                                      int token_bytes, int scale_bytes) {
    TORCH_CHECK(ranks > 0 && experts > 0 && experts % ranks == 0 && topk > 0 && capacity >= 0);
    TORCH_CHECK(token_bytes > 0 && scale_bytes >= 0);
    DispatchConfig c;
    c.num_ranks = ranks;
    c.num_sms = get_num_sms();
    c.num_experts = experts;
    c.num_experts_per_rank = experts / ranks;
    c.num_topk = topk;
    c.nmtpr = capacity;
    c.token_bytes = token_bytes;
    c.sf_bytes = scale_bytes;
    return c;
}

// Select W4 dispatch threads and tiles from the transport byte count.
// W4A16 selects its actual MMA tile separately in w4a16_dispatch_tile.
inline DispatchConfig w4_dispatch_config(DispatchConfig c, int intermediate, int tokens,
                                         int bn, int bk, int wm, int wn, int stages,
                                         const std::optional<Overrides>& overrides) {
    TORCH_CHECK(intermediate > 0 && bn > 0 && bk > 0 && tokens >= 0 && tokens <= c.nmtpr);
    const int epr = c.num_experts_per_rank, bytes = c.token_bytes, sms = c.num_sms;
    bn = std::min(bn, intermediate);
    bk = std::min(bk, bytes);
    TORCH_CHECK(intermediate % bn == 0 && bytes % bk == 0 && bk % 16 == 0);
    const int max_bn = bn, total_m = std::max(1, tokens * std::min(c.num_topk, epr));
    const int em = std::max(1, ceil_div(total_m, epr));
    int bm, selected_sms;
    std::tuple<int, int, int> smem;
    std::tie(selected_sms, bm, bn, bk, wm, wn, stages, smem) =
        fp4_config(total_m, em, intermediate, bytes, epr, sms, GemmType::MGroupedNoPad, bn);
    // Apply measured persistent-GEMM corrections in shape-based bins.
    const bool wide = max_bn >= 256 && intermediate % 256 == 0 && bytes % 128 == 0;
    if (wide && epr >= sms && em >= 2 && em <= 5) {
        bm = 256;
        bn = 256;
        bk = 128;
        wm = 64;
        wn = 64;
        stages = 3;
    } else if (wide && epr <= sms / 2 && em >= 16 && em <= 32) {
        bm = 64;
        bn = 256;
        bk = 128;
        wm = 32;
        wn = 64;
        stages = 3;
    }
    if (bn < 128) {
        TORCH_CHECK(intermediate % 128 == 0 && (bm == 16 || bm == 32 || bm == 64));
        bn = 128;
        bk = 128;
        wm = bm == 16 ? 16 : 32;
        wn = 64;
        stages = bm == 16 ? 3 : 2;
    }
    TORCH_CHECK(intermediate % bn == 0 && bytes % bk == 0);
    c.num_threads = threads(bm, bn, wm, wn);
    c.block_m = bm;
    c.chunk_blocks = sms;
    c.gemm1_n = intermediate;
    c.gemm1_block_n = bn;
    c.gemm1_block_k = bk;
    c.gemm1_warp_m = wm;
    c.gemm1_warp_n = wn;
    c.gemm1_stages = stages;
    if (overrides) {
        for (const auto& [key, val] : *overrides) {
            const auto found =
                std::find_if(std::begin(dispatch_config_fields), std::end(dispatch_config_fields),
                             [&](const auto& field) { return key == field.first; });
            TORCH_CHECK(found != std::end(dispatch_config_fields), "Unknown MegaMoE config key: ", key);
            c.*(found->second) = val;
        }
        if (!overrides->count("NUM_THREADS") &&
            (overrides->count("BLOCK_M") || overrides->count("GEMM1_BLOCK_N") ||
             overrides->count("GEMM1_WARP_M") || overrides->count("GEMM1_WARP_N")))
            c.num_threads = threads(c.block_m, c.gemm1_block_n, c.gemm1_warp_m, c.gemm1_warp_n);
    }
    return c;
}
inline DispatchConfig fp4_dispatch_config(int ranks, int experts, int topk, int hidden, int intermediate,
                                          int capacity, int tokens, int bn = 256, int bk = 128, int wm = 32,
                                          int wn = 64, int stages = 3,
                                          const std::optional<Overrides>& overrides = std::nullopt) {
    TORCH_CHECK(hidden > 0 && hidden % 32 == 0);
    return w4_dispatch_config(dispatch_config(ranks, experts, topk, capacity, hidden / 2, hidden / 32),
                              intermediate, tokens, bn, bk, wm, wn, stages, overrides);
}
inline DispatchConfig w4a16_dispatch_config(int ranks, int experts, int topk, int hidden, int intermediate,
                                            int capacity, int tokens, int bm) {
    TORCH_CHECK(hidden > 0 && hidden % 32 == 0);
    TORCH_CHECK(bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256);
    Overrides overrides{{"BLOCK_M", bm}};
    if (bm < 64) overrides["GEMM1_WARP_M"] = bm;
    auto c = w4_dispatch_config(dispatch_config(ranks, experts, topk, capacity, hidden * 2, 0),
                                intermediate, tokens, 256, 128, 32, 64, 3, overrides);
    // The FP4 selector provides a scheduling hint, but its narrow N warp
    // shapes are not W4A16 MMA shapes. Normalize against the actual N tile
    // before sizing transport or applying a later block-M tuning override.
    const int bn = intermediate % 256 == 0 ? 256 : 128;
    const int nwarps = bn / 64;
    const int mwarps = bm < 64 ? 1 : std::clamp(c.num_threads / (nwarps * 32), 1, 4);
    c.gemm1_block_n = bn;
    c.gemm1_warp_m = bm / mwarps;
    c.gemm1_warp_n = 64;
    c.num_threads = mwarps * nwarps * 32;
    return c;
}
inline DispatchConfig w4a16_buffer_config(int ranks, int experts, int topk, int hidden, int intermediate,
                                         int capacity, std::optional<int> bucket, std::optional<int> requested) {
    // Every rank uses the same allocation bucket, including ranks whose later
    // forwards are empty. Transport protocols cannot be selected independently.
    const auto profile = bucket
        ? tuning_config({Precision::W4A16, ranks, experts, topk, hidden, intermediate / 2,
                         *bucket, "situ"}).buffer : BufferTuning{};
    auto c = w4a16_dispatch_config(ranks, experts, topk, hidden, intermediate, capacity, capacity,
                                   requested.value_or(profile.block_m.value_or(64)));
    c.enable_remote_push = profile.push_dispatch;
    return c;
}
inline int fp8_dispatch_block_m(std::optional<int> requested, bool blockwise) {
    const int bm = requested.value_or(blockwise ? 128 : 256);
    if (blockwise) {
        TORCH_CHECK(bm == 128, "FP8 blockwise dispatch requires block_m=128; "
                    "the GEMM1 compute tile is selected separately");
    } else {
        TORCH_CHECK(bm == 128 || bm == 256,
                    "FP8 channelwise dispatch/GEMM1 requires block_m=128 or 256");
    }
    return bm;
}

inline DispatchConfig fp8_dispatch_config(int ranks, int experts, int topk, int hidden, int capacity,
                                          std::optional<int> requested_block_m, bool blockwise) {
    TORCH_CHECK(hidden > 0 && hidden % (blockwise ? 128 : 16) == 0);
    const int bm = fp8_dispatch_block_m(requested_block_m, blockwise);
    auto c = dispatch_config(ranks, experts, topk, capacity, hidden, blockwise ? 4 * hidden / 128 : 4);
    c.num_sms = integer("SGLANG_PPU_FP8_MEGA_MOE_NUM_SMS", get_num_sms());
    TORCH_CHECK(c.num_sms > 0 && c.num_sms <= get_num_sms());
    c.num_threads = blockwise ? 256 : 512;
    c.block_m = bm;
    return c;
}
inline bool fp8_unit_scale() {
    return get_env<std::string>("DG_MOE_SCATTER_FP8_STATIC_SCALE_ONE", "0") == "1" &&
           get_env<std::string>("DG_MOE_SCATTER_FP8_SIDE_SCALE", "1") == "1" &&
           get_env<std::string>("DG_MOE_SCATTER_FP8_SIDE_SCALE_DYNAMIC", "0") != "1";
}
inline Config fp4_scatter_config(int m, int n, int k, int groups, std::optional<Config> requested,
                                 bool low_latency, bool masked, std::optional<bool> fp8,
                                 std::optional<int> expected) {
    auto validate = [n](const Config& config) {
        const int bn = std::get<2>(config);
        TORCH_CHECK(bn > 0 && bn % 256 == 0 && n % bn == 0,
                    "FP4 scatter requires complete 256-column tiles");
        return config;
    };
    if (requested)
        return validate(*requested);
    TORCH_CHECK(groups > 0 && n > 0 && n % 256 == 0,
                "FP4 scatter requires N divisible by 256");
    const int sms = get_num_sms(), em = expected.value_or(ceil_div(m, groups));
    int bm, bn = 256, bk, wm, wn, stages;
    const auto override = get_env<std::string>("DG_MOE_SCATTER_CONFIG");
    if (!override.empty()) {
        std::istringstream input(override);
        std::string field;
        std::vector<int> fields;
        while (std::getline(input, field, ','))
            fields.push_back(std::stoi(field));
        TORCH_CHECK(fields.size() == 6, "DG_MOE_SCATTER_CONFIG requires six integers");
        bm = fields[0];
        bn = fields[1];
        bk = fields[2];
        wm = fields[3];
        wn = fields[4];
        stages = fields[5];
    } else if (low_latency) {
        const bool quant = fp8.value_or(get_env<std::string>("DG_MOE_SCATTER_FP8", "0") == "1");
        // Select measured low-latency tile configurations by group count and expected M.
        if (!quant && groups == 12 && em <= 64) {
            bm = 64;
            bk = 64;
            wm = 32;
            wn = 64;
            stages = 3;
        } else if (em <= 16) {
            bm = 16;
            bk = quant ? 128 : 64;
            wm = 16;
            wn = 64;
            stages = 2;
        } else if (em <= 32) {
            bm = 32;
            bk = quant ? 128 : 64;
            wm = 32;
            wn = 32;
            stages = 2;
        } else if (em <= 64) {
            bm = 64;
            bk = quant ? 128 : 64;
            wm = 32;
            wn = 64;
            stages = quant ? 3 : 2;
        } else if (em <= 128) {
            bm = 128;
            bk = 64;
            wm = 64;
            wn = 64;
            stages = 3;
        } else {
            bm = 256;
            bk = 128;
            wm = 64;
            wn = 64;
            stages = 3;
        }
    } else {
        auto config = fp4_config(m, em, n, k, groups, sms,
                                 masked ? GemmType::MGroupedMasked : GemmType::MGroupedNoPad);
        auto& selected_n = std::get<2>(config);
        selected_n = std::max(256, std::min(selected_n, n) / 256 * 256);
        while (n % selected_n)
            selected_n -= 256;
        std::get<7>(config) = fp4_smem(std::get<6>(config), std::get<1>(config), selected_n,
                                     std::get<4>(config), std::get<5>(config), std::get<3>(config));
        return validate(config);
    }
    return validate({sms, bm, bn, bk, wm, wn, stages, fp4_smem(stages, bm, bn, wm, wn, bk)});
}
// GEMM1 metadata and GEMM2 must agree on row padding. Output quantization
// changes the scatter epilogue, not this shared default tile selection.
inline Config fp4_pipeline_scatter_config(int m, int n, int k, int groups,
                                          std::optional<Config> requested, std::optional<int> expected,
                                          const std::optional<TuningConfig>& tuning) {
    TORCH_CHECK(groups > 0, "FP4 scatter requires at least one expert group");
    const int em = expected ? std::max(1, *expected)
                   : tuning ? std::max(1, ceil_div(tuning->workload.tokens * tuning->workload.topk, groups))
                            : ceil_div(m, groups);
    if (!requested && get_env<std::string>("DG_MOE_SCATTER_CONFIG").empty() &&
        tuning && tuning->gemm2.tile) {
        const auto& t = *tuning->gemm2.tile;
        requested = Config{get_num_sms(), t[0], t[1], t[2], t[3], t[4], t[5],
                           fp4_smem(t[5], t[0], t[1], t[3], t[4], t[2])};
    }
    return fp4_scatter_config(m, n, k, groups, requested, true, false, false, em);
}

inline void fp8_pull_config(DispatchConfig& c, std::optional<bool> paired,
                            std::optional<bool> triple = std::nullopt,
                            const std::string& default_mode = "triple", const Gemm1Tuning& tuning = {}) {
    TORCH_CHECK(!(paired.value_or(false) && triple.value_or(false)));
    const auto mode = paired || triple ? (paired.value_or(false)   ? "paired"
                                          : triple.value_or(false) ? "triple"
                                                                   : default_mode)
                                       : get_env<std::string>("MEGA_MOE_PULL_MODE",
                                             tuning.pull_mode ? pull_mode_name(*tuning.pull_mode) : default_mode);
    TORCH_CHECK(mode == "single" || mode == "paired" || mode == "triple");
    const int pulls = integer("MEGA_MOE_PULL_SMS", tuning.pull_sms.value_or(6));
    const int team = mode == "triple" ? 3 : mode == "paired" ? 2 : 1;
    TORCH_CHECK(pulls > 0 && pulls < c.num_sms && pulls % team == 0, "Invalid FP8 pull SM partition");
    c.num_pull_sms = pulls;
    c.paired_block_pull = int(mode == "paired");
    c.triple_block_pull = int(mode == "triple");
}

inline void fp8_gemm1_config(DispatchConfig& c, std::optional<bool> paired,
                             std::optional<bool> triple, const Gemm1Tuning& tuning,
                             std::optional<int> block_m = std::nullopt) {
    // Pull reads unpadded source rows; only this rank's landing pool and GEMM
    // metadata use BM. Workspace offsets and peer handshakes are independent
    // of BM, so the same allocation supports per-call and per-rank selection.
    c.block_m = fp8_dispatch_block_m(block_m.value_or(tuning.block_m.value_or(c.block_m)), false);
    c.gemm1_compute_m = c.block_m;
    c.gemm1_block_n = 256;
    c.gemm1_block_k = 128;
    c.gemm1_warp_m = c.gemm1_compute_m / 4;
    c.gemm1_warp_n = 64;
    c.gemm1_stages = 4;
    c.num_threads = threads(c.gemm1_compute_m, c.gemm1_block_n, c.gemm1_warp_m, c.gemm1_warp_n);
    c.gemm1_dynamic_queue = tuning.dynamic_queue.value_or(false);
    fp8_pull_config(c, paired, triple, "triple", tuning);
}

inline int blockwise_compute_m(std::optional<int> expected_m, int groups, int n, int sms) {
    TORCH_CHECK(groups > 0 && n > 0 && sms > 0 && (!expected_m || *expected_m >= 0));
    if (!expected_m || *expected_m <= 128) return 128;
    const int m128 = ceil_div(*expected_m, 128), m192 = ceil_div(*expected_m, 192);
    if (groups * m192 * ceil_div(n, 256) < sms && m128 > m192) return 128;
    // Empirical fallback: BM192 may trade up to 12.5% more padded rows for
    // lower per-row overhead. This is a tuning budget, not a kernel constraint.
    constexpr int padding_budget_numerator = 9, padding_budget_denominator = 8;
    return int64_t(m192) * 192 * padding_budget_denominator <=
           int64_t(m128) * 128 * padding_budget_numerator ? 192 : 128;
}

inline void fp8_blockwise_gemm1_config(DispatchConfig& c, int tokens, int n, bool quant,
                                       const Gemm1Tuning& tuning) {
    c.block_m = fp8_dispatch_block_m(c.block_m, true);
    const int groups = c.num_experts_per_rank;
    const int em = ceil_div(int64_t(tokens) * c.num_topk, int64_t(groups));
    c.gemm1_compute_m = tuning.block_m.value_or(quant ? blockwise_compute_m(em, groups, n, c.num_sms) : 128);
    TORCH_CHECK(c.gemm1_compute_m == 128 ||
                (quant && (c.gemm1_compute_m == 32 || c.gemm1_compute_m == 192)),
                "FP8 blockwise GEMM1 requires compute block_m=128, or 32/192 with FP8 output");
    TORCH_CHECK(!tuning.tile, "blockwise GEMM1 accepts block_m, not a full tile override");
    c.gemm1_block_n = quant ? 256 : 128;
    c.gemm1_block_k = 128;
    c.gemm1_warp_m = std::max(16, c.gemm1_compute_m / 4);
    c.gemm1_warp_n = c.gemm1_block_n / 4;
    c.gemm1_stages = 4;
    c.num_threads = threads(c.gemm1_compute_m, c.gemm1_block_n, c.gemm1_warp_m, c.gemm1_warp_n);
    const double waves = double(groups) * ceil_div(em, c.gemm1_compute_m) * ceil_div(n, 256) / c.num_sms;
    // The fallback favors a queue for medium-depth BM192 workloads;
    // keep an explicit on/off choice authoritative over this empirical window.
    constexpr int queue_min_waves = 32, queue_max_waves = 64;
    c.gemm1_dynamic_queue = tuning.dynamic_queue.value_or(
        quant && c.gemm1_compute_m == 192 && waves >= queue_min_waves && waves <= queue_max_waves);
    // Bound dedicated pull CTAs to a quarter of the grid. Wider GEMMs leave
    // more computation per pulled row, so they need a smaller pull fraction.
    constexpr int pull_grid_divisor = 4, pull_balance_columns = 512;
    const int limit = std::max(1, c.num_sms / pull_grid_divisor);
    int pulls = std::min(limit, std::max(1, ceil_div(c.num_sms * pull_balance_columns, n + pull_balance_columns)));
    pulls = std::min(limit, pulls * std::max(ceil_div(c.num_ranks, 32), c.gemm1_dynamic_queue ? 2 : 1));
    c.pull_sms = integer("MEGA_MOE_PULL_SMS", tuning.pull_sms.value_or(pulls));
    TORCH_CHECK(c.pull_sms > 0 && c.pull_sms < c.num_sms);
}

inline ScatterConfig fp8_blockwise_scatter_config(int n, int k, int groups, int masters,
                                                   std::optional<int> expected,
                                                   const Gemm2Tuning& tuning) {
    ScatterConfig c;
    c.n = n;
    c.k = k;
    c.num_groups = groups;
    c.num_master_groups = masters;
    c.block_m = blockwise_compute_m(expected, groups, n, get_num_sms());
    c.num_stages = std::min(c.block_m == 192 ? 4 : 3, std::max(2, k / 128));
    if (tuning.tile) {
        const auto& t = *tuning.tile;
        TORCH_CHECK(t[1] == 256 && t[2] == 128 && t[3] == std::max(16, t[0] / 4) && t[4] == 64,
                    "unsupported blockwise GEMM2 tile");
        c.block_m = t[0];
        c.num_stages = t[5];
    }
    TORCH_CHECK((c.block_m == 32 || c.block_m == 128 || c.block_m == 192) &&
                c.num_stages >= 2 && c.num_stages <= 4);
    c.block_n = 256;
    c.block_k = 128;
    c.warp_m = std::max(16, c.block_m / 4);
    c.warp_n = 64;
    c.num_threads = threads(c.block_m, c.block_n, c.warp_m, c.warp_n);
    c.scratch_rows = c.block_m == 192 || c.num_stages == 4 ? 32 : 0;
    return c;
}

// Keep dispatch scheduling independent of buffer ownership and kernel argument assembly.
inline void fp4_dispatch_schedule(DispatchConfig& c, const DispatchConfig& capacity, int num_tokens,
                                  bool dynamic, bool bf16, bool dispatch_only, bool split, bool glu,
                                  bool quantize_output, const Gemm1Tuning& tuning = {}) {
    const bool gemm = !dispatch_only && flag("MEGA_MOE_ENABLE_GEMM1", true);
    const int push_threshold = integer("MEGA_MOE_BF16_PUSH_MIN_TOKENS", 512);
    TORCH_CHECK(push_threshold >= 0);
    const bool push =
        dispatch_only && flag("MEGA_MOE_ENABLE_REMOTE_PUSH", bf16 && num_tokens >= push_threshold && !split);
    TORCH_CHECK(!push || (!split && capacity.sf_bytes == 0));
    const bool pull = !push && flag("MEGA_MOE_ENABLE_REMOTE_PULL", true);
    const int sms = c.num_sms, serial_rows = integer("MEGA_MOE_SERIAL_MAX_RECV_ROWS", 2 * sms);
    TORCH_CHECK(serial_rows >= 0);
    const bool latency =
        flag("MEGA_MOE_LATENCY_OPTIMIZED",
             dynamic && num_tokens * c.num_topk <= serial_rows && glu && quantize_output && gemm && pull);
    const bool specialized = !dispatch_only && !latency && flag("MEGA_MOE_SM_SPECIALIZATION", true);
    const int producers = latency ? 0 : integer("MEGA_MOE_PRODUCER_WARPS", push && bf16 ? 8 : 0);
    TORCH_CHECK(producers >= 0 && producers <= 16);
    c.gemm1_dynamic_queue = specialized && gemm && pull && producers == 0 &&
                            tuning.dynamic_queue.value_or(true);
    const int slots = dispatch_pull_slots(c.num_threads + producers * 32, c.token_bytes, c.sf_bytes);
    int pulls, gemms;
    if (specialized) {
        pulls = integer("MEGA_MOE_PULL_SMS", std::getenv("MEGA_MOE_GEMM_SMS")
                                                 ? sms - integer("MEGA_MOE_GEMM_SMS", 0)
                                                 : tuning.pull_sms.value_or(std::min(6, sms - 1)));
        gemms = integer("MEGA_MOE_GEMM_SMS", sms - pulls);
    } else {
        pulls = latency ? sms : integer("MEGA_MOE_PULL_SMS", sms);
        gemms = latency ? sms : integer("MEGA_MOE_GEMM_SMS", sms);
    }
    TORCH_CHECK(pulls >= 1 && pulls <= sms && gemms >= 1 && gemms <= sms);
    TORCH_CHECK(!specialized || ((gemm || pull) && pulls + gemms == sms));
    c.enable_gemm1 = gemm;
    c.enable_remote_pull = pull;
    c.enable_remote_push = push;
    c.producer_warps = producers;
    c.pull_sms = pulls;
    c.gemm_sms = gemms;
    c.sm_specialization = specialized;
    c.pull_slots = slots;
}

inline void w4a16_dispatch_tile(DispatchConfig& c, int n) {
    const int bn = n % 256 == 0 ? 256 : 128, warps = c.num_threads / 32, mwarps = warps / (bn / 64);
    TORCH_CHECK(warps * 32 == c.num_threads && warps % (bn / 64) == 0 &&
                (mwarps == 1 || mwarps == 2 || mwarps == 4));
    c.pull_slots = dispatch_pull_slots(c.num_threads, c.token_bytes, c.sf_bytes);
    c.w4_block_m = c.block_m < 64 ? c.block_m * mwarps : 64 * mwarps;
    c.w4_block_n = bn;
    c.w4_block_k = 128;
    c.w4_warp_m = c.block_m < 64 ? c.block_m : 64;
    c.w4_warp_n = 64;
    c.w4_warp_k = 128;
    c.w4_stages = 2;
}

struct W4A16Gemm1Config {
    DispatchConfig dispatch;
    int gemm_threads;
    bool separate_dispatch;
};

inline W4A16Gemm1Config w4a16_gemm1_config(
    DispatchConfig c, int n, const Gemm1Tuning& tuning,
    const std::optional<Overrides>& overrides) {
    c.gemm1_n = n;
    c.gemm1_dynamic_queue = tuning.dynamic_queue.value_or(false);
    if (tuning.separate_dispatch || c.enable_remote_push) {
        TORCH_CHECK((tuning.tile || c.enable_remote_push) && !c.gemm1_dynamic_queue,
                    "separate W4A16 dispatch requires an MMA tile and no fused work queue");
        if (tuning.tile) {
            const auto& t = *tuning.tile;
            c.w4_block_m = t[0]; c.w4_block_n = t[1]; c.w4_block_k = t[2];
            c.w4_warp_m = t[3]; c.w4_warp_n = t[4]; c.w4_warp_k = t[2];
            c.w4_stages = t[5];
        } else {
            // An empty or unmatched rank still follows the workspace's push
            // protocol, with the ordinary W4A16 GEMM geometry.
            w4a16_dispatch_tile(c, n);
        }
        const int gemm_threads = threads(c.w4_block_m, c.w4_block_n, c.w4_warp_m, c.w4_warp_n);
        c.num_threads = tuning.dispatch_threads.value_or(c.num_threads);
        c.pull_slots = dispatch_pull_slots(c.num_threads, c.token_bytes, c.sf_bytes);
        return {c, gemm_threads, true};
    }
    apply_gemm1_tuning(c, tuning, overrides);
    // The measured queue profile owns its pull/compute partition.
    TORCH_CHECK(!c.gemm1_dynamic_queue || tuning.pull_sms,
                "W4A16 dynamic queue requires a pull CTA count in the tuning profile");
    c.num_pull_sms = c.gemm1_dynamic_queue ? *tuning.pull_sms : c.num_sms;
    TORCH_CHECK(!c.gemm1_dynamic_queue || (c.num_pull_sms > 0 && c.num_pull_sms < c.num_sms));
    w4a16_dispatch_tile(c, n);
    return {c, c.num_threads, false};
}

inline bool channel_pair_staging(const Config& config, const std::string& kernel, int m, int n, int k,
                                 int groups, int em, int world_size, int max_tokens, int num_topk,
                                 bool rolling) {
    const auto [ns, bm, bn, bk, wm, wn, stages, smem] = config;
    return channel_pair_staging_default(m, n, k, groups, em, world_size, max_tokens, num_topk) &&
           bm == 128 && bn == 256 && bk == 128 && wm == 32 && wn == 64 && stages == 4 &&
           kernel == "OverlapPrologue" && !rolling;
}

inline std::pair<Config, ScatterConfig> channel_scatter_policy(
    int m, int n, int k, int groups, int masters, int world, int capacity, int topk, bool padded,
    std::optional<Config> requested, std::optional<std::string> kernel_override,
    std::optional<bool> rolling_override, std::optional<int> sms_override,
    const Gemm2Tuning& tuning = {}) {
    const int em = ceil_div(m, groups), physical_sms = get_num_sms();
    int sms = sms_override.value_or(integer("DG_CHANNELWISE_SCATTER_NUM_SMS", physical_sms));
    TORCH_CHECK(sms > 0 && sms <= physical_sms);
    std::string kernel = kernel_override.value_or("Default");
    TORCH_CHECK(kernel == "Default" || kernel == "OverlapPrologue");
    if (!requested && tuning.tile &&
        (!tuning.capacity || *tuning.capacity == capacity) &&
        (!tuning.requires_no_replicas || masters == groups)) {
        const auto& t = *tuning.tile;
        requested = Config{sms, t[0], t[1], t[2], t[3], t[4], t[5],
            deep_gemm_int8::get_smem_config(t[5], k, t[0], t[1], t[2], 1)};
    }
    if (!requested)
        requested = deep_gemm_int8::get_best_configs(
            em, n, k, groups, sms, GemmType::MGroupedNoPad, 256,
            std::nullopt, nullptr, /*fixed_block_n=*/256);
    const auto [ns, bm, bn, bk, wm, wn, stages, smem] = *requested;
    TORCH_CHECK(bn == 256 && n % bn == 0, "channelwise scatter requires complete 256-column tiles");
    const bool rolling = rolling_override.value_or(
        bm == 256 && bn == 256 && wm == 64 && wn == 64 && kernel == "Default");
    TORCH_CHECK(!rolling || (bm == 256 && bn == 256 && wm == 64 && wn == 64));
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
    c.world_size = world;
    c.bm128_overlap_pair_staging = channel_pair_staging(
        *requested, kernel, m, n, k, groups, em, world, capacity, topk, rolling);
    c.kernel_type = kernel;
    c.use_padded_valid_rows = padded;
    c.rolling_pair_scatter = rolling;
    return {*requested, c};
}

inline auto w4a16_scatter_config(int em, int n, int k,
                                 std::optional<deep_gemm_w4a16_common::PPUW4ConfigTuple> configs,
                                 const Gemm2Tuning& tuning = {}) {
    TORCH_CHECK(n > 0 && n % 256 == 0, "W4A16 scatter requires N divisible by 256");
    if (!configs && tuning.tile) {
        const auto& t = *tuning.tile;
        configs = deep_gemm_w4a16_common::PPUW4ConfigTuple{
            get_num_sms(), t[0], t[1], t[2], t[3], t[4], t[2], t[5]};
    }
    auto c = deep_gemm_w4a16_common::ppu_w4_config(em, n, k, configs);
    if (!configs) {
        c.block_n = std::max(256, std::min(c.block_n, n) / 256 * 256);
        while (n % c.block_n)
            c.block_n -= 256;
        c.warp_k = c.block_k;
    }
    TORCH_CHECK(c.block_n > 0 && c.block_n % 256 == 0 && n % c.block_n == 0,
                "W4A16 scatter requires complete 256-column tiles");
    return c;
}

inline int combine_num_threads(int tokens, int hidden) {
    // One CTA per token. Use the actual local row count, not buffer capacity
    // or dispatch padding, to balance row parallelism and work per thread.
    if (tokens >= 512)
        return 256;
    if (tokens <= 64 && hidden > 4096)
        return 1024;
    return 512;
}

} // namespace deep_gemm::mega_moe_config
