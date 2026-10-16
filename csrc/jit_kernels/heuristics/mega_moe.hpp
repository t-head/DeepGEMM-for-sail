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

namespace deep_gemm::mega_moe_config {
struct DispatchConfig {
    int block_m = 0;
    int chunk_blocks = 0;
    int clamp = 0;
    float clamp_value = 0;
    bool enable_gemm1 = false;
    bool enable_remote_pull = false;
    bool enable_remote_push = false;
    bool quantize_output = false;
    bool fused_situ = false;
    bool fused_swiglu = false;
    int gemm1_block_k = 0;
    int gemm1_block_n = 0;
    int gemm1_n = 0;
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
    int pack_routes = 0;
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
    int tail_gemm_blocks = 0;
    int tail_gemm_percent = 0;
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
    int groups_per_block = 0;
    int hidden = 0;
    int num_sms = 0;
    int topk = 0;
    int vec_block_threads = 0;
};

struct ScatterConfig {
    int block_k = 0;
    int block_m = 0;
    int block_n = 0;
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
    int num_stages = 0;
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
                         GemmType type = GemmType::GroupedNoPad, int max_bn = 256) {
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

// Preserve the W4 dispatch thread/tile policy in terms of transport bytes.
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
        fp4_config(total_m, em, intermediate, bytes, epr, sms, GemmType::GroupedNoPad, bn);
    // Preserve the measured persistent-GEMM corrections in shape-based bins.
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
    return w4_dispatch_config(dispatch_config(ranks, experts, topk, capacity, hidden * 2, 0),
                              intermediate, tokens, 256, 128, 32, 64, 3, Overrides{{"BLOCK_M", bm}});
}
inline DispatchConfig fp8_dispatch_config(int ranks, int experts, int topk, int hidden, int capacity,
                                          int bm, bool blockwise) {
    TORCH_CHECK(hidden > 0 && hidden % (blockwise ? 128 : 16) == 0);
    TORCH_CHECK(bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256);
    auto c = dispatch_config(ranks, experts, topk, capacity, hidden, blockwise ? 4 * hidden / 128 : 4);
    c.num_sms = integer("SGLANG_PPU_FP8_MEGA_MOE_NUM_SMS", get_num_sms());
    TORCH_CHECK(c.num_sms > 0 && c.num_sms <= get_num_sms());
    c.num_threads = bm == 256 ? 512 : 256;
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
        // Keep the existing tuned bins; this refactor does not retune kernels.
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
                                 masked ? GemmType::GroupedMasked : GemmType::GroupedNoPad);
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
inline void fp8_pull_config(DispatchConfig& c, std::optional<bool> paired,
                            std::optional<bool> triple = std::nullopt,
                            const std::string& default_mode = "triple") {
    TORCH_CHECK(!(paired.value_or(false) && triple.value_or(false)));
    const auto mode = paired || triple ? (paired.value_or(false)   ? "paired"
                                          : triple.value_or(false) ? "triple"
                                                                   : default_mode)
                                       : get_env<std::string>("MEGA_MOE_PULL_MODE", default_mode);
    TORCH_CHECK(mode == "single" || mode == "paired" || mode == "triple");
    const int pulls = integer("MEGA_MOE_PULL_SMS", 6), team = mode == "triple" ? 3 : mode == "paired" ? 2 : 1;
    TORCH_CHECK(pulls > 0 && pulls < c.num_sms && pulls % team == 0, "Invalid FP8 pull SM partition");
    c.num_pull_sms = pulls;
    c.paired_block_pull = int(mode == "paired");
    c.triple_block_pull = int(mode == "triple");
}

inline std::optional<std::pair<Config, std::string>> channel_scatter_config(int expected_m, int n, int k,
                                                                            int groups, int sms) {
    if (n != 6144 || k != 2048 || groups < 32 || groups > 34)
        return std::nullopt;
    const int bm = expected_m <= 16 ? 32 : expected_m <= 32 ? 64 : expected_m <= 128 ? 128 : 256;
    const int wm = bm == 256 ? 64 : 32, stages = expected_m <= 32 ? 3 : 4;
    Config config{sms, bm, 256,    128,
                  wm,  64, stages, deep_gemm_int8::get_smem_config(stages, k, bm, 256, 128, 1)};
    return std::make_pair(config, bm == 256 ? "Default" : "OverlapPrologue");
}
// Keep dispatch scheduling independent of buffer ownership and kernel argument assembly.
inline void fp4_dispatch_schedule(DispatchConfig& c, const DispatchConfig& capacity, int num_tokens,
                                  bool dynamic, bool bf16, bool dispatch_only, bool split, bool glu,
                                  bool quantize_output) {
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
    const int max_slots = c.num_threads / 32 + producers;
    const int slots = latency ? std::min(2, max_slots)
                              : integer("MEGA_MOE_PULL_SLOTS", std::min(specialized ? 16 : 8, max_slots));
    TORCH_CHECK(slots >= 1 && slots <= max_slots);
    const bool tail_default = specialized && gemm && pull && !std::getenv("MEGA_MOE_TAIL_GEMM_BLOCKS") &&
                              !std::getenv("MEGA_MOE_TAIL_GEMM_PERCENT");
    const int tail_blocks = latency ? 0 : integer("MEGA_MOE_TAIL_GEMM_BLOCKS", tail_default ? 19 : 0);
    const int tail_percent = latency ? 0 : integer("MEGA_MOE_TAIL_GEMM_PERCENT", tail_default ? 27 : 0);
    TORCH_CHECK(tail_blocks >= 0 && tail_percent >= 0 && tail_percent <= 100);
    int pulls, gemms;
    if (specialized) {
        pulls = integer("MEGA_MOE_PULL_SMS", std::getenv("MEGA_MOE_GEMM_SMS")
                                                 ? sms - integer("MEGA_MOE_GEMM_SMS", 0)
                                                 : std::min(6, sms - 1));
        gemms = integer("MEGA_MOE_GEMM_SMS", sms - pulls);
    } else {
        pulls = latency ? sms : integer("MEGA_MOE_PULL_SMS", sms);
        gemms = latency ? sms : integer("MEGA_MOE_GEMM_SMS", sms);
    }
    TORCH_CHECK(pulls >= 1 && pulls <= sms && gemms >= 1 && gemms <= sms);
    TORCH_CHECK(!specialized || ((gemm || pull) && pulls + gemms == sms));
    TORCH_CHECK(!(tail_blocks || tail_percent) || (specialized && gemm && pull));
    c.enable_gemm1 = gemm;
    c.enable_remote_pull = pull;
    c.enable_remote_push = push;
    c.producer_warps = producers;
    c.pull_sms = pulls;
    c.gemm_sms = gemms;
    c.sm_specialization = specialized;
    c.pull_slots = slots;
    c.tail_gemm_blocks = tail_blocks;
    c.tail_gemm_percent = tail_percent;
}

inline void w4a16_dispatch_tile(DispatchConfig& c, int n) {
    const int bn = n % 256 == 0 ? 256 : 128, warps = c.num_threads / 32, mwarps = warps / (bn / 64);
    TORCH_CHECK(warps * 32 == c.num_threads && warps % (bn / 64) == 0 &&
                (mwarps == 1 || mwarps == 2 || mwarps == 4));
    c.pull_slots = std::min(8, warps);
    c.w4_block_m = 64 * mwarps;
    c.w4_block_n = bn;
    c.w4_block_k = 128;
    c.w4_warp_m = 64;
    c.w4_warp_n = 64;
    c.w4_warp_k = 128;
    c.w4_stages = 2;
}

// GLM prefill quantization policy; explicit API/environment overrides take priority.
inline bool fp8_separate_quant(const DispatchConfig& c, int num_tokens, int masters, int n, int hidden,
                               std::optional<bool> separate) {
    bool split = c.swiglu_oai && num_tokens >= 1024 && c.num_ranks == 8 && c.num_experts == 128 &&
                 c.num_topk == 4 && masters == 16 && n == 6144 && hidden == 6144 && !c.paired_block_pull;
    if (separate)
        split = *separate;
    else {
        const auto mode = get_env<std::string>("MEGA_MOE_QUANT_MODE", "auto");
        TORCH_CHECK(mode == "auto" || mode == "fused" || mode == "separate");
        if (mode != "auto")
            split = mode == "separate";
    }
    return split;
}

inline bool channel_pair_staging(const Config& config, const std::string& kernel, int m, int n, int k,
                                 int groups, int em, int world_size, int max_tokens, int num_topk,
                                 bool rolling) {
    const auto [ns, bm, bn, bk, wm, wn, stages, smem] = config;
    return m == 4096 && n == 6144 && k == 2048 && groups == 32 && em == 128 && bm == 128 && bn == 256 &&
           bk == 128 && wm == 32 && wn == 64 && stages == 4 && kernel == "OverlapPrologue" &&
           world_size == 8 && max_tokens == 512 && num_topk == 8 && !rolling;
}

inline auto w4a16_scatter_config(int em, int n, int k,
                                 const std::optional<deep_gemm_w4a16_common::PPUW4ConfigTuple>& configs) {
    TORCH_CHECK(n > 0 && n % 256 == 0, "W4A16 scatter requires N divisible by 256");
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

inline bool chunked_reduce(int tokens, bool fp8 = false) {
    const char* value = std::getenv(fp8 ? "DG_MOE_FP8_REDUCE_CHUNKED" : "DG_MOE_REDUCE_CHUNKED");
    if (fp8 && value == nullptr)
        value = std::getenv("DG_MOE_REDUCE_CHUNKED");
    return value != nullptr ? std::atoi(value) != 0 : (!fp8 && tokens >= 64 && tokens <= 2048);
}

} // namespace deep_gemm::mega_moe_config
