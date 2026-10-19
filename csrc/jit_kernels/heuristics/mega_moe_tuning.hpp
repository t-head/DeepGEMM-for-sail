#pragma once

#include <array>
#include <optional>
#include <string>

namespace deep_gemm::mega_moe_config {

enum class Precision { W4A4, W4A16, FP8Channelwise, FP8Blockwise };
enum class PullMode { Single, Paired, Triple };
using Tile = std::array<int, 6>; // BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, stages

struct TuningWorkload {
    Precision precision;
    int ranks, experts, topk, hidden, intermediate, tokens;
    std::string activation;
};

struct Gemm1Tuning {
    std::optional<Tile> tile{};
    std::optional<int> block_m{};
    std::optional<int> pull_sms{};
    std::optional<PullMode> pull_mode{};
    std::optional<bool> separate_quant{};
};

struct Gemm2Tuning {
    std::optional<Tile> tile{};
};

struct CombineTuning {
    std::optional<bool> chunked{};
};

struct TuningConfig {
    TuningWorkload workload;
    std::string profile;
    Gemm1Tuning gemm1;
    Gemm2Tuning gemm2;
    CombineTuning combine;
};

// These are measured overrides, not complete kernel configurations. Empty
// fields retain the existing family-specific selector. Token counts match
// exactly: no nearest-model matching or interpolation into unmeasured ranges.
struct TuningEntry {
    const char* model;
    Precision precision;
    int ranks, experts, topk, hidden, intermediate;
    const char* activation;
    std::array<Gemm1Tuning, 3> gemm1; // 1024, 2048, 4096 tokens per rank
    std::array<Gemm2Tuning, 3> gemm2{};
    std::array<CombineTuning, 3> combine{};
};

inline TuningConfig select_tuning(const TuningWorkload& w, const std::string& device, int sms,
                                   bool enabled = true) {
    TuningConfig result{w, {}, {}, {}, {}};
    if (!enabled || device != "ZW-M890P" || sms != 39)
        return result;
    const int bucket = w.tokens == 1024 ? 0 : w.tokens == 2048 ? 1 : w.tokens == 4096 ? 2 : -1;
    if (bucket < 0)
        return result;
    using P = Precision;
    using M = PullMode;
    static const TuningEntry entries[] = {
        {"deepseek_v4_pro", P::W4A4, 8, 384, 6, 7168, 3072, "swiglu",
         {{{Tile{128, 256, 128, 64, 64, 3}}, {}, {{}, {}, 4}}}},
        {"deepseek_v4_flash", P::W4A4, 4, 256, 6, 4096, 2048, "swiglu",
         {{{Tile{256, 256, 128, 64, 64, 3}}, {}, {{}, {}, 8}}}},
        {"glm52", P::W4A4, 8, 256, 8, 6144, 2048, "swiglu",
         {{{{}, {}, 12}, {{}, {}, 8}, {{}, {}, 8}}}},
        {"kimi_k3", P::W4A4, 16, 896, 16, 3584, 3072, "situ",
         {{{{}, {}, 8}, {{}, {}, 8}, {{}, {}, 8}}}},
        {"qwen38_2_4t", P::W4A4, 16, 512, 10, 8192, 2048, "swiglu",
         {{{{}, {}, 4}, {}, {}}}, {}, {{{false}, {}, {}}}},
        {"kimi_k3", P::W4A16, 16, 896, 16, 3584, 3072, "situ",
         {{{{}, 128}, {{}, 256}, {{}, 256}}}},
        {"minimax_m3", P::FP8Channelwise, 8, 128, 4, 6144, 3072, "swiglu_oai",
         {{{{}, {}, 6, M::Paired}, {{}, {}, 6, M::Paired}, {{}, {}, 6, M::Paired}}}},
        {"glm52", P::FP8Channelwise, 8, 256, 8, 6144, 2048, "swiglu",
         {{{{}, {}, {}, {}, true}, {{}, {}, 8, M::Paired, true}, {{}, {}, 8, M::Paired, true}}}},
        {"glm53_flash", P::FP8Channelwise, 4, 288, 8, 4096, 2048, "swiglu",
         {{{{}, {}, 3, {}, true}, {{}, {}, {}, {}, true}, {{}, {}, {}, {}, true}}},
         {{{Tile{128, 256, 128, 32, 64, 3}}, {}, {}}}},
        {"qwen38_flash", P::FP8Channelwise, 4, 512, 10, 2560, 640, "swiglu",
         {{{{}, {}, 9, M::Triple}, {{}, {}, 12}, {{}, {}, 12}}},
         {{{Tile{128, 256, 128, 32, 64, 3}}, {}, {}}}},
    };
    for (const auto& entry : entries) {
        if (w.precision == entry.precision && w.ranks == entry.ranks && w.experts == entry.experts &&
            w.topk == entry.topk && w.hidden == entry.hidden && w.intermediate == entry.intermediate &&
            w.activation == entry.activation) {
            result.profile = entry.model;
            result.gemm1 = entry.gemm1[bucket];
            result.gemm2 = entry.gemm2[bucket];
            result.combine = entry.combine[bucket];
            break;
        }
    }
    return result;
}

inline const char* pull_mode_name(PullMode mode) {
    return mode == PullMode::Paired ? "paired" : mode == PullMode::Triple ? "triple" : "single";
}

} // namespace deep_gemm::mega_moe_config
