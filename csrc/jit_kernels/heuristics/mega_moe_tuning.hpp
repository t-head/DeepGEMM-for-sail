#pragma once

#include <array>
#include <initializer_list>
#include <optional>
#include <string>

namespace deep_gemm::mega {

enum class Precision { W4A16, W4A8, FP8Channelwise, FP8Blockwise };
using Tile = std::array<int, 6>; // BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, stages

struct TuningWorkload {
    Precision precision;
    int ranks, experts, topk, hidden, intermediate, tokens;
    std::string activation;
};

struct Gemm1Tuning {
    std::optional<Tile> tile{};
    std::optional<int> block_m{};
    std::optional<int> pull_ctas{};
};

struct Gemm2Tuning {
    std::optional<Tile> tile{};
    // Optional applicability limits for measured scatter tiles. G1 is unaffected.
    std::optional<int> capacity{};
    bool requires_no_replicas = false;
};

struct BufferTuning {
    std::optional<int> block_m{};
};

struct TuningConfig {
    TuningWorkload workload;
    std::string profile;
    Gemm1Tuning gemm1;
    Gemm2Tuning gemm2;
    BufferTuning buffer;
};

// These are measured overrides, not complete kernel configurations. Empty
// fields use the family-specific selector. Tokens match exactly unless an
// entry explicitly retains measured buckets with inclusive upper bounds.
struct TuningCase {
    int tokens;
    Gemm1Tuning gemm1{};
    Gemm2Tuning gemm2{};
    BufferTuning buffer{};
};

struct TuningEntry {
    const char* model;
    Precision precision;
    int ranks, experts, topk, hidden, intermediate;
    const char* activation;
    std::initializer_list<TuningCase> cases;
    // Only the W4A8 profiles use their existing measured token buckets.
    // Inputs above the final bound fall back to the family selector.
    bool token_buckets = false;
};

inline TuningConfig select_tuning(const TuningWorkload& w) {
    TuningConfig result{w, {}, {}, {}, {}};
    using P = Precision;
    static const Tile channelwise_g1_decode{128, 256, 128, 32, 64, 5};
    static const TuningEntry entries[] = {
        {"minimax_m3",
         P::FP8Channelwise,
         8,
         128,
         4,
         6144,
         3072,
         "swiglu_oai",
         {{32, {channelwise_g1_decode}},
          {64, {channelwise_g1_decode}},
          {128, {channelwise_g1_decode}},
          {1024, {{}, {}, 3}}, {2048, {{}, {}, 3}}, {4096, {{}, {}, 3}}}},
        {"glm52",
         P::FP8Channelwise,
         8,
         256,
         8,
         6144,
         2048,
         "swiglu",
         {{32, {{}, 128}, {Tile{128, 256, 128, 32, 64, 4}, 4096, true}},
          {64, {channelwise_g1_decode}},
          {128, {channelwise_g1_decode}},
          {1024, {{}, {}, 9}},
          {2048, {{}, {}, 9}},
          {4096, {{}, {}, 9}}}},
        {"glm53_flash",
         P::FP8Channelwise,
         4,
         288,
         8,
         4096,
         2048,
         "swiglu",
         {{32, {Tile{16, 128, 128, 16, 32, 4}, {}, 3}, {Tile{16, 256, 64, 16, 32, 3}}},
          {64, {Tile{32, 256, 128, 16, 32, 5}, {}, 3}, {Tile{16, 256, 64, 16, 32, 3}}},
          {128, {Tile{32, 256, 128, 16, 32, 5}, {}, 3}, {Tile{32, 256, 64, 32, 64, 4}}},
          {1024, {{}, 128, 6}, {Tile{128, 256, 128, 32, 64, 3}}}, {2048}, {4096}}},
        {"qwen38_flash",
         P::FP8Channelwise,
         4,
         512,
         10,
         2560,
         640,
         "swiglu",
         {{32, {Tile{16, 128, 128, 16, 32, 4}, {}, 6}, {Tile{16, 256, 64, 16, 64, 2}}},
          {64, {Tile{32, 256, 128, 16, 32, 5}, {}, 6}, {Tile{16, 256, 64, 16, 64, 2}}},
          {128, {Tile{32, 256, 128, 16, 32, 5}, {}, 6}, {Tile{16, 256, 64, 16, 64, 2}}},
          {1024, {{}, {}, 9}, {Tile{128, 256, 128, 32, 64, 3}}}, {2048, {{}, {}, 12}}, {4096, {{}, {}, 12}}}},
    };
    for (const auto& entry : entries) {
        if (w.precision == entry.precision && w.ranks == entry.ranks && w.experts == entry.experts &&
            w.topk == entry.topk && w.hidden == entry.hidden && w.intermediate == entry.intermediate &&
            w.activation == entry.activation) {
            for (const auto& c : entry.cases) {
                if (w.tokens == c.tokens || (entry.token_buckets && w.tokens <= c.tokens)) {
                    result.profile = entry.model;
                    result.gemm1 = c.gemm1;
                    result.gemm2 = c.gemm2;
                    result.buffer = c.buffer;
                    return result;
                }
            }
        }
    }
    return result;
}

} // namespace deep_gemm::mega
