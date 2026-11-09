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
    static const Tile small16{16, 256, 128, 16, 64, 2};
    static const Tile small32{32, 256, 128, 32, 64, 2};
    static const Tile reuse128{128, 256, 128, 128, 32, 2};
    static const Tile scatter128{128, 256, 128, 64, 64, 2};
    static const Tile channelwise_g1_decode{128, 256, 128, 32, 64, 5};
    // Transport padding remains independent of the GEMM1 compute tile.
    static const std::initializer_list<TuningCase> k3_ep32 = {
        {4, {small16, {}, 36}, {small16}, {16}},    {8, {small16, {}, 36}, {small16}, {16}},
        {16, {small16, {}, 36}, {small16}, {16}},   {32, {small32, {}, 36}, {small32}, {32}},
        {64, {reuse128, {}, 18}, {small32}, {16}},  {128, {reuse128, {}, 18}, {scatter128}, {16}},
        {1024, {reuse128}, {}, {64}}, {2048, {reuse128}, {}, {64}},
        {4096, {reuse128}, {}, {64}},
    };
    static const Tile w4a8_g1_decode{16, 256, 128, 16, 64, 2};
    static const Tile w4a8_g2_decode{16, 256, 128, 16, 64, 2};
    static const Tile w4a8_g2_decode3{16, 256, 128, 16, 64, 3};
    static const Tile w4a8_g1_small{16, 1024, 128, 16, 64, 3};
    static const Tile w4a8_g1_medium{32, 1024, 128, 32, 64, 3};
    static const Tile w4a8_g1_large{64, 1024, 128, 64, 64, 3};
    static const Tile w4a8_g2_small{16, 512, 128, 16, 64, 3};
    static const Tile w4a8_g2_medium{32, 512, 128, 32, 64, 3};
    static const Tile w4a8_g2_large{64, 512, 128, 64, 64, 3};
    static const Tile w4a8_native160{160, 256, 128, 160, 32, 3};
    static const TuningEntry entries[] = {
        {"kimi_k3", P::W4A8, 8, 896, 16, 3584, 3072, "situ",
         {{4, {w4a8_g1_decode}, {w4a8_g2_decode}},
          {8, {w4a8_g1_decode}, {w4a8_g2_decode3}},
          {64, {w4a8_g1_decode}, {w4a8_g2_decode}},
          {128, {Tile{32, 512, 128, 32, 64, 3}}, {Tile{32, 256, 128, 32, 64, 3}}},
          {1024},
          {2048, {w4a8_native160}, {w4a8_native160}}}, true},
        {"kimi_k3", P::W4A8, 16, 896, 16, 3584, 3072, "situ",
         {{32, {w4a8_g1_decode}, {w4a8_g2_decode}}}},
        {"kimi_k3", P::W4A8, 16, 896, 16, 3584, 3072, "situ",
         {{16, {w4a8_g1_small}, {w4a8_g2_small}},
          {64, {w4a8_g1_medium}, {w4a8_g2_medium}},
          {4096, {w4a8_g1_large}, {w4a8_g2_large}}}, true},
        {"kimi_k3", P::W4A8, 32, 896, 16, 3584, 3072, "situ",
         {{8, {Tile{16, 512, 128, 16, 64, 2}, {}, 12}, {w4a8_g2_decode}},
          {16, {Tile{16, 512, 128, 16, 64, 3}}, {w4a8_g2_decode}},
          {32, {Tile{32, 512, 128, 32, 64, 3}, {}, 12}, {Tile{32, 256, 128, 32, 64, 3}}},
          {64, {w4a8_g1_large, {}, 12}, {Tile{64, 256, 128, 64, 64, 2}}},
          {128, {Tile{64, 512, 128, 64, 64, 2}, {}, 12}, {w4a8_g2_large}},
          {4096, {w4a8_g1_large}, {w4a8_g2_large}}}, true},

        {"kimi_k3",
         P::W4A16,
         8,
         896,
         16,
         3584,
         3072,
         "situ",
         {{1024, {reuse128}, {}, {64}},
          {2048, {reuse128}, {}, {64}},
          {4096, {reuse128}, {}, {64}}}},
        {"kimi_k3",
         P::W4A16,
         16,
         896,
         16,
         3584,
         3072,
         "situ",
         {{1024, {reuse128}}, {2048, {reuse128}},
          {4096, {reuse128}, {}, {256}}}},
        {"kimi_k3", P::W4A16, 32, 896, 16, 3584, 3072, "situ", k3_ep32},
        {"kimi_k3", P::W4A16, 32, 928, 16, 3584, 3072, "situ", k3_ep32},
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
        {"minimax_m3_blockwise",
         P::FP8Blockwise,
         8,
         384,
         4,
         7168,
         5120,
         "swiglu_oai",
         {{32, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {64, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {128, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {1024, {{}, {}, 4}}, {4096, {{}, {}, 4}}}},
        {"minimax_m3_blockwise",
         P::FP8Blockwise,
         16,
         384,
         4,
         7168,
         5120,
         "swiglu_oai",
         {{32, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {64, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {128, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}}}},
        {"minimax_m3_blockwise",
         P::FP8Blockwise,
         32,
         384,
         4,
         7168,
         5120,
         "swiglu_oai",
         {{32, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {64, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {1024, {{}, {}, 4}}, {4096, {{}, {}, 3}}}},
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
