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
    return result;
}

} // namespace deep_gemm::mega
