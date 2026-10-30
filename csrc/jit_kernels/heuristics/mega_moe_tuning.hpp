#pragma once

#include <array>
#include <initializer_list>
#include <optional>
#include <string>

namespace deep_gemm::mega_moe_config {

using Tile = std::array<int, 6>; // BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, stages

struct TuningWorkload {
    int ranks, experts, topk, hidden, intermediate, tokens;
    std::string activation;
};

struct Gemm1Tuning {
    std::optional<Tile> tile{};
    std::optional<int> pull_ctas{};
    std::optional<bool> dynamic_queue{}; // Unset: backend policy; false: explicitly disabled.
    std::optional<int> ctas_per_sm{}; // Target; launch is capped by the kernel's actual occupancy.
};

struct Gemm2Tuning {
    std::optional<Tile> tile{};
    // M tiles interleaved while traversing N; unset keeps the backend default.
    std::optional<int> m_tiles_per_group{};
};

struct TuningConfig {
    TuningWorkload workload;
    std::string profile;
    Gemm1Tuning gemm1;
    Gemm2Tuning gemm2;
};

// Partial overrides: omitted/empty fields retain the heuristic selection.
// Match actual tokens per rank exactly, not workspace capacity; no interpolation.
struct TuningCase {
    int tokens;
    Gemm1Tuning gemm1{};
    Gemm2Tuning gemm2{};
};

struct TuningEntry {
    const char* model;
    int ranks, experts, topk, hidden, intermediate;
    const char* activation;
    std::initializer_list<TuningCase> cases;
};

inline TuningConfig select_tuning(const TuningWorkload& w) {
    TuningConfig result{w, {}, {}, {}};
    // Entry: {label, EP, global physical experts, top-k, H, I, activation, cases}.
    // I is the activated width (GEMM1 outputs 2I); label is diagnostic, not a key.
    // Case: {tokens/rank, {G1 tile, total pull CTAs, dynamic queue, target CTAs/SM},
    //                    {G2 tile, M tiles per scheduling group}}.
    // Tiles use the BM/BN/BK/WM/WN/stages order defined above.
    // Measured with tests/bench_mega_moe.py on ZW-M890P (39 CUs). This records
    // measurement provenance, not a hardware restriction: pull counts are capped
    // to available CUs, tiles are resource-checked, and launch uses actual occupancy.
    static const TuningEntry entries[] = {
        {"deepseek_v4_pro", 8, 384, 6, 7168, 3072, "swiglu",
         {{32, {{}, {}, {}, 4}},
          {64, {{}, {}, {}, 4}, {Tile{32, 256, 64, 32, 64, 2}}},
          {128, {{}, {}, {}, 4}, {Tile{32, 256, 64, 32, 64, 2}}},
          {1024, {{}, 4}, {{}, 1}},
          {2048, {{}, 4}},
          {4096, {{}, 4}}}},
        {"deepseek_v4_flash", 4, 256, 6, 4096, 2048, "swiglu",
         {{32, {{}, {}, {}, 4}},
          {64, {{}, {}, {}, 4}},
          {128, {{}, {}, {}, 4}, {Tile{32, 256, 64, 32, 64, 2}}},
          {1024, {{}, 8}},
          {4096, {{}, 8}}}},
        {"glm52", 8, 256, 8, 6144, 2048, "swiglu",
         {{32, {{}, 32, {}, 4}},
          {64, {Tile{32, 256, 128, 32, 64, 3}, 32, {}, 2}, {Tile{32, 256, 64, 32, 64, 3}}},
          {128, {{}, {}, {}, 2}, {Tile{64, 256, 64, 32, 64, 3}}},
          {2048, {{}, 8}, {{}, 1}},
          {4096, {{}, 8}, {{}, 1}}}},
        {"kimi_k3", 8, 896, 16, 3584, 3072, "situ",
         {{32, {{}, {}, {}, 4}},
          {64, {{}, {}, {}, 4}},
          {128, {{}, {}, {}, 4}},
          {2048, {{}, 4}},
          {4096, {{}, 8}}}},
        {"kimi_k3", 16, 896, 16, 3584, 3072, "situ",
         {{32, {{}, {}, {}, 4}},
          {64, {{}, {}, {}, 4}},
          {128, {Tile{64, 256, 64, 32, 64, 3}, {}, {}, 2}},
          {1024, {{}, 4}},
          {2048, {{}, 8}},
          {4096, {{}, 8}}}},
        {"qwen38_2_4t", 8, 512, 10, 8192, 2048, "swiglu",
         {{32, {{}, 12, {}, 4}},
          {64, {{}, {}, {}, 4}},
          {128, {Tile{64, 256, 64, 32, 64, 3}, {}, {}, 2}},
          {4096, {{}, 8}, {{}, 1}}}},
        {"qwen38_2_4t", 16, 512, 10, 8192, 2048, "swiglu",
         {{32, {{}, {}, {}, 4}},
          {64, {{}, {}, {}, 4}},
          {128, {{}, {}, {}, 2}},
          {2048, {{}, 8}, {{}, 1}},
          {4096, {{}, 8}, {{}, 1}}}},
        {"mimo_v26_pro", 8, 384, 8, 6144, 2048, "swiglu",
         {{32, {{}, 12, {}, 4}},
          {64, {{}, 24, {}, 4}, {Tile{32, 256, 64, 32, 32, 2}}},
          {128, {Tile{64, 256, 64, 32, 64, 3}, {}, {}, 2}, {Tile{64, 256, 64, 32, 64, 3}}},
          {2048, {{}, 8}},
          {4096, {{}, 8}, {{}, 1}},
          {8192, {}, {Tile{128, 256, 64, 64, 64, 3}}}}},
    };
    for (const auto& entry : entries) {
        if (w.ranks == entry.ranks && w.experts == entry.experts &&
            w.topk == entry.topk && w.hidden == entry.hidden && w.intermediate == entry.intermediate &&
            w.activation == entry.activation) {
            for (const auto& c : entry.cases) {
                if (w.tokens == c.tokens) {
                    result.profile = entry.model;
                    result.gemm1 = c.gemm1;
                    result.gemm2 = c.gemm2;
                    return result;
                }
            }
        }
    }
    return result;
}

} // namespace deep_gemm::mega_moe_config
