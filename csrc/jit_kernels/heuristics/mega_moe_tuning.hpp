#pragma once

#include <array>
#include <initializer_list>
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
    std::optional<bool> dynamic_queue{}; // Unset: backend policy; false: explicitly disabled.
    bool separate_dispatch = false;
    std::optional<int> dispatch_threads{};
};

struct Gemm2Tuning {
    std::optional<Tile> tile{};
    // M tiles interleaved while traversing N; unset keeps the backend default.
    std::optional<int> m_tiles_per_group{};
    // Optional applicability limits for measured scatter tiles. G1 is unaffected.
    std::optional<int> capacity{};
    bool requires_no_replicas = false;
};

struct BufferTuning {
    std::optional<int> block_m{};
    // Communication direction is chosen collectively when the workspace is
    // allocated, never from a rank-local token count during forward.
    bool push_dispatch = false;
};

struct TuningConfig {
    TuningWorkload workload;
    std::string profile;
    Gemm1Tuning gemm1;
    Gemm2Tuning gemm2;
    BufferTuning buffer;
};

// These are measured overrides, not complete kernel configurations. Empty
// fields use the family-specific selector. Token counts match
// exactly: no nearest-model matching or interpolation into unmeasured ranges.
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
};

inline TuningConfig select_tuning(const TuningWorkload& w, const std::string& device, int sms,
                                   bool enabled = true) {
    TuningConfig result{w, {}, {}, {}, {}};
    if (!enabled || device != "ZW-M890P" || sms != 39)
        return result;
    using P = Precision;
    using M = PullMode;
    // Separate dispatch and GEMM launches keep their own CTA geometry. The
    // buffer tile describes transport padding, independently of the MMA tile.
    const auto separate_gemm1 = [](const Tile& tile, int dispatch_threads = 512) {
        Gemm1Tuning c;
        c.tile = tile;
        c.dynamic_queue = false;
        c.separate_dispatch = true;
        c.dispatch_threads = dispatch_threads;
        return c;
    };
    static const Tile small16{16, 256, 128, 16, 64, 2};
    static const Tile small32{32, 256, 128, 32, 64, 2};
    static const Tile reuse128{128, 256, 128, 128, 32, 2};
    static const Tile scatter128{128, 256, 128, 64, 64, 2};
    // K3 EP32: separate CTA geometry for transport and GEMM. Prefill uses
    // source push only when the common allocation bucket selects it.
    static const std::initializer_list<TuningCase> k3_ep32 = {
        {4,   separate_gemm1(small16),  {small16},    {16}},
        {8,   separate_gemm1(small16),  {small16},    {16}},
        {16,  separate_gemm1(small16),  {small16},    {16}},
        {32,  separate_gemm1(small32),  {small32},    {32}},
        {64,  separate_gemm1(reuse128), {small32},    {16}},
        {128, separate_gemm1(reuse128), {scatter128}, {16}},
        {1024, separate_gemm1(reuse128, 1024), {}, {64, true}},
        {2048, separate_gemm1(reuse128, 1024), {}, {64, true}},
        {4096, separate_gemm1(reuse128, 1024), {}, {64, true}},
    };
    static const TuningEntry entries[] = {
        {"deepseek_v4_pro", P::W4A4, 8, 384, 6, 7168, 3072, "swiglu",
         {{32, {Tile{16, 256, 128, 16, 64, 3}, {}, 3}, {Tile{16, 256, 128, 16, 64, 3}}},
          {1024, {Tile{256, 256, 128, 64, 64, 3}, {}, 4}, {{}, 1}},
          {2048, {{}, {}, 4}}, {4096, {{}, {}, 4}}}},
        {"deepseek_v4_flash", P::W4A4, 4, 256, 6, 4096, 2048, "swiglu",
         {{1024, {Tile{256, 256, 128, 64, 64, 3}, {}, 8}},
          {2048, {Tile{256, 256, 64, 64, 64, 4}, {}, 6}},
          {4096, {{}, {}, 8}}}},
        {"glm52", P::W4A4, 8, 256, 8, 6144, 2048, "swiglu",
         {{1024, {Tile{256, 256, 128, 64, 64, 3}, {}, 6}},
          {2048, {Tile{256, 256, 128, 64, 64, 3}, {}, 8}, {{}, 1}},
          {4096, {Tile{256, 256, 128, 64, 64, 3}, {}, 8}, {{}, 1}}}},
        {"kimi_k3", P::W4A4, 8, 896, 16, 3584, 3072, "situ",
         {{2048, {{}, {}, 4}}, {4096, {{}, {}, 8}}}},
        {"kimi_k3", P::W4A4, 16, 896, 16, 3584, 3072, "situ",
         {{1024, {{}, {}, 4}}, {2048, {{}, {}, 8}}, {4096, {{}, {}, 8}}}},
        {"qwen38_2_4t", P::W4A4, 8, 512, 10, 8192, 2048, "swiglu",
         {{1024, {Tile{256, 256, 128, 64, 64, 3}, {}, 6}},
          {2048, {Tile{256, 256, 128, 64, 64, 3}, {}, 6}},
          {4096, {Tile{256, 256, 128, 64, 64, 3}, {}, 8}, {{}, 1}}}},
        {"qwen38_2_4t", P::W4A4, 16, 512, 10, 8192, 2048, "swiglu",
         {{1024, {{}, {}, 6}}, {2048, {{}, {}, 8}, {{}, 1}}, {4096, {{}, {}, 8}, {{}, 1}}}},
        {"kimi_k3", P::W4A16, 8, 896, 16, 3584, 3072, "situ",
         {{1024, separate_gemm1(reuse128), {}, {64}},
          {2048, separate_gemm1(reuse128), {}, {64}},
          {4096, separate_gemm1(reuse128), {}, {64}}}},
        {"kimi_k3", P::W4A16, 16, 896, 16, 3584, 3072, "situ",
         {{1024, separate_gemm1(reuse128)}, {2048, {{}, 256}}, {4096, {{}, 256}}}},
        {"kimi_k3", P::W4A16, 32, 896, 16, 3584, 3072, "situ", k3_ep32},
        {"kimi_k3", P::W4A16, 32, 928, 16, 3584, 3072, "situ", k3_ep32},
        {"minimax_m3", P::FP8Channelwise, 8, 128, 4, 6144, 3072, "swiglu_oai",
         {{1024, {{}, {}, 3, M::Triple}}, {2048, {{}, {}, 3, M::Triple}},
          {4096, {{}, {}, 3, M::Triple}}}},
        {"minimax_m3_blockwise", P::FP8Blockwise, 8, 384, 4, 7168, 5120, "swiglu_oai",
         {{64, {{}, 32, {}, {}, true}, {Tile{32, 256, 128, 16, 64, 3}}},
          {1024, {{}, {}, 4}}, {4096, {{}, {}, 4}}}},
        {"minimax_m3_blockwise", P::FP8Blockwise, 16, 384, 4, 7168, 5120, "swiglu_oai",
         {{64, {{}, 32, {}, {}, true}, {Tile{32, 256, 128, 16, 64, 3}}}}},
        {"minimax_m3_blockwise", P::FP8Blockwise, 32, 384, 4, 7168, 5120, "swiglu_oai",
         {{1024, {{}, {}, 4}}, {4096, {{}, {}, 3}}}},
        {"glm52", P::FP8Channelwise, 8, 256, 8, 6144, 2048, "swiglu",
         {{32, {{}, 128, {}, {}, true}, {Tile{128, 256, 128, 32, 64, 4}, {}, 4096, true}},
          {1024, {{}, {}, 9, M::Triple, true}},
          {2048, {{}, {}, 9, M::Triple}}, {4096, {{}, {}, 9, M::Triple}}}},
        {"glm53_flash", P::FP8Channelwise, 4, 288, 8, 4096, 2048, "swiglu",
         {{1024, {{}, 128, 6, {}, true}, {Tile{128, 256, 128, 32, 64, 3}}},
          {2048}, {4096}}},
        {"qwen38_flash", P::FP8Channelwise, 4, 512, 10, 2560, 640, "swiglu",
         {{1024, {{}, {}, 9, M::Triple}, {Tile{128, 256, 128, 32, 64, 3}}},
          {2048, {{}, {}, 12}}, {4096, {{}, {}, 12}}}},
        {"mimo_v26_pro", P::W4A4, 8, 384, 8, 6144, 2048, "swiglu",
         {{2048, {{}, {}, 8}}, {4096, {{}, {}, 8}, {{}, 1}},
          {8192, {}, {Tile{128, 256, 64, 64, 64, 3}}}}},
    };
    for (const auto& entry : entries) {
        if (w.precision == entry.precision && w.ranks == entry.ranks && w.experts == entry.experts &&
            w.topk == entry.topk && w.hidden == entry.hidden && w.intermediate == entry.intermediate &&
            w.activation == entry.activation) {
            for (const auto& c : entry.cases) {
                if (w.tokens == c.tokens) {
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

inline bool channel_pair_staging_default(int m, int n, int k, int groups, int expected_m,
                                          int ranks, int capacity, int topk) {
    struct Entry { int m, n, k, groups, expected_m, ranks, capacity, topk; };
    static constexpr Entry entries[] = {
        {4096, 6144, 2048, 32, 128, 8, 512, 8},
    };
    for (const auto& entry : entries)
        if (m == entry.m && n == entry.n && k == entry.k && groups == entry.groups &&
            expected_m == entry.expected_m && ranks == entry.ranks && capacity == entry.capacity &&
            topk == entry.topk)
            return true;
    return false;
}

inline const char* pull_mode_name(PullMode mode) {
    return mode == PullMode::Paired ? "paired" : mode == PullMode::Triple ? "triple" : "single";
}

} // namespace deep_gemm::mega_moe_config
