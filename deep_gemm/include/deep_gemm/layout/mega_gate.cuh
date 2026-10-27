#pragma once
#include <cstdint>

namespace deep_gemm::layout::mega_gate {
constexpr uint32_t BLOCK_K = 64;
constexpr uint32_t kBlockTokens = 16;
constexpr uint32_t kExpertAlignment = 128;
constexpr uint32_t kNumThreads = 256;
constexpr uint32_t kNumMaxTokens = 1u << 20;
constexpr uint32_t kNumExpertsPerLaneVector = 4;
constexpr uint32_t kNumExpertsPerWave = 128;

// Same routing layout as upstream DeepGEMM 5081f851 (MIT).
struct RoutingArgs {
    const int* to_physical_map;
    const int* logical_count;
    int64_t* topk_idx;
    int64_t* unmapped_topk_idx;
    float* topk_weights;
    int64_t unmapped_topk_idx_stride;
    uint32_t num_routed_experts;
    uint32_t num_shared_experts;
    uint32_t num_duplicate_experts;
    uint32_t rank_idx;
    float routed_scaling_factor;
};

struct KernelArgs {
    const void* x;
    const void* weight;
    const float* bias;
    const float* image_bias;
    const bool* image_token_mask;
    const bool* mask;
    const bool* fix_routing_mask;
    const bool* force_random;
    RoutingArgs routing;
    float* scratch; // FP32 scores; producer selects legacy or token-tile-major K planes
    int* score_barriers; // One zero-initialized completion counter per token tile.
    uint32_t num_tokens;
    uint32_t scratch_token_base = 0; // Shared scratch starts at the current CTA's token tile.
};
} // namespace deep_gemm::layout::mega_gate

namespace deep_gemm {
namespace mega_gate_layout = layout::mega_gate;
}
