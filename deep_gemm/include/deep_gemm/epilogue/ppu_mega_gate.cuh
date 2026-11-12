// Ported from DeepGEMM 5081f8513af57e61696de4f4f04b0e707ffb503e (MIT).
#pragma once
#include <cmath>
#include <cstdint>
#include <hggc_runtime.h>
#include <acrand_kernel.h>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/tix/utils.cuh>

namespace deep_gemm::epilogue::mega_gate {

using namespace layout::mega_gate;

__device__ __forceinline__ void device_check(bool condition) {
    if (!condition) __builtin_trap();
}

// Store GEMM scores and defer any invalid-score trap until synchronization completes.
template <bool UseSigmoid, int S>
__device__ __forceinline__ void store_checked_scores(const KernelArgs& a, int token, int expert, float* dst, float lo, float hi,
                                                      bool check_activated, bool& gate_is_valid) {
    bool valid;
    if constexpr (UseSigmoid) {
        // The aligned score pair is either wholly inside the expert range or wholly padding.
        if (token >= a.num_tokens || expert >= a.routing.num_routed_experts) {
            valid = true;
        } else if constexpr (S == 1) {
            valid = !isnan(lo) && !isnan(hi);
        } else {
            // Learned routing checks the reduced logits. Fixed/random routes skip that loop.
            const bool skips_reduction_check = (a.fix_routing_mask && a.fix_routing_mask[token]) ||
                                               (a.force_random && a.force_random[token]);
            valid = !skips_reduction_check || (isfinite(lo) && isfinite(hi));
        }
    } else {
        valid = check_activated ? lo < INFINITY && hi < INFINITY : isfinite(lo) && isfinite(hi);
    }
    // Masked input may contain NaN/inf. The final MMA tile can extend past the mask allocation.
    gate_is_valid &= valid || (a.mask && (token >= a.num_tokens || !a.mask[token]));
    *reinterpret_cast<float2*>(dst) = make_float2(lo, hi);
}

__device__ __forceinline__ float sqrt_softplus(const float& value) {
    const auto softplus_precise = log1pf(expf(value));
    const auto softplus = value > 20.0f ? value : softplus_precise;
    return sqrtf(softplus);
}

template <uint32_t kNumValues>
__device__ __forceinline__ void sqrt_softplus(float (&values)[kNumValues]) {
    float exponentials[kNumValues], softplus[kNumValues];
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
        exponentials[value_idx] = expf(values[value_idx]);
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
        softplus[value_idx] = log1pf(exponentials[value_idx]);
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
        values[value_idx] = sqrtf(values[value_idx] > 20.0f ? values[value_idx] : softplus[value_idx]);
}

template <bool UseSigmoid>
__device__ __forceinline__ float gate_score(float value) {
    if constexpr (UseSigmoid)
        return 1.0f / (1.0f + expf(-value));
    else
        return sqrt_softplus(value);
}

template <bool UseSigmoid, uint32_t N>
__device__ __forceinline__ void gate_score(float (&values)[N]) {
    if constexpr (UseSigmoid) {
        // Match the vector sigmoid schedule in DeepGEMM 66081d4c (sm100_mega_gate.cuh).
        float exponentials[N];
        #pragma unroll
        for (uint32_t i = 0; i < N; ++i)
            exponentials[i] = expf(-values[i]);
        #pragma unroll
        for (uint32_t i = 0; i < N; ++i)
            values[i] = 1.0f / (1.0f + exponentials[i]);
    } else {
        sqrt_softplus(values);
    }
}

__device__ __forceinline__ void warp_reduce_best(const float& score, int& expert_idx) {
    const auto best_score = tix::reduce_max_sync(score);
    // Ties resolve to the smallest expert index
    const auto tied_expert_idx = score == best_score and expert_idx >= 0 ?
        static_cast<uint32_t>(expert_idx) : 0xffffffffu;
    expert_idx = static_cast<int>(__reduce_min_sync(0xffffffff, tied_expert_idx));
}

__device__ __forceinline__ void compare_swap_best_first(float& lhs_score, int& lhs_expert_idx, float& rhs_score, int& rhs_expert_idx) {
    if (rhs_score > lhs_score or (rhs_score == lhs_score and rhs_expert_idx < lhs_expert_idx)) {
        const auto score = lhs_score;
        const auto expert_idx = lhs_expert_idx;
        lhs_score = rhs_score, lhs_expert_idx = rhs_expert_idx;
        rhs_score = score, rhs_expert_idx = expert_idx;
    }
}

__device__ __forceinline__ uint32_t sort4_best_first(float& score_0, float& score_1, float& score_2, float& score_3) {
    int idx_0 = 0, idx_1 = 1, idx_2 = 2, idx_3 = 3;
    compare_swap_best_first(score_0, idx_0, score_1, idx_1);
    compare_swap_best_first(score_2, idx_2, score_3, idx_3);
    compare_swap_best_first(score_0, idx_0, score_2, idx_2);
    compare_swap_best_first(score_1, idx_1, score_3, idx_3);
    compare_swap_best_first(score_1, idx_1, score_2, idx_2);
    return static_cast<uint32_t>(idx_0 | idx_1 << 2 | idx_2 << 4 | idx_3 << 6);
}

template <uint32_t kNumExpertWaves, uint32_t kNumTopk>
__device__ __forceinline__ void select_warp_topk(float (&scores)[kNumExpertWaves * kNumExpertsPerLaneVector], int& selected_expert_idx) {
    static_assert(kNumExpertsPerLaneVector == 4, "The lane vector is sorted with a 4-element network");
    uint32_t permutations[kNumExpertWaves];
    #pragma unroll
    for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
        const auto local_expert_base_idx = expert_wave_idx * kNumExpertsPerLaneVector;
        permutations[expert_wave_idx] = sort4_best_first(scores[local_expert_base_idx + 0],
                                                         scores[local_expert_base_idx + 1],
                                                         scores[local_expert_base_idx + 2],
                                                         scores[local_expert_base_idx + 3]);
    }

    // Every round, each lane offers the best unconsumed candidate of each wave (a 3-bit cursor into the
    // 2-bit permutation), the warp picks the best one and only the winning lane advances that cursor
    const auto lane_idx = (threadIdx.x % 32);
    uint32_t packed_cursors = 0;
    #pragma unroll
    for (uint32_t output_idx = 0; output_idx < kNumTopk; ++ output_idx) {
        auto best_score = -INFINITY;
        int best_expert_idx = -1;
        uint32_t best_expert_wave_idx = 0;
        #pragma unroll
        for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
            const auto cursor = packed_cursors >> (expert_wave_idx * 3) & 7;
            const auto local_expert_base_idx = expert_wave_idx * kNumExpertsPerLaneVector;
            const auto candidate_score = cursor == 4 ? -INFINITY :
                                         cursor & 2 ? (cursor & 1 ? scores[local_expert_base_idx + 3] : scores[local_expert_base_idx + 2]) :
                                                      (cursor & 1 ? scores[local_expert_base_idx + 1] : scores[local_expert_base_idx + 0]);
            const auto candidate_offset = permutations[expert_wave_idx] >> (cursor * 2) & 3;
            const auto candidate_expert_idx = static_cast<int>(expert_wave_idx * kNumExpertsPerWave +
                                                               lane_idx * kNumExpertsPerLaneVector + candidate_offset);
            if (candidate_score > best_score) {
                best_score = candidate_score;
                best_expert_idx = candidate_expert_idx;
                best_expert_wave_idx = expert_wave_idx;
            }
        }
        const auto local_best_expert_idx = best_expert_idx;
        warp_reduce_best(best_score, best_expert_idx);
        if (local_best_expert_idx == best_expert_idx)
            packed_cursors += 1u << (best_expert_wave_idx * 3);
        if (lane_idx == output_idx)
            selected_expert_idx = best_expert_idx;
    }
}

template <uint32_t kNumExpertWaves>
__device__ __forceinline__ float select_warp_expert_value(const float (&values)[kNumExpertWaves * kNumExpertsPerLaneVector],
                                              const int& expert_idx) {
    const auto selected_value_idx = expert_idx >= 0 ?
        expert_idx / static_cast<int>(kNumExpertsPerWave) * static_cast<int>(kNumExpertsPerLaneVector) +
        expert_idx % static_cast<int>(kNumExpertsPerLaneVector) : -1;
    const auto source_lane_idx = expert_idx >= 0 ?
        static_cast<uint32_t>(expert_idx % static_cast<int>(kNumExpertsPerWave) / static_cast<int>(kNumExpertsPerLaneVector)) : 0u;
    auto selected_value = 0.0f;
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumExpertWaves * kNumExpertsPerLaneVector; ++ value_idx) {
        const auto exchanged = __shfl_sync(0xffffffff, values[value_idx], source_lane_idx);
        if (selected_value_idx == static_cast<int>(value_idx))
            selected_value = exchanged;
    }
    return selected_value;
}

template <uint32_t kNumTopk, bool kHasUnmappedTopkIdx>
__device__ __forceinline__ void store_token_outputs(const RoutingArgs& args, const uint32_t& global_token_idx, const uint32_t& lane_idx,
                                        const int64_t& physical_expert_idx, const float& weight, const int64_t& logical_expert_idx) {
    const auto num_physical_topk = kNumTopk + args.num_shared_experts;
    if (lane_idx < num_physical_topk) {
        const auto output_idx = global_token_idx * num_physical_topk + lane_idx;
        args.topk_idx[output_idx] = physical_expert_idx;
        args.topk_weights[output_idx] = weight;
    }
    if constexpr (kHasUnmappedTopkIdx) {
        if (lane_idx < kNumTopk)
            args.unmapped_topk_idx[global_token_idx * args.unmapped_topk_idx_stride + lane_idx] = logical_expert_idx;
    }
}

template <uint32_t kNumTopk, bool kHasUnmappedTopkIdx, bool kHasPhysicalMap, bool UseSigmoid>
__device__ __forceinline__ bool store_topk_token(const RoutingArgs& args, const int* routed_logical_count,
                                     const uint32_t& global_token_idx, const uint32_t& lane_idx,
                                     int selected_expert_idx, const float& selected_unbiased_score) {
    static_assert(kNumTopk <= 32, "Top-k slots exceed a warp");
    const bool valid = lane_idx >= kNumTopk or
        (selected_expert_idx >= 0 and selected_expert_idx < static_cast<int>(args.num_routed_experts));
    // Like upstream, keep an invalid selection in bounds until the final gate assertion.
    if (lane_idx < kNumTopk)
        selected_expert_idx = min(max(selected_expert_idx, 0), static_cast<int>(args.num_routed_experts) - 1);
    const auto num_physical_topk = kNumTopk + args.num_shared_experts;
    // The shared experts follow the routed ones and take the slots after the top-k
    if (lane_idx >= kNumTopk and lane_idx < num_physical_topk)
        selected_expert_idx = static_cast<int>(lane_idx + args.num_routed_experts - kNumTopk);

    auto physical_expert_idx = selected_expert_idx;
    if constexpr (kHasPhysicalMap) {
        if (lane_idx < num_physical_topk) {
            const auto logical_expert_idx = static_cast<uint32_t>(selected_expert_idx);
            // The SMEM cache only covers the routed experts
            const auto num_duplicates = static_cast<uint32_t>((lane_idx < kNumTopk ? routed_logical_count : args.logical_count)[logical_expert_idx]);
            device_check(num_duplicates > 0 and num_duplicates <= args.num_duplicate_experts);
            // Spread the tokens of a logical expert over its duplicates, differently on every rank
            const auto token_hash = args.rank_idx + global_token_idx * 23333u;
            // Exact for every power-of-two count; retain the general modulo fallback.
            auto duplicate_idx = token_hash & (num_duplicates - 1u);
            if ((num_duplicates & (num_duplicates - 1u)) != 0)
                duplicate_idx = token_hash % num_duplicates;
            physical_expert_idx = args.to_physical_map[logical_expert_idx * args.num_duplicate_experts + duplicate_idx];
        }
    }

    // The smallest shuffle width covering the routed lanes, the other lanes hold zero
    constexpr uint32_t kSumWidth = kNumTopk <= 8 ? 8 : (kNumTopk <= 16 ? 16 : 32);
    const float sum = math::warp_reduce_sum<kSumWidth>(selected_unbiased_score);
    // For any K split count, GLM divides by the sum of the selected, unbiased sigmoid scores.
    // Even a tiny positive sum gives routed weights that sum to 1 before scaling.
    // When all scores are zero, a denominator of 1 keeps all weights zero.
    // DeepSeek V4.1 sqrtsoftplus uses sum + 1e-20.
    const float topk_sum = UseSigmoid ? (sum > 0.0f ? sum : 1.0f) : sum + 1e-20f;
    const auto selected_weight = lane_idx < kNumTopk ? selected_unbiased_score / topk_sum * args.routed_scaling_factor : 1.0f;
    store_token_outputs<kNumTopk, kHasUnmappedTopkIdx>(args, global_token_idx, lane_idx,
                                                       physical_expert_idx, selected_weight, selected_expert_idx);
    return valid;
}

template <uint32_t kNumTopk, bool kHasUnmappedTopkIdx, bool kHasPhysicalMap>
__device__ __forceinline__ void store_random_topk_token(const RoutingArgs& args, const uint32_t& global_token_idx, const uint32_t& lane_idx) {
    const auto num_physical_topk = kNumTopk + args.num_shared_experts;
    auto num_physical_experts = args.num_routed_experts + args.num_shared_experts;
    if constexpr (kHasPhysicalMap) {
        auto local_count = 0u;
        for (uint32_t logical_expert_idx = lane_idx; logical_expert_idx < num_physical_experts; logical_expert_idx += 32)
            local_count += static_cast<uint32_t>(args.logical_count[logical_expert_idx]);
        num_physical_experts = math::warp_reduce_sum<32>(local_count);
    }
    device_check(num_physical_experts >= num_physical_topk);

    // Sample `num_physical_topk` distinct experts: lane `i` draws from `n - i`, then ranks below every earlier draw
    // that is not larger shift up by one
    acrandStatePhilox4_32_10_t rng_state;
    acrand_init(args.rank_idx, global_token_idx * 32 + lane_idx, 0, &rng_state);
    const auto random_weight = acrand_uniform(&rng_state);
    constexpr uint32_t kInvalidIdx = 0xffffffffu;
    auto candidate_idx = lane_idx < num_physical_topk ?
        acrand(&rng_state) % (num_physical_experts - lane_idx) : kInvalidIdx;
    auto selected_expert_idx = kInvalidIdx;
    for (uint32_t output_idx = 0; output_idx < num_physical_topk; ++ output_idx) {
        const auto min_candidate_idx = __reduce_min_sync(0xffffffff, candidate_idx);
        const auto min_lane_idx = __reduce_min_sync(0xffffffff, candidate_idx == min_candidate_idx ? lane_idx : kInvalidIdx);
        if (candidate_idx != kInvalidIdx) {
            if (min_lane_idx == lane_idx) {
                selected_expert_idx = candidate_idx;
                candidate_idx = kInvalidIdx;
            } else if (candidate_idx >= min_candidate_idx and min_lane_idx < lane_idx) {
                ++ candidate_idx;
            }
        }
    }
    store_token_outputs<kNumTopk, kHasUnmappedTopkIdx>(args, global_token_idx, lane_idx,
                                                       static_cast<int64_t>(selected_expert_idx), random_weight, -1);
}

} // namespace deep_gemm::epilogue::mega_gate
