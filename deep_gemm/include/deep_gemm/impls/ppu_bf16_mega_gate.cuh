#pragma once
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <hggc_mma.h>
#include <cooperative_groups.h>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/epilogue/ppu_mega_gate.cuh>
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
#include <deep_gemm/cute_free/copy/aiu_g2s_atom.cuh>
#include <deep_gemm/cute_free/copy/tsm_s2r_atom.cuh>
#include <deep_gemm/cute_free/mma/mma_atom.cuh>
#include <deep_gemm/cute_free/mainloop/mainloop.cuh>
#include <deep_gemm/cute_free/epilogue/warp_k_reduce.cuh>
#endif

namespace deep_gemm {
namespace mega_gate_detail {

__device__ __forceinline__ uint32_t ordered_score_key(float value) {
    const uint32_t bits = __float_as_uint(value == 0.0f ? 0.0f : value);
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}
__device__ __forceinline__ uint32_t warp_max_key(uint32_t key) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    return __reduce_max_sync(0xffffffff,key);
#else
    #pragma unroll
    for (int offset=1;offset<32;offset*=2) {
        const uint32_t other=__shfl_xor_sync(0xffffffff,key,offset);
        key=key>other?key:other;
    }
    return key;
#endif
}

struct ScoreBarrier {
    // Only for a co-resident cooperative grid. Counts advance monotonically:
    // a fast producer may enter the next round before another waiter observes this one.
    __device__ static void arrive_and_wait(int* counter, int target) {
        asm volatile("ppu.fence.acq_rel.gpu;" ::: "memory");
        atomicAdd(counter, 1);
        int state;
        do {
            asm volatile("ppu.ld.global.acquire.gpu.b32 %0, [%1];"
                         : "=r"(state) : "l"(counter) : "memory");
        } while (state < target);
    }

    // Called by thread 0 AFTER the scratch writers have joined at __syncthreads.
    // Same release-fence/atomic + acquire-load pattern as ACTLIZE cutlass::Barrier.
    // The last producer becomes the consumer; earlier producers never spin.
    __device__ static bool arrive_last(int* counter, int producers) {
        asm volatile("ppu.fence.acq_rel.gpu;" ::: "memory");
        const int old = atomicAdd(counter, 1);
        if (old != producers - 1) return false;
        int state;
        asm volatile("ppu.ld.global.acquire.gpu.b32 %0, [%1];"
                     : "=r"(state) : "l"(counter) : "memory");
        epilogue::mega_gate::device_check(state == producers);
        return true;
    }
};

template<int E, bool CompactScores, int ScoreStride, int ScoreBlockTokens>
__device__ __forceinline__ float* score_ptr(const mega_gate_layout::KernelArgs& a, int token, int expert) {
    const int row = token - a.scratch_token_base;
    const int64_t offset = CompactScores ? int64_t(row) * a.routing.num_routed_experts :
        (int64_t(row / ScoreBlockTokens) * ScoreStride * ScoreBlockTokens + row % ScoreBlockTokens) * E;
    return a.scratch + offset + expert;
}

template<int E, int S, bool CompactScores=false, int ScoreStride=S, int ScoreBlockTokens=1>
__device__ __forceinline__ float reduced_score(const mega_gate_layout::KernelArgs& a, int token, int expert) {
    static_assert(!CompactScores or S == 1);
    const float* p = score_ptr<E, CompactScores, ScoreStride, ScoreBlockTokens>(a, token, expert);
    float value = p[0];
    #pragma unroll
    for (int s = 1; s < S; ++s) value += p[s * ScoreBlockTokens * E];
    return value;
}

// Keep each sorted four-entry list's head in a fixed register slot. Ordered
// keys avoid the per-round float/key round trip and packed-cursor gathers.
template<int Waves, int Topk>
__device__ __forceinline__ void select_merge_topk(float (&scores)[Waves * 4], int& selected) {
    const int lane = threadIdx.x % 32;
    uint32_t keys[Waves * 4], permutations[Waves];
    #pragma unroll
    for (int w = 0; w < Waves; ++w) {
        permutations[w] = epilogue::mega_gate::sort4_best_first(
            scores[w*4], scores[w*4+1], scores[w*4+2], scores[w*4+3]);
        #pragma unroll
        for (int v = 0; v < 4; ++v) keys[w*4+v] = ordered_score_key(scores[w*4+v]);
    }
    #pragma unroll
    for (int output = 0; output < Topk; ++output) {
        uint32_t best = keys[0];
        int best_wave = 0;
        #pragma unroll
        for (int w = 1; w < Waves; ++w) {
            if (keys[w*4] > best) best = keys[w*4], best_wave = w;
        }
        int offered = 0;
        #pragma unroll
        for (int w = 0; w < Waves; ++w)
            if (best_wave == w) offered = w*128 + lane*4 + int(permutations[w] & 3);
        const uint32_t winner_key = warp_max_key(best);
        const uint32_t winner = epilogue::mega_gate::warp_min(
            best == winner_key ? (uint32_t(offered) << 5) | uint32_t(lane) : 0xffffffffu);
        if (lane == output) selected = int(winner >> 5);
        #pragma unroll
        for (int w = 0; w < Waves; ++w) {
            if (lane == int(winner & 31) and best_wave == w) {
                #pragma unroll
                for (int v = 0; v < 3; ++v) keys[w*4+v] = keys[w*4+v+1];
                keys[w*4+3] = 0x007fffffu;
                permutations[w] >>= 2;
            }
        }
    }
}

// Exact selection over a small lane-local list. Invalid padded candidates have
// expert -1; equal scores select the smaller global expert index.
template<int Values, int Topk>
__device__ __forceinline__ void select_records(float (&scores)[Values], float (&ranks)[Values],
        int (&experts)[Values], int& selected, float& selected_score) {
    const int lane = threadIdx.x % 32;
    uint32_t keys[Values];
    #pragma unroll
    for (int v = 0; v < Values; ++v) keys[v] = ordered_score_key(ranks[v]);
    #pragma unroll
    for (int output = 0; output < Topk; ++output) {
        uint32_t best = 0x007fffffu, offered = 0xffffffffu;
        float score = 0.0f;
        #pragma unroll
        for (int v = 0; v < Values; ++v) {
            const uint32_t expert = uint32_t(experts[v]);
            if (experts[v] >= 0 and (keys[v] > best or (keys[v] == best and expert < offered)))
                best = keys[v], offered = expert, score = scores[v];
        }
        const uint32_t best_key = warp_max_key(best);
        const uint32_t winner = epilogue::mega_gate::warp_min(
            best == best_key and offered != 0xffffffffu ? (offered << 5) | uint32_t(lane) : 0xffffffffu);
        const float winner_score = __shfl_sync(0xffffffff, score, winner & 31);
        if (lane == output) {
            selected = winner == 0xffffffffu ? -1 : int(winner >> 5);
            selected_score = winner == 0xffffffffu ? 0.0f : winner_score;
        }
        #pragma unroll
        for (int v = 0; v < Values; ++v)
            if (winner != 0xffffffffu and experts[v] == int(winner >> 5)) experts[v] = -1;
    }
}

template<int E, int S, int N, int Topk, int Threads, int M>
__device__ __forceinline__ void store_group_candidates(const mega_gate_layout::KernelArgs& a,
        int token_base, int expert_base) {
    using namespace epilogue::mega_gate;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int Values = N / 32;
    for (int token = token_base + warp; token < token_base + M and token < a.num_tokens; token += Threads / 32) {
        float scores[Values], ranks[Values];
        int experts[Values];
        bool finite = true;
        #pragma unroll
        for (int v = 0; v < Values; ++v) {
            const int expert = expert_base + lane + v * 32;
            scores[v] = reduced_score<E, S>(a, token, expert);
            finite &= isfinite(scores[v]);
            experts[v] = expert < a.routing.num_routed_experts ? expert : -1;
        }
        device_check(finite);
        sqrt_softplus(scores);
        #pragma unroll
        for (int v = 0; v < Values; ++v) {
            ranks[v] = experts[v] >= 0 ? scores[v] + a.bias[experts[v]] : -INFINITY;
            finite &= experts[v] < 0 or isfinite(ranks[v]);
        }
        device_check(finite);
        int selected = -1;
        float value = 0.0f;
        select_records<Values, Topk>(scores, ranks, experts, selected, value);
        __syncwarp();
        if (lane < Topk) {
            uint2* records = reinterpret_cast<uint2*>(a.scratch + int64_t(token) * S * E + expert_base);
            records[lane] = make_uint2(__float_as_uint(value), uint32_t(selected));
        }
    }
}

template<int E, int S, int N, int Topk, int Threads, int M, bool HasPhysicalMap, bool HasUnmapped>
__device__ __forceinline__ void merge_group_candidates(const mega_gate_layout::KernelArgs& a, int token_base) {
    using namespace epilogue::mega_gate;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int Count = (E / N) * Topk, Values = (Count + 31) / 32;
    for (int token = token_base + warp; token < token_base + M and token < a.num_tokens; token += Threads / 32) {
        float scores[Values], ranks[Values];
        int experts[Values];
        #pragma unroll
        for (int v = 0; v < Values; ++v) {
            const int candidate = lane + v * 32;
            uint2 record = make_uint2(0, 0xffffffffu);
            if (candidate < Count) {
                const uint2* records = reinterpret_cast<const uint2*>(
                    a.scratch + int64_t(token) * S * E + (candidate / Topk) * N);
                record = records[candidate % Topk];
            }
            scores[v] = __uint_as_float(record.x);
            experts[v] = int(record.y);
            ranks[v] = experts[v] >= 0 ? scores[v] + a.bias[experts[v]] : -INFINITY;
        }
        int selected = -1;
        float value = 0.0f;
        select_records<Values, Topk>(scores, ranks, experts, selected, value);
        device_check(lane >= Topk or (selected >= 0 and selected < a.routing.num_routed_experts));
        store_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, a.routing.logical_count,
            token, lane, selected, value);
    }
}

} // namespace mega_gate_detail

// Also used by the acBLASLt + fused Top-k benchmark, with S=1.
template<int E, int S, int Topk, bool HasPhysicalMap, bool HasUnmapped, int Threads, int BlockTokens=16,
         bool CompactScores=false, int ScoreStride=S, bool PreActivated=false, int Features=-1, int ScoreBlockTokens=1, bool WarpLocal=false, int RowGroups=1>
__device__ __forceinline__ void ppu_mega_gate_topk_impl(mega_gate_layout::KernelArgs a, int token_base, int row_group=0) {
    using namespace mega_gate_layout;
    using namespace epilogue::mega_gate;
    constexpr int M = BlockTokens, Warps = Threads / 32;
    if constexpr (!(Features & 1)) a.mask = nullptr;
    if constexpr (!(Features & 2)) a.force_random = nullptr;
    if constexpr (!(Features & 4)) a.fix_routing_mask = nullptr;
    if constexpr (!(Features & 8)) a.bias = nullptr;
    if constexpr (!(Features & 16)) a.image_token_mask = nullptr, a.image_bias = nullptr;
    if constexpr (!(Features & 32)) a.routing.num_shared_experts = 0;
    const bool score_for_ranking = Features < 0 ? (a.bias or a.image_token_mask) : bool(Features & 24);
    const int lane = threadIdx.x % 32, warp = WarpLocal ? 0 : threadIdx.x / 32;
    constexpr bool PruneCandidates = !PreActivated and !CompactScores and E >= 256 and ((M >= 32 and M > Warps) or (WarpLocal and Topk * 2 <= E / 32)) and Features >= 0 and (Features & 8) and !(Features & 16);
    constexpr bool CacheBias = Features >= 0 and (Features & 8) and !(Features & 16) and (M > Warps or WarpLocal);
    float bias_cache[E / 32];
    if constexpr (CacheBias) {
        #pragma unroll
        for (int v=0;v<E/32;++v) {
            const int expert=(v/4)*128+lane*4+v%4;
            bias_cache[v]=expert<a.routing.num_routed_experts?a.bias[expert]:-INFINITY;
        }
    }
    int seed_expert = -1;
    float seed_bias = 0.0f;
    bool seed_bias_finite = true;
    if constexpr (PruneCandidates) {
        constexpr int Waves = E / kNumExpertsPerWave, Values = Waves * kNumExpertsPerLaneVector;
        float best_bias = -INFINITY;
        int offered = -1;
        #pragma unroll
        for (int v=0;v<Values;++v) {
            const int expert=(v/4)*128+lane*4+v%4;
            const float bias=expert<a.routing.num_routed_experts?bias_cache[v]:-INFINITY;
            seed_bias_finite &= expert>=a.routing.num_routed_experts or isfinite(bias);
            if (bias>best_bias) best_bias=bias,offered=expert;
        }
        // Keep the single-row cooperative path unchanged. Reused rows can
        // amortize the ordered seed keys across multiple token iterations.
        if constexpr (WarpLocal) {
            #pragma unroll
            for (int output=0;output<Topk;++output) {
                int winner=offered;
                const float best=warp_max(best_bias);
                winner=int(warp_min(best_bias==best and offered>=0?uint32_t(offered):0xffffffffu));
                if (lane==output) seed_expert=winner,seed_bias=best;
                if (offered==winner) offered=-1,best_bias=-INFINITY;
            }
        } else {
            uint32_t bias_key = mega_gate_detail::ordered_score_key(best_bias);
            #pragma unroll
            for (int output=0;output<Topk;++output) {
                const uint32_t best = mega_gate_detail::warp_max_key(bias_key);
                const int winner = int(warp_min(bias_key==best and offered>=0?uint32_t(offered):0xffffffffu));
                if (lane==output) {
                    seed_expert=winner;
                    seed_bias=__uint_as_float(best & 0x80000000u ? best ^ 0x80000000u : ~best);
                }
                if (offered==winner) offered=-1,bias_key=0x007fffffu;
            }
        }
    }
    bool worth_pruning = true;
    if constexpr (PruneCandidates) {
        const float bias_floor = -warp_max(lane < Topk ? -seed_bias : -INFINITY);
        int eligible = 0;
        #pragma unroll
        for (int v=0;v<E/32;++v) {
            const int expert=(v/4)*128+lane*4+v%4;
            eligible += expert<a.routing.num_routed_experts and bias_cache[v]>=bias_floor;
        }
        // ponytail: diffuse bias uses the full selector; a larger compact
        // list is only worthwhile if this fallback becomes a bottleneck.
        worth_pruning = warp_sum<32>(eligible) <= 64;
    }
    // Bias-only multi-row warps reuse the same seeds for every token.
    // Other routing controls must keep validation after masked/fixed/random rows.
    if constexpr (PruneCandidates and !WarpLocal and Features == 8) {
        if (worth_pruning) {
            device_check(seed_bias_finite);
            device_check(lane >= Topk or (seed_expert >= 0 and seed_expert < a.routing.num_routed_experts));
        }
    }
    // A cooperative expert group distributes rows across all producer CTAs.
    // The ordinary last-arrival path retains RowGroups=1.
    for (int token = token_base + warp * RowGroups + row_group;
         token < token_base + M and token < a.num_tokens; token += Warps * RowGroups) {
        if (a.mask and !a.mask[token]) {
            store_token_outputs<Topk, HasUnmapped>(a.routing, token, lane, -1, 0.0f, -1);
            continue;
        }
        if (a.force_random and a.force_random[token]) {
            store_random_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, token, lane);
            continue;
        }
        if (a.fix_routing_mask and a.fix_routing_mask[token]) {
            int selected = -1;
            float value = 0.0f;
            if (lane < Topk) {
                const int64_t fixed = a.routing.unmapped_topk_idx[int64_t(token) * a.routing.unmapped_topk_idx_stride + lane];
                device_check(fixed >= 0 and fixed < a.routing.num_routed_experts);
                selected = int(fixed);
                value = mega_gate_detail::reduced_score<E,S,CompactScores,ScoreStride,ScoreBlockTokens>(a, token, selected);
                if (!PreActivated or !score_for_ranking) value = sqrt_softplus(value);
                device_check(isfinite(value));
            }
            store_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, a.routing.logical_count, token, lane, selected, value);
            continue;
        }
        const float* bias = a.image_token_mask and a.image_token_mask[token] ? a.image_bias : a.bias;
        // Upstream activates all rows when either bias path exists, including
        // non-image rows of an image-only invocation.
        constexpr int Waves = E / kNumExpertsPerWave, Values = Waves * kNumExpertsPerLaneVector;
        float seed_logit = 0.0f;
        if constexpr (PruneCandidates and !WarpLocal) {
            if (worth_pruning) {
                if constexpr (Features != 8) {
                    device_check(seed_bias_finite);
                    device_check(lane >= Topk or (seed_expert >= 0 and seed_expert < a.routing.num_routed_experts));
                }
                if (lane < Topk)
                    seed_logit = mega_gate_detail::reduced_score<E,S,CompactScores,ScoreStride,ScoreBlockTokens>(a, token, seed_expert);
            }
        }
        float values[Values], ranking[Values];
        bool finite = true;
        #pragma unroll
        for (int v = 0; v < Values; ++v) {
            const int expert = (v / 4) * 128 + lane * 4 + v % 4;
            values[v] = CompactScores and expert >= a.routing.num_routed_experts ? 0.0f :
                mega_gate_detail::reduced_score<E,S,CompactScores,ScoreStride,ScoreBlockTokens>(a, token, expert);
            finite &= isfinite(values[v]);
        }
        device_check(finite);
        if constexpr (PruneCandidates) {
            if (worth_pruning) {
                if constexpr (WarpLocal) {
                    device_check(seed_bias_finite);
                    device_check(lane >= Topk or (seed_expert >= 0 and seed_expert < a.routing.num_routed_experts));
                    if (lane < Topk)
                        seed_logit = mega_gate_detail::reduced_score<E,S,CompactScores,ScoreStride,ScoreBlockTokens>(a, token, seed_expert);
                }
                float seed_rank = INFINITY;
                if (lane < Topk) {
                    seed_rank = sqrt_softplus(seed_logit) + seed_bias;
                    // The multi-row path already checked raw logits and bias;
                    // its activation is finite and below 2^64 (see rank below).
                    if constexpr (WarpLocal) device_check(isfinite(seed_rank));
                }
                const float threshold = -warp_max(-seed_rank);
                uint32_t candidates[Values];
                int count = 0;
                #pragma unroll
                for (int v = 0; v < Values; ++v) {
                    const int expert = (v / 4) * 128 + lane * 4 + v % 4;
                    // softplus(x) <= max(x,0)+3/4, and softplus''(x) <= 1/4.
                    // sqrt(u) <= u/2+1/2 and sqrt(u) <= u/4+1 (u >= 0).
                    // Their linear/quadratic bounds only prune; output scores
                    // still use the original expf/log1pf/sqrtf implementation.
                    const float positive = fmaxf(values[v], 0.0f);
                    const float quadratic = (0.0625f * values[v] + 0.25f) * values[v] + 0.875f;
                    const float upper = fminf(quadratic, fminf(0.5f * positive + 0.875f, 0.25f * positive + 1.1875f)) + bias_cache[v];
                    candidates[v] = __ballot_sync(0xffffffff, expert < a.routing.num_routed_experts and upper >= threshold);
                    count += __popc(candidates[v]);
                }
                if (count <= 64) {
                    device_check(count >= Topk);
                    // All raw row values are now in registers. Reuse the row for
                    // the compact list; no extra shared/global workspace is needed.
                    float* compact = mega_gate_detail::score_ptr<E, false, ScoreStride, ScoreBlockTokens>(a, token, 0);
                    uint2* records = reinterpret_cast<uint2*>(compact);
                    __syncwarp();
                    int prefix = 0;
                    #pragma unroll
                    for (int v = 0; v < Values; ++v) {
                        const bool active = candidates[v] & (1u << lane);
                        const int position = prefix + __popc(candidates[v] & ((1u << lane) - 1));
                        // Inactive lanes use disjoint, unused row slots, keeping
                        // these vector stores branch-free without write races.
                        records[active ? position : E/2-32+lane] = make_uint2(__float_as_uint(values[v]), (v / 4) * 128 + lane * 4 + v % 4);
                        prefix += __popc(candidates[v]);
                    }
                    __syncwarp();
                    int experts[2];
                    float scores[2], ranks[2];
                    #pragma unroll
                    for (int v=0;v<2;++v) {
                        const uint2 item = lane+v*32 < count ? records[lane+v*32] : make_uint2(0, 0xffffffffu);
                        experts[v] = int(item.y);
                        scores[v] = __uint_as_float(item.x);
                    }
                    if constexpr (!WarpLocal) {
                        // One candidate per lane: select indices first and
                        // shuffle each output score only once after selection.
                        if (count <= 32) {
                            // Issue the independent bias load before the activation chain.
                            // Padded lanes read slot zero; their rank stays -INFINITY.
                            const float candidate_bias = a.bias[experts[0] >= 0 ? experts[0] : 0];
                            const float score = sqrt_softplus(scores[0]);
                            const float rank = experts[0] >= 0 ? score + candidate_bias : -INFINITY;
                            // Finite raw logits and bias were checked above.
                            // sqrt(softplus(finite FP32)) <= 2^64, so adding
                            // any finite FP32 bias cannot overflow FP32.
                            uint32_t key = mega_gate_detail::ordered_score_key(rank);
                            const uint32_t location = experts[0] >= 0 ? (uint32_t(experts[0]) << 5) | uint32_t(lane) : 0xffffffffu;
                            int selected = -1, selected_lane = 0;
                            #pragma unroll 1
                            for (int output=0; output<Topk; ++output) {
                                const uint32_t best = mega_gate_detail::warp_max_key(key);
                                const uint32_t winner = warp_min(key == best ? location : 0xffffffffu);
                                const int source = winner & 31;
                                if (lane == output) selected = int(winner >> 5), selected_lane = source;
                                if (lane == source) key = 0x007fffffu;
                            }
                            const float chosen_score = __shfl_sync(0xffffffff, score, selected_lane);
                            const float selected_score = lane < Topk ? chosen_score : 0.0f;
                            store_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, a.routing.logical_count, token, lane, selected, selected_score);
                            continue;
                        }
                    }
                    if (count <= 32) scores[0] = sqrt_softplus(scores[0]);
                    else sqrt_softplus(scores);
                    bool finite = true;
                    #pragma unroll
                    for (int v=0;v<2;++v) {
                        ranks[v] = experts[v] >= 0 ? scores[v]+a.bias[experts[v]] : -INFINITY;
                        finite &= experts[v]<0 or isfinite(ranks[v]);
                    }
                    device_check(finite);
                    if (ranks[1] > ranks[0] or (ranks[1]==ranks[0] and experts[1]<experts[0])) {
                        float tmp=ranks[0];ranks[0]=ranks[1];ranks[1]=tmp;
                        tmp=scores[0];scores[0]=scores[1];scores[1]=tmp;
                        int idx=experts[0];experts[0]=experts[1];experts[1]=idx;
                    }
                    uint32_t rank_keys[2]={mega_gate_detail::ordered_score_key(ranks[0]),mega_gate_detail::ordered_score_key(ranks[1])};
                    int selected=-1;
                    float selected_score=0.0f;
                    #pragma unroll 1
                    for (int output=0;output<Topk;++output) {
                        const uint32_t best=mega_gate_detail::warp_max_key(rank_keys[0]);
                        const uint32_t local=rank_keys[0]==best and experts[0]>=0?(uint32_t(experts[0])<<5)|uint32_t(lane):0xffffffffu;
                        const uint32_t winner=warp_min(local);
                        const int source=winner&31;
                        const float winner_score=__shfl_sync(0xffffffff,scores[0],source);
                        if (lane==output) selected=int(winner>>5),selected_score=winner_score;
                        if (lane==source) {
                            rank_keys[0]=rank_keys[1];experts[0]=experts[1];scores[0]=scores[1];
                            rank_keys[1]=0x007fffffu;experts[1]=-1;scores[1]=0.0f;
                        }
                    }
                    store_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, a.routing.logical_count, token, lane, selected, selected_score);
                    continue;
                }
            }
        }
        if (score_for_ranking and !PreActivated) sqrt_softplus(values);
        #pragma unroll
        for (int v = 0; v < Values; ++v) {
            const int expert = (v / 4) * 128 + lane * 4 + v % 4;
            ranking[v] = expert < a.routing.num_routed_experts ? values[v] + (CacheBias ? bias_cache[v] : (bias ? bias[expert] : 0.0f)) : -INFINITY;
            finite &= expert >= a.routing.num_routed_experts or isfinite(ranking[v]);
        }
        device_check(finite);
        int selected = -1;
        if constexpr ((PreActivated and ScoreStride > 1 and M >= 32) or WarpLocal)
            mega_gate_detail::select_merge_topk<Waves, Topk>(ranking, selected);
        else
            select_warp_topk<Waves, Topk>(ranking, selected);
        device_check(lane >= Topk or (selected >= 0 and selected < a.routing.num_routed_experts));
        float value = 0.0f;
        if constexpr (PreActivated and ScoreStride > 1 and M >= 32) {
            if (lane < Topk)
                value = mega_gate_detail::reduced_score<E, 1, CompactScores, ScoreStride, ScoreBlockTokens>(a, token, selected);
        } else {
            value = select_warp_expert_value<Waves>(values, selected);
        }
        if (!score_for_ranking and lane < Topk) value = sqrt_softplus(value);
        store_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, a.routing.logical_count, token, lane, selected, value);
    }
}

template<int H, int E, int S, int Topk, bool HasPhysicalMap, bool HasUnmapped,
         int N=128, int K=64, int Stages=2, int WarpK=1, bool UseAiu=false, int M=16,
         bool ProducerActivation=false, int Features=-1, int WarpNOverride=0, int WarpMOverride=0, bool Cooperative=false>
__device__ void ppu_bf16_mega_gate_impl(mega_gate_layout::KernelArgs a) {
    using namespace mega_gate_layout;
    using namespace epilogue::mega_gate;
    // Specialize optional routing controls so unused RNG/control paths cannot
    // inflate the GEMM kernel's register footprint.
    if constexpr (!(Features & 1)) a.mask = nullptr;
    if constexpr (!(Features & 2)) a.force_random = nullptr;
    if constexpr (!(Features & 4)) a.fix_routing_mask = nullptr;
    if constexpr (!(Features & 8)) a.bias = nullptr;
    if constexpr (!(Features & 16)) a.image_token_mask = nullptr, a.image_bias = nullptr;
    if constexpr (!(Features & 32)) a.routing.num_shared_experts = 0;
    constexpr int WarpM = WarpMOverride ? WarpMOverride : M >= 128 ? 64 : M >= 32 ? 32 : 16;
    constexpr int WarpN = WarpNOverride ? WarpNOverride : N >= 256 ? 64 : M >= 32 ? 32 : 16;
    constexpr int WarpsM = M / WarpM, WarpsN = N / WarpN, WarpsMN = WarpsM * WarpsN;
    constexpr int Threads = WarpsMN * WarpK * 32;
    constexpr int Warps = Threads / 32, Groups = E / N, Producers = Groups * S;
    constexpr bool PersistentGroups = Cooperative and S == 1 and Groups > 1;
    static_assert(H % (S * K) == 0 and E % N == 0 and E <= 512);
    constexpr bool PruneAfterReduction = S > 1 and Groups == 1 and E >= 256 and M >= 32 and M > Warps and
        Features >= 0 and (Features & 8) and !(Features & 16);
    // Wide-K routers benefit from contiguous rows within each K plane; shorter
    // reductions retain the cheaper legacy indexing at the measured crossover.
    constexpr bool GroupedScores = PruneAfterReduction and H >= 16 * E;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    // Short, K-split grids traverse M before N and K, as in BF16 GEMM.
    // Adjacent CTAs reuse the same weight tile across token tiles; producer
    // identity and per-token ScoreBarrier ownership remain unchanged.
    constexpr bool MFirst = UseAiu and S > 1 and M >= 32;
    const int token_tiles = PersistentGroups ? (int(a.num_tokens) + M - 1) / M : gridDim.x / Producers;
    const int first_token_tile = MFirst ? blockIdx.x % token_tiles : blockIdx.x / Producers;
    const int producer = MFirst ? blockIdx.x / token_tiles : blockIdx.x % Producers;
    const int split = producer / Groups;
    const int expert_base = producer % Groups * N;
    a.scratch_token_base = 0;
    const auto* x = static_cast<const __ppu_bfloat16*>(a.x);
    const auto* weight = static_cast<const __ppu_bfloat16*>(a.weight);
    __shared__ int fallback_last;
    extern __shared__ __align__(128) char smem[];
    // GEMM has drained the pipeline before ScoreBarrier uses this flag.
    // Reuse its storage so a full-capacity pipeline needs no extra static SMEM.
    int& last = UseAiu ? *reinterpret_cast<int*>(smem) : fallback_last;
    if constexpr (PersistentGroups) {
        static_assert(UseAiu);
        if (producer == 0 and tid == 0) a.score_barriers[first_token_tile] = 0;
        // Initialization is part of this kernel, including Graph replays.
        cooperative_groups::this_grid().sync();
    }
    const int tile_stride = PersistentGroups ? gridDim.x / Producers : token_tiles;
    const int rounds = PersistentGroups ? (token_tiles - first_token_tile + tile_stride - 1) / tile_stride : 1;
    for (int round = 0; round < rounds; ++round) {
    const int token_tile = first_token_tile + round * tile_stride;
    const int token_base = token_tile * M;
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    if constexpr (UseAiu) {
        // Reuse the library's AIU + swizzled TSM + register-buffered MMA pipeline.
        using Element = cutlass::bfloat16_t;
        using Mainloop = mainloop::Mainloop<Element,
            copy::AiuG2SAtom<Element, M, K>, copy::AiuG2SAtom<Element, N, K>,
            copy::TsmS2RAtom<Element, M, K, false>, copy::TsmS2RAtom<Element, N, K, true>,
            mma::Bf16MmaAtom, M, N, K, WarpM, WarpN, K / WarpK, Stages, true, false>;
        typename Mainloop::Params params{
            reinterpret_cast<const Element*>(x) + split * (H / S),
            reinterpret_cast<const Element*>(weight) + split * (H / S),
            int(a.num_tokens), int(a.routing.num_routed_experts), H / S, H, H};
        float acc[WarpM / 16][WarpN / 16][8] = {};
        const int warp_mn = warp % WarpsMN, warp_k = warp / WarpsMN;
        const int warp_m = warp_mn / WarpsN, warp_n = warp_mn % WarpsN;
        Mainloop::run(params, smem, warp, warp_m, warp_n, warp_k, token_base, expert_base, acc);
        if constexpr (Cooperative and WarpK > 1 and
                      ((WarpM / 16) * (WarpN / 16) * 8) % (2 * WarpK) == 0 and
                      WarpK * M * N <= Stages * (M + N) * K / 2) {
            // Distribute the final warp-K reduction and partial stores over all
            // warp groups, preserving the original ascending-K addition order.
            constexpr int Values = (WarpM / 16) * (WarpN / 16) * 8;
            constexpr int Plane = WarpsMN * 32 * Values;
            static_assert(WarpK * Plane <= Stages * (M + N) * K / 2);
            static_assert(Values % (2 * WarpK) == 0);
            float* partial = reinterpret_cast<float*>(smem);
            #pragma unroll
            for (int mi = 0; mi < WarpM / 16; ++mi)
                #pragma unroll
                for (int ni = 0; ni < WarpN / 16; ++ni)
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int e = (mi * (WarpN / 16) + ni) * 8 + r;
                        partial[warp_k * Plane + warp_mn * 32 * Values + e * 32 + lane] = acc[mi][ni][r];
                    }
            __syncthreads();
            #pragma unroll
            for (int v = 0; v < Values / (2 * WarpK); ++v) {
                const int e = (v * WarpK + warp_k) * 2;
                const int base = warp_mn * 32 * Values + e * 32 + lane;
                float lo = partial[base], hi = partial[base + 32];
                #pragma unroll
                for (int k = 1; k < WarpK; ++k) {
                    lo += partial[k * Plane + base];
                    hi += partial[k * Plane + base + 32];
                }
                const int mi = e / (8 * (WarpN / 16)), ni = (e / 8) % (WarpN / 16), r = e % 8;
                const int token = token_base + warp_m * WarpM + mi * 16 + lane / 4 + ((r / 2) % 2) * 8;
                const int expert = expert_base + warp_n * WarpN + ni * 16 + (lane % 4) * 2 + (r / 4) * 8;
                const int64_t score_row = GroupedScores ?
                    (int64_t(token / M) * S + split) * M + token % M : int64_t(token) * S + split;
                *reinterpret_cast<float2*>(a.scratch + score_row * E + expert) = make_float2(lo, hi);
            }
        } else {
        epilogue::WarpKReduce<WarpK, WarpsMN, WarpM / 16, WarpN / 16, 8, Stages * (M + N) * K / 2>::run(
            acc, reinterpret_cast<float*>(smem), warp_mn, warp_k, lane);
        if constexpr (Producers == 1) {
            a.scratch = reinterpret_cast<float*>(smem);
            a.scratch_token_base = token_base;
        }
        if (warp_k == 0) {
            #pragma unroll
            for (int mi = 0; mi < WarpM / 16; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < WarpN / 16; ++ni) {
                    #pragma unroll
                    for (int r = 0; r < 8; r += 2) {
                        const int token = token_base + warp_m * WarpM + mi * 16 + lane / 4 + ((r / 2) % 2) * 8;
                        const int expert = expert_base + warp_n * WarpN + ni * 16 + (lane % 4) * 2 + (r / 4) * 8;
                        const int row = token - a.scratch_token_base;
                        const int64_t score_row = GroupedScores ?
                            (int64_t(row / M) * S + split) * M + row % M : int64_t(row) * S + split;
                        *reinterpret_cast<float2*>(a.scratch + score_row * E + expert) =
                            make_float2(acc[mi][ni][r], acc[mi][ni][r + 1]);
                    }
                }
            }
        }
        }
    } else
#endif
    {
        static_assert(WarpK == 1 and M == 16, "Native tiling requires PPU 1.5");
        __shared__ __align__(32) __ppu_bfloat16 xs[M * K], ws[N * K];
        __shared__ __align__(32) float partial[M * N];

        awmma::fragment<awmma::matrix_a,16,16,16,__ppu_bfloat16,awmma::row_major> af;
        awmma::fragment<awmma::matrix_b,16,16,16,__ppu_bfloat16,awmma::col_major> bf;
        awmma::fragment<awmma::accumulator,16,16,16,float> acc;
        awmma::fill_fragment(acc, 0.0f);
        // Preserve the generic PPU path, including contiguous tensors with unaligned storage offsets.
        for (int k = split * (H / S); k < (split + 1) * (H / S); k += K) {
            for (int i = tid; i < M * K; i += Threads) {
                const int token = token_base + i / K;
                xs[i] = token < a.num_tokens ? x[int64_t(token) * H + k + i % K] : __float2bfloat16(0.0f);
            }
            for (int i = tid; i < N * K; i += Threads) {
                const int expert = expert_base + i / K;
                ws[i] = expert < a.routing.num_routed_experts ? weight[int64_t(expert) * H + k + i % K] : __float2bfloat16(0.0f);
            }
            __syncthreads();
            #pragma unroll
            for (int kk = 0; kk < K; kk += 16) {
                awmma::load_matrix_sync(af, xs + kk, K);
                awmma::load_matrix_sync(bf, ws + warp * 16 * K + kk, K);
                awmma::mma_sync(acc, af, bf, acc);
            }
            __syncthreads();
        }
        awmma::store_matrix_sync(partial + warp * 16, acc, N, awmma::mem_row_major);
        __syncthreads();
        if constexpr (Producers == 1) {
            a.scratch = partial;
            a.scratch_token_base = token_base;
        } else {
            for (int i = tid; i < M * N; i += Threads) {
                const int token = token_base + i / N;
                a.scratch[int64_t(token) * S * E + split * E + expert_base + i % N] = partial[i];
            }
        }
    }
    __syncthreads();
    if constexpr (PersistentGroups) {
        if (tid == 0)
            mega_gate_detail::ScoreBarrier::arrive_and_wait(a.score_barriers + first_token_tile, (round + 1) * Producers);
        __syncthreads();
        ppu_mega_gate_topk_impl<E, S, Topk, HasPhysicalMap, HasUnmapped, Threads, M,
                                false, S, false, Features, 1, false, Producers>(a, token_base, producer);
        __syncthreads();
        continue;
    }
    if constexpr (Cooperative and !PersistentGroups) {
        static_assert(UseAiu and S > 1);
        // The cooperative launch guarantees that the entire producer grid can
        // reach this device-wide barrier. Every CTA then routes its own rows.
        cooperative_groups::this_grid().sync();
        constexpr int RowsPerProducer = (M + Producers - 1) / Producers;
        for (int row = producer * RowsPerProducer + warp;
             row < (producer + 1) * RowsPerProducer and row < M; row += Warps) {
            const int token = token_base + row;
            if (token < a.num_tokens)
                ppu_mega_gate_topk_impl<E, S, Topk, HasPhysicalMap, HasUnmapped, 32, 1,
                                        false, S, false, Features, GroupedScores ? M : 1, true>(a, token);
        }
        return;
    }
    if constexpr (Producers == 1) {
        ppu_mega_gate_topk_impl<E, S, Topk, HasPhysicalMap, HasUnmapped, Threads, M, false, S, false, Features>(a, token_base);
        return;
    }
    // Raw-score pruning avoids activating every expert after K reduction.
    if constexpr (ProducerActivation and !PruneAfterReduction) {
        int* counters = a.score_barriers + token_base / M * (Groups + 1);
        if constexpr (S > 1) {
            if (tid == 0)
                last = mega_gate_detail::ScoreBarrier::arrive_last(counters + 1 + expert_base / N, S);
            __syncthreads();
            if (!last) return;
        }
        // Each warp handles one row, and the final candidate set fits one
        // entry per lane. Larger candidate sets use the full-score selector.
        // Bias-only routing has no masked/fixed-row scratch to preserve.
        if constexpr (S > 1 and Groups > 1 and M >= 32 and M <= Warps and N % 32 == 0 and
                      2 * Topk <= N and Groups * Topk <= 32 and Features == 8) {
            mega_gate_detail::store_group_candidates<E, S, N, Topk, Threads, M>(a, token_base, expert_base);
            __syncthreads();
            if (tid == 0) last = mega_gate_detail::ScoreBarrier::arrive_last(counters, Groups);
            __syncthreads();
            if (!last) return;
            mega_gate_detail::merge_group_candidates<E, S, N, Topk, Threads, M, HasPhysicalMap, HasUnmapped>(a, token_base);
            return;
        }
        // Each expert tile reduces its K partials and activates its own scores.
        // The final token-tile consumer only merges Top-k across expert tiles.
        constexpr int ValuesPerThread = M * N / Threads;
        constexpr int Batch = ValuesPerThread < 8 ? ValuesPerThread : 8;
        static_assert(ValuesPerThread > 0 and M * N % Threads == 0);
        const bool activate = Features < 0 ? (a.bias or a.image_token_mask) : bool(Features & 24);
        for (int base = 0; base < ValuesPerThread; base += Batch) {
            float values[Batch];
            bool finite = true;
            #pragma unroll
            for (int v = 0; v < Batch; ++v) {
                const int linear = tid + (base + v) * Threads;
                const int token = token_base + linear / N, expert = expert_base + linear % N;
                values[v] = mega_gate_detail::reduced_score<E,S>(a, token, expert);
                // Masked/random/fixed rows must retain the original validation behavior.
                const bool checked = token < a.num_tokens and (!a.mask or a.mask[token]) and
                    (!a.force_random or !a.force_random[token]) and (!a.fix_routing_mask or !a.fix_routing_mask[token]);
                finite &= !checked or isfinite(values[v]);
            }
            device_check(finite);
            if (activate) sqrt_softplus(values);
            #pragma unroll
            for (int v = 0; v < Batch; ++v) {
                const int linear = tid + (base + v) * Threads;
                const int token = token_base + linear / N, expert = expert_base + linear % N;
                a.scratch[int64_t(token) * S * E + expert] = values[v];
            }
        }
        __syncthreads();
        if (tid == 0) last = mega_gate_detail::ScoreBarrier::arrive_last(counters, Groups);
        __syncthreads();
        if (!last) return;
        ppu_mega_gate_topk_impl<E, 1, Topk, HasPhysicalMap, HasUnmapped, Threads, M, false, S, true, Features>(a, token_base);
        return;
    }
    if (tid == 0)
        last = mega_gate_detail::ScoreBarrier::arrive_last(a.score_barriers + token_base / M, Producers);
    __syncthreads(); // Broadcast both the last-arrival decision and acquired scratch visibility.
    if (!last) return;

    ppu_mega_gate_topk_impl<E, S, Topk, HasPhysicalMap, HasUnmapped, Threads, M, false, S, false,
                            Features, GroupedScores ? M : 1>(a, token_base);
    }
}
} // namespace deep_gemm
