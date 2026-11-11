#pragma once
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <cooperative_groups.h>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/tix/utils.cuh>
#include <deep_gemm/epilogue/ppu_mega_gate.cuh>

#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/epilogue/ppu_mega_gate.cuh>
#include <deep_gemm/tix/utils.cuh>

namespace deep_gemm::mega_gate_shared {
__device__ __forceinline__ uint32_t ordered_score_key(float value) {
    const uint32_t bits = __float_as_uint(value == 0.0f ? 0.0f : value);
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}
__device__ __forceinline__ uint32_t warp_max_key(uint32_t key) {
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ == 100 || __HGGC_ARCH__ >= 150)
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
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
        for (int v = 0; v < 4; ++v) keys[w*4+v] = ordered_score_key(scores[w*4+v]);
#else
        for (int v = 0; v < 4; ++v)
            keys[w*4+v] = scores[w*4+v] == scores[w*4+v] ? ordered_score_key(scores[w*4+v]) : 0x007fffffu;
#endif
    }
    // Bound instruction growth in the per-warp merge loop.
    #if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    #pragma unroll 4
    #else
    #pragma unroll
    #endif
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
        const uint32_t winner = __reduce_min_sync(0xffffffff,
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
            best == winner_key ? (uint32_t(offered) << 5) | uint32_t(lane) : 0xffffffffu);
#else
            best == winner_key and best > 0x007fffffu ? (uint32_t(offered) << 5) | uint32_t(lane) : 0xffffffffu);
#endif
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


template<int E, int S, int Topk, bool HasPhysicalMap, bool HasUnmapped, int Threads, int BlockTokens=16,
         bool CompactScores=false, int ScoreStride=S, bool PreActivated=false, int Features=-1, int ScoreBlockTokens=1, bool WarpLocal=false, int RowGroups=1>
__device__ __forceinline__ void ppu_mega_gate_topk_impl(mega_gate_layout::KernelArgs a, int token_base, bool& gate_is_valid, int row_group=0) {
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
                const float best=tix::reduce_max_sync(best_bias);
                winner=int(__reduce_min_sync(0xffffffff, best_bias==best and offered>=0?uint32_t(offered):0xffffffffu));
                if (lane==output) seed_expert=winner,seed_bias=best;
                if (offered==winner) offered=-1,best_bias=-INFINITY;
            }
        } else {
            uint32_t bias_key = mega_gate_shared::ordered_score_key(best_bias);
            #pragma unroll
            for (int output=0;output<Topk;++output) {
                const uint32_t best = mega_gate_shared::warp_max_key(bias_key);
                const int winner = int(__reduce_min_sync(0xffffffff, bias_key==best and offered>=0?uint32_t(offered):0xffffffffu));
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
        const float bias_floor = -tix::reduce_max_sync(lane < Topk ? -seed_bias : -INFINITY);
        int eligible = 0;
        #pragma unroll
        for (int v=0;v<E/32;++v) {
            const int expert=(v/4)*128+lane*4+v%4;
            eligible += expert<a.routing.num_routed_experts and bias_cache[v]>=bias_floor;
        }
        // ponytail: diffuse bias uses the full selector; a larger compact
        // list is only worthwhile if this fallback becomes a bottleneck.
        // Non-finite bias is legal upstream; use the full selector for those rows.
        worth_pruning = math::warp_reduce_sum<32>(seed_bias_finite ? eligible : E) <= 64;
    }
    // A cooperative expert group distributes rows across all producer CTAs.
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
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
                const bool valid = fixed >= 0 and fixed < a.routing.num_routed_experts;
                gate_is_valid &= valid;
                selected = valid ? int(fixed) : 0;
#else
                device_check(fixed >= 0 and fixed < a.routing.num_routed_experts);
                selected = int(fixed);
#endif
                value = mega_gate_shared::reduced_score<E,S,CompactScores,ScoreStride,ScoreBlockTokens>(a, token, selected);
                if (!PreActivated or !score_for_ranking) value = sqrt_softplus(value);
            }
            gate_is_valid &= store_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, a.routing.logical_count, token, lane, selected, value);
            continue;
        }
        const float* bias = a.image_token_mask and a.image_token_mask[token] ? a.image_bias : a.bias;
        // Upstream activates all rows when either bias path exists, including
        // non-image rows of an image-only invocation.
        constexpr int Waves = E / kNumExpertsPerWave, Values = Waves * kNumExpertsPerLaneVector;
        float seed_logit = 0.0f;
        if constexpr (PruneCandidates and !WarpLocal) {
            if (worth_pruning) {
                if (lane < Topk)
                    seed_logit = mega_gate_shared::reduced_score<E,S,CompactScores,ScoreStride,ScoreBlockTokens>(a, token, seed_expert);
            }
        }
        float values[Values], ranking[Values];
        #pragma unroll
        for (int v = 0; v < Values; ++v) {
            const int expert = (v / 4) * 128 + lane * 4 + v % 4;
            values[v] = CompactScores and expert >= a.routing.num_routed_experts ? 0.0f :
                mega_gate_shared::reduced_score<E,S,CompactScores,ScoreStride,ScoreBlockTokens>(a, token, expert);
        }
        if constexpr (PruneCandidates) {
            if (worth_pruning) {
                if constexpr (WarpLocal) {
                    if (lane < Topk)
                        seed_logit = mega_gate_shared::reduced_score<E,S,CompactScores,ScoreStride,ScoreBlockTokens>(a, token, seed_expert);
                }
                float seed_rank = INFINITY;
                if (lane < Topk) {
                    seed_rank = sqrt_softplus(seed_logit) + seed_bias;
                }
                const float threshold = -tix::reduce_max_sync(-seed_rank);
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
                if (count >= Topk and count <= 64) {
                    // All raw row values are now in registers. Reuse the row for
                    // the compact list; no extra shared/global workspace is needed.
                    float* compact = mega_gate_shared::score_ptr<E, false, ScoreStride, ScoreBlockTokens>(a, token, 0);
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
                            uint32_t key = mega_gate_shared::ordered_score_key(rank);
                            const uint32_t location = experts[0] >= 0 ? (uint32_t(experts[0]) << 5) | uint32_t(lane) : 0xffffffffu;
                            int selected = -1, selected_lane = 0;
                            #if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
                            #pragma unroll
                            #else
                            #pragma unroll 1
                            #endif
                            for (int output=0; output<Topk; ++output) {
                                const uint32_t best = mega_gate_shared::warp_max_key(key);
                                const uint32_t winner = __reduce_min_sync(0xffffffff, key == best ? location : 0xffffffffu);
                                const int source = winner & 31;
                                if (lane == output) selected = int(winner >> 5), selected_lane = source;
                                if (lane == source) key = 0x007fffffu;
                            }
                            const float chosen_score = __shfl_sync(0xffffffff, score, selected_lane);
                            const float selected_score = lane < Topk ? chosen_score : 0.0f;
                            gate_is_valid &= store_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, a.routing.logical_count, token, lane, selected, selected_score);
                            continue;
                        }
                    }
                    if (count <= 32) scores[0] = sqrt_softplus(scores[0]);
                    else sqrt_softplus(scores);
                    #pragma unroll
                    for (int v=0;v<2;++v) {
                        ranks[v] = experts[v] >= 0 ? scores[v]+a.bias[experts[v]] : -INFINITY;
                    }
                    if (ranks[1] > ranks[0] or (ranks[1]==ranks[0] and experts[1]<experts[0])) {
                        float tmp=ranks[0];ranks[0]=ranks[1];ranks[1]=tmp;
                        tmp=scores[0];scores[0]=scores[1];scores[1]=tmp;
                        int idx=experts[0];experts[0]=experts[1];experts[1]=idx;
                    }
                    uint32_t rank_keys[2]={mega_gate_shared::ordered_score_key(ranks[0]),mega_gate_shared::ordered_score_key(ranks[1])};
                    int selected=-1;
                    float selected_score=0.0f;
                    #if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
                    #pragma unroll
                    #else
                    #pragma unroll 1
                    #endif
                    for (int output=0;output<Topk;++output) {
                        const uint32_t best=mega_gate_shared::warp_max_key(rank_keys[0]);
                        const uint32_t local=rank_keys[0]==best and experts[0]>=0?(uint32_t(experts[0])<<5)|uint32_t(lane):0xffffffffu;
                        const uint32_t winner=__reduce_min_sync(0xffffffff, local);
                        const int source=winner&31;
                        const float winner_score=__shfl_sync(0xffffffff,scores[0],source);
                        if (lane==output) selected=int(winner>>5),selected_score=winner_score;
                        if (lane==source) {
                            rank_keys[0]=rank_keys[1];experts[0]=experts[1];scores[0]=scores[1];
                            rank_keys[1]=0x007fffffu;experts[1]=-1;scores[1]=0.0f;
                        }
                    }
                    gate_is_valid &= store_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, a.routing.logical_count, token, lane, selected, selected_score);
                    continue;
                }
            }
        }
        if (score_for_ranking and !PreActivated) sqrt_softplus(values);
        #pragma unroll
        for (int v = 0; v < Values; ++v) {
            const int expert = (v / 4) * 128 + lane * 4 + v % 4;
            ranking[v] = expert < a.routing.num_routed_experts ? values[v] + (CacheBias ? bias_cache[v] : (bias ? bias[expert] : 0.0f)) : -INFINITY;
        }
        int selected = -1;
        if constexpr ((PreActivated and ScoreStride > 1 and M >= 32) or WarpLocal)
            mega_gate_shared::select_merge_topk<Waves, Topk>(ranking, selected);
        else
            select_warp_topk<Waves, Topk>(ranking, selected);
        float value = 0.0f;
        if constexpr (PreActivated and ScoreStride > 1 and M >= 32) {
            if (lane < Topk)
                value = mega_gate_shared::reduced_score<E, 1, CompactScores, ScoreStride, ScoreBlockTokens>(a, token, selected);
        } else {
            value = select_warp_expert_value<Waves>(values, selected);
        }
        if (!score_for_ranking and lane < Topk) value = sqrt_softplus(value);
        gate_is_valid &= store_topk_token<Topk, HasUnmapped, HasPhysicalMap>(a.routing, a.routing.logical_count, token, lane, selected, value);
    }
}

} // namespace deep_gemm::mega_gate_shared


// Use the same PPU001 CollectiveMma and operand layouts as BF16GemmCutlass3Runtime.
#ifndef BF16_HGRTC
#define BF16_HGRTC
#endif
#include <deep_gemm/impls/bf16_gemm_cutlass3.cuh>
#include <deep_gemm/epilogue/warp_on_k_reduction.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/epilogue/ppu_mega_gate.cuh>

namespace deep_gemm::mega_gate_810e::mega_gate_detail {

template<int H, int E, int S, int M, int N, int K, int WarpM, int WarpN, int WarpK, int Stages, bool GroupedScores, bool Cooperative>
__device__ __forceinline__ void cutlass3_mainloop(
        mega_gate_layout::KernelArgs& args, char* smem, int token_base, int expert_base, int split, bool& gate_is_valid,
        bool check_activated) {
    using namespace cute;
    using Element = cutlass::bfloat16_t;
    using Arch = cutlass::arch::PPU0010;
    using TileShape = Shape<Int<M>, Int<N>, Int<K>>;
    using OperandStride = cute::Stride<int64_t, Int<1>, int64_t>;
    constexpr int WarpsM = M / WarpM, WarpsN = N / WarpN;
    using Instruction = typename cutlass::gemm::config::GetAiuMmaInst<Arch, Element, Element, float>::type;
    using Mma = TiledMMA<MMA_Atom<Instruction>, Layout<Shape<Int<WarpsM>, Int<WarpsN>, Int<WarpK>>>,
                        Tile<Int<WarpsM * 16>, Int<WarpsN * 16>, Int<WarpK * 16>>>;
    using A = cutlass::gemm::config::DefaultGemm_AIU_Operand<Arch, Element, false, Int<M>, Int<K>, false, M * K>;
    using B = cutlass::gemm::config::DefaultGemm_AIU_Operand<Arch, Element, false, Int<N>, Int<K>, true, N * K>;
    using Mainloop = cutlass::gemm::collective::CollectiveMma<
        Arch, cutlass::gemm::MainloopPPUAiuOpt<Stages, cutlass::gemm::KernelAiuMultistage>, TileShape,
        Element, OperandStride, Element, OperandStride, Mma,
        typename A::GmemTiledCopy, typename A::SmemLayoutAtom, typename A::SmemCopyAtom, cute::identity,
        typename B::GmemTiledCopy, typename B::SmemLayoutAtom, typename B::SmemCopyAtom, cute::identity>;
    static_assert(sizeof(typename Mainloop::SharedStorage) <= Stages * (M + N) * K * sizeof(Element));
    const auto* x = reinterpret_cast<const Element*>(args.x) + split * (H / S);
    const auto* weight = reinterpret_cast<const Element*>(args.weight) + split * (H / S);
    typename Mainloop::Params params{x, {H, Int<1>{}, 0}, weight, {H, Int<1>{}, 0}};
    const auto problem_shape = make_shape(int(args.num_tokens), int(args.routing.num_routed_experts), H / S, 1);
    const auto block = make_coord(token_base / M, expert_base / N, _, 0);
    Mainloop mainloop(params, take<0, 3>(problem_shape));
    const auto loads = mainloop.load_init(problem_shape, block, params, 0, 0, x, weight);
    const auto gA = get<0>(loads);
    const auto residue = make_tuple(int(args.num_tokens) - token_base,
                                   int(args.routing.num_routed_experts) - expert_base, 0);
    Mma mma;
    auto accum = make_fragment_like<float>(partition_fragment_C(mma, make_shape(Int<M>{}, Int<N>{})));
    clear(accum);
    auto k_iter = make_coord_iterator(shape<2>(gA));
    mainloop(accum, loads, accum, k_iter, size<2>(gA), residue,
             int(threadIdx.x), int(threadIdx.x) / 32, smem);

    constexpr int Values = WarpM * WarpN / 32;
    constexpr int Plane = M * N, ThreadsMN = WarpsM * WarpsN * 32;
    constexpr bool Distributed = Cooperative && S > 1 && WarpK > 2 && E / N * S <= 2 * M && Values % WarpK == 0 &&
        WarpK * Plane * sizeof(float) <= Stages * (M + N) * K * sizeof(Element);
    const int warp_k = int(threadIdx.x) / ThreadsMN;
    float* partial = reinterpret_cast<float*>(smem);
    // Match the 890's cooperative reduction: all K groups share the stores.
    // The mainloop has drained, and only global scores are written in this path.
    if constexpr (Distributed) {
        static_assert(S > 1);
        CUTE_UNROLL
        for (int i = 0; i < Values; ++i)
            partial[warp_k * Plane + i * ThreadsMN + int(threadIdx.x) % ThreadsMN] = accum(i);
        __syncthreads();
    } else if constexpr (WarpK > 1) {
        using Reduction = cutlass::gemm::kernel::WarpOnKReductionPolicy<
            M, N, K, WarpM, WarpN, K / WarpK, Stages, sizeof(Element)>;
        static_assert(Reduction::UseStrategyA, "Warp-K partials must fit the reusable mainloop buffer");
        static_assert(size(accum) == Reduction::AccumPerThread);
        cutlass::gemm::kernel::warp_on_k_reduce<Reduction>(
            accum, reinterpret_cast<float*>(smem), int(threadIdx.x) / 32, int(threadIdx.x) % 32);
    }

    // Keep FP32 accumulators: the BF16 Dense GEMM output conversion is not part of routing.
    constexpr bool LocalScores = S == 1 && E == N;
    if constexpr (LocalScores) {
        args.scratch = reinterpret_cast<float*>(smem);
        args.scratch_token_base = token_base;
    }
    if constexpr (WarpK > 1 && !Distributed)
        if (warp_k != 0) return;
    const auto coordinates = mma.get_thread_slice(int(threadIdx.x)).partition_C(
        make_identity_tensor(make_shape(Int<M>{}, Int<N>{})));
    CUTE_UNROLL
    for (int v = 0; v < Values / (Distributed ? WarpK : 1); ++v) {
        const int i = Distributed ? v * WarpK + warp_k : v;
        float value = accum(i);
        if constexpr (Distributed) {
            const int base = i * ThreadsMN + int(threadIdx.x) % ThreadsMN;
            value = partial[base];
            CUTE_UNROLL
            for (int k = 1; k < WarpK; ++k) {
                value += partial[k * Plane + base];
            }
        }
        const auto coord = coordinates(i);
        const int token = token_base + get<0>(coord);
        const int expert = expert_base + get<1>(coord);
        const int row = token - args.scratch_token_base;
        const int64_t score_row = GroupedScores ?
            (int64_t(row / M) * S + split) * M + row % M : int64_t(row) * S + split;
        // PPU0010 accumulator neighbors are strided in N; retain its tensor mapping.
        gate_is_valid &= check_activated ? value < INFINITY : isfinite(value);
        args.scratch[score_row * E + expert] = value;
    }
}

}  // namespace deep_gemm::mega_gate_810e::mega_gate_detail

namespace deep_gemm::mega_gate_810e {
template<int H, int E, int S, int Topk, bool HasPhysicalMap, bool HasUnmapped,
         int N=128, int K=64, int Stages=2, int WarpK=1, int M=16,
         int Features=-1, int WarpNOverride=0, int WarpMOverride=0, bool Cooperative=false,
         bool PersistentTokens=false>
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
    static_assert(Producers == 1 or Cooperative);
    static_assert(!PersistentTokens or (Cooperative and Producers > 1));
    static_assert(H % (S * K) == 0 and E % N == 0 and E <= 512);
    constexpr bool PruneAfterReduction = S > 1 and Groups == 1 and E >= 256 and M >= 32 and M > Warps and
        Features >= 0 and (Features & 8) and !(Features & 16);
    // Wide-K routers benefit from contiguous rows within each K plane; shorter
    // reductions retain the cheaper legacy indexing at the measured crossover.
    constexpr bool GroupedScores = PruneAfterReduction and H >= 16 * E;
    const int tid = threadIdx.x, warp = tid / 32;
    const int token_tiles = PersistentTokens ? (int(a.num_tokens) + M - 1) / M : gridDim.x / Producers;
    const int first_token_tile = blockIdx.x / Producers;
    const int producer = blockIdx.x % Producers;
    const int split = producer / Groups;
    const int expert_base = producer % Groups * N;
    a.scratch_token_base = 0;
    extern __shared__ __align__(128) char smem[];
    const int tile_stride = PersistentTokens ? gridDim.x / Producers : token_tiles;
    // Every CTA executes the same number of grid barriers, including idle
    // groups in the final round. Each round writes disjoint scratch rows.
    const int rounds = PersistentTokens ? (token_tiles + tile_stride - 1) / tile_stride : 1;
    bool gate_is_valid = true;
    const bool check_activated = S == 1 and (Features < 0 ? (a.bias or a.image_token_mask) : bool(Features & 24));
    for (int round = 0; round < rounds; ++round) {
    const int token_tile = first_token_tile + round * tile_stride;
    const int token_base = token_tile * M;
    if (!PersistentTokens or token_tile < token_tiles) {
        mega_gate_detail::cutlass3_mainloop<H, E, S, M, N, K, WarpM, WarpN, WarpK, Stages, GroupedScores, Cooperative>(
            a, smem, token_base, expert_base, split, gate_is_valid, check_activated);
    } // Active token tile; idle groups must still join the grid barrier.
    __syncthreads();
    if constexpr (Cooperative) {
        static_assert(Producers > 1);
        // The cooperative launch guarantees that the entire producer grid can
        // reach this device-wide barrier. Every CTA then routes its own rows.
        cooperative_groups::this_grid().sync();
        constexpr int RowsPerProducer = (M + Producers - 1) / Producers;
        for (int row = producer * RowsPerProducer + warp;
             row < (producer + 1) * RowsPerProducer and row < M; row += Warps) {
            const int token = token_base + row;
            if (token < a.num_tokens)
                mega_gate_shared::ppu_mega_gate_topk_impl<E, S, Topk, HasPhysicalMap, HasUnmapped, 32, 1,
                                        false, S, false, Features, GroupedScores ? M : 1, true>(a, token, gate_is_valid);
        }
        if constexpr (PersistentTokens) {
            // All routing warps finish before this CTA reuses its mainloop SMEM.
            __syncthreads();
            continue;
        }
        break;
    }
    if constexpr (Producers == 1) {
        mega_gate_shared::ppu_mega_gate_topk_impl<E, S, Topk, HasPhysicalMap, HasUnmapped, Threads, M, false, S, false, Features>(a, token_base, gate_is_valid);
        break;
    }
    }
    // Finish every cooperative round before any thread reports invalid scores.
    device_check(gate_is_valid);
}
} // namespace deep_gemm::mega_gate_810e
