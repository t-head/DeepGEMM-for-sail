#pragma once

#include <hggc_bf16.h>
#include <cutlass/numeric_conversion.h>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/comm/int8_dispatch.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/int8_mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/tix/ld_st.cuh>

namespace deep_gemm {

// Instantiated only for channelwise INT8. All call sites share this CTA state;
// FP4 has no extra shared allocation or work.
template <class Storage>
__device__ __forceinline__ uint32_t* token_scale_batch_state() {
    __shared__ uint32_t state[2]; // current batch sender, sticky sender for release
    return state;
}

// The source-rank slice is already local. Complete warps send adjacent vectors
// to each peer, instead of one remote scalar store for every expert route.
template <class Scale, uint32_t NumRanks, uint32_t SourceCapacity>
__device__ __forceinline__ void broadcast_token_scales(
    Scale* staging, uint32_t tokens, uint32_t rank,
    const layout::SymBuffer<>& symmetric, uint32_t token_begin = 0) {
    static_assert(sizeof(Scale) == sizeof(float));
    static_assert(SourceCapacity % 4 == 0);
    const auto* source = staging + uint64_t(rank) * SourceCapacity + token_begin;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = threadIdx.x / 32;
    for (uint32_t peer = warp; peer < NumRanks; peer += blockDim.x / 32) {
        if (peer == rank) continue;
        auto* target = symmetric.map(staging, peer) + uint64_t(rank) * SourceCapacity + token_begin;
        for (uint32_t v = lane; v < tokens / 4; v += 32)
            __stbl(reinterpret_cast<int4*>(target) + v,
                   __ldbl(reinterpret_cast<const int4*>(source) + v));
        // Do not read an uninitialized token when the count is not a vector.
        if (lane < tokens % 4) {
            const uint32_t token = tokens / 4 * 4 + lane;
            __stbl(target + token, __ldbl(source + token));
        }
    }
    // As in FP4, stores remain in flight while this CTA scatters payload.
    // Every scale-writing warp releases them in the shared Phase 3 handoff.
    __syncwarp();
}

// Quantize once, claim expert slots, and publish using the shared round protocol.
template <class Quantizer, int Hidden, int TopK, uint32_t SourceCapacity,
          uint32_t NumRanks, uint32_t NumExperts, int Intermediate,
          int64_t StagingScales = 0, int64_t StagingCounts = 0, int64_t StagingTokens = 0,
          class TopkIdx = int32_t>
__device__ __forceinline__ void publish_moe_tokens(
    const __ppu_bfloat16* __restrict__ input,    // [num_tokens, Hidden]
    const TopkIdx* __restrict__ topk_ids,       // [num_tokens, topk]
    const float* __restrict__ topk_weights,
    uint32_t num_tokens, void* base,
    const layout::SymBuffer<>& sym_buffer) {

    using Storage = typename Quantizer::Storage;
    using Scale = typename Quantizer::Scale;
    using BufferLayout = layout::Int8MegaMoEBuffer;
    constexpr int kRowBytes = Storage::row_bytes(Hidden);
    constexpr int kScales = Storage::scale_bytes(Hidden) / sizeof(Scale);
    constexpr bool kBroadcastScales = Storage::kTokenScaleBroadcast;
    // This batches independent token scales for communication, never along K.
    constexpr uint32_t kScaleBatchTokens = 32;
    constexpr int kQuantWarps = 7;
    constexpr bool kSeparateClaimWarp = Quantizer::template separate_claim_warp<Hidden>;
    static_assert(kRowBytes % 16 == 0);
    static_assert(SourceCapacity > 0,
                  "SourceCapacity must be positive");
    static_assert(TopK >= 1 && TopK <= 32,
                  "TopK must be in [1, 32]");
    static_assert(is_supported_megamoe_rank_count(NumRanks),
                  "quant requires a power-of-two rank count in [1, 64]");
    static_assert(NumExperts > 0 &&
                      NumExperts % NumRanks == 0,
                  "quant requires a positive uniform expert shard");
    // Shapes are fixed by JIT, so every buffer offset is a compile-time constant.
    constexpr BufferLayout buffers(NumRanks, NumExperts, SourceCapacity, TopK, Hidden, Intermediate,
                                   {StagingScales, StagingCounts, StagingTokens});
    auto* routing_topk_ids = buffers.topk_idx.template data<int64_t>(base);
    auto* routing_topk_weights = buffers.topk_weights.template data<float>(base);
    auto* dispatch_counts = buffers.dispatch_counts[0].template data<uint32_t>(base);
    auto* dispatch_data = buffers.dispatch_data.data(base);
    auto* dispatch_retire = buffers.dispatch_retire.template data<uint32_t>(base);
    auto* staging_scales = buffers.staging_scales[0].template data<Scale>(base);
    auto* staging_counts = buffers.staging_counts[0].template data<uint32_t>(base);
    auto* staging_tokens = buffers.staging_tokens[0].template data<uint32_t>(base);
    volatile uint32_t* preprocess_ready_flags = buffers.preprocess_ready.template data<uint32_t>(base);

    constexpr uint32_t kNumLocalExperts = NumExperts / NumRanks;
    uint32_t rank_idx = sym_buffer.rank_idx;
    constexpr uint32_t kCountBlocks = kNumLocalExperts;
    __shared__ uint32_t s_last_cta;
    uint32_t generation = 0;
    __shared__ uint32_t s_generation;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    bool is_count_cta = blockIdx.x < kCountBlocks;

    constexpr uint64_t kCountStride = buffers.dispatch_counts[0].bytes / sizeof(uint32_t);
    constexpr uint64_t kStagingCountStride = buffers.staging.counts / sizeof(uint32_t);
    constexpr uint64_t kScaleStride = buffers.staging.scales / sizeof(Scale);
    constexpr uint64_t kTokenStride = buffers.staging.tokens / sizeof(uint32_t);
    // CTA leaders advance the generation left by the preceding GEMM.
    constexpr int kClaimWarp = kSeparateClaimWarp ? kQuantWarps : 0;
    bool generation_leader = is_count_cta
        ? threadIdx.x == 0
        : warp_id == kClaimWarp && lane_id == 0;
    uint32_t current = 0u;
    if (generation_leader) {
        // The stream orders the preceding GEMM, so a cached state load is enough.
        current = *preprocess_ready_flags & ~kPreprocessMetadataReadyBit;
        // Wrap to 2 to keep parity alternating during Graph replay.
        current = current >= kFusedRoundGenerationLimit
            ? 2u : current + 1u;
        s_generation = current;
    }
    if (!is_count_cta && warp_id == kClaimWarp)
        generation = __shfl_sync(0xffffffffu, current, 0);

    bool claim_lane = kSeparateClaimWarp
        ? (warp_id == kQuantWarps && lane_id < TopK)
        : (threadIdx.x < TopK);

    // Shared memory: quantized payload data + scales.
    __shared__ __align__(16) uint8_t s_data[kRowBytes];
    __shared__ __align__(16) Scale s_scales[kScales];  // 16B-aligned for int4 push
    __shared__ uint32_t s_slot[TopK];
    __shared__ int s_expert[TopK];
    // Each dedicated count CTA accumulates one local expert across all ranks.
    __shared__ uint32_t s_pushed_counts[NumRanks];

    // Initialize before the final CTA handoff.
    if (threadIdx.x == 0) {
        s_last_cta = 0;
        if constexpr (kBroadcastScales)
            token_scale_batch_state<Storage>()[1] = 0;
    }

    int token_block_idx = static_cast<int>(blockIdx.x) -
        static_cast<int>(kCountBlocks);
    int token_block_count = static_cast<int>(gridDim.x) -
        static_cast<int>(kCountBlocks);

    // One CTA per local expert counts routes; Phase 3 releases the counts.
    if (is_count_cta) {
        if (threadIdx.x < NumRanks)
            s_pushed_counts[threadIdx.x] = 0u;
        __syncthreads();
        generation = s_generation;

        if constexpr (kBroadcastScales) {
            static_assert(buffers.stage1_scale_storage.bytes >=
                ((SourceCapacity + kScaleBatchTokens - 1) / kScaleBatchTokens) * sizeof(uint32_t));
            // Stage1 output scales are dead during pre-dispatch and are not
            // written again until G1 has completed. Reuse their prefix locally.
            if (blockIdx.x == 0 && warp_id == 0) {
                auto* batch_counts = buffers.stage1_scale_storage.template data<uint32_t>(base);
                const uint32_t batches = (num_tokens + kScaleBatchTokens - 1) / kScaleBatchTokens;
                for (uint32_t batch = lane_id; batch < batches; batch += 32)
                    __stbl(batch_counts + batch, 0u);
                __syncwarp();
                if (lane_id == 0) {
                    __ppu_threadfence_uncache();
                    // Words0/1 retire rounds; word2 tags this local initialization.
                    __stbl(dispatch_retire + 2, generation);
                }
            }
        }
        uint32_t local_expert = blockIdx.x;
        uint32_t num_routes = num_tokens * TopK;
        for (uint32_t i = threadIdx.x; i < num_routes;
             i += blockDim.x) {
            const auto expert_idx = __ldg(topk_ids + i);
            if (expert_idx >= 0 && expert_idx < NumExperts) {
                const uint32_t expert = static_cast<uint32_t>(expert_idx);
                uint32_t owner = expert / kNumLocalExperts;
                uint32_t owner_local = expert - owner * kNumLocalExperts;
                if (owner_local == local_expert)
                    atomicAdd(s_pushed_counts + owner, 1u);
            }
        }
        __syncthreads();

        if (threadIdx.x < NumRanks) {
            uint32_t owner = threadIdx.x;
            uint32_t* dst = sym_buffer.map(staging_counts, owner) +
                (generation & 1u) * kStagingCountStride + rank_idx * kNumLocalExperts + local_expert;
            // The arrival flag makes this round's count visible to the consumer.
            __stbl(dst, s_pushed_counts[owner]);
        }
        // Clear stale routes when a later publication uses fewer source tokens.
        for (uint32_t slot = num_routes + blockIdx.x * blockDim.x + threadIdx.x;
             slot < SourceCapacity * TopK; slot += kCountBlocks * blockDim.x) {
            routing_topk_ids[slot] = -1;
            routing_topk_weights[slot] = 0.0f;
        }
    }
    // Each token CTA handles several tokens, then releases them once in Phase 3.
    for (int token_idx = token_block_idx;
         !is_count_cta && token_idx < num_tokens;
         token_idx += token_block_count) {
        const __ppu_bfloat16* token_data = input + static_cast<int64_t>(token_idx) * Hidden;

        // Phase 0: claim unique valid top-k slots while other warps quantize.
        // Use warp 7 for specialized shapes, otherwise the first TopK threads.
        if (claim_lane) {
            int t = kSeparateClaimWarp ? lane_id : threadIdx.x;
            int route_index = token_idx * TopK + t;
            const auto expert_idx = topk_ids[route_index];
            // Check the full input width before narrowing valid IDs for storage.
            bool accepted = expert_idx >= 0 && expert_idx < NumExperts;
            routing_topk_ids[route_index] = accepted ? expert_idx : -1;
            routing_topk_weights[route_index] = accepted
                ? __ldg(topk_weights + route_index)
                : 0.0f;
            s_expert[t] = accepted ? expert_idx : -1;
            if (accepted) {
                uint32_t* counts = dispatch_counts + (generation & 1u) * kCountStride;
                // The last retiring CTA clears this parity after all slot claims.
                s_slot[t] = atomicAdd(&counts[expert_idx], 1u);
            } else {
                s_slot[t] = SourceCapacity;
            }
        }

        Quantizer::template run<Hidden>(token_data, s_data, s_scales);
        // Make slot claims, payload data and scales available to all warps.
        __syncthreads();
        generation = s_generation;

        if constexpr (kBroadcastScales) {
            static_assert(kScales == 1, "INT8 uses one scale for the complete K row");
            const uint32_t batch_begin = uint32_t(token_idx) / kScaleBatchTokens * kScaleBatchTokens;
            const uint32_t batch_tokens = min(kScaleBatchTokens, num_tokens - batch_begin);
            if (threadIdx.x == 0) {
                while (tix::ld_ready_flag_uncached(dispatch_retire + 2) != generation) { }
                __stbl(staging_scales + (generation & 1u) * kScaleStride +
                       uint64_t(rank_idx) * SourceCapacity + token_idx, s_scales[0]);
                // Release this token's scale before the last batch member reads it.
                __ppu_threadfence_uncache();
                auto* counts = buffers.stage1_scale_storage.template data<uint32_t>(base);
                const uint32_t ticket = atomicAdd(counts + batch_begin / kScaleBatchTokens, 1u);
                const bool last = ticket + 1 == batch_tokens;
                token_scale_batch_state<Storage>()[0] = last;
                if (last) token_scale_batch_state<Storage>()[1] = 1;
            }
            __syncthreads();
            if (token_scale_batch_state<Storage>()[0])
                broadcast_token_scales<Scale, NumRanks, SourceCapacity>(
                    staging_scales + (generation & 1u) * kScaleStride,
                    batch_tokens, rank_idx, sym_buffer, batch_begin);
        }

        // Push scales before storing the source row; Phase 3 releases both.
        constexpr int kPayloadWarps = 8;
        constexpr int kDestinationRounds =
            (TopK + kPayloadWarps - 1) / kPayloadWarps;
        // One destination row per warp; TopK > 8 needs multiple rounds.
        #pragma unroll
        for (int round = 0; round < kDestinationRounds; ++round) {
            int t = warp_id + round * kPayloadWarps;
            if (t < TopK) {
                int expert_idx = s_expert[t];
                uint32_t slot = s_slot[t];
                if (expert_idx >= 0 && slot < SourceCapacity) {
                    uint32_t expert = expert_idx;
                    uint32_t owner = expert / kNumLocalExperts;
                    uint32_t local_expert = expert - owner * kNumLocalExperts;
                    uint64_t expert_rank = local_expert * NumRanks + rank_idx;
                    uint64_t staging_idx = expert_rank * SourceCapacity + slot;
                    if constexpr (!kBroadcastScales) {
                        Scale* staging = sym_buffer.map(staging_scales, owner) +
                            (generation & 1u) * kScaleStride;
                        if constexpr (sizeof(Scale) * kScales >= 16) {
                            static_assert((sizeof(Scale) * kScales) % 16 == 0);
                            auto* scale_dst = reinterpret_cast<int4*>(staging + staging_idx * kScales);
                            auto* scale_src = reinterpret_cast<const int4*>(s_scales);
                            for (int v = lane_id; v < sizeof(Scale) * kScales / 16; v += 32)
                                __stbl(scale_dst + v, scale_src[v]);
                        } else {
                            for (int v = lane_id; v < kScales; v += 32)
                                __stbl(staging + staging_idx * kScales + v, s_scales[v]);
                        }
                    }
                    // Token metadata shares the scales' release and arrival flag.
                    if (lane_id == 0) {
                        uint32_t* token_dst = sym_buffer.map(staging_tokens, owner) +
                            (generation & 1u) * kTokenStride;
                        uint32_t token_topk = token_idx * TopK + t;
                        __stbl(token_dst + staging_idx, token_topk);
                    }
                }
            }
        }

        // Phase 2: store one row per source token; route metadata selects it on pull.
        constexpr int kPayloadVectors = kRowBytes / 16;
        const int4* s_payload_v = reinterpret_cast<const int4*>(s_data);
        constexpr int kPayloadWriterWarps = TopK < kPayloadWarps ? TopK : kPayloadWarps;
        // Only these warps participate in the existing Phase 3 release.
        if (warp_id < kPayloadWriterWarps) {
            auto* payload_out = reinterpret_cast<int4*>(dispatch_data + uint64_t(token_idx) * kRowBytes);
            for (int i = threadIdx.x; i < kPayloadVectors; i += kPayloadWriterWarps * 32)
                payload_out[i] = s_payload_v[i];
        }

        // Wait only before reusing shared memory for the next token.
        if (token_idx + token_block_count < num_tokens) {
            __syncthreads();
        }
    }

    if (!is_count_cta && num_tokens == 0) {
        // Empty token CTAs still join the final handoff.
        __syncthreads();
        generation = s_generation;
    }

    // Phase 3: release payloads, then let the last CTA signal consumers.
    uint32_t* round_retire_counter = dispatch_retire + (generation & 1u);
    // Each writer warp releases its stores before retiring.
    __syncwarp();
    // Count CTAs release all warps; token CTAs release their payload warps.
    bool pushed_scales = false;
    if constexpr (kBroadcastScales)
        pushed_scales = token_scale_batch_state<Storage>()[1] != 0;
    if ((threadIdx.x & 31) == 0 &&
        (is_count_cta || warp_id < TopK || pushed_scales)) {
        // Release peer stores at system scope without flushing L1.
        __ppu_threadfence_system_uncache();
    }
    // Wait for every warp's release before taking a retire ticket.
    __syncthreads();

    if (threadIdx.x == 0) {
        // The last ticket resets the counter for the next round.
        uint32_t ticket = atomicInc(
            round_retire_counter, gridDim.x - 1);
        if (ticket == gridDim.x - 1) {
            // All CTAs have released; GEMM will set the ready bit after metadata.
            __stbl(const_cast<uint32_t*>(preprocess_ready_flags),
                   generation);
            s_last_cta = 1;
        }
    }
    // Share the last-ticket result with the notification lanes.
    __syncthreads();
    if (s_last_cta) {
        // Clear slot counters before reusing this parity.
        uint32_t* counts = dispatch_counts + (generation & 1u) * kCountStride;
        for (uint32_t expert = threadIdx.x;
             expert < NumExperts; expert += blockDim.x)
            counts[expert] = 0u;
        comm::rank_arrive(layout::SymBuffer<NumRanks>(sym_buffer),
                         static_cast<layout::MegaMoESignals*>(base)->dispatch_arrivals, generation);
    }
}

}  // namespace deep_gemm

namespace deep_gemm {

template <int Threads>
__device__ __forceinline__ float int8_row_amax(float value, float* warp_max, float epsilon) {
    static_assert(Threads % 32 == 0 && Threads <= 1024);
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    const int lane = threadIdx.x % 32;
    if (lane == 0) warp_max[threadIdx.x / 32] = value;
    __syncthreads();
    value = lane < Threads / 32 ? warp_max[lane] : epsilon;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

// Per-token INT8 scales follow the existing A8W8 GEMM contract.
template <bool kInputAligned>
struct Int8TokenQuantizer {
    using Storage = layout::Int8Storage;
    using Scale = float;
    template <int Hidden> static constexpr bool separate_claim_warp = false;

    template <int Hidden, bool NormalQuantization = false>
    __device__ static __forceinline__ void run(
        const __ppu_bfloat16* input, uint8_t* data, float* scales) {
        // Legacy DeepEP QUANT_SIZE_CHANNEL: vector loads, amax >= 1e-6,
        // multiplier 127/amax, FP32 dequant scale amax*(1/127), round-to-nearest.
        static_assert(Hidden % 8 == 0);
        __shared__ float warp_max[8];
        constexpr float epsilon = NormalQuantization ? 1.0e-10f : 1.0e-6f;
        float amax = epsilon;
        for (int i = threadIdx.x; i < Hidden / 8; i += 256) {
            int4 raw = tix::load_bf16x8<kInputAligned>(input + i * 8);
            const auto* values = reinterpret_cast<const __ppu_bfloat16*>(&raw);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                amax = fmaxf(amax, fabsf(__bfloat162float(values[j])));
        }
        amax = int8_row_amax<256>(amax, warp_max, epsilon);
        // Match dense Legacy's Triton division at half-integer boundaries.
        const float multiplier = NormalQuantization ? 127.0f * __ppu_rcpf(amax) : 127.0f / amax;
        if (threadIdx.x == 0)
            scales[0] = amax * (1.0f / 127.0f);
        for (int i = threadIdx.x; i < Hidden / 8; i += 256) {
            int4 raw = tix::load_bf16x8<kInputAligned>(input + i * 8);
            const auto* values = reinterpret_cast<const __ppu_bfloat16*>(&raw);
            int2 packed;
            auto* quantized = reinterpret_cast<int8_t*>(&packed);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float value = __bfloat162float(values[j]) * multiplier;
                if constexpr (NormalQuantization)
                    quantized[j] = static_cast<int8_t>(roundf(value));
                else
                    quantized[j] = cutlass::NumericConverter<int8_t, float>{}(value);
            }
            reinterpret_cast<int2*>(data)[i] = packed;
        }
    }
};

// Original dense Legacy uses round-to-nearest-away and a smaller amax floor.
// Publication and buffer layout are identical; only the quantizer differs.
template <bool kInputAligned>
struct NormalInt8TokenQuantizer : Int8TokenQuantizer<kInputAligned> {
    template <int Hidden>
    __device__ static __forceinline__ void run(
        const __ppu_bfloat16* input, uint8_t* data, float* scales) {
        Int8TokenQuantizer<kInputAligned>::template run<Hidden, true>(input, data, scales);
    }
};

}  // namespace deep_gemm
