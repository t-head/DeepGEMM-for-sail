#pragma once

#include <cstdint>
#include <type_traits>
#include <hggc_bf16.h>
#include <cutlass/numeric_conversion.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/tix/ld_st.cuh>
#include <deep_gemm/tix/utils.cuh>

namespace deep_gemm {

// Pack two FP32 values into one FP4 byte.
__device__ __forceinline__ uint8_t cvt_f32x2_to_fp4x2(float hi, float lo) {
    uint8_t b;
    asm volatile("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;\n"
                 : "=c"(b)
                 : "f"(hi), "f"(lo));
    return b;
}

__device__ __forceinline__ __ppu_bfloat16 fast_pow2_bf16(int x) {
    uint16_t bits = static_cast<uint16_t>((x + 127) << 7);
    return *reinterpret_cast<__ppu_bfloat16*>(&bits);
}

// Return the BF16 quantization multiplier and UE8M0 dequantization scale.
__device__ __forceinline__ void calculate_mxfp4_scales_bf16(
    __ppu_bfloat16 amax, __ppu_bfloat16& scale, uint8_t& scale_inv) {
    constexpr float kAmaxInvMXFP4 = 0.16666667f;  // 1/6.0, max representable E2M1 value
    __ppu_bfloat16 scale_inv_bf16 = __hmul(amax, __float2bfloat16(kAmaxInvMXFP4));
    uint16_t scale_inv_bits = *reinterpret_cast<uint16_t*>(&scale_inv_bf16);
    // Round the magnitude up to a UE8M0 exponent.
    uint16_t biased_exp_scale_inv = static_cast<uint16_t>(
        ((scale_inv_bits & 0x7fffu) + 0x7fu) >> 7);
    int exp_scale_inv = biased_exp_scale_inv - 127;
    scale = fast_pow2_bf16(-exp_scale_inv);
    scale_inv = static_cast<uint8_t>(biased_exp_scale_inv);
}

// Per-token INT8 scales follow the existing A8W8 GEMM contract.
template <int Hidden, bool kInputAligned>
__device__ __forceinline__ void quantize_int8_token(
    const __ppu_bfloat16* input, uint8_t* data, float* scales) {
    // Legacy DeepEP QUANT_SIZE_CHANNEL: vector loads, amax >= 1e-6,
    // multiplier 127/amax, FP32 dequant scale amax*(1/127), round-to-nearest.
    static_assert(Hidden % 8 == 0);
    __shared__ float warp_max[8];
    constexpr float epsilon = 1.0e-6f;
    float amax = epsilon;
    for (int i = threadIdx.x; i < Hidden / 8; i += 256) {
        int4 raw = tix::load_bf16x8<kInputAligned>(input + i * 8);
        const auto* values = reinterpret_cast<const __ppu_bfloat16*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            amax = fmaxf(amax, fabsf(__bfloat162float(values[j])));
    }
    amax = tix::block_reduce_max<256>(amax, warp_max, epsilon);
    const float multiplier = 127.0f / amax;
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
            quantized[j] = cutlass::NumericConverter<int8_t, float>{}(value);
        }
        reinterpret_cast<int2*>(data)[i] = packed;
    }
}

// The source-rank slice is already local. Complete warps send adjacent vectors
// to each peer, instead of one remote scalar store for every expert route.
template <uint32_t NumRanks, uint32_t SourceCapacity>
__device__ __forceinline__ void broadcast_token_scales(
    float* staging, uint32_t tokens, uint32_t rank,
    const layout::SymBuffer<>& symmetric, uint32_t token_begin = 0) {
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

// Claim routes and publish quantized source rows through one round protocol.
template <bool Int8, int Hidden, int TopK, uint32_t SourceCapacity,
          uint32_t NumRanks, uint32_t NumExperts, int Intermediate,
          int64_t StagingScales, int64_t StagingCounts, int64_t StagingTokens, typename TopkIdx, bool kInputAligned>
__global__ void mega_moe_pre_dispatch(
    const __ppu_bfloat16* __restrict__ input,    // [num_tokens, Hidden]
    const TopkIdx* __restrict__ topk_ids,       // [num_tokens, topk]
    const float* __restrict__ topk_weights,
    uint32_t num_tokens, layout::SymBuffer<> sym_buffer) {

    static_assert(Hidden >= 1024 && Hidden % 512 == 0,
                  "MegaMoE hidden must be a multiple of 512 and at least 1024");
    constexpr int kElementsPerThread = 16; // 16 BF16 values = 2 int4 loads
    constexpr int kScaleGroupSize = 32;
    constexpr int kLanesPerScale = kScaleGroupSize / kElementsPerThread;  // 2
    constexpr int kQuantChunks = Hidden / kElementsPerThread;
    using Scale = std::conditional_t<Int8, float, uint16_t>;
    using BufferLayout = layout::MegaMoEBufferLayout<Int8>;
    constexpr int kRowBytes = BufferLayout::Storage::row_bytes(Hidden);
    constexpr int kScales = BufferLayout::Storage::scale_bytes(Hidden) / sizeof(Scale);
    // INT8 batches independent token scales for communication, never along K.
    constexpr uint32_t kScaleBatchTokens = 32;
    // For shapes such as Hidden=7168, reserve warp 7 for slot claims.
    constexpr int kQuantWarps = 7;
    constexpr int kQuantThreads = kQuantWarps * 32;
    constexpr bool kSeparateClaimWarp = !Int8 && (kQuantChunks % kQuantThreads) == 0;
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
    auto* base = sym_buffer.get_base_ptr<layout::MegaMoESignals*>();
    auto* routing_topk_ids = buffers.topk_idx.template data<int64_t>(base);
    auto* routing_topk_weights = buffers.topk_weights.template data<float>(base);
    auto* dispatch_counts = buffers.dispatch_counts[0].template data<uint32_t>(base);
    auto* dispatch_data = buffers.dispatch_data.data(base);
    auto* dispatch_retire = buffers.dispatch_retire.template data<uint32_t>(base);
    auto* staging_scales = buffers.staging_scales[0].template data<Scale>(base);
    auto* staging_tokens = buffers.staging_tokens[0].template data<uint32_t>(base);
    volatile uint32_t* preprocess_ready_flags = buffers.preprocess_ready.template data<uint32_t>(base);

    constexpr uint32_t kNumLocalExperts = NumExperts / NumRanks;
    uint32_t rank_idx = sym_buffer.rank_idx;
    constexpr uint32_t kCountBlocks = kNumLocalExperts;
    uint32_t generation = 0;
    __shared__ uint32_t s_generation;
    __shared__ uint32_t s_scale_batch[2]; // current batch sender, sticky sender for release

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    bool is_count_cta = blockIdx.x < kCountBlocks;

    constexpr uint64_t kCountStride = buffers.dispatch_counts[0].bytes / sizeof(uint32_t);
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

    __shared__ __align__(16) uint8_t s_data[kRowBytes];
    __shared__ __align__(16) Scale s_scales[kScales];
    __shared__ uint32_t s_slot[TopK];
    __shared__ int s_expert[TopK];
    __shared__ bool s_push_scale[TopK];
    // Each dedicated count CTA accumulates one local expert across all ranks.
    __shared__ uint32_t s_pushed_counts[NumRanks];

    if constexpr (Int8)
        if (threadIdx.x == 0)
            s_scale_batch[1] = 0;

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

        if constexpr (Int8) {
            static_assert(buffers.l2_acts_sf.bytes >=
                ((SourceCapacity + kScaleBatchTokens - 1) / kScaleBatchTokens) * sizeof(uint32_t));
            // Stage1 output scales are dead during pre-dispatch and are not
            // written again until G1 has completed. Reuse their prefix locally.
            if (blockIdx.x == 0 && warp_id == 0) {
                auto* batch_counts = buffers.l2_acts_sf.template data<uint32_t>(base);
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
        const uint32_t num_routes = num_tokens * TopK;
        const uint32_t local_expert = blockIdx.x;
        for (uint32_t i = threadIdx.x; i < num_routes; i += blockDim.x) {
            const auto route = __ldg(topk_ids + i);
            if (route >= 0 && uint64_t(route) < NumExperts) {
                const uint32_t expert = static_cast<uint32_t>(route);
                const uint32_t owner = expert / kNumLocalExperts;
                if (expert - owner * kNumLocalExperts == local_expert)
                    atomicAdd(s_pushed_counts + owner, 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x < NumRanks) {
            auto* counts = buffers.staging_counts[generation & 1u].template data<uint32_t>(base);
            __stbl(sym_buffer.map(counts, threadIdx.x) + sym_buffer.rank_idx * kNumLocalExperts + local_expert,
                   s_pushed_counts[threadIdx.x]);
        }
        auto* ids = buffers.topk_idx.template data<int64_t>(base);
        auto* weights = buffers.topk_weights.template data<float>(base);
        for (uint32_t slot = num_routes + blockIdx.x * blockDim.x + threadIdx.x;
             slot < SourceCapacity * TopK; slot += kNumLocalExperts * blockDim.x) {
            ids[slot] = -1;
            weights[slot] = 0.0f;
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
            const auto route = topk_ids[route_index];
            bool accepted = route >= 0 && uint64_t(route) < NumExperts;
            // Narrow only after validating the full input ID.
            int expert_idx = accepted ? static_cast<int>(route) : -1;
            routing_topk_ids[route_index] = accepted ? expert_idx : -1;
            routing_topk_weights[route_index] = accepted
                ? __ldg(topk_weights + route_index)
                : 0.0f;
            s_expert[t] = accepted ? expert_idx : -1;
            uint32_t slot = SourceCapacity;
            if (accepted) {
                uint32_t* counts = dispatch_counts + (generation & 1u) * kCountStride;
                // The last retiring CTA clears this parity after all slot claims.
                slot = atomicAdd(&counts[expert_idx], 1u);
            }
            s_slot[t] = slot;
            if constexpr (!Int8) {
                uint32_t owner = slot < SourceCapacity ? uint32_t(expert_idx) / kNumLocalExperts : NumRanks;
                const uint32_t peers = __match_any_sync(0xffffffffu >> (32 - TopK), owner);
                s_push_scale[t] = (peers & __ppu_read_lanemask_lt()) == 0;
            }
        }

        if constexpr (Int8) {
            quantize_int8_token<Hidden, kInputAligned>(token_data, s_data, s_scales);
        } else {
            // Phase 1: preload BF16 vectors, then quantize into shared memory.
            constexpr int kActiveQuantThreads = kSeparateClaimWarp ? kQuantThreads : 256;
            constexpr int kQuantIterations =
                (kQuantChunks + kActiveQuantThreads - 1) / kActiveQuantThreads;
            int4 raw[kQuantIterations][2];
            #pragma unroll
            for (int it = 0; it < kQuantIterations; ++it) {
                int i = it * kActiveQuantThreads + threadIdx.x;
                if (threadIdx.x < kActiveQuantThreads && i < kQuantChunks) {
                    const auto* ptr = token_data + i * kElementsPerThread;
                    raw[it][0] = tix::load_bf16x8<kInputAligned>(ptr);
                    raw[it][1] = tix::load_bf16x8<kInputAligned>(ptr + 8);
                }
            }
            #pragma unroll
            for (int it = 0; it < kQuantIterations; ++it) {
                int i = it * kActiveQuantThreads + threadIdx.x;
                if (threadIdx.x >= kActiveQuantThreads || i >= kQuantChunks) continue;
                __ppu_bfloat162 local_v2[kElementsPerThread / 2];

                __ppu_bfloat162 amax2 = __float2bfloat162_rn(0.0f);
                for (int j = 0; j < kElementsPerThread / 2; ++j) {
                    local_v2[j] = reinterpret_cast<const __ppu_bfloat162*>(&raw[it][j / 4])[j % 4];
                    amax2 = __hmax2(amax2, __habs2(local_v2[j]));
                }
                __ppu_bfloat16 amax = __hmax(amax2.x, amax2.y);

                amax = __hmax(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 2));

                // Both lanes use the same scale; only the leader stores it.
                __ppu_bfloat16 scale;
                uint8_t scale_inv;
                calculate_mxfp4_scales_bf16(amax, scale, scale_inv);
                if ((lane_id & (kLanesPerScale - 1)) == 0) {
                    reinterpret_cast<uint8_t*>(s_scales)[i / kLanesPerScale] = scale_inv;
                }
                __ppu_bfloat162 scale2 = __halves2bfloat162(scale, scale);
                int2 int_value;
                uint8_t* fp4x2_values = reinterpret_cast<uint8_t*>(&int_value);
                for (int j = 0; j < kElementsPerThread / 2; ++j) {
                    __ppu_bfloat162 scaled = __hmul2(local_v2[j], scale2);
                    float2 f2 = __bfloat1622float2(scaled);
                    fp4x2_values[j] = cvt_f32x2_to_fp4x2(f2.y, f2.x);
                }

                reinterpret_cast<int2*>(s_data)[i] = int_value;
            }
        }
        // Make slot claims, payload data and scales available to all warps.
        __syncthreads();
        generation = s_generation;

        if constexpr (Int8) {
            const uint32_t batch_begin = uint32_t(token_idx) / kScaleBatchTokens * kScaleBatchTokens;
            const uint32_t batch_tokens = min(kScaleBatchTokens, num_tokens - batch_begin);
            if (threadIdx.x == 0) {
                while (tix::ld_ready_flag_uncached(dispatch_retire + 2) != generation) { }
                __stbl(staging_scales + (generation & 1u) * kScaleStride +
                       uint64_t(rank_idx) * SourceCapacity + token_idx, s_scales[0]);
                // Release this token's scale before the last batch member reads it.
                __ppu_threadfence_uncache();
                auto* counts = buffers.l2_acts_sf.template data<uint32_t>(base);
                const uint32_t ticket = atomicAdd(counts + batch_begin / kScaleBatchTokens, 1u);
                const bool last = ticket + 1 == batch_tokens;
                s_scale_batch[0] = last;
                if (last) s_scale_batch[1] = 1;
            }
            __syncthreads();
            if (s_scale_batch[0])
                broadcast_token_scales<NumRanks, SourceCapacity>(
                    staging_scales + (generation & 1u) * kScaleStride,
                    batch_tokens, rank_idx, sym_buffer, batch_begin);
        }

        // Publish scales and route metadata before the shared Phase 3 release.
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
                    if constexpr (!Int8) {
                        if (s_push_scale[t]) {
                            uint16_t* staging = sym_buffer.map(staging_scales, owner) +
                                (generation & 1u) * kScaleStride;
                            uint64_t scale_row = uint64_t(rank_idx) * SourceCapacity + token_idx;
                            auto* scale_dst = reinterpret_cast<int4*>(staging + scale_row * kScales);
                            auto* scale_src = reinterpret_cast<const int4*>(s_scales);
                            constexpr int kScaleVectors = kScales / 8;
                            for (int v = lane_id; v < kScaleVectors; v += 32)
                                __stbl(scale_dst + v, scale_src[v]);
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

        // Phase 2: store each source row once; route metadata selects it on pull.
        constexpr int kPayloadVectors = kRowBytes / 16;
        const int4* s_payload_v = reinterpret_cast<const int4*>(s_data);
        constexpr int kPayloadWriterWarps = TopK < kPayloadWarps ? TopK : kPayloadWarps;
        // Use only warps already released by the existing Phase 3 handoff.
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

    bool pushed_scales = false;
    if constexpr (Int8)
        pushed_scales = !is_count_cta && token_block_idx < num_tokens && s_scale_batch[1] != 0;
    comm::pre_dispatch_release<NumRanks, NumExperts>(
        *base, sym_buffer, generation, is_count_cta || warp_id < TopK || pushed_scales);
}

}  // namespace deep_gemm
