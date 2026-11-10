#pragma once

#include <cstdint>
#include <hggc_bf16.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/tix/ld_st.cuh>

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

// Store one FP4 row per source token and push its scales once per destination rank.
template <int Hidden, int TopK, uint32_t SourceCapacity,
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
    constexpr int kFp4Words = Hidden / 8;
    constexpr int kPackedScales = Hidden / 64;  // Two UE8M0 scales per uint16_t.
    // For shapes such as Hidden=7168, reserve warp 7 for slot claims.
    constexpr int kQuantWarps = 7;
    constexpr int kQuantThreads = kQuantWarps * 32;
    constexpr bool kSeparateClaimWarp = (kQuantChunks % kQuantThreads) == 0;
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
    constexpr layout::MegaMoEBuffer buffers(NumRanks, NumExperts, SourceCapacity, TopK, Hidden, Intermediate,
                                           {StagingScales, StagingCounts, StagingTokens});
    auto* base = sym_buffer.get_base_ptr<layout::MegaMoESignals*>();
    auto* routing_topk_ids = buffers.topk_idx.data<int64_t>(base);
    auto* routing_topk_weights = buffers.topk_weights.data<float>(base);
    auto* dispatch_counts = buffers.dispatch_counts[0].data<uint32_t>(base);
    auto* dispatch_fp4 = buffers.dispatch_fp4.data(base);
    auto* dispatch_retire = buffers.dispatch_retire.data<uint32_t>(base);
    auto* staging_scales = buffers.staging_scales[0].data<uint16_t>(base);
    auto* staging_counts = buffers.staging_counts[0].data<uint32_t>(base);
    auto* staging_tokens = buffers.staging_tokens[0].data<uint32_t>(base);
    volatile uint32_t* preprocess_ready_flags = buffers.preprocess_ready.data<uint32_t>(base);

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

    // Shared memory: quantized FP4 data + packed scales.
    __shared__ __align__(16) int s_fp4_data[kFp4Words];
    __shared__ __align__(16) uint16_t s_scales[kPackedScales];  // 16B-aligned for int4 push
    __shared__ uint32_t s_slot[TopK];
    __shared__ int s_expert[TopK];
    __shared__ bool s_push_scale[TopK];
    // Each dedicated count CTA accumulates one local expert across all ranks.
    __shared__ uint32_t s_pushed_counts[NumRanks];

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

        uint32_t local_expert = blockIdx.x;
        uint32_t num_routes = num_tokens * TopK;
        for (uint32_t i = threadIdx.x; i < num_routes;
             i += blockDim.x) {
            const auto route = __ldg(topk_ids + i);
            if (route >= 0 && uint64_t(route) < NumExperts) {
                uint32_t expert = static_cast<uint32_t>(route);
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
                (generation & 1u) * (buffers.staging.counts / sizeof(uint32_t)) +
                rank_idx * kNumLocalExperts + local_expert;
            // The arrival flag makes this round's count visible to the consumer.
            __stbl(dst, s_pushed_counts[owner]);
        }
        // Fill [num_tokens, SourceCapacity) with (-1, 0).
        for (uint32_t slot = num_routes + blockIdx.x * blockDim.x + threadIdx.x;
             slot < SourceCapacity * TopK;
             slot += kCountBlocks * blockDim.x) {
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
            uint32_t owner = slot < SourceCapacity ? uint32_t(expert_idx) / kNumLocalExperts : NumRanks;
            const uint32_t peers = __match_any_sync(0xffffffffu >> (32 - TopK), owner);
            s_push_scale[t] = (peers & __ppu_read_lanemask_lt()) == 0;
        }

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

            reinterpret_cast<int2*>(s_fp4_data)[i] = int_value;
        }
        // Make slot claims, FP4 data and scales available to all warps.
        __syncthreads();
        generation = s_generation;

        // Push scales before storing the source FP4 row; Phase 3 releases both.
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
                    if (s_push_scale[t]) {
                        uint16_t* staging = sym_buffer.map(staging_scales, owner) +
                            (generation & 1u) * (buffers.staging.scales / sizeof(uint16_t));
                        uint64_t scale_row = uint64_t(rank_idx) * SourceCapacity + token_idx;
                        auto* scale_dst = reinterpret_cast<int4*>(staging + scale_row * kPackedScales);
                        auto* scale_src = reinterpret_cast<const int4*>(s_scales);
                        constexpr int kScaleVectors = kPackedScales / 8;
                        for (int v = lane_id; v < kScaleVectors; v += 32)
                            __stbl(scale_dst + v, scale_src[v]);
                    }
                    // Token metadata shares the scales' release and arrival flag.
                    if (lane_id == 0) {
                        uint32_t* token_dst = sym_buffer.map(staging_tokens, owner) +
                            (generation & 1u) * (buffers.staging.tokens / sizeof(uint32_t));
                        uint32_t token_topk = token_idx * TopK + t;
                        __stbl(token_dst + staging_idx, token_topk);
                    }
                }
            }
        }

        // Phase 2: store each source row once; route metadata selects it on pull.
        constexpr int kFp4Vectors = kFp4Words / 4;
        const int4* s_fp4_v = reinterpret_cast<const int4*>(s_fp4_data);
        constexpr int kFp4WriterWarps = TopK < kPayloadWarps ? TopK : kPayloadWarps;
        // Use only warps already released by the existing Phase 3 handoff.
        if (warp_id < kFp4WriterWarps) {
            auto* fp4_out = reinterpret_cast<int4*>(dispatch_fp4 + uint64_t(token_idx) * (Hidden / 2));
            for (int i = threadIdx.x; i < kFp4Vectors; i += kFp4WriterWarps * 32)
                fp4_out[i] = s_fp4_v[i];
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
    if ((threadIdx.x & 31) == 0 &&
        (is_count_cta || warp_id < TopK)) {
        // Release peer stores at system scope without flushing L1.
        __ppu_threadfence_system_uncache();
    }
    // Wait for every warp's release before taking a retire ticket.
    __syncthreads();

    if (threadIdx.x == 0) {
        // The last ticket resets the counter for the next round.
        uint32_t ticket = atomicInc(
            round_retire_counter, gridDim.x - 1);
        const bool is_last_cta = ticket == gridDim.x - 1;
        if (is_last_cta) {
            // All CTAs have released; GEMM will set the ready bit after metadata.
            __stbl(const_cast<uint32_t*>(preprocess_ready_flags),
                   generation);
        }
        s_last_cta = is_last_cta;
    }
    // Share the last-ticket result with the notification lanes.
    __syncthreads();
    if (s_last_cta) {
        // Clear slot counters before reusing this parity.
        uint32_t* counts = dispatch_counts + (generation & 1u) * kCountStride;
        for (uint32_t expert = threadIdx.x;
             expert < NumExperts; expert += blockDim.x)
            counts[expert] = 0u;
        comm::rank_arrive(sym_buffer, base->dispatch_arrivals, generation);
    }
}

}  // namespace deep_gemm
