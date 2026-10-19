#pragma once

#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <hggc_fp8.h>

namespace deep_gemm::mega_moe::detail {

constexpr uint64_t kBarrierTimeoutCycles = 600000000000ULL;

__device__ __forceinline__ int32_t load_barrier_flag(const int32_t* ptr) {
    int32_t value;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ void combine_fence() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

// Every rank enqueues this kernel after its scatter on the same stream. Each
// flag points to one zero-initialized, peer-visible int32 generation counter.
// Only block (0, 0) publishes arrival and polls the peer counters. The local
// done flag is cleared on the launch stream before each invocation, including
// graph replay. Empty ranks must launch this kernel too.
__device__ __forceinline__ void combine_barrier(int32_t** flags, int rank, int world, int32_t* done) {
    const int tid = threadIdx.x;
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        if (tid < 32) {
            int32_t generation = 0;
            // Order completed scatter writes before publishing this rank.
            combine_fence();
            if (tid == 0) {
                generation = static_cast<int32_t>(static_cast<uint32_t>(load_barrier_flag(flags[rank])) + 1u);
                asm volatile("st.volatile.global.s32 [%0], %1;"
                             :: "l"(flags[rank]), "r"(generation) : "memory");
            }
            generation = __shfl_sync(0xffffffffu, generation, 0);
            const auto start = clock64();
            while (true) {
                const int32_t peer = tid < world ? load_barrier_flag(flags[tid]) : generation;
                // Peers can be one generation ahead. Unsigned subtraction also
                // handles int32 wraparound without signed overflow.
                const bool arrived = static_cast<int32_t>(static_cast<uint32_t>(peer) -
                                                         static_cast<uint32_t>(generation)) >= 0;
                if (__all_sync(0xffffffffu, arrived))
                    break;
                if (clock64() - start > kBarrierTimeoutCycles) {
                    if (tid == 0)
                        printf("MegaMoE combine barrier timeout: rank=%d generation=%d\n", rank, generation);
                    asm volatile("trap;");
                }
            }
            combine_fence();
            if (tid == 0)
                atomicExch(done, 1);
        }
    } else if (tid == 0) {
        const auto start = clock64();
        while (load_barrier_flag(done) == 0) {
            if (clock64() - start > kBarrierTimeoutCycles)
                asm volatile("trap;");
        }
    }
    // One thread acquires completion for the CTA; the converged CTA barrier
    // orders all consumers after that acquire. Data loads bypass the L1 cache.
    if (tid == 0)
        combine_fence();
#if __HGGC_ARCH__ == 150
    __ppu_barrier_sync_nocnt(0, 1);
#else
    __syncthreads();
#endif
}

// BF16 and FP8 share the barrier, row traversal, masking, accumulation and
// output conversion. Only the vector load/dequantization depends on dtype.
template <bool kFp8, bool kScaled>
__device__ __forceinline__ void load_combine_vector(
    const void* buf, const uint64_t* scales, int64_t row, int hidden, int vector,
    int scale_tiles, int scale_block_n, float& weight, float2 (&values)[4]) {
    const int64_t offset = row * hidden + vector * 8;
    if constexpr (kFp8) {
        const uint2 packed = __ldcg(reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(buf) + offset));
        const uint16_t pairs[4] = {static_cast<uint16_t>(packed.x), static_cast<uint16_t>(packed.x >> 16),
                                   static_cast<uint16_t>(packed.y), static_cast<uint16_t>(packed.y >> 16)};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half2 half = static_cast<__half2>(
                __hg_cvt_fp8x2_to_halfraw2(static_cast<__hg_fp8x2_storage_t>(pairs[i]), __HG_E4M3));
            values[i] = __half22float2(half);
        }
        if constexpr (kScaled) {
            const uint64_t packed_scale = __ldcg(scales + row * scale_tiles + vector * 8 / scale_block_n);
            weight *= __uint_as_float(static_cast<uint32_t>(packed_scale & 0xffffu) << 16);
        }
    } else {
        const float4 packed = __ldcg(reinterpret_cast<const float4*>(static_cast<const __ppu_bfloat16*>(buf) + offset));
        const auto* pairs = reinterpret_cast<const __ppu_bfloat162*>(&packed);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            values[i] = __bfloat1622float2(pairs[i]);
    }
}

template <int kTopk, bool kFp8, bool kScaled, int kChunkVecs, int kThreads>
__device__ __forceinline__ void barrier_reduce_kernel(
    const void* buf, const uint64_t* scales, const float* weights, __ppu_bfloat16* output,
    int tokens, int hidden, int capacity, int scale_tiles, int scale_block_n, int fp8_exp_shift,
    int32_t** flags, int rank, int world, int32_t* done) {
    combine_barrier(flags, rank, world, done);
    const int token = blockIdx.x;
    if (token >= tokens)
        return;
    const int begin = kChunkVecs ? blockIdx.y * kChunkVecs : 0;
    const int end = kChunkVecs ? min(begin + kChunkVecs, hidden / 8) : hidden / 8;
    float route_weights[kTopk];
    #pragma unroll
    for (int slot = 0; slot < kTopk; ++slot) {
        route_weights[slot] = __ldg(weights + static_cast<int64_t>(token) * kTopk + slot);
        if constexpr (kFp8 && !kScaled)
            route_weights[slot] *= ldexpf(1.0f, fp8_exp_shift);
    }
    for (int vector = begin + threadIdx.x; vector < end; vector += kThreads) {
        float2 acc[4] = {};
        #pragma unroll
        for (int slot = 0; slot < kTopk; ++slot) {
            // Masked routes need not have a scatter producer. In particular,
            // do not read stale NaNs and then multiply them by zero.
            if (route_weights[slot] == 0.0f)
                continue;
            const int64_t row = static_cast<int64_t>(slot) * capacity + token;
            float weight = route_weights[slot];
            float2 values[4];
            load_combine_vector<kFp8, kScaled>(buf, scales, row, hidden, vector,
                                               scale_tiles, scale_block_n, weight, values);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i].x += weight * values[i].x;
                acc[i].y += weight * values[i].y;
            }
        }
        float4 packed;
        auto* pairs = reinterpret_cast<__ppu_bfloat162*>(&packed);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            pairs[i] = __float22bfloat162_rn(acc[i]);
        *reinterpret_cast<float4*>(output + static_cast<int64_t>(token) * hidden + vector * 8) = packed;
    }
}

} // namespace deep_gemm::mega_moe::detail
