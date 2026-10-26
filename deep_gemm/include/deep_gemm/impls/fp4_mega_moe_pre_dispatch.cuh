#pragma once

#include <cutlass/cutlass.h>
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <cstdint>
#include <math.h>
#include <deep_gemm/tix/cvt.cuh>
#include <deep_gemm/tix/utils.cuh>
#include <deep_gemm/impls/mega_moe_io.cuh>

namespace deep_gemm::mega_moe::fp4 {

template <uint32_t HIDDEN, uint32_t TOP_K, uint32_t BLOCK_THREADS = 256>
CUTLASS_DEVICE void fp4_mega_moe_pre_dispatch_kernel_device(
    const __ppu_bfloat16* __restrict__ hidden_states,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    uint8_t* __restrict__ staged_tokens,
    uint8_t* __restrict__ staged_scales,
    int64_t* __restrict__ input_topk_idx,
    float* __restrict__ input_topk_weights,
    uint32_t num_tokens) {

    static_assert(HIDDEN % 64 == 0, "HIDDEN must be a multiple of 64 for paired MXFP4 scales");
    // Four lanes quantize one 32-value MXFP4 group; each lane loads eight BF16 values.
    constexpr uint32_t kVecElems = 8;
    constexpr uint32_t kThreadsPerGroup = 4;
    constexpr uint32_t kTokenBytes = HIDDEN / 2;
    constexpr uint32_t kNumGroups = HIDDEN / 32;
    constexpr uint32_t kSfBytes = kNumGroups;
    constexpr uint32_t kNumVecs = HIDDEN / kVecElems;
    static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be warp-aligned");
    static_assert(BLOCK_THREADS <= kNumVecs, "BLOCK_THREADS must not exceed HIDDEN/8");
    constexpr bool kEvenVecSchedule = (kNumVecs % BLOCK_THREADS) == 0;
    constexpr uint32_t kIters = kNumVecs / BLOCK_THREADS;

    const uint32_t tid = threadIdx.x;
    for (uint32_t token = blockIdx.x; token < num_tokens; token += gridDim.x) {
        const auto* row = hidden_states + static_cast<uint64_t>(token) * HIDDEN;
        uint32_t* token_words = reinterpret_cast<uint32_t*>(
            staged_tokens + static_cast<uint64_t>(token) * kTokenBytes);
        uint8_t* sf_row = staged_scales + static_cast<uint64_t>(token) * kSfBytes;

        if constexpr (kEvenVecSchedule) {
            uint4 cur_bits = detail::load_bf16x8(row + tid * kVecElems);
            #pragma unroll 1
            for (uint32_t iter = 0; iter < kIters; ++iter) {
                const uint32_t vec = tid + iter * BLOCK_THREADS;
                uint4 next_bits = cur_bits;
                if (iter + 1u < kIters) {
                    next_bits = detail::load_bf16x8(row + (tid + (iter + 1u) * BLOCK_THREADS) * kVecElems);
                }
                const uint4 bits = cur_bits;
                const __ppu_bfloat162* pairs = reinterpret_cast<const __ppu_bfloat162*>(&bits);

                float vals[kVecElems];
                float local_max = 0.0f;
                #pragma unroll
                for (uint32_t i = 0; i < kVecElems / 2; ++i) {
                    const float2 f = __bfloat1622float2(pairs[i]);
                    vals[2 * i + 0] = f.x;
                    vals[2 * i + 1] = f.y;
                    local_max = fmaxf(local_max, fmaxf(fabsf(f.x), fabsf(f.y)));
                }

                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 1));
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 2));
                local_max = fmaxf(local_max, 1e-10f);

                const float dequant_scale = local_max * (1.0f / 6.0f);
                const uint32_t scale_exp = (__float_as_uint(dequant_scale) + 0x007fffffu) & 0x7f800000u;
                const float scale_f = __uint_as_float(scale_exp);
                const float inv_scale = tix::reciprocal_approx_ftz(scale_f);

                #pragma unroll
                for (uint32_t i = 0; i < kVecElems; ++i)
                    vals[i] *= inv_scale;
                token_words[vec] = tix::cvt_rte_fp4x8_f32(vals);

                if ((vec & (kThreadsPerGroup - 1u)) == 0u) {
                    sf_row[vec / kThreadsPerGroup] = static_cast<uint8_t>(scale_exp >> 23);
                }
                cur_bits = next_bits;
            }
        } else {
            for (uint32_t vec = tid; vec < kNumVecs; vec += BLOCK_THREADS) {
                const uint4 bits = detail::load_bf16x8(row + vec * kVecElems);
                const __ppu_bfloat162* pairs = reinterpret_cast<const __ppu_bfloat162*>(&bits);

                float vals[kVecElems];
                float local_max = 0.0f;
                #pragma unroll
                for (uint32_t i = 0; i < kVecElems / 2; ++i) {
                    const float2 f = __bfloat1622float2(pairs[i]);
                    vals[2 * i + 0] = f.x;
                    vals[2 * i + 1] = f.y;
                    local_max = fmaxf(local_max, fmaxf(fabsf(f.x), fabsf(f.y)));
                }

                const uint32_t mask = __activemask();
                local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, 1));
                local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, 2));
                local_max = fmaxf(local_max, 1e-10f);

                const float dequant_scale = local_max * (1.0f / 6.0f);
                const uint32_t scale_exp = (__float_as_uint(dequant_scale) + 0x007fffffu) & 0x7f800000u;
                const float scale_f = __uint_as_float(scale_exp);
                const float inv_scale = tix::reciprocal_approx_ftz(scale_f);

                #pragma unroll
                for (uint32_t i = 0; i < kVecElems; ++i)
                    vals[i] *= inv_scale;
                token_words[vec] = tix::cvt_rte_fp4x8_f32(vals);

                if ((vec & (kThreadsPerGroup - 1u)) == 0u) {
                    sf_row[vec / kThreadsPerGroup] = static_cast<uint8_t>(scale_exp >> 23);
                }
            }
        }

        for (uint32_t k = tid; k < TOP_K; k += BLOCK_THREADS) {
            const uint64_t off = static_cast<uint64_t>(token) * TOP_K + k;
            const auto expert = topk_ids[off];
            input_topk_idx[off] = expert;
            // Combine can consume this refreshed view without reading unwritten slots.
            input_topk_weights[off] = expert >= 0 ? topk_weights[off] : 0.0f;
        }
    }
}

template <uint32_t HIDDEN, uint32_t TOP_K, uint32_t BLOCK_THREADS = 256>
__global__ __launch_bounds__(BLOCK_THREADS, 4) void
fp4_mega_moe_pre_dispatch_kernel(
    const __ppu_bfloat16* __restrict__ hidden_states,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    uint8_t* __restrict__ staged_tokens,
    uint8_t* __restrict__ staged_scales,
    int64_t* __restrict__ input_topk_idx,
    float* __restrict__ input_topk_weights,
    uint32_t num_tokens) {
    fp4_mega_moe_pre_dispatch_kernel_device<HIDDEN, TOP_K, BLOCK_THREADS>(hidden_states, topk_ids, topk_weights, staged_tokens, staged_scales, input_topk_idx, input_topk_weights, num_tokens);
}

}  // namespace deep_gemm::mega_moe::fp4
