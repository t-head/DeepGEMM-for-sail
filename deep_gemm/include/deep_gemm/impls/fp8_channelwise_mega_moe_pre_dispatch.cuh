#pragma once

#include <cutlass/cutlass.h>
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <cstdint>
#include <math.h>

namespace deep_gemm::mega_moe::fp8_channelwise {

// BF16 -> E4M3 row quantization and local token-quad staging for the FP8
// MegaMoE path. The quantization contract intentionally matches the existing
// per-token FP8 operator: per-token absmax / 448.f followed by
// cvt.rn.satfinite.e4m3x2.f32. Writing router metadata from the same CTA
// removes temporary FP8/scale tensors and the four follow-up copies.
template <uint32_t HIDDEN, uint32_t TOP_K, uint32_t BLOCK_THREADS = 256>
CUTLASS_DEVICE void fp8_mega_moe_pre_dispatch_kernel_device(
        const __ppu_bfloat16* __restrict__ hidden_states,
        const int64_t* __restrict__ topk_ids,
        const float* __restrict__ topk_weights,
        uint8_t* __restrict__ staged_tokens,
        float* __restrict__ staged_scales,
        int64_t* __restrict__ input_topk_idx,
        float* __restrict__ input_topk_weights,
        uint32_t num_tokens) {
    static_assert(HIDDEN % 8 == 0,
                  "FP8 pre-dispatch vectorizes eight BF16 values at a time");
    static_assert(BLOCK_THREADS % 32 == 0,
                  "FP8 pre-dispatch CTA must contain whole warps");

    constexpr uint32_t kVecElems = 8;
    constexpr uint32_t kNumVecs = HIDDEN / kVecElems;
    constexpr float kFp8E4m3Max = 448.0f;

    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (token >= num_tokens)
        return;

    const __ppu_bfloat16* row =
        hidden_states + static_cast<uint64_t>(token) * HIDDEN;
    const uint4* row_vec = reinterpret_cast<const uint4*>(row);

    float local_max = 0.0f;
    for (uint32_t vec = tid; vec < kNumVecs; vec += BLOCK_THREADS) {
        const uint4 bits = row_vec[vec];
        const __ppu_bfloat162* pairs =
            reinterpret_cast<const __ppu_bfloat162*>(&bits);
#pragma unroll
        for (uint32_t i = 0; i < kVecElems / 2; ++i) {
            const float2 values = __bfloat1622float2(pairs[i]);
            local_max = fmaxf(local_max, fabsf(values.x));
            local_max = fmaxf(local_max, fabsf(values.y));
        }
    }

    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    __shared__ alignas(16) float warp_max[BLOCK_THREADS / 32];
    __shared__ alignas(16) float shared_scale;

    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 2));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 4));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 8));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 16));
    if (lane == 0)
        warp_max[warp] = local_max;
    __syncthreads();

    if (warp == 0) {
        float row_max = lane < (BLOCK_THREADS / 32) ? warp_max[lane] : 0.0f;
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 1));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 2));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 4));
        if (lane == 0) {
            const float scale = row_max / kFp8E4m3Max;
            shared_scale = scale;
            staged_scales[token] = scale;
        }
    }
    __syncthreads();
    const float scale_inv = shared_scale == 0.0f ? 0.0f : 1.0f / shared_scale;

    uint8_t* output_row =
        staged_tokens + static_cast<uint64_t>(token) * HIDDEN;
    uint16_t* output_packed = reinterpret_cast<uint16_t*>(output_row);
    for (uint32_t vec = tid; vec < kNumVecs; vec += BLOCK_THREADS) {
        const uint4 bits = row_vec[vec];
        const __ppu_bfloat162* pairs =
            reinterpret_cast<const __ppu_bfloat162*>(&bits);
#pragma unroll
        for (uint32_t i = 0; i < kVecElems / 2; ++i) {
            const float2 values = __bfloat1622float2(pairs[i]);
            const float q0 = values.x * scale_inv;
            const float q1 = values.y * scale_inv;
            uint16_t packed;
            asm volatile(
                "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                : "=h"(packed) : "f"(q1), "f"(q0));
            output_packed[vec * (kVecElems / 2) + i] = packed;
        }
    }

    if (tid < TOP_K) {
        const uint64_t off = static_cast<uint64_t>(token) * TOP_K + tid;
        input_topk_idx[off] = topk_ids[off];
        input_topk_weights[off] = topk_weights[off];
    }
}

template <uint32_t HIDDEN, uint32_t TOP_K, uint32_t BLOCK_THREADS = 256>
__global__ __launch_bounds__(BLOCK_THREADS) void
fp8_mega_moe_pre_dispatch_kernel(
        const __ppu_bfloat16* __restrict__ hidden_states,
        const int64_t* __restrict__ topk_ids,
        const float* __restrict__ topk_weights,
        uint8_t* __restrict__ staged_tokens,
        float* __restrict__ staged_scales,
        int64_t* __restrict__ input_topk_idx,
        float* __restrict__ input_topk_weights,
        uint32_t num_tokens) {
    fp8_mega_moe_pre_dispatch_kernel_device<HIDDEN, TOP_K, BLOCK_THREADS>(hidden_states, topk_ids, topk_weights, staged_tokens, staged_scales, input_topk_idx, input_topk_weights, num_tokens);
}

}  // namespace deep_gemm::mega_moe::fp8_channelwise
