#pragma once

// Per-token group quantization shared by blockwise FP8 pre-dispatch.
#include <hggc_bf16.h>
#include <hggc_fp8.h>
#include <math.h>
#include <ppu_include.hpp>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm::mega_moe::fp8_blockwise {

constexpr int kGroupK = 128;
constexpr int kThreads = 256;

__device__ __forceinline__ float warp_absmax(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return fmaxf(value, 1e-10f);
}

// Four consecutive values per lane, with the same 1e-10 absmax floor and
// saturated round-to-nearest E4M3 conversion as per-token-group quantization.
__device__ __forceinline__ void quantize_group(
        const float (&values)[4], __hg_fp8_e4m3* output,
        float* scale, int lane) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        amax = fmaxf(amax, fabsf(values[i]));
    const float sf = warp_absmax(amax) / 448.0f;
    if (lane == 0)
        *scale = sf;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        output[lane * 4 + i] = __hg_fp8_e4m3(values[i] / sf);
}

}  // namespace deep_gemm::mega_moe::fp8_blockwise
