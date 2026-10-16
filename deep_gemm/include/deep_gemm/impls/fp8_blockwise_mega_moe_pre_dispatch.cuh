#pragma once

// Publish 1x128-quantized activations and routes to the MegaMoE input buffer.
#include <deep_gemm/impls/fp8_blockwise_moe_utils.cuh>

namespace deep_gemm::mega_moe::fp8_blockwise {

template <int kHidden, int kTopk>
CUTLASS_DEVICE void pre_dispatch_kernel_device(
        const __ppu_bfloat16* input, const int64_t* ids, const float* weights,
        __hg_fp8_e4m3* output, float* scales,
        int64_t* output_ids, float* output_weights, int num_tokens) {
    static_assert(kHidden % kGroupK == 0);
    constexpr int kScales = kHidden / kGroupK;
    const int row = blockIdx.x;
    if (row >= num_tokens)
        return;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int group = warp; group < kScales; group += kThreads / 32) {
        float values[4];
        const int64_t base = int64_t(row) * kHidden + group * kGroupK;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            values[i] = __bfloat162float(input[base + lane * 4 + i]);
        quantize_group(values, output + base,
                       scales + int64_t(row) * kScales + group, lane);
    }
    for (int slot = threadIdx.x; slot < kTopk; slot += kThreads) {
        const int64_t index = int64_t(row) * kTopk + slot;
        output_ids[index] = ids[index];
        output_weights[index] = weights[index];
    }
}

template <int kHidden, int kTopk>
__global__ __launch_bounds__(kThreads) void pre_dispatch_kernel(
        const __ppu_bfloat16* input, const int64_t* ids, const float* weights,
        __hg_fp8_e4m3* output, float* scales,
        int64_t* output_ids, float* output_weights, int num_tokens) {
    pre_dispatch_kernel_device<kHidden, kTopk>(input, ids, weights, output, scales, output_ids, output_weights, num_tokens);
}

}  // namespace deep_gemm::mega_moe::fp8_blockwise
