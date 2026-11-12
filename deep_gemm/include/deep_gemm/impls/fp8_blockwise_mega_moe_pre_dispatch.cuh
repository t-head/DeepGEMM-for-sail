#pragma once

// Publish 1x128-quantized activations and routes to the MegaMoE input buffer.
#include <deep_gemm/impls/fp8_blockwise_moe_utils.cuh>
#include <deep_gemm/impls/mega_moe_io.cuh>

namespace deep_gemm::mega_moe::fp8_blockwise {

template <int kHidden, int kTopk, typename Index = int64_t>
CUTLASS_DEVICE void pre_dispatch_kernel_device(
        const __ppu_bfloat16* input, const Index* ids, const float* weights,
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
        output_weights[index] = ids[index] >= 0 ? weights[index] : 0.0f;
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

namespace deep_gemm::mega_moe::fp8_blockwise {

template <int Hidden, int TopK, typename TopkIndex>
__global__ __launch_bounds__(256, 1) void sm89_fp8_blockwise_mega_moe_pre_dispatch(
    __ppu_bfloat16* hidden_states, const TopkIndex* topk_ids, float* topk_weights, __hg_fp8_e4m3* input_token,
    float* input_scale, int64_t* input_topk_idx, float* input_topk_weights, int num_tokens, int capacity) {
    if (blockIdx.x >= num_tokens) {
        detail::invalidate_unused_routes<TopK>(input_topk_idx, input_topk_weights, num_tokens, capacity, num_tokens);
        return;
    }
    mega_moe::fp8_blockwise::pre_dispatch_kernel_device<Hidden, TopK>(hidden_states, topk_ids, topk_weights,
                                                                      input_token, input_scale, input_topk_idx,
                                                                      input_topk_weights, (uint32_t)num_tokens);
}

} // namespace deep_gemm::mega_moe::fp8_blockwise
