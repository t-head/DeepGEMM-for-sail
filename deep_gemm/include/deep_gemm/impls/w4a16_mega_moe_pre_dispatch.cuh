#pragma once

#include <cutlass/cutlass.h>
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <deep_gemm/impls/mega_moe_io.cuh>
#include <cstdint>

namespace deep_gemm::mega_moe::w4a16 {

// Stage BF16 activations and routing metadata in one launch. Input and output
// views may alias when the upstream producer writes directly to the buffer.
// The launcher omits token-copy CTAs when activations are already staged.
template <int kHidden, int kTopk, typename Index = int64_t>
CUTLASS_DEVICE void pre_dispatch_kernel_device(
    const __ppu_bfloat16* hidden_states, const Index* topk_ids, const float* topk_weights,
    __ppu_bfloat16* staged_tokens, int64_t* staged_ids, float* staged_weights,
    int num_tokens, int capacity, int token_blocks, int vector_size) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (blockIdx.x < token_blocks) {
        if (index * vector_size < int64_t(num_tokens) * kHidden) {
            if (vector_size == 8)
                reinterpret_cast<uint4*>(staged_tokens)[index] = reinterpret_cast<const uint4*>(hidden_states)[index];
            else
                staged_tokens[index] = hidden_states[index];
        }
    } else {
        const int64_t route = index - int64_t(token_blocks) * blockDim.x;
        if (route < int64_t(num_tokens) * kTopk) {
            if (static_cast<const void*>(topk_ids) != static_cast<const void*>(staged_ids))
                staged_ids[route] = topk_ids[route];
            staged_weights[route] = topk_ids[route] >= 0 ? topk_weights[route] : 0.0f;
        }
        detail::invalidate_unused_routes<kTopk>(staged_ids, staged_weights, num_tokens, capacity, token_blocks);
    }
}

}  // namespace deep_gemm::mega_moe::w4a16

namespace deep_gemm::mega_moe::w4a16 {

template <int Hidden, int TopK, int Threads, typename TopkIndex>
__global__ __launch_bounds__(Threads) void sm89_bf16_fp4_mega_moe_pre_dispatch(
    const __ppu_bfloat16* hidden_states, const TopkIndex* topk_ids, const float* topk_weights,
    __ppu_bfloat16* staged_tokens, int64_t* staged_topk_idx, float* staged_topk_weights, int num_tokens,
    int capacity, int token_blocks, int vector_size) {
    deep_gemm::mega_moe::w4a16::pre_dispatch_kernel_device<Hidden, TopK>(
        hidden_states, topk_ids, topk_weights, staged_tokens, staged_topk_idx, staged_topk_weights, num_tokens,
        capacity, token_blocks, vector_size);
}

} // namespace deep_gemm::mega_moe::w4a16
