#pragma once

#include <cutlass/cutlass.h>
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <cstdint>

namespace deep_gemm::mega_moe::w4a16 {

// Stage BF16 activations and routing metadata in one launch. Input and output
// views may alias when the upstream producer writes directly to the buffer.
// The launcher omits activation/routing CTAs for inputs that are already staged.
template <int kHidden, int kTopk>
CUTLASS_DEVICE void pre_dispatch_kernel_device(
    const __ppu_bfloat16* hidden_states, const int64_t* topk_ids, const float* topk_weights,
    __ppu_bfloat16* staged_tokens, int64_t* staged_ids, float* staged_weights,
    int num_tokens, int token_blocks, int vector_size) {
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
            if (topk_ids != staged_ids)
                staged_ids[route] = topk_ids[route];
            if (topk_weights != staged_weights)
                staged_weights[route] = topk_weights[route];
        }
    }
}

}  // namespace deep_gemm::mega_moe::w4a16
