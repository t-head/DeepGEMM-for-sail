#pragma once

#include <cstdint>
#include <hggc_bf16.h>
#include <hggc_runtime.h>

#ifndef DG_MOE_FAST_MATH
#define DG_MOE_FAST_MATH 1
#endif

namespace deep_gemm::mega_moe::detail {

template <int TopK>
__device__ __forceinline__ void invalidate_unused_routes(
    int64_t* ids, float* weights, int num_tokens, int capacity, int first_block) {
    const int start = num_tokens * TopK + (int(blockIdx.x) - first_block) * blockDim.x + threadIdx.x;
    const int stride = (int(gridDim.x) - first_block) * blockDim.x;
    for (int slot = start; slot < capacity * TopK; slot += stride) {
        ids[slot] = -1;
        weights[slot] = 0.0f;
    }
}

// Contiguous BF16 views can have a storage offset that is not 16-byte aligned.
// Keep vector accesses for aligned views; scalar accesses preserve the same bits.
__device__ __forceinline__ uint4 load_bf16x8(const __ppu_bfloat16* ptr) {
    if ((reinterpret_cast<uintptr_t>(ptr) & 15u) == 0)
        return *reinterpret_cast<const uint4*>(ptr);
    uint4 packed;
    auto* values = reinterpret_cast<__ppu_bfloat16*>(&packed);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        values[i] = ptr[i];
    return packed;
}

__device__ __forceinline__ void store_bf16x8(__ppu_bfloat16* ptr, const float4& packed) {
    if ((reinterpret_cast<uintptr_t>(ptr) & 15u) == 0) {
        *reinterpret_cast<float4*>(ptr) = packed;
    } else {
        const auto* values = reinterpret_cast<const __ppu_bfloat16*>(&packed);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            ptr[i] = values[i];
    }
}

} // namespace deep_gemm::mega_moe::detail
