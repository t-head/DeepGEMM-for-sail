#pragma once

#include <cstdint>
#include <hggc_bf16.h>
#include <hggc_runtime.h>

namespace deep_gemm::mega_moe::detail {

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
