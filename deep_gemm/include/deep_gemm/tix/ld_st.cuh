#pragma once

#include <cstdint>
#include <hggc_bf16.h>
#include <hggc_runtime.h>

// HGGC inline-asm shared-memory load primitives, lifted verbatim
// from impls/mqa_logits_utils.cuh.

namespace deep_gemm::tix {

__device__ __forceinline__ uint32_t ld_ready_flag_uncached(const volatile uint32_t* ptr) {
    asm volatile("" ::: "memory");
    return __ldbl(const_cast<const uint32_t*>(ptr));
}

// A contiguous BF16 view may start at an offset of just one element.
template <bool kAligned>
__device__ __forceinline__ int4 load_bf16x8(const __ppu_bfloat16* ptr) {
    if constexpr (kAligned)
        return *reinterpret_cast<const int4*>(ptr);
    int4 packed;
    auto* lanes = reinterpret_cast<__ppu_bfloat16*>(&packed);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        lanes[i] = ptr[i];
    return packed;
}

template <bool kAligned>
__device__ __forceinline__ void store_bf16x8(__ppu_bfloat16* ptr, const int4& packed) {
    if constexpr (kAligned) {
        *reinterpret_cast<int4*>(ptr) = packed;
        return;
    }
    const auto* lanes = reinterpret_cast<const __ppu_bfloat16*>(&packed);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        ptr[i] = lanes[i];
}

__device__ __forceinline__ float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ppu.ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __inline__ uint32_t ld_shared_u32(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ppu.ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void red_add_rel_sys(
        int32_t* address, int32_t value) {
    asm volatile("ppu.atom.add.release.sys.global.s32 _, [%0], %1;"
                 :: "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t ld_acq(
        const uint32_t* address) {
    uint32_t value;
    asm volatile("ppu.ld.acquire.gpu.global.b32 %0, [%1];"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ int32_t ld_acq_sys(
        const int32_t* address) {
    int32_t value;
    asm volatile("ppu.ld.acquire.sys.global.s32 %0, [%1];"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ uint32_t atomic_add_rel(
        uint32_t* address, uint32_t value) {
    uint32_t old_value;
    asm volatile("ppu.atom.add.release.gpu.global.u32 %0, [%1], %2;"
                 : "=r"(old_value) : "l"(address), "r"(value) : "memory");
    return old_value;
}

__device__ __forceinline__ void red_add(
        int32_t* address, int32_t value) {
    asm volatile("ppu.atom.add.relaxed.gpu.global.s32 _, [%0], %1;"
                 :: "l"(address), "r"(value) : "memory");
}

} // namespace deep_gemm::tix
