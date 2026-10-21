#pragma once

// Layout and communication helpers shared by MegaMoE.

#include <cstdint>
#include <hggc_runtime.h>
#include <deep_gemm/common/math.cuh>

// ---- Macro fallbacks (no-op if cutlass / PPU utils already defined them) ----
#ifndef CUTLASS_HOST_DEVICE
#define CUTLASS_HOST_DEVICE __host__ __device__ __forceinline__
#endif

#ifndef CUTLASS_DEVICE
#define CUTLASS_DEVICE __device__ __forceinline__
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond)                                                          \
do {                                                                                    \
    if (not (cond)) {                                                                   \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond);  \
        asm("ppu.trap;");                                                               \
    }                                                                                   \
} while (0)
#endif

#ifndef DG_UNIFIED_ASSERT
#if defined(__HGGC_ARCH__)
#define DG_UNIFIED_ASSERT(cond) DG_DEVICE_ASSERT(cond)
#else
#define DG_UNIFIED_ASSERT(cond) ((void) 0)
#endif
#endif

namespace deep_gemm::math {

/// Pointer arithmetic helpers
template <typename dtype_t = void>
CUTLASS_HOST_DEVICE dtype_t* advance_ptr(void* ptr, const uint64_t num_bytes) {
    return reinterpret_cast<dtype_t*>(static_cast<uint8_t*>(ptr) + num_bytes);
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T, bool kDoCeilAlignment = true>
CUTLASS_HOST_DEVICE T align(T a, T b) {
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_max(T a, T b) {
    return a > b ? a : b;
}

} // namespace deep_gemm::math
