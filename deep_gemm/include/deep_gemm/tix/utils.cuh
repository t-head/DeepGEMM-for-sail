#pragma once

#if defined(__HGGC__)

namespace deep_gemm::tix {

// Approximate reciprocal with flush-to-zero semantics for MXFP4 scales.
__device__ __forceinline__ float reciprocal_approx_ftz(float value) {
    float result;
    asm("ppu.rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ int atomic_add_release_global(int* addr, int value) {
    int ret;
    asm volatile ("ppu.atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(value));
    return ret;
}

} // namespace deep_gemm::tix

#endif  // __HGGC__
