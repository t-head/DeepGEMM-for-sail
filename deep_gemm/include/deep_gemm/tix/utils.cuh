#pragma once

#if defined(__HGGC__)

namespace deep_gemm::tix {

__device__ __forceinline__ int atomic_add_release_global(int* addr, int value) {
    int ret;
    asm volatile ("ppu.atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(value));
    return ret;
}

} // namespace deep_gemm::tix

#endif  // __HGGC__
