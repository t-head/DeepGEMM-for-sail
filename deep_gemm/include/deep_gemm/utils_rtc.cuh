#pragma once

#ifdef __CLION_IDE__
__host__ __device__ __forceinline__ void host_device_printf(const char* format, ...) { asm volatile("trap;"); }
#define printf host_device_printf
#endif

__device__ __forceinline__ int atomic_add_release_global(int* addr, int value) {
    int ret;
    asm volatile ("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(value));
    return ret;
}

enum class GemmType {
    DenseGemm,
    GroupedContiguous,
    GroupedMasked,
    GroupedNoPad,
    GroupedFused,
    BatchGemm,
    FusedDispatch,
};

const char* GemmTypeS[] = { "DenseGemm", "GroupedContiguous", "GroupedMasked", "GroupedNoPad", "GroupedFused", "BatchGemm", "FusedDispatch"};

enum class KernelType {
    Default,
    MultistageOnN,
    MoeDynamicTile,
    OverlapPrologue,
    OverlapMainloop
};

const char* KernelTypeS[] = { "Default", "MultistageOnN", "MoeDynamicTile", "OverlapPrologue", "OverlapMainloop"};


template <typename T>
__device__ __host__ constexpr inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}
