#pragma once


enum class GemmType {
    DenseGemm,
    GroupedContiguous,
    GroupedMasked,
    GroupedNoPad,
    GroupedFused,
};

const char* GemmTypeS[] = { "DenseGemm", "GroupedContiguous", "GroupedMasked", "GroupedNoPad", "GroupedFused"};

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
