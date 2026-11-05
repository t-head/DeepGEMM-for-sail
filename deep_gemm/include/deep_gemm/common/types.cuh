#pragma once

#include <cstdint>
#include <cutlass/detail/helper_macros.hpp>

namespace deep_gemm {

enum class MajorType { K = 0, MN = 1 };

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
    KGroupedContiguousWithPsumLayout    = 6,
    MGroupedNoPad                       = 7,
    MGroupedFused                       = 8,
    MegaMoE                             = 9,
};

constexpr CUTLASS_HOST_DEVICE bool is_m_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::MGroupedContiguous:                  return true;
        case GemmType::MGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

constexpr CUTLASS_HOST_DEVICE bool is_k_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::KGroupedContiguous:                  return true;
        case GemmType::KGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

inline constexpr const char* GemmTypeS[] = {
    "DenseGemm", "GroupedContiguous", "GroupedMasked", "KGroupedContiguous", "BatchGemm",
    "GroupedContiguous", "KGroupedContiguous", "GroupedNoPad", "GroupedFused", "MegaMoE"
};

enum class MegaMoEPhaseKind { Linear1, Linear2 };

namespace scatter_pipeline {
inline constexpr int kComputeThreads = 256;
inline constexpr int kScatterThreads = 256;
inline constexpr int kCtaThreads = kComputeThreads + kScatterThreads;
}  // namespace scatter_pipeline

struct MegaMoEReduceParams {
    const std::int64_t* ids;
    const float* weights;
    void* output;
    int tokens;
};
static_assert(sizeof(MegaMoEReduceParams) == 32);

enum class FP4DynamicTileId : int {
    Disabled  = -1,
    LargeEM   = 0,
    LargeK    = 1,
    LargeK_G2 = 2,
    SmallEM   = 3
};

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};

enum class KernelStrategy {
    Default,
    MultistageOnN,
    MoeDynamicTile,
    OverlapPrologue,
    OverlapMainloop,
};

inline constexpr const char* KernelStrategyS[] = {
    "Default", "MultistageOnN", "MoeDynamicTile", "OverlapPrologue", "OverlapMainloop"
};

} // namespace deep_gemm
