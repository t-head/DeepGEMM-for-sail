#pragma once

namespace deep_gemm {

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
};

inline constexpr const char* GemmTypeS[] = {
    "DenseGemm", "GroupedContiguous", "GroupedMasked", "KGroupedContiguous", "BatchGemm",
    "GroupedContiguous", "KGroupedContiguous", "GroupedNoPad", "GroupedFused"
};

enum class FP4DynamicTileId : int {
    Disabled  = -1,
    LargeEM   = 0,
    LargeK    = 1,
    LargeK_G2 = 2,
    SmallEM   = 3
};

enum class KernelType {
    Default,
    MultistageOnN,
    MoeDynamicTile,
    OverlapPrologue,
    OverlapMainloop
};

inline constexpr const char* KernelTypeS[] = { "Default", "MultistageOnN", "MoeDynamicTile", "OverlapPrologue", "OverlapMainloop"};

} // namespace deep_gemm
