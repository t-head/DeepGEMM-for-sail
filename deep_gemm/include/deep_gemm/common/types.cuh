#pragma once


enum class GemmType {
    DenseGemm,
    GroupedContiguous,
    GroupedMasked,
    GroupedNoPad,
    GroupedFused,
    BatchGemm,
};

const char* GemmTypeS[] = { "DenseGemm", "GroupedContiguous", "GroupedMasked", "GroupedNoPad", "GroupedFused", "BatchGemm"};

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

const char* KernelTypeS[] = { "Default", "MultistageOnN", "MoeDynamicTile", "OverlapPrologue", "OverlapMainloop"};
