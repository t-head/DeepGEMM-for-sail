#pragma once

#include <torch/version.h>
#include <hggc.h>

// `torch::kFloat8_e4m3fn` is supported since PyTorch 2.1
// #define DG_FP8_COMPATIBLE (TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 1))

// `hgTensorMapEncodeTiled` is supported since HGGC Driver API 12.1
#define DG_TENSORMAP_COMPATIBLE (HGGC_VERSION >= 12010)