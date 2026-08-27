#pragma once

#include <torch/version.h>
#include <hggc.h>

// `torch::kFloat8_e4m3fn` is supported since PyTorch 2.1
// #define DG_FP8_COMPATIBLE (TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 1))

// `hgTensorMapEncodeTiled` is supported since HGGC Driver API 12.1
#define DG_TENSORMAP_COMPATIBLE (HGGC_VERSION >= 12010)

// `acblasGetErrorString` is supported since HGGC Runtime API 11.4.2
#define DG_ACBLAS_GET_ERROR_STRING_COMPATIBLE (HGGCRT_VERSION >= 11042)

// `ACBLASLT_MATMUL_DESC_FAST_ACCUM` and `ACBLASLT_MATMUL_DESC_SM_COUNT_TARGET`
#define DG_ACBLASLT_ADVANCED_FEATURES_COMPATIBLE (HGGCRT_VERSION >= 11080)