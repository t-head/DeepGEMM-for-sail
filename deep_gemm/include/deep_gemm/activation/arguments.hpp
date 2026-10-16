#pragma once
#include <cstdint>

namespace deep_gemm {
enum class ActivationOutputMode { BF16, FP8_CHANNEL, MXFP4, MXFP8, FP8_BLOCK };

// Each channel thread owns eight consecutive columns; wider rows use more rounds.
constexpr int activation_fp8_channel_threads(int n) {
    return n >= 8192 ? 1024 : ((n + 255) / 256) * 32;
}

// Shared host/device ABI for all activation kernels.
struct ActivationParams {
    const void* input;
    void* output;
    void* scales;
    const int32_t* masked_m;
    int rows_per_batch;
    int row_slots;
    float limit;
    float weight_inv_scale;
    float beta;
    float linear_beta;
};
} // namespace deep_gemm
