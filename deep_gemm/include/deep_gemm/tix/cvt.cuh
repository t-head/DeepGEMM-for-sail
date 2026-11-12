#pragma once

#include <cstdint>

namespace deep_gemm::tix {

// Single bf16x2 convert of two f32 sources (r = {hi, lo}); the .relu variant
// clamps negatives to zero at convert time (how the avg epilogues apply ReLU).
// Lifted verbatim from impls/mqa_logits_utils.cuh.

__forceinline__ __device__
uint32_t cvt_rte_bf16x2_f32(float hi, float lo) {
    uint32_t r;
    asm volatile("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__forceinline__ __device__
uint32_t cvt_rte_bf16x2_f32_relu(float hi, float lo) {
    uint32_t r;
    asm volatile("ppu.cvt.rtte.bf16x2.f32.relu %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// Shared with the ordinary FP4 post-quantization epilogue. Low input occupies
// the low nibble; saturation and round-to-nearest-even match the GEMM format.
__forceinline__ __device__
uint8_t cvt_rte_fp4x2_f32(float hi, float lo) {
    uint8_t packed;
    asm("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;\n"
        : "=c"(packed) : "f"(hi), "f"(lo));
    return packed;
}

__forceinline__ __device__
uint32_t cvt_rte_fp4x8_f32(const float* values) {
    const uint32_t b0 = cvt_rte_fp4x2_f32(values[1], values[0]);
    const uint32_t b1 = cvt_rte_fp4x2_f32(values[3], values[2]);
    const uint32_t b2 = cvt_rte_fp4x2_f32(values[5], values[4]);
    const uint32_t b3 = cvt_rte_fp4x2_f32(values[7], values[6]);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

} // namespace deep_gemm::tix
