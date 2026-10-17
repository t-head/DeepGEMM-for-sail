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

} // namespace deep_gemm::tix
