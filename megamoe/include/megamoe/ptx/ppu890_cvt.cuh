#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace megamoe::ptx::ppu890 {

// FP4 (E2M1) conversion wrapper for the W4A4 requantization path.

__device__ __forceinline__ uint8_t cvt_f4x2_f32x2(float hi, float lo) {
    uint16_t packed;
    asm volatile("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;\n" : "=h"(packed) : "f"(hi), "f"(lo));
    return static_cast<uint8_t>(packed);
}

}  // namespace megamoe::ptx::ppu890