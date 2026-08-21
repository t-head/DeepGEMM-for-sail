#pragma once

#include <cstdint>
#include "cutlass/bfloat16.h"

namespace deep_gemm {
namespace epilogue {

// =============================================================================
// Element traits for the epilogue (accumulator -> global memory conversion).
//
// Maps an OUTPUT data type to its storage width, packed register width, vector
// width, and the f32->Element conversion/pack instruction. This is the axis
// that distinguishes a bf16 epilogue from an fp16 one: the accumulator CLayout
// is shared (see cute_free/mma/mma_atom.cuh), while the output conversion lives here.
// Adding a new output type only requires a new specialization.
// =============================================================================

template <typename Element>
struct EpilogueElementTraits;

template <>
struct EpilogueElementTraits<cutlass::bfloat16_t> {
  using StorageType = uint16_t;        // in-memory element width
  using PackedType  = uint32_t;        // one cvt produces this many bits
  static constexpr int kVecWidth = 2;  // elements packed per cvt instruction

  /// Convert two f32 accumulators into one packed bf16x2 register.
  /// The instruction packs {hi, lo} into the high/low half respectively.
  static __device__ __forceinline__ PackedType cvt_pack(float hi, float lo) {
    PackedType packed;
    asm("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n"
        : "=r"(packed) : "f"(hi), "f"(lo));
    return packed;
  }
};

} // namespace epilogue
} // namespace deep_gemm
