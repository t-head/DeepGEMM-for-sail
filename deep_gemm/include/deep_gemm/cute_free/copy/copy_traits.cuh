#pragma once

#include "cutlass/bfloat16.h"

namespace deep_gemm {
namespace copy {

// =============================================================================
// Element traits for copy atoms.
//
// Maps a data type to the instruction-level metadata required by the PPU copy
// primitives (G2S / S2R). Adding a new data type (e.g. fp16, fp8) only requires
// a new specialization here plus the matching arch-level instruction wrapper.
// =============================================================================

template <typename Element>
struct CopyElementTraits;

template <>
struct CopyElementTraits<cutlass::bfloat16_t> {
  static constexpr int kSize = 2;            // bytes per element
  static constexpr const char* kSuffix = "b16";  // aiu bulk tensor / ldmatrix suffix
};

} // namespace copy
} // namespace deep_gemm
