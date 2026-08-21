#pragma once

#include <cstdint>
#include "cutlass/bfloat16.h"

namespace deep_gemm {
namespace mma {

// =============================================================================
// MMA atoms.
//
// An MMA atom wraps a single warp-level matrix-multiply-accumulate instruction
// and exposes its shape / fragment layout as compile-time constants. The
// mainloop and epilogue consume these constants instead of hard-coding them,
// so a different data type or MMA shape only requires a new atom.
// =============================================================================

// -----------------------------------------------------------------------------
// Accumulator CLayout of the PPU tc02 16x16x16 MMA instruction.
//
// This is a property of the instruction SHAPE, independent of the operand data
// type: a bf16 / fp16 / fp8 MMA that issues this same 16x16x16 instruction
// shares exactly this accumulator layout. Keeping it separate from the
// data-type-specific atom (below) lets a new element type reuse it verbatim.
//
// Thread decomposition of the 16x16 output tile (32 threads/warp):
//   t_row = lane / kThreadsCol   in [0, kThreadsRow)
//   t_col = lane % kThreadsCol   in [0, kThreadsCol)
// Each thread owns kColsPerThread contiguous columns, plus two row halves
// separated by kHalfMmaM and two column halves separated by kHalfMmaN.
// -----------------------------------------------------------------------------
struct Tc02Mma16x16x16CLayout {
  static constexpr int kMmaM = 16;
  static constexpr int kMmaN = 16;
  static constexpr int kMmaK = 16;
  static constexpr int kAccumPerThread = 8;         // f32 accumulator regs per thread
  static constexpr int kThreadsRow    = 8;          // lane / 4 -> 0..7
  static constexpr int kThreadsCol    = 4;          // lane % 4 -> 0..3
  static constexpr int kColsPerThread = 2;          // contiguous output cols per thread
  static constexpr int kHalfMmaM      = kMmaM / 2;  // row half-tile offset (8)
  static constexpr int kHalfMmaN      = kMmaN / 2;  // col half-tile offset (8)
};

/// 16x16x16 bf16 MMA (PPU 1.0 tc01 / PPU 1.5 tc02).
/// 32 threads cooperate on one MMA; per thread:
///   A = 4 x uint32 (packed bf16 pairs), B = 4 x uint32, C/D = 8 x float.
struct Bf16MmaAtom {
  // Accumulator layout is shared with any element type using this instruction.
  using CLayout = Tc02Mma16x16x16CLayout;

  // Re-export shape/layout constants from the CLayout for caller convenience.
  static constexpr int kMmaM = CLayout::kMmaM;
  static constexpr int kMmaN = CLayout::kMmaN;
  static constexpr int kMmaK = CLayout::kMmaK;
  static constexpr int kAccumPerThread = CLayout::kAccumPerThread;

  // Operand fragment packing -- data-type dependent (bf16 = 16-bit elements).
  static constexpr int kFragA = 4;          // uint32 registers per thread
  static constexpr int kFragB = 4;

  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using AccumType = float;

  /// d = a * b + c   (one 16x16x16 tile)
  static __device__ __forceinline__ void mma(
      float (&d)[8],
      uint32_t const (&a)[4],
      uint32_t const (&b)[4],
      float const (&c)[8]) {
    // Register class: "=r"/"r" for accumulators (CuTe's constraint style) was
    // A/B-tested against the original "=f"/"f" on the 7x3 warp shape, where the
    // backend migrates accumulator groups between k-steps (~120 v.madl acc->acc
    // copies per k_tile). SASS came out IDENTICAL: the backend maps both
    // constraint letters to the same register class, so the migration is a
    // grouped-allocation (8 consecutive regs per tile) packing failure, not a
    // class effect. Kept as "f" to match the accumulator type.
#if __HGGC_ARCH__ == 100
    asm volatile(
        "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32  "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
            "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
              "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
              "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
#elif __HGGC_ARCH__ == 150
    asm volatile(
        "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32  "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
            "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
              "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
              "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
#endif
  }
};

} // namespace mma
} // namespace deep_gemm
