#pragma once

#include <cstdint>

namespace deep_gemm {
namespace arch {

/// Convert shared memory pointer to uint32_t for AIU asm instructions.
/// Used by G2S (AIU bulk tensor copy) to encode the shared destination address.
__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {
  uint32_t addr;
  asm("{ .reg .u64 smem_ptr; ppu.cvta.to.shared.u64 smem_ptr, %1; ppu.cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(addr) : "l"(ptr));
  return addr;
}

/// Convert shared memory pointer to tsm_add base (divide by 16) + coord offset,
/// all in scalar registers. Uses "=s" output and "s" input for coord_offset
/// to force the entire chain into scalar ALU (IALU reduction).
__device__ __forceinline__ int smem_ptr_to_tsm_base(void const* ptr, int coord_offset) {
  int tsm_base;
  asm volatile(
      "{ .reg .u64 sp; .reg .u32 u;\n"
      "  ppu.cvta.to.shared.u64 sp, %1;\n"
      "  ppu.cvt.u32.u64 u, sp;\n"
      "  ppu.shr.u32 u, u, 4;\n"
      "  ppu.add.u32 %0, u, %2;\n"
      "}\n"
      : "=s"(tsm_base) : "l"(ptr), "s"(coord_offset));
  return tsm_base;
}

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
// PPU0010 ldmatrix needs the shared-memory base, row, and K sub-tile separately.
struct TsmCoord {
  int row;
  int k_cube;
};
struct TsmBase {
  uintptr_t ptr;
  int coord;
  int k_cube;
  __device__ __forceinline__ TsmBase operator+(int x) const { return {ptr, coord + x, k_cube}; }
};
__device__ __forceinline__ TsmBase make_tsm_base(void const* ptr, TsmCoord coord) {
  return {reinterpret_cast<uintptr_t>(ptr), coord.row, coord.k_cube};
}
#else
using TsmCoord = int;
using TsmBase = int;

__device__ __forceinline__ TsmBase make_tsm_base(void const* ptr, TsmCoord coord) {
  return smem_ptr_to_tsm_base(ptr, coord);
}
#endif

} // namespace arch
} // namespace deep_gemm
