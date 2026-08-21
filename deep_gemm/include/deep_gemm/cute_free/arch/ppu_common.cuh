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

} // namespace arch
} // namespace deep_gemm
