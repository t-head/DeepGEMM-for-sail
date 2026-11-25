#pragma once
#include <cstdint>
#include <cutlass/bfloat16.h>
#include <deep_gemm/cute_free/arch/ppu_common.cuh>

namespace deep_gemm::arch {
template <int CUBE_H, int CUBE_W, int Policy = 1>
__device__ __forceinline__ void aiu_bulk_tensor_load_bf16(
    cutlass::bfloat16_t* smem_dst, cutlass::bfloat16_t const* gmem_base,
    int rows, int ld, int row_offset, int col_offset) {
  static_assert(Policy == 1 || Policy == 3, "Unsupported BF16 AIU cache policy");
  const uint32_t s = smem_ptr_to_uint(smem_dst);
  asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
      :: "r"(s), "l"(gmem_base), "r"(1), "r"(rows), "r"(ld),
         "r"(0), "r"(row_offset), "r"(col_offset),
         "r"(1), "r"(CUBE_H), "r"(CUBE_W), "r"(1));
}
} // namespace deep_gemm::arch
