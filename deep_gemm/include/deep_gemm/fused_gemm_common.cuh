#pragma once
#include <cstdint>

#include "cute/int_tuple.hpp"

// Shared host/device definitions for the fused MoE GEMM kernels and the C++
// JIT host runtimes: the kernel argument structs and the shared-memory size
// formulas live here once, so the host side cannot drift from the device side.
// The smem formulas are CUTE_HOST_DEVICE constexpr functions — the same
// annotation cute::round_up / cute::ceil_div use — so both the device kernels
// and the host runtimes holding runtime tile sizes call them directly.
// Requires the actlize include dirs, which both the setup.py host build and
// the JIT device compiler provide.

namespace deep_gemm {

struct GemmArgs {
  const void *__restrict__ a_ptr;
  const void *__restrict__ b_ptr;
  void *__restrict__ c_ptr;

  const int *__restrict__ expert_ids_and_cumsum;
  const int *__restrict__ sorted_token_ids;
  const int *__restrict__ aligned_num_m_blocks;
  uint32_t shape_m;
};

struct QuantGemmArgs : public GemmArgs{
  const void *__restrict__ scale_a_ptr;
  const void *__restrict__ scale_b_ptr;
};

// tsm.ld.swzl need 128B aligned
CUTE_HOST_DEVICE constexpr uint32_t smem_a_size(uint32_t elem_size, uint32_t num_stages,
                                                uint32_t block_m, uint32_t block_k) {
    return cute::round_up(num_stages * block_m * block_k * elem_size, 128);
}

CUTE_HOST_DEVICE constexpr uint32_t smem_b_size(uint32_t elem_size, uint32_t num_stages,
                                                uint32_t block_n, uint32_t block_k) {
    return cute::round_up(num_stages * block_n * block_k * elem_size, 128);
}

CUTE_HOST_DEVICE constexpr uint32_t gemm_smem_total_size(uint32_t elem_size, uint32_t num_stages,
                                                         uint32_t block_m, uint32_t block_n,
                                                         uint32_t block_k) {
    return smem_a_size(elem_size, num_stages, block_m, block_k) +
           smem_b_size(elem_size, num_stages, block_n, block_k);
}

// Blockwise-quantization scale segments; note scale_b aligns to 256B, not 128B.
CUTE_HOST_DEVICE constexpr uint32_t smem_scale_a_size(uint32_t num_stages, uint32_t block_m,
                                                      uint32_t block_k) {
    return cute::round_up(num_stages * block_m * cute::ceil_div(block_k, 128) * sizeof(float), 128);
}

CUTE_HOST_DEVICE constexpr uint32_t smem_scale_b_size(uint32_t num_stages, uint32_t block_n,
                                                      uint32_t block_k) {
    return cute::round_up(num_stages * cute::ceil_div(block_n, 128) * cute::ceil_div(block_k, 128) * sizeof(float), 256);
}

CUTE_HOST_DEVICE constexpr uint32_t blkwise_smem_total_size(uint32_t elem_size, uint32_t num_stages,
                                                            uint32_t block_m, uint32_t block_n,
                                                            uint32_t block_k) {
    return gemm_smem_total_size(elem_size, num_stages, block_m, block_n, block_k) +
           smem_scale_a_size(num_stages, block_m, block_k) +
           smem_scale_b_size(num_stages, block_n, block_k);
}

} // namespace deep_gemm
