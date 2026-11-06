#pragma once

#include <hggc_runtime.h>
#include <deep_gemm/layout/dispatch_layout.cuh>

namespace deep_gemm {

__device__ __forceinline__ uint32_t ld_copy_ready_uncached(
        const volatile uint32_t* ptr) {
  asm volatile("" ::: "memory");
  return __ldbl(const_cast<const uint32_t*>(ptr));
}

// Wait for this generation's requested stripe or a later completed stripe.
// The caller controls participating threads and joins asynchronous consumers.
__device__ __forceinline__ void wait_copy_stripe(
        const volatile uint32_t* flags, uint32_t generation, uint32_t stripe) {
  const uint32_t tagged_ready = (generation << 3) | stripe;
  constexpr uint32_t kLastStripe = 7u;
  while (ld_copy_ready_uncached(flags) - tagged_ready > kLastStripe - stripe) { }
}

} // namespace deep_gemm
