#pragma once

#include <hggc_runtime.h>
#include <deep_gemm/layout/dispatch_layout.cuh>
#include <deep_gemm/tix/ld_st.cuh>

namespace deep_gemm {

// Wait for this generation's requested stripe or a later completed stripe.
// The caller controls participating threads and joins asynchronous consumers.
__device__ __forceinline__ void wait_copy_stripe(
        const volatile uint32_t* flags, uint32_t generation, uint32_t stripe) {
  const uint32_t tagged_ready = (generation << 3) | stripe;
  constexpr uint32_t kLastStripe = 7u;
  while (tix::ld_ready_flag_uncached(flags) - tagged_ready > kLastStripe - stripe) { }
}

} // namespace deep_gemm
