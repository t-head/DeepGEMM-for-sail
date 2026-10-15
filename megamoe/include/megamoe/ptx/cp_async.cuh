#pragma once

#include <cstdint>

namespace megamoe::ptx {

// One 16B global -> shared copy. It is not complete until the issuing thread's cp_async_wait<> for its
// group; a thread that reads back only the bytes it copied needs no block barrier.
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

}  // namespace megamoe::ptx