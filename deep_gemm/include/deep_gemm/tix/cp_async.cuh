#pragma once

#include <cstdint>

namespace deep_gemm::tix {

// HGGC cp.async (async global -> shared copy) primitives, lifted verbatim from
// impls/gemvt.cuh. Shared by gemvt.cuh and blockwise_gemvt.cuh.

__device__ __forceinline__
uint32_t SmemU32Addr(const void *smemptr) {
    uint32_t u32addr;
    asm (
        "{.reg .u64 u64addr;\n"
        " ppu.cvta.to.shared.u64 u64addr, %1;\n"
        " ppu.cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(u32addr)
        : "l"(smemptr)
    );
    return u32addr;
}

__device__ __forceinline__
void LdgSts128(const void* smemPtr,
               const void *gmemPtr,
               bool guard = true) {
    auto smemAddr = SmemU32Addr(smemPtr);
    asm volatile (
        "{.reg.pred p;\n"
        " ppu.cmpp.ne.b32 p, %2, 0;\n"
        " @p ppu.cp.async.cg.shared.global [%0], [%1], 16;}\n"
        :
        : "r"(smemAddr), "l"(gmemPtr), "r"((int)guard)
    );
}

__device__ __forceinline__
void LdgStsGroupCommit() {
    asm volatile ("ppu.cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__
void LdgStsGroupWait() {
    asm volatile ("ppu.cp.async.wait_group %0;\n" : : "n"(N));
}

} // namespace deep_gemm::tix
