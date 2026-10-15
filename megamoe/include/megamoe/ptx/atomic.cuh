#pragma once

#include <cuda_bf16.h>

#include <megamoe/common/exception.cuh>
#include <cuda/std/cstdint>

namespace megamoe::ptx {

// Atomic and ordered memory operations (atom/red, acquire/release/volatile
// loads and stores) -- the primitives behind cross-SM / cross-rank sync.

// Completion tag for the fire-and-forget release reductions: the high bit flips once every
// unit contribution has landed (the dispatch handoff and grid_sync users).
constexpr uint32_t kFinishSumTag = 0x80000000u;

DG_DEVICE uint64_t ld_volatile(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

DG_DEVICE uint32_t ld_volatile(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

DG_DEVICE void st_volatile(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("st.volatile.global.b32 [%0], %1;" ::"l"(ptr), "r"(value));
}

DG_DEVICE uint32_t atomic_add(const uint32_t* ptr, const uint32_t& value) {
    uint32_t ret;
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

DG_DEVICE uint32_t atomic_add_rel(const uint32_t* ptr, const uint32_t& value) {
    uint32_t ret;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

DG_DEVICE uint32_t atomic_exch_rel(const uint32_t* ptr, const uint32_t& value) {
    uint32_t ret;
    asm volatile("atom.release.gpu.global.exch.b32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

DG_DEVICE void red_add(const int* ptr, const int& value) {
    asm volatile("red.gpu.global.add.s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

DG_DEVICE void red_add(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(value));
}

DG_DEVICE void red_add_rel(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(value));
}

DG_DEVICE void red_add_rel_sys(const int* ptr, const int& value) {
    asm volatile("red.release.sys.global.add.s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

DG_DEVICE uint32_t ld_acq_sys(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

DG_DEVICE uint32_t ld_acq_gpu(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

}  // namespace megamoe::ptx
