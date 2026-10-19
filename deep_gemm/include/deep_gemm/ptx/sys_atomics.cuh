#pragma once

// System-scope (.sys) and GPU-scope PTX primitives used by the mega-MoE
// cross-rank "lock" mechanism. All instructions below have been verified to
// compile AND run correctly on PPU (ZW-M890P, cc 8.9), see
// tests/test_ppu_sys_scope.py.
#include <deep_gemm/layout/mega_moe_common.cuh>

namespace deep_gemm::ptx {

// ---- acquire loads ----
CUTLASS_DEVICE uint32_t ld_acq(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE int ld_acq_sys(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint32_t ld_acq_sys(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint64_t ld_acq_sys(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint64_t ld_acq_gpu(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

// ---- atomic add (returns old value) ----
CUTLASS_DEVICE uint32_t atomic_add_rel(const uint32_t* ptr, const uint32_t& value) {
    uint32_t ret;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

CUTLASS_DEVICE uint64_t atomic_add_sys(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.sys.global.add.u64 %0, [%1], %2;"
                 : "=l"(ret)
                 : "l"(ptr), "l"(value));
    return ret;
}

// ---- reduction (no return value) ----
CUTLASS_DEVICE void red_add(const int* ptr, const int& value) {
    asm volatile("red.gpu.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_add(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_add_rel(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_add_rel_sys(const int* ptr, const int& value) {
    asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_or_rel_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.sys.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

CUTLASS_DEVICE void red_or_rel_gpu(uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.gpu.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

} // namespace deep_gemm::ptx
