#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

namespace deep_gemm::tix {

// PPU scope/order spellings follow the SDK's
// hggc/std/detail/libcxx/include/support/atomic/atomic_hggc_generated.h.
// The memory clobbers order compiler memory accesses as well as the device
// acquire/release operations. Generic addresses also accept mapped peer memory.
CUTLASS_DEVICE uint32_t ld_acq(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

CUTLASS_DEVICE int ld_acq_sys(const int* ptr) {
    int ret;
    asm volatile("ppu.ld.acquire.sys.b32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

CUTLASS_DEVICE uint32_t ld_acq_sys(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ppu.ld.acquire.sys.b32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

CUTLASS_DEVICE uint64_t ld_acq_sys(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ppu.ld.acquire.sys.b64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory");
    return ret;
}

CUTLASS_DEVICE uint32_t atomic_add_rel(const uint32_t* ptr, uint32_t value) {
    uint32_t ret;
    asm volatile("ppu.atom.add.release.gpu.u32 %0, [%1], %2;"
                 : "=r"(ret) : "l"(ptr), "r"(value) : "memory");
    return ret;
}

CUTLASS_DEVICE uint64_t atomic_add_sys(const uint64_t* ptr, uint64_t value) {
    uint64_t ret;
    asm volatile("ppu.atom.add.sys.u64 %0, [%1], %2;"
                 : "=l"(ret) : "l"(ptr), "l"(value) : "memory");
    return ret;
}

// Use the SDK's atom forms when the caller does not need the old value too.
CUTLASS_DEVICE void red_add(const uint32_t* ptr, uint32_t value) {
    uint32_t ignored;
    asm volatile("ppu.atom.add.gpu.u32 %0, [%1], %2;"
                 : "=r"(ignored) : "l"(ptr), "r"(value) : "memory");
}

CUTLASS_DEVICE void red_add_rel(const uint32_t* ptr, uint32_t value) {
    atomic_add_rel(ptr, value);
}

CUTLASS_DEVICE void red_add_rel_sys(const int* ptr, int value) {
    uint32_t ignored;
    asm volatile("ppu.atom.add.release.sys.u32 %0, [%1], %2;"
                 : "=r"(ignored) : "l"(ptr), "r"(static_cast<uint32_t>(value)) : "memory");
}

CUTLASS_DEVICE int32_t load_volatile(const int32_t* ptr) {
    int32_t value;
    asm volatile("ppu.ld.volatile.b32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

CUTLASS_DEVICE void store_volatile(int32_t* ptr, int32_t value) {
    asm volatile("ppu.st.volatile.b32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
}

CUTLASS_DEVICE void fence_acq_rel_sys() {
    asm volatile("ppu.fence.acq_rel.sys;" ::: "memory");
}

} // namespace deep_gemm::tix
