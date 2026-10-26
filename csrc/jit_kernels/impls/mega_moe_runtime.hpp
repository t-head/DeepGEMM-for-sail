#pragma once

#include <map>
#include <mutex>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"

namespace deep_gemm {

inline int get_kernel_occupancy(const std::shared_ptr<KernelRuntime>& runtime,
                                int threads, int dynamic_smem = 0) {
    // Occupancy must use the same dynamic shared-memory limit as launch.
    construct_launch_config(runtime->kernel, current_stream(), dynamic_smem, dim3(1), dim3(threads));
    int blocks = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_UNIFIED_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, runtime->kernel, threads, dynamic_smem));
#else
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, runtime->kernel, threads, dynamic_smem));
#endif
    return blocks;
}

// Shared storage depends on the instantiated device collectives. Read it once
// per loaded kernel, before graph capture, without extending KernelRuntime.
inline int get_kernel_smem_size(const std::shared_ptr<KernelRuntime>& runtime) {
    using Key = std::weak_ptr<KernelRuntime>;
    static std::map<Key, int, std::owner_less<Key>> sizes;
    static std::mutex mutex;
    const std::lock_guard<std::mutex> guard(mutex);
    const auto found = sizes.find(Key(runtime));
    if (found != sizes.end())
        return found->second;
    size_t bytes = 0;
    void* address = nullptr;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_UNIFIED_CHECK(hggcLibraryGetGlobal(&address, &bytes, runtime->library,
                                      "ppu_mega_moe_smem_bytes"));
#else
    HGdeviceptr device_address = 0;
#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
    DG_HGGC_UNIFIED_CHECK(hgLibraryGetGlobal(&device_address, &bytes, runtime->library,
                                    "ppu_mega_moe_smem_bytes"));
#else
    DG_HGGC_UNIFIED_CHECK(hgModuleGetGlobal(&device_address, &bytes, runtime->library,
                                   "ppu_mega_moe_smem_bytes"));
#endif
    address = reinterpret_cast<void*>(device_address);
#endif
    DG_HOST_ASSERT(bytes == sizeof(uint32_t));
    uint32_t smem = 0;
    DG_HGGC_RUNTIME_CHECK(hggcMemcpy(&smem, address, sizeof(smem), hggcMemcpyDeviceToHost));
    DG_HOST_ASSERT(smem <= 262144);
    return sizes.emplace(Key(runtime), static_cast<int>(smem)).first->second;
}

} // namespace deep_gemm
