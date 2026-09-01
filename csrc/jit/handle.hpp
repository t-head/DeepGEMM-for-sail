#pragma once

#include <hggc.h>
#include <hggc_runtime.h>
#include <dlfcn.h>
#include <filesystem>

#include "../utils/exception.hpp"
#include "../utils/compatibility.hpp"

namespace deep_gemm {

// Lazy loading all driver symbols
static void* get_driver_handle() {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = dlopen("libhggc.so", RTLD_LAZY | RTLD_LOCAL);
        DG_HOST_ASSERT(handle != nullptr and "Failed to load HGGC driver `libhggc.so`");
    }
    return handle;
}

// Macro to define wrapper functions named `lazy_hg{API name}`
#define DECL_LAZY_HGGC_DRIVER_FUNCTION(name)                                                                           \
    template <typename... Args>                                                                                        \
    static auto lazy_##name(Args&&... args)->decltype(name(args...)) {                                                 \
        using FuncType = decltype(&name);                                                                              \
        static FuncType func = nullptr;                                                                                \
        if (func == nullptr) {                                                                                         \
            func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), #name));                                      \
            DG_HOST_ASSERT(func != nullptr and "Failed to load HGGC driver API");                                      \
        }                                                                                                              \
        return func(std::forward<decltype(args)>(args)...);                                                            \
    }

DECL_LAZY_HGGC_DRIVER_FUNCTION(hgGetErrorName);
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgGetErrorString);
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgFuncSetAttribute);
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgModuleLoad);
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgModuleUnload);
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgModuleGetFunction);
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgLibraryLoadFromFile);
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgLibraryUnload);
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgKernelGetFunction);
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgLaunchKernelEx);

#if DG_TENSORMAP_COMPATIBLE
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgTensorMapEncodeTiled);
#endif

#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)

// Use HGGC runtime API
using LibraryHandle = hggcLibrary_t;
using KernelHandle = hggcKernel_t;
using LaunchConfigHandle = hggcLaunchConfig_t;
using LaunchAttrHandle = hggcLaunchAttribute;

#define DG_HGGC_CHECK DG_HGGC_RUNTIME_CHECK

static KernelHandle load_kernel(const std::filesystem::path& hgbin_path, const std::string& func_name,
                                LibraryHandle* library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel{};
    DG_HGGC_RUNTIME_CHECK(
        hggcLibraryLoadFromFile(&library, hgbin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    DG_HGGC_RUNTIME_CHECK(hggcLibraryGetKernel(&kernel, library, func_name.c_str()));

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
    const auto error = hggcLibraryUnload(library);
    DG_HOST_ASSERT(error == hggcSuccess or error == hggcErrorHggcrtUnloading);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel, const hggcStream_t& stream,
                                                  const int& smem_size, const dim3& grid_dim, const dim3& block_dim) {
    if (smem_size > 0)
        DG_HGGC_RUNTIME_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    LaunchConfigHandle config;
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    config.numAttrs = 0;
    config.attrs = nullptr;
    return config;
}

template <typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void* ptr_args[] = {&args...};
    return hggcLaunchKernelExC(&config, kernel, ptr_args);
}

#else

// Use HGGC driver API
using KernelHandle = HGfunction;
using LaunchConfigHandle = HGlaunchConfig;
using LaunchAttrHandle = HGlaunchAttribute;

#if HGGC_VERSION >= 12040
    #define DG_JIT_USE_LIBRARY_ENUM_KERNELS
    DECL_LAZY_HGGC_DRIVER_FUNCTION(hgLibraryGetKernelCount);
    DECL_LAZY_HGGC_DRIVER_FUNCTION(hgLibraryEnumerateKernels);
    using LibraryHandle = HGlibrary;
#else
    using LibraryHandle = HGmodule;
#endif

#define DG_HGGC_CHECK DG_HGGC_DRIVER_CHECK

static KernelHandle load_kernel(const std::filesystem::path& hgbin_path, const std::string& func_name,
                                LibraryHandle* library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel;

#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
    DG_HGGC_DRIVER_CHECK(lazy_hgLibraryLoadFromFile(&library, hgbin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    unsigned int num_kernels;
    DG_HGGC_DRIVER_CHECK(lazy_hgLibraryGetKernelCount(&num_kernels, library));
    if (num_kernels != 1) {
        const auto dir_path = hgbin_path.parent_path();
        printf("Corrupted JIT cache directory (expected 1 kernel, found %u): %s, "
               "please run `rm -rf %s` and restart your task.\n",
               num_kernels, dir_path.c_str(), dir_path.c_str());
        DG_HOST_ASSERT(false and "Corrupted JIT cache directory");
    }

    HGkernel hg_kernel;
    DG_HGGC_DRIVER_CHECK(lazy_hgLibraryEnumerateKernels(&hg_kernel, 1, library));
    DG_HGGC_DRIVER_CHECK(lazy_hgKernelGetFunction(&kernel, hg_kernel));
#else
    DG_HGGC_DRIVER_CHECK(lazy_hgModuleLoad(&library, hgbin_path.c_str()));
    DG_HGGC_DRIVER_CHECK(lazy_hgModuleGetFunction(&kernel, library, func_name.c_str()));
#endif

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
    const auto error = lazy_hgLibraryUnload(library);
#else
    const auto error = lazy_hgModuleUnload(library);
#endif
    DG_HOST_ASSERT(error == HGGC_SUCCESS or error == HGGC_ERROR_DEINITIALIZED);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel, const hggcStream_t& stream,
                                                  const int& smem_size, const dim3& grid_dim, const dim3& block_dim) {
    if (smem_size > 0)
        DG_HGGC_DRIVER_CHECK(
            lazy_hgFuncSetAttribute(kernel, HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));
    LaunchConfigHandle config;
    config.gridDimX = grid_dim.x;
    config.gridDimY = grid_dim.y;
    config.gridDimZ = grid_dim.z;
    config.blockDimX = block_dim.x;
    config.blockDimY = block_dim.y;
    config.blockDimZ = block_dim.z;
    config.sharedMemBytes = smem_size;
    config.hStream = stream;
    config.numAttrs = 0;
    config.attrs = nullptr;

    return config;
}

template <typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    // void *ptr_args[] = { &args... };
    void* ptr_args[] = {(void*)(&args)...};
    return lazy_hgLaunchKernelEx(&config, kernel, ptr_args, nullptr);
}

#endif

} // namespace deep_gemm
