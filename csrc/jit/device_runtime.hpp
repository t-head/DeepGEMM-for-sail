#pragma once

#include <iostream>
#include <string>
#include <acblasLt.h>
#include <hggc_runtime_api.h>
#include <torch/version.h>
#include <torch/torch.h>

#include "../utils/exception.hpp"
#include "../utils/lazy_init.hpp"

// Not support currently
#define PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE 0

namespace deep_gemm {

class DeviceRuntime {
    int num_sms = 0, tc_util = 0;
    bool enable_pdl = false;
    std::shared_ptr<hggcDeviceProp> cached_prop;

    // acBLASLt utils
    static constexpr size_t kAcblasLtWorkspaceSize = 32 * 1024 * 1024;

public:
    acblasLtHandle_t acblaslt_handle = nullptr;
    torch::Tensor acblaslt_workspace;
    bool use_pytorch_managed_acblaslt_handle;
    bool use_temp_acblaslt_workspace;

    explicit DeviceRuntime() {
        // Whether to use PyTorch acBLASLt
        // By default, we don't use it,
        // as `at::cuda::getCurrentCUDABlasLtHandle` has large CPU overhead with some PyTorch versions
        use_pytorch_managed_acblaslt_handle = get_env<int>("DG_USE_PYTORCH_ACBLASLT_HANDLE", 0) > 0;
#if not PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE
        DG_HOST_ASSERT(not use_pytorch_managed_acblaslt_handle and "PyTorch does not support to get acBLASLt handle");
#endif

        // Whether to create workspace tensor on each call instead of holding one.
        // Enabled by compute-sanitizer tests, which trigger runtime errors
        // when the workspace tensor is destructed after driver shutdown.
        use_temp_acblaslt_workspace = get_env<int>("DG_USE_TEMP_ACBLASLT_WORKSPACE", 0) > 0;

        if (not use_pytorch_managed_acblaslt_handle)
            DG_ACBLASLT_CHECK(acblasLtCreate(&acblaslt_handle));

        if (not use_temp_acblaslt_workspace)
            acblaslt_workspace = torch::empty({kAcblasLtWorkspaceSize}, dtype(torch::kByte).device(at::kCUDA));
    }

    ~DeviceRuntime() noexcept(false) {
        if (not use_pytorch_managed_acblaslt_handle)
            DG_ACBLASLT_CHECK(acblasLtDestroy(acblaslt_handle));
    }

    acblasLtHandle_t get_acblaslt_handle() const {
#if PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE
        if (use_pytorch_managed_acblaslt_handle)
            DG_HOST_UNREACHABLE("PyTorch-managed acBLASLt handle not yet available on PPU");
#endif
        return acblaslt_handle;
    }

    torch::Tensor get_acblaslt_workspace() {
        if (use_temp_acblaslt_workspace)
            return torch::empty({kAcblasLtWorkspaceSize}, dtype(torch::kByte).device(at::kCUDA));
        return acblaslt_workspace;
    }

    std::shared_ptr<hggcDeviceProp> get_prop() {
        if (cached_prop == nullptr) {
            int device_idx;
            hggcDeviceProp prop;
            DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&device_idx));
            DG_HGGC_RUNTIME_CHECK(hggcGetDeviceProperties(&prop, device_idx));
            cached_prop = std::make_shared<hggcDeviceProp>(prop);
        }
        return cached_prop;
    }

    std::pair<int, int> get_arch_pair() {
        const auto prop = get_prop();
        return {prop->major, prop->minor};
    }

    std::string get_arch(const bool& number_only = false,
                         const bool& support_arch_family = false) {
        const auto [major, minor] = get_arch_pair();
        if (major == 10 and minor != 1) {
            if (number_only)
                return "100";
            return support_arch_family ? "100f" : "100a";
        }
        return std::to_string(major * 10 + minor) + (number_only ? "" : "a");
    }

    int get_arch_major() {
        return get_arch_pair().first;
    }

    void set_num_sms(const int& new_num_sms) {
        DG_HOST_ASSERT(0 <= new_num_sms and new_num_sms <= get_prop()->multiProcessorCount);
        num_sms = new_num_sms;
    }

    int get_num_sms() {
        if (num_sms == 0) {
            const auto prop = get_prop();
            std::cout << "device_props.name:" << prop->name << std::endl;
            const std::string device_name(prop->name);
            // Synced from the legacy free function deep_gemm::get_num_sms() (utils/utils.hpp):
            // ZW810E/ZW610E expose a reduced SM budget, other devices use the full count.
            if (device_name.find("ZW810E") != std::string::npos or device_name.find("ZW610E") != std::string::npos) {
                num_sms = 20;
            } else {
                num_sms = prop->multiProcessorCount;
            }
        }
        return num_sms;
    }

    int get_l2_cache_size() {
        return get_prop()->l2CacheSize;
    }

    void set_tc_util(const int& new_tc_util) {
        DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
        tc_util = new_tc_util;
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }

    void set_pdl(const bool& new_enable_pdl) {
        if (new_enable_pdl) {
            static bool warned = false;
            if (not warned) {
                warned = true;
                printf("\033[33mWarning: PDL is not supported on PPU 1.0/1.5, `set_pdl(True)` has no effect\033[0m\n");
            }
        }
        enable_pdl = new_enable_pdl;
    }

    bool get_pdl() const {
        return enable_pdl;
    }
};

inline auto device_runtime = LazyInit<DeviceRuntime>([](){ return std::make_shared<DeviceRuntime>(); });

} // namespace deep_gemm
