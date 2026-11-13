#pragma once

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <acblasLt.h>
#include <hggc_ad.h>
#include <hggc_runtime_api.h>
#include <torch/version.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "../utils/exception.hpp"
#include "../utils/lazy_init.hpp"

#define PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE (TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 3))

namespace deep_gemm {

class DeviceRuntime {
    int tc_util = 0;
    bool enable_pdl = false;
    struct DeviceState {
        std::shared_ptr<hggcDeviceProp> prop;
        int num_sms = 0;
        bool numSmsExplicit = false;
        int cachedTensorCUs = 0;
    };
    std::mutex device_states_mutex;
    std::unordered_map<int, DeviceState> device_states;

    // Caller holds device_states_mutex. Device guards change the current device,
    // so both hardware properties and caller-selected SM budgets must follow it.
    DeviceState& current_device_state() {
        int device_idx;
        DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&device_idx));
        auto& state = device_states[device_idx];
        if (state.prop == nullptr) {
            hggcDeviceProp prop;
            DG_HGGC_RUNTIME_CHECK(hggcGetDeviceProperties(&prop, device_idx));
            state.prop = std::make_shared<hggcDeviceProp>(prop);
        }
        return state;
    }

    // These helpers also require device_states_mutex; do not call the locking
    // public getters while accessing a device state.
    int get_tensor_cu_count(DeviceState& state) {
        if (state.cachedTensorCUs == 0) {
            const auto& prop = state.prop;
            const std::string name(prop->name);
            int count = prop->multiProcessorCount;
            if (name.find("ZW810E") != std::string::npos || name.find("ZW610E") != std::string::npos) {
                int ordinal;
                HGdevice device;
                DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&ordinal));
                DG_HOST_ASSERT(hgDeviceGet(&device, ordinal) == HGGC_SUCCESS);
                DG_HOST_ASSERT(hgDeviceGetAttributeAD(&count, HGAD_DEVICE_ATTRIBUTE_TENSOR_MULTIPROCESSOR_COUNT,
                                                      device) == HGGC_SUCCESS);
            }
            DG_HOST_ASSERT(count > 0 && count <= prop->multiProcessorCount);
            state.cachedTensorCUs = count;
        }
        return state.cachedTensorCUs;
    }

    int get_num_sms(DeviceState& state) {
        if (state.num_sms == 0) {
            const auto& prop = state.prop;
            std::cout << "device_props.name:" << prop->name << std::endl;
            const std::string device_name(prop->name);
            if (device_name.find("ZW810E") != std::string::npos) {
                state.num_sms = get_tensor_cu_count(state);
            } else if (device_name.find("ZW610E") != std::string::npos) {
                // Preserve the legacy default on 610E.
                state.num_sms = 20;
            } else {
                state.num_sms = prop->multiProcessorCount;
            }
        }
        return state.num_sms;
    }

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
            return at::cuda::getCurrentCUDABlasLtHandle();
#endif
        return acblaslt_handle;
    }

    torch::Tensor get_acblaslt_workspace() const {
        if (use_temp_acblaslt_workspace)
            return torch::empty({kAcblasLtWorkspaceSize}, dtype(torch::kByte).device(at::kCUDA));
        return acblaslt_workspace;
    }

    std::shared_ptr<hggcDeviceProp> get_prop() {
        std::lock_guard<std::mutex> lock(device_states_mutex);
        return current_device_state().prop;
    }

    std::pair<int, int> get_arch_pair() {
        const auto prop = get_prop();
        return {prop->major, prop->minor};
    }

    std::string get_arch() {
        const auto [major, minor] = get_arch_pair();
        return std::to_string(major * 10 + minor);
    }

    int get_arch_major() {
        return get_arch_pair().first;
    }

    int get_tensor_cu_count() {
        std::lock_guard<std::mutex> lock(device_states_mutex);
        return get_tensor_cu_count(current_device_state());
    }

    void set_num_sms(const int& new_num_sms) {
        std::lock_guard<std::mutex> lock(device_states_mutex);
        auto& state = current_device_state();
        DG_HOST_ASSERT(0 <= new_num_sms and new_num_sms <= state.prop->multiProcessorCount);
        state.num_sms = new_num_sms;
        state.numSmsExplicit = new_num_sms != 0;
    }

    int get_num_sms() {
        std::lock_guard<std::mutex> lock(device_states_mutex);
        return get_num_sms(current_device_state());
    }

    // On 810E, settings up to the Tensor count remove the same number of SIMT CUs.
    // Larger settings directly specify the SIMT budget; Tensor paths clamp separately.
    // Other devices retain their existing explicit/automatic budget semantics.
    int get_num_simt_sms() {
        std::lock_guard<std::mutex> lock(device_states_mutex);
        auto& state = current_device_state();
        const auto& prop = state.prop;
        if (std::string(prop->name).find("ZW810E") != std::string::npos) {
            const int sms = get_num_sms(state), tensorCUs = get_tensor_cu_count(state);
            return sms <= tensorCUs ? prop->multiProcessorCount - tensorCUs + sms : sms;
        }
        return state.numSmsExplicit ? get_num_sms(state) : prop->multiProcessorCount;
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

// The stream the caller (vLLM/SGLang) is currently on, as an HGGC handle. Torch hands it back
// as `CUstream_st*` while HGGC takes `HGstream_st*` -- unrelated pointer types naming the same
// object, so `reinterpret_cast` is the only well-formed conversion. Kept here instead of at every
// launch site; after cudafy the two types coincide and the cast degenerates to a no-op.
inline hggcStream_t current_stream() {
    return reinterpret_cast<hggcStream_t>(at::cuda::getCurrentCUDAStream().stream());
}

} // namespace deep_gemm
