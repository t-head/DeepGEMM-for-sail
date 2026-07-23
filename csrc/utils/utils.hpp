#pragma once

#include <string>
#include <torch/version.h>
#include <hggc_runtime_api.h>
#include "math.hpp"
#include "system.hpp"

namespace deep_gemm {

torch::Tensor get_col_major_tma_aligned_tensor(const torch::Tensor& x) {
    assert(x.dim() == 2 || x.dim() == 3);

    bool remove_dim = false;
    int64_t m = x.size(-2);
    int64_t n = x.size(-1);
    auto dtype = x.dtype();
    auto device = x.device();

    int64_t element_size = x.element_size();
    // int64_t aligned_m = get_tma_aligned_size(m, element_size);
    int64_t aligned_m = m;
    torch::Tensor x_view = x;

    if (x.dim() == 2) {
        if (x.stride(0) == 1 && x.stride(1) == aligned_m) {
            return x;
        }
        x_view = x.unsqueeze(0);
        remove_dim = true;
    }

    int64_t b = x_view.size(0);

    if (x_view.stride(0) == aligned_m * n && x_view.stride(1) == 1 && x_view.stride(2) == aligned_m) {
        return remove_dim ? x_view.squeeze(0) : x_view;
    }

    auto options = torch::TensorOptions().dtype(dtype).device(device);
    torch::Tensor aligned_x = torch::transpose(torch::empty({b, n, aligned_m}, options), 1, 2);

    aligned_x.slice(1, 0, m).copy_(x_view);

    aligned_x = aligned_x.slice(1, 0, m);

    return remove_dim ? aligned_x.squeeze(0) : aligned_x;
}

int get_num_sms() {
    static int* _num_sms = nullptr;
    if (_num_sms == nullptr) {
        _num_sms = new int(0);
        hggcDeviceProp device_props;
        hggcGetDeviceProperties(&device_props, 0);
        std::cout << "device_props.name:" << device_props.name << std::endl;
        std::string device_name(device_props.name);
        if (device_name.find("ZW810E") != std::string::npos || device_name.find("ZW610E") != std::string::npos) {
            *_num_sms = 20;
        } else {
            *_num_sms = device_props.multiProcessorCount;
        }
    }
    return *_num_sms;
}

bool is_ppu1v5_device() {
    hggcDeviceProp device_prop;
    hggcGetDeviceProperties(&device_prop, 0);
    if (device_prop.major == 8 && device_prop.minor == 9) {
        return true;
    } else {
        return false;
    }
}

int get_sm_count() {
    hggcDeviceProp device_prop;
    hggcGetDeviceProperties(&device_prop, 0);
    return device_prop.multiProcessorCount;
}

std::unordered_map<std::string, int> get_extra_info(int m = 0, int n = 0, int k = 0, int dtype = 1,
                                                    const std::string& api_type = "dense") {
    std::unordered_map<std::string, int> extra_info;
    extra_info["use_actlize_v100"] = is_ppu1v5_device() || get_env<int>("DG_USE_ACTLIZE_V100", 0);
    extra_info["use_multistage_on_N"] = get_env<int>("DG_USE_MULTISTAGE_ON_N", 0);
    extra_info["use_moe_dynamic_tile"] = get_env<int>("DG_USE_MOE_DYNAMIC_TILE", 0);

    return extra_info;
}

int get_m_alignment_for_contiguous_layout() {
    /*
    When we do a grouped GEMM in contiguous format, LHS are grouped into several batches along the M axis.
    Since we deal with exactly one sub-matrix of RHS for each GEMM block, batch sizes above should align well
        with GEMM block shape.
    Returns:
        Group-level alignment requirement for grouped contiguous layout, which is always 128.
    */
    return 128;
}

static dim3 get_grid_shape(int sm_count) {
    return dim3(sm_count, 1, 1);
}

static dim3 get_block_shape() {
    return dim3(128, 1, 1);
}

int32_t next_power_of_two(uint32_t n) {
    if (n == 0)
        return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

} // namespace deep_gemm
