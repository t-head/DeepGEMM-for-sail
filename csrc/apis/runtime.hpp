#pragma once

#include <variant>

#include "../jit/compiler.hpp"
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/heuristics/runtime.hpp"

namespace deep_gemm::runtime {

static void register_apis(pybind11::module_& m) {
    m.def("set_num_sms", [&](const int& new_num_sms) {
        device_runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return device_runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        device_runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return device_runtime->get_tc_util();
    });
    m.def("set_pdl", [&](const bool& new_enable_pdl) {
        device_runtime->set_pdl(new_enable_pdl);
    });
    m.def("get_pdl", [&]() {
        return device_runtime->get_pdl();
    });
    m.def("set_ignore_compile_dims", [&](const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
    });
    m.def("set_block_size_multiple_of", [&](const std::variant<int, std::tuple<int, int>>& new_value) {
        if (std::holds_alternative<int>(new_value)) {
            auto x = std::get<int>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, x);
        } else {
            auto [x, y] = std::get<std::tuple<int, int>>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, y);
        }
    });

    // COMPILE MODE (mirrors Python jit/runtime.py set/get_compile_mode; ONLY_COMPILE = 1 compiles but skips launch)
    m.def("set_compile_mode", [](const int& mode) {
        set_compile_mode(mode);
    });
    m.def("get_compile_mode", []() {
        return get_compile_mode();
    });

    m.def("init", [&](const std::string& library_root_path, const std::string& sdk_home_path) {
        Compiler::prepare_init(library_root_path, sdk_home_path);
        KernelRuntime::prepare_init(sdk_home_path);
    });
}

extern "C"
{
    void deep_gemm_runtime_init(void* library_root_path_ptr, void * sdk_home_path_ptr)
    {
        using namespace deep_gemm::runtime;
        DG_HOST_ASSERT(library_root_path_ptr);
        DG_HOST_ASSERT(sdk_home_path_ptr);
        auto library_root_path = *reinterpret_cast<std::string*>(library_root_path_ptr);
        auto sdk_home_path = *reinterpret_cast<std::string*>(sdk_home_path_ptr);
        Compiler::prepare_init(library_root_path, sdk_home_path);
        KernelRuntime::prepare_init(sdk_home_path);
    }
}
} // namespace deep_gemm::runtime