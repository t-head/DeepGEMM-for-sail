#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include "../jit_kernels/impls/epilogue_class.hpp"

namespace deep_gemm::epilogue_class {

// Register the upstream interface type; no custom implementations are exposed.
static void register_apis(pybind11::module_& m) {
    const auto submodule = m.def_submodule("epilogue", "GEMM epilogue classes");
    pybind11::class_<EpilogueClass, std::shared_ptr<EpilogueClass>>(submodule, "Epilogue");
}

} // namespace deep_gemm::epilogue_class
