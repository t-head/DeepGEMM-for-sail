#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include "../jit_kernels/impls/epilogue_class.hpp"

namespace deep_gemm::epilogue_class {

// User-facing classes for the GEMM APIs' epilogue argument.
static void register_apis(pybind11::module_& m) {
    const auto submodule = m.def_submodule("epilogue", "GEMM epilogue classes");
    pybind11::class_<EpilogueClass, std::shared_ptr<EpilogueClass>>(submodule, "Epilogue");
    pybind11::class_<FP8CastWithClampEpilogue, EpilogueClass, std::shared_ptr<FP8CastWithClampEpilogue>>(
        submodule, "FP8CastWithClamp",
        "Cast BF16-rounded GEMM output to FP8, optionally clamping before scaling. "
        "clamped_count is an optional single-element CUDA uint32 tensor; "
        "the total is written asynchronously.")
        .def(pybind11::init<const torch::Tensor&, const std::optional<float>&, const std::optional<torch::Tensor>&>(),
             pybind11::arg("sfd"), pybind11::arg("clamp_value") = std::nullopt,
             pybind11::arg("clamped_count") = std::nullopt);
    pybind11::class_<FP8PerTokenCastEpilogue, EpilogueClass, std::shared_ptr<FP8PerTokenCastEpilogue>>(
        submodule, "FP8PerTokenCast",
        "PPU per-token FP8 output with one FP32 scale per flattened output row.")
        .def(pybind11::init<const torch::Tensor&>(), pybind11::arg("sfd"));
}

} // namespace deep_gemm::epilogue_class
