#pragma once

#include <torch/extension.h>

#include <variant>

#include "../jit_kernels/heuristics/common_a8w4.hpp"
#include "../jit_kernels/heuristics/common_fp4.hpp"
#include "../utils/layout.hpp"
#include "../utils/utils.hpp"

namespace deep_gemm::layout {

static torch::Tensor transform_sf_into_required_layout(
        const torch::Tensor& sf,
        const int& mn, const int& k,
        const std::variant<std::tuple<int, int, int>, std::tuple<int, int>>& recipe,
        const std::optional<int>& num_groups,
        const std::optional<bool>& is_sfa,
        const bool& disable_ue8m0_cast,
        const std::optional<torch::Tensor>& psum_layout = std::nullopt) {
    const auto arch_major = device_runtime->get_arch_major();

    if (arch_major == 8)
        return sf.contiguous();

    DG_HOST_UNREACHABLE("Unknown SF transformation");
}

static void register_apis(pybind11::module_& m) {
    m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout,
          py::arg("sf"), py::arg("mn"), py::arg("k"), py::arg("recipe"),
          py::arg("num_groups") = std::nullopt,
          py::arg("is_sfa") = std::nullopt,
          py::arg("disable_ue8m0_cast") = false,
          py::arg("psum_layout") = std::nullopt);

    // FP4 AUXILIARY KERNELs
    m.def("preprocess_mxfp4_scales",
          &deep_gemm_fp4_common::preprocess_mxfp4_scales,
          py::arg("scale"));
    m.def("preprocess_mxfp4_weight_for_act_and_quant_fusing",
          &deep_gemm_fp4_common::preprocess_mxfp4_weight_for_act_and_quant_fusing,
          py::arg("weight"), py::arg("weight_scale"));

    // A8W4 AUXILIARY KERNELs
    m.def("pack_a8w4_weight",
          &deep_gemm_a8w4_common::pack_a8w4_weight,
          py::arg("weight"));
    m.def("preprocess_a8w4_weight",
          &deep_gemm_a8w4_common::preprocess_a8w4_weight,
          py::arg("weight"));
    
    m.def("get_tma_aligned_size", &get_tma_aligned_size);
    m.def("get_mk_alignment_for_contiguous_layout", &get_mk_alignment_for_contiguous_layout);
    m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor, py::arg("x"));
    m.def("get_col_major_tensor", &get_col_major_tensor, py::arg("x"));
    m.def("set_mk_alignment_for_contiguous_layout", [](const int& new_value) {
        if (new_value != get_mk_alignment_for_contiguous_layout())
            DG_HOST_UNREACHABLE("PPU 1.0/1.5 only supports the default contiguous-layout alignment");
    });
    m.def("get_theoretical_mk_alignment_for_contiguous_layout", [](const std::optional<int>&) {
        return get_mk_alignment_for_contiguous_layout();
    }, py::arg("expected_m") = std::nullopt);

    // Unsupported UE8M0
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor", [](const torch::Tensor&) -> torch::Tensor {
        DG_HOST_UNREACHABLE("UE8M0-packed scaling factors are not supported on PPU 1.0/1.5");
    }, py::arg("sf"));
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
          [](const torch::Tensor&, const torch::Tensor&, const std::optional<std::vector<int>>&,
             const int&, const int&, const bool&) -> torch::Tensor {
        DG_HOST_UNREACHABLE("UE8M0-packed scaling factors are not supported on PPU 1.0/1.5");
    }, py::arg("sf"), py::arg("grouped_layout"), py::arg("ks_cpu"), py::arg("gran_k"),
       py::arg("k_alignment"), py::arg("use_psum_layout") = false);
}

} // namespace deep_gemm::layout
