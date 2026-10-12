#pragma once

#include <torch/extension.h>

#include <variant>

#include "../jit_kernels/heuristics/common_a8w4_int8.hpp"
#include "../jit_kernels/heuristics/common_fp4.hpp"
#include "../jit_kernels/heuristics/common_preprocess.hpp"
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

    // Deprecated scale-only alias of the unified MXFP4 preprocessing path.
    m.def("preprocess_mxfp4_scales",
          &deep_gemm_preprocess_common::preprocess_mxfp4_scales,
          py::arg("scale"));
    py::enum_<deep_gemm_preprocess_common::PreprocessMode>(m, "PreprocessMode")
        .value("MXFP4", deep_gemm_preprocess_common::PreprocessMode::MXFP4)
        .value("W4A16_MXFP4", deep_gemm_preprocess_common::PreprocessMode::W4A16_MXFP4)
        .value("W4A16_MXFP4_MMA", deep_gemm_preprocess_common::PreprocessMode::W4A16_MXFP4_MMA)
        .value("W4A8_MXFP4_MMA", deep_gemm_preprocess_common::PreprocessMode::W4A8_MXFP4_MMA)
        .value("W4A16_INT4", deep_gemm_preprocess_common::PreprocessMode::W4A16_INT4)
        .value("FP8_CHANNEL", deep_gemm_preprocess_common::PreprocessMode::FP8_CHANNEL)
        .value("FP8_BLOCK", deep_gemm_preprocess_common::PreprocessMode::FP8_BLOCK)
        .value("INT8", deep_gemm_preprocess_common::PreprocessMode::INT8)
        .value("BF16", deep_gemm_preprocess_common::PreprocessMode::BF16)
        .value("W4A8_INT8", deep_gemm_preprocess_common::PreprocessMode::W4A8_INT8);
    m.def("preprocess_weights_and_scales",
          [](const deep_gemm_preprocess_common::PreprocessMode mode, bool fuse_activation,
             const std::optional<torch::Tensor>& weight, const std::optional<const torch::Tensor>& weight_scale) -> py::tuple {
              auto result = deep_gemm_preprocess_common::preprocess_weights_and_scales(
                  mode, fuse_activation, weight.value_or(torch::Tensor{}), weight_scale);
              const py::object prepared_weight = result.weight.defined() ? py::cast(result.weight) : py::none();
              const py::object prepared_weight_scale = result.scale.defined() ? py::cast(result.scale) : py::none();
              if (mode == deep_gemm_preprocess_common::PreprocessMode::W4A8_MXFP4_MMA)
                  return py::make_tuple(prepared_weight, result.scale, result.inv_scale);
              return py::make_tuple(prepared_weight, prepared_weight_scale);
          },
          py::arg("mode"), py::arg("fuse_activation"), py::arg("weight") = std::nullopt,
          py::arg("weight_scale") = std::nullopt);
    m.def("preprocess_mxfp4_weight_for_act_and_quant_fusing",
          &deep_gemm_preprocess_common::preprocess_mxfp4_weight_for_act_and_quant_fusing,
          py::arg("weight"), py::arg("weight_scale"));
    m.def("preprocess_a8w4_weight",
          &deep_gemm_preprocess_common::preprocess_a8w4_weight,
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
