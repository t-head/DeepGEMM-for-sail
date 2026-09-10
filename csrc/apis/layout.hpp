#pragma once

#include <torch/extension.h>

#include "../jit_kernels/heuristics/common_fp4.hpp"
#include "../utils/utils.hpp"

namespace deep_gemm::layout {

static void register_apis(pybind11::module_& m) {
    // FP4 AUXILIARY KERNELs
    m.def("preprocess_mxfp4_scales",
          &deep_gemm_fp4_common::preprocess_mxfp4_scales,
          py::arg("scale"));
    m.def("preprocess_mxfp4_weight_for_act_and_quant_fusing",
          &deep_gemm_fp4_common::preprocess_mxfp4_weight_for_act_and_quant_fusing,
          py::arg("weight"), py::arg("weight_scale"));

    // LAYOUT UTILITIES
    m.def("get_col_major_tma_aligned_tensor",
          &deep_gemm::get_col_major_tma_aligned_tensor,
          py::arg("x"));
    m.def("get_col_major_tensor",
          &deep_gemm::get_col_major_tensor,
          py::arg("x"));
    m.def("get_m_alignment_for_contiguous_layout",
          &deep_gemm::get_m_alignment_for_contiguous_layout);
}

} // namespace deep_gemm::layout
