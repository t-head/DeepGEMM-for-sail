#pragma once

#include <torch/extension.h>

#include "../jit_kernels/heuristics/common_fp4.hpp"

namespace deep_gemm::layout {

static void register_apis(pybind11::module_& m) {
    // FP4 AUXILIARY KERNELs
    m.def("preprocess_mxfp4_scales",
          &deep_gemm_fp4_common::preprocess_mxfp4_scales,
          py::arg("scale"));
    m.def("preprocess_mxfp4_weight_for_act_and_quant_fusing",
          &deep_gemm_fp4_common::preprocess_mxfp4_weight_for_act_and_quant_fusing,
          py::arg("weight"), py::arg("weight_scale"));
}

} // namespace deep_gemm::layout
