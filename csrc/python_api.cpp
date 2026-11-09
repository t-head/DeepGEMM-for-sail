#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/attention.hpp"
#include "apis/activation.hpp"
#include "apis/einsum.hpp"
#include "apis/epilogue_class.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega_moe.hpp"
#include "apis/int8_mega_moe.hpp"
#include "apis/runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    deep_gemm::epilogue_class::register_apis(m);
    deep_gemm::attention::register_apis(m);
    deep_gemm::activation::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::hyperconnection::register_apis(m);
    deep_gemm::mega_mhc::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::mega_int8::register_apis(m);
    deep_gemm::runtime::register_apis(m);
}
