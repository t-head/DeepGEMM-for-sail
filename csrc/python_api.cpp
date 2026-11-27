// Native PPU half types must precede Torch CUDA-compatible headers.
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#define __CUDA_FP16_H__
#define __CUDA_FP16_HPP__
#define __CUDA_BF16_H__
#define __CUDA_BF16_HPP__
using __nv_bfloat16 = __ppu_bfloat16;
using __nv_bfloat162 = __ppu_bfloat162;

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/attention.hpp"
#include "apis/activation.hpp"
#include "apis/einsum.hpp"
#include "apis/epilogue_class.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/mega_gate.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega_moe.hpp"
#include "apis/runtime.hpp"
#include "apis/mega_moe_training.hpp"

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
    deep_gemm::mega_gate::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::runtime::register_apis(m);
    deep_gemm::mega_moe_training::register_apis(m);
    deep_gemm::mega_moe_training::register_mega_moe_apis(m);
    deep_gemm::mega_moe_training::register_mega_moe_dw_block128_apis(m);
    deep_gemm::mega_moe_training::register_dw_activation_apis(m);
    deep_gemm::mega_moe_training::register_stable_routes_apis(m);
    deep_gemm::mega_moe_training::register_fp8_reuse_apis(m);
    deep_gemm::mega_moe_training::register_fp8_gradient_apis(m);
    deep_gemm::mega_moe_training::register_mega_moe_fp8_apis(m);
    deep_gemm::mega_moe_training::register_fp8_wgrad_nt_apis(m);
}
