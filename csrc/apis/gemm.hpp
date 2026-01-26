#pragma once

#include "../utils/compatibility.hpp"
#include <torch/extension.h>

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/ppu10500_fp8_gemm.hpp"
#include "../jit_kernels/impls/ppu10500_bf16_gemm.hpp"

#endif 

// #include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"

namespace deep_gemm::gemm {

static bool early_return(const int& m, const int &n, const int& k,
                         const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Do nothing if the problem is empty
    if (m == 0 or n == 0)
        return true;

    // Checks
    const bool& is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
    // if (c.has_value()) {
    //     check_major_type_cd(c.value());
    //     DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    //     DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
    // }

    // No accumulation
    if (k == 0) {
        if (not is_cd_same)
            c.has_value() ? d.copy_(c.value()) : d.zero_();
        return true;
    }

    // With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
    if (c.has_value() and not is_cd_same)
        d.copy_(c.value());
    return false;
}

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
static void fp8_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d, std::optional<ConfigTuple> config = std::nullopt, at::cuda::CUDAStream stream = at::cuda::getDefaultCUDAStream()) {
    // Shape must be `[M, K] @ [N, K].T`

    // C/D must be N-major
    check_major_type_cd(d);
    // Type and shape checks
    const auto& [m , k ] = get_shape<2>(a.first);
    const auto& [n , k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    // if (early_return(m, n, k, d))
    //     return;

    // Dispatch into different implements
    ppu10500_fp8_gemm(a.first, a.second, b.first, b.second, d, m, n, k, config, stream);
}

static void fp8_gemm_nt_no_stream(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d, std::optional<ConfigTuple> config = std::nullopt) {
    fp8_gemm_nt(a, b, d, config);
}

static void fp8_gemm_nt_no_config(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d, at::cuda::CUDAStream stream = at::cuda::getDefaultCUDAStream()) {
    fp8_gemm_nt(a, b, d, std::nullopt);
}

static void fp8_gemm_nt_no_stream_config(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d) {
    fp8_gemm_nt(a, b, d);
}

static void gemm_bf16_bf16_bf16_nt(const torch::Tensor& a,
                        const torch::Tensor& b, const torch::Tensor& d,
                        at::cuda::CUDAStream stream = at::cuda::getDefaultCUDAStream()) {
    const auto& [m , k ] = get_shape<2>(a);
    const auto& [n , k_] = get_shape<2>(b);
    const auto& [m_, n_] = get_shape<2>(d);

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(a.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    if (m == 0) {
        return;
    }
    bf16_gemm(a, b, d, m, n, k, stream);
}

static void gemm_bf16_bf16_bf16_nt_no_stream(const torch::Tensor& a,
                        const torch::Tensor& b, const torch::Tensor& d) {
    gemm_bf16_bf16_bf16_nt(a, b, d, at::cuda::getDefaultCUDAStream());
}
#endif


static void register_apis(pybind11::module_& m) {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    // m.def("gemm_bf16_bf16_bf16_nt", &gemm_bf16_bf16_bf16_nt,
    //       py::arg("a"), py::arg("b"), py::arg("d"), py::arg("stream"));
    // m.def("gemm_bf16_bf16_bf16_nt", &gemm_bf16_bf16_bf16_nt_no_stream,
    //       py::arg("a"), py::arg("b"), py::arg("d"));
    // FP8 GEMMs
    m.def("gemm_fp8_fp8_bf16_nt", &fp8_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("config"), py::arg("stream"));
    m.def("gemm_fp8_fp8_bf16_nt", &fp8_gemm_nt_no_stream,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("config"));
    m.def("gemm_fp8_fp8_bf16_nt", &fp8_gemm_nt_no_config,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("stream"));
    m.def("gemm_fp8_fp8_bf16_nt", &fp8_gemm_nt_no_stream_config,
        py::arg("a"), py::arg("b"), py::arg("d"));
#endif
}

} // namespace deep_gemm::gemm
