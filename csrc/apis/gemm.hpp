#pragma once

#include "../utils/compatibility.hpp"
#include <torch/extension.h>
#include "../jit_kernels/impls/fp8_gemm.hpp"

#include "../jit_kernels/impls/bf16_gemm.hpp"
#include "../jit_kernels/impls/int8_gemm.hpp"
// #include "layout.hpp"

namespace deep_gemm::gemm {
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
extern "C"
{
void fp8_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d, std::optional<ConfigTuple> config = std::nullopt) {
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

    fp8_gemm(a.first, a.second, b.first, b.second, d, m, n, k, config);
}

void gemm_bf16_bf16_bf16_nt(const torch::Tensor& a,
                        const torch::Tensor& b, const torch::Tensor& d,
                        std::optional<ConfigTuple> config = std::nullopt) {
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
    bf16_gemm(a, b, d, m, n, k, config);
}

void gemm_int8_int8_bf16_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d, std::optional<ConfigTuple> config = std::nullopt) {
    const auto& [m , k ] = get_shape<2>(a.first);
    const auto& [n , k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kInt8);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kInt8);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(a.first.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(b.first.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    if (m == 0) {
        return;
    }
    int8_gemm(a.first, a.second, b.first, b.second, d, m, n, k, config);
}
}
static void register_apis(pybind11::module_& m) {
    m.def("gemm_bf16_bf16_bf16_nt", &gemm_bf16_bf16_bf16_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("config") = std::nullopt);
    m.def("gemm_int8_int8_bf16_nt", &gemm_int8_int8_bf16_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("config") = std::nullopt);
    // FP8 GEMMs
    m.def("gemm_fp8_fp8_bf16_nt", &fp8_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("config") = std::nullopt);
}

} // namespace deep_gemm::gemm
