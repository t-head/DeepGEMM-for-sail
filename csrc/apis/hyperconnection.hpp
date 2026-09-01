#pragma once

#include "../utils/compatibility.hpp"
#include "../utils/layout.hpp"
#include <torch/extension.h>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/tf32_hc_prenorm_gemm.hpp"
#endif

namespace deep_gemm::hyperconnection {

#if DG_TENSORMAP_COMPATIBLE
void tf32_hc_prenorm_gemm_nt(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                             const torch::Tensor& sqr_sum, std::optional<int> num_splits = std::nullopt,
                             std::optional<ConfigTuple> configs = std::nullopt) {
    // A and B must be K-major, D must be N-major
    DG_HOST_ASSERT(get_major_type_ab(a) == MajorType::K);
    DG_HOST_ASSERT(get_major_type_ab(b) == MajorType::K);
    check_major_type_cd(d);

    const auto& [m, k] = get_shape<2>(a);
    const auto& [n, k_] = get_shape<2>(b);

    DG_HOST_ASSERT(k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat32);
    TORCH_CHECK(a.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "rhs must be contiguous");

    // NOTES: the split-K partials are reduced in place, so `d` and `sqr_sum` always hold a single copy
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT((num_splits.has_value() ? (d.sizes() == std::vector<int64_t>{1, m, n})
                                           : (d.sizes() == std::vector<int64_t>{m, n})));
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");

    DG_HOST_ASSERT(sqr_sum.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT((num_splits.has_value() ? (sqr_sum.sizes() == std::vector<int64_t>{1, m})
                                           : (sqr_sum.sizes() == std::vector<int64_t>{m})));
    TORCH_CHECK(sqr_sum.is_contiguous(), "sqr_sum must be contiguous");

    if (m == 0) {
        return;
    }

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 8) {
        tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
#endif

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    // TF32 GEMMs
    m.def("tf32_hc_prenorm_gemm", &tf32_hc_prenorm_gemm_nt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("sqr_sum"), py::arg("num_splits") = std::nullopt, py::arg("configs") = std::nullopt);
#endif
}

} // namespace deep_gemm::hyperconnection
