#pragma once

#include "../utils/compatibility.hpp"
#include "../utils/layout.hpp"
#include <torch/extension.h>
#include "../jit_kernels/impls/einsum.hpp"
#include "gemm.hpp"

namespace deep_gemm::einsum {
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

extern "C" {

void int8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
              const torch::Tensor& b, const torch::Tensor& sfb,
              const torch::Tensor& d,
              const std::optional<torch::Tensor>& c,
              std::optional<std::tuple<int, int, int>> recipe,
              const std::string& compiled_dims = "",
              std::optional<ConfigTuple> configs = std::nullopt) {
    // Shape must be `[B, M, K] @ [B, N, K].T`
    DG_HOST_ASSERT(a.stride(-1) == 1 or a.stride(-2) == 1);
    DG_HOST_ASSERT(b.stride(-1) == 1 or b.stride(-2) == 1);
    DG_HOST_ASSERT(d.stride(-1) == 1);
    const auto& [batch_size, m, k] = get_shape<3>(a);
    const auto& [batch_size_, n, k_] = get_shape<3>(b);
    const auto& [m_, batch_size__, n_] = get_shape<3>(d);

    // Type and shape checks
    DG_HOST_ASSERT(batch_size == batch_size_ && batch_size_ == batch_size__);
    DG_HOST_ASSERT(m == m_ && n == n_ && k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kInt8 || a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kInt8 || b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(a.scalar_type() == b.scalar_type());
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(sfb.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (batch_size == 0 or gemm::early_return(m, n, k, d, c))
        return;

    int8_bmm_impl(a, sfa, b, sfb, d, c, configs);
}

void fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
             const torch::Tensor& b, const torch::Tensor& sfb,
             const torch::Tensor& d,
             const std::optional<torch::Tensor>& c,
             std::optional<std::tuple<int, int, int>> recipe,
             const std::string& compiled_dims = "",
             std::optional<ConfigTuple> configs = std::nullopt) {
    // Shape must be `[B, M, K] @ [B, N, K].T`
    DG_HOST_ASSERT(a.stride(-1) == 1 or a.stride(-2) == 1);
    DG_HOST_ASSERT(b.stride(-1) == 1 or b.stride(-2) == 1);
    DG_HOST_ASSERT(d.stride(-1) == 1);
    const auto& [batch_size, m, k] = get_shape<3>(a);
    const auto& [batch_size_, n, k_] = get_shape<3>(b);
    const auto& [m_, batch_size__, n_] = get_shape<3>(d);

    // Per-tensor scale fallback
    if (sfa.sizes() == std::vector<int64_t>{batch_size, m, 1} &&
        sfb.sizes() == std::vector<int64_t>{batch_size_, n, 1}) {
        return int8_bmm(a, sfa, b, sfb, d, c, recipe, compiled_dims, configs);
    }

    DG_HOST_ASSERT(batch_size == batch_size_ && batch_size_ == batch_size__);
    DG_HOST_ASSERT(m == m_ && n == n_ && k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sfb.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (batch_size == 0 or gemm::early_return(m, n, k, d, c))
        return;

    fp8_bmm_impl(a, sfa, b, sfb, d, c, configs);
}

void int8_einsum(const std::string& expr,
                 const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::tuple<int, int, int>& recipe) {
    // Some hardcoded Einstein sum kernels
    // const auto arch_major = device_runtime->get_arch_major();
    if (expr == "bhr,hdr->bhd") {
        // Permute dims to satisfy the order of (batch_size, m, n, k)
        // (batch_size, m, n, k): (h, b, d, r)
        auto& lhs = a.first;
        auto& lhs_scales = a.second;
        auto [perm_a, perm_sfa] = fused_permute(lhs, lhs_scales);
        int8_bmm(perm_a, perm_sfa, b.first, b.second, d, c, recipe, "nk");
    } else if (expr == "bhd,hdr->bhr") {
        throw std::runtime_error("bhd,hdr->bhr is not yet supported in PPU int8_einsum.");
    } else if (expr == "bhd,bhr->hdr") {
        throw std::runtime_error("bhd,bhr->hdr is not yet supported in PPU int8_einsum.");
    } else {
        throw std::runtime_error("unsupported expr expression: " + expr);
    }
}

void fp8_einsum(const std::string& expr,
                const std::pair<torch::Tensor, torch::Tensor>& a,
                const std::pair<torch::Tensor, torch::Tensor>& b,
                const torch::Tensor& d,
                const std::optional<torch::Tensor>& c,
                const std::tuple<int, int, int>& recipe) {
    // Some hardcoded Einstein sum kernels
    // const auto arch_major = device_runtime->get_arch_major();
    if (expr == "bhr,hdr->bhd") {
        // Permute dims to satisfy the order of (batch_size, m, n, k)
        // (batch_size, m, n, k): (h, b, d, r)
        auto& lhs = a.first;
        auto& lhs_scales = a.second;
        auto [perm_a, perm_sfa] = fused_permute(lhs, lhs_scales);
        fp8_bmm(perm_a, perm_sfa, b.first, b.second, d, c, recipe, "nk");
    } else if (expr == "bhd,hdr->bhr") {
        throw std::runtime_error("bhd,hdr->bhr is not yet supported in PPU fp8_einsum.");
    } else if (expr == "bhd,bhr->hdr") {
        throw std::runtime_error("bhd,bhr->hdr is not yet supported in PPU fp8_einsum.");
    } else {
        throw std::runtime_error("unsupported expr expression: " + expr);
    }
}

}

static void register_apis(pybind11::module_& m) {
    m.def("fp8_einsum", &fp8_einsum, py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt, py::arg("recipe") = std::vector<int>{1, 128, 128});
    m.def("int8_einsum", &int8_einsum, py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt, py::arg("recipe") = std::vector<int>{1, 1, 128});
}

} // namespace deep_gemm::einsum
