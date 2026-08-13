#pragma once

#include "../utils/compatibility.hpp"
#include <torch/extension.h>
#include "../jit_kernels/impls/fp8_gemm.hpp"

#include "../jit_kernels/impls/bf16_gemm.hpp"
#include "../jit_kernels/impls/int8_gemm.hpp"
#include "../jit_kernels/impls/m_grouped_bf16_gemm.hpp"
#include "../jit_kernels/impls/m_grouped_fp8_gemm.hpp"
#include "../jit_kernels/impls/m_grouped_int8_gemm.hpp"
// #include "layout.hpp"
#include "../jit_kernels/impls/fp4_gemm.hpp"
#include "../jit_kernels/impls/m_grouped_fp4_gemm.hpp"
#include "../jit_kernels/impls/tf32_hc_prenorm_gemm.hpp"

namespace deep_gemm::gemm {
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
extern "C" {
void gemm_bf16_bf16_bf16_nt(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                            std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& [m, k] = get_shape<2>(a);
    const auto& [n, k_] = get_shape<2>(b);
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
    bf16_gemm(a, b, d, m, n, k, configs);
}

void gemm_int8_int8_bf16_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& d,
                            std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [n, k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT((a.first.scalar_type() == torch::kInt8) || (a.first.scalar_type() == torch::kFloat8_e4m3fn));
    DG_HOST_ASSERT((b.first.scalar_type() == torch::kInt8) || (b.first.scalar_type() == torch::kFloat8_e4m3fn));
    DG_HOST_ASSERT(a.first.scalar_type() == b.first.scalar_type());
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT((a.second.size(0) == m) && (a.second.scalar_type() == torch::kFloat32));
    DG_HOST_ASSERT((b.second.size(0) == n) && (b.second.scalar_type() == torch::kFloat32));
    TORCH_CHECK(a.first.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(b.first.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    if (m == 0) {
        return;
    }
    int8_gemm(a.first, a.second, b.first, b.second, d, m, n, k, configs);
}

void fp8_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a, const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d, std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs_scales = a.second;
    const auto& rhs_scales = b.second;
    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [n, k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);

    if ((lhs_scales.sizes() == std::vector<int64_t>{m, 1}) && (rhs_scales.sizes() == std::vector<int64_t>{n, 1})) {
        return gemm_int8_int8_bf16_nt(a, b, d, configs);
    }
    // DG_HOST_ASSERT(k % 128 == 0);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, (k + 127) / 128}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{(n + 127) / 128, (k + 127) / 128}));
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(a.first.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(b.first.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");
    fp8_gemm(a.first, a.second, b.first, b.second, d, m, n, k, configs);
}

void fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const std::optional<torch::Tensor>& bias,
                 const torch::Tensor& d,
                 std::optional<ConfigTuple> config = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;
    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [n, k_] = get_shape<2>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);

    // Type and shape checks
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");

    // Handle bias - create empty tensor if not provided
    torch::Tensor bias_tensor;
    if (bias.has_value() && bias->defined() && bias->numel() > 0) {
        bias_tensor = *bias;
        DG_HOST_ASSERT(bias_tensor.scalar_type() == torch::kFloat32);
    } else {
        bias_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(lhs.device()));
    }

    if (m == 0) return;

    fp4_gemm(lhs, lhs_scales, rhs, rhs_scales, bias_tensor, d, m, n, k, config);
}

void m_grouped_gemm_int8_int8_bf16_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d, const torch::Tensor& m_indices,
                                                 std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    DG_HOST_ASSERT(m == m_ && m_ == m__ && k == k_ && n == n_);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, 1}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kInt8 || lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kInt8 || rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs.scalar_type() == rhs.scalar_type());
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (m == 0) {
        return;
    }

    m_grouped_gemm_int8_int8_bf16_nt_contiguous_impl(lhs, lhs_scales, rhs, rhs_scales, d, m_indices, m, n, k,
                                                     num_groups, configs);
}

std::pair<int, int> m_grouped_gemm_int8_int8_bf16_nt_masked(
    const std::pair<torch::Tensor, torch::Tensor>& a, const std::pair<torch::Tensor, torch::Tensor>& b,
    const torch::Tensor& d, const torch::Tensor& masked_m, int expected_m,
    std::optional<ConfigTuple> configs = std::nullopt, std::optional<int> max_block_n = 256,
    std::optional<bool> enable_sbo_overlap = false, std::optional<const torch::Tensor> signal = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    at::Tensor signal_tensor;
    if (signal.has_value() && signal->defined()) {
        signal_tensor = *signal;
    } else {
        signal_tensor = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(d.device()));
    }

    const auto& [num_groups, m, k] = get_shape<3>(lhs);
    const auto& [num_groups_, n, k_] = get_shape<3>(rhs);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    int num_groups___ = masked_m.numel();

    // Type and shape checks (matching Python implementation)
    DG_HOST_ASSERT(num_groups == num_groups_ && num_groups_ == num_groups__ && num_groups__ == num_groups___);
    DG_HOST_ASSERT(m == m_ && n == n_ && k == k_);
    DG_HOST_ASSERT(expected_m > 0 && m > 0 && n > 0 && k > 0 && num_groups > 0);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{num_groups, m, 1}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kInt8 || lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kInt8 || rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs.scalar_type() == rhs.scalar_type());
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (enable_sbo_overlap.value_or(false)) {
        TORCH_CHECK(signal_tensor.defined(), "signal must be defined when enable_sbo_overlap is true");
        TORCH_CHECK(signal_tensor.is_contiguous(), "signal must be contiguous");
        TORCH_CHECK(signal_tensor.scalar_type() == torch::kInt32, "signal must be int32");
    }

    return m_grouped_gemm_int8_int8_bf16_nt_masked_impl(lhs, lhs_scales, rhs, rhs_scales, d, masked_m, m, n, k,
                                                        num_groups, expected_m, configs, max_block_n.value_or(256),
                                                        enable_sbo_overlap.value_or(false), signal_tensor);
}

void m_grouped_gemm_int8_int8_bf16_nt_nopad(const std::pair<torch::Tensor, torch::Tensor>& a,
                                            const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& d,
                                            const torch::Tensor& m_indices,
                                            std::optional<const torch::Tensor> m_rows = std::nullopt,
                                            std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    // Type and shape checks (matching Python implementation)
    DG_HOST_ASSERT(m == m_ && m_ == m__ && k == k_ && n == n_);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, 1}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kInt8 || lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kInt8 || rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs.scalar_type() == rhs.scalar_type());
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (m == 0) {
        return;
    }

    m_grouped_gemm_int8_int8_bf16_nt_nopad_impl(lhs, lhs_scales, rhs, rhs_scales, d, m_indices, m, n, k, num_groups,
                                                m_rows, configs);
}

void m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                               const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& d,
                                               const torch::Tensor& m_indices,
                                               std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    if ((lhs_scales.sizes() == std::vector<int64_t>{m, 1}) &&
        (rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1})) {
        return m_grouped_gemm_int8_int8_bf16_nt_contiguous(a, b, d, m_indices, configs);
    }

    // Type and shape checks (matching Python implementation)
    DG_HOST_ASSERT(m == m_ && m_ == m__ && k == k_ && n == n_);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, (k + 127) / 128}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, (n + 127) / 128, (k + 127) / 128}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (m == 0) {
        return;
    }

    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous_impl(lhs, lhs_scales, rhs, rhs_scales, d, m_indices, m, n, k, num_groups,
                                                   configs);
}

std::pair<int, int> m_grouped_gemm_fp8_fp8_bf16_nt_masked(
    const std::pair<torch::Tensor, torch::Tensor>& a, const std::pair<torch::Tensor, torch::Tensor>& b,
    const torch::Tensor& d, const torch::Tensor& masked_m, int expected_m,
    std::optional<ConfigTuple> configs = std::nullopt, std::optional<int> max_block_n = 256,
    std::optional<bool> enable_sbo_overlap = false, std::optional<const torch::Tensor> signal = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    at::Tensor signal_tensor;
    if (signal.has_value() && signal->defined()) {
        signal_tensor = *signal;
    } else {
        signal_tensor = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(d.device()));
    }

    const auto& [num_groups, m, k] = get_shape<3>(lhs);
    const auto& [num_groups_, n, k_] = get_shape<3>(rhs);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    int num_groups___ = masked_m.numel();

    if ((lhs_scales.sizes() == std::vector<int64_t>{num_groups, m, 1}) &&
        (rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1})) {
        return m_grouped_gemm_int8_int8_bf16_nt_masked(a, b, d, masked_m, expected_m, configs, max_block_n,
                                                       enable_sbo_overlap, signal);
    }

    // Type and shape checks (matching Python implementation)
    DG_HOST_ASSERT(num_groups == num_groups_ && num_groups_ == num_groups__ && num_groups__ == num_groups___);
    DG_HOST_ASSERT(m == m_ && n == n_ && k == k_);
    DG_HOST_ASSERT(expected_m > 0 && m > 0 && n > 0 && k > 0 && num_groups > 0);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{num_groups, m, (k + 127) / 128}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, (n + 127) / 128, (k + 127) / 128}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (enable_sbo_overlap.value_or(false)) {
        TORCH_CHECK(signal_tensor.defined(), "signal must be defined when enable_sbo_overlap is true");
        TORCH_CHECK(signal_tensor.is_contiguous(), "signal must be contiguous");
        TORCH_CHECK(signal_tensor.scalar_type() == torch::kInt32, "signal must be int32");
    }

    return m_grouped_gemm_fp8_fp8_bf16_nt_masked_impl(lhs, lhs_scales, rhs, rhs_scales, d, masked_m, m, n, k,
                                                      num_groups, expected_m, configs, max_block_n.value_or(256),
                                                      enable_sbo_overlap.value_or(false), signal_tensor);
}

void m_grouped_gemm_fp8_fp8_bf16_nt_nopad(const std::pair<torch::Tensor, torch::Tensor>& a,
                                          const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& d,
                                          const torch::Tensor& m_indices,
                                          std::optional<const torch::Tensor> m_rows = std::nullopt,
                                          std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    if ((lhs_scales.sizes() == std::vector<int64_t>{m, 1}) &&
        (rhs_scales.sizes() == std::vector<int64_t>{num_groups, n, 1})) {
        return m_grouped_gemm_int8_int8_bf16_nt_nopad(a, b, d, m_indices, m_rows, configs);
    }

    // Type and shape checks (matching Python implementation)
    DG_HOST_ASSERT(m == m_ && m_ == m__ && k == k_ && n == n_);
    DG_HOST_ASSERT((lhs_scales.sizes() == std::vector<int64_t>{m, (k + 127) / 128}));
    DG_HOST_ASSERT((rhs_scales.sizes() == std::vector<int64_t>{num_groups, (n + 127) / 128, (k + 127) / 128}));

    DG_HOST_ASSERT(lhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(lhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(rhs_scales.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);

    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    if (m == 0) {
        return;
    }

    m_grouped_gemm_fp8_fp8_bf16_nt_nopad_impl(lhs, lhs_scales, rhs, rhs_scales, d, m_indices, m, n, k, num_groups,
                                              m_rows, configs);
}

void m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                                 const torch::Tensor& out, const torch::Tensor& m_indices,
                                                 std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(out);
    int m__ = m_indices.numel();

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(lhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(rhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(out.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.dtype() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");

    if (m == 0) {
        return;
    }

    m_grouped_gemm_bf16_bf16_bf16_nt_contiguous_impl(lhs, rhs, out, m_indices, m, n, k, num_groups, configs);
}

std::pair<int, int> m_grouped_gemm_bf16_bf16_bf16_nt_masked(
    const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out, const torch::Tensor& masked_m,
    int expected_m, std::optional<ConfigTuple> configs = std::nullopt, std::optional<int> max_block_n = 256,
    std::optional<bool> enable_sbo_overlap = false, std::optional<const torch::Tensor> signal = std::nullopt) {
    at::Tensor signal_tensor;
    if (signal.has_value() && signal->defined()) {
        signal_tensor = *signal;
    } else {
        signal_tensor = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(out.device()));
    }
    const auto& [num_groups, m, k] = get_shape<3>(lhs);
    const auto& [num_groups_, n, k_] = get_shape<3>(rhs);
    const auto& [num_groups__, m_, n_] = get_shape<3>(out);
    int num_groups___ = masked_m.numel();

    DG_HOST_ASSERT(num_groups == num_groups_ && num_groups_ == num_groups__ && num_groups__ == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(lhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(rhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(out.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.dtype() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");

    if (enable_sbo_overlap.value_or(false)) {
        TORCH_CHECK(signal_tensor.defined(), "signal must be defined when enable_sbo_overlap is true");
        TORCH_CHECK(signal_tensor.is_contiguous(), "signal must be contiguous");
        TORCH_CHECK(signal_tensor.scalar_type() == torch::kInt32, "signal must be int32");
    }

    return m_grouped_gemm_bf16_bf16_bf16_nt_masked_impl(lhs, rhs, out, masked_m, m, n, k, num_groups, expected_m,
                                                        configs, max_block_n.value_or(256),
                                                        enable_sbo_overlap.value_or(false), signal_tensor);
}

void m_grouped_gemm_bf16_bf16_bf16_nt_nopad(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                            const torch::Tensor& out, const torch::Tensor& m_indices,
                                            std::optional<const torch::Tensor> m_rows = std::nullopt,
                                            std::optional<ConfigTuple> configs = std::nullopt) {
    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(out);
    int m__ = m_indices.numel();

    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_ and m__ == m_);
    DG_HOST_ASSERT(lhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(rhs.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(out.dtype() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.dtype() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");

    if (m == 0) {
        return;
    }

    m_grouped_gemm_bf16_bf16_bf16_nt_nopad_impl(lhs, rhs, out, m_indices, m, n, k, num_groups, m_rows, configs);
}

static void m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& d,
    const torch::Tensor& m_indices,
    std::optional<const torch::Tensor> m_rows = std::nullopt,
    std::optional<ConfigTuple> config = std::nullopt) {

    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    const auto& [m, k] = get_shape<2>(lhs);
    const auto& [num_groups, n, k_] = get_shape<3>(rhs);
    const auto& [m_, n_] = get_shape<2>(d);
    int m__ = m_indices.numel();

    DG_HOST_ASSERT(m == m_ && m_ == m__ && k == k_ && n == n_);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");

    torch::Tensor bias_tensor;
    if (bias.has_value() && bias->defined() && bias->numel() > 0) {
        bias_tensor = *bias;
    } else {
        bias_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(lhs.device()));
    }

    torch::Tensor m_rows_tensor;
    if (m_rows.has_value() && m_rows->defined()) {
        m_rows_tensor = *m_rows;
    } else {
        m_rows_tensor = torch::Tensor();  // undefined
    }

    if (m == 0) return;

    m_grouped_gemm_fp4_fp4_bf16_nt_nopad_impl(lhs, lhs_scales, rhs, rhs_scales, bias_tensor, d,
                                                m_indices, m_rows_tensor, m, n, k, num_groups, config);
}

static void m_grouped_gemm_fp4_fp4_bf16_nt_masked(
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& d,
    const torch::Tensor& masked_m,
    int expected_m,
    std::optional<ConfigTuple> config = std::nullopt,
    std::optional<int> max_block_n = 256,
    std::optional<bool> enable_sbo_overlap = false,
    std::optional<const torch::Tensor> signal = std::nullopt) {

    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;

    at::Tensor signal_tensor;
    if (signal.has_value() && signal->defined()) {
        signal_tensor = *signal;
    } else {
        signal_tensor = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(d.device()));
    }

    const auto& [num_groups, m, k] = get_shape<3>(lhs);
    const auto& [num_groups_, n, k_] = get_shape<3>(rhs);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    int num_groups___ = masked_m.numel();

    DG_HOST_ASSERT(num_groups == num_groups_ && num_groups_ == num_groups__ && num_groups__ == num_groups___);
    DG_HOST_ASSERT(m == m_ && n == n_ && k == k_);
    DG_HOST_ASSERT(expected_m > 0 && m > 0 && n > 0 && k > 0 && num_groups > 0);
    DG_HOST_ASSERT(lhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(rhs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);
    TORCH_CHECK(lhs.is_contiguous(), "lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(), "rhs must be contiguous");
    TORCH_CHECK(d.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");

    torch::Tensor bias_tensor;
    if (bias.has_value() && bias->defined() && bias->numel() > 0) {
        bias_tensor = *bias;
    } else {
        bias_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(lhs.device()));
    }

    if (enable_sbo_overlap.value_or(false)) {
        TORCH_CHECK(signal_tensor.defined(), "signal must be defined when enable_sbo_overlap is true");
        TORCH_CHECK(signal_tensor.is_contiguous(), "signal must be contiguous");
        TORCH_CHECK(signal_tensor.scalar_type() == torch::kInt32, "signal must be int32");
    }

    m_grouped_gemm_fp4_fp4_bf16_nt_masked_impl(lhs, lhs_scales, rhs, rhs_scales, bias_tensor, d,
                                                 masked_m, m, n, k, num_groups, expected_m, config,
                                                 max_block_n.value_or(256), enable_sbo_overlap.value_or(false),
                                                 signal_tensor);
}

void tf32_hc_prenorm_gemm_nt(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                             const torch::Tensor& sqr_sum, std::optional<int> num_splits = std::nullopt,
                             std::optional<ConfigTuple> configs = std::nullopt) {
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

    tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k);
}
}

static void register_apis(pybind11::module_& m) {
    // BF16 GEMMs
    m.def("gemm_bf16_bf16_bf16_nt", &gemm_bf16_bf16_bf16_nt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_bf16_bf16_bf16_nt_contiguous", &m_grouped_gemm_bf16_bf16_bf16_nt_contiguous, py::arg("lhs"),
          py::arg("rhs"), py::arg("out"), py::arg("m_indices"), py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_bf16_bf16_bf16_nt_masked", &m_grouped_gemm_bf16_bf16_bf16_nt_masked, py::arg("lhs"),
          py::arg("rhs"), py::arg("out"), py::arg("masked_m"), py::arg("expected_m"), py::arg("configs") = std::nullopt,
          py::arg("max_block_n") = 256, py::arg("enable_sbo_overlap") = false, py::arg("signal") = std::nullopt);
    m.def("m_grouped_gemm_bf16_bf16_bf16_nt_nopad", &m_grouped_gemm_bf16_bf16_bf16_nt_nopad, py::arg("lhs"),
          py::arg("rhs"), py::arg("out"), py::arg("m_indices"), py::arg("m_rows") = std::nullopt,
          py::arg("configs") = std::nullopt);
    // INT8 GEMMS
    m.def("gemm_int8_int8_bf16_nt", &gemm_int8_int8_bf16_nt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_int8_int8_bf16_nt_contiguous", &m_grouped_gemm_int8_int8_bf16_nt_contiguous, py::arg("a"),
          py::arg("b"), py::arg("d"), py::arg("m_indices"), py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_int8_int8_bf16_nt_masked", &m_grouped_gemm_int8_int8_bf16_nt_masked, py::arg("a"),
          py::arg("b"), py::arg("d"), py::arg("masked_m"), py::arg("expected_m"), py::arg("configs") = std::nullopt,
          py::arg("max_block_n") = 256, py::arg("enable_sbo_overlap") = false, py::arg("signal") = std::nullopt);
    m.def("m_grouped_gemm_int8_int8_bf16_nt_nopad", &m_grouped_gemm_int8_int8_bf16_nt_nopad, py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("m_indices"), py::arg("m_rows") = std::nullopt, py::arg("configs") = std::nullopt);
    // FP8 GEMMs
    m.def("gemm_fp8_fp8_bf16_nt", &fp8_gemm_nt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_fp8_fp8_bf16_nt_contiguous", &m_grouped_gemm_fp8_fp8_bf16_nt_contiguous, py::arg("a"),
          py::arg("b"), py::arg("d"), py::arg("m_indices"), py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_fp8_fp8_bf16_nt_masked", &m_grouped_gemm_fp8_fp8_bf16_nt_masked, py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("masked_m"), py::arg("expected_m"), py::arg("configs") = std::nullopt,
          py::arg("max_block_n") = 256, py::arg("enable_sbo_overlap") = false, py::arg("signal") = std::nullopt);
    m.def("m_grouped_gemm_fp8_fp8_bf16_nt_nopad", &m_grouped_gemm_fp8_fp8_bf16_nt_nopad, py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("m_indices"), py::arg("m_rows") = std::nullopt, py::arg("configs") = std::nullopt);
    // FP4 GEMMs
    m.def("gemm_fp4_fp4_bf16_nt", &fp4_gemm_nt, py::arg("a"), py::arg("b"), py::arg("bias"),
          py::arg("d"), py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_fp4_fp4_bf16_nt_nopad", &m_grouped_gemm_fp4_fp4_bf16_nt_nopad, py::arg("a"),
          py::arg("b"), py::arg("bias"), py::arg("d"), py::arg("m_indices"),
          py::arg("m_rows") = std::nullopt, py::arg("configs") = std::nullopt);
    m.def("m_grouped_gemm_fp4_fp4_bf16_nt_masked", &m_grouped_gemm_fp4_fp4_bf16_nt_masked, py::arg("a"),
          py::arg("b"), py::arg("bias"), py::arg("d"), py::arg("masked_m"), py::arg("expected_m"),
          py::arg("configs") = std::nullopt, py::arg("max_block_n") = 256,
          py::arg("enable_sbo_overlap") = false, py::arg("signal") = std::nullopt);
    // TF32 GEMMs
    m.def("tf32_hc_prenorm_gemm", &tf32_hc_prenorm_gemm_nt, py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("sqr_sum"), py::arg("num_splits") = std::nullopt, py::arg("configs") = std::nullopt);
}

} // namespace deep_gemm::gemm
