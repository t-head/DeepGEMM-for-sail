#pragma once
#include <cmath>
#include <limits>
#include <torch/extension.h>
#include "../utils/utils.hpp"
#include "../jit_kernels/impls/activation.hpp"

namespace deep_gemm::activation {
static std::pair<int64_t, int64_t> check_input(const torch::Tensor& input, float swiglu_limit) {
    DG_HOST_ASSERT(is_ppu1v5_device());
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == torch::kBFloat16,
                "activation input must be contiguous BF16 on a PPU device");
    TORCH_CHECK(input.dim() >= 2 && input.size(-1) > 0 && input.size(-1) % 2 == 0,
                "activation input must have shape [..., 2*N]");
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0,
                "swiglu_limit must be finite and nonnegative");
    const int64_t n = input.size(-1) / 2;
    const int64_t rows = input.numel() / (2 * n);
    TORCH_CHECK(n <= std::numeric_limits<int>::max() / 2 && rows <= std::numeric_limits<int>::max(),
                "activation dimensions exceed kernel indexing limits");
    return {n, rows};
}
static ActivationMaskArgs check_mask(const torch::Tensor& input, const torch::Tensor& masked_m,
                                     const std::optional<int>& expected_m) {
    TORCH_CHECK(input.dim() == 3, "masked activation requires [E,T,2*N] input");
    TORCH_CHECK(masked_m.is_cuda() && masked_m.device() == input.device() && masked_m.is_contiguous() &&
                masked_m.scalar_type() == torch::kInt32 && masked_m.dim() == 1 &&
                masked_m.numel() == input.size(0), "masked_m must be contiguous int32 [E] on the input device");
    TORCH_CHECK(!expected_m.has_value() || *expected_m >= 0, "expected_m must be nonnegative");
    const int64_t capacity = input.size(1);
    TORCH_CHECK(capacity <= std::numeric_limits<int>::max(), "masked row capacity exceeds indexing limits");
    return make_activation_mask(input, masked_m, expected_m);
}
static ActivationSoftcapArgs check_softcap(float beta, const std::optional<float>& linear_beta) {
    TORCH_CHECK(std::isfinite(beta) && beta > 0, "SiTU beta must be finite and positive");
    TORCH_CHECK(!linear_beta || (std::isfinite(*linear_beta) && *linear_beta > 0),
                "SiTU linear_beta must be finite and positive, or None");
    return {true, beta, linear_beta.value_or(0.0f)};
}

enum class ActivationQuantMode { FP8_CHANNEL, MXFP4, MXFP8, FP8_BLOCK, BF16 };

static py::object dispatch_activation(
        ActivationQuantMode mode, const torch::Tensor& input,
        const std::optional<torch::Tensor>& masked_m, const std::optional<int>& expected_m,
        float limit, float inverse, const std::optional<bool>& reorder_scale,
        ActivationSoftcapArgs softcap = {}) {
    TORCH_CHECK(masked_m || !expected_m, "expected_m requires masked_m");
    TORCH_CHECK(mode == ActivationQuantMode::FP8_CHANNEL || inverse == 1.0f,
                "weight_inv_scale is only supported for FP8_CHANNEL");
    TORCH_CHECK(mode == ActivationQuantMode::MXFP4 || !reorder_scale,
                "reorder_scale is only supported for MXFP4");
    auto mask = masked_m ? check_mask(input, *masked_m, expected_m) : ActivationMaskArgs{};
    const auto [n, rows] = check_input(input, limit);
    if (mode == ActivationQuantMode::FP8_CHANNEL)
        TORCH_CHECK(std::isfinite(inverse) && inverse > 0, "inverse scale must be finite and positive");
    if (mode == ActivationQuantMode::MXFP4) {
        TORCH_CHECK(n % 2 == 0, "MXFP4 requires even output N");
        TORCH_CHECK(!reorder_scale.value_or(true) || input.dim() == 2 || input.dim() == 3,
                    "PPU1.5 scale layout requires rank 2 or 3");
    }
    if (mode == ActivationQuantMode::FP8_BLOCK)
        TORCH_CHECK(n % 128 == 0, "FP8_BLOCK requires output N divisible by 128");
    switch (mode) {
        case ActivationQuantMode::BF16:
            return py::cast(run_activation(input, n, rows, limit, mask, softcap));
        case ActivationQuantMode::FP8_CHANNEL:
            return py::cast(run_activation_fp8_channel(input, n, rows, limit, inverse, mask, softcap));
        case ActivationQuantMode::MXFP4:
            return py::cast(run_activation_mxfp4(input, n, rows, limit, reorder_scale.value_or(true), mask, softcap));
        case ActivationQuantMode::FP8_BLOCK:
            return py::cast(run_activation_tiled<true>(input, n, rows, limit, mask, softcap));
        case ActivationQuantMode::MXFP8:
            return py::cast(run_activation_mx<false>(input, n, rows, limit, mask, softcap));
    }
    TORCH_CHECK(false, "unsupported activation quantization mode");
}

static void register_apis(pybind11::module_& m) {
    py::enum_<ActivationQuantMode>(m, "ActivationQuantMode")
        .value("BF16", ActivationQuantMode::BF16)
        .value("FP8_CHANNEL", ActivationQuantMode::FP8_CHANNEL)
        .value("MXFP4", ActivationQuantMode::MXFP4)
        .value("MXFP8", ActivationQuantMode::MXFP8)
        .value("FP8_BLOCK", ActivationQuantMode::FP8_BLOCK);
    using Mask = std::optional<torch::Tensor>;
    using Hint = std::optional<int>;
    using Limit = std::optional<float>;
    using Reorder = std::optional<bool>;
    m.def("silu_and_mul", [](ActivationQuantMode mode, const torch::Tensor& input, Mask mask, Hint hint,
                              Limit limit, float inverse, Reorder reorder) {
        return dispatch_activation(mode, input, mask, hint, limit.value_or(0.0f), inverse, reorder);
    }, py::arg("mode"), py::arg("input"), py::kw_only(),
       py::arg("masked_m") = std::nullopt, py::arg("expected_m") = std::nullopt,
       py::arg("swiglu_limit") = std::nullopt, py::arg("weight_inv_scale") = 1.0f,
       py::arg("reorder_scale") = std::nullopt,
       "SiLU(gate)*up; mode is required; BF16 returns a tensor, other modes return (values, scales). "
       "FP8_BLOCK uses group128 FP32 scales (masked scales are M-major with rows aligned to 4). "
       "MXFP4 reorder_scale defaults to true (uint16 M-major); false returns raw uint8 scales.");
    m.def("situ_and_mul", [](ActivationQuantMode mode, const torch::Tensor& input, float beta, Limit linear_beta,
                              Mask mask, Hint hint, float inverse, Reorder reorder) {
        return dispatch_activation(mode, input, mask, hint, 0.0f, inverse, reorder,
                                   check_softcap(beta, linear_beta));
    }, py::arg("mode"), py::arg("input"), py::kw_only(), py::arg("beta"), py::arg("linear_beta") = std::nullopt,
       py::arg("masked_m") = std::nullopt,
       py::arg("expected_m") = std::nullopt, py::arg("weight_inv_scale") = 1.0f,
       py::arg("reorder_scale") = std::nullopt,
       "SiTU(gate)*up; mode is required; BF16 returns a tensor, other modes return (values, scales).");
}
} // namespace deep_gemm::activation
