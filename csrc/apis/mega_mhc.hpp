#pragma once

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <climits>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "../jit_kernels/impls/mega_mhc.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::mega_mhc {

static void mega_mhc(
    const torch::Tensor& x, const torch::Tensor& residual,
    const std::optional<torch::Tensor>& shifted_prev_mix,
    const torch::Tensor& post_mix, const torch::Tensor& comb_res_mix,
    const torch::Tensor& fn, const torch::Tensor& mix_scales, const torch::Tensor& mix_bases,
    int hc_mult, float hc_norm_eps, float hc_pre_eps, float hc_post_scale,
    float sinkhorn_eps, int num_sinkhorn_iters, const torch::Tensor& rmsnorm_weight,
    float rmsnorm_eps, float rmsnorm_scale, const torch::Tensor& new_residual,
    const std::optional<torch::Tensor>& new_prev_mix, const torch::Tensor& new_post_mix,
    const torch::Tensor& new_comb_res_mix, const std::optional<torch::Tensor>& y_bf16,
    const std::optional<torch::Tensor>& y_fp8, const std::optional<torch::Tensor>& y_gemm_sf,
    const std::optional<torch::Tensor>& y_routed_sf, const std::optional<torch::Tensor>& y_shared_sf,
    int shared_sf_block_m, const std::optional<torch::Tensor>& y_token_sf) {
    DG_HOST_ASSERT(y_routed_sf.has_value() == y_shared_sf.has_value());
    DG_HOST_ASSERT(y_bf16.has_value() or y_fp8.has_value());
    const int num_sf = y_gemm_sf.has_value() + y_routed_sf.has_value() + y_token_sf.has_value();
    DG_HOST_ASSERT(num_sf == static_cast<int>(y_fp8.has_value()));
    DG_HOST_ASSERT(y_shared_sf.has_value() ? shared_sf_block_m > 0 : shared_sf_block_m == 0);
    DG_HOST_ASSERT(hc_mult == 4);
    const bool shifted = shifted_prev_mix.has_value();
    DG_HOST_ASSERT(shifted == new_prev_mix.has_value());
    DG_HOST_ASSERT(x.dim() == 2 and x.is_cuda());
    const int64_t t64 = x.size(0), h64 = x.size(1);
    DG_HOST_ASSERT(shifted ? h64 == 5120 : (h64 == 4096 or h64 == 7168));
    DG_HOST_ASSERT(t64 <= INT_MAX);
    const int t = t64, h = h64;
    DG_HOST_ASSERT(num_sinkhorn_iters >= 1);
    DG_HOST_ASSERT(std::isfinite(hc_norm_eps) and std::isfinite(hc_pre_eps) and
                   std::isfinite(hc_post_scale) and std::isfinite(sinkhorn_eps) and
                   std::isfinite(rmsnorm_eps) and std::isfinite(rmsnorm_scale));
    DG_HOST_ASSERT(hc_norm_eps > 0 and sinkhorn_eps > 0 and rmsnorm_eps > 0);
    const auto check_dense = [&](const torch::Tensor& v, at::IntArrayRef shape, torch::ScalarType dtype) {
        DG_HOST_ASSERT(v.sizes() == shape);
        DG_HOST_ASSERT(v.scalar_type() == dtype);
        DG_HOST_ASSERT(v.is_contiguous());
        DG_HOST_ASSERT(v.device() == x.device());
    };
    check_dense(x, {t, h}, torch::kBFloat16);
    check_dense(residual, {t, 4, h}, torch::kBFloat16);
    check_dense(post_mix, {t, 4, 1}, torch::kFloat);
    check_dense(comb_res_mix, {t, 4, 4}, torch::kFloat);
    check_dense(fn, {24, 4 * h}, torch::kFloat);
    check_dense(mix_scales, {3}, torch::kFloat);
    check_dense(mix_bases, {24}, torch::kFloat);
    check_dense(rmsnorm_weight, {h}, torch::kBFloat16);
    check_dense(new_residual, {t, 4, h}, torch::kBFloat16);
    check_dense(new_post_mix, {t, 4, 1}, torch::kFloat);
    check_dense(new_comb_res_mix, {t, 4, 4}, torch::kFloat);
    if (shifted) {
        check_dense(*shifted_prev_mix, {t, 4, 1}, torch::kFloat);
        check_dense(*new_prev_mix, {t, 4, 1}, torch::kFloat);
    }
    if (y_bf16.has_value()) check_dense(*y_bf16, {t, h}, torch::kBFloat16);
    if (y_fp8.has_value()) check_dense(*y_fp8, {t, h}, torch::kFloat8_e4m3fn);
    if (y_token_sf.has_value()) check_dense(*y_token_sf, {t, 1}, torch::kFloat);
    const int64_t shared_rows = y_shared_sf.has_value() ?
        ceil_div<int64_t>(t64, shared_sf_block_m) * align<int64_t>(shared_sf_block_m, 128) : 0;
    for (const auto& sf : {y_gemm_sf, y_routed_sf, y_shared_sf}) {
        if (sf.has_value()) {
            DG_HOST_ASSERT(sf->sizes() == at::IntArrayRef({t, h / 128}));
            DG_HOST_ASSERT(sf->scalar_type() == torch::kInt);
            DG_HOST_ASSERT(sf->device() == x.device());
        }
    }
    if (y_gemm_sf.has_value()) {
        DG_HOST_ASSERT(y_gemm_sf->stride(0) == 1);
        DG_HOST_ASSERT(y_gemm_sf->stride(1) >= t);
        DG_HOST_ASSERT(y_gemm_sf->stride(1) % get_tma_aligned_size(1, sizeof(uint32_t)) == 0);
    }
    if (y_routed_sf.has_value())
        DG_HOST_ASSERT(y_routed_sf->is_contiguous());
    if (y_shared_sf.has_value()) {
        DG_HOST_ASSERT(y_shared_sf->stride(0) == 1);
        DG_HOST_ASSERT(y_shared_sf->stride(1) >= shared_rows);
        const auto required_elements = y_shared_sf->storage_offset() + shared_rows +
                                       (h / 128 - 1) * y_shared_sf->stride(1);
        const auto required_bytes = static_cast<uint64_t>(required_elements) * y_shared_sf->element_size();
        DG_HOST_ASSERT(required_bytes <= y_shared_sf->storage().nbytes());
    }
    if (!t) return;
    const auto aligned = [](const torch::Tensor& v, uintptr_t bytes) {
        return reinterpret_cast<uintptr_t>(v.data_ptr()) % bytes == 0;
    };
    for (const auto& v : {x, residual, fn, rmsnorm_weight, new_residual,
                          post_mix, comb_res_mix}) {
        DG_HOST_ASSERT(aligned(v, 16));
    }
    if (shifted) DG_HOST_ASSERT(aligned(*shifted_prev_mix, 16));
    if (y_bf16.has_value()) DG_HOST_ASSERT(aligned(*y_bf16, 16));
    if (y_fp8.has_value()) DG_HOST_ASSERT(aligned(*y_fp8, 8));

    using ByteRange = std::pair<uintptr_t, uintptr_t>;
    const auto byte_range = [](const torch::Tensor& v) -> ByteRange {
        const auto begin = reinterpret_cast<uintptr_t>(v.data_ptr());
        int64_t span = 1;
        for (int i = 0; i < v.dim(); ++i) span += (v.size(i) - 1) * v.stride(i);
        return {begin, begin + span * v.element_size()};
    };
    std::vector<ByteRange> output_ranges;
    const auto check_overlap = [&](const ByteRange& range) {
        for (const auto& output : output_ranges) {
            DG_HOST_ASSERT(range.first >= output.second or output.first >= range.second);
        }
    };
    for (const auto& v : {new_residual, new_post_mix, new_comb_res_mix}) {
        const auto range = byte_range(v);
        check_overlap(range);
        output_ranges.push_back(range);
    }
    for (const auto& v : {new_prev_mix, y_bf16, y_fp8, y_gemm_sf, y_routed_sf, y_shared_sf, y_token_sf}) {
        if (!v) continue;
        auto range = byte_range(*v);
        if (y_shared_sf && v->data_ptr() == y_shared_sf->data_ptr())
            range.second = range.first + (shared_rows + (h / 128 - 1) * v->stride(1)) * sizeof(int32_t);
        check_overlap(range);
        output_ranges.push_back(range);
    }
    for (const auto& v : {x, residual, post_mix, comb_res_mix, fn,
                          mix_scales, mix_bases, rmsnorm_weight}) {
        check_overlap(byte_range(v));
    }
    if (shifted) check_overlap(byte_range(*shifted_prev_mix));
    const c10::cuda::CUDAGuard guard(x.device());
    const auto arch = get_ppu_arch();
    DG_HOST_ASSERT(arch == AC_PPU0010 or arch == AC_PPU0015);
    const bool tc01 = arch == AC_PPU0010;
    DG_HOST_ASSERT((not tc01 or shifted) and "Normal Mega-mHC requires M890");
    if (tc01) DG_HOST_ASSERT(t64 <= INT_MAX / 26);
    DG_HOST_ASSERT((not tc01 or not y_fp8.has_value()) and
                   "810E Mega-mHC supports BF16 output only");
    const auto plan = select_mega_mhc_plan(t, h, num_sinkhorn_iters, get_num_sms(), tc01,
        y_bf16.has_value(), y_fp8.has_value(), shared_sf_block_m, y_token_sf.has_value(),
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount, shifted);
    const auto& y_primary_sf = y_token_sf.has_value() ? y_token_sf :
                               y_routed_sf.has_value() ? y_routed_sf : y_gemm_sf;
    mhc::Args args{
        static_cast<const __ppu_bfloat16*>(x.data_ptr()), static_cast<const __ppu_bfloat16*>(residual.data_ptr()),
        static_cast<const __ppu_bfloat16*>(rmsnorm_weight.data_ptr()), shifted ? shifted_prev_mix->data_ptr<float>() : nullptr, post_mix.data_ptr<float>(),
        comb_res_mix.data_ptr<float>(), fn.data_ptr<float>(), mix_scales.data_ptr<float>(), mix_bases.data_ptr<float>(),
        static_cast<__ppu_bfloat16*>(new_residual.data_ptr()), y_bf16 ? static_cast<__ppu_bfloat16*>(y_bf16->data_ptr()) : nullptr,
        shifted ? new_prev_mix->data_ptr<float>() : nullptr, new_post_mix.data_ptr<float>(), new_comb_res_mix.data_ptr<float>(),
        nullptr, nullptr, t,
        hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, rmsnorm_eps, rmsnorm_scale,
        y_fp8 ? static_cast<uint8_t*>(y_fp8->data_ptr()) : nullptr,
        y_primary_sf ? static_cast<int32_t*>(y_primary_sf->data_ptr()) : nullptr,
        y_shared_sf ? y_shared_sf->data_ptr<int32_t>() : nullptr,
        y_primary_sf ? y_primary_sf->stride(0) : 0, y_primary_sf ? y_primary_sf->stride(1) : 0,
        y_shared_sf ? y_shared_sf->stride(1) : 0};
    run_mega_mhc(plan, args, x.options());
}

static void register_apis(pybind11::module_& m) {
    m.def("mega_mhc", &mega_mhc,
          py::arg("x"), py::arg("residual"), py::arg("shifted_prev_mix"), py::arg("post_mix"),
          py::arg("comb_res_mix"), py::arg("fn"), py::arg("mix_scales"), py::arg("mix_bases"),
          py::arg("hc_mult"), py::arg("hc_norm_eps"), py::arg("hc_pre_eps"), py::arg("hc_post_scale"),
          py::arg("sinkhorn_eps"), py::arg("num_sinkhorn_iters"), py::arg("rmsnorm_weight"),
          py::arg("rmsnorm_eps"), py::arg("rmsnorm_scale"), py::arg("new_residual"), py::arg("new_prev_mix"),
          py::arg("new_post_mix"), py::arg("new_comb_res_mix"), py::arg("y_bf16") = std::nullopt,
          py::arg("y_fp8") = std::nullopt, py::arg("y_gemm_sf") = std::nullopt,
          py::arg("y_routed_sf") = std::nullopt, py::arg("y_shared_sf") = std::nullopt,
          py::arg("shared_sf_block_m") = 0, py::arg("y_token_sf") = std::nullopt);
}

} // namespace deep_gemm::mega_mhc
