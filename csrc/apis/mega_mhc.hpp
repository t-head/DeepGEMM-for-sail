#pragma once

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "../jit_kernels/impls/mega_mhc.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::mega_mhc {

static int select_num_splits(int tokens, int sms, int iterations) {
    if (iterations == 20 && tokens >= 128 && tokens <= 32768) {
        const int groups = ceil_div(tokens, 32), capacity = 8 * sms;
        if (tokens >= 1024 && tokens % 32 == 0)
            return groups * 9 <= capacity ? 18 : 16;
        for (int splits : {80, 40, 32, 16, 12, 10, 8})
            if (groups * (splits / 2) <= capacity) return splits;
        int selected = 16, best = INT_MAX;
        for (int splits : {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40}) {
            const int waves = ceil_div(groups * (splits / 2), capacity);
            const int cost = 3 * waves * ceil_div(160, splits) + splits;
            if (cost <= best) { best = cost; selected = splits; }
        }
        const int s16_waves = ceil_div(groups * 8, capacity);
        if (groups > capacity && 30 * s16_waves + 16 <= best + 2) selected = 16;
        return selected;
    }
    const int blocks = std::max(1, ceil_div(tokens, 64));
    if (sms == 39 && blocks <= 2) return 80 / blocks;
    if (sms == 39 && blocks == 16) return 8;
    return std::max(1, std::min(16, 20 * sms / blocks));
}

static MegaMhcPlan select_mega_mhc_plan(int tokens, int hidden, int iterations, int sms,
                                       bool tc01, bool bf16, bool fp8,
                                       int shared_sf_block_m, bool per_token_fp8, int physical_sms) {
    DG_HOST_ASSERT(tokens > 0 and sms > 0);
    int splits = tc01 ? 0 : select_num_splits(tokens, sms, iterations);
    if (tc01) {
        splits = 80;
        if (tokens >= 128) {
            int64_t best = INT64_MAX;
            for (int candidate : {4, 5, 6, 7, 8, 10, 12, 16, 20, 32, 40, 80}) {
                const int64_t waves = ceil_div<int64_t>(ceil_div(tokens, 64) * int64_t(candidate), 8 * sms);
                const int64_t cost = waves * (64 * ceil_div(80, candidate) + 20) + candidate;
                if (cost < best) { best = cost; splits = candidate; }
            }
        }
    }
    const bool ring = !tc01 && splits >= 8 && splits <= 80 && splits % 2 == 0 &&
                      tokens >= 128 && tokens <= 32768;
    const int ring_m = splits == 16 && tokens < 1024 && ceil_div(tokens, 16) <= sms ? 16 : 32;
    const bool single = iterations == 20 && ring;
    const int groups = ceil_div(tokens, ring_m), capacity = 8 * sms;
    const bool local = single && groups * (splits / 2) <= capacity;
    int task_splits = 2;
    if (single && !local && tokens % 32 != 0) {
        const int fine_waves = ceil_div(groups * splits, capacity);
        const int pair_waves = ceil_div(groups * (splits / 2), capacity);
        if (100 * fine_waves * ceil_div(80, splits) <
            98 * pair_waves * ceil_div(160, splits)) task_splits = 1;
    }
    MegaMhcPlan plan{
        {hidden, splits, iterations, ring_m, task_splits, tokens % 64 == 0,
         single, local, tokens % ring_m != 0, tokens % (ring_m / 8) != 0, local || tokens != 1248,
         bf16, fp8, shared_sf_block_m, per_token_fp8},
        {dim3(local ? groups : single ? std::min(groups * (splits / task_splits), capacity) :
              std::min(4 * sms, ceil_div(tokens, 64) * splits), local ? splits / 2 : 1),
         dim3(single ? ring_m * 2 : 128), 0}, sms,
        tc01 ? 0 : ceil_div<int64_t>(tokens, ring ? ring_m : 64) + 2,
        ceil_div<int64_t>(tokens, 64) * splits * 64 * 26 + (tc01 ? 24 * 4 * hidden : 0) +
            (ring ? (align<int64_t>(tokens, ring_m) - tokens) * 5 * hidden / 2 : 0)};
    auto& c = plan.config;
    c.tc01 = tc01;
    if (tc01) {
        // Pre-rounding amortizes a weight conversion over at least three M tiles.
        c.tc01_round_b = ceil_div(tokens, 64) >= 3;
        // Favor higher occupancy once the wider BF16 tile fills a resident wave.
        c.tc01_pre_chunk = tokens > 8 * physical_sms ? 512 : 1024;
        plan.pre_launch = {dim3(tokens), dim3(128), 0};
        plan.finish_launch = {dim3(ceil_div(tokens, splits < 32 ? 8 : 4)), dim3(128), 0};
        plan.launch_args = {dim3(ceil_div(tokens, 64), splits), dim3(128), 0};
    }
    return plan;
}

static void mega_mhc(
    const torch::Tensor& x, const torch::Tensor& residual,
    const torch::Tensor& shifted_prev_mix,
    const torch::Tensor& post_mix, const torch::Tensor& comb_res_mix,
    const torch::Tensor& fn, const torch::Tensor& mix_scales, const torch::Tensor& mix_bases,
    int hc_mult, float hc_norm_eps, float hc_pre_eps, float hc_post_scale,
    float sinkhorn_eps, int num_sinkhorn_iters, const torch::Tensor& rmsnorm_weight,
    float rmsnorm_eps, float rmsnorm_scale, const torch::Tensor& new_residual,
    const torch::Tensor& new_prev_mix, const torch::Tensor& new_post_mix,
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
    DG_HOST_ASSERT(x.dim() == 2 and x.is_cuda());
    const int64_t t64 = x.size(0), h64 = x.size(1);
    DG_HOST_ASSERT(h64 == 5120);
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
    check_dense(shifted_prev_mix, {t, 4, 1}, torch::kFloat);
    check_dense(new_prev_mix, {t, 4, 1}, torch::kFloat);
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
                          shifted_prev_mix, post_mix, comb_res_mix}) {
        DG_HOST_ASSERT(aligned(v, 16));
    }
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
    for (const auto& v : {new_residual, new_post_mix, new_comb_res_mix, new_prev_mix}) {
        const auto range = byte_range(v);
        check_overlap(range);
        output_ranges.push_back(range);
    }
    for (const auto& v : {y_bf16, y_fp8, y_gemm_sf, y_routed_sf, y_shared_sf, y_token_sf}) {
        if (!v) continue;
        auto range = byte_range(*v);
        if (y_shared_sf && v->data_ptr() == y_shared_sf->data_ptr())
            range.second = range.first + (shared_rows + (h / 128 - 1) * v->stride(1)) * sizeof(int32_t);
        check_overlap(range);
        output_ranges.push_back(range);
    }
    for (const auto& v : {x, residual, post_mix, comb_res_mix, fn,
                          mix_scales, mix_bases, rmsnorm_weight, shifted_prev_mix}) {
        check_overlap(byte_range(v));
    }
    const c10::cuda::CUDAGuard guard(x.device());
    const auto arch = get_ppu_arch();
    DG_HOST_ASSERT(arch == AC_PPU0010 or arch == AC_PPU0015);
    const bool tc01 = arch == AC_PPU0010;
    if (tc01) DG_HOST_ASSERT(t64 <= INT_MAX / 26);
    DG_HOST_ASSERT((not tc01 or not y_fp8.has_value()) and
                   "810E Mega-mHC supports BF16 output only");
    const auto plan = select_mega_mhc_plan(t, h, num_sinkhorn_iters, get_num_sms(), tc01,
        y_bf16.has_value(), y_fp8.has_value(), shared_sf_block_m, y_token_sf.has_value(),
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    const auto& y_primary_sf = y_token_sf.has_value() ? y_token_sf :
                               y_routed_sf.has_value() ? y_routed_sf : y_gemm_sf;
    mhc::Args args{
        static_cast<const __ppu_bfloat16*>(x.data_ptr()), static_cast<const __ppu_bfloat16*>(residual.data_ptr()),
        static_cast<const __ppu_bfloat16*>(rmsnorm_weight.data_ptr()), shifted_prev_mix.data_ptr<float>(), post_mix.data_ptr<float>(),
        comb_res_mix.data_ptr<float>(), fn.data_ptr<float>(), mix_scales.data_ptr<float>(), mix_bases.data_ptr<float>(),
        static_cast<__ppu_bfloat16*>(new_residual.data_ptr()), y_bf16 ? static_cast<__ppu_bfloat16*>(y_bf16->data_ptr()) : nullptr,
        new_prev_mix.data_ptr<float>(), new_post_mix.data_ptr<float>(), new_comb_res_mix.data_ptr<float>(),
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
