#pragma once

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../jit_kernels/impls/mega_mhc.hpp"

namespace deep_gemm::mega_mhc {

static int select_num_splits(int tokens, int sms, int iterations) {
    if (iterations == 20 && tokens >= 128 && tokens <= 32768) {
        const int groups = (tokens + 31) / 32, capacity = 8 * sms;
        if (tokens >= 1024 && tokens % 32 == 0)
            return groups * 9 <= capacity ? 18 : 16;
        for (int splits : {80, 40, 32, 16, 12, 10, 8})
            if (groups * (splits / 2) <= capacity) return splits;
        int selected = 16, best = INT_MAX;
        for (int splits : {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40}) {
            const int waves = (groups * (splits / 2) + capacity - 1) / capacity;
            const int cost = 3 * waves * ((160 + splits - 1) / splits) + splits;
            if (cost <= best) { best = cost; selected = splits; }
        }
        const int s16_waves = (groups * 8 + capacity - 1) / capacity;
        if (groups > capacity && 30 * s16_waves + 16 <= best + 2) selected = 16;
        return selected;
    }
    const int blocks = std::max(1, (tokens + 63) / 64);
    if (sms == 39 && blocks <= 2) return 80 / blocks;
    if (sms == 39 && blocks == 16) return 8;
    return std::max(1, std::min(16, 20 * sms / blocks));
}

struct Workspace {
    torch::Tensor partials, counters;
};
using WorkspaceCache = std::list<std::pair<uintptr_t, Workspace>>;
static constexpr size_t kWorkspaceStreams = 16;
static std::unordered_map<int, WorkspaceCache> workspaces;
static std::mutex submit_mutex;

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
    int shared_sf_block_m) {
    if (y_fp8 || y_gemm_sf || y_routed_sf || y_shared_sf || shared_sf_block_m != 0)
        throw py::value_error("Mega-mHC only supports shifted BF16 output; FP8 output is not supported");
    if (!y_bf16) throw py::value_error("Mega-mHC requires y_bf16 output");
    if (hc_mult != 4 || x.dim() != 2 || !x.is_cuda())
        throw py::value_error("Mega-mHC requires device x[T,H] and hc_mult=4");
    const int64_t t64 = x.size(0), h64 = x.size(1);
    if (h64 != 5120)
        throw py::value_error("Mega-mHC requires V4.1 Flash hidden=5120");
    if (t64 > INT_MAX) throw py::value_error("Too many tokens");
    const int t = t64, h = h64;
    if (num_sinkhorn_iters < 1 || !std::isfinite(hc_norm_eps) || !std::isfinite(hc_pre_eps) ||
        !std::isfinite(hc_post_scale) || !std::isfinite(sinkhorn_eps) ||
        !std::isfinite(rmsnorm_eps) || !std::isfinite(rmsnorm_scale))
        throw py::value_error("Invalid iteration count or scalar");
    if (hc_norm_eps <= 0 || sinkhorn_eps <= 0 || rmsnorm_eps <= 0)
        throw py::value_error("RMS and Sinkhorn epsilons must be positive");
    const auto dense = [&](const torch::Tensor& v, at::IntArrayRef shape, torch::ScalarType dtype) {
        if (v.sizes() != shape || v.scalar_type() != dtype || !v.is_contiguous() || v.device() != x.device())
            throw py::value_error("Invalid Mega-mHC tensor shape, dtype, strides or device");
    };
    dense(x, {t, h}, torch::kBFloat16);
    dense(residual, {t, 4, h}, torch::kBFloat16);
    dense(post_mix, {t, 4, 1}, torch::kFloat);
    dense(comb_res_mix, {t, 4, 4}, torch::kFloat);
    dense(fn, {24, 4 * h}, torch::kFloat);
    dense(mix_scales, {3}, torch::kFloat);
    dense(mix_bases, {24}, torch::kFloat);
    dense(rmsnorm_weight, {h}, torch::kBFloat16);
    dense(new_residual, {t, 4, h}, torch::kBFloat16);
    dense(new_post_mix, {t, 4, 1}, torch::kFloat);
    dense(new_comb_res_mix, {t, 4, 4}, torch::kFloat);
    dense(shifted_prev_mix, {t, 4, 1}, torch::kFloat);
    dense(new_prev_mix, {t, 4, 1}, torch::kFloat);
    dense(*y_bf16, {t, h}, torch::kBFloat16);
    if (!t) return;
    const auto aligned = [](const torch::Tensor& v, uintptr_t bytes) {
        return reinterpret_cast<uintptr_t>(v.data_ptr()) % bytes == 0;
    };
    for (const auto& v : {x, residual, fn, rmsnorm_weight, new_residual,
                          shifted_prev_mix, post_mix, comb_res_mix, *y_bf16}) {
        if (!aligned(v, 16))
            throw py::value_error("Mega-mHC vector inputs and outputs must be 16-byte aligned");
    }

    using ByteRange = std::pair<uintptr_t, uintptr_t>;
    const auto byte_range = [](const torch::Tensor& v) -> ByteRange {
        const auto begin = reinterpret_cast<uintptr_t>(v.data_ptr());
        return {begin, begin + v.numel() * v.element_size()};
    };
    std::vector<ByteRange> output_ranges;
    const auto check_overlap = [&](const ByteRange& range) {
        for (const auto& output : output_ranges) {
            if (range.first < output.second && output.first < range.second)
                throw py::value_error("Mega-mHC outputs must not overlap inputs or other outputs");
        }
    };
    for (const auto& v : {new_residual, new_post_mix, new_comb_res_mix, *y_bf16, new_prev_mix}) {
        const auto range = byte_range(v);
        check_overlap(range);
        output_ranges.push_back(range);
    }
    for (const auto& v : {x, residual, post_mix, comb_res_mix, fn,
                          mix_scales, mix_bases, rmsnorm_weight, shifted_prev_mix}) {
        check_overlap(byte_range(v));
    }
    const c10::cuda::CUDAGuard guard(x.device());
    std::lock_guard<std::mutex> lock(submit_mutex);
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    if (prop->major != 8 || prop->minor != 9 || std::string(prop->name).find("M890") == std::string::npos)
        throw std::runtime_error("Mega-mHC requires Zhenwu M890P");
    const int sms = get_num_sms();
    const int splits = select_num_splits(t, sms, num_sinkhorn_iters);
    // Graph captures need private buffers to avoid concurrent replay races.
    Workspace graph_workspace;
    auto* selected = &graph_workspace;
    if (c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None) {
        auto& cache = workspaces[x.get_device()];
        const auto stream = reinterpret_cast<uintptr_t>(current_stream());
        const auto it = std::find_if(cache.begin(), cache.end(),
                                     [&](const auto& entry) { return entry.first == stream; });
        if (it == cache.end()) {
            cache.emplace_front(stream, Workspace{});
            if (cache.size() > kWorkspaceStreams) cache.pop_back();
        } else {
            cache.splice(cache.begin(), cache, it);
        }
        selected = &cache.front().second;
    }
    auto& ws = *selected;
    const auto ensure_capacity = [&](torch::Tensor& buffer, int64_t count, torch::ScalarType dtype,
                                     bool zero = false) {
        if (!buffer.defined() || buffer.numel() < count) {
            const auto options = x.options().dtype(dtype);
            buffer = zero ? torch::zeros({count}, options) : torch::empty({count}, options);
        }
    };
    const bool ring = (splits >= 8 && splits <= 80 && splits % 2 == 0) && t >= 128 && t <= 32768;
    const int ring_m = splits == 16 && t < 1024 && (t + 15) / 16 <= sms ? 16 : 32;
    ensure_capacity(ws.partials, ((t64 + 63) / 64) * splits * 64 * 26 +
        (ring ? (ring_m - t % ring_m) % ring_m * 5 * h / 2 : 0), torch::kFloat);
    ensure_capacity(ws.counters, (ring ? (t64 + ring_m - 1) / ring_m : (t64 + 63) / 64) + 2, torch::kInt, true);
    const auto& pre = shifted_prev_mix;
    const auto& npre = new_prev_mix;
    const auto& y = *y_bf16;
    const bool single = num_sinkhorn_iters == 20 && ring;
    const int groups = (t + ring_m - 1) / ring_m, capacity = 8 * sms;
    const bool local = single && groups * (splits / 2) <= capacity;
    int task_splits = 2;
    if (single && !local && t % 32 != 0) {
        const int fine_waves = (groups * splits + capacity - 1) / capacity;
        const int pair_waves = (groups * (splits / 2) + capacity - 1) / capacity;
        if (100 * fine_waves * ((80 + splits - 1) / splits) <
            98 * pair_waves * ((160 + splits - 1) / splits)) task_splits = 1;
    }
    MegaMhcRuntime::Args args{
        {h, splits, num_sinkhorn_iters, ring_m, task_splits, t % 64 == 0,
         single, local, t % ring_m != 0, t % (ring_m / 8) != 0, local || t != 1248},
        {dim3(local ? groups : single ? std::min(groups * (splits / task_splits), capacity) : std::min(4 * sms, (t + 63) / 64 * splits),
              local ? splits / 2 : 1),
         dim3(single ? ring_m * 2 : 128), 0},
        {static_cast<const __ppu_bfloat16*>(x.data_ptr()), static_cast<const __ppu_bfloat16*>(residual.data_ptr()),
         static_cast<const __ppu_bfloat16*>(rmsnorm_weight.data_ptr()), pre.data_ptr<float>(), post_mix.data_ptr<float>(),
         comb_res_mix.data_ptr<float>(), fn.data_ptr<float>(), mix_scales.data_ptr<float>(), mix_bases.data_ptr<float>(),
         static_cast<__ppu_bfloat16*>(new_residual.data_ptr()), static_cast<__ppu_bfloat16*>(y.data_ptr()),
         npre.data_ptr<float>(), new_post_mix.data_ptr<float>(), new_comb_res_mix.data_ptr<float>(),
         ws.partials.data_ptr<float>(), ws.counters.data_ptr<int>(), t,
         hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, rmsnorm_eps, rmsnorm_scale}};
    launch_mega_mhc(args, sms, prop->multiProcessorCount, prop->cooperativeLaunch);
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
          py::arg("shared_sf_block_m") = 0);
}

} // namespace deep_gemm::mega_mhc
