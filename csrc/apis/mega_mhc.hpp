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
    // Scratch for the computed pre when the caller provides no buffer.
    torch::Tensor new_pre;
};
using WorkspaceCache = std::list<std::pair<uintptr_t, Workspace>>;
static constexpr size_t kWorkspaceStreams = 16;
static std::unordered_map<int, WorkspaceCache> workspaces;
static std::mutex submit_mutex;

// Per-hidden-size contract. A shape that reuses the non-shift kernel needs only
// another row here; the shifted schedule is planned by plan_shifted().
struct Spec {
    int hidden;
    bool non_shift;
    bool requires_prev;
    int64_t max_rows;
    int splits, iterations, tile_m, task_splits;  // non-shift plan only
};
static constexpr Spec kSpecs[] = {
    {4096, true, false, 131072, 8, 20, 32, 2},
    {5120, false, true, INT_MAX, 0, 0, 0, 0},
};

struct Plan {
    MegaMhcRuntime::Config config;
    dim3 grid, block;
    int64_t partials, counters;
};

static Plan plan_non_shift(const Spec& spec, int tokens) {
    const int blocks = (tokens + 63) / 64;
    return {{spec.hidden, spec.splits, spec.iterations, spec.tile_m, spec.task_splits,
             false, false, false, false, false, true, true},
            dim3(std::min(156, blocks * 8)), dim3(128),
            static_cast<int64_t>(blocks) * spec.splits * 64 * 26, blocks + 2};
}

static Plan plan_shifted(int hidden, int tokens, int sms, int iterations) {
    const int splits = select_num_splits(tokens, sms, iterations);
    const bool ring = (splits >= 8 && splits <= 80 && splits % 2 == 0) &&
                      tokens >= 128 && tokens <= 32768;
    const int ring_m = splits == 16 && tokens < 1024 && (tokens + 15) / 16 <= sms ? 16 : 32;
    const bool single = iterations == 20 && ring;
    const int groups = (tokens + ring_m - 1) / ring_m, capacity = 8 * sms;
    const bool local = single && groups * (splits / 2) <= capacity;
    int task_splits = 2;
    if (single && !local && tokens % 32 != 0) {
        const int fine_waves = (groups * splits + capacity - 1) / capacity;
        const int pair_waves = (groups * (splits / 2) + capacity - 1) / capacity;
        if (100 * fine_waves * ((80 + splits - 1) / splits) <
            98 * pair_waves * ((160 + splits - 1) / splits))
            task_splits = 1;
    }
    const int blocks = (tokens + 63) / 64;
    return {{hidden, splits, iterations, ring_m, task_splits,
             tokens % 64 == 0, single, local, tokens % ring_m != 0,
             tokens % (ring_m / 8) != 0, local || tokens != 1248},
            dim3(local ? groups
                       : single ? std::min(groups * (splits / task_splits), capacity)
                                : std::min(4 * sms, blocks * splits),
                 local ? splits / 2 : 1),
            dim3(single ? ring_m * 2 : 128),
            static_cast<int64_t>(blocks) * splits * 64 * 26 +
                (ring ? (ring_m - tokens % ring_m) % ring_m * 5 * hidden / 2 : 0),
            (ring ? (tokens + ring_m - 1) / ring_m : blocks) + 2};
}

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
    int shared_sf_block_m) {
    if (x.dim() != 2 || !x.is_cuda())
        throw py::value_error("MegaMHC requires device x[T,H]");
    if (hc_mult != 4)
        throw py::value_error("MegaMHC requires hc_mult=4");
    if (!y_bf16)
        throw py::value_error("MegaMHC requires y_bf16 output");
    if (y_fp8 || y_gemm_sf || y_routed_sf || y_shared_sf || shared_sf_block_m != 0)
        throw py::value_error("MegaMHC only supports bf16 output");

    const int64_t t64 = x.size(0), h = x.size(1);
    const Spec* spec = nullptr;
    for (const auto& candidate : kSpecs)
        if (candidate.hidden == h) spec = &candidate;
    if (!spec)
        throw py::value_error("MegaMHC only supports hidden sizes 4096 and 5120");
    if (t64 > spec->max_rows || (spec->non_shift && t64 == 0))
        throw py::value_error("MegaMHC token count out of range for this hidden size");
    if (spec->requires_prev && (!shifted_prev_mix || !new_prev_mix))
        throw py::value_error("Shifted MegaMHC requires shifted_prev_mix and new_prev_mix");
    if (num_sinkhorn_iters < 1 || !std::isfinite(hc_norm_eps) || !std::isfinite(hc_pre_eps) ||
        !std::isfinite(hc_post_scale) || !std::isfinite(sinkhorn_eps) ||
        !std::isfinite(rmsnorm_eps) || !std::isfinite(rmsnorm_scale))
        throw py::value_error("Invalid iteration count or scalar");
    if (hc_norm_eps <= 0 || sinkhorn_eps <= 0 || rmsnorm_eps <= 0)
        throw py::value_error("MegaMHC epsilons must be positive");
    if (spec->non_shift && (num_sinkhorn_iters != spec->iterations ||
                            hc_post_scale != 2.f || rmsnorm_scale != 1.f))
        throw py::value_error(
            "Non-shifted MegaMHC fixes 20 Sinkhorn iterations, hc_post_scale=2, rmsnorm_scale=1");

    const auto dense = [&](const torch::Tensor& v, at::IntArrayRef shape, torch::ScalarType dtype) {
        if (v.sizes() != shape || v.scalar_type() != dtype || !v.is_contiguous() ||
            v.device() != x.device())
            throw py::value_error("Invalid MegaMHC tensor shape, dtype, strides or device");
    };
    dense(x, {t64, h}, torch::kBFloat16);
    dense(residual, {t64, 4, h}, torch::kBFloat16);
    dense(post_mix, {t64, 4, 1}, torch::kFloat);
    dense(comb_res_mix, {t64, 4, 4}, torch::kFloat);
    dense(fn, {24, 4 * h}, torch::kFloat);
    dense(mix_scales, {3}, torch::kFloat);
    dense(mix_bases, {24}, torch::kFloat);
    dense(rmsnorm_weight, {h}, torch::kBFloat16);
    dense(new_residual, {t64, 4, h}, torch::kBFloat16);
    dense(new_post_mix, {t64, 4, 1}, torch::kFloat);
    dense(new_comb_res_mix, {t64, 4, 4}, torch::kFloat);
    dense(*y_bf16, {t64, h}, torch::kBFloat16);
    if (shifted_prev_mix) dense(*shifted_prev_mix, {t64, 4, 1}, torch::kFloat);
    if (new_prev_mix) dense(*new_prev_mix, {t64, 4, 1}, torch::kFloat);
    if (!t64) return;

    const auto require_aligned = [](const torch::Tensor& v) {
        if (reinterpret_cast<uintptr_t>(v.data_ptr()) % 16)
            throw py::value_error("MegaMHC tensors must be 16-byte aligned");
    };
    for (const auto* v : {&x, &residual, &post_mix, &comb_res_mix, &fn, &mix_scales,
                          &mix_bases, &rmsnorm_weight, &new_residual, &new_post_mix,
                          &new_comb_res_mix, &*y_bf16})
        require_aligned(*v);
    if (shifted_prev_mix) require_aligned(*shifted_prev_mix);
    if (new_prev_mix) require_aligned(*new_prev_mix);

    using ByteRange = std::pair<uintptr_t, uintptr_t>;
    const auto byte_range = [](const torch::Tensor& v) -> ByteRange {
        const auto begin = reinterpret_cast<uintptr_t>(v.data_ptr());
        return {begin, begin + v.numel() * v.element_size()};
    };
    std::vector<ByteRange> outputs;
    const auto forbid_overlap = [&](const torch::Tensor& v) {
        const auto range = byte_range(v);
        for (const auto& other : outputs)
            if (range.first < other.second && other.first < range.second)
                throw py::value_error("MegaMHC outputs must not overlap inputs or other outputs");
    };
    const auto add_output = [&](const torch::Tensor& v) {
        forbid_overlap(v);
        outputs.push_back(byte_range(v));
    };
    add_output(new_residual);
    add_output(new_post_mix);
    add_output(new_comb_res_mix);
    add_output(*y_bf16);
    if (new_prev_mix) add_output(*new_prev_mix);
    for (const auto* v : {&x, &residual, &post_mix, &comb_res_mix, &fn, &mix_scales,
                          &mix_bases, &rmsnorm_weight})
        forbid_overlap(*v);
    if (shifted_prev_mix) forbid_overlap(*shifted_prev_mix);

    const c10::cuda::CUDAGuard guard(x.device());
    std::lock_guard<std::mutex> lock(submit_mutex);
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    if (prop->major != 8 || prop->minor != 9 ||
        std::string(prop->name).find("M890") == std::string::npos)
        throw std::runtime_error("MegaMHC requires Zhenwu M890P");

    const int t = static_cast<int>(t64);
    const int sms = get_num_sms();
    const Plan plan = spec->non_shift
        ? plan_non_shift(*spec, t)
        : plan_shifted(static_cast<int>(h), t, sms, num_sinkhorn_iters);

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
    ensure_capacity(ws.partials, plan.partials, torch::kFloat);
    ensure_capacity(ws.counters, plan.counters, torch::kInt, true);

    float* pre = spec->requires_prev ? shifted_prev_mix->data_ptr<float>() : nullptr;
    float* npre;
    if (new_prev_mix) {
        npre = new_prev_mix->data_ptr<float>();
    } else {
        ensure_capacity(ws.new_pre, t64 * 4, torch::kFloat);
        npre = ws.new_pre.data_ptr<float>();
    }

    MegaMhcRuntime::Args args{
        plan.config,
        {plan.grid, plan.block, 0},
        {static_cast<const __ppu_bfloat16*>(x.data_ptr()),
         static_cast<const __ppu_bfloat16*>(residual.data_ptr()),
         static_cast<const __ppu_bfloat16*>(rmsnorm_weight.data_ptr()), pre,
         post_mix.data_ptr<float>(), comb_res_mix.data_ptr<float>(),
         fn.data_ptr<float>(), mix_scales.data_ptr<float>(), mix_bases.data_ptr<float>(),
         static_cast<__ppu_bfloat16*>(new_residual.data_ptr()),
         static_cast<__ppu_bfloat16*>(y_bf16->data_ptr()), npre,
         new_post_mix.data_ptr<float>(), new_comb_res_mix.data_ptr<float>(),
         ws.partials.data_ptr<float>(), ws.counters.data_ptr<int>(), t,
         hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, rmsnorm_eps, rmsnorm_scale},
        false};
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
