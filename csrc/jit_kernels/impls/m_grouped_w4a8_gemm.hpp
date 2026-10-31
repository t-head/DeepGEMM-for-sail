#pragma once
#include <torch/python.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <optional>
#include <map>
#include <mutex>
#include <string>
#include <set>
#include <type_traits>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_w4a8.hpp"
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include "w4a8_streamed_gemm.hpp"
#include "fp8_gemm.hpp" // `ComputeBlockInfoKernelRuntime`

namespace deep_gemm::w4fa8 {
static bool is_w4a8_capturing() {
    hggcStreamCaptureStatus status = hggcStreamCaptureStatusNone;
    DG_HGGC_RUNTIME_CHECK(hggcStreamIsCapturing(current_stream(), &status));
    return status != hggcStreamCaptureStatusNone;
}

// Host mirror of the device-only FusedGemmScheduler::Params.
struct W4A8FusedSchedulerArguments {
    const int32_t* aligned_num_m_blocks;
    const int32_t* expert_ids_and_cumsum;
};

template <class Scheduler> struct Arguments {
    const void* a;
    const void* b;
    const void* scale;
    void* d;
    const float* a_scale;
    int num_token;
    float inv_scale;
    const int* sorted_token_ids;
    Scheduler scheduler;
};

template <class Scheduler> class Runtime final : public LaunchRuntime<Runtime<Scheduler>> {
  public:
    struct Args {
        W4A8Config config;
        int n, k, groups;
        GemmType kind;
        LaunchArgs launch_args;
        Arguments<Scheduler> params;
        NativeStrategy strategy = NativeStrategy::Default;
        int device = 0;
        bool conditional = false;
        const int32_t* enabled = nullptr;
        std::pair<int, int> cache_variant() const { return {int(strategy) + 4 * int(conditional), int(kind)}; }
    };
    static const char* name() { return "m_grouped_gemm_w4fa8_mma"; }
    static std::string generate_impl(const Args& a) {
        const auto& c = a.config;
        return fmt::format(
            R"(
#include <deep_gemm/impls/w4fa8_gemm.cuh>
using Kernel = cutlass::gemm::kernel::W4A8GEMM_MMA<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::GemmType::{}, {}, {}, deep_gemm::w4fa8::NativeStrategy::{}, {}>;
static_assert(sizeof(Kernel::SharedStorage) == {}, "shared memory mismatch");
static_assert(Kernel::MaxThreadsPerBlock == {}, "thread count mismatch");
static_assert(sizeof(Kernel::Params) == {}, "kernel argument size mismatch");
extern "C" __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void m_grouped_gemm_w4fa8_mma(Kernel::Params params{}) {{
    extern __shared__ char smem[];
    {}
    Kernel{{}}(params, smem);
}}
)",
            a.n, a.k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, a.groups, c.num_stages,
            get_gemm_type_name(a.kind), c.n_expand, a.params.inv_scale != 1.0f ? "true" : "false",
            a.strategy == NativeStrategy::Routed    ? "Routed"
            : a.strategy == NativeStrategy::Compact ? "Compact"
                                                    : "Default",
            a.strategy == NativeStrategy::Routed
                ? fmt::format("deep_gemm::w4fa8::W4A8RoutedScheduler<{}, {}, {}, {}{}>", a.n, a.k, c.block_m, c.block_n,
                              a.kind == GemmType::MGroupedFused ? ", true" : "")
                : "void",
            a.launch_args.smem_size, get_num_threads(c), sizeof(Arguments<Scheduler>),
            a.conditional ? ", const int32_t* enabled" : "",
            a.conditional ? "if (__ldg(enabled) == 0) return;" : "");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if (args.conditional)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params, args.enabled));
        else
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static DgProfParam make_profile(GemmType kind, int groups, int m, int n, int k, int expected_m, int topk,
                                int32_t* rows) {
    DgProfParam prof;
    if (ProfilingInterface::Instance().get_op_info()) {
        if (kind == GemmType::MGroupedFused)
            prof.set_fused_moe_params("w4fa8_mma", "group", groups, m, topk, n, k, rows, current_stream());
        else {
            prof.set_params(kind, false, "w4fa8_mma", groups, m, n, k, expected_m, rows, current_stream());
            prof.add_params(std::string("quant_type"), std::string("group"));
        }
        prof.add_params(std::string("group_size"), 32);
    }
    return prof;
}

template <class Scheduler>
static void launch(const W4A8Config& c, int m, int n, int k, int groups, GemmType kind, int expected_m, int topk,
                   int32_t* m_rows, const Arguments<Scheduler>& params) {
    using R = Runtime<Scheduler>;
    const int threads = get_num_threads(c), smem = get_smem_size(c, k);
    typename R::Args args{c, n, k, groups, kind, {dim3(c.num_sms), dim3(threads), smem}, params};
    const auto& runtime = compiler->build("m_grouped_gemm_w4fa8_mma", R::generate(args), !is_w4a8_capturing());
    construct_launch_config(runtime->kernel, current_stream(), smem, args.launch_args.grid_dim,
                            args.launch_args.block_dim);
    int blocks = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, runtime->kernel, threads, smem));
    DG_HOST_ASSERT(blocks > 0);
    args.launch_args.grid_dim = dim3(c.num_sms < 20 ? c.num_sms : c.num_sms * std::min(8, blocks));
    auto prof = make_profile(kind, groups, m, n, k, expected_m, topk, m_rows);
    if (const char* log = std::getenv("show_log"); log && isdigit(*log)) {
        int regs = 0, stack = 0;
        hgFuncGetAttribute(&regs, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel);
        hgFuncGetAttribute(&stack, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel);
        printf("[W4FA8] group:%d problem:[%d,%d,%d] block:[%d,%d,%d] warp:[%d,%d,%d] stages:%d smem:%d vreg:%d "
               "stack:%d tb_per_sm:%d threads:%d n_expand:%d\n",
               groups, m, n, k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, c.num_stages, smem, regs,
               stack, blocks, threads, c.n_expand);
    }
    ProfilingInterface::Instance().instrument(true, prof);
    R::launch(runtime, args);
    ProfilingInterface::Instance().instrument(false, prof);
}

// Ordinary optimized warmup also prepares its native fallback. A different
// caller stream can then capture safely without creating streamed resources.
static void prepare_native_fallback(const W4A8Config& c, int n, int k, int groups, GemmType kind) {
    TORCH_CHECK(kind != GemmType::MGroupedNoPad || groups <= 1024, "W4FA8 nopad supports at most 1024 experts");
    check_config(c, n, k, kind);
    using Key = std::tuple<int, int, int, int, int, W4A8ConfigTuple>;
    static std::mutex mutex;
    static std::set<Key> ready;
    int device = 0;
    DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&device));
    const Key key{device, n, k, groups, int(kind), pack_config(c)};
    std::lock_guard<std::mutex> lock(mutex);
    if (ready.count(key) || is_w4a8_capturing())
        return;
    auto prepare = [&](auto scheduler) {
        using R = Runtime<decltype(scheduler)>;
        typename R::Args args{c, n, k, groups, kind,
                              {dim3(c.num_sms), dim3(get_num_threads(c)), get_smem_size(c, k)}, {}};
        args.params.inv_scale = 1.f;
        const auto runtime = compiler->build(R::name(), R::generate(args));
        construct_launch_config(runtime->kernel, current_stream(), args.launch_args.smem_size,
                                args.launch_args.grid_dim, args.launch_args.block_dim);
    };
    if (kind == GemmType::MGroupedFused)
        prepare(W4A8FusedSchedulerArguments{});
    else {
        prepare(TileSchedulerArguments{});
        if (groups >= 128 && groups <= 1024)
            compiler->build("computeBlockInfoKernel", ComputeBlockInfoKernelRuntime::generate(c.block_m));
    }
    ready.insert(key);
}

static torch::Tensor prepare_nopad_layout(const torch::Tensor& rows, int groups, int m, const W4A8Config& c) {
    if (groups < 128)
        return rows;
    TORCH_CHECK(groups <= 1024, "W4FA8 nopad supports at most 1024 experts");
    auto scratch = at::empty({(groups + ceil_div(m + 1 - groups, c.block_m)) * 4}, rows.options());
    auto args = ComputeBlockInfoKernelRuntime::Args{
        .launch_attr_args = {reinterpret_cast<const uint32_t*>(rows.data_ptr<int32_t>()), static_cast<uint32_t>(groups),
                             reinterpret_cast<uint32_t*>(scratch.data_ptr<int32_t>())},
        .launch_args = {1, std::max(32, next_power_of_two(groups)), 0}};
    const auto& runtime = compiler->build("computeBlockInfoKernel", ComputeBlockInfoKernelRuntime::generate(c.block_m),
                                          !is_w4a8_capturing());
    ComputeBlockInfoKernelRuntime::launch(runtime, args);
    return scratch;
}

static void run(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows, GemmType kind,
                int expected_m, const W4A8Config& c, float inv_scale = 1.0f, const torch::Tensor& expert_ids = {},
                const torch::Tensor& token_ids = {}, const torch::Tensor& aligned_blocks = {}) {
    TORCH_CHECK(std::isfinite(inv_scale) && inv_scale > 0, "inv_scale must be finite and positive");
    const int groups = b.size(0), n = b.size(1), k = a.size(-1);
    const int m = kind == GemmType::MGroupedMasked ? a.size(1) : a.size(0);
    check_config(c, n, k, kind);
    if (kind == GemmType::MGroupedFused) {
        const Arguments<W4A8FusedSchedulerArguments> params{
            a.data_ptr(),
            b.data_ptr(),
            scale.data_ptr(),
            d.data_ptr(),
            a_scale.data_ptr<float>(),
            m,
            inv_scale,
            token_ids.data_ptr<int32_t>(),
            {aligned_blocks.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>()}};
        launch(c, m, n, k, groups, kind, expected_m, d.size(0) / m, rows.data_ptr<int32_t>(), params);
    } else {
        auto prepared = kind == GemmType::MGroupedNoPad ? prepare_nopad_layout(rows, groups, m, c) : rows;
        auto layout = prepared.data_ptr<int32_t>();
        const Arguments<TileSchedulerArguments> params{a.data_ptr(),
                                                       b.data_ptr(),
                                                       scale.data_ptr(),
                                                       d.data_ptr(),
                                                       a_scale.data_ptr<float>(),
                                                       0,
                                                       inv_scale,
                                                       nullptr,
                                                       TileSchedulerArguments(static_cast<uint32_t>(m), layout)};
        launch(c, m, n, k, groups, kind, expected_m, 0, rows.data_ptr<int32_t>(), params);
    }
}

// Specialized kernels use the same compiler, device settings and launch cache
// as every other native GEMM. Only compiled handles/resource counts are cached;
// tensors, routing values and decoded weights are always per call.
struct SpecializedKernel {
    std::shared_ptr<KernelRuntime> runtime;
    int active_blocks;
};

template <class R>
static std::optional<SpecializedKernel> prepare_specialized(const typename R::Args& args, bool capturing,
                                                            bool check_local = true) {
    using Key = std::tuple<int, int, int, int, W4A8ConfigTuple, std::pair<int, int>>;
    static std::mutex mutex;
    static std::map<Key, std::optional<SpecializedKernel>> kernels;
    const Key key{args.device, args.n, args.k, args.groups, pack_config(args.config), args.cache_variant()};
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = kernels.find(key); it != kernels.end())
        return it->second;
    // Module loading and resource initialization are kept outside capture.
    if (capturing)
        return std::nullopt;
    const auto runtime = compiler->build(R::name(), R::generate(args));
    construct_launch_config(runtime->kernel, current_stream(), args.launch_args.smem_size, args.launch_args.grid_dim,
                            args.launch_args.block_dim);
    int local_bytes = 0;
    DG_HGGC_UNIFIED_CHECK(hgFuncGetAttribute(&local_bytes, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel));
    if (check_local && !fits_compiled_strategy(local_bytes))
        return kernels.emplace(key, std::nullopt).first->second;
    int blocks = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, runtime->kernel, args.launch_args.block_dim.x, args.launch_args.smem_size));
    if (blocks == 0)
        return kernels.emplace(key, std::nullopt).first->second;
    return kernels.emplace(key, SpecializedKernel{runtime, blocks}).first->second;
}

struct RoutedSchedulerArguments {
    const uint8_t* meta;
};
using RoutedRuntime = Runtime<RoutedSchedulerArguments>;

class RoutedMetadataRuntime final : public LaunchRuntime<RoutedMetadataRuntime> {
  public:
    struct Args {
        W4A8Config config;
        int n, k, groups, device, variant, token_block_m;
        LaunchArgs launch_args;
        RoutedMetadataArguments params;
        bool force_small = false;
        std::pair<int, int> cache_variant() const { return {variant + 2 * int(force_small), token_block_m}; }
    };
    static const char* name() { return "w4a8_routed_metadata"; }
    static std::string generate_impl(const Args& a) {
        const auto lo = routed_config(true), hi = routed_config(false);
        return fmt::format(R"(
#include <deep_gemm/scheduler/w4a8_routed_scheduler.cuh>
extern "C" __global__ __launch_bounds__(256)
void w4a8_routed_metadata(deep_gemm::w4fa8::RoutedMetadataArguments p) {{
    deep_gemm::w4fa8::build_w4a8_routed_metadata<{}, {}, {}, {}>(
        p.rows, p.lower, p.upper);
}}
// device={}
)",
                           a.groups, lo.block_m, a.force_small ? lo.block_m : hi.block_m, a.token_block_m, a.device);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

template <class Scheduler>
static bool run_compact(const W4A8Config& c, int m, int n, int k, int groups, GemmType kind,
                        const Arguments<Scheduler>& params, const torch::Tensor& rows, int device, bool capturing) {
    using R = Runtime<Scheduler>;
    const bool fused = kind == GemmType::MGroupedFused;
    typename R::Args args{
        c,      n,
        k,      groups,
        kind,   {dim3(c.num_sms), dim3(get_num_threads(c)), specialized_smem_size(c, k, false, !fused && groups <= 64)},
        params, small_m_strategy(c, kind),
        device};
    const auto prepared = prepare_specialized<R>(args, capturing);
    if (!prepared)
        return false;
    torch::Tensor layout;
    if constexpr (std::is_same_v<Scheduler, TileSchedulerArguments>) {
        layout = prepare_nopad_layout(rows, groups, m, c);
        args.params.scheduler = TileSchedulerArguments(uint32_t(m), layout.data_ptr<int32_t>());
    }
    args.launch_args.grid_dim =
        dim3(c.num_sms * (fused ? prepared->active_blocks : std::min(8, prepared->active_blocks)));
    auto prof = make_profile(kind, groups, fused ? params.num_token : m, n, k, ceil_div(m, groups),
                             fused ? m / params.num_token : 0, rows.data_ptr<int32_t>());
    ProfilingInterface::Instance().instrument(true, prof);
    R::launch(prepared->runtime, args);
    ProfilingInterface::Instance().instrument(false, prof);
    return true;
}

static bool run_routed(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                       const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows,
                       const torch::Tensor& tokens, GemmType kind, const W4A8Config& original, bool capturing) {
    const int n = b.size(1), k = a.size(-1), groups = b.size(0), m = d.size(0), device = a.get_device();
    const bool fused = kind == GemmType::MGroupedFused;
    const auto lo = routed_config(true), hi = routed_config(false);
    RoutedMetadataRuntime::Args meta{lo, n, k, groups, device, int(!fused), original.block_m, {dim3(1), dim3(256), 0},
                                     {}};
    RoutedRuntime::Args low{lo,    n,
                            k,     groups,
                            kind,  {dim3(lo.num_sms), dim3(get_num_threads(lo)), specialized_smem_size(lo, k, true)},
                            {},    NativeStrategy::Routed,
                            device};
    RoutedRuntime::Args high{hi,    n,
                             k,     groups,
                             kind,  {dim3(hi.num_sms), dim3(get_num_threads(hi)), specialized_smem_size(hi, k, true)},
                             {},    NativeStrategy::Routed,
                             device};
    const auto low_kernel = prepare_specialized<RoutedRuntime>(low, capturing);
    // Fused bounds live operand state with M160; NoPad can also use M192.
    const auto high_kernel =
        fused ? std::optional<SpecializedKernel>{} : prepare_specialized<RoutedRuntime>(high, capturing);
    if (!low_kernel)
        return false;
    meta.force_small = !high_kernel;
    const auto metadata = prepare_specialized<RoutedMetadataRuntime>(meta, capturing);
    if (!metadata)
        return false;
    auto buffer = [&](int bm) {
        return at::empty({int64_t(kRoutedMetadataBytes) + int64_t(32) * (ceil_div(m, bm) + groups)}, b.options());
    };
    auto lower = buffer(lo.block_m), upper = buffer(hi.block_m);
    meta.params = {rows.data_ptr<int32_t>(), lower.data_ptr<uint8_t>(), upper.data_ptr<uint8_t>()};
    auto prof = make_profile(kind, groups, fused ? a.size(0) : m, n, k, ceil_div(m, groups),
                             fused ? d.size(0) / a.size(0) : 0, rows.data_ptr<int32_t>());
    ProfilingInterface::Instance().instrument(true, prof);
    RoutedMetadataRuntime::launch(metadata->runtime, meta);
    auto invoke = [&](RoutedRuntime::Args& args, const SpecializedKernel& prepared, const torch::Tensor& layout) {
        args.params = {a.data_ptr(),
                       b.data_ptr<uint8_t>(),
                       scale.data_ptr<uint8_t>(),
                       d.data_ptr(),
                       a_scale.data_ptr<float>(),
                       int(a.size(0)),
                       1.0f,
                       fused ? tokens.data_ptr<int32_t>() : nullptr,
                       {layout.data_ptr<uint8_t>()}};
        args.launch_args.grid_dim = dim3(args.config.num_sms * std::min(8, prepared.active_blocks));
        RoutedRuntime::launch(prepared.runtime, args);
    };
    invoke(low, *low_kernel, lower);
    if (high_kernel)
        invoke(high, *high_kernel, upper);
    ProfilingInterface::Instance().instrument(false, prof);
    return true;
}

class AdaptiveMetadataRuntime final : public LaunchRuntime<AdaptiveMetadataRuntime> {
  public:
    struct Args {
        W4A8Config config;
        int n, k, groups, device, native_m, token_block_m;
        LaunchArgs launch_args;
        AdaptiveMetadataArguments params;
        std::pair<int, int> cache_variant() const { return {native_m, token_block_m}; }
    };
    static const char* name() { return "w4a8_adaptive_metadata"; }
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/scheduler/w4a8_routed_scheduler.cuh>
extern "C" __global__ __launch_bounds__(256)
void w4a8_adaptive_metadata(deep_gemm::w4fa8::AdaptiveMetadataArguments p) {{
    deep_gemm::w4fa8::build_w4a8_adaptive_metadata<{}, {}, {}, {}>(p);
}}
// device={}
)", a.groups, a.native_m, a.config.block_m, a.token_block_m, a.device);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

template <class Launch> static float measure_candidate(Launch&& run) {
    hggcEvent_t begin{}, end{};
    std::array<float, 5> samples;
    try {
        DG_HGGC_RUNTIME_CHECK(hggcEventCreate(&begin));
        DG_HGGC_RUNTIME_CHECK(hggcEventCreate(&end));
        run();
        run();
        for (auto& value : samples) {
            DG_HGGC_RUNTIME_CHECK(hggcEventRecord(begin, current_stream()));
            run();
            DG_HGGC_RUNTIME_CHECK(hggcEventRecord(end, current_stream()));
            DG_HGGC_RUNTIME_CHECK(hggcEventSynchronize(end));
            DG_HGGC_RUNTIME_CHECK(hggcEventElapsedTime(&value, begin, end));
        }
    } catch (...) {
        if (begin)
            hggcEventDestroy(begin);
        if (end)
            hggcEventDestroy(end);
        throw;
    }
    DG_HGGC_RUNTIME_CHECK(hggcEventDestroy(begin));
    DG_HGGC_RUNTIME_CHECK(hggcEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    return samples[2];
}

template <class NativeScheduler>
static bool run_adaptive_rows(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                              const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows,
                              const torch::Tensor& tokens, GemmType kind, const W4A8Config& c, bool capturing,
                              const Arguments<NativeScheduler>& native_params) {
    using Native = Runtime<NativeScheduler>;
    const int m = d.size(0), n = b.size(1), k = a.size(-1), groups = b.size(0), device = a.get_device();
    const bool fused = kind == GemmType::MGroupedFused;
    const auto wide_config = routed_config(true);
    using CostKey = std::tuple<int, int, int, int, int, int, int, W4A8ConfigTuple>;
    struct Costs {
        float native_rate, wide_rate;
        bool adaptive;
    };
    static std::mutex mutex;
    static std::map<CostKey, Costs> costs;
    const CostKey key{device, n, k, groups, m, int(a.size(0)), int(kind), pack_config(c)};
    std::optional<Costs> ready;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (const auto it = costs.find(key); it != costs.end())
            ready = it->second;
    }
    if (capturing && !ready)
        return false;
    if (ready && !ready->adaptive) {
        auto params = native_params;
        torch::Tensor layout;
        if constexpr (std::is_same_v<NativeScheduler, TileSchedulerArguments>) {
            layout = prepare_nopad_layout(rows, groups, m, c);
            params.scheduler = TileSchedulerArguments(uint32_t(m), layout.data_ptr<int32_t>());
        }
        launch(c, fused ? int(a.size(0)) : m, n, k, groups, kind, ceil_div(m, groups),
               fused ? m / a.size(0) : 0, rows.data_ptr<int32_t>(), params);
        return true;
    }
    typename Native::Args native{c, n, k, groups, kind,
                                  {dim3(c.num_sms), dim3(get_num_threads(c)), get_smem_size(c, k)},
                                  native_params, NativeStrategy::Default, device, true};
    // This is the original native scheduler/fragment body. Its existing spill
    // behavior remains legal; only the wider compact candidate has a spill cap.
    const auto native_kernel = prepare_specialized<Native>(native, capturing, false);
    auto baseline = native;
    baseline.conditional = false;
    const auto baseline_kernel = prepare_specialized<Native>(baseline, capturing, false);
    RoutedRuntime::Args wide{wide_config, n, k, groups, kind,
                             {dim3(c.num_sms), dim3(get_num_threads(wide_config)),
                              specialized_smem_size(wide_config, k, true)},
                             {}, NativeStrategy::Routed, device, true};
    const auto wide_kernel = prepare_specialized<RoutedRuntime>(wide, capturing);
    if (!native_kernel || !baseline_kernel || !wide_kernel)
        return false;
    AdaptiveMetadataRuntime::Args meta{wide_config, n, k, groups, device, c.block_m, c.block_m,
                                      {dim3(1), dim3(256), 0}, {}};
    const auto metadata = prepare_specialized<AdaptiveMetadataRuntime>(meta, capturing);
    if (!metadata)
        return false;
    auto prof = make_profile(kind, groups, fused ? a.size(0) : m, n, k, ceil_div(m, groups),
                             fused ? m / a.size(0) : 0, rows.data_ptr<int32_t>());
    ProfilingInterface::Instance().instrument(true, prof);
    auto work = at::empty({int64_t(kRoutedMetadataBytes) + int64_t(32) * (ceil_div(m, wide_config.block_m) + groups)},
                          b.options());
    auto control = at::empty({4}, rows.options());
    meta.params = {rows.data_ptr<int32_t>(), work.data_ptr<uint8_t>(), control.data_ptr<int32_t>(), 1, 1};
    native.enabled = control.data_ptr<int32_t>();
    native.launch_args.grid_dim = dim3(c.num_sms < 20 ? c.num_sms : c.num_sms * std::min(8, native_kernel->active_blocks));
    torch::Tensor layout;
    if constexpr (std::is_same_v<NativeScheduler, TileSchedulerArguments>) {
        layout = groups < 128 ? rows : at::empty({(groups + ceil_div(m + 1 - groups, c.block_m)) * 4}, rows.options());
        native.params.scheduler = TileSchedulerArguments(uint32_t(m), layout.data_ptr<int32_t>());
        if (groups >= 128)
            meta.params.native_layout = layout.data_ptr<int32_t>();
    }
    baseline.params = native.params;
    baseline.launch_args.grid_dim =
        dim3(c.num_sms < 20 ? c.num_sms : c.num_sms * std::min(8, baseline_kernel->active_blocks));
    wide.enabled = reinterpret_cast<const int32_t*>(work.data_ptr<uint8_t>()) + 1;
    wide.params = {a.data_ptr(), b.data_ptr<uint8_t>(), scale.data_ptr<uint8_t>(), d.data_ptr(),
                    a_scale.data_ptr<float>(), int(a.size(0)), 1.f, fused ? tokens.data_ptr<int32_t>() : nullptr,
                    {work.data_ptr<uint8_t>()}};
    wide.launch_args.grid_dim = dim3(c.num_sms * std::min(8, wide_kernel->active_blocks));
    auto run_native = [&]() { Native::launch(native_kernel->runtime, native); };
    auto run_wide = [&]() { RoutedRuntime::launch(wide_kernel->runtime, wide); };
    auto run_baseline = [&]() { Native::launch(baseline_kernel->runtime, baseline); };
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = costs.find(key);
        if (it == costs.end()) {
            // This path is unreachable during capture. Calibrate genuine
            // kernels on the caller only once; no tensor values are cached.
            meta.params.native_rate = 0;
            AdaptiveMetadataRuntime::launch(metadata->runtime, meta);
            const float native_ms = measure_candidate(run_native);
            meta.params.native_rate = 1;
            meta.params.wide_rate = 0;
            AdaptiveMetadataRuntime::launch(metadata->runtime, meta);
            const float wide_ms = measure_candidate(run_wide);
            // The native wrapper is inactive under these forced-wide rates.
            // Measure selection overhead as well as the actual ungated native
            // kernel, so short calls need not pay a metadata/branch tax.
            const float overhead_ms = measure_candidate([&]() {
                AdaptiveMetadataRuntime::launch(metadata->runtime, meta);
                run_native();
            });
            const float baseline_ms = measure_candidate(run_baseline);
            int counts[2];
            DG_HGGC_RUNTIME_CHECK(hggcMemcpy(counts, control.data_ptr<int32_t>() + 1, sizeof(counts),
                                           hggcMemcpyDeviceToHost));
            it = costs.emplace(key, Costs{native_ms / std::max(1, counts[0]),
                                          wide_ms / std::max(1, counts[1]),
                                          wide_ms + overhead_ms < baseline_ms &&
                                          overhead_ms <= baseline_ms * kMaxAdaptiveSelectionOverhead}).first;
        }
        ready = it->second;
    }
    if (ready->adaptive) {
        meta.params.native_rate = ready->native_rate;
        meta.params.wide_rate = ready->wide_rate;
        AdaptiveMetadataRuntime::launch(metadata->runtime, meta);
        run_native();
        run_wide();
    } else {
        run_baseline();
    }
    ProfilingInterface::Instance().instrument(false, prof);
    return true;
}

static bool try_run_specialized(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                                const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows,
                                GemmType kind, const W4A8Config& c, const torch::Tensor& tokens = {},
                                const torch::Tensor& experts = {}, const torch::Tensor& aligned = {}) {
    if (is_compile_only() || !supports_native_strategy())
        return false;
    const int m = a.size(0), n = b.size(1), k = a.size(-1), groups = b.size(0);
    const bool fused_layout = kind != GemmType::MGroupedFused || (tokens.dim() == 2 && tokens.size(1) == c.block_m &&
                                                                  default_fused_config(c, d.size(0), n, k, groups));
    const bool small = fused_layout && small_m_eligible(groups, d.size(0), n, k, kind) &&
                       !(kind == GemmType::MGroupedNoPad && c.block_m == 128);
    const bool routed = routed_eligible(groups, d.size(0), n, k, kind, c) &&
                        (kind != GemmType::MGroupedFused || (tokens.dim() == 2 && tokens.size(1) == c.block_m));
    const bool adaptive = fused_layout && adaptive_rows_eligible(groups, d.size(0), n, k);
    if (!small && !routed && !adaptive)
        return false;
    hggcStreamCaptureStatus status = hggcStreamCaptureStatusNone;
    DG_HGGC_RUNTIME_CHECK(hggcStreamIsCapturing(current_stream(), &status));
    const bool capturing = status != hggcStreamCaptureStatusNone;
    if (adaptive) {
        if (kind == GemmType::MGroupedFused) {
            const Arguments<W4A8FusedSchedulerArguments> params{
                a.data_ptr(), b.data_ptr(), scale.data_ptr(), d.data_ptr(), a_scale.data_ptr<float>(), m, 1.f,
                tokens.data_ptr<int32_t>(), {aligned.data_ptr<int32_t>(), experts.data_ptr<int32_t>()}};
            if (run_adaptive_rows(a, a_scale, b, scale, d, rows, tokens, kind, c, capturing, params))
                return true;
        } else {
            const Arguments<TileSchedulerArguments> params{
                a.data_ptr(), b.data_ptr(), scale.data_ptr(), d.data_ptr(), a_scale.data_ptr<float>(), 0, 1.f,
                nullptr, TileSchedulerArguments(uint32_t(m), rows.data_ptr<int32_t>())};
            if (run_adaptive_rows(a, a_scale, b, scale, d, rows, tokens, kind, c, capturing, params))
                return true;
        }
    }
    if (small && (!routed || ceil_div(int(d.size(0)), groups) <= 64)) {
        const int output_m = d.size(0);
        const auto compact = small_m_config(ceil_div(output_m, groups), n, kind, groups);
        if (kind == GemmType::MGroupedFused) {
            const Arguments<W4A8FusedSchedulerArguments> params{
                a.data_ptr(), b.data_ptr(), scale.data_ptr(), d.data_ptr(), a_scale.data_ptr<float>(), m,
                1.0f, tokens.data_ptr<int32_t>(), {aligned.data_ptr<int32_t>(), experts.data_ptr<int32_t>()}};
            return run_compact(compact, output_m, n, k, groups, kind, params, rows, a.get_device(), capturing);
        }
        const Arguments<TileSchedulerArguments> params{
            a.data_ptr(), b.data_ptr(), scale.data_ptr(), d.data_ptr(), a_scale.data_ptr<float>(), 0,
            1.0f, nullptr, TileSchedulerArguments(uint32_t(m), rows.data_ptr<int32_t>())};
        return run_compact(compact, m, n, k, groups, kind, params, rows, a.get_device(), capturing);
    }
    return routed && run_routed(a, a_scale, b, scale, d, rows, tokens, kind, c, capturing);
}

} // namespace deep_gemm::w4fa8
