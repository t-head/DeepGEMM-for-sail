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
        int device = 0;
        std::optional<NativeKernelConfig> kernel_config;
        NativeKernelConfig kernel_spec() const {
            return kernel_config.value_or(NativeKernelConfig{config, FragmentPolicy::FullK});
        }
        std::pair<int, int> cache_variant() const {
            const auto c = kernel_spec();
            return {int(c.fragment) + 2 * int(c.a_load()) + 16 * int(params.inv_scale != 1.f), int(kind)};
        }
    };
    static const char* name() { return "m_grouped_gemm_w4fa8_mma"; }
    static std::string generate_impl(const Args& a) {
        const auto kernel = a.kernel_spec();
        const auto& c = kernel.tile;
        return fmt::format(
            R"(
#include <deep_gemm/impls/w4fa8_gemm.cuh>
using Kernel = cutlass::gemm::kernel::W4A8GEMM_MMA<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::GemmType::{}, {}, {}, deep_gemm::w4fa8::FragmentPolicy::{}, deep_gemm::w4fa8::ALoadPolicy::{}>;
static_assert(sizeof(Kernel::SharedStorage) == {}, "shared memory mismatch");
static_assert(Kernel::MaxThreadsPerBlock == {}, "thread count mismatch");
static_assert(sizeof(Kernel::Params) == {}, "kernel argument size mismatch");
extern "C" __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void m_grouped_gemm_w4fa8_mma(Kernel::Params params) {{
    extern __shared__ char smem[];
    Kernel{{}}(params, smem);
}}
)",
            a.n, a.k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, a.groups, c.num_stages,
            get_gemm_type_name(a.kind), c.n_expand, a.params.inv_scale != 1.0f ? "true" : "false",
            kernel.fragment == FragmentPolicy::FullK ? "FullK" : "RollingK32",
            kernel.a_load() == ALoadPolicy::Bulk ? "Bulk" : "Ring2",
            a.launch_args.smem_size, get_num_threads(c), sizeof(Arguments<Scheduler>));
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
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


// Specialized kernels use the same compiler, device settings and launch cache
// as every other native GEMM. Only compiled handles/resource counts are cached;
// tensors, routing values and decoded weights are always per call.
struct SpecializedKernel {
    std::shared_ptr<KernelRuntime> runtime;
    int active_blocks;
};

template <class R>
static std::optional<SpecializedKernel> lookup_native_kernel(const typename R::Args& args, bool capturing,
                                                            bool check_local = true) {
    using Key = std::tuple<int, int, int, int, W4A8ConfigTuple, std::pair<int, int>>;
    static std::mutex mutex;
    static std::map<Key, std::optional<SpecializedKernel>> kernels;
    const Key key{args.device, args.n, args.k, args.groups, pack_config(args.config), args.cache_variant()};
    std::unique_lock<std::mutex> lock(mutex);
    if (const auto it = kernels.find(key); it != kernels.end())
        return it->second;
    // Module loading and resource initialization are kept outside capture.
    if (capturing)
        return std::nullopt;
    lock.unlock();
    const auto runtime = compiler->build(R::name(), R::generate(args));
    construct_launch_config(runtime->kernel, current_stream(), args.launch_args.smem_size, args.launch_args.grid_dim,
                            args.launch_args.block_dim);
    int local_bytes = 0;
    DG_HGGC_UNIFIED_CHECK(hgFuncGetAttribute(&local_bytes, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel));
    std::optional<SpecializedKernel> prepared;
    if (!check_local || fits_compiled_strategy(local_bytes)) {
        int blocks = 0;
        DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, runtime->kernel, args.launch_args.block_dim.x, args.launch_args.smem_size));
        if (blocks > 0) prepared = SpecializedKernel{runtime, blocks};
    }
    lock.lock();
    return kernels.emplace(key, prepared).first->second;
}

static void log_native_config(const W4A8Config& c, int m, int n, int k, int groups,
                              const SpecializedKernel& prepared) {
    if (const char* log = std::getenv("show_log"); log && isdigit(*log)) {
        int regs = 0, stack = 0;
        hgFuncGetAttribute(&regs, HG_FUNC_ATTRIBUTE_NUM_REGS, prepared.runtime->kernel);
        hgFuncGetAttribute(&stack, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, prepared.runtime->kernel);
        printf("[W4FA8] group:%d problem:[%d,%d,%d] block:[%d,%d,%d] warp:[%d,%d,%d] stages:%d smem:%d vreg:%d "
               "stack:%d tb_per_sm:%d threads:%d n_expand:%d\n",
               groups, m, n, k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, c.num_stages,
               get_smem_size(c, k), regs, stack, prepared.active_blocks, get_num_threads(c), c.n_expand);
    }
}

template <class Scheduler>
static void launch(const W4A8Config& c, int m, int n, int k, int groups, GemmType kind, int expected_m, int topk,
                   int32_t* m_rows, const Arguments<Scheduler>& params) {
    using R = Runtime<Scheduler>;
    const int threads = get_num_threads(c), smem = get_smem_size(c, k);
    typename R::Args args{c, n, k, groups, kind, {dim3(c.num_sms), dim3(threads), smem}, params};
    DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&args.device));
    const auto prepared = lookup_native_kernel<R>(args, is_w4a8_capturing(), false);
    TORCH_CHECK(prepared.has_value(), "W4FA8 native plan must be warmed outside graph capture");
    const auto& runtime = prepared->runtime;
    const int blocks = prepared->active_blocks;
    args.launch_args.grid_dim = dim3(c.num_sms < 20 ? c.num_sms : c.num_sms * std::min(8, blocks));
    auto prof = make_profile(kind, groups, m, n, k, expected_m, topk, m_rows);
    log_native_config(c, m, n, k, groups, *prepared);
    ProfilingInterface::Instance().instrument(true, prof);
    R::launch(runtime, args);
    ProfilingInterface::Instance().instrument(false, prof);
}

static torch::Tensor prepare_nopad_layout(const torch::Tensor& rows, int groups, int m, const W4A8Config& c) {
    if (groups < 128)
        return rows;
    TORCH_CHECK(groups <= 1024, "W4FA8 nopad supports at most 1024 experts");
    auto scratch = at::empty({int64_t(4) * (1 + groups + ceil_div(m, c.block_m))}, rows.options());
    auto args = ComputeBlockInfoKernelRuntime::Args{
        .launch_attr_args = {reinterpret_cast<const uint32_t*>(rows.data_ptr<int32_t>()), static_cast<uint32_t>(groups),
                             reinterpret_cast<uint32_t*>(scratch.data_ptr<int32_t>())},
        .launch_args = {1, std::max(32, next_power_of_two(groups)), 0}};
    using Key = std::pair<int, int>;
    static std::mutex mutex;
    static std::map<Key, std::shared_ptr<KernelRuntime>> kernels;
    int device = 0;
    DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&device));
    std::shared_ptr<KernelRuntime> runtime;
    {
        std::unique_lock<std::mutex> lock(mutex);
        const Key key{device, c.block_m};
        const auto found = kernels.find(key);
        if (found != kernels.end()) runtime = found->second;
        else {
            TORCH_CHECK(!is_w4a8_capturing(), "W4FA8 nopad layout must be warmed outside graph capture");
            lock.unlock();
            runtime = compiler->build("computeBlockInfoKernel", ComputeBlockInfoKernelRuntime::generate(c.block_m));
            lock.lock();
            runtime = kernels.emplace(key, runtime).first->second;
        }
    }
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

template <class Scheduler>
static bool run_compact(const NativeKernelConfig& spec, int m, int n, int k, int groups, GemmType kind,
                        const Arguments<Scheduler>& params, const torch::Tensor& rows, int device, bool capturing) {
    using R = Runtime<Scheduler>;
    const bool fused = kind == GemmType::MGroupedFused;
    const auto& c = spec.tile;
    typename R::Args args{
        c,      n,
        k,      groups,
        kind,   {dim3(c.num_sms), dim3(get_num_threads(c)), get_smem_size(c, k)},
        params, device, spec};
    const auto prepared = lookup_native_kernel<R>(args, capturing);
    if (!prepared)
        return false;
    torch::Tensor layout;
    if constexpr (std::is_same_v<Scheduler, TileSchedulerArguments>) {
        if (kind == GemmType::MGroupedNoPad) {
            layout = prepare_nopad_layout(rows, groups, m, c);
            args.params.scheduler = TileSchedulerArguments(uint32_t(m), layout.data_ptr<int32_t>());
        }
    }
    args.launch_args.grid_dim =
        dim3(c.num_sms * (fused ? prepared->active_blocks : std::min(8, prepared->active_blocks)));
    auto prof = make_profile(kind, groups, fused ? params.num_token : m, n, k, ceil_div(m, groups),
                             fused ? m / params.num_token : 0, rows.data_ptr<int32_t>());
    log_native_config(c, fused ? params.num_token : m, n, k, groups, *prepared);
    ProfilingInterface::Instance().instrument(true, prof);
    R::launch(prepared->runtime, args);
    ProfilingInterface::Instance().instrument(false, prof);
    return true;
}

static void execute_plan(const ExecutionPlan& plan, const torch::Tensor& a, const torch::Tensor& a_scale,
                         const torch::Tensor& b, const torch::Tensor& scale, const torch::Tensor& d,
                         const torch::Tensor& rows, GemmType kind, int expected_m, float inv_scale,
                         const torch::Tensor& experts = {}, const torch::Tensor& tokens = {}, const torch::Tensor& aligned = {}) {
    const int n = b.size(1), k = a.size(-1), groups = b.size(0), m = kind == GemmType::MGroupedMasked ? a.size(1) : d.size(0);
    check_plan(plan, m, n, k, groups, kind);
    if (const auto* streamed_config = std::get_if<StreamedConfig>(&plan)) {
        const auto& c = *streamed_config;
        TORCH_CHECK(streamed::run_streamed(a, a_scale, b, scale, d, rows, c,
                                          kind == GemmType::MGroupedFused ? streamed::FusedInputs{experts, tokens, aligned}
                                                                          : streamed::FusedInputs{}, inv_scale),
                    "W4FA8 streamed plan must be warmed outside graph capture");
        return;
    }
    const auto& kernel = std::get<NativeKernelConfig>(plan);
    using ReadyKey = std::tuple<int, uintptr_t, int, int, int, int, int, W4A8ConfigTuple, FragmentPolicy, bool>;
    static std::mutex ready_mutex;
    static std::set<ReadyKey> ready;
    const ReadyKey key{a.get_device(), reinterpret_cast<uintptr_t>(current_stream()), m, n, k, groups, int(kind), pack_config(kernel.tile), kernel.fragment, inv_scale != 1.f};
    const bool capturing = is_w4a8_capturing();
    if (capturing) {
        std::lock_guard<std::mutex> lock(ready_mutex);
        TORCH_CHECK(ready.count(key), "W4FA8 native plan must be warmed outside graph capture on this caller stream");
    }

    if (kernel.fragment == FragmentPolicy::FullK) {
        run(a, a_scale, b, scale, d, rows, kind, expected_m, kernel.tile, inv_scale, experts, tokens, aligned);
    } else if (kind == GemmType::MGroupedFused) {
        const Arguments<W4A8FusedSchedulerArguments> params{
            a.data_ptr(), b.data_ptr(), scale.data_ptr(), d.data_ptr(), a_scale.data_ptr<float>(), int(a.size(0)),
            inv_scale, tokens.data_ptr<int32_t>(), {aligned.data_ptr<int32_t>(), experts.data_ptr<int32_t>()}};
        TORCH_CHECK(run_compact(kernel, m, n, k, groups, kind, params, rows, a.get_device(), is_w4a8_capturing()),
                    "W4FA8 native plan must be warmed outside graph capture and fit compiled resources");
    } else {
        const Arguments<TileSchedulerArguments> params{
            a.data_ptr(), b.data_ptr(), scale.data_ptr(), d.data_ptr(), a_scale.data_ptr<float>(), 0,
            inv_scale, nullptr, TileSchedulerArguments(uint32_t(m), rows.data_ptr<int32_t>())};
        TORCH_CHECK(run_compact(kernel, m, n, k, groups, kind, params, rows, a.get_device(), is_w4a8_capturing()),
                    "W4FA8 native plan must be warmed outside graph capture and fit compiled resources");
    }
    if (!capturing && !is_compile_only()) {
        std::lock_guard<std::mutex> lock(ready_mutex);
        ready.insert(key);
    }

}

} // namespace deep_gemm::w4fa8
