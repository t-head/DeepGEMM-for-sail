#pragma once
#include <torch/python.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <optional>
#include <map>
#include <mutex>
#include <string>

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
    };
    static std::string generate_impl(const Args& a) {
        const auto& c = a.config;
        return fmt::format(R"(
#include <deep_gemm/impls/w4fa8_gemm.cuh>
using Kernel = cutlass::gemm::kernel::W4A8GEMM_MMA<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::GemmType::{}, {}{}>;
static_assert(sizeof(Kernel::SharedStorage) == {}, "shared memory mismatch");
static_assert(Kernel::MaxThreadsPerBlock == {}, "thread count mismatch");
static_assert(sizeof(Kernel::Params) == {}, "kernel argument size mismatch");
extern "C" __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void m_grouped_gemm_w4fa8_mma(Kernel::Params params) {{
    extern __shared__ char smem[];
    Kernel{{}}(params, smem);
}}
)",
                           a.n, a.k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, a.groups,
                           c.num_stages, get_gemm_type_name(a.kind), c.n_expand,
                           a.params.inv_scale == 1.0f ? "" : ", true", get_smem_size(c, a.k), get_num_threads(c),
                           sizeof(Arguments<Scheduler>));
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

template <class Scheduler>
static void launch(const W4A8Config& c, int m, int n, int k, int groups, GemmType kind, int expected_m, int topk,
                   int32_t* m_rows, const Arguments<Scheduler>& params) {
    using R = Runtime<Scheduler>;
    const int threads = get_num_threads(c), smem = get_smem_size(c, k);
    typename R::Args args{c, n, k, groups, kind, {dim3(c.num_sms), dim3(threads), smem}, params};
    const auto& runtime = compiler->build("m_grouped_gemm_w4fa8_mma", R::generate(args));
    construct_launch_config(runtime->kernel, current_stream(), smem, args.launch_args.grid_dim,
                            args.launch_args.block_dim);
    int blocks = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, runtime->kernel, threads, smem));
    DG_HOST_ASSERT(blocks > 0);
    args.launch_args.grid_dim = dim3(c.num_sms < 20 ? c.num_sms : c.num_sms * std::min(8, blocks));
    DgProfParam prof;
    if (ProfilingInterface::Instance().get_op_info()) {
        if (kind == GemmType::MGroupedFused)
            prof.set_fused_moe_params("w4fa8_mma", "group", groups, m, topk, n, k, m_rows, current_stream());
        else {
            prof.set_params(kind, false, "w4fa8_mma", groups, m, n, k, expected_m, m_rows, current_stream());
            prof.add_params(std::string("quant_type"), std::string("group"));
        }
        prof.add_params(std::string("group_size"), 32);
    }
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
        torch::Tensor scratch;
        auto layout = rows.data_ptr<int32_t>();
        if (kind == GemmType::MGroupedNoPad && groups >= 128) {
            TORCH_CHECK(groups <= 1024, "W4FA8 nopad supports at most 1024 experts");
            scratch = at::empty({(groups + ceil_div(m + 1 - groups, c.block_m)) * 4}, rows.options());
            auto args = ComputeBlockInfoKernelRuntime::Args{
                .launch_attr_args = {reinterpret_cast<const uint32_t*>(layout), static_cast<uint32_t>(groups),
                                     reinterpret_cast<uint32_t*>(scratch.data_ptr<int32_t>())},
                .launch_args = {1, std::max(32, next_power_of_two(groups)), 0}};
            const auto& code = ComputeBlockInfoKernelRuntime::generate(c.block_m);
            const auto& runtime = compiler->build("computeBlockInfoKernel", code);
            ComputeBlockInfoKernelRuntime::launch(runtime, args);
            layout = scratch.data_ptr<int32_t>();
        }
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
static std::optional<SpecializedKernel> prepare_specialized(const typename R::Args& args, bool capturing) {
    using Key = std::tuple<int, int, int, int, W4A8ConfigTuple, std::pair<int, int>>;
    static std::mutex mutex;
    static std::map<Key, SpecializedKernel> kernels;
    const Key key{args.device, args.n, args.k, args.groups, pack_config(args.config), args.cache_variant()};
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = kernels.find(key); it != kernels.end())
        return it->second;
    // Module loading and resource initialization are kept outside capture.
    if (capturing)
        return std::nullopt;
    const auto runtime = compiler->build(R::name(), R::generate(args));
    int blocks = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, runtime->kernel, args.launch_args.block_dim.x, args.launch_args.smem_size));
    TORCH_CHECK(blocks > 0, "W4A8 specialization has no resident CTA");
    return kernels.emplace(key, SpecializedKernel{runtime, blocks}).first->second;
}

struct SmallMArguments {
    const void* a;
    const void* b;
    const void* scale;
    void* d;
    const float* a_scale;
    int num_token;
    const int* sorted_token_ids;
    TileSchedulerArguments scheduler;
};

class SmallMRuntime final : public LaunchRuntime<SmallMRuntime> {
  public:
    struct Args {
        W4A8Config config;
        int n, k, groups, device, variant = 0;
        LaunchArgs launch_args;
        SmallMArguments params;
        std::pair<int, int> cache_variant() const { return {variant, 0}; }
    };
    static const char* name() { return "m_grouped_gemm_w4fa8_mma"; }
    static std::string generate_impl(const Args& a) {
        const auto& c = a.config;
        return fmt::format(R"(
#include <deep_gemm/impls/w4a8_small_m.cuh>
using Kernel = cutlass::gemm::kernel::W4A8SmallMGemm<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::GemmType::MGroupedNoPad, {}, {}, {}>;
static_assert(sizeof(Kernel::SharedStorage) == {}, "small-M shared memory mismatch");
static_assert(sizeof(Kernel::Params) == {}, "small-M parameter ABI mismatch");
extern "C" __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, 1)
void m_grouped_gemm_w4fa8_mma(Kernel::Params params) {{
    extern __shared__ char smem[];
    Kernel{{}}(params, smem);
}}
// device={}
)",
                           a.n, a.k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, a.groups,
                           c.num_stages, c.n_expand, kSmallMPolicy.k_batch,
                           kSmallMPolicy.schedule_bound ? "true" : "false", a.launch_args.smem_size,
                           sizeof(SmallMArguments), a.device);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

class RoutedMetadataRuntime final : public LaunchRuntime<RoutedMetadataRuntime> {
  public:
    struct Args {
        W4A8Config config;
        int n, k, groups, device, variant, token_block_m;
        LaunchArgs launch_args;
        RoutedMetadataArguments params;
        std::pair<int, int> cache_variant() const { return {variant, token_block_m}; }
    };
    static const char* name() { return "w4a8_routed_metadata"; }
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/scheduler/w4a8_routed_scheduler.cuh>
extern "C" __global__ __launch_bounds__(256)
void w4a8_routed_metadata(deep_gemm::w4fa8::RoutedMetadataArguments p) {{
    deep_gemm::w4fa8::build_w4a8_routed_metadata<{}, {}, {}, {}, {}>(
        p.rows, p.fast_lo, p.fast_hi, p.old_lo, p.old_hi, p.max_extra_rows);
}}
// device={}
)",
                           a.groups, routed_config(true, GemmType::MGroupedNoPad).block_m,
                           routed_config(false, GemmType::MGroupedNoPad).block_m, a.token_block_m,
                           a.variant ? "true" : "false", a.device);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

class RoutedRuntime final : public LaunchRuntime<RoutedRuntime> {
  public:
    struct Args {
        W4A8Config config;
        int n, k, groups, device, variant;
        GemmType kind;
        LaunchArgs launch_args;
        RoutedGemmArguments params;
        std::pair<int, int> cache_variant() const { return {variant, 0}; }
    };
    static const char* name() { return "m_grouped_gemm_w4a8_routed"; }
    static std::string generate_impl(const Args& a) {
        const auto& c = a.config;
        const bool fused = a.kind == GemmType::MGroupedFused;
        const int distance = fused ? -1 : 1;
        const bool preload = !fused && c.block_m == routed_config(true, a.kind).block_m;
        const bool reload = fused && c.block_k == 128;
        const auto types =
            fmt::format(R"(
#include <deep_gemm/impls/w4a8_routed.cuh>
using Kernel = cutlass::gemm::kernel::W4A8RoutedGemm<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::GemmType::{}, {}, {}, false, {}>;
using Fast = cutlass::gemm::kernel::W4A8RoutedGemm<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::GemmType::{}, {}, {}, true, {}>;
static_assert(sizeof(Kernel::SharedStorage) == {}, "routed shared memory mismatch");
)",
                        a.n, a.k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, a.groups, c.num_stages,
                        get_gemm_type_name(a.kind), distance, preload ? "true" : "false", reload ? "true" : "false",
                        a.n, a.k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, a.groups, c.num_stages,
                        get_gemm_type_name(a.kind), distance, preload ? "true" : "false", reload ? "true" : "false",
                        a.launch_args.smem_size);
        const std::string invoke = fused ? R"(
    Kernel::Params args{reinterpret_cast<const Kernel::ElementA*>(p.a),p.b,p.scale,
        reinterpret_cast<Kernel::ElementD*>(p.d),p.a_scale,p.num_token,p.sorted_token_ids,{p.old_metadata}};
    Kernel{}(args,smem);
)"
                                         : R"(
    const auto* range=reinterpret_cast<const int32_t*>(p.fast_metadata);
    if (__ldg(range+1)>__ldg(range)) {
        Fast::Params args{reinterpret_cast<const Fast::ElementA*>(p.a),p.b,p.scale,
            reinterpret_cast<Fast::ElementD*>(p.d),p.a_scale,p.num_token,nullptr,{p.fast_metadata}};
        Fast{}(args,smem);
    } else {
        Kernel::Params args{reinterpret_cast<const Kernel::ElementA*>(p.a),p.b,p.scale,
            reinterpret_cast<Kernel::ElementD*>(p.d),p.a_scale,p.num_token,nullptr,{p.old_metadata}};
        Kernel{}(args,smem);
    }
)";
        return types + fmt::format(R"(
extern "C" __global__ __launch_bounds__({}, 1)
void m_grouped_gemm_w4a8_routed(deep_gemm::w4fa8::RoutedGemmArguments p) {{
    extern __shared__ __align__(128) char smem[];
{}
}}
// device={}
)",
                                   get_num_threads(c), invoke, a.device);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static bool run_small_m(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                        const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows, bool capturing) {
    const int m = a.size(0), n = b.size(1), k = a.size(1), groups = b.size(0);
    const auto c = small_m_config(ceil_div(m, groups));
    const int threads = get_num_threads(c), smem = specialized_smem_size(c, k, false);
    SmallMRuntime::Args args{c,
                             n,
                             k,
                             groups,
                             a.get_device(),
                             0,
                             {dim3(c.num_sms), dim3(threads), smem},
                             {a.data_ptr(), b.data_ptr(), scale.data_ptr(), d.data_ptr(), a_scale.data_ptr<float>(), 0,
                              nullptr, TileSchedulerArguments(uint32_t(m), rows.data_ptr<int32_t>())}};
    const auto prepared = prepare_specialized<SmallMRuntime>(args, capturing);
    if (!prepared)
        return false;
    args.launch_args.grid_dim = dim3(c.num_sms * std::min(8, prepared->active_blocks));
    SmallMRuntime::launch(prepared->runtime, args);
    return true;
}

static bool run_routed(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                       const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows,
                       const torch::Tensor& tokens, GemmType kind, const W4A8Config& original, bool capturing) {
    const int n = b.size(1), k = a.size(-1), groups = b.size(0), m = d.size(0), device = a.get_device();
    const bool fused = kind == GemmType::MGroupedFused;
    const auto lo = routed_config(true, kind), hi = routed_config(false, kind);
    RoutedMetadataRuntime::Args meta{lo, n, k, groups, device, int(!fused), original.block_m, {dim3(1), dim3(256), 0},
                                     {}};
    RoutedRuntime::Args low{
        lo,     n,         k,    groups,
        device, int(kind), kind, {dim3(lo.num_sms), dim3(get_num_threads(lo)), specialized_smem_size(lo, k, true)},
        {}};
    RoutedRuntime::Args high{
        hi,     n,         k,    groups,
        device, int(kind), kind, {dim3(hi.num_sms), dim3(get_num_threads(hi)), specialized_smem_size(hi, k, true)},
        {}};
    const auto metadata = prepare_specialized<RoutedMetadataRuntime>(meta, capturing);
    const auto low_kernel = prepare_specialized<RoutedRuntime>(low, capturing);
    const auto high_kernel = prepare_specialized<RoutedRuntime>(high, capturing);
    if (!metadata || !low_kernel || !high_kernel)
        return false;
    auto buffer = [&](int bm) {
        return at::empty({int64_t(kRoutedMetadataBytes) + int64_t(32) * (ceil_div(m, bm) + groups)}, b.options());
    };
    auto fast_lo = buffer(lo.block_m), fast_hi = buffer(hi.block_m);
    auto old_lo = fused ? fast_lo : buffer(lo.block_m), old_hi = fused ? fast_hi : buffer(hi.block_m);
    meta.params = {rows.data_ptr<int32_t>(),   fast_lo.data_ptr<uint8_t>(), fast_hi.data_ptr<uint8_t>(),
                   old_lo.data_ptr<uint8_t>(), old_hi.data_ptr<uint8_t>(),  m / 1024};
    RoutedMetadataRuntime::launch(metadata->runtime, meta);
    auto invoke = [&](RoutedRuntime::Args& args, const SpecializedKernel& prepared, const torch::Tensor& fast,
                      const torch::Tensor& fallback) {
        args.params = {a.data_ptr(),
                       b.data_ptr<uint8_t>(),
                       scale.data_ptr<uint8_t>(),
                       d.data_ptr(),
                       a_scale.data_ptr<float>(),
                       int(a.size(0)),
                       fused ? tokens.data_ptr<int32_t>() : nullptr,
                       fast.data_ptr<uint8_t>(),
                       fallback.data_ptr<uint8_t>()};
        args.launch_args.grid_dim = dim3(args.config.num_sms * std::min(8, prepared.active_blocks));
        RoutedRuntime::launch(prepared.runtime, args);
    };
    invoke(low, *low_kernel, fast_lo, old_lo);
    invoke(high, *high_kernel, fast_hi, old_hi);
    return true;
}

static bool try_run_specialized(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                                const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows,
                                GemmType kind, const W4A8Config& c, const torch::Tensor& tokens = {}) {
    if (is_compile_only() || !use_tuned_device())
        return false;
    const int m = a.size(0), n = b.size(1), k = a.size(-1), groups = b.size(0);
    const bool small = kind == GemmType::MGroupedNoPad && small_m_eligible(groups, m, n, k);
    const bool routed =
        has_routed_profile(groups, m, n, k, kind) && (kind != GemmType::MGroupedFused || is_default_routed_config(c));
    if (!small && !routed)
        return false;
    hggcStreamCaptureStatus status = hggcStreamCaptureStatusNone;
    DG_HGGC_RUNTIME_CHECK(hggcStreamIsCapturing(current_stream(), &status));
    const bool capturing = status != hggcStreamCaptureStatusNone;
    if (small)
        return run_small_m(a, a_scale, b, scale, d, rows, capturing);
    return run_routed(a, a_scale, b, scale, d, rows, tokens, kind, c, capturing);
}

} // namespace deep_gemm::w4fa8
