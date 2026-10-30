#pragma once
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <set>
#include <tuple>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdint>
#include <deep_gemm/common/w4a8_arguments.cuh>
#include "int8_gemm.hpp"

namespace deep_gemm::w4fa8::streamed {
using StreamedInt8 = INT8GemmCutlass3Runtime;
// Match the existing mixed FP8 collective ABI; only the scheduler adds a GPU row range.
struct StreamedCoreParams {
    StreamedInt8::GemmUniversalMode mode;
    StreamedInt8::GemmProblemSize problem_shape;
    StreamedInt8::MainLoopArguments mainloop;
    StreamedInt8::EpilogueParamNoTsm epilogue;
    cutlass::KernelHardwareInfo hw_info;
    w4fa8_streamed::GroupedRowRangeArguments scheduler;
    void* workspace = nullptr;
    int32_t* signal = nullptr;
};

class StreamedDecodeRuntime final : public LaunchRuntime<StreamedDecodeRuntime> {
  public:
    struct Args {
        int n, k, tile_n, groups;
        LaunchArgs launch_args;
        w4fa8_streamed::DecodeParams params;
        bool skip_fence = false;
        bool empty_flat = false;
    };
    static bool full_n_fast(const Args& a) {
        return a.tile_n == a.n && a.n >= 3584 && a.k >= 3072 && uint64_t(a.groups) * a.n * a.k <= UINT32_MAX;
    }
    static std::string kernel_name(const Args& a) {
        // Reuse the established FP8 compiler options for this integer-to-FP8 decoder.
        if (a.empty_flat)
            return a.skip_fence ? "w4a8_streamed_decode_nonempty_free" : "w4a8_streamed_decode_nonempty_fenced";
        if (!full_n_fast(a))
            return "w4a8_streamed_decode";
        return a.skip_fence ? "fp8_deep_gemm_w4a8_tail_decode_fulln" : "fp8_deep_gemm_w4a8_streamed_decode_fulln";
    }
    static std::string generate_impl(const Args& a) {
        const bool fast = full_n_fast(a);
        if (a.empty_flat)
            return fmt::format(R"(
#include <deep_gemm/impls/w4a8_streamed_decode.cuh>
extern "C" __global__ __maxnreg__(32)
void {}(deep_gemm::w4fa8_streamed::DecodeParams params) {{
    deep_gemm::w4fa8_empty_flat::decode_empty_flat_body<{}, {}, {}, {}, 256, {}>(
        params.packed, params.scales, params.output, params.n_begin, params.all_rows, params.group_start);
}}
)",
                               kernel_name(a), a.n, a.k, a.tile_n, a.groups, a.skip_fence ? "true" : "false");
        return fmt::format(R"(
#include <deep_gemm/impls/{}>
extern "C" __global__ {}
void {}(deep_gemm::w4fa8_streamed::DecodeParams params) {{
    deep_gemm::w4fa8_streamed::{}<{}, {}, {}, {}{}>(
        params.packed, params.scales, params.output, params.n_begin, params.all_rows, params.group_start);
}}
)",
                           "w4a8_streamed_decode.cuh", fast ? "__maxnreg__(32)" : "__launch_bounds__(256)",
                           kernel_name(a), fast ? "decode_fulln_rows32_body" : "decode_tile_body", a.n, a.k, a.tile_n,
                           a.groups, fast && a.skip_fence ? ", true" : "");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.params));
    }
};

class StreamedCoreRuntime final : public LaunchRuntime<StreamedCoreRuntime> {
  public:
    struct Args {
        int n, k, tile_n, groups;
        LaunchArgs launch_args;
        StreamedCoreParams params;
    };
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/w4a8_streamed_core.cuh>
using Traits = deep_gemm::w4fa8_streamed::CoreTraits<{}, {}, {}, {}>;
using Kernel = typename Traits::Kernel;
using Params = typename Kernel::Params;
static_assert(Kernel::MaxThreadsPerBlock == 512, "streamed thread count mismatch");
static_assert(Kernel::SharedStorageSize == 262144, "streamed shared memory mismatch");
static_assert(sizeof(Params) == {}, "streamed core argument size mismatch");
static_assert(offsetof(Params, mainloop) == {}, "streamed mainloop ABI mismatch");
static_assert(offsetof(Params, epilogue) == {}, "streamed epilogue ABI mismatch");
static_assert(offsetof(Params, scheduler) == {}, "streamed scheduler ABI mismatch");
extern "C" __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, 1)
void fp8_grouped_deep_gemm_w4a8_streamed_core(Params params) {{
    extern __shared__ char smem[];
    Kernel{{}}(params, smem);
}}
)",
                           a.n, a.k, a.tile_n, a.groups, sizeof(StreamedCoreParams),
                           offsetof(StreamedCoreParams, mainloop), offsetof(StreamedCoreParams, epilogue),
                           offsetof(StreamedCoreParams, scheduler));
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.params));
    }
};

// Full-width tiles remain a strict subset of the experts and are decoded anew.
static void run_streamed_serial(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                                const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows,
                                int tile_n, int num_sms) {
    const int groups = b.size(0), n = b.size(1), k = a.size(1), m = a.size(0);
    constexpr int chunk = 14, threads = 512, core_smem = 256 * 1024;
    DgProfParam prof;
    if (ProfilingInterface::Instance().get_op_info()) {
        prof.set_params(GemmType::MGroupedNoPad, false, "w4fa8_streamed", groups, m, n, k, ceil_div(m, groups),
                        rows.data_ptr<int32_t>(), current_stream());
        prof.add_params(std::string("group_size"), 32);
    }
    ProfilingInterface::Instance().instrument(true, prof);
    auto scratch = at::empty({int64_t(chunk) * tile_n * k + 8}, b.options());
    auto ones = at::ones({chunk, tile_n, 1}, a_scale.options());
    auto output = reinterpret_cast<cutlass::bfloat16_t*>(d.data_ptr());
    const auto stride_input = cute::Stride<int64_t, cute::Int<1>, int64_t>{k, cute::Int<1>{}, 0};
    const auto stride_output = cute::Stride<int64_t, cute::Int<1>, int64_t>{n, cute::Int<1>{}, 0};
    const int remainder = groups % chunk;
    for (int local_groups : {chunk, remainder}) {
        if (!local_groups || local_groups > groups)
            continue;
        StreamedDecodeRuntime::Args decode{
            n,
            k,
            tile_n,
            local_groups,
            {dim3(std::min<int64_t>((int64_t(local_groups) * tile_n * k / 16 + 255) / 256, int64_t(num_sms) * 8)),
             dim3(256), 0},
            {}};
        StreamedCoreRuntime::Args core{n, k, tile_n, local_groups, {dim3(num_sms), dim3(threads), core_smem}, {}};
        const auto& decoder =
            compiler->build(StreamedDecodeRuntime::kernel_name(decode), StreamedDecodeRuntime::generate(decode));
        const auto& gemm =
            compiler->build("fp8_grouped_deep_gemm_w4a8_streamed_core", StreamedCoreRuntime::generate(core));
        construct_launch_config(gemm->kernel, current_stream(), core_smem, core.launch_args.grid_dim,
                                core.launch_args.block_dim);
        int blocks = 0;
        DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, gemm->kernel, threads, core_smem));
        DG_HOST_ASSERT(blocks > 0);
        core.launch_args.grid_dim = dim3(num_sms * blocks);
        for (int group_start = 0; group_start < groups; group_start += chunk) {
            if (std::min(chunk, groups - group_start) != local_groups)
                continue;
            auto range =
                reinterpret_cast<const int32_t*>(scratch.data_ptr<uint8_t>() + int64_t(local_groups) * tile_n * k);
            for (int start = 0; start < n; start += tile_n) {
                decode.params = {b.data_ptr<uint8_t>() + int64_t(group_start) * n * k / 2,
                                 scale.data_ptr<uint8_t>() + int64_t(group_start) * n * k / 32,
                                 scratch.data_ptr<uint8_t>(),
                                 start,
                                 rows.data_ptr<int32_t>(),
                                 group_start};
                StreamedDecodeRuntime::launch(decoder, decode);
                core.params = {StreamedInt8::GemmUniversalMode::kGemm,
                               {m, tile_n, k, 1},
                               {a.data_ptr(), stride_input, scratch.data_ptr(), stride_input, a_scale.data_ptr<float>(),
                                ones.data_ptr<float>()},
                               {{}, output + start, stride_output, output + start, stride_output},
                               {a.get_device(), num_sms},
                               {uint32_t(m), rows.data_ptr<int32_t>() + group_start, range},
                               nullptr,
                               nullptr};
                StreamedCoreRuntime::launch(gemm, core);
            }
        }
    }
    ProfilingInterface::Instance().instrument(false, prof);
}
// Resources contain stream/event identities only. Every call allocates and
// overwrites its own pair of partial weight tiles.
struct StreamedAsyncContext {
    int device;
    hggcStream_t compute = nullptr, decode = nullptr;
    hggcEvent_t fork = nullptr;
    std::vector<hggcEvent_t> decoded, consumed;
    std::mutex submit_mutex;
    // Shape readiness is CPU metadata only; no decoded weights are retained.
    using ShapeKey = std::tuple<int, int, int, int, int, bool>;
    std::set<ShapeKey> warmed_shapes;

    explicit StreamedAsyncContext(int device_) : device(device_) {}
    StreamedAsyncContext(const StreamedAsyncContext&) = delete;
    StreamedAsyncContext& operator=(const StreamedAsyncContext&) = delete;

    void initialize() {
        int least_priority = 0, greatest_priority = 0;
        DG_HGGC_RUNTIME_CHECK(hggcDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
        DG_HGGC_RUNTIME_CHECK(hggcStreamCreateWithPriority(&compute, hggcStreamNonBlocking, greatest_priority));
        DG_HGGC_RUNTIME_CHECK(hggcStreamCreateWithPriority(&decode, hggcStreamNonBlocking, least_priority));
        DG_HGGC_RUNTIME_CHECK(hggcEventCreateWithFlags(&fork, hggcEventDisableTiming));
    }

    void reserve_events(int parts) {
        decoded.reserve(parts);
        consumed.reserve(parts);
        auto grow = [parts](std::vector<hggcEvent_t>& events) {
            while (events.size() < size_t(parts)) {
                hggcEvent_t event = nullptr;
                DG_HGGC_RUNTIME_CHECK(hggcEventCreateWithFlags(&event, hggcEventDisableTiming));
                events.push_back(event);
            }
        };
        grow(decoded);
        grow(consumed);
    }

    ~StreamedAsyncContext() noexcept {
        // Runtime teardown can precede static destruction. Cleanup must not throw.
        int previous = device;
        if (hggcGetDevice(&previous) != hggcSuccess || hggcSetDevice(device) != hggcSuccess)
            return;
        for (auto event : decoded)
            hggcEventDestroy(event);
        for (auto event : consumed)
            hggcEventDestroy(event);
        if (fork)
            hggcEventDestroy(fork);
        if (compute)
            hggcStreamDestroy(compute);
        if (decode)
            hggcStreamDestroy(decode);
        hggcSetDevice(previous);
    }
};

static std::shared_ptr<StreamedAsyncContext> streamed_async_context(int device, hggcStream_t caller, bool create) {
    using Key = std::pair<int, uintptr_t>;
    static std::mutex mutex;
    static std::map<Key, std::shared_ptr<StreamedAsyncContext>> contexts;
    std::lock_guard<std::mutex> lock(mutex);
    const Key key{device, reinterpret_cast<uintptr_t>(caller)};
    auto found = contexts.find(key);
    if (found != contexts.end())
        return found->second;
    if (!create)
        return nullptr;
    // Own partially initialized handles immediately, including failure cleanup.
    auto context = std::make_shared<StreamedAsyncContext>(device);
    context->initialize();
    contexts.emplace(key, context);
    return context;
}

template <typename Runtime, typename Args>
static void launch_streamed_on(const std::shared_ptr<KernelRuntime>& runtime, const Args& args, hggcStream_t stream) {
    const auto& launch = args.launch_args;
    auto config = construct_launch_config(runtime->kernel, stream, launch.smem_size, launch.grid_dim, launch.block_dim);
    Runtime::launch_impl(runtime->kernel, config, args);
}

static bool run_streamed_async(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                               const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows,
                               int num_sms, const std::shared_ptr<StreamedAsyncContext>& context, bool capturing,
                               bool allow_sparse) {
    const int groups = b.size(0), n = b.size(1), k = a.size(1), m = a.size(0);
    constexpr int threads = 512, core_smem = 256 * 1024;
    const auto schedule = make_streamed_plan(m, n, k, groups, num_sms);
    const auto& plan = schedule.groups;
    const int chunk = schedule.max_groups, parts = int(plan.size());
    const bool ramp = schedule.ramp, short_call = schedule.short_call;
    const auto caller = current_stream();
    // The entire event submission is atomic with respect to other host callers
    // using this caller stream; previously queued waits retain their event record.
    std::lock_guard<std::mutex> lock(context->submit_mutex);
    const StreamedAsyncContext::ShapeKey key{n, k, groups, num_sms, chunk, ramp};
    // A captured call may only reuse resources and kernels warmed on this caller.
    if (capturing && (!context->warmed_shapes.count(key) || context->decoded.size() < size_t(parts) ||
                      context->consumed.size() < size_t(parts)))
        return false;
    if (!capturing)
        context->reserve_events(parts);

    struct Kernels {
        StreamedDecodeRuntime::Args decode;
        StreamedCoreRuntime::Args core;
        std::shared_ptr<KernelRuntime> decoder, gemm;
    };
    std::map<int, Kernels> kernels;
    for (const auto& op : plan)
        kernels.emplace(op.second, Kernels{});
    for (auto& entry : kernels) {
        const int local_groups = entry.first;
        auto& pair = entry.second;
        pair.decode = {n, k, n, local_groups, {dim3(1), dim3(256), 0}, {}};
        // Row32 lanes cover1024 K values per warp. Only the short-call tail path
        // uses the measured fence-free decoder; other paths retain V4.
        pair.decode.skip_fence = short_call && (k % 1024 != 0);
        // Ordinary calls prune empty experts; captured calls retain the exact
        // accepted decoder. Both variants are prepared before capture.
        pair.decode.empty_flat = allow_sparse && short_call && !capturing &&
                                 nonempty_streamed_eligible(groups, m, n, k, num_sms) &&
                                 context->warmed_shapes.count(key);
        pair.core = {n, k, n, local_groups, {dim3(num_sms), dim3(threads), core_smem}, {}};
        pair.decoder = compiler->build(StreamedDecodeRuntime::kernel_name(pair.decode),
                                       StreamedDecodeRuntime::generate(pair.decode));
        if (!capturing && !context->warmed_shapes.count(key)) {
            auto graph_decode = pair.decode;
            graph_decode.empty_flat = false;
            compiler->build(StreamedDecodeRuntime::kernel_name(graph_decode),
                            StreamedDecodeRuntime::generate(graph_decode));
        }
        pair.gemm =
            compiler->build("fp8_grouped_deep_gemm_w4a8_streamed_core", StreamedCoreRuntime::generate(pair.core));
        construct_launch_config(pair.gemm->kernel, context->compute, core_smem, pair.core.launch_args.grid_dim,
                                pair.core.launch_args.block_dim);
        int blocks = 0;
        DG_HGGC_UNIFIED_CHECK(
            hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, pair.gemm->kernel, threads, core_smem));
        DG_HOST_ASSERT(blocks > 0);
        pair.core.launch_args.grid_dim = dim3(num_sms * blocks);
    }

    // A newly capturing caller still uses the serial14 fallback. Loading a
    // previously unseen module during capture is unsupported by SDK2.2, so an
    // ordinary adaptive4 warmup also prepares that fallback without launching it.
    if (!capturing && (ramp || chunk != 14) && !context->warmed_shapes.count(key)) {
        for (int local_groups : {14, groups % 14}) {
            if (!local_groups)
                continue;
            StreamedDecodeRuntime::Args decode{n, k, n, local_groups, {dim3(1), dim3(256), 0}, {}};
            StreamedCoreRuntime::Args core{n, k, n, local_groups, {dim3(num_sms), dim3(threads), core_smem}, {}};
            compiler->build(StreamedDecodeRuntime::kernel_name(decode), StreamedDecodeRuntime::generate(decode));
            const auto fallback =
                compiler->build("fp8_grouped_deep_gemm_w4a8_streamed_core", StreamedCoreRuntime::generate(core));
            construct_launch_config(fallback->kernel, caller, core_smem, core.launch_args.grid_dim,
                                    core.launch_args.block_dim);
        }
    }
    if (!capturing)
        context->warmed_shapes.insert(key);

    DgProfParam prof;
    if (ProfilingInterface::Instance().get_op_info()) {
        prof.set_params(GemmType::MGroupedNoPad, false, "w4fa8_streamed_async", groups, m, n, k, ceil_div(m, groups),
                        rows.data_ptr<int32_t>(), caller);
        prof.add_params(std::string("group_size"), 32);
    }
    ProfilingInterface::Instance().instrument(true, prof);
    // Both buffers are bounded strict subsets of the experts, decoded anew on
    // every invocation. Largest historical case: 2 * 294 MiB, plus metadata.
    std::array<torch::Tensor, 2> scratch{at::empty({int64_t(chunk) * n * k + 8}, b.options()),
                                         at::empty({int64_t(chunk) * n * k + 8}, b.options())};
    auto ones = at::ones({chunk, n, 1}, a_scale.options());
    auto output = reinterpret_cast<cutlass::bfloat16_t*>(d.data_ptr());
    const auto stride_input = cute::Stride<int64_t, cute::Int<1>, int64_t>{k, cute::Int<1>{}, 0};
    const auto stride_output = cute::Stride<int64_t, cute::Int<1>, int64_t>{n, cute::Int<1>{}, 0};

    const int overlap_per_sm = int64_t(m) >= int64_t(groups) * 2048 ? 1 : 2;
    auto decode = [&](int index) {
        const int group_start = plan[index].first;
        const int local_groups = plan[index].second;
        auto& pair = kernels.at(local_groups);
        auto args = pair.decode;
        const int64_t blocks = (int64_t(local_groups) * n * k / 16 + 255) / 256;
        // Bound interference while the compute stream is active.
        args.launch_args.grid_dim =
            dim3(std::min<int64_t>(blocks, int64_t(num_sms) * (index == 0 ? 8 : overlap_per_sm)));
        args.params = {b.data_ptr<uint8_t>() + int64_t(group_start) * n * k / 2,
                       scale.data_ptr<uint8_t>() + int64_t(group_start) * n * k / 32,
                       scratch[index & 1].data_ptr<uint8_t>(),
                       0,
                       rows.data_ptr<int32_t>(),
                       group_start};
        launch_streamed_on<StreamedDecodeRuntime>(pair.decoder, args, context->decode);
    };
    auto core = [&](int index) {
        const int group_start = plan[index].first;
        const int local_groups = plan[index].second;
        auto& pair = kernels.at(local_groups);
        auto args = pair.core;
        const auto& buffer = scratch[index & 1];
        auto range = reinterpret_cast<const int32_t*>(buffer.data_ptr<uint8_t>() + int64_t(local_groups) * n * k);
        args.params = {StreamedInt8::GemmUniversalMode::kGemm,
                       {m, n, k, 1},
                       {a.data_ptr(), stride_input, buffer.data_ptr(), stride_input, a_scale.data_ptr<float>(),
                        ones.data_ptr<float>()},
                       {{}, output, stride_output, output, stride_output},
                       {a.get_device(), num_sms},
                       {uint32_t(m), rows.data_ptr<int32_t>() + group_start, range},
                       nullptr,
                       nullptr};
        launch_streamed_on<StreamedCoreRuntime>(pair.gemm, args, context->compute);
    };

    try {
        // Input preparation and ones fill precede the fork on the caller stream.
        DG_HGGC_RUNTIME_CHECK(hggcEventRecord(context->fork, caller));
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(context->compute, context->fork, 0));
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(context->decode, context->fork, 0));
        decode(0);
        DG_HGGC_RUNTIME_CHECK(hggcEventRecord(context->decoded[0], context->decode));
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(context->compute, context->decoded[0], 0));
        for (int index = 0; index < parts; ++index) {
            core(index);
            DG_HGGC_RUNTIME_CHECK(hggcEventRecord(context->consumed[index], context->compute));
            if (index + 1 < parts) {
                // Do not overwrite a slot until its preceding reader completes.
                if (index >= 1)
                    DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(context->decode, context->consumed[index - 1], 0));
                decode(index + 1);
                DG_HGGC_RUNTIME_CHECK(hggcEventRecord(context->decoded[index + 1], context->decode));
                DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(context->compute, context->decoded[index + 1], 0));
            }
        }
        // All internal readers/writers join before the allocator can reuse any
        // caller-stream storage or subsequent caller work can modify inputs.
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(caller, context->consumed[parts - 1], 0));
    } catch (...) {
        // Keep local tensors alive if submission fails after work has started.
        // Every queued wait references a record already submitted above.
        // Capture records nodes without executing them; only live submissions drain.
        if (!capturing) {
            hggcStreamSynchronize(context->compute);
            hggcStreamSynchronize(context->decode);
        }
        throw;
    }
    ProfilingInterface::Instance().instrument(false, prof);
    return true;
}

static void run_streamed(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                         const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows, int tile_n,
                         int num_sms, bool allow_sparse) {
    const c10::cuda::CUDAGuard device_guard(a.device());
    hggcStreamCaptureStatus capture = hggcStreamCaptureStatusNone;
    DG_HGGC_RUNTIME_CHECK(hggcStreamIsCapturing(current_stream(), &capture));
    // Cold capture keeps the established serial route. Warm capture uses only
    // preexisting caller-owned streams/events and already compiled kernel shapes.
    if (tile_n == b.size(1) && full_width_streamed(b.size(1), a.size(1), b.size(0)) &&
        get_compile_mode() != static_cast<int>(CompileMode::ONLY_COMPILE) &&
        get_env<std::string>("HGGC_WARM_UP").empty()) {
        auto context = streamed_async_context(a.get_device(), current_stream(), capture == hggcStreamCaptureStatusNone);
        if (context && run_streamed_async(a, a_scale, b, scale, d, rows, num_sms, context,
                                          capture != hggcStreamCaptureStatusNone, allow_sparse))
            return;
    }
    run_streamed_serial(a, a_scale, b, scale, d, rows, tile_n, num_sms);
}
} // namespace deep_gemm::w4fa8::streamed
