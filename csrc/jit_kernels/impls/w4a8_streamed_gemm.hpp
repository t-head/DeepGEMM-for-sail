#pragma once
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <set>
#include <tuple>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <deep_gemm/common/w4a8_arguments.cuh>
#include "int8_gemm.hpp"

namespace deep_gemm::w4fa8::streamed {
using StreamedInt8 = INT8GemmCutlass3Runtime;
// Match the existing mixed FP8 collective ABI; only the scheduler adds a GPU row range.
struct ScaledMainloopArguments : StreamedInt8::MainLoopArguments { float inv_scale = 1.f; };
template <bool ApplyInvScale> struct StreamedCoreParamsT {
    StreamedInt8::GemmUniversalMode mode;
    StreamedInt8::GemmProblemSize problem_shape;
    std::conditional_t<ApplyInvScale, ScaledMainloopArguments, StreamedInt8::MainLoopArguments> mainloop;
    StreamedInt8::EpilogueParamNoTsm epilogue;
    cutlass::KernelHardwareInfo hw_info;
    w4fa8_streamed::GroupedRowRangeArguments scheduler;
    void* workspace = nullptr;
    int32_t* signal = nullptr;
};

using StreamedCoreParams = StreamedCoreParamsT<false>;
using ScaledCoreParams = StreamedCoreParamsT<true>;

struct FusedInputs {
    torch::Tensor experts, tokens, aligned;
};

class StreamedGatherRuntime final : public LaunchRuntime<StreamedGatherRuntime> {
  public:
    struct Args {
        int k, groups;
        LaunchArgs launch_args;
        w4fa8_streamed::GatherParams params;
    };
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/w4a8_streamed_gemm.cuh>
extern "C" __global__ __launch_bounds__(256)
void w4a8_streamed_gather(deep_gemm::w4fa8_streamed::GatherParams p) {{
    deep_gemm::w4fa8_streamed::gather_body<{}, deep_gemm::w4fa8_streamed::kBlockM, {}>(p, 0);
}}
)",
                           a.k, a.groups);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.params));
    }
};

class StreamedDecodeRuntime final : public LaunchRuntime<StreamedDecodeRuntime> {
  public:
    struct Args {
        int n, k, tile_n, groups;
        LaunchArgs launch_args;
        w4fa8_streamed::DecodeParams params;
        bool skip_fence = false;
        bool empty_flat = false;
        bool fuse_gather = false;
        w4fa8_streamed::GatherParams gather{};
    };
    static bool full_n_fast(const Args& a) {
        return a.tile_n == a.n && a.n % 512 == 0 && a.k % 128 == 0 && uint64_t(a.groups) * a.n * a.k <= UINT32_MAX;
    }
    static std::string kernel_name(const Args& a) {
        // Reuse the established FP8 compiler options for this integer-to-FP8 decoder.
        const std::string suffix = a.fuse_gather ? "_fused" : "";
        if (a.empty_flat)
            return (a.skip_fence ? "w4a8_streamed_decode_nonempty_free" : "w4a8_streamed_decode_nonempty_fenced") +
                   suffix;
        if (!full_n_fast(a))
            return "w4a8_streamed_decode" + suffix;
        return (a.skip_fence ? "fp8_deep_gemm_w4a8_tail_decode_fulln" : "fp8_deep_gemm_w4a8_streamed_decode_fulln") +
               suffix;
    }
    static std::string gather_call(const Args& a) {
        return a.fuse_gather
                   ? fmt::format("deep_gemm::w4fa8_streamed::gather_body<{}, deep_gemm::w4fa8_streamed::kBlockM, {}>(gather, params.group_start);", a.k, a.groups)
                   : "";
    }
    static std::string generate_impl(const Args& a) {
        const bool fast = full_n_fast(a);
        const std::string gather_arg = a.fuse_gather ? ", deep_gemm::w4fa8_streamed::GatherParams gather" : "";
        if (a.empty_flat)
            return fmt::format(R"(
#include <deep_gemm/impls/w4a8_streamed_gemm.cuh>
extern "C" __global__ __maxnreg__(32)
void {}(deep_gemm::w4fa8_streamed::DecodeParams params{}) {{
    deep_gemm::w4fa8_empty_flat::decode_empty_flat_body<{}, {}, {}, {}, 256, {}>(
        params.packed, params.scales, params.output, params.n_begin, params.all_rows, params.group_start);
    {}
}}
)",
                               kernel_name(a), gather_arg, a.n, a.k, a.tile_n, a.groups,
                               a.skip_fence ? "true" : "false", gather_call(a));
        return fmt::format(R"(
#include <deep_gemm/impls/w4a8_streamed_gemm.cuh>
extern "C" __global__ {}
void {}(deep_gemm::w4fa8_streamed::DecodeParams params{}) {{
    deep_gemm::w4fa8_streamed::{}<{}, {}, {}, {}{}>(
        params.packed, params.scales, params.output, params.n_begin, params.all_rows, params.group_start);
    {}
}}
)",
                           fast ? "__maxnreg__(32)" : "__launch_bounds__(256)",
                           kernel_name(a), gather_arg, fast ? "decode_fulln_rows32_body" : "decode_tile_body", a.n, a.k,
                           a.tile_n, a.groups, fast && a.skip_fence ? ", true" : "", gather_call(a));
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        if (a.fuse_gather)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.params, a.gather));
        else
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.params));
    }
};

class StreamedCoreRuntime final : public LaunchRuntime<StreamedCoreRuntime> {
  public:
    struct Args {
        int n, k, tile_n, groups;
        LaunchArgs launch_args;
        StreamedCoreParams params;
        bool apply_inv_scale = false;
        float inv_scale = 1.f;
    };
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/w4a8_streamed_gemm.cuh>
using Traits = deep_gemm::w4fa8_streamed::CoreTraits<{}, {}, {}, {}, deep_gemm::w4fa8_streamed::kBlockM, 256, 64, 64, 4, {}>;
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
                           a.n, a.k, a.tile_n, a.groups, a.apply_inv_scale ? "true" : "false",
                           a.apply_inv_scale ? sizeof(ScaledCoreParams) : sizeof(StreamedCoreParams),
                           a.apply_inv_scale ? offsetof(ScaledCoreParams, mainloop) : offsetof(StreamedCoreParams, mainloop),
                           a.apply_inv_scale ? offsetof(ScaledCoreParams, epilogue) : offsetof(StreamedCoreParams, epilogue),
                           a.apply_inv_scale ? offsetof(ScaledCoreParams, scheduler) : offsetof(StreamedCoreParams, scheduler));
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        if (a.apply_inv_scale) {
            ScaledCoreParams p{a.params.mode, a.params.problem_shape, {a.params.mainloop, a.inv_scale},
                               a.params.epilogue, a.params.hw_info, a.params.scheduler, a.params.workspace, a.params.signal};
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p));
        } else {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.params));
        }
    }
};

// Resources contain stream/event identities only. Every call allocates and
// overwrites its own pair of partial weight tiles.
struct StreamedKernels {
    StreamedDecodeRuntime::Args decode;
    StreamedCoreRuntime::Args core;
    std::shared_ptr<KernelRuntime> decoder, gemm;
};
struct StreamedPreparedPlan {
    std::map<std::pair<int, int>, StreamedKernels> kernels;
    std::shared_ptr<KernelRuntime> gather;
    bool warmed = false;
};

static void log_streamed_config(const StreamedConfig& execution, int m, int n, int k, int groups,
                                int grouped_rows, int chunk, int parts, const StreamedPreparedPlan& prepared) {
    if (const char* log = std::getenv("show_log"); log && isdigit(*log)) {
        const auto& c = execution.core;
        printf("[W4FA8 streamed] group:%d problem:[%d,%d,%d] rows:%d decode_n:%d expert_chunk:%d parts:%d "
               "schedule:%s gather:%s empty_experts:%s\n",
               groups, m, n, k, grouped_rows, execution.decode_n, chunk, parts,
               execution.schedule == ChunkSchedule::Ramp ? "ramp" : "fixed",
               execution.gather == GatherPolicy::None ? "none" : execution.gather == GatherPolicy::Full ? "full" : "chunk",
               execution.empty_experts == EmptyExpertPolicy::Nonempty ? "nonempty" : "dense");
        for (const auto& entry : prepared.kernels) {
            const auto& core = entry.second;
            int regs = 0, stack = 0;
            hgFuncGetAttribute(&regs, HG_FUNC_ATTRIBUTE_NUM_REGS, core.gemm->kernel);
            hgFuncGetAttribute(&stack, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, core.gemm->kernel);
            printf("[W4FA8 streamed core] group:%d n:%d block:[%d,%d,%d] warp:[%d,%d,%d] stages:%d "
                   "smem:%d vreg:%d stack:%d tb_per_sm:%u threads:%u\n",
                   entry.first.first, entry.first.second, c.block_m, c.block_n, c.block_k,
                   c.warp_m, c.warp_n, c.warp_k, c.num_stages, core.core.launch_args.smem_size, regs, stack,
                   core.core.launch_args.grid_dim.x / c.num_sms, core.core.launch_args.block_dim.x);
        }
    }
}

struct StreamedAsyncContext {
    int device;
    hggcStream_t compute = nullptr, decode = nullptr;
    hggcEvent_t fork = nullptr;
    std::vector<hggcEvent_t> decoded, consumed;
    std::mutex submit_mutex;
    // Shape readiness is CPU metadata only; no decoded weights are retained.
    using ShapeKey = std::tuple<int, int, int, int, int, int, int, bool, bool, bool, bool, int, DecodeFencePolicy>;
    std::map<ShapeKey, StreamedPreparedPlan> prepared_plans;

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

// Prepare every declared chunk/tail and numerical variant on the actual caller stream.
static bool run_streamed(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& b,
                         const torch::Tensor& scale, const torch::Tensor& d, const torch::Tensor& rows,
                         const StreamedConfig& execution, const FusedInputs& fused = {}, float inv_scale = 1.f) {
    const auto caller = current_stream();
    hggcStreamCaptureStatus capture = hggcStreamCaptureStatusNone;
    DG_HGGC_RUNTIME_CHECK(hggcStreamIsCapturing(caller, &capture));
    const bool capturing = capture != hggcStreamCaptureStatusNone;
    auto context = streamed_async_context(a.get_device(), caller, !capturing);
    TORCH_CHECK(context, "W4FA8 streamed plan must be warmed outside graph capture on this caller stream");
    const int num_sms = execution.core.num_sms, tile_n = execution.decode_n;
    const int groups = b.size(0), n = b.size(1), k = a.size(1), m = d.size(0);
    constexpr int threads = 512, core_smem = 256 * 1024;
    const bool is_fused = fused.tokens.defined();
    check_streamed_config(execution, m, n, k, groups, is_fused ? GemmType::MGroupedFused : GemmType::MGroupedNoPad);
    const bool allow_sparse = execution.empty_experts == EmptyExpertPolicy::Nonempty;
    struct Work { int group_start, groups, n_begin, n_count; };
    std::vector<Work> plan;
    for (int nb = 0; nb < n; nb += tile_n)
        for (const auto& group : execution.group_chunks)
            plan.push_back({group.first, group.second, nb, std::min(tile_n, n - nb)});
    const int chunk = execution.expert_chunk, parts = int(plan.size());
    const bool ramp = execution.schedule == ChunkSchedule::Ramp;
    const bool chunk_gather = execution.gather == GatherPolicy::Chunk;
    // The entire event submission is atomic with respect to other host callers
    // using this caller stream; previously queued waits retain their event record.
    std::lock_guard<std::mutex> lock(context->submit_mutex);
    const StreamedAsyncContext::ShapeKey key{m, n, k, groups, num_sms, chunk, tile_n, ramp, allow_sparse, is_fused, inv_scale != 1.f, int(execution.gather), execution.decode_fence};
    // A captured call may only reuse resources and kernels warmed on this caller.
    auto prepared = context->prepared_plans.find(key);
    if (capturing && (prepared == context->prepared_plans.end() || !prepared->second.warmed || context->decoded.size() < size_t(parts) ||
                      context->consumed.size() < size_t(parts)))
        return false;
    StreamedGatherRuntime::Args gather_args{k, groups, {dim3(num_sms * 4), dim3(256), 0}, {}};
    if (!capturing) context->reserve_events(parts);
    if (prepared == context->prepared_plans.end()) {
        TORCH_CHECK(!capturing, "W4FA8 streamed variant must be warmed outside graph capture");
        StreamedPreparedPlan next;
        for (const auto& op : plan)
            next.kernels.emplace(std::make_pair(op.groups, op.n_count), StreamedKernels{});
        for (auto& entry : next.kernels) {
            const int local_groups = entry.first.first, local_n = entry.first.second;
            auto& pair = entry.second;
            pair.decode = {n, k, local_n, local_groups, {dim3(1), dim3(256), 0}, {}};
            pair.decode.skip_fence = execution.decode_fence == DecodeFencePolicy::FenceFree;
            // Active experts are read on the device on every call, including
            // Graph replay; empty experts never require decoded weight traffic.
            pair.decode.empty_flat = allow_sparse && local_n == n;
            pair.decode.fuse_gather = chunk_gather && is_fused;
            pair.core = {n, k, local_n, local_groups, {dim3(num_sms), dim3(threads), core_smem}, {}};
            pair.core.apply_inv_scale = inv_scale != 1.f;
            pair.decoder = compiler->build(StreamedDecodeRuntime::kernel_name(pair.decode),
                                           StreamedDecodeRuntime::generate(pair.decode));
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
        if (is_fused && !chunk_gather)
            next.gather = compiler->build("w4a8_streamed_gather", StreamedGatherRuntime::generate(gather_args));
        // Publish only a complete plan. Capture never enters a compiler or module loader.
        prepared = context->prepared_plans.emplace(key, std::move(next)).first;
    }
    if (is_compile_only()) return true;
    const auto& kernels = prepared->second.kernels;
    const auto& gather_kernel = prepared->second.gather;
    log_streamed_config(execution, int(a.size(0)), n, k, groups, m, chunk, parts, prepared->second);

    DgProfParam prof;
    if (ProfilingInterface::Instance().get_op_info()) {
        if (is_fused)
            prof.set_fused_moe_params("w4fa8_mma", "group", groups, a.size(0), m / a.size(0), n, k,
                                      rows.data_ptr<int32_t>(), caller);
        else {
            prof.set_params(GemmType::MGroupedNoPad, false, "w4fa8_mma", groups, m, n, k, ceil_div(m, groups),
                            rows.data_ptr<int32_t>(), caller);
            prof.add_params(std::string("quant_type"), std::string("group"));
        }
        prof.add_params(std::string("group_size"), 32);
    }
    ProfilingInterface::Instance().instrument(true, prof);
    // Both buffers are bounded strict subsets of the experts, decoded anew on
    // every invocation. Largest historical case: 2 * 294 MiB, plus metadata.
    std::array<torch::Tensor, 2> scratch{at::empty({int64_t(chunk) * tile_n * k + 8}, b.options()),
                                         at::empty({int64_t(chunk) * tile_n * k + 8}, b.options())};
    // The inherited collective ABI carries a B-scale pointer, but the streamed
    // mainloop reads only A scales: decoded E5M2 weights include their scales.
    auto b_scale_storage = at::empty({chunk, tile_n, 1}, a_scale.options());
    torch::Tensor core_a = a, core_scale = a_scale;
    w4fa8_streamed::GatherParams gather{};
    if (is_fused) {
        core_a = at::empty({m, k}, a.options());
        core_scale = at::empty({m, 1}, a_scale.options());
        gather = {reinterpret_cast<const uint8_t*>(a.data_ptr()),
                  a_scale.data_ptr<float>(),
                  reinterpret_cast<uint8_t*>(core_a.data_ptr()),
                  core_scale.data_ptr<float>(),
                  fused.experts.data_ptr<int32_t>(),
                  fused.tokens.data_ptr<int32_t>(),
                  fused.aligned.data_ptr<int32_t>(),
                  int(a.size(0))};
        if (!chunk_gather) {
            gather_args.params = gather;
            launch_streamed_on<StreamedGatherRuntime>(gather_kernel, gather_args, caller);
        }
    }
    auto output = reinterpret_cast<cutlass::bfloat16_t*>(d.data_ptr());
    const auto stride_input = cute::Stride<int64_t, cute::Int<1>, int64_t>{k, cute::Int<1>{}, 0};
    const auto stride_output = cute::Stride<int64_t, cute::Int<1>, int64_t>{n, cute::Int<1>{}, 0};

    const int overlap_per_sm = int64_t(m) >= int64_t(groups) * (is_fused ? 4096 : 2048) ? 1 : 2;
    auto decode = [&](int index) {
        const auto& work = plan[index];
        const int group_start = work.group_start, local_groups = work.groups;
        const auto& pair = kernels.at({local_groups, work.n_count});
        auto args = pair.decode;
        const int64_t blocks = (int64_t(local_groups) * work.n_count * k / 16 + 255) / 256;
        // Bound interference while the compute stream is active.
        args.launch_args.grid_dim =
            dim3(std::min<int64_t>(blocks, int64_t(num_sms) * (index == 0 ? 8 : overlap_per_sm)));
        args.params = {b.data_ptr<uint8_t>() + int64_t(group_start) * n * k / 2,
                       scale.data_ptr<uint8_t>() + int64_t(group_start) * n * k / 32,
                       scratch[index & 1].data_ptr<uint8_t>(),
                       work.n_begin,
                       rows.data_ptr<int32_t>(),
                       group_start};
        args.gather = gather;
        launch_streamed_on<StreamedDecodeRuntime>(pair.decoder, args, context->decode);
    };
    auto core = [&](int index) {
        const auto& work = plan[index];
        const int group_start = work.group_start, local_groups = work.groups;
        const auto& pair = kernels.at({local_groups, work.n_count});
        auto args = pair.core;
        const auto& buffer = scratch[index & 1];
        auto range = reinterpret_cast<const int32_t*>(buffer.data_ptr<uint8_t>() + int64_t(local_groups) * work.n_count * k);
        args.inv_scale = inv_scale;
        args.params = {StreamedInt8::GemmUniversalMode::kGemm,
                       {m, work.n_count, k, 1},
                       {core_a.data_ptr(), stride_input, buffer.data_ptr(), stride_input, core_scale.data_ptr<float>(),
                        b_scale_storage.data_ptr<float>()},
                       {{}, output + work.n_begin, stride_output, output + work.n_begin, stride_output},
                       {a.get_device(), num_sms},
                       {uint32_t(m), rows.data_ptr<int32_t>() + group_start, range},
                       nullptr,
                       nullptr};
        launch_streamed_on<StreamedCoreRuntime>(pair.gemm, args, context->compute);
    };

    try {
        // Input preparation precedes the fork on the caller stream.
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
    if (!capturing)
        prepared->second.warmed = true;
    return true;
}

} // namespace deep_gemm::w4fa8::streamed
