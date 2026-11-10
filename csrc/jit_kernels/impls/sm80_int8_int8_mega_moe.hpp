#pragma once

#include <array>
#include "../../utils/layout.hpp"
#include <cstring>
#include <memory>
#include <vector>
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/scheduler/sm80_mega_moe.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>
#include "int8_gemm.hpp"
#include "mega_moe_common.hpp"
#include "../heuristics/sm80_mega_moe.hpp"
#if __has_include(<hggc_ad.h>)
#include <hggc_ad.h>
#endif

namespace deep_gemm::mega_int8 {

using mega_moe_common::for_ranks;
using mega_moe_common::get_kernel_occupancy;

static layout::MegaMoEStaging get_mega_moe_staging(
        int ranks, int experts, int capacity, int hidden,
        const std::array<int, 3>& staging_config = {}) {
    TORCH_CHECK(experts > 0 && experts <= layout::MegaMoESignals::kNumMaxExperts,
                "MegaMoE expert count exceeds the shared protocol capacity");
    const auto required = layout::int8_staging_for_shape(ranks, experts, capacity, hidden);
    if (staging_config == std::array<int, 3>{}) return required;
    const auto [owner_experts, owner_capacity, owner_hidden] = staging_config;
    TORCH_CHECK(owner_experts > 0 && owner_experts % ranks == 0 &&
                owner_experts <= layout::MegaMoESignals::kNumMaxExperts &&
                owner_capacity > 0 && owner_capacity % layout::kInt8TokenAlignment == 0 &&
                owner_hidden >= 2048 && owner_hidden % 512 == 0,
                "invalid MegaMoE allocation staging configuration");
    const auto owner = layout::int8_staging_for_shape(
        ranks, owner_experts, owner_capacity, owner_hidden);
    TORCH_CHECK(required.scales <= owner.scales && required.counts <= owner.counts &&
                required.tokens <= owner.tokens, "MegaMoE view exceeds allocation staging capacity");
    return owner;
}

template <typename Runtime>
static void submit_mega_gemm(const std::shared_ptr<KernelRuntime>& runtime,
                             const typename Runtime::Args& args, const char* dtype, int groups,
                             hggcStream_t stream = current_stream()) {
    DgProfParam profile;
    if (deep_gemm::profiling_enabled()) {
        const auto& shape = args.kernel_params.problem_shape;
        profile.set_params(GemmType::MegaMoE, false, dtype, groups,
            cute::get<0>(shape), cute::get<1>(shape), cute::get<2>(shape), 0, nullptr, stream);
        profile.add_params("stage", args.kernel_name);
    }
    auto profiling_guard = deep_gemm::show_gemm_ut(profile);
    Runtime::launch(runtime, args, stream);
}

template <typename Runtime, typename Tile>
static void log_mega_gemm(const std::shared_ptr<KernelRuntime>& runtime,
                          const typename Runtime::Args& args, const Tile& tile,
                          int groups, int num_sms) {
    if (!get_env<int>("show_log", 0))
        return;
    const auto [registers, stack_bytes] = mega_moe_common::get_kernel_resources(runtime);
    const auto& shape = args.kernel_params.problem_shape;
    const auto grid = args.launch_args.grid_dim.x;
    printf("[%s] groups:%d problem:[%d,%d,%d] num_sms:%d blocks_per_sm:%u grid:%u\n"
           "tile:[%d,%d,%d] warp:[%d,%d] stages:%d smem:%d vreg:%d stack:%d\n",
        args.kernel_name, groups, cute::get<0>(shape), cute::get<1>(shape), cute::get<2>(shape),
        num_sms, grid / num_sms, grid, tile.block_m, tile.block_n, tile.block_k,
        tile.warp_m, tile.warp_n, tile.num_stages, args.launch_args.smem_size, registers, stack_bytes);
}

class TokenPublishRuntime final : public LaunchRuntime<TokenPublishRuntime> {
public:
    struct Args {
        int hidden, topk, capacity, ranks, experts, intermediate;
        LaunchArgs launch_args;
        const void* input;
        const void* ids;
        const float* weights;
        uint32_t tokens;
        layout::SymBuffer<> sym_buffer;
        layout::MegaMoEStaging staging{};
        bool int64_ids = false;
    };

    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/int8_mega_moe_pre_dispatch.cuh>
using TopkIdx = {};
extern "C" __global__ void megamoe_quant(
    const __ppu_bfloat16* input, const TopkIdx* ids, const float* weights,
    uint32_t tokens, void* base, deep_gemm::layout::SymBuffer<> sym_buffer) {{
    deep_gemm::publish_moe_tokens<deep_gemm::Int8TokenQuantizer<{}>, {}, {}, {}, {}, {}, {}, {}, {}, {}, TopkIdx>(
        input, ids, weights, tokens, base, sym_buffer);
}}
)", a.int64_ids ? "int64_t" : "int32_t", is_bf16x8_aligned(a.input), a.hidden, a.topk, a.capacity, a.ranks, a.experts, a.intermediate,
            a.staging.scales, a.staging.counts, a.staging.tokens);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.input, a.ids, a.weights, a.tokens,
                                   a.sym_buffer.get_base_ptr(), a.sym_buffer));
    }
};



#if defined(HGGCAD_VERSION)
DECL_LAZY_HGGC_DRIVER_FUNCTION(hgLaunchKernelExAD);
#endif

// Copy runs on the vector-CU pool without a launch-completion event.
template <class Derived>
class CopyLaunchRuntime : public LaunchRuntime<Derived> {
public:
    template <class Args>
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
#if defined(HGGCAD_VERSION)
        HGlaunchAttributeAD attr{};
        attr.id = HGAD_LAUNCH_ATTRIBUTE_YIELD_MULTIPROCESSOR;
        attr.value.yieldMultiprocessor = HGAD_YIELD_MULTIPROCESSOR_INTENSIVE;
        HGlaunchConfigAD launch_config{};
        launch_config.gridDimX = a.launch_args.grid_dim.x;
        launch_config.gridDimY = a.launch_args.grid_dim.y;
        launch_config.gridDimZ = a.launch_args.grid_dim.z;
        launch_config.blockDimX = a.launch_args.block_dim.x;
        launch_config.blockDimY = a.launch_args.block_dim.y;
        launch_config.blockDimZ = a.launch_args.block_dim.z;
        launch_config.sharedMemBytes = a.launch_args.smem_size;
        launch_config.attrs = &attr;
        launch_config.numAttrs = 1;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
        HGfunction function{};
        DG_HGGC_DRIVER_CHECK(lazy_hgKernelGetFunction(&function, reinterpret_cast<HGkernel>(kernel)));
        launch_config.hStream = reinterpret_cast<HGstream>(config.stream);
#else
        const HGfunction function = kernel;
        launch_config.hStream = config.hStream;
#endif
        void* params[] = {&a.scheduler};
        DG_HGGC_DRIVER_CHECK(lazy_hgLaunchKernelExAD(&launch_config, function, params, nullptr));
#else
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.scheduler));
#endif
    }
};

// Independent callers must not reuse mutable event state. Each capture stream
// is warmed up before capture, and every fork is rejoined on the caller stream.
struct CopyStreamResources {
    int device;
    hggcStream_t caller, copy = nullptr;
    hggcEvent_t fork = nullptr, done = nullptr;

    CopyStreamResources(int device_, hggcStream_t caller_) : device(device_), caller(caller_) {
        try {
            DG_HGGC_RUNTIME_CHECK(hggcStreamCreateWithFlags(&copy, hggcStreamNonBlocking));
            DG_HGGC_RUNTIME_CHECK(hggcEventCreateWithFlags(&fork, hggcEventDisableTiming));
            DG_HGGC_RUNTIME_CHECK(hggcEventCreateWithFlags(&done, hggcEventDisableTiming));
        } catch (...) {
            if (done) hggcEventDestroy(done);
            if (fork) hggcEventDestroy(fork);
            if (copy) hggcStreamDestroy(copy);
            throw;
        }
    }
    CopyStreamResources(const CopyStreamResources&) = delete;
    CopyStreamResources& operator=(const CopyStreamResources&) = delete;
    ~CopyStreamResources() {
        int previous = device;
        if (hggcGetDevice(&previous) != hggcSuccess || hggcSetDevice(device) != hggcSuccess)
            return;
        hggcEventDestroy(done);
        hggcEventDestroy(fork);
        hggcStreamDestroy(copy);
        if (previous != device) hggcSetDevice(previous);
    }
};

static CopyStreamResources& copy_stream_resources(hggcStream_t caller) {
    int device = 0;
    DG_HGGC_RUNTIME_CHECK(hggcGetDevice(&device));
    static thread_local std::vector<std::unique_ptr<CopyStreamResources>> resources;
    for (const auto& entry : resources)
        if (entry->device == device && entry->caller == caller)
            return *entry;
    hggcStreamCaptureStatus capture;
    DG_HGGC_RUNTIME_CHECK(hggcStreamIsCapturing(caller, &capture));
    TORCH_CHECK(capture == hggcStreamCaptureStatusNone,
                "Warm up split-copy MegaMoE on the capture stream before Graph capture");
    resources.emplace_back(std::make_unique<CopyStreamResources>(device, caller));
    return *resources.back();
}


struct Int8GatedEpilogueParams : INT8GemmCutlass3Runtime::EpilogueParamNoTsm {
    float activation_limit = 0.0f;
};

struct Int8ScatterEpilogueParams {
    INT8GemmCutlass3Runtime::LinearCombinationArgs thread{};
    const void* src_info = nullptr;
    const int32_t* masked_m = nullptr;
    uint32_t token_capacity = 0;
};

template <int Ranks>
struct Int8ScatterParams : INT8GemmCutlass3Runtime::GemmKernelParamsT<Int8ScatterEpilogueParams> {
    layout::SymBuffer<Ranks> scatter_buffer{};
    layout::MegaMoESignals* signals = nullptr;
};

template <MegaMoEPhaseKind Stage, int Ranks>
class GemmRuntime final : public LaunchRuntime<GemmRuntime<Stage, Ranks>> {
public:
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;
    static constexpr auto kName = kIsLinear1 ? "megamoe_int8_gemm1" : "megamoe_int8_scatter";
    static constexpr auto kReduceName = "megamoe_int8_reduce";
    static constexpr int kRanks = Ranks;
    using Params = std::conditional_t<kIsLinear1,
        INT8GemmCutlass3Runtime::GemmKernelParamsT<Int8GatedEpilogueParams,
                                                sched::Int8MegaMoESchedulerArguments>,
        Int8ScatterParams<Ranks>>;
    using Config = std::conditional_t<kIsLinear1, MegaMoEStage1Config, MegaMoEStageConfig>;
    struct Args {
        Config tile;
        int hidden, intermediate, groups, capacity, topk;
        LaunchArgs launch_args;
        Params kernel_params{};
        const int64_t* reduce_ids = nullptr;
        const float* reduce_weights = nullptr;
        void* reduce_output = nullptr;
        int tokens = 0;
        bool normal_quantization = false, apply_clamp = false, fast_math = true;
        uint32_t activation_limit_bits = 0xffffffffu;
        GatedActivation activation = GatedActivation::SwiGLU;
        uint32_t beta_bits = 0, linear_beta_bits = 0, alpha_bits = 0;
        const char* kernel_name = kName;
        bool split_sync = false;
    };

    static std::string generate_impl(const Args& a) {
        const auto& c = a.tile;
        const auto arch = is_ppu1v5_device() ? "PPU0015" : "PPU0010";
        // Keep the default SwiGLU specialization unchanged. Other modes pass
        // their FP4-compatible activation parameters and math policy explicitly.
        const auto activation = a.activation == GatedActivation::SwiGLU ? "SwiGLU" :
            a.activation == GatedActivation::SiTU ? "SiTU" :
            a.activation == GatedActivation::SwiGLUStep ? "SwiGLUStep" : "SwiGLUOAI";
        const auto activation_suffix = a.activation == GatedActivation::SwiGLU && a.fast_math ? std::string{} :
            fmt::format(",deep_gemm::GatedActivation::{},{}u,{}u,{}u,{}",
                        activation, a.beta_bits, a.linear_beta_bits, a.alpha_bits, a.fast_math);
        const bool wait_scatter_pair = !kIsLinear1 && Ranks == 16 && c.block_n == 256 &&
            device_runtime->get_arch_pair() == std::make_pair(8, 0) &&
            get_env<std::string>("DG_MEGAMOE_SM80_EP16_SCATTER_WAIT") == "1";
        std::string linear;
        if constexpr (kIsLinear1) {
            linear = fmt::format("deep_gemm::MegaMoEInt8Linear1<cutlass::arch::{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},cutlass::bfloat16_t,true,{},{},{}u{}>",
                arch, a.intermediate * 2, a.hidden, c.block_m, c.block_n, c.block_k,
                c.warp_m, c.warp_n, c.num_stages, a.groups, Ranks, a.capacity, a.topk,
                c.copy_blocks, c.k_tiles_per_flag, a.normal_quantization, a.apply_clamp, a.activation_limit_bits, activation_suffix);
        } else {
            linear = fmt::format("deep_gemm::MegaMoEInt8Linear2<cutlass::arch::{},{},{},{},{},{},{},{},{},{},{},{},{},{}>",
                arch, a.hidden, a.intermediate, c.block_m, c.block_n, c.block_k,
                c.warp_m, c.warp_n, c.num_stages, a.groups, Ranks, a.capacity, a.topk, wait_scatter_pair);
        }
        const bool reduce_only = a.split_sync && std::strcmp(a.kernel_name, kReduceName) == 0;
        const std::string execute = kIsLinear1 ? "Kernel{}(params, smem);" : reduce_only ? fmt::format(
            "deep_gemm::mega_moe_int8_reduce_impl<Kernel,{},{},{},{},{}>(params, ids, weights, output, tokens, params.signals->scatter_writer_mask);",
            a.hidden, a.topk, a.groups, Ranks, is_bf16x8_aligned(a.reduce_output)) : fmt::format(
            "deep_gemm::mega_moe_int8_scatter_impl<Kernel,{},{},{},{},{}>(params, smem, ids, weights, output, tokens);",
            a.hidden, a.topk, a.groups, Ranks, is_bf16x8_aligned(a.reduce_output));
        return fmt::format(R"(
#define DG_MEGAMOE_OWNER_MODE 1
#define DG_MEGAMOE_SITU_USE_RCP {}
#define DG_MEGAMOE_SPLIT_SYNC {}
#include <deep_gemm/impls/sm80_int8_int8_mega_moe.cuh>
using Linear = {};
using Kernel = Linear::Kernel;
static_assert(sizeof(Kernel::Params) == {});
static_assert(offsetof(Kernel::Params, mainloop) == {});
static_assert(offsetof(Kernel::Params, epilogue) == {});
static_assert(offsetof(Kernel::Params, scheduler) == {});
static_assert(Kernel::SharedStorageSize == {});
extern "C" __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock) void {}(
    Kernel::Params params{}) {{
    extern __shared__ char smem[];
    {}
}}
)", static_cast<int>(kIsLinear1 && a.activation == GatedActivation::SiTU),
            static_cast<int>(a.split_sync),
            linear, sizeof(Params), offsetof(Params, collective_mainloop_params),
            offsetof(Params, collective_epilogue_params), offsetof(Params, scheduler),
            a.launch_args.smem_size, a.kernel_name,
            kIsLinear1 ? "" : ", const int64_t* ids, const float* weights, __ppu_bfloat16* output, int tokens",
            execute);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        if constexpr (kIsLinear1)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.kernel_params));
        else
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.kernel_params,
                                               a.reduce_ids, a.reduce_weights, a.reduce_output, a.tokens));
    }
};

class Int8MegaMoEQuantRuntime final : public LaunchRuntime<Int8MegaMoEQuantRuntime> {
public:
    struct Args {
        int intermediate;
        bool normal_quantization;
        int64_t expected_rows;
        LaunchArgs launch_args;
        const void* input;
        int8_t* output;
        float* scale;
        const uint32_t* rows;
    };
    static std::string generate_impl(const Args& a) {
        // Short batches need a CTA per row to keep the launch grid populated.
        const bool warp_rows = device_runtime->get_arch_pair() == std::make_pair(8, 0) &&
            a.expected_rows >= int64_t(a.launch_args.grid_dim.x) * (a.launch_args.block_dim.x / 32);
        const auto quantize = warp_rows
            ? fmt::format("deep_gemm::int8_quant_activated_warp<{},256,{}>(input, output, scale, rows);",
                          a.intermediate, a.normal_quantization)
            : fmt::format("deep_gemm::int8_silu_mul_quant_impl<{},256,false,{},true>(input, output, scale, rows, 0.0f);",
                          a.intermediate, a.normal_quantization);
        return fmt::format(R"(
#include <deep_gemm/impls/int8_mega_moe_activation.cuh>
extern "C" __global__ __launch_bounds__(256) void megamoe_int8_quant(
        const __ppu_bfloat16* input, int8_t* output, float* scale,
        const uint32_t* rows) {{
    {}
}}
)", quantize);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.input, a.output, a.scale, a.rows));
    }
};

class Int8MegaMoECopyRuntime final : public CopyLaunchRuntime<Int8MegaMoECopyRuntime> {
public:
    struct Args {
        int n, k, block_m, block_k, groups, ranks, k_tiles_per_flag, capacity, topk, threads;
        LaunchArgs launch_args;
        sched::Int8MegaMoESchedulerArguments scheduler;
        bool split_sync = false;
    };
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#define DG_MEGAMOE_SPLIT_SYNC {}
#include <deep_gemm/impls/int8_mega_moe_dispatch.cuh>
using Kernel = deep_gemm::sm80::MegaMoECopyKernel<deep_gemm::sm80::Int8FusedDispatch<{}>,
    {},{},{},{},{},{},{},{},{},{}>;
static_assert(Kernel::SharedStorageSize == {});
static_assert(sizeof(Kernel::Arguments) == {});
extern "C" __global__ __launch_bounds__({}) void megamoe_int8_dispatch_copy(
        Kernel::Arguments scheduler) {{
    extern __shared__ char smem[];
    Kernel::run(scheduler, smem);
}}
)", static_cast<int>(a.split_sync), a.block_k, a.n, a.k, a.block_m, a.block_k, a.groups, a.ranks,
            a.k_tiles_per_flag, a.capacity, a.topk, a.threads,
            a.launch_args.smem_size, sizeof(a.scheduler), a.threads);
    }
};

template <typename Runtime>
static void launch_gemm(typename Runtime::Args& args, int num_sms) {
    auto runtime = compiler->build(args.kernel_name, Runtime::generate(args));
    const int blocks = get_kernel_occupancy(runtime, args.launch_args.block_dim.x, args.launch_args.smem_size);
    args.launch_args.grid_dim.x = num_sms * blocks;
    args.kernel_params.hw_info.cu_count = args.launch_args.grid_dim.x;
    submit_mega_gemm<Runtime>(runtime, args, "int8", args.groups);
    log_mega_gemm<Runtime>(runtime, args, args.tile, args.groups, num_sms);
}

template <typename Runtime>
static void launch_int8_split_stage1(typename Runtime::Args& args, int num_sms) {
    TORCH_CHECK(device_runtime->get_prop()->concurrentKernels,
                "Split-copy MegaMoE requires concurrent kernel execution");
    auto& tile = args.tile;
    const int threads = args.launch_args.block_dim.x;
    const int copy_threads = get_int8_mega_moe_copy_threads(
        threads, get_env<int>("DG_MEGAMOE_COPY_THREADS", 0));
    const int copy_ctas = tile.copy_blocks;
    DG_HOST_ASSERT(copy_ctas > 0 && copy_ctas <= device_runtime->get_prop()->multiProcessorCount);
    DG_HOST_ASSERT(copy_threads >= 64 && copy_threads >= Runtime::kRanks &&
                   copy_threads % 32 == 0 && copy_threads <= device_runtime->get_prop()->maxThreadsPerBlock);
    constexpr int ranks = Runtime::kRanks;
    const int row_map_bytes = ceil_div(4 * ranks + 8 * tile.block_m, 16) * 16;
    const int smem = std::max(row_map_bytes, 4 * args.groups * (ranks + 4));
    Int8MegaMoECopyRuntime::Args copy_args{
        args.intermediate * 2, args.hidden, tile.block_m, tile.block_k, args.groups, ranks,
        tile.k_tiles_per_flag, args.capacity, args.topk, copy_threads,
        {dim3(copy_ctas), dim3(copy_threads), smem}, args.kernel_params.scheduler, args.split_sync};

    // Every persistent GEMM CTA computes. Copy uses a separate grid and its
    // own intensive launch instead of taking slots from the Tensor CU budget.
    tile.copy_blocks = 0;
    const std::array<int64_t, 26> key{
        args.hidden, args.intermediate, args.groups, args.capacity, args.topk,
        tile.block_m, tile.block_n, tile.block_k, tile.warp_m, tile.warp_n, tile.num_stages,
        tile.k_tiles_per_flag, copy_ctas, num_sms, threads, copy_threads, args.launch_args.smem_size,
        args.normal_quantization, args.apply_clamp, args.activation_limit_bits,
        static_cast<int>(args.activation), args.beta_bits, args.linear_beta_bits, args.alpha_bits, args.fast_math,
        args.split_sync};
    HGcontext context = nullptr;
    DG_HGGC_DRIVER_CHECK(hgCtxGetCurrent(&context));
    const auto* active_compiler = compiler.operator->();
    struct Plan {
        std::array<int64_t, 26> key;
        HGcontext context;
        const Compiler* compiler;
        std::string signature, flags;
        std::shared_ptr<KernelRuntime> compute, copy;
        int compute_grid;
    };
    static thread_local std::vector<Plan> plans;
    const bool cache_plan = get_env<int>("DG_MEGAMOE_CACHE_SPLIT_PLAN", 1) != 0;
    const Plan* cached = nullptr;
    if (cache_plan) {
        for (const auto& plan : plans) {
            if (plan.key == key && plan.context == context && plan.compiler == active_compiler &&
                plan.signature == active_compiler->signature && plan.flags == active_compiler->flags) {
                cached = &plan;
                break;
            }
        }
    }
    std::shared_ptr<KernelRuntime> compute, copy;
    if (cached) {
        compute = cached->compute;
        copy = cached->copy;
        args.launch_args.grid_dim.x = cached->compute_grid;
    } else {
        compute = active_compiler->build(args.kernel_name, Runtime::generate(args));
        const int blocks = get_kernel_occupancy(compute, threads, args.launch_args.smem_size);
        args.launch_args.grid_dim.x = num_sms * blocks;
        copy = active_compiler->build("megamoe_int8_dispatch_copy", Int8MegaMoECopyRuntime::generate(copy_args));
        get_kernel_occupancy(copy, copy_threads, smem);
        if (cache_plan)
            plans.push_back({key, context, active_compiler, active_compiler->signature,
                             active_compiler->flags, compute, copy,
                             static_cast<int>(args.launch_args.grid_dim.x)});
    }
    args.kernel_params.hw_info.cu_count = args.launch_args.grid_dim.x;
    log_mega_gemm<Runtime>(compute, args, tile, args.groups, num_sms);
    if (get_env<int>("show_log", 0))
        printf("[megamoe_int8_dispatch_copy] grid:%d threads:%d smem:%d compute_grid:%u\n",
               copy_ctas, copy_threads, smem, args.launch_args.grid_dim.x);
    if (is_compile_only())
        return;

    const auto caller = current_stream();
    auto& streams = copy_stream_resources(caller);
    // Both streams wait for the caller's inputs. GEMM consumes copy stripes
    // through device-ready flags; join copy before reusing the workspace.
    DG_HGGC_RUNTIME_CHECK(hggcEventRecord(streams.fork, caller));
    DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(streams.copy, streams.fork, 0));
    try {
        Int8MegaMoECopyRuntime::launch(copy, copy_args, streams.copy);
        submit_mega_gemm<Runtime>(compute, args, "int8", args.groups, caller);
        DG_HGGC_RUNTIME_CHECK(hggcEventRecord(streams.done, streams.copy));
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(caller, streams.done, 0));
    } catch (...) {
        hggcEventRecord(streams.done, streams.copy);
        hggcStreamWaitEvent(caller, streams.done, 0);
        throw;
    }
}

static auto make_packed_stride(int m, int n) {
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    return cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, n, 1));
}

static LaunchArgs make_launch_args(const MegaMoEStageConfig& config, bool scatter) {
    const int smem = config.num_stages * (config.block_m + config.block_n) * config.block_k;
    const int epilogue = scatter
        ? 2 * (config.block_m / config.warp_m) * 16 * config.block_n * int(sizeof(float)) : 0;
    return {dim3(1), dim3((config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32),
            std::max(smem, epilogue)};
}

template <int Ranks>
static auto make_linear1_args(
        const std::tuple<torch::Tensor, torch::Tensor>& w1, int intermediate,
        const layout::Int8MegaMoEBuffer& buffers, uint8_t* base, const layout::SymBuffer<>& symmetric,
        int capacity, int topk, const MegaMoEStage1Config& config, int num_sms,
        GatedActivation activation, float limit, bool normal_quantization,
        float beta, float linear_beta, float alpha, bool fast_math,
        int32_t* cumulative_local_expert_recv_stats) {
    using Runtime = GemmRuntime<MegaMoEPhaseKind::Linear1, Ranks>;
    const auto& [weight, scale] = w1;
    const int hidden = weight.size(2), groups = weight.size(0);
    typename Runtime::Args args{config, hidden, intermediate, groups, capacity, topk,
                                make_launch_args(config, false)};
    args.normal_quantization = normal_quantization;
    args.apply_clamp = std::isfinite(limit);
    args.activation = activation;
    args.fast_math = fast_math;
    if (activation != GatedActivation::SwiGLU) {
        std::memcpy(&args.beta_bits, &beta, sizeof(beta));
        std::memcpy(&args.linear_beta_bits, &linear_beta, sizeof(linear_beta));
        std::memcpy(&args.alpha_bits, &alpha, sizeof(alpha));
    }
    // Preserve the finite-clamp specialization and its split-plan cache key.
    if (args.apply_clamp && !is_ppu1v5_device())
        std::memcpy(&args.activation_limit_bits, &limit, sizeof(limit));
    auto& params = args.kernel_params;
    params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    params.problem_shape = {Ranks * capacity, intermediate * 2, hidden, 1};
    params.collective_mainloop_params = {
        buffers.local_data.data(base), make_packed_stride(Ranks * capacity, hidden),
        weight.data_ptr(), make_packed_stride(intermediate * 2, hidden),
        buffers.local_sfa.data<float>(base), scale.data_ptr<float>()};
    auto* out = buffers.activated.data<cutlass::bfloat16_t>(base);
    static_cast<INT8GemmCutlass3Runtime::EpilogueParamNoTsm&>(params.collective_epilogue_params) = {
        {}, out, make_packed_stride(Ranks * capacity, intermediate),
        out, make_packed_stride(Ranks * capacity, intermediate)};
    params.collective_epilogue_params.activation_limit = limit;
    params.hw_info = {weight.get_device(), num_sms};
    params.scheduler = sched::make_int8_mega_moe_scheduler(buffers, base, Ranks * capacity);
    params.scheduler.sym_buffer = symmetric;
    params.scheduler.local_source_rank = symmetric.rank_idx;
    params.scheduler.cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats;
    params.scheduler.num_copy_blocks = config.copy_blocks;
    return args;
}

template <int Ranks>
static void launch_linear1(
        typename GemmRuntime<MegaMoEPhaseKind::Linear1, Ranks>::Args& args,
        const layout::Int8MegaMoEBuffer& buffers,
        uint8_t* base, int num_tokens, int num_sms) {
    if (args.split_sync)
        mega_moe_common::launch_sync("mega_moe_dispatch_sync", Ranks,
            reinterpret_cast<layout::MegaMoESignals*>(base));
    launch_int8_split_stage1<GemmRuntime<MegaMoEPhaseKind::Linear1, Ranks>>(args, num_sms);

    Int8MegaMoEQuantRuntime::Args quantization{
        args.intermediate, args.normal_quantization, int64_t(num_tokens) * args.topk,
        {dim3(num_sms * 4), dim3(256), 0}, buffers.activated.data(base),
        buffers.stage1_data.data<int8_t>(base), buffers.stage1_scale_storage.data<float>(base),
        buffers.expert_base.data<uint32_t>(base) + args.groups};
    auto kernel = compiler->build("megamoe_int8_quant", Int8MegaMoEQuantRuntime::generate(quantization));
    const int vector_grid = device_runtime->get_prop()->multiProcessorCount * 4;
    if (device_runtime->get_arch_pair() == std::make_pair(8, 0) &&
        vector_grid > quantization.launch_args.grid_dim.x &&
        quantization.expected_rows >= int64_t(vector_grid) * (quantization.launch_args.block_dim.x / 32))
        quantization.launch_args.grid_dim.x = vector_grid;
    Int8MegaMoEQuantRuntime::launch(kernel, quantization);
}

template <int Ranks>
static auto make_linear2_args(
        const std::tuple<torch::Tensor, torch::Tensor>& w2, const layout::Int8MegaMoEBuffer& buffers,
        uint8_t* base, const std::vector<int64_t>& pointers, int rank, int capacity, int topk,
        int* masked_m, const MegaMoEStageConfig& config, int num_sms) {
    using Runtime = GemmRuntime<MegaMoEPhaseKind::Linear2, Ranks>;
    const auto& [weight, scale] = w2;
    const int hidden = weight.size(1), intermediate = weight.size(2), groups = weight.size(0);
    typename Runtime::Args args{config, hidden, intermediate, groups, capacity, topk,
                                make_launch_args(config, true)};
    auto& params = args.kernel_params;
    params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    params.problem_shape = {static_cast<int>(buffers.compact), hidden, intermediate, 1};
    params.collective_mainloop_params = {
        buffers.stage1_data.data(base), make_packed_stride(buffers.compact, intermediate),
        weight.data_ptr(), make_packed_stride(hidden, intermediate),
        buffers.stage1_scale_storage.data<float>(base), scale.data_ptr<float>()};
    params.collective_epilogue_params = {
        {}, buffers.src_info.data(base), masked_m, static_cast<uint32_t>(capacity)};
    params.hw_info = {weight.get_device(), num_sms};
    params.scheduler = TileSchedulerArguments(buffers.compact, masked_m);
    params.scatter_buffer = layout::SymBuffer<Ranks>(pointers, rank);
    params.scatter_buffer.base += buffers.scatter_storage.offset;
    params.signals = reinterpret_cast<layout::MegaMoESignals*>(base);
    return args;
}

template <int Ranks>
static void launch_linear2(typename GemmRuntime<MegaMoEPhaseKind::Linear2, Ranks>::Args& args, int num_sms) {
    using Runtime = GemmRuntime<MegaMoEPhaseKind::Linear2, Ranks>;
    launch_gemm<Runtime>(args, num_sms);
    if (args.split_sync) {
        mega_moe_common::launch_sync("mega_moe_scatter_sync", Ranks, args.kernel_params.signals);
        args.kernel_name = Runtime::kReduceName;
        launch_gemm<Runtime>(args, num_sms);
    }
}

template <int Ranks>
static void int8_mega_moe_impl(
        const torch::Tensor& y,
        const std::tuple<torch::Tensor, torch::Tensor>& w1,
        const std::tuple<torch::Tensor, torch::Tensor>& w2,
        uint8_t* base, const layout::Int8MegaMoEBuffer& buffers, const std::vector<int64_t>& pointers,
        int rank, int capacity, int topk, int expected_m, float limit, bool normal_quantization,
        GatedActivation activation = GatedActivation::SwiGLU,
        float beta = 0, float linear_beta = 0, float alpha = 1.702f,
        int32_t* cumulative_local_expert_recv_stats = nullptr,
        bool fast_math = true) {
    const int hidden = std::get<0>(w1).size(2), intermediate = std::get<0>(w2).size(2);
    const int groups = std::get<0>(w1).size(0), num_sms = device_runtime->get_num_sms();
    const auto config = select_int8_mega_moe({num_sms, Ranks, groups, capacity, hidden, intermediate, expected_m},
        get_env<int>("DG_MEGAMOE_COPY_THREADS", 0), normal_quantization);
    const bool split_sync = get_env<int>("DG_MEGAMOE_SPLIT_SYNC", 0) != 0;

    auto l1_args = make_linear1_args<Ranks>(w1, intermediate, buffers, base, layout::SymBuffer<>(pointers, rank),
        capacity, topk, config.l1, num_sms, activation, limit, normal_quantization,
        beta, linear_beta, alpha, fast_math, cumulative_local_expert_recv_stats);
    l1_args.split_sync = split_sync;
    launch_linear1<Ranks>(l1_args, buffers, base, y.size(0), num_sms);

    auto l2_args = make_linear2_args<Ranks>(w2, buffers, base, pointers, rank, capacity, topk,
        l1_args.kernel_params.scheduler.grouped_layout, config.l2, num_sms);
    l2_args.reduce_ids = buffers.topk_idx.data<int64_t>(base);
    l2_args.reduce_weights = buffers.topk_weights.data<float>(base);
    l2_args.reduce_output = y.data_ptr();
    l2_args.tokens = y.size(0);
    l2_args.split_sync = split_sync;
    launch_linear2<Ranks>(l2_args, num_sms);
}

static void int8_mega_moe_quant_impl(const torch::Tensor& x,
        const torch::Tensor& ids, const torch::Tensor& weights,
        const std::vector<int64_t>& pointers, int rank, int ranks, int experts, int capacity,
        int intermediate, layout::MegaMoEStaging staging) {
    using Runtime = mega_int8::TokenPublishRuntime;
    Runtime::Args args{int(x.size(1)), int(ids.size(1)), capacity, ranks, experts, intermediate,
        {dim3(1), dim3(256), 0}, x.data_ptr(), ids.data_ptr(), weights.data_ptr<float>(),
        static_cast<uint32_t>(x.size(0)), layout::SymBuffer<>(pointers, rank), staging,
        ids.scalar_type() == torch::kInt64};
    const auto runtime = compiler->build("megamoe_quant", Runtime::generate(args));
    const int groups = experts / ranks;
    const int resident = mega_int8::get_kernel_occupancy(runtime, 256, 0) * device_runtime->get_prop()->multiProcessorCount;
    TORCH_CHECK(groups < resident, "INT8 MegaMoE publication needs a resident token CTA");
    args.launch_args.grid_dim.x = groups + std::max(1, std::min(int(args.tokens), resident - groups));
    Runtime::launch(runtime, args);
}

static void int8_int8_mega_moe_impl(uint8_t* base, const layout::Int8MegaMoEBuffer& buffers,
        int capacity, int topk,
        const std::vector<int64_t>& pointers, int rank, const torch::Tensor& y,
        const std::tuple<torch::Tensor, torch::Tensor>& w1,
        const std::tuple<torch::Tensor, torch::Tensor>& w2,
        const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
        int expected_m,
        GatedActivation activation, float limit, float beta, float linear_beta, float alpha, bool fast_math) {
    mega_int8::for_ranks(pointers.size(), [&](auto r) {
        mega_int8::int8_mega_moe_impl<decltype(r)::value>(y, w1, w2, base, buffers, pointers,
            rank, capacity, topk, expected_m, limit, false,
            activation, beta, linear_beta, alpha,
            cumulative_local_expert_recv_stats ? cumulative_local_expert_recv_stats->data_ptr<int32_t>() : nullptr,
            fast_math);
    });
}

}  // namespace deep_gemm::mega_int8
