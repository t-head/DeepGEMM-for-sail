#pragma once

#include <array>
#include <limits>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>
#include "fp4_gemm.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

static layout::MegaMoEStaging get_staging_capacity(
    int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk,
    int hidden, int intermediate_hidden, const std::array<int, 3>& staging_config) {
    DG_HOST_ASSERT(is_supported_megamoe_rank_count(num_ranks));
    DG_HOST_ASSERT(num_experts > 0 && num_experts <= layout::MegaMoESignals::kNumMaxExperts &&
                   num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 && num_max_tokens_per_rank % layout::kTokenAlignment == 0 &&
                   num_topk >= 1 && num_topk <= 32);
    DG_HOST_ASSERT(hidden >= 1024 && hidden % 512 == 0 && intermediate_hidden > 0 && intermediate_hidden % 64 == 0);
    // With aligned capacity and uniform expert shards, each expert needs capacity / 16 BM16 flags.
    DG_HOST_ASSERT(int64_t(num_experts) * (num_max_tokens_per_rank / 16) <= layout::MegaMoESignals::kNumMaxCopyBlocks);
    // Keep the existing signed 32-bit bound on the full routing space.
    DG_HOST_ASSERT(int64_t(num_ranks) * num_max_tokens_per_rank * num_topk <= std::numeric_limits<int>::max());
    if (staging_config == std::array<int, 3>{} ||
        staging_config == std::array<int, 3>{num_experts, num_max_tokens_per_rank, hidden})
        return {};

    // Aliases use the allocation owner's bank strides for peer writes.
    const auto [owner_experts, owner_capacity, owner_hidden] = staging_config;
    const auto owner = layout::MegaMoEStaging::for_shape(num_ranks, owner_experts, owner_capacity, owner_hidden);
    const auto required = layout::MegaMoEStaging::for_shape(num_ranks, num_experts, num_max_tokens_per_rank, hidden);
    DG_HOST_ASSERT(required.scales <= owner.scales && required.counts <= owner.counts && required.tokens <= owner.tokens);
    return owner;
}

static void check_mega_moe_buffer(const torch::Tensor& buffer, const layout::MegaMoEBuffer& mega_buffer) {
    TORCH_CHECK(buffer.scalar_type() == torch::kUInt8 && buffer.dim() == 1 &&
                buffer.is_contiguous() && buffer.storage_offset() == 0 &&
                buffer.numel() >= mega_buffer.get_num_bytes(), "invalid MegaMoE symmetric buffer size/layout");
}

static int get_kernel_occupancy(const std::shared_ptr<KernelRuntime>& runtime, int threads, int dynamic_smem) {
    construct_launch_config(runtime->kernel, current_stream(), dynamic_smem, dim3(1), dim3(threads));
    int blocks = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, runtime->kernel, threads, dynamic_smem));
#else
    DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, runtime->kernel, threads, dynamic_smem));
#endif
    DG_HOST_ASSERT(blocks > 0);
    return blocks;
}

// Rows and reduce column chunks start on 16-byte boundaries relative to the tensor base.
static bool is_bf16x8_aligned(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

class MXFP4MegaMoEPreDispatchRuntime final : public LaunchRuntime<MXFP4MegaMoEPreDispatchRuntime> {
public:
    struct Args {
        int hidden, num_topk, num_max_tokens_per_rank, num_ranks, num_experts, intermediate_hidden;
        LaunchArgs launch_args;
        const void* hidden_states;
        const torch::Tensor& topk_idx;
        const float* topk_weights;
        uint32_t num_tokens;
        layout::SymBuffer<> sym_buffer;
        layout::MegaMoEStaging staging;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/fp4_mega_moe_pre_dispatch.cuh>
using TopkIdx = {};
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mega_moe_pre_dispatch<{}, {}, {}, {}, {}, {}, {}, {}, {}, TopkIdx, {}>);
}}
)", args.topk_idx.scalar_type() == torch::kInt32 ? "int32_t" : "int64_t",
    args.hidden, args.num_topk, args.num_max_tokens_per_rank, args.num_ranks, args.num_experts, args.intermediate_hidden,
    args.staging.scales, args.staging.counts, args.staging.tokens, is_bf16x8_aligned(args.hidden_states));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.hidden_states, args.topk_idx.data_ptr(),
                                            args.topk_weights, args.num_tokens, args.sym_buffer));
    }
};

struct ScatterEpilogueParams {
    decltype(FP4GemmRuntime::EpilogueArgsNoTsm{}.thread) thread{};
    const layout::TokenSrcMetadata* src_info = nullptr;
    const int32_t* masked_m = nullptr;
    uint32_t token_capacity = 0;
};

// Mirror the device epilogue ABI, with an activation-specific parameter tail.
// SwiGLU stays byte-for-byte compatible with the grouped FP4 epilogue.
template <GatedActivation Activation>
struct Linear1EpilogueArguments {
    decltype(FP4GemmRuntime::EpilogueArgsNoTsm{}.thread) thread{};
    const float* ptr_C = nullptr;
    cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;
    uint8_t* ptr_D = nullptr;
    cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    uint16_t* ptr_SFD = nullptr;
    uint32_t shape_m = 0;
    typename GatedActivationOp<Activation>::Params activation{};
};
static_assert(sizeof(Linear1EpilogueArguments<GatedActivation::SwiGLU>) ==
              sizeof(FP4GemmRuntime::EpilogueArgsSiluAndMulPostQuant));
static_assert(offsetof(Linear1EpilogueArguments<GatedActivation::SwiGLU>, activation) ==
              offsetof(FP4GemmRuntime::EpilogueArgsSiluAndMulPostQuant, swiglu_limit));

template <MegaMoEPhaseKind Stage, int Ranks, GatedActivation Activation = GatedActivation::SwiGLU>
class GemmRuntime final : public LaunchRuntime<GemmRuntime<Stage, Ranks, Activation>> {
public:
    static_assert(Stage == MegaMoEPhaseKind::Linear1 || Stage == MegaMoEPhaseKind::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;

    struct LaunchInfo {
        int n, k, block_m, block_n, block_k, warp_m, warp_n, num_groups, num_stages, num_sms;
        int copy_blocks = 0, k_tiles_per_flag = 0, source_capacity = 1, topk = 1;
        bool apply_clamp = false, wide_offsets = false, warp_pipeline = false;
        bool has_linear_beta = true;
        bool fast_math = true;
    };

    struct Args {
        const char* kernel_name;
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        FP4GemmRuntime::MainLoopArguments mainloop{};
        std::conditional_t<kIsLinear1, Linear1EpilogueArguments<Activation>, ScatterEpilogueParams> epilogue{};
        std::conditional_t<kIsLinear1, sched::MegaMoESchedulerArguments, TileSchedulerArguments> scheduler{};
        layout::MegaMoESignals* signals = nullptr;
        layout::SymBuffer<Ranks> scatter_buffer{};
        MegaMoEReplicaWeights replica_weights{};
        const int64_t* reduce_ids = nullptr;
        const float* reduce_weights = nullptr;
        void* reduce_output = nullptr;
        int tokens = 0, topk = 0;
    };

    static std::string generate_impl(const Args& a) {
        const auto& info = a.launch_info;
        int column_splits = 1;
        int reduce_group_threads = 0;
        if constexpr (!kIsLinear1) {
            const int parts = ceil_div(info.n, 8 * static_cast<int>(a.launch_args.block_dim.x));
            if (a.tokens <= 128 && info.n % parts == 0 && (info.n / parts) % 256 == 0)
                column_splits = parts;
            constexpr int kGroupThreads = 128;
            const int cta_threads = static_cast<int>(a.launch_args.block_dim.x);
            // Group only when the reduce work can cover at least one CTA per SM.
            if (cta_threads > kGroupThreads && cta_threads % kGroupThreads == 0 &&
                a.tokens * column_splits >= info.num_sms * (cta_threads / kGroupThreads))
                reduce_group_threads = kGroupThreads;
        }
        const auto kernel_args = kIsLinear1 ? std::string{"Kernel"} :
            fmt::format("Kernel, {}, {}, {}, {}, {}, {}, {}", Ranks, a.topk, info.n, info.num_groups,
                        column_splits, reduce_group_threads, is_bf16x8_aligned(a.reduce_output));
        return fmt::format(R"(
#define DG_MEGAMOE_OWNER_MODE {}
#define DG_MEGAMOE_OWNER_READER_OPT {}
#define DG_MEGAMOE_SITU_USE_RCP {}
#include <deep_gemm/impls/sm89_fp4_fp4_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEMxfp4Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::GatedActivation::{}, {}, {}, {}>;
using Kernel = Builder::Kernel;
static_assert(Kernel::SharedStorageSize == {}, "MegaMoE shared storage size differs from host");
static_assert(sizeof(Kernel::MainloopArguments) == {}, "MegaMoE mainloop size differs from host");
static_assert(sizeof(Kernel::EpilogueParams) == {}, "MegaMoE epilogue size differs from host");
static_assert(sizeof(Kernel::SchedulerParams) == {}, "MegaMoE scheduler size differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::{}<{}>);
}}
)", static_cast<int>(!kIsLinear1), static_cast<int>(!kIsLinear1),
            static_cast<int>(kIsLinear1 && Activation == GatedActivation::SiTU),
            info.n, info.k, info.block_m, info.block_n, info.block_k, info.warp_m, info.warp_n, info.num_groups, info.num_stages,
            Ranks, kIsLinear1 ? "deep_gemm::MegaMoEPhaseKind::Linear1" : "deep_gemm::MegaMoEPhaseKind::Linear2",
            info.copy_blocks, info.k_tiles_per_flag, info.source_capacity, info.topk,
            info.apply_clamp, info.wide_offsets, info.warp_pipeline,
            Activation == GatedActivation::SiTU ? "SiTU" :
            Activation == GatedActivation::SwiGLUStep ? "SwiGLUStep" :
            Activation == GatedActivation::SwiGLUOAI ? "SwiGLUOAI" : "SwiGLU", info.has_linear_beta, info.fast_math,
            a.replica_weights.weights != nullptr,
            a.launch_args.smem_size, sizeof(a.mainloop), sizeof(a.epilogue), sizeof(a.scheduler),
            a.kernel_name, kernel_args);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        if constexpr (kIsLinear1) {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.mainloop, a.epilogue, a.scheduler,
                                                a.signals, a.replica_weights));
        } else {
            MegaMoEReduceParams reduce{a.reduce_ids, a.reduce_weights, a.reduce_output, a.tokens};
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.mainloop, a.epilogue, a.scheduler,
                                                a.signals, a.scatter_buffer, reduce, a.replica_weights));
        }
    }
};

template <typename F>
static void for_ranks(int ranks, F&& f) {
    switch (ranks) {
        case 1: return f(std::integral_constant<int, 1>{});
        case 2: return f(std::integral_constant<int, 2>{});
        case 4: return f(std::integral_constant<int, 4>{});
        case 8: return f(std::integral_constant<int, 8>{});
        case 16: return f(std::integral_constant<int, 16>{});
        case 32: return f(std::integral_constant<int, 32>{});
        case 64: return f(std::integral_constant<int, 64>{});
        default: DG_HOST_UNREACHABLE("unsupported MegaMoE rank count");
    }
}

static FP4GemmRuntime::MainLoopArguments make_mainloop_arguments(int m, int n, int k, int sfa_m,
                        uint8_t* lhs, uint16_t* lhs_scales,
                        const torch::Tensor& rhs, const torch::Tensor& rhs_scales) {
    using S = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using SF = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    return {
        cute::make_shape(m, n, k), lhs,
        cutlass::make_cute_packed_stride(S{}, cute::make_shape(m, k, 1)),
        rhs.data_ptr<uint8_t>(), cutlass::make_cute_packed_stride(S{}, cute::make_shape(n, k, 1)),
        lhs_scales,
        cutlass::make_cute_packed_stride(SF{}, cute::make_shape(sfa_m, ceil_div(k, 32), 1)),
        rhs_scales.data_ptr<uint16_t>(),
        cutlass::make_cute_packed_stride(SF{}, cute::make_shape(n, ceil_div(k, 32), 1))};
}

template <typename Runtime>
static void launch_gemm(typename Runtime::Args& args, int expected_m) {
    const auto& info = args.launch_info;
    const auto name = args.kernel_name;
    const auto runtime = compiler->build(name, Runtime::generate(args));
    const int blocks = get_kernel_occupancy(runtime, args.launch_args.block_dim.x, args.launch_args.smem_size);
    args.launch_args.grid_dim.x = info.num_sms * blocks;
    DgProfParam profile;
    auto& profiling = ProfilingInterface::Instance();
    if (profiling.get_op_info()) {
        profile.set_params(GemmType::MegaMoE, false, "fp4", info.num_groups,
            cute::get<0>(args.mainloop.problem_shape), info.n, info.k, expected_m, nullptr, current_stream());
        profile.add_params("stage", name);
    }
    profiling.instrument(true, profile);
    Runtime::launch(runtime, args);
    profiling.instrument(false, profile);
    if (get_env<int>("show_log", 0)) {
        int registers = 0, stack_bytes = 0;
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
        hggcFuncAttributes attributes{};
        DG_HGGC_RUNTIME_CHECK(hggcFuncGetAttributes(&attributes, runtime->kernel));
        registers = attributes.numRegs;
        stack_bytes = attributes.localSizeBytes;
#else
        DG_HGGC_DRIVER_CHECK(hgFuncGetAttribute(&registers, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel));
        DG_HGGC_DRIVER_CHECK(hgFuncGetAttribute(&stack_bytes, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel));
#endif
        printf("[%s] groups:%d problem:[%d,%d,%d] num_sms:%d blocks_per_sm:%d grid:%u\n"
               "tile:[%d,%d,%d] warp:[%d,%d] stages:%d smem:%d vreg:%d stack:%d\n",
            name, info.num_groups, cute::get<0>(args.mainloop.problem_shape), info.n, info.k,
            info.num_sms, blocks, args.launch_args.grid_dim.x, info.block_m, info.block_n, info.block_k,
            info.warp_m, info.warp_n, info.num_stages, args.launch_args.smem_size, registers, stack_bytes);
    }
}

template <int Ranks, GatedActivation Activation>
static auto make_linear1_args(
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const MegaMoEReplicaWeights& replicas, int num_experts_per_rank,
    uint8_t* l2_acts, uint16_t* l2_acts_sf, int compact_m, int num_topk,
    const sched::MegaMoESchedulerArguments& scheduler, layout::MegaMoESignals* signals,
    const typename GatedActivationOp<Activation>::Params& activation_params,
    const MegaMoEL1Config& config, bool fast_math) {
    using Runtime = GemmRuntime<MegaMoEPhaseKind::Linear1, Ranks, Activation>;
    bool apply_clamp = false;
    bool has_linear_beta = true;
    if constexpr (Activation == GatedActivation::SiTU)
        has_linear_beta = activation_params.linear_beta > 0.0f;
    if constexpr (Activation == GatedActivation::SwiGLUOAI)
        apply_clamp = std::isfinite(activation_params.limit);
    else if constexpr (Activation != GatedActivation::SiTU)
        apply_clamp = std::isfinite(activation_params);
    DG_HOST_ASSERT(config.num_sms > config.copy_blocks &&
                   config.num_sms <= device_runtime->get_prop()->multiProcessorCount);
    const int n = l1_weights.size(1), k = l1_weights.size(2);
    const int m = scheduler.shape_m;
    using S = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    return typename Runtime::Args{
        .kernel_name = "sm89_fp4_fp4_mega_moe_l1",
        .launch_info = {.n = n, .k = k, .block_m = config.block_m, .block_n = config.block_n, .block_k = config.block_k,
                        .warp_m = config.warp_m, .warp_n = config.warp_n, .num_groups = num_experts_per_rank, .num_stages = config.num_stages,
                        .num_sms = config.num_sms,
                        .copy_blocks = config.copy_blocks, .k_tiles_per_flag = config.k_tiles_per_flag,
                        .source_capacity = m / Ranks, .topk = num_topk,
                        .apply_clamp = apply_clamp, .has_linear_beta = has_linear_beta, .fast_math = fast_math},
        .launch_args = {dim3(1), dim3((config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32), config.smem_size},
        .mainloop = make_mainloop_arguments(m, n, k, compact_m,
            scheduler.local_fp4_buf, scheduler.local_sfa_buf, l1_weights, l1_weights_sf),
        .epilogue = {
            .stride_C = cutlass::make_cute_packed_stride(S{}, cute::make_shape(m, 0, 1)),
            .ptr_D = l2_acts,
            .stride_D = cutlass::make_cute_packed_stride(S{}, cute::make_shape(m, n, 1)),
            .ptr_SFD = l2_acts_sf, .shape_m = static_cast<uint32_t>(compact_m),
            .activation = activation_params},
        .scheduler = scheduler, .signals = signals,
        .replica_weights = replicas};
}

template <int Ranks>
static void launch_linear1(
    GatedActivation activation, float activation_clamp, float situ_beta, float situ_linear_beta, float activation_alpha,
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const MegaMoEReplicaWeights& replicas, int num_experts_per_rank,
    uint8_t* l2_acts, uint16_t* l2_acts_sf, int compact_m, int num_topk,
    const sched::MegaMoESchedulerArguments& scheduler, layout::MegaMoESignals* signals,
    const MegaMoEL1Config& config, bool fast_math, int expected_m) {
    auto launch = [&](auto activation_tag) {
        constexpr auto kind = decltype(activation_tag)::value;
        using Runtime = GemmRuntime<MegaMoEPhaseKind::Linear1, Ranks, kind>;
        typename GatedActivationOp<kind>::Params activation_params;
        if constexpr (kind == GatedActivation::SiTU)
            activation_params = {situ_beta, situ_linear_beta};
        else if constexpr (kind == GatedActivation::SwiGLUOAI)
            activation_params = {activation_clamp, activation_alpha};
        else
            activation_params = activation_clamp;

        auto args = make_linear1_args<Ranks, kind>(l1_weights, l1_weights_sf, replicas, num_experts_per_rank, l2_acts, l2_acts_sf,
                                                   compact_m, num_topk, scheduler, signals, activation_params, config, fast_math);
        launch_gemm<Runtime>(args, expected_m);
    };

    if (activation == GatedActivation::SiTU)
        launch(std::integral_constant<GatedActivation, GatedActivation::SiTU>{});
    else if (activation == GatedActivation::SwiGLUStep)
        launch(std::integral_constant<GatedActivation, GatedActivation::SwiGLUStep>{});
    else if (activation == GatedActivation::SwiGLUOAI)
        launch(std::integral_constant<GatedActivation, GatedActivation::SwiGLUOAI>{});
    else
        launch(std::integral_constant<GatedActivation, GatedActivation::SwiGLU>{});
}

template <int Ranks>
static auto make_linear2_args(
    uint8_t* l2_acts, uint16_t* l2_acts_sf, int m,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const MegaMoEReplicaWeights& replicas, int num_experts_per_rank,
    int32_t* masked_m, const layout::TokenSrcMetadata* src_info, const layout::SymBuffer<Ranks>& scatter_buffer,
    layout::MegaMoESignals* signals, int num_max_tokens_per_rank, int num_topk, MegaMoEL2Config config) {
    using Runtime = GemmRuntime<MegaMoEPhaseKind::Linear2, Ranks>;
    const int n = l2_weights.size(1), k = l2_weights.size(2);
    const bool warp_pipeline = config.block_m == 256 && config.block_n == 256;
    if (warp_pipeline) {
        config.block_m = 128;
        config.smem_size = std::get<0>(deep_gemm_fp4_common::get_smem_config_fp4(
            config.num_stages, config.block_m, config.block_n, config.warp_m,
            config.warp_n, config.block_k, n, false, false)) + 3 * 32 * 256 * 2;
    }
    return typename Runtime::Args{
        .kernel_name = "sm89_fp4_fp4_mega_moe_l2",
        .launch_info = {.n = n, .k = k, .block_m = config.block_m, .block_n = config.block_n, .block_k = config.block_k,
                        .warp_m = config.warp_m, .warp_n = config.warp_n, .num_groups = num_experts_per_rank, .num_stages = config.num_stages,
                        .num_sms = config.num_sms,
                        .wide_offsets = static_cast<int64_t>(num_topk) * num_max_tokens_per_rank * n > UINT32_MAX,
                        .warp_pipeline = warp_pipeline},
        .launch_args = {dim3(1), dim3(warp_pipeline ? scatter_pipeline::kCtaThreads :
            (config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32), config.smem_size},
        .mainloop = make_mainloop_arguments(m, n, k, m, l2_acts, l2_acts_sf, l2_weights, l2_weights_sf),
        .epilogue = {.src_info = src_info, .masked_m = masked_m,
                     .token_capacity = static_cast<uint32_t>(num_max_tokens_per_rank)},
        .scheduler = TileSchedulerArguments(m, masked_m), .signals = signals,
        .scatter_buffer = scatter_buffer, .replica_weights = replicas};
}

static void fp4_mega_moe_pre_dispatch_impl(
    const torch::Tensor& hidden_states, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
    const std::vector<int64_t>& sym_buffer_ptrs,
    int num_ranks, int num_experts, int num_max_tokens_per_rank, int rank_idx,
    int hidden, int intermediate_hidden, int num_topk, const layout::MegaMoEStaging& staging) {
    MXFP4MegaMoEPreDispatchRuntime::Args args{hidden, num_topk, num_max_tokens_per_rank, num_ranks, num_experts, intermediate_hidden,
        {dim3(1), dim3(256), 0}, hidden_states.data_ptr(), topk_idx, topk_weights.data_ptr<float>(),
        static_cast<uint32_t>(hidden_states.size(0)), layout::SymBuffer<>(sym_buffer_ptrs, rank_idx), staging};
    const auto runtime = compiler->build("mega_moe_pre_dispatch", MXFP4MegaMoEPreDispatchRuntime::generate(args));
    const int count_blocks = num_experts / num_ranks;
    const int resident = get_kernel_occupancy(runtime, 256, 0) * device_runtime->get_prop()->multiProcessorCount;
    const int token_blocks = std::max(1, std::min(static_cast<int>(args.num_tokens), std::max(1, resident - count_blocks)));
    args.launch_args.grid_dim.x = count_blocks + token_blocks;
    MXFP4MegaMoEPreDispatchRuntime::launch(runtime, args);
}

static void sm89_fp4_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const MegaMoEReplicaWeights& l1_replicas, const MegaMoEReplicaWeights& l2_replicas,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    uint8_t* base, const layout::MegaMoEBuffer& mega_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    int rank_idx, int num_max_tokens_per_rank,
    int num_experts_per_rank, int num_tokens, int num_topk, int hidden, int intermediate_hidden,
    float activation_clamp, int expected_m, GatedActivation activation,
    float situ_beta, float situ_linear_beta, float activation_alpha, bool fast_math) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto config = get_mega_moe_config(num_ranks, num_experts_per_rank, num_max_tokens_per_rank,
                                            hidden, intermediate_hidden, expected_m, device_runtime->get_num_sms());
    const auto& l1_config = config.l1;
    const auto& l2_config = config.l2;

    auto* grouped_layout = mega_buffer.grouped_layout.data<int32_t>(base);
    auto* masked_m = grouped_layout + (1 + mega_buffer.blocks) * 4;
    sched::MegaMoESchedulerArguments scheduler(num_ranks * num_max_tokens_per_rank, masked_m);
    scheduler.staging = mega_buffer.staging;
    scheduler.copy_grouped_layout = grouped_layout;
    scheduler.expert_rank_prefix = mega_buffer.expert_rank_prefix.data<uint32_t>(base);
    scheduler.local_fp4_buf = mega_buffer.local_fp4.data(base);
    scheduler.local_sfa_buf = mega_buffer.local_sfa.data<uint16_t>(base);
    scheduler.copy_ready_flags = mega_buffer.copy_ready.data<uint32_t>(base);
    scheduler.dispatch_fp4 = mega_buffer.dispatch_fp4.data(base);
    scheduler.dispatch_token_topk = mega_buffer.staging_tokens[0].data<uint32_t>(base);
    scheduler.expert_base = mega_buffer.expert_base.data<uint32_t>(base);
    scheduler.src_info = mega_buffer.src_info.data<layout::TokenSrcMetadata>(base);
    scheduler.sym_buffer = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx);
    scheduler.num_copy_blocks = l1_config.copy_blocks;
    scheduler.cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats
        ? cumulative_local_expert_recv_stats->data_ptr<int32_t>() : nullptr;
    auto* signals = reinterpret_cast<layout::MegaMoESignals*>(base);
    auto* l2_acts = mega_buffer.l2_acts.data(base);
    auto* l2_acts_sf = mega_buffer.l2_acts_sf.data<uint16_t>(base);
    const int compact_m = static_cast<int>(mega_buffer.compact);
    for_ranks(num_ranks, [&](auto r) {
        constexpr int Ranks = decltype(r)::value;
        using Linear2Runtime = GemmRuntime<MegaMoEPhaseKind::Linear2, Ranks>;
        auto scatter_buffer = layout::SymBuffer<Ranks>(sym_buffer_ptrs, rank_idx);
        scatter_buffer.base += mega_buffer.scatter_storage.offset;
        auto l2_args = make_linear2_args<Ranks>(
            l2_acts, l2_acts_sf, compact_m, l2_weights, l2_weights_sf, l2_replicas, num_experts_per_rank,
            masked_m, scheduler.src_info, scatter_buffer, signals, num_max_tokens_per_rank, num_topk, l2_config);
        l2_args.reduce_ids = mega_buffer.topk_idx.data<int64_t>(base);
        l2_args.reduce_weights = mega_buffer.topk_weights.data<float>(base);
        l2_args.reduce_output = y.data_ptr();
        l2_args.tokens = num_tokens;
        l2_args.topk = num_topk;

        launch_linear1<Ranks>(activation, activation_clamp, situ_beta, situ_linear_beta, activation_alpha,
                              l1_weights, l1_weights_sf, l1_replicas, num_experts_per_rank,
                              l2_acts, l2_acts_sf, compact_m, num_topk, scheduler, signals, l1_config, fast_math, expected_m);
        launch_gemm<Linear2Runtime>(l2_args, expected_m);
    });
}

}  // namespace deep_gemm::mega
