#pragma once

#include <array>
#include "../../utils/layout.hpp"
#include <limits>
#include <string_view>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>
#include "fp4_gemm.hpp"
#include "mega_moe_runtime.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

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

static constexpr char kReduceKernelName[] = "sm89_fp4_fp4_mega_moe_reduce";

template <MegaMoEPhaseKind Stage, int Ranks, GatedActivation Activation = GatedActivation::SwiGLU>
class GemmRuntime final : public LaunchRuntime<GemmRuntime<Stage, Ranks, Activation>> {
public:
    static_assert(Stage == MegaMoEPhaseKind::Linear1 || Stage == MegaMoEPhaseKind::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;

    using SharedParams = FP4GemmRuntime::GemmKernelParamsT<
        std::conditional_t<kIsLinear1, Linear1EpilogueArguments<Activation>, FP4GemmRuntime::EpilogueArgsNoTsm>>;

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
        bool split_sync = false;
        int shared_n = 0, shared_k = 0;
        int32_t* shared_control = nullptr;
        SharedParams shared_params{};
    };

    static std::string generate_impl(const Args& a) {
        const auto& info = a.launch_info;
        mega::ReduceConfig reduce;
        if constexpr (!kIsLinear1)
            reduce = mega::select_reduce_config(
                a.tokens, info.n, static_cast<int>(a.launch_args.block_dim.x), info.num_sms);
        const char* stage = kIsLinear1 ? "deep_gemm::MegaMoEPhaseKind::Linear1" : "deep_gemm::MegaMoEPhaseKind::Linear2";
        const char* activation = Activation == GatedActivation::SiTU ? "SiTU" :
                Activation == GatedActivation::SwiGLUStep ? "SwiGLUStep" :
                Activation == GatedActivation::SwiGLUOAI ? "SwiGLUOAI" : "SwiGLU";
        std::string shared_type;
        if (a.shared_n) {
            shared_type = fmt::format(R"(
using SharedKernel = deep_gemm::MegaMoEMxfp4Kernel<{}, {}, {}, {}, {}, {}, {}, 1, {}, 1, {},
    {}, 0, 1, 1, {}, false, false, deep_gemm::GatedActivation::{}, {}, {}, false, true>::Kernel;
static_assert(sizeof(SharedKernel::Params) == {});
static_assert(SharedKernel::MaxThreadsPerBlock == Kernel::MaxThreadsPerBlock);
static_assert(SharedKernel::SharedStorageSize <= {});
)", a.shared_n, a.shared_k, info.warp_pipeline ? 256 : info.block_m, info.block_n, info.block_k,
                info.warp_m, info.warp_n, info.num_stages, stage, kIsLinear1 ? info.copy_blocks : 0,
                info.apply_clamp, activation, info.has_linear_beta, info.fast_math, sizeof(SharedParams), a.launch_args.smem_size);
        }
        const auto kernel_args = kIsLinear1 ? std::string{"Kernel"} :
            fmt::format("Kernel, {}, {}, {}, {}, {}, {}, {}", Ranks, a.topk, info.n, info.num_groups,
                        reduce.column_splits, reduce.group_threads, is_bf16x8_aligned(a.reduce_output));
        return fmt::format(R"(
#define DG_MEGAMOE_HAS_SHARED {}
#define DG_MEGAMOE_OWNER_MODE {}
#define DG_MEGAMOE_OWNER_READER_OPT {}
#define DG_MEGAMOE_SPLIT_SYNC {}
#include <deep_gemm/impls/sm89_fp4_fp4_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEMxfp4Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::GatedActivation::{}, {}, {}, {}>;
using Kernel = Builder::Kernel;
{}
static_assert(Kernel::SharedStorageSize <= {}, "MegaMoE shared storage size differs from host");
static_assert(sizeof(Kernel::MainloopArguments) == {}, "MegaMoE mainloop size differs from host");
static_assert(sizeof(Kernel::EpilogueParams) == {}, "MegaMoE epilogue size differs from host");
static_assert(sizeof(Kernel::SchedulerParams) == {}, "MegaMoE scheduler size differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::{}<{}{}>);
}}
)", static_cast<int>(a.shared_n > 0), static_cast<int>(!kIsLinear1), static_cast<int>(!kIsLinear1),
            static_cast<int>(a.split_sync),
            info.n, info.k, info.block_m, info.block_n, info.block_k, info.warp_m, info.warp_n, info.num_groups, info.num_stages,
            Ranks, kIsLinear1 ? "deep_gemm::MegaMoEPhaseKind::Linear1" : "deep_gemm::MegaMoEPhaseKind::Linear2",
            info.copy_blocks, info.k_tiles_per_flag, info.source_capacity, info.topk,
            info.apply_clamp, info.wide_offsets, info.warp_pipeline,
            activation, info.has_linear_beta, info.fast_math,
            a.replica_weights.weights != nullptr, shared_type,
            a.launch_args.smem_size, sizeof(a.mainloop), sizeof(a.epilogue), sizeof(a.scheduler),
            a.kernel_name, kernel_args, a.shared_n ? ", SharedKernel" : "");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        if constexpr (kIsLinear1) {
            if (a.shared_n) {
                DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.mainloop, a.epilogue, a.scheduler,
                                                a.signals, a.replica_weights, a.shared_params, a.shared_control));
            } else {
                DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.mainloop, a.epilogue, a.scheduler,
                                                    a.signals, a.replica_weights));
            }
        } else {
            MegaMoEReduceParams reduce{a.reduce_ids, a.reduce_weights, a.reduce_output, a.tokens};
            if (a.split_sync && std::string_view(a.kernel_name) == kReduceKernelName) {
                if (a.shared_n) {
                    DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.scatter_buffer, a.epilogue.token_capacity,
                                                    a.signals, reduce, a.shared_params, a.shared_control));
                } else {
                    DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.scatter_buffer, a.epilogue.token_capacity,
                                                        a.signals, reduce));
                }
            } else {
                if (a.shared_n) {
                    DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.mainloop, a.epilogue, a.scheduler,
                                                    a.signals, a.scatter_buffer, reduce, a.replica_weights, a.shared_params, a.shared_control));
                } else {
                    DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.mainloop, a.epilogue, a.scheduler,
                                                        a.signals, a.scatter_buffer, reduce, a.replica_weights));
                }
            }
        }
    }
};

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
    const MegaMoEGemmProfile profile{
        "fp4", info.num_groups, cute::get<0>(args.mainloop.problem_shape), info.n, info.k, expected_m,
        {info.num_sms, info.block_m, info.block_n, info.block_k, info.warp_m, info.warp_n,
         info.num_stages, args.launch_args.smem_size}};
    launch_profiled<Runtime>(runtime, args, profile);
    log_gemm(runtime, name, args.launch_args, profile, blocks);
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
            scheduler.local_data_buf, scheduler.local_sfa_buf, l1_weights, l1_weights_sf),
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
    const MegaMoEL1Config& config, bool fast_math, int expected_m, bool split_sync,
    int num_tokens, int capacity, uint8_t* base, const layout::MegaMoEBuffer& buffers,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1) {
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
        if (shared_l1) {
            const auto& [weights, scales] = *shared_l1;
            args.shared_n = weights.size(0);
            args.shared_k = weights.size(1);
            args.shared_control = buffers.shared_control.data<int32_t>(base);
            auto& p = args.shared_params;
            p.problem_shape = {num_tokens, args.shared_n, args.shared_k, 1};
            p.collective_mainloop_params = make_mainloop_arguments(num_tokens, args.shared_n,
                args.shared_k, capacity, buffers.dispatch_data.data(base),
                buffers.shared_l1_sf.data<uint16_t>(base), weights, scales);
            using S = cute::Stride<int64_t, cute::Int<1>, int64_t>;
            p.collective_epilogue_params = {
                .stride_C = cutlass::make_cute_packed_stride(S{}, cute::make_shape(num_tokens, 0, 1)),
                .ptr_D = buffers.shared_l2_acts.data(base),
                .stride_D = cutlass::make_cute_packed_stride(S{}, cute::make_shape(num_tokens, args.shared_n, 1)),
                .ptr_SFD = buffers.shared_l2_sf.data<uint16_t>(base),
                .shape_m = static_cast<uint32_t>(capacity), .activation = activation_params};
            p.scheduler = TileSchedulerArguments(num_tokens);
        }
        args.split_sync = split_sync;
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
    float situ_beta, float situ_linear_beta, float activation_alpha, bool fast_math,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto config = get_mega_moe_config(num_ranks, num_experts_per_rank, num_max_tokens_per_rank,
                                            hidden, intermediate_hidden, expected_m, device_runtime->get_num_sms());
    const auto& l1_config = config.l1;
    const auto& l2_config = config.l2;
    const bool split_sync = get_env<int>("DG_MEGAMOE_SPLIT_SYNC", 0) != 0;

    auto scheduler = make_mega_moe_scheduler<uint16_t>(mega_buffer, base, num_ranks * num_max_tokens_per_rank);
    auto* masked_m = scheduler.grouped_layout;
    scheduler.sym_buffer = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx);
    scheduler.num_copy_blocks = l1_config.copy_blocks;
    scheduler.cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats
        ? cumulative_local_expert_recv_stats->data_ptr<int32_t>() : nullptr;
    auto* signals = reinterpret_cast<layout::MegaMoESignals*>(base);
    auto* l2_acts = mega_buffer.l2_acts.data(base);
    auto* l2_acts_sf = mega_buffer.l2_acts_sf.data<uint16_t>(base);
    const int compact_m = static_cast<int>(mega_buffer.compact);
    if (split_sync) launch_sync("mega_moe_dispatch_sync", num_ranks, signals);
    for_ranks(num_ranks, [&](auto r) {
        constexpr int Ranks = decltype(r)::value;
        using Linear2Runtime = GemmRuntime<MegaMoEPhaseKind::Linear2, Ranks>;
        auto scatter_buffer = layout::SymBuffer<Ranks>(sym_buffer_ptrs, rank_idx);
        scatter_buffer.base += mega_buffer.scatter_storage.offset;
        auto l2_args = make_linear2_args<Ranks>(
            l2_acts, l2_acts_sf, compact_m, l2_weights, l2_weights_sf, l2_replicas, num_experts_per_rank,
            masked_m, scheduler.src_info, scatter_buffer, signals, num_max_tokens_per_rank, num_topk, l2_config);
        if (shared_l1) {
            const auto& [w2, sf2] = *shared_l2;
            auto* control = mega_buffer.shared_control.data<int32_t>(base);
            l2_args.shared_control = control;
            using S = cute::Stride<int64_t, cute::Int<1>, int64_t>;
            l2_args.shared_n = w2.size(0);
            l2_args.shared_k = w2.size(1);
            auto& p2 = l2_args.shared_params;
            p2.problem_shape = {num_tokens, l2_args.shared_n, l2_args.shared_k, 1};
            p2.collective_mainloop_params = make_mainloop_arguments(num_tokens, l2_args.shared_n,
                l2_args.shared_k, num_max_tokens_per_rank, mega_buffer.shared_l2_acts.data(base), mega_buffer.shared_l2_sf.data<uint16_t>(base), w2, sf2);
            p2.collective_epilogue_params.ptr_D = mega_buffer.shared_output.data<cutlass::bfloat16_t>(base);
            p2.collective_epilogue_params.stride_C = cutlass::make_cute_packed_stride(S{}, cute::make_shape(num_tokens, hidden, 1));
            p2.collective_epilogue_params.stride_D = p2.collective_epilogue_params.stride_C;
            p2.scheduler = TileSchedulerArguments(num_tokens);
            // Shared GEMM2 is dense: use all 16 warps on BM256, while routed
            // GEMM2 keeps its BM128 compute/scatter pipeline. Storage is reused.
            if (l2_args.launch_info.warp_pipeline)
                l2_args.launch_args.smem_size = std::max(l2_args.launch_args.smem_size,
                    std::get<0>(deep_gemm_fp4_common::get_smem_config_fp4(
                        l2_config.num_stages, 256, l2_config.block_n, l2_config.warp_m,
                        l2_config.warp_n, l2_config.block_k, hidden, false, false)));
        }
        l2_args.reduce_ids = mega_buffer.topk_idx.data<int64_t>(base);
        l2_args.reduce_weights = mega_buffer.topk_weights.data<float>(base);
        l2_args.reduce_output = y.data_ptr();
        l2_args.tokens = num_tokens;
        l2_args.topk = num_topk;
        l2_args.split_sync = split_sync;

        launch_linear1<Ranks>(activation, activation_clamp, situ_beta, situ_linear_beta, activation_alpha,
                              l1_weights, l1_weights_sf, l1_replicas, num_experts_per_rank,
                              l2_acts, l2_acts_sf, compact_m, num_topk, scheduler, signals, l1_config, fast_math, expected_m, split_sync,
                              num_tokens, num_max_tokens_per_rank, base, mega_buffer, shared_l1);
        if (split_sync) {
            launch_gemm<Linear2Runtime>(l2_args, expected_m);
            launch_sync("mega_moe_scatter_sync", num_ranks, signals);
            l2_args.kernel_name = kReduceKernelName;
        }
        launch_gemm<Linear2Runtime>(l2_args, expected_m);
    });
}

}  // namespace deep_gemm::mega
