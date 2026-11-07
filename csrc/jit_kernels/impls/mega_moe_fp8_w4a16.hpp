#pragma once

#include "mega_moe_combine.hpp"
#include "mega_moe_fp8_w4a16_dispatch.hpp"
#include "mega_moe_fp8_w4a16_scatter.hpp"

namespace deep_gemm::mega_moe_fp8_w4a16 {

struct RoutedMoEWorkspace {
    torch::Tensor dispatched_tokens, l2_acts, l2_acts_sf, bf16_scratch, block_info;
    torch::Tensor m_rows, valid_m_rows, src_m_offsets, token_meta;
    torch::Tensor combine, peer_offsets, dispatch_counts;
    torch::Tensor topk_idx, topk_weights, barrier_flag, barrier_done;
    layout::SymBuffer<> sym_buffer{};
};

struct W4A16ScatterParams {
    const void* ptr_a;
    const void* ptr_b;
    const void* ptr_scale;
    const void* ptr_b_replicas;
    const void* ptr_scale_replicas;
    uint32_t num_master_groups;
    void* ptr_d;
    int num_token, num_topk;
    const int32_t* sorted_token_ids;
    const int64_t* packed_src_info;
    const int64_t* sym_offsets;
    const int32_t* valid_m_rows;
    int rank_idx, num_max_tokens_per_rank;
    float situ_beta, situ_linear_beta;
    TileSchedulerArguments scheduler;
};

struct W4A16ScatterTag {
    using Runtime = PPUW4A16MegaMoERuntime;
};

struct RoutedMoEStatsTag {
    using Runtime = PPUFP8MegaMoERuntime;
};
inline void update_recv_stats(const RoutedMoEWorkspace& buffer, const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    if (!cumulative_local_expert_recv_stats)
        return;
    PreDispatchConfig c{256, 0, 0, 0};
    launch<RoutedMoEStatsTag>(
        "mega_moe_update_recv_stats", R"(
extern "C" __device__ __constant__ unsigned int ppu_mega_moe_smem_bytes = 0;
extern "C" __global__ void PPU_KERNEL_ENTRY(const int* valid,int* cumulative,int groups) {
    const int expert=blockIdx.x*blockDim.x+threadIdx.x;
    if (expert<groups) atomicAdd(cumulative+expert,valid[expert]);
}
)",
        c, std::make_tuple(buffer.valid_m_rows.data_ptr<int32_t>(),
                           cumulative_local_expert_recv_stats->data_ptr<int32_t>(), int(buffer.valid_m_rows.numel())),
        dim3(ceil_div(int(buffer.valid_m_rows.numel()), 256)), 256);
}

inline void w4a16_scatter_impl(const RoutedMoEWorkspace& buffer, const torch::Tensor& weights, const torch::Tensor& scales,
                               const torch::Tensor& replicas, const torch::Tensor& replica_scales,
                               const deep_gemm_w4a16_common::W4A16Config& config, int groups, int hidden,
                               int intermediate_hidden, int rank_idx, int num_max_tokens_per_rank, int num_topk, int num_ranks,
                               const std::string& kernel_name) {
    const int m = buffer.l2_acts.size(0);
    auto* layout = scatter_layout(groups, buffer.m_rows, buffer.block_info);
    W4A16ScatterParams params{buffer.l2_acts.data_ptr(),
                              weights.data_ptr(),
                              scales.data_ptr(),
                              replicas.data_ptr(),
                              replica_scales.data_ptr(),
                              uint32_t(weights.size(0)),
                              buffer.combine.data_ptr(),
                              m,
                              num_topk,
                              nullptr,
                              buffer.token_meta.data_ptr<int64_t>(),
                              buffer.peer_offsets.data_ptr<int64_t>(),
                              buffer.valid_m_rows.data_ptr<int32_t>(),
                              rank_idx,
                              num_max_tokens_per_rank,
                              0,
                              0,
                              {uint32_t(m), layout}};
    ScatterConfig c;
    c.num_ranks = num_ranks;
    c.n = hidden;
    c.k = intermediate_hidden;
    c.block_m = config.block_m;
    c.block_n = config.block_n;
    c.block_k = config.block_k;
    c.warp_m = config.warp_m;
    c.warp_n = config.warp_n;
    c.num_groups = groups;
    c.num_stages = config.num_stages;
    auto source = std::string(R"(
#include <deep_gemm/impls/w4a16_gemm_cutlass3.cuh>
using Kernel = cutlass::gemm::kernel::W4A16GEMM_MMA<
    {N},{K},{BLOCK_M},{BLOCK_N},{BLOCK_K},{WARP_M},{WARP_N},{BLOCK_K},
    {NUM_GROUPS},{NUM_STAGES},deep_gemm::GemmType::MGroupedNoPad,1,true,false,true,false,{NUM_RANKS}>;
extern "C" __device__ __constant__ uint32_t ppu_mega_moe_smem_bytes = sizeof(Kernel::SharedStorage);
static_assert(sizeof(Kernel::Params) == {ABI_SIZE}, "W4A16 scatter parameter ABI mismatch");
extern "C" __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock,1)
void PPU_KERNEL_ENTRY(typename Kernel::Params params) {
    extern __shared__ char smem[];
    Kernel op;
    op(params,smem);
}
)");
    replace_source_field(source, "ABI_SIZE", std::to_string(sizeof(params)));
    launch<W4A16ScatterTag>(kernel_name, source, c, std::make_tuple(params), dim3(config.num_sms),
                            deep_gemm_w4a16_common::get_num_threads(config, true), true);
}

struct PPUFP8MegaMoERuntime::Args {
    KernelPhase kernel_phase;
    const RoutedMoEWorkspace& sym_buffer;
    const torch::Tensor& y;
    DispatchConfig l1_config;
    ScatterConfig l2_config;
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights;
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights;
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_replicas;
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_replicas;
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats;
    Precision precision;
    int rank_idx, num_tokens;
};

struct PPUW4A16MegaMoERuntime::Args {
    KernelPhase kernel_phase;
    const RoutedMoEWorkspace& sym_buffer;
    const torch::Tensor& y;
    DispatchConfig l1_config;
    deep_gemm_w4a16_common::W4A16Config l2_config;
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights;
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights;
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_replicas;
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_replicas;
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats;
    int rank_idx, num_tokens;
    float situ_beta, situ_linear_beta;
};

inline void combine_mega_moe(const RoutedMoEWorkspace& buffer, const DispatchConfig& config,
                             const torch::Tensor& y, int rank_idx, int num_tokens) {
    auto args = make_mega_moe_combine_args(
        buffer.combine, buffer.topk_weights.narrow(0, 0, num_tokens), buffer.topk_idx, config.num_experts,
        num_tokens, y.size(1), config.num_topk, config.nmtpr, buffer.sym_buffer, buffer.barrier_flag,
        rank_idx, config.num_ranks, buffer.barrier_done, buffer.dispatch_counts, std::nullopt, 0, y);
    const auto runtime = compiler->build("ppu_mega_moe_barrier_reduce", MegaMoECombineRuntime::generate(args));
    reset_done(buffer.barrier_done);
    MegaMoECombineRuntime::launch(runtime, args);
}

inline void PPUFP8MegaMoERuntime::compile_and_launch(const std::string& tag, const Args& args) {
    const auto& buffer = args.sym_buffer;
    const auto& config = args.l1_config;
    const auto& [l1_weights, l1_weights_sf] = args.l1_weights;
    const auto& [l2_weights, l2_weights_sf] = args.l2_weights;
    const auto& [l1_weights_replicas, l1_weights_replicas_sf] = args.l1_weights_replicas;
    const auto& [l2_weights_replicas, l2_weights_replicas_sf] = args.l2_weights_replicas;
    if (args.kernel_phase == KernelPhase::Linear1) {
        if (args.precision == Precision::FP8Blockwise) {
            fp8_blockwise_mega_moe_dispatch_gemm1_kernel(
                config, buffer.sym_buffer, buffer.dispatched_tokens, args.rank_idx, args.num_tokens,
                l1_weights, l1_weights_sf, buffer.l2_acts, buffer.l2_acts_sf,
                config.activation_clamp, config.activation_alpha, l1_weights_replicas, l1_weights_replicas_sf,
                buffer.block_info, tag);
        } else {
            fp8_channelwise_mega_moe_dispatch_gemm1_kernel(
                config, buffer.sym_buffer, args.rank_idx, args.num_tokens, l1_weights, l1_weights_sf,
                l1_weights_replicas, l1_weights_replicas_sf, l1_weights.size(0), buffer.bf16_scratch,
                config.activation_clamp, buffer.block_info, tag);
            fp8_channelwise_mega_moe_quantize_local_pool_kernel(
                config, buffer.sym_buffer, args.rank_idx, buffer.bf16_scratch, buffer.l2_acts, buffer.l2_acts_sf);
        }
        update_recv_stats(buffer, args.cumulative_local_expert_recv_stats);
        return;
    }

    TORCH_CHECK(args.kernel_phase == KernelPhase::Linear2, "Invalid MegaMoE kernel phase");
    const int hidden = args.y.size(1);
    const auto output = buffer.combine.narrow(0, 0, hidden).view({1, hidden});
    if (args.precision == Precision::FP8Blockwise) {
        fp8_blockwise_mega_moe_scatter_kernel(
            args.l2_config, buffer.l2_acts, buffer.l2_acts_sf, l2_weights, l2_weights_sf, output,
            buffer.m_rows, buffer.valid_m_rows, buffer.src_m_offsets, buffer.token_meta, buffer.combine,
            buffer.peer_offsets, buffer.block_info, buffer.l2_acts.size(0), config.nmtpr, config.num_topk,
            args.rank_idx, config.num_ranks, config.num_sms, l2_weights_replicas, l2_weights_replicas_sf, tag);
    } else {
        fp8_channelwise_mega_moe_scatter_kernel(
            args.l2_config, buffer.l2_acts, buffer.l2_acts_sf, l2_weights, l2_weights_sf,
            l2_weights_replicas, l2_weights_replicas_sf, output, buffer.l2_acts.size(0), buffer.m_rows,
            buffer.block_info, config.num_sms, buffer.token_meta, buffer.combine, hidden * 2, config.nmtpr,
            config.num_topk, buffer.peer_offsets, args.rank_idx, config.num_ranks, buffer.valid_m_rows,
            buffer.src_m_offsets, buffer.valid_m_rows, config.num_experts_per_rank, tag);
    }
    combine_mega_moe(buffer, config, args.y, args.rank_idx, args.num_tokens);
}

inline void PPUW4A16MegaMoERuntime::compile_and_launch(const std::string& tag, const Args& args) {
    const auto& buffer = args.sym_buffer;
    auto config = args.l1_config;
    const auto& [l1_weights, l1_weights_sf] = args.l1_weights;
    const auto& [l2_weights, l2_weights_sf] = args.l2_weights;
    const auto& [l1_weights_replicas, l1_weights_replicas_sf] = args.l1_weights_replicas;
    const auto& [l2_weights_replicas, l2_weights_replicas_sf] = args.l2_weights_replicas;
    if (args.kernel_phase == KernelPhase::Linear1) {
        w4a16_mega_moe_dispatch_gemm1_kernel(
            config, buffer.sym_buffer, args.rank_idx, args.num_tokens, l1_weights, l1_weights_sf,
            l1_weights_replicas, l1_weights_replicas_sf, l1_weights.size(0), buffer.l2_acts,
            args.situ_beta, args.situ_linear_beta, tag);
        update_recv_stats(buffer, args.cumulative_local_expert_recv_stats);
        return;
    }

    TORCH_CHECK(args.kernel_phase == KernelPhase::Linear2, "Invalid MegaMoE kernel phase");
    w4a16_scatter_impl(
        buffer, l2_weights, l2_weights_sf, l2_weights_replicas, l2_weights_replicas_sf, args.l2_config,
        config.num_experts_per_rank, args.y.size(1), l1_weights.size(1) / 2, args.rank_idx,
        config.nmtpr, config.num_topk, config.num_ranks, tag);
    combine_mega_moe(buffer, config, args.y, args.rank_idx, args.num_tokens);
}

inline void fp8_fp8_mega_moe_impl(
    const torch::Tensor& y, const std::tuple<torch::Tensor, torch::Tensor>& l1_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_replicas,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_replicas,
    const RoutedMoEWorkspace& sym_buffer, int rank_idx, int num_tokens,
    const std::string& activation, float activation_clamp, bool fast_math, float activation_alpha,
    const RoutedMoEConfigs& configs, Precision precision,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    auto config = configs.dispatch;
    config.swiglu_oai = activation == "swiglu_oai";
    config.activation_alpha = activation_alpha;
    config.activation_clamp = activation_clamp;
    config.fast_math = fast_math;
    const auto args = PPUFP8MegaMoERuntime::Args{
        PPUFP8MegaMoERuntime::KernelPhase::Linear1, sym_buffer, y, config, configs.scatter,
        l1_weights, l2_weights, l1_weights_replicas, l2_weights_replicas,
        cumulative_local_expert_recv_stats, precision, rank_idx, num_tokens};
    const auto launch_with_phase = [&](PPUFP8MegaMoERuntime::KernelPhase kernel_phase, const char* kernel_name) {
        auto split_args = args;
        split_args.kernel_phase = kernel_phase;
        PPUFP8MegaMoERuntime::compile_and_launch(kernel_name, split_args);
    };
    launch_with_phase(PPUFP8MegaMoERuntime::KernelPhase::Linear1,
                      precision == Precision::FP8Blockwise ? "fp8_blockwise_mega_moe_dispatch_gemm1"
                                                         : "fp8_channelwise_mega_moe_dispatch_gemm1");
    launch_with_phase(PPUFP8MegaMoERuntime::KernelPhase::Linear2,
                      precision == Precision::FP8Blockwise
                          ? "fp8_blockwise_mega_moe_scatter"
                          : "m_grouped_gemm_fp8_channelwise_moe_scatter");
}

inline void bf16_fp4_mega_moe_impl(
    const torch::Tensor& y, const std::tuple<torch::Tensor, torch::Tensor>& l1_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_replicas,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_replicas,
    const RoutedMoEWorkspace& sym_buffer, int rank_idx, int num_tokens, bool fast_math,
    float situ_beta, float situ_linear_beta, const RoutedMoEConfigs& configs,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    auto config = configs.dispatch;
    config.fast_math = fast_math;
    const auto args = PPUW4A16MegaMoERuntime::Args{
        PPUW4A16MegaMoERuntime::KernelPhase::Linear1, sym_buffer, y, config, configs.w4_scatter,
        l1_weights, l2_weights, l1_weights_replicas, l2_weights_replicas,
        cumulative_local_expert_recv_stats, rank_idx, num_tokens, situ_beta, situ_linear_beta};
    const auto launch_with_phase = [&](PPUW4A16MegaMoERuntime::KernelPhase kernel_phase, const char* kernel_name) {
        auto split_args = args;
        split_args.kernel_phase = kernel_phase;
        PPUW4A16MegaMoERuntime::compile_and_launch(kernel_name, split_args);
    };
    launch_with_phase(PPUW4A16MegaMoERuntime::KernelPhase::Linear1, "w4a16_mega_moe_dispatch_gemm1");
    launch_with_phase(PPUW4A16MegaMoERuntime::KernelPhase::Linear2, "m_grouped_gemm_w4a16_moe_scatter");
}

} // namespace deep_gemm::mega_moe_fp8_w4a16
