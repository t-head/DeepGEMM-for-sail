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
                config.activation_clamp, config.swiglu_alpha, l1_weights_replicas, l1_weights_replicas_sf, tag);
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

inline void fp8_fp8_mega_moe_impl(
    const torch::Tensor& y, const std::tuple<torch::Tensor, torch::Tensor>& l1_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_replicas,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_replicas,
    const RoutedMoEWorkspace& sym_buffer, int rank_idx, int num_tokens,
    const std::string& activation, float activation_clamp, bool fast_math, float swiglu_alpha,
    const RoutedMoEConfigs& configs, Precision precision,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    auto config = configs.dispatch;
    config.swiglu_oai = activation == "swiglu_oai";
    config.swiglu_alpha = swiglu_alpha;
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

} // namespace deep_gemm::mega_moe_fp8_w4a16
