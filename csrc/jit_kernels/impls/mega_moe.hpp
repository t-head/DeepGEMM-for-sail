#pragma once

#include "mega_moe_dispatch.hpp"
#include "mega_moe_scatter.hpp"
#include "mega_moe_combine.hpp"

namespace deep_gemm {

// Typed aliases into the single library-owned allocation. Constructing these
// views allocates host Tensor metadata, never device storage.
struct MegaMoEWorkspace {
    DispatchConfig dispatch_config;
    int sfb_block_n;
    layout::SymBuffer<1> sym_buffer;
    torch::Tensor l2_acts, l2_acts_sf, sfa_scratch;
    torch::Tensor m_rows, valid_m_rows, src_m_offsets, token_meta;
    std::optional<torch::Tensor> block_info;
    torch::Tensor combine, dispatch_counts, topk_weights, topk_idx;
    torch::Tensor barrier_flag, barrier_done;
};

class PPUFP4MegaMoERuntime final : public LaunchRuntime<PPUFP4MegaMoERuntime> {
public:
    enum class KernelPhase {
        Linear1,
        Linear2
    };

    struct Args {
        KernelPhase kernel_phase;
        const MegaMoEWorkspace& sym_buffer;
        const torch::Tensor& y;
        DispatchConfig l1_config;
        mega_moe_config::Config l2_config;
        const std::tuple<torch::Tensor, torch::Tensor>& l1_weights;
        const std::tuple<torch::Tensor, torch::Tensor>& l2_weights;
        const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_replicas;
        const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_replicas;
        int m_tiles_per_group;
        const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats;
        int rank_idx, num_tokens;
    };

    // Distinct argument types keep LaunchRuntime's include hash separate for
    // the two phases, which instantiate different device headers.
    static std::string generate_impl(const FP4MegaMoELinear1Args& args) {
        return generate_fp4_mega_moe_linear1(args);
    }

    static std::string generate_impl(const FP4MegaMoELinear2Args& args) {
        return generate_fp4_mega_moe_linear2(args);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config,
                            FP4MegaMoELinear1Args args) {
        auto& p = args.kernel_params;
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p.sym_buffer, p.rank_idx, p.num_tokens,
            p.l1_weights, p.l1_weights_sf, p.l1_weights_replicas, p.l1_weights_replicas_sf,
            p.num_master_experts, p.sfa_scratch, p.l2_acts, p.l2_acts_sf, p.cumulative_local_expert_recv_stats));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config,
                            FP4MegaMoELinear2Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto& buffer = args.sym_buffer;
        const auto& [l1_weights, l1_weights_sf] = args.l1_weights;
        const auto& [l2_weights, l2_weights_sf] = args.l2_weights;
        const auto& [l1_weights_replicas, l1_weights_replicas_sf] = args.l1_weights_replicas;
        const auto& [l2_weights_replicas, l2_weights_replicas_sf] = args.l2_weights_replicas;
        if (args.kernel_phase == KernelPhase::Linear1) {
            auto config = args.l1_config;
            const auto prepare = [&] {
                return make_fp4_mega_moe_linear1_args(
                    config, buffer.sym_buffer, args.rank_idx, args.num_tokens,
                    l1_weights, l1_weights_sf, l1_weights_replicas, l1_weights_replicas_sf, l1_weights.size(0),
                    buffer.sfa_scratch, buffer.l2_acts, buffer.l2_acts_sf, args.cumulative_local_expert_recv_stats);
            };
            auto phase_args = prepare();
            auto runtime = compiler->build(tag, generate(phase_args));
            phase_args.launch_args.smem_size = get_kernel_smem_size(runtime);
            const int occupancy = get_kernel_occupancy(runtime, phase_args.launch_args.block_dim.x,
                                                       phase_args.launch_args.smem_size);
            TORCH_CHECK(occupancy >= 1,
                        "MegaMoE launch cannot keep a block resident");
            const int hardware_sms = device_runtime->get_prop()->multiProcessorCount;
            // Every CTA must remain resident at the grid barriers. A measured
            // target can be reduced by register limits, including explicit tiles.
            if (config.num_ctas > hardware_sms * occupancy) {
                config.num_ctas = hardware_sms * occupancy;
                config.gemm_ctas = config.num_ctas - config.pull_ctas;
                TORCH_CHECK(config.gemm_ctas > 0, "MegaMoE needs a resident compute CTA");
                phase_args = prepare();
                runtime = compiler->build(tag, generate(phase_args));
                phase_args.launch_args.smem_size = get_kernel_smem_size(runtime);
                TORCH_CHECK(config.num_ctas <= hardware_sms *
                            get_kernel_occupancy(runtime, phase_args.launch_args.block_dim.x,
                                                 phase_args.launch_args.smem_size),
                            "MegaMoE grid exceeds kernel residency");
            }
            launch(runtime, phase_args);
            return;
        }

        TORCH_CHECK(args.kernel_phase == KernelPhase::Linear2, "Invalid MegaMoE kernel phase");
        const int num_rows = buffer.l2_acts.size(0);
        const auto& config = args.l1_config;
        const int hidden = config.token_bytes * 2, intermediate_hidden = config.gemm1_n / 2;
        if (num_rows > 0) {
            // Scheduler preprocessing consumes GEMM1 metadata and must be enqueued
            // here, after Linear1, on the same stream as GEMM2/scatter.
            auto phase_args = make_fp4_mega_moe_linear2_args(
                buffer.l2_acts, buffer.l2_acts_sf, l2_weights, l2_weights_sf, l2_weights_replicas, l2_weights_replicas_sf,
                buffer.m_rows, buffer.valid_m_rows, buffer.token_meta, buffer.combine, buffer.sym_buffer,
                num_rows, hidden, intermediate_hidden / 2, config.num_experts_per_rank, l2_weights.size(0),
                args.rank_idx, config.num_ranks, config.nmtpr, hidden, config.num_topk, false, 0,
                args.l2_config, std::nullopt, std::nullopt, 0,
                buffer.src_m_offsets, args.m_tiles_per_group, buffer.block_info);
            const auto code = generate(phase_args);
            const auto runtime = compiler->build(tag, code);
            phase_args.launch_args.smem_size = get_kernel_smem_size(runtime);
            const int occupancy = get_kernel_occupancy(runtime, phase_args.launch_args.block_dim.x,
                                                       phase_args.launch_args.smem_size);
            TORCH_CHECK(occupancy > 0, "scatter launch cannot keep a block resident");
            phase_args.launch_args.grid_dim.x *= occupancy;
            phase_args.kernel_params.hw_info = {0, static_cast<int>(phase_args.launch_args.grid_dim.x)};
            launch(runtime, phase_args);
        }
        // Linear2 owns completion on the same stream, including empty ranks.
        // Keep reduce separate so it can use its own resource budget.
        auto combine_args = make_mega_moe_combine_args(
            buffer.combine, buffer.topk_weights.narrow(0, 0, args.num_tokens), buffer.topk_idx, config.num_experts,
            args.num_tokens, hidden, config.num_topk, config.nmtpr, buffer.sym_buffer, buffer.barrier_flag,
            args.rank_idx, config.num_ranks, buffer.barrier_done, buffer.dispatch_counts,
            std::nullopt, 0, args.y);
        using CombineRuntime = MegaMoECombineRuntime;
        const auto code = CombineRuntime::generate(combine_args);
        const auto runtime = compiler->build("ppu_mega_moe_barrier_reduce", code);
        reset_done(buffer.barrier_done);
        CombineRuntime::launch(runtime, combine_args);
    }
};

inline void fp4_fp4_mega_moe_impl(
    const torch::Tensor& y, const std::tuple<torch::Tensor, torch::Tensor>& l1_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_replicas,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_replicas, const MegaMoEWorkspace& sym_buffer,
    int rank_idx, int num_tokens, const std::string& activation, float activation_clamp,
    bool fast_math, float situ_beta, float situ_linear_beta,
    const mega_moe_config::MegaMoEConfigs& configs,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    auto dispatch_config = sym_buffer.dispatch_config;
    TORCH_CHECK(sym_buffer.sfb_block_n % dispatch_config.gemm1_block_n == 0);
    dispatch_config.sfb_layout_block_n = sym_buffer.sfb_block_n;
    dispatch_config.activation_clamp = activation_clamp;
    dispatch_config.fast_math = fast_math;
    dispatch_config.fused_swiglu = activation == "silu_and_mul" || activation == "silu_and_mul_step";
    dispatch_config.silu_then_clamp = activation == "silu_and_mul_step";
    dispatch_config.fused_situ = activation == "situ_and_mul";
    dispatch_config.situ_beta = situ_beta;
    dispatch_config.situ_linear_beta = situ_linear_beta;
    const auto args = PPUFP4MegaMoERuntime::Args{
        PPUFP4MegaMoERuntime::KernelPhase::Linear1, sym_buffer, y, dispatch_config, configs.scatter,
        l1_weights, l2_weights, l1_weights_replicas, l2_weights_replicas,
        configs.m_tiles_per_group, cumulative_local_expert_recv_stats,
        rank_idx, num_tokens};
    const auto launch_with_phase = [&](PPUFP4MegaMoERuntime::KernelPhase kernel_phase, const char* kernel_name) {
        auto split_args = args;
        split_args.kernel_phase = kernel_phase;
        PPUFP4MegaMoERuntime::compile_and_launch(kernel_name, split_args);
    };
    launch_with_phase(PPUFP4MegaMoERuntime::KernelPhase::Linear1, "fp4_mega_moe_dispatch_gemm1");
    launch_with_phase(PPUFP4MegaMoERuntime::KernelPhase::Linear2, "m_grouped_gemm_fp4_moe_scatter");
}

} // namespace deep_gemm
