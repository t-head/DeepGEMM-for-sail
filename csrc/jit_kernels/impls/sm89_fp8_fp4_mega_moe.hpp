#pragma once

#include <type_traits>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>

#include "mega_moe_dispatch.hpp"
#include "mega_moe_combine.hpp"

namespace deep_gemm::mega {

using W4A8Weights = std::tuple<torch::Tensor, torch::Tensor, float>;

struct FP8FP4Linear1Params {
    layout::SymBuffer<> sym_buffer;
    int num_tokens;
    const uint8_t *weights, *scales;
    float inv_scale;
    const uint8_t *replicas, *replica_scales;
    float replica_inv_scale;
    int master_experts;
    at::BFloat16* output;
    float activation_clamp, activation_alpha, situ_beta, situ_linear_beta;
    uint4* block_info;
};

struct FP8FP4Linear2Params {
    const uint8_t* acts;
    const float* act_scales;
    const uint8_t *weights, *scales;
    float inv_scale;
    const uint8_t *replicas, *replica_scales;
    float replica_inv_scale;
    int master_experts;
    int32_t* rows;
    const int32_t* valid_rows;
    const int64_t* routes;
    at::BFloat16* output;
    const int64_t* peer_offsets;
    int capacity, pool_rows, rank_idx;
};

template <MegaMoEPhaseKind Stage, GatedActivation Activation = GatedActivation::SwiGLU>
class FP8FP4MegaMoEGemmRuntime final : public LaunchRuntime<FP8FP4MegaMoEGemmRuntime<Stage, Activation>> {
public:
    static_assert(Stage == MegaMoEPhaseKind::Linear1 || Stage == MegaMoEPhaseKind::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;
    using Params = std::conditional_t<kIsLinear1, FP8FP4Linear1Params, FP8FP4Linear2Params>;
    using LaunchInfo = std::conditional_t<kIsLinear1, Gemm1Config, ScatterConfig>;

    struct Args {
        const char* kernel_name;
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        Params kernel_params{};
        layout::RoutedMoEStaging staging;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.launch_info;
        if constexpr (kIsLinear1) {
            const auto& dispatch = c.dispatch;
            const auto& tile = c.tile;
            TORCH_CHECK(dispatch.num_ranks > 0 && dispatch.num_experts % dispatch.num_ranks == 0,
                        "invalid MegaMoE layout");
            const layout::RoutedMoEInputLayout inputs(dispatch.num_ranks, dispatch.num_experts, dispatch.nmtpr,
                                                     dispatch.num_topk, dispatch.token_bytes, dispatch.sf_bytes, args.staging);
            return fmt::format(R"(
#define DG_MOE_FAST_MATH {}
#include <deep_gemm/impls/sm89_fp8_fp4_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEFp8Fp4Gemm1Kernel<{}, {}, {}, {}, {}, {}, deep_gemm::GatedActivation({}), deep_gemm::w4fa8::FragmentPolicy::{}, deep_gemm::w4fa8::ALoadPolicy::{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_fp4_mega_moe_l1<Builder>);
}}
)",
                int(c.fast_math), dispatch.block_m, tile.block_n, dispatch.token_bytes, tile.warp_m, tile.warp_n,
                tile.num_stages, int(Activation),
                c.fragment == w4fa8::FragmentPolicy::FullK ? "FullK" : "RollingK32",
                !w4fa8::uses_ring_a(tile.warp_m) ? "Bulk" : "Ring2", dispatch.num_experts, c.num_threads,
                dispatch.num_ranks, c.num_ctas, dispatch.num_experts_per_rank, dispatch.num_topk, c.n, dispatch.pull_ctas,
                dispatch.pull_group_size, c.metadata.block_m, c.metadata.valid_rows, inputs.topk_idx, inputs.input_tokens,
                inputs.input_scales, inputs.dispatched_tokens, inputs.dispatched_scales, inputs.gemm2_metadata,
                dispatch.nmtpr, args.staging.counts, args.staging.tokens, args.launch_args.smem_size);
        } else {
            return fmt::format(R"(
#include <deep_gemm/impls/sm89_fp8_fp4_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEFp8Fp4Gemm2Kernel<{}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::w4fa8::FragmentPolicy::{}, deep_gemm::w4fa8::ALoadPolicy::{}, {}, {}>;
using Kernel = typename Builder::Kernel;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_fp4_mega_moe_l2<Builder>);
}}
)",
                c.n, c.k, c.block_m, c.block_n, c.warp_m, c.warp_n, c.num_groups, c.num_stages,
                c.fragment == w4fa8::FragmentPolicy::FullK ? "FullK" : "RollingK32",
                !w4fa8::uses_ring_a(c.warp_m) ? "Bulk" : "Ring2", c.num_threads, c.num_ranks,
                args.launch_args.smem_size);
        }
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        const auto& p = args.kernel_params;
        if constexpr (kIsLinear1) {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
                p.sym_buffer, p.num_tokens, p.weights, p.scales, p.inv_scale, p.replicas, p.replica_scales,
                p.replica_inv_scale, p.master_experts, p.output, p.activation_clamp, p.activation_alpha,
                p.situ_beta, p.situ_linear_beta, p.block_info));
        } else {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
                p.acts, p.act_scales, p.weights, p.scales, p.inv_scale, p.replicas, p.replica_scales,
                p.replica_inv_scale, p.master_experts, p.rows, p.valid_rows, p.routes, p.output,
                p.peer_offsets, p.capacity, p.pool_rows, p.rank_idx));
        }
    }
};

template <GatedActivation Activation>
static auto make_fp8_fp4_linear1_args(
    const W4A8Weights& weights, const W4A8Weights& replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const layout::SymBuffer<>& sym_buffer,
    int num_tokens, float activation_clamp, float activation_alpha, float situ_beta, float situ_linear_beta,
    const Gemm1Config& config) {
    using Runtime = FP8FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1, Activation>;
    const auto& [w, s, inv] = weights;
    const auto& [rw, rs, rinv] = replicas;
    return typename Runtime::Args{
        .kernel_name = "w4a8_mega_moe_dispatch_gemm1", .launch_info = config,
        .launch_args = {dim3(config.num_ctas), dim3(config.num_threads), config.tile.smem_size},
        .kernel_params = {
            .sym_buffer = sym_buffer, .num_tokens = num_tokens,
            .weights = device_pointer<uint8_t>(w), .scales = device_pointer<uint8_t>(s), .inv_scale = inv,
            .replicas = device_pointer<uint8_t>(rw), .replica_scales = device_pointer<uint8_t>(rs), .replica_inv_scale = rinv,
            .master_experts = static_cast<int>(w.size(0)),
            .output = math::advance_ptr<at::BFloat16>(base, buffer.bf16_scratch),
            .activation_clamp = activation_clamp, .activation_alpha = activation_alpha,
            .situ_beta = situ_beta, .situ_linear_beta = situ_linear_beta,
            .block_info = math::advance_ptr<uint4>(base, buffer.block_info)},
        .staging = buffer.dispatch.staging};
}

static auto make_fp8_fp4_linear2_args(
    const W4A8Weights& weights, const W4A8Weights& replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, int rank_idx,
    const DispatchConfig& dispatch, const ScatterConfig& config) {
    using Runtime = FP8FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2>;
    const auto& [w, s, inv] = weights;
    const auto& [rw, rs, rinv] = replicas;
    const layout::Gemm2Metadata meta(base + buffer.gemm2_metadata, config.num_groups);
    return Runtime::Args{
        .kernel_name = "w4a8_mega_moe_scatter", .launch_info = config,
        .launch_args = {dim3(config.num_sms), dim3(config.num_threads), config.smem_size},
        .kernel_params = {
            .acts = base + buffer.l2_acts, .act_scales = math::advance_ptr<float>(base, buffer.l2_scales),
            .weights = device_pointer<uint8_t>(w), .scales = device_pointer<uint8_t>(s), .inv_scale = inv,
            .replicas = device_pointer<uint8_t>(rw), .replica_scales = device_pointer<uint8_t>(rs), .replica_inv_scale = rinv,
            .master_experts = static_cast<int>(w.size(0)),
            .rows = scatter_layout(config.num_groups, meta.rows, math::advance_ptr<int32_t>(base, buffer.block_info)),
            .valid_rows = meta.valid_rows, .routes = math::advance_ptr<int64_t>(base, buffer.packed_routes),
            .output = math::advance_ptr<at::BFloat16>(base, buffer.combine),
            .peer_offsets = math::advance_ptr<int64_t>(base, buffer.peer_offsets),
            .capacity = dispatch.nmtpr, .pool_rows = static_cast<int>(buffer.dispatch.num_max_pool_tokens),
            .rank_idx = rank_idx}};
}

inline void fp8_fp4_mega_moe_impl(
    const torch::Tensor& y, const W4A8Weights& l1_weights, const W4A8Weights& l2_weights,
    const W4A8Weights& l1_weights_replicas, const W4A8Weights& l2_weights_replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const layout::SymBuffer<>& sym_buffer,
    int rank_idx, int num_tokens, GatedActivation activation, float activation_clamp, bool fast_math,
    float activation_alpha, float situ_beta, float situ_linear_beta,
    const MegaMoEPrecisionConfigs& configs, const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    using Linear2Runtime = FP8FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2>;
    auto config = configs.l1;
    config.fast_math = fast_math;
    const int rows = buffer.dispatch.num_max_pool_tokens;
    const auto l2_args = make_fp8_fp4_linear2_args(
        l2_weights, l2_weights_replicas, base, buffer, rank_idx, config.dispatch, configs.l2);
    auto launch_linear1 = [&](auto activation_tag) {
        constexpr auto kind = decltype(activation_tag)::value;
        using Runtime = FP8FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1, kind>;
        const auto args = make_fp8_fp4_linear1_args<kind>(
            l1_weights, l1_weights_replicas, base, buffer, sym_buffer, num_tokens,
            activation_clamp, activation_alpha, situ_beta, situ_linear_beta, config);
        launch_mega_moe_kernel<Runtime, true>(
            args, gemm_profile("w4fa8_mma", rows, configs.expected_m, config));
    };
    if (activation == GatedActivation::SiTU)
        launch_linear1(std::integral_constant<GatedActivation, GatedActivation::SiTU>{});
    else if (activation == GatedActivation::SwiGLUStep)
        launch_linear1(std::integral_constant<GatedActivation, GatedActivation::SwiGLUStep>{});
    else if (activation == GatedActivation::SwiGLUOAI)
        launch_linear1(std::integral_constant<GatedActivation, GatedActivation::SwiGLUOAI>{});
    else
        launch_linear1(std::integral_constant<GatedActivation, GatedActivation::SwiGLU>{});
    fp8_channelwise_mega_moe_quantize_local_pool_kernel(config, sym_buffer, base, buffer);
    update_recv_stats(base, buffer, config.dispatch.num_experts_per_rank, cumulative_local_expert_recv_stats);
    launch_mega_moe_kernel<Linear2Runtime, false, 8>(
        l2_args, gemm_profile("w4fa8_mma", rows, configs.expected_m, configs.l2));
    combine_mega_moe(base, buffer, config.dispatch, sym_buffer, y, rank_idx, num_tokens);
}

} // namespace deep_gemm::mega
