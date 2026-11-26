#pragma once

#include <type_traits>

#include "../heuristics/common_preprocess.hpp"
#include "../heuristics/mega_moe.hpp"
#include "nvfp4_decode.hpp"
#include "mega_moe_dispatch.hpp"
#include "mega_moe_combine.hpp"
#include "sm89_fp8_fp8_mega_moe.hpp"

namespace deep_gemm::mega {

using NVFP4Weights = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
using NVFP4Workspace = std::pair<torch::Tensor, torch::Tensor>;

struct FP8NVFP4Linear1Params {
    layout::SymBuffer<> sym_buffer;
    uint8_t* local_pool;
    int num_tokens;
    const int *weights, *replica_weights;
    const uint8_t *native_scales, *replica_native_scales;
    const float *outer_scales, *replica_outer_scales;
    uint8_t* output;
    float* output_scales;
    float activation_clamp, activation_alpha, situ_beta, situ_linear_beta;
    uint4* block_info;
};

struct FP8NVFP4Linear2Params {
    const uint8_t* acts;
    const float* act_scales;
    const int *weights, *replica_weights;
    const uint8_t *native_scales, *replica_native_scales;
    const float *outer_scales, *replica_outer_scales;
    int master_experts;
    int32_t* rows;
    const int32_t* offsets;
    const int64_t *routes, *peer_offsets;
    at::BFloat16* output;
    int capacity, pool_rows, rank_idx;
};

inline NVFP4Weights prepare_nvfp4_mega_moe_weights(
    const std::tuple<torch::Tensor, torch::Tensor>& weights, const torch::Tensor& outer_scales,
    bool interleave, deep_gemm_preprocess_common::PreprocessMode mode) {
    using deep_gemm_preprocess_common::interleave_gate_up_rows;
    const auto& [weight, native_scales] = weights;
    const auto w = interleave ? interleave_gate_up_rows(weight, 128) : weight;
    // Reorder native E4M3 bytes without a numerical conversion.
    const auto s = interleave
        ? interleave_gate_up_rows(native_scales.view(torch::kUInt8), 128).view(torch::kFloat8_e4m3fn)
        : native_scales;
    const auto outer = interleave ? interleave_gate_up_rows(outer_scales) : outer_scales;
    const auto prepared = deep_gemm_preprocess_common::preprocess_weights_and_scales(mode, false, w, s);
    return {prepared.weight, prepared.scale, outer};
}

template <MegaMoEPhaseKind Stage>
class FP8NVFP4MegaMoEGemmRuntime final : public LaunchRuntime<FP8NVFP4MegaMoEGemmRuntime<Stage>> {
public:
    static_assert(Stage == MegaMoEPhaseKind::Linear1 || Stage == MegaMoEPhaseKind::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;
    using LaunchInfo = std::conditional_t<kIsLinear1, Gemm1Config, ScatterConfig>;
    using Params = std::conditional_t<kIsLinear1, FP8NVFP4Linear1Params, FP8NVFP4Linear2Params>;
    struct Args {
        const char* kernel_name;
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        Params kernel_params;
        uint64_t routing_bytes = 0;
    };
    static std::string generate_impl(const Args& args) {
        const auto& c = args.launch_info;
        std::string builder;
        if constexpr (kIsLinear1) {
            const auto& d = c.dispatch;
            const layout::RoutedMoEInputLayout inputs(d.num_ranks, d.num_experts, d.nmtpr,
                d.num_topk, d.token_bytes, d.sf_bytes, args.routing_bytes);
            const auto& t = c.tile;
            const auto backend = fmt::format("deep_gemm::mega_moe::nvfp4::Gemm1<{}, {}, {}, {}, {}, {}, {}, {}, {}>",
                c.n, d.token_bytes, t.block_m, t.block_n, t.block_k, t.warp_m, t.warp_n, c.warp_k, t.num_stages);
            builder = fmt::format("MegaMoEFp8Nvfp4Gemm1Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, "
                "deep_gemm::GatedActivation({}), {}, {}, {}, {}, {}, {}, {}, {}>",
                backend, d.num_ranks, c.num_ctas,
                d.num_experts, d.num_topk, d.pull_ctas, c.master_experts, c.metadata.block_m,
                args.kernel_params.activation_clamp >= 0, int(c.activation), inputs.topk_idx, inputs.input_tokens,
                inputs.input_scales, inputs.dispatched_tokens, inputs.dispatched_scales, inputs.gemm2_metadata,
                d.nmtpr, args.routing_bytes);
        } else {
            builder = fmt::format("MegaMoEFp8Nvfp4Gemm2Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                c.n, c.k, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k,
                c.num_groups, c.num_stages, c.num_ranks);
        }
        const bool fast_math = [&]() {
            if constexpr (kIsLinear1) return c.fast_math;
            else return true;
        }();
        return fmt::format(R"(
#define DG_MOE_FAST_MATH {}
#include <deep_gemm/impls/sm89_fp8_nvfp4_mega_moe.cuh>
using Builder = deep_gemm::{};
static_assert(Builder::shared_memory_bytes() == {}, "NVFP4 MegaMoE shared-memory estimate mismatch");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_nvfp4_mega_moe_{}<Builder>);
}}
)", int(fast_math), builder, args.launch_args.smem_size, kIsLinear1 ? "l1" : "l2");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        const auto& p = args.kernel_params;
        if constexpr (kIsLinear1) {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
                p.sym_buffer, p.local_pool, p.num_tokens, p.weights, p.native_scales, p.outer_scales,
                p.replica_weights, p.replica_native_scales, p.replica_outer_scales,
                p.output, p.output_scales, p.activation_clamp, p.activation_alpha, p.situ_beta, p.situ_linear_beta, p.block_info));
        } else {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
                p.acts, p.act_scales, p.weights, p.native_scales, p.outer_scales,
                p.replica_weights, p.replica_native_scales, p.replica_outer_scales, p.master_experts,
                p.rows, p.offsets, p.routes, p.peer_offsets, p.output, p.capacity, p.pool_rows, p.rank_idx));
        }
    }
};

template <class Params>
inline void set_nvfp4_weights(Params& p,
                              const NVFP4Weights& weights, const NVFP4Weights& replicas) {
    p.weights = device_pointer<int>(std::get<0>(weights));
    p.native_scales = device_pointer<uint8_t>(std::get<1>(weights));
    p.outer_scales = device_pointer<float>(std::get<2>(weights));
    p.replica_weights = device_pointer<int>(std::get<0>(replicas));
    p.replica_native_scales = device_pointer<uint8_t>(std::get<1>(replicas));
    p.replica_outer_scales = device_pointer<float>(std::get<2>(replicas));
}

inline std::tuple<torch::Tensor, torch::Tensor> decode_nvfp4_mega_moe_weights(
    const NVFP4Weights& weights, const torch::Tensor& workspace) {
    nvfp4::detail::decode_fp4_fp8(std::get<0>(weights), std::get<1>(weights), workspace);
    return {workspace, std::get<2>(weights)};
}

inline void fp8_nvfp4_mega_moe_impl(
    const torch::Tensor& y, const NVFP4Weights& l1, const NVFP4Weights& l2,
    const NVFP4Weights& r1, const NVFP4Weights& r2,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const layout::SymBuffer<>& transport,
    int rank_idx, int num_tokens, GatedActivation activation, float activation_clamp, bool fast_math,
    float activation_alpha, float situ_beta, float situ_linear_beta, const MegaMoEPrecisionConfigs& configs,
    const std::optional<torch::Tensor>& stats,
    const std::optional<NVFP4Workspace>& workspace,
    const std::optional<NVFP4Workspace>& replica_workspace) {
    if (workspace) {
        const auto w1 = decode_nvfp4_mega_moe_weights(l1, workspace->first);
        const auto w2 = decode_nvfp4_mega_moe_weights(l2, workspace->second);
        const auto rw1 = replica_workspace ? decode_nvfp4_mega_moe_weights(r1, replica_workspace->first) : w1;
        const auto rw2 = replica_workspace ? decode_nvfp4_mega_moe_weights(r2, replica_workspace->second) : w2;
        fp8_fp8_mega_moe_impl(y, w1, w2, rw1, rw2, base, buffer, transport,
            rank_idx, num_tokens, activation, activation_clamp, fast_math, activation_alpha, situ_beta, situ_linear_beta,
            configs, Precision::FP8Blockwise, stats);
        return;
    }
    const auto& a = configs.l1;
    const auto& b = configs.l2;
    const int rows = buffer.dispatch.num_max_pool_tokens;
    // Retain the native W4A8 family tag used by the JIT scheduling policy.
    typename FP8NVFP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1>::Args g1{
        "w4a8_nvfp4_mega_moe_dispatch_gemm1", a,
        {dim3(a.num_ctas), dim3(a.num_threads), a.tile.smem_size}, {}, buffer.dispatch.routing_bytes};
    g1.launch_info.activation = activation; g1.launch_info.fast_math = fast_math;
    auto& p = g1.kernel_params;
    p.sym_buffer = transport; p.local_pool = base + buffer.dispatched_tokens;
    p.num_tokens = num_tokens;
    p.output = base + buffer.l2_acts;
    p.output_scales = math::advance_ptr<float>(base, buffer.l2_scales);
    p.block_info = math::advance_ptr<uint4>(base, buffer.block_info);
    p.activation_clamp = activation_clamp; p.activation_alpha = activation_alpha; p.situ_beta = situ_beta; p.situ_linear_beta = situ_linear_beta;
    set_nvfp4_weights(p, l1, r1);
    typename FP8NVFP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2>::Args g2{
        "w4a8_nvfp4_mega_moe_scatter", b, {dim3(b.num_sms), dim3(b.num_threads), b.smem_size}, {}};
    auto& q = g2.kernel_params;
    q.acts = base + buffer.l2_acts;
    q.act_scales = math::advance_ptr<float>(base, buffer.l2_scales);
    q.output = math::advance_ptr<at::BFloat16>(base, buffer.combine);
    q.routes = math::advance_ptr<int64_t>(base, buffer.packed_routes);
    q.peer_offsets = math::advance_ptr<int64_t>(base, buffer.peer_offsets);
    const layout::Gemm2Metadata meta(base + buffer.gemm2_metadata, b.num_groups);
    q.rows = scatter_layout(b.num_groups, meta.valid_rows, math::advance_ptr<int32_t>(base, buffer.block_info));
    q.offsets = meta.offsets;
    q.rank_idx = rank_idx; q.capacity = a.dispatch.nmtpr; q.pool_rows = rows;
    q.master_experts = std::get<0>(l2).size(0);
    set_nvfp4_weights(q, l2, r2);
    launch_mega_moe_kernel<FP8NVFP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1>, true>(
        g1, gemm_profile("w4a8_nvfp4", rows, configs.expected_m, a));
    update_recv_stats(base, buffer, a.dispatch.num_experts_per_rank, stats);
    launch_mega_moe_kernel<FP8NVFP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2>, false, 8>(
        g2, gemm_profile("w4a8_nvfp4", rows, configs.expected_m, b));
    combine_mega_moe(base, buffer, a.dispatch, transport, y, rank_idx, num_tokens);
}

} // namespace deep_gemm::mega
