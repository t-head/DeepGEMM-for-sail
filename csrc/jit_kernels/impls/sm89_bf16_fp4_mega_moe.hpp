#pragma once

#include <cstddef>
#include <type_traits>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include "mega_moe_dispatch.hpp"
#include "mega_moe_combine.hpp"

namespace deep_gemm::mega {

class BF16FP4MegaMoEPreDispatchRuntime final : public LaunchRuntime<BF16FP4MegaMoEPreDispatchRuntime> {
public:
    struct Params {
        void* hidden_states;
        void* topk_ids;
        void* topk_weights;
        void* staged_tokens;
        void* staged_topk_idx;
        void* staged_topk_weights;
        int num_tokens, capacity;
        int token_blocks;
        int vector_size;
    };

    struct Args {
        const char* kernel_name;
        PreDispatchConfig launch_info;
        LaunchArgs launch_args;
        Params kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.launch_info;
        return fmt::format(R"DGMEGA(#include <deep_gemm/impls/w4a16_mega_moe_pre_dispatch.cuh>
using TopkIndex = {};
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mega_moe::w4a16::sm89_bf16_fp4_mega_moe_pre_dispatch<{}, {}, {}, TopkIndex>);
}}
)DGMEGA",
            c.int32_indices ? "int32_t" : "int64_t", c.hidden, c.num_topk, c.block_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        const auto& p = args.kernel_params;
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
            p.hidden_states, p.topk_ids, p.topk_weights, p.staged_tokens,
            p.staged_topk_idx, p.staged_topk_weights, p.num_tokens, p.capacity, p.token_blocks,
            p.vector_size));
    }
};

inline void w4a16_mega_moe_pre_dispatch_impl(
    const torch::Tensor& hidden_states, const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights, const torch::Tensor& staged_tokens,
    const torch::Tensor& staged_topk_idx, const torch::Tensor& staged_topk_weights,
    int num_tokens, int hidden, int num_topk) {
    const bool copy_tokens = hidden_states.data_ptr() != staged_tokens.data_ptr();
    auto c = w4a16_pre_dispatch_config(hidden, num_topk);
    c.int32_indices = topk_idx.scalar_type() == torch::kInt32;
    const int vector =
        (hidden % 8 == 0 &&
         ((reinterpret_cast<uintptr_t>(hidden_states.data_ptr()) | reinterpret_cast<uintptr_t>(staged_tokens.data_ptr())) &
          15) == 0)
            ? 8
            : 1;
    const int token_blocks = copy_tokens ? ceil_div(int64_t(num_tokens) * hidden / vector, int64_t(c.block_threads)) : 0;
    const int capacity = staged_topk_idx.size(0);
    const int route_blocks = std::max(ceil_div(num_tokens * num_topk, c.block_threads),
                                      pre_dispatch_tail_blocks(num_tokens, capacity, num_topk, c.block_threads));
    const BF16FP4MegaMoEPreDispatchRuntime::Args args{
        "w4a16_mega_moe_pre_dispatch", c, {dim3(token_blocks + route_blocks), dim3(c.block_threads), 0},
        {hidden_states.data_ptr(), topk_idx.data_ptr(), topk_weights.data_ptr(), staged_tokens.data_ptr(),
         staged_topk_idx.data_ptr(), staged_topk_weights.data_ptr(), num_tokens, capacity, token_blocks,
         vector}};
    launch_mega_moe_kernel<BF16FP4MegaMoEPreDispatchRuntime>(args);
}

struct BF16FP4Linear1Params {
    layout::SymBuffer<> sym_buffer;
    int rank_idx, num_tokens;
    const uint8_t *weights, *scales, *replicas, *replica_scales;
    int master_experts;
    at::BFloat16* activated;
    GatedActivationOp<GatedActivation::SiTU>::Params activation;
};

// Mirror the ordinary W4A16 extended parameter layout used by scatter.
struct BF16FP4Linear2Params {
    const at::BFloat16* ptr_a;
    const uint8_t *ptr_b, *ptr_scale, *ptr_b_replicas, *ptr_scale_replicas;
    uint32_t num_master_groups;
    at::BFloat16* ptr_d;
    int num_token, num_topk;
    const int32_t* sorted_token_ids = nullptr;
    const int64_t *packed_src_info, *sym_offsets;
    const int32_t* valid_m_rows;
    int rank_idx, num_max_tokens_per_rank;
    float situ_beta = 0, situ_linear_beta = 0;
    TileSchedulerArguments scheduler;
};

template <MegaMoEPhaseKind Stage>
class BF16FP4MegaMoEGemmRuntime final : public LaunchRuntime<BF16FP4MegaMoEGemmRuntime<Stage>> {
public:
    static_assert(Stage == MegaMoEPhaseKind::Linear1 || Stage == MegaMoEPhaseKind::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;
    using Params = std::conditional_t<kIsLinear1, BF16FP4Linear1Params, BF16FP4Linear2Params>;
    using LaunchInfo = std::conditional_t<kIsLinear1, Gemm1Config, ScatterConfig>;

    struct Args {
        const char* kernel_name;
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        Params kernel_params{};
    };

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        if constexpr (kIsLinear1) {
            const auto& dispatch = info.dispatch;
            const auto& tile = info.tile;
            TORCH_CHECK(dispatch.num_ranks > 0 && dispatch.num_experts % dispatch.num_ranks == 0,
                        "invalid MegaMoE layout");
            const layout::RoutedMoEInputLayout inputs(dispatch.num_ranks, dispatch.num_experts, dispatch.nmtpr,
                                                     dispatch.num_topk, dispatch.token_bytes, dispatch.sf_bytes);
            return fmt::format(R"(
#define DG_MOE_FAST_MATH {}
#include <deep_gemm/impls/sm89_bf16_fp4_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEBf16Fp4Gemm1Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_bf16_fp4_mega_moe_l1<Builder>);
}}
)", int(info.fast_math), info.n, dispatch.token_bytes / 2,
                tile.block_m, tile.block_n, tile.block_k, tile.warp_m, tile.warp_n, info.warp_k,
                dispatch.num_experts_per_rank, tile.num_stages, dispatch.num_ranks, info.num_ctas, info.num_threads,
                dispatch.block_m, dispatch.num_topk, dispatch.nmtpr, dispatch.pull_ctas, dispatch.pull_group_size, dispatch.pull_slots,
                inputs.dispatch.num_max_pool_tokens, inputs.input_tokens, inputs.topk_idx,
                inputs.dispatched_tokens, inputs.gemm2_metadata, args.launch_args.smem_size);
        } else {
            return fmt::format(R"(
#include <deep_gemm/impls/sm89_bf16_fp4_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEBf16Fp4Gemm2Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;
using Kernel = Builder::Kernel;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static_assert(sizeof(Kernel::Params) == {}, "MegaMoE parameter size differs from host");
static_assert(alignof(Kernel::Params) == {}, "MegaMoE parameter alignment differs from host");
static_assert(offsetof(Kernel::Params, ptr_b_replicas) == {}, "MegaMoE replica offset differs from host");
static_assert(offsetof(Kernel::Params, ptr_d) == {}, "MegaMoE output offset differs from host");
static_assert(offsetof(Kernel::Params, packed_src_info) == {}, "MegaMoE route offset differs from host");
static_assert(offsetof(Kernel::Params, sym_offsets) == {}, "MegaMoE peer offset differs from host");
static_assert(offsetof(Kernel::Params, valid_m_rows) == {}, "MegaMoE valid rows offset differs from host");
static_assert(offsetof(Kernel::Params, rank) == {}, "MegaMoE rank offset differs from host");
static_assert(offsetof(Kernel::Params, max_tokens) == {}, "MegaMoE capacity offset differs from host");
static_assert(offsetof(Kernel::Params, scheduler) == {}, "MegaMoE scheduler offset differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_bf16_fp4_mega_moe_l2<Builder>);
}}
)", info.n, info.k, info.block_m, info.block_n, info.block_k, info.warp_m, info.warp_n,
                info.num_groups, info.num_stages, info.num_ranks, args.launch_args.smem_size,
                sizeof(Params), alignof(Params), offsetof(Params, ptr_b_replicas), offsetof(Params, ptr_d),
                offsetof(Params, packed_src_info), offsetof(Params, sym_offsets), offsetof(Params, valid_m_rows),
                offsetof(Params, rank_idx), offsetof(Params, num_max_tokens_per_rank), offsetof(Params, scheduler));
        }
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        const auto& p = args.kernel_params;
        if constexpr (kIsLinear1) {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
                p.sym_buffer, p.rank_idx, p.num_tokens, p.weights, p.scales, p.replicas, p.replica_scales,
                p.master_experts, p.activated, p.activation.beta, p.activation.linear_beta));
        } else {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p));
        }
    }
};

static auto make_bf16_fp4_linear1_args(
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const layout::SymBuffer<>& sym_buffer,
    int rank_idx, int num_tokens, const GatedActivationOp<GatedActivation::SiTU>::Params& activation_params,
    const Gemm1Config& config) {
    using Runtime = BF16FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1>;
    const auto& [weight, scale] = l1_weights;
    const auto& [replica_weight, replica_scale] = replicas;
    return Runtime::Args{
        .kernel_name = "w4a16_mega_moe_dispatch_gemm1", .launch_info = config,
        .launch_args = {dim3(config.num_ctas), dim3(config.num_threads), config.tile.smem_size},
        .kernel_params = {
            .sym_buffer = sym_buffer, .rank_idx = rank_idx, .num_tokens = num_tokens,
            .weights = device_pointer<uint8_t>(weight), .scales = device_pointer<uint8_t>(scale),
            .replicas = device_pointer<uint8_t>(replica_weight), .replica_scales = device_pointer<uint8_t>(replica_scale),
            .master_experts = static_cast<int>(weight.size(0)),
            .activated = math::advance_ptr<at::BFloat16>(base, buffer.l2_acts), .activation = activation_params}};
}

static auto make_bf16_fp4_linear2_args(
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, int rank_idx,
    const DispatchConfig& dispatch, const ScatterConfig& config) {
    using Runtime = BF16FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2>;
    const auto& [weight, scale] = l2_weights;
    const auto& [replica_weight, replica_scale] = replicas;
    const int m = buffer.dispatch.num_max_pool_tokens;
    const layout::Gemm2Metadata meta(base + buffer.gemm2_metadata, dispatch.num_experts_per_rank);
    auto* grouped_layout = scatter_layout(dispatch.num_experts_per_rank, meta.rows,
                                          math::advance_ptr<int32_t>(base, buffer.block_info));
    return Runtime::Args{
        .kernel_name = "m_grouped_gemm_w4a16_moe_scatter", .launch_info = config,
        .launch_args = {dim3(config.num_sms), dim3(config.num_threads), config.smem_size},
        .kernel_params = {
            .ptr_a = math::advance_ptr<at::BFloat16>(base, buffer.l2_acts), .ptr_b = device_pointer<uint8_t>(weight),
            .ptr_scale = device_pointer<uint8_t>(scale), .ptr_b_replicas = device_pointer<uint8_t>(replica_weight),
            .ptr_scale_replicas = device_pointer<uint8_t>(replica_scale),
            .num_master_groups = static_cast<uint32_t>(weight.size(0)),
            .ptr_d = math::advance_ptr<at::BFloat16>(base, buffer.combine),
            .num_token = m, .num_topk = dispatch.num_topk,
            .packed_src_info = math::advance_ptr<int64_t>(base, buffer.packed_routes),
            .sym_offsets = math::advance_ptr<int64_t>(base, buffer.peer_offsets), .valid_m_rows = meta.valid_rows,
            .rank_idx = rank_idx, .num_max_tokens_per_rank = dispatch.nmtpr,
            .scheduler = {static_cast<uint32_t>(m), grouped_layout}}};
}

inline void bf16_fp4_mega_moe_impl(
    const torch::Tensor& y, const std::tuple<torch::Tensor, torch::Tensor>& l1_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_replicas,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const layout::SymBuffer<>& sym_buffer,
    int rank_idx, int num_tokens, bool fast_math, float situ_beta, float situ_linear_beta,
    const MegaMoEPrecisionConfigs& configs,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    using Linear1Runtime = BF16FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1>;
    using Linear2Runtime = BF16FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2>;
    auto config = configs.l1;
    config.fast_math = fast_math;
    const auto l1_args = make_bf16_fp4_linear1_args(
        l1_weights, l1_weights_replicas, base, buffer, sym_buffer, rank_idx, num_tokens,
        {situ_beta, situ_linear_beta}, config);
    const auto l2_args = make_bf16_fp4_linear2_args(
        l2_weights, l2_weights_replicas, base, buffer, rank_idx, config.dispatch, configs.l2);
    const int rows = buffer.dispatch.num_max_pool_tokens;
    launch_mega_moe_kernel<Linear1Runtime, true>(
        l1_args, gemm_profile("w4fa16_mma", rows, configs.expected_m, config));
    update_recv_stats(base, buffer, config.dispatch.num_experts_per_rank, cumulative_local_expert_recv_stats);
    launch_mega_moe_kernel<Linear2Runtime, false, 8>(
        l2_args, gemm_profile("w4fa16_mma", rows, configs.expected_m, configs.l2));
    combine_mega_moe(base, buffer, config.dispatch, sym_buffer, y, rank_idx, num_tokens);
}

} // namespace deep_gemm::mega
