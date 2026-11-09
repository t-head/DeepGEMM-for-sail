#pragma once

#include <cstddef>
#include <type_traits>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>

#include "mega_moe_dispatch.hpp"
#include "mega_moe_combine.hpp"
#include "fp8_gemm.hpp"
#include "int8_gemm.hpp"
#include "fp4_gemm.hpp"

namespace deep_gemm::mega {

struct FP8ChannelwiseLinear1Params {
    layout::SymBuffer<> sym_buffer;
    int rank_idx, num_tokens;
    const uint8_t* weights;
    const float* scales;
    const uint8_t* replicas;
    const float* replica_scales;
    int master_experts;
    at::BFloat16* output;
    float activation_clamp, activation_alpha;
    uint4* block_info;
};

struct FP8BlockwiseLinear1Params {
    layout::SymBuffer<> sym_buffer;
    uint8_t* local_pool;
    int rank_idx, num_tokens;
    const uint8_t* weights;
    const float* scales;
    uint8_t* output;
    float* output_scales;
    float activation_clamp, activation_alpha;
    const uint8_t* replicas;
    const float* replica_scales;
    uint4* block_info;
};

using PackedStride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using Problem = FP4GemmRuntime::GemmProblemSize;
using Mode = FP4GemmRuntime::GemmUniversalMode;

using ScatterScheduler = TileSchedulerArguments;

struct ChannelScatter {
    void* combine_base = nullptr;
    uint32_t hidden_bytes = 0, max_tokens_per_rank = 0, num_topk = 0;
    const int64_t* packed_src_info = nullptr;
    const int32_t* valid_m_rows = nullptr;
    const int64_t* peer_offsets = nullptr;
    uint32_t rank_idx = 0, num_ranks = 0;
    const int32_t* row_group_indices = nullptr;
    uint32_t num_groups = 0;
};

struct BlockScatter {
    ChannelScatter base;
    const int32_t* src_m_offsets = nullptr;
};

template <typename Epilogue, typename Scatter> struct ScatterEpilogue {
    Epilogue base;
    Scatter scatter;
    void* base_ptr_D = nullptr;
};

struct FP8ChannelwiseLinear2Params {
    Mode mode = Mode::kGemm;
    Problem problem_shape;
    DenseINT8GemmCutlass3Runtime::MainLoopArguments mainloop;
    DenseINT8GemmCutlass3Runtime::EpilogueArgs epilogue;
    cutlass::KernelHardwareInfo hw_info;
    ScatterScheduler scheduler;
    void* workspace = nullptr;
    int32_t* signal = nullptr;
    ChannelScatter scatter;
    const void* rhs_replicas = nullptr;
    const float* rhs_replica_scales = nullptr;
    uint32_t num_master_groups = 0;
    const int32_t* src_m_offsets = nullptr;
};

struct FP8BlockwiseLinear2Params {
    Mode mode = Mode::kGemm;
    Problem problem_shape;
    FP8GemmRuntime::MainLoopArguments mainloop;
    ScatterEpilogue<FP8GemmRuntime::EpilogueArgs, BlockScatter> epilogue;
    cutlass::KernelHardwareInfo hw_info;
    struct Scheduler {
        ScatterScheduler base;
        const int32_t* pool_row_offsets = nullptr;
    } scheduler;
    void* workspace = nullptr;
    int32_t* signal = nullptr;
    const void* rhs_replicas = nullptr;
    const float* rhs_replica_scales = nullptr;
    uint32_t num_master_groups = 0;
};

inline PackedStride packed_stride(int m_rows, int cols) {
    return cutlass::make_cute_packed_stride(PackedStride{}, cute::make_shape(m_rows, cols, 1));
}

template <MegaMoEPhaseKind Stage, Precision Quantization>
class FP8MegaMoEGemmRuntime final : public LaunchRuntime<FP8MegaMoEGemmRuntime<Stage, Quantization>> {
public:
    static_assert(Stage == MegaMoEPhaseKind::Linear1 || Stage == MegaMoEPhaseKind::Linear2,
                  "unsupported MegaMoE stage");
    static_assert(Quantization == Precision::FP8Channelwise || Quantization == Precision::FP8Blockwise,
                  "unsupported FP8 MegaMoE quantization");
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;
    static constexpr bool kIsBlockwise = Quantization == Precision::FP8Blockwise;
    using Params = std::conditional_t<kIsLinear1,
        std::conditional_t<kIsBlockwise, FP8BlockwiseLinear1Params, FP8ChannelwiseLinear1Params>,
        std::conditional_t<kIsBlockwise, FP8BlockwiseLinear2Params, FP8ChannelwiseLinear2Params>>;
    using LaunchInfo = std::conditional_t<kIsLinear1, Gemm1Config, ScatterConfig>;

    struct Args {
        const char* kernel_name;
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        Params kernel_params{};
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.launch_info;
        if constexpr (kIsLinear1) {
            const auto& dispatch = c.dispatch;
            const auto& tile = c.tile;
            TORCH_CHECK(dispatch.num_ranks > 0 && dispatch.num_experts % dispatch.num_ranks == 0,
                        "invalid MegaMoE layout");
            const layout::RoutedMoEInputLayout inputs(dispatch.num_ranks, dispatch.num_experts, dispatch.nmtpr,
                                                     dispatch.num_topk, dispatch.token_bytes, dispatch.sf_bytes);
            if constexpr (kIsBlockwise) {
                return fmt::format(R"(
#define DG_MOE_FAST_MATH {}
#include <deep_gemm/impls/sm89_fp8_fp8_mega_moe_blockwise.cuh>
using Builder = deep_gemm::MegaMoEFp8BlockwiseGemm1Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_blockwise_mega_moe_l1<Builder>);
}}
)",
                    int(c.fast_math), c.n, tile.block_m, c.num_threads, inputs.input_tokens,
                    inputs.input_scales, inputs.topk_idx, inputs.dispatched_tokens, inputs.dispatched_scales,
                    inputs.gemm2_metadata, dispatch.pull_ctas, int(args.kernel_params.activation_clamp >= 0),
                    c.master_experts, c.swiglu_oai, c.metadata.block_m, dispatch.nmtpr, c.num_ctas, dispatch.num_experts,
                    dispatch.num_ranks, dispatch.num_topk, dispatch.token_bytes, args.launch_args.smem_size);
            } else {
                return fmt::format(R"(
#define DG_MOE_FAST_MATH {}
#include <deep_gemm/impls/sm89_fp8_fp8_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEFp8ChannelwiseGemm1Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_channelwise_mega_moe_l1<Builder>);
}}
)",
                    int(c.fast_math), c.num_threads, inputs.input_tokens, inputs.input_scales, inputs.topk_idx,
                    inputs.dispatched_tokens, inputs.dispatched_scales, inputs.gemm2_metadata, c.swiglu_oai,
                    dispatch.pull_group_size, c.metadata.block_m, c.metadata.valid_rows, dispatch.block_m, tile.block_k,
                    tile.block_n, c.n, tile.num_stages, tile.warp_m, tile.warp_n, dispatch.nmtpr,
                    c.num_ctas, dispatch.num_experts, dispatch.num_experts_per_rank, dispatch.num_ranks, dispatch.num_topk, dispatch.pull_ctas,
                    dispatch.token_bytes, args.launch_args.smem_size);
            }
        } else {
            if constexpr (kIsBlockwise) {
                return fmt::format(R"(
#include <deep_gemm/impls/sm89_fp8_fp8_mega_moe_blockwise.cuh>
using Builder = deep_gemm::MegaMoEFp8BlockwiseGemm2Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;
using Kernel = typename Builder::Kernel;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static_assert(sizeof(Kernel::Params) == {}, "scatter Params ABI mismatch");
static_assert(alignof(Kernel::Params) == {}, "scatter Params alignment mismatch");
static_assert(sizeof(Kernel::MainloopParams) == {}, "scatter mainloop ABI mismatch");
static_assert(sizeof(Kernel::EpilogueParams) == {}, "scatter epilogue ABI mismatch");
static_assert(offsetof(Kernel::Params,mainloop) == {}, "scatter mainloop offset mismatch");
static_assert(offsetof(Kernel::Params,epilogue) == {}, "scatter epilogue offset mismatch");
static_assert(offsetof(Kernel::Params,hw_info) == {}, "scatter hardware offset mismatch");
static_assert(offsetof(Kernel::Params,scheduler) == {}, "scatter scheduler offset mismatch");
static_assert(offsetof(Kernel::Params,workspace) == {}, "scatter workspace offset mismatch");
static_assert(offsetof(Kernel::Params,signal) == {}, "scatter signal offset mismatch");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_blockwise_mega_moe_l2<Builder>);
}}
)",
                    c.n, c.k, c.num_groups, c.num_master_groups, c.num_stages, c.block_m, c.scratch_rows,
                    c.num_ranks, c.num_threads, c.smem_size, c.smem_size, sizeof(Params), alignof(Params),
                    sizeof(Params{}.mainloop), sizeof(Params{}.epilogue), offsetof(Params, mainloop),
                    offsetof(Params, epilogue), offsetof(Params, hw_info), offsetof(Params, scheduler),
                    offsetof(Params, workspace), offsetof(Params, signal));
            } else {
                return fmt::format(R"(
#include <deep_gemm/impls/sm89_fp8_fp8_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEFp8ChannelwiseGemm2Kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;
using Kernel = typename Builder::Kernel;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static_assert(sizeof(Kernel::Params) == {}, "scatter Params ABI mismatch");
static_assert(alignof(Kernel::Params) == {}, "scatter Params alignment mismatch");
static_assert(sizeof(Kernel::MainloopParams) == {}, "scatter mainloop ABI mismatch");
static_assert(sizeof(Kernel::EpilogueParams) == {}, "scatter epilogue ABI mismatch");
static_assert(offsetof(Kernel::Params,mainloop) == {}, "scatter mainloop offset mismatch");
static_assert(offsetof(Kernel::Params,epilogue) == {}, "scatter epilogue offset mismatch");
static_assert(offsetof(Kernel::Params,hw_info) == {}, "scatter hardware offset mismatch");
static_assert(offsetof(Kernel::Params,scheduler) == {}, "scatter scheduler offset mismatch");
static_assert(offsetof(Kernel::Params,workspace) == {}, "scatter workspace offset mismatch");
static_assert(offsetof(Kernel::Params,signal) == {}, "scatter signal offset mismatch");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_fp8_mega_moe_l2<Builder>);
}}
)",
                    c.use_padded_valid_rows, c.check_packed_routes, c.n, c.k, c.block_m, c.block_n, c.block_k,
                    c.warp_m, c.warp_n, c.num_groups, c.num_stages, c.num_ranks, c.smem_size, c.smem_size,
                    sizeof(Params), alignof(Params), sizeof(Params{}.mainloop), sizeof(Params{}.epilogue),
                    offsetof(Params, mainloop), offsetof(Params, epilogue), offsetof(Params, hw_info),
                    offsetof(Params, scheduler), offsetof(Params, workspace), offsetof(Params, signal));
            }
        }
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        if constexpr (kIsLinear1) {
            const auto& p = args.kernel_params;
            if constexpr (kIsBlockwise) {
                DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
                    p.sym_buffer, p.local_pool, p.rank_idx, p.num_tokens, p.weights, p.scales, p.output,
                    p.output_scales, p.activation_clamp, p.activation_alpha, p.replicas, p.replica_scales, p.block_info));
            } else {
                DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
                    p.sym_buffer, p.rank_idx, p.num_tokens, p.weights, p.scales, p.replicas, p.replica_scales,
                    p.master_experts, p.output, p.activation_clamp, p.activation_alpha, p.block_info));
            }
        } else {
            auto params = args.kernel_params;
            params.hw_info = {0, static_cast<int>(args.launch_args.grid_dim.x)};
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, params));
        }
    }
};

template <Precision Quantization>
static auto make_fp8_fp8_linear1_args(
    const std::tuple<torch::Tensor, torch::Tensor>& weights,
    const std::tuple<torch::Tensor, torch::Tensor>& replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const layout::SymBuffer<>& sym_buffer,
    int rank_idx, int num_tokens, float activation_clamp, float activation_alpha, const Gemm1Config& config) {
    using Runtime = FP8MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1, Quantization>;
    const auto& [weight, scale] = weights;
    const auto& [replica_weight, replica_scale] = replicas;
    typename Runtime::Args args{
        .kernel_name = Runtime::kIsBlockwise ? "fp8_blockwise_mega_moe_dispatch_gemm1" : "fp8_channelwise_mega_moe_dispatch_gemm1",
        .launch_info = config,
        .launch_args = {dim3(config.num_ctas), dim3(config.num_threads), config.tile.smem_size}};
    auto& p = args.kernel_params;
    p.sym_buffer = sym_buffer;
    p.rank_idx = rank_idx;
    p.num_tokens = num_tokens;
    p.weights = device_pointer<uint8_t>(weight);
    p.scales = device_pointer<float>(scale);
    p.replicas = device_pointer<uint8_t>(replica_weight);
    p.replica_scales = device_pointer<float>(replica_scale);
    p.activation_clamp = activation_clamp;
    p.activation_alpha = activation_alpha;
    p.block_info = math::advance_ptr<uint4>(base, buffer.block_info);
    if constexpr (Runtime::kIsBlockwise) {
        p.local_pool = base + buffer.dispatched_tokens;
        p.output = base + buffer.l2_acts;
        p.output_scales = math::advance_ptr<float>(base, buffer.l2_scales);
    } else {
        p.master_experts = weight.size(0);
        p.output = math::advance_ptr<at::BFloat16>(base, buffer.bf16_scratch);
    }
    return args;
}

template <Precision Quantization>
static auto make_fp8_fp8_linear2_args(
    const std::tuple<torch::Tensor, torch::Tensor>& weights,
    const std::tuple<torch::Tensor, torch::Tensor>& replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, int rank_idx,
    const DispatchConfig& dispatch, const ScatterConfig& config) {
    using Runtime = FP8MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2, Quantization>;
    const auto& [weight, scale] = weights;
    const auto& [replica_weight, replica_scale] = replicas;
    const int m = buffer.dispatch.num_max_pool_tokens, n = config.n, k = config.k, groups = config.num_groups;
    const layout::Gemm2Metadata meta(base + buffer.gemm2_metadata, groups);
    const bool valid_rows = Runtime::kIsBlockwise || config.use_padded_valid_rows;
    auto* grouped_layout = scatter_layout(groups, valid_rows ? meta.valid_rows : meta.rows,
                                          math::advance_ptr<int32_t>(base, buffer.block_info));
    const auto sa = packed_stride(m, k), sb = packed_stride(n, k), sd = packed_stride(m, n);
    const auto problem = cute::make_shape(m, n, k, 1);
    const int threads = Runtime::kIsBlockwise ? config.num_threads
                       : (config.block_m / config.warp_m) * (config.block_n / config.warp_n) * 32;
    typename Runtime::Args args{
        .kernel_name = Runtime::kIsBlockwise ? "fp8_blockwise_mega_moe_scatter" : "m_grouped_gemm_fp8_channelwise_moe_scatter",
        .launch_info = config, .launch_args = {dim3(config.num_sms), dim3(threads), config.smem_size}};
    auto& p = args.kernel_params;
    p.problem_shape = {m, n, k, 1};
    p.rhs_replicas = device_pointer(replica_weight);
    p.rhs_replica_scales = device_pointer<float>(replica_scale);
    p.num_master_groups = config.num_master_groups;
    auto* output = math::advance_ptr<cutlass::bfloat16_t>(base, buffer.combine);
    ChannelScatter scatter{
        output, uint32_t(n * 2), uint32_t(dispatch.nmtpr), uint32_t(dispatch.num_topk),
        math::advance_ptr<int64_t>(base, buffer.packed_routes), meta.valid_rows,
        math::advance_ptr<int64_t>(base, buffer.peer_offsets), uint32_t(rank_idx), uint32_t(dispatch.num_ranks),
        Runtime::kIsBlockwise ? nullptr : meta.valid_rows, uint32_t(groups)};
    if constexpr (Runtime::kIsBlockwise) {
        p.mainloop = {math::advance_ptr<cutlass::float_e4m3_t>(base, buffer.l2_acts), sa,
                      device_pointer<cutlass::float_e4m3_t>(weight), sb, 4,
                      math::advance_ptr<float>(base, buffer.l2_scales), FP8GemmRuntime::ScaleConfig::tile_atom_to_shape_SFA(problem),
                      device_pointer<float>(scale), FP8GemmRuntime::ScaleConfig::tile_atom_to_shape_SFB(problem)};
        p.epilogue.base = {{1, 0}, nullptr, sd, output, sd};
        p.epilogue.scatter = {scatter, meta.offsets};
        p.epilogue.base_ptr_D = output;
        p.scheduler = {{uint32_t(m), grouped_layout}, meta.offsets};
    } else {
        p.mainloop = {base + buffer.l2_acts, sa, device_pointer(weight), sb,
                      math::advance_ptr<float>(base, buffer.l2_scales), device_pointer<float>(scale)};
        p.epilogue = {{1, 0}, output, sd, output, sd};
        p.scheduler = {uint32_t(m), grouped_layout};
        p.scatter = scatter;
        p.src_m_offsets = meta.offsets;
    }
    return args;
}

inline void fp8_fp8_mega_moe_impl(
    const torch::Tensor& y, const std::tuple<torch::Tensor, torch::Tensor>& l1_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_replicas,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const layout::SymBuffer<>& sym_buffer,
    int rank_idx, int num_tokens, GatedActivation activation, float activation_clamp, bool fast_math, float activation_alpha,
    const MegaMoEPrecisionConfigs& configs, Precision precision,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    auto config = configs.l1;
    config.swiglu_oai = activation == GatedActivation::SwiGLUOAI;
    config.fast_math = fast_math;
    auto launch = [&](auto quantization_tag) {
        constexpr auto kind = decltype(quantization_tag)::value;
        using Linear1Runtime = FP8MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1, kind>;
        using Linear2Runtime = FP8MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2, kind>;
        const auto l1_args = make_fp8_fp8_linear1_args<kind>(
            l1_weights, l1_weights_replicas, base, buffer, sym_buffer, rank_idx, num_tokens,
            activation_clamp, activation_alpha, config);
        const auto l2_args = make_fp8_fp8_linear2_args<kind>(
            l2_weights, l2_weights_replicas, base, buffer, rank_idx, config.dispatch, configs.l2);
        const int rows = buffer.dispatch.num_max_pool_tokens;
        launch_mega_moe_kernel<Linear1Runtime, true>(
            l1_args, gemm_profile("fp8", rows, configs.expected_m, config));
        if constexpr (kind == Precision::FP8Channelwise)
            fp8_channelwise_mega_moe_quantize_local_pool_kernel(config, sym_buffer, base, buffer);
        update_recv_stats(base, buffer, config.dispatch.num_experts_per_rank, cumulative_local_expert_recv_stats);
        if (rows > 0)
            launch_mega_moe_kernel<Linear2Runtime, false, -1>(
                l2_args, gemm_profile("fp8", rows, configs.expected_m, configs.l2));
    };
    if (precision == Precision::FP8Blockwise)
        launch(std::integral_constant<Precision, Precision::FP8Blockwise>{});
    else
        launch(std::integral_constant<Precision, Precision::FP8Channelwise>{});
    combine_mega_moe(base, buffer, config.dispatch, sym_buffer, y, rank_idx, num_tokens);
}

} // namespace deep_gemm::mega
