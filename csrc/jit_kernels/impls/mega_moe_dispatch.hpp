#pragma once

#include "mega_moe_runtime.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

inline int pre_dispatch_tail_blocks(int num_tokens, int capacity, int num_topk, int block_threads) {
    return std::min(get_num_sms(), ceil_div((capacity - num_tokens) * num_topk, block_threads));
}

class FP8ChannelwiseMegaMoEPreDispatchRuntime final : public LaunchRuntime<FP8ChannelwiseMegaMoEPreDispatchRuntime> {
public:
    struct Params {
        void* hidden_states;
        void* topk_ids;
        void* topk_weights;
        void* staged_tokens;
        void* staged_scales;
        void* staged_topk_idx;
        void* staged_topk_weights;
        int num_tokens, capacity;
    };

    struct Args {
        const char* kernel_name;
        PreDispatchConfig launch_info;
        LaunchArgs launch_args;
        Params kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.launch_info;
        return fmt::format(R"DGMEGA(#include <deep_gemm/impls/fp8_channelwise_mega_moe_pre_dispatch.cuh>
using TopkIndex = {};
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mega_moe::fp8_channelwise::sm89_fp8_channelwise_mega_moe_pre_dispatch<{}, {}, TopkIndex>);
}}
)DGMEGA",
            c.int32_indices ? "int32_t" : "int64_t", c.hidden, c.num_topk);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        const auto& p = args.kernel_params;
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
            p.hidden_states, p.topk_ids, p.topk_weights, p.staged_tokens,
            p.staged_scales, p.staged_topk_idx, p.staged_topk_weights, p.num_tokens, p.capacity));
    }
};

inline void fp8_channelwise_mega_moe_pre_dispatch_impl(
    int hidden, int num_topk, const torch::Tensor& hidden_states,
    const torch::Tensor& topk_ids, const torch::Tensor& topk_weights,
    const torch::Tensor& input_token, const torch::Tensor& input_scale,
    const torch::Tensor& input_topk_idx, const torch::Tensor& input_topk_weights,
    int num_tokens, const std::string& entry = "fp8_channelwise_mega_moe_pre_dispatch") {
    PreDispatchConfig config;
    config.hidden = hidden;
    config.num_topk = num_topk;
    config.int32_indices = topk_ids.scalar_type() == torch::kInt32;
    constexpr int block_threads = 256;
    const int capacity = input_topk_idx.size(0);
    const int tail_blocks = pre_dispatch_tail_blocks(num_tokens, capacity, num_topk, block_threads);
    const FP8ChannelwiseMegaMoEPreDispatchRuntime::Args args{
        entry.c_str(), config, {dim3(num_tokens + tail_blocks), dim3(block_threads), 0},
        {device_pointer(hidden_states), device_pointer(topk_ids), device_pointer(topk_weights), device_pointer(input_token),
         device_pointer(input_scale), device_pointer(input_topk_idx), device_pointer(input_topk_weights), num_tokens, capacity}};
    launch_mega_moe_kernel<FP8ChannelwiseMegaMoEPreDispatchRuntime>(args);
}

class FP8ChannelwiseMegaMoEQuantizeRuntime final : public LaunchRuntime<FP8ChannelwiseMegaMoEQuantizeRuntime> {
public:
    struct LaunchInfo {
        int groups, n, num_threads;
        uint64_t metadata_offset;
        bool fast_math;
    };
    struct Params {
        int64_t sym_buffer_base;
        const at::BFloat16* input;
        uint8_t* output;
        float* scales;
    };
    struct Args {
        const char* kernel_name;
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        Params kernel_params;
    };
    static std::string generate_impl(const Args& args) {
        const auto& c = args.launch_info;
        return fmt::format(R"DGMEGA(#define DG_MOE_FAST_MATH {}
#include <deep_gemm/impls/fp8_channelwise_mega_moe_dispatch_gemm1.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mega_moe::fp8_channelwise::sm89_fp8_mega_moe_quantize_local_pool<{}, {} / 2, {}, {}>);
}}
)DGMEGA",
            int(c.fast_math), c.groups, c.n, c.num_threads, c.metadata_offset);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        const auto& p = args.kernel_params;
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
            p.sym_buffer_base, p.input, p.output, p.scales));
    }
};

inline void fp8_channelwise_mega_moe_quantize_local_pool_kernel(
    const Gemm1Config& config, const layout::SymBuffer<>& sym_buffer,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer) {
    // Keep the existing four-CTA/SM, four-warp row-quantization budget.
    const FP8ChannelwiseMegaMoEQuantizeRuntime::Args args{
        "fp8_channelwise_mega_moe_quantize_local_pool",
        {config.dispatch.num_experts_per_rank, config.n, 128, buffer.gemm2_metadata, config.fast_math},
        {dim3(config.tile.num_sms * 4), dim3(128), 0},
        {sym_buffer.base, math::advance_ptr<at::BFloat16>(base, buffer.bf16_scratch),
         base + buffer.l2_acts, math::advance_ptr<float>(base, buffer.l2_scales)}};
    launch_mega_moe_kernel<FP8ChannelwiseMegaMoEQuantizeRuntime>(args);
}

class MegaMoERecvStatsRuntime final : public LaunchRuntime<MegaMoERecvStatsRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        const int32_t* valid_rows;
        int32_t* cumulative;
        int groups;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_gemm/impls/mega_moe_transport.cuh>
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mega_moe::detail::mega_moe_update_recv_stats<int32_t>);
}
)";
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.valid_rows, args.cumulative, args.groups));
    }
};

inline void update_recv_stats(uint8_t* base, const layout::RoutedMoEBuffer& buffer, int groups,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    if (!cumulative_local_expert_recv_stats)
        return;
    const layout::Gemm2Metadata meta(base + buffer.gemm2_metadata, groups);
    const MegaMoERecvStatsRuntime::Args args{
        {dim3(ceil_div(groups, 256)), dim3(256), 0}, meta.valid_rows,
        cumulative_local_expert_recv_stats->data_ptr<int32_t>(), groups};
    const auto runtime = compiler->build("mega_moe_update_recv_stats", MegaMoERecvStatsRuntime::generate(args));
    MegaMoERecvStatsRuntime::launch(runtime, args);
}

} // namespace deep_gemm::mega
