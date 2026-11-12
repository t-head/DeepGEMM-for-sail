#pragma once

#include <torch/extension.h>
#include "mega_moe_runtime.hpp"
#include <deep_gemm/layout/sym_buffer.cuh>
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

class MegaMoECombineRuntime final : public LaunchRuntime<MegaMoECombineRuntime> {
public:
    struct LaunchInfo {
        int num_topk;
        bool fp8, scaled;
    };

    struct KernelParams {
        const void* combine_buffer;
        const uint64_t* combine_scales;
        const float* topk_weights;
        const int64_t* topk_idx;
        at::BFloat16* y;
        int num_tokens, hidden, num_max_tokens_per_rank, num_experts;
        int scale_tiles, scale_block_n, shift;
        layout::SymBuffer<> sym_buffer;
        int32_t* barrier_flag;
        int rank_idx, num_ranks;
        int32_t* barrier_done;
        uint64_t* dispatch_counts;
        int num_dispatch_counts;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_combine.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mega_moe::detail::sm89_mega_moe_barrier_reduce<{}, {}, {}, {}>);
}}
)", info.num_topk, info.fp8, info.scaled, args.launch_args.block_dim.x);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        auto& p = args.kernel_params;
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p.combine_buffer, p.combine_scales,
            p.topk_weights, p.topk_idx, p.y, p.num_tokens, p.hidden, p.num_max_tokens_per_rank, p.num_experts, p.scale_tiles,
            p.scale_block_n, p.shift, p.sym_buffer, p.barrier_flag, p.rank_idx, p.num_ranks, p.barrier_done,
            p.dispatch_counts, p.num_dispatch_counts));
    }
};

inline void reset_done(int32_t* barrier_done) {
    if (get_compile_mode() == static_cast<int>(CompileMode::COMPILE_AND_RUN) &&
        get_env<std::string>("HGGC_WARM_UP").empty())
        DG_HGGC_RUNTIME_CHECK(hggcMemsetAsync(barrier_done, 0, sizeof(int32_t), current_stream()));
}

inline MegaMoECombineRuntime::Args make_mega_moe_combine_args(
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const DispatchConfig& config,
    const layout::SymBuffer<>& sym_buffer, const torch::Tensor& y, int rank_idx, int num_tokens) {
    const int hidden = y.size(1), threads = combine_num_threads(num_tokens, hidden);
    const int count_words = 2 * config.num_experts + config.num_experts_per_rank;
    return {
        {config.num_topk, false, false}, {dim3(std::max(num_tokens, 1), 1), dim3(threads), 0},
        {base + buffer.combine, nullptr, math::advance_ptr<float>(base, buffer.topk_weights),
         math::advance_ptr<int64_t>(base, buffer.topk_idx), y.data_ptr<at::BFloat16>(),
         num_tokens, hidden, config.nmtpr, config.num_experts, 0, 0, 0,
         sym_buffer, math::advance_ptr<int32_t>(base, buffer.barrier_flag), rank_idx, config.num_ranks,
         math::advance_ptr<int32_t>(base, buffer.barrier_done),
         math::advance_ptr<uint64_t>(base, layout::Workspace::kNumBarrierSignalBytes), count_words}};
}

inline void combine_mega_moe(
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const DispatchConfig& config,
    const layout::SymBuffer<>& sym_buffer, const torch::Tensor& y, int rank_idx, int num_tokens) {
    auto args = make_mega_moe_combine_args(base, buffer, config, sym_buffer, y, rank_idx, num_tokens);
    const auto runtime = compiler->build("ppu_mega_moe_barrier_reduce", MegaMoECombineRuntime::generate(args));
    reset_done(math::advance_ptr<int32_t>(base, buffer.barrier_done));
    MegaMoECombineRuntime::launch(runtime, args);
}

} // namespace deep_gemm::mega
