#pragma once

#include <torch/extension.h>
#include "mega_moe_runtime.hpp"
#include <deep_gemm/layout/sym_buffer.cuh>
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

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
        layout::SymBuffer<1> sym_buffer;
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
        const int threads = args.launch_args.block_dim.x;
        const auto body = fmt::format(
            "barrier_reduce_kernel<{}, {}, {}, {}>(b, s, w, ids, y, t, h, c, e, st, sn, shift, sym, flag, r, ws, d, counts, nc);",
            info.num_topk, info.fp8, info.scaled, threads);
        return "#include <deep_gemm/impls/mega_moe_combine.cuh>\n"
               "extern \"C\" __global__ void __launch_bounds__(" + std::to_string(threads) +
               ") ppu_mega_moe_barrier_reduce("
               "const void* b, const uint64_t* s, const float* w, const int64_t* ids, __ppu_bfloat16* y, int t, int h, int c, int e, "
               "int st, int sn, int shift, deep_gemm::layout::SymBuffer<1> sym, int32_t* flag, int r, int ws, int32_t* d, uint64_t* counts, int nc"
               ") {\nusing namespace deep_gemm::mega_moe::detail;\n" + body + "\n}\n";
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        auto& p = args.kernel_params;
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p.combine_buffer, p.combine_scales,
            p.topk_weights, p.topk_idx, p.y, p.num_tokens, p.hidden, p.num_max_tokens_per_rank, p.num_experts, p.scale_tiles,
            p.scale_block_n, p.shift, p.sym_buffer, p.barrier_flag, p.rank_idx, p.num_ranks, p.barrier_done,
            p.dispatch_counts, p.num_dispatch_counts));
    }
};

inline void reset_done(const torch::Tensor& barrier_done) {
    TORCH_CHECK(barrier_done.is_cuda() && barrier_done.scalar_type() == torch::kInt32 && barrier_done.numel() > 0);
    if (get_compile_mode() == static_cast<int>(CompileMode::COMPILE_AND_RUN) &&
        get_env<std::string>("HGGC_WARM_UP").empty())
        DG_HGGC_RUNTIME_CHECK(hggcMemsetAsync(barrier_done.data_ptr(), 0, sizeof(int32_t), current_stream()));
}

inline MegaMoECombineRuntime::Args make_mega_moe_combine_args(
    const torch::Tensor& combine_buffer, const torch::Tensor& topk_weights,
    const torch::Tensor& topk_idx, int num_experts, int num_tokens,
    int hidden, int num_topk, int num_max_tokens_per_rank, const layout::SymBuffer<1>& sym_buffer,
    const torch::Tensor& barrier_flag,
    int rank_idx, int num_ranks, const torch::Tensor& barrier_done, const torch::Tensor& dispatch_counts,
    const std::optional<torch::Tensor>& combine_scales, int scale_block_n, const torch::Tensor& y) {
    const bool fp8 = combine_buffer.scalar_type() == torch::kFloat8_e4m3fn;
    const bool scaled = combine_scales.has_value();
    const int threads = mega_moe_config::combine_num_threads(num_tokens, hidden);
    const int scale_tiles = scaled ? ceil_div(hidden, scale_block_n) : 0;
    constexpr int shift = 0;
    return {
        {num_topk, fp8, scaled}, {dim3(std::max(num_tokens, 1), 1), dim3(threads), 0},
        {combine_buffer.data_ptr(), scaled ? reinterpret_cast<const uint64_t*>(combine_scales->data_ptr()) : nullptr,
         topk_weights.data_ptr<float>(), topk_idx.data_ptr<int64_t>(), y.data_ptr<at::BFloat16>(),
         num_tokens, hidden, num_max_tokens_per_rank, num_experts, scale_tiles, scale_block_n, shift,
         sym_buffer, barrier_flag.data_ptr<int32_t>(), rank_idx, num_ranks, barrier_done.data_ptr<int32_t>(),
         static_cast<uint64_t*>(dispatch_counts.data_ptr()), static_cast<int>(dispatch_counts.nbytes() / sizeof(uint64_t))}};
}

} // namespace deep_gemm
