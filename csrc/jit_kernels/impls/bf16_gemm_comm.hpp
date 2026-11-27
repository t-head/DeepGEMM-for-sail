#pragma once

#include "bf16_gemm.hpp"
#include "gemm_comm.hpp"
#include "../heuristics/common_bf16_gemm_comm.hpp"
#include <deep_gemm/gemm_comm/bf16_gemm_comm.cuh>

namespace deep_gemm::gemm_comm {
enum class Bf16GemmRSPhase { Fused, PeerShard, LocalShard };

template <class Native> class Bf16GemmRSRuntime final : public LaunchRuntime<Bf16GemmRSRuntime<Native>> {
  public:
    static constexpr const char* DataType = "bf16";
    static constexpr const char* QuantType = "non_quantized";
    struct Args : Native::Args {
        int m, n, k, rank, world;
        Bf16GemmRSKernelConfig plan{};
        Bf16GemmRSPhase phase = Bf16GemmRSPhase::Fused;
    };
    static auto compile_key(const Args &args) {
        if constexpr (std::is_same_v<Native, DenseBF16GemmCuteFreeRuntimeT<cutlass::bfloat16_t>>) {
            return std::make_tuple(Native::compile_key(args), args.m, args.n, args.k, args.rank, args.world, int(args.phase),
                                   args.launch_args.smem_size, is_ppu1v5_device(), args.plan.shared_packets, args.plan.compact_min_m);
        } else {
            // The old-generation native BF16 runtime has no compile_key.
            const auto &i = args.launch_info;
            const auto &shape = is_ppu1v5_device() ? args.kernel_params.no_tsm.problem_shape
                                                     : args.kernel_params.with_tsm.problem_shape;
            return std::make_tuple(i.block_m, i.block_n, i.block_k, i.warp_m, i.warp_n, i.num_groups, i.num_stages,
                                   i.gemm_type, i.kernel_strategy, i.kernel_name, i.enable_sbo_overlap,
                                   cute::get<1>(shape), cute::get<2>(shape), args.m, args.n, args.k, args.rank,
                                   args.world, args.launch_args.smem_size, is_ppu1v5_device(), args.plan.shared_packets, args.plan.compact_min_m);
        }
    }

    static std::string generate_impl(const Args &args) {
        auto source = Native::generate_impl(args);
        const std::string header = is_ppu1v5_device() ? "deep_gemm/impls/bf16_gemm_cute_free.cuh"
                                                      : "deep_gemm/impls/bf16_gemm_cutlass3.cuh";
        const std::string type = is_ppu1v5_device() ? "BF16GemmCuteFreeKernel<"
                                                    : "cutlass::gemm::kernel::DeepGemmUniversal<";
        auto pos = source.find(header);
        TORCH_CHECK(pos != std::string::npos, "native BF16 header missing");
        source.replace(pos, header.size(), "deep_gemm/gemm_comm/bf16_gemm_comm.cuh");
        pos = source.find(type);
        TORCH_CHECK(pos != std::string::npos, "native BF16 kernel missing");
        source.replace(pos, type.size(), "gemm_comm::Bf16GemmRS<");
        const std::string phase = args.phase == Bf16GemmRSPhase::PeerShard ? "#define DG_BF16_RS_PEER_SHARD\n" :
            args.phase == Bf16GemmRSPhase::LocalShard ? "#define DG_BF16_RS_LOCAL_SHARD\n" : "";
        return phase + "#define DG_BF16_RS_SHARED_PACKETS " + std::to_string(args.plan.shared_packets) + "\n" +
               gemm_rs_defines(args, 0, args.plan.compact_min_m) + source;
    }
    static void launch_impl(const KernelHandle &kernel, const LaunchConfigHandle &config, Args args) {
        if constexpr (std::is_same_v<Native, DenseBF16GemmCuteFreeRuntimeT<cutlass::bfloat16_t>>) {
            const auto& p = args.kernel_params;
            // Shapes and contiguous strides are compiled into the fused kernel.
            const Bf16GemmRSParams params{p.collective_mainloop_params.ptr_A, p.collective_mainloop_params.ptr_B,
                                          p.collective_epilogue_params.ptr_D, p.workspace};
            struct { Bf16GemmRSParams kernel_params; } launch{params};
            launch_gemm_rs(kernel, config, launch);
        } else {
            launch_gemm_rs(kernel, config, args);
        }
    }
};

// The split path resolves both grids before either kernel can run.
using Bf16GemmRSCuteFreeRuntime = Bf16GemmRSRuntime<DenseBF16GemmCuteFreeRuntimeT<cutlass::bfloat16_t>>;
struct Bf16GemmRSLaunchPlan {
    using Launch = GemmRSLaunchPlan<Bf16GemmRSCuteFreeRuntime::Args>;
    Launch producer;
    std::optional<Launch> local;
};

inline Bf16GemmRSLaunchPlan resolve_bf16_gemm_rs_launch(Bf16GemmRSCuteFreeRuntime::Args producer, int tiles) {
    using Runtime = Bf16GemmRSCuteFreeRuntime;
    const auto& info = producer.launch_info;
    const auto& plan = producer.plan;
    const int local_m = producer.m / producer.world;
    if (plan.split_owner_shards) {
        const int owner_tiles = ceil_div(local_m, info.block_m) * ceil_div(producer.n, info.block_n);
        producer.phase = Bf16GemmRSPhase::PeerShard;
        auto local = producer;
        local.phase = Bf16GemmRSPhase::LocalShard;
        local.launch_info.kernel_name = "bf16_gemm_rs_local_gemm";
        auto local_plan = resolve_gemm_rs_launch<Runtime>(std::move(local), owner_tiles);
        TORCH_CHECK(local_plan.args.launch_args.grid_dim.x > 0, "BF16 local GEMM cannot reside");
        return {resolve_gemm_rs_launch<Runtime>(std::move(producer), owner_tiles), std::move(local_plan)};
    }
    // The one-tile-per-CTA path can reduce in-kernel only when every CTA
    // fits in the compiled kernel's resident grid.
    const int input_blocks = int(gemm_rs_smem_capacity() / plan.smem_size);
    if (!plan.nonpersistent && !plan.input_overlap && producer.world >= 8 && producer.m <= info.block_m && local_m < 3 &&
            plan.shared_packets == 0 && info.warp_k == info.block_k && tiles <= int64_t(producer.launch_args.grid_dim.x) * input_blocks) {
        auto single = producer;
        single.launch_info.nonpersistent = true;
        auto single_plan = resolve_gemm_rs_launch<Runtime>(std::move(single), tiles);
        if (single_plan.args.launch_args.grid_dim.x == tiles)
            return {std::move(single_plan), std::nullopt};
    }
    return {resolve_gemm_rs_launch<Runtime>(std::move(producer), tiles), std::nullopt};
}

inline auto launch_bf16_gemm_rs(const torch::Tensor &a, const torch::Tensor &b, const torch::Tensor &out,
                                const torch::Tensor &state, int m, int rank, int world,
                                const Bf16GemmRSConfig &selected, bool launch = true) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && b.size(0) <= INT_MAX && b.size(1) <= INT_MAX,
                "BF16 GEMM/RS requires int32 matrices");
    const int n = b.size(0), k = b.size(1);
    validate_gemm_rs_shape(m, n, k, world, false);
    TORCH_CHECK(rank >= 0 && rank < world && a.sizes() == std::vector<int64_t>({m, k}),
                "invalid BF16 GEMM/RS LHS/rank");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16 &&
                    out.scalar_type() == torch::kBFloat16 && out.sizes() == std::vector<int64_t>({m / world, n}),
                "BF16 GEMM/RS requires BF16 operands and output shard");
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && b.is_contiguous() && out.is_contiguous() && state.is_contiguous() &&
                    state.scalar_type() == torch::kLong && state.numel() == GemmRSStateWords &&
                    a.device() == b.device() && a.device() == out.device() && a.device() == state.device(),
                "invalid BF16 GEMM/RS operands/state");
    validate_gemm_rs_output_alignment(out);
    auto [config, wk] = selected;
    const auto [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    validate_bf16_gemm_rs_config(config);
    TORCH_CHECK(wk >= 64 && wk % 64 == 0 && bk % wk == 0 && (is_ppu1v5_device() || wk == bk), "invalid BF16 warp K");
    TORCH_CHECK(int64_t(bm / wm) * (bn / wn) * (bk / wk) * 32 <=
                    at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                "BF16 GEMM/RS warp K exceeds block capacity");
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    const auto sa = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, k, 1));
    const auto sb = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(n, k, 1));
    const auto sd = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, n, 1));
    auto *pa = reinterpret_cast<cutlass::bfloat16_t *>(a.data_ptr<at::BFloat16>());
    auto *pb = reinterpret_cast<cutlass::bfloat16_t *>(b.data_ptr<at::BFloat16>());
    auto *pd = reinterpret_cast<cutlass::bfloat16_t *>(out.data_ptr<at::BFloat16>());
    const int tiles = ((m + bm - 1) / bm) * ((n + bn - 1) / bn);
    if (is_ppu1v5_device()) {
        using Native = DenseBF16GemmCuteFreeRuntimeT<cutlass::bfloat16_t>;
        const Native::GemmArguments ga{.mode = cutlass::gemm::GemmUniversalMode::kGemm,
                                       .problem_shape = {m, n, k, 1},
                                       .mainloopargs = {pa, sa, pb, sb},
                                       .epilogueargs = {{1, 0}, nullptr, sd, pd, sd},
                                       .hw_info = {0, sms},
                                       .scheduler = {(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr}};
        const auto plan = get_bf16_gemm_rs_kernel_config(selected, m, n, k, world);
        const Native::Args base{
            .launch_info = {bm, bn, bk, wm, wn, wk, true, stages, "Normal", "Default", "bf16_gemm_rs", false,
                            plan.input_overlap, plan.nonpersistent, plan.b_cache_policy},
            .launch_args = {dim3(sms), dim3(plan.threads), plan.smem_size},
            .kernel_params = Native::to_underlying_arguments_rtc(ga, state.data_ptr())};
        using Runtime = Bf16GemmRSCuteFreeRuntime;
        const auto launch_plan = resolve_bf16_gemm_rs_launch({base, m, n, k, rank, world, plan}, tiles);
        // Record the operation once, including when it launches two kernels.
        const auto& producer = launch_plan.producer;
        const auto resources = inspect_or_launch_gemm_rs<Runtime>(producer, launch, producer.args.plan.shared_packets, wk);
        if (launch && launch_plan.local)
            Runtime::launch(launch_plan.local->runtime, launch_plan.local->args);
        return resources;
    }
    using Native = BF16GemmCutlass3RuntimeT<cutlass::bfloat16_t>;
    const Native::GemmArguments ga{.mode = cutlass::gemm::GemmUniversalMode::kGemm,
                                   .problem_shape = {m, n, k, 1},
                                   .mainloopargs = {pa, sa, pb, sb},
                                   .epilogueargs = {{1, 0}, nullptr, sd, pd, sd},
                                   .hw_info = {0, sms},
                                   .scheduler = {(uint32_t)m, nullptr},
                                   .signal = nullptr};
    const Native::Args base{.launch_info = {bm, bn, bk, wm, wn, 1, stages, "Normal", "Default", "bf16_gemm_rs", false},
                            .launch_args = {dim3(sms), dim3((bm / wm) * (bn / wn) * 32), std::get<0>(smem)},
                            .kernel_params = Native::to_underlying_arguments_rtc(ga, state.data_ptr())};
    using Runtime = Bf16GemmRSRuntime<Native>;
    const auto plan = resolve_gemm_rs_launch<Runtime>({base, m, n, k, rank, world}, tiles);
    return inspect_or_launch_gemm_rs<Runtime>(plan, launch, 0, wk);
}


} // namespace deep_gemm::gemm_comm
