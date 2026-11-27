#pragma once

#include "int8_gemm.hpp"
#include "gemm_comm.hpp"
#include "../heuristics/common_fp8_channelwise_gemm_comm.hpp"

namespace deep_gemm::gemm_comm {

class Fp8ChannelwiseGemmRSRuntime final : public LaunchRuntime<Fp8ChannelwiseGemmRSRuntime> {
  public:
    static constexpr const char* DataType = "fp8";
    static constexpr const char* QuantType = "channel";
    using Native = DenseINT8GemmCutlass3Runtime;
    struct Args : Native::Args {
        int m, n, k, rank, world, shared_packets;
    };

    static auto compile_key(const Args& args) {
        return std::make_tuple(Native::compile_key(args), args.m, args.n, args.k, args.rank, args.world,
                               args.shared_packets, args.launch_args.smem_size);
    }

    static std::string generate_impl(const Args& args) {
        auto source = Native::generate_impl(args);
        const std::string header = "deep_gemm/impls/int8_densegemm_cutlass3.cuh";
        const std::string kernel = "cutlass::gemm::kernel::DenseGemmKernel<";
        auto offset = source.find(header);
        TORCH_CHECK(offset != std::string::npos, "native dense header missing");
        source.replace(offset, header.size(), "deep_gemm/gemm_comm/fp8_channelwise_gemm_comm.cuh");
        offset = source.find(kernel);
        TORCH_CHECK(offset != std::string::npos, "native dense kernel type missing");
        source.replace(offset, kernel.size(), "gemm_comm::Fp8ChannelwiseGemmRS<");
        // The native occupancy estimate excludes communication registers.
        const std::string bounds = "GemmOcc::kMinBlocksPerMultiprocessor";
        offset = source.find(bounds);
        TORCH_CHECK(offset != std::string::npos, "native dense launch bounds missing");
        source.replace(offset, bounds.size(), "GemmKernel::MinBlocksPerMultiprocessor");
        return gemm_rs_defines(args, args.shared_packets) + source;
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        launch_gemm_rs(kernel, config, args);
    }
};

inline auto make_channelwise_gemm_args(const std::pair<torch::Tensor, torch::Tensor>& a,
                                       const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                                       int m, int sms) {
    using Native = DenseINT8GemmCutlass3Runtime;
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    const int n = b.first.size(0), k = b.first.size(1);
    const auto sa = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, k, 1));
    const auto sb = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(n, k, 1));
    const auto sd = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, n, 1));
    Native::GemmArguments args{};
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = {m, n, k, 1};
    args.mainloopargs = {a.first.data_ptr(), sa, b.first.data_ptr(), sb, a.second.data_ptr<float>(),
                         b.second.data_ptr<float>()};
    args.epilogueargs = {{1, 0}, nullptr, sd, reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>()), sd};
    args.hw_info = {0, sms};
    args.scheduler = {(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr};
    return args;
}

inline auto resolve_channelwise_gemm_rs_plan(
        const Fp8GemmRSConfig& config, const Fp8ChannelwiseGemmRSRuntime::Native::GemmKernelParams& params,
        int m, int n, int k, int rank, int world) {
    using Runtime = Fp8ChannelwiseGemmRSRuntime;
    const auto& [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    const int warps = (bm / wm) * (bn / wn), threads = warps * 32;
    const int columns = ceil_div(n, bn), tiles = ceil_div(m, bm) * columns;
    const int64_t capacity = gemm_rs_smem_capacity();
    int shared_size = std::get<0>(smem);
    const int output_bytes = bm * bn * 2;
    if (m <= bm && is_single_owner_packet(world, m / world, bn) &&
            shared_size + output_bytes <= capacity / ceil_div(columns, sms))
        shared_size += output_bytes;

    const Runtime::Args args{
        {{bm, bn, bk, wm, wn, bk, false, stages, "Normal", "Default", "fp8_channelwise_gemm_rs", false},
         {dim3(sms), dim3(threads), shared_size},
         params,
         "cutlass::float_e4m3_t"},
        m, n, k, rank, world, 0};
    auto direct = resolve_gemm_rs_launch<Runtime>(args, tiles);
    const int direct_grid = int(direct.args.launch_args.grid_dim.x);
    TORCH_CHECK(direct_grid > 0, "GEMM/RS kernel cannot reside");

    // A resident single-wave tile has no successor work to overlap.
    if (direct_grid < tiles && can_pipeline_channelwise_gemm_rs(config, m, world)) {
        const int available = int((capacity - shared_size) / (threads * 16));
        const int packets = get_gemm_rs_preferred_shared_packets(config, available);
        if (packets > 0) {
            auto candidate = args;
            candidate.shared_packets = packets;
            candidate.launch_args.smem_size += packets * threads * 16;
            auto staged = resolve_gemm_rs_launch<Runtime>(std::move(candidate), tiles);
            const int grid = int(staged.args.launch_args.grid_dim.x);
            if (grid >= direct_grid && grid < tiles)
                return staged;
        }
    }
    return direct;
}

inline auto launch_channelwise_gemm_rs(const std::pair<torch::Tensor, torch::Tensor>& a,
                                       const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                                       const torch::Tensor& state, int m, int rank, int world,
                                       const Fp8GemmRSConfig& config, bool launch = true) {
    const auto [n, k] = validate_fp8_gemm_rs(a, b, out, state, m, rank, world, true);
    validate_gemm_rs_config(config, true);
    using Runtime = Fp8ChannelwiseGemmRSRuntime;
    const int sms = std::get<0>(config);
    const auto native_args = make_channelwise_gemm_args(a, b, out, m, sms);
    const auto params = Runtime::Native::to_underlying_arguments_rtc(native_args, state.data_ptr());
    const auto plan = resolve_channelwise_gemm_rs_plan(config, params, m, n, k, rank, world);
    return inspect_or_launch_gemm_rs<Runtime>(plan, launch, plan.args.shared_packets);
}

} // namespace deep_gemm::gemm_comm
