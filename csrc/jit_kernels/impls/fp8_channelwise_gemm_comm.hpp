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

inline auto launch_channelwise_gemm_rs(const std::pair<torch::Tensor, torch::Tensor>& a,
                                       const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                                       const torch::Tensor& state, int m, int rank, int world,
                                       const Fp8GemmRSConfig& config, bool launch = true) {
    const auto [n, k] = validate_fp8_gemm_rs(a, b, out, state, m, rank, world, true);
    validate_gemm_rs_config(config, true);
    const auto [sms, bm, bn, bk, wm, wn, stages, smem] = config;
    using Native = DenseINT8GemmCutlass3Runtime;
    const auto native_args = make_channelwise_gemm_args(a, b, out, m, sms);
    Native::Args base{
        .launch_info = {bm, bn, bk, wm, wn, bk, false, stages, "Normal", "Default",
                         "fp8_channelwise_gemm_rs", false},
        .launch_args = {dim3(sms), dim3((bm / wm) * (bn / wn) * 32), std::get<0>(smem)},
        .kernel_params = Native::to_underlying_arguments_rtc(native_args, state.data_ptr()),
        .type_info = "cutlass::float_e4m3_t"};
    using Runtime = Fp8ChannelwiseGemmRSRuntime;
    Runtime::Args args{base, m, n, k, rank, world, 0};
    const int tiles = ((m + bm - 1) / bm) * ((n + bn - 1) / bn);
    auto runtime = compiler->build_cached<Runtime>(args.launch_info.kernel_name, args);
    int grid = get_gemm_rs_grid(runtime, args.launch_args, tiles);
    TORCH_CHECK(grid > 0, "GEMM/RS kernel cannot reside");
    if (grid < tiles && can_pipeline_channelwise_gemm_rs(config, m, world)) {
        const int threads = int(base.launch_args.block_dim.x);
        const int available = int((gemm_rs_smem_capacity() - base.launch_args.smem_size) / (threads * 16));
        const int shared_packets = get_gemm_rs_preferred_shared_packets(config, available);
        if (shared_packets > 0) {
            auto sender_args = args;
            sender_args.shared_packets = shared_packets;
            sender_args.launch_args.smem_size += shared_packets * threads * 16;
            auto sender = compiler->build_cached<Runtime>(sender_args.launch_info.kernel_name, sender_args);
            const int sender_grid = get_gemm_rs_grid(sender, sender_args.launch_args, tiles);
            // Keep compute residency and require successor work to overlap.
            if (sender_grid >= grid && sender_grid < tiles) {
                args = std::move(sender_args);
                runtime = std::move(sender);
                grid = sender_grid;
            }
        }
    }
    args.launch_args.grid_dim.x = grid;
    return inspect_or_launch_gemm_rs<Runtime>(runtime, args, launch, args.shared_packets, 0, sms);
}



} // namespace deep_gemm::gemm_comm
