#pragma once

#include "int8_gemm.hpp"
#include "gemm_comm.hpp"

namespace deep_gemm::gemm_comm {

class Fp8ChannelwiseGemmRSRuntime final : public LaunchRuntime<Fp8ChannelwiseGemmRSRuntime> {
  public:
    using Args = Fp8GemmRSArgs<INT8GemmCutlass3Runtime>;

    static std::string generate_impl(const Args& args) {
        const auto source = INT8GemmCutlass3Runtime::generate_impl(
            {args.launch_info, args.launch_args, args.kernel_params, "cutlass::float_e4m3_t"},
            "deep_gemm/gemm_comm/fp8_channelwise_gemm_comm.cuh", "gemm_comm::Fp8ChannelwiseGemmRS",
            "GemmKernel::MinBlocksPerMultiprocessor");
        return fmt::format("#define DG_GEMM_RS_M {}\n#define DG_GEMM_RS_WORLD {}\n"
                           "#define DG_GEMM_RS_RANK {}\n"
                           "#define DG_GEMM_RS_SPECIALIZE {}\n",
                           args.m, args.world, args.rank, int(args.specialize)) +
               source;
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        launch_gemm_rs(kernel, config, args);
    }
};

inline void launch_channelwise_gemm_rs(const std::pair<torch::Tensor, torch::Tensor>& a,
                                       const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                                       const torch::Tensor& state, int m, int rank, int world,
                                       const Fp8GemmRSConfig& rs_config) {
    const auto [n, k] = validate_fp8_gemm_rs(a, b, out, state, m, rank, world, true);
    const auto& [config, specialize] = rs_config;
    auto [num_sms, bm, bn, bk, wm, wn, stages, smem] = config;
    using Runtime = INT8GemmCutlass3Runtime;
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    const auto stride_a = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, k, 1));
    const auto stride_b = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(n, k, 1));
    const auto stride_d = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, n, 1));
    auto* output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    const auto gemm_args =
        Runtime::GemmArguments{.mode = cutlass::gemm::GemmUniversalMode::kGemm,
                               .problem_shape = {m, n, k, 1},
                               .mainloopargs = {a.first.data_ptr(), stride_a, b.first.data_ptr(), stride_b,
                                                a.second.data_ptr<float>(), b.second.data_ptr<float>()},
                               .epilogueargs = {{1, 0}, nullptr, stride_d, output, stride_d},
                               .hw_info = {0, num_sms},
                               .scheduler = {(uint32_t)m, nullptr},
                               .signal = nullptr};
    const auto params = Runtime::to_underlying_arguments_rtc(gemm_args, state.data_ptr());
    const dim3 block((bm / wm) * (bn / wn) * 32);
    auto args = Fp8ChannelwiseGemmRSRuntime::Args{
        .launch_info = {bm, bn, bk, wm, wn, 1, stages, "Normal", "Default", "fp8_channelwise_gemm_rs", false},
        .launch_args = {dim3(num_sms), block, std::get<0>(smem)},
        .kernel_params = params};
    args.m = m;
    args.rank = rank;
    args.world = world;
    args.specialize = specialize;
    const auto code = Fp8ChannelwiseGemmRSRuntime::generate(args);
    const auto runtime = compiler->build("fp8_channelwise_gemm_rs", code);
    int blocks_per_cu = 0;
    const auto status =
        hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, runtime->kernel, block.x, std::get<0>(smem));
    TORCH_CHECK(status == HGGC_SUCCESS && blocks_per_cu > 0, "failed to query GEMM/RS kernel residency");
    args.launch_args.grid_dim.x *= blocks_per_cu;
    const int tiles = ((m + bm - 1) / bm) * ((n + bn - 1) / bn);
    args.launch_args.grid_dim.x = std::min<unsigned int>(args.launch_args.grid_dim.x, tiles);
    Fp8ChannelwiseGemmRSRuntime::launch(runtime, args);
}

inline void register_fp8_channelwise_gemm_comm_apis(pybind11::module_& m) {
    m.def("_gemm_comm_fp8_channelwise_gemm_rs_config", &get_channelwise_gemm_rs_config, pybind11::arg("m"),
          pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("world"), pybind11::arg("configs") = pybind11::none(),
          pybind11::arg("preset") = "decode");
    m.def("_gemm_comm_fp8_channelwise_gemm_rs", &launch_channelwise_gemm_rs);
}
} // namespace deep_gemm::gemm_comm
