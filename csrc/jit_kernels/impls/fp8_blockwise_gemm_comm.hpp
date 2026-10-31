#pragma once

#include <cstring>
#include "fp8_gemm.hpp"
#include "gemm_comm.hpp"
#include <deep_gemm/gemm_comm/collectives.cuh>

namespace deep_gemm::gemm_comm {

class Fp8BlockwiseGemmRSRuntime final : public LaunchRuntime<Fp8BlockwiseGemmRSRuntime> {
  public:
    using Args = Fp8GemmRSArgs<FP8GemmRuntime>;

    static std::string generate_impl(const Args& args) {
        const auto source = FP8GemmRuntime::generate_impl(
            {args.launch_info, args.launch_args, args.kernel_params},
            "deep_gemm/gemm_comm/fp8_blockwise_gemm_comm.cuh", "gemm_comm::Fp8BlockwiseGemmRS",
            args.direct_sfb ? 64 : 128);
        return fmt::format("#define DG_GEMM_RS_DIRECT_SFB {}\n"
                           "#define DG_GEMM_RS_M {}\n"
                           "#define DG_GEMM_RS_RANK {}\n#define DG_GEMM_RS_WORLD {}\n",
                           int(args.direct_sfb), args.m, args.rank, args.world) +
               source;
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        launch_gemm_rs(kernel, config, args);
    }
};

inline void launch_blockwise_gemm_rs(const std::pair<torch::Tensor, torch::Tensor>& a,
                                     const std::pair<torch::Tensor, torch::Tensor>& b, const torch::Tensor& out,
                                     const torch::Tensor& state, int m, int rank, int world,
                                     const Fp8GemmRSConfig& rs_config) {
    const auto [n, k] = validate_fp8_gemm_rs(a, b, out, state, m, rank, world, false);
    const auto& config = rs_config.first;
    TORCH_CHECK(std::get<3>(config) == 128, "blockwise FP8 GEMM/RS requires BK128");
    const auto& lhs = a.first;
    const auto& lhs_scales = a.second;
    const auto& rhs = b.first;
    const auto& rhs_scales = b.second;
    // BN192 needs N64 loader coordinates while preserving original W128 scales.
    // This is a tile property shared by all fused-tail launches.
    const bool direct_sfb = std::get<2>(config) == 192;
    auto lhs_scales_aligned = get_mn_major_tma_aligned_tensor(lhs_scales);
    const auto& [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = config;
    auto SMSIZE = std::get<0>(smem_config);
    uint32_t kNumGroups = 1;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using ScaleGranularityShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig =
        decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(
            ScaleGranularityShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA()); // Layout type for SFA matrix operand
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    if (direct_sfb) {
        using Granularity64 = cute::Shape<cute::_1, cute::_64, cute::_128>;
        using Config64 =
            decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<Granularity64, false, true>(Granularity64{}));
        using SFB64 = decltype(Config64::deduce_layoutSFB());
        static_assert(sizeof(SFB64) == sizeof(LayoutSFB) && alignof(SFB64) == alignof(LayoutSFB),
                      "SFB64/128 erased ABI mismatch");
        static_assert(cute::is_same_v<LayoutSFA, decltype(Config64::deduce_layoutSFA())>,
                      "N-scale refinement must preserve LHS layout");
        const auto refined = Config64::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
        // Layout static constants occupy no bytes. Copy all runtime counts and
        // strides together; preserve every runtime count and stride.
        std::memcpy(&layout_SFB, &refined, sizeof(refined));
    }
    static constexpr GemmType kGemmType = GemmType::Normal;

    cutlass::float_e4m3_t* converted_input_b =
        reinterpret_cast<cutlass::float_e4m3_t*>(rhs.data_ptr<at::Float8_e4m3fn>());
    cutlass::float_e4m3_t* converted_input_a =
        reinterpret_cast<cutlass::float_e4m3_t*>(lhs.data_ptr<at::Float8_e4m3fn>());
    cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    float* scales_a_ptr = lhs_scales_aligned.data_ptr<float>();
    float* scales_b_ptr = rhs_scales.data_ptr<float>();
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);
    const auto gemm_args =
        FP8GemmRuntime::GemmArguments{.mode = cutlass::gemm::GemmUniversalMode::kGemm,
                                      .problem_shape = {m, n, k, 1},
                                      .mainloopargs = {converted_input_a, stride_A, converted_input_b, stride_B, 4,
                                                       scales_a_ptr, layout_SFA, scales_b_ptr, layout_SFB},
                                      .epilogueargs =
                                          {
                                              {1, 0},
                                              nullptr,
                                              stride_D,
                                              converted_output,
                                              stride_D,
                                          },
                                      .hw_info = hw_info,
                                      .scheduler = {(uint32_t)m, nullptr},
                                      .signal = nullptr};

    FP8GemmRuntime::GemmKernelParams params = FP8GemmRuntime::to_underlying_arguments_rtc(gemm_args, state.data_ptr());

    const std::string kernel_name = "fp8_blockwise_gemm_rs";
    auto args =
        Fp8BlockwiseGemmRSRuntime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                          num_stages, "Normal", "Default", kernel_name, false},
                                          .launch_args = {grid, block, SMSIZE},
                                          .kernel_params = params};

    args.direct_sfb = direct_sfb;
    args.m = m;
    args.rank = rank;
    args.world = world;
    const auto& code = Fp8BlockwiseGemmRSRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;
    TORCH_CHECK(result == HGGC_SUCCESS && blocks_per_cu > 0, "failed to query GEMM/RS kernel residency");
    {
        const int n_step = block_n;
        const int tasks = ((m + block_m - 1) / block_m) * ((n + n_step - 1) / n_step);
        args.launch_args.grid_dim.x = std::min<unsigned int>(args.launch_args.grid_dim.x, tasks);
    }
    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("fp8"), kNumGroups, m, n, k, 0, nullptr,
                                  current_stream());
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    Fp8BlockwiseGemmRSRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[DenseGemm_FP8:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n, block_k,
               warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }
}
inline void register_fp8_blockwise_gemm_comm_apis(pybind11::module_& m) {
    m.def("_gemm_comm_fp8_blockwise_gemm_rs_config", &get_blockwise_gemm_rs_config, pybind11::arg("m"),
          pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("world"), pybind11::arg("configs") = pybind11::none(),
          pybind11::arg("preset") = "decode");
    m.def("_gemm_comm_fp8_blockwise_gemm_rs", &launch_blockwise_gemm_rs);
}
} // namespace deep_gemm::gemm_comm
