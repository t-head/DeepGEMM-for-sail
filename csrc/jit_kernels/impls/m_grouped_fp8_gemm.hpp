#pragma once

#include <torch/python.h>
#include <cstdint>
#include <hggc_fp8.h>
#include "cute/tensor.hpp"
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../../utils/layout_type_name.hpp"
#include "cute/arch/mma.hpp"
#include "../heuristics/common_fp8.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "../../../deep_gemm/include/deep_gemm/utils_rtc.cuh"
#include "../../../deep_gemm/include/deep_gemm/profiling_interface.hpp"
#include "int8_gemm.hpp"
#include "fp8_gemm.hpp"
#include "m_grouped_int8_gemm.hpp"

using namespace deep_gemm_fp8_common;
namespace deep_gemm {

// =============================================================================
// Grouped GEMM API Implementations
// =============================================================================

using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

static void m_grouped_gemm_fp8_fp8_bf16_nt_contiguous_impl(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                                                           const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                                                           const torch::Tensor& out, const torch::Tensor& m_indices,
                                                           const int& m, const int& n, const int& k,
                                                           const int& num_groups, std::optional<ConfigTuple> configs) {
    auto lhs_scales_aligned = get_col_major_tma_aligned_tensor(lhs_scales);
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    int num_sms = get_num_sms();

    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp8_common::get_smem_config(nst, k, bm, bn, bk));
    } else {
        selected_config = deep_gemm_fp8_common::get_best_configs(m, n, k, num_groups, num_sms, true);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);
    // std::cout << "num_sms_new is " << num_sms_new << " block_m is " << block_m << " block_n is " << block_n << "
    // block_k is " << block_k << std::endl; std::cout << " warp_m is " << warp_m << " warp_n is " << warp_n << "
    // num_stages is " << num_stages << " SMSIZE is " << SMSIZE << std::endl;
    int expected_m = ceil_div(m, num_groups);

    int kNumGroups = num_groups;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using ScaleGranularityShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig =
        decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(
            ScaleGranularityShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA()); // Layout type for SFA matrix operand
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    static constexpr bool kEnableMultistageOnN = false;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    int32_t* grouped_layout = reinterpret_cast<int32_t*>(m_indices.data_ptr<int32_t>());

    static constexpr GemmType kGemmType = GemmType::GroupedContiguous;

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
    const auto gemm_args = FP8GemmRuntime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
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
        .scheduler = {(uint32_t)m, grouped_layout},
        .signal = nullptr};

    FP8GemmRuntime::GemmKernelParams params = FP8GemmRuntime::to_underlying_arguments_rtc(gemm_args, nullptr);

    auto args =
        FP8GemmRuntime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                                             "GroupedContiguous", "Default", "fp8_grouped_deep_gemm_contiguous", false},
                             .launch_args = {grid, block, SMSIZE},
                             .kernel_params = params};

    const auto& code = FP8GemmRuntime::generate(args);
    const auto& runtime = compiler->build("fp8_grouped_deep_gemm_contiguous", code, block.x, SMSIZE);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    hggcStream_t stream = (hggcStream_t)0;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("fp8"), kNumGroups, m, n, k, expected_m, grouped_layout, stream);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    FP8GemmRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);
    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[GroupedContiguous:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n,
               block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }
}

static std::pair<int, int> m_grouped_gemm_fp8_fp8_bf16_nt_masked_impl(
    const torch::Tensor& lhs, const torch::Tensor& lhs_scales, const torch::Tensor& rhs,
    const torch::Tensor& rhs_scales, const torch::Tensor& out, const torch::Tensor& masked_m, const int& m,
    const int& n, const int& k, const int& num_groups, const int& expected_m, std::optional<ConfigTuple> configs,
    int max_block_n, bool enable_sbo_overlap, const torch::Tensor& signal) {
    auto lhs_scales_aligned = get_col_major_tma_aligned_tensor(lhs_scales);
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");

    int num_sms = get_num_sms();

    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp8_common::get_smem_config(nst, k, bm, bn, bk));
    } else {
        selected_config = deep_gemm_fp8_common::get_best_configs(expected_m, n, k, num_groups, num_sms, false, true, max_block_n);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);
    // std::cout << "num_sms_new is " << num_sms_new << " block_m is " << block_m << " block_n is " << block_n << "
    // block_k is " << block_k << std::endl; std::cout << " warp_m is " << warp_m << " warp_n is " << warp_n << "
    // num_stages is " << num_stages << " SMSIZE is " << SMSIZE << std::endl;

    int kNumGroups = num_groups;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using ScaleGranularityShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig =
        decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(
            ScaleGranularityShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA()); // Layout type for SFA matrix operand
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    static constexpr bool kEnableMultistageOnN = false;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    static constexpr GemmType kGemmType = GemmType::GroupedMasked;

    int32_t* grouped_layout = reinterpret_cast<int32_t*>(masked_m.data_ptr<int32_t>());
    int32_t* signal_ptr = reinterpret_cast<int32_t*>(signal.data_ptr<int32_t>());

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
    const auto gemm_args = FP8GemmRuntime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
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
        .scheduler = {(uint32_t)m, grouped_layout},
        .signal = signal_ptr};

    FP8GemmRuntime::GemmKernelParams params = FP8GemmRuntime::to_underlying_arguments_rtc(gemm_args, nullptr);

    auto args = FP8GemmRuntime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                                                     "GroupedMasked", "Default", "fp8_grouped_deep_gemm_masked",
                                                     enable_sbo_overlap},
                                     .launch_args = {grid, block, SMSIZE},
                                     .kernel_params = params};

    const auto& code = FP8GemmRuntime::generate(args);
    const auto& runtime = compiler->build("fp8_grouped_deep_gemm_masked", code, block.x, SMSIZE);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    hggcStream_t stream = (hggcStream_t)0;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("fp8"), kNumGroups, m, n, k, expected_m, grouped_layout, stream);
    }

    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    FP8GemmRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[GroupedMasked:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n,
               block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }
    return std::make_pair(block_m, ceil_div(n, block_n));
}

static void m_grouped_gemm_fp8_fp8_bf16_nt_nopad_impl(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                                                      const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                                                      const torch::Tensor& out, const torch::Tensor& m_indices,
                                                      const int& m, const int& n, const int& k, const int& num_groups,
                                                      std::optional<const torch::Tensor> m_rows,
                                                      std::optional<ConfigTuple> configs) {
    auto lhs_scales_aligned = get_col_major_tma_aligned_tensor(lhs_scales);
    int num_sms = get_num_sms();
    int expected_m = ceil_div(m, num_groups);

    ConfigTuple cfg;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp8_common::get_smem_config(nst, k, bm, bn, bk));
    } else {
        cfg = deep_gemm_fp8_common::get_best_configs(expected_m, n, k, num_groups, num_sms);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = cfg;
    auto extra_info = get_extra_info();
    auto SMSIZE = std::get<0>(smem_config);
    // std::cout << "num_sms_new is " << num_sms_new << " block_m is " << block_m << " block_n is " << block_n << "
    // block_k is " << block_k << std::endl; std::cout << " warp_m is " << warp_m << " warp_n is " << warp_n << "
    // num_stages is " << num_stages << " SMSIZE is " << SMSIZE << std::endl;
    static constexpr GemmType kGemmType = GemmType::GroupedNoPad;
    int kNumGroups = num_groups;
    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using ScaleGranularityShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig =
        decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(
            ScaleGranularityShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    cutlass::float_e4m3_t* input_b = reinterpret_cast<cutlass::float_e4m3_t*>(rhs.data_ptr<at::Float8_e4m3fn>());
    cutlass::float_e4m3_t* input_a = reinterpret_cast<cutlass::float_e4m3_t*>(lhs.data_ptr<at::Float8_e4m3fn>());
    cutlass::bfloat16_t* output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    float* scales_a_ptr = lhs_scales_aligned.data_ptr<float>();
    float* scales_b_ptr = rhs_scales.data_ptr<float>();

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);

    at::Tensor m_rows_tensor;
    if (!m_rows.has_value() || !m_rows->defined()) {
        at::Tensor counts = at::bincount(m_indices);
        int64_t min_n = std::min<int64_t>(counts.size(0), num_groups);
        at::Tensor experts_for_rows =
            at::zeros({num_groups}, at::TensorOptions().dtype(at::kInt).device(m_indices.device()));
        if (min_n > 0) {
            experts_for_rows.narrow(0, 0, min_n).copy_(counts.narrow(0, 0, min_n).to(at::kInt));
        }
        m_rows_tensor = experts_for_rows;
    } else {
        m_rows_tensor = *m_rows;
    }

    int64_t block_m_info_size = (num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4;
    at::Tensor block_m_info =
        at::empty({block_m_info_size}, at::TensorOptions().dtype(at::kInt).device(m_rows_tensor.device()));

    int* layout_info = reinterpret_cast<int32_t*>(m_rows_tensor.data_ptr<int32_t>());
    bool kIsNoPadPreprocessLayout = kNumGroups >= 128;
    if (kIsNoPadPreprocessLayout) {
        uint32_t block_size = std::max(32, next_power_of_two(kNumGroups));
        auto compute_block_info_args = ComputeBlockInfoKernelRuntime::Args{
            .launch_attr_args = {reinterpret_cast<const uint32_t*>(m_rows_tensor.data_ptr<int32_t>()), kNumGroups,
                                 reinterpret_cast<uint32_t*>(block_m_info.data_ptr<int32_t>())},
            .launch_args = {1, block_size, 0},
        };
        const auto& code_blockinfo = ComputeBlockInfoKernelRuntime::generate(block_m);
        const auto& runtime_blockinfo = compiler->build("computeBlockInfoKernel", code_blockinfo);
        ComputeBlockInfoKernelRuntime::launch(runtime_blockinfo, compute_block_info_args);
        layout_info = reinterpret_cast<int32_t*>(block_m_info.data_ptr<int32_t>());
    }

    const auto gemm_args = FP8GemmRuntime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, 1},
        .mainloopargs = {input_a, stride_A, input_b, stride_B, 4, scales_a_ptr, layout_SFA, scales_b_ptr, layout_SFB},
        .epilogueargs =
            {
                {1, 0},
                nullptr,
                stride_D,
                output,
                stride_D,
            },
        .hw_info = hw_info,
        .scheduler = {(uint32_t)m, layout_info},
        .signal = nullptr};

    FP8GemmRuntime::GemmKernelParams params = FP8GemmRuntime::to_underlying_arguments_rtc(gemm_args, nullptr);

    auto args = FP8GemmRuntime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                                                     "GroupedNoPad", "Default", "fp8_grouped_deep_gemm_NoPad", false},
                                     .launch_args = {grid, block, SMSIZE},
                                     .kernel_params = params};

    const auto& code = FP8GemmRuntime::generate(args);
    const auto& runtime = compiler->build("fp8_grouped_deep_gemm_NoPad", code, block.x, SMSIZE);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;

    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("fp8"), kNumGroups, m, n, k, expected_m, m_rows_tensor.data_ptr<int32_t>(),
                                  (hggcStream_t)0);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    FP8GemmRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[GroupedNoPad:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n,
               block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }
}

} // namespace deep_gemm
