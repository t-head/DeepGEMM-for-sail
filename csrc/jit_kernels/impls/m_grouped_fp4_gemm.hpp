#pragma once

#include "fp4_gemm.hpp"
#include "fp8_gemm.hpp"

using namespace deep_gemm_fp4_common;
namespace deep_gemm {

// =============================================================================
// Grouped FP4 GEMM API Implementations
// =============================================================================

static void m_grouped_gemm_fp4_fp4_bf16_nt_nopad_impl(
    const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
    const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
    const torch::Tensor& bias, const torch::Tensor& out,
    const torch::Tensor& m_indices, const torch::Tensor& m_rows,
    const int& m, const int& n, const int& k, const int& num_groups,
    std::optional<ConfigTuple> configs = std::nullopt) {
    // --- Scales layout check & preprocess ---
    torch::Tensor lhs_scales_t = lhs_scales;
    torch::Tensor rhs_scales_t = rhs_scales;
    if (!check_mxfp4_scales_layout(lhs_scales_t, /*is_sfa=*/true)) {
        lhs_scales_t = preprocess_mxfp4_scales(lhs_scales_t);
    }
    if (!check_mxfp4_scales_layout(rhs_scales_t, /*is_sfa=*/false)) {
        rhs_scales_t = post_preprocess_mxfp4_scales(rhs_scales_t);
        if (!check_mxfp4_scales_layout(rhs_scales_t, /*is_sfa=*/false)) {
            rhs_scales_t = preprocess_mxfp4_scales(rhs_scales_t);
        }
    }

    // Scale stride checks (validate post-preprocessing result)
    DG_HOST_ASSERT(lhs_scales_t.stride(0) == 1 || lhs_scales_t.size(0) == 1);
    DG_HOST_ASSERT(rhs_scales_t.stride(1) == 1 || rhs_scales_t.size(1) == 1);

    if (m == 0) return;

    int num_sms = get_num_sms();
    int expected_m = ceil_div(m, num_groups);

    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp4_common::get_smem_config_fp4(nst, bm, bn, wm, wn, bk, 1));
    } else {
        selected_config = deep_gemm_fp4_common::get_best_configs(m, expected_m, n, k, num_groups, num_sms,
                                                                  true /*is_grouped_nopad*/, false /*is_grouped_masked*/);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = num_groups;
    static constexpr GemmType kGemmType = GemmType::GroupedNoPad;

    bool hasBias = bias.numel() > 0;

    // N_EXPAND logic
    int n_expand = 1;
    if (k <= 512 && n % (block_n * 4) == 0 && !hasBias) {
        n_expand = 4;
    }

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideSFA = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    using StrideSFB = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    // A/B data strides: float4_t packed as uint8, M/N-major
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));

    // SFA is M-major (ColumnMajor), shape (m, ceil_div(k, 32))
    auto stride_SFA = cutlass::make_cute_packed_stride(StrideSFA{}, cute::make_shape(m, ceil_div(k, 32), 1));
    // SFB is N-major (transposed), shape (n, ceil_div(k, 32))
    auto stride_SFB = cutlass::make_cute_packed_stride(StrideSFB{}, cute::make_shape(n, ceil_div(k, 32), 1));

    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, 0, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    // Get data pointers
    uint8_t* ptr_A = lhs.data_ptr<uint8_t>();
    uint16_t* ptr_scale_A = lhs_scales_t.data_ptr<uint16_t>();
    uint8_t* ptr_B = rhs.data_ptr<uint8_t>();
    uint16_t* ptr_scale_B = rhs_scales_t.data_ptr<uint16_t>();
    cutlass::bfloat16_t* ptr_D = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    float* ptr_C = nullptr;

    // Compute m_rows from m_indices if not provided
    at::Tensor m_rows_tensor;
    if (!m_rows.defined() || m_rows.numel() == 0) {
        at::Tensor counts = at::bincount(m_indices);
        int64_t min_n = std::min<int64_t>(counts.size(0), num_groups);
        at::Tensor experts_for_rows =
            at::zeros({num_groups}, at::TensorOptions().dtype(at::kInt).device(m_indices.device()));
        if (min_n > 0) {
            experts_for_rows.narrow(0, 0, min_n).copy_(counts.narrow(0, 0, min_n).to(at::kInt));
        }
        m_rows_tensor = experts_for_rows;
    } else {
        m_rows_tensor = m_rows;
    }

    // Compute block_m_info
    int64_t block_m_info_size = (num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4;
    at::Tensor block_m_info =
        at::empty({block_m_info_size}, at::TensorOptions().dtype(at::kInt).device(m_rows_tensor.device()));

    // Grouped layout: start with m_rows, may be replaced by block_m_info for preprocessing
    int32_t* layout_info = reinterpret_cast<int32_t*>(m_rows_tensor.data_ptr<int32_t>());

    // ComputeBlockInfoKernel preprocessing for NoPad with large num_groups
    bool kIsNoPadPreprocessLayout = kNumGroups >= 128;
    if (kIsNoPadPreprocessLayout) {
        uint32_t block_size = std::max(32, next_power_of_two(kNumGroups));
        auto compute_block_info_args = ComputeBlockInfoKernelRuntime::Args{
            .launch_attr_args = {reinterpret_cast<const uint32_t*>(m_rows_tensor.data_ptr<int32_t>()), (uint32_t)kNumGroups,
                                 reinterpret_cast<uint32_t*>(block_m_info.data_ptr<int32_t>())},
            .launch_args = {1, block_size, 0},
        };
        const auto& code_blockinfo = ComputeBlockInfoKernelRuntime::generate(block_m);
        const auto& runtime_blockinfo = compiler->build("computeBlockInfoKernel", code_blockinfo);
        ComputeBlockInfoKernelRuntime::launch(runtime_blockinfo, compute_block_info_args);
        layout_info = reinterpret_cast<int32_t*>(block_m_info.data_ptr<int32_t>());
    }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;

    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);

    int32_t* signal_ptr = nullptr;

    FP4GemmRuntime::MainLoopArguments mainloop_params{
        cute::make_shape(m, n, k), ptr_A, stride_A, ptr_B, stride_B,
        ptr_scale_A, stride_SFA, ptr_scale_B, stride_SFB};

    FP4GemmRuntime::Args args{};
    args.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                        n, k, "GroupedNoPad", "fp4_grouped_deep_gemm_nopad", hasBias, n_expand, false};
    args.launch_args = {grid, block, SMSIZE};

    if (hasBias) {
        auto& params = args.kernel_params.bias;
        params = {};
        params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        params.problem_shape = {m, n, k, 1};
        params.collective_mainloop_params = mainloop_params;
        params.collective_epilogue_params = {
            .thread = {1.0f, 0.0f, nullptr, nullptr, {}},
            .ptr_C = ptr_C,
            .stride_C = stride_C,
            .ptr_D = ptr_D,
            .stride_D = stride_D,
            .ptr_Bias = reinterpret_cast<float const*>(bias.data_ptr<float>()),
            .stride_Bias = {},
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)expected_m, layout_info);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    } else if (n % 2 == 0) {
        auto& params = args.kernel_params.no_bias;
        params = {};
        params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        params.problem_shape = {m, n, k, 1};
        params.collective_mainloop_params = mainloop_params;
        params.collective_epilogue_params = {
            .thread = {1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
                       1.0f, 1.0f, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr},
            .ptr_C = ptr_C,
            .stride_C = stride_C,
            .ptr_D = ptr_D,
            .stride_D = stride_D,
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)expected_m, layout_info);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    } else {
        auto& params = args.kernel_params.no_bias_tsm;
        params = {};
        params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        params.problem_shape = {m, n, k, 1};
        params.collective_mainloop_params = mainloop_params;
        params.collective_epilogue_params = {
            .thread = {1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
                       1.0f, 1.0f, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr},
            .ptr_C = ptr_C,
            .stride_C = stride_C,
            .ptr_D = ptr_D,
            .stride_D = stride_D,
            .ptr_Bias = nullptr,
            .stride_Bias = {},
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)expected_m, layout_info);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    }

    const auto& code = FP4GemmRuntime::generate(args);
    const auto& runtime = compiler->build("fp4_grouped_deep_gemm_nopad", code, block.x, SMSIZE);
    const auto& kernel = runtime->kernel;

    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("fp4"), kNumGroups, m, n, k, expected_m, m_rows_tensor.data_ptr<int32_t>(),
                                  (hggcStream_t)0);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    FP4GemmRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[GroupedNoPad-FP4:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n,
               block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("vreg:%d, stack:%d\n", int(numRegs), int(localSize));
    }
}

static void m_grouped_gemm_fp4_fp4_bf16_nt_masked_impl(
    const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
    const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
    const torch::Tensor& bias, const torch::Tensor& out,
    const torch::Tensor& masked_m, const int& m, const int& n, const int& k,
    const int& num_groups, const int& expected_m,
    std::optional<ConfigTuple> configs = std::nullopt,
    int max_block_n = 256,
    bool enable_sbo_overlap = false,
    const torch::Tensor& signal = torch::Tensor()) {
    // --- Scales layout check & preprocess ---
    torch::Tensor lhs_scales_t = lhs_scales;
    torch::Tensor rhs_scales_t = rhs_scales;
    if (!check_mxfp4_scales_layout(lhs_scales_t, /*is_sfa=*/true)) {
        lhs_scales_t = preprocess_mxfp4_scales(lhs_scales_t);
    }
    if (!check_mxfp4_scales_layout(rhs_scales_t, /*is_sfa=*/false)) {
        rhs_scales_t = post_preprocess_mxfp4_scales(rhs_scales_t);
        if (!check_mxfp4_scales_layout(rhs_scales_t, /*is_sfa=*/false)) {
            rhs_scales_t = preprocess_mxfp4_scales(rhs_scales_t);
        }
    }

    // Scale stride checks (validate post-preprocessing result)
    DG_HOST_ASSERT(lhs_scales_t.stride(1) == 1 || lhs_scales_t.size(1) == 1);
    DG_HOST_ASSERT(rhs_scales_t.stride(1) == 1 || rhs_scales_t.size(1) == 1);

    int num_sms = get_num_sms();

    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp4_common::get_smem_config_fp4(nst, bm, bn, wm, wn, bk, 1));
    } else {
        selected_config = deep_gemm_fp4_common::get_best_configs(m, expected_m, n, k, num_groups, num_sms,
                                                                  false /*is_grouped_nopad*/, true /*is_grouped_masked*/,
                                                                  max_block_n);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = num_groups;
    static constexpr GemmType kGemmType = GemmType::GroupedMasked;

    bool hasBias = bias.numel() > 0;

    // N_EXPAND logic for masked
    int n_expand = 1;
    if (k <= 512 && expected_m > 2 && n % (block_n * 4) == 0 && !hasBias) {
        n_expand = 4;
    }

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideSFA = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    using StrideSFB = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    // A/B data strides
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));

    // SFA is M-major (ColumnMajor), shape (m, ceil_div(k, 32))
    auto stride_SFA = cutlass::make_cute_packed_stride(StrideSFA{}, cute::make_shape(m, ceil_div(k, 32), 1));
    // SFB is N-major (transposed), shape (n, ceil_div(k, 32))
    auto stride_SFB = cutlass::make_cute_packed_stride(StrideSFB{}, cute::make_shape(n, ceil_div(k, 32), 1));

    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, 0, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    // Get data pointers
    uint8_t* ptr_A = lhs.data_ptr<uint8_t>();
    uint16_t* ptr_scale_A = lhs_scales_t.data_ptr<uint16_t>();
    uint8_t* ptr_B = rhs.data_ptr<uint8_t>();
    uint16_t* ptr_scale_B = rhs_scales_t.data_ptr<uint16_t>();
    cutlass::bfloat16_t* ptr_D = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    float* ptr_C = nullptr;

    // Grouped layout for masked: masked_m contains per-group row counts
    int32_t* grouped_layout = reinterpret_cast<int32_t*>(masked_m.data_ptr<int32_t>());

    // Signal pointer for SBO overlap
    int32_t* signal_ptr = signal.defined() && signal.numel() > 0
                              ? signal.data_ptr<int32_t>()
                              : nullptr;

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;

    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);

    FP4GemmRuntime::MainLoopArguments mainloop_params_masked{
        cute::make_shape(m, n, k), ptr_A, stride_A, ptr_B, stride_B,
        ptr_scale_A, stride_SFA, ptr_scale_B, stride_SFB};

    FP4GemmRuntime::Args args{};
    args.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                        n, k, "GroupedMasked", "fp4_grouped_deep_gemm_masked", hasBias, n_expand, enable_sbo_overlap};
    args.launch_args = {grid, block, SMSIZE};

    if (hasBias) {
        auto& params = args.kernel_params.bias;
        params = {};
        params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        params.problem_shape = {m, n, k, 1};
        params.collective_mainloop_params = mainloop_params_masked;
        params.collective_epilogue_params = {
            .thread = {1.0f, 0.0f, nullptr, nullptr, {}},
            .ptr_C = ptr_C,
            .stride_C = stride_C,
            .ptr_D = ptr_D,
            .stride_D = stride_D,
            .ptr_Bias = reinterpret_cast<float const*>(bias.data_ptr<float>()),
            .stride_Bias = {},
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)m, grouped_layout);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    } else if (n % 2 == 0) {
        auto& params = args.kernel_params.no_bias;
        params = {};
        params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        params.problem_shape = {m, n, k, 1};
        params.collective_mainloop_params = mainloop_params_masked;
        params.collective_epilogue_params = {
            .thread = {1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
                       1.0f, 1.0f, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr},
            .ptr_C = ptr_C,
            .stride_C = stride_C,
            .ptr_D = ptr_D,
            .stride_D = stride_D,
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)m, grouped_layout);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    } else {
        auto& params = args.kernel_params.no_bias_tsm;
        params = {};
        params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        params.problem_shape = {m, n, k, 1};
        params.collective_mainloop_params = mainloop_params_masked;
        params.collective_epilogue_params = {
            .thread = {1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
                       1.0f, 1.0f, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr},
            .ptr_C = ptr_C,
            .stride_C = stride_C,
            .ptr_D = ptr_D,
            .stride_D = stride_D,
            .ptr_Bias = nullptr,
            .stride_Bias = {},
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)m, grouped_layout);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    }

    const auto& code = FP4GemmRuntime::generate(args);
    const auto& runtime = compiler->build("fp4_grouped_deep_gemm_masked", code, block.x, SMSIZE);
    const auto& kernel = runtime->kernel;

    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    hggcStream_t stream = (hggcStream_t)0;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("fp4"), kNumGroups, m, n, k, expected_m, grouped_layout,
                                  stream);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    FP4GemmRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[GroupedMasked-FP4:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d, hasBias:%d\n", block_m, block_n,
               block_k, expected_m, warp_m, warp_n, block_k, num_stages, hasBias);
        printf("vreg:%d, stack:%d\n", int(numRegs), int(localSize));
    }
}

} // namespace deep_gemm
