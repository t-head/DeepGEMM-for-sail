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
    std::optional<ConfigTuple> configs = std::nullopt,
    const torch::Tensor& out_scale = torch::Tensor(),
    double swiglu_limit = 0.0) {
    // When `out_scale` is provided, silu_and_mul + mxfp4 post-quant are fused into the epilogue.
    const bool enable_act_and_quant_fusing = out_scale.defined() && out_scale.numel() > 0;
    const bool hasBias = bias.numel() > 0;

    int num_sms = get_num_sms();
    int expected_m = ceil_div(m, num_groups);

    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp4_common::get_smem_config_fp4(nst, bm, bn, wm, wn, bk, n, hasBias, enable_act_and_quant_fusing));
    } else {
        selected_config = deep_gemm_fp4_common::get_best_configs(
            m, expected_m, n, k, num_groups, num_sms, hasBias, enable_act_and_quant_fusing, GemmType::GroupedNoPad);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = num_groups;
    static constexpr GemmType kGemmType = GemmType::GroupedNoPad;

    // N_EXPAND logic, keep in sync with m_grouped_gemm_fp4.py (nopad).
    // The fused epilogue requires n_expand == 1 (EpilogueTraits::is_valid_config).
    int n_expand = 1;
    if (k <= 512 && n % (block_n * 4) == 0 && !hasBias && !enable_act_and_quant_fusing) {
        n_expand = 4;
    }
    if (k <= 128 && n % (block_n * 8) == 0 && !hasBias && !enable_act_and_quant_fusing) {
        n_expand = 8;
    }

    // These mirror EpilogueTraits<SiluAndMulPostQuantFp4>::is_valid_config(n, block_n, n_expand,
    // hasBias) in utils_rtc.cuh.
    // `!hasBias` plus `n % 64 == 0` (which implies n is even) are also what keep the TSM fallback
    // from claiming this shape ahead of the fused epilogue in the branch chain below.
    if (enable_act_and_quant_fusing) {
        DG_HOST_ASSERT(!hasBias);
        DG_HOST_ASSERT(n_expand == 1);
        DG_HOST_ASSERT(block_n >= 64 && block_n % 64 == 0);
        DG_HOST_ASSERT(n % 64 == 0);
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
    uint16_t* ptr_scale_A = lhs_scales.data_ptr<uint16_t>();
    uint8_t* ptr_B = rhs.data_ptr<uint8_t>();
    uint16_t* ptr_scale_B = rhs_scales.data_ptr<uint16_t>();
    // In the fused epilogue `out` is uint8 (packed mxfp4) rather than bfloat16.
    cutlass::bfloat16_t* ptr_D = enable_act_and_quant_fusing
                                     ? nullptr
                                     : reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    uint8_t* ptr_D_fused = enable_act_and_quant_fusing ? out.data_ptr<uint8_t>() : nullptr;
    uint16_t* ptr_SFD = enable_act_and_quant_fusing ? out_scale.data_ptr<uint16_t>() : nullptr;
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
    // NOTE: mirrors DeepGemmScheduler::kIsNoPadPreprocessLayout in scheduler_cutlass3.cuh,
    // which is `(kGemmType == GroupedNoPad || kGemmType == GroupedFused) && kNumGroups >= 128`.
    // kGemmType is GroupedNoPad here, so only the group-count term is evaluated.
    // Keep both sides in sync when that definition changes.
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
                        n, k, "GroupedNoPad", "fp4_grouped_deep_gemm_nopad", hasBias, n_expand, false,
                        enable_act_and_quant_fusing ? "SiluAndMulPostQuantFp4" : "Default",
                        enable_act_and_quant_fusing && swiglu_limit > 0.0};
    args.launch_args = {grid, block, SMSIZE};

    // Branch exactly like the device-side CollectiveEpilogue conditional in generate_impl:
    //   hasBias || SHAPE_N % 2 != 0 ? WithTsm : (fused ? SiluAndMulPostQuant : NoTsm)
    // The TSM fallback is checked first so host and device can never disagree on the layout.
    if (hasBias || n % 2 != 0) {
        auto& params = args.kernel_params.with_tsm;
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
            .ptr_Bias = hasBias ? reinterpret_cast<float const*>(bias.data_ptr<float>()) : nullptr,
            .stride_Bias = {},
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)m, layout_info);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    } else if (enable_act_and_quant_fusing) {
        auto& params = args.kernel_params.silu_and_mul_post_quant;
        params = {};
        params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        params.problem_shape = {m, n, k, 1};
        params.collective_mainloop_params = mainloop_params;
        params.collective_epilogue_params = {
            .thread = {1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
                       1.0f, 1.0f, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr},
            .ptr_C = ptr_C,
            .stride_C = stride_C,
            .ptr_D = ptr_D_fused,
            .stride_D = stride_D,
            .ptr_SFD = ptr_SFD,
            .shape_m = (uint32_t)m,
            .swiglu_limit = (float)swiglu_limit,
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)m, layout_info);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    } else {
        auto& params = args.kernel_params.no_tsm;
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
        params.scheduler = TileSchedulerArguments((uint32_t)m, layout_info);
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

// Returns (block_m, ceil_div(n, block_n)), matching the Python interface: the SBO-overlap
// signal protocol check consumes both values.
static std::pair<int, int> m_grouped_gemm_fp4_fp4_bf16_nt_masked_impl(
    const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
    const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
    const torch::Tensor& bias, const torch::Tensor& out,
    const torch::Tensor& masked_m, const int& m, const int& n, const int& k,
    const int& num_groups, const int& expected_m,
    std::optional<ConfigTuple> configs = std::nullopt,
    int max_block_n = 256,
    bool enable_sbo_overlap = false,
    const torch::Tensor& signal = torch::Tensor(),
    const torch::Tensor& out_scale = torch::Tensor(),
    double swiglu_limit = 0.0) {
    // When `out_scale` is provided, silu_and_mul + mxfp4 post-quant are fused into the epilogue.
    const bool enable_act_and_quant_fusing = out_scale.defined() && out_scale.numel() > 0;
    const bool hasBias = bias.numel() > 0;

    int num_sms = get_num_sms();

    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp4_common::get_smem_config_fp4(nst, bm, bn, wm, wn, bk, n, hasBias, enable_act_and_quant_fusing));
    } else {
        selected_config = deep_gemm_fp4_common::get_best_configs(
            m, expected_m, n, k, num_groups, num_sms, hasBias, enable_act_and_quant_fusing, GemmType::GroupedMasked, max_block_n);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = num_groups;
    static constexpr GemmType kGemmType = GemmType::GroupedMasked;

    // N_EXPAND logic for masked, keep in sync with m_grouped_gemm_fp4.py (masked).
    // The fused epilogue requires n_expand == 1 (EpilogueTraits::is_valid_config).
    int n_expand = 1;
    if (k <= 512 && expected_m > 2 && n % (block_n * 4) == 0 && !hasBias && !enable_act_and_quant_fusing) {
        n_expand = 4;
    }

    // These mirror EpilogueTraits<SiluAndMulPostQuantFp4>::is_valid_config(n, block_n, n_expand,
    // hasBias) in utils_rtc.cuh.
    // `!hasBias` plus `n % 64 == 0` (which implies n is even) are also what keep the TSM fallback
    // from claiming this shape ahead of the fused epilogue in the branch chain below.
    if (enable_act_and_quant_fusing) {
        DG_HOST_ASSERT(!hasBias);
        DG_HOST_ASSERT(n_expand == 1);
        DG_HOST_ASSERT(block_n >= 64 && block_n % 64 == 0);
        DG_HOST_ASSERT(n % 64 == 0);
    }

    // MoE dynamic tile, mirroring Fp4Gemm::run's `if constexpr (kEnableMoeDynamicTile)` branch.
    // The kernel picks its own tile shape, so Python fixes the block config here purely to avoid
    // spawning extra JIT variants; we do the same so the smem estimate below matches.
    const auto [enable_moe_dynamic_tile, dynamic_tile_id] =
        deep_gemm_fp4_common::select_moe_dynamic_tile(n, k, expected_m, hasBias);
    deep_gemm_fp4_common::DynamicTileLaunchConst dyn_launch{};
    if (enable_moe_dynamic_tile) {
        // select_moe_dynamic_tile() already refuses a bias.
        DG_HOST_ASSERT(!hasBias);
        // fix the block config to avoid unnecessary jit compile
        block_m = 128; block_n = 128; block_k = 64; warp_m = 64; warp_n = 64; num_stages = 3;
        // Both the SharedStorageSize and the block thread count are decided inside the kernel by
        // kDynamicTileId, not by these block values, so neither get_smem_config_fp4() nor the usual
        // (block_m/warp_m)*(block_n/warp_n)*32 formula applies. Use the hgcc-probed table.
        dyn_launch = deep_gemm_fp4_common::dynamic_tile_launch_const(dynamic_tile_id);
        SMSIZE = dyn_launch.shared_storage_size;
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
    uint16_t* ptr_scale_A = lhs_scales.data_ptr<uint16_t>();
    uint8_t* ptr_B = rhs.data_ptr<uint8_t>();
    uint16_t* ptr_scale_B = rhs_scales.data_ptr<uint16_t>();
    // In the fused epilogue `out` is uint8 (packed mxfp4) rather than bfloat16.
    cutlass::bfloat16_t* ptr_D = enable_act_and_quant_fusing
                                     ? nullptr
                                     : reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    uint8_t* ptr_D_fused = enable_act_and_quant_fusing ? out.data_ptr<uint8_t>() : nullptr;
    uint16_t* ptr_SFD = enable_act_and_quant_fusing ? out_scale.data_ptr<uint16_t>() : nullptr;
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

    // ----- DynamicTile early-out: completely separate runtime & kernel -----
    if (enable_moe_dynamic_tile) {
        FP4DynamicTileRuntime::Args dyn_args{};
        dyn_args.launch_info = {n, k, kNumGroups, dynamic_tile_id, "GroupedMasked",
                                "fp4_grouped_deep_gemm_masked_dynamic_tile",
                                enable_act_and_quant_fusing ? "SiluAndMulPostQuantFp4" : "Default",
                                enable_act_and_quant_fusing && swiglu_limit > 0.0};
        // The dynamic-tile kernel's block shape is get_block_shape() == MaxThreadsPerBlock, which
        // varies per kDynamicTileId and is unrelated to the fixed 128x128 block config above.
        dim3 const dyn_block = dyn_launch.block_threads;
        dyn_args.launch_args = {grid, dyn_block, SMSIZE};

        if (enable_act_and_quant_fusing) {
            auto& params = dyn_args.kernel_params.dynamic_tile_silu_and_mul_post_quant;
            params = {};
            params.ptr_A = ptr_A;
            params.stride_A = stride_A;
            params.ptr_B = ptr_B;
            params.stride_B = stride_B;
            params.ptr_scale_A = ptr_scale_A;
            params.stride_SFA = stride_SFA;
            params.ptr_scale_B = ptr_scale_B;
            params.stride_SFB = stride_SFB;
            params.epi_params = {
                .thread = {1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
                        1.0f, 1.0f, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr},
                .ptr_C = ptr_C,
                .stride_C = stride_C,
                .ptr_D = ptr_D_fused,
                .stride_D = stride_D,
                .ptr_SFD = ptr_SFD,
                .shape_m = (uint32_t)m,
                .swiglu_limit = (float)swiglu_limit,
            };
            params.shape_m = (uint32_t)m;
            params.grouped_layout = grouped_layout;
        } else {
            auto& params = dyn_args.kernel_params.dynamic_tile_no_tsm;
            params = {};
            params.ptr_A = ptr_A;
            params.stride_A = stride_A;
            params.ptr_B = ptr_B;
            params.stride_B = stride_B;
            params.ptr_scale_A = ptr_scale_A;
            params.stride_SFA = stride_SFA;
            params.ptr_scale_B = ptr_scale_B;
            params.stride_SFB = stride_SFB;
            params.epi_params = {
                .thread = {1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
                        1.0f, 1.0f, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr},
                .ptr_C = ptr_C,
                .stride_C = stride_C,
                .ptr_D = ptr_D,
                .stride_D = stride_D,
            };
            params.shape_m = (uint32_t)m;
            params.grouped_layout = grouped_layout;
        }

        const auto& code = FP4DynamicTileRuntime::generate(dyn_args);
        const auto& runtime = compiler->build("fp4_grouped_deep_gemm_masked_dynamic_tile", code, dyn_block.x, SMSIZE);
        const auto& kernel = runtime->kernel;

        int blocks_per_cu = 0;
        HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, dyn_block.x, SMSIZE);
        dyn_args.launch_args.grid_dim.x *= blocks_per_cu;

        hggcStream_t stream = (hggcStream_t)0;
        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(kGemmType, false, std::string("fp4"), kNumGroups, m, n, k, expected_m,
                                      grouped_layout, stream);
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        FP4DynamicTileRuntime::launch(runtime, dyn_args);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        return {block_m, ceil_div(n, block_n)};
    }
    // ----- End DynamicTile early-out -----

    FP4GemmRuntime::MainLoopArguments mainloop_params_masked{
        cute::make_shape(m, n, k), ptr_A, stride_A, ptr_B, stride_B,
        ptr_scale_A, stride_SFA, ptr_scale_B, stride_SFB};

    FP4GemmRuntime::Args args{};
    args.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                        n, k, "GroupedMasked", "fp4_grouped_deep_gemm_masked", hasBias, n_expand, enable_sbo_overlap,
                        enable_act_and_quant_fusing ? "SiluAndMulPostQuantFp4" : "Default",
                        enable_act_and_quant_fusing && swiglu_limit > 0.0};
    args.launch_args = {grid, block, SMSIZE};

    // Branch exactly like the device-side CollectiveEpilogue conditional in generate_impl:
    //   hasBias || SHAPE_N % 2 != 0 ? WithTsm : (fused ? SiluAndMulPostQuant : NoTsm)
    // The TSM fallback is checked first so host and device can never disagree on the layout.
    if (hasBias || n % 2 != 0) {
        auto& params = args.kernel_params.with_tsm;
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
            .ptr_Bias = hasBias ? reinterpret_cast<float const*>(bias.data_ptr<float>()) : nullptr,
            .stride_Bias = {},
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)m, grouped_layout);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    } else if (enable_act_and_quant_fusing) {
        auto& params = args.kernel_params.silu_and_mul_post_quant;
        params = {};
        params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        params.problem_shape = {m, n, k, 1};
        params.collective_mainloop_params = mainloop_params_masked;
        params.collective_epilogue_params = {
            .thread = {1.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
                       1.0f, 1.0f, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr},
            .ptr_C = ptr_C,
            .stride_C = stride_C,
            .ptr_D = ptr_D_fused,
            .stride_D = stride_D,
            .ptr_SFD = ptr_SFD,
            .shape_m = (uint32_t)m,
            .swiglu_limit = (float)swiglu_limit,
        };
        params.hw_info = hw_info;
        params.scheduler = TileSchedulerArguments((uint32_t)m, grouped_layout);
        params.workspace = nullptr;
        params.signal = signal_ptr;
    } else {
        auto& params = args.kernel_params.no_tsm;
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

    return {block_m, ceil_div(n, block_n)};
}

} // namespace deep_gemm
