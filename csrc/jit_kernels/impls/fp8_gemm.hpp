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
#include "../../../deep_gemm/include/deep_gemm/scheduler_cutlass3.cuh"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "../../../deep_gemm/include/deep_gemm/utils_rtc.cuh"
#include "../../../deep_gemm/include/deep_gemm/profiling_interface.hpp"
#include "int8_gemm.hpp"

using namespace deep_gemm_fp8_common;
namespace deep_gemm {

class ComputeBlockInfoKernelRuntime final : public LaunchRuntime<ComputeBlockInfoKernelRuntime> {
public:
    struct ComputeBlockInfoArguments {
        const uint32_t* grouped_layout;
        const uint32_t num_groups;
        uint32_t* block_m_info;
    };

    struct Args {
        ComputeBlockInfoArguments launch_attr_args;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const int BlockM) {
        return fmt::format(
            R"(
extern "C"
__global__ void computeBlockInfoKernel(
    const uint32_t* __restrict__ group_num_list,
    const uint32_t group_num,
    uint32_t* __restrict__ block_info)
{{
    constexpr int32_t BlockM = {};
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp_id = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t num_warps = blockDim.x / 32;
    uint32_t group_val = tid < group_num ? group_num_list[tid] : 0;
    uint32_t block_val = (group_val + BlockM - 1) / BlockM;
    uint32_t warp_group_scan = group_val;
    uint32_t warp_block_scan = block_val;

    for (uint32_t offset = 1; offset < 32; offset *= 2) {{
        uint32_t tmp_group = __shfl_up_sync(0xFFFFFFFF, warp_group_scan, offset);
        uint32_t tmp_block = __shfl_up_sync(0xFFFFFFFF, warp_block_scan, offset);
        if (lane >= offset) {{
            warp_group_scan += tmp_group;
            warp_block_scan += tmp_block;
        }}
    }}

    __shared__ uint32_t warp_group_totals[32];
    __shared__ uint32_t warp_block_totals[32];

    if (lane == 31) {{
        warp_group_totals[warp_id] = warp_group_scan;
        warp_block_totals[warp_id] = warp_block_scan;
    }}
    __syncthreads();

    __shared__ uint32_t warp_group_prefix[32];
    __shared__ uint32_t warp_block_prefix[32];

    if (warp_id == 0) {{
        uint32_t group_sum = 0;
        uint32_t block_sum = 0;
        for (uint32_t w = 0; w < num_warps; ++w) {{
            warp_group_prefix[w] = group_sum;
            warp_block_prefix[w] = block_sum;
            group_sum += warp_group_totals[w];
            block_sum += warp_block_totals[w];
        }}
        if (tid == 0) {{
            *block_info = block_sum;
        }}
    }}
    __syncthreads();

    uint32_t group_prefix = warp_group_prefix[warp_id];
    uint32_t block_prefix = warp_block_prefix[warp_id];
    uint32_t warp_group_exclusive = warp_group_scan - group_val;
    uint32_t warp_block_exclusive = warp_block_scan - block_val;
    uint32_t global_group_prefix = group_prefix + warp_group_exclusive;
    uint32_t global_block_prefix = block_prefix + warp_block_exclusive;

    uint32_t base_offset = global_block_prefix * 4;
    uint32_t* output_info = block_info + 4;
    for (uint32_t i = 0; i < block_val; ++i) {{
        uint32_t block_idx = base_offset + i * 4;
        output_info[block_idx]     = tid;          // group_idx
        output_info[block_idx + 1] = group_val;    // group_num
        output_info[block_idx + 2] = global_block_prefix; // block_in_group
        output_info[block_idx + 3] = global_group_prefix; // prefix_group_sum
    }}
}}
)",
            BlockM);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.launch_attr_args.grouped_layout,
                                            args.launch_attr_args.num_groups, args.launch_attr_args.block_m_info));
    }
};

class FP8GemmRuntime final : public LaunchRuntime<FP8GemmRuntime> {
public:
    using ScaleGranularityShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig =
        decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(
            ScaleGranularityShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA()); // Layout type for SFA matrix operand
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

    using GemmProblemSize = cute::tuple<int32_t, int32_t, int32_t, int32_t>;

    struct MainLoopArguments {
        cutlass::float_e4m3_t* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        cutlass::float_e4m3_t* ptr_B;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
        uint32_t mma_promotion_interval = 4;
        float* ptr_scale_A;
        LayoutSFA layout_SFA;
        float* ptr_scale_B;
        LayoutSFB layout_SFB;
    };

    struct LinearCombinationArgs {
        float alpha = 1.0f;               ///< scales accumulators
        float beta = 0.0f;                ///< scales source tensor
        float const* alpha_ptr = nullptr; ///< pointer to accumulator scalar - if not null, loads it from memory
        float const* beta_ptr = nullptr;  ///< pointer to source scalar - if not null, loads it from memory
        float const* const* alpha_ptr_array = nullptr; ///< array of pointers to accumulator scalar per group/batch
        float const* const* beta_ptr_array = nullptr;  ///< array of pointers to source scalar per group/batch
                                                       // float scale_a = float(1);
                                                       // float scale_b = float(1);
                                                       // float scale_c = float(1);
                                                       // float scale_d = float(1);
                                                       // float const* scale_a_ptr = nullptr;
                                                       // float const* scale_b_ptr = nullptr;
                                                       // float const* scale_c_ptr = nullptr;
                                                       // float const* scale_d_ptr = nullptr;
    };

    // Epilogue
    struct EpilogueArgs {
        LinearCombinationArgs callback;
        cutlass::bfloat16_t* ptr_C;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;

        cutlass::bfloat16_t* ptr_D;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n, num_groups, num_stages;
        std::string gemm_type, kKernelType, kernel_name;
        bool kEnableSboOverlap;
    };

    // Main Arguments
    struct GemmArguments {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        MainLoopArguments mainloopargs;
        EpilogueArgs epilogueargs;
        cutlass::KernelHardwareInfo hw_info;
        TileSchedulerArguments scheduler{};
        int32_t* signal{nullptr};
    };

    using CollectiveMainloopParams = MainLoopArguments;
    using CollectiveEpilogueParams = EpilogueArgs;

    struct GemmKernelParams {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        CollectiveMainloopParams collective_mainloop_params;
        CollectiveEpilogueParams collective_epilogue_params;
        cutlass::KernelHardwareInfo hw_info;
        TileSchedulerArguments scheduler;
        void* workspace{nullptr}; // workspace,
        int32_t* signal{nullptr};
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
    };

    static GemmKernelParams to_underlying_arguments_rtc(GemmArguments args, void* workspace) {
        auto problem_shape = args.problem_shape;
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.cu_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST(
                "  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

        uint32_t problem_shape_m = cute::get<0>(problem_shape_MNKL);

        return {args.mode, problem_shape,  args.mainloopargs, args.epilogueargs,
                hw_info,   args.scheduler, workspace,         args.signal};
    }

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#define FP8_HGRTC
#include <fp8_gemm.cuh>
namespace deep_gemm {{
using namespace cute;
using cutlass::KernelHardwareInfo;

constexpr int SHAPE_N = {};
constexpr int SHAPE_K = {};
constexpr int BLOCK_M = {};
constexpr int BLOCK_N = {};
constexpr int BLOCK_K = {};
constexpr int NUM_GROUPS = {};
constexpr int WARP_M = {};
constexpr int WARP_N = {};
constexpr int STAGES = {};

static constexpr GemmType kGemmType = GemmType::{};
static constexpr KernelType kKernelType = KernelType::{}; //Default;
static constexpr bool kEnableSboOverlap = {};

using ScaleGranularityShape = cute::Shape<cute::_1,cute::_128,cute::_128>;
using ScaleConfig         = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(ScaleGranularityShape{{}}));
using LayoutSFA           = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
using LayoutSFB           = decltype(ScaleConfig::deduce_layoutSFB());
static constexpr bool kUseNStageKernel = SHAPE_K <= 512 && (SHAPE_N % (BLOCK_N * KernelAiuMultistageOnN::N_EXPAND) == 0)
                                          && (BLOCK_K == 128) && (SHAPE_K % BLOCK_K == 0) && STAGES == 2;
constexpr int N_EXPAND = kUseNStageKernel ? KernelAiuMultistageOnN::N_EXPAND : 1;

using TileScheduler = DeepGemmScheduler<
  kGemmType,
  SHAPE_N,
  SHAPE_K,
  BLOCK_M,
  BLOCK_N * N_EXPAND,
  NUM_GROUPS
>;

static constexpr bool UseAIU = true;
// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
static constexpr int AlignmentA  = UseAIU ? 1 : 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
static constexpr int AlignmentB  = UseAIU ? 1 : 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = cutlass::bfloat16_t;
using         LayoutD     = cutlass::layout::RowMajor;
static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

// C matrix configuration
using         ElementC    = ElementD;
using         LayoutC     = LayoutD;
static constexpr int AlignmentC  = AlignmentD;

using ArchTag = cutlass::arch::PPU0015;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ElementScalar    = ElementCompute;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, cutlass::arch::OpClassTensorOp,
  ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
  ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
  ElementAccumulator,  // ElementAccumulator
  Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>,
  Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>,
  Int<STAGES>,
  cutlass::gemm::KernelAiuMultistageWithBlockWiseScale
>::CollectiveOp;

using EpilogueDispatchPolicy = cutlass::epilogue::EpilogueSimtVectorized;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
using CollectiveEpilogueWithTsm = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>, 
    Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>,
    EpilogueTileType,
    ElementCompute, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueDispatchPolicy
  >::CollectiveOp;

// Epilogue
static constexpr bool IsAligedN = SHAPE_N % BLOCK_N == 0 ? true : false;
// reduce vreg to use ScaleType::Nothing for alpha=1 & beta=0
using CollectiveEpilogueNoTsm = typename cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::EpilogueDefault,
    IsAligedN>;
static constexpr bool EpilogueWithTsm = false;
using CollectiveEpilogue = typename cutlass::platform::conditional<
    EpilogueWithTsm,
    CollectiveEpilogueWithTsm,
    CollectiveEpilogueNoTsm
>::type;

using GemmKernel = DeepGemmUniversal<
  Shape<int,int,int,int>,
  CollectiveMainloop,
  CollectiveEpilogue,
  TileScheduler,
  kEnableSboOverlap,
  kUseNStageKernel
>;

extern "C"
__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmKernel::MinBlocksPerMultiprocessor)
__global__ void {}(
  typename GemmKernel::Params params
) {{
  extern __shared__ char smem[];
  GemmKernel op;
  op(params, smem);
}}
}}
)",
            cute::get<1>(args.kernel_params.problem_shape), cute::get<2>(args.kernel_params.problem_shape),
            args.launch_info.block_m, args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_groups,
            args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.num_stages, args.launch_info.gemm_type,
            args.launch_info.kKernelType, args.launch_info.kEnableSboOverlap, args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
static void fp8_gemm(const torch::Tensor& lhs, const torch::Tensor& lhs_scales, const torch::Tensor& rhs,
                     const torch::Tensor& rhs_scales, const torch::Tensor& out, const int& m, const int& n,
                     const int& k, std::optional<ConfigTuple> configs = std::nullopt) {
    auto lhs_scales_aligned = get_col_major_tma_aligned_tensor(lhs_scales);
    int num_sms = get_num_sms();

    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp8_common::get_smem_config(nst, k, bm, bn, bk));
    } else {
        selected_config = deep_gemm_fp8_common::get_best_configs(m, n, k, 1, num_sms);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
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
    static constexpr bool kEnableMultistageOnN = false;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    int* grouped_layout = nullptr;
    int* block_m_info = nullptr;
    int* layout_info = grouped_layout;

    static constexpr GemmType kGemmType = GemmType::DenseGemm;
    constexpr static bool kIsNoPadPreprocessLayout = kGemmType == GemmType::GroupedNoPad && kNumGroups >= 128;
    if (kIsNoPadPreprocessLayout) {
        uint32_t block_size = std::max(32, next_power_of_two(kNumGroups));
        auto compute_block_info_args = ComputeBlockInfoKernelRuntime::Args{
            .launch_attr_args = {reinterpret_cast<const uint32_t*>(grouped_layout), 256,
                                 reinterpret_cast<uint32_t*>(block_m_info)},
            .launch_args = {1, block_size, 0},
        };
        const auto& code_blockinfo = ComputeBlockInfoKernelRuntime::generate(block_m);
        const auto& runtime_blockinfo = compiler->build("computeBlockInfoKernel", code_blockinfo);
        ComputeBlockInfoKernelRuntime::launch(runtime_blockinfo, compute_block_info_args);
    }

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

    auto args = FP8GemmRuntime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                                                     "DenseGemm", "Default", "fp8_deep_gemm", false},
                                     .launch_args = {grid, block, SMSIZE},
                                     .kernel_params = params};

    const auto& code = FP8GemmRuntime::generate(args);
    const auto& runtime = compiler->build("fp8_deep_gemm", code, block.x, SMSIZE);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;
    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("fp8"), kNumGroups, m, n, k, 0, grouped_layout,
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

        printf("[DenseGemm_FP8:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n, block_k,
               warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }
}
} // namespace deep_gemm
