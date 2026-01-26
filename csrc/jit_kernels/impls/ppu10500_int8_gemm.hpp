#pragma once

#include <torch/python.h>
#include <cstdint> 
#include <cuda_fp8.h>
#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/python2cpp.hpp"
#include "../../utils/layout_type_name.hpp"
#include "cute/arch/mma.hpp"
#include "../heuristics/common_int8.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include "ppu/cutlass/detail/blockwise_scale_layout.hpp"
#include "../../../deep_gemm/include/deep_gemm/utils_rtc.cuh"
#include "epilogue.hpp"
#include "runtime_utils.hpp"
#include "../../../deep_gemm/include/deep_gemm/profiling_interface.hpp"

using namespace deep_gemm_int8;
namespace deep_gemm {

// class ComputeBlockInfoKernelRuntime final: public LaunchRuntime<ComputeBlockInfoKernelRuntime> {
// public:

//     struct ComputeBlockInfoArguments {
//       const uint32_t* grouped_layout;  // 分组布局指针
//       uint32_t num_groups;              // 分组数量
//       uint32_t* block_m_info;          // Block信息输出指针
//     };

//     struct Args {
//       ComputeBlockInfoArguments launch_attr_args;
//       LaunchArgs launch_args;
//     };

//     static std::string generate_impl(const int BlockM) {
//         return fmt::format(R"(
// #include "cutlass/cutlass.h"
// #include "cutlass/device_kernel.h"
// extern "C" 
// __global__ void computeBlockInfoKernel(
//     const uint32_t* __restrict__ group_num_list,
//     const uint32_t group_num,
//     uint32_t* __restrict__ block_info)
// {{
//     constexpr int32_t BlockM = {};
//     const uint32_t tid = threadIdx.x;
//     const uint32_t lane = cutlass::canonical_lane_idx();
//     const uint32_t warp_id = cutlass::canonical_warp_idx_sync();
//     const uint32_t num_warps = blockDim.x / 32;
//     uint32_t group_val = tid < group_num ? group_num_list[tid] : 0;
//     uint32_t block_val = (group_val + BlockM - 1) / BlockM;
//     uint32_t warp_group_scan = group_val;
//     uint32_t warp_block_scan = block_val;

//     for (uint32_t offset = 1; offset < 32; offset *= 2) {{
//         uint32_t tmp_group = __shfl_up_sync(0xFFFFFFFF, warp_group_scan, offset);
//         uint32_t tmp_block = __shfl_up_sync(0xFFFFFFFF, warp_block_scan, offset);
//         if (lane >= offset) {{
//             warp_group_scan += tmp_group;
//             warp_block_scan += tmp_block;
//         }}
//     }}

//     __shared__ uint32_t warp_group_totals[32];
//     __shared__ uint32_t warp_block_totals[32];

//     if (lane == 31) {{
//         warp_group_totals[warp_id] = warp_group_scan;
//         warp_block_totals[warp_id] = warp_block_scan;
//     }}

//     __syncthreads();

//     __shared__ uint32_t warp_group_prefix[32];
//     __shared__ uint32_t warp_block_prefix[32];

//     if (warp_id == 0) {{
//         uint32_t group_sum = 0;
//         uint32_t block_sum = 0;
//         for (uint32_t w = 0; w < num_warps; ++w) {{
//             warp_group_prefix[w] = group_sum;
//             warp_block_prefix[w] = block_sum;
//             group_sum += warp_group_totals[w];
//             block_sum += warp_block_totals[w];
//         }}
//         if (tid == 0) {{
//             *block_info = block_sum;
//         }}
//     }}
//     __syncthreads();

//     uint32_t group_prefix = warp_group_prefix[warp_id];
//     uint32_t block_prefix = warp_block_prefix[warp_id];
//     uint32_t warp_group_exclusive = warp_group_scan - group_val;
//     uint32_t warp_block_exclusive = warp_block_scan - block_val;
//     uint32_t global_group_prefix = group_prefix + warp_group_exclusive;
//     uint32_t global_block_prefix = block_prefix + warp_block_exclusive;

//     uint32_t base_offset = global_block_prefix * 4;
//     uint32_t* output_info = block_info + 4;
//     for (uint32_t i = 0; i < block_val; ++i) {{
//         uint32_t block_idx = base_offset + i * 4;
//         output_info[block_idx]     = tid;          // group_idx
//         output_info[block_idx + 1] = group_val;    // group_num
//         output_info[block_idx + 2] = i;            // block_in_group
//         output_info[block_idx + 3] = global_group_prefix; // prefix_group_sum
//     }}
// }}
// )",
//         BlockM);
// }

//     static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
//         // TODO: optimize `args` copy
//         DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.launch_attr_args));
//     }
// };

class PPU10500INT8GemmRuntime final: public LaunchRuntime<PPU10500INT8GemmRuntime> {
public:
    using ScaleGranularityShape = cute::Shape<cute::_1,cute::_128,cute::_128>;
    using ScaleConfig         = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(ScaleGranularityShape{}));
    using LayoutSFA           = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
    using LayoutSFB           = decltype(ScaleConfig::deduce_layoutSFB());
    using ElementA            = cutlass::float_e4m3_t;                          // Element type for A matrix operand
    using LayoutA             = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    using ElementB            = cutlass::float_e4m3_t;                          // Element type for B matrix operand
    using LayoutB             = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    using ElementD            = cutlass::bfloat16_t;
    using LayoutD             = cutlass::layout::RowMajor;
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

    // 问题尺寸

    using GemmProblemSize = cute::tuple<int32_t,int32_t,int32_t,int32_t>;

    struct MainLoopArguments {
      cutlass::float_e4m3_t const* ptr_A;
      cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
      cutlass::float_e4m3_t const* ptr_B;
      cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
      float const * ptr_scale_A;
      float const * ptr_scale_B;
    };

    struct LinearCombinationArgs {
      float alpha = 1.0f;                         ///< scales accumulators
      float beta = 0.0f;                         ///< scales source tensor
      float const *alpha_ptr = nullptr;              ///< pointer to accumulator scalar - if not null, loads it from memory
      float const *beta_ptr = nullptr;               ///< pointer to source scalar - if not null, loads it from memory
      float const* const* alpha_ptr_array = nullptr; ///< array of pointers to accumulator scalar per group/batch
      float const* const* beta_ptr_array = nullptr;  ///< array of pointers to source scalar per group/batch
// #if SUPPORT_FP8_SCALING
      float scale_a = float(1);
      float scale_b = float(1);
      float scale_c = float(1);
      float scale_d = float(1);
      float const* scale_a_ptr = nullptr;
      float const* scale_b_ptr = nullptr;
      float const* scale_c_ptr = nullptr;
      float const* scale_d_ptr = nullptr;
// #endif
    };

    // Epilogue
    struct EpilogueArgs {
      LinearCombinationArgs callback;
      cutlass::bfloat16_t * ptr_C;        // 通常为 nullptr（in-place D）
      cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;   // 通常等于 stride_D

      cutlass::bfloat16_t * ptr_D;
      cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
      int block_m, block_n, block_k, warp_m, warp_n, num_groups, num_stages;
      std::string kernel_name;
    };

    struct TileSchedulerArguments {
      int* grouped_layout;
      uint32_t shape_m;
    };

    // 主 Arguments 结构体
    struct GemmArguments {
      GemmUniversalMode mode;
      GemmProblemSize problem_shape;
      MainLoopArguments mainloopargs;
      EpilogueArgs epilogueargs;
      cutlass::KernelHardwareInfo hw_info;
      TileSchedulerArguments scheduler{};
      int32_t* signal{nullptr};
    };

    struct TileSchedulerParams {
      int* grouped_layout;
      uint32_t shape_m;
    };

    using CollectiveMainloopParams = MainLoopArguments;
    using CollectiveEpilogueParams = EpilogueArgs;

    struct GemmKernelParams {
      GemmUniversalMode mode;
      GemmProblemSize problem_shape;
      CollectiveMainloopParams collective_mainloop_params;
      CollectiveEpilogueParams collective_epilogue_params;
      cutlass::KernelHardwareInfo hw_info;
      TileSchedulerParams tile_scheduler_params;
      void* workspace{nullptr};//workspace,
      int32_t* signal{nullptr};
    };

    struct Args {
      GemmArguments gemm_args;
      LaunchInfo launch_info;
      LaunchArgs launch_args;
      GemmKernelParams kernel_params;
    };

    static GemmKernelParams to_underlying_arguments_rtc(GemmArguments args, void* workspace, int* grouped_layout) {
      auto problem_shape = args.problem_shape;
      auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
      // Get SM count if needed, otherwise use user supplied SM count
      int sm_count = args.hw_info.sm_count;
      if (sm_count <= 0) {
        CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
            "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
        sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
      }

      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

      cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

      uint32_t problem_shape_m = cute::get<0>(problem_shape_MNKL);
      TileSchedulerParams scheduler = {grouped_layout, problem_shape_m};

      return {
        args.mode,
        problem_shape,
        args.mainloopargs,
        args.epilogueargs,
        hw_info,
        scheduler,
        workspace,
        args.signal
      };

    }

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define FP8_NVRTC
#include <int8_gemm_cutlass3.cuh>
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

using ArchTag = cutlass::arch::Sm80;

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand

// D matrix configuration
using         ElementD    = cutlass::bfloat16_t;
using         LayoutD     = cutlass::layout::RowMajor;

// C matrix configuration
using         ElementC    = ElementD;
using         LayoutC     = LayoutD;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ElementScalar    = ElementCompute;

using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
static constexpr int WarpOnM = BLOCK_M / WARP_M;
static constexpr int WarpOnN = BLOCK_N / WARP_N;
constexpr bool EnableMultistageOnN_ = false;
constexpr bool kEnableSboOverlap = false;
constexpr bool kEnableMoeDynamicTile = false;

using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ElementA,ElementB,ElementAccumulator>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
    Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _32>>;       // 1x1x1 value group

constexpr int EnableMultistageOnN = EnableMultistageOnN_
                                    && (SHAPE_N % (BLOCK_N) == 0)
                                    && (SHAPE_K > (BLOCK_K * STAGES));
static constexpr int N_EXPAND = EnableMultistageOnN ? cutlass::gemm::KernelAiuMultistageOnN::N_EXPAND : 1;

using KernelSchedule = typename cutlass::platform::conditional<
    EnableMultistageOnN,
    cutlass::gemm::KernelAiuMultistageOnN,
    cutlass::gemm::KernelAiuMultistage
>::type;
using DispatchPolicy = cutlass::gemm::MainloopAcomputeAiuA8W8<STAGES, KernelSchedule>;
static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
static constexpr int TSM_LD_NUM = BLOCK_M == 8 ? 2 : 4;

using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false>;
using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true>;
// A
using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
// B
using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

// Mainloop
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
>;

// Epilogue
static constexpr bool IsAligedN = SHAPE_N % BLOCK_N == 0 ? true : false;
// reduce vreg to use ScaleType::Nothing for alpha=1 & beta=0
using CollectiveEpilogue_noTsm = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::EpilogueDefault,
    IsAligedN>;

static constexpr int AlignmentC = 16 / sizeof(ElementC);
using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute>;
using EpilogueSchedule = typename cutlass::epilogue::EpilogueSimtVectorized;
using CollectiveEpilogue_withTsm = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp,
    TileShape, WarpShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    EpilogueSchedule,
    DefaultOperation
>::CollectiveOp;

static constexpr bool EpilogueWithTsm = false;
using CollectiveEpilogue = typename cutlass::platform::conditional<
    EpilogueWithTsm,
    CollectiveEpilogue_withTsm,
    CollectiveEpilogue_noTsm
>::type;

static constexpr GemmType kGemmType = GemmType::DenseGemm;

using TileScheduler = DeepGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, NUM_GROUPS>;
using GemmKernel = cutlass::gemm::kernel::DeepGemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler,
    kEnableSboOverlap>;


// Kernel 函数定义
extern "C" 
__launch_bounds__(512)
__global__ void {}(
  typename GemmKernel::Params params
) {{
  extern __shared__ char smem[];
  int* grouped_layout = nullptr;
  GemmKernel op;
  op(params, smem);
}}
}}
)",
        cute::get<1>(args.gemm_args.problem_shape), cute::get<2>(args.gemm_args.problem_shape),
        args.launch_info.block_m, args.launch_info.block_n, args.launch_info.block_k,
        args.launch_info.num_groups, 
        args.launch_info.warp_m, args.launch_info.warp_n,
        args.launch_info.num_stages,
        args.launch_info.kernel_name
        );
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
static void gemm_a8w8_per_channel_nt(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                              const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                              const torch::Tensor& out,
                              const int& m, const int& n, const int& k, std::optional<ConfigTuple> config = std::nullopt, at::cuda::CUDAStream stream = at::cuda::getDefaultCUDAStream()) {

    // # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");
    if (m == 0) {
        return;
    }
    int num_sms = get_num_sms();

    ConfigTuple selected_config;
    if (config.has_value()) {
      selected_config = *config;
    } else {
      selected_config = deep_gemm_int8::get_best_configs(m, n, k, 1, num_sms);
    }
    
    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;

    auto SMSIZE = std::get<0>(smem_config);
    uint32_t kNumGroups = 1;
    
    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using ScaleGranularityShape = cute::Shape<cute::_1,cute::_128,cute::_128>;
    using ScaleConfig         = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(ScaleGranularityShape{}));
    using LayoutSFA           = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
    using LayoutSFB           = decltype(ScaleConfig::deduce_layoutSFB());
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
    // constexpr static bool kIsNoPadPreprocessLayout = kGemmType == GemmType::GroupedNoPad && kNumGroups >= 128;
    // if (kIsNoPadPreprocessLayout) {
    //   uint32_t block_size = std::max(32, next_power_of_two(kNumGroups));
    //   auto compute_block_info_args = ComputeBlockInfoKernelRuntime::Args{
    //     .launch_attr_args = {reinterpret_cast<const uint32_t*>(grouped_layout), kNumGroups, reinterpret_cast<uint32_t*>(block_m_info)},
    //     .launch_args = LaunchArgs(1, block_size, 0),
    //   };
    //   const auto& code_blockinfo = ComputeBlockInfoKernelRuntime::generate(block_m);
    //   const auto& runtime_blockinfo = compiler->build("computeBlockInfoKernel", code_blockinfo);
    //   ComputeBlockInfoKernelRuntime::launch(runtime_blockinfo, compute_block_info_args, stream);
    // }

    cutlass::float_e4m3_t* converted_input_b = reinterpret_cast<cutlass::float_e4m3_t*>(rhs.data_ptr<at::Float8_e4m3fn>());
    cutlass::float_e4m3_t* converted_input_a = reinterpret_cast<cutlass::float_e4m3_t*>(lhs.data_ptr<at::Float8_e4m3fn>());
    cutlass::bfloat16_t * converted_output = reinterpret_cast<cutlass::bfloat16_t *>(out.data_ptr<at::BFloat16>());
    float* scales_a_ptr = lhs_scales.data_ptr<float>();
    float* scales_b_ptr = rhs_scales.data_ptr<float>();
    // TODO get hw info from real env
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.sm_count);
    const auto gemm_args = PPU10500INT8GemmRuntime::GemmArguments{
      .mode = cutlass::gemm::GemmUniversalMode::kGemm,
      .problem_shape = {m, n, k, 1},
      .mainloopargs = {converted_input_a, stride_A, converted_input_b, stride_B,
        scales_a_ptr, scales_b_ptr},
      .epilogueargs = {
        {1, 0},
        nullptr, stride_D,
        converted_output, stride_D,
      },
      .hw_info = hw_info,
      .scheduler = {},
      .signal = nullptr
    };

    PPU10500INT8GemmRuntime::GemmKernelParams params = PPU10500INT8GemmRuntime::to_underlying_arguments_rtc(gemm_args, nullptr, grouped_layout);
    auto args = PPU10500INT8GemmRuntime::Args{
      .gemm_args = gemm_args,
      .launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages, "sm100_int8_deep_gemm_1d1d"},
      .launch_args = LaunchArgs(grid, block, SMSIZE),
      .kernel_params = params
    };
    const auto& code = PPU10500INT8GemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_int8_deep_gemm_1d1d", code, block.x, SMSIZE);
    const auto& max_block_per_cu = compiler->get_max_block_per_cu();
    std::cout << "max_block_per_cu is " << max_block_per_cu << std::endl;
    hw_info.sm_count = hw_info.sm_count * max_block_per_cu;
    grid = get_grid_shape(hw_info.sm_count);
    args.launch_args.grid_dim = grid;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()){
        dg_prof_params.set_params(
            kGemmType, false, std::string("int8"), kNumGroups, m, n, k, 0,
            grouped_layout, stream
        );
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    PPU10500INT8GemmRuntime::launch(runtime, args, stream);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);
}
} // namespace deep_gemm
