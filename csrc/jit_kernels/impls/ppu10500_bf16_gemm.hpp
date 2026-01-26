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
#include "../heuristics/common_bf16.hpp"
// #include "../heuristics/gemm_search_space.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include "ppu/cutlass/detail/blockwise_scale_layout.hpp"


#include "epilogue.hpp"
#include "runtime_utils.hpp"
using namespace deep_gemm_bf16;
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

class BF16GemmCutlass3Runtime final: public LaunchRuntime<BF16GemmCutlass3Runtime> {
public:
    using ElementA            = cutlass::bfloat16_t;                          // Element type for A matrix operand
    using LayoutA             = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    using ElementB            = cutlass::bfloat16_t;                          // Element type for B matrix operand
    using LayoutB             = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    using ElementD            = cutlass::bfloat16_t;
    using LayoutD             = cutlass::layout::RowMajor;
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

    // 问题尺寸

    using GemmProblemSize = cute::tuple<int32_t,int32_t,int32_t,int32_t>;

    struct MainLoopArguments {
      cutlass::bfloat16_t const* ptr_A;
      cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
      cutlass::bfloat16_t const* ptr_B;
      cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
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

    using CollectiveEpilogueParams = EpilogueArgs;
    using TileSchedulerParams = TileSchedulerArguments;
    using CollectiveMainloopParams = MainLoopArguments;

    struct GemmKernelParams {
      GemmUniversalMode mode;
      GemmProblemSize problem_shape;
      CollectiveMainloopParams collective_mainloop_params;
      CollectiveEpilogueParams collective_epilogue_params;
      cutlass::KernelHardwareInfo hw_info;
      TileSchedulerArguments scheduler{};
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
      // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
      // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
      // subtile will not be used, therefore separate reduction will not be enabled.
      // uint32_t problem_shape_m = cute::get<0>(problem_shape_MNKL);
      // TileSchedulerParams scheduler = {grouped_layout, problem_shape_m};
      return {
        args.mode,
        problem_shape,
        args.mainloopargs, // CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
        args.epilogueargs,
        hw_info,
        args.scheduler,
        workspace,
        args.signal
      };

    }

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define BF16_NVRTC
#include <bf16_gemm_cutlass3.cuh>
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


static constexpr KernelType kKernelType = KernelType::Default;
using ElementA    = cutlass::bfloat16_t;
using ElementB    = cutlass::bfloat16_t;
using ElementC    = cutlass::bfloat16_t;
using LayoutA     = cutlass::layout::RowMajor;
using LayoutB     = cutlass::layout::ColumnMajor;
using LayoutC     = cutlass::layout::RowMajor;
using ElementD    = ElementC;
using LayoutD     = cutlass::layout::RowMajor;
using ElementCompute      = float;
using ElementScalar       = ElementCompute;
using LinearCombOutType   = ElementD;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ArchTag = cutlass::arch::Sm80;

using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
static constexpr int WarpOnM = BLOCK_M / WARP_M;
static constexpr int WarpOnN = BLOCK_N / WARP_N;
static constexpr bool kEnableSboOverlap = false;

using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<cutlass::bfloat16_t, cutlass::bfloat16_t, float>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
    Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _16>>;       // 1x1x1 value group

static constexpr int N_EXPAND = 1;//kKernelType == KernelType::MultistageOnN && (SHAPE_N % (BLOCK_N) == 0) ? cutlass::gemm::KernelAiuMultistageOnN::N_EXPAND : 1;
using KernelSchedule = cute::conditional_t<
    kKernelType == KernelType::OverlapMainloop,
    cutlass::gemm::KernelAiuMultistageOverlapMainloop,
    cute::conditional_t<
      kKernelType == KernelType::OverlapPrologue,
      cutlass::gemm::KernelAiuMultistageOverlapPrologue,
      cute::conditional_t<
        kKernelType == KernelType::MultistageOnN,
        cutlass::gemm::KernelAiuMultistageOnN,
        cutlass::gemm::KernelAiuMultistage>>>;
using DispatchPolicy = cute::conditional_t<
    kKernelType == KernelType::OverlapMainloop,
    cutlass::gemm::MainloopAcomputeOverlapMainloop<STAGES, KernelSchedule>,
    cute::conditional_t<
      kKernelType == KernelType::OverlapPrologue,
      cutlass::gemm::MainloopAcomputeOverlapPrologue<STAGES, KernelSchedule>,
      cutlass::gemm::MainloopAcomputeAiuOpt<STAGES, KernelSchedule>>>;

static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
static constexpr int TSM_LD_NUM = BLOCK_M == 8 ? 2 : 4;

static constexpr int SmemLayoutStageStrideA = kKernelType == KernelType::OverlapMainloop || kKernelType == KernelType::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_M * BLOCK_K;
static constexpr int SmemLayoutStageStrideB = kKernelType == KernelType::OverlapMainloop || kKernelType == KernelType::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_N * BLOCK_K;
using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;
// using t1 = DefaultOperandB::xhzhao;
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
using CollectiveEpilogue_noTsm = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 8, float, float>,
    cutlass::gemm::EpilogueDefault>;

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

using TileScheduler = DeepGemmScheduler<GemmType::DenseGemm, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, NUM_GROUPS>;
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

class BF16GemmRuntime final: public LaunchRuntime<BF16GemmRuntime> {
public:
    using ElementA            = cutlass::bfloat16_t;                          // Element type for A matrix operand
    using LayoutA             = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    using ElementB            = cutlass::bfloat16_t;                          // Element type for B matrix operand
    using LayoutB             = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    using ElementD            = cutlass::bfloat16_t;
    using LayoutD             = cutlass::layout::RowMajor;
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

    // 问题尺寸

    using GemmProblemSize = cute::tuple<int32_t,int32_t,int32_t,int32_t>;

    struct MainLoopArguments {
      cutlass::bfloat16_t const* ptr_A;
      cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
      cutlass::bfloat16_t const* ptr_B;
      cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
    };

    struct LinearCombinationArgs {
      float alpha;                  ///< scales accumulators
      float beta;                   ///< scales source tensor
      float const *alpha_ptr = nullptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
      float const *beta_ptr = nullptr;        ///< pointer to source scalar - if not null, loads it from memory
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
    // struct GemmArguments {
    //     LinearCombinationArgs output_op;

    //     typename Mma::IteratorA::TensorRef ref_A;
    //     typename Mma::IteratorB::TensorRef ref_B;

    //     typename Epilogue::OutputTileIterator::TensorRef ref_D;

    //     int64_t gemm_m;
    //     int64_t gemm_n;
    //     int64_t gemm_k;

    //     int* grouped_layout;

    //     int problem_count {0};
    //     int threadblock_count {0};

    //     int32_t* signal;

    //     // in-order to compatility with base_group
    //     cutlass::gemm::GemmCoord* host_problem_sizes {nullptr};

    // };

    struct ProblemVisitorParams {
        int const* grouped_layout;
        int64_t gemm_n;
        int64_t gemm_k;
        int64_t gemm_m;
        int32_t problem_count;
    };

    struct GemmKernelParams {
        ProblemVisitorParams problem_visitor;
        int threadblock_count;
        int problem_count;

        LinearCombinationArgs output_op;

        cutlass::bfloat16_t const* ptr_A;
        torch::Layout params_A;
        cutlass::bfloat16_t const* ptr_B;
        torch::Layout params_B;//typename Mma::IteratorB::Params params_B;
        cutlass::bfloat16_t const* ptr_D;
        torch::Layout params_D; //typename Epilogue::OutputTileIterator::Params

        int32_t* signal;
    };

    struct Args {
      // GemmArguments gemm_args;
      LaunchInfo launch_info;
      LaunchArgs launch_args;
      GemmKernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <bf16_gemm.cuh>
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


using ThreadblockShape = cutlass::gemm::GemmShape<BLOCK_M, BLOCK_N, BLOCK_K>;
        using WarpShape = cutlass::gemm::GemmShape<WARP_M, WARP_N, BLOCK_K>;
        using ElementType = cutlass::bfloat16_t;

        using OperatorClass = cutlass::arch::OpClassTensorOp;
        using ElementAccumulator = float;

        static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<ElementType>::value;
        static constexpr int LimitedPerAccessC_ = ((ThreadblockShape::kN) * 8 / (ThreadblockShape::kN / WarpShape::kN) / 32);
        static constexpr int ElementsPerAccessC = LimitedPerAccessC_ < ElementsPerAccess ? LimitedPerAccessC_ : ElementsPerAccess;
        static constexpr int ThreadblockK = ThreadblockShape::kK;
        using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

        using EpilogueOp = typename cutlass::epilogue::thread::LinearCombination<ElementType, ElementsPerAccessC,
                ElementAccumulator, ElementAccumulator>;

        using DefaultGemm = typename aiu::gemm::kernel::DefaultGemmGrouped<ElementType, cutlass::layout::RowMajor, ElementsPerAccess,
                                                                            ElementType, cutlass::layout::ColumnMajor, ElementsPerAccess,
                                                                            ElementType, cutlass::layout::RowMajor, ElementAccumulator,
                                                                            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape, WarpShape,
                                                                            InstructionShape, EpilogueOp,
                                                                            cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, kNumStages,
                                                                            cutlass::gemm::kernel::GroupScheduleMode::kDeepGemm,
                                                                            cutlass::arch::OpMultiplyAdd>::GemmKernel;

        using ProblemVisitor = Scheduler<kGemmType, SHAPE_N, ThreadblockShape>;

        using GemmKernel = GemmKernel<typename DefaultGemm::Mma, typename DefaultGemm::Epilogue, ProblemVisitor, kEnableSboOverlap>;


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
        // TODO fake
        args.launch_info.block_m, args.launch_info.block_n, //cute::get<1>(args.gemm_args.problem_shape), cute::get<2>(args.gemm_args.problem_shape),
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

static void bf16_gemm(const torch::Tensor& lhs,
                      const torch::Tensor& rhs,
                      const torch::Tensor& out,
                      const int& m, const int& n, const int& k,
                      at::cuda::CUDAStream stream = at::cuda::getDefaultCUDAStream()) {

    int num_sms = get_num_sms();
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    std::vector<int> shape = {m, n, k};

    bool all_ok = true;
    for (int64_t a : shape) {
    if (!(a >= 4096 && (a % 64 == 0))) { all_ok = false; break; }
    }

    std::string dev_name(device_props.name);
    bool zw_ok = (dev_name.find("ZW810E") != std::string::npos) ||
                (dev_name.find("ZW810")  != std::string::npos);
                
    using Config = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

    Config cfg;
    if (all_ok && zw_ok) {
        cfg = get_gemm_best_configs_v2(shape, 2, num_sms);
    } else {
        std::cout << "\ngoing there" << std::endl;
        cfg = deep_gemm_bf16::get_best_configs(m, n, k, 1, num_sms);
    }
    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = cfg;
    auto extra_info = get_extra_info();    

    auto SMSIZE = std::get<0>(smem_config);
    // auto SWIZZLE_D_MODE = std::get<1>(smem_config);
    // auto BLOCK_N_PADDING = std::get<2>(smem_config);
    // printf("num_sms_new is %d, block_m is %d, block_n is %d, block_k is %d, warp_m is %d, warp_n is %d,  num_stages is %d SMSIZE is %d", num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, SMSIZE);

    int kNumGroups = 1;
    
    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));


    int* grouped_layout = nullptr;
    // int* block_m_info = nullptr;
    int* layout_info = grouped_layout;

    // if (false) { // TileScheduler::kIsNoPadPreprocessLayout
    //   uint32_t block_size = std::max(32, next_power_of_two(kNumGroups));
    //   auto compute_block_info_args = ComputeBlockInfoKernelRuntime::Args{
    //     .launch_attr_args = {reinterpret_cast<const uint32_t*>(grouped_layout), kNumGroups, reinterpret_cast<uint32_t*>(block_m_info)},
    //     .launch_args = LaunchArgs(1, block_size, 0),
    //   };
    //   const auto& code = ComputeBlockInfoKernelRuntime::generate(block_m);
    //   const auto& runtime = compiler->build("ComputeBlockInfoKernel", code);
    //   ComputeBlockInfoKernelRuntime::launch(runtime, compute_block_info_args, stream);
    //   layout_info = block_m_info;
    // }

    cutlass::bfloat16_t* input_b = reinterpret_cast<cutlass::bfloat16_t*>(rhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_a = reinterpret_cast<cutlass::bfloat16_t*>(lhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());

    // TODO get hw info from real env
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = num_sms_new;   // * 2;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;  //get_block_shape();
    dim3 grid = get_grid_shape(hw_info.sm_count);
    if (extra_info["use_cutlass3"]) {
      std::cout << "come into bf16 cutlass3" << std::endl;
      const auto gemm_args = BF16GemmCutlass3Runtime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, 1},
        .mainloopargs = {input_a, stride_A, input_b, stride_B},
        .epilogueargs = {
          {1, 0},
          output, stride_D,
          output, stride_D,
        },
        .hw_info = hw_info,
        .scheduler = {layout_info, (uint32_t)m},
        .signal = nullptr
      };

      BF16GemmCutlass3Runtime::GemmKernelParams params = BF16GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr, grouped_layout);
      auto args = BF16GemmCutlass3Runtime::Args{
        .gemm_args = gemm_args,
        .launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages, "ppu10500_bf16_gemm"},
        .launch_args = LaunchArgs(grid, block, SMSIZE),
        .kernel_params = params
      };
      const auto& code = BF16GemmCutlass3Runtime::generate(args);
      const auto& runtime = compiler->build("ppu10500_bf16_gemm", code, block.x, SMSIZE);
      const auto& max_block_per_cu = compiler->get_max_block_per_cu();

      hw_info.sm_count = hw_info.sm_count * max_block_per_cu;
      grid = get_grid_shape(hw_info.sm_count);
      args.launch_args.grid_dim = grid;

      BF16GemmCutlass3Runtime::launch(runtime, args, stream);
      
    } else {

      // typename EpilogueOp::Params epilogue_op(
      //       ElementAccumulator(1.f), ElementAccumulator(0.f));

      const int threadblock_count = num_sms < 20 ? num_sms : num_sms;// * max_active_tb_num;

      // const auto gemm_args = BF16GemmRuntime::arguments {
      //   .output_op = epilogue_op,
      //   .ref_A = input_a,
      //   .ref_B = input_b,
      //   .ref_D = output,
      //   .gemm_m = m,
      //   .gemm_n = n,
      //   .gemm_k = k,
      //   .grouped_layout = layout_info,
      //   .problem_count = kNumGroups,
      //   .threadblock_count = threadblock_count,
      //   .signal = nullptr
      // }

      BF16GemmRuntime::GemmKernelParams params = BF16GemmRuntime::GemmKernelParams {
        .problem_visitor = {layout_info, n, k, m},
        .threadblock_count = threadblock_count,
        .problem_count = kNumGroups,
        .output_op = {float(1), float(0)},
        .ptr_A = input_a,
        .params_A = lhs.layout(),
        .ptr_B = input_b,
        .params_B = rhs.layout(),
        .ptr_D = output,
        .params_D = out.layout(),
        .signal = nullptr
      };

      auto args = BF16GemmRuntime::Args{
        // .gemm_args = gemm_args,
        .launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages, "ppu10000_bf16_gemm"},
        .launch_args = LaunchArgs(grid, block, SMSIZE),
        .kernel_params = params
      };

      const auto& code = BF16GemmRuntime::generate(args);
      const auto& runtime = compiler->build("ppu10000_bf16_gemm", code, block.x, SMSIZE);
      const auto& max_block_per_cu = compiler->get_max_block_per_cu();

      std::cout << "grid: " << grid.x << "block: " << block.x << std::endl;
      hw_info.sm_count = hw_info.sm_count * max_block_per_cu;
      grid = get_grid_shape(hw_info.sm_count);
      args.launch_args.grid_dim = grid;

      BF16GemmRuntime::launch(runtime, args, stream);
    }
}
} // namespace deep_gemm
