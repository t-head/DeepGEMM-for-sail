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
#include "../heuristics/common_int8.hpp"
#include "../heuristics/predicated_tile_iterator_params.hpp"
#include "../../../deep_gemm/include/deep_gemm/scheduler_cutlass3.cuh"
#include "../../../deep_gemm/include/deep_gemm/densegemm_scheduler_cutlass3.cuh"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "../../../deep_gemm/include/deep_gemm/utils_rtc.cuh"
#include "../../../deep_gemm/include/deep_gemm/profiling_interface.hpp"

using namespace deep_gemm_int8;
namespace deep_gemm {

class DenseINT8GemmCutlass3Runtime final : public LaunchRuntime<DenseINT8GemmCutlass3Runtime> {
public:
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
    using GemmProblemSize = cute::tuple<int32_t, int32_t, int32_t, int32_t>;

    struct MainLoopArguments {
        const void* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        const void* ptr_B;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
        float const* ptr_scale_A;
        float const* ptr_scale_B;
    };

    struct LinearCombinationArgs {
        float alpha = 1.0f;
        float beta = 0.0f;
        float const* alpha_ptr = nullptr;
        float const* beta_ptr = nullptr;
        float const* const* alpha_ptr_array = nullptr;
        float const* const* beta_ptr_array = nullptr;
        float scale_a = float(1);
        float scale_b = float(1);
        float scale_c = float(1);
        float scale_d = float(1);
        float const* scale_a_ptr = nullptr;
        float const* scale_b_ptr = nullptr;
        float const* scale_c_ptr = nullptr;
        float const* scale_d_ptr = nullptr;
    };

    struct EpilogueArgs {
        LinearCombinationArgs callback;
        cutlass::bfloat16_t* ptr_C;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;

        cutlass::bfloat16_t* ptr_D;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n;
        int warp_k;           // WARP_K tile size for K-dim split (= block_k / WarpOnK). Used in WarpShape as Int<WARP_K>.
        bool kDenseS2Opt;
        int num_stages;
        std::string gemm_type, kKernelType, kernel_name;
        bool kEnableSboOverlap;
    };

    struct GemmArguments {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        MainLoopArguments mainloopargs;
        EpilogueArgs epilogueargs;
        cutlass::KernelHardwareInfo hw_info;
        DenseGemmTileSchedulerArguments scheduler{};
        // Carried for API uniformity with bf16/fp8 GemmArguments and the grouped INT8 paths.
        // The dense path leaves it null; to_underlying_arguments_rtc ignores it (GemmKernelParams has no signal field).
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
        DenseGemmTileSchedulerArguments scheduler;
        void* workspace{nullptr}; // workspace,
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
        std::string type_info;
    };

    static GemmKernelParams to_underlying_arguments_rtc(GemmArguments args, void* workspace) {
        auto problem_shape = args.problem_shape;
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);

        int sm_count = args.hw_info.cu_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST(
                "  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

        return {args.mode, problem_shape,  args.mainloopargs, args.epilogueargs,
                hw_info,   args.scheduler, workspace};
    }

    static std::string generate_impl(const Args& args) {
        // IsAlignedN is derived from the runtime problem N and block_n (n % block_n == 0),
        // aligned with DenseBF16GemmCutlass3Runtime::generate_impl. The int8 dense epilogue
        // (DefaultEpilogueNoTsm, ElementD = bfloat16_t) predicates the N boundary on block_n,
        // so the same expression applies with correct int8 semantics.
        const int shape_n = cute::get<1>(args.kernel_params.problem_shape);
        const bool is_aligned_n = (shape_n % args.launch_info.block_n == 0);
        return fmt::format(R"(
#define INT8_HGRTC
#include <int8_densegemm_cutlass3.cuh>
namespace deep_gemm {{
using namespace cute;
using cutlass::KernelHardwareInfo;

using ElementAB = {};
using ElementAccumulator = cute::conditional_t<
    cute::is_same_v<ElementAB, int8_t>,
    int32_t,
    float
>;
constexpr int BLOCK_M = {};
constexpr int BLOCK_N = {};
constexpr int BLOCK_K = {};
constexpr int WARP_M = {};
constexpr int WARP_N = {};
constexpr int WARP_K = {};
constexpr int STAGES = {};
constexpr bool kDenseS2Opt = {};

using ArchTag = cutlass::arch::PPU0015;
using ElementA = ElementAB;
using ElementB = ElementAB;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementD = cutlass::bfloat16_t;
using LayoutD = cutlass::layout::RowMajor;
using ElementC = ElementD;
using LayoutC = LayoutD;
using ElementCompute = float;

using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<WARP_K>>;
static constexpr int WarpOnM = BLOCK_M / WARP_M;
static constexpr int WarpOnN = BLOCK_N / WARP_N;
static constexpr int WarpOnK = BLOCK_K / WARP_K;

using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementA, ElementB, ElementAccumulator>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, Int<WarpOnK>>>,
    Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, Int<WarpOnK * 32>>>;

using KernelSchedule = cutlass::gemm::KernelAiuMultistage;
using DispatchPolicy = cutlass::gemm::MainloopPPUAiuA8W8<STAGES, KernelSchedule, kDenseS2Opt>;

static constexpr bool TransA = false;
static constexpr bool TransB = false;
static constexpr int SmemLayoutStageStrideA = BLOCK_M * BLOCK_K;
static constexpr int SmemLayoutStageStrideB = BLOCK_N * BLOCK_K;
using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;

using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    ArchTag, DispatchPolicy, TileShape,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;

static constexpr bool IsAlignedN = {};
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::EpilogueDefault,
    IsAlignedN>;

using TileScheduler = DenseGemmScheduler<BLOCK_M, BLOCK_N>;
using GemmKernel = cutlass::gemm::kernel::DenseGemmKernel<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler>;

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
                       args.type_info,
                       args.launch_info.block_m,
                       args.launch_info.block_n,
                       args.launch_info.block_k,
                       args.launch_info.warp_m,
                       args.launch_info.warp_n,
                       args.launch_info.warp_k,
                       args.launch_info.num_stages,
                       args.launch_info.kDenseS2Opt,
                       is_aligned_n ? "true" : "false",
                       args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& configs, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, configs, args.kernel_params));
    }
};

// MoE grouped-path runtime restored to baseline 753f28f form (no warp_k / dense_s2_opt).
// Used by m_grouped_int8_gemm.hpp; generates DeepGemmUniversal + DeepGemmScheduler kernels.
class INT8GemmCutlass3Runtime final : public LaunchRuntime<INT8GemmCutlass3Runtime> {
public:
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
    using GemmProblemSize = cute::tuple<int32_t, int32_t, int32_t, int32_t>;

    struct MainLoopArguments {
        const void* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        const void* ptr_B;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
        float const* ptr_scale_A;
        float const* ptr_scale_B;
    };

    struct LinearCombinationArgs {
        float alpha = 1.0f;
        float beta = 0.0f;
        float const* alpha_ptr = nullptr;
        float const* beta_ptr = nullptr;
        float const* const* alpha_ptr_array = nullptr;
        float const* const* beta_ptr_array = nullptr;
        float scale_a = float(1);
        float scale_b = float(1);
        float scale_c = float(1);
        float scale_d = float(1);
        float const* scale_a_ptr = nullptr;
        float const* scale_b_ptr = nullptr;
        float const* scale_c_ptr = nullptr;
        float const* scale_d_ptr = nullptr;
    };

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
        std::string type_info;
    };

    static GemmKernelParams to_underlying_arguments_rtc(GemmArguments args, void* workspace) {
        auto problem_shape = args.problem_shape;
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);

        int sm_count = args.hw_info.cu_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST(
                "  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

        return {args.mode, problem_shape,  args.mainloopargs, args.epilogueargs,
                hw_info,   args.scheduler, workspace,         args.signal};
    }

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define INT8_HGRTC
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

using ArchTag = cutlass::arch::PPU0015;

using         ElementA    = {};
using         LayoutA     = cutlass::layout::RowMajor;

using         ElementB    = ElementA;
using         LayoutB     = cutlass::layout::ColumnMajor;

using         ElementD    = cutlass::bfloat16_t;
using         LayoutD     = cutlass::layout::RowMajor;

using         ElementC    = ElementD;
using         LayoutC     = LayoutD;

using ElementAccumulator  = cute::conditional_t<
    cute::is_same_v<ElementA, int8_t>,
    int32_t,
    float
>;
using ElementCompute      = float;
using ElementScalar    = ElementCompute;

using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
static constexpr int WarpOnM = BLOCK_M / WARP_M;
static constexpr int WarpOnN = BLOCK_N / WARP_N;

constexpr bool kEnableSboOverlap = {};
constexpr KernelType kKernelType = KernelType::{};

using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementA,ElementB,ElementAccumulator>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
    Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _32>>;       // 1x1x1 value group
constexpr int EnableMultistageOnN = kKernelType == KernelType::MultistageOnN
                                    && (SHAPE_N % (BLOCK_N) == 0)
                                    && (SHAPE_K > (BLOCK_K * STAGES));
static constexpr int N_EXPAND = EnableMultistageOnN ? KernelAiuMultistageOnN::N_EXPAND : 1;
constexpr int EnableOverlapPrologue = kKernelType == KernelType::OverlapPrologue;
                                          // && (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(int8_t) >= WarpOnM * 16 * BLOCK_N * sizeof(int32_t); // to impl

using KernelSchedule = cute::conditional_t<
    EnableOverlapPrologue,
    KernelAiuMultistageOverlapPrologue,
    cute::conditional_t<
      EnableMultistageOnN,
      KernelAiuMultistageOnN,
      cutlass::gemm::KernelAiuMultistage>>;
using DispatchPolicy = cute::conditional_t<
    EnableOverlapPrologue,
    cutlass::gemm::MainloopPPUAiuA8W8OverlapPrologue<STAGES, KernelSchedule>,
    cutlass::gemm::MainloopPPUAiuA8W8<STAGES, KernelSchedule>>;
static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
static constexpr int TSM_LD_NUM = BLOCK_M == 8 ? 2 : 4;
static constexpr int SmemLayoutStageStrideA = EnableOverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_M * BLOCK_K;
static constexpr int SmemLayoutStageStrideB = EnableOverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_N * BLOCK_K;
using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;

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
    ArchTag,
    DispatchPolicy, TileShape,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
>;
// Epilogue
static constexpr bool IsAlignedN = SHAPE_N % BLOCK_N == 0 ? true : false;
// reduce vreg to use ScaleType::Nothing for alpha=1 & beta=0
using CollectiveEpilogue_noTsm = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::EpilogueDefault,
    IsAlignedN>;
static constexpr int AlignmentC = 16 / sizeof(ElementC);
using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute>;
using EpilogueSchedule = typename cutlass::epilogue::EpilogueSimtVectorized;
using CollectiveEpilogue_withTsm = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
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

static constexpr GemmType kGemmType = GemmType::{};

using TileScheduler = DeepGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, NUM_GROUPS>;
using GemmKernel = cutlass::gemm::kernel::DeepGemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler,
    kEnableSboOverlap>;

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
                           cute::get<1>(args.kernel_params.problem_shape),
                           cute::get<2>(args.kernel_params.problem_shape), args.launch_info.block_m,
                           args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_groups,
                           args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.num_stages,
                           args.type_info, args.launch_info.kEnableSboOverlap, args.launch_info.kKernelType,
                           args.launch_info.gemm_type, args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& configs, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, configs, args.kernel_params));
    }
};

class INT8GemmRuntime final : public LaunchRuntime<INT8GemmRuntime> {
public:
    struct LinearCombinationArgs {
        float alpha = 1.0;                ///< scales accumulators
        float beta = 0.0;                 ///< scales source tensor
        float const* alpha_ptr = nullptr; ///< pointer to accumulator scalar - if not null, loads it from memory
        float const* beta_ptr = nullptr;  ///< pointer to source scalar - if not null, loads it from memory
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n, num_groups, num_stages, shape_n, shape_k;
        std::string gemm_type, kernel_name;
        bool kEnableSboOverlap;
    };

    struct ProblemVisitorParams {
        int const* grouped_layout;
        int64_t gemm_n;
        int64_t gemm_k;
        int64_t gemm_m;
        int32_t problem_count;
    };

    struct EpilogueVisitorParams {
        LinearCombinationArgs linearargs;
        int64_t batch_stride_alpha = 0;
        int64_t batch_stride_C = 0;
        int64_t batch_stride_D = 0;
    };

    struct PredicatedTileIteratorParams {
        int64_t stride = 0; ///< stride in bytes between rows

        int64_t increment_row = 0;     ///< increment quantity (in bytes) to advance when moving between rows
        int64_t increment_group = 0;   ///< increment quantity (in bytes) to advance when moving to the next group
        int64_t increment_cluster = 0; ///< increment quantity (in bytes) to advance when moving to the next cluster

        int64_t advance_row = 0;     ///< amount to add to move to the next 'row' position
        int64_t advance_group = 0;   ///< amount to add to move to the next 'group' position
        int64_t advance_cluster = 0; ///< amount to add to move to the next 'cluster' position
        int64_t advance_tile = 0;    ///< amount to add to move to the next 'tile'
    };

    struct GemmKernelParams {
        ProblemVisitorParams problem_visitor;
        int threadblock_count;
        int problem_count;

        const void* ptr_A;
        cutlass::layout::RowMajor params_A;
        const void* ptr_B;
        cutlass::layout::ColumnMajor params_B;
        cutlass::bfloat16_t const* ptr_D;
        PredicatedTileIteratorParams params_D;
        float* ptr_alpha_col;
        float* ptr_alpha_row;
        PredicatedTileIteratorParams params_alpha_col;
        PredicatedTileIteratorParams params_alpha_row;

        int64_t batch_stride_A;
        int64_t batch_stride_B;

        EpilogueVisitorParams epilogue_visitor_params;

        int32_t* signal;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define FP8_HGRTC
#include <int8_gemm.cuh>
namespace deep_gemm {{

constexpr int SHAPE_N = {};
constexpr int SHAPE_K = {};
constexpr int BLOCK_M = {};
constexpr int BLOCK_N = {};
constexpr int BLOCK_K = {};
constexpr int NUM_GROUPS = {};
constexpr int WARP_M = {};
constexpr int WARP_N = {};
constexpr int STAGES = {};
constexpr bool kEnableSboOverlap = {};
static constexpr GemmType kGemmType = GemmType::{};

using ThreadblockShape = cutlass::gemm::GemmShape<BLOCK_M, BLOCK_N, BLOCK_K>;
using WarpShape = cutlass::gemm::GemmShape<WARP_M, WARP_N, BLOCK_K>;
using ElementType = int8_t;

using OperatorClass = cutlass::arch::OpClassTensorOp;
using ElementAccumulator = int32_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementCompute = float;

static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<ElementType>::value;
static constexpr int LimitedPerAccessC_ = ((ThreadblockShape::kN) * 8 / (ThreadblockShape::kN / WarpShape::kN) / 32);
static constexpr int ElementsPerAccessC = LimitedPerAccessC_ < ElementsPerAccess ? LimitedPerAccessC_ : ElementsPerAccess;
static constexpr int ThreadblockK = ThreadblockShape::kK;
static constexpr int ScalePerAccess = 128 / cutlass::sizeof_bits<ElementCompute>::value;

using InstructionShape = cutlass::gemm::GemmShape<16, 16, 32>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, ElementsPerAccessC, ElementAccumulator, ElementCompute, \
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling, cutlass::FloatRoundStyle::round_to_nearest>;

using DefaultGemm = typename aiu::gemm::kernel::DefaultGemmGrouped<ElementType, cutlass::layout::RowMajor, ElementsPerAccess,
                                                                    ElementType, cutlass::layout::ColumnMajor, ElementsPerAccess,
                                                                    ElementType, cutlass::layout::RowMajor, ElementAccumulator,
                                                                    cutlass::arch::OpClassTensorOp, cutlass::arch::PPU0010, ThreadblockShape, WarpShape,
                                                                    InstructionShape, EpilogueOp,
                                                                    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, STAGES,
                                                                    cutlass::gemm::kernel::GroupScheduleMode::kDeepGemm,
                                                                    cutlass::arch::OpMultiplyAdd>::GemmKernel;

using ProblemVisitor = Scheduler<kGemmType, SHAPE_N, ThreadblockShape, NUM_GROUPS>;

using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
        typename DefaultGemm::Epilogue::OutputTileIterator::ThreadMap::Shape,
        typename DefaultGemm::Epilogue::OutputTileIterator::ThreadMap::Count,
        DefaultGemm::Epilogue::OutputTileIterator::ThreadMap::kThreads,
        DefaultGemm::Epilogue::OutputTileIterator::kElementsPerAccess, cutlass::sizeof_bits<ElementOutput>::value>,
    ElementCompute>;

// Epilogue
using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<ThreadblockShape,
    DefaultGemm::kThreadCount, AlphaColTileIterator, typename DefaultGemm::Epilogue::OutputTileIterator,
    ElementAccumulator, ElementCompute, EpilogueOp>;

/// Epilogue
using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor,
    typename DefaultGemm::Epilogue>::Epilogue;
// GEMM
using Gemm_Kernel = GemmKernel<typename DefaultGemm::Mma, Epilogue, ProblemVisitor, kEnableSboOverlap>;

extern "C"
__launch_bounds__(DefaultGemm::kThreadCount)
__global__ void {}(
  typename Gemm_Kernel::Params params
) {{
  extern __shared__ char smem[];
  using SharedStorage = typename Gemm_Kernel::SharedStorage;
  int* grouped_layout = nullptr;
  Gemm_Kernel op;
  op(params, *reinterpret_cast<SharedStorage*>(smem));
}}
}}
)",
                           args.launch_info.shape_n, args.launch_info.shape_k, args.launch_info.block_m,
                           args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_groups,
                           args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.num_stages,
                           args.launch_info.kEnableSboOverlap, args.launch_info.gemm_type,
                           args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& configs, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, configs, args.kernel_params));
    }
};

// 8-element public boundary ConfigTuple (matches Python/vLLM stable contract).
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
// 10-element internal working config: adds warp_k and dense_s2_opt (never exposed via pybind).
using WorkConfigTuple = std::tuple<int, int, int, int, int, int, int, int, bool, std::tuple<int, int, int>>;
static void gemm_a8w8_per_channel_nt(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                                     const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                                     const torch::Tensor& out, const int& m, const int& n, const int& k,
                                     std::optional<ConfigTuple> configs = std::nullopt,
                                     hggcStream_t stream = (hggcStream_t)0) {
    TORCH_CHECK(rhs_scales.is_contiguous(), "rhs_scales must be contiguous");
    if (m == 0) {
        return;
    }
    int num_sms = get_num_sms();

    WorkConfigTuple selected_config;
    if (configs.has_value()) {
        // Explicit config path: unpack the 8-element public tuple; force warp_k = block_k, dense_s2_opt = false
        // (mirrors deep_gemm/jit_kernels/gemm.py lines 406-408/422; adaptive is NOT re-evaluated here).
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, /*warp_k=*/bk, nst, /*dense_s2_opt=*/false,
            deep_gemm_int8::get_smem_config(nst, k, bm, bn, bk, 1));
    } else {
        // Heuristic path: get_best_configs now returns the baseline 8-tuple; adaptive warp_k/dense_s2_opt
        // injection is done here (dense-only), mirroring deep_gemm/jit_kernels/gemm.py.
        auto [ns, bm, bn, bk, wm, wn, nst, sc] = deep_gemm_int8::get_best_configs(m, n, k, 1, num_sms);
        int warp_k = bk;  // default: WarpOnK=1
        bool dense_s2_opt = false;
        if (is_ppu1v5_device() && (deep_gemm_adaptive::is_int8_adaptive_shape(m, n, k) || deep_gemm_adaptive::int8_adaptive_enabled())) {
            dense_s2_opt = true;
            // warp_k remains = block_k (WarpOnK=1 for INT8; adaptive warp_k TODO)
        }
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, warp_k, nst, dense_s2_opt, sc);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, dense_s2_opt, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);
    // std::cout << "num_sms_new is " << num_sms_new << " block_m is " << block_m << " block_n is " << block_n << "
    // block_k is " << block_k << std::endl; std::cout << " warp_m is " << warp_m << " warp_n is " << warp_n << "
    // num_stages is " << num_stages << " SMSIZE is " << SMSIZE << std::endl;
    uint32_t kNumGroups = 1;
    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    static constexpr bool kEnableMultistageOnN = false;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    int* grouped_layout = nullptr;
    int* block_m_info = nullptr;
    int* layout_info = grouped_layout;

    static constexpr GemmType kGemmType = GemmType::DenseGemm;

    float* scales_a_ptr = lhs_scales.data_ptr<float>();
    float* scales_b_ptr = rhs_scales.data_ptr<float>();
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);
    auto extra_info = get_extra_info();
    torch::Dtype dtype = lhs.dtype().toScalarType();

    const void* converted_input_a = nullptr;
    const void* converted_input_b = nullptr;
    cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    std::string type_info;
    std::string kernel_name;
    std::string profile_type;

    if (dtype == torch::kInt8) {
        converted_input_a = lhs.data_ptr<int8_t>();
        converted_input_b = rhs.data_ptr<int8_t>();
        type_info = "int8_t";
        kernel_name = "int8_deep_gemm";
        profile_type = "int8";
    } else {
        converted_input_a = reinterpret_cast<const void*>(lhs.data_ptr<at::Float8_e4m3fn>());
        converted_input_b = reinterpret_cast<const void*>(rhs.data_ptr<at::Float8_e4m3fn>());
        type_info = "cutlass::float_e4m3_t";
        kernel_name = "fp8_deep_gemm";
        profile_type = "fp8";
    }

    if (extra_info["use_cutlass3"]) {
        // Dense-specific JIT cache key (distinct from MoE grouped kernels) to avoid cache collision.
        std::string dense_kernel_name = (dtype == torch::kInt8) ? "int8_dense_gemm" : "fp8_dense_gemm";
        const auto gemm_args = DenseINT8GemmCutlass3Runtime::GemmArguments{
            .mode = cutlass::gemm::GemmUniversalMode::kGemm,
            .problem_shape = {m, n, k, 1},
            .mainloopargs = {converted_input_a, stride_A, converted_input_b, stride_B, scales_a_ptr, scales_b_ptr},
            .epilogueargs =
                {
                    {1, 0},
                    nullptr,
                    stride_D,
                    converted_output,
                    stride_D,
                },
            .hw_info = hw_info,
            .scheduler = {(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr},
        };

        DenseINT8GemmCutlass3Runtime::GemmKernelParams params =
            DenseINT8GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

        auto args = DenseINT8GemmCutlass3Runtime::Args{
            .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, dense_s2_opt, num_stages, "DenseGemm", "Default",
                            dense_kernel_name, false},
            .launch_args = {grid, block, SMSIZE},
            .kernel_params = params,
            .type_info = type_info,
        };
        const auto& code = DenseINT8GemmCutlass3Runtime::generate(args);
        const auto& runtime = compiler->build(dense_kernel_name, code, block.x, SMSIZE);
        const auto& kernel = runtime->kernel;
        int blocks_per_cu = 0;
        HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
        args.launch_args.grid_dim.x *= blocks_per_cu;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(kGemmType, false, profile_type, kNumGroups, m, n, k, 0, grouped_layout, stream);
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        DenseINT8GemmCutlass3Runtime::launch(runtime, args);

        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            int numRegs = 0, localSize = 0;
            hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
            hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

            printf("[DenseGemm_INT8:]\n");
            printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
                   args.launch_args.grid_dim.x);
            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n, block_k,
                   warp_m, warp_n, block_k, num_stages);
            printf("SMSIZE:%d, vreg:%d, stack:%d\n",int(SMSIZE), int(numRegs), int(localSize));
        }

    } else {
        int64_t stride, increment_row, increment_group, increment_cluster;
        int64_t advance_row, advance_group, advance_cluster, advance_tile;
        using ElementType = int8_t;
        int ElementsPerAccess = 128 / cutlass::sizeof_bits<ElementType>::value;
        int LimitedPerAccessC_ = (block_n * 8 / (block_n / warp_n) / 32);
        int ElementsPerAccessC = LimitedPerAccessC_ < ElementsPerAccess ? LimitedPerAccessC_ : ElementsPerAccess;
        deep_gemm::compute_predicated_tile_iterator_params(block_m, block_n, block_k, // block_m, block_n, block_k
                                                           warp_m, warp_n, block_k,   // warp_m, warp_n, warp_k
                                                           ElementsPerAccessC,        // elements_per_access
                                                           16,                        // element_size_bits (8 for int8)
                                                           n,                         // shape_n (runtime value)
                                                           &stride, &increment_row, &increment_group,
                                                           &increment_cluster, &advance_row, &advance_group,
                                                           &advance_cluster, &advance_tile);
        // printf("block_m is %ld, block_n is %ld, block_k is %ld, warp_m is %ld, warp_n is %ld, ElementsPerAccessC is
        // %ld, n is %ld", block_m, block_n, block_k, warp_m, warp_n, ElementsPerAccessC, n); printf("stride is %ld,
        // increment_row is %ld, increment_group is %ld, increment_cluster is %ld, advance_row is %ld, advance_group is
        // %ld, advance_cluster is %ld, advance_tile is %ld", stride, increment_row, increment_group, increment_cluster,
        // advance_row, advance_group, advance_cluster, advance_tile);

        INT8GemmRuntime::EpilogueVisitorParams temp;
        const int threadblock_count = num_sms;
        INT8GemmRuntime::GemmKernelParams params = INT8GemmRuntime::GemmKernelParams{
            .problem_visitor = {layout_info, n, k, m, kNumGroups},
            .threadblock_count = threadblock_count,
            .problem_count = kNumGroups,
            .ptr_A = converted_input_a,
            .params_A = cutlass::layout::RowMajor(k),
            .ptr_B = converted_input_b,
            .params_B = cutlass::layout::ColumnMajor(k),
            .ptr_D = converted_output,
            .params_D = {stride, increment_row, increment_group, increment_cluster, advance_row, advance_group,
                         advance_cluster, advance_tile},
            .ptr_alpha_col = scales_b_ptr,
            .ptr_alpha_row = scales_a_ptr,
            .params_alpha_col = {0, 0, 0, 0, 0, 0, 0, 0},
            .params_alpha_row = {0, 0, 0, 0, 0, 0, 0, 0},
            .batch_stride_A = 0,
            .batch_stride_B = 0,
            .epilogue_visitor_params = temp,
            .signal = nullptr,
        };
        auto args =
            INT8GemmRuntime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                         num_stages, n, k, "DenseGemm", "int8_deep_gemm", false},
                                         .launch_args = {grid, block, SMSIZE},
                                         .kernel_params = params};
        const auto& code = INT8GemmRuntime::generate(args);
        const auto& runtime = compiler->build("int8_deep_gemm", code, block.x, SMSIZE);
        const auto& kernel = runtime->kernel;
        int blocks_per_cu = 0;
        HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
        args.launch_args.grid_dim.x *= blocks_per_cu;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(kGemmType, false, std::string("int8"), kNumGroups, m, n, k, 0, grouped_layout,
                                      (hggcStream_t)0);
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        INT8GemmRuntime::launch(runtime, args);

        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            int numRegs = 0, localSize = 0;
            hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
            hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

            printf("[DenseGemm_INT8:]\n");
            printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
                   args.launch_args.grid_dim.x);
            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n, block_k,
                   warp_m, warp_n, block_k, num_stages);
            printf("SMSIZE:%d, vreg:%d, stack:%d\n",int(SMSIZE), int(numRegs), int(localSize));
        }
    }
}

static void int8_gemm(const torch::Tensor& lhs, const torch::Tensor& lhs_scales, const torch::Tensor& rhs,
                      const torch::Tensor& rhs_scales, const torch::Tensor& out, const int& m, const int& n,
                      const int& k, std::optional<ConfigTuple> configs = std::nullopt) {
    gemm_a8w8_per_channel_nt(lhs, lhs_scales, rhs, rhs_scales, out, m, n, k, configs);
}
} // namespace deep_gemm
