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
#include "../heuristics/common_bf16.hpp"
// #include "../heuristics/gemm_search_space.hpp"
#include "../../../deep_gemm/include/deep_gemm/scheduler_cutlass3.cuh"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"

using namespace deep_gemm_bf16_common;
namespace deep_gemm {

class BF16GemmCutlass3Runtime final : public LaunchRuntime<BF16GemmCutlass3Runtime> {
public:
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
    using GemmProblemSize = cute::tuple<int32_t, int32_t, int32_t, int32_t>;

    struct MainLoopArguments {
        cutlass::bfloat16_t const* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        cutlass::bfloat16_t const* ptr_B;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
    };

    struct LinearCombinationArgs {
        float alpha = 1.0f;               ///< scales accumulators
        float beta = 0.0f;                ///< scales source tensor
        float const* alpha_ptr = nullptr; ///< pointer to accumulator scalar - if not null, loads it from memory
        float const* beta_ptr = nullptr;  ///< pointer to source scalar - if not null, loads it from memory
        float const* const* alpha_ptr_array = nullptr; ///< array of pointers to accumulator scalar per group/batch
        float const* const* beta_ptr_array = nullptr;  ///< array of pointers to source scalar per group/batch
        float scale_a = float(1);
        float scale_b = float(1);
        float scale_c = float(1);
        float scale_d = float(1);
        float const* scale_a_ptr = nullptr;
        float const* scale_b_ptr = nullptr;
        float const* scale_c_ptr = nullptr;
        float const* scale_d_ptr = nullptr;
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
    using CollectiveMainloopParams = MainLoopArguments;

    struct GemmKernelParams {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        CollectiveMainloopParams collective_mainloop_params;
        CollectiveEpilogueParams collective_epilogue_params;
        cutlass::KernelHardwareInfo hw_info;
        TileSchedulerArguments scheduler{};
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

        return {args.mode, problem_shape,  args.mainloopargs, args.epilogueargs,
                hw_info,   args.scheduler, workspace,         args.signal};
    }

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#define BF16_HGRTC
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

static constexpr GemmType kGemmType = GemmType::{};
static constexpr KernelType kKernelType = KernelType::{}; //Default;
static constexpr bool kEnableSboOverlap = {};// false;

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
using ArchTag = cutlass::arch::PPU0015;

using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
static constexpr int WarpOnM = BLOCK_M / WARP_M;
static constexpr int WarpOnN = BLOCK_N / WARP_N;

using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, cutlass::bfloat16_t, cutlass::bfloat16_t, float>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
    Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _16>>;       // 1x1x1 value group

static constexpr int N_EXPAND = kKernelType == KernelType::MultistageOnN && (SHAPE_N % (BLOCK_N) == 0) ? KernelAiuMultistageOnN::N_EXPAND : 1;

using KernelSchedule = cute::conditional_t<
    kKernelType == KernelType::OverlapMainloop,
    KernelAiuMultistageOverlapMainloop,
    cute::conditional_t<
      kKernelType == KernelType::OverlapPrologue,
      KernelAiuMultistageOverlapPrologue,
      cute::conditional_t<
        kKernelType == KernelType::MultistageOnN,
        KernelAiuMultistageOnN,
        cutlass::gemm::KernelAiuMultistage>>>;
using DispatchPolicy = cute::conditional_t<
    kKernelType == KernelType::OverlapMainloop,
    cutlass::gemm::MainloopPPUOverlapMainloop<STAGES, KernelSchedule>,
    cute::conditional_t<
      kKernelType == KernelType::OverlapPrologue,
      cutlass::gemm::MainloopPPUOverlapPrologue<STAGES, KernelSchedule>,
      cutlass::gemm::MainloopPPUAiuOpt<STAGES, KernelSchedule>>>;

static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
static constexpr int TSM_LD_NUM = BLOCK_M == 8 ? 2 : 4;

static constexpr int SmemLayoutStageStrideA = kKernelType == KernelType::OverlapMainloop || kKernelType == KernelType::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_M * BLOCK_K;
static constexpr int SmemLayoutStageStrideB = kKernelType == KernelType::OverlapMainloop || kKernelType == KernelType::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_N * BLOCK_K;
using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;
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
    ArchTag,
    DispatchPolicy, TileShape,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
>;

// Epilogue
static constexpr bool IsAligedN = SHAPE_N % BLOCK_N == 0 ? true : false;
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
            cute::get<1>(args.kernel_params.problem_shape), cute::get<2>(args.kernel_params.problem_shape),
            args.launch_info.block_m, args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_groups,
            args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.num_stages, args.launch_info.gemm_type,
            args.launch_info.kKernelType, args.launch_info.kEnableSboOverlap, args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

class BF16GemmRuntime final : public LaunchRuntime<BF16GemmRuntime> {
public:
    struct LinearCombinationArgs {
        float alpha;                      ///< scales accumulators
        float beta;                       ///< scales source tensor
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

        LinearCombinationArgs output_op;

        cutlass::bfloat16_t const* ptr_A;
        cutlass::layout::RowMajor params_A;
        cutlass::bfloat16_t const* ptr_B;
        cutlass::layout::ColumnMajor params_B;
        cutlass::bfloat16_t const* ptr_D;
        PredicatedTileIteratorParams params_D;

        int32_t* signal;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#define BF16_HGRTC
#include <bf16_gemm.cuh>
namespace deep_gemm {{
// using namespace cute;
// using cutlass::KernelHardwareInfo;

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
static constexpr bool kEnableSboOverlap = {};// false;

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
                                                                    cutlass::arch::OpClassTensorOp, cutlass::arch::PPU0010, ThreadblockShape, WarpShape,
                                                                    InstructionShape, EpilogueOp,
                                                                    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, STAGES,
                                                                    cutlass::gemm::kernel::GroupScheduleMode::kDeepGemm,
                                                                    cutlass::arch::OpMultiplyAdd>::GemmKernel;

using ProblemVisitor = Scheduler<kGemmType, SHAPE_N, ThreadblockShape, NUM_GROUPS>;
using Gemm_Kernel = GemmKernel<typename DefaultGemm::Mma, typename DefaultGemm::Epilogue, ProblemVisitor, kEnableSboOverlap>;

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
            args.launch_info.shape_n, args.launch_info.shape_k, args.launch_info.block_m, args.launch_info.block_n,
            args.launch_info.block_k, args.launch_info.num_groups, args.launch_info.warp_m, args.launch_info.warp_n,
            args.launch_info.num_stages, args.launch_info.gemm_type, args.launch_info.kEnableSboOverlap,
            args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
static void bf16_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out, const int& m,
                      const int& n, const int& k, std::optional<ConfigTuple> configs = std::nullopt) {
    int num_sms = get_num_sms();
    hggcDeviceProp device_props;
    hggcGetDeviceProperties(&device_props, 0);
    std::vector<int> shape = {m, n, k};
    static constexpr GemmType kGemmType = GemmType::DenseGemm;

    using Config = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

    Config cfg;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else {
        bool shape_large_aligned = true;
        for (int64_t a : shape) {
            if (!(a >= 4096 && (a % 64 == 0))) {
                shape_large_aligned = false;
                break;
            }
        }

        std::string dev_name(device_props.name);
        bool is_ppu0010_device = (dev_name.find("ZW810E") != std::string::npos) || (dev_name.find("ZW810") != std::string::npos);

        if (shape_large_aligned && is_ppu0010_device) {
            cfg = get_gemm_best_configs_v2(shape, 2, num_sms);
        } else {
            cfg = deep_gemm_bf16_common::get_best_configs(m, n, k, 1, num_sms);
        }
    }
    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = cfg;
    auto extra_info = get_extra_info();
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = 1;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    int* grouped_layout = nullptr;
    int* layout_info = grouped_layout;

    cutlass::bfloat16_t* input_b = reinterpret_cast<cutlass::bfloat16_t*>(rhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_a = reinterpret_cast<cutlass::bfloat16_t*>(lhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());

    // TODO get hw info from real env
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);
    bool kEnableSboOverlap = false;
    if (extra_info["use_cutlass3"]) {
        const auto gemm_args = BF16GemmCutlass3Runtime::GemmArguments{
            .mode = cutlass::gemm::GemmUniversalMode::kGemm,
            .problem_shape = {m, n, k, 1},
            .mainloopargs = {input_a, stride_A, input_b, stride_B},
            .epilogueargs =
                {
                    {1, 0},
                    output,
                    stride_D,
                    output,
                    stride_D,
                },
            .hw_info = hw_info,
            .scheduler = {(uint32_t)m, layout_info},
            .signal = nullptr};

        BF16GemmCutlass3Runtime::GemmKernelParams params =
            BF16GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

        auto args = BF16GemmCutlass3Runtime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                                  num_stages, "DenseGemm", "Default", "bf16_deep_gemm",
                                                                  kEnableSboOverlap},
                                                  .launch_args = {grid, block, SMSIZE},
                                                  .kernel_params = params};

        const auto& code = BF16GemmCutlass3Runtime::generate(args);
        const auto& runtime = compiler->build("bf16_deep_gemm", code, block.x, SMSIZE);
        const auto& kernel = runtime->kernel;
        int blocks_per_cu = 0;
        HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
        args.launch_args.grid_dim.x *= blocks_per_cu;
        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, grouped_layout,
                                      (hggcStream_t)0);
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        BF16GemmCutlass3Runtime::launch(runtime, args);

        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            int numRegs = 0, localSize = 0;
            hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
            hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

            printf("[DenseGemm_BF16:]\n");
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
        using ElementType = cutlass::bfloat16_t;

        int ElementsPerAccess = 128 / cutlass::sizeof_bits<ElementType>::value;
        int LimitedPerAccessC_ = (block_n * 8 / (block_n / warp_n) / 32);
        int ElementsPerAccessC = LimitedPerAccessC_ < ElementsPerAccess ? LimitedPerAccessC_ : ElementsPerAccess;

        deep_gemm::compute_predicated_tile_iterator_params(block_m, block_n,
                                                           block_k, // block_m, block_n, block_k
                                                           warp_m, warp_n,
                                                           block_k,            // warp_m, warp_n, warp_k
                                                           ElementsPerAccessC, // elements_per_access
                                                           16,                 // element_size_bits (16 for bfloat16)
                                                           n,                  // shape_n (runtime value)
                                                           &stride, &increment_row, &increment_group,
                                                           &increment_cluster, &advance_row, &advance_group,
                                                           &advance_cluster, &advance_tile);
        const int threadblock_count = num_sms < 20 ? num_sms : num_sms; // * max_active_tb_num;
        // printf("block_m is %ld, block_n is %ld, block_k is %ld, warp_m is %ld, warp_n is %ld, ElementsPerAccessC is
        // %ld, n is %ld", block_m, block_n, block_k, warp_m, warp_n, ElementsPerAccessC, n); printf("stride is %ld,
        // increment_row is %ld, increment_group is %ld, increment_cluster is %ld, advance_row is %ld, advance_group is
        // %ld, advance_cluster is %ld, advance_tile is %ld", stride, increment_row, increment_group, increment_cluster,
        // advance_row, advance_group, advance_cluster, advance_tile);
        BF16GemmRuntime::GemmKernelParams params = BF16GemmRuntime::GemmKernelParams{
            .problem_visitor = {layout_info, n, k, m, kNumGroups},
            .threadblock_count = threadblock_count,
            .problem_count = kNumGroups,
            .output_op = {float(1), float(0)},
            .ptr_A = input_a,
            .params_A = cutlass::layout::RowMajor(k),
            .ptr_B = input_b,
            .params_B = cutlass::layout::ColumnMajor(k),
            .ptr_D = output,
            .params_D = {stride, increment_row, increment_group, increment_cluster, advance_row, advance_group,
                         advance_cluster, advance_tile}, // cutlass::layout::RowMajor(n),
            .signal = nullptr};

        auto args =
            BF16GemmRuntime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages, n,
                                                  k, "DenseGemm", "bf16_deep_gemm", kEnableSboOverlap},
                                  .launch_args = {grid, block, SMSIZE},
                                  .kernel_params = params};

        const auto& code = BF16GemmRuntime::generate(args);
        const auto& runtime = compiler->build("bf16_deep_gemm", code, block.x, SMSIZE);
        const auto& kernel = runtime->kernel;
        int blocks_per_cu = 0;
        HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
        args.launch_args.grid_dim.x *= blocks_per_cu;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, grouped_layout,
                                      (hggcStream_t)0);
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        BF16GemmRuntime::launch(runtime, args);

        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            int numRegs = 0, localSize = 0;
            hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
            hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

            printf("[DenseGemm_BF16:]\n");
            printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
                   args.launch_args.grid_dim.x);
            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n, block_k,
                   warp_m, warp_n, block_k, num_stages);
            printf("SMSIZE:%d, vreg:%d, stack:%d\n",int(SMSIZE), int(numRegs), int(localSize));
        }
    }
}
} // namespace deep_gemm
