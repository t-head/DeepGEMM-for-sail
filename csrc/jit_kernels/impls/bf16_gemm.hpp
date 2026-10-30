#pragma once

#include <torch/python.h>
#include <cstdint>
#include <limits>
#include <type_traits>
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
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/scheduler/densegemm_scheduler_cutlass3.cuh>
#include <deep_gemm/common/gemm_occ_model.cuh>
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"

using namespace deep_gemm_bf16_common;
namespace deep_gemm {

// Output type is part of the generated source (and therefore the JIT cache key).
// Grouped GEMMs retain the BF16 specialization through the alias below.
template <typename ElementOutput>
class BF16GemmCutlass3RuntimeT final : public LaunchRuntime<BF16GemmCutlass3RuntimeT<ElementOutput>> {
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
        ElementOutput* ptr_C;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;

        ElementOutput* ptr_D;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n, num_groups, num_stages;
        std::string gemm_type, kernel_strategy, kernel_name;
        bool enable_sbo_overlap;
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

    // One epilogue-params layout per device generation (must match the device Params bytes).
    using EpilogueParamNoTsm = EpilogueArgs;  // PPU1.5: the args are the params, as-is
    struct EpilogueParamWithTsm {
        struct {} thread{};
        ElementOutput* ptr_C{};
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C{};
        ElementOutput* ptr_D{};
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D{};
    };

    template <typename EpilogueParamsT>
    struct GemmKernelParamsT {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        CollectiveMainloopParams collective_mainloop_params;
        EpilogueParamsT collective_epilogue_params;
        cutlass::KernelHardwareInfo hw_info;
        TileSchedulerArguments scheduler{};
        void* workspace{nullptr}; // workspace,
        int32_t* signal{nullptr};
    };
    // Same prefix for both generations; the member filled by to_underlying_arguments_rtc must match the device compile.
    union KernelParams {
        GemmKernelParamsT<EpilogueParamWithTsm> with_tsm;
        GemmKernelParamsT<EpilogueParamNoTsm> no_tsm;
    };
    using GemmKernelParams = KernelParams;

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

        // Both epilogues only convert the accumulator; enforce the {alpha=1, beta=0} contract.
        DG_HOST_ASSERT(args.epilogueargs.callback.alpha == 1.0f && args.epilogueargs.callback.alpha_ptr == nullptr &&
                       args.epilogueargs.callback.beta == 0.0f && args.epilogueargs.callback.beta_ptr == nullptr);

        // Same decision source as the injected kWithTsmEpilogue; must match the device compile.
        // Initialize the selected union member directly to start its lifetime.
        if (is_ppu1v5_device()) {
            return KernelParams{.no_tsm = GemmKernelParamsT<EpilogueParamNoTsm>{
                args.mode, problem_shape, args.mainloopargs, args.epilogueargs,
                hw_info, args.scheduler, workspace, args.signal}};
        } else {
            // Map the flat inputs on the host (once per launch); the device converts nothing.
            return KernelParams{.with_tsm = GemmKernelParamsT<EpilogueParamWithTsm>{
                args.mode, problem_shape, args.mainloopargs,
                {{},
                 args.epilogueargs.ptr_C, args.epilogueargs.stride_C,
                 args.epilogueargs.ptr_D, args.epilogueargs.stride_D},
                hw_info, args.scheduler, workspace, args.signal}};
        }
    }

    static std::string generate_impl(const Args& args) {
        // Query device hardware constants from the driver and inject into generated kernel (see gemm_occ_model.cuh).
        const PpuHwParams& hw = PpuHwParams::instance();
        // One snapshot drives both the union member read and the injected epilogue flag.
        const bool is_ppu1v5 = is_ppu1v5_device();
        // Read the common head (mode/problem_shape/mainloop) through the matching member.
        const GemmProblemSize problem_shape = is_ppu1v5 ? args.kernel_params.no_tsm.problem_shape
                                                        : args.kernel_params.with_tsm.problem_shape;
        // Injected into the generated source; same source as to_underlying_arguments_rtc.
        const bool with_tsm_epilogue = not is_ppu1v5;
        // sizeof of what the driver copies; the device asserts it equals its own Params size.
        const auto host_params_size = with_tsm_epilogue ? sizeof(args.kernel_params.with_tsm)
                                                        : sizeof(args.kernel_params.no_tsm);
        return fmt::format(
            R"(
#define BF16_HGRTC
#include <deep_gemm/impls/bf16_gemm_cutlass3.cuh>
#include <deep_gemm/common/gemm_occ_model.cuh>
namespace deep_gemm {{
using namespace cute;
using cutlass::KernelHardwareInfo;

// Injected device hardware constants (host-side hggcDeviceGetAttribute query).
constexpr int kHwTsmPerCu         = {13};
constexpr int kHwMaxThreadsPerCta = {14};
constexpr int kHwMaxWarpsPerCu    = {15};
constexpr int kHwTotalVregPerCu   = {16};

constexpr int SHAPE_N = {0};
constexpr int SHAPE_K = {1};
constexpr int BLOCK_M = {2};
constexpr int BLOCK_N = {3};
constexpr int BLOCK_K = {4};
constexpr int NUM_GROUPS = {5};
constexpr int WARP_M = {6};
constexpr int WARP_N = {7};
constexpr int STAGES = {8};

static constexpr GemmType kGemmType = GemmType::{9};
static constexpr KernelStrategy kKernelStrategy = KernelStrategy::{10}; //Default;
static constexpr bool kEnableSboOverlap = {11};// false;

using ElementA    = cutlass::bfloat16_t;
using ElementB    = cutlass::bfloat16_t;
using ElementC    = {19};
using LayoutA     = cutlass::layout::RowMajor;
using LayoutB     = cutlass::layout::ColumnMajor;
using LayoutC     = cutlass::layout::RowMajor;
using ElementD    = ElementC;
using LayoutD     = cutlass::layout::RowMajor;
using ElementCompute      = float;
using ElementScalar       = ElementCompute;
using LinearCombOutType   = ElementD;
using OperatorClass = cutlass::arch::OpClassTensorOp;
#if __HGGC_ARCH__ == 100
using ArchTag = cutlass::arch::PPU0010;
#else
using ArchTag = cutlass::arch::PPU0015;
#endif

using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
static constexpr int WarpOnM = BLOCK_M / WARP_M;
static constexpr int WarpOnN = BLOCK_N / WARP_N;

using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, cutlass::bfloat16_t, cutlass::bfloat16_t, float>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
    Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _16>>;       // 1x1x1 value group

static constexpr int N_EXPAND = kKernelStrategy == KernelStrategy::MultistageOnN && (SHAPE_N % (BLOCK_N) == 0) ? KernelAiuMultistageOnN::N_EXPAND : 1;

using KernelSchedule = cute::conditional_t<
    kKernelStrategy == KernelStrategy::OverlapMainloop,
    KernelAiuMultistageOverlapMainloop,
    cute::conditional_t<
      kKernelStrategy == KernelStrategy::OverlapPrologue,
      KernelAiuMultistageOverlapPrologue,
      cute::conditional_t<
        kKernelStrategy == KernelStrategy::MultistageOnN,
        KernelAiuMultistageOnN,
        cutlass::gemm::KernelAiuMultistage>>>;
using DispatchPolicy = cute::conditional_t<
    kKernelStrategy == KernelStrategy::OverlapMainloop,
    cutlass::gemm::MainloopPPUOverlapMainloop<STAGES, KernelSchedule>,
    cute::conditional_t<
      kKernelStrategy == KernelStrategy::OverlapPrologue,
      cutlass::gemm::MainloopPPUOverlapPrologue<STAGES, KernelSchedule>,
      cutlass::gemm::MainloopPPUAiuOpt<STAGES, KernelSchedule>>>;

static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
static constexpr int TSM_LD_NUM = BLOCK_M == 8 ? 2 : 4;

static constexpr int SmemLayoutStageStrideA = kKernelStrategy == KernelStrategy::OverlapMainloop || kKernelStrategy == KernelStrategy::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_M * BLOCK_K;
static constexpr int SmemLayoutStageStrideB = kKernelStrategy == KernelStrategy::OverlapMainloop || kKernelStrategy == KernelStrategy::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_N * BLOCK_K;
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
static constexpr bool IsAlignedN = SHAPE_N % BLOCK_N == 0 ? true : false;
using CollectiveEpilogue_noTsm = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::EpilogueDefault,
    IsAlignedN>;

// TSM epilogue for alpha=1, beta=0;
static constexpr int AlignmentC = 16 / sizeof(ElementC);
using DefaultOperation = cutlass::epilogue::fusion::PPUEVT<
    cutlass::epilogue::fusion::PPUCompute<cutlass::epilogue::thread::Identity, ElementD, ElementCompute,
                                       cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::PPUAccFetch>;
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

// Injected epilogue choice (is_ppu1v5_device()); matches to_underlying_arguments_rtc.
static constexpr bool kWithTsmEpilogue = {17};
using CollectiveEpilogue = typename cutlass::platform::conditional<
    kWithTsmEpilogue,
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

// Host/device Params ABI check (the driver copies sizeof(device Params) bytes from the host struct).
static_assert(sizeof(GemmKernel::Params) == {18}, "host/device kernel Params size mismatch");

// Derive MinBlocksPerMultiprocessor from GemmOccModel with device hardware
// constants injected by the host-side query (see kHw* above).
using GemmOcc = GemmOccModel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K, GemmKernel::SharedStorageSize,
                            cute::sizeof_bits_v<ElementA>, cute::sizeof_bits_v<ElementB>,
                            cute::sizeof_bits_v<ElementCompute>,
                            kHwTsmPerCu, kHwMaxThreadsPerCta, kHwMaxWarpsPerCu, kHwTotalVregPerCu>;

extern "C"
__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmOcc::kMinBlocksPerMultiprocessor)
__global__ void {12}(
  typename GemmKernel::Params params
) {{
  extern __shared__ char smem[];
  GemmKernel op;
  op(params, smem);
}}
}}
)",
            cute::get<1>(problem_shape), cute::get<2>(problem_shape),
            args.launch_info.block_m, args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_groups,
            args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.num_stages, args.launch_info.gemm_type,
            args.launch_info.kernel_strategy, args.launch_info.enable_sbo_overlap, args.launch_info.kernel_name,
            hw.tsm_per_cu, hw.max_threads_per_cta, hw.max_warps_per_cu, hw.total_vreg_per_cu,
            with_tsm_epilogue, host_params_size,
            std::is_same_v<ElementOutput, float> ? "float" : "cutlass::bfloat16_t");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

using BF16GemmCutlass3Runtime = BF16GemmCutlass3RuntimeT<cutlass::bfloat16_t>;

// Dedicated BF16 DenseGemm runtime (standalone kernel, no SHAPE_N/K template params).
// Architecture-aligned with DenseINT8GemmCutlass3Runtime. The MoE grouped-path runtime
// (BF16GemmCutlass3Runtime) is intentionally left untouched so that m_grouped_bf16_gemm.hpp
// stays byte-identical to baseline 753f28f.
template <typename ElementOutput>
class DenseBF16GemmCutlass3RuntimeT final : public LaunchRuntime<DenseBF16GemmCutlass3RuntimeT<ElementOutput>> {
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
        ElementOutput* ptr_C;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;

        ElementOutput* ptr_D;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n;
        int warp_k;           // WARP_K tile size for K-dim split (= block_k / WarpOnK). Used in WarpShape as Int<WARP_K>.
        bool dense_s2_opt;
        int num_stages;
        std::string gemm_type, kernel_strategy, kernel_name;
        bool enable_sbo_overlap;
        bool overlap_prologue; // next-tile prologue overlap, enabled by host when wave > 3 (acblas parity)
    };

    struct GemmArguments {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        MainLoopArguments mainloopargs;
        EpilogueArgs epilogueargs;
        cutlass::KernelHardwareInfo hw_info;
        DenseGemmTileSchedulerArguments scheduler{};
        // Carried for API uniformity with bf16/fp8 GemmArguments and the grouped BF16 paths.
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
        // Query device hardware constants from the driver and inject into generated kernel (see gemm_occ_model.cuh).
        const PpuHwParams& hw = PpuHwParams::instance();
        // Byte-identical to the original BF16DenseGemmHelper::generate_code output:
        // IsAlignedN is derived from the runtime problem N and block_n (n % block_n == 0).
        const int shape_n = cute::get<1>(args.kernel_params.problem_shape);
        const bool is_aligned_n = (shape_n % args.launch_info.block_n == 0);
        return fmt::format(
            R"(
#define BF16_HGRTC
#include <deep_gemm/impls/bf16_densegemm_cutlass3.cuh>
#include <deep_gemm/common/gemm_occ_model.cuh>
namespace deep_gemm {{
using namespace cute;
using cutlass::KernelHardwareInfo;

// Injected device hardware constants (host-side hggcDeviceGetAttribute query).
constexpr int kHwTsmPerCu         = {11};
constexpr int kHwMaxThreadsPerCta = {12};
constexpr int kHwMaxWarpsPerCu    = {13};
constexpr int kHwTotalVregPerCu   = {14};

constexpr int BLOCK_M = {0};
constexpr int BLOCK_N = {1};
constexpr int BLOCK_K = {2};
constexpr int WARP_M = {3};
constexpr int WARP_N = {4};
constexpr int WARP_K = {5};
constexpr int kNumStages = {6};

using ElementAB = cutlass::bfloat16_t;
using ElementAcc = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ElementC = {15};
using ElementD = ElementC;
using LayoutD = cutlass::layout::RowMajor;
using ElementCompute = float;
using ArchTag = cutlass::arch::PPU0015;

using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using WarpShape_t = Shape<Int<WARP_M>, Int<WARP_N>, Int<WARP_K>>;
static constexpr int WarpOnM = BLOCK_M / WARP_M;
static constexpr int WarpOnN = BLOCK_N / WARP_N;
static constexpr int WarpOnK = BLOCK_K / WARP_K;
static_assert(BLOCK_K % WARP_K == 0, "BLOCK_K must be divisible by WARP_K");

using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementAB, ElementAB, ElementAcc>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, Int<WarpOnK>>>,
    Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, Int<WarpOnK * 16>>>;

using KernelSchedule = cutlass::gemm::KernelAiuMultistage;
using DispatchPolicy = cutlass::gemm::MainloopPPUAiuOpt<kNumStages, KernelSchedule, {7}, {10}>;

static constexpr bool TransA = false;
static constexpr bool TransB = false;

// OverlapPrologue interleaves A/B stages in one contiguous smem region, so the
// TSM swizzle-load atom needs the combined stage stride (validated grouped-path pattern).
static constexpr bool OverlapPrologue = {10};
static constexpr int SmemLayoutStageStrideA = OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_M * BLOCK_K;
static constexpr int SmemLayoutStageStrideB = OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_N * BLOCK_K;
using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementAB, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementAB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;

using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    ArchTag,
    DispatchPolicy, TileShape,
    ElementAB, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementAB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity
>;

static constexpr bool IsAlignedN = {8};
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::EpilogueDefault,
    IsAlignedN>;

using TileScheduler = DenseGemmScheduler<BLOCK_M, BLOCK_N>;
using GemmKernel = cutlass::gemm::kernel::BF16DenseGemmKernel<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler>;

using GemmOcc = GemmOccModel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, GemmKernel::SharedStorageSize,
                            cute::sizeof_bits_v<ElementAB>, cute::sizeof_bits_v<ElementAB>,
                            cute::sizeof_bits_v<ElementCompute>,
                            kHwTsmPerCu, kHwMaxThreadsPerCta, kHwMaxWarpsPerCu, kHwTotalVregPerCu>;

extern "C"
__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmOcc::kMinBlocksPerMultiprocessor)
__global__ void {9}(
  typename GemmKernel::Params params
) {{
  extern __shared__ char smem[];
  GemmKernel op;
  op(params, smem);
}}
}}
)",
            args.launch_info.block_m, args.launch_info.block_n, args.launch_info.block_k,
            args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.warp_k,
            args.launch_info.num_stages,
            args.launch_info.dense_s2_opt ? "true" : "false",
            is_aligned_n ? "true" : "false",
            args.launch_info.kernel_name,
            args.launch_info.overlap_prologue ? "true" : "false",
            hw.tsm_per_cu, hw.max_threads_per_cta, hw.max_warps_per_cu, hw.total_vreg_per_cu,
            std::is_same_v<ElementOutput, float> ? "float" : "cutlass::bfloat16_t");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& configs, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, configs, args.kernel_params));
    }
};

template <typename ElementOutput>
class DenseBF16GemmCuteFreeRuntimeT final : public LaunchRuntime<DenseBF16GemmCuteFreeRuntimeT<ElementOutput>> {
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
        ElementOutput* ptr_C;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;

        ElementOutput* ptr_D;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n;
        int warp_k;           // WARP_K tile size for K-dim split (= block_k / WarpOnK). Used in WarpShape as Int<WARP_K>.
        bool dense_s2_opt;
        int num_stages;
        std::string gemm_type, kernel_strategy, kernel_name;
        bool enable_sbo_overlap;
        bool overlap_prologue;
    };

    struct GemmArguments {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        MainLoopArguments mainloopargs;
        EpilogueArgs epilogueargs;
        cutlass::KernelHardwareInfo hw_info;
        DenseGemmTileSchedulerArguments scheduler{};
        // Carried for API uniformity with bf16/fp8 GemmArguments and the grouped BF16 paths.
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
        const auto& info = args.launch_info;
        const int shape_n = cute::get<1>(args.kernel_params.problem_shape);
        const bool is_aligned_n = (shape_n % info.block_n == 0);

        return fmt::format(
            R"(
#define BF16_HGRTC
#include <deep_gemm/impls/bf16_gemm_cute_free.cuh>

namespace deep_gemm {{

using GemmTypeTag = std::integral_constant<GemmType, GemmType::{0}>;

using Kernel = BF16GemmCuteFreeKernel<{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {11}, {9}, GemmTypeTag{12}>;

}} // namespace deep_gemm

extern "C"
__launch_bounds__(deep_gemm::Kernel::MaxThreadsPerBlock,
                  deep_gemm::Kernel::MinBlocksPerMultiprocessor)
__global__ void {10}(typename deep_gemm::Kernel::Params params) {{
  extern __shared__ char smem[];
  deep_gemm::Kernel kernel;
  kernel(params, smem);
}}
)",
            info.gemm_type,
            info.block_m, info.block_n, info.block_k,
            info.warp_m, info.warp_n, info.warp_k, info.num_stages,
            info.dense_s2_opt ? "true" : "false",
            is_aligned_n ? "true" : "false",
            info.kernel_name,
            info.overlap_prologue ? "true" : "false",
            std::is_same_v<ElementOutput, float> ? ", float" : "");
    }

};

// Dense GEMV (m == 1) JIT runtime. Host-side mirror of the device GemvDenseArgs
// in deep_gemm/impls/gemv_dense.cuh -- the two PODs must stay field-for-field
// identical (same convention as GemvRuntime::GemvtArgs vs gemvt.cuh), since the
// kernel receives it by value.
class DenseGemvRuntime final : public LaunchRuntime<DenseGemvRuntime> {
public:
    struct GemvDenseArgs {
        int N;
        int K;
        const void* x_ptr;   // lhs [1, K]
        const void* w_ptr;   // rhs [N, K] row-major
        void* y_ptr;         // out [1, N]
        int64_t stride_wn;   // W row stride (elements, == K for contiguous rhs)
    };

    struct LaunchInfo {
        int block_x, block_y, k_per_thread;
        int m;   // 1 or 2
        int npt; // W rows per thread (m == 2 policy: 2 shares the x load pair)
        std::string kernel_name;
        bool output_fp32 = false;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemvDenseArgs kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        const bool is_m2 = args.launch_info.m == 2;
        const char* entry = is_m2 ? "gemv_dense_m2_kernel_impl" : "gemv_dense_kernel_impl";
        // m2 entry carries a trailing NPT template arg (1 or 2 W rows/thread); m1 has none.
        const std::string npt_arg = is_m2 ? fmt::format(", {}", args.launch_info.npt) : "";
        return fmt::format(
            R"(
#include <deep_gemm/impls/gemv_dense.cuh>

namespace deep_gemm {{

extern "C" __global__
void {}(const GemvDenseArgs args) {{
    {}<__ppu_bfloat16, {}, float,
       int4, int4, {}, {}, {}{}>(args);
}}

}}
)",
            args.launch_info.kernel_name, entry,
            args.launch_info.output_fp32 ? "float" : "__ppu_bfloat16",
            args.launch_info.block_x, args.launch_info.block_y, args.launch_info.k_per_thread,
            npt_arg);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// bf16 dense GEMV fast path (m == 1 or 2). Returns true when launched; false
// means the shape is uncovered and the caller must fall back to the tile path.
static bool gemv_dense_bf16(const torch::Tensor& lhs, const torch::Tensor& rhs,
                            const torch::Tensor& out, const int& m, const int& n, const int& k) {
    DG_HOST_ASSERT(m == 1 || m == 2);
    TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous(),
                "dense gemv requires contiguous tensors");

    int block_x = 0, block_y = 0, k_per_thread = 0, npt = 1;
    if (!deep_gemm_bf16_common::dense_gemv_select_configs(m, n, k, rhs.data_ptr(), lhs.data_ptr(),
                                                          block_x, block_y, k_per_thread, npt))
        return false;

    DenseGemvRuntime::GemvDenseArgs params;
    params.N = n;
    params.K = k;
    params.x_ptr = lhs.data_ptr<at::BFloat16>();
    params.w_ptr = rhs.data_ptr<at::BFloat16>();
    params.y_ptr = out.data_ptr();
    params.stride_wn = k;

    dim3 block(block_x, block_y);
    dim3 grid(ceil_div(n, block_y * npt));

    // m == 2 -> dual-accumulator variant; npt == 2 walks two W rows/thread sharing
    // (x0, x1), each block owning block_y * npt rows. Names split so a trace shows npt.
    std::string kernel_name = m == 1 ? "gemv_dense_bf16"
                                     : (npt == 2 ? "gemv_dense_m2r2_bf16" : "gemv_dense_m2_bf16");
    const bool output_fp32 = out.scalar_type() == torch::kFloat;
    if (output_fp32) kernel_name += "_fp32";
    auto args = DenseGemvRuntime::Args{
        .launch_info = {block_x, block_y, k_per_thread, m, npt, kernel_name, output_fp32},
        .launch_args = {grid, block, 0},
        .kernel_params = params,
    };
    const auto& code = DenseGemvRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    const auto& kernel = runtime->kernel;

    static constexpr GemmType kGemmType = GemmType::Normal;
    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("bf16"), 1, m, n, k, 0, nullptr,
                                  current_stream());
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);
    DenseGemvRuntime::launch(runtime, args);
    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[DenseGemm_BF16_GemV:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", 1, m, n, k);
        printf("BlockX:%d, BlockY:%d, k_per_thread:%d, npt:%d\n", block_x, block_y, k_per_thread,
               npt);
        printf("threadblock_count:%d, vreg:%d, stack:%d\n", (int)grid.x, int(numRegs), int(localSize));
    }
    return true;
}

// 8-element public boundary ConfigTuple (matches Python/vLLM stable contract).
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
template <typename ElementOutput>
static void bf16_gemm_impl(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out, const int& m,
                           const int& n, const int& k, std::optional<ConfigTuple> configs = std::nullopt) {
    int num_sms = get_num_sms();
    static constexpr GemmType kGemmType = GemmType::Normal;

    using Config = std::tuple<int, int, int, int, int, int, int, int, std::tuple<int, int, int>>;

    Config cfg;
    if (configs.has_value()) {
        // Explicit config path: unpack the 8-element public tuple; force warp_k = block_k
        // (mirrors deep_gemm/jit_kernels/gemm.py lines 406-408/422;
        // adaptive is NOT re-evaluated here).
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, /*warp_k=*/bk, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else {
        bool from_runtime_lut = false;
        auto best = deep_gemm_bf16_common::get_best_configs(
            m, n, k, 1, num_sms, kGemmType, 256, &from_runtime_lut);
        auto [ns, bm, bn, bk, wm, wn, nst, sc] = best;
        int warp_k = bk;
        if (!from_runtime_lut && is_ppu1v5_device() &&
            deep_gemm_adaptive::bf16_adaptive_enabled(m, n, k)) {
            warp_k = std::get<6>(deep_gemm_adaptive::get_adaptive_configs(m, n, k, num_sms));
        }
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, warp_k, nst, sc);
    }
    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, smem_config] = cfg;
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

    // TODO get hw info from real env
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);
    bool enable_sbo_overlap = false;
    auto* output = static_cast<ElementOutput*>(out.data_ptr());
    if (is_ppu1v5_device()) {
        using CuteRuntime = DenseBF16GemmCuteFreeRuntimeT<ElementOutput>;
        using CutlassRuntime = DenseBF16GemmCutlass3RuntimeT<ElementOutput>;
        // Dense GEMV fast path (m == 1 or 2): SIMT avoids the tile path's BM=16
        // padding (15/16 lane waste at m==1, 14/16 at m==2). Explicit configs force tile.
        if ((m == 1 || m == 2) && !configs.has_value() && gemv_dense_bf16(lhs, rhs, out, m, n, k)) {
            return;
        }

        // Runtime path selection: DG_USE_CUTE=0 selects original CUTLASS 3 path
        bool use_cute_free = true;
        if (const char* env_ct = std::getenv("DG_USE_CUTE")) {
            if (std::string(env_ct) == "0") use_cute_free = false;
        }

        // Shared occupancy/overlap heuristic (identical for both paths). Computed
        // once here so both the CuteFree and Cutlass3 branches reuse it.
        // Step 1: compute blocks_per_cu purely from the SMEM budget so waves and
        // overlap_on can be decided BEFORE any kernel is compiled -- no probe
        // compile / occupancy query is needed.
        // SMEM-based occupancy heuristic (PPU 890P / ppu1.5: 256 KB shared memory per CU)
        // NOTE: the CuteFree path sets smem_cute_free = SMSIZE, so SMSIZE is equivalent.
        constexpr int kSmemPerCU = 256 * 1024;
        int blocks_per_cu = kSmemPerCU / SMSIZE;
        if (blocks_per_cu < 1) blocks_per_cu = 1;

        // Step 2: compute waves and decide overlap. Overlap-prologue gating
        // (acblas parity): wave = ceil(tiles / (num_sms * blocks_per_cu)); enable
        // next-tile prologue overlap only when wave > 3 AND k <= 2048, since the
        // overlap carries overhead. K-sweep A/B (M=2048, N=4096..40960): benefit
        // is +3.8% at K=1024 and +2.8% at K=2048, but drops to ~1% at K=4096 and
        // to noise level (<=0.2%) at K>=8192 -- the larger mainloop amortizes
        // the hidden prologue. acblas uses the same K<=2048 cutoff.
        const int tiles_m = (m + block_m - 1) / block_m;
        const int tiles_n = (n + block_n - 1) / block_n;
        const int num_tiles = tiles_m * tiles_n;
        const int blocks_per_wave = num_sms_new * blocks_per_cu;
        const int waves = blocks_per_wave > 0 ? (num_tiles + blocks_per_wave - 1) / blocks_per_wave : 0;
        const bool overlap_on = (waves > 3) && (k <= 2048);

        if (use_cute_free) {
        // --- CuteFree path (default) ---
        // warp_k comes from the adaptive tile selector (std::get<6> of
        // adaptive_cfg); WarpOnK (block_k / warp_k) is gated inside
        // get_warp_k (memory-bound region only + env/guard checks).
        const int warps_k = block_k / warp_k;
        dim3 const block_cute_free = (block_m / warp_m) * (block_n / warp_n) * warps_k * 32;
        int smem_cute_free = SMSIZE;

        // generate_impl only reads problem_shape (for IsAlignedN) + launch_info.
        const auto gemm_args = typename CuteRuntime::GemmArguments{
            .mode = cutlass::gemm::GemmUniversalMode::kGemm,
            .problem_shape = {m, n, k, 1},
            .mainloopargs = {input_a, stride_A, input_b, stride_B},
            .epilogueargs =
                {
                    {1.0f, 0.0f},
                    output,
                    stride_D,
                    output,
                    stride_D,
                },
            .hw_info = hw_info,
            .scheduler = {(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr},
        };

        typename CuteRuntime::GemmKernelParams params =
            CuteRuntime::to_underlying_arguments_rtc(gemm_args, nullptr);

        dim3 grid_cute_free;
        grid_cute_free.x = blocks_per_cu * num_sms_new;

        // Step 3: build the kernel name for the matching variant (explicit '='
        // assignment avoids _ovlp_ovlp on re-entry) and compile ONLY that one
        // kernel. Single compilation, single launch.
        std::string cute_free_kernel_name =
            overlap_on ? "bf16_deep_gemm_cute_free_ovlp" : "bf16_deep_gemm_cute_free";
        auto args = typename CuteRuntime::Args{
            .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, /*dense_s2_opt=*/true, num_stages,
                            "Normal", "Default", cute_free_kernel_name, enable_sbo_overlap, overlap_on},
            .launch_args = {grid, block_cute_free, smem_cute_free},
            .kernel_params = params,
        };

        auto code = CuteRuntime::generate(args);
        auto runtime = compiler->build(cute_free_kernel_name, code);
        if (is_compile_only())
            return;
        auto kernel = runtime->kernel;

        // Must stay byte-identical to GemmKernel::Params (cute_free/kernel/gemm.cuh).
        struct CuteFreeParams {
            cutlass::bfloat16_t const* ptr_A;
            cutlass::bfloat16_t const* ptr_B;
            ElementOutput* ptr_D;
            int M, N, K;
            int lda, ldb, ldd;
            DenseGemmTileSchedulerArguments scheduler;
        };

        CuteFreeParams hparams;
        hparams.ptr_A = input_a;
        hparams.ptr_B = input_b;
        hparams.ptr_D = output;
        hparams.M = m;
        hparams.N = n;
        hparams.K = k;
        hparams.lda = k;    // A is row-major MxK
        hparams.ldb = k;    // B is col-major KxN, stored as row-major NxK
        hparams.ldd = n;    // D is row-major MxN
        hparams.scheduler = DenseGemmTileSchedulerArguments{(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr};

        const auto& stream = current_stream();
        auto config = construct_launch_config(kernel, stream, smem_cute_free, grid_cute_free, block_cute_free);

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, nullptr, stream);
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, hparams));

        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            int numRegs = 0, localSize = 0;
            hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
            hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

            printf("[DenseGemm_BF16_CuteFree:]\n");
            printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
                   grid_cute_free.x);
            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n,
                   block_k, warp_m, warp_n, warp_k, num_stages);
            printf("SMSIZE:%d, vreg:%d, stack:%d\n", smem_cute_free, int(numRegs), int(localSize));
            printf("waves:%d, overlap_on:%d\n", waves, int(overlap_on));
        }
        } else {
            // --- Cutlass3 path (DG_USE_CUTE=0) ---
            // warp_k is obtained exactly as in the CuteFree path (adaptive
            // tile selector, std::get<6> of adaptive_cfg); all WarpOnK gating
            // (memory-bound region, K-depth gate, fat-tile cap, BK whitelist)
            // lives inside get_warp_k, so both
            // paths see identical BM/BN/BK/S/warp_k for the same shape.
            // The JIT-codegened TiledMMA carries Int<WarpOnK> in its thread
            // layout, so the threadblock must launch WarpOnK extra warp rows;
            // BF16DenseGemmKernel performs the partial-sum combine via
            // warp_on_k_reduce when WarpOnK > 1.
            const int warps_k = block_k / warp_k;
            dim3 const block_cutlass3 = (block_m / warp_m) * (block_n / warp_n) * warps_k * 32;
            const auto gemm_args = typename CutlassRuntime::GemmArguments{
                .mode = cutlass::gemm::GemmUniversalMode::kGemm,
                .problem_shape = {m, n, k, 1},
                .mainloopargs = {input_a, stride_A, input_b, stride_B},
                .epilogueargs =
                    {
                        {1.0f, 0.0f},
                        output,
                        stride_D,
                        output,
                        stride_D,
                    },
                .hw_info = hw_info,
                .scheduler = {(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr},
            };

            typename CutlassRuntime::GemmKernelParams params =
                CutlassRuntime::to_underlying_arguments_rtc(gemm_args, nullptr);

            // Single compilation: overlap decision is baked into the generated code.
            auto args = typename CutlassRuntime::Args{
                .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, /*dense_s2_opt=*/true, num_stages,
                                "Normal", "Default", "bf16_dense_gemm", false, /*overlap_prologue=*/overlap_on},
                .launch_args = {grid, block_cutlass3, SMSIZE},
                .kernel_params = params,
            };
            auto code = CutlassRuntime::generate(args);
            auto runtime = compiler->build("bf16_dense_gemm", code);
            auto kernel = runtime->kernel;

            args.launch_args.grid_dim.x *= blocks_per_cu;
            // Persistent DenseGemm scheduler reads final grid extent from hw_info.cu_count
            args.kernel_params.hw_info.cu_count = args.launch_args.grid_dim.x;

            DgProfParam dg_prof_params;
            if (ProfilingInterface::Instance().get_op_info()) {
                dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, nullptr,
                                          current_stream());
            }
            ProfilingInterface::Instance().instrument(true, dg_prof_params);

            CutlassRuntime::launch(runtime, args);

            ProfilingInterface::Instance().instrument(false, dg_prof_params);

            char* pEnv_params = std::getenv("show_log");
            if (pEnv_params && isdigit(*pEnv_params)) {
                int numRegs = 0, localSize = 0;
                hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
                hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

                printf("[DenseGemm_BF16_Cutlass3:]\n");
                printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
                printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
                       args.launch_args.grid_dim.x);
                printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n",
                       block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages);
                printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
                printf("waves:%d, overlap_prologue:%d\n", waves, int(overlap_on));
            }
        }
    } else {
        // Keep input/mainloop configuration unchanged and specialize only the epilogue.
        auto launch = [&](auto* output) {
            using OutputType = std::remove_pointer_t<decltype(output)>;
            using Runtime = BF16GemmCutlass3RuntimeT<OutputType>;
            // The TSM epilogue shares storage with the BF16 mainloop. FP32 stores
            // can require twice the BF16 epilogue storage, including explicit configs.
            const int smem_size = std::max(SMSIZE, int(block_m * block_n * sizeof(OutputType)));
            const auto gemm_args = typename Runtime::GemmArguments{
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

            auto params = Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

            auto args = typename Runtime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                               num_stages, "Normal", "Default", "bf16_deep_gemm",
                                                               enable_sbo_overlap},
                                               .launch_args = {grid, block, smem_size},
                                               .kernel_params = params};

            const auto& code = Runtime::generate(args);
            const auto& runtime = compiler->build("bf16_deep_gemm", code);
            const auto& kernel = runtime->kernel;
            int blocks_per_cu = 0;
            HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, smem_size);
            args.launch_args.grid_dim.x *= blocks_per_cu;
            DgProfParam dg_prof_params;
            if (ProfilingInterface::Instance().get_op_info()) {
                dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, grouped_layout,
                                          current_stream());
            }
            ProfilingInterface::Instance().instrument(true, dg_prof_params);

            Runtime::launch(runtime, args);

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
                printf("SMSIZE:%d, vreg:%d, stack:%d\n",smem_size, int(numRegs), int(localSize));
            }
        };
        if (out.scalar_type() == torch::kFloat) {
            launch(out.data_ptr<float>());
        } else {
            launch(reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>()));
        }
    }
}
// Public launcher: the input type remains BF16; only the output is dispatched.
static void bf16_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                      const int& m, const int& n, const int& k,
                      std::optional<ConfigTuple> configs = std::nullopt) {
    if (out.scalar_type() == torch::kFloat)
        bf16_gemm_impl<float>(lhs, rhs, out, m, n, k, configs);
    else
        bf16_gemm_impl<cutlass::bfloat16_t>(lhs, rhs, out, m, n, k, configs);
}

// 810E direct, native batch, split-K and TSM launchers.
// Rows are contiguous intervals. Test the nearest row in the other dimension
// for each batch difference (or vice versa), preserving legal interleaved views.
static bool model_810e_batch_output_overlaps(const torch::Tensor& d) {
    const int64_t batch = d.size(0), m = d.size(1), n = d.size(2);
    const int64_t bs = d.stride(0), ld = d.stride(1);
    if (batch <= 1) return false; // Row overlap was checked by the caller.
    if (bs < n) return true;
    if (m <= 1 || bs >= (m - 1) * ld + n || ld >= (batch - 1) * bs + n) return false;
    const bool batch_first = batch <= m;
    const int64_t count = batch_first ? batch : m, step = batch_first ? bs : ld;
    const int64_t other_count = batch_first ? m : batch, other_step = batch_first ? ld : bs;
    for (int64_t i = 1; i < count; ++i) {
        const int64_t offset = i * step;
        const int64_t j = std::min(offset / other_step, other_count - 1);
        if (offset - j * other_step < n ||
            (j + 1 < other_count && (j + 1) * other_step - offset < n)) return true;
    }
    return false;
}

// Validate native strided batches. Read-only inputs may
// broadcast batches; output batches must never race on the same address.
static bool model_810e_check_batch(const torch::Tensor& a, const torch::Tensor& b,
                                  const torch::Tensor& d) {
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && d.dim() == 3, "810E batch requires rank-3 tensors");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16 &&
                d.scalar_type() == torch::kBFloat16, "810E batch requires BF16 inputs and output");
    TORCH_CHECK(a.is_cuda() && b.device() == a.device() && d.device() == a.device(),
                "810E batch tensors must be on the same accelerator");
    TORCH_CHECK(!is_ppu1v5_device(), "810E batch targets PPU 1.0");
    TORCH_CHECK(a.size(0) == b.size(0) && a.size(0) == d.size(0) && a.size(1) == d.size(1) &&
                b.size(1) == d.size(2) && a.size(2) == b.size(2),
                "810E batch expects A[B,M,K], B[B,N,K], D[B,M,N]");
    if (a.size(0) == 0 || a.size(1) == 0 || b.size(1) == 0) return true;
    for (auto size : {a.size(0), a.size(1), a.size(2), b.size(1)})
        TORCH_CHECK(size <= std::numeric_limits<int>::max(), "810E batch dimensions exceed int32");
    TORCH_CHECK(a.size(0) <= device_runtime->get_prop()->maxGridSize[2], "810E batch exceeds grid.z limit");
    TORCH_CHECK(a.stride(2) == 1 && b.stride(2) == 1 && d.stride(2) == 1,
                "810E batch requires contiguous K in inputs and contiguous N in output");
    TORCH_CHECK(a.stride(0) >= 0 && b.stride(0) >= 0 && d.stride(0) > 0 &&
                a.stride(1) >= a.size(2) && b.stride(1) >= b.size(2) && d.stride(1) >= d.size(2),
                "810E batch requires nonnegative batch strides and nonoverlapping rows");
    TORCH_CHECK(!model_810e_batch_output_overlaps(d), "810E batch output batches overlap");
    if (a.size(2) == 0) { d.zero_(); return true; }
    // Every supported tile has BK >= 32 and at least two complete K stages.
    TORCH_CHECK(a.size(2) >= 64 && a.size(2) % 32 == 0,
                "810E batch requires K >= 64 and divisible by 32");
    TORCH_CHECK(a.stride(0) % 8 == 0 && b.stride(0) % 8 == 0 &&
                a.stride(1) % 8 == 0 && b.stride(1) % 8 == 0 &&
                (reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0,
                "810E batch requires 16-byte-aligned input rows and batches");
    return false;
}

// N=1 is the transpose of a one-row GEMV, and the output has the same physical
// order. Swapping pointers is sufficient; no input/output transpose is launched.
static bool model_810e_bf16_n1(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    if (is_ppu1v5_device() || b.size(0) != 1) return false;
    const int m = a.size(0), k = a.size(1);
    TORCH_CHECK(k % 8 == 0 && (reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0,
                "810E N=1 BF16 GEMV requires K divisible by 8 and 16-byte-aligned inputs");
    constexpr int bx = 128, unroll = 2;
    const int by = 128 / bx;
    const bool fp32 = d.scalar_type() == torch::kFloat;
    const std::string name = fp32 ? "model_810e_bf16_n1_fp32" : "model_810e_bf16_n1";
    DenseGemvRuntime::Args args{
        {bx, by, unroll, 1, 1, name, fp32},
        {dim3(ceil_div(m, by)), dim3(bx, by), 0},
        {m, k, b.data_ptr(), a.data_ptr(), d.data_ptr(), k}};
    auto code = DenseGemvRuntime::generate(args);
    auto runtime = compiler->build(name, code);
    DenseGemvRuntime::launch(runtime, args);
    return true;
}

class Model810eBatchedRuntime final : public LaunchRuntime<Model810eBatchedRuntime> {
public:
    using Base = BF16GemmCutlass3Runtime;
    using Args = Base::Args;
    static void replace_once(std::string& code, const std::string& before, const std::string& after) {
        const auto pos = code.find(before);
        TORCH_CHECK(pos != std::string::npos, "810E batched JIT template is incompatible with BF16 runtime");
        code.replace(pos, before.size(), after);
    }
    static std::string generate_impl(const Args& args) {
        // Reuse the exact architecture/mainloop/Params declarations. Keeping one
        // generator prevents host/device ABI drift as the dense runtime evolves.
        // Only the custom entry below runs. It reads M/N/K and all strides from
        // Params, and never instantiates the standard scheduler or epilogue's
        // operator(). These dummy dimensions affect unused type declarations
        // only; canonicalizing them lets different N/K cases share one binary.
        // Keep the actual launch arguments untouched, including batch count.
        auto codegen_args = args;
        cute::get<1>(codegen_args.kernel_params.with_tsm.problem_shape) = 128;
        cute::get<2>(codegen_args.kernel_params.with_tsm.problem_shape) = 128;
        auto code = Base::generate_impl(codegen_args);
        replace_once(code, "GemmKernel::SharedStorageSize,", "sizeof(typename CollectiveMainloop::SharedStorage),");
        replace_once(code, "GemmKernel op;\n  op(params, smem);",
            "Model810eBatchedKernel<CollectiveMainloop> op;\n  op(params, smem);");
        return code;
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

template <typename Output>
class Model810eDenseRuntime final : public LaunchRuntime<Model810eDenseRuntime<Output>> {
public:
    using Base = BF16GemmCutlass3RuntimeT<Output>;
    using Args = typename Base::Args;
    static std::string generate_impl(const Args& args) {
        // The replacement kernel uses runtime dimensions. SHAPE_N/K belong to
        // the unused standard scheduler/epilogue declarations, so normalize
        // only this codegen copy to share JIT kernels across problem shapes.
        auto codegen_args = args;
        cute::get<1>(codegen_args.kernel_params.with_tsm.problem_shape) = 128;
        cute::get<2>(codegen_args.kernel_params.with_tsm.problem_shape) = 128;
        auto code = Base::generate_impl(codegen_args);
        const int splits = cute::get<3>(args.kernel_params.with_tsm.problem_shape);
        Model810eBatchedRuntime::replace_once(code, "GemmKernel::SharedStorageSize,",
            "sizeof(typename CollectiveMainloop::SharedStorage),");
        Model810eBatchedRuntime::replace_once(code, "GemmKernel op;\n  op(params, smem);",
            fmt::format("Model810eDenseKernel<CollectiveMainloop, ElementC, {}> op;\n  op(params, smem);", splits));
        return code;
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

template <typename Output>
class Model810eReduceRuntime final : public LaunchRuntime<Model810eReduceRuntime<Output>> {
public:
    struct Params { const float* partial; void* d; int64_t count; };
    struct Args { LaunchArgs launch_args; Params params; int splits; };
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define BF16_HGRTC
#include <deep_gemm/impls/bf16_gemm_cutlass3.cuh>
extern "C" __global__ void model_810e_splitk_reduce(deep_gemm::Model810eReduceParams p) {{
    deep_gemm::model_810e_reduce<{}, {}>(p);
}}
)", std::is_same_v<Output, float> ? "float" : "__ppu_bfloat16", args.splits);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

template <typename Output>
static void model_810e_bf16_dense(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                 int bm, int bn, int bk, int wm, int wn, int stages, int splits) {
    const int m = a.size(0), n = b.size(0), k = a.size(1);
    const auto pow2 = [](int x) { return x > 0 && (x & (x - 1)) == 0; };
    TORCH_CHECK(bm >= 16 && bn >= 16 && bk >= 32 && pow2(bm) && pow2(bn) && pow2(bk) &&
                wm >= 16 && wn >= 16 && bm % wm == 0 && bn % wn == 0 &&
                stages >= 2 && stages <= 8, "invalid 810E BF16 tile (requires BK>=32)");
    TORCH_CHECK(splits >= 1 && splits <= 64 && k % bk == 0 && (k / bk) / splits >= stages,
                "810E BF16 requires 1<=splits<=64, K%BK=0, and at least stages BK tiles per split");
    const int threads = (bm / wm) * (bn / wn) * 32;
    const int smem = (bm + bn) * bk * 2 * stages;
    TORCH_CHECK(threads <= device_runtime->get_prop()->maxThreadsPerBlock &&
                smem <= device_runtime->get_prop()->sharedMemPerMultiprocessor,
                "810E BF16 tile exceeds device resources");
    using Runtime = Model810eDenseRuntime<Output>;
    using Base = typename Runtime::Base;
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    const Stride sa{k, cute::Int<1>{}, int64_t(m) * k};
    const Stride sb{k, cute::Int<1>{}, int64_t(n) * k};
    const Stride sd{n, cute::Int<1>{}, int64_t(m) * n};
    auto* output = static_cast<Output*>(d.data_ptr());
    torch::Tensor workspace;
    if (splits > 1)
        workspace = torch::empty({int64_t(splits) * m * n}, a.options().dtype(torch::kFloat));
    cutlass::KernelHardwareInfo hw{};
    hw.device_id = a.get_device(); hw.cu_count = get_num_sms();
    typename Base::GemmArguments gemm_args{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, splits},
        .mainloopargs = {reinterpret_cast<const cutlass::bfloat16_t*>(a.data_ptr<at::BFloat16>()), sa,
                         reinterpret_cast<const cutlass::bfloat16_t*>(b.data_ptr<at::BFloat16>()), sb},
        .epilogueargs = {{1, 0}, output, sd, output, sd},
        .hw_info = hw, .scheduler = {(uint32_t)m, nullptr}, .signal = nullptr};
    typename Runtime::Args args{
        .launch_info = {bm, bn, bk, wm, wn, 1, stages, "Normal", "Default", "model_810e_bf16_dense", false},
        .launch_args = {dim3(ceil_div(n, bn), ceil_div(m, bm), splits), dim3(threads), smem},
        .kernel_params = Base::to_underlying_arguments_rtc(gemm_args, splits > 1 ? workspace.data_ptr<float>() : nullptr)};
    auto code = Runtime::generate(args);
    auto runtime = compiler->build("model_810e_bf16_dense", code);
    Runtime::launch(runtime, args);
    if (splits > 1) {
        using Reduce = Model810eReduceRuntime<Output>;
        const int64_t count = int64_t(m) * n;
        typename Reduce::Args reduce_args{
            {dim3((count + 255) / 256), dim3(256), 0},
            {workspace.data_ptr<float>(), output, count}, splits};
        auto reduce_code = Reduce::generate(reduce_args);
        auto reduce_runtime = compiler->build("model_810e_splitk_reduce", reduce_code);
        Reduce::launch(reduce_runtime, reduce_args);
    }
}

static std::pair<int, int> model_810e_batch_k_config(const int k) {
    const int bk = k >= 128 && k % 64 == 0 ? 64 : 32;
    return {bk, std::min(3, k / bk)};
}

static void model_810e_bf16_strided_batched(const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& d, std::optional<ConfigTuple> configs = std::nullopt) {
    const int batch = a.size(0), m = a.size(1), n = b.size(1), k = a.size(2);
    auto [bk, stages] = model_810e_batch_k_config(k);
    int bm = m <= 64 ? 32 : 128, bn = m <= 64 ? 64 : 128;
    int wm = m <= 64 ? 16 : 32, wn = m <= 64 ? 32 : 64;
    int sms = get_num_sms();
    if (configs) {
        auto [ns, cbm, cbn, cbk, cwm, cwn, cstages, unused] = *configs;
        sms = ns; bm = cbm; bn = cbn; bk = cbk; wm = cwm; wn = cwn; stages = cstages;
    }
    const auto pow2 = [](int x) { return x > 0 && (x & (x - 1)) == 0; };
    TORCH_CHECK(bm >= 16 && bn >= 16 && bk >= 32 && pow2(bm) && pow2(bn) && pow2(bk) &&
                wm >= 16 && wn >= 16 && bm % wm == 0 && bn % wn == 0 &&
                stages >= 2 && stages <= 8, "invalid 810E BF16 batched tile (requires BK>=32)");
    TORCH_CHECK(k % bk == 0 && k / bk >= stages,
                "810E BF16 batched tile requires K%BK=0 and at least stages BK tiles");
    const int threads = (bm / wm) * (bn / wn) * 32;
    const int smem = (bm + bn) * bk * 2 * stages;
    TORCH_CHECK(threads <= device_runtime->get_prop()->maxThreadsPerBlock &&
                smem <= device_runtime->get_prop()->sharedMemPerMultiprocessor,
                "810E BF16 batched tile exceeds device resources");
    using Runtime = Model810eBatchedRuntime;
    using Base = Runtime::Base;
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    const Stride sa{a.stride(1), cute::Int<1>{}, a.stride(0)};
    const Stride sb{b.stride(1), cute::Int<1>{}, b.stride(0)};
    const Stride sd{d.stride(1), cute::Int<1>{}, d.stride(0)};
    auto* output = reinterpret_cast<cutlass::bfloat16_t*>(d.data_ptr<at::BFloat16>());
    cutlass::KernelHardwareInfo hw{};
    hw.device_id = a.get_device(); hw.cu_count = sms;
    Base::GemmArguments gemm_args{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, batch},
        .mainloopargs = {reinterpret_cast<const cutlass::bfloat16_t*>(a.data_ptr<at::BFloat16>()), sa,
                         reinterpret_cast<const cutlass::bfloat16_t*>(b.data_ptr<at::BFloat16>()), sb},
        .epilogueargs = {{1, 0}, output, sd, output, sd},
        .hw_info = hw, .scheduler = {(uint32_t)m, nullptr}, .signal = nullptr};
    Runtime::Args args{
        .launch_info = {bm, bn, bk, wm, wn, batch, stages, "Normal", "Default", "model_810e_bf16_batch", false},
        .launch_args = {dim3(ceil_div(n, bn), ceil_div(m, bm), batch), dim3(threads), smem},
        .kernel_params = Base::to_underlying_arguments_rtc(gemm_args, nullptr)};
    auto code = Runtime::generate(args);
    auto runtime = compiler->build("model_810e_bf16_batch", code);
    Runtime::launch(runtime, args);
}

template <typename Output, bool SplitK>
class Model810eTsmRuntime final : public LaunchRuntime<Model810eTsmRuntime<Output, SplitK>> {
public:
    using Base = BF16GemmCutlass3RuntimeT<Output>;
    // Host-only scheduling options stay outside device Params. Code generation
    // and launch dimensions consume the same values selected by the tile table.
    struct Args : Base::Args {
        bool grid_m_first = false;
        bool peel_first_k = false;
        bool static_m36 = false;
        bool static_k256 = false;
    };
    static std::string generate_impl(const Args& args) {
        auto codegen_args = static_cast<const typename Base::Args&>(args);
        // The custom kernel and EVT epilogue use real dimensions from Params
        // (or the validated M36 specialization). The unused standard scheduler
        // remains a fixed dummy type.
        cute::get<1>(codegen_args.kernel_params.with_tsm.problem_shape) = 128;
        cute::get<2>(codegen_args.kernel_params.with_tsm.problem_shape) = 128;
        auto code = Base::generate_impl(codegen_args);
        const int splits = SplitK ? cute::get<3>(args.kernel_params.with_tsm.problem_shape) : 1;
        Model810eBatchedRuntime::replace_once(code, "GemmKernel op;\n  op(params, smem);",
            fmt::format("static_assert(GemmKernel::SharedStorageSize <= {}, \"810E TSM launch storage is too small\");\n"
                        "  Model810eTsmKernel<{}, CollectiveEpilogue, {}, {}, {}, {}> op;\n  op(params, smem);",
                        args.launch_args.smem_size,
                        args.peel_first_k ? fmt::format("deep_gemm::Model810eFirstKPeel<CollectiveMainloop, {}>",
                                                       args.static_k256 ? "true" : "false") : "CollectiveMainloop",
                        SplitK ? "true" : "false", splits,
                        args.grid_m_first ? "true" : "false", args.static_m36 ? "true" : "false"));
        return code;
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Rank-3 views: dense uses a singleton input batch; split-K D has `splits`
// batches of FLOAT partials. No packing or copies are performed by this helper.
template <typename Output, bool SplitK>
static void model_810e_tsm_launch(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    int bm, int bn, int bk, int wm, int wn, int stages, int splits, int sms,
    bool grid_m_first = false,
    bool peel_first_k = false,
    bool static_m36 = false,
    bool static_k256 = false) {
    const int m = a.size(1), n = b.size(1), k = a.size(2);
    const int groups = SplitK ? splits : a.size(0);
    TORCH_CHECK(!static_m36 || (!SplitK && groups == 64 && m == 36 && n == 512 && k == 256 &&
                !is_ppu1v5_device() &&
                std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos),
                "Static M36 specialization requires native batch=64, M=36, N=512, K=256");
    TORCH_CHECK(!static_k256 || (!SplitK && groups == 64 && m == 36 && n == 512 && k == 256 &&
                bk == 64 && (stages == 2 || stages == 3) && static_m36 && peel_first_k &&
                !is_ppu1v5_device() &&
                std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos),
                "Static K256 specialization requires native 810E batch=64/M36/N512/K256, BK64, stages 2 or 3, "
                "static_m36=true and peel_first_k=true");
    const auto pow2 = [](int x) { return x > 0 && (x & (x - 1)) == 0; };
    // This mainloop advances its pipeline at K_BLOCK_MAX-2. BK16 has only
    // one register K block on 810E and cannot advance that state machine.
    TORCH_CHECK(bm >= 16 && bn >= 16 && bk >= 32 && pow2(bm) && pow2(bn) && pow2(bk) &&
                wm >= 16 && wn >= 16 && bm % wm == 0 && bn % wn == 0 &&
                stages >= 2 && stages <= 8, "invalid 810E TSM tile (requires BK>=32)");
    TORCH_CHECK(splits >= 1 && splits <= 64 && k % bk == 0 && (k / bk) / splits >= stages,
                "810E TSM requires K%BK=0 and at least stages BK tiles per split");
    const int threads = (bm / wm) * (bn / wn) * 32;
    const int main_smem = (bm + bn) * bk * 2 * stages;
    // The EVT stages WarpOnM*16 rows at a time, in FP32. Reserve one alignment
    // unit for its empty callback storage/padding; a generated static_assert
    // checks this bound against the actual CUTLASS SharedStorage type.
    const int epi_smem = (bm / wm) * 16 * bn * 4 + 128;
    const int smem = std::max(main_smem, epi_smem);
    TORCH_CHECK(threads <= device_runtime->get_prop()->maxThreadsPerBlock &&
                smem <= device_runtime->get_prop()->sharedMemPerMultiprocessor,
                "810E TSM tile exceeds device resources");
    using Runtime = Model810eTsmRuntime<Output, SplitK>;
    using Base = typename Runtime::Base;
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    const Stride sa{a.stride(1), cute::Int<1>{}, a.stride(0)};
    const Stride sb{b.stride(1), cute::Int<1>{}, b.stride(0)};
    const Stride sd{d.stride(1), cute::Int<1>{}, d.stride(0)};
    auto* output = static_cast<Output*>(d.data_ptr());
    cutlass::KernelHardwareInfo hw{};
    hw.device_id = a.get_device(); hw.cu_count = sms;
    typename Base::GemmArguments gemm_args{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, groups},
        .mainloopargs = {reinterpret_cast<const cutlass::bfloat16_t*>(a.data_ptr<at::BFloat16>()), sa,
                         reinterpret_cast<const cutlass::bfloat16_t*>(b.data_ptr<at::BFloat16>()), sb},
        .epilogueargs = {{1, 0}, output, sd, output, sd},
        .hw_info = hw, .scheduler = {(uint32_t)m, nullptr}, .signal = nullptr};
    const std::string name = SplitK ? "model_810e_tsm_dense" : "model_810e_tsm_batch";
    typename Base::Args base_args{
        .launch_info = {bm, bn, bk, wm, wn, 1, stages, "Normal", "Default", name, false},
        .launch_args = {grid_m_first ? dim3(ceil_div(m, bm), ceil_div(n, bn), groups) :
                                      dim3(ceil_div(n, bn), ceil_div(m, bm), groups), dim3(threads), smem},
        .kernel_params = Base::to_underlying_arguments_rtc(gemm_args, nullptr)};
    typename Runtime::Args args{base_args, grid_m_first, peel_first_k,
                               static_m36, static_k256};
    auto code = Runtime::generate(args);
    auto runtime = compiler->build(name, code);
    Runtime::launch(runtime, args);
}

template <typename Output>
static void model_810e_bf16_dense_tsm(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    int bm, int bn, int bk, int wm, int wn, int stages, int splits,
    bool grid_m_first = false,
    bool peel_first_k = false,
    bool static_m36 = false,
    bool static_k256 = false) {
    TORCH_CHECK(!static_m36 && !static_k256,
                "Static M36/K256 specialization requires native strided batches");
    const int m = a.size(0), n = b.size(0);
    const int alignment = 16 / sizeof(Output);
    // Standard EVT predicates only the vector's first element. Keep narrow or
    // unaligned output rows on the audited per-element path, including N=4.
    if (n < 8 || n % alignment != 0 || (reinterpret_cast<uintptr_t>(d.data_ptr()) & 15) != 0) {
        model_810e_bf16_dense<Output>(a, b, d, bm, bn, bk, wm, wn, stages, splits);
        return;
    }
    TORCH_CHECK(splits >= 1 && splits <= 64, "810E TSM requires 1<=splits<=64");
    const auto av = a.unsqueeze(0), bv = b.unsqueeze(0);
    if (splits == 1) {
        model_810e_tsm_launch<Output, true>(av, bv, d.unsqueeze(0), bm, bn, bk, wm, wn, stages, 1, get_num_sms(),
                                          grid_m_first, peel_first_k);
        return;
    }
    auto partial = torch::empty({int64_t(splits), int64_t(m), int64_t(n)}, a.options().dtype(torch::kFloat));
    model_810e_tsm_launch<float, true>(av, bv, partial, bm, bn, bk, wm, wn, stages, splits, get_num_sms(),
                                     grid_m_first, peel_first_k);
    using Reduce = Model810eReduceRuntime<Output>;
    const int64_t count = int64_t(m) * n;
    typename Reduce::Args args{{dim3((count + 255) / 256), dim3(256), 0},
                              {partial.data_ptr<float>(), d.data_ptr(), count}, splits};
    auto code = Reduce::generate(args);
    auto runtime = compiler->build("model_810e_splitk_reduce", code);
    Reduce::launch(runtime, args);
}

static void model_810e_bf16_strided_batched_tsm(const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& d, std::optional<ConfigTuple> configs = std::nullopt,
    bool grid_m_first = false,
    bool peel_first_k = false,
    bool static_m36 = false,
    bool static_k256 = false) {
    const int m = a.size(1), n = b.size(1);
    if (n < 8 || n % 8 != 0 || d.stride(1) % 8 != 0 || d.stride(0) % 8 != 0 ||
        (reinterpret_cast<uintptr_t>(d.data_ptr()) & 15) != 0) {
        TORCH_CHECK(!static_m36 && !static_k256,
                    "Static M36/K256 specialization requires an aligned TSM epilogue");
        model_810e_bf16_strided_batched(a, b, d, configs);
        return;
    }
    auto [bk, stages] = model_810e_batch_k_config(int(a.size(2)));
    int bm = m <= 64 ? 32 : 128, bn = m <= 64 ? 64 : 128;
    int wm = m <= 64 ? 16 : 32, wn = m <= 64 ? 32 : 64;
    int sms = get_num_sms();
    if (configs) {
        auto [ns, cbm, cbn, cbk, cwm, cwn, cstages, unused] = *configs;
        sms = ns; bm = cbm; bn = cbn; bk = cbk; wm = cwm; wn = cwn; stages = cstages;
    }
    model_810e_tsm_launch<cutlass::bfloat16_t, false>(a, b, d, bm, bn, bk, wm, wn, stages, 1, sms,
                                                  grid_m_first, peel_first_k, static_m36, static_k256);
}

static bool model_810e_dispatch_device() {
    return std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos;
}

static ConfigTuple model_810e_dispatch_config(const Model810eDispatchTile& tile) {
    // The standard launcher recomputes shared storage from the first seven
    // fields; native batch launchers do likewise. No cached smem estimate leaks
    // across output dtypes or implementations.
    return {get_num_sms(), tile.bm, tile.bn, tile.bk, tile.wm, tile.wn, tile.stages, {0, 0, 0}};
}

template <typename Output>
static void model_810e_dispatch_dense_entry(const Model810eDenseDispatchEntry& entry,
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    const auto& t = entry.tile;
    switch (entry.kind) {
        case Model810eDispatchKind::Direct:
            model_810e_bf16_dense<Output>(a, b, d, t.bm, t.bn, t.bk, t.wm, t.wn, t.stages, t.splits);
            return;
        case Model810eDispatchKind::TSM:
            model_810e_bf16_dense_tsm<Output>(a, b, d, t.bm, t.bn, t.bk, t.wm, t.wn, t.stages, t.splits,
                                             t.grid_m_first != 0, t.peel_first_k != 0,
                                             t.static_m36 != 0, t.static_k256 != 0);
            return;
        case Model810eDispatchKind::Hybrid: {
            const auto& tail = entry.tail_tile;
            DG_HOST_ASSERT(entry.split_m > 0 && entry.split_m < entry.m);
            DG_HOST_ASSERT(t.splits == 1 && tail.splits == 1);
            model_810e_bf16_dense_tsm<Output>(a.narrow(0, 0, entry.split_m), b,
                d.narrow(0, 0, entry.split_m), t.bm, t.bn, t.bk, t.wm, t.wn, t.stages, 1,
                t.grid_m_first != 0, t.peel_first_k != 0,
                t.static_m36 != 0, t.static_k256 != 0);
            model_810e_bf16_dense_tsm<Output>(a.narrow(0, entry.split_m, entry.m - entry.split_m), b,
                d.narrow(0, entry.split_m, entry.m - entry.split_m),
                tail.bm, tail.bn, tail.bk, tail.wm, tail.wn, tail.stages, 1,
                tail.grid_m_first != 0, tail.peel_first_k != 0,
                tail.static_m36 != 0, tail.static_k256 != 0);
            return;
        }
        case Model810eDispatchKind::Standard:
            // Call the lower-level launcher with an explicit tuple. Calling
            // the public bf16_gemm_nt entry here would re-enter this table.
            bf16_gemm(a, b, d, int(entry.m), int(entry.n), int(entry.k), model_810e_dispatch_config(t));
            return;
    }
}

static bool model_810e_dispatch_dense(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    if constexpr (kModel810eDenseDispatch.size() == 0) return false;
    if (!model_810e_dispatch_device()) return false;
    // Native input descriptors require aligned bases; preserve the
    // original fallback for unusual contiguous views with a storage offset.
    if ((reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) != 0) return false;
    const auto output = d.scalar_type() == torch::kFloat ? Model810eOutputType::FP32 : Model810eOutputType::BF16;
    for (const auto& entry : kModel810eDenseDispatch) {
        if (entry.m != a.size(0) || entry.n != b.size(0) || entry.k != a.size(1) || entry.output != output) continue;
        if (output == Model810eOutputType::FP32)
            model_810e_dispatch_dense_entry<float>(entry, a, b, d);
        else
            model_810e_dispatch_dense_entry<cutlass::bfloat16_t>(entry, a, b, d);
        return true;
    }
    return false;
}

static bool model_810e_dispatch_batch(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    if constexpr (kModel810eBatchDispatch.size() == 0) return false;
    if (!model_810e_dispatch_device()) return false;
    const auto output = d.scalar_type() == torch::kFloat ? Model810eOutputType::FP32 : Model810eOutputType::BF16;
    for (const auto& entry : kModel810eBatchDispatch) {
        if (entry.batch != a.size(0) || entry.m != a.size(1) || entry.n != b.size(1) || entry.k != a.size(2) ||
            entry.lhs_ld != a.stride(1) || entry.rhs_ld != b.stride(1) || entry.out_ld != d.stride(1) ||
            entry.lhs_batch_stride != a.stride(0) || entry.rhs_batch_stride != b.stride(0) ||
            entry.out_batch_stride != d.stride(0) || entry.output != output) continue;
        const auto configs = model_810e_dispatch_config(entry.tile);
        if (entry.kind == Model810eDispatchKind::Direct)
            model_810e_bf16_strided_batched(a, b, d, configs);
        else if (entry.kind == Model810eDispatchKind::TSM)
            model_810e_bf16_strided_batched_tsm(a, b, d, configs,
                entry.tile.grid_m_first != 0, entry.tile.peel_first_k != 0,
                entry.tile.static_m36 != 0, entry.tile.static_k256 != 0);
        else
            return false; // No standard native strided-batch launcher exists.
        return true;
    }
    return false;
}

} // namespace deep_gemm
