#pragma once

#include <torch/python.h>
#include <algorithm>
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
#include "../heuristics/bf16_memory_policy.hpp"
// #include "../heuristics/gemm_search_space.hpp"
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/scheduler/densegemm_scheduler_cutlass3.cuh>
#include <deep_gemm/common/gemm_occ_model.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
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

    using InputStrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    struct MainLoopArguments {
        cutlass::bfloat16_t const* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        cutlass::bfloat16_t const* ptr_B;
        InputStrideB stride_B;
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
        bool denseS2Opt = false;
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
using LayoutB     = {20};
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

using KernelSchedule = cute::conditional_t<kKernelStrategy == KernelStrategy::MultistageOnN, KernelAiuMultistageOnN, cutlass::gemm::KernelAiuMultistage>;
using DispatchPolicy = cutlass::gemm::MainloopPPUAiuOpt<STAGES, KernelSchedule>;

static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
static constexpr int TSM_LD_NUM = BLOCK_M == 8 ? 2 : 4;

static constexpr int SmemLayoutStageStrideA = BLOCK_M * BLOCK_K;
static constexpr int SmemLayoutStageStrideB = BLOCK_N * BLOCK_K;
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
    {21},
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
            std::is_same_v<ElementOutput, float> ? "float" : "cutlass::bfloat16_t",
            "cutlass::layout::ColumnMajor",
            args.launch_info.denseS2Opt ? "BF16GemmMainloop810E<CollectiveMainloop>" : "CollectiveMainloop");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

using BF16GemmCutlass3Runtime = BF16GemmCutlass3RuntimeT<cutlass::bfloat16_t>;

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
        bool nonpersistent = false;
        int b_cache_policy = 1;
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

    // ElementOutput specializes this runtime and its cache. Keep every codegen
    // field here, including alignment and next-tile prologue overlap.
    static auto compile_key(const Args& args) {
        const auto& info = args.launch_info;
        const int shape_n = cute::get<1>(args.kernel_params.problem_shape);
        const bool is_aligned_n = shape_n % info.block_n == 0;
        const bool use_packed_store = shape_n % 2 == 0 &&
            reinterpret_cast<uintptr_t>(args.kernel_params.collective_epilogue_params.ptr_D) % (2 * sizeof(ElementOutput)) == 0;
        return std::make_tuple(info.gemm_type, info.block_m, info.block_n, info.block_k,
                               info.warp_m, info.warp_n, info.warp_k, info.num_stages,
                               info.dense_s2_opt, is_aligned_n, use_packed_store, info.kernel_name, info.overlap_prologue,
                               info.b_cache_policy, info.nonpersistent);
    }

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        const int shape_n = cute::get<1>(args.kernel_params.problem_shape);
        const bool is_aligned_n = (shape_n % info.block_n == 0);
        const bool use_packed_store = shape_n % 2 == 0 &&
            reinterpret_cast<uintptr_t>(args.kernel_params.collective_epilogue_params.ptr_D) % (2 * sizeof(ElementOutput)) == 0;

        return fmt::format(
            R"(
#include <deep_gemm/impls/bf16_gemm_cute_free.cuh>

namespace deep_gemm {{

using GemmTypeTag = std::integral_constant<GemmType, GemmType::{0}>;

using Kernel = BF16GemmCuteFreeKernel<{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {11}, {9}, {13}, GemmTypeTag, {12}, {14}, {15}>;

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
            std::is_same_v<ElementOutput, float> ? "float" : "cutlass::bfloat16_t",
            use_packed_store ? "true" : "false",
            info.b_cache_policy, info.nonpersistent ? "true" : "false");
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
        bool staticK = false;
        bool pairLoad = false;
        bool persistent = false;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemvDenseArgs kernel_params;
    };

    static auto compile_key(const Args& args) {
        const auto& info = args.launch_info;
        return std::make_tuple(info.block_x, info.block_y, info.k_per_thread,
                               info.m == 2, info.npt, info.kernel_name, info.output_fp32,
                               info.staticK, info.pairLoad, info.persistent, info.staticK ? args.kernel_params.K : 0);
    }

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
    {}
    {}
    {}<__ppu_bfloat16, {}, float,
       {}, {}, {}, {}, {}{}>({}, blockIndex);
    {}
}}

}}
)",
            args.launch_info.kernel_name,
            args.launch_info.staticK ? fmt::format(
                "GemvDenseArgs fixed_args = args; fixed_args.K = {}; fixed_args.stride_wn = {};",
                args.kernel_params.K, args.kernel_params.K) : "",
            args.launch_info.persistent ? fmt::format(
                "for (int64_t blockIndex = blockIdx.x; blockIndex < (int64_t(args.N) + {0} - 1) / {0}; blockIndex += gridDim.x) {{",
                args.launch_info.block_y * (is_m2 ? args.launch_info.npt : 1)) :
                "const int blockIndex = blockIdx.x;",
            entry,
            args.launch_info.output_fp32 ? "float" : "__ppu_bfloat16",
            args.launch_info.pairLoad ? "int" : "int4",
            args.launch_info.pairLoad ? "int" : "int4",
            args.launch_info.block_x, args.launch_info.block_y, args.launch_info.k_per_thread,
            npt_arg, args.launch_info.staticK ? "fixed_args" : "args",
            args.launch_info.persistent ? "__syncthreads(); }" : "");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// bf16 dense GEMV fast path (m == 1 or 2). Returns true when launched; false
// means the shape is uncovered and the caller must fall back to the tile path.
static bool gemv_dense_bf16(const torch::Tensor& lhs, const torch::Tensor& rhs,
                            const torch::Tensor& out, const int& m, const int& n, const int& k,
                            bool transposed_n1 = false) {
    DG_HOST_ASSERT(m == 1 || m == 2);
    TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous(),
                "dense gemv requires contiguous tensors");

    int block_x = 0, block_y = 0, k_per_thread = 0, npt = 1;
    if (!deep_gemm_bf16_common::dense_gemv_select_configs(m, n, k, rhs.data_ptr(), lhs.data_ptr(),
                                                          block_x, block_y, k_per_thread, npt, transposed_n1))
        return false;
    const bool is810e = std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos;
    const bool pairLoad = m == 1 && dense_gemv_pair_load(n, k, is810e);
    if (pairLoad) {
        block_x = 16;
        block_y = 2;
        k_per_thread = k / 32;
    }

    DenseGemvRuntime::GemvDenseArgs params;
    params.N = n;
    params.K = k;
    params.x_ptr = lhs.data_ptr<at::BFloat16>();
    params.w_ptr = rhs.data_ptr<at::BFloat16>();
    params.y_ptr = out.data_ptr();
    params.stride_wn = k;

    dim3 block(block_x, block_y);
    dim3 grid(ceil_div(n, block_y * npt));
    const bool persistent = is810e && device_runtime->get_num_simt_sms() < device_runtime->get_prop()->multiProcessorCount &&
                            grid.x > unsigned(device_runtime->get_num_simt_sms());
    if (persistent) grid.x = device_runtime->get_num_simt_sms();

    // m == 2 -> dual-accumulator variant; npt == 2 walks two W rows/thread sharing
    // (x0, x1), each block owning block_y * npt rows. Names split so a trace shows npt.
    std::string kernel_name = m == 1 ? "gemv_dense_bf16"
                                     : (npt == 2 ? "gemv_dense_m2r2_bf16" : "gemv_dense_m2_bf16");
    const bool output_fp32 = out.scalar_type() == torch::kFloat;
    if (output_fp32) kernel_name += "_fp32";
    // Bound code growth to short reductions; constant K removes loop bounds
    // and row-stride arithmetic without specializing M/N or changing the ABI.
    const bool staticK = k <= 1024 && is810e;
    auto args = DenseGemvRuntime::Args{
        .launch_info = {block_x, block_y, k_per_thread, m, npt, kernel_name, output_fp32, staticK, pairLoad, persistent},
        .launch_args = {grid, block, 0},
        .kernel_params = params,
    };
    const auto& runtime = compiler->build_cached<DenseGemvRuntime>(kernel_name, args);
    const auto& kernel = runtime->kernel;

    static constexpr GemmType kGemmType = GemmType::Normal;
    DgProfParam dg_prof_params;
    if (deep_gemm::profiling_enabled()) {
        dg_prof_params.set_params(kGemmType, false, std::string(out.scalar_type() == torch::kFloat ? "bf16_bf16_fp32" : "bf16"), 1, transposed_n1 ? n : m, transposed_n1 ? 1 : n, k, 0, nullptr,
                                  current_stream());
    }
    {
        auto profiling_guard = deep_gemm::show_gemm_ut(dg_prof_params);
        DenseGemvRuntime::launch(runtime, args);
    }

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
    const int num_sms = get_num_tensor_sms_810e();
    // A GEMV grid can occupy every CU, so use the persistent tile scheduler
    // when the caller restricts the CU budget.
    const bool limited_n1 = n == 1 && is_ppu1v5_device() &&
                            num_sms < device_runtime->get_prop()->multiProcessorCount;
    if (n == 1 && !configs.has_value() && is_ppu1v5_device() && !limited_n1 &&
        gemv_dense_bf16(rhs, lhs, out, 1, m, k, true))
        return;
    static constexpr GemmType kGemmType = GemmType::Normal;

    using Config = std::tuple<int, int, int, int, int, int, int, int, std::tuple<int, int, int>>;

    Config cfg;
    bool use_memory_policies = false;
    if (configs.has_value()) {
        // Explicit config path: unpack the 8-element public tuple and force warp_k = block_k;
        // adaptive selection is not re-evaluated here.
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, /*warp_k=*/bk, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else {
        int selected_warp_k = 0;
        bool from_runtime_lut = false;
        auto best = deep_gemm_bf16_common::get_best_configs(
            m, n, k, 1, num_sms, kGemmType, 256, &from_runtime_lut, &selected_warp_k);
        auto [ns, bm, bn, bk, wm, wn, nst, sc] = best;
        int warp_k = selected_warp_k > 0 ? selected_warp_k : bk;
        if (!from_runtime_lut && selected_warp_k > 0) {
            use_memory_policies = memory_policies_enabled();
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
        // Dense GEMV fast path (m == 1 or 2): SIMT avoids the tile path's BM=16
        // padding (15/16 lane waste at m==1, 14/16 at m==2). Explicit configs force tile.
        if ((m == 1 || m == 2) && !configs.has_value() && !limited_n1 &&
            gemv_dense_bf16(lhs, rhs, out, m, n, k)) {
            return;
        }

        // Step 1: compute blocks_per_cu purely from the SMEM budget so waves and
        // overlap_on can be decided BEFORE any kernel is compiled -- no probe
        // compile / occupancy query is needed.
        // SMEM-based occupancy heuristic (PPU 890P / ppu1.5: 256 KB shared memory per CU)
        // NOTE: the CuteFree path sets smem_cute_free = SMSIZE, so SMSIZE is equivalent.
        constexpr int kSmemPerCU = 256 * 1024;
        int blocks_per_cu = kSmemPerCU / SMSIZE;
        if (blocks_per_cu < 1) blocks_per_cu = 1;
        if (limited_n1) {
            // A grid with at most num_sms CTAs cannot occupy more than that
            // many CUs, regardless of where the hardware schedules them.
            blocks_per_cu = 1;
            num_sms_new = std::min(num_sms_new, num_sms);
        }

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
        MemoryExecutionPolicy memory_policy;
        if (use_memory_policies) {
            const auto prop = device_runtime->get_prop();
            memory_policy = get_memory_execution_policy(m, n, block_m, block_n,
                num_sms_new, blocks_per_cu, num_sms, prop->multiProcessorCount);
        }

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
        const bool nonpersistent = memory_policy.nonpersistent;
        DG_HOST_ASSERT(!nonpersistent || !overlap_on);
        grid_cute_free = nonpersistent ? dim3(num_tiles, 1, 1)
                                     : dim3(blocks_per_cu * num_sms_new, 1, 1);

        // Step 3: build the kernel name for the matching variant (explicit '='
        // assignment avoids _ovlp_ovlp on re-entry) and compile ONLY that one
        // kernel. Single compilation, single launch.
        std::string cute_free_kernel_name =
            overlap_on ? "bf16_deep_gemm_cute_free_ovlp" : "bf16_deep_gemm_cute_free";
        auto args = typename CuteRuntime::Args{
            .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, /*dense_s2_opt=*/true, num_stages,
                            "Normal", "Default", cute_free_kernel_name, enable_sbo_overlap, overlap_on,
                            nonpersistent, memory_policy.b_cache_policy},
            .launch_args = {grid, block_cute_free, smem_cute_free},
            .kernel_params = params,
        };

        auto runtime = compiler->build_cached<CuteRuntime>(cute_free_kernel_name, args);
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
        if (deep_gemm::profiling_enabled()) {
            dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, nullptr, stream);
        }
        {
            auto profiling_guard = deep_gemm::show_gemm_ut(dg_prof_params);
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, hparams));
        }

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
            printf("b_cache_policy:%d, nonpersistent:%d, tiles_m:%d, tiles_n:%d\n",
                   memory_policy.b_cache_policy, int(nonpersistent), tiles_m, tiles_n);
        }
    } else {
        // Keep input/mainloop configuration unchanged and specialize only the epilogue.
        auto launch = [&](auto* output) {
            using OutputType = std::remove_pointer_t<decltype(output)>;
            using Runtime = BF16GemmCutlass3RuntimeT<OutputType>;
            // Standard and custom TSM kernels share the same FP32 EVT staging.
            const int smem_size = get_smem_size_bf16_810e(block_m, block_n, block_k, warp_m, num_stages);
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

            // Opt in only this dense PPU 1.0 caller; grouped runtimes keep
            // the default mainloop, including their separate overlap policies.
            args.launch_info.denseS2Opt =
                std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos;
            auto code = Runtime::generate(args);
            code += fmt::format("\nstatic_assert(deep_gemm::GemmKernel::SharedStorageSize <= {}, "
                                "\"BF16 dense launch storage is too small\");\n", smem_size);
            const auto& runtime = compiler->build("bf16_deep_gemm", code);
            const auto& kernel = runtime->kernel;
            int blocks_per_cu = 0;
            HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, smem_size);
            args.launch_args.grid_dim.x *= blocks_per_cu;
            if (args.launch_info.denseS2Opt && num_sms_new < get_tensor_cu_count_810e())
                args.launch_args.grid_dim.x = num_sms_new;
            DgProfParam dg_prof_params;
            if (deep_gemm::profiling_enabled()) {
                dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, grouped_layout,
                                          current_stream());
            }
            {
                auto profiling_guard = deep_gemm::show_gemm_ut(dg_prof_params);
                Runtime::launch(runtime, args);
            }

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
// Follow the existing show_log convention without changing the shared JIT API.
template <typename Args>
static void show_bf16_gemm_launch_810e(const Args& args, int m, int n, int k, int numSms) {
    const char* show_log = std::getenv("show_log");
    if (!show_log || !isdigit(*show_log)) return;
    const auto& tile = args.launch_info;
    const auto& launch = args.launch_args;
    printf("[DenseGemm_BF16:]\n");
    printf("group:1, problem:[%d, %d, %d], kernel:%s\n", m, n, k, tile.kernel_name.c_str());
    printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d], num_stages:%d\n",
           tile.block_m, tile.block_n, tile.block_k, tile.warp_m, tile.warp_n, tile.num_stages);
    printf("num_sms:%d, grid:[%u, %u, %u], block:[%u, %u, %u], SMSIZE:%d\n",
           numSms, launch.grid_dim.x, launch.grid_dim.y, launch.grid_dim.z,
           launch.block_dim.x, launch.block_dim.y, launch.block_dim.z, launch.smem_size);
}

// N=1 is the transpose of a one-row GEMV, and the output has the same physical
// order. Swapping pointers is sufficient; no input/output transpose is launched.
static bool gemv_dense_bf16_810e(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    if (is_ppu1v5_device() || b.size(0) != 1) return false;
    const int m = a.size(0), k = a.size(1);
    TORCH_CHECK(k % 8 == 0 && (reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0,
                "810E N=1 BF16 GEMV requires K divisible by 8 and 16-byte-aligned inputs");
    const bool is810e = std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos;
    int bx, by, unroll;
    bool pairLoad;
    dense_gemv_select_configs_810e(m, k, is810e, bx, by, unroll, pairLoad);
    const bool fp32 = d.scalar_type() == torch::kFloat;
    const std::string name = fp32 ? "gemv_dense_bf16_fp32_810e" : "gemv_dense_bf16_810e";
    DenseGemvRuntime::Args args{
        {bx, by, unroll, 1, 1, name, fp32, is810e && k <= 1024, pairLoad},
        {dim3(ceil_div(m, by)), dim3(bx, by), 0},
        {m, k, b.data_ptr(), a.data_ptr(), d.data_ptr(), k}};
    if (is810e && device_runtime->get_num_simt_sms() < device_runtime->get_prop()->multiProcessorCount &&
        args.launch_args.grid_dim.x > unsigned(device_runtime->get_num_simt_sms())) {
        args.launch_info.persistent = true;
        args.launch_args.grid_dim.x = device_runtime->get_num_simt_sms();
    }
    auto code = DenseGemvRuntime::generate(args);
    auto runtime = compiler->build(name, code);
    DgProfParam params;
    if (profiling_enabled())
        params.set_params(GemmType::Normal, false, fp32 ? "bf16_bf16_fp32" : "bf16",
                          1, m, 1, k, 0, nullptr, current_stream());
    auto profilingGuard = show_gemm_ut(params);
    DenseGemvRuntime::launch(runtime, args);
    const char* show_log = std::getenv("show_log");
    if (show_log && isdigit(*show_log)) {
        printf("[DenseGemm_BF16_GemV:]\n");
        printf("group:1, problem:[%d, 1, %d], kernel:%s\n", m, k, name.c_str());
        printf("BlockX:%d, BlockY:%d, k_per_thread:%d, num_sms:%d, threadblock_count:%u\n",
               bx, by, unroll, device_runtime->get_num_simt_sms(), args.launch_args.grid_dim.x);
    }
    return true;
}

// Share source substitutions across the dense Direct and TSM runtimes.
static void replace_bf16_gemm_source(std::string& code, const std::string& before, const std::string& after) {
    const auto pos = code.find(before);
    TORCH_CHECK(pos != std::string::npos, "810E JIT template is incompatible with BF16 runtime");
    code.replace(pos, before.size(), after);
}

template <typename Output>
class BF16GemmDirectRuntime810ET final : public LaunchRuntime<BF16GemmDirectRuntime810ET<Output>> {
public:
    using Base = BF16GemmCutlass3RuntimeT<Output>;
    using Args = typename Base::Args;
    static std::string generate_impl(const Args& args) {
        // The replacement kernel uses runtime dimensions. SHAPE_N/K belong to
        // the unused standard scheduler/epilogue declarations, so normalize
        // only this codegen copy to share JIT kernels across problem shapes.
        auto codegenArgs = args;
        cute::get<1>(codegenArgs.kernel_params.with_tsm.problem_shape) = 128;
        cute::get<2>(codegenArgs.kernel_params.with_tsm.problem_shape) = 128;
        auto code = Base::generate_impl(codegenArgs);
        const int splits = cute::get<3>(args.kernel_params.with_tsm.problem_shape);
        replace_bf16_gemm_source(code, "GemmKernel::SharedStorageSize,",
            "sizeof(typename CollectiveMainloop::SharedStorage),");
        replace_bf16_gemm_source(code, "GemmKernel op;\n  op(params, smem);",
            fmt::format("BF16GemmDirectKernel810E<BF16GemmMainloop810E<CollectiveMainloop>, ElementC, {}, {}> op;\n  op(params, smem);",
                splits, args.kernel_params.with_tsm.hw_info.cu_count < get_tensor_cu_count_810e() ? "true" : "false"));
        return code;
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

template <typename Output>
class BF16GemmSplitKReduceRuntime810ET final : public LaunchRuntime<BF16GemmSplitKReduceRuntime810ET<Output>> {
public:
    struct Params { const float* partial; void* d; int64_t count; };
    struct Args { LaunchArgs launch_args; Params params; int splits; };
    static LaunchArgs launch_args(int64_t count) {
        const int sms = device_runtime->get_num_simt_sms();
        TORCH_CHECK(sms > 0, "810E split-K reduction requires a positive CU budget");
        const auto prop = device_runtime->get_prop();
        const bool bounded = sms < prop->multiProcessorCount;
        const int threads = bounded ? std::min(1024, prop->maxThreadsPerBlock) : 256;
        int64_t blocks = ceil_div(count, int64_t(threads));
        if (bounded)
            blocks = std::min<int64_t>(blocks, sms);
        TORCH_CHECK(blocks > 0 && blocks <= prop->maxGridSize[0],
                    "810E split-K reduction exceeds grid.x limit");
        return {dim3(static_cast<unsigned int>(blocks)), dim3(threads), 0};
    }
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/bf16_gemm_cutlass3.cuh>
extern "C" __global__ void bf16_gemm_split_k_reduce_810e(deep_gemm::BF16GemmSplitKReduceParams810E p) {{
    deep_gemm::bf16_gemm_split_k_reduce_impl_810e<{}, {}>(p);
}}
)", std::is_same_v<Output, float> ? "float" : "__ppu_bfloat16", args.splits);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

template <typename Output>
static void bf16_gemm_direct_810e(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
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
    using Runtime = BF16GemmDirectRuntime810ET<Output>;
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
    hw.device_id = a.get_device(); hw.cu_count = get_num_tensor_sms_810e();
    typename Base::GemmArguments gemmArgs{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, splits},
        .mainloopargs = {reinterpret_cast<const cutlass::bfloat16_t*>(a.data_ptr<at::BFloat16>()), sa,
                         reinterpret_cast<const cutlass::bfloat16_t*>(b.data_ptr<at::BFloat16>()), sb},
        .epilogueargs = {{1, 0}, output, sd, output, sd},
        .hw_info = hw, .scheduler = {(uint32_t)m, nullptr}, .signal = nullptr};
    typename Runtime::Args args{
        .launch_info = {bm, bn, bk, wm, wn, 1, stages, "Normal", "Default", "bf16_gemm_direct_810e", false},
        .launch_args = {dim3(ceil_div(n, bn), ceil_div(m, bm), splits), dim3(threads), smem},
        .kernel_params = Base::to_underlying_arguments_rtc(gemmArgs, splits > 1 ? workspace.data_ptr<float>() : nullptr)};
    if (get_num_tensor_sms_810e() < get_tensor_cu_count_810e()) {
        TORCH_CHECK(get_num_tensor_sms_810e() > 0, "810E dense GEMM requires a positive CU budget");
        const int64_t tiles = int64_t(ceil_div(m, bm)) * ceil_div(n, bn) * splits;
        args.launch_args.grid_dim = dim3(std::min<int64_t>(get_num_tensor_sms_810e(), tiles));
    }
    auto code = Runtime::generate(args);
    auto runtime = compiler->build("bf16_gemm_direct_810e", code);
    Runtime::launch(runtime, args);
    show_bf16_gemm_launch_810e(args, m, n, k, get_num_tensor_sms_810e());
    if (splits > 1) {
        using Reduce = BF16GemmSplitKReduceRuntime810ET<Output>;
        const int64_t count = int64_t(m) * n;
        typename Reduce::Args reduceArgs{
            Reduce::launch_args(count),
            {workspace.data_ptr<float>(), output, count}, splits};
        auto reduceCode = Reduce::generate(reduceArgs);
        auto reduceRuntime = compiler->build("bf16_gemm_split_k_reduce_810e", reduceCode);
        Reduce::launch(reduceRuntime, reduceArgs);
    }
}

// Validate the resolved grid, including reduced-budget persistent launches,
// before generating or compiling a kernel.
static void check_bf16_gemm_grid_810e(const dim3& grid) {
    const auto prop = device_runtime->get_prop();
    TORCH_CHECK(grid.x > 0 && grid.y > 0 && grid.z > 0 &&
                grid.x <= uint64_t(prop->maxGridSize[0]) &&
                grid.y <= uint64_t(prop->maxGridSize[1]) &&
                grid.z <= uint64_t(prop->maxGridSize[2]),
                "810E GEMM grid exceeds device limits: (", grid.x, ", ", grid.y, ", ", grid.z, ")");
}

template <typename Output, bool SplitK>
class BF16GemmTsmRuntime810ET final : public LaunchRuntime<BF16GemmTsmRuntime810ET<Output, SplitK>> {
public:
    using Base = BF16GemmCutlass3RuntimeT<Output>;
    // Host-only scheduling options stay outside device Params. Code generation
    // and launch dimensions consume the same values selected by the dispatch heuristic.
    struct Args : Base::Args {
        int tileOrder = 0;
        bool peelFirstK = false;
        bool unrollK = false;
        bool narrowNTail = false;
    };
    static std::string generate_impl(const Args& args) {
        auto codegenArgs = static_cast<const typename Base::Args&>(args);
        // The custom kernel and EVT epilogue use real dimensions from Params
        // (or a validated static K). The unused standard scheduler
        // remains a fixed dummy type.
        cute::get<1>(codegenArgs.kernel_params.with_tsm.problem_shape) = 128;
        cute::get<2>(codegenArgs.kernel_params.with_tsm.problem_shape) = 128;
        auto code = Base::generate_impl(codegenArgs);
        const int splits = SplitK ? cute::get<3>(args.kernel_params.with_tsm.problem_shape) : 1;
        auto kernelType = fmt::format("BF16GemmTsmKernel810E<{}, CollectiveEpilogue, {}, {}, {}, {}, {}>",
                                      fmt::format("deep_gemm::BF16GemmMainloop810E<CollectiveMainloop, {}, {}>",
                                                  args.peelFirstK ? "true" : "false",
                                                  args.unrollK ? cute::get<2>(args.kernel_params.with_tsm.problem_shape) / args.launch_info.block_k : 0),
                                      SplitK ? "true" : "false", splits,
                                      args.tileOrder,
                                      args.unrollK ? cute::get<2>(args.kernel_params.with_tsm.problem_shape) : 0,
                                      args.kernel_params.with_tsm.hw_info.cu_count < get_tensor_cu_count_810e() ? "true" : "false");
        if (args.narrowNTail) {
            // Both alternatives use 256 threads. The smaller mainloop reuses
            // the primary allocation and writes only valid scalar outputs.
            replace_bf16_gemm_source(code, "extern \"C\"", R"(
using TailTile = Shape<_128, _16, _64>;
using TailMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<_8, _1, _1>>,
                        Tile<_128, _16, _16>>;
using TailOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
    ArchTag, ElementB, false, _16, _64, true, 16 * 64>;
using TailCollective = cutlass::gemm::collective::CollectiveMma<
    ArchTag, DispatchPolicy, TailTile,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>, TailMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    typename TailOperandB::GmemTiledCopy, typename TailOperandB::SmemLayoutAtom,
    typename TailOperandB::SmemCopyAtom, cute::identity>;
static_assert(sizeof(typename TailCollective::SharedStorage) <= GemmKernel::SharedStorageSize);
extern "C")");
            kernelType = fmt::format("BF16GemmNarrowNTailKernel810E<{}, {}, {}>", kernelType,
                fmt::format("deep_gemm::BF16GemmMainloop810E<TailCollective, {}>",
                            args.peelFirstK ? "true" : "false"),
                args.tileOrder);
        }
        replace_bf16_gemm_source(code, "GemmKernel op;\n  op(params, smem);",
            fmt::format("static_assert(GemmKernel::SharedStorageSize <= {}, \"810E TSM launch storage is too small\");\n"
                        "  {} op;\n  op(params, smem);", args.launch_args.smem_size, kernelType));
        return code;
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Singleton input views share the dense/split-K parameter layout. The output
// has one plane per FP32 split-K partial. This helper performs no input copies.
template <typename Output, bool SplitK>
static void bf16_gemm_tsm_impl_810e(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                    int bm, int bn, int bk, int wm, int wn, int stages, int splits, int sms,
                                    int tileOrder = 0,
                                    bool peelFirstK = false,
                                    bool unrollK = false,
                                    bool narrowNTail = false) {
    const int m = a.size(1), n = b.size(1), k = a.size(2);
    TORCH_CHECK(a.size(0) == 1 && b.size(0) == 1, "810E TSM expects singleton dense input views");
    const int groups = SplitK ? splits : 1;
    TORCH_CHECK(!unrollK || (!SplitK && peelFirstK && k <= 512 && k % bk == 0 &&
                             k / bk >= stages && k / bk <= 16 && !is_ppu1v5_device() &&
                             std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos),
                "Invalid dense K-loop count");
    const auto pow2 = [](int x) { return x > 0 && (x & (x - 1)) == 0; };
    // This mainloop advances its pipeline at K_BLOCK_MAX-2. BK16 has only
    // one register K block on 810E and cannot advance that state machine.
    // Non-power-of-two M tiles keep one or two M warps and four N warps.
    // The vector epilogue requires this power-of-two warp grid, not power-of-two BM.
    const bool compactM = bn == 128 && wn == 32 &&
        (stages == 2 || stages == 3) &&
        ((bk == 64 && ((bm == 80 && wm == 80) || (bm == 96 && wm == 48) ||
                       (bm == 112 && wm == 112) || (bm == 160 && wm == 80) ||
                       (bm == 192 && wm == 96) || (bm == 224 && wm == 112))) ||
         (bk == 128 && ((bm == 160 && wm == 80) || (bm == 192 && wm == 96))));
    TORCH_CHECK(bm >= 16 && bn >= 16 && bk >= 32 && (pow2(bm) || compactM) && pow2(bn) && pow2(bk) &&
                wm >= 16 && wn >= 16 && bm % wm == 0 && bn % wn == 0 &&
                stages >= 2 && stages <= 8, "invalid 810E TSM tile (requires BK>=32)");
    // A complete K product may use AIU padding for its last tile only when
    // the descriptor width is the logical K. Padded input rows and split-K
    // partitions retain the complete-tile requirement.
    const bool fullKTail = !SplitK && k % 32 == 0 && a.stride(1) == k && b.stride(1) == k;
    TORCH_CHECK(splits >= 1 && splits <= 64 && (k % bk == 0 || fullKTail) &&
                (k / bk) / splits >= stages,
                "810E TSM requires complete prologue tiles and aligned split-K or contiguous full-K inputs");
    const int threads = (bm / wm) * (bn / wn) * 32;
    TORCH_CHECK(!narrowNTail || (!SplitK && groups == 1 && splits == 1 &&
                                 sms >= get_tensor_cu_count_810e() && bm == 128 && bn == 128 && bk == 64 &&
                                 threads == 256 && (tileOrder == 0 || tileOrder == 1) &&
                                 n % 128 > 0 && n % 128 <= 16 && !unrollK),
                "Invalid 810E narrow N-tail launch");
    const int64_t smem = get_smem_size_bf16_810e(bm, bn, bk, wm, stages);
    TORCH_CHECK(threads <= device_runtime->get_prop()->maxThreadsPerBlock &&
                smem <= device_runtime->get_prop()->sharedMemPerMultiprocessor,
                "810E TSM tile exceeds device resources");
    using Runtime = BF16GemmTsmRuntime810ET<Output, SplitK>;
    using Base = typename Runtime::Base;
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    const Stride sa{a.stride(1), cute::Int<1>{}, a.stride(0)};
    const Stride sb{b.stride(1), cute::Int<1>{}, b.stride(0)};
    const Stride sd{d.stride(1), cute::Int<1>{}, d.stride(0)};
    auto* output = static_cast<Output*>(d.data_ptr());
    cutlass::KernelHardwareInfo hw{};
    hw.device_id = a.get_device(); hw.cu_count = sms;
    typename Base::GemmArguments gemmArgs{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, groups},
        .mainloopargs = {reinterpret_cast<const cutlass::bfloat16_t*>(a.data_ptr<at::BFloat16>()), sa,
                         reinterpret_cast<const cutlass::bfloat16_t*>(b.data_ptr<at::BFloat16>()), sb},
        .epilogueargs = {{1, 0}, output, sd, output, sd},
        .hw_info = hw, .scheduler = {(uint32_t)m, nullptr}, .signal = nullptr};
    const std::string name = SplitK ? "bf16_gemm_tsm_split_k_810e" : "bf16_gemm_tsm_810e";
    typename Base::Args baseArgs{
        .launch_info = {bm, bn, bk, wm, wn, 1, stages, "Normal", "Default", name, false},
        .launch_args = {tileOrder == 1 ? dim3(ceil_div(m, bm), ceil_div(n, bn), groups) :
                                      dim3(ceil_div(n, bn), ceil_div(m, bm), groups), dim3(threads), int(smem)},
        .kernel_params = Base::to_underlying_arguments_rtc(gemmArgs, nullptr)};
    typename Runtime::Args args{baseArgs, tileOrder, peelFirstK,
                               unrollK, narrowNTail};
    if (sms < get_tensor_cu_count_810e()) {
        TORCH_CHECK(sms > 0, "810E TSM requires a positive CU budget");
        const int64_t tiles = int64_t(ceil_div(m, bm)) * ceil_div(n, bn) * groups;
        args.launch_args.grid_dim = dim3(std::min<int64_t>(sms, tiles));
    }
    check_bf16_gemm_grid_810e(args.launch_args.grid_dim);
    auto code = Runtime::generate(args);
    auto runtime = compiler->build(name, code);
    Runtime::launch(runtime, args);
    show_bf16_gemm_launch_810e(args, m, n, k, sms);
}

template <typename Output>
static void bf16_gemm_tsm_810e(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                               int bm, int bn, int bk, int wm, int wn, int stages, int splits,
                               int tileOrder = 0,
                               bool peelFirstK = false,
                               bool unrollK = false,
                               bool narrowNTail = false) {
    TORCH_CHECK(!unrollK || splits == 1, "Dense K unrolling requires a complete K product");
    const int m = a.size(0), n = b.size(0), k = a.size(1);
    const int alignment = 16 / sizeof(Output);
    // Standard EVT predicates only the vector's first element. Keep narrow or
    // unaligned output rows on the audited per-element path, including N=4.
    if (n < 8 || n % alignment != 0 || (reinterpret_cast<uintptr_t>(d.data_ptr()) & 15) != 0) {
        bf16_gemm_direct_810e<Output>(a, b, d, bm, bn, bk, wm, wn, stages, splits);
        return;
    }
    TORCH_CHECK(splits >= 1 && splits <= 64, "810E TSM requires 1<=splits<=64");
    const auto av = a.unsqueeze(0), bv = b.unsqueeze(0);
    if (splits == 1) {
        // Complete BK tiles let the deep, large-M pipeline eliminate K-tail
        // bookkeeping. A single partition still writes the final output once.
        // Other tiles retain the complete-K path, including padded K tails.
        if (bm >= 256 && stages >= 4 && k % bk == 0 && !unrollK)
            bf16_gemm_tsm_impl_810e<Output, true>(av, bv, d.unsqueeze(0), bm, bn, bk, wm, wn, stages, 1,
                                                  get_num_tensor_sms_810e(), tileOrder, peelFirstK);
        else
            bf16_gemm_tsm_impl_810e<Output, false>(av, bv, d.unsqueeze(0), bm, bn, bk, wm, wn, stages, 1,
                                                   get_num_tensor_sms_810e(), tileOrder, peelFirstK, unrollK, narrowNTail);
        return;
    }
    auto partial = torch::empty({int64_t(splits), int64_t(m), int64_t(n)}, a.options().dtype(torch::kFloat));
    bf16_gemm_tsm_impl_810e<float, true>(av, bv, partial, bm, bn, bk, wm, wn, stages, splits, get_num_tensor_sms_810e(),
                                         tileOrder, peelFirstK);
    using Reduce = BF16GemmSplitKReduceRuntime810ET<Output>;
    const int64_t count = int64_t(m) * n;
    typename Reduce::Args args{Reduce::launch_args(count),
                              {partial.data_ptr<float>(), d.data_ptr(), count}, splits};
    auto code = Reduce::generate(args);
    auto runtime = compiler->build("bf16_gemm_split_k_reduce_810e", code);
    Reduce::launch(runtime, args);
}

static bool is_810e_device() {
    return std::string(device_runtime->get_prop()->name).find("ZW810E") != std::string::npos;
}

template <typename Output>
static void bf16_gemm_impl_810e(const ConfigResult810E& entry,
                                const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    const auto& t = entry.tile;
    if (entry.kind == BF16GemmKernelType810E::Standard) {
        bf16_gemm_impl<Output>(a, b, d, int(entry.m), int(entry.n), int(entry.k), get_best_configs_810e(t));
        return;
    }
    DgProfParam params;
    if (profiling_enabled())
        params.set_params(GemmType::Normal, false, d.scalar_type() == torch::kFloat ? "bf16_bf16_fp32" : "bf16",
                          1, int(entry.m), int(entry.n), int(entry.k), 0, nullptr, current_stream());
    auto profilingGuard = show_gemm_ut(params);
    switch (entry.kind) {
        case BF16GemmKernelType810E::Direct:
            bf16_gemm_direct_810e<Output>(a, b, d, t.bm, t.bn, t.bk, t.wm, t.wn, t.stages, t.splits);
            return;
        case BF16GemmKernelType810E::TSM:
            bf16_gemm_tsm_810e<Output>(a, b, d, t.bm, t.bn, t.bk, t.wm, t.wn, t.stages, t.splits,
                                       t.tileOrder, t.peelFirstK != 0,
                                       t.unrollK != 0, t.narrowNTail);
            return;
        case BF16GemmKernelType810E::Hybrid: {
            const auto& tail = entry.tailTile;
            DG_HOST_ASSERT(entry.splitM > 0 && entry.splitM < entry.m);
            DG_HOST_ASSERT(t.splits == 1 && tail.splits == 1);
            bf16_gemm_tsm_810e<Output>(a.narrow(0, 0, entry.splitM), b,
                d.narrow(0, 0, entry.splitM), t.bm, t.bn, t.bk, t.wm, t.wn, t.stages, 1,
                t.tileOrder, t.peelFirstK != 0,
                t.unrollK != 0);
            bf16_gemm_tsm_810e<Output>(a.narrow(0, entry.splitM, entry.m - entry.splitM), b,
                d.narrow(0, entry.splitM, entry.m - entry.splitM),
                tail.bm, tail.bn, tail.bk, tail.wm, tail.wn, tail.stages, 1,
                tail.tileOrder, tail.peelFirstK != 0,
                tail.unrollK != 0);
            return;
        }
        case BF16GemmKernelType810E::Standard:
            DG_HOST_UNREACHABLE("Standard configuration is handled before custom dispatch");
    }
}

static bool bf16_gemm_dispatch_810e(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    if (!is_810e_device()) return false;
    // Compare the Tensor budget against the Tensor pool, not general CUs. Keep ordinary
    // low-budget products on the standard persistent launcher. Its vector
    // epilogue cannot cover partial vectors, so those use persistent Direct.
    const int outputAlignment = d.scalar_type() == torch::kFloat ? 4 : 8;
    const bool perElementOutput = b.size(0) % outputAlignment != 0 ||
                                   (reinterpret_cast<uintptr_t>(d.data_ptr()) & 15) != 0;
    const bool lowBudget = get_num_tensor_sms_810e() < get_tensor_cu_count_810e();
    if (a.size(0) <= 2 && gemv_dense_bf16(a, b, d, int(a.size(0)), int(b.size(0)), int(a.size(1))))
        return true;
    if (lowBudget && !perElementOutput) return false;
    // Custom AIU mainloops require complete BK32 tiles and two prologue tiles.
    const bool supportedInputs = a.size(1) >= 64 && a.size(1) % 32 == 0 &&
        (reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0;
    TORCH_CHECK(!perElementOutput || supportedInputs,
                "810E partial-vector outputs outside the GEMV path require K>=64, K%32=0 and aligned inputs");
    if (!supportedInputs || b.size(0) < 2) return false;
    const auto output = d.scalar_type() == torch::kFloat ? torch::kFloat : torch::kBFloat16;
    auto plan = get_adaptive_configs_bf16_810e(a.size(0), b.size(0), a.size(1), output,
                                               perElementOutput, lowBudget);
    const auto gridFits = [&](const BF16GemmConfig810E& t, int64_t rows) {
        if (lowBudget) return true;  // A one-dimensional grid of at most sms CTAs.
        const int64_t mt = ceil_div(rows, int64_t(t.bm)), nt = ceil_div(b.size(0), int64_t(t.bn));
        const auto prop = device_runtime->get_prop();
        return (t.tileOrder == 1 ? mt : nt) <= prop->maxGridSize[0] &&
               (t.tileOrder == 1 ? nt : mt) <= prop->maxGridSize[1] &&
               t.splits <= prop->maxGridSize[2];
    };
    // The existing persistent launcher also covers outputs whose tile grid
    // exceeds a physical grid dimension.
    if (plan.kind != BF16GemmKernelType810E::Standard &&
        (!gridFits(plan.tile, plan.splitM ? plan.splitM : a.size(0)) ||
         (plan.splitM && !gridFits(plan.tailTile, a.size(0) - plan.splitM)))) {
        TORCH_CHECK(b.size(0) >= 8 && b.size(0) % outputAlignment == 0 &&
                    (reinterpret_cast<uintptr_t>(d.data_ptr()) & 15) == 0,
                    "810E unaligned output exceeds the direct kernel grid limits");
        return false;
    }
    if (output == torch::kFloat)
        bf16_gemm_impl_810e<float>(plan, a, b, d);
    else
        bf16_gemm_impl_810e<cutlass::bfloat16_t>(plan, a, b, d);
    return true;
}


static void bf16_gemm_810e(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                           int m, int n, int k, std::optional<ConfigTuple> configs) {
    if (n == 1 && !configs.has_value() && gemv_dense_bf16_810e(a, b, d)) return;
    if (!configs.has_value() && bf16_gemm_dispatch_810e(a, b, d)) return;
    TORCH_CHECK(is_ppu1v5_device() || n % 2 == 0, "BF16 dense GEMM on PPU1.0 requires even N, got ", n);
    if (!is_ppu1v5_device()) {
        TORCH_CHECK(k >= 64 && k % 32 == 0 &&
                    (reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
                    (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0,
                    "BF16 tiled GEMM requires K>=64, K%32=0 and 16-byte-aligned inputs");
        TORCH_CHECK(n % (16 / d.element_size()) == 0 && (reinterpret_cast<uintptr_t>(d.data_ptr()) & 15) == 0,
                    "BF16 tiled GEMM requires vector-aligned output");
        if (!configs) configs = get_best_configs_810e(m, n, k, get_num_tensor_sms_810e());
        check_configs_810e(configs, k);
    }
    if (d.scalar_type() == torch::kFloat)
        bf16_gemm_impl<float>(a, b, d, m, n, k, configs);
    else
        bf16_gemm_impl<cutlass::bfloat16_t>(a, b, d, m, n, k, configs);
}

static void bf16_gemm(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                      const int& m, const int& n, const int& k,
                      std::optional<ConfigTuple> configs = std::nullopt,
                      const std::optional<torch::Tensor>& c = std::nullopt) {
    if (!is_ppu1v5_device()) {
        configs = get_runtime_configs_bf16_810e(m, n, k, configs);
        check_configs_810e(configs, k);
        if (configs && !c)
            TORCH_CHECK(n % (16 / d.element_size()) == 0 &&
                        (reinterpret_cast<uintptr_t>(d.data_ptr()) & 15) == 0,
                        "BF16 dense explicit/LUT configs require vector-aligned output");
    }
    if (c) {
        // Preserve C until the FP32 product is complete, including C == D.
        // The only BF16 conversion follows addition; compile-only preserves C/D.
        auto accum = torch::empty({m, n}, d.options().dtype(torch::kFloat));
        bf16_gemm(a, b, accum, m, n, k, configs);
        if (!is_compile_only()) {
            accum.add_(*c);
            d.copy_(accum);
        }
        return;
    }
    if (!is_ppu1v5_device())
        bf16_gemm_810e(a, b, d, m, n, k, configs);
    else if (d.scalar_type() == torch::kFloat)
        bf16_gemm_impl<float>(a, b, d, m, n, k, configs);
    else
        bf16_gemm_impl<cutlass::bfloat16_t>(a, b, d, m, n, k, configs);
}

} // namespace deep_gemm
