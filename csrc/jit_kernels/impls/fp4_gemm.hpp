#pragma once

#include <torch/python.h>
#include <cstdint>
#include <cstring>
#include <string>
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
#include "../heuristics/common_fp4.hpp"
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include <deep_gemm/common/utils_rtc.cuh>
#include <deep_gemm/common/profiling_interface.cuh>

using namespace deep_gemm_fp4_common;
namespace deep_gemm {

class FP4GemmRuntime final : public LaunchRuntime<FP4GemmRuntime> {
public:
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
    using GemmProblemSize = cute::tuple<int32_t, int32_t, int32_t, int32_t>;

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n, num_groups, num_stages;
        int n, k;
        std::string gemm_type, kernel_name;
        bool hasBias;
        int n_expand;
        bool kEnableSboOverlap;
        // Epilogue selection. "Default" or "SiluAndMulPostQuantFp4"; apply_swiglu_limit only
        // matters for the fused epilogue.
        std::string epilogue_type = "Default";
        bool apply_swiglu_limit = false;
    };

    struct MainLoopArguments {
        cute::Shape<int, int, int> problem_shape;
        uint8_t* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        uint8_t* ptr_B;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
        uint16_t* ptr_scale_A;
        cute::Stride<cute::Int<1>, int64_t, int64_t> stride_SFA;
        uint16_t* ptr_scale_B;
        cute::Stride<cute::Int<1>, int64_t, int64_t> stride_SFB;
    };

    // One host mirror per device-side CollectiveEpilogue, matching the three types selected in
    // Fp4Gemm::run (fp4_gemm_cutlass3.cuh).
    //
    //   device type                             condition                    Epilogue::Params
    //   CollectiveEpilogueWithTsm               hasBias || SHAPE_N % 2 != 0         96 B
    //   CollectiveEpilogueNoTsm                 fast path                          136 B
    //   CollectiveEpilogueSiluAndMulPostQuant   fused, derives from NoTsm          152 B

    // hasBias || SHAPE_N % 2 != 0 -> CollectiveEpilogueWithTsm
    struct EpilogueArgsWithTsm {
        struct {
            float alpha = 1.0f;
            float beta = 0.0f;
            float const* alpha_ptr = nullptr;
            float const* beta_ptr = nullptr;
            char _pad[8] = {};  // EmptyArguments(1B) + 7B padding = 32B total
        } thread;
        float const* ptr_C = nullptr;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;
        cutlass::bfloat16_t* ptr_D = nullptr;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
        float const* ptr_Bias = nullptr;  // null when hasBias == false
        cute::Stride<cute::Int<0>, cute::Int<1>, int64_t> stride_Bias;
    };

    // fast path -> CollectiveEpilogueNoTsm
    struct EpilogueArgsNoTsm {
        struct {
            float alpha = 1.0f;
            float beta = 0.0f;
            float const* alpha_ptr = nullptr;
            float const* beta_ptr = nullptr;
            float const* const* alpha_ptr_array = nullptr;
            float const* const* beta_ptr_array = nullptr;
            // SUPPORT_FP8_SCALING fields (48B)
            float scale_a = 1.0f;
            float scale_b = 1.0f;
            float scale_c = 1.0f;
            float scale_d = 1.0f;
            float const* scale_a_ptr = nullptr;
            float const* scale_b_ptr = nullptr;
            float const* scale_c_ptr = nullptr;
            float const* scale_d_ptr = nullptr;
        } thread;  // 88 bytes
        float const* ptr_C = nullptr;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;
        cutlass::bfloat16_t* ptr_D = nullptr;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    // fused -> CollectiveEpilogueSiluAndMulPostQuant, whose Arguments derives from the NoTsm ones
    // and appends ptr_SFD / shape_m / swiglu_limit.
    struct EpilogueArgsSiluAndMulPostQuant {
        struct {
            float alpha = 1.0f;
            float beta = 0.0f;
            float const* alpha_ptr = nullptr;
            float const* beta_ptr = nullptr;
            float const* const* alpha_ptr_array = nullptr;
            float const* const* beta_ptr_array = nullptr;
            // SUPPORT_FP8_SCALING fields (48B)
            float scale_a = 1.0f;
            float scale_b = 1.0f;
            float scale_c = 1.0f;
            float scale_d = 1.0f;
            float const* scale_a_ptr = nullptr;
            float const* scale_b_ptr = nullptr;
            float const* scale_c_ptr = nullptr;
            float const* scale_d_ptr = nullptr;
        } thread;  // 88 bytes
        float const* ptr_C = nullptr;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;
        uint8_t* ptr_D = nullptr;  // EpilogueTraits<SiluAndMulPostQuantFp4>::ElementD
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
        uint16_t* ptr_SFD = nullptr;
        uint32_t shape_m = 0;
        float swiglu_limit = 0.0f;
    };

    // Pin the three layouts above. The device-side sizes were read back from hgcc on the generated
    // kernel; the fused one is expressed relative to NoTsm because its Arguments literally derives
    // from them and appends three fields.
    static_assert(sizeof(EpilogueArgsWithTsm) == 96, "WithTsm epilogue params layout changed");
    static_assert(sizeof(EpilogueArgsNoTsm) == 136, "NoTsm epilogue params layout changed");
    static_assert(sizeof(EpilogueArgsSiluAndMulPostQuant) ==
                      sizeof(EpilogueArgsNoTsm) + sizeof(uint16_t*) + sizeof(uint32_t) + sizeof(float),
                  "fused epilogue params must be the NoTsm layout plus ptr_SFD/shape_m/swiglu_limit");

    template <typename EpilogueArgsT>
    struct GemmKernelParamsT {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        MainLoopArguments collective_mainloop_params;
        EpilogueArgsT collective_epilogue_params;
        cutlass::KernelHardwareInfo hw_info;
        TileSchedulerArguments scheduler;
        void* workspace{nullptr};
        int32_t* signal{nullptr};
        // actlize_v1.0.0 sets SAIL_SYNC_IN_CE=1 (accutlass.hpp), which appends this field to the
        // N_EXPAND == 1 specialization of DeepGemmUniversal::Params. Never read here, but must stay:
        // the driver copies sizeof(device Params) bytes out of this struct.
        int32_t* ptr_sync_ce{nullptr};
    };

    union KernelParams {
        GemmKernelParamsT<EpilogueArgsWithTsm> with_tsm;
        GemmKernelParamsT<EpilogueArgsNoTsm> no_tsm;
        GemmKernelParamsT<EpilogueArgsSiluAndMulPostQuant> silu_and_mul_post_quant;
    };

    // fused -> CollectiveEpilogueSiluAndMulPostQuant, whose Arguments derives from the NoTsm ones
    // and appends ptr_SFD / shape_m / swiglu_limit.
    template <typename EpilogueArgsT>
    struct DynamicTileParamsT {
        uint8_t* ptr_A = nullptr;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        uint8_t* ptr_B = nullptr;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
        uint16_t* ptr_scale_A = nullptr;
        cute::Stride<cute::Int<1>, int64_t, int64_t> stride_SFA;
        uint16_t* ptr_scale_B = nullptr;
        cute::Stride<cute::Int<1>, int64_t, int64_t> stride_SFB;
        EpilogueArgsT epi_params;
        uint32_t shape_m = 0;
        int32_t* grouped_layout = nullptr;
    };

    union DynamicTileKernelParams {
        DynamicTileParamsT<EpilogueArgsNoTsm> dynamic_tile_no_tsm;
        DynamicTileParamsT<EpilogueArgsSiluAndMulPostQuant> dynamic_tile_silu_and_mul_post_quant;
    };

    static_assert(sizeof(DynamicTileParamsT<EpilogueArgsNoTsm>) == 248
        && sizeof(DynamicTileParamsT<EpilogueArgsSiluAndMulPostQuant>) == 264, "DynamicTile params layout changed");
    static_assert(offsetof(DynamicTileParamsT<EpilogueArgsNoTsm>, epi_params) == 96
        && offsetof(DynamicTileParamsT<EpilogueArgsSiluAndMulPostQuant>, epi_params) == 96, "DynamicTile params layout changed");
    static_assert(offsetof(DynamicTileParamsT<EpilogueArgsNoTsm>, shape_m) == 232
        && offsetof(DynamicTileParamsT<EpilogueArgsSiluAndMulPostQuant>, shape_m) == 248, "DynamicTile shape_m moved");
    static_assert(offsetof(DynamicTileParamsT<EpilogueArgsNoTsm>, grouped_layout) == 240
        && offsetof(DynamicTileParamsT<EpilogueArgsSiluAndMulPostQuant>, grouped_layout) == 256, "DynamicTile grouped_layout moved");

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        // Mirrors the guard in Fp4Gemm::run: the fused epilogue reuses the mainloop's shared storage,
        // so its activation tile must fit in what the kernel already reserves.
        // Emitted only for the fused variant instead of using `if constexpr` in the generated kernel:
        // everything there is concrete, so naming ::get_shared_storage_size() in a discarded branch
        // would still instantiate the DefaultEpilogueNoTsm base and trip `kOutputAlignment == 2`
        // whenever AlignmentD is 1 (the bias / odd-N configurations).
        const std::string fused_smem_guard =
            args.launch_info.epilogue_type == "SiluAndMulPostQuantFp4"
                ? "static_assert(CollectiveEpilogueSiluAndMulPostQuant::get_shared_storage_size(BLOCK_M, BLOCK_N)\n"
                  "                  <= GemmKernel::SharedStorageSize,\n"
                  "              \"fused epilogue act tile exceeds the kernel shared storage\");\n"
                : "";
        return fmt::format(
            R"(
#include <deep_gemm/impls/fp4_gemm_cutlass3.cuh>
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
static constexpr bool kEnableSboOverlap = {};
static constexpr bool hasBias = {};
constexpr int N_EXPAND = {};
static constexpr EpilogueType kEpilogueType = EpilogueType::{};
static constexpr bool kApplySwigluLimit = {};

// A matrix configuration
using         ElementA    = cutlass::float4_t;
using         LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 1;

// B matrix configuration
using         ElementB    = cutlass::float4_t;
using         LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 1;

// C matrix configuration
using         ElementC    = float;
using         LayoutC     = cutlass::layout::RowMajor;
constexpr int AlignmentC  = 1;

// D matrix configuration
using         ElementD    = typename EpilogueTraits<kEpilogueType>::ElementD;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Core kernel configurations
using ElementAccumulator  = float;
using ElementCompute      = float;
using ElementBias         = float;
using ElementScalar       = ElementCompute;

using WarpOnM = Int<BLOCK_M / WARP_M>;
using WarpOnN = Int<BLOCK_N / WARP_N>;
static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
static constexpr bool TransA = false;
static constexpr bool TransB = false;
using ArchTag = cutlass::arch::PPU0015;

using DispatchPolicy = cutlass::gemm::MainloopWithScalePPU0015Aiu<STAGES>;
using GemmOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false>;
using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true>;

using TransformA = cute::identity;
using TransformB = cute::identity;

// Use Aiu for SFA
using ElementSFA = uint16_t;
using LayoutSFA = cutlass::layout::ColumnMajor;
constexpr bool TransSFA = true;
constexpr int MinAiuContElemSize = 32 / (sizeof_bits<ElementSFA>::value / 8);

constexpr int ScaleGranularityK = 32;
constexpr int ScaleMsPerTile = BLOCK_M;
constexpr int ScaleKsPerTile = BLOCK_K / ScaleGranularityK;

constexpr int SFATileM = TransSFA ? cute::max(ScaleMsPerTile, MinAiuContElemSize) : ScaleMsPerTile;
constexpr int SFATileK = TransSFA ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);

constexpr bool swap = true;
constexpr int StageStride = 0;
constexpr bool swzl = false;
using GemmOperandSFA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementSFA, TransSFA, Int<SFATileM>, Int<SFATileK>, swap, StageStride, swzl>;

// Use Aiu for SFB
using ElementSFB = uint16_t;
using LayoutSFB = cutlass::layout::RowMajor;
constexpr bool TransSFB = true;
constexpr int MinAiuContElemSizeSFB = 32 / (sizeof_bits<ElementSFB>::value / 8);

constexpr int ScaleNsPerTile = BLOCK_N;
constexpr int ScaleKsPerTileSFB = ScaleKsPerTile;

constexpr int SFBTileN = TransSFB ? cute::max(ScaleNsPerTile, MinAiuContElemSizeSFB) : ScaleNsPerTile;
constexpr int SFBTileK = TransSFB ? ScaleKsPerTileSFB : cute::max(ScaleKsPerTileSFB, MinAiuContElemSizeSFB);

using GemmOperandSFB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementSFB, TransSFB, Int<SFBTileN>, Int<SFBTileK>, swap, StageStride, swzl>;

using MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;

using TiledMma = cute::TiledMMA<
    cute::MMA_Atom<MmaInst>,
    cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMmaScaleFp4<
    DispatchPolicy, Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
    typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB,
    ElementSFA, cutlass::detail::TagToStrideA_t<LayoutSFA>, typename GemmOperandSFA::GmemTiledCopy, typename GemmOperandSFA::SmemLayoutAtom,
    ElementSFB, cutlass::detail::TagToStrideB_t<LayoutSFB>, typename GemmOperandSFB::GmemTiledCopy, typename GemmOperandSFB::SmemLayoutAtom>;

using EpilogueOutputOp = typename cutlass::platform::conditional<
    hasBias || (SHAPE_N % 2 != 0),
    cutlass::epilogue::thread::LinearCombinationBiasElementwise<ElementD, ElementAccumulator, ElementCompute, ElementD, ElementD, AlignmentD, cutlass::epilogue::thread::Identity<float>, cutlass::plus<ElementCompute>, false, ElementBias>,
    cutlass::epilogue::thread::LinearCombination<ElementD, 2, ElementAccumulator, ElementCompute, cutlass::epilogue::thread::ScaleType::Nothing, cutlass::FloatRoundStyle::round_to_nearest, ElementC>
  >::type;

using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<AlignmentC * sizeof(ElementC) * 8>;
using GemmEpilogueConfiguration = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, float, AlignmentC, Int<BLOCK_M>, Int<BLOCK_N>, WarpOnM, ThreadNum>;
static constexpr bool IsAligedN = SHAPE_N % BLOCK_N == 0 ? true : false;

using CollectiveEpilogueWithTsm = typename cutlass::epilogue::collective::Epilogue<
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOutputOp,
    typename GemmEpilogueConfiguration::SmemLayoutO,
    Copy_Atom<EpilogueCopyInst,float>,
    typename GemmEpilogueConfiguration::GmemTiledCopyO,
    Copy_Atom<EpilogueCopyInst,ElementC>
>;

using CollectiveEpilogueNoTsm = typename cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOutputOp,
    cutlass::gemm::EpilogueDefault,
    IsAligedN
>;

using CollectiveEpilogueSiluAndMulPostQuant = typename cutlass::epilogue::collective::EpilogueSiluAndMulPostQuant<
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOutputOp,
    cutlass::gemm::EpilogueDefault,
    IsAligedN,
    kApplySwigluLimit
>;

// Keep the selection identical to Fp4Gemm::run in fp4_gemm_cutlass3.cuh
using CollectiveEpilogue = typename cutlass::platform::conditional<
    hasBias || (SHAPE_N % 2 != 0),
    CollectiveEpilogueWithTsm,
    typename cutlass::platform::conditional<
        kEpilogueType == EpilogueType::SiluAndMulPostQuantFp4,
        CollectiveEpilogueSiluAndMulPostQuant,
        CollectiveEpilogueNoTsm
    >::type
>::type;

// NOTE: kEpilogueType must be passed explicitly -- the scheduler derives SHAPE_N_OUT and
// curr_offset_c_scales() from it, so leaving it at the default would silently break the
// fused epilogue's SFD addressing.
using TileScheduler = DeepGemmScheduler<
    kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, NUM_GROUPS,
    ceil_div(SHAPE_N, BLOCK_N * N_EXPAND), 2, kEpilogueType>;

using GemmKernel = typename deep_gemm::DeepGemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler,
    hasBias,
    N_EXPAND,
    kEnableSboOverlap,
    kEpilogueType
>;

// This kernel declares `extern __shared__`, so its dynamic smem size comes from the launch, and the
// host can only supply an estimate (get_smem_config_fp4) -- it cannot see SharedStorageSize.
// Assert the two agree: an under-estimate corrupts memory, an over-estimate silently costs
// occupancy. The Python path is immune because Fp4Gemm::run launches with SharedStorageSize
// directly, so a drift here would show up as "Python correct, C++ JIT wrong".
static_assert(GemmKernel::SharedStorageSize == {},
              "host get_smem_config_fp4() disagrees with the kernel's SharedStorageSize -- "
              "update csrc/jit_kernels/heuristics/common_fp4.hpp");

// Fused-epilogue guard, emitted below only when kEpilogueType is SiluAndMulPostQuantFp4: that
// epilogue reuses the mainloop's shared storage, so its activation tile has to fit inside
// GemmKernel::SharedStorageSize. Nothing is emitted for the other epilogues -- naming
// CollectiveEpilogueSiluAndMulPostQuant here would instantiate it for every variant (see
// `fused_smem_guard` in generate_impl for why `if constexpr` cannot be used instead).
{}
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
            args.launch_info.n, args.launch_info.k,
            args.launch_info.block_m, args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_groups,
            args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.num_stages, args.launch_info.gemm_type,
            args.launch_info.kEnableSboOverlap, args.launch_info.hasBias, args.launch_info.n_expand,
            args.launch_info.epilogue_type, args.launch_info.apply_swiglu_limit,
            args.launch_args.smem_size,
            fused_smem_guard,
            args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& configs, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, configs, args.kernel_params));
    }
};

// ============================================================================
// MoE dynamic-tile kernel
// ============================================================================
// Kept as a separate runtime rather than a branch inside FP4GemmRuntime: Fp4DeepGemmDynamicTile
// shares no template parameters with DeepGemmUniversal and builds its collectives and tile
// scheduler internally, so the generated code needs none of the CollectiveMainloop /
// CollectiveEpilogue / TileScheduler aliases. Emitting one combined template would instantiate
// types the dynamic-tile kernel never uses and make the placeholder list far harder to keep in sync.
//
// Mirrors the `if constexpr (kEnableMoeDynamicTile)` branch of Fp4Gemm::run. Only the Default
// epilogue is supported there (the kernel static_asserts it), which is why this path reuses
// FP4GemmRuntime::EpilogueArgsNoTsm and does not carry a fused variant.
class FP4DynamicTileRuntime final : public LaunchRuntime<FP4DynamicTileRuntime> {
public:
    struct LaunchInfo {
        int n, k, num_groups;
        // FP4DynamicTileId enumerator name: "LargeEM", "LargeK", "LargeK_G2" or "SmallEM".
        std::string dynamic_tile_id;
        std::string gemm_type, kernel_name;
        // Epilogue selection. "Default" or "SiluAndMulPostQuantFp4"; apply_swiglu_limit only
        // matters for the fused epilogue.
        std::string epilogue_type = "Default";
        bool apply_swiglu_limit = false;
    };

    using DynamicTileKernelParams = FP4GemmRuntime::DynamicTileKernelParams;

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        DynamicTileKernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <deep_gemm/impls/fp4_gemm_cutlass3_dynamic.cuh>
namespace deep_gemm {{
using namespace cute;

constexpr int SHAPE_N = {};
constexpr int SHAPE_K = {};
constexpr int NUM_GROUPS = {};
static constexpr auto kDynamicTileId = FP4DynamicTileId::{};
static constexpr GemmType kGemmType = GemmType::{};
static constexpr EpilogueType kEpilogueType = EpilogueType::{};
static constexpr bool kApplySwigluLimit = {};

using ElementD = typename EpilogueTraits<kEpilogueType>::ElementD;
// The dynamic-tile kernel picks its own tile shape from kDynamicTileId, so no block/warp constants
// are needed here. Element types must still match what the host filled into DynamicTileKernelParams.
using GemmKernel = cutlass::gemm::kernel::Fp4DeepGemmDynamicTile<
    kGemmType,
    cutlass::float4_t,      // ElementA
    cutlass::float4_t,      // ElementB
    float,                  // ElementC
    ElementD,               // ElementD
    float,                  // ElementAccumulator
    float,                  // ElementCompute
    SHAPE_N, SHAPE_K, NUM_GROUPS,
    kDynamicTileId,
    kEpilogueType,
    kApplySwigluLimit
>;

// The dynamic-tile kernel decides its own tile shape from kDynamicTileId, so get_smem_config_fp4()
// cannot compute SharedStorageSize. The host looks up a table (dynamic_tile_shared_storage_size)
// populated by hgcc probing each variant; this static_assert pins the expectation so any device-
// side change fails to compile rather than corrupting memory at launch.
static_assert(GemmKernel::SharedStorageSize == {},
              "host dynamic_tile_shared_storage_size disagrees with the kernel's SharedStorageSize "
              "-- update the table in csrc/jit_kernels/heuristics/common_fp4.hpp");

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
            args.launch_info.n, args.launch_info.k, args.launch_info.num_groups,
            args.launch_info.dynamic_tile_id, args.launch_info.gemm_type,
            args.launch_info.epilogue_type, args.launch_info.apply_swiglu_limit,
            args.launch_args.smem_size,
            args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& configs, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, configs, args.kernel_params));
    }
};

using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
static void fp4_gemm(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                     const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                     const torch::Tensor& bias, const torch::Tensor& out,
                     const int& m, const int& n, const int& k,
                     std::optional<ConfigTuple> configs = std::nullopt) {
    int num_sms = get_num_sms();

    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp4_common::get_smem_config_fp4(nst, bm, bn, wm, wn, bk, 1));
    } else {
        selected_config = deep_gemm_fp4_common::get_best_configs(m, m, n, k, 1, num_sms);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);
    uint32_t kNumGroups = 1;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideSFA = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    using StrideSFB = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    static constexpr GemmType kGemmType = GemmType::DenseGemm;

    // A/B data strides: float4_t packed as uint8, M/N-major
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));

    // SFA is M-major (ColumnMajor), shape (m, ceil_div(k, 32))
    auto stride_SFA = cutlass::make_cute_packed_stride(StrideSFA{}, cute::make_shape(m, ceil_div(k, 32), 1));
    // SFB is N-major (transposed), shape (n, ceil_div(k, 32))
    auto stride_SFB = cutlass::make_cute_packed_stride(StrideSFB{}, cute::make_shape(n, ceil_div(k, 32), 1));

    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, 0, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    // Determine bias
    bool hasBias = bias.numel() > 0;

    // Get data pointers
    uint8_t* ptr_A = lhs.data_ptr<uint8_t>();
    uint16_t* ptr_scale_A = lhs_scales.data_ptr<uint16_t>();
    uint8_t* ptr_B = rhs.data_ptr<uint8_t>();
    uint16_t* ptr_scale_B = rhs_scales.data_ptr<uint16_t>();
    cutlass::bfloat16_t* ptr_D = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    float* ptr_C = nullptr; // C is not used for in-place output

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;

    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);

    int* grouped_layout = nullptr;

    // N_EXPAND is always 1 for Dense GEMM
    int n_expand = 1;

    FP4GemmRuntime::Args args{};
    args.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                        n, k, "DenseGemm", "fp4_deep_gemm", hasBias, n_expand, false,
                        "Default", false};
    args.launch_args = {grid, block, SMSIZE};

    FP4GemmRuntime::MainLoopArguments mainloop_params{
        cute::make_shape(m, n, k), ptr_A, stride_A, ptr_B, stride_B,
        ptr_scale_A, stride_SFA, ptr_scale_B, stride_SFB};

    // Branch exactly like the device-side CollectiveEpilogue conditional in generate_impl:
    // the TSM fallback claims `hasBias || SHAPE_N % 2 != 0` first, everything else is the fast path.
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
        params.scheduler = TileSchedulerArguments((uint32_t)m, grouped_layout);
        params.workspace = nullptr;
        params.signal = nullptr;
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
        params.scheduler = TileSchedulerArguments((uint32_t)m, grouped_layout);
        params.workspace = nullptr;
        params.signal = nullptr;
    }

    const auto& code = FP4GemmRuntime::generate(args);
    const auto& runtime = compiler->build("fp4_deep_gemm", code, block.x, SMSIZE);
    const auto& kernel = runtime->kernel;

    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("fp4"), kNumGroups, m, n, k, 0, grouped_layout,
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

        printf("[DenseGemm_FP4:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d, hasBias:%d\n", block_m, block_n, block_k,
               warp_m, warp_n, block_k, num_stages, hasBias);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }
}

} // namespace deep_gemm
