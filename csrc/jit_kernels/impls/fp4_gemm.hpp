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
#include "../../../deep_gemm/include/deep_gemm/scheduler_cutlass3.cuh"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include "../../../deep_gemm/include/deep_gemm/utils_rtc.cuh"
#include "../../../deep_gemm/include/deep_gemm/profiling_interface.hpp"

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

    // hasBias=true: LinearCombinationBiasElementwise::Params (32B) + sm70 epilogue with bias
    struct EpilogueArgsBias {
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
        float const* ptr_Bias = nullptr;
        cute::Stride<cute::Int<0>, cute::Int<1>, int64_t> stride_Bias;
    };

    // hasBias=false, N%2==0: LinearCombination::Params (88B with SUPPORT_FP8_SCALING) + DefaultEpilogueNoTsm (no bias fields)
    struct EpilogueArgsNoBias {
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

    // hasBias=false, N%2!=0: LinearCombination::Params (88B with SUPPORT_FP8_SCALING) + sm70 epilogue (has bias pointer fields, but set to nullptr)
    struct EpilogueArgsNoBiasWithTsm {
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
        float const* ptr_Bias = nullptr;
        cute::Stride<cute::Int<0>, cute::Int<1>, int64_t> stride_Bias;
    };

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
    };

    union KernelParams {
        GemmKernelParamsT<EpilogueArgsBias> bias;
        GemmKernelParamsT<EpilogueArgsNoBias> no_bias;
        GemmKernelParamsT<EpilogueArgsNoBiasWithTsm> no_bias_tsm;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <fp4_gemm_cutlass3.cuh>
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
using         ElementD    = cutlass::bfloat16_t;
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
    ArchTag, sDispatchPolicy, Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
    typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB,
    ElementSFA, cutlass::detail::TagToStrideA_t<LayoutSFA>, typename GemmOperandSFA::GmemTiledCopy, typename GemmOperandSFA::SmemLayoutAtom,
    ElementSFB, cutlass::detail::TagToStrideB_t<LayoutSFB>, typename GemmOperandSFB::GmemTiledCopy, typename GemmOperandSFB::SmemLayoutAtom>;

using EpilogueOutputOp = typename cutlass::platform::conditional<
    hasBias,
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

using CollectiveEpilogue = typename cutlass::platform::conditional<
    hasBias || (SHAPE_N % 2 != 0),
    CollectiveEpilogueWithTsm,
    CollectiveEpilogueNoTsm
>::type;

using TileScheduler = DeepGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, NUM_GROUPS>;

using GemmKernel = typename deep_gemm::DeepGemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler,
    hasBias,
    N_EXPAND,
    kEnableSboOverlap
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
            args.launch_info.n, args.launch_info.k,
            args.launch_info.block_m, args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_groups,
            args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.num_stages, args.launch_info.gemm_type,
            args.launch_info.kEnableSboOverlap, args.launch_info.hasBias, args.launch_info.n_expand,
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
    // --- Scales layout check & preprocess ---
    torch::Tensor lhs_scales_t = lhs_scales;
    torch::Tensor rhs_scales_t = rhs_scales;
    if (!check_mxfp4_scales_layout(lhs_scales_t)) {
        lhs_scales_t = preprocess_mxfp4_scales(lhs_scales_t);
    }
    if (!check_mxfp4_scales_layout(rhs_scales_t)) {
        rhs_scales_t = post_preprocess_mxfp4_scales(rhs_scales_t);
        if (!check_mxfp4_scales_layout(rhs_scales_t)) {
            rhs_scales_t = preprocess_mxfp4_scales(rhs_scales_t);
        }
    }

    // Scale stride checks (validate post-preprocessing result)
    DG_HOST_ASSERT(check_mxfp4_scales_layout(lhs_scales_t));
    DG_HOST_ASSERT(check_mxfp4_scales_layout(rhs_scales_t));

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
    uint16_t* ptr_scale_A = lhs_scales_t.data_ptr<uint16_t>();
    uint8_t* ptr_B = rhs.data_ptr<uint8_t>();
    uint16_t* ptr_scale_B = rhs_scales_t.data_ptr<uint16_t>();
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
                        n, k, "DenseGemm", "fp4_deep_gemm", hasBias, n_expand, false};
    args.launch_args = {grid, block, SMSIZE};

    FP4GemmRuntime::MainLoopArguments mainloop_params{
        cute::make_shape(m, n, k), ptr_A, stride_A, ptr_B, stride_B,
        ptr_scale_A, stride_SFA, ptr_scale_B, stride_SFB};

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
        params.scheduler = TileSchedulerArguments((uint32_t)m, grouped_layout);
        params.workspace = nullptr;
        params.signal = nullptr;
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
        params.scheduler = TileSchedulerArguments((uint32_t)m, grouped_layout);
        params.workspace = nullptr;
        params.signal = nullptr;
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
