#pragma once

// =============================================================================
// BF16 x FP32 DenseGemm (cute-free dual-B kernel + PPU1.0 fallback).
//
// Computes  D = A @ (W_high + W_low * scale)^T  with FP32 accumulation, where
// the FP32 weight is supplied as two BF16 matrices:
//     W_high = W_fp32.to(torch.bfloat16)
//     W_low  = ((W_fp32 - W_high.float()) / scale).to(torch.bfloat16)
// (scale is typically 1/256). Inputs / layouts:
//     A (lhs)     : BF16, row-major [m, k]
//     W_high / W_low : BF16, row-major [n, k]  (NT layout: rhs.T is col-major)
//     D (out)     : BF16 or FP32, row-major [m, n]
//
// PPU1.5 (890P): fused cute-free kernel -- a single interleaved K pipeline
//     per output tile, each stage carrying A + W_high + W_low together (A is
//     streamed once), two back-to-back MMAs per k_step into two FP32
//     accumulators, combined in registers and stored once.
//     Small M also uses fused dual-BF16 MMA; no K splitting or composed two-pass GEMV.
// PPU1.0 (810E): fallback composition of two BF16 GEMMs through an FP32
//     intermediate (still BF16 tensor-core rate, higher precision than
//     BF16 x BF16 because the low part recovers the residual bits).
// =============================================================================

#include <torch/python.h>
#include <cstdint>
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
#include <deep_gemm/common/profiling_interface.cuh>
#include "../heuristics/common_bf16.hpp"
#include <deep_gemm/scheduler/densegemm_scheduler_cutlass3.cuh>
#include "bf16_gemm.hpp"

using namespace deep_gemm_bf16_common;
namespace deep_gemm {

// JIT runtime for the fused dual-B cute-free kernel. Mirrors
// DenseBF16GemmCuteFreeRuntimeT; the host-side GemmKernelParams must stay
// byte-identical to kernel::GemmBf16xFp32Kernel::Params (the kernel receives
// it by value).
template <typename ElementOutput>
class DenseBf16xFp32GemmCuteFreeRuntimeT final : public LaunchRuntime<DenseBf16xFp32GemmCuteFreeRuntimeT<ElementOutput>> {
public:
    struct GemmKernelParams {
        cutlass::bfloat16_t const* ptr_A;
        cutlass::bfloat16_t const* ptr_B_high;
        cutlass::bfloat16_t const* ptr_B_low;
        ElementOutput* ptr_D;
        float low_scale;
        int M, N, K;
        int lda, ldb, ldd;
        DenseGemmTileSchedulerArguments scheduler;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n;
        int warp_k;
        int num_stages;
        std::string gemm_type, kernel_name;
        bool full_k_prefetch, early_g2s, mma_s2r_overlap, bounded_kloop;
        int kloop_unroll;
        bool kloop_chunk;
        bool overlap_prologue, valid_epilogue, packed_dual_n, static_router, static_stages, streamed_low, standard_s2, streamed_pairs;
        int min_blocks;
        bool pairs_m_first;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        const int shape_n = args.kernel_params.N;
        const bool is_aligned_n = (shape_n % info.block_n == 0);

        return fmt::format(
            R"(
#define BF16_HGRTC
#define DG_DUAL_B_FULL_K_PREFETCH {13}
#define DG_DUAL_B_EARLY_G2S {14}
#define DG_DUAL_B_MMA_S2R_OVERLAP {15}
#define DG_DUAL_B_BOUNDED_KLOOP {16}
#define DG_DUAL_B_KLOOP_UNROLL {17}
#define DG_DUAL_B_KLOOP_CHUNK {18}
#define DG_DUAL_B_OVERLAP_PROLOGUE {19}
#define DG_DUAL_B_VALID_EPILOGUE {20}
#define DG_DUAL_B_PACKED_N {21}
#define DG_DUAL_B_ROUTER_M {22}
#define DG_DUAL_B_ROUTER_N {23}
#define DG_DUAL_B_ROUTER_K {24}
#define DG_DUAL_B_STATIC_STAGES {25}
#define DG_DUAL_B_STREAMED_LOW {26}
#define DG_DUAL_B_STREAMED_PAIRS {28}
#define DG_DUAL_B_PAIRS_M_FIRST {30}
#include <deep_gemm/impls/bf16xfp32_gemm_cute_free.cuh>

namespace deep_gemm {{

using GemmTypeTag = std::integral_constant<GemmType, GemmType::{0}>;

using Kernel = BF16xFp32GemmCuteFreeKernel<{1}, {2}, {3}, {4}, {5}, {6}, {7}, {27}, {8}, GemmTypeTag{9}>;
static_assert(sizeof(typename Kernel::Params) == {11}, "host/device Params ABI mismatch");
static_assert(Kernel::SharedStorageSize == {12}, "host/device shared storage mismatch");

}} // namespace deep_gemm

extern "C"
__launch_bounds__(deep_gemm::Kernel::MaxThreadsPerBlock, {29})
__global__ void {10}(typename deep_gemm::Kernel::Params params) {{
  extern __shared__ char smem[];
  deep_gemm::Kernel kernel;
  kernel(params, smem);
}}
)",
            info.gemm_type,
            info.block_m, info.block_n, info.block_k,
            info.warp_m, info.warp_n, info.warp_k, info.num_stages,
            is_aligned_n ? "true" : "false",
            std::is_same_v<ElementOutput, float> ? ", float" : "",
            info.kernel_name, sizeof(GemmKernelParams),
            (info.num_stages * (info.block_m + (info.packed_dual_n ? 1 : 2) * info.block_n) * info.block_k * 2 + 127) & ~127,
            info.full_k_prefetch ? 1 : 0, info.early_g2s ? 1 : 0, info.mma_s2r_overlap ? 1 : 0,
            info.bounded_kloop ? 1 : 0, info.kloop_unroll, info.kloop_chunk ? 1 : 0,
            info.overlap_prologue ? 1 : 0, info.valid_epilogue ? 1 : 0, info.packed_dual_n ? 1 : 0,
            info.static_router ? args.kernel_params.M : 0,
            info.static_router ? args.kernel_params.N : 0,
            info.static_router ? args.kernel_params.K : 0,
            info.static_stages ? 1 : 0, info.streamed_low ? 1 : 0,
            info.standard_s2 ? "false" : "true", info.streamed_pairs ? 1 : 0, info.min_blocks,
            info.pairs_m_first ? 1 : 0);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Composition fallback: out = x@w_high^T + scale * x@w_low^T via two plain BF16
// GEMMs through an FP32 intermediate (BF16 tensor-core rate, FP32 accumulation;
// the low part recovers the residual bits). Only PPU1.0 uses the composed path.
static void bf16xfp32_gemm_composed(const torch::Tensor& lhs, const torch::Tensor& w_high,
                                    const torch::Tensor& w_low, const torch::Tensor& out,
                                    const double& scale, const int& m, const int& n, const int& k,
                                    std::optional<ConfigTuple> configs = std::nullopt) {
    torch::Tensor out_f32 = (out.scalar_type() == torch::kFloat)
        ? out
        : torch::empty({m, n}, out.options().dtype(torch::kFloat));
    bf16_gemm(lhs, w_high, out_f32, m, n, k, configs);
    torch::Tensor low_f32 = torch::empty({m, n}, out_f32.options());
    bf16_gemm(lhs, w_low, low_f32, m, n, k, configs);
    out_f32.add_(low_f32, scale);
    if (out.scalar_type() != torch::kFloat)
        out.copy_(out_f32);
}

// The dual-B cost tree uses only the isolated tuning set; predictions do not guarantee occupancy or performance.
// small_m comes from ff054cbc and general from d63dc371; models are trained separately without merging cross-version timings.
// New defaults still require independent validation; held-out shapes and standard BF16 configs remain untouched.
struct DualBCostNode { int feature; double value; int left, right; };
struct DualBPolicyTile { int bm, bn, bk, wm, wn, stages, resource_mask; };
static constexpr DualBCostNode kDualBSmallTree[] = {
    {12, 64.5, 1, 8}, {2, 448, 2, 5}, {1, 8960, 3, 4}, {-1, 0.0099584228623437188, -1, -1}, {-1, 0.043727468153095861, -1, -1},
    {5, 96, 6, 7}, {-1, 0.03911221710570089, -1, -1}, {-1, 0.11965673220396099, -1, -1}, {11, 21, 9, 12}, {5, 96, 10, 11},
    {-1, 0.21503684186316341, -1, -1}, {-1, 0.49816058624203219, -1, -1}, {11, 96.5, 13, 14}, {-1, 0.14537200177300427, -1, -1}, {-1, 0.046436451557076687, -1, -1},
    {11, 27.5, 16, 23}, {2, 1280, 17, 20}, {0, 9.5, 18, 19}, {-1, 0.003217793308876222, -1, -1}, {-1, 0.067007444056045792, -1, -1},
    {11, 13.5, 21, 22}, {-1, 0.29631582856485161, -1, -1}, {-1, 0.15670747545378483, -1, -1}, {14, 108.5, 24, 27}, {12, 111.5, 25, 26},
    {-1, 0.036664846977092595, -1, -1}, {-1, 0.08440292289152479, -1, -1}, {15, 1.5, 28, 29}, {-1, -0.00042625959456151717, -1, -1}, {-1, 0.026262273146532664, -1, -1},
    {11, 57, 31, 38}, {12, 20, 32, 35}, {15, 1.5, 33, 34}, {-1, 0.0028054807166245058, -1, -1}, {-1, 0.045276384941897022, -1, -1},
    {11, 17.5, 36, 37}, {-1, 0.17063073351268493, -1, -1}, {-1, 0.068699894398790179, -1, -1}, {1, 15040, 39, 42}, {1, 13696, 40, 41},
    {-1, 0.012638229316805022, -1, -1}, {-1, -0.015310083423884012, -1, -1}, {16, 0.22265625, 43, 44}, {-1, 0.0056680652640038378, -1, -1}, {-1, 0.047445034935193017, -1, -1},
    {12, 160, 46, 51}, {18, 16.644122383252817, 47, 50}, {1, 96, 48, 49}, {-1, -0.057988548022100822, -1, -1}, {-1, 0.010474988438180659, -1, -1},
    {-1, 0.13598156891134636, -1, -1}, {1, 2112, 52, 55}, {5, 160, 53, 54}, {-1, 0.14216304405247657, -1, -1}, {-1, 0.32248355729340838, -1, -1},
    {11, 23.5, 56, 57}, {-1, 0.099709630782448322, -1, -1}, {-1, 0.014626074523049537, -1, -1}, {2, 10240, 59, 66}, {16, 0.41015625, 60, 63},
    {2, 7200, 61, 62}, {-1, -0.00010996793696298911, -1, -1}, {-1, -0.056474922261294164, -1, -1}, {2, 224, 64, 65}, {-1, -0.0081224677437740939, -1, -1},
    {-1, 0.026435467767535106, -1, -1}, {11, 11.5, 67, 68}, {-1, 0.16786800112881517, -1, -1}, {1, 5568, 69, 70}, {-1, 0.03774079948974618, -1, -1},
    {-1, 0.004837574680657479, -1, -1}, {12, 10, 72, 79}, {15, 2.5, 73, 76}, {2, 224, 74, 75}, {-1, -0.012342833121029725, -1, -1},
    {-1, -0.0033751023798808174, -1, -1}, {12, 1.5, 77, 78}, {-1, -0.010451733159555157, -1, -1}, {-1, 0.035539567624849937, -1, -1}, {11, 57, 80, 83},
    {1, 736, 81, 82}, {-1, -0.026424539503696467, -1, -1}, {-1, 0.041355139567263464, -1, -1}, {11, 78.5, 84, 85}, {-1, -0.020679335870522439, -1, -1},
    {-1, 0.0072793103219313694, -1, -1}, {18, 16.543718976708668, 87, 94}, {15, 1.5, 88, 91}, {1, 1984, 89, 90}, {-1, 0.013218229948897013, -1, -1},
    {-1, -0.010681143486218948, -1, -1}, {11, 51.5, 92, 93}, {-1, 0.057226140222226884, -1, -1}, {-1, 0.006431492590768622, -1, -1}, {-1, 0.070436056264550448, -1, -1},
    {5, 160, 96, 101}, {1, 96, 97, 98}, {-1, -0.039053314041814127, -1, -1}, {12, 40.5, 99, 100}, {-1, -0.0051662952957788255, -1, -1},
    {-1, 0.0057918923195167194, -1, -1}, {1, 4480, 102, 105}, {2, 704, 103, 104}, {-1, -0.0090304470959656317, -1, -1}, {-1, 0.073406201853807432, -1, -1},
    {11, 39.5, 106, 107}, {-1, -0.02821340934349317, -1, -1}, {-1, 0.015148361963647839, -1, -1}, {9, 2.5, 109, 116}, {1, 1984, 110, 113},
    {1, 864, 111, 112}, {-1, -0.0034639487509667433, -1, -1}, {-1, 0.056628020937461866, -1, -1}, {15, 1.5, 114, 115}, {-1, -0.0033284986693840287, -1, -1},
    {-1, 0.0090734191936406704, -1, -1}, {15, 1.5, 117, 120}, {6, 96, 118, 119}, {-1, -0.0044149091317863915, -1, -1}, {-1, -0.029913910404962996, -1, -1},
    {1, 13280, 121, 122}, {-1, 0.013341871779865828, -1, -1}, {-1, -0.010528486932745258, -1, -1}, {5, 224, 124, 131}, {11, 42.5, 125, 128},
    {11, 28.5, 126, 127}, {-1, -0.00035565003638978046, -1, -1}, {-1, -0.025594128602195489, -1, -1}, {11, 46.5, 129, 130}, {-1, 0.06259172703389039, -1, -1},
    {-1, -0.00010533746342174164, -1, -1}, {1, 3904, 132, 135}, {2, 704, 133, 134}, {-1, -0.0055592381129851388, -1, -1}, {-1, 0.081560910852196203, -1, -1},
    {1, 10176, 136, 137}, {-1, -0.009390463436123743, -1, -1}, {-1, 0.017579008622023589, -1, -1}, {1, 96, 139, 142}, {0, 4, 140, 141},
    {-1, -0.071078712880385961, -1, -1}, {-1, 0.020417377785614442, -1, -1}, {11, 7.5, 143, 146}, {2, 2272, 144, 145}, {-1, -0.0046946689249455177, -1, -1},
    {-1, 0.063494489982547586, -1, -1}, {14, 273.5, 147, 148}, {-1, 0.0015190165229781864, -1, -1}, {-1, -0.007264835242651476, -1, -1}, {9, 2.5, 150, 157},
    {2, 4128, 151, 154}, {11, 62.5, 152, 153}, {-1, 0.0028872433535043988, -1, -1}, {-1, -0.0049270889981320835, -1, -1}, {1, 2944, 155, 156},
    {-1, 0.03226110338739583, -1, -1}, {-1, 0.0026895288987542365, -1, -1}, {1, 1152, 158, 161}, {12, 52.5, 159, 160}, {-1, -0.0006420992486795697, -1, -1},
    {-1, -0.037632985797217876, -1, -1}, {11, 10.5, 162, 163}, {-1, 0.032439790695501777, -1, -1}, {-1, -0.0017244929691564827, -1, -1},
};
static constexpr int kDualBSmallRoots[] = {0, 15, 30, 45, 58, 71, 86, 95, 108, 123, 138, 149};
static constexpr DualBPolicyTile kDualBSmallTiles[] = {
    {16, 16, 64, 16, 16, 2, 10}, {16, 16, 64, 16, 16, 3, 10},
    {16, 16, 128, 16, 16, 2, 10}, {16, 16, 128, 16, 16, 3, 10},
    {16, 32, 64, 16, 16, 2, 10}, {16, 32, 64, 16, 16, 3, 10},
    {16, 32, 128, 16, 16, 3, 10}, {16, 64, 64, 16, 16, 2, 10},
    {16, 64, 64, 16, 16, 3, 10}, {16, 64, 128, 16, 16, 2, 10},
    {16, 64, 128, 16, 16, 3, 10}, {16, 128, 64, 16, 16, 2, 15},
    {16, 128, 64, 16, 16, 3, 15}, {16, 128, 128, 16, 16, 2, 15},
    {16, 128, 128, 16, 16, 3, 15}, {16, 192, 64, 16, 16, 2, 15},
    {16, 192, 64, 16, 16, 3, 15}, {16, 192, 128, 16, 16, 2, 15},
    {16, 256, 64, 16, 16, 2, 15}, {16, 256, 64, 16, 16, 3, 15},
};
static constexpr DualBCostNode kDualBGeneralTree[] = {
    {7, 24, 1, 8}, {13, 125.5, 2, 5}, {12, 20, 3, 4}, {-1, 0.055252434556024493, -1, -1}, {-1, 0.12172442593769797, -1, -1},
    {20, 0.0390625, 6, 7}, {-1, 0.28447186661474072, -1, -1}, {-1, 0.071204523803150219, -1, -1}, {9, 2.5, 9, 12}, {8, 24, 10, 11},
    {-1, 0.1657663709953231, -1, -1}, {-1, 0.086676123577219377, -1, -1}, {13, 14.5, 13, 14}, {-1, 0.14370721612866352, -1, -1}, {-1, 0.045499854907794346, -1, -1},
    {4, 112, 16, 23}, {13, 196.5, 17, 20}, {2, 1280, 18, 19}, {-1, 0.024961538337649896, -1, -1}, {-1, 0.084990424084797511, -1, -1},
    {12, 14, 21, 22}, {-1, 0.098508603394564159, -1, -1}, {-1, 0.19964054557964919, -1, -1}, {7, 24, 24, 27}, {2, 2560, 25, 26},
    {-1, 0.10362992311196129, -1, -1}, {-1, 0.16997521020189987, -1, -1}, {13, 10.5, 28, 29}, {-1, 0.14889415288436472, -1, -1}, {-1, 0.035098178966903079, -1, -1},
    {9, 2.5, 31, 38}, {2, 1280, 32, 35}, {21, 17, 33, 34}, {-1, -0.016287485884668804, -1, -1}, {-1, 0.032691994803257705, -1, -1},
    {4, 160, 36, 37}, {-1, 0.089736023921428099, -1, -1}, {-1, 0.027322646534734554, -1, -1}, {5, 224, 39, 42}, {13, 25.5, 40, 41},
    {-1, 0.025643564709853336, -1, -1}, {-1, -0.016735888943915558, -1, -1}, {10, 9.5, 43, 44}, {-1, 0.0044117396640201163, -1, -1}, {-1, 0.049664363533236895, -1, -1},
    {13, 113.5, 46, 53}, {11, 25.5, 47, 50}, {12, 28, 48, 49}, {-1, -0.0028399664123777758, -1, -1}, {-1, 0.046374016865398264, -1, -1},
    {15, 2.5, 51, 52}, {-1, -0.026969931628337114, -1, -1}, {-1, 0.0093751387711930936, -1, -1}, {7, 24, 54, 57}, {12, 96.5, 55, 56},
    {-1, 0.066079863679743139, -1, -1}, {-1, 0.16129870483327813, -1, -1}, {17, 0.9174107142857143, 58, 59}, {-1, 0.030293119465585223, -1, -1}, {-1, -0.0067102238451927547, -1, -1},
    {2, 640, 61, 68}, {7, 24, 62, 65}, {13, 606, 63, 64}, {-1, 0.0061805135230688108, -1, -1}, {-1, 0.06563195527322159, -1, -1},
    {13, 735, 66, 67}, {-1, -0.022300045707664359, -1, -1}, {-1, -0.0021116663084663912, -1, -1}, {20, 0.044270833333333329, 69, 72}, {14, 77.5, 70, 71},
    {-1, 0.019834205017644772, -1, -1}, {-1, 0.10431738393709172, -1, -1}, {20, 0.057291666666666664, 73, 74}, {-1, -0.042689764810755371, -1, -1}, {-1, -0.097923770835599749, -1, -1},
    {4, 160, 76, 83}, {10, 9.5, 77, 80}, {12, 320, 78, 79}, {-1, -0.00040312508494953824, -1, -1}, {-1, 0.07572253131547807, -1, -1},
    {12, 48.5, 81, 82}, {-1, 0.015544397070462599, -1, -1}, {-1, 0.051661492672453617, -1, -1}, {6, 96, 84, 87}, {18, 1.3357706316076872, 85, 86},
    {-1, -0.021755140768040516, -1, -1}, {-1, 0.035751630924327267, -1, -1}, {13, 21, 88, 89}, {-1, 0.075742076451353169, -1, -1}, {-1, 0.0060612387376362169, -1, -1},
    {9, 2.5, 91, 98}, {21, 17, 92, 95}, {14, 50.5, 93, 94}, {-1, -0.0028496033581614589, -1, -1}, {-1, -0.03518299368786762, -1, -1},
    {10, 184, 96, 97}, {-1, 0.014718477399028117, -1, -1}, {-1, 0.073214453190983575, -1, -1}, {13, 21.5, 99, 102}, {1, 512, 100, 101},
    {-1, -0.029557614748923766, -1, -1}, {-1, 0.052004202467931186, -1, -1}, {13, 635, 103, 104}, {-1, -0.019475177650773782, -1, -1}, {-1, 0.0029808321079977324, -1, -1},
    {12, 160, 106, 113}, {13, 632.5, 107, 110}, {18, 1.3519639287872449, 108, 109}, {-1, -0.0065918543224486766, -1, -1}, {-1, 0.01807538242501406, -1, -1},
    {17, 0.59449404761904767, 111, 112}, {-1, 0.074469427273618966, -1, -1}, {-1, 0.0072425223412760503, -1, -1}, {5, 224, 114, 117}, {14, 42, 115, 116},
    {-1, -0.011970000851888129, -1, -1}, {-1, 0.036019274013404393, -1, -1}, {18, 1.0656958400156005, 118, 119}, {-1, 0.023574230580959744, -1, -1}, {-1, 0.10077349324408467, -1, -1},
    {21, 17, 121, 128}, {5, 96, 122, 125}, {10, 1.5, 123, 124}, {-1, -0.053049931185065395, -1, -1}, {-1, -0.018356406873000684, -1, -1},
    {13, 14.5, 126, 127}, {-1, 0.056600963535309405, -1, -1}, {-1, -0.0086718981659576425, -1, -1}, {14, 39.5, 129, 132}, {13, 78.5, 130, 131},
    {-1, -0.01039421139438873, -1, -1}, {-1, 0.00718398768119443, -1, -1}, {13, 54.5, 133, 134}, {-1, 0.12923072732645491, -1, -1}, {-1, 0.012904268741160276, -1, -1},
    {13, 18.5, 136, 143}, {18, 1.6352874613744179, 137, 140}, {1, 1056, 138, 139}, {-1, -0.01042498711211011, -1, -1}, {-1, 0.033550684923461652, -1, -1},
    {12, 59.5, 141, 142}, {-1, 0.034008662335569033, -1, -1}, {-1, 0.14421077807820343, -1, -1}, {13, 39.5, 144, 147}, {13, 24.5, 145, 146},
    {-1, 4.3428999167866641e-05, -1, -1}, {-1, -0.040429943034457252, -1, -1}, {17, 0.8046875, 148, 149}, {-1, 0.0091391317408547994, -1, -1}, {-1, -0.0092563132244940765, -1, -1},
    {12, 5, 151, 158}, {19, 1.1666666666666665, 152, 155}, {12, 1.5, 153, 154}, {-1, -0.031434493496357623, -1, -1}, {-1, -0.010425915659376913, -1, -1},
    {13, 433.5, 156, 157}, {-1, 0.0056470379150235263, -1, -1}, {-1, 0.064256892905221116, -1, -1}, {16, 0.9140625, 159, 162}, {0, 96.5, 160, 161},
    {-1, -0.011391362584692656, -1, -1}, {-1, 0.012213081625716537, -1, -1}, {18, 1.010854910353411, 163, 164}, {-1, -0.021455447522689162, -1, -1}, {-1, -0.003121897474080391, -1, -1},
    {4, 160, 166, 173}, {22, 0.5, 167, 170}, {12, 224, 168, 169}, {-1, -0.006602785752642561, -1, -1}, {-1, 0.022166575641414776, -1, -1},
    {2, 320, 171, 172}, {-1, -0.0073547163478294977, -1, -1}, {-1, 0.010826282504617583, -1, -1}, {1, 224, 174, 177}, {0, 3072, 175, 176},
    {-1, 0.071907274681433112, -1, -1}, {-1, -0.0021197537245304229, -1, -1}, {8, 24, 178, 179}, {-1, 0.0043516030272784034, -1, -1}, {-1, -0.014154686095226127, -1, -1},
};
static constexpr int kDualBGeneralRoots[] = {0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165};
static constexpr DualBPolicyTile kDualBGeneralTiles[] = {
    {32, 64, 64, 16, 16, 2, 10}, {32, 64, 64, 16, 16, 4, 10}, {32, 64, 128, 16, 16, 2, 10}, {32, 64, 128, 16, 16, 4, 10},
    {32, 128, 64, 16, 16, 2, 15}, {32, 128, 64, 16, 64, 2, 15}, {32, 128, 64, 32, 32, 2, 15}, {32, 128, 128, 16, 16, 2, 15},
    {32, 192, 64, 16, 16, 4, 15}, {32, 192, 64, 16, 32, 4, 15}, {32, 192, 64, 32, 16, 4, 15}, {32, 192, 64, 32, 32, 2, 15},
    {32, 192, 128, 16, 16, 2, 15}, {32, 256, 64, 16, 32, 3, 15}, {32, 256, 64, 32, 32, 3, 15}, {48, 64, 64, 16, 16, 2, 10},
    {48, 64, 128, 16, 16, 2, 10}, {48, 128, 64, 16, 16, 3, 15}, {48, 128, 64, 48, 16, 2, 5}, {48, 128, 128, 16, 16, 2, 15},
    {48, 192, 64, 16, 32, 2, 15}, {48, 192, 64, 16, 32, 4, 15}, {48, 192, 64, 16, 48, 2, 15}, {48, 192, 64, 16, 48, 4, 15},
    {48, 192, 64, 48, 16, 2, 15}, {48, 192, 128, 16, 32, 2, 15}, {48, 256, 64, 16, 64, 3, 15}, {48, 256, 64, 48, 16, 2, 15},
    {64, 64, 64, 16, 16, 2, 10}, {64, 64, 128, 16, 16, 2, 10}, {64, 64, 128, 16, 16, 3, 10}, {64, 64, 128, 16, 16, 4, 10},
    {64, 128, 64, 16, 64, 2, 5}, {64, 128, 64, 32, 32, 2, 5}, {64, 128, 128, 16, 16, 2, 15}, {64, 192, 64, 16, 32, 2, 15},
    {64, 192, 64, 16, 32, 4, 15}, {64, 192, 64, 32, 16, 3, 15}, {64, 192, 64, 32, 48, 2, 5}, {64, 192, 128, 16, 32, 2, 15},
    {96, 64, 64, 16, 16, 2, 10}, {96, 64, 64, 16, 16, 4, 10}, {96, 64, 64, 16, 16, 5, 10}, {96, 64, 64, 48, 16, 3, 10},
    {96, 128, 64, 16, 32, 2, 15}, {96, 128, 128, 16, 32, 2, 15}, {96, 192, 64, 16, 48, 2, 15}, {96, 192, 64, 32, 32, 4, 15},
    {96, 192, 64, 48, 16, 3, 15}, {96, 192, 64, 48, 16, 4, 15}, {96, 192, 64, 48, 32, 2, 5}, {96, 192, 64, 48, 48, 2, 5},
    {96, 192, 128, 16, 48, 2, 15}, {96, 256, 64, 32, 32, 2, 15}, {96, 256, 64, 32, 32, 3, 15}, {96, 256, 64, 32, 64, 3, 15},
    {96, 256, 64, 48, 16, 2, 15}, {96, 256, 64, 48, 16, 3, 15}, {96, 256, 64, 48, 32, 2, 15}, {96, 256, 64, 48, 32, 3, 15},
    {128, 128, 64, 32, 32, 2, 10}, {128, 128, 64, 64, 16, 2, 10}, {128, 128, 128, 16, 32, 2, 15},
    {128, 192, 64, 32, 32, 2, 15}, {128, 192, 64, 32, 32, 3, 15}, {128, 192, 64, 32, 32, 4, 15}, {128, 192, 64, 32, 48, 2, 15},
    {128, 192, 64, 64, 16, 2, 15}, {128, 192, 64, 64, 16, 4, 15}, {128, 256, 64, 32, 32, 2, 15}, {128, 256, 64, 32, 32, 3, 15},
    {128, 256, 64, 32, 64, 2, 15}, {128, 256, 64, 32, 64, 3, 15}, {128, 256, 64, 64, 16, 3, 15}, {128, 256, 64, 64, 32, 2, 15},
    {128, 256, 64, 64, 32, 3, 15}, {192, 64, 64, 32, 16, 4, 10}, {192, 64, 64, 32, 16, 5, 10}, {192, 128, 64, 32, 32, 2, 15},
    {192, 128, 64, 48, 32, 2, 10}, {192, 128, 128, 48, 16, 2, 15}, {192, 192, 64, 32, 48, 2, 8}, {192, 192, 64, 32, 48, 3, 13},
    {192, 192, 64, 48, 32, 2, 15}, {192, 192, 64, 48, 32, 3, 15}, {192, 192, 64, 48, 48, 2, 15}, {192, 192, 64, 48, 48, 3, 15},
    {256, 128, 64, 32, 32, 3, 15}, {256, 128, 64, 32, 32, 4, 15}, {256, 128, 64, 32, 64, 2, 15}, {256, 128, 64, 32, 64, 3, 10},
    {256, 128, 64, 64, 16, 3, 15}, {256, 128, 64, 64, 16, 4, 15}, {256, 128, 64, 64, 32, 2, 15}, {256, 128, 128, 32, 32, 2, 15},
};

template <size_t NodeCount, size_t RootCount, size_t TileCount>
static DualBPolicyTile select_dual_b_tile(int m, int n, int k, int num_sms, bool fp32,
    const DualBCostNode (&nodes)[NodeCount], const int (&roots)[RootCount],
    const DualBPolicyTile (&tiles)[TileCount]) {
    double best = 1e300;
    DualBPolicyTile selected{16, 16, 64, 16, 16, 2, 15};
    for (const auto& t : tiles) {
        const int bit = (fp32 ? 2 : 0) + (n % t.bn == 0 ? 1 : 0);
        if (!(t.resource_mask & (1 << bit))) continue;
        const int smem = t.stages * (t.bm + 2 * t.bn) * t.bk * 2;
        const int warps = (t.bm / t.wm) * (t.bn / t.wn);
        const int we = (warps + 7) / 8;
        const int limit = we == 3 ? 168 : std::min(256, 512 / we);
        const double regs = 2.0 * t.wm * t.wn / 32 + (t.wm + t.wn) / 2.0 + 20;
        if (smem > 262144 || warps * 32 > 1024 || regs > limit) continue;
        const int mt = ceil_div(m, t.bm), nt = ceil_div(n, t.bn), kt = ceil_div(k, t.bk);
        const int count = mt * nt;
        const int grid = std::min(count, (262144 / smem) * std::min(num_sms, count));
        double features[] = {double(m), double(n), double(k), double(fp32), double(t.bm), double(t.bn),
            double(t.bk), double(t.wm), double(t.wn), double(t.stages), double(mt), double(nt), double(kt),
            double(count), double(grid), double(ceil_div(count, grid)), smem / 262144.0, regs / limit,
            double(mt) * t.bm * nt * t.bn / (double(m) * n), double(kt) * t.bk / k,
            1.0 / t.bm + 2.0 / t.bn, double(warps), double(n % t.bn != 0)};
        double score = 0;
        for (const int root : roots) {
            int index = root;
            while (nodes[index].feature >= 0) {
                const auto& node = nodes[index];
                index = features[node.feature] <= node.value ? node.left : node.right;
            }
            score += nodes[index].value;
        }
        if (score < best) { best = score; selected = t; }
    }
    return selected;
}

static DualBPolicyTile get_dual_b_policy_tile(int m, int n, int k, int num_sms, bool fp32) {
    // Cache only the last feature-based tile selection per thread to reduce repeated host overhead; never cache inputs or outputs.
    struct Cache { int m = 0, n = 0, k = 0, sms = 0; bool fp32 = false; DualBPolicyTile tile{}; };
    static thread_local Cache cache;
    if (cache.m == m && cache.n == n && cache.k == k && cache.sms == num_sms && cache.fp32 == fp32)
        return cache.tile;
    const auto tile = m <= 16
        ? select_dual_b_tile(m, n, k, num_sms, fp32, kDualBSmallTree, kDualBSmallRoots, kDualBSmallTiles)
        : select_dual_b_tile(m, n, k, num_sms, fp32, kDualBGeneralTree, kDualBGeneralRoots, kDualBGeneralTiles);
    cache = Cache{m, n, k, num_sms, fp32, tile};
    return tile;
}

// The PPU1.5 dual-B fused launcher validates configs against dual accumulators and the dual-B SMEM budget.
// Unsupported explicit configs raise an error instead of falling back to composition.
template <typename ElementOutput>
static void bf16xfp32_gemm_cute_free_impl(const torch::Tensor& lhs, const torch::Tensor& w_high,
                                          const torch::Tensor& w_low, const torch::Tensor& out,
                                          const float& scale, const int& m, const int& n, const int& k,
                                          std::optional<ConfigTuple> configs = std::nullopt) {
    int num_sms = get_num_sms();
    static constexpr GemmType kGemmType = GemmType::Normal;
    const char* packed_env = std::getenv("DG_BF16XFP32_PACKED_DUAL_N");
    DG_HOST_ASSERT(packed_env == nullptr || std::string(packed_env) == "0" || std::string(packed_env) == "1");
    auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && *value == '1';
    };
    const bool other_pipeline = enabled("DG_BF16XFP32_FULL_K_PREFETCH")
        || enabled("DG_BF16XFP32_EARLY_G2S") || enabled("DG_BF16XFP32_MMA_S2R_OVERLAP")
        || enabled("DG_BF16XFP32_BOUNDED_KLOOP") || enabled("DG_BF16XFP32_OVERLAP_PROLOGUE")
        || enabled("DG_BF16XFP32_VALID_EPILOGUE") || enabled("DG_BF16XFP32_STATIC_STAGES")
        || enabled("DG_BF16XFP32_STREAMED_LOW") || enabled("DG_BF16XFP32_STANDARD_S2")
        || enabled("DG_BF16XFP32_STREAMED_PAIRS") || enabled("DG_BF16XFP32_LAUNCH_BOUNDS2")
        || enabled("DG_BF16XFP32_PAIRS_M_FIRST");
    // Auto-enable only validated router defaults; explicit configs keep their kernel, 0 disables, and 1 opts into the experiment.
    const bool packed_dual_n = (packed_env != nullptr && *packed_env == '1')
        || (packed_env == nullptr && !other_pipeline && !configs.has_value()
            && std::is_same_v<ElementOutput, float> && (m == 8 || m == 16)
            && n == 384 && k == 7168);

    // Specialize dimensions and strides for exact router shapes to remove repeated DMA address calculations without changing K order.
    // Add M16 specialization only to the packed default path; explicit configs retain their behavior unless the experiment is enabled.
    const char* static_env = std::getenv("DG_BF16XFP32_STATIC_ROUTER");
    DG_HOST_ASSERT(static_env == nullptr || std::string(static_env) == "0" || std::string(static_env) == "1");
    const bool static_router = (static_env == nullptr || *static_env == '1')
        && std::is_same_v<ElementOutput, float>
        && (m == 8 || (m == 16 && packed_dual_n
            && (!configs.has_value() || (static_env != nullptr && *static_env == '1')))
            || m == 128 || m == 256 || m == 512 || m == 4096 || m == 16384) && n == 384 && k == 7168;

    using Config = std::tuple<int, int, int, int, int, int, int, int, std::tuple<int, int, int>>;

    Config cfg;
    if (configs.has_value()) {
        // Explicit config path: unpack the 8-element public tuple; force warp_k = block_k
        // (same contract as the plain BF16 dense path).
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, /*warp_k=*/bk, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else if (std::is_same_v<ElementOutput, float> && (m == 8 || m == 16)
               && n == 384 && k == 7168) {
        // Router decode specialization validated with 64-layer rotation on the same card; larger BK reduces stage sync while one warp covers full K.
        // Apply only to exact shapes with FP32 output; explicit configs, standard BF16, and PPU1.0 paths remain unchanged.
        cfg = packed_dual_n
            ? std::make_tuple(std::min(num_sms, 39), 16, 16, 512,
                              16, 16, 512, 4, std::make_tuple(0, 0, 0))
            : std::make_tuple(std::min(num_sms, 24), 16, 16, 1024,
                              16, 16, 1024, 2, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 128 && n == 384 && k == 7168) {
        // Short-to-medium router prefill: increase output-tile parallelism while both FP32 accumulation chains cover full K.
        cfg = std::make_tuple(std::min(num_sms, 39), 64, 32, 128,
                              16, 16, 128, 4, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 256 && n == 384 && k == 7168) {
        // Exact M256 uses a full M tile and deeper stages; both modes passed interleaved tuning without changing K order.
        cfg = std::make_tuple(std::min(num_sms, 39), 128, 32, 128,
                              16, 16, 128, 5, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 512 && n == 384 && k == 7168) {
        // Exact M512 increases warp-M reuse while preserving separate high/low FP32 accumulation chains over full K.
        cfg = std::make_tuple(std::min(num_sms, 39), 128, 64, 64,
                              32, 16, 64, 4, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 4096 && n == 384 && k == 7168) {
        // Target the router prefill regression point with a smaller M tile chosen by independent rotation tuning; keep the general cost tree unchanged.
        cfg = std::make_tuple(std::min(num_sms, 64), 128, 192, 64,
                              32, 32, 64, 4, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 16384 && n == 384 && k == 7168) {
        // Long router prefill uses a larger warp-N tile; both chains cover full K, with no spills in isolated rotation validation.
        cfg = std::make_tuple(std::min(num_sms, 64), 256, 128, 64,
                              64, 32, 64, 4, std::make_tuple(0, 0, 0));
    } else if (m == 512 && n == 2048 && k == 4096) {
        // Both output types were measured spill-free at this anchor; default-path and deep validation still need reruns.
        cfg = std::make_tuple(std::min(num_sms, 32), 128, 256, 64,
                              64, 32, 64, 3, std::make_tuple(0, 0, 0));
    } else {
        const auto t = get_dual_b_policy_tile(m, n, k, num_sms, std::is_same_v<ElementOutput, float>);
        cfg = std::make_tuple(std::min(num_sms, ceil_div(m, t.bm) * ceil_div(n, t.bn)),
            t.bm, t.bn, t.bk, t.wm, t.wn, t.bk, t.stages, std::make_tuple(0, 0, 0));
    }
    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, smem_config] = cfg;
    (void)smem_config;
    DG_HOST_ASSERT(!packed_dual_n || (std::is_same_v<ElementOutput, float>
                   && (m == 8 || m == 16 || m == 128 || m == 256) && n == 384 && k == 7168
                   && block_m > 0 && block_n > 0 && block_k >= 64 && block_k % 64 == 0
                   && block_m % 16 == 0 && block_n % 16 == 0 && warp_m == 16 && warp_n == 16
                   && (block_m / 16) * (block_n / 16) <= 32));

    // Budget the full dual-B SMEM footprint while preserving stage pruning for explicit configs.
    constexpr int kSmemPerCU = 256 * 1024;
    constexpr int kElemBytes = static_cast<int>(sizeof(cutlass::bfloat16_t));
    auto dual_smem_bytes = [&](int stages) {
        int bytes = stages * (block_m + (packed_dual_n ? 1 : 2) * block_n) * block_k * kElemBytes;
        return (bytes + 127) & ~127;
    };
    while (num_stages > 2 && dual_smem_bytes(num_stages) > kSmemPerCU)
        --num_stages;
    if (!configs.has_value() && dual_smem_bytes(num_stages) > kSmemPerCU) {
        // Single-B configs may use excessive BK; select a valid fused default rather than rejecting a supported shape.
        block_k = std::min(block_k, 128);
        if (dual_smem_bytes(num_stages) > kSmemPerCU) block_k = 64;
        warp_k = block_k;
    }
    const int SMSIZE = dual_smem_bytes(num_stages);
    // Never disguise composition as fusion; report explicit configs that exceed the budget as unsupported.
    DG_HOST_ASSERT(num_sms_new > 0 && num_stages >= 2);
    DG_HOST_ASSERT(SMSIZE <= kSmemPerCU);
    DG_HOST_ASSERT(warp_k == block_k);

    cutlass::bfloat16_t* input_a = reinterpret_cast<cutlass::bfloat16_t*>(lhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_b_high = reinterpret_cast<cutlass::bfloat16_t*>(w_high.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_b_low = reinterpret_cast<cutlass::bfloat16_t*>(w_low.data_ptr<at::BFloat16>());
    auto* output = static_cast<ElementOutput*>(out.data_ptr());

    using Runtime = DenseBf16xFp32GemmCuteFreeRuntimeT<ElementOutput>;

    typename Runtime::GemmKernelParams hparams;
    hparams.ptr_A = input_a;
    hparams.ptr_B_high = input_b_high;
    hparams.ptr_B_low = input_b_low;
    hparams.ptr_D = output;
    hparams.low_scale = scale;
    hparams.M = m;
    hparams.N = n;
    hparams.K = k;
    hparams.lda = k;   // A is row-major MxK
    hparams.ldb = k;   // B is col-major KxN, stored as row-major NxK
    hparams.ldd = n;   // D is row-major MxN
    hparams.scheduler = DenseGemmTileSchedulerArguments{(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr};

    const int warps_k = block_k / warp_k;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * warps_k * 32;

    // SMEM-based occupancy heuristic on the dual-B footprint.
    int blocks_per_cu = kSmemPerCU / SMSIZE;
    if (blocks_per_cu < 1) blocks_per_cu = 1;

    dim3 grid;
    // Cap the persistent grid at the output-tile count to avoid idle CTAs.
    grid.x = std::min(blocks_per_cu * num_sms_new,
                      ceil_div(m, block_m) * ceil_div(n, packed_dual_n ? block_n / 2 : block_n));

    // Output dtype is part of the JIT cache key (separate kernel name).
    std::string kernel_name = "bf16xfp32_gemm_cute_free";
    if (std::is_same_v<ElementOutput, float>) kernel_name += "_fp32";
    if (packed_dual_n) kernel_name += "_packed_dual_n";
    if (static_router) kernel_name += "_static_router_m" + std::to_string(m);
    // Exact long-prefill experiment: specialize only outer stage indices; each chain still advances through full K in order.
    const char* stages_env = std::getenv("DG_BF16XFP32_STATIC_STAGES");
    DG_HOST_ASSERT(stages_env == nullptr || std::string(stages_env) == "0" || std::string(stages_env) == "1");
    const bool static_stages = stages_env != nullptr && *stages_env == '1';
    DG_HOST_ASSERT(!static_stages || (static_router && !packed_dual_n
        && std::is_same_v<ElementOutput, float> && m == 16384 && n == 384 && k == 7168));
    if (static_stages) kernel_name += "_static_stages";
    // Load low fragments on demand without preconverting or premerging weights; both FP32 accumulators cover full K.
    const char* low_env = std::getenv("DG_BF16XFP32_STREAMED_LOW");
    DG_HOST_ASSERT(low_env == nullptr || std::string(low_env) == "0" || std::string(low_env) == "1");
    const bool streamed_low = low_env != nullptr && *low_env == '1';
    DG_HOST_ASSERT(!streamed_low || (static_router && !packed_dual_n
        && std::is_same_v<ElementOutput, float> && m == 16384 && n == 384 && k == 7168));
    if (streamed_low) kernel_name += "_streamed_low";
    const char* pairs_env = std::getenv("DG_BF16XFP32_STREAMED_PAIRS");
    DG_HOST_ASSERT(pairs_env == nullptr || std::string(pairs_env) == "0" || std::string(pairs_env) == "1");
    const bool streamed_pairs = pairs_env != nullptr && *pairs_env == '1';
    DG_HOST_ASSERT(!streamed_pairs || streamed_low);
    if (streamed_pairs) kernel_name += "_streamed_pairs";
    // Fusion-only M-subtile scheduling experiment, disabled by default; preserve each output's K-step accumulation order.
    const char* mfirst_env = std::getenv("DG_BF16XFP32_PAIRS_M_FIRST");
    DG_HOST_ASSERT(mfirst_env == nullptr || std::string(mfirst_env) == "0" || std::string(mfirst_env) == "1");
    const bool pairs_m_first = mfirst_env != nullptr && *mfirst_env == '1';
    DG_HOST_ASSERT(!pairs_m_first || (streamed_pairs && static_router && !packed_dual_n
        && std::is_same_v<ElementOutput, float> && m == 16384 && n == 384 && k == 7168));
    if (pairs_m_first) kernel_name += "_pairs_m_first";
    // The exact two-stage experiment reuses the standard wait/barrier path; synchronize all three copies before stage reuse.
    const char* s2_env = std::getenv("DG_BF16XFP32_STANDARD_S2");
    DG_HOST_ASSERT(s2_env == nullptr || std::string(s2_env) == "0" || std::string(s2_env) == "1");
    const bool standard_s2 = s2_env != nullptr && *s2_env == '1';
    DG_HOST_ASSERT(!standard_s2 || (static_router && !packed_dual_n && !streamed_low
        && std::is_same_v<ElementOutput, float> && m == 16384 && n == 384 && k == 7168
        && num_stages == 2));
    if (standard_s2) kernel_name += "_standard_s2";
    // The dual-CTA resource experiment sets only the launch_bounds minimum; static SMEM estimates do not prove actual residency.
    const char* bounds_env = std::getenv("DG_BF16XFP32_LAUNCH_BOUNDS2");
    DG_HOST_ASSERT(bounds_env == nullptr || std::string(bounds_env) == "0" || std::string(bounds_env) == "1");
    const bool launch_bounds2 = bounds_env != nullptr && *bounds_env == '1';
    DG_HOST_ASSERT(!launch_bounds2 || (static_router && !packed_dual_n
        && std::is_same_v<ElementOutput, float> && m == 16384 && n == 384 && k == 7168
        && SMSIZE <= kSmemPerCU / 2 && block.x <= 512));
    const int min_blocks = launch_bounds2 ? 2 : 1;
    if (launch_bounds2) kernel_name += "_launch_bounds2";
    // The experiment preserves the public config tuple; full-K-tile register prefetch requires one MMA tile per warp.
    const char* prefetch_env = std::getenv("DG_BF16XFP32_FULL_K_PREFETCH");
    DG_HOST_ASSERT(prefetch_env == nullptr || std::string(prefetch_env) == "0"
                   || std::string(prefetch_env) == "1");
    const bool full_k_prefetch = prefetch_env != nullptr && *prefetch_env == '1'
        && warp_m == 16 && warp_n == 16 && warp_k == block_k && block_k <= 128;
    if (full_k_prefetch) kernel_name += "_full_k_prefetch";
    const char* early_env = std::getenv("DG_BF16XFP32_EARLY_G2S");
    DG_HOST_ASSERT(early_env == nullptr || std::string(early_env) == "0" || std::string(early_env) == "1");
    const bool early_g2s = (early_env != nullptr && *early_env == '1')
        || (early_env == nullptr && !other_pipeline && !packed_dual_n && !configs.has_value()
            && std::is_same_v<ElementOutput, float> && m == 128 && n == 384 && k == 7168);
    DG_HOST_ASSERT(!early_g2s || !full_k_prefetch);
    if (early_g2s) kernel_name += "_early_g2s";
    const char* overlap_env = std::getenv("DG_BF16XFP32_MMA_S2R_OVERLAP");
    DG_HOST_ASSERT(overlap_env == nullptr || std::string(overlap_env) == "0" || std::string(overlap_env) == "1");
    const bool mma_s2r_overlap = overlap_env != nullptr && *overlap_env == '1';
    DG_HOST_ASSERT(!mma_s2r_overlap || (!early_g2s && !full_k_prefetch));
    if (mma_s2r_overlap) kernel_name += "_mma_s2r_overlap";
    const char* kloop_env = std::getenv("DG_BF16XFP32_BOUNDED_KLOOP");
    DG_HOST_ASSERT(kloop_env == nullptr || std::string(kloop_env) == "0" || std::string(kloop_env) == "1");
    const bool bounded_kloop = kloop_env != nullptr && *kloop_env == '1';
    DG_HOST_ASSERT(!bounded_kloop || (!early_g2s && !full_k_prefetch && !mma_s2r_overlap));
    const char* unroll_env = std::getenv("DG_BF16XFP32_KLOOP_UNROLL");
    DG_HOST_ASSERT(unroll_env == nullptr || std::string(unroll_env) == "1"
                   || std::string(unroll_env) == "2" || std::string(unroll_env) == "4");
    const int kloop_unroll = bounded_kloop && unroll_env != nullptr ? *unroll_env - '0' : 1;
    const char* chunk_env = std::getenv("DG_BF16XFP32_KLOOP_CHUNK");
    DG_HOST_ASSERT(chunk_env == nullptr || std::string(chunk_env) == "0" || std::string(chunk_env) == "1");
    const bool kloop_chunk = bounded_kloop && chunk_env != nullptr && *chunk_env == '1';
    DG_HOST_ASSERT(!kloop_chunk || kloop_unroll == 2 || kloop_unroll == 4);
    DG_HOST_ASSERT(!streamed_low || (!full_k_prefetch && !early_g2s && !mma_s2r_overlap && !bounded_kloop));
    if (bounded_kloop) {
        kernel_name += "_bounded_kloop";
        if (kloop_chunk) kernel_name += "_c" + std::to_string(kloop_unroll);
        else if (kloop_unroll > 1) kernel_name += "_u" + std::to_string(kloop_unroll);
    }

    const char* prologue_env = std::getenv("DG_BF16XFP32_OVERLAP_PROLOGUE");
    const char* epilogue_env = std::getenv("DG_BF16XFP32_VALID_EPILOGUE");
    DG_HOST_ASSERT(prologue_env == nullptr || std::string(prologue_env) == "0" || std::string(prologue_env) == "1");
    DG_HOST_ASSERT(epilogue_env == nullptr || std::string(epilogue_env) == "0" || std::string(epilogue_env) == "1");
    // Exact long-prefill default experiment: overlap the next tile's stage0 with output merging; preserve explicit-config behavior.
    const bool router_overlap = !other_pipeline && !packed_dual_n && !configs.has_value()
        && std::is_same_v<ElementOutput, float> && m == 16384 && n == 384 && k == 7168;
    const bool overlap_prologue = (prologue_env != nullptr && *prologue_env == '1')
        || (prologue_env == nullptr && router_overlap);
    const bool valid_epilogue = (epilogue_env != nullptr && *epilogue_env == '1')
        || (epilogue_env == nullptr && router_overlap);
    DG_HOST_ASSERT(!(overlap_prologue || valid_epilogue)
                   || (!full_k_prefetch && !early_g2s && !mma_s2r_overlap && !bounded_kloop));
    if (overlap_prologue) kernel_name += "_overlap_prologue";
    if (valid_epilogue) kernel_name += "_valid_epilogue";
    DG_HOST_ASSERT(!packed_dual_n || (!full_k_prefetch && !early_g2s && !mma_s2r_overlap
                   && !bounded_kloop && !overlap_prologue && !valid_epilogue));

    auto args = typename Runtime::Args{
        .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages,
                        "Normal", kernel_name, full_k_prefetch, early_g2s, mma_s2r_overlap, bounded_kloop, kloop_unroll, kloop_chunk,
                        overlap_prologue, valid_epilogue, packed_dual_n, static_router, static_stages, streamed_low, standard_s2, streamed_pairs, min_blocks, pairs_m_first},
        .launch_args = {grid, block, SMSIZE},
        .kernel_params = hparams,
    };

    const auto& code = Runtime::generate(args);
    auto runtime = compiler->build(kernel_name, code);
    if (is_compile_only())
        return;
    auto kernel = runtime->kernel;

    const auto& stream = current_stream();
    auto config = construct_launch_config(kernel, stream, SMSIZE, grid, block);

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("bf16xfp32"), 1, m, n, k, 0, nullptr, stream);
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, hparams));

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[DenseGemm_BF16xFP32_CuteFree:]\n");
        printf("problem:[%d, %d, %d], scale:%g\n", m, n, k, scale);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu, grid.x);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n",
               block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", SMSIZE, int(numRegs), int(localSize));
        printf("dual_b_static_stages:%d, streamed_low:%d\n", int(static_stages), int(streamed_low));
        printf("dual_b_standard_s2:%d\n", int(standard_s2));
        printf("dual_b_streamed_pairs:%d\n", int(streamed_pairs));
        printf("dual_b_launch_bounds_min_blocks:%d\n", min_blocks);
        const char* bounded_name = kloop_chunk
            ? (kloop_unroll == 4 ? "bounded_kloop_c4" : "bounded_kloop_c2")
            : (kloop_unroll == 4 ? "bounded_kloop_u4" : (kloop_unroll == 2 ? "bounded_kloop_u2" : "bounded_kloop"));
        const char* dense_pipeline = overlap_prologue
            ? (valid_epilogue ? "overlap_valid" : "overlap_prologue") : "valid_epilogue";
        printf("dual_b_pipeline:%s\n", packed_dual_n ? "packed_dual_n" : (overlap_prologue || valid_epilogue) ? dense_pipeline
               : (bounded_kloop ? bounded_name : (mma_s2r_overlap ? "mma_s2r_overlap"
               : (full_k_prefetch ? "full_k_prefetch" : (early_g2s ? "early_g2s" : "double_buffer")))));
    }
}

// Public launcher: dispatches on the device generation.
//   PPU1.5 : All M use dual-B fusion with independent full-K FP32 accumulation and BF16 or FP32 output.
//   PPU1.0 : composed two-GEMM fallback (out = x@w_high^T, then out +=
//            scale * x@w_low^T).
// `configs` (optional) follows the 8-element public ConfigTuple contract of
// the plain BF16 dense path.
static void bf16xfp32_gemm(const torch::Tensor& lhs, const torch::Tensor& w_high, const torch::Tensor& w_low,
                          const torch::Tensor& out, const double& scale,
                          const int& m, const int& n, const int& k,
                          std::optional<ConfigTuple> configs = std::nullopt) {
    if (is_ppu1v5_device()) {
        if (out.scalar_type() == torch::kFloat)
            bf16xfp32_gemm_cute_free_impl<float>(lhs, w_high, w_low, out, static_cast<float>(scale),
                                                 m, n, k, configs);
        else
            bf16xfp32_gemm_cute_free_impl<cutlass::bfloat16_t>(lhs, w_high, w_low, out, static_cast<float>(scale),
                                                              m, n, k, configs);
        return;
    }

    // --- PPU1.0 fallback: compose two BF16 GEMMs (both accumulate in FP32). ---
    bf16xfp32_gemm_composed(lhs, w_high, w_low, out, scale, m, n, k, configs);
}

} // namespace deep_gemm
