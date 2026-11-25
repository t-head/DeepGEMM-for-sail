#pragma once

// =============================================================================
// BF16 x FP32 DenseGemm.
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
//     Exact FP32 router decode partitions K across two warps within each CTA,
//     reduces high/low partials in FP32, and stores once without a global workspace.
// PPU1.0 (810E): fallback composition of two BF16 GEMMs through an FP32
//     intermediate.
// =============================================================================

#include <torch/python.h>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include "cutlass/bfloat16.h"
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/common/dual_b_resources.hpp>
#include "../heuristics/common_bf16.hpp"
#include <deep_gemm/scheduler/densegemm_scheduler_cutlass3.cuh>
#include "bf16_gemm.hpp"

using namespace deep_gemm_bf16_common;
namespace deep_gemm {

// JIT runtime for the fused dual-B cute-free kernel.
// Follows the plain BF16 runtime's source-generation and launch structure;
// the host-side GemmKernelParams must stay byte-identical to
// kernel::GemmBf16xFp32Kernel::Params (the kernel receives it by value).
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
        bool early_g2s, mma_s2r_overlap;
        bool overlap_prologue, valid_epilogue, packed_dual_n, static_router, static_stages;
        int decode_split_k;
        bool packed_store, cta_k, one_tile;
        int k_order;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
    };

    static auto compile_key(const Args& args) {
        const auto& i = args.launch_info;
        const auto& p = args.kernel_params;
        return std::make_tuple(i.block_m, i.block_n, i.block_k, i.warp_m, i.warp_n, i.warp_k,
            i.num_stages, i.gemm_type, i.kernel_name, i.early_g2s, i.mma_s2r_overlap,
            i.overlap_prologue, i.valid_epilogue, i.packed_dual_n, i.static_router,
            i.static_stages, i.decode_split_k, i.packed_store, i.cta_k,
            i.one_tile, i.k_order, p.N % i.block_n == 0,
            i.static_router ? p.M : 0, i.static_router ? p.N : 0, i.static_router ? p.K : 0);
    }

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        const int shape_n = args.kernel_params.N;
        const bool is_aligned_n = (shape_n % info.block_n == 0);

        return fmt::format(
            R"(
#define BF16_HGRTC
#define DG_DUAL_B_FULL_K_PREFETCH 0
#define DG_DUAL_B_EARLY_G2S {13}
#define DG_DUAL_B_MMA_S2R_OVERLAP {14}
#define DG_DUAL_B_BOUNDED_KLOOP 0
#define DG_DUAL_B_KLOOP_UNROLL 1
#define DG_DUAL_B_KLOOP_CHUNK 0
#define DG_DUAL_B_OVERLAP_PROLOGUE {15}
#define DG_DUAL_B_VALID_EPILOGUE {16}
#define DG_DUAL_B_PACKED_N {17}
#define DG_DUAL_B_ROUTER_M {18}
#define DG_DUAL_B_ROUTER_N {19}
#define DG_DUAL_B_ROUTER_K {20}
#define DG_DUAL_B_STATIC_STAGES {21}
#define DG_DUAL_B_STREAMED_LOW 0
#define DG_DUAL_B_STREAMED_PAIRS 0
#define DG_DUAL_B_PAIRS_M_FIRST 0
#define DG_DUAL_B_DECODE_SPLIT_K {22}
{23}#include <deep_gemm/impls/bf16xfp32_gemm_cute_free.cuh>

namespace deep_gemm {{

using GemmTypeTag = std::integral_constant<GemmType, GemmType::{0}>;

using Kernel = BF16xFp32GemmCuteFreeKernel<{1}, {2}, {3}, {4}, {5}, {6}, {7}, true, {8}, GemmTypeTag{9}>;
static_assert(sizeof(typename Kernel::Params) == {11}, "host/device Params ABI mismatch");
static_assert(Kernel::SharedStorageSize == {12}, "host/device shared storage mismatch");

}} // namespace deep_gemm

extern "C"
__launch_bounds__(deep_gemm::Kernel::MaxThreadsPerBlock, 1)
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
            dual_b::shared_storage_bytes(info.block_m, info.block_n, info.block_k,
                info.warp_m, info.warp_n, info.warp_k, info.num_stages,
                info.packed_dual_n, info.decode_split_k, info.cta_k),
            info.early_g2s ? 1 : 0, info.mma_s2r_overlap ? 1 : 0,
            info.overlap_prologue ? 1 : 0, info.valid_epilogue ? 1 : 0, info.packed_dual_n ? 1 : 0,
            info.static_router ? args.kernel_params.M : 0,
            info.static_router ? args.kernel_params.N : 0,
            info.static_router ? args.kernel_params.K : 0,
            info.static_stages ? 1 : 0, info.decode_split_k,
            info.packed_store ? fmt::format(
                "#define DG_BFOUT_PACKED_STORE 1\n#define DG_BFOUT_CTA_K {}\n"
                "#define DG_BFOUT_ONE_TILE {}\n#define DG_DUAL_B_K_ORDER {}\n",
                int(info.cta_k), int(info.one_tile), info.k_order) : "");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};


namespace dual_b {
enum Pipeline : unsigned {
    PackedStore = 1 << 0, StaticShape = 1 << 1, PackedN = 1 << 2,
    OneTile = 1 << 3, CtaK = 1 << 4, MmaS2ROverlap = 1 << 5,
    OverlapPrologue = 1 << 6, ValidEpilogue = 1 << 7, StaticStages = 1 << 8,
    EarlyG2S = 1 << 9
};

struct ExactKey {
    int m, n, k;
    at::ScalarType output_dtype;
    acArch_t soc;
    int num_sms;

    constexpr bool operator==(const ExactKey& other) const {
        return m == other.m && n == other.n && k == other.k
            && output_dtype == other.output_dtype && soc == other.soc
            && num_sms == other.num_sms;
    }
};

struct ExactConfig {
    ExactKey key;
    int bm, bn, bk, wm, wn, wk, stages, grid;
    unsigned pipeline;
    int decode_split_k, k_order;

    bool has(Pipeline flag) const { return (pipeline & flag) != 0; }
};

static const ExactConfig* exact_config(const ExactKey& key, bool explicit_config, bool use_hgrtc) {
    if (explicit_config) return nullptr;
    static constexpr ExactConfig lut[] = {
        {{8,     384, 7168, at::kBFloat16, AC_PPU0015, 39},  16,  16, 128, 16, 16, 128, 4, 48, PackedStore | StaticShape | PackedN, 2, 0},
        {{16,    384, 7168, at::kBFloat16, AC_PPU0015, 39},  16,  16, 128, 16, 16, 128, 4, 48, PackedStore | StaticShape | PackedN, 2, 0},
        {{128,   384, 7168, at::kBFloat16, AC_PPU0015, 39},  48,  32, 128, 16, 16, 128, 5, 36, PackedStore | StaticShape | OneTile, 1, 0},
        {{256,   384, 7168, at::kBFloat16, AC_PPU0015, 39},  96,  32, 128, 48, 16,  64, 6, 36, PackedStore | StaticShape | OneTile | CtaK, 1, 0},
        {{512,   384, 7168, at::kBFloat16, AC_PPU0015, 39},  96,  64,  64, 48, 16,  64, 5, 36, PackedStore | StaticShape | OneTile | MmaS2ROverlap, 1, 0},
        {{1024,  384, 7168, at::kBFloat16, AC_PPU0015, 39}, 192,  64,  64, 48, 32,  64, 4, 36, PackedStore | StaticShape | OneTile | MmaS2ROverlap, 1, 0},
        {{2048,  384, 7168, at::kBFloat16, AC_PPU0015, 39}, 256,  96,  64, 32, 48,  64, 4, 32, PackedStore | StaticShape | OverlapPrologue, 1, 0},
        {{4096,  384, 7168, at::kBFloat16, AC_PPU0015, 39}, 320, 128,  64, 80, 32,  64, 3, 39, PackedStore, 1, 0},
        {{8192,  384, 7168, at::kBFloat16, AC_PPU0015, 39}, 320, 128,  64, 80, 32,  64, 3, 39, PackedStore | StaticShape | OverlapPrologue | ValidEpilogue, 1, 1},
        {{16384, 384, 7168, at::kBFloat16, AC_PPU0015, 39}, 256, 128,  64, 64, 32,  64, 4, 39, PackedStore | StaticShape | OverlapPrologue | ValidEpilogue | StaticStages, 1, 6},
        {{32768, 384, 7168, at::kBFloat16, AC_PPU0015, 39}, 256, 128,  64, 64, 16,  64, 4,  0, 0, 1, 0},
        {{8,     384, 7168, at::kFloat,   AC_PPU0015, 39},  16,  16, 128, 16, 16, 128, 4,  0, StaticShape | PackedN, 2, 0},
        {{16,    384, 7168, at::kFloat,   AC_PPU0015, 39},  16,  16, 128, 16, 16, 128, 4,  0, StaticShape | PackedN, 2, 0},
        {{128,   384, 7168, at::kFloat,   AC_PPU0015, 39},  64,  32, 128, 16, 16, 128, 4,  0, StaticShape | EarlyG2S, 1, 0},
        {{256,   384, 7168, at::kFloat,   AC_PPU0015, 39},  96,  32,  64, 32, 16,  64, 7,  0, StaticShape | EarlyG2S, 1, 0},
        {{512,   384, 7168, at::kFloat,   AC_PPU0015, 39}, 128,  64,  64, 32, 16,  64, 4,  0, StaticShape, 1, 0},
        {{1024,  384, 7168, at::kFloat,   AC_PPU0015, 39}, 192,  64,  64, 32, 16,  64, 4,  0, 0, 1, 0},
        {{2048,  384, 7168, at::kFloat,   AC_PPU0015, 39}, 128, 192,  64, 32, 32,  64, 4,  0, 0, 1, 0},
        {{4096,  384, 7168, at::kFloat,   AC_PPU0015, 39}, 320, 128,  64, 80, 32,  64, 2,  0, StaticShape | ValidEpilogue, 1, 0},
        {{8192,  384, 7168, at::kFloat,   AC_PPU0015, 39}, 320, 128,  64, 80, 32,  64, 3,  0, 0, 1, 0},
        {{16384, 384, 7168, at::kFloat,   AC_PPU0015, 39}, 256, 128,  64, 64, 32,  64, 4,  0, StaticShape | OverlapPrologue | ValidEpilogue, 1, 0},
        {{32768, 384, 7168, at::kFloat,   AC_PPU0015, 39}, 256, 128,  64, 64, 16,  64, 4,  0, 0, 1, 0},
        {{512,  2048, 4096, at::kBFloat16, AC_PPU0015, 39}, 128, 256,  64, 64, 32,  64, 3,  0, 0, 1, 0},
        {{512,  2048, 4096, at::kFloat,   AC_PPU0015, 39}, 128, 256,  64, 64, 32,  64, 3,  0, 0, 1, 0},
    };
    for (const auto& entry : lut) {
        if (entry.key == key) {
            if (use_hgrtc && entry.has(PackedStore)) return nullptr;
            return &entry;
        }
    }
    return nullptr;
}
} // namespace dual_b

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

// Fixed tile repertoire, analogous to the ordinary BF16 selector's
// hardcoded candidates. The output/N admissibility whitelist is retained
// separately from the analytic resource bounds; it predicts no performance.
static constexpr unsigned kBf16TailN = 1, kBf16AlignedN = 2;
static constexpr unsigned kFp32TailN = 4, kFp32AlignedN = 8;
static constexpr unsigned kAnyN = kBf16TailN | kBf16AlignedN | kFp32TailN | kFp32AlignedN;
static constexpr unsigned kAlignedN = kBf16AlignedN | kFp32AlignedN;
static constexpr unsigned kTailN = kBf16TailN | kFp32TailN;
struct DualBPolicyTile { int bm, bn, bk, wm, wn, stages; unsigned supported_n; };
static constexpr DualBPolicyTile kDualBSmallTiles[] = {
    {16, 16, 64, 16, 16, 2, kAlignedN}, {16, 16, 64, 16, 16, 3, kAlignedN},
    {16, 16, 128, 16, 16, 2, kAlignedN}, {16, 16, 128, 16, 16, 3, kAlignedN},
    {16, 32, 64, 16, 16, 2, kAlignedN}, {16, 32, 64, 16, 16, 3, kAlignedN},
    {16, 32, 128, 16, 16, 3, kAlignedN}, {16, 64, 64, 16, 16, 2, kAlignedN},
    {16, 64, 64, 16, 16, 3, kAlignedN}, {16, 64, 128, 16, 16, 2, kAlignedN},
    {16, 64, 128, 16, 16, 3, kAlignedN}, {16, 128, 64, 16, 16, 2, kAnyN},
    {16, 128, 64, 16, 16, 3, kAnyN}, {16, 128, 128, 16, 16, 2, kAnyN},
    {16, 128, 128, 16, 16, 3, kAnyN}, {16, 192, 64, 16, 16, 2, kAnyN},
    {16, 192, 64, 16, 16, 3, kAnyN}, {16, 192, 128, 16, 16, 2, kAnyN},
    {16, 256, 64, 16, 16, 2, kAnyN}, {16, 256, 64, 16, 16, 3, kAnyN},
};
static constexpr DualBPolicyTile kDualBGeneralTiles[] = {
    {32, 64, 64, 16, 16, 2, kAlignedN}, {32, 64, 64, 16, 16, 4, kAlignedN}, {32, 64, 128, 16, 16, 2, kAlignedN}, {32, 64, 128, 16, 16, 4, kAlignedN},
    {32, 128, 64, 16, 16, 2, kAnyN}, {32, 128, 64, 16, 64, 2, kAnyN}, {32, 128, 64, 32, 32, 2, kAnyN}, {32, 128, 128, 16, 16, 2, kAnyN},
    {32, 192, 64, 16, 16, 4, kAnyN}, {32, 192, 64, 16, 32, 4, kAnyN}, {32, 192, 64, 32, 16, 4, kAnyN}, {32, 192, 64, 32, 32, 2, kAnyN},
    {32, 192, 128, 16, 16, 2, kAnyN}, {32, 256, 64, 16, 32, 3, kAnyN}, {32, 256, 64, 32, 32, 3, kAnyN}, {48, 64, 64, 16, 16, 2, kAlignedN},
    {48, 64, 128, 16, 16, 2, kAlignedN}, {48, 128, 64, 16, 16, 3, kAnyN}, {48, 128, 64, 48, 16, 2, kTailN}, {48, 128, 128, 16, 16, 2, kAnyN},
    {48, 192, 64, 16, 32, 2, kAnyN}, {48, 192, 64, 16, 32, 4, kAnyN}, {48, 192, 64, 16, 48, 2, kAnyN}, {48, 192, 64, 16, 48, 4, kAnyN},
    {48, 192, 64, 48, 16, 2, kAnyN}, {48, 192, 128, 16, 32, 2, kAnyN}, {48, 256, 64, 16, 64, 3, kAnyN}, {48, 256, 64, 48, 16, 2, kAnyN},
    {64, 64, 64, 16, 16, 2, kAlignedN}, {64, 64, 128, 16, 16, 2, kAlignedN}, {64, 64, 128, 16, 16, 3, kAlignedN}, {64, 64, 128, 16, 16, 4, kAlignedN},
    {64, 128, 64, 16, 64, 2, kTailN}, {64, 128, 64, 32, 32, 2, kTailN}, {64, 128, 128, 16, 16, 2, kAnyN}, {64, 192, 64, 16, 32, 2, kAnyN},
    {64, 192, 64, 16, 32, 4, kAnyN}, {64, 192, 64, 32, 16, 3, kAnyN}, {64, 192, 64, 32, 48, 2, kTailN}, {64, 192, 128, 16, 32, 2, kAnyN},
    {96, 64, 64, 16, 16, 2, kAlignedN}, {96, 64, 64, 16, 16, 4, kAlignedN}, {96, 64, 64, 16, 16, 5, kAlignedN}, {96, 64, 64, 48, 16, 3, kAlignedN},
    {96, 128, 64, 16, 32, 2, kAnyN}, {96, 128, 128, 16, 32, 2, kAnyN}, {96, 192, 64, 16, 48, 2, kAnyN}, {96, 192, 64, 32, 32, 4, kAnyN},
    {96, 192, 64, 48, 16, 3, kAnyN}, {96, 192, 64, 48, 16, 4, kAnyN}, {96, 192, 64, 48, 32, 2, kTailN}, {96, 192, 64, 48, 48, 2, kTailN},
    {96, 192, 128, 16, 48, 2, kAnyN}, {96, 256, 64, 32, 32, 2, kAnyN}, {96, 256, 64, 32, 32, 3, kAnyN}, {96, 256, 64, 32, 64, 3, kAnyN},
    {96, 256, 64, 48, 16, 2, kAnyN}, {96, 256, 64, 48, 16, 3, kAnyN}, {96, 256, 64, 48, 32, 2, kAnyN}, {96, 256, 64, 48, 32, 3, kAnyN},
    {128, 128, 64, 32, 32, 2, kAlignedN}, {128, 128, 64, 64, 16, 2, kAlignedN}, {128, 128, 128, 16, 32, 2, kAnyN},
    {128, 192, 64, 32, 32, 2, kAnyN}, {128, 192, 64, 32, 32, 3, kAnyN}, {128, 192, 64, 32, 32, 4, kAnyN}, {128, 192, 64, 32, 48, 2, kAnyN},
    {128, 192, 64, 64, 16, 2, kAnyN}, {128, 192, 64, 64, 16, 4, kAnyN}, {128, 256, 64, 32, 32, 2, kAnyN}, {128, 256, 64, 32, 32, 3, kAnyN},
    {128, 256, 64, 32, 64, 2, kAnyN}, {128, 256, 64, 32, 64, 3, kAnyN}, {128, 256, 64, 64, 16, 3, kAnyN}, {128, 256, 64, 64, 32, 2, kAnyN},
    {128, 256, 64, 64, 32, 3, kAnyN}, {192, 64, 64, 32, 16, 4, kAlignedN}, {192, 64, 64, 32, 16, 5, kAlignedN}, {192, 128, 64, 32, 32, 2, kAnyN},
    {192, 128, 64, 48, 32, 2, kAlignedN}, {192, 128, 128, 48, 16, 2, kAnyN}, {192, 192, 64, 32, 48, 2, kFp32AlignedN}, {192, 192, 64, 32, 48, 3, kTailN | kFp32AlignedN},
    {192, 192, 64, 48, 32, 2, kAnyN}, {192, 192, 64, 48, 32, 3, kAnyN}, {192, 192, 64, 48, 48, 2, kAnyN}, {192, 192, 64, 48, 48, 3, kAnyN},
    {256, 128, 64, 32, 32, 3, kAnyN}, {256, 128, 64, 32, 32, 4, kAnyN}, {256, 128, 64, 32, 64, 2, kAnyN}, {256, 128, 64, 32, 64, 3, kAlignedN},
    {256, 128, 64, 64, 16, 3, kAnyN}, {256, 128, 64, 64, 16, 4, kAnyN}, {256, 128, 64, 64, 32, 2, kAnyN}, {256, 128, 128, 32, 32, 2, kAnyN},
};

// A + B_high + B_low staged in BF16, and two FP32 accumulator tiles.
// These are conservative static bounds, not measured hardware occupancy.
static bool dual_b_tile_supported(const DualBPolicyTile& t, int n, bool fp32) {
    if (t.bm <= 0 || t.bn <= 0 || t.bk < 64 || t.bk % 64 || t.stages < 2
        || t.wm <= 0 || t.wn <= 0 || t.wm % 16 || t.wn % 16
        || t.bm % t.wm || t.bn % t.wn) return false;
    const unsigned required = fp32
        ? (n % t.bn == 0 ? kFp32AlignedN : kFp32TailN)
        : (n % t.bn == 0 ? kBf16AlignedN : kBf16TailN);
    if (!(t.supported_n & required)) return false;
    const int warps = (t.bm / t.wm) * (t.bn / t.wn);
    if (warps < 1 || warps > 32) return false;
    const int we = ceil_div(warps, 8);
    const int register_limit = we == 3 ? 168 : std::min(256, 512 / we);
    const int registers = 2 * t.wm * t.wn / 32 + (t.wm + t.wn) / 2 + 20;
    const int smem = (t.stages * (t.bm + 2 * t.bn) * t.bk * 2 + 127) & ~127;
    return smem <= 256 * 1024 && registers <= register_limit;
}

// White-box analogue of adaptive_tile_selector.hpp's wave/makespan policy.
// Ordinary BF16 uses BM*BN / 96 versus BM+BN. Dual-B does twice the
// MMA work and reads two B operands: BM*BN / 48 versus BM+2*BN.
// This fixed analytic ratio is not fitted to timing samples. M/N/K padding
// pays for complete tiles; WE imbalance pays for unused eight-warp slots.
// Exact measured workload overrides are kept separate from this heuristic.
template <size_t TileCount>
static DualBPolicyTile select_dual_b_tile(int m, int n, int k, int num_sms, bool fp32,
    const DualBPolicyTile (&tiles)[TileCount]) {
    DG_HOST_ASSERT(m > 0 && n > 0 && k > 0 && num_sms > 0);
    using Rank = std::tuple<double, double, int, int, int, int, int, int, int>;
    Rank best{1e300, 1e300, 0, 0, 0, 0, 0, 0, 0};
    DualBPolicyTile selected{16, 128, 64, 16, 16, 2, kAnyN};
    bool found = false;
    for (const auto& t : tiles) {
        if (!dual_b_tile_supported(t, n, fp32)) continue;
        const int warps = (t.bm / t.wm) * (t.bn / t.wn);
        const double we_penalty = double(ceil_div(warps, 8) * 8) / warps;
        const double blocks = double(ceil_div(m, t.bm)) * ceil_div(n, t.bn);
        const double waves = std::ceil(blocks / num_sms);
        const int k_tiles = ceil_div(k, t.bk);
        const double padded_k = double(k_tiles) * t.bk;
        const double compute = double(t.bm) * t.bn * we_penalty / 48.0;
        const double operands = t.bm + 2.0 * t.bn;
        const double makespan = waves * padded_k * std::max(compute, operands);
        const double traffic = blocks * (2.0 * padded_k * operands
            + (fp32 ? 4.0 : 2.0) * t.bm * t.bn);
        // Equal makespan: less traffic, fewer K-loop iterations, enough stages
        // to cover the pipeline, then more register reuse and stable geometry.
        const int useful_stages = std::min(t.stages, std::max(2, k_tiles));
        const Rank rank{makespan, traffic, k_tiles, -useful_stages,
                        -(t.wm * t.wn), t.bm, t.bn, t.wm, t.wn};
        if (!found || rank < best) { best = rank; selected = t; found = true; }
    }
    DG_HOST_ASSERT(found);
    return selected;
}

static DualBPolicyTile get_dual_b_policy_tile(int m, int n, int k, int num_sms, bool fp32) {
    // Cache only shape, SM count and output dtype; never input/output pointers.
    struct Cache { int m = 0, n = 0, k = 0, sms = 0; bool fp32 = false; DualBPolicyTile tile{}; };
    static thread_local Cache cache;
    if (cache.m == m && cache.n == n && cache.k == k && cache.sms == num_sms && cache.fp32 == fp32)
        return cache.tile;
    const auto tile = m <= 16
        ? select_dual_b_tile(m, n, k, num_sms, fp32, kDualBSmallTiles)
        : select_dual_b_tile(m, n, k, num_sms, fp32, kDualBGeneralTiles);
    cache = Cache{m, n, k, num_sms, fp32, tile};
    return tile;
}

// The PPU1.5 dual-B fused launcher validates configs against dual accumulators and the dual-B SMEM budget.
// Unsupported explicit configs raise an error instead of falling back to composition.
template <typename ElementOutput>
static void bf16xfp32_gemm_cute_free_impl(const torch::Tensor& lhs, const torch::Tensor& w_high,
                                          const torch::Tensor& w_low, const torch::Tensor& out,
                                          const double& scale, const int& m, const int& n, const int& k,
                                          std::optional<ConfigTuple> configs = std::nullopt) {
    int num_sms = get_num_sms();
    const dual_b::ExactKey key{m, n, k, out.scalar_type(), get_ppu_arch(), num_sms};
    const auto* exact = dual_b::exact_config(key, configs.has_value(), get_env<int>("DG_JIT_USE_HGRTC", 0));
    const bool cta_k = exact && exact->has(dual_b::CtaK);
    const bool packed_store = exact && exact->has(dual_b::PackedStore);
    const bool packed_dual_n = exact && exact->has(dual_b::PackedN);
    const bool static_router = exact && exact->has(dual_b::StaticShape);
    const bool static_stages = exact && exact->has(dual_b::StaticStages);
    const bool early_g2s = exact && exact->has(dual_b::EarlyG2S);
    const bool mma_s2r_overlap = exact && exact->has(dual_b::MmaS2ROverlap);
    const bool overlap_prologue = exact && exact->has(dual_b::OverlapPrologue);
    const bool valid_epilogue = exact && exact->has(dual_b::ValidEpilogue);
    const int decode_split_k = exact ? exact->decode_split_k : 1;
    static constexpr GemmType kGemmType = GemmType::Normal;
    DG_HOST_ASSERT(decode_split_k == 1 || (packed_dual_n && static_router
        && (decode_split_k == 2 || decode_split_k == 4 || decode_split_k == 8)));
    DG_HOST_ASSERT(!static_stages || (static_router && !packed_dual_n));
    DG_HOST_ASSERT(!mma_s2r_overlap || !early_g2s);
    DG_HOST_ASSERT(!(overlap_prologue || valid_epilogue) || (!early_g2s && !mma_s2r_overlap));
    DG_HOST_ASSERT(!packed_dual_n || (!cta_k && !early_g2s && !mma_s2r_overlap
        && !overlap_prologue && !valid_epilogue));

    using Config = std::tuple<int, int, int, int, int, int, int, int>;
    Config cfg;
    if (configs.has_value()) {
        // Explicit config path: unpack the 8-element public tuple; force warp_k = block_k
        // (same contract as the plain BF16 dense path).
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, /*warp_k=*/bk, nst);
    } else if (exact) {
        cfg = std::make_tuple(num_sms, exact->bm, exact->bn, exact->bk,
            exact->wm, exact->wn, exact->wk, exact->stages);
    } else {
        const auto t = get_dual_b_policy_tile(m, n, k, num_sms, std::is_same_v<ElementOutput, float>);
        cfg = std::make_tuple(std::min(num_sms, ceil_div(m, t.bm) * ceil_div(n, t.bn)),
            t.bm, t.bn, t.bk, t.wm, t.wn, t.bk, t.stages);
    }
    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages] = cfg;
    DG_HOST_ASSERT(!packed_dual_n || ((std::is_same_v<ElementOutput, float> || packed_store)
                   && block_m > 0 && block_n > 0 && block_k >= 64 && block_k % 64 == 0
                   && block_m % 16 == 0 && block_n % 16 == 0 && warp_m == 16 && warp_n == 16
                   && (block_m / 16) * (block_n / 16) * decode_split_k <= 32));

    // Budget the full dual-B SMEM footprint while preserving stage pruning for explicit configs.
    const int kSmemPerCU = static_cast<int>(device_runtime->get_prop()->sharedMemPerMultiprocessor);
    constexpr int kElemBytes = static_cast<int>(sizeof(cutlass::bfloat16_t));
    DG_HOST_ASSERT(block_m > 0 && block_n > 0 && block_k > 0
        && warp_m > 0 && warp_n > 0 && warp_k > 0);
    DG_HOST_ASSERT(block_m % warp_m == 0 && block_n % warp_n == 0 && block_k % warp_k == 0);
    auto dual_smem_bytes = [&](int stages) {
        return dual_b::shared_storage_bytes(block_m, block_n, block_k, warp_m, warp_n,
            warp_k, stages, packed_dual_n, decode_split_k, cta_k, kElemBytes);
    };
    while (configs.has_value() && num_stages > 2 && dual_smem_bytes(num_stages) > kSmemPerCU)
        --num_stages;
    const int SMSIZE = dual_smem_bytes(num_stages);
    // Never disguise composition as fusion; report explicit configs that exceed the budget as unsupported.
    DG_HOST_ASSERT(num_sms_new > 0 && num_stages >= 2);
    DG_HOST_ASSERT(SMSIZE <= kSmemPerCU);
    DG_HOST_ASSERT(cta_k || warp_k == block_k);

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
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * warps_k * decode_split_k * 32;

    // SMEM-based occupancy heuristic on the dual-B footprint.
    int blocks_per_cu = kSmemPerCU / SMSIZE;
    if (blocks_per_cu < 1) blocks_per_cu = 1;

    dim3 grid;
    // Cap the persistent grid at the output-tile count to avoid idle CTAs.
    const int output_tiles = ceil_div(m, block_m) * ceil_div(n, packed_dual_n ? block_n / 2 : block_n);
    grid.x = exact && exact->grid > 0 ? exact->grid : std::min(blocks_per_cu * num_sms_new, output_tiles);
    DG_HOST_ASSERT(!exact || (grid.x <= output_tiles && (!exact->has(dual_b::OneTile) || grid.x == output_tiles)));

    // Output dtype is part of the JIT cache key (separate kernel name).
    std::string kernel_name = "bf16xfp32_gemm_cute_free";
    if (std::is_same_v<ElementOutput, float>) kernel_name += "_fp32";
    if (packed_dual_n) kernel_name += "_packed_dual_n";
    if (static_router) kernel_name += "_static_router_m" + std::to_string(m);
    if (decode_split_k > 1) kernel_name += "_decode_sk" + std::to_string(decode_split_k);
    if (static_stages) kernel_name += "_static_stages";
    if (early_g2s) kernel_name += "_early_g2s";
    if (mma_s2r_overlap) kernel_name += "_mma_s2r_overlap";
    if (overlap_prologue) kernel_name += "_overlap_prologue";
    if (valid_epilogue) kernel_name += "_valid_epilogue";

    auto args = typename Runtime::Args{
        .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages,
                        "Normal", kernel_name, early_g2s, mma_s2r_overlap,
                        overlap_prologue, valid_epilogue, packed_dual_n, static_router, static_stages,
                        decode_split_k, packed_store, cta_k,
                        exact && exact->has(dual_b::OneTile), exact ? exact->k_order : 0},
        .launch_args = {grid, block, SMSIZE},
        .kernel_params = hparams,
    };

    auto runtime = compiler->build_cached<Runtime>(kernel_name, args);
    if (is_compile_only())
        return;
    auto kernel = runtime->kernel;

    const auto& stream = current_stream();

    DgProfParam dg_prof_params;
    if (deep_gemm::profiling_enabled()) {
        dg_prof_params.set_params(kGemmType, false, std::string("bf16xfp32"), 1, m, n, k, 0, nullptr, stream);
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

        printf("[DenseGemm_BF16xFP32_CuteFree:]\n");
        printf("problem:[%d, %d, %d], scale:%g\n", m, n, k, scale);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu, grid.x);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n",
               block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", SMSIZE, int(numRegs), int(localSize));
        printf("dual_b_decode_split_k:%d\n", decode_split_k);
        printf("dual_b_exact_lut:%d, cta_k:%d, k_order:%d\n",
               int(exact != nullptr), int(cta_k), args.launch_info.k_order);
        printf("dual_b_static_router:%d, static_stages:%d\n", int(static_router), int(static_stages));
        const char* dense_pipeline = overlap_prologue
            ? (valid_epilogue ? "overlap_valid" : "overlap_prologue") : "valid_epilogue";
        printf("dual_b_pipeline:%s\n", packed_dual_n ? "packed_dual_n" : (overlap_prologue || valid_epilogue) ? dense_pipeline
               : (mma_s2r_overlap ? "mma_s2r_overlap" : (early_g2s ? "early_g2s" : "double_buffer")));
    }
}

// Public launcher: dispatches on the device generation.
//   PPU1.5 : All M use dual-B fusion with FP32 accumulation and BF16 or FP32 output.
//            Exact FP32 decode may partition K and reduce both accumulation chains inside the CTA.
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
            bf16xfp32_gemm_cute_free_impl<float>(lhs, w_high, w_low, out, scale, m, n, k, configs);
        else
            bf16xfp32_gemm_cute_free_impl<cutlass::bfloat16_t>(lhs, w_high, w_low, out, scale, m, n, k, configs);
        return;
    }

    // --- PPU1.0 fallback: compose two BF16 GEMMs (both accumulate in FP32). ---
    bf16xfp32_gemm_composed(lhs, w_high, w_low, out, scale, m, n, k, configs);
}

} // namespace deep_gemm
