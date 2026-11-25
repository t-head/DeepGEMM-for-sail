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
        bool full_k_prefetch, early_g2s, mma_s2r_overlap, bounded_kloop;
        int kloop_unroll;
        bool kloop_chunk;
        bool overlap_prologue, valid_epilogue, packed_dual_n, static_router, static_stages, streamed_low, standard_s2, streamed_pairs;
        int min_blocks;
        bool pairs_m_first;
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
            i.num_stages, i.gemm_type, i.kernel_name, i.full_k_prefetch, i.early_g2s,
            i.mma_s2r_overlap, i.bounded_kloop, i.kloop_unroll, i.kloop_chunk,
            i.overlap_prologue, i.valid_epilogue, i.packed_dual_n, i.static_router,
            i.static_stages, i.streamed_low, i.standard_s2, i.streamed_pairs,
            i.min_blocks, i.pairs_m_first, i.decode_split_k, i.packed_store, i.cta_k,
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
#define DG_DUAL_B_DECODE_SPLIT_K {31}
{32}#include <deep_gemm/impls/bf16xfp32_gemm_cute_free.cuh>

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
            dual_b::shared_storage_bytes(info.block_m, info.block_n, info.block_k,
                info.warp_m, info.warp_n, info.warp_k, info.num_stages,
                info.packed_dual_n, info.decode_split_k, info.cta_k),
            info.full_k_prefetch ? 1 : 0, info.early_g2s ? 1 : 0, info.mma_s2r_overlap ? 1 : 0,
            info.bounded_kloop ? 1 : 0, info.kloop_unroll, info.kloop_chunk ? 1 : 0,
            info.overlap_prologue ? 1 : 0, info.valid_epilogue ? 1 : 0, info.packed_dual_n ? 1 : 0,
            info.static_router ? args.kernel_params.M : 0,
            info.static_router ? args.kernel_params.N : 0,
            info.static_router ? args.kernel_params.K : 0,
            info.static_stages ? 1 : 0, info.streamed_low ? 1 : 0,
            info.standard_s2 ? "false" : "true", info.streamed_pairs ? 1 : 0, info.min_blocks,
            info.pairs_m_first ? 1 : 0, info.decode_split_k,
            info.packed_store ? fmt::format(
                "#define DG_BFOUT_PACKED_STORE 1\n#define DG_BFOUT_CTA_K {}\n"
                "#define DG_BFOUT_ONE_TILE {}\n#define DG_DUAL_B_K_ORDER {}\n",
                int(info.cta_k), int(info.one_tile), info.k_order) : "");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};


static bool dual_b_no_experimental_overrides() {
    for (const char* name : {
        "DG_BF16XFP32_PACKED_DUAL_N", "DG_BF16XFP32_STATIC_ROUTER", "DG_BF16XFP32_DECODE_SPLIT_K",
        "DG_BF16XFP32_FULL_K_PREFETCH", "DG_BF16XFP32_EARLY_G2S", "DG_BF16XFP32_MMA_S2R_OVERLAP",
        "DG_BF16XFP32_BOUNDED_KLOOP", "DG_BF16XFP32_OVERLAP_PROLOGUE", "DG_BF16XFP32_VALID_EPILOGUE",
        "DG_BF16XFP32_STATIC_STAGES", "DG_BF16XFP32_STREAMED_LOW", "DG_BF16XFP32_STANDARD_S2",
        "DG_BF16XFP32_STREAMED_PAIRS", "DG_BF16XFP32_LAUNCH_BOUNDS2", "DG_BF16XFP32_PAIRS_M_FIRST",
        "DG_BF16XFP32_KLOOP_UNROLL", "DG_BF16XFP32_KLOOP_CHUNK"}) {
        if (std::getenv(name) != nullptr) return false;
    }
    return true;
}

namespace dual_b {
enum Pipeline : unsigned {
    PackedStore = 1 << 0, StaticShape = 1 << 1, PackedN = 1 << 2,
    OneTile = 1 << 3, CtaK = 1 << 4, MmaS2ROverlap = 1 << 5,
    OverlapPrologue = 1 << 6, ValidEpilogue = 1 << 7, StaticStages = 1 << 8
};

struct ExactConfig {
    int m, bm, bn, bk, wm, wn, wk, stages, grid;
    unsigned pipeline;
    int decode_split_k, k_order;

    bool has(Pipeline flag) const { return (pipeline & flag) != 0; }
};

static const ExactConfig* exact_config(int m, int n, int k, int sms, double scale,
                                       bool fp32_output, bool explicit_config) {
    if (fp32_output || explicit_config || n != 384 || k != 7168 || sms != 39
        || scale != 1.0 / 256.0 || get_env<int>("DG_JIT_USE_HGRTC", 0)
        || !dual_b_no_experimental_overrides())
        return nullptr;
    static constexpr ExactConfig lut[] = {
        {8,     16,  16, 128, 16, 16, 128, 4, 48, PackedStore | StaticShape | PackedN, 2, 0},
        {16,    16,  16, 128, 16, 16, 128, 4, 48, PackedStore | StaticShape | PackedN, 2, 0},
        {128,   48,  32, 128, 16, 16, 128, 5, 36, PackedStore | StaticShape | OneTile, 1, 0},
        {256,   96,  32, 128, 48, 16,  64, 6, 36, PackedStore | StaticShape | OneTile | CtaK, 1, 0},
        {512,   96,  64,  64, 48, 16,  64, 5, 36, PackedStore | StaticShape | OneTile | MmaS2ROverlap, 1, 0},
        {1024, 192,  64,  64, 48, 32,  64, 4, 36, PackedStore | StaticShape | OneTile | MmaS2ROverlap, 1, 0},
        {2048, 256,  96,  64, 32, 48,  64, 4, 32, PackedStore | StaticShape | OverlapPrologue, 1, 0},
        {4096, 320, 128,  64, 80, 32,  64, 3, 39, PackedStore, 1, 0},
        {8192, 320, 128,  64, 80, 32,  64, 3, 39, PackedStore | StaticShape | OverlapPrologue | ValidEpilogue, 1, 1},
        {16384,256, 128,  64, 64, 32,  64, 4, 39, PackedStore | StaticShape | OverlapPrologue | ValidEpilogue | StaticStages, 1, 6},
    };
    for (const auto& entry : lut)
        if (entry.m == m) return &entry;
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

// Exact default geometries captured from the 7169b71 regression baseline.
// Only these measured router points bypass the generic heuristic; no nearest
// shape lookup or extrapolation. Existing specialized router paths take priority.
static std::optional<DualBPolicyTile> get_dual_b_measured_tile(int m, int n, int k, int num_sms, bool fp32) {
    if (num_sms != 39 || n != 384 || k != 7168) return std::nullopt;
    struct Entry { int m; bool fp32; DualBPolicyTile tile; };
    static constexpr Entry entries[] = {
        {1024, true, {192, 64, 64, 32, 16, 4, kFp32AlignedN}},
        {2048, true, {128, 192, 64, 32, 32, 4, kFp32AlignedN}},
        {32768, true, {256, 128, 64, 64, 16, 4, kAnyN}},
        {32768, false, {256, 128, 64, 64, 16, 4, kAnyN}},
    };
    for (const auto& e : entries)
        if (e.m == m && e.fp32 == fp32) return e.tile;
    return std::nullopt;
}

static DualBPolicyTile get_dual_b_policy_tile(int m, int n, int k, int num_sms, bool fp32) {
    // Cache only shape, SM count and output dtype; never input/output pointers.
    struct Cache { int m = 0, n = 0, k = 0, sms = 0; bool fp32 = false; DualBPolicyTile tile{}; };
    static thread_local Cache cache;
    if (cache.m == m && cache.n == n && cache.k == k && cache.sms == num_sms && cache.fp32 == fp32)
        return cache.tile;
    const auto measured = get_dual_b_measured_tile(m, n, k, num_sms, fp32);
    const auto tile = measured.has_value() ? *measured : (m <= 16
        ? select_dual_b_tile(m, n, k, num_sms, fp32, kDualBSmallTiles)
        : select_dual_b_tile(m, n, k, num_sms, fp32, kDualBGeneralTiles));
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
    const auto* exact = dual_b::exact_config(m, n, k, num_sms, scale,
        std::is_same_v<ElementOutput, float>, configs.has_value());
    const bool cta_k = exact && exact->has(dual_b::CtaK);
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
    const bool packed_dual_n = exact ? exact->has(dual_b::PackedN) : (packed_env != nullptr && *packed_env == '1')
        || (packed_env == nullptr && !other_pipeline && !configs.has_value()
            && std::is_same_v<ElementOutput, float> && (m == 8 || m == 16)
            && n == 384 && k == 7168);

    // Specialize dimensions and strides for exact router shapes to remove repeated DMA address calculations without changing K order.
    // Add M16 specialization only to the packed default path; explicit configs retain their behavior unless the experiment is enabled.
    const char* static_env = std::getenv("DG_BF16XFP32_STATIC_ROUTER");
    DG_HOST_ASSERT(static_env == nullptr || std::string(static_env) == "0" || std::string(static_env) == "1");
    const bool static_router = exact ? exact->has(dual_b::StaticShape) : (static_env == nullptr || *static_env == '1')
        && std::is_same_v<ElementOutput, float>
        && (m == 8 || (m == 16 && packed_dual_n
            && (!configs.has_value() || (static_env != nullptr && *static_env == '1')))
            || m == 128 || m == 256 || m == 512 || m == 4096 || m == 16384) && n == 384 && k == 7168;

    // Validated exact decode default: partition K among warps and reduce high/low FP32 chains inside the CTA.
    // Explicit configs and disabled packed/static paths retain one full-K partition unless opted in.
    const char* split_env = std::getenv("DG_BF16XFP32_DECODE_SPLIT_K");
    DG_HOST_ASSERT(split_env == nullptr || std::string(split_env) == "1" || std::string(split_env) == "2"
                   || std::string(split_env) == "4" || std::string(split_env) == "8");
    const bool split_default = !configs.has_value() && !other_pipeline && packed_dual_n && static_router
        && std::is_same_v<ElementOutput, float> && (m == 8 || m == 16) && n == 384 && k == 7168;
    const int decode_split_k = exact ? exact->decode_split_k
        : split_env == nullptr ? (split_default ? 2 : 1) : *split_env - '0';
    DG_HOST_ASSERT(decode_split_k == 1 || (packed_dual_n && static_router
                   && (std::is_same_v<ElementOutput, float> || exact) && (m == 8 || m == 16)
                   && n == 384 && k == 7168));

    using Config = std::tuple<int, int, int, int, int, int, int, int, std::tuple<int, int, int>>;

    Config cfg;
    if (configs.has_value()) {
        // Explicit config path: unpack the 8-element public tuple; force warp_k = block_k
        // (same contract as the plain BF16 dense path).
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, /*warp_k=*/bk, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else if (exact) {
        cfg = std::make_tuple(num_sms, exact->bm, exact->bn, exact->bk,
            exact->wm, exact->wn, exact->wk, exact->stages, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && (m == 8 || m == 16)
               && n == 384 && k == 7168) {
        // Two K partitions with BK128/stage4 reduce decode latency in both hot and 64-layer rotation.
        // The split=1 override restores legacy packed geometry; explicit configs, BF16 output, and PPU1.0 remain unchanged.
        const int decode_bk = decode_split_k > 1 ? 128 : 512;
        cfg = packed_dual_n
            ? std::make_tuple(std::min(num_sms, 39), 16, 16, decode_bk,
                              16, 16, decode_bk, 4, std::make_tuple(0, 0, 0))
            : std::make_tuple(std::min(num_sms, 24), 16, 16, 1024,
                              16, 16, 1024, 2, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 128 && n == 384 && k == 7168) {
        // Short-to-medium router prefill: increase output-tile parallelism while both FP32 accumulation chains cover full K.
        cfg = std::make_tuple(std::min(num_sms, 39), 64, 32, 128,
                              16, 16, 128, 4, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 256 && n == 384 && k == 7168) {
        // Exact M256 uses 36 output tiles and a smaller dual-B SMEM footprint, validated against unrestricted FP32 BLAS.
        // Preserve the opt-in packed geometry; the default keeps both independent FP32 chains over full K.
        cfg = packed_dual_n
            ? std::make_tuple(std::min(num_sms, 39), 128, 32, 128,
                              16, 16, 128, 5, std::make_tuple(0, 0, 0))
            : std::make_tuple(std::min(num_sms, 39), 96, 32, 64,
                              32, 16, 64, 7, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 512 && n == 384 && k == 7168) {
        // Exact M512 increases warp-M reuse while preserving separate high/low FP32 accumulation chains over full K.
        cfg = std::make_tuple(std::min(num_sms, 39), 128, 64, 64,
                              32, 16, 64, 4, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 4096 && n == 384 && k == 7168) {
        cfg = std::make_tuple(std::min(num_sms, 39), 320, 128, 64,
                              80, 32, 64, 2, std::make_tuple(0, 0, 0));
    } else if (std::is_same_v<ElementOutput, float> && m == 8192 && n == 384 && k == 7168) {
        cfg = std::make_tuple(std::min(num_sms, 39), 320, 128, 64,
                              80, 32, 64, 3, std::make_tuple(0, 0, 0));
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
    DG_HOST_ASSERT(!packed_dual_n || ((std::is_same_v<ElementOutput, float> || exact)
                   && (m == 8 || m == 16 || m == 128 || m == 256) && n == 384 && k == 7168
                   && block_m > 0 && block_n > 0 && block_k >= 64 && block_k % 64 == 0
                   && block_m % 16 == 0 && block_n % 16 == 0 && warp_m == 16 && warp_n == 16
                   && (block_m / 16) * (block_n / 16) * decode_split_k <= 32));

    // Budget the full dual-B SMEM footprint while preserving stage pruning for explicit configs.
    constexpr int kSmemPerCU = 256 * 1024;
    constexpr int kElemBytes = static_cast<int>(sizeof(cutlass::bfloat16_t));
    DG_HOST_ASSERT(block_m > 0 && block_n > 0 && block_k > 0
        && warp_m > 0 && warp_n > 0 && warp_k > 0);
    DG_HOST_ASSERT(block_m % warp_m == 0 && block_n % warp_n == 0 && block_k % warp_k == 0);
    auto dual_smem_bytes = [&](int stages) {
        return dual_b::shared_storage_bytes(block_m, block_n, block_k, warp_m, warp_n,
            warp_k, stages, packed_dual_n, decode_split_k, cta_k, kElemBytes);
    };
    while (!exact && num_stages > 2 && dual_smem_bytes(num_stages) > kSmemPerCU)
        --num_stages;
    if (!exact && !configs.has_value() && dual_smem_bytes(num_stages) > kSmemPerCU) {
        // Single-B configs may use excessive BK; select a valid fused default rather than rejecting a supported shape.
        block_k = std::min(block_k, 128);
        if (dual_smem_bytes(num_stages) > kSmemPerCU) block_k = 64;
        warp_k = block_k;
    }
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
    grid.x = exact ? exact->grid : std::min(blocks_per_cu * num_sms_new, output_tiles);
    DG_HOST_ASSERT(!exact || (grid.x <= output_tiles && (!exact->has(dual_b::OneTile) || grid.x == output_tiles)));

    // Output dtype is part of the JIT cache key (separate kernel name).
    std::string kernel_name = "bf16xfp32_gemm_cute_free";
    if (std::is_same_v<ElementOutput, float>) kernel_name += "_fp32";
    if (packed_dual_n) kernel_name += "_packed_dual_n";
    if (static_router) kernel_name += "_static_router_m" + std::to_string(m);
    if (decode_split_k > 1) kernel_name += "_decode_sk" + std::to_string(decode_split_k);
    // Exact long-prefill experiment: specialize only outer stage indices; each chain still advances through full K in order.
    const char* stages_env = std::getenv("DG_BF16XFP32_STATIC_STAGES");
    DG_HOST_ASSERT(stages_env == nullptr || std::string(stages_env) == "0" || std::string(stages_env) == "1");
    const bool static_stages = exact ? exact->has(dual_b::StaticStages) : stages_env != nullptr && *stages_env == '1';
    DG_HOST_ASSERT(!static_stages || (static_router && !packed_dual_n
        && (std::is_same_v<ElementOutput, float> || exact) && m == 16384 && n == 384 && k == 7168));
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
            && std::is_same_v<ElementOutput, float> && (m == 128 || m == 256) && n == 384 && k == 7168);
    DG_HOST_ASSERT(!early_g2s || !full_k_prefetch);
    if (early_g2s) kernel_name += "_early_g2s";
    const char* overlap_env = std::getenv("DG_BF16XFP32_MMA_S2R_OVERLAP");
    DG_HOST_ASSERT(overlap_env == nullptr || std::string(overlap_env) == "0" || std::string(overlap_env) == "1");
    const bool mma_s2r_overlap = exact ? exact->has(dual_b::MmaS2ROverlap) : overlap_env != nullptr && *overlap_env == '1';
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
    const bool overlap_prologue = exact ? exact->has(dual_b::OverlapPrologue) : (prologue_env != nullptr && *prologue_env == '1')
        || (prologue_env == nullptr && router_overlap);
    const bool router_valid_epilogue = !other_pipeline && !packed_dual_n && !configs.has_value()
        && std::is_same_v<ElementOutput, float> && m == 4096 && n == 384 && k == 7168;
    const bool valid_epilogue = exact ? exact->has(dual_b::ValidEpilogue) : (epilogue_env != nullptr && *epilogue_env == '1')
        || (epilogue_env == nullptr && (router_overlap || router_valid_epilogue));
    DG_HOST_ASSERT(!(overlap_prologue || valid_epilogue)
                   || (!full_k_prefetch && !early_g2s && !mma_s2r_overlap && !bounded_kloop));
    if (overlap_prologue) kernel_name += "_overlap_prologue";
    if (valid_epilogue) kernel_name += "_valid_epilogue";
    DG_HOST_ASSERT(!packed_dual_n || (!full_k_prefetch && !early_g2s && !mma_s2r_overlap
                   && !bounded_kloop && !overlap_prologue && !valid_epilogue));

    auto args = typename Runtime::Args{
        .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages,
                        "Normal", kernel_name, full_k_prefetch, early_g2s, mma_s2r_overlap, bounded_kloop, kloop_unroll, kloop_chunk,
                        overlap_prologue, valid_epilogue, packed_dual_n, static_router, static_stages, streamed_low, standard_s2, streamed_pairs, min_blocks, pairs_m_first, decode_split_k,
                        exact && exact->has(dual_b::PackedStore), cta_k,
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
