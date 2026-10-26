#pragma once

// ============================================================================
// PPU MegaMoE GEMM1 FP4 Tensor Cell path.
// ----------------------------------------------------------------------------
// The fused dispatch/GEMM1 kernel uses the MXFP4 mainloop
// `CollectiveMmaScaleFp4::operator()` and a BF16 activation/quantization epilogue.
//
// This header pulls in the full CUTLASS3 stack via fp4_gemm_cutlass3.cuh
// for the FP4 Tensor Cell mainloop.
// ============================================================================

#include <cstdint>
#include <hggc_runtime.h>
#include <hggc_bf16.h>

#include <cutlass/fast_math.h>
#include <deep_gemm/tix/cvt.cuh>
#include <deep_gemm/impls/fp4_gemm_cutlass3.cuh>

namespace deep_gemm::mega_moe::fp4 {

// Match hggc_bf16.hpp's packed tanh operation directly: the CUDA-compatible
// headers gate their equivalent on SM90, while this kernel targets PPU0015.
__device__ __forceinline__ __ppu_bfloat162 ppu_tanh_approx_bf16x2(
        const __ppu_bfloat162 value) {
    __ppu_bfloat162 result;
    asm volatile("ppu.tanh.approx.bf16x2 %0, %1;"
                 : "=r"(__BFLOAT162_TO_UI(result))
                 : "r"(__BFLOAT162_TO_CUI(value)));
    return result;
}

// ----------------------------------------------------------------------------
// SF uint8[rows][kSfBytes] (row-major) -> uint16[rows][kSfBytes/2] MN-major:
//   dst[j*rows + r] = src[r*kSfBytes + 2j] | (src[r*kSfBytes + 2j+1] << 8)
// Matches host preprocess_mxfp4_scales (view(uint16).t().contiguous().t()).
// Padding rows (r >= valid_rows) are written 0 (scale 0 -> contributes 0).
//
// Uses shared memory tiled transpose with large tiles (kTileM=rows, kTileK=32)
// to minimize __syncthreads() count (2 syncs per K-tile × 4 K-tiles = 8 total).
// Both HBM reads and writes are coalesced. The +1 K-dimension padding
// eliminates bank conflicts on the transposed smem read.
//
// smem_buf reuses the mainloop's dynamic shared memory — safe because the
// conversion completes (with __syncthreads) before the mainloop begins.
// ----------------------------------------------------------------------------
template <int kSfBytes, int kTileK = 32>
__device__ __forceinline__ void convert_sf_u8_to_u16_mn_major(
    uint16_t* __restrict__ dst, const uint8_t* __restrict__ src,
    int rows, int valid_rows, int thread_idx, int num_threads,
    char* smem_buf) {
    constexpr int kCols = kSfBytes / 2;
    // Full-height tile: [rows][kTileK + 1] u16. +1 padding for bank-conflict-free transpose.
    uint16_t (*stile)[kTileK + 1] =
        reinterpret_cast<uint16_t(*)[kTileK + 1]>(smem_buf);
    const int tile_elements = rows * kTileK;

    for (int tk = 0; tk < kCols; tk += kTileK) {
        // Phase 1: coalesced read from src (row-major) → smem tile
        // Consecutive threads read consecutive j within a row → coalesced HBM reads.
        for (int i = thread_idx; i < tile_elements; i += num_threads) {
            const int r = i / kTileK;
            const int j = i % kTileK;
            const int gj = tk + j;
            uint16_t v = 0;
            if (r < valid_rows && gj < kCols) {
                const uint16_t lo = src[r * kSfBytes + 2 * gj];
                const uint16_t hi = src[r * kSfBytes + 2 * gj + 1];
                v = static_cast<uint16_t>(lo | (hi << 8));
            }
            stile[r][j] = v;
        }
        __syncthreads();

        // Phase 2: coalesced write from smem (transposed) → dst (column-major)
        // Iterate j outer, r inner → consecutive threads write consecutive r → coalesced HBM writes.
        for (int i = thread_idx; i < tile_elements; i += num_threads) {
            const int j = i / rows;
            const int r = i % rows;
            const int gj = tk + j;
            if (gj < kCols) {
                dst[gj * rows + r] = stile[r][j];
            }
        }
        __syncthreads();
    }
}

// Minimal k-tile iterator: operator() uses `*it` / `++it` to walk integer k idx.
struct KTileIter {
    int idx;
    CUTLASS_DEVICE int operator*() const { return idx; }
    CUTLASS_DEVICE KTileIter& operator++() { ++idx; return *this; }
};

// ----------------------------------------------------------------------------
// All CUTLASS3 types for the gemm1 mxfp4 mainloop, parameterized by the fixed
// tile. Mirrors the type list extracted from Fp4Gemm::run and validated by the
// standalone test.
//   ElementA = float4_t (RowMajor)  /  ElementB = float4_t (ColumnMajor)
//   ElementSFA/SFB = uint16_t (SFA ColumnMajor / SFB RowMajor), Trans = true.
// ----------------------------------------------------------------------------
template <int BlockM, int BlockN, int BlockK, int WarpM = 32, int WarpN = 64,
          int Stages = 2, int SfbBlockN = BlockN>
struct Gemm1MmaConfig {
    static constexpr bool WideSFB = SfbBlockN != BlockN;
    using ElementA = cutlass::float4_t;
    using LayoutA  = cutlass::layout::RowMajor;
    using ElementB = cutlass::float4_t;
    using LayoutB  = cutlass::layout::ColumnMajor;

    using WarpOnM = cute::Int<BlockM / WarpM>;   // 2
    using WarpOnN = cute::Int<BlockN / WarpN>;   // 2
    static constexpr int ThreadNum = (BlockM / WarpM) * (BlockN / WarpN) * 32;  // 128
    static constexpr bool TransA = false;
    static constexpr bool TransB = false;

    using DispatchPolicy = cutlass::gemm::MainloopWithScalePPU0015Aiu<Stages>;
    using GemmOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        cutlass::arch::PPU0015, ElementA, TransA, cute::Int<BlockM>, cute::Int<BlockK>, false>;
    using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        cutlass::arch::PPU0015, ElementB, TransB, cute::Int<BlockN>, cute::Int<BlockK>, true>;

    using TransformA = cute::identity;
    using TransformB = cute::identity;

    // SFA: uint16, M-Major (ColumnMajor).
    using ElementSFA = uint16_t;
    using LayoutSFA  = cutlass::layout::ColumnMajor;
    static constexpr bool TransSFA = true;
    static constexpr int MinAiuContElemSize = 16;          // 32 / (16bit/8)
    static constexpr int ScaleGranularityK  = 32;
    static constexpr int ScaleMsPerTile = BlockM;
    static constexpr int ScaleKsPerTile = BlockK / ScaleGranularityK;  // 4
    static constexpr int SFATileM = TransSFA ? cute::max(ScaleMsPerTile, MinAiuContElemSize) : ScaleMsPerTile;
    static constexpr int SFATileK = TransSFA ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);
    static constexpr bool swap = true;
    static constexpr int  StageStride = 0;
    static constexpr bool swzl = false;
    using GemmOperandSFA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        cutlass::arch::PPU0015, ElementSFA, TransSFA, cute::Int<SFATileM>, cute::Int<SFATileK>, swap, StageStride, swzl>;

    // SFB: uint16, N-Major (RowMajor).
    using ElementSFB = uint16_t;
    using LayoutSFB  = cutlass::layout::RowMajor;
    static constexpr bool TransSFB = true;
    static constexpr int ScaleNsPerTile = SfbBlockN;
    static constexpr int SFBTileN = TransSFB ? cute::max(ScaleNsPerTile, MinAiuContElemSize) : ScaleNsPerTile;
    static constexpr int SFBTileK = TransSFB ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);
    using GemmOperandSFB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        cutlass::arch::PPU0015, ElementSFB, TransSFB, cute::Int<SFBTileN>, cute::Int<SFBTileK>, swap, StageStride, swzl>;

    using MmaInst  = PPU0015_16x16x64_F32F4F4F32_TN;
    using TiledMma = cute::TiledMMA<
        cute::MMA_Atom<MmaInst>,
        cute::Layout<cute::Shape<WarpOnM, WarpOnN, cute::_1>>>;

    using StrideA   = cutlass::detail::TagToStrideA_t<LayoutA>;
    using StrideB   = cutlass::detail::TagToStrideB_t<LayoutB>;
    using StrideSFA = cutlass::detail::TagToStrideA_t<LayoutSFA>;
    using StrideSFB = cutlass::detail::TagToStrideB_t<LayoutSFB>;

    using TileShapeMNK = cute::Shape<cute::Int<BlockM>, cute::Int<BlockN>, cute::Int<BlockK>>;

    using DefaultCollectiveMainloop = typename cutlass::gemm::collective::CollectiveMmaScaleFp4<
        DispatchPolicy, TileShapeMNK,
        ElementA, StrideA,
        ElementB, StrideB,
        TiledMma,
        typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
        typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB,
        ElementSFA, StrideSFA, typename GemmOperandSFA::GmemTiledCopy, typename GemmOperandSFA::SmemLayoutAtom,
        ElementSFB, StrideSFB, typename GemmOperandSFB::GmemTiledCopy, typename GemmOperandSFB::SmemLayoutAtom,
        FP4DynamicTileId::Disabled, -1>;
    using WideCollectiveMainloop = typename cutlass::gemm::collective::CollectiveMmaScaleFp4<
        DispatchPolicy, TileShapeMNK,
        ElementA, StrideA,
        ElementB, StrideB,
        TiledMma,
        typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
        typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB,
        ElementSFA, StrideSFA, typename GemmOperandSFA::GmemTiledCopy, typename GemmOperandSFA::SmemLayoutAtom,
        ElementSFB, StrideSFB, typename GemmOperandSFB::GmemTiledCopy, typename GemmOperandSFB::SmemLayoutAtom,
        FP4DynamicTileId::Disabled, SfbBlockN>;
    using CollectiveMainloop = cute::conditional_t<
        WideSFB, WideCollectiveMainloop, DefaultCollectiveMainloop>;
};

// ----------------------------------------------------------------------------
// Compile-time stage picker: select the largest stage count that fits the
// shared-memory cap, matching get_best_configs / get_smem_config_fp4.
// Gemm1MmaConfig::CollectiveMainloop::SharedStorage grows ~linearly with Stages,
// so we descend from the desired upper bound `DesiredStages` and stop at the
// first stage whose SharedStorage fits `kSmemCap`. This MUST be evaluated
// identically on the host (hggcFuncSetAttribute smem sizing) and the device
// (kernel template instantiation) -- evaluating it in one place (here) keeps
// the two in lock-step and avoids the host/device smem mismatch that makes a
// larger device stage read past the smaller host-sized buffer.
template <int BlockM, int BlockN, int BlockK, int WarpM, int WarpN,
          int S, int kSmemCap, int SfbBlockN = BlockN>
struct Gemm1StageFits {
    static constexpr bool value =
        (int)sizeof(typename Gemm1MmaConfig<BlockM, BlockN, BlockK, WarpM, WarpN, S, SfbBlockN>
                    ::CollectiveMainloop::SharedStorage) <= kSmemCap;
};

template <int BlockM, int BlockN, int BlockK, int WarpM, int WarpN,
          int DesiredStages, int kSmemCap, int SfbBlockN = BlockN>
struct Gemm1PickStage {
    static_assert(DesiredStages >= 2, "GEMM1 requires at least two pipeline stages");
    static constexpr int value =
        Gemm1StageFits<BlockM, BlockN, BlockK, WarpM, WarpN, DesiredStages,
                      kSmemCap, SfbBlockN>::value
        ? DesiredStages
        : Gemm1PickStage<BlockM, BlockN, BlockK, WarpM, WarpN, DesiredStages - 1,
                         kSmemCap, SfbBlockN>::value;
};

// The mainloop needs double buffering; reject tiles that cannot fit two stages.
template <int BlockM, int BlockN, int BlockK, int WarpM, int WarpN,
          int kSmemCap, int SfbBlockN>
struct Gemm1PickStage<BlockM, BlockN, BlockK, WarpM, WarpN, 2,
                      kSmemCap, SfbBlockN> {
    static_assert(Gemm1StageFits<BlockM, BlockN, BlockK, WarpM, WarpN, 2,
                                kSmemCap, SfbBlockN>::value,
                  "GEMM1 minimum two-stage pipeline exceeds shared-memory capacity");
    static constexpr int value = 2;
};

// ----------------------------------------------------------------------------
// Run one BlockM x N gemm1 tile on the FP4 Tensor Core, writing D (bf16, only
// rows m < valid_m). a_ptr / b_ptr are packed FP4 row bases; sfa_u16 is the
// device-converted M-major SFA (BlockM rows), sfb_u16 the host-converted
// N-major SFB. `smem_buf` must be >= sizeof(CollectiveMainloop::SharedStorage).
// M/N/K are the (compile-time-valued) BlockM / N / K_packed.
// ----------------------------------------------------------------------------
template <class Cfg, bool kQuantizeOutput = false,
          bool kSituTanhBf16x2 = false,
          bool kFusedSitu = false,
          bool kSiluThenClamp = false, bool kFastMath = true>
__device__ __forceinline__ void gemm1_mma_block(
    const uint8_t*  __restrict__ a_ptr,
    const uint16_t* __restrict__ sfa_u16,
    const uint8_t*  __restrict__ b_ptr,
    const uint16_t* __restrict__ sfb_u16,
    int sfb_n,
    int sfb_n_offset,
    __ppu_bfloat16*  __restrict__ d_ptr,
    int ld_d, int valid_m, int M, int N, int K,
    int thread_idx, char* smem_buf,
    float clamp_limit = 0.0f,             // >0: clamp; 0: no clamp (default)
    bool is_gate = false,                // gate: clamp max only; up: clamp both (non-fused mode)
    bool fused_swiglu = false,           // fused: pair mma_n (gate,up) and compute silu(clamp(gate))*clamp(up)
    bool fused_situ = false,             // fused: Kimi K3 SiTU-GLU on the same gate/up pair
    float situ_beta = 4.0f,
    float situ_linear_beta = 25.0f,
    uint8_t* __restrict__ fp4_output = nullptr,    // [M, N/4] packed FP4 (kQuantizeOutput)
    uint8_t* __restrict__ scale_output = nullptr,  // byte view of packed/transposed uint16 E8M0 scales
    int ld_fp4 = 0,                      // leading dimension of fp4_output (bytes)
    int ld_scale = 0) {                  // byte stride between adjacent uint16 scale-pair rows

    using namespace cute;
    using CollectiveMainloop = typename Cfg::CollectiveMainloop;
    using ElementA   = typename Cfg::ElementA;
    using ElementB   = typename Cfg::ElementB;
    using StrideA    = typename Cfg::StrideA;
    using StrideB    = typename Cfg::StrideB;
    using StrideSFA  = typename Cfg::StrideSFA;
    using StrideSFB  = typename Cfg::StrideSFB;
    using TiledMma   = typename Cfg::TiledMma;
    using TileShapeMNK = typename Cfg::TileShapeMNK;

    StrideA   dA   = cutlass::make_cute_packed_stride(StrideA{},   cute::make_shape(M, K, 1));
    StrideB   dB   = cutlass::make_cute_packed_stride(StrideB{},   cute::make_shape(N, K, 1));
    StrideSFA dSFA = cutlass::make_cute_packed_stride(StrideSFA{}, cute::make_shape(M, cute::ceil_div(K, 32), 1));
    StrideSFB dSFB;
    if constexpr (Cfg::WideSFB) {
        dSFB = cutlass::make_cute_packed_stride(
            StrideSFB{}, cute::make_shape(sfb_n, cute::ceil_div(K, 32), 1));
    } else {
        dSFB = cutlass::make_cute_packed_stride(
            StrideSFB{}, cute::make_shape(N, cute::ceil_div(K, 32), 1));
    }

    typename CollectiveMainloop::Arguments args{
        {M, N, K},
        reinterpret_cast<ElementA const*>(a_ptr), dA,
        reinterpret_cast<ElementB const*>(b_ptr), dB,
        sfa_u16, dSFA,
        sfb_u16, dSFB
    };

    auto problem_shape_MNKL = make_shape(M, N, K, Int<1>{});
    auto blk_coord_mnkl     = make_coord(0, 0, _, 0);

    // Keep the storage tile shape, but never fetch padded activation rows.
    CollectiveMainloop collective(args, make_shape(valid_m, N, K, Int<1>{}));
    auto load_inputs = collective.load_init(problem_shape_MNKL, blk_coord_mnkl, args);

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShapeMNK{}));
    clear(accum);

    auto residue_mnk = make_tuple(M, N, K - size<1>(gA) * size<2>(gA));

    KTileIter k_tile_iter{0};
    int k_tile_count = size<2>(gA);

    if constexpr (Cfg::WideSFB) {
        collective(accum, load_inputs, accum, k_tile_iter, k_tile_count,
                   residue_mnk, thread_idx, smem_buf, sfb_n_offset);
    } else {
        collective(accum, load_inputs, accum, k_tile_iter, k_tile_count,
                   residue_mnk, thread_idx, smem_buf);
    }

    // Fused activations use eight gate rows followed by eight up rows and
    // compact [M, N/2] output. The JIT embeds the selected mode and SiTU betas,
    // allowing the compiler to remove the inactive activation branch.
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor mD = make_tensor(make_gmem_ptr(d_ptr), make_shape(M, N), make_stride(ld_d, Int<1>{}));
    Tensor gD = local_tile(mD, take<0, 2>(TileShapeMNK{}), make_coord(0, 0));
    Tensor tCgD = thr_mma.partition_C(gD);

    Tensor cD  = make_identity_tensor(make_shape(M, N));
    Tensor gcD = local_tile(cD, take<0, 2>(TileShapeMNK{}), make_coord(0, 0));
    Tensor tCcD = thr_mma.partition_C(gcD);
    int lane_idx = thread_idx % 32;
    const bool fused_glu = fused_swiglu || fused_situ || kFusedSitu;

    if (fused_glu) {
        // Match the public fast_math contract while retaining the PPU sigmoid
        // instruction for the default path. This does not change MMA precision.
        const auto sigmoid = [](float value) {
            if constexpr (kFastMath)
                return __ppu_sgmdf(value);
            else
                return 1.0f / (1.0f + expf(-value));
        };
        constexpr int kTileM = int(cute::size<0>(TileShapeMNK{}));
        constexpr int kTileN = int(cute::size<1>(TileShapeMNK{}));
        constexpr int kCompactN = kTileN / 2;
        constexpr int kCompactStride = kCompactN;
        auto* compact_smem = reinterpret_cast<__ppu_bfloat16*>(smem_buf);
        if constexpr (kQuantizeOutput) {
            static_assert(kTileN % 128 == 0,
                          "fused MXFP4 tile N must contain complete scale pairs");
            static_assert(kTileM * kCompactStride * sizeof(__ppu_bfloat16) <=
                              sizeof(typename CollectiveMainloop::SharedStorage),
                          "compact epilogue must fit in mainloop shared memory");
        }

        // G8 stores gate/up in adjacent N fragments. Use the MMA
        // fragment coordinates for both BF16 output and BF16 quantization staging.
        CUTLASS_PRAGMA_UNROLL
        for (int warp_m = 0; warp_m < size<1>(tCgD); ++warp_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < size<1>(shape<0>(tCgD)); ++mma_m) {
                const int base_m = get<0>(tCcD(
                    make_coord(Int<0>{}, mma_m, Int<0>{}), warp_m, Int<0>{}));
                if (base_m >= valid_m)
                    continue;
                CUTLASS_PRAGMA_UNROLL
                for (int warp_n = 0; warp_n < size<2>(tCgD); ++warp_n) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mma_n = 0; mma_n < size<2>(shape<0>(tCgD)); mma_n += 2) {
                        auto gate_chunk = accum(make_coord(_, mma_m, mma_n), warp_m, warp_n);
                        auto up_chunk = accum(make_coord(_, mma_m, mma_n + 1), warp_m, warp_n);
                        const auto* g_acc = reinterpret_cast<const float*>(gate_chunk.data());
                        const auto* u_acc = reinterpret_cast<const float*>(up_chunk.data());
                        constexpr int kVec = int(cute::size(gate_chunk));
                        const int tile_n = get<1>(tCcD(
                            make_coord(Int<0>{}, mma_m, mma_n), warp_m, warp_n));
                        const int compact_n = (tile_n / 16) * 8 + tile_n % 8;
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kVec / 2; ++i) {
                            auto bg = __float22bfloat162_rn(make_float2(g_acc[2 * i], g_acc[2 * i + 1]));
                            auto bu = __float22bfloat162_rn(make_float2(u_acc[2 * i], u_acc[2 * i + 1]));
                            if (clamp_limit >= 0.0f) {
                                const auto limit = __float2bfloat162_rn(clamp_limit);
                                const auto neg_limit = __float2bfloat162_rn(-clamp_limit);
                                if constexpr (!kSiluThenClamp)
                                    bg = __hmin2(bg, limit);
                                bu = __hmin2(__hmax2(bu, neg_limit), limit);
                            }
                            const float2 gf = __bfloat1622float2(bg);
                            const float2 uf = __bfloat1622float2(bu);
                            float lhs0, lhs1, rhs0, rhs1;
                            if (kFusedSitu || fused_situ) {
                                const float inv_beta = 1.0f / situ_beta;
                                const float inv_linear_beta = 1.0f / situ_linear_beta;
                                if constexpr (!kFastMath) {
                                    lhs0 = situ_beta * tanhf(gf.x / situ_beta) * sigmoid(gf.x);
                                    lhs1 = situ_beta * tanhf(gf.y / situ_beta) * sigmoid(gf.y);
                                    rhs0 = situ_linear_beta * tanhf(uf.x / situ_linear_beta);
                                    rhs1 = situ_linear_beta * tanhf(uf.y / situ_linear_beta);
                                } else if constexpr (kSituTanhBf16x2) {
                                    const auto gate_arg = __hmul2(bg, __float2bfloat162_rn(inv_beta));
                                    const auto up_arg = __hmul2(bu, __float2bfloat162_rn(inv_linear_beta));
                                    const float2 gate_tanh = __bfloat1622float2(ppu_tanh_approx_bf16x2(gate_arg));
                                    const float2 up_tanh = __bfloat1622float2(ppu_tanh_approx_bf16x2(up_arg));
                                    lhs0 = situ_beta * gate_tanh.x * __ppu_sgmdf(gf.x);
                                    lhs1 = situ_beta * gate_tanh.y * __ppu_sgmdf(gf.y);
                                    rhs0 = situ_linear_beta * up_tanh.x;
                                    rhs1 = situ_linear_beta * up_tanh.y;
                                } else {
                                    lhs0 = situ_beta * cutlass::fast_tanh(gf.x * inv_beta) * __ppu_sgmdf(gf.x);
                                    lhs1 = situ_beta * cutlass::fast_tanh(gf.y * inv_beta) * __ppu_sgmdf(gf.y);
                                    rhs0 = situ_linear_beta * cutlass::fast_tanh(uf.x * inv_linear_beta);
                                    rhs1 = situ_linear_beta * cutlass::fast_tanh(uf.y * inv_linear_beta);
                                }
                            } else {
                                lhs0 = gf.x * sigmoid(gf.x);
                                lhs1 = gf.y * sigmoid(gf.y);
                                rhs0 = uf.x;
                                rhs1 = uf.y;
                            }
                            if constexpr (kSiluThenClamp) {
                                if (clamp_limit >= 0.0f) {
                                    lhs0 = fminf(lhs0, clamp_limit);
                                    lhs1 = fminf(lhs1, clamp_limit);
                                }
                            }
                            const int row = base_m + i * 8;
                            if (row >= valid_m)
                                continue;
                            if constexpr (kQuantizeOutput) {
                                auto* out = compact_smem + uint64_t(row) * kCompactStride + compact_n;
                                out[0] = __float2bfloat16(lhs0 * rhs0);
                                out[1] = __float2bfloat16(lhs1 * rhs1);
                            } else {
                                auto* out = d_ptr + uint64_t(row) * ld_d + compact_n;
                                out[0] = __float2bfloat16(lhs0 * rhs0);
                                out[1] = __float2bfloat16(lhs1 * rhs1);
                            }
                        }
                    }
                }
            }
        }
        // BF16 output is written directly. MXFP4 stages BF16 activations in
        // shared memory, then remaps each subwarp to one row for quantization.
        if constexpr (kQuantizeOutput) {
            __syncthreads();

            // Phase 2 mirrors silu_and_mul_post_quant_mxfp4's mapping:
            // 16 contiguous elements/thread and 2 threads/32-element group.
            // Multiple independent rows share a warp. This turns each group reduction
            // into a single xor-1 shuffle and enables an 8-byte vector store.
            constexpr int kElemPerThread = 16;
            constexpr int kThreadsPerGroup = 32 / kElemPerThread;
            constexpr int kThreadsPerRow = kCompactN / kElemPerThread;
            constexpr int kRowsPerWarp = 32 / kThreadsPerRow;
            static_assert(kThreadsPerGroup == 2,
                          "subwarp quantizer expects two threads per group");
            static_assert(kThreadsPerRow * kElemPerThread == kCompactN,
                          "subwarp quantizer must cover the compact tile");
            static_assert(32 % kThreadsPerRow == 0,
                          "an integer number of rows must fit in one warp");
            const int warp_idx = thread_idx / 32;
            constexpr int kNumWarps = Cfg::ThreadNum / 32;
            const int row_in_warp = lane_idx / kThreadsPerRow;
            const int thread_in_row = lane_idx % kThreadsPerRow;
            const int grp = thread_in_row / kThreadsPerGroup;
            const int lane_in_group = thread_in_row % kThreadsPerGroup;
            const int elem_base = grp * 32 + lane_in_group * kElemPerThread;
            for (int row_base = warp_idx * kRowsPerWarp;
                 row_base < valid_m;
                row_base += kNumWarps * kRowsPerWarp) {
                const int m_row = row_base + row_in_warp;
                const bool row_valid = m_row < valid_m;
                float values[kElemPerThread];
                float local_am = 0.0f;
                if (row_valid) {
                    const auto* src = reinterpret_cast<const __ppu_bfloat162*>(
                        compact_smem
                        + (uint64_t)m_row * kCompactStride + elem_base);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kElemPerThread / 2; ++i) {
                        const float2 value2 = __bfloat1622float2(src[i]);
                        values[2 * i] = value2.x;
                        values[2 * i + 1] = value2.y;
                        local_am = fmaxf(
                            local_am,
                            fmaxf(fabsf(value2.x), fabsf(value2.y)));
                    }
                }
                const float peer_am =
                    __shfl_xor_sync(0xffffffffu, local_am, 1);
                if (!row_valid)
                    continue;
                const float am = fmaxf(fmaxf(local_am, peer_am), 1e-10f);
                const uint32_t ds_int =
                    __float_as_uint(am * (1.0f / 6.0f));
                const uint32_t ds_exp =
                    (ds_int + 0x007FFFFFu) & 0x7F800000u;
                if (lane_in_group == 0) {
                    // scale_output is the byte view of physical
                    // [scale_pair, num_pool] uint16 storage. Adjacent 32-value
                    // groups write the low/high byte of one uint16 pair.
                    const int scale_pair = grp / 2;
                    const int scale_lo_hi = grp % 2;
                    scale_output[(uint64_t)scale_pair * ld_scale
                                 + (uint64_t)m_row * sizeof(uint16_t)
                                 + scale_lo_hi] =
                        static_cast<uint8_t>(ds_exp >> 23);
                }
                const float sf_inv = __uint_as_float(0x7F000000u - ds_exp);
                float q[kElemPerThread];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kElemPerThread; ++i)
                    q[i] = values[i] * sf_inv;

                const uint32_t packed0 = tix::cvt_rte_fp4x8_f32(q);
                const uint32_t packed1 = tix::cvt_rte_fp4x8_f32(q + 8);
                auto* out = reinterpret_cast<uint2*>(
                    fp4_output + (uint64_t)m_row * ld_fp4 + elem_base / 2);
                *out = make_uint2(packed0, packed1);
            }
            __syncthreads();
        }

    } else {
        // Unfused epilogue: write all N columns with optional gate/up clamping.
        CUTLASS_PRAGMA_UNROLL
        for (int warp_m = 0; warp_m < size<1>(tCgD); ++warp_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < size<1>(shape<0>(tCgD)); ++mma_m) {
                if (get<0>(tCcD(make_coord(Int<0>{}, mma_m, Int<0>{}), warp_m, Int<0>{})) < valid_m) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int warp_n = 0; warp_n < size<2>(tCgD); ++warp_n) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int mma_n = 0; mma_n < size<2>(shape<0>(tCgD)); ++mma_n) {
                            auto acc_chunk = accum(make_coord(_, mma_m, mma_n), warp_m, warp_n);
                            auto dst_chunk = tCgD(make_coord(_, mma_m, mma_n), warp_m, warp_n);
                            auto* acc = reinterpret_cast<const float*>(acc_chunk.data());
                            auto* dst = reinterpret_cast<uint32_t*>(dst_chunk.data().get());
                            constexpr int kVec = int(cute::size(acc_chunk));
                            CUTLASS_PRAGMA_UNROLL
                            for (int i = 0; i < kVec / 2; ++i) {
                                float v0 = acc[2*i], v1 = acc[2*i+1];
                                if (clamp_limit >= 0.0f) {
                                    if (is_gate) {
                                        v0 = fminf(v0, clamp_limit);
                                        v1 = fminf(v1, clamp_limit);
                                    } else {
                                        v0 = fminf(fmaxf(v0, -clamp_limit), clamp_limit);
                                        v1 = fminf(fmaxf(v1, -clamp_limit), clamp_limit);
                                    }
                                }
                                const uint32_t packed = tix::cvt_rte_bf16x2_f32(v1, v0);
                                dst[i] = packed;
                            }
                            if constexpr (kVec % 2 == 1) {
                                float v = acc[kVec-1];
                                if (clamp_limit >= 0.0f) {
                                    v = is_gate ? fminf(v, clamp_limit)
                                                : fminf(fmaxf(v, -clamp_limit), clamp_limit);
                                }
                                reinterpret_cast<__ppu_bfloat16*>(dst)[kVec-1] = __float2bfloat16(v);
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace deep_gemm::mega_moe::fp4
