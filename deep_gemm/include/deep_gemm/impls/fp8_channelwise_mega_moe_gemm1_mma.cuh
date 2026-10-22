#pragma once

// Device-side channel-FP8 GEMM1 block used by the cooperative MegaMoE kernel.
// The CUTLASS A8W8 collective performs E4M3xE4M3 MMA and applies the per-row
// activation scale plus per-output-channel weight scale before returning FP32
// accumulators.  The custom epilogue consumes G=8 interleaved gate/up weights
// and writes the compact BF16 SwiGLU result.

#include <cstdint>
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>

namespace deep_gemm::mega_moe::fp8_channelwise {

struct KTileIter {
    int idx;
    CUTLASS_DEVICE int operator*() const { return idx; }
    CUTLASS_DEVICE KTileIter& operator++() {
        ++idx;
        return *this;
    }
};

template <
    int BlockM, int BlockN, int BlockK, int WarpM = 64, int WarpN = 64,
    int Stages = 4>
struct Gemm1MmaConfig {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementAcc = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ArchTag = cutlass::arch::PPU0015;

    static_assert(BlockM % WarpM == 0 && BlockN % WarpN == 0,
                  "CTA tile must be divisible by warp tile");
    static constexpr int WarpOnM = BlockM / WarpM;
    static constexpr int WarpOnN = BlockN / WarpN;
    static constexpr int ThreadNum = WarpOnM * WarpOnN * 32;

    using TileShapeMNK =
        cute::Shape<cute::Int<BlockM>, cute::Int<BlockN>, cute::Int<BlockK>>;
    using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<
        ArchTag, ElementA, ElementB, ElementAcc>::type;
    using TiledMma = cute::TiledMMA<
        cute::MMA_Atom<MmaInst>,
        cute::Layout<
            cute::Shape<cute::Int<WarpOnM>, cute::Int<WarpOnN>, cute::_1>>,
        cute::Tile<
            cute::Int<WarpOnM * 16>, cute::Int<WarpOnN * 16>, cute::_32>>;

    using DispatchPolicy = cutlass::gemm::MainloopPPUAiuA8W8<
        Stages, cutlass::gemm::KernelAiuMultistage>;
    static constexpr bool TransA = false;
    static constexpr bool TransB = false;
    using OperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        ArchTag, ElementA, TransA, cute::Int<BlockM>, cute::Int<BlockK>,
        false>;
    using OperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        ArchTag, ElementB, TransB, cute::Int<BlockN>, cute::Int<BlockK>,
        true>;

    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        ArchTag,
        DispatchPolicy,
        TileShapeMNK,
        ElementA,
        StrideA,
        ElementB,
        StrideB,
        TiledMma,
        typename OperandA::GmemTiledCopy,
        typename OperandA::SmemLayoutAtom,
        typename OperandA::SmemCopyAtom,
        cute::identity,
        typename OperandB::GmemTiledCopy,
        typename OperandB::SmemLayoutAtom,
        typename OperandB::SmemCopyAtom,
        cute::identity>;
};

template <class Cfg, bool kSwiGLUOAI = false>
__device__ __forceinline__ void gemm1_swiglu_mma_block(
        const typename Cfg::ElementA* __restrict__ a_ptr,
        const float* __restrict__ scale_a,
        const typename Cfg::ElementB* __restrict__ b_ptr,
        const float* __restrict__ scale_b,
        __ppu_bfloat16* __restrict__ d_ptr,
        int ld_d, int valid_m, int M, int N, int K,
        int thread_idx, char* smem_buf, float swiglu_limit,
        float swiglu_alpha, uint32_t* queue_counter = nullptr,
        uint32_t* next_ticket = nullptr, uint32_t* prefetched_work = nullptr) {
    using namespace cute;
    using CollectiveMainloop = typename Cfg::CollectiveMainloop;
    using StrideA = typename Cfg::StrideA;
    using StrideB = typename Cfg::StrideB;
    using TiledMma = typename Cfg::TiledMma;
    using TileShapeMNK = typename Cfg::TileShapeMNK;

    StrideA dA = cutlass::make_cute_packed_stride(
        StrideA{}, make_shape(M, K, 1));
    StrideB dB = cutlass::make_cute_packed_stride(
        StrideB{}, make_shape(N, K, 1));
    typename CollectiveMainloop::Arguments args{
        a_ptr, dA, b_ptr, dB, scale_a, scale_b};

    auto problem_shape_mnkl = make_shape(M, N, K, Int<1>{});
    auto block_coord_mnkl = make_coord(0, 0, _, 0);
    CollectiveMainloop collective(args, make_shape(valid_m, N, K, Int<1>{}));
    auto load_inputs = collective.load_init(
        problem_shape_mnkl, block_coord_mnkl, args,
        M, 0, 0, a_ptr, b_ptr);
    Tensor gA = get<0>(load_inputs);

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(
        tiled_mma, take<0, 2>(TileShapeMNK{}));
    clear(accum);
    auto residue_mnk = make_tuple(
        valid_m, N, K - size<1>(gA) * size<2>(gA));
    KTileIter k_tile_iter{0};
    collective(
        accum, load_inputs, accum, k_tile_iter, size<2>(gA),
        residue_mnk, thread_idx, cutlass::canonical_warp_idx_sync(), smem_buf,
        queue_counter, next_ticket);

    // Collective's final barrier publishes the ticket; the epilogue
    // barrier below retires every thread's read before the next MMA uses SMEM.
    if (prefetched_work != nullptr) *prefetched_work = *next_ticket;
    // G=8 preprocessing stores [gate8, up8, gate8, up8, ...]. The physical
    // C-fragment therefore pairs consecutive mma_n slices. Match the production
    // numerical order: FP32 accum -> BF16 -> optional BF16 clamp -> SwiGLU ->
    // BF16 compact output.
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor mD = make_tensor(
        make_gmem_ptr(d_ptr), make_shape(M, N),
        make_stride(ld_d, Int<1>{}));
    Tensor gD = local_tile(
        mD, take<0, 2>(TileShapeMNK{}), make_coord(0, 0));
    Tensor tCgD = thr_mma.partition_C(gD);
    Tensor cD = make_identity_tensor(make_shape(M, N));
    Tensor gcD = local_tile(
        cD, take<0, 2>(TileShapeMNK{}), make_coord(0, 0));
    Tensor tCcD = thr_mma.partition_C(gcD);

    CUTLASS_PRAGMA_UNROLL
    for (int warp_m = 0; warp_m < size<1>(tCgD); ++warp_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < size<1>(shape<0>(tCgD)); ++mma_m) {
            const int base_m = get<0>(tCcD(
                make_coord(Int<0>{}, mma_m, Int<0>{}),
                warp_m, Int<0>{}));
            if (base_m >= valid_m)
                continue;
            CUTLASS_PRAGMA_UNROLL
            for (int warp_n = 0; warp_n < size<2>(tCgD); ++warp_n) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0;
                     mma_n < size<2>(shape<0>(tCgD)); mma_n += 2) {
                    auto gate_chunk = accum(
                        make_coord(_, mma_m, mma_n), warp_m, warp_n);
                    auto up_chunk = accum(
                        make_coord(_, mma_m, mma_n + 1), warp_m, warp_n);
                    const auto* gate =
                        reinterpret_cast<const float*>(gate_chunk.data());
                    const auto* up =
                        reinterpret_cast<const float*>(up_chunk.data());
                    constexpr int kVec = int(cute::size(gate_chunk));
                    const int tile_n = get<1>(tCcD(
                        make_coord(Int<0>{}, mma_m, mma_n),
                        warp_m, warp_n));
                    const int compact_n =
                        (tile_n / 16) * 8 + tile_n % 8;
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kVec / 2; ++i) {
                        const int m_row = base_m + i * 8;
                        if (m_row >= valid_m)
                            continue;
                        __ppu_bfloat162 bg =
                            __float22bfloat162_rn(
                                make_float2(gate[2 * i], gate[2 * i + 1]));
                        __ppu_bfloat162 bu =
                            __float22bfloat162_rn(
                                make_float2(up[2 * i], up[2 * i + 1]));
                        if (swiglu_limit > 0.0f) {
                            const __ppu_bfloat16 limit =
                                __float2bfloat16(swiglu_limit);
                            const __ppu_bfloat162 hi = {limit, limit};
                            const __ppu_bfloat162 lo = {
                                __float2bfloat16(-swiglu_limit),
                                __float2bfloat16(-swiglu_limit)};
                            bg = __hmin2(bg, hi);
                            bu = __hmin2(__hmax2(bu, lo), hi);
                        }
                        const float2 gf = __bfloat1622float2(bg);
                        const float2 uf = __bfloat1622float2(bu);
                        // SwiGLU uses an FP32 gate/sigmoid product. M3 applies
                        // FP32 alpha and up bias after BF16 projection/clamp.
                        const float lhs0 = gf.x * __ppu_sgmdf(
                            kSwiGLUOAI ? swiglu_alpha * gf.x : gf.x);
                        const float lhs1 = gf.y * __ppu_sgmdf(
                            kSwiGLUOAI ? swiglu_alpha * gf.y : gf.y);
                        __ppu_bfloat16* out =
                            d_ptr + static_cast<uint64_t>(m_row) * ld_d
                            + compact_n;
                        out[0] = __float2bfloat16(
                            lhs0 * (kSwiGLUOAI ? uf.x + 1.0f : uf.x));
                        out[1] = __float2bfloat16(
                            lhs1 * (kSwiGLUOAI ? uf.y + 1.0f : uf.y));
                    }
                }
            }
        }
    }
    __syncthreads();
}

}  // namespace deep_gemm::mega_moe::fp8_channelwise
