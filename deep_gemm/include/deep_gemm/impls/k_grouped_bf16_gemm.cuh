#pragma once

#include <deep_gemm/common/fused_gemm_common.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/scheduler/k_grouped_scheduler.cuh>

#include <cute/ppu_tensor_mix.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/atom/mma_traits_ppu0015.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0015_aiu.hpp>
#include <cute/algorithm/ppu_copy.hpp>
#include <deep_gemm/common/utils_cutlass3.cuh>

namespace deep_gemm {
using namespace cute;

template <class _SrcT, class _DstT, GemmType kGemmType,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N,
          uint32_t BLOCK_SIZE, int kNumStages, bool kWithAccumulation>
__device__ __forceinline__ void bf16_k_grouped_gemm_impl(const KGroupedGemmArgs args) {
    static_assert(is_k_grouped_contiguous(kGemmType), "K-grouped GEMM type required");
    static_assert(kWithAccumulation, "K-grouped wgrad requires beta=1");
    static_assert(SHAPE_M > 0 and SHAPE_N > 0 and kNumGroups > 0, "M, N and groups must be positive");
    static_assert(kNumStages >= 2, "The multistage pipeline requires at least two stages");
    static_assert(BLOCK_K > 0 && 128 % BLOCK_K == 0,
                  "BLOCK_K must divide the 128-element K-group alignment");

    constexpr uint32_t STRIDE_CM = SHAPE_N;
    constexpr uint32_t STRIDE_CG = SHAPE_M * SHAPE_N;

    using SrcT = typename ToCutlassType<_SrcT>::type;
    using DstT = _DstT;
    using AccT = float;
    using TileShape = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>>;
#if __HGGC_ARCH__ == 100
    using ArchTag = cutlass::arch::PPU0010;
#else
    using ArchTag = cutlass::arch::PPU0015;
#endif
    using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, SrcT, SrcT, AccT>::type;
    using TiledMma = cute::TiledMMA<
        cute::MMA_Atom<MmaInst>,
        cute::Layout<Shape<Int<BLOCK_M / WARP_M>, Int<BLOCK_N / WARP_N>, _1>>>;
    static_assert(BLOCK_SIZE == 32 * (BLOCK_M / WARP_M) * (BLOCK_N / WARP_N));
    static_assert(BLOCK_SIZE >= 64, "Split A/B AIU copy requires at least two warps");
    using TileScheduler = KGroupedScheduler<kGemmType, SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups>;

    constexpr uint32_t kSmemASize = smem_a_size(sizeof(SrcT), kNumStages, BLOCK_M, BLOCK_K);
    extern __shared__ __align__(128) uint8_t smem_buffer[];
    SrcT* smem_a = reinterpret_cast<SrcT*>(smem_buffer);
    SrcT* smem_b = reinterpret_cast<SrcT*>(smem_buffer + kSmemASize);

    const uint32_t thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();

    // Both inputs are physically [K, M/N]. Use AIU's Trans=true descriptors
    // and hardware swizzle, paired with A/B's respective TSM-to-MMA layouts.
    using GemmOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        ArchTag, SrcT, true, cute::Int<BLOCK_M>, cute::Int<BLOCK_K>, false>;
    using TilerA = typename GemmOperandA::GmemTiledCopy::Tiler_MN;
    using SmemLayoutA = decltype(tile_to_shape(typename GemmOperandA::SmemLayoutAtom{},
        Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));
    typename GemmOperandA::GmemTiledCopy gmem_tiled_copy_A;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    Tensor sA = cute::make_tensor(cute::make_smem_ptr(smem_a), SmemLayoutA{});
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
    auto stride_A = cute::make_stride(_1{}, Int<SHAPE_M>{});

    using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
        ArchTag, SrcT, true, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>, true>;
    using TilerB = typename GemmOperandB::GmemTiledCopy::Tiler_MN;
    using SmemLayoutB = decltype(tile_to_shape(typename GemmOperandB::SmemLayoutAtom{},
        Shape<cute::Int<BLOCK_N>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));
    typename GemmOperandB::GmemTiledCopy gmem_tiled_copy_B;
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor sB = cute::make_tensor(cute::make_smem_ptr(smem_b), SmemLayoutB{});
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);
    auto stride_B = cute::make_stride(_1{}, Int<SHAPE_N>{});
    static_assert(cute::cosize_v<SmemLayoutA> * sizeof(SrcT) <= kSmemASize);
    static_assert(cute::cosize_v<SmemLayoutB> * sizeof(SrcT) <=
                  smem_b_size(sizeof(SrcT), kNumStages, BLOCK_N, BLOCK_K));

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    using SmemCopyAtomA = typename GemmOperandA::SmemCopyAtom;
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));

    using SmemCopyAtomB = typename GemmOperandB::SmemCopyAtom;
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));

    // Initialize fixed descriptor fields once; each group supplies its K extent.
    gmem_tiled_copy_A.desc_.template init<SrcT, true, get<0>(TilerA{}), get<1>(TilerA{})>(
        nullptr, SHAPE_M, 0, stride_A);
    gmem_tiled_copy_B.desc_.template init<SrcT, true, get<0>(TilerB{}), get<1>(TilerB{})>(
        nullptr, SHAPE_N, 0, stride_B);

    auto blk_mn_shape = Shape<Int<BLOCK_M>, Int<BLOCK_N>>{};
    Tensor cC = make_identity_tensor(blk_mn_shape);
    Tensor tCcC = thr_mma.partition_C(cC);
    CUTE_STATIC_ASSERT_V(size(tCcC) == size(accum));

    const auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
    const auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);
    CUTE_STATIC_ASSERT_V(K_BLOCK_MAX > Int<1>{});
    CUTE_STATIC_ASSERT_V(size<2>(tCrA_copy_view) == size<2>(tCrB_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) % size<2>(tCrA_copy_view) == Int<0>{});

    // Block scheduler, following fused_moe_gemm.cuh.
    uint32_t curr_group_idx, m_block_idx, n_block_idx;
    TileScheduler scheduler(args.grouped_layout, SHAPE_M);
    while (scheduler.fetch_next_work(curr_group_idx, m_block_idx, n_block_idx)) {
        const uint32_t group_k = scheduler.current_shape_k;
        gmem_tiled_copy_A.desc_.dim_h = group_k;
        gmem_tiled_copy_B.desc_.dim_h = group_k;

        const uint32_t c_row_base = m_block_idx * BLOCK_M;
        const uint32_t num_valid_rows = SHAPE_M % BLOCK_M == 0
            ? BLOCK_M : min(BLOCK_M, SHAPE_M - c_row_base);
        const uint32_t blk_n_offset = n_block_idx * BLOCK_N;
        DstT* c_group = reinterpret_cast<DstT*>(args.c_ptr) + static_cast<size_t>(curr_group_idx) * STRIDE_CG;

        // Bind A/B to the current group's base and current M/N tile.
        auto blk_coord_mnkl = make_coord(m_block_idx, n_block_idx, _, _1{});
        const uint32_t k_begin = scheduler.current_k_start;
        const SrcT* gmem_a = reinterpret_cast<const SrcT*>(args.a_ptr) + static_cast<size_t>(k_begin) * SHAPE_M;
        const SrcT* gmem_b = reinterpret_cast<const SrcT*>(args.b_ptr) + static_cast<size_t>(k_begin) * SHAPE_N;
        auto shape_A = cute::make_shape(Int<SHAPE_M>{}, group_k);
        Tensor mA_mk = cute::make_tensor(cute::make_gmem_ptr(gmem_a), shape_A, stride_A);
        Tensor gA = cute::local_tile(cute::make_mix_tensor_like(mA_mk), TileShape{},
                                     take<0,3>(blk_coord_mnkl), Step<_1,X,_1>{});
        Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
        auto shape_B = cute::make_shape(Int<SHAPE_N>{}, group_k);
        Tensor mB_nk = cute::make_tensor(cute::make_gmem_ptr(gmem_b), shape_B, stride_B);
        Tensor gB = cute::local_tile(cute::make_mix_tensor_like(mB_nk), TileShape{},
                                     take<0,3>(blk_coord_mnkl), Step<X,_1,_1>{});
        Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

        // Prologue: actlize multistage reserves one stage for the first refill.
        int k_tile_iter = 0;
        int k_tile_count = group_k / BLOCK_K;  // Each group K is 128-aligned.
        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < kNumStages - 1; ++k_pipe) {
            if (k_tile_count > 0) {
                copy_aiu<false>(
                    gmem_tiled_copy_A, tAgA(_,_,_,k_tile_iter), tAsA(_,_,_,k_pipe),
                    gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,k_pipe), warp_idx);
                ++k_tile_iter;
            }
            cp_async_fence();
            --k_tile_count;
        }
        clear(accum);

        // Mainloop: actlize ppu_mma_multistage.hpp, S-1 prologue and k_block=0 refill.
        int smem_pipe_read = 0;
        int smem_pipe_write = kNumStages - 1;
        Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
        Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

        cp_async_wait<kNumStages-2>();
        __syncthreads();
        copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));

        // Peel the first K tile as in actlize so the zero accumulator can
        // become overwrite MMA instructions instead of explicit register clears.
        auto compute_k_tile = [&]() {
            for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                if (k_block == K_BLOCK_MAX - 1) {
                    tCsA_p = tCsA(_,_,_,smem_pipe_read);
                    tCsB_p = tCsB(_,_,_,smem_pipe_read);
                    cp_async_wait<kNumStages-2>();
                    __syncthreads();
                }

                // Refill the free stage before consuming the current register fragment.
                if (k_block == 0) {
                    if (k_tile_count > 0) {
                        copy_aiu<false>(
                            gmem_tiled_copy_A, tAgA(_,_,_,k_tile_iter), tAsA(_,_,_,smem_pipe_write),
                            gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,smem_pipe_write), warp_idx);
                    }
                    cp_async_fence();
                    --k_tile_count;
                    ++k_tile_iter;
                    smem_pipe_write = smem_pipe_read;
                    ++smem_pipe_read;
                    smem_pipe_read = (smem_pipe_read == kNumStages) ? 0 : smem_pipe_read;
                }

                // actlize AIU_BEFORE_TSMLD=true for a 256x256 CTA tile.
                // Prefetch the next register fragment, then consume this one.
                const auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
                copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
                CUTLASS_PRAGMA_UNROLL
                for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; ++k_loop) {
                    const auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
                    cute::gemm(tiled_mma, accum, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), accum);
                }
            });
        };
        compute_k_tile();
        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > -(kNumStages - 1)) {
            compute_k_tile();
        }
        cp_async_wait<0>();
        __syncthreads();

        // Epilogue: reuse fused MoE's no-TSM store with FP32 beta=1.
        using TCcC = decltype(tCcC);
        if constexpr (cute::is_same_v<AccT, float> && cute::is_same_v<DstT, float> &&
                      SHAPE_M % BLOCK_M == 0 && SHAPE_N % BLOCK_N == 0 &&
                      SHAPE_N % 4 == 0 && size<0,2>(TCcC{}) == 2) {
            if ((reinterpret_cast<uintptr_t>(c_group) & 15) == 0) {
                epilogue_no_tsm_float4<SHAPE_N>(
                    accum, tCcC, c_group, c_row_base, blk_n_offset);
            } else {
                epilogue_no_tsm<AccT, DstT, SHAPE_N, BLOCK_N, STRIDE_CM, kWithAccumulation, false>(
                    accum, tCcC, c_group, c_row_base, num_valid_rows, blk_n_offset);
            }
        } else {
            epilogue_no_tsm<AccT, DstT, SHAPE_N, BLOCK_N, STRIDE_CM, kWithAccumulation, false>(
                accum, tCcC, c_group, c_row_base, num_valid_rows, blk_n_offset);
        }
    }
}

template <class _SrcT, class _DstT, GemmType kGemmType,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N,
          uint32_t BLOCK_SIZE, int kNumStages, bool kWithAccumulation>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
bf16_k_grouped_gemm_kernel(const KGroupedGemmArgs args) {
    bf16_k_grouped_gemm_impl<_SrcT, _DstT, kGemmType, SHAPE_M, SHAPE_N, kNumGroups,
                             BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_SIZE, kNumStages, kWithAccumulation>(args);
}

}  // namespace deep_gemm
