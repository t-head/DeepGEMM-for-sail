#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include "utils.cuh"
#include "profiling_interface.hpp"

#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0015_aiu.hpp"
#include "cute/algorithm/copy.hpp"

#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/gemm/collective/ppu_promotion_with_scale_accumulation.hpp"

#include "fused_scheduler.cuh"
#include "fused_gemm_util.cuh"
#include "utils_cutlass3.h"

using namespace cute;

namespace deep_gemm {

template <class _SrcT, GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N,
          uint32_t BLOCK_SIZE, int kNumStages, int N_EXPAND>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void
fp8_blockwise_quant_gemm_fused_moe_kernel(const QuantGemmArgs args) {
    static constexpr uint32_t GROUP_M = 1;
    static constexpr uint32_t GROUP_N = 128;
    static constexpr uint32_t GROUP_K = 128;
    static constexpr uint32_t ScaleMsPerTile = cute::ceil_div(Int<BLOCK_M>{}, Int<1>{});
    static constexpr uint32_t ScaleNsPerTile = cute::ceil_div(Int<BLOCK_N>{}, Int<GROUP_N>{});
    static constexpr uint32_t ScaleKsPerTile = cute::ceil_div(Int<BLOCK_K>{}, Int<GROUP_K>{});

    constexpr uint32_t STRIDE_AM = SHAPE_K;
    constexpr uint32_t STRIDE_BE = SHAPE_N * SHAPE_K;
    constexpr uint32_t STRIDE_CM = SHAPE_N;
    constexpr uint32_t STRIDE_BSE = SHAPE_N / GROUP_N * SHAPE_K / GROUP_K;

    using SrcT = typename ToCutlassType<_SrcT>::type;
    using DstT = cutlass::bfloat16_t;
    using AccT = float;
    using ElementScale = float;
    using TileShape = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>>;
    using WarpShape = cute::Shape<cute::Int<WARP_M>, cute::Int<WARP_N>, cute::Int<BLOCK_K>>;
#if __HGGC_ARCH__ == 100
    using ArchTag = cutlass::arch::PPU0010;
#else
    using ArchTag = cutlass::arch::PPU0015;
#endif

    using MmaInst = typename cutlass::gemm::config::GetMmaInst<ArchTag, SrcT, SrcT, AccT>::type;
    using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape< Int<BLOCK_M / WARP_M>, Int<BLOCK_N / WARP_N>, _1>>>;

    using TileScheduler = FusedGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, kNumGroups>;

    // Shared memory
    using TsmCfg = BlkwiseQuantGemmSmemConfig<SrcT, kNumStages, BLOCK_M, BLOCK_N, BLOCK_K>;
    extern __shared__ __align__(128) uint8_t smem_buffer[];
    SrcT* smem_a = reinterpret_cast<SrcT*>(smem_buffer);
    SrcT* smem_b = reinterpret_cast<SrcT*>(smem_buffer + TsmCfg::kSmemASize);
    ElementScale* smem_scale_a = reinterpret_cast<ElementScale*>(smem_buffer + (TsmCfg::kSmemASize + TsmCfg::kSmemBSize));
    ElementScale* smem_scale_b = reinterpret_cast<ElementScale*>(smem_buffer + (TsmCfg::kTotalSize - TsmCfg::kSmemScaleBSize));

    uint32_t thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    uint32_t shape_m = args.shape_m;

    // load A from hbm to tsm: use async copy.
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<SrcT>::value;
    using ACopyInst = cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t>;
    using GemmOperandA = cutlass::gemm::config::Gemm_Hybrid_Operand<
            ArchTag, SrcT, false, AlignmentA, cute::Int<BLOCK_K>, BLOCK_SIZE,
            ACopyInst, cute::Int<BLOCK_M>>;

    using TilerA = typename GemmOperandA::GmemTiledCopy::Tiler_MN;
    using SmemLayoutA = decltype(tile_to_shape(typename GemmOperandA::SmemLayoutAtom{},
            Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));

    typename GemmOperandA::GmemTiledCopy gmem_tiled_copy_A;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    Tensor sA = cute::make_tensor(cute::make_smem_ptr(smem_a), SmemLayoutA{});
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);

    // load B from hbm to tsm: use aiu load.
    using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<
            ArchTag, SrcT, false, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>, true>;
    using SmemLayoutB = decltype(tile_to_shape(typename GemmOperandB::SmemLayoutAtom{},
            Shape<cute::Int<BLOCK_N>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));

    // init aiu desc
    typename GemmOperandB::GmemTiledCopy gmem_tiled_copy_B;
    using TilerB = typename GemmOperandB::GmemTiledCopy::Tiler_MN;
    auto shape_B = cute::make_shape(Int<SHAPE_N>{}, Int<SHAPE_K>{});
    auto stride_B = cute::make_shape((int)SHAPE_K, _1{});
    gmem_tiled_copy_B.desc_.template init<SrcT, false, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, SHAPE_N, SHAPE_K, stride_B);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor sB = cute::make_tensor(cute::make_smem_ptr(smem_b), SmemLayoutB{});
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);

    // --------------------------------------- scale operand -------------------------------------//
    using ScaleCopyInst = cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<ElementScale>;
    auto copy_scaleA_to_tsm = [&](int pipe_write, uint32_t k_idx, const int* blk_token_base) {
      static constexpr uint32_t NumThreads_Needed = ScaleMsPerTile * ScaleKsPerTile;
      CUTLASS_PRAGMA_UNROLL
      for(uint32_t tid = thread_idx; tid < NumThreads_Needed; tid += BLOCK_SIZE) {
        uint32_t tid_k = tid / ScaleMsPerTile;
        uint32_t tid_m = tid % ScaleMsPerTile;
        uint32_t token_offset = __ldg(blk_token_base + tid_m);
        bool token_mask = token_offset < shape_m;
        // scale a is m-major, stride asm = 1, stride_ask = shape_m;
        ElementScale* src_ptr = (ElementScale*)args.scale_a_ptr + (tid_k + k_idx) * shape_m + token_offset;
        ElementScale* dst_ptr = (ElementScale*)smem_scale_a + (tid_k + pipe_write) * BLOCK_M + tid_m;
        ScaleCopyInst::copy(*src_ptr, *dst_ptr, token_mask);
      }
    };

    // load scaleB from hbm to tsm: use aiu load.
    constexpr uint32_t MaxAiuContElemSize = 128 / (sizeof_bits<ElementScale>::value / 8);     // 32
    constexpr uint32_t MinAiuContElemSize = 32 / (sizeof_bits<ElementScale>::value / 8);      // 8
    constexpr bool TransSFB = false;  // k-major is false
    static constexpr uint32_t SFBTileN = TransSFB ? cute::max(ScaleNsPerTile, MinAiuContElemSize) : ScaleNsPerTile;
    static constexpr uint32_t SFBTileK = TransSFB ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);
    using GemmOperandSFB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementScale, TransSFB, Int<SFBTileN>, Int<SFBTileK>, true, 1, false>;
    using RealSmemLayoutAtomSFB = Layout<Shape<Int<SFBTileN>, Int<SFBTileK>>, Stride<Int<SFBTileK>, _1>>;
    using RealSmemLayoutSFB = decltype(tile_to_shape(RealSmemLayoutAtomSFB{},
            Shape<cute::Int<SFBTileN>, cute::Int<SFBTileK>, cute::Int<kNumStages>>{}));

    // init aiu desc
    typename GemmOperandSFB::GmemTiledCopy gmem_tiled_copy_SFB;
    using TilerSFB = typename GemmOperandSFB::GmemTiledCopy::Tiler_MN;
    constexpr uint32_t SCALE_N = cute::ceil_div(Int<SHAPE_N>{}, Int<GROUP_N>{});
    constexpr uint32_t SCALE_K = cute::ceil_div(Int<SHAPE_K>{}, Int<GROUP_K>{});
    auto shape_SFB = cute::make_shape(Int<SCALE_N>{}, Int<SCALE_K>{});
    auto stride_SFB = cute::make_shape((int)SCALE_K, _1{});
    gmem_tiled_copy_SFB.desc_.template init<ElementScale, TransSFB, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(nullptr, SCALE_N, SCALE_K, stride_SFB);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_slice(thread_idx);
    Tensor sSFB = cute::make_tensor(cute::make_smem_ptr(smem_scale_b), RealSmemLayoutSFB{});
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{})); // (MMA,MMA_M,MMA_N)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    Tensor cC = make_identity_tensor(Shape<Int<BLOCK_M>, Int<BLOCK_N>>{});
    Tensor tCcC = thr_mma.partition_C(cC);
    CUTE_STATIC_ASSERT_V(size(tCcC) == size(accum),
            "Accumulator count must have the same destination element count.");

    //
    // Copy Atom retiling
    //
    using SmemCopyAtomA = typename GemmOperandA::SmemCopyAtom;
    using SmemCopyAtomB = typename GemmOperandB::SmemCopyAtom;

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // ---------------- scale ------------------- //
    using ScaleConfig = ::cutlass::detail::PPUBlockwiseScaleConfig<GROUP_M, GROUP_N, GROUP_K>;
    using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(TileShape{}));
    using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(TileShape{}));

    using SmemLayoutSFA = decltype(make_layout(
      append(shape(SmemLayoutAtomSFA{}), Int<kNumStages>{}),
      append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
    ));

    using SmemLayoutSFB = decltype(make_layout(
      append(shape(SmemLayoutAtomSFB{}), Int<kNumStages>{}),
      append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
    ));

    auto layout_sSFA_copy = make_layout(
      make_shape(
        shape<0>(SmemLayoutSFA{}),
        get<1>(TileShape{}),
        make_shape(
          shape<1>(SmemLayoutSFA{}),
          shape<2>(SmemLayoutSFA{}))
      ),
      make_stride(
        stride<0>(SmemLayoutSFA{}),
        _0{},
        make_stride(
          stride<1>(SmemLayoutSFA{}),
          stride<2>(SmemLayoutSFA{}))
      )
    );

    auto layout_sSFB_copy = make_layout(
      make_shape(
        get<0>(TileShape{}),
        get<0>(shape(SmemLayoutSFB{})),       // n-broadcast
        make_shape(
          get<1>(shape(SmemLayoutSFB{})),     // k-broadcast
          get<2>(shape(SmemLayoutSFB{})))
      ),
      make_stride(
        _0{},
        make_stride(_0{}, get<0>(stride(RealSmemLayoutSFB{}))),
        make_stride(
          get<1>(stride(SmemLayoutSFB{})),
          get<2>(stride(RealSmemLayoutSFB{})))
      )
    );

    Tensor sSFA_copy = make_tensor(cute::make_smem_ptr(smem_scale_a), layout_sSFA_copy);
    Tensor sSFB_copy = make_tensor(cute::make_smem_ptr(smem_scale_b), layout_sSFB_copy);

    Tensor tCsSFA = tiled_mma.get_slice(thread_idx).partition_C(sSFA_copy);
    Tensor tCsSFB = tiled_mma.get_slice(thread_idx).partition_C(sSFB_copy);
    Tensor tCrSFA = make_fragment_like<ElementScale>(tCsSFA(_, _, _, _0{}));
    Tensor tCrSFB = make_fragment_like<ElementScale>(tCsSFB(_, _, _, _0{}));

    static constexpr int ScalePromotionInterval = GROUP_K / size<2>(typename TiledMma::AtomShape_MNK{});
    using FrgTensorC = decltype(accum);
    using EngineAccum = typename FrgTensorC::engine_type;
    using LayoutAccum = typename FrgTensorC::layout_type;
    cutlass::gemm::collective::MixAccumulation<EngineAccum, LayoutAccum, AccT> accumulation(
      accum, ScalePromotionInterval, size<2>(tCrA));

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    TileScheduler deep_scheduler(args.aligned_num_m_blocks, args.expert_ids_and_cumsum);

    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto n_coord = n_block_idx * N_EXPAND;
      auto blk_coord_mnkl = make_coord(m_block_idx, n_coord, _, _1{});
      const int* blk_token_base = args.sorted_token_ids + deep_scheduler.cumsum_m_block_idx * BLOCK_M;

      // gmem_b in block
      SrcT* gmem_b = (SrcT*)args.b_ptr + deep_scheduler.curr_group_idx * STRIDE_BE;
      Tensor mB_nk = cute::make_tensor(cute::make_gmem_ptr(gmem_b), shape_B, stride_B);
      Tensor gB = cute::local_tile(cute::make_mix_tensor_like(mB_nk), TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

      // gmem_scale b in block
      ElementScale* gmem_scale_b = (ElementScale*)args.scale_b_ptr + deep_scheduler.curr_group_idx * STRIDE_BSE;
      static constexpr int n_factor = cute::ceil_div(Int<GROUP_N>{}, Int<BLOCK_N>{});
      Tensor mSFB_nk = cute::make_tensor(cute::make_gmem_ptr(gmem_scale_b), shape_SFB, stride_SFB);
      Tensor gSFB = cute::local_tile(cute::make_mix_tensor_like(mSFB_nk), make_tile(Int<ScaleNsPerTile>{}, Int<ScaleKsPerTile>{}),
                                     make_coord(n_coord / n_factor,_));
      Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);

      static constexpr int K_TILE_COUNT = ceil_div(SHAPE_K, BLOCK_K);

      int k_tile_iter  = 0;
      int k_tile_count = K_TILE_COUNT;

      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < kNumStages; ++k_pipe) {
        if (k_tile_count > 0) {
          copy_A_to_tsm<SrcT, ACopyInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM>(
              tAsA(_,_,_,k_pipe), args.a_ptr,
              blk_token_base, BLOCK_K * k_tile_iter, shape_m, thread_idx);
          copy_scaleA_to_tsm(k_pipe, k_tile_iter, blk_token_base);
          if (warp_idx == 0) {
            copy(gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,k_pipe));
          } else if (warp_idx == 1) {
            copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_tile_iter), tSFBsSFB(_,_,_,k_pipe));
          }
          ++k_tile_iter;
        }
        cp_async_fence();
        --k_tile_count;
      }
      int k_tile_count_reset = k_tile_count;

      //
      // PIPELINED MAIN LOOP
      //

      // Current pipe index in smem to read from
      int smem_pipe_read  = 0;
      // Current pipe index in smem to write to
      int smem_pipe_write = 0;

      Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
      Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

      // Size of the register pipeline
      auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
      auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);

      const auto base_tSFBgSFB_data = tSFBgSFB.data();  // valu copy iterator
      for (int n_iter = 0; n_iter < N_EXPAND; ++n_iter) {
        clear(accum);
        int k_iter = 0;
        uint32_t blk_n_offset = (n_coord + n_iter) * BLOCK_N;
        k_tile_count = k_tile_count_reset;
        // PREFETCH register pipeline
        if constexpr(K_BLOCK_MAX > 1) {
          // Wait until our first prefetched tile is loaded in
          cp_async_wait<kNumStages-1>();
          __syncthreads();

          // Prefetch the first rmem from the first k-tile
          copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
          copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
        }
        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > -(kNumStages)) {
          copy(tCsSFA(_,_,_,make_coord(_0{}, smem_pipe_read)), tCrSFA);
          copy(tCsSFB(_,_,_,make_coord(_0{}, smem_pipe_read)), tCrSFB);
          clear(accumulation());
          if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
            tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
          }
          if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
            ElementScale scale_b = tCrSFB(_0{});
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
              filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
            }
          }
          if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
            ElementScale scale_a = tCrSFA(_0{});
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
              filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
            }
          }
          // Pipeline the outer products with a static for loop.
          //
          // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
          for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
            // Load A, B shmem->regs for k_block+1
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == K_BLOCK_MAX - 1) {
              // Commit the smem for smem_pipe_read
              if constexpr (kNumStages == 1) {
                cp_async_wait<kNumStages-1>();
              } else {
                cp_async_wait<kNumStages-2>();
              }
              __syncthreads();

              if (k_tile_count > 0 || n_iter < N_EXPAND - 1) {
                if (k_tile_iter >= K_TILE_COUNT) {
                  if (n_iter < N_EXPAND - 1) {
                    // load for next n_iter, avoid invalid page
                    tBgB.data() = tBgB.data() + SHAPE_K * BLOCK_N;
                    tSFBgSFB.data() = base_tSFBgSFB_data + (SHAPE_K / GROUP_K) * ((n_iter + 1) * BLOCK_N / GROUP_N) ;
                  }
                  k_tile_iter = 0;
                }
                if constexpr (SHAPE_K > BLOCK_K * kNumStages) {
                  copy_A_to_tsm<SrcT, ACopyInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM>(
                      tAsA(_,_,_,smem_pipe_write), args.a_ptr,
                      blk_token_base, BLOCK_K * k_tile_iter, shape_m, thread_idx);
                  copy_scaleA_to_tsm(smem_pipe_write, k_tile_iter, blk_token_base);
                }
                if (warp_idx == 0) {
                  copy(gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,smem_pipe_write));
                } else if (warp_idx == 1) {
                  copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_tile_iter), tSFBsSFB(_,_,_,smem_pipe_write));
                }
              }
              cp_async_fence();
              --k_tile_count;
              ++k_tile_iter;

              // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
              ++smem_pipe_read;
              smem_pipe_read = (smem_pipe_read == kNumStages) ? 0 : smem_pipe_read;
              smem_pipe_write = smem_pipe_read;

              // Slice the smem_pipe_read smem
              tCsA_p = tCsA(_,_,_,smem_pipe_read);
              tCsB_p = tCsB(_,_,_,smem_pipe_read);
            }
            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
            copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
            copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

            CUTLASS_PRAGMA_UNROLL
            for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
              auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
              // gemm for one tiled_mma atom on K
              cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), accumulation());
            }
          }); // for_each
          // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
          if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
            ElementScale scale_ab = tCrSFA(_0{});
            accumulation.scale(scale_ab);
          }
          if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
            accumulation.scale(tCrSFA);
          }
          if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
            accumulation.scale(tCrSFB);
          }
          if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
            accumulation.scale(tCrSFA, tCrSFB);
          }
        }
        // acc write back
        epilogue_no_tsm<AccT, DstT, SHAPE_N, BLOCK_N, STRIDE_CM>(accum, tCcC, args.c_ptr,
            deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset);
      }

    }
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N, int32_t kNumStages,
          GemmType kGemmType, bool kEnableSboOverlap = false,
          KernelType kKernelType = KernelType::Default>
class FusedMoeGemmWithBlkwiseQuant {
    using SrcT = __hg_fp8_e4m3;
    using DstT = __ppu_bfloat16;

public:
    FusedMoeGemmWithBlkwiseQuant() = default;

    static void run(DstT* gmem_d, SrcT* gmem_a, SrcT* gmem_b, float* gmem_sa, float* gmem_sb,
                    int* m_rows, int* expert_ids_and_cumsum, int* sorted_token_ids,
                    int* aligned_num_m_blocks, uint32_t shape_m, uint32_t topk,
                    hggcStream_t stream, int num_sms) {

        QuantGemmArgs args;

        args.a_ptr = (void *)gmem_a;
        args.b_ptr = (void *)gmem_b;
        args.c_ptr = (void *)gmem_d;
        args.scale_a_ptr = (void *)gmem_sa;
        args.scale_b_ptr = (void *)gmem_sb;

        args.expert_ids_and_cumsum = expert_ids_and_cumsum;
        args.sorted_token_ids = sorted_token_ids;
        args.aligned_num_m_blocks = aligned_num_m_blocks;
        args.shape_m = shape_m;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()){
            // check src type
            std::string data_type = "fp8";
            dg_prof_params.set_params(
                GemmType::GroupedFused, false, data_type, kNumGroups, shape_m, SHAPE_N, SHAPE_K, 1,
                m_rows, stream
            );
        }
        static constexpr int Stages = SHAPE_K < BLOCK_K * kNumStages ? SHAPE_K / BLOCK_K : kNumStages;
        // dispatch and launch kernel
        constexpr int BlockSize = BLOCK_M / WARP_M * BLOCK_N / WARP_N * 32;
        auto launch_kernel = [&](auto n_expand_) {
          using N_EXPAND_T = decltype(n_expand_);
          static constexpr bool kUseNStageKernel = n_expand_ > 1 && (SHAPE_N % (BLOCK_N * n_expand_) == 0);
          static constexpr int N_EXPAND = kUseNStageKernel ? n_expand_ : 1;

          auto device_func = fp8_blockwise_quant_gemm_fused_moe_kernel<
                SrcT, kGemmType, SHAPE_N, SHAPE_K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BlockSize, Stages, N_EXPAND>;
          constexpr int smem_size = BlkwiseQuantGemmSmemConfig<SrcT, Stages, BLOCK_M, BLOCK_N, BLOCK_K>::kTotalSize;
          CHECK_HGGC(hggcFuncSetAttribute(device_func, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          int max_blocks_per_cu = -1;
          CHECK_HGGC(hggcOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu, device_func, BlockSize, smem_size));

          int sm_count = num_sms * max_blocks_per_cu;
          dim3 grid(sm_count, 1, 1);

          const char* pEnv_params = std::getenv("show_log");
          if (pEnv_params && std::atoi(pEnv_params) == 1) {
              hggcFuncAttributes attr;
              hggcFuncGetAttributes(&attr, device_func);

              printf("[FusedMoeGemmWithBlkwiseQuant-FP8:]\n");
              printf("group:%d, problem:[%d, %d, %d], gemm_type:%s, kernel_type:%s\n",
                  kNumGroups, shape_m, SHAPE_N, SHAPE_K, GemmTypeS[static_cast<int>(kGemmType)], KernelTypeS[static_cast<int>(kKernelType)]);

              printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], Stages:%d, N_EXPAND:%d\n",
                  BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K, Stages, N_EXPAND);

              printf("grid:%d, vreg:%d, smem_size: %d, tb_per_cu:%d, stack:%d\n",
                  sm_count, int(attr.numRegs), smem_size, max_blocks_per_cu, int(attr.localSizeBytes));
          }
          ProfilingInterface::Instance().instrument(true, dg_prof_params);
          device_func<<<grid, BlockSize, smem_size, stream>>>(args);
          ProfilingInterface::Instance().instrument(false, dg_prof_params);
        };
        if constexpr(SHAPE_K <= 512 && (BLOCK_K == 128) && (SHAPE_K % BLOCK_K == 0)) {
          int expected_m = ceil_div(shape_m * topk, kNumGroups);
          int wave = ceil_div(ceil_div(expected_m, BLOCK_M) * ceil_div(SHAPE_N, BLOCK_N), num_sms);
          switch (wave) {
          case 1:
            launch_kernel(_1{});
            break;
          case 2:
            launch_kernel(_2{});
            break;
          default:
            launch_kernel(_4{});
            break;
          }
        } else {
          launch_kernel(_1{});
        }
    }
};


} // namespace deep_gemm


#pragma clang diagnostic pop
