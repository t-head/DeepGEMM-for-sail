#pragma once

#include "cutlass/cutlass.h"
#include "ppu/cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/collective/collective_mma.hpp"

#include "ppu/cutlass/gemm/collective/ppu_fp8_accumulation.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class DispatchPolicy_,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMmaBlockWise {
  //
  // Type Aliases
  //
  using DispatchPolicy = DispatchPolicy_;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ElementScale = float;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  constexpr static uint32_t CTA_M = shape<0>(TileShape{});
  constexpr static uint32_t CTA_N = shape<1>(TileShape{});
  constexpr static uint32_t CTA_K = shape<2>(TileShape{});
  // ScaleA
  using GmemTiledCopyScaleA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementScale>{},
                    Layout<Shape <Int<CTA_M / 4>, _1>>{},
                    Layout<Shape < _4,_1>>{}));

  // ScaleB
  using GmemTiledCopyScaleB = decltype(
    make_tiled_copy(Copy_Atom<cute::DefaultCopy, ElementScale>{},
                    Layout<Shape <_1,_1>>{},
                    Layout<Shape <_1,_1>>{}));

  using SmemLayoutAtomScale = Layout<Shape<_4, _1>>;
  using SmemLayoutScaleA = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(Int<CTA_M>{}, Int<1>{}, Int<DispatchPolicy::Stages>{})));   // assert CTA_SCALE_K = 1, gs >= cta_k
  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  struct SharedStorage {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleA>> smem_scale_a;
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScale const* ptr_scale_A;
    StrideScale dScaleA;
    ElementScale const* ptr_scale_B;
    StrideB dScaleB;
    uint32_t group_size = 128;
    uint32_t mma_promotion_interval = 4;
  };

  // Device side kernel params
  using Params = Arguments;
  // store params to const scale tensor in mainloop
  // avoid modify kernel file for fp4
  Params params_;

  // sm90 realization put TMA_A into params directly
  // put gmem_tiled_copy here and copy desc in kernel to simplify rtc usage
  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;

  GmemTiledCopyScaleA gmem_tiled_copy_scaleA;
  GmemTiledCopyScaleB gmem_tiled_copy_scaleB;
  uint32_t scale_k = 0;
  uint32_t reload_factor = 0;

  uint32_t mma_promotion_interval = 4;

  //
  // Methods
  //

  template <class ProblemShape_MNKL, class BlockCoord_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, BlockCoord_MNKL const& blk_coord_mnkl, Params const& params) {
    params_ = params;
    static constexpr bool TransA = is_static<decltype(get<1>(params.dA))>::value ? false : true;
    static constexpr bool TransB = is_static<decltype(get<1>(params.dB))>::value ? false : true;

    auto [M,N,K,L] = problem_shape_MNKL;
    auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;

    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;

    gmem_tiled_copy_A.desc_.template init<ElementA, TransA, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, M, K, params.dA);
    gmem_tiled_copy_B.desc_.template init<ElementB, TransB, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, N, K, params.dB);
    // mma_promotion_interval = params.mma_promotion_interval;

    // update desc.ptr to current batch, use y_offset as batch will cause smaller bit wide
    // gmem_tiled_copy_A.desc_.gmem_ptr += batch_idx * get<2>(params.dA) * sizeof(ElementA);
    // gmem_tiled_copy_B.desc_.gmem_ptr += batch_idx * get<2>(params.dB) * sizeof(ElementB);

    gmem_tiled_copy_scaleA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementScale>{},
                    Layout<Shape <Int<CTA_M / 4>, _1>>{},
                    Layout<Shape < _4,_1>>{});

    gmem_tiled_copy_scaleB = make_tiled_copy(Copy_Atom<cute::DefaultCopy, ElementScale>{},
                    Layout<Shape <_1,_1>>{},
                    Layout<Shape <_1,_1>>{});

    // init scale param
    // scale_k = cute::ceil_div(K, params.group_size);
    scale_k = cute::ceil_div(K, 128);
    reload_factor = cute::ceil_div(128, get<2>(TileShape{}));
    // load init A
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_A), make_shape(M,K,L), params.dA);   // (m,k,l)
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,l_coord));                                 // (m,k)
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)

    // load init B
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.ptr_B), make_shape(N,K,L), params.dB);   //(n,k,l)
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));                                 // (n,k)
    Tensor gB = local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

    // load init scale A/B
    Tensor mScaleA_mkl = make_tensor(make_gmem_ptr(params.ptr_scale_A), make_shape(M,scale_k,L), params.dScaleA);   // (m,scale_k,l)
    Tensor mScaleA_mk = mScaleA_mkl(_,_,l_coord);                                                                    // (m,scale_k)
    Tensor gScaleA = local_tile(mScaleA_mk, make_shape(Int<CTA_M>{}, Int<1>{}), make_coord(m_coord, _));    // (BLK_M, 1, scale_k) (128,1)(1)

    int scale_n = cute::ceil_div(N, _128{});
    Tensor mScaleB_nkl = make_tensor(make_gmem_ptr(params.ptr_scale_B), make_shape(scale_n, scale_k,L), params.dScaleB);                   // (scale_n,scale_k,l) (2,2)
    Tensor mScaleB_nk = mScaleB_nkl(_,_,l_coord);                                                                    // (scale_n,scale_k)
    constexpr int n_factor = cute::ceil_div(_128{}, Int<CTA_N>{});
    Tensor gScaleB = local_tile(mScaleB_nk, make_shape(_1{}, _1{}), make_coord(n_coord / n_factor, _));    // (BLK_N, 1, scale_k)

    return cute::make_tuple(gA, gB, gScaleA, gScaleB);
  }

  template <class ProblemShape>
  CUTLASS_DEVICE
  CollectiveMmaBlockWise(Params params, ProblemShape problem_shape_MNK) {
    static constexpr bool TransA = is_static<decltype(get<1>(params.dA))>::value ? false : true;
    static constexpr bool TransB = is_static<decltype(get<1>(params.dB))>::value ? false : true;

    auto M = get<0>(problem_shape_MNK);
    auto N = get<1>(problem_shape_MNK);
    auto K = get<2>(problem_shape_MNK);

    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;

    gmem_tiled_copy_A.desc_.init<ElementA, TransA, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, M, K, params.dA);
    gmem_tiled_copy_B.desc_.init<ElementB, TransB, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, N, K, params.dB);

    mma_promotion_interval = params.mma_promotion_interval;
  };

  CUTLASS_DEVICE
  CollectiveMmaBlockWise() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class TensorC,
    class TensorD,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      TensorC gScaleA,
      TensorD gScaleB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopSm80CpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopSm80CpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    Tensor sSA = make_tensor(make_smem_ptr(storage.smem_scale_a.data()), SmemLayoutScaleA{});
    Tensor fSB = make_fragment_like<ElementScale>(make_layout(make_shape(_1{}, _1{}, Int<DispatchPolicy::Stages>{})));

    auto gmem_thr_copy_scaleA = gmem_tiled_copy_scaleA.get_slice(thread_idx % (Int<CTA_M  / 4>{}));
    auto gmem_thr_copy_scaleB = gmem_tiled_copy_scaleB.get_slice(32); // use second warp
    Tensor tSgSA = gmem_thr_copy_scaleA.partition_S(gScaleA);
    Tensor tSsSA = gmem_thr_copy_scaleA.partition_D(sSA);
    Tensor tSgSB = gmem_thr_copy_scaleB.partition_S(gScaleB);
    Tensor tSfSB = gmem_thr_copy_scaleB.partition_D(fSB);

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      copy_aiu(
        gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe),
        gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe),
        warp_idx
      );
      uint32_t scale_load_k = *k_tile_iter / reload_factor;
      copy(gmem_tiled_copy_scaleA, tSgSA(_,_,_,scale_load_k), tSsSA(_,_,_,k_pipe));
      copy(gmem_tiled_copy_scaleB, tSgSB(_,_,_,scale_load_k), tSfSB(_,_,_,k_pipe));
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) { ++k_tile_iter; }
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    Tensor mma_acc = make_fragment_like<ElementAccumulator>(accum);
    clear(mma_acc);
    // scale A
    Tensor tCrSA = make_fragment_like<ElementScale>(thr_mma.partition_fragment_C(sSA(_,_,Int<0>{})));

    using SmemCopyAtomScale = Copy_Atom<cute::DefaultCopy, ElementScale>;
    auto smem_tiled_copy_ScaleA   = make_tiled_copy_C(SmemCopyAtomScale{}, tiled_mma);
    auto smem_thr_copy_ScaleA     = smem_tiled_copy_ScaleA.get_thread_slice(thread_idx);
    Tensor tCsSA                  = smem_thr_copy_ScaleA.partition_S(sSA);
    Tensor tCrSA_copy_view        = smem_thr_copy_ScaleA.retile_D(tCrSA);

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    Tensor tCsSA_p = tCsSA(_,_,_,smem_pipe_read);
    // Tensor tCsSB_p = tCsSB(_,_,_,smem_pipe_read);

    int fSB_pipe_read = 0;

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
    auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_ScaleA, tCsSA_p(_,_,Int<0>{}), tCrSA_copy_view(_,_,Int<0>{}));
    }

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -(DispatchPolicy::Stages-1)) {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        if (k_block == K_BLOCK_MAX - 1) {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);
          tCsSA_p = tCsSA(_,_,_,smem_pipe_read);
          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
        }

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0) {
          copy_aiu(
            gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write),
            gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write),
            warp_idx
          );
          uint32_t scale_load_k = *k_tile_iter / reload_factor;
          copy(gmem_tiled_copy_scaleA, tSgSA(_,_,_,scale_load_k), tSsSA(_,_,_,smem_pipe_write));
          copy(gmem_tiled_copy_scaleB, tSgSB(_,_,_,scale_load_k), tSfSB(_,_,_,smem_pipe_write));
          cp_async_fence();
          // Advance the tile
          --k_tile_count;
          if (k_tile_count > 0) { ++k_tile_iter; }

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

CUTLASS_PRAGMA_UNROLL
        for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
          auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
          // Transform before compute
          cute::transform(tCrA(_,_,atom_idx), TransformA{});
          cute::transform(tCrB(_,_,atom_idx), TransformB{});
          // gemm for one tiled_mma atom on K
          cute::gemm(tiled_mma, mma_acc, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), mma_acc);
        }
        // dequantize & load next group
        if (k_block_next == 0) {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < cute::size<2>(accum); ++n) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < cute::size<1>(accum); ++m) {
              // CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < cute::size<0>(accum); ++i) {
                accum(i, m, n) += mma_acc(i, m, n) * tCrSA(i,m,Int<0>{}) * fSB(0,0,fSB_pipe_read);
              }
            }
          }
          clear(mma_acc);
          fSB_pipe_read = (fSB_pipe_read + 1) % DispatchPolicy::Stages;
          copy(smem_tiled_copy_ScaleA, tCsSA_p(_,_,Int<0>{}), tCrSA_copy_view(_,_,Int<0>{}));
        }
      }); // for_each
    }

    // TODO: original cutlass3 miss this sync
    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
