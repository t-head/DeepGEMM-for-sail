#pragma once

#include <deep_gemm/common/utils_cutlass3.cuh>
#include <deep_gemm/mma/fp4_mma.cuh>

namespace cutlass::gemm {

// OverlapPrologue dispatch policy for FP4. Stages and ClusterShape stay defined by the
// actlize policy; the only addition is the schedule tag that selects the kernel.
template <int Stages_, typename Schedule_ = KernelAiuMultistageOverlapPrologue>
struct MainloopWithScalePPU0015AiuOverlapPrologue : MainloopWithScalePPU0015Aiu<Stages_> {
  using Schedule = Schedule_;
};

} // namespace cutlass::gemm

namespace cutlass::gemm::collective {

using namespace cute;

// OverlapPrologue variant of CollectiveMmaScaleFp4: continue the successor tile's loads from the
// tail of the current mainloop, so every configuration that does not enable it runs the
// unmodified base mainloop.
template <class Base_>
struct CollectiveMmaScaleFp4OverlapPrologue : Base_ {
  using Base = Base_;

  using Params = typename Base::Params;
  using DispatchPolicy = typename Base::DispatchPolicy;
  using TileShape = typename Base::TileShape;
  using TiledMma = typename Base::TiledMma;
  using SharedStorage = typename Base::SharedStorage;
  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;
  using SmemLayoutSFA = typename Base::SmemLayoutSFA;
  using SmemLayoutSFB = typename Base::SmemLayoutSFB;
  using SmemCopyAtomA = typename Base::SmemCopyAtomA;
  using SmemCopyAtomB = typename Base::SmemCopyAtomB;
  using TransformA = typename Base::TransformA;
  using TransformB = typename Base::TransformB;

  using Base::params_;
  using Base::gmem_tiled_copy_A;
  using Base::gmem_tiled_copy_B;
  using Base::gmem_tiled_copy_SFA;
  using Base::gmem_tiled_copy_SFB;
  using Base::warp_on_m;
  using Base::SplitAIU;
  using Base::AIU_SFA_WARP;
  using Base::AIU_SFB_WARP;
  using Base::SfaScaleBlocks;
  using Base::cal_ldmat_offset;
  using Base::sf_s2r;
  using Base::sync_compute_group;
  using Base::load_init;

  struct PipelineState {
    int smem_pipe_read = 0;
    bool preloaded = false;
  };

  // Structural contract only: no fused SFA readiness protocol, and more than Stages K-blocks so
  // the tail can issue a full set of successor loads. Which launch shapes may use the prefetch is
  // host policy (deep_gemm_fp4_common::kOverlapPrologueMeasured) and is deliberately not mirrored
  // here, so a measured row takes effect without touching device headers.
  template <int ShapeK>
  static constexpr bool use_overlap_prologue() {
    return SfaScaleBlocks == 0 &&
           cute::ceil_div(ShapeK, int(size<2>(TileShape{}))) > DispatchPolicy::Stages;
  }

  template <class ProblemShape>
  CUTLASS_DEVICE
  CollectiveMmaScaleFp4OverlapPrologue(Params params, ProblemShape problem_shape_MNKL, int batch_idx = 0)
      : Base(params, problem_shape_MNKL, batch_idx) {}

  // Compute one scheduled FP4 tile with overlap-prologue continuation. The caller owns
  // traversal and the epilogue. Without a preloaded first block and without a
  // successor there is nothing to continue, so defer to the untouched base path.
  template <class ProblemShape, class BlockCoord, class Fragment, class NextFn>
  static CUTLASS_DEVICE auto compute_tile(Params const& params, ProblemShape const& problem,
      BlockCoord const& coord, Fragment& accumulators, int thread_idx, char* smem,
      PipelineState* pipeline, NextFn const& next_fn, bool has_next) {
    if (!pipeline->preloaded && !has_next) {
      auto residue = Base::compute_tile(params, problem, coord, accumulators, thread_idx, smem);
      *pipeline = PipelineState{};
      return residue;
    }
    CollectiveMmaScaleFp4OverlapPrologue mma(params, problem);
    auto inputs = mma.load_init(problem, coord, params);
    auto const& a = get<0>(inputs);
    auto const& b = get<1>(inputs);
    auto residue = make_tuple(
        static_cast<uint32_t>(get<0>(problem)) - size<0>(a) * get<0>(coord),
        get<1>(problem) - size<0>(b) * get<1>(coord),
        get<2>(problem) - size<1>(a) * size<2>(a));
    clear(accumulators);
    mma(accumulators, inputs, accumulators, cute::make_coord_iterator(shape<2>(a)),
        size<2>(a), residue, thread_idx, smem, pipeline, next_fn, has_next);
    return residue;
  }

  /// Perform a collective-scoped matrix multiply-accumulate, always continuing into
  /// the successor tile's first Stages K-blocks from the tail of the current one.
  template <
    class... Ts,
    class FrgTensorD,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK,
    class NextFn
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf,
      PipelineState* pipeline,
      NextFn const& next_fn,
      bool has_next)
  {
    using namespace cute;
    constexpr int Stages = DispatchPolicy::Stages;
    int pipeline_base = 0;
    bool preloaded = false;
    static_assert(SfaScaleBlocks == 0, "overlap-prologue loads bypass the fused SFA readiness protocol");
    pipeline_base = pipeline->smem_pipe_read;
    preloaded = pipeline->preloaded;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    int aiu_warp_group_thread_idx = warp_idx * 32;

    int warp_m_idx = warp_idx % warp_on_m;
    int warp_n_idx = warp_idx / warp_on_m;

    auto M = get<0>(params_.problem_shape);
    auto N = get<1>(params_.problem_shape);
    auto K = get<2>(params_.problem_shape);

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

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

    Tensor gSFA = get<2>(load_inputs);
    Tensor gSFB = get<3>(load_inputs);

    Tensor sSFA = make_tensor(make_smem_ptr(storage.smem_sfa.data()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(storage.smem_sfb.data()), SmemLayoutSFB{});

    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(thread_idx);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);

    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    auto copy_to_tsm = [&](auto& mma, auto const& a, auto const& b,
                           auto const& sfa, auto const& sfb, int slot, int k_idx)
        __attribute__((always_inline)) {
      copy_aiu<SplitAIU>(
        mma.gmem_tiled_copy_A, a(_,_,_,k_idx), tAsA(_,_,_,slot),
        mma.gmem_tiled_copy_B, b(_,_,_,k_idx), tBsB(_,_,_,slot),
        warp_idx
      );
      if (warp_idx == AIU_SFA_WARP) {
        copy(mma.gmem_tiled_copy_SFA, sfa(_,_,_,k_idx), tSFAsSFA(_,_,_,slot));
      }
      if (warp_idx == AIU_SFB_WARP) {
        copy(mma.gmem_tiled_copy_SFB, sfb(_,_,_,k_idx), tSFBsSFB(_,_,_,slot));
      }
    };

    auto load_current = [&](int slot) __attribute__((always_inline)) {
      if (k_tile_count > 0) {
        copy_to_tsm(*this, tAgA, tBgB, tSFAgSFA, tSFBgSFB, slot, *k_tile_iter);
        ++k_tile_iter;
      }
    };

    if (!preloaded) {
      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < Stages; ++k_pipe) {
        int slot = pipeline_base + k_pipe;
        load_current(slot >= Stages ? slot - Stages : slot);
        cp_async_fence();
        --k_tile_count;
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < Stages; ++k_pipe) {
        ++k_tile_iter;
      }
      k_tile_count -= Stages;
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

    // tsm ld swzl needn't distinguish params inner warp
    // use original smem_tiled_copy to avoid use specific tailed_layout_tv like below, use warp_idx*32 to get scaler layout of current warp
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    smem_tiled_copy_A.smem_base_ = storage.smem_a.data();
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                 // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    smem_tiled_copy_B.smem_base_ = storage.smem_b.data();
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // create rSFA refer to tCsA/tCrA
    constexpr auto warp_iter_num_sfa = size<1>(tCsA);
    constexpr auto block_iter_num_sfa = size<2>(tCsA);
    constexpr auto warp_iter_stride_sfa = [&]() constexpr {
      if constexpr (cute::is_scaled_basis<decltype(stride<1>(tCsA))>::value) {
        return stride<1>(tCsA).value();
      } else {
        return stride<1>(tCsA);
      }
    }();
    constexpr int ldmat_iter_num_sfa = cute::ceil_div(warp_iter_num_sfa, Int<4>{});
    Tensor tCrSFA = make_tensor<uint32_t>(Shape<Int<1>, Int<ldmat_iter_num_sfa>, Int<block_iter_num_sfa>>{});
    static_assert(warp_iter_num_sfa <= 4, "warp_iter_num_sfa > 4 is not valid now.");
    // create rSFB refer to tCsB/tCrB
    constexpr auto warp_iter_num_sfb = size<1>(tCsB);
    constexpr auto block_iter_num_sfb = size<2>(tCsB);
    constexpr auto warp_iter_stride_sfb = [&]() constexpr {
      if constexpr (cute::is_scaled_basis<decltype(stride<1>(tCsB))>::value) {
        return stride<1>(tCsB).value();
      } else {
        return stride<1>(tCsB);
      }
    }();
    constexpr int ldmat_iter_num_sfb = cute::ceil_div(warp_iter_num_sfb, Int<4>{});
    Tensor tCrSFB = make_tensor<uint32_t>(Shape<Int<1>, Int<ldmat_iter_num_sfb>, Int<block_iter_num_sfb>>{});
    static_assert(warp_iter_num_sfb <= 4, "warp_iter_num_sfb > 4 is not valid now.");

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = pipeline_base;
    // Current pipe index in smem to write to
    int smem_pipe_write = pipeline_base;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);
    static_assert(K_BLOCK_MAX > 1, "K_BLOCK_MAX should be large than 1.");

    int base_offset_sfa = 0;
    int base_offset_sfb = 0;
    cal_ldmat_offset(base_offset_sfa, SmemLayoutSFA{}, lane_idx, warp_m_idx, Int<warp_iter_num_sfa>{}, Int<warp_iter_stride_sfa>{});
    cal_ldmat_offset(base_offset_sfb, SmemLayoutSFB{}, lane_idx, warp_n_idx, Int<warp_iter_num_sfb>{}, Int<warp_iter_stride_sfb>{});
    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      if (!preloaded) {
        cp_async_wait<Stages-1>();
        sync_compute_group();
      }

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
      sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, smem_pipe_read);
      sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, smem_pipe_read);
    }

    auto process_kblock_iterations = [&](auto k_block, auto const& load_fn)
        __attribute__((always_inline)) {
      if constexpr (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tCsA_p = tCsA(_,_,_,smem_pipe_read);
        tCsB_p = tCsB(_,_,_,smem_pipe_read);
        copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
        sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, smem_pipe_read);
        sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, smem_pipe_read);
      } else {
        auto k_block_next = k_block + Int<1>{};  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, k_block_next, smem_pipe_read);
        sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, k_block_next, smem_pipe_read);
      }
      // Transform before compute
      cute::transform(tCrA(_,_,k_block), TransformA{});
      cute::transform(tCrB(_,_,k_block), TransformB{});
      cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrSFA(_,_,k_block), tCrB(_,_,k_block), tCrSFB(_,_,k_block), src_accum);
      if constexpr (k_block == K_BLOCK_MAX - 2) {
        cp_async_wait<Stages-2>();
        sync_compute_group();
        load_fn(smem_pipe_write);
        cp_async_fence();

        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
        smem_pipe_write = smem_pipe_read;
      }
    };
    auto run_iteration = [&](auto const& load_fn) __attribute__((always_inline)) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block)
          __attribute__((always_inline)) {
        process_kblock_iterations(k_block, load_fn);
      });
    };

    run_iteration(load_current);
    --k_tile_count;
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      run_iteration(load_current);
    }

    {
      // Keep successor descriptors out of the steady-state register live range.
      auto [next_params, next_problem, next_coord] = next_fn();
      Base next_mma(next_params, next_problem);
      auto next_inputs = next_mma.load_init(next_problem, next_coord, next_params);
      auto next_thr_copy_A = next_mma.gmem_tiled_copy_A.get_slice(thread_idx);
      auto next_thr_copy_B = next_mma.gmem_tiled_copy_B.get_slice(thread_idx);
      Tensor next_tAgA = next_thr_copy_A.partition_S(get<0>(next_inputs));
      Tensor next_tBgB = next_thr_copy_B.partition_S(get<1>(next_inputs));
      auto next_thr_copy_SFA = next_mma.gmem_tiled_copy_SFA.get_thread_slice(thread_idx);
      auto next_thr_copy_SFB = next_mma.gmem_tiled_copy_SFB.get_thread_slice(thread_idx);
      Tensor next_tSFAgSFA = next_thr_copy_SFA.partition_S(get<2>(next_inputs));
      Tensor next_tSFBgSFB = next_thr_copy_SFB.partition_S(get<3>(next_inputs));
      int next_k_load = 0;
      auto load_next = [&](int slot) __attribute__((always_inline)) {
        if (has_next) {
          copy_to_tsm(next_mma, next_tAgA, next_tBgB, next_tSFAgSFA, next_tSFBgSFB,
                      slot, next_k_load);
          ++next_k_load;
        }
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > -Stages; --k_tile_count) {
        run_iteration(load_next);
      }
    }
    if (has_next) {
      pipeline->smem_pipe_read = smem_pipe_read;
      pipeline->preloaded = true;
      return;
    }

    cp_async_wait<0>();
    sync_compute_group();
    if (pipeline != nullptr) {
      *pipeline = PipelineState{};
    }
  }
};

} // namespace cutlass::gemm::collective
