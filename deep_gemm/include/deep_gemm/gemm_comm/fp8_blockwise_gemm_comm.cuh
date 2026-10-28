#pragma once

#include <deep_gemm/impls/fp8_gemm.cuh>
#include <deep_gemm/gemm_comm/collectives.cuh>
#if DG_GEMM_RS_DIRECT_SFB
#include <deep_gemm/impls/direct_sfb_mainloop.cuh>
#endif

namespace deep_gemm::gemm_comm {

// Reuse native mainloop/storage/parameter types; only packet stores and
// collective completion replace the dense GEMM epilogue.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_,
          class TileScheduler_, bool kEnableSboOverlap, bool kUseNStageKernel>
class Fp8BlockwiseGemmRS {
public:
#if DG_GEMM_RS_DIRECT_SFB
  using CollectiveMainloop = cutlass::gemm::collective::DirectSfbMainloop<CollectiveMainloop_>;
  static_assert(sizeof(typename CollectiveMainloop::Params) == sizeof(typename CollectiveMainloop_::Params));
  static_assert(alignof(typename CollectiveMainloop::Params) == alignof(typename CollectiveMainloop_::Params));
  using DirectMainloopParams = typename CollectiveMainloop::Params;
  using NativeMainloopParams = typename CollectiveMainloop_::Params;
  static_assert(offsetof(DirectMainloopParams, ptr_scale_B) == offsetof(NativeMainloopParams, ptr_scale_B));
  static_assert(offsetof(DirectMainloopParams, layout_SFB) == offsetof(NativeMainloopParams, layout_SFB));
  static_assert(sizeof(typename CollectiveMainloop::Arguments) == sizeof(typename CollectiveMainloop_::Arguments));
  static_assert(sizeof(typename CollectiveMainloop::SharedStorage) == sizeof(typename CollectiveMainloop_::SharedStorage));
#else
  using CollectiveMainloop = CollectiveMainloop_;
#endif

  using Base = DeepGemmUniversal<ProblemShape_, CollectiveMainloop, CollectiveEpilogue_, TileScheduler_,
                                 kEnableSboOverlap, kUseNStageKernel>;
  static_assert(!kEnableSboOverlap && !kUseNStageKernel);
  static_assert(TileScheduler_::GEMM_TYPE == GemmType::Normal);
  using ProblemShape = typename Base::ProblemShape;
  using TileShape = typename Base::TileShape;
  using TiledMma = typename Base::TiledMma;
  using TileScheduler = typename Base::TileScheduler;
  using StrideA = typename Base::StrideA;
  using StrideB = typename Base::StrideB;
  using StrideC = typename Base::StrideC;
  using StrideD = typename Base::StrideD;
  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using ElementScale = typename Base::ElementScale;
  using LayoutSFA = typename Base::LayoutSFA;
  using MainloopParams = typename Base::MainloopParams;
  using Params = typename Base::Params;
  static constexpr int SharedStorageSize = Base::SharedStorageSize;
  static constexpr int MaxThreadsPerBlock = Base::MaxThreadsPerBlock;

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;
    gemm_comm::GemmRSState gemm_rs_state{static_cast<const int64_t*>(params.workspace)};
    const int gemm_rs_epoch = gemm_comm::begin(gemm_rs_state, smem_buf);
    TileScheduler deep_scheduler{params.scheduler};

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)
    auto strides = make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, M));
    uint32_t m_block_idx, n_block_idx;
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx;
      auto l_coord = 0;
      M = deep_scheduler.curr_problem_m();
      auto offset_scalea = deep_scheduler.curr_offset_scalea();
      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx);
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;

      const ElementScale* ptr_scale_A = reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_A) + offset_scalea;
      const ElementScale* ptr_scale_B = reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_B) + offset_b / 128 / 128;
      auto mk_layout = make_layout(
        make_shape(make_shape(Int<1>{}, M),
                  make_shape(Int<128>{}, cute::ceil_div(K, 128))),
        strides
      );
      LayoutSFA layout_SFA = make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      CollectiveMainloop collective_mainloop;
#if DG_GEMM_RS_DIRECT_SFB
      // Keep the original W128 scale pointer base; only the local loader maps
      // the native N64 coordinates back to those original values.
      collective_mainloop.sfb_n_origin64 = int(n_coord) * (size<1>(blk_shape) / 64);
#endif
      // update actual global ptr offset
      auto layout_sfb_local = params.mainloop.layout_SFB;
      MainloopParams update_params = {
        ptr_A, params.mainloop.dA, ptr_B, params.mainloop.dB, 4,
        ptr_scale_A, layout_SFA,
        ptr_scale_B, layout_sfb_local
      };
      problem_shape_MNKL = ProblemShape{M, N, K, L};
      auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);
      static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

      Tensor gA = get<0>(load_inputs);
      // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
      TiledMma tiled_mma;
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
      clear(accumulators);

      auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
      int  k_tile_count = size<2>(gA);

      // Perform the collective scoped MMA
      collective_mainloop(
        update_params,
        load_inputs,
        accumulators,
        k_tile_iter, k_tile_count,
        thread_idx,
        smem_buf
      );
      gemm_comm::store_packets<DG_GEMM_RS_M, TileScheduler::SHAPE_N, DG_GEMM_RS_WORLD, DG_GEMM_RS_RANK>(
          gemm_rs_state, gemm_rs_epoch, accumulators, tiled_mma, blk_shape, int(m_coord), int(n_coord)
          );

    } // Scheduler work fetch loop
    gemm_comm::finish<DG_GEMM_RS_M, TileScheduler::SHAPE_N, DG_GEMM_RS_WORLD, DG_GEMM_RS_RANK, DG_GEMM_RS_DIRECT_SFB>(gemm_rs_state, gemm_rs_epoch, reinterpret_cast<__ppu_bfloat16*>(params.epilogue.ptr_D));
  }
};

} // namespace deep_gemm::gemm_comm
