#pragma once

#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
#include <deep_gemm/gemm_comm/collectives.cuh>

namespace deep_gemm::gemm_comm {

// Channel scales are applied by the native A8W8 mainloop before packet stores.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_,
          class TileScheduler_, bool kEnableSboOverlap>
class Fp8ChannelwiseGemmRS {
public:
  using Base = cutlass::gemm::kernel::DeepGemmUniversal<ProblemShape_, CollectiveMainloop_,
                                                     CollectiveEpilogue_, TileScheduler_, kEnableSboOverlap>;
  static_assert(!kEnableSboOverlap && TileScheduler_::GEMM_TYPE == GemmType::Normal);
  using ProblemShape = typename Base::ProblemShape;
  using CollectiveMainloop = typename Base::CollectiveMainloop;
  using TileShape = typename Base::TileShape;
  using TiledMma = typename Base::TiledMma;
  using TileScheduler = typename Base::TileScheduler;
  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using ElementCompute = typename Base::ElementCompute;
  using SharedStorage = typename Base::SharedStorage;
  using Params = typename Base::Params;
  static constexpr int SharedStorageSize = Base::SharedStorageSize;
  static constexpr int MaxThreadsPerBlock = Base::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor = Base::MinBlocksPerMultiprocessor;
  static constexpr int N = TileScheduler::SHAPE_N;
  static constexpr int K = TileScheduler::SHAPE_K;

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;
    const int thread_idx = int(threadIdx.x);
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    GemmRSState state{static_cast<const int64_t*>(params.workspace)};
    const int epoch = begin(state, smem_buf);
    TileScheduler scheduler{params.scheduler};
    auto blk_shape = TileShape{};
    uint32_t m_block_idx, n_block_idx;
    while (scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      const auto m_coord = [&] {
        if constexpr (DG_GEMM_RS_SPECIALIZE && DG_GEMM_RS_M <= size<0>(TileShape{}))
          return uint32_t(0);
        else
          return m_block_idx;
      }();
      const auto M = [&] {
        if constexpr (DG_GEMM_RS_SPECIALIZE)
          return uint32_t(DG_GEMM_RS_M);
        else
          return scheduler.curr_problem_m();
      }();
      auto problem_shape = ProblemShape{M, N, K, 1};
      const auto offset_m = scheduler.curr_offset_m();
      const auto offset_a = scheduler.curr_offset_a();
      const auto offset_b = scheduler.curr_offset_b(m_block_idx);
      const auto* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const auto* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;
      auto block_coord = make_coord(m_coord, n_block_idx, _, 0);
      CollectiveMainloop mainloop{params.mainloop, take<0, 3>(problem_shape)};
      auto inputs = mainloop.load_init(problem_shape, block_coord, params.mainloop,
                                      M, offset_m, offset_b, ptr_A, ptr_B);
      Tensor gA = get<0>(inputs);
      Tensor gB = get<1>(inputs);
      const auto residue_m = M - size<0>(gA) * m_coord;
      const auto residue_n = [&] {
        if constexpr (N % size<1>(TileShape{}) == 0)
          return size<1>(TileShape{});
        else
          return N - size<0>(gB) * n_block_idx;
      }();
      const auto residue_k = K - size<1>(gA) * size<2>(gA);
      auto residue = make_tuple(residue_m, residue_n, residue_k);
      TiledMma mma;
      Tensor accumulators = make_fragment_like<ElementCompute>(partition_fragment_C(mma, take<0, 2>(blk_shape)));
      clear(accumulators);
      auto k_iter = make_coord_iterator(shape<2>(gA));
      int k_count = size<2>(gA);
      mainloop(accumulators, inputs, accumulators, k_iter, k_count,
               residue, thread_idx, warp_idx, smem_buf);
      store_packets<DG_GEMM_RS_M, N, DG_GEMM_RS_WORLD, DG_GEMM_RS_RANK>(
          state, epoch, accumulators, mma, blk_shape, int(m_coord), int(n_block_idx));
    }
    finish<DG_GEMM_RS_M, N, DG_GEMM_RS_WORLD, DG_GEMM_RS_RANK, true>(
        state, epoch, reinterpret_cast<__ppu_bfloat16*>(params.epilogue.ptr_D));
  }
};

} // namespace deep_gemm::gemm_comm
