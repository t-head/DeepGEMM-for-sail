#pragma once

#include <type_traits>
#include <deep_gemm/mma/fp4_mma_overlap_prologue.cuh>

// Included from the tail of fp4_gemm_cutlass3.cuh, which must come first (same as the INT8/BF16
// overlap-prologue headers): the base cannot include this file and be included back.
namespace deep_gemm {

// OverlapPrologue kernel: the base kernel with only the traversal rewritten -- the successor tile is
// fetched before the current mainloop and its first Stages K-blocks are loaded from that mainloop's
// tail; Params, storage and launch geometry are inherited. Parameters mirror the base kernel so the
// selector's two branches line up, with SBO pinned false (no SBO=true base specialization yet).
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool kHasBias, int kNumExpand, bool kEnableSboOverlap, EpilogueType kEpilogueType>
class Fp4OverlapPrologue
    : public DeepGemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_,
                               kHasBias, kNumExpand, false, kEpilogueType> {
public:
  using Base = DeepGemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_,
                                 kHasBias, kNumExpand, false, kEpilogueType>;

  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;
  using ProblemShape = typename Base::ProblemShape;
  using CollectiveMainloop = typename Base::CollectiveMainloop;
  using CollectiveEpilogue = typename Base::CollectiveEpilogue;
  using TileScheduler = typename Base::TileScheduler;
  using TileShape = typename Base::TileShape;
  using TiledMma = typename Base::TiledMma;
  using MainloopParams = typename Base::MainloopParams;
  using StrideA = typename Base::StrideA;
  using StrideB = typename Base::StrideB;
  using StrideC = typename Base::StrideC;
  using StrideD = typename Base::StrideD;

  using Base::N_PREFETCH_CACHELINE;

  // The host only selects this kernel where overlap-prologue continuation is legal; these
  // assert the contract instead of silently falling back.
  static_assert(TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad ||
                TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked,
                "FP4 OverlapPrologue supports NoPad and Masked only.");
  static_assert(std::is_empty<typename CollectiveEpilogue::SharedStorage>::value,
                "FP4 OverlapPrologue requires an epilogue that does not use shared memory.");
  static_assert(!kHasBias,
                "FP4 OverlapPrologue requires the no-bias path: bias selects the TSM epilogue, whose "
                "shared storage the prefetch would overwrite.");
  static_assert(kNumExpand == 1,
                "FP4 OverlapPrologue traverses one N block per tile (the N_EXPAND==1 scheduler path).");
  static_assert(!kEnableSboOverlap,
                "FP4 OverlapPrologue does not implement the SBO signal protocol yet; wire it up here "
                "before enabling the strategy on an SBO kernel.");
  static_assert(kEpilogueType == EpilogueType::Default,
                "FP4 OverlapPrologue requires the plain epilogue: the fused act+quant tile aliases the "
                "mainloop storage, so the successor's prefetch would clobber it. Odd N is covered by the "
                "TSM fallback and the empty-storage assert above.");
  static_assert(CollectiveMainloop::template use_overlap_prologue<TileScheduler::SHAPE_K>(),
                "FP4 OverlapPrologue requires a mainloop that supports the overlap-prologue prefetch.");

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    if constexpr (TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    }
    if constexpr (TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad) {
      // each warp prefetch one cacheline
      int warp_idx = cutlass::canonical_warp_idx_sync();
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx << 5)));
      }
    }

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    TileScheduler deep_scheduler{params.scheduler};

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)

    typename CollectiveMainloop::PipelineState pipeline;
    uint32_t m_block_idx, n_block_idx;
    // Mainloop parameters of the work the scheduler currently points at.
    auto mainloop_params = [&](uint32_t m) __attribute__((always_inline)) {
      return MainloopParams{
        {m, N, K},
        params.mainloop.ptr_A + deep_scheduler.curr_offset_a(), params.mainloop.dA,
        params.mainloop.ptr_B + deep_scheduler.curr_offset_b(m_block_idx), params.mainloop.dB,
        params.mainloop.ptr_scale_A + deep_scheduler.curr_offset_mxfp4_scalea(), params.mainloop.dSFA,
        params.mainloop.ptr_scale_B + deep_scheduler.curr_offset_mxfp4_scaleb(m_block_idx), params.mainloop.dSFB
      };
    };
    bool has_work = deep_scheduler.fetch_next_work(m_block_idx, n_block_idx);
    #pragma clang loop licm(disable)
    while (has_work) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx;
      auto l_coord = 0;

      uint32_t M = deep_scheduler.curr_problem_m();
      auto problem_shape_MNKL = ProblemShape{M, N, K, L};
      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      MainloopParams update_params = mainloop_params(M);
      // Taken before the scheduler advances to the successor.
      auto params_epilogue_local = params.epilogue;
      params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c();

      TiledMma tiled_mma;
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
      // Scope the successor lookup so its state dies before the epilogue is built. Hoisting this
      // closure out of the block cost 0.25-0.65% on the long-K tiles (A/B, case1: +19 instr and
      // ~14% more samples stalled at the per-K-block commit barrier; store traffic unchanged).
      auto residue_mnk = [&]() __attribute__((always_inline)) {
        has_work = deep_scheduler.fetch_next_work(m_block_idx, n_block_idx);
        // Invoked in the tail of the current tile, after the scheduler has advanced.
        auto next_tile = [&]() __attribute__((always_inline)) {
          uint32_t next_m = deep_scheduler.curr_problem_m();
          // m_block_idx/n_block_idx name the successor; m_coord/n_coord still snapshot
          // the tile being computed.
          return make_tuple(mainloop_params(next_m), ProblemShape{next_m, N, K, L},
                            make_coord(m_block_idx, n_block_idx, _, l_coord));
        };
        return CollectiveMainloop::compute_tile(
            update_params, problem_shape_MNKL, blk_coord_mnkl, accumulators,
            thread_idx, smem_buf, &pipeline, next_tile, has_work);
      }();

      CollectiveEpilogue epilogue{params_epilogue_local, shared_storage.tensors.epilogue};
      epilogue(problem_shape_MNKL, blk_shape, blk_coord_mnkl, accumulators,
               tiled_mma, residue_mnk, thread_idx, (char*)&shared_storage.tensors.epilogue);
    }
  }
};

// The overlap kernel is the fixed NoPad/Masked, default-epilogue, no-bias, N_EXPAND=1 variant;
// everywhere else the tag resolves to the untouched base kernel. It always runs with the matching
// overlap collective: the kernel asserts `use_overlap_prologue`, which only that collective defines.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_,
          class TileScheduler_, bool hasBias, int nExpand, bool kEnableSboOverlap,
          EpilogueType kEpilogueType>
struct Fp4GemmKernelSelector {
  using type = cute::conditional_t<
      cute::is_base_of_v<KernelAiuMultistageOverlapPrologue, typename CollectiveMainloop_::DispatchPolicy::Schedule>,
      Fp4OverlapPrologue<ProblemShape_,
                         cutlass::gemm::collective::CollectiveMmaScaleFp4OverlapPrologue<CollectiveMainloop_>,
                         CollectiveEpilogue_, TileScheduler_, hasBias, nExpand, kEnableSboOverlap,
                         kEpilogueType>,
      DeepGemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_,
                        hasBias, nExpand, kEnableSboOverlap, kEpilogueType>>;
};

} // namespace deep_gemm
