#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#ifndef INT8_HGRTC
    #include "profiling_interface.hpp"
#endif
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/ppu_util.hpp"
#include "tools/util/include/cutlass/util/packed_stride.hpp"
#include "scheduler_cutlass3.cuh"

#include "ppu_include.hpp"
#include "utils_cutlass3.h"
#include "utils_rtc.cuh"

using namespace cute;

namespace cutlass::gemm {

template<int Stages_, typename Schedule_ = KernelAiuMultistage>
struct MainloopPPUAiuA8W8 {
  constexpr static int Stages = Stages_;
#if __HGGC_ARCH__ == 100
  using ArchTag = arch::PPU0010;
#else
  using ArchTag = arch::PPU0015;
#endif
  using Schedule = Schedule_;
  using ClusterShape = Shape<_1,_1,_1>;
};

template<int Stages_, typename Schedule_ = KernelAiuMultistageOverlapPrologue>
struct MainloopPPUAiuA8W8OverlapPrologue {
  constexpr static int Stages = Stages_;
#if __HGGC_ARCH__ == 100
  using ArchTag = arch::PPU0010;
#else
  using ArchTag = arch::PPU0015;
#endif
  using Schedule = Schedule_;
  using ClusterShape = Shape<_1,_1,_1>;
};

} // namespace cutlass::gemm

#include "int8_gemm_cutlass3_dynamic.cuh"

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////
template <
  class ProblemShapeOrThreadblockMma_, // (m, n, k) or (m, n, k, l)
  class CollectiveMainloopOrEpilogue_,
  class CollectiveEpilogueOrThreadblockSwizzle_,
  class TileScheduler_ = void,
  bool kEnableSboOverlap = false,
  class Enable = void
>
class DeepGemmUniversal;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_,
  bool kEnableSboOverlap
>
class DeepGemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  kEnableSboOverlap,
  cute::enable_if_t<cute::is_base_of_v<KernelAiuMultistageOnN, typename CollectiveMainloop_::DispatchPolicy::Schedule>>> {
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using ElementScale = float;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t N = TileScheduler::SHAPE_N;
  static constexpr uint32_t K = TileScheduler::SHAPE_K;
  static constexpr uint32_t N_PREFETCH_CACHELINE = cute::ceil_div(TileScheduler::kNumGroups, 32); // numGroups * sizeof(int) / 128 Byte = cacheline
  static constexpr uint32_t N_EXPAND = 4;

  constexpr static uint32_t CTA_M = shape<0>(TileShape{});
  constexpr static uint32_t CTA_N = shape<1>(TileShape{});
  constexpr static uint32_t CTA_K = shape<2>(TileShape{});
  using ScaleCopyAtomWidth = cute::uint32_t;
  constexpr static uint32_t ScaleGranularity = sizeof(ScaleCopyAtomWidth);
  // ScaleA
  using GmemTiledCopyScaleA = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_M / ScaleGranularity>, _1>>{},
                    Layout<Shape <Int<ScaleGranularity>,_1>>{}));

  // ScaleB
  using GmemTiledCopyScaleB = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_N / ScaleGranularity>, _1>>{},
                    Layout<Shape <Int<ScaleGranularity>,_1>>{}));

  using SmemLayoutAtomScale = Layout<Shape<Int<ScaleGranularity>, _1>>;
  using SmemLayoutScaleA = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(Int<CTA_M>{}, Int<1>{}, Int<1>{})));   // assert CTA_SCALE_K = 1, gs >= cta_k
  using SmemLayoutScaleB = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(Int<CTA_N>{}, Int<1>{}, Int<1>{})));   // assert CTA_SCALE_K = 1, gs >= cta_k


  // Kernel level shared memory storage
  struct SharedStorage {
    // Mainloop and epilogue don't use smem concurrently since kernel is non-persistent, so we can use a union
    struct SharedTensorStorage {
      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
      using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;

      MainloopSharedStorage mainloop;
      EpilogueSharedStorage epilogue;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    int32_t* signal{nullptr};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    void* workspace{nullptr};
    int32_t* signal{nullptr};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.cu_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
    // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
    // subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = 1; //CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      args.scheduler,
      workspace,
      args.signal
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hggcStream_t stream = nullptr,
    HostAdapter* host_adapter = nullptr) {
    return Status::kSuccess;
  }

  // // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    return dim3(params.hw_info.cu_count, 1, 1);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    // printf("run ppu aiu deepgemm persistent!!!");
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    int warp_idx = canonical_warp_idx_sync();
    if (TileScheduler::GEMM_TYPE == GemmType::GroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    } else if (TileScheduler::GEMM_TYPE == GemmType::GroupedNoPad) {
      // each warp prefetch one cacheline
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx<<5)));
      }
    }

    // if (thread0()) {
    //   printf("EpilogueSharedStorage size = %d\n", sizeof(CollectiveEpilogue::SharedStorage));
    // }

    // int warp_idx = cutlass::canonical_warp_idx_sync();
    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    // auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    // auto M = 16; //get<0>(problem_shape_MNKL);
    // auto N = get<1>(problem_shape_MNKL);
    // auto K = get<2>(problem_shape_MNKL);
    // auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    constexpr int BlockM = get<0>(TileShape{})();
    constexpr int BlockN = get<1>(TileShape{})();
    constexpr int BlockK = get<2>(TileShape{})();
    using TileShapeExpand = Shape<Int<BlockM>, Int<BlockN * N_EXPAND>, Int<BlockK>>;
    static_assert(BlockN * N_EXPAND == TileScheduler::BLOCK_N, "BlockN x N_EXPAND is not same as TileScheduler::BLOCK_N");
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)
    auto tile_shape = TileShape{};

    TileScheduler deep_scheduler(params.scheduler);

    uint32_t m_block_idx, n_block_idx;
    constexpr uint32_t L = 1;
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      // print("run here");
      // printf("m_block_idx is %d",m_block_idx);
      // printf("n_block_idx is %d",n_block_idx);

      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx * N_EXPAND;
      auto l_coord = 0;

      uint32_t M = deep_scheduler.curr_problem_m();

      auto problem_shape_MNKL = ProblemShape{M, N, K, L};

      auto offset_m = deep_scheduler.curr_offset_m();
    //   auto expert_id = deep_scheduler.problem_index();
      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b();
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      CollectiveMainloop collective_mma(params.mainloop, take<0, 3>(problem_shape_MNKL));
      auto load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, params.mainloop,
                                                  M, offset_m, offset_b, ptr_A, ptr_B);
      // Extract out partitioned A and B.
      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);
      Tensor gScaleA = get<2>(load_inputs);
      Tensor gScaleB = get<3>(load_inputs);

      // Compute tile residues for predication
      auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
      auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
      auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

      // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
      TiledMma tiled_mma;
      Tensor accumulators = make_fragment_like<ElementCompute>(partition_fragment_C(tiled_mma, take<0,2>(blk_shape)));


      using accum_type = decltype(accumulators);
      accum_type& accum = accumulators;
      accum_type& src_accum = accumulators;
      using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
      using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
      using GmemTiledCopyA = typename CollectiveMainloop::GmemTiledCopyA;
      using GmemTiledCopyB = typename CollectiveMainloop::GmemTiledCopyB;
      using SmemCopyAtomA = typename CollectiveMainloop::SmemCopyAtomA;
      using SmemCopyAtomB = typename CollectiveMainloop::SmemCopyAtomB;
      using TransformA = typename CollectiveMainloop::TransformA;
      using TransformB = typename CollectiveMainloop::TransformB;

      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
      GmemTiledCopyA gmem_tiled_copy_A = collective_mma.gmem_tiled_copy_A;
      GmemTiledCopyB gmem_tiled_copy_B = collective_mma.gmem_tiled_copy_B;
      GmemTiledCopyScaleA gmem_tiled_copy_scaleA = collective_mma.gmem_tiled_copy_scaleA;
      GmemTiledCopyScaleB gmem_tiled_copy_scaleB = collective_mma.gmem_tiled_copy_scaleB;

      // Construct shared memory tiles
      MainloopSharedStorage& storage = *reinterpret_cast<MainloopSharedStorage*>(smem_buf);
      Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
      // Partition the copying of A and B tiles across the threads
      auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
      auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

      Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
      Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
      Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

      // Tile MMA compute thread partitions and allocate accumulators
      // TiledMma tiled_mma;
      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
      Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
      Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

      CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
      CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
      CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
      CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
      CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

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

      Tensor sSA = make_tensor(make_smem_ptr(storage.smem_scale_a.data()), SmemLayoutScaleA{});
      Tensor sSB = make_tensor(make_smem_ptr(storage.smem_scale_b.data()), SmemLayoutScaleB{});

      auto gmem_thr_copy_scaleA = gmem_tiled_copy_scaleA.get_slice(thread_idx % (Int<CTA_M  / 4>{}));
      auto gmem_thr_copy_scaleB = gmem_tiled_copy_scaleB.get_slice(thread_idx % (Int<CTA_N  / 4>{}));

      Tensor tSgSA = gmem_thr_copy_scaleA.partition_S(gScaleA);
      Tensor tSsSA = gmem_thr_copy_scaleA.partition_D(sSA);
      Tensor tSgSB = gmem_thr_copy_scaleB.partition_S(gScaleB);
      Tensor tSsSB = gmem_thr_copy_scaleB.partition_D(sSB);

      if (warp_idx <= 1) {
        copy(gmem_tiled_copy_scaleA, tSgSA(_,_,_,0), tSsSA(_,_,_,0));
        copy(gmem_tiled_copy_scaleB, tSgSB(_,_,_,0), tSsSB(_,_,_,0));
      }

      auto k_tile_iter  = 0; //cute::make_coord_iterator(shape<2>(gA));
      int  k_tile_count = size<2>(gA);
      constexpr int K_TILE_COUNT = (K + BlockK - 1) / BlockK;

      auto k_tile_iter_reset = k_tile_iter;

      int num_n_copy_done = 0;
      // Prologue, Start async loads for all pipes but the last
      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages; ++k_pipe) {
        copy_aiu(
          gmem_tiled_copy_A, tAgA(_,_,_,k_tile_iter), tAsA(_,_,_,k_pipe),
          gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,k_pipe),
          warp_idx
        );
        cp_async_fence();
        --k_tile_count;
        ++k_tile_iter;

        // if (k_tile_iter == K_TILE_COUNT) {
        //   num_n_copy_done++;
        //   tBgB.data() = tBgB.data() + K * BlockN;
        //   k_tile_iter = 0;
        //   // if (thread0()) { printf("        update tBgB.data()\n"); }
        // }
      }
      auto k_tile_count_reset = k_tile_count;

      // scale A/B
      using SmemCopyLayoutScaleB = decltype(tile_to_shape(Layout<Shape<_1, Int<ScaleGranularity>>>{},
              make_shape(Int<1>{}, Int<CTA_N>{}, Int<DispatchPolicy::Stages>{})));
      Tensor sSB_copy = make_tensor(make_smem_ptr(storage.smem_scale_b.data()), SmemCopyLayoutScaleB{});
      Tensor tCrSA = make_fragment_like<ElementScale>(thr_mma.partition_fragment_C(sSA(_,_,Int<0>{})));
      Tensor tCrSB = make_fragment_like<ElementScale>(thr_mma.partition_fragment_C(sSB_copy(_,_,Int<0>{})));

      using SmemCopyAtomScale = Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<cutlass::sizeof_bits<ElementScale>::value>, ElementScale>;
      auto smem_tiled_copy_ScaleA   = make_tiled_copy_C(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_ScaleA     = smem_tiled_copy_ScaleA.get_thread_slice(thread_idx);
      Tensor tCsSA                  = smem_thr_copy_ScaleA.partition_S(sSA);
      Tensor tCrSA_copy_view        = smem_thr_copy_ScaleA.retile_D(tCrSA);

      auto smem_tiled_copy_ScaleB   = make_tiled_copy_C(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_ScaleB     = smem_tiled_copy_ScaleB.get_thread_slice(thread_idx);
      Tensor tCsSB                  = smem_thr_copy_ScaleB.partition_S(sSB_copy);
      Tensor tCrSB_copy_view        = smem_thr_copy_ScaleB.retile_D(tCrSB);

      // Current pipe index in smem to read from
      int smem_pipe_read  = 0;
      // Current pipe index in smem to write to
      int smem_pipe_write = 0;

      Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
      Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);
      Tensor tCsSA_p = tCsSA(_,_,_,smem_pipe_read);
      Tensor tCsSB_p = tCsSB(_,_,_,smem_pipe_read);

      // Size of the register pipeline
      auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
      auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);

      Tensor mma_acc = make_fragment_like<ElementAccumulator>(accum);
      clear(mma_acc);

      // PREFETCH register pipeline
      if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<DispatchPolicy::Stages-1>();
        __syncthreads();
        // Prefetch the first rmem from the first k-tile
        copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
        copy(smem_tiled_copy_ScaleA, tCsSA_p(_,_,Int<0>{}), tCrSA_copy_view(_,_,Int<0>{}));
        copy(smem_tiled_copy_ScaleB, tCsSB_p(_,Int<0>{},_), tCrSB_copy_view(_,Int<0>{},_));
      }

      for (int n_iter = 0; n_iter < N_EXPAND; n_iter++) {
        auto blk_coord_mnkl = make_coord(m_coord, n_coord + n_iter, _, l_coord);

        // if (thread0()) {
        //   print("mA_mkl: "); print(mA_mkl); print("\n");
        //   print("mB_nkl: "); print(mB_nkl); print("\n");
        //   // print("gA: "); print_tensor(gA); print("\n");
        //   // print("gB: "); print_tensor(gB); print("\n");
        // }

        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > -(DispatchPolicy::Stages)) {
          // Pipeline the outer products with a static for loop.
          //
          // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
          for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
              // Slice the smem_pipe_read smem
              tCsA_p = tCsA(_,_,_,smem_pipe_read);
              tCsB_p = tCsB(_,_,_,smem_pipe_read);
            }

            bool is_last_k_block = (k_block == K_BLOCK_MAX - 1) && (k_tile_count == -DispatchPolicy::Stages);
            if (!is_last_k_block) {
              // Load A, B shmem->regs for k_block+1
              auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
              copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
              copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
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

            // Copy gmem to smem after computing gemm on each k-pipe
            if (k_block == K_BLOCK_MAX - 2) {

              // Commit the smem for smem_pipe_read
              cp_async_wait<DispatchPolicy::Stages-2>();

              __syncthreads();

              if (k_tile_count > 0 || n_iter < N_EXPAND - 1) {
                // if (thread0()) {
                //   printf("K_TILE_COUNT = %d, n_iter = %d, k_tile_iter = %d, k_tile_count = %d, smem_pipe_write = %d\n",
                //       K_TILE_COUNT, n_iter, k_tile_iter, k_tile_count, smem_pipe_write);
                // }

                if (k_tile_iter == K_TILE_COUNT) {
                  num_n_copy_done++;
                  // if (thread0()) {
                  //   printf("    num_n_copy_done = %d, k_tile_iter_reset = %d\n", num_n_copy_done, k_tile_iter_reset);
                  // }
                  if (num_n_copy_done < N_EXPAND) {
                    // load for next n_iter, avoid invalid page
                    tBgB.data() = tBgB.data() + K * BlockN;
                    tSgSB.data() = tSgSB.data() + BlockN;

                    if (warp_idx <= 1) {
                      copy(gmem_tiled_copy_scaleB, tSgSB(_,_,_,0), tSsSB(_,_,_,0));
                    }
                    // if (thread0()) { printf("        update tBgB.data()\n"); }
                  }
                  k_tile_iter = k_tile_iter_reset;
                }

                copy_aiu(
                  gmem_tiled_copy_A, tAgA(_,_,_,k_tile_iter), tAsA(_,_,_,smem_pipe_write),
                  gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,smem_pipe_write),
                  warp_idx
                );
              }
              cp_async_fence();

              --k_tile_count;
              ++k_tile_iter;
              // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
              ++smem_pipe_read;
              smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
              smem_pipe_write = smem_pipe_read;
            }
          }); // for_each

        } // while k_tile_count

        // dequantize
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < cute::size<2>(accumulators); ++n) {
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < cute::size<1>(accumulators); ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size<0>(accumulators); ++i) {
              accumulators(i,m,n) = mma_acc(i,m,n) * tCrSA(i,m,Int<0>{}) * tCrSB(i,Int<0>{},n);
            // accumulators(i,m,n) = mma_acc(i,m,n);
            }
          }
        }

        // if(thread0()) {
        //   printf("    mainloop with a8w8, mma_acc[0] = %.4f, tCrSA[0] = %.8f, tCrSB[0] = %.8f, accum[0] = %.8f\n",
        //     (float)mma_acc[0], tCrSA(0,0,0), tCrSB(0,0,0), accumulators[0]);
        // }

        // update params.epilogue for ptrC and ptrD
        auto params_epilogue_local = params.epilogue;
        params_epilogue_local.ptr_C += deep_scheduler.curr_offset_c();
        params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c();

        // Epilogue and write to gD
        CollectiveEpilogue epilogue{params_epilogue_local, shared_storage.tensors.epilogue};
        epilogue(
          problem_shape_MNKL,
          blk_shape,
          blk_coord_mnkl,
          accumulators,
          tiled_mma,
          residue_mnk,
          thread_idx,
          (char*)&shared_storage.tensors.epilogue
        );

        // PREFETCH register pipeline
        if (n_iter < N_EXPAND - 1) {
          clear(mma_acc);
          k_tile_count = k_tile_count_reset;

          // Wait until our first prefetched tile is loaded in
          cp_async_wait<DispatchPolicy::Stages-1>();
          __syncthreads();

          // Prefetch the first rmem from the first k-tile
          copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
          copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
          copy(smem_tiled_copy_ScaleB, tCsSB_p(_,Int<0>{},_), tCrSB_copy_view(_,Int<0>{},_));
        }

      } // for n_iter

      if constexpr(kEnableSboOverlap && TileScheduler::GEMM_TYPE == GemmType::GroupedMasked) {
        cp_async_wait<0>();
        __syncthreads();

        if (threadIdx.x == 0) {
          atomic_add_release_global(params.signal + deep_scheduler.curr_group_idx
                  * ceil_div(deep_scheduler.params.shape_m, TileScheduler::BLOCK_M) + m_block_idx, 1);
        }
      }
    } // Scheduler work fetch loop
  }

};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Arch_,
  int Stages,
  class KernelSchedule,
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
struct CollectiveMma<
    Arch_,
    MainloopPPUAiuA8W8<Stages, KernelSchedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopPPUAiuA8W8<Stages, KernelSchedule>;
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
  using ArchTag = typename DispatchPolicy::ArchTag;

  using ElementScale = float;

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

  // tile256x256 && !NT trans, use sigle warp to issue AIU LOAD
  static constexpr bool SplitAIU = size<0>(TileShape{}) != 256 || size<1>(TileShape{}) != 256
              || (is_same_v<StrideA, cutlass::detail::TagToStrideA_t<layout::ColumnMajor>>
                  && is_same_v<StrideB, cutlass::detail::TagToStrideB_t<layout::RowMajor>>);

  constexpr static uint32_t CTA_M = shape<0>(TileShape{});
  constexpr static uint32_t CTA_N = shape<1>(TileShape{});
  constexpr static uint32_t CTA_K = shape<2>(TileShape{});

  // b32x4 is not avaliable for nopad interface, as the scale is not 16 byte alinged
  // maybe b32x4 has better perf for the other interface
  using ScaleCopyAtomWidth = cute::uint32_t;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  constexpr static uint32_t ScaleGranularity = sizeof(ScaleCopyAtomWidth) / sizeof(float);
  static constexpr int ScaleMsPerThread = cute::ceil_div(size<0>(TileShape{}), Int<MaxThreadsPerBlock * ScaleGranularity>{});
  static constexpr int ScaleNsPerThread = cute::ceil_div(size<1>(TileShape{}), Int<MaxThreadsPerBlock * ScaleGranularity>{});

  static constexpr int MmaTileM = TiledMma().template tile_size_mnk<0>();
  static constexpr int MmaTileN = TiledMma().template tile_size_mnk<1>();

  // ScaleA
  using GmemTiledCopyScaleA = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_M / ScaleMsPerThread>, _1>>{},
                    Layout<Shape <Int<ScaleMsPerThread>,_1>>{}));

  // ScaleB
  using GmemTiledCopyScaleB = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_N / ScaleNsPerThread>, _1>>{},
                    Layout<Shape <Int<ScaleNsPerThread>,_1>>{}));

  using SmemLayoutAtomScale = Layout<Shape<Int<ScaleGranularity>, _1>>;
  using SmemLayoutScaleA = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(Int<CTA_M>{}, Int<1>{}, Int<1>{})));   // assert CTA_SCALE_K = 1, gs >= cta_k
  using SmemLayoutScaleB = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(Int<CTA_N>{}, Int<1>{}, Int<1>{})));   // assert CTA_SCALE_K = 1, gs >= cta_k

  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  struct SharedStorage {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScale const* ptr_scale_A;
    ElementScale const* ptr_scale_B;
  };

  // Device side kernel params
  using Params = Arguments;

  // sm90 realization put TMA_A into params directly
  // put gmem_tiled_copy here and copy desc in kernel to simplify rtc usage
  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;

  GmemTiledCopyScaleA gmem_tiled_copy_scaleA;
  GmemTiledCopyScaleB gmem_tiled_copy_scaleB;
  //
  // Methods
  //

  template <class ProblemShape>
  CUTLASS_DEVICE
  CollectiveMma(Params params, ProblemShape problem_shape_MNK) {
    static constexpr bool TransA = is_static<decltype(get<1>(params.dA))>::value ? false : true;
    static constexpr bool TransB = is_static<decltype(get<1>(params.dB))>::value ? false : true;

    auto M = get<0>(problem_shape_MNK);
    auto N = get<1>(problem_shape_MNK);
    auto K = get<2>(problem_shape_MNK);

    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;

    gmem_tiled_copy_A.desc_.template init<ElementA, TransA, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, M, K, params.dA);
    gmem_tiled_copy_B.desc_.template init<ElementB, TransB, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, N, K, params.dB);

    gmem_tiled_copy_scaleA = make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_M / ScaleMsPerThread>, _1>>{},
                    Layout<Shape < Int<ScaleMsPerThread>,_1>>{});

    gmem_tiled_copy_scaleB = make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_N / ScaleNsPerThread>, _1>>{},
                    Layout<Shape < Int<ScaleNsPerThread>,_1>>{});
  };

  template <class ProblemShape_MNKL, class BlockCoord_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, BlockCoord_MNKL const& blk_coord_mnkl, Params const& params,
            int M, int offset_m, int64_t offset_b, ElementA const* ptr_A, ElementB const* ptr_B) {
    auto [_M,N,K,L] = problem_shape_MNKL;
    auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;
    // load init A
    Tensor mA_mkl = make_tensor(make_gmem_ptr(ptr_A), make_shape(M,K,L), params.dA);   // (m,k,l)
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,l_coord));                                 // (m,k)
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)

    // load init B
    Tensor mB_nkl = make_tensor(make_gmem_ptr(ptr_B), make_shape(N,K,L), params.dB);   //(n,k,l)
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));                                 // (n,k)
    Tensor gB = local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

    // // load init scale A/B
    Tensor mSFA_mk = make_tensor(make_gmem_ptr(params.ptr_scale_A + offset_m), make_shape(M,1));      // (n,scale_k,l)
    auto sfa_shape = make_shape(Int<CTA_M>{}, Int<1>{});
    Tensor gSFA = local_tile(mSFA_mk, sfa_shape, make_coord(m_coord, _));    // (BLK_N, 1, scale_k)
    Tensor iSFA_mk = make_identity_tensor(sfa_shape);
    Tensor cSFA = local_tile(iSFA_mk, sfa_shape, make_coord(m_coord, _));

    Tensor mSFB_nk = make_tensor(make_gmem_ptr(params.ptr_scale_B + offset_b / K), make_shape(N,1));           // (n,scale_k,l)
    auto sfb_shape = make_shape(Int<CTA_N>{}, Int<1>{});
    Tensor gSFB = local_tile(mSFB_nk, sfb_shape, make_coord(n_coord, _));    // (BLK_N, 1, scale_k)
    Tensor iSFB_nk = make_identity_tensor(sfb_shape);
    Tensor cSFB = local_tile(iSFB_nk, sfb_shape, make_coord(n_coord, _));

    // if (thread0()) {
    //   int expert_id = offset_b / K / N;
    //   printf("M = %d, m_coord = %d, n_coord = %d, l_coord = %d, offset_m = %d, expert_id = %d, ptr_A = %p, ptr_B = %p, scale_a = %p, scale_b = %p\n",
    //           M, m_coord, n_coord, l_coord, offset_m, expert_id, ptr_A, ptr_B, params.ptr_scale_A + offset_m, params.ptr_scale_B + expert_id * N);
    // }

    return cute::make_tuple(gA, gB, cute::make_tuple(gSFA, cSFA), cute::make_tuple(gSFB, cSFB));
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, hggcStream_t stream, HostAdapter* host_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }
  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class... Ts,
    class FrgTensorD,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // if (thread0()) {
    //   printf("smem total size = %d, smem_a = %d, smem_b = %d, smem_scale_a = %d, smem_scale_b = %d\n",
    //     sizeof(storage), sizeof(storage.smem_a), sizeof(storage.smem_b), sizeof(storage.smem_scale_a), sizeof(storage.smem_scale_b));
    // }

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

    Tensor gSFA = get<2, 0>(load_inputs);
    Tensor gSFB = get<3, 0>(load_inputs);
    Tensor cSFA = get<2, 1>(load_inputs);
    Tensor cSFB = get<3, 1>(load_inputs);

    int last_stage = k_tile_count % DispatchPolicy::Stages;
    ElementScale * scale_a_smem_ptr = reinterpret_cast<ElementScale*>(tAsA(_,_,_,last_stage).data().get());
    ElementScale * scale_b_smem_ptr = reinterpret_cast<ElementScale*>(tBsB(_,_,_,last_stage).data().get());
    Tensor sSA = make_tensor(make_smem_ptr(scale_a_smem_ptr), SmemLayoutScaleA{});
    Tensor sSB = make_tensor(make_smem_ptr(scale_b_smem_ptr), SmemLayoutScaleB{});

    auto gmem_thr_copy_scaleA = gmem_tiled_copy_scaleA.get_slice(thread_idx);
    auto gmem_thr_copy_scaleB = gmem_tiled_copy_scaleB.get_slice(thread_idx);

    Tensor tSgSA = gmem_thr_copy_scaleA.partition_S(gSFA);
    Tensor tSsSA = gmem_thr_copy_scaleA.partition_D(sSA);
    Tensor tSFAcSFA = gmem_thr_copy_scaleA.partition_S(cSFA);

    Tensor tSgSB = gmem_thr_copy_scaleB.partition_S(gSFB);
    Tensor tSsSB = gmem_thr_copy_scaleB.partition_D(sSB);
    Tensor tSFBcSFB = gmem_thr_copy_scaleB.partition_S(cSFB);

    Tensor tSpSA = make_tensor<bool>(shape(tSsSA));
    Tensor tSpSB = make_tensor<bool>(shape(tSsSB));
    for (int i = 0; i < size(tSpSA); ++i) {
      tSpSA(i) = get<0>(tSFAcSFA(i)) < get<0>(residue_mnk);
    }
    for (int i = 0; i < size(tSpSB); ++i) {
      tSpSB(i) = get<0>(tSFBcSFB(i)) < get<1>(residue_mnk);
    }
    bool scaleCopySendDone = false;

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages; ++k_pipe) {
      if (k_tile_count > 0) {
        copy_aiu<SplitAIU>(
          gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe),
          gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe),
          warp_idx
        );
      }
      cp_async_fence();
      --k_tile_count;
      ++k_tile_iter;
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

    // scale A/B
    using SmemCopyLayoutScaleB = decltype(tile_to_shape(Layout<Shape<_1, Int<ScaleNsPerThread>>>{},
            make_shape(Int<1>{}, Int<CTA_N>{}, Int<DispatchPolicy::Stages>{})));
    Tensor sSB_copy = make_tensor(sSB.data(), SmemCopyLayoutScaleB{});
    Tensor tCrSA = make_fragment_like<ElementScale>(thr_mma.partition_fragment_C(sSA(_,_,Int<0>{})));
    Tensor tCrSB = make_fragment_like<ElementScale>(thr_mma.partition_fragment_C(sSB_copy(_,_,Int<0>{})));

    using SmemCopyAtomScale = Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<cutlass::sizeof_bits<ElementScale>::value>, ElementScale>;
    auto smem_tiled_copy_ScaleA   = make_tiled_copy_C(SmemCopyAtomScale{}, tiled_mma);
    auto smem_thr_copy_ScaleA     = smem_tiled_copy_ScaleA.get_thread_slice(thread_idx);
    Tensor tCsSA                  = smem_thr_copy_ScaleA.partition_S(sSA);
    Tensor tCrSA_copy_view        = smem_thr_copy_ScaleA.retile_D(tCrSA);

    auto smem_tiled_copy_ScaleB   = make_tiled_copy_C(SmemCopyAtomScale{}, tiled_mma);
    auto smem_thr_copy_ScaleB     = smem_tiled_copy_ScaleB.get_thread_slice(thread_idx);
    Tensor tCsSB                  = smem_thr_copy_ScaleB.partition_S(sSB_copy);
    Tensor tCrSB_copy_view        = smem_thr_copy_ScaleB.retile_D(tCrSB);

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = 0;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);
    Tensor tCsSA_p = tCsSA(_,_,_,smem_pipe_read);
    Tensor tCsSB_p = tCsSB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
    auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-1>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
    }

    auto process_kblock_iterations = [&](int k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
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
        // Transform before compute
        cute::transform(tCrA(_,_,atom_idx), TransformA{});
        cute::transform(tCrB(_,_,atom_idx), TransformB{});
        // gemm for one tiled_mma atom on K
        cute::gemm(tiled_mma, mma_acc, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), mma_acc);
      }

      // Copy gmem to smem after computing gemm on each k-pipe
      if (k_block == K_BLOCK_MAX - 2) {

        // Commit the smem for smem_pipe_read
        cp_async_wait<DispatchPolicy::Stages-2>();

        __syncthreads();

        if (k_tile_count > 0) {
          copy_aiu<SplitAIU>(
            gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write),
            gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write),
            warp_idx
          );
        } else if (scaleCopySendDone == false) {
          copy_if(gmem_tiled_copy_scaleA, tSpSA, tSgSA(_,_,_,0), tSsSA(_,_,_,0));
          copy_if(gmem_tiled_copy_scaleB, tSpSB, tSgSB(_,_,_,0), tSsSB(_,_,_,0));
          scaleCopySendDone = true;
        }
        cp_async_fence();

        --k_tile_count;
        ++k_tile_iter;
        // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        smem_pipe_write = smem_pipe_read;
      }
    };

    for_each(make_int_sequence<K_BLOCK_MAX >{}, [&] (auto k_block) {
      process_kblock_iterations(k_block);
    }); // for_each

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -(DispatchPolicy::Stages)) {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX >{}, [&] (auto k_block) {
        process_kblock_iterations(k_block);
      }); // for_each
    }

    // TODO: original cutlass3 miss this sync
    cp_async_wait<0>();
    __syncthreads();

    // if (thread0()) {
    //   print("    gSFA="); print(gSFA); print('\n');
    //   print("    tSgSA="); print(tSgSA); print('\n');
    //   print_tensor(tSgSA);
    //   print("    tSsSA="); print(tSsSA); print('\n');
    //   print_tensor(tSsSA);
    // }
    copy(smem_tiled_copy_ScaleA, tCsSA_p(_,_,Int<0>{}), tCrSA_copy_view(_,_,Int<0>{}));
    copy(smem_tiled_copy_ScaleB, tCsSB_p(_,Int<0>{},_), tCrSB_copy_view(_,Int<0>{},_));

    // dequantize
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < cute::size<2>(accum); ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < cute::size<1>(accum); ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size<0>(accum); ++i) {
          accum(i,m,n) = mma_acc(i,m,n) * tCrSA(i,m,Int<0>{}) * tCrSB(i,Int<0>{},n);
          // accum(i,m,n) += mma_acc(i,m,n);
        }
      }
    }

    __syncthreads();
    // if(thread0()) {
    //   printf("    mainloop with a8w8, mma_acc[0] = %.4f, tCrSA[0] = %.8f, accum[0] = %.8f\n",
    //     (float)mma_acc[0], tCrSA(0,0,0), accum[0]);
    // }

  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////


namespace cutlass::gemm::kernel {

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_,
  bool kEnableSboOverlap
>
class DeepGemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  kEnableSboOverlap,
  cute::enable_if_t<cute::is_base_of_v<KernelAiuMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>>> {
public:
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t N = TileScheduler::SHAPE_N;
  static constexpr uint32_t K = TileScheduler::SHAPE_K;
  static constexpr uint32_t N_PREFETCH_CACHELINE = cute::ceil_div(TileScheduler::kNumGroups, 32); // numGroups * sizeof(int) / 128 Byte = cacheline

  // Kernel level shared memory storage
  struct SharedStorage {
    // Mainloop and epilogue don't use smem concurrently since kernel is non-persistent, so we can use a union
    union SharedTensorStorage {
      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
      using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;

      MainloopSharedStorage mainloop;
      EpilogueSharedStorage epilogue;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    int32_t* signal{nullptr};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    void* workspace{nullptr};
    int32_t* signal{nullptr};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.cu_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
    // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
    // subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = 1; //CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      args.scheduler,
      workspace,
      args.signal
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hggcStream_t stream = nullptr,
    HostAdapter* host_adapter = nullptr) {
    return Status::kSuccess;
  }

  // // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    return dim3(params.hw_info.cu_count, 1, 1);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    // printf("run ppu aiu deepgemm persistent!!!");
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    int warp_idx = canonical_warp_idx_sync();
    if (TileScheduler::GEMM_TYPE == GemmType::GroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    } else if (TileScheduler::GEMM_TYPE == GemmType::GroupedNoPad) {
      // each warp prefetch one cacheline
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx<<5)));
      }
    }

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)

    TileScheduler deep_scheduler(params.scheduler);

    uint32_t m_block_idx, n_block_idx;
    constexpr uint32_t L = 1;
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {

      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx;
      auto l_coord = 0;

      uint32_t M = deep_scheduler.curr_problem_m();

      auto problem_shape_MNKL = ProblemShape{M, N, K, L};

      auto offset_m = deep_scheduler.curr_offset_m();
      auto expert_id = deep_scheduler.problem_index();
      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx);

      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;

      // if (thread0()) {
      //   printf("M = %d, N = %d, K = %d, offset_m = %d, expert_id = %d, offset_a = %d, offset_b = %d, m_coord = %d, n_coord = %d, ptr_A = %p, ptr_B = %p\n",
      //       M, N, K, offset_m, expert_id, offset_a, offset_b, m_coord, n_coord, ptr_A, ptr_B);
      // }

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      CollectiveMainloop collective_mma(params.mainloop, take<0, 3>(problem_shape_MNKL));

      auto load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, params.mainloop,
                                                  M, offset_m, offset_b, ptr_A, ptr_B);
      // Extract out partitioned A and B.
      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);

      // Compute tile residues for predication
      auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
      auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
      auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

      // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
      TiledMma tiled_mma;
      Tensor accumulators = make_fragment_like<ElementCompute>(partition_fragment_C(tiled_mma, take<0,2>(blk_shape))); // (MMA,MMA_M,MMA_N)
      clear(accumulators);

      auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
      int  k_tile_count = size<2>(gA);

      // Perform the collective scoped MMA
      collective_mma(
        accumulators,
        load_inputs,
        accumulators,
        k_tile_iter, k_tile_count,
        residue_mnk,
        thread_idx,
        smem_buf
      );

      // update params.epilogue for ptrC and ptrD
      auto params_epilogue_local = params.epilogue;
      params_epilogue_local.ptr_C += deep_scheduler.curr_offset_c();
      params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c();
      
      // Epilogue and write to gD
      CollectiveEpilogue epilogue{params_epilogue_local, shared_storage.tensors.epilogue};
      epilogue(
        problem_shape_MNKL,
        blk_shape,
        blk_coord_mnkl,
        accumulators,
        tiled_mma,
        residue_mnk,
        thread_idx,
        (char*)&shared_storage.tensors.epilogue
      );

      if constexpr(kEnableSboOverlap && TileScheduler::GEMM_TYPE == GemmType::GroupedMasked) {
        cp_async_wait<0>();
        __syncthreads();

        if (threadIdx.x == 0) {
          atomic_add_release_global(params.signal + deep_scheduler.curr_group_idx
                  * ceil_div(deep_scheduler.params.shape_m, TileScheduler::BLOCK_M) + m_block_idx, 1);
        }
      }
    } // Scheduler work fetch loop
  }

};

} // namespace cutlass::gemm::kernel

#include "int8_gemm_cutlass3_overlap_prologue.cuh"

namespace deep_gemm {

template <typename ElementAB, typename ElementAcc,
          uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N,
          uint32_t kNumGroups, uint32_t kNumStages,
          GemmType kGemmType,
          bool kEnableSboOverlap = false,
          KernelType kKernelType = KernelType::Default>
class Gemm {

public:
    Gemm() = default;

    static void run(__ppu_bfloat16* gmem_d, int* grouped_layout, int* block_m_info,
                    uint32_t shape_m, uint32_t expected_m, ElementAB* gmem_a, float* scales_a,
                    ElementAB * gmem_b, float* scales_b,
                    hggcStream_t stream, int num_sms, uint32_t smem_size, int32_t* signal = nullptr) {
        using ElementA    = ElementAB;
        using ElementB    = ElementAB;
        using ElementC    = cutlass::bfloat16_t;
        using LayoutA     = cutlass::layout::RowMajor;
        using LayoutB     = cutlass::layout::ColumnMajor;
        using LayoutC     = cutlass::layout::RowMajor;
        using ElementD    = ElementC;
        using LayoutD     = cutlass::layout::RowMajor;
        using ElementCompute      = float;
        using ElementScalar       = ElementCompute;
        using LinearCombOutType   = ElementD;
        using OperatorClass = cutlass::arch::OpClassTensorOp;
#if __HGGC_ARCH__ == 100
        using ArchTag = cutlass::arch::PPU0010;
#else
        using ArchTag = cutlass::arch::PPU0015;
#endif

        int max_blocks_per_cu = 0;
        int smem_size_kernel = 0;
        bool kIsNoPadPreprocessLayout = false;
        hggcFuncAttributes attr;
        int DynamicTildId = 0;

        if constexpr (kKernelType == KernelType::MoeDynamicTile) {
          auto launch_dynamic_tile_kernel = [&](auto gemm_kernel_) {
            using GemmKernel = decltype(gemm_kernel_);
            using TileScheduler = typename GemmKernel::TileScheduler;
            kIsNoPadPreprocessLayout = TileScheduler::kIsNoPadPreprocessLayout;
            DynamicTildId = GemmKernel::KernelAiuDynamicTile::DynamicTildId;

            using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
            using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
            using StrideD = cutlass::detail::TagToStrideC_t<LayoutD>;

            StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape((int)shape_m, (int)SHAPE_K, 1));
            StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape((int)SHAPE_N, (int)SHAPE_K, 1));
            StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape((int)shape_m, (int)SHAPE_N, 1));

            using Epilogue = typename GemmKernel::CollectiveEpilogue;
            using ProblemShape = Shape<int,int,int,int>;
            auto problem_shape_MNKL = ProblemShape{32, SHAPE_N, SHAPE_K, 1};
            typename Epilogue::Arguments arg_epilogue = {{1.0f, 0.0f}, (ElementD*)gmem_d, stride_D, (ElementD*)gmem_d, stride_D};
            auto params_epilogue = Epilogue::to_underlying_arguments(problem_shape_MNKL, arg_epilogue, nullptr);

            int* layout_info = grouped_layout;
            // compute block_m_info
            if (TileScheduler::kIsNoPadPreprocessLayout) {
                uint32_t block_size = max(32, next_power_of_two(kNumGroups));
                computeBlockInfoKernel<TileScheduler::BLOCK_M><<<1, block_size, 0, stream>>>(
                  reinterpret_cast<const uint32_t*>(grouped_layout), kNumGroups, reinterpret_cast<uint32_t*>(block_m_info));
                layout_info = block_m_info;
            }

            typename GemmKernel::Arguments arguments {
                (ElementA*)gmem_a, stride_A, (ElementB*)gmem_b, stride_B, scales_a, scales_b,
                params_epilogue, shape_m, layout_info
            };

            max_blocks_per_cu = compute_occupancy_for_kernel<GemmKernel>();
            dim3 const block = GemmKernel::get_block_shape();
            dim3 const grid(num_sms * max_blocks_per_cu, 1, 1);
            smem_size_kernel = GemmKernel::SharedStorageSize;

            DgProfParam dg_prof_params;
            if (ProfilingInterface::Instance().get_op_info()){
                dg_prof_params.set_params(
                    kGemmType, false, std::string("int8"), kNumGroups, shape_m, SHAPE_N, SHAPE_K, expected_m,
                    grouped_layout, stream
                );
            }
            ProfilingInterface::Instance().instrument(true, dg_prof_params);
            cutlass::device_kernel<GemmKernel><<<grid, block, smem_size_kernel, stream>>>(arguments);
            ProfilingInterface::Instance().instrument(false, dg_prof_params);

            hggcFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>);
          };
          if (expected_m > 73) {
            constexpr bool kLargeEM = true;
            using GemmKernel = cutlass::gemm::kernel::DeepGemmDynamicTile<
                kGemmType, ElementA, ElementB, ElementD, ElementAcc, ElementCompute,
                SHAPE_N, SHAPE_K, kNumGroups, kLargeEM>;
            launch_dynamic_tile_kernel(GemmKernel{});
          } else {
            constexpr bool kLargeEM = false;
            using GemmKernel = cutlass::gemm::kernel::DeepGemmDynamicTile<
                kGemmType, ElementA, ElementB, ElementD, ElementAcc, ElementCompute,
                SHAPE_N, SHAPE_K, kNumGroups, kLargeEM>;
            launch_dynamic_tile_kernel(GemmKernel{});
          }
        } else {
          using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
          using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
          static constexpr int WarpOnM = BLOCK_M / WARP_M;
          static constexpr int WarpOnN = BLOCK_N / WARP_N;

          using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementAB,ElementAB,ElementAcc>::type;
          using TiledMma = TiledMMA<
              MMA_Atom<MmaInst>,
              Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
              Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _32>>;       // 1x1x1 value group

          constexpr int EnableMultistageOnN = kKernelType == KernelType::MultistageOnN
                                              && (SHAPE_N % (BLOCK_N) == 0)
                                              && (SHAPE_K > (BLOCK_K * kNumStages));
          static constexpr int N_EXPAND = EnableMultistageOnN ? KernelAiuMultistageOnN::N_EXPAND : 1;
          constexpr int EnableOverlapPrologue = kKernelType == KernelType::OverlapPrologue;
                                          // && (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(int8_t) >= WarpOnM * 16 * BLOCK_N * sizeof(int32_t); // to impl

          using KernelSchedule = cute::conditional_t<
              EnableOverlapPrologue,
              KernelAiuMultistageOverlapPrologue,
              cute::conditional_t<
                EnableMultistageOnN,
                KernelAiuMultistageOnN,
                cutlass::gemm::KernelAiuMultistage>>;

          using DispatchPolicy = cute::conditional_t<
              EnableOverlapPrologue,
              cutlass::gemm::MainloopPPUAiuA8W8OverlapPrologue<kNumStages, KernelSchedule>,
              cutlass::gemm::MainloopPPUAiuA8W8<kNumStages, KernelSchedule>>;

          static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
          static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
          static constexpr int TSM_LD_NUM = BLOCK_M == 8 ? 2 : 4;

          static constexpr int SmemLayoutStageStrideA = EnableOverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_M * BLOCK_K;
          static constexpr int SmemLayoutStageStrideB = EnableOverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_N * BLOCK_K;
          using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
          using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;
          // A
          using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
          using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
          using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
          // B
          using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
          using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
          using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

          // Mainloop
          using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
              ArchTag,
              DispatchPolicy, TileShape,
              ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
              ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
              TiledMma,
              GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
              GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
          >;

          // Epilogue
          static constexpr bool IsAligedN = SHAPE_N % BLOCK_N == 0 ? true : false;
          // reduce vreg to use ScaleType::Nothing for alpha=1 & beta=0
          using CollectiveEpilogue_noTsm = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
              cutlass::detail::TagToStrideA_t<LayoutC>,
              cutlass::detail::TagToStrideA_t<LayoutC>,
              cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
              cutlass::gemm::EpilogueDefault,
              IsAligedN>;

          static constexpr int AlignmentC = 16 / sizeof(ElementC);
          using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute>;
          using EpilogueSchedule = typename cutlass::epilogue::EpilogueSimtVectorized;
          using CollectiveEpilogue_withTsm = typename cutlass::epilogue::collective::CollectiveBuilder<
              ArchTag, cutlass::arch::OpClassTensorOp,
              TileShape, WarpShape,
              cutlass::epilogue::collective::EpilogueTileAuto,
              float, float,
              ElementC, LayoutC, AlignmentC,
              ElementC, LayoutC, AlignmentC,
              EpilogueSchedule,
              DefaultOperation
          >::CollectiveOp;

          static constexpr bool EpilogueWithTsm = false;
          using CollectiveEpilogue = typename cutlass::platform::conditional<
              EpilogueWithTsm,
              CollectiveEpilogue_withTsm,
              CollectiveEpilogue_noTsm
          >::type;

          using TileScheduler = DeepGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, kNumGroups>;
          using GemmKernel = cutlass::gemm::kernel::DeepGemmUniversal<
              Shape<int,int,int,int>,
              CollectiveMainloop,
              CollectiveEpilogue,
              TileScheduler,
              kEnableSboOverlap>;

          using StrideA = typename GemmKernel::StrideA;
          using StrideB = typename GemmKernel::StrideB;
          using StrideC = typename GemmKernel::StrideC;
          using StrideD = typename GemmKernel::StrideD;

          kIsNoPadPreprocessLayout = TileScheduler::kIsNoPadPreprocessLayout;

          int* layout_info = grouped_layout;
          // compute block_m_info
          if (TileScheduler::kIsNoPadPreprocessLayout) {
              uint32_t block_size = max(32, next_power_of_two(kNumGroups));
              computeBlockInfoKernel<BLOCK_M><<<1, block_size, 0, stream>>>(reinterpret_cast<const uint32_t*>(grouped_layout), kNumGroups, reinterpret_cast<uint32_t*>(block_m_info));
              layout_info = block_m_info;
          }

          StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape((int)shape_m, (int)SHAPE_K, 1));
          StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape((int)SHAPE_N, (int)SHAPE_K, 1));
          StrideD stride_D;
          if constexpr (kGemmType == GemmType::BatchGemm) {
            // BHD output: D[b,h,d] has offset b*H*D + h*D + d
            // In the kernel's (batch=h, M=b, N=d) model:
            // row stride = H*D = kNumGroups * SHAPE_N
            // col stride = 1
            // batch stride = D = SHAPE_N
            stride_D = StrideD{int64_t(kNumGroups) * SHAPE_N, cute::Int<1>{}, int64_t(SHAPE_N)};
          } else {
            stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape((int)shape_m, (int)SHAPE_N, 1));
          }
          auto stride_C = stride_D;
          max_blocks_per_cu = compute_occupancy_for_kernel<GemmKernel>();

          cutlass::KernelHardwareInfo hw_info;
          hw_info.device_id = 0;
          hw_info.cu_count = num_sms * max_blocks_per_cu;

          typename GemmKernel::Arguments arguments{
              cutlass::gemm::GemmUniversalMode::kGemm,
              {shape_m, SHAPE_N, SHAPE_K, 1},
              {(ElementA*)gmem_a, stride_A, (ElementB*)gmem_b, stride_B, scales_a, scales_b},
              {{1.0f, 0.0f}, (ElementC*)gmem_d, stride_C, (ElementD*)gmem_d, stride_D},
              hw_info, {shape_m, layout_info}, signal
          };

          arguments.epilogue.thread.alpha = 1;
          arguments.epilogue.thread.beta = 0;
          auto params = GemmKernel::to_underlying_arguments(arguments, nullptr);

          dim3 const block = GemmKernel::get_block_shape();
          dim3 const grid = GemmKernel::get_grid_shape(params);
          smem_size_kernel = GemmKernel::SharedStorageSize;

          DgProfParam dg_prof_params;
          if (ProfilingInterface::Instance().get_op_info()){
              dg_prof_params.set_params(
                  kGemmType, false, std::string("int8"), kNumGroups, shape_m, SHAPE_N, SHAPE_K, expected_m,
                  grouped_layout, stream
              );
          }
          ProfilingInterface::Instance().instrument(true, dg_prof_params);
          cutlass::device_kernel<GemmKernel><<<grid, block, smem_size_kernel, stream>>>(params);
          ProfilingInterface::Instance().instrument(false, dg_prof_params);

          hggcFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>);
        }

        const int threadblock_count = num_sms * max_blocks_per_cu;
        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
          static const bool kEnableMoeDynamicTile = kKernelType == KernelType::MoeDynamicTile;
          printf("[GemmGrouped-A8W8:]\n");
          printf("group:%d, problem:[%d, %d, %d], expected_m:%d, gemm_type:%s, kIsNoPadPreprocessLayout:%d, kernel_type:%s\n",
              kNumGroups, shape_m, SHAPE_N, SHAPE_K, expected_m, GemmTypeS[static_cast<int>(kGemmType)],
              kIsNoPadPreprocessLayout, KernelTypeS[static_cast<int>(kKernelType)]);
          if constexpr (!kEnableMoeDynamicTile) {
            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
                BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K, kNumStages);
          } else {
            printf("DynamicTildId:%d\n", DynamicTildId);
          }
          printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms, max_blocks_per_cu, threadblock_count);
          printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
        }

    }
};
};  // namespace deep_gemm

#pragma clang diagnostic pop
