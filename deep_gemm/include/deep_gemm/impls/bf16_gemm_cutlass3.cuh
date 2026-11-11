#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"


#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/tix/utils.cuh>

// #include <iostream>

#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/dispatch_policy.hpp>

#include <cute/algorithm/functional.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/tensor_predicate.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>

#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/detail/layout.hpp>

#include <cute/ppu_util.hpp>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>

#include <ppu_include.hpp>

using namespace cute;

namespace cutlass::gemm {


template<int Stages_, typename Schedule_ = KernelAiuMultistage>
struct MainloopPPUAiuOpt {
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


namespace cutlass::gemm::kernel {
using deep_gemm::GemmType;

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

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
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
  static constexpr uint32_t N_EXPAND = DispatchPolicy::Schedule::N_EXPAND;

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
    if (TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    } else if (TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad) {
      // each warp prefetch one cacheline
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx<<5)));
      }
    }

    // if (thread0()) {
    //   printf("EpilogueSharedStorage size = %d\n", sizeof(CollectiveEpilogue::SharedStorage));
    // }

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
      auto expert_id = deep_scheduler.problem_index();
      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx);
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      CollectiveMainloop collective_mma(params.mainloop, take<0, 3>(problem_shape_MNKL));
      auto load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, params.mainloop,
                                                  offset_m, expert_id, ptr_A, ptr_B);
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
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(tile_shape)); // (MMA,MMA_M,MMA_N)

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

        if (k_tile_iter == K_TILE_COUNT) {
          num_n_copy_done++;
          tBgB.data() = tBgB.data() + K * BlockN;
          k_tile_iter = 0;
          // if (thread0()) { printf("        update tBgB.data()\n"); }
        }
      }
      auto k_tile_count_reset = k_tile_count;

      // Current pipe index in smem to read from
      int smem_pipe_read = 0;
      // Current pipe index in smem to write to
      int smem_pipe_write = 0;

      Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
      Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

      // Size of the register pipeline
      auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
      auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);

      clear(accumulators);

      // PREFETCH register pipeline
      if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<DispatchPolicy::Stages-1>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
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
              cute::gemm(tiled_mma, accum, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), src_accum);
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

        // if (thread0()) {
        //   printf("accumulators[0] = %.4f\n", accumulators[0]);
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
          clear(accumulators);
          k_tile_count = k_tile_count_reset;

          // Wait until our first prefetched tile is loaded in
          cp_async_wait<DispatchPolicy::Stages-1>();
          __syncthreads();

          // Prefetch the first rmem from the first k-tile
          copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
          copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
        }

      } // for n_iter

      if constexpr(kEnableSboOverlap && TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
        cp_async_wait<0>();
        __syncthreads();

        if (threadIdx.x == 0) {
          deep_gemm::tix::atomic_add_release_global(params.signal + deep_scheduler.curr_group_idx
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
    MainloopPPUAiuOpt<Stages, KernelSchedule>,
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
  using DispatchPolicy = MainloopPPUAiuOpt<Stages, KernelSchedule>;
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
  };

  // Device side kernel params
  using Params = Arguments;

  // Copy descriptors are initialized on device rather than passed through Params.
  // put gmem_tiled_copy here and copy desc in kernel to simplify rtc usage
  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;

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
  };

  template <class ProblemShape_MNKL, class BlockCoord_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, BlockCoord_MNKL const& blk_coord_mnkl, Params const& params,
            int offset_m, int expert_id, ElementA const* ptr_A, ElementB const* ptr_B) {
    auto [M,N,K,L] = problem_shape_MNKL;
    auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;
    // load init A
    Tensor mA_mkl = make_tensor(make_gmem_ptr(ptr_A), make_shape(M,K,L), params.dA);   // (m,k,l)
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,l_coord));                                 // (m,k)
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)

    // load init B
    Tensor mB_nkl = make_tensor(make_gmem_ptr(ptr_B), make_shape(N,K,L), params.dB);   //(n,k,l)
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));                                 // (n,k)
    Tensor gB = local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

    return cute::make_tuple(gA, gB);
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
    bool PeelFirstK = false,
    bool StaticK256 = false,
    class... Ts,
    class FrgTensorD,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK,
    class PrefetchC = decltype(nullptr)
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      int warp_idx,
      char *smem_buf,
      PrefetchC prefetch_c = nullptr) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

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

    constexpr bool stage_c = !std::is_same_v<PrefetchC, decltype(nullptr)>;
    // Commit one group per ring position, including slots unused by short K.
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages; ++k_pipe) {
      if (!stage_c || k_tile_count > 0) {
        copy_aiu(
          gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe),
          gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe),
          warp_idx
        );
      } else if constexpr (stage_c) {
        prefetch_c(k_pipe);
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

    //
    // PIPELINED MAIN LOOP
    //
    // if (thread0()) {
    //     print("tiled_mma = "); print(tiled_mma); print("\n");
    // }

    // Current pipe index in smem to read from
    int smem_pipe_read = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = 0;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

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

    // Keep one BK iteration implementation for normal, peeled and static-K
    // schedules. Peeling exposes the zero accumulator to the first MMA.
    auto process_k_tile = [&](auto final_c, auto steady_ab) {
      const int live_stage = smem_pipe_read;
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        if (k_block == K_BLOCK_MAX - 1) {
          // On the last tile the next-stage register preload is unused. Keep
          // that read on the live A/B stage while the retired slots receive C.
          int next_stage = smem_pipe_read;
          if constexpr (decltype(final_c)::value) next_stage = live_stage;
          tCsA_p = tCsA(_,_,_,next_stage);
          tCsB_p = tCsB(_,_,_,next_stage);
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
          cute::gemm(tiled_mma, accum, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), src_accum);
          // if (thread0()) {
          //   print("tCrA(_,_,atom_idx) = "); print_tensor(tCrA(_,_,atom_idx)); print("\n");
          //   print("tCrB(_,_,atom_idx) = "); print_tensor(tCrB(_,_,atom_idx)); print("\n");
          //   // print("src_accum = "); print_tensor(src_accum); print("\n");
          //   print("accum = "); print_tensor(accum); print("\n");
          // }
        }

        // Copy gmem to smem after computing gemm on each k-pipe
        if (k_block == K_BLOCK_MAX - 2) {

          // The final C tile already has its A/B operands. A wait here would
          // stall its remaining MMA on C, which is only consumed by the epilogue.
          if constexpr (!decltype(final_c)::value) {
            cp_async_wait<DispatchPolicy::Stages-2>();
            __syncthreads();
          }

          if (decltype(steady_ab)::value || k_tile_count > 0) {
            copy_aiu(
              gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write),
              gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write),
              warp_idx
            );
          } else if constexpr (stage_c) {
            // All warps have read this slot for the last time. Use its empty
            // refill group for C; the next A/B slot is already ready to read.
            if constexpr (!decltype(final_c)::value) prefetch_c(smem_pipe_write);
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

    };
    if constexpr (StaticK256) {
      static_assert(PeelFirstK && std::is_same_v<ArchTag, cutlass::arch::PPU0010>);
      CUTE_STATIC_ASSERT_V(size<2>(TileShape{}) == Int<64>{});
      static_assert(DispatchPolicy::Stages == 2 || DispatchPolicy::Stages == 3);
      static_assert(!stage_c);  // C staging uses the PPU0015 collective.
      for_each(make_int_sequence<4>{}, [&](auto) { process_k_tile(std::false_type{}, std::false_type{}); });
    } else {
      constexpr int normal_end = int(stage_c) - DispatchPolicy::Stages;
      if constexpr (PeelFirstK) {
        CUTE_STATIC_ASSERT_V(K_BLOCK_MAX > Int<1>{});
        if (k_tile_count > normal_end) process_k_tile(std::false_type{}, std::false_type{});
      }
      if constexpr (stage_c && DispatchPolicy::Stages > 2) {
        // Keep C addresses and the drain decisions out of the steady-state refill.
        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > 0) process_k_tile(std::false_type{}, std::true_type{});
      }
      CUTLASS_PRAGMA_NO_UNROLL
      while (k_tile_count > normal_end) process_k_tile(std::false_type{}, std::false_type{});
      if constexpr (stage_c) {
        if (k_tile_count > -DispatchPolicy::Stages) process_k_tile(std::true_type{}, std::false_type{});
      }
    }

    // TODO: original cutlass3 miss this sync
    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////


namespace cutlass::gemm::kernel {
// The grouped C epilogues expose CStage: void for a direct read, or a
// policy that prefetches C as A/B slots retire.
template <class Epilogue, class = void>
struct GroupedCPolicy {
  static constexpr bool with_c = false;
  using Stage = void;
};

template <class Epilogue>
struct GroupedCPolicy<Epilogue, std::void_t<typename Epilogue::CStage>> {
  static constexpr bool with_c = true;
  using Stage = typename Epilogue::CStage;
};


using deep_gemm::GemmType;

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
  using CPolicy = GroupedCPolicy<CollectiveEpilogue>;
  using CStage = typename CPolicy::Stage;
  static constexpr bool kStageC = !std::is_void_v<CStage>;
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
    if (TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    } else if (TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad) {
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
    if constexpr (CPolicy::with_c && TileScheduler::GEMM_TYPE == GemmType::MGroupedContiguous) {
      // Contiguous C and D have physical row coordinates and no batch offset.
      deep_scheduler.curr_group_idx = 0;
    }

    uint32_t m_coord, n_coord;
    uint32_t l_coord = 0;
    constexpr uint32_t L = 1;

    while (deep_scheduler.fetch_next_work(m_coord, n_coord)) {

      uint32_t M = deep_scheduler.curr_problem_m();

      auto problem_shape_MNKL = ProblemShape{M, N, K, L};

      auto offset_m = deep_scheduler.curr_offset_m();
      auto expert_id = deep_scheduler.problem_index();
      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_coord);
      if constexpr (CPolicy::with_c && TileScheduler::GEMM_TYPE == GemmType::MGroupedContiguous) {
        // Skip padded tiles without an assigned expert before forming the B address.
        if (uint64_t(offset_b) >= uint64_t(TileScheduler::kNumGroups) * N * K) continue;
      }
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;

      // if (thread0()) {
      //   printf("M = %d, N = %d, K = %d, offset_m = %d, expert_id = %d, offset_a = %d, offset_b = %d, m_coord = %d, n_coord = %d, ptr_A = %p, ptr_B = %p\n",
      //       M, N, K, offset_m, expert_id, offset_a, offset_b, m_coord, n_coord, ptr_A, ptr_B);
      // }

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      CollectiveMainloop collective_mma(params.mainloop, take<0, 3>(problem_shape_MNKL));

      auto load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, params.mainloop,
                                                  offset_m, expert_id, ptr_A, ptr_B);
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

      if constexpr (kStageC) {
        const auto* group_c = params.epilogue.ptr_C + deep_scheduler.curr_offset_c();
        collective_mma(accumulators, load_inputs, accumulators, k_tile_iter, k_tile_count,
                       residue_mnk, thread_idx, warp_idx, smem_buf, [&](int slot) {
                         CStage::prefetch(smem_buf, group_c,
                                          m_coord * TileScheduler::BLOCK_M,
                                          n_coord * TileScheduler::BLOCK_N, M, N, slot);
                       });
      } else {
        collective_mma(accumulators, load_inputs, accumulators, k_tile_iter, k_tile_count,
                       residue_mnk, thread_idx, warp_idx, smem_buf);
      }

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
      if constexpr (kStageC) {
        // No warp may overwrite C with the next output tile's A/B before its peers finish reading.
        __syncthreads();
      }
      // if (thread0()) {
      //   printf("accumulators[0] = %.4f\n", accumulators[0]);
      // }

      if constexpr(kEnableSboOverlap && TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
        cp_async_wait<0>();
        __syncthreads();

        if (threadIdx.x == 0) {
          deep_gemm::tix::atomic_add_release_global(params.signal + deep_scheduler.curr_group_idx
                  * ceil_div(deep_scheduler.params.shape_m, TileScheduler::BLOCK_M) + m_coord, 1);
        }
      }
    } // Scheduler work fetch loop
  }

};

} // namespace cutlass::gemm::kernel



#pragma clang diagnostic pop

namespace deep_gemm {

struct Model810eReduceParams {
    const float* partial;
    void* d;
    int64_t count;
};

template <typename Output, int Splits>
__device__ void model_810e_reduce(const Model810eReduceParams p) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= p.count) return;
    float value = 0.0f;
    #pragma unroll
    for (int s = 0; s < Splits; ++s)
        value += p.partial[int64_t(s) * p.count + i];
    // BF16 conversion occurs only after all FP32 split partials are summed.
    static_cast<Output*>(p.d)[i] = Output(value);
}


template <class Mainloop, typename Output, int Splits>
struct Model810eDenseKernel {
    template <class Params>
    CUTLASS_DEVICE void operator()(Params const& p, char* smem) {
        using namespace cute;
        using Tile = typename Mainloop::TileShape;
        using Mma = typename Mainloop::TiledMma;
        constexpr int BM = size<0>(Tile{}), BN = size<1>(Tile{}), BK = size<2>(Tile{});
        const int m = get<0>(p.problem_shape), n = get<1>(p.problem_shape);
        const int k = get<2>(p.problem_shape);
        const int mi = int(blockIdx.y), ni = int(blockIdx.x), split = int(blockIdx.z);
        // Partition complete BK tiles. The API enforces K%BK=0 and at least one
        // tile per split, so AIU never reads across a logical row's K boundary.
        const int begin = ((k / BK) * split / Splits) * BK;
        const int length = ((k / BK) * (split + 1) / Splits) * BK - begin;
        auto problem = make_shape(m, n, length, 1);
        auto coord = make_coord(mi, ni, _, 0);
        Mainloop loop(p.mainloop, take<0, 3>(problem));
        auto inputs = loop.load_init(problem, coord, p.mainloop, 0, 0,
                                     p.mainloop.ptr_A + begin, p.mainloop.ptr_B + begin);
        auto gA = get<0>(inputs);
        auto residue = make_tuple(m - mi * BM, n - ni * BN,
                                  length - size<1>(gA) * size<2>(gA));
        Mma mma;
        auto accum = make_fragment_like<float>(partition_fragment_C(mma, take<0, 2>(Tile{})));
        clear(accum);
        auto iter = make_coord_iterator(shape<2>(gA));
        loop(accum, inputs, accum, iter, size<2>(gA), residue,
             int(threadIdx.x), cutlass::canonical_warp_idx_sync(), smem);
        auto coords = mma.get_slice(int(threadIdx.x)).partition_C(
            make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));
        CUTE_UNROLL
        for (int i = 0; i < size(accum); ++i) {
            const int row = mi * BM + get<0>(coords(i));
            const int col = ni * BN + get<1>(coords(i));
            if (row < m && col < n) {
                const int64_t offset = int64_t(row) * n + col;
                if constexpr (Splits == 1)
                    p.epilogue.ptr_D[offset] = Output(accum(i));
                else
                    static_cast<float*>(p.workspace)[int64_t(split) * m * n + offset] = accum(i);
            }
        }
    }
};

template <class Mainloop>
struct Model810eBatchedKernel {
    template <class Params>
    CUTLASS_DEVICE void operator()(Params const& p, char* smem) {
        using namespace cute;
        using Tile = typename Mainloop::TileShape;
        using Mma = typename Mainloop::TiledMma;
        constexpr int BM = size<0>(Tile{}), BN = size<1>(Tile{});
        const int m = get<0>(p.problem_shape), n = get<1>(p.problem_shape);
        const int k = get<2>(p.problem_shape);
        const int batch = int(blockIdx.z), mi = int(blockIdx.y), ni = int(blockIdx.x);
        auto problem = make_shape(m, n, k, 1);
        auto coord = make_coord(mi, ni, _, 0);
        Mainloop loop(p.mainloop, take<0, 3>(problem));
        auto inputs = loop.load_init(problem, coord, p.mainloop, 0, 0,
            p.mainloop.ptr_A + int64_t(batch) * get<2>(p.mainloop.dA),
            p.mainloop.ptr_B + int64_t(batch) * get<2>(p.mainloop.dB));
        auto gA = get<0>(inputs);
        auto residue = make_tuple(m - mi * BM, n - ni * BN,
                                  k - size<1>(gA) * size<2>(gA));
        Mma mma;
        auto accum = make_fragment_like<float>(partition_fragment_C(mma, take<0, 2>(Tile{})));
        clear(accum);
        auto iter = make_coord_iterator(shape<2>(gA));
        loop(accum, inputs, accum, iter, size<2>(gA), residue,
             int(threadIdx.x), cutlass::canonical_warp_idx_sync(), smem);
        // TiledMma supplies the 810E lane mapping; CUDA/890P formulas differ.
        auto coords = mma.get_slice(int(threadIdx.x)).partition_C(
            make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));
        CUTE_UNROLL
        for (int i = 0; i < size(accum); ++i) {
            const int row = mi * BM + get<0>(coords(i));
            const int col = ni * BN + get<1>(coords(i));
            if (row < m && col < n) {
                const int64_t offset = int64_t(batch) * get<2>(p.epilogue.dD) +
                                       int64_t(row) * get<0>(p.epilogue.dD) + col;
                p.epilogue.ptr_D[offset] = cutlass::bfloat16_t(accum(i));
            }
        }
    }
};

template <class Base, bool StaticK256 = false>
struct Model810eFirstKPeel : Base {
    using Base::Base;
    template <class... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
        Base::template operator()<true, StaticK256>(static_cast<Args&&>(args)...);
    }
};

// Input descriptors retain the caller's row strides. Batch and split-K both
// use grid.z, but only batch changes input bases; split-K changes the K interval.
template <class Mainloop, class Epilogue, bool SplitK, int Splits, bool GridMFirst, bool StaticM36 = false>
struct Model810eTsmKernel {
    template <class Params>
    CUTLASS_DEVICE void operator()(Params const& p, char* smem) {
        using namespace cute;
        using Tile = typename Mainloop::TileShape;
        using Mma = typename Mainloop::TiledMma;
        constexpr int BM = size<0>(Tile{}), BN = size<1>(Tile{}), BK = size<2>(Tile{});
        const int m = StaticM36 ? 36 : get<0>(p.problem_shape), n = StaticM36 ? 512 : get<1>(p.problem_shape);
        const int k = StaticM36 ? 256 : get<2>(p.problem_shape), groups = get<3>(p.problem_shape);
        const int mi = GridMFirst ? int(blockIdx.x) : int(blockIdx.y);
        const int ni = GridMFirst ? int(blockIdx.y) : int(blockIdx.x);
        const int zi = int(blockIdx.z);
        const int begin = SplitK ? ((k / BK) * zi / Splits) * BK : 0;
        const int length = SplitK ? ((k / BK) * (zi + 1) / Splits) * BK - begin : k;
        const auto* ptr_a = p.mainloop.ptr_A + (SplitK ? int64_t(begin) : int64_t(zi) * get<2>(p.mainloop.dA));
        const auto* ptr_b = p.mainloop.ptr_B + (SplitK ? int64_t(begin) : int64_t(zi) * get<2>(p.mainloop.dB));
        const auto input_problem = make_shape(m, n, length, 1);
        const auto input_coord = make_coord(mi, ni, _, 0);
        Mainloop loop(p.mainloop, take<0, 3>(input_problem));
        auto inputs = loop.load_init(input_problem, input_coord, p.mainloop, 0, 0, ptr_a, ptr_b);
        auto gA = get<0>(inputs);
        auto residue = make_tuple(m - mi * BM, n - ni * BN,
                                  length - size<1>(gA) * size<2>(gA));
        Mma mma;
        auto accum = make_fragment_like<float>(partition_fragment_C(mma, take<0, 2>(Tile{})));
        clear(accum);
        auto iter = make_coord_iterator(shape<2>(gA));
        loop(accum, inputs, accum, iter, size<2>(gA), residue,
             int(threadIdx.x), cutlass::canonical_warp_idx_sync(), smem);

        // The AIU mainloop ends with cp_async_wait<0>() and __syncthreads().
        // Reuse its storage for the software-swizzled epilogue only after that
        // barrier. These are fresh register writes, not reads of AIU data.
        auto& epi_storage = *reinterpret_cast<typename Epilogue::SharedStorage*>(smem);
        Epilogue epilogue{p.epilogue, epi_storage};
        // Output strides and z coordinate are real. For split-K the host builds
        // a FLOAT epilogue and D[splits,M,N], so no partial BF16 rounding occurs.
        epilogue(make_shape(m, n, length, groups), Tile{}, make_coord(mi, ni, _, zi),
                 accum, mma, residue, int(threadIdx.x), smem);
    }
};

} // namespace deep_gemm
