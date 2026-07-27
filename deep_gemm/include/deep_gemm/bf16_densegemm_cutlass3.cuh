#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

/// \file bf16_densegemm_cutlass3.cuh
/// \brief Standalone BF16 DenseGemm kernel — SHAPE_N/SHAPE_K are runtime arguments.
/// Extracted from bf16_gemm_cutlass3.cuh, removing MoeDynamicTile, GroupedMasked,
/// and all non-DenseGemm paths.

// Self-contained BF16 DenseGemm header: the MainloopPPUAiuOpt dispatch policy
// and its CollectiveMma specialization are copied verbatim below (from
// bf16_gemm_cutlass3.cuh) so this header no longer depends on the group-GEMM
// header. The direct underlying includes below mirror bf16_gemm_cutlass3.cuh.
#ifndef BF16_HGRTC
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

#include "ppu_include.hpp"
#include "utils_cutlass3.h"
#include "utils_rtc.cuh"
#include "warp_on_k_reduction.hpp"

#include "densegemm_scheduler_cutlass3.cuh"

using namespace cute;

////////////////////////////////////////////////////////////////////////////////
// MainloopPPUAiuOpt dispatch policy + CollectiveMma specialization
// (copied verbatim from bf16_gemm_cutlass3.cuh; group GEMM keeps its own copy).
////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {

template<int Stages_, typename Schedule_ = KernelAiuMultistage, bool DenseS2Opt_ = false>
struct MainloopPPUAiuOpt {
  constexpr static int Stages = Stages_;
  using ArchTag = arch::PPU0015;
  using Schedule = Schedule_;
  using ClusterShape = Shape<_1,_1,_1>;
  static constexpr bool DenseS2Opt = DenseS2Opt_;
};

} // namespace cutlass::gemm

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Arch_,
  int Stages,
  class KernelSchedule,
  bool DenseS2Opt,
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
    MainloopPPUAiuOpt<Stages, KernelSchedule, DenseS2Opt>,
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

  // sm90 realization put TMA_A into params directly
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
            int offset_m, ElementA const* ptr_A, ElementB const* ptr_B) {
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

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages; ++k_pipe) {
      copy_aiu(
        gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe),
        gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe),
        warp_idx
      );
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

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -(DispatchPolicy::Stages)) {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        if (k_block == K_BLOCK_MAX - 1) {
          if constexpr (DenseS2Opt && DispatchPolicy::Stages == 2) {
            cp_async_wait<1>();
            __syncthreads();
          }
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

          if constexpr (DenseS2Opt) {
            if constexpr (DispatchPolicy::Stages > 2) {
              cp_async_wait<DispatchPolicy::Stages-2>();
              __syncthreads();
            } else {
              __syncthreads();
            }
          } else {
            cp_async_wait<DispatchPolicy::Stages-2>();
            __syncthreads();
          }

          if (k_tile_count > 0) {
            copy_aiu(
              gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write),
              gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write),
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

    }

    // TODO: original cutlass3 miss this sync
    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

////////////////////////////////////////////////////////////////////////////////
// DenseGemm-specific kernel class — reads N, K from scheduler params at runtime
////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

/// Simplified DeepGemmUniversal for BF16 DenseGemm only.
/// N and K are read at runtime from TileSchedulerParams (no compile-time SHAPE_N/K).
template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class BF16DenseGemmKernel {
public:
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

  static constexpr int WarpOnM = CUTE_STATIC_V(size<1>(typename TiledMma::ThrLayoutVMNK{}));
  static constexpr int WarpOnN = CUTE_STATIC_V(size<2>(typename TiledMma::ThrLayoutVMNK{}));
  static constexpr int WarpOnK = CUTE_STATIC_V(size<3>(typename TiledMma::ThrLayoutVMNK{}));

  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Kernel level shared memory storage
  struct SharedStorage {
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
  };

  // Convert to underlying arguments
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    int sm_count = args.hw_info.cu_count;
    if (sm_count <= 0) {
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    void* mainloop_workspace = nullptr;

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      args.scheduler,
      workspace
    };
  }

  static bool can_implement(Arguments const& args) {
    return (args.mode == GemmUniversalMode::kGemm);
  }

  static size_t get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hggcStream_t stream = nullptr,
    HostAdapter* host_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3(params.hw_info.cu_count, 1, 1);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using X = Underscore;
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    // Runtime N and K from scheduler params
    const uint32_t N = params.scheduler.shape_n;
    const uint32_t K = params.scheduler.shape_k;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L].");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L].");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L].");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L].");

    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};

    TileScheduler deep_scheduler(params.scheduler);

    uint32_t m_coord, n_coord;
    uint32_t l_coord = 0;
    constexpr uint32_t L = 1;

    while (deep_scheduler.fetch_next_work(m_coord, n_coord)) {

      uint32_t M = deep_scheduler.curr_problem_m();
      auto problem_shape_MNKL = ProblemShape{M, N, K, L};

      auto offset_m = deep_scheduler.curr_offset_m();
      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_coord);

      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      CollectiveMainloop collective_mma(params.mainloop, take<0, 3>(problem_shape_MNKL));

      auto load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, params.mainloop,
                                                  offset_m, ptr_A, ptr_B);
      // Extract out partitioned A and B.
      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);

      // Compute tile residues for predication
      auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);
      auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);
      auto k_residue   = K - size<1>(gA) * size<2>(gA);
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

      // Allocate the tiled_mma and accumulators
      TiledMma tiled_mma;
      Tensor accumulators = make_fragment_like<ElementCompute>(partition_fragment_C(tiled_mma, take<0,2>(blk_shape)));
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

      int warp_idx = canonical_warp_idx_sync();
      constexpr int WarpsPerK = WarpOnM * WarpOnN;
      const int warp_k_idx = (WarpOnK > 1) ? (warp_idx / WarpsPerK) : 0;
      if constexpr (WarpOnK > 1) {
        using ReductionPolicy_ = cutlass::gemm::kernel::WarpOnKReductionPolicy<
            CUTE_STATIC_V(get<0>(TileShape{})), CUTE_STATIC_V(get<1>(TileShape{})), CUTE_STATIC_V(get<2>(TileShape{})),
            CUTE_STATIC_V(get<0>(TileShape{})) / WarpOnM,
            CUTE_STATIC_V(get<1>(TileShape{})) / WarpOnN,
            CUTE_STATIC_V(get<2>(TileShape{})) / WarpOnK,
            DispatchPolicy::Stages,
            int(sizeof(ElementA))>;
        int lane_idx = threadIdx.x % 32;
        cutlass::gemm::kernel::warp_on_k_reduce<ReductionPolicy_>(
            accumulators, reinterpret_cast<float*>(smem_buf), warp_idx, lane_idx);
      }

      if (warp_k_idx == 0) {
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
      }
    } // Scheduler work fetch loop
  }
};

} // namespace cutlass::gemm::kernel

////////////////////////////////////////////////////////////////////////////////
// BF16 DenseGemm host-side wrapper class
////////////////////////////////////////////////////////////////////////////////

namespace deep_gemm {

template <typename ElementAB, typename ElementAcc,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N, uint32_t WARP_K,
          uint32_t kNumStages,
          bool kDenseS2Opt = false,
          KernelType kKernelType = KernelType::Default,
          bool IsAlignedN_ = true>
class BF16DenseGemm {
public:
    static_assert(kKernelType == KernelType::Default, "BF16 DenseGemm only supports the Default kernel type");

    BF16DenseGemm() = default;

    static void run(__ppu_bfloat16* gmem_d,
                    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                    __ppu_bfloat16* gmem_a, __ppu_bfloat16* gmem_b,
                    hggcStream_t stream, int num_sms, uint32_t smem_size) {
        using ElementA    = cutlass::bfloat16_t;
        using ElementB    = cutlass::bfloat16_t;
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
        using ArchTag = cutlass::arch::PPU0015;

        using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
        using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<WARP_K>>;
        static constexpr int WarpOnM = BLOCK_M / WARP_M;
        static constexpr int WarpOnN = BLOCK_N / WARP_N;
        static constexpr int WarpOnK = BLOCK_K / WARP_K;

        using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, cutlass::bfloat16_t, cutlass::bfloat16_t, float>::type;
        using TiledMma = TiledMMA<
            MMA_Atom<MmaInst>,
            Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, Int<WarpOnK>>>,
            Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, Int<WarpOnK * 16>>>;

        static_assert(BLOCK_K % WARP_K == 0, "BLOCK_K must be divisible by WARP_K");

        using KernelSchedule = cutlass::gemm::KernelAiuMultistage;

        using DispatchPolicy = cutlass::gemm::MainloopPPUAiuOpt<kNumStages, KernelSchedule, kDenseS2Opt>;

        static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
        static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;

        static constexpr int SmemLayoutStageStrideA = BLOCK_M * BLOCK_K;
        static constexpr int SmemLayoutStageStrideB = BLOCK_N * BLOCK_K;
        using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
        using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;
        // A
        using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
        using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
        using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
        // B
        using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
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

        // Epilogue — IsAlignedN driven by template param (Python JIT passes n % block_n == 0)
        static constexpr bool IsAlignedN = IsAlignedN_;
        using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
            cutlass::detail::TagToStrideA_t<LayoutC>,
            cutlass::detail::TagToStrideA_t<LayoutC>,
            cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
            cutlass::gemm::EpilogueDefault,
            IsAlignedN>;

        // DenseGemm scheduler — no SHAPE_N/K template params
        using TileScheduler = DenseGemmScheduler<BLOCK_M, BLOCK_N>;
        using GemmKernel = cutlass::gemm::kernel::BF16DenseGemmKernel<
            Shape<int,int,int,int>,
            CollectiveMainloop,
            CollectiveEpilogue,
            TileScheduler>;

        using StrideA = typename GemmKernel::StrideA;
        using StrideB = typename GemmKernel::StrideB;
        using StrideC = typename GemmKernel::StrideC;
        using StrideD = typename GemmKernel::StrideD;

        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape((int)shape_m, (int)shape_k, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape((int)shape_n, (int)shape_k, 1));
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape((int)shape_m, (int)shape_n, 1));
        auto stride_C = stride_D;

        int max_blocks_per_cu = compute_occupancy_for_kernel<GemmKernel>();

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.cu_count = num_sms * max_blocks_per_cu;

        typename GemmKernel::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {(int)shape_m, (int)shape_n, (int)shape_k, 1},
            {(ElementA*)gmem_a, stride_A, (ElementB*)gmem_b, stride_B},
            {{1.0f, 0.0f}, (ElementC*)gmem_d, stride_C, (ElementD*)gmem_d, stride_D},
            hw_info,
            {shape_m, shape_n, shape_k, nullptr}
        };

        arguments.epilogue.thread.alpha = 1;
        arguments.epilogue.thread.beta = 0;
        auto params = GemmKernel::to_underlying_arguments(arguments, nullptr);

        dim3 const block = GemmKernel::get_block_shape();
        dim3 const grid = GemmKernel::get_grid_shape(params);
        int smem_size_kernel = GemmKernel::SharedStorageSize;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(
                GemmType::DenseGemm, false, std::string("bf16"), 1, shape_m, shape_n, shape_k, shape_m,
                nullptr, stream
            );
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        cutlass::device_kernel<GemmKernel><<<grid, block, smem_size_kernel, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        hggcFuncAttributes attr;
        hggcFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>);

        const int threadblock_count = num_sms * max_blocks_per_cu;
        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            printf("[BF16DenseGemm-Standalone:]\n");
            printf("problem:[%d, %d, %d]\n", shape_m, shape_n, shape_k);
            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
                BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumStages);
            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms, max_blocks_per_cu, threadblock_count);
            printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
        }
    }
};

}  // namespace deep_gemm

#pragma clang diagnostic pop
