#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#include "cutlass/cutlass.h"
#include "utils.cuh"
#include <iostream>
#include "profiling_interface.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/tensor.hpp"
#include "cute/ppu_util.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "ppu_include.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/config/gemm_configs.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cutlass/numeric_conversion.h"
#include "tools/util/include/cutlass/util/host_tensor.h"
#include "tools/util/include/cutlass/util/packed_stride.hpp"
#include "scheduler_cutlass3.cuh"
#include "utils_cutlass3.h"
#include "fp4_gemm_cutlass3_dynamic.cuh"
#include "fp4_mma.cuh"
#include <math.h>

using namespace cute;

namespace deep_gemm {
using cutlass::KernelHardwareInfo;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_,
  bool hasBias = false,
  int nExpand = 1,
  bool kEnableSboOverlap = false
>
class DeepGemmUniversal;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_,
  bool hasBias_,
  int nExpand
>
class DeepGemmUniversal <
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  hasBias_,
  nExpand,
  false
> {
  public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using ElementSFA  = typename CollectiveMainloop::ElementSFA;
  using StrideSFA = typename CollectiveMainloop::StrideSFA;
  using ElementSFB  = typename CollectiveMainloop::ElementSFB;
  using StrideSFB = typename CollectiveMainloop::StrideSFB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopParams = typename CollectiveMainloop::Params;

  using ThrLayoutVMNK = typename TiledMma::ThrLayoutVMNK;
  using GmemTiledCopySFA = typename CollectiveMainloop::GmemTiledCopySFA;
  using GmemTiledCopySFB = typename CollectiveMainloop::GmemTiledCopySFB;
  static constexpr int warp_on_m = CollectiveMainloop::warp_on_m;
  static constexpr int warp_on_n = CollectiveMainloop::warp_on_n;

  // warp_num == 1, use sigle warp to issue AIU LOAD
  static constexpr bool SplitAIU = CollectiveMainloop::SplitAIU;
  static constexpr int AIU_SFA_WARP = CollectiveMainloop::AIU_SFA_WARP;
  static constexpr int AIU_SFB_WARP = CollectiveMainloop::AIU_SFB_WARP;

  // FLUX would deliver in own private PersistentScheduler for gemm+communication kernel
  // static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
  //   "SM70 kernel does not support specializing the tile scheduler.");
  // using TileSchedulerTag = TileScheduler_;
  // using TileScheduler = typename detail::TileSchedulerSelector<
  //   TileScheduler_, ArchTag, TileShape,
  //   cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
  using KernelHardwareInfo = cutlass::KernelHardwareInfo;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  static constexpr int BlockM = size<0>(TileShape{});
  static constexpr int BlockN = size<1>(TileShape{});
  static constexpr int BlockK = size<2>(TileShape{});
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr int N_EXPAND = nExpand;
  static constexpr uint32_t N_PREFETCH_CACHELINE = cute::ceil_div(TileScheduler::kNumGroups, 32);   // numGroups * sizeof(int) / 128 Byte = cacheline
  static constexpr int Stages = DispatchPolicy::Stages;

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

  // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

#if SAIL_SYNC_IN_CE
  // only 256/256 tile enable sync tb in CE
  static constexpr bool EnableSyncCE = (size<0>(TileShape{}) == 256) && (size<1>(TileShape{}) == 256) && (!cute::is_same_v<cutlass::float4_t, ElementA>) && (!cute::is_same_v<cutlass::float4_t, ElementB>);
#endif

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    int32_t* signal{nullptr};
#if SAIL_SYNC_IN_CE
    int *ptr_sync_ce = nullptr;
#endif

#if SAIL_SYNC_IN_CE
    Arguments(
      GemmUniversalMode mode_ = GemmUniversalMode{},
      ProblemShape problem_shape_ = ProblemShape{},
      MainloopArguments mainloop_ = MainloopArguments{},
      EpilogueArguments epilogue_ = EpilogueArguments{},
      KernelHardwareInfo hw_info_ = KernelHardwareInfo{},
      TileSchedulerArguments scheduler_ = TileSchedulerArguments{},
      int32_t* signal_ = nullptr)
    : mode(mode_), problem_shape(problem_shape_), mainloop(mainloop_),
      epilogue(epilogue_), hw_info(hw_info_), scheduler(scheduler_), signal(signal_){}
#endif
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
#if SAIL_SYNC_IN_CE
    int *ptr_sync_ce = nullptr;
#endif
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace, int* grouped_layout) {
    (void) workspace;
    auto problem_shape = args.problem_shape;
    if constexpr (cutlass::gemm::kernel::detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      cute::get<0>(problem_shape) = cute::get<1>(args.problem_shape);
      cute::get<1>(problem_shape) = cute::get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.cu_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    constexpr uint32_t NumEpilogueSubTiles = 1;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    void* scheduler_workspace = workspace_ptr;

    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(grouped_layout,
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);
    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
      hw_info,
      scheduler,
      workspace,
      args.signal
    };
  }

  static bool
  can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
  }

  static int
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = cutlass::round_nearest(workspace_size,  cutlass::MinWorkspaceAlignment);

    return workspace_size;
  }

  static
  cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hggcStream_t stream = nullptr,
                       cutlass::HostAdapter* host_adapter = nullptr) {
    cutlass::Status status = cutlass::Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, host_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = cutlass::round_nearest(workspace_offset,  cutlass::MinWorkspaceAlignment);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    return status;
  }

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
    using namespace cute;
    using X = Underscore;

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    int aiu_warp_group_thread_idx = warp_idx * 32;
    int warp_m_idx = warp_idx % warp_on_m;
    int warp_n_idx = warp_idx / warp_on_m;

    if constexpr (TileScheduler::GEMM_TYPE == GemmType::GroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    }
    if constexpr (TileScheduler::GEMM_TYPE == GemmType::GroupedNoPad) {
      // each warp prefetch one cacheline
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx << 5)));
      }
    }

    TileScheduler deep_scheduler{params.scheduler};

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto N = Int<TileScheduler::SHAPE_N>{};
    auto K = Int<TileScheduler::SHAPE_K>{};
    constexpr int L = 1;

    // Preconditions
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)

    uint32_t m_block_idx, n_block_idx;
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx * N_EXPAND;
      auto l_coord = 0;

      uint32_t M = deep_scheduler.curr_problem_m();

      auto problem_shape_MNKL = ProblemShape{M, N, K, L};

      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx);
      auto offset_scalea = deep_scheduler.curr_offset_mxfp4_scalea();
      auto offset_scaleb = deep_scheduler.curr_offset_mxfp4_scaleb(m_block_idx);
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;
      const ElementSFA* ptr_scale_A = reinterpret_cast<const ElementSFA*>(params.mainloop.ptr_scale_A) + offset_scalea;
      const ElementSFB* ptr_scale_B = reinterpret_cast<const ElementSFB*>(params.mainloop.ptr_scale_B) + offset_scaleb;

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      // update actual global ptr offset
      MainloopParams update_params = {
        {M, N, K}, ptr_A, params.mainloop.dA, ptr_B, params.mainloop.dB,
        ptr_scale_A, params.mainloop.dSFA, ptr_scale_B, params.mainloop.dSFB
      };
      CollectiveMainloop collective_mma(update_params, problem_shape_MNKL);
      auto load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);

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
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)

      auto k_tile_iter  = 0;
      int  k_tile_count = size<2>(gA);

      using FrgTensorC = decltype(accumulators);
      FrgTensorC& accum = accumulators;

      using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
      using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
      using GmemTiledCopyA = typename CollectiveMainloop::GmemTiledCopyA;
      using GmemTiledCopyB = typename CollectiveMainloop::GmemTiledCopyB;
      using SmemCopyAtomA = typename CollectiveMainloop::SmemCopyAtomA;
      using SmemCopyAtomB = typename CollectiveMainloop::SmemCopyAtomB;
      using TransformA = typename CollectiveMainloop::TransformA;
      using TransformB = typename CollectiveMainloop::TransformB;

      GmemTiledCopyA& gmem_tiled_copy_A = collective_mma.gmem_tiled_copy_A;
      GmemTiledCopyB& gmem_tiled_copy_B = collective_mma.gmem_tiled_copy_B;

      // Construct shared memory tiles
      Tensor sA = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

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

      // ========= fp4 scale gmem tiledCopy =========
      using SmemLayoutSFA = typename CollectiveMainloop::SmemLayoutSFA;
      using SmemLayoutSFB = typename CollectiveMainloop::SmemLayoutSFB;

      GmemTiledCopySFA& gmem_tiled_copy_SFA = collective_mma.gmem_tiled_copy_SFA;
      GmemTiledCopySFB& gmem_tiled_copy_SFB = collective_mma.gmem_tiled_copy_SFB;

      Tensor gSFA = get<2>(load_inputs);
      Tensor gSFB = get<3>(load_inputs);

      Tensor sSFA = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_sfa.data()), SmemLayoutSFA{});
      Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_sfb.data()), SmemLayoutSFB{});

      auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(thread_idx);
      auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);

      Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
      Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

      Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
      Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

      // Tile MMA compute thread partitions and allocate accumulators
      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
      Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
      Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

      CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
      CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
      CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

      //
      // Copy Atom retiling
      //

      // tsm ld swzl needn't distinguish params inner warp
      // use original smem_tiled_copy to avoid use specific tailed_layout_tv like below, use warp_idx*32 to get scaler layout of current warp
      auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
      smem_tiled_copy_A.smem_base_ = shared_storage.tensors.mainloop.smem_a.data();
      auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
      Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                 // (CPY,CPY_M,CPY_K,PIPE)
      Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
      CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
      CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

      auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
      smem_tiled_copy_B.smem_base_ = shared_storage.tensors.mainloop.smem_b.data();
      auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
      Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
      Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
      CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
      CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

      // ========= fp4 scale smem tiledCopy =========
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

      auto copy_to_tsm = [&](int k_pipe_write, int k_idx, int warp_idx) {
        copy_aiu<SplitAIU>(
          gmem_tiled_copy_A, tAgA(_,_,_,k_idx), tAsA(_,_,_,k_pipe_write),
          gmem_tiled_copy_B, tBgB(_,_,_,k_idx), tBsB(_,_,_,k_pipe_write),
          warp_idx
        );
        if (warp_idx == AIU_SFA_WARP) {
          copy(gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,k_idx), tSFAsSFA(_,_,_,k_pipe_write));
        }
        if (warp_idx == AIU_SFB_WARP) {
          copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_idx), tSFBsSFB(_,_,_,k_pipe_write));
        }
      };

      int n_tile_iter = 0;
      constexpr int K_TILE_COUNT = (K + BlockK - 1) / BlockK;

      // Prologue, Start async loads for all pipes but the last
      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < Stages; ++k_pipe) {
        if (k_tile_iter >= K_TILE_COUNT) {
          ++n_tile_iter;
          if (n_tile_iter < N_EXPAND) {
            tBgB.data() = tBgB.data() + K * BlockN;
            // gSFB is N-Major
            tSFBgSFB.data() = tSFBgSFB.data() + BlockN;
          }
          k_tile_iter = 0;
        }
        if (n_tile_iter < N_EXPAND) {
          copy_to_tsm(k_pipe, k_tile_iter, warp_idx);
        }
        ++k_tile_iter;
        cp_async_fence();
      }

      // Current pipe index in smem to read from
      int smem_pipe_read  = 0;
      // Current pipe index in smem to write to
      int smem_pipe_write = 0;

      Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
      Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

      // Size of the register pipeline
      auto K_BLOCK_MAX = size<2>(tCrA_copy_view);

      // tsm->vreg prefetch
      int base_offset_sfa = 0;
      int base_offset_sfb = 0;
      CollectiveMainloop::cal_ldmat_offset(base_offset_sfa, SmemLayoutSFA{}, lane_idx, warp_m_idx, Int<warp_iter_num_sfa>{}, Int<warp_iter_stride_sfa>{});
      CollectiveMainloop::cal_ldmat_offset(base_offset_sfb, SmemLayoutSFB{}, lane_idx, warp_n_idx, Int<warp_iter_num_sfb>{}, Int<warp_iter_stride_sfb>{});
      cp_async_wait<Stages - 1>();
      __syncthreads();
      copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
      CollectiveMainloop::sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, 0);
      CollectiveMainloop::sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, 0);
      CUTLASS_PRAGMA_NO_UNROLL
      for (int n_iter = 0; n_iter < N_EXPAND; n_iter++) {
        clear(accum);
        auto blk_coord_mnkl = make_coord(m_coord, n_coord + n_iter, _, l_coord);

        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_iter = 0; k_iter < K_TILE_COUNT; k_iter++) {
          // Pipeline the outer products with a static for loop.
          // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
          for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
            if constexpr (k_block == K_BLOCK_MAX - 1) {
              if (k_tile_iter >= K_TILE_COUNT) {
                if (n_tile_iter < N_EXPAND - 1) {
                  tBgB.data() = tBgB.data() + K * BlockN;
                  // gSFB is N-Major
                  tSFBgSFB.data() = tSFBgSFB.data() + BlockN;
                }
                ++n_tile_iter;
                k_tile_iter = 0;
              }
              if (n_tile_iter < N_EXPAND) {
                __syncthreads();
                copy_to_tsm(smem_pipe_write, k_tile_iter, warp_idx);
              }
              cp_async_fence();
              ++k_tile_iter;
              ++smem_pipe_read;
              smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
              tCsA_p = tCsA(_,_,_,smem_pipe_read);
              tCsB_p = tCsB(_,_,_,smem_pipe_read);
              if constexpr (K_BLOCK_MAX > 1) {
                cp_async_wait<Stages - 1>();
                __syncthreads();
                copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
                copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
                CollectiveMainloop::sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, smem_pipe_read);
                CollectiveMainloop::sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, smem_pipe_read);
              }
              smem_pipe_write = smem_pipe_read;
            } else {
              auto k_block_next = k_block + Int<1>{};
              copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
              copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
              CollectiveMainloop::sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, k_block_next, smem_pipe_read);
              CollectiveMainloop::sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, k_block_next, smem_pipe_read);
            }
            // Transform before compute
            cute::transform(tCrA(_,_,k_block), TransformA{});
            cute::transform(tCrB(_,_,k_block), TransformB{});
            // gemm for one tiled_mma atom on K
            cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrSFA(_,_,k_block), tCrB(_,_,k_block), tCrSFB(_,_,k_block), accum);
            if constexpr (K_BLOCK_MAX == 1) {
              cp_async_wait<Stages - 1>();
              __syncthreads();
              copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
              copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
              CollectiveMainloop::sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, smem_pipe_read);
              CollectiveMainloop::sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, smem_pipe_read);
            }
          });
        }
        auto params_epilogue_local = params.epilogue;
        params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c();
        if constexpr (hasBias_) {
          params_epilogue_local.ptr_Bias += deep_scheduler.curr_offset_mxfp4_c();
        }
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
    }
  }
};

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_,
  bool hasBias_
>
class DeepGemmUniversal <
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  hasBias_,
  1,
  false
> {
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using ElementSFA  = typename CollectiveMainloop::ElementSFA;
  using StrideSFA = typename CollectiveMainloop::StrideSFA;
  using ElementSFB  = typename CollectiveMainloop::ElementSFB;
  using StrideSFB = typename CollectiveMainloop::StrideSFB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopParams = typename CollectiveMainloop::Params;

  // FLUX would deliver in own private PersistentScheduler for gemm+communication kernel
  // static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
  //   "SM70 kernel does not support specializing the tile scheduler.");
  // using TileSchedulerTag = TileScheduler_;
  // using TileScheduler = typename detail::TileSchedulerSelector<
  //   TileScheduler_, ArchTag, TileShape,
  //   cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
  using KernelHardwareInfo = cutlass::KernelHardwareInfo;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t NumMmaWarpGroups = 1;

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

  // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t N_PREFETCH_CACHELINE = cute::ceil_div(TileScheduler::kNumGroups, 32);   // numGroups * sizeof(int) / 128 Byte = cacheline

#if SAIL_SYNC_IN_CE
  // only 256/256 tile enable sync tb in CE
  static constexpr bool EnableSyncCE = (size<0>(TileShape{}) == 256) && (size<1>(TileShape{}) == 256) && (!cute::is_same_v<cutlass::float4_t, ElementA>) && (!cute::is_same_v<cutlass::float4_t, ElementB>);
#endif

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    int32_t* signal{nullptr};
#if SAIL_SYNC_IN_CE
    int *ptr_sync_ce = nullptr;
#endif

#if SAIL_SYNC_IN_CE
    Arguments(
      GemmUniversalMode mode_ = GemmUniversalMode{},
      ProblemShape problem_shape_ = ProblemShape{},
      MainloopArguments mainloop_ = MainloopArguments{},
      EpilogueArguments epilogue_ = EpilogueArguments{},
      KernelHardwareInfo hw_info_ = KernelHardwareInfo{},
      TileSchedulerArguments scheduler_ = TileSchedulerArguments{},
      int32_t* signal_ = nullptr)
    : mode(mode_), problem_shape(problem_shape_), mainloop(mainloop_),
      epilogue(epilogue_), hw_info(hw_info_), scheduler(scheduler_), signal(signal_){}
#endif
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
    int32_t* signal{nullptr};
#if SAIL_SYNC_IN_CE
    int *ptr_sync_ce = nullptr;
#endif
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace, int* grouped_layout) {
    (void) workspace;
    auto problem_shape = args.problem_shape;
    if constexpr (cutlass::gemm::kernel::detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      cute::get<0>(problem_shape) = cute::get<1>(args.problem_shape);
      cute::get<1>(problem_shape) = cute::get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.cu_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    constexpr uint32_t NumEpilogueSubTiles = 1;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    void* scheduler_workspace = workspace_ptr;

    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(grouped_layout,
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);

    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
      hw_info,
      scheduler,
      workspace,
      args.signal
    };
  }

  static bool
  can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
  }

  static int
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = cutlass::round_nearest(workspace_size,  cutlass::MinWorkspaceAlignment);

    return workspace_size;
  }

  static
  cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hggcStream_t stream = nullptr,
                       cutlass::HostAdapter* host_adapter = nullptr) {
    cutlass::Status status = cutlass::Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, host_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = cutlass::round_nearest(workspace_offset,  cutlass::MinWorkspaceAlignment);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    return status;
  }

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
    using namespace cute;
    using X = Underscore;

    TileScheduler deep_scheduler{params.scheduler};

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

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

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)

    if constexpr (TileScheduler::GEMM_TYPE == GemmType::GroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    }
    if constexpr (TileScheduler::GEMM_TYPE == GemmType::GroupedNoPad) {
      // each warp prefetch one cacheline
      int warp_idx = cutlass::canonical_warp_idx_sync();
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx << 5)));
      }
    }

    uint32_t m_block_idx, n_block_idx;
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx;
      auto l_coord = 0;

      uint32_t M = deep_scheduler.curr_problem_m();

      auto problem_shape_MNKL = ProblemShape{M, N, K, L};

      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx);
      auto offset_scalea = deep_scheduler.curr_offset_mxfp4_scalea();
      auto offset_scaleb = deep_scheduler.curr_offset_mxfp4_scaleb(m_block_idx);
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + offset_b;
      const ElementSFA* ptr_scale_A = reinterpret_cast<const ElementSFA*>(params.mainloop.ptr_scale_A) + offset_scalea;
      const ElementSFB* ptr_scale_B = reinterpret_cast<const ElementSFB*>(params.mainloop.ptr_scale_B) + offset_scaleb;

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      // update actual global ptr offset
      MainloopParams update_params = {
        {M, N, K}, ptr_A, params.mainloop.dA, ptr_B, params.mainloop.dB,
        ptr_scale_A, params.mainloop.dSFA, ptr_scale_B, params.mainloop.dSFB
      };
      CollectiveMainloop collective_mma(update_params, problem_shape_MNKL);
      auto load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);

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
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
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

      // Update the pointer to C and D of MXFP4 in the epilogue parameters
      auto params_epilogue_local = params.epilogue;
      params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c();
      if constexpr (hasBias_) {
        params_epilogue_local.ptr_Bias += deep_scheduler.curr_offset_mxfp4_c();
      }

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
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace deep_gemm {

template <int32_t ShapeN, int32_t ShapeK,
          int32_t BlockM, int32_t BlockN, int32_t BlockK,
          int32_t WarpM, int32_t WarpN,
          int32_t kNumGroups, int32_t kNumStages, GemmType kGemmType,
          bool kEnableSboOverlap = false, bool hasBias = false, int NExpand = 1,
          FP4DynamicTileId kDynamicTileId = FP4DynamicTileId::Disabled>
class Fp4Gemm {
  static constexpr bool kEnableMoeDynamicTile = (kDynamicTileId != FP4DynamicTileId::Disabled);
  static_assert((BlockM == 16) || (BlockM == 32) || (BlockM == 64) || (BlockM == 128) || (BlockM == 256), "BlockM should only be in [16, 32, 64, 128, 256].");
  static_assert((BlockN == 16) || (BlockN == 32) || (BlockN == 64) || (BlockN == 128) || (BlockN == 256), "BlockM should only be in [16, 32, 64, 128, 256].");
  static_assert((BlockK % 32 == 0), "BlockK must be divideable by 32.");
  static_assert((WarpM <= 64) && (WarpM % 16 == 0), "WarpM must be divideable by 16.");
  static_assert((WarpN % 16 == 0), "WarpN must be divideable by 16.");

public:
    Fp4Gemm() = default;

    static void run(uint8_t *a_ptr, uint16_t *scale_a, uint8_t *b_ptr, uint16_t *scale_b,
                    float *c_ptr, __ppu_bfloat16 *d_ptr, int shape_m, int *grouped_layout, int *block_m_info, uint32_t expected_m,
                    hggcStream_t stream, int num_sms, uint32_t smem_size, int32_t* signal = nullptr) {
        constexpr int N_EXPAND = NExpand;

        // A matrix configuration
        using         ElementA    = cutlass::float4_t;                          // Element type for A matrix operand
        using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
        constexpr int AlignmentA  = 1;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

        // B matrix configuration
        using         ElementB    = cutlass::float4_t;                          // Element type for B matrix operand
        using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
        constexpr int AlignmentB  = 1;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

        // C matrix configuration
        using         ElementC    = float;                          // Element type for C and D matrix operands
        using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
        constexpr int AlignmentC  = 1;
        constexpr int AlignmentD  = 1;
        // D matrix configuration
        using         ElementD    = cutlass::bfloat16_t;
        using         LayoutD     = LayoutC;
        using ElementAccumulator = float;
        using ElementCompute = float;

        int max_blocks_per_cu = 0;
        int smem_size_kernel = 0;
        bool kIsNoPadPreprocessLayout = false;
        int DynamicTildId = 0; // for printf, cast from FP4DynamicTileId enum
        int warp_num = 1;

        hggcFuncAttributes attr;
        if constexpr (kEnableMoeDynamicTile) {
            auto launch_dynamic_tile_kernel = [&](auto gemm_kernel_) {
                using GemmKernel = decltype(gemm_kernel_);
                using TileScheduler = typename GemmKernel::TileScheduler;
                kIsNoPadPreprocessLayout = TileScheduler::kIsNoPadPreprocessLayout;
                DynamicTildId = static_cast<int>(GemmKernel::KernelAiuFp4DynamicTile::DynamicTildId);

                using StrideA   = typename GemmKernel::StrideA;
                using StrideB   = typename GemmKernel::StrideB;
                using StrideSFA = typename GemmKernel::StrideSFA;
                using StrideSFB = typename GemmKernel::StrideSFB;
                using StrideC   = typename GemmKernel::StrideC;
                using StrideD   = typename GemmKernel::StrideD;
                using ElementSFA = typename GemmKernel::ElementSFA;
                using ElementSFB = typename GemmKernel::ElementSFB;
                using StrideBias  = Stride<_0,_1,int64_t>;
                StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape((int)shape_m, (int)ShapeK, 1));
                StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape((int)ShapeN, (int)ShapeK, 1));
                StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(shape_m, 0, 1));
                StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape((int)shape_m, (int)ShapeN, 1));
                StrideSFA stride_SFA = cutlass::make_cute_packed_stride(StrideSFA{}, cute::make_shape((int)shape_m, cute::ceil_div(ShapeK, 32), 1));
                StrideSFB stride_SFB = cutlass::make_cute_packed_stride(StrideSFB{}, cute::make_shape((int)ShapeN, cute::ceil_div(ShapeK, 32), 1));

                using Epilogue = typename GemmKernel::CollectiveEpilogue;
                using ProblemShape = Shape<int,int,int,int>;
                auto problem_shape_MNKL = ProblemShape{32, ShapeN, ShapeK, 1};
                StrideBias stride_Bias = {};
                cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(d_ptr);
                typename Epilogue::Arguments arg_epilogue = [&]() -> auto {
                    return typename Epilogue::Arguments{{1.0f, 0.0f}, c_ptr, stride_C, converted_output, stride_D};
                }();

                auto params_epilogue = Epilogue::to_underlying_arguments(problem_shape_MNKL, arg_epilogue, nullptr);

                int* layout_info = grouped_layout;
                if (TileScheduler::kIsNoPadPreprocessLayout) {
                    uint32_t block_size = max(32, next_power_of_two(kNumGroups));
                    computeBlockInfoKernel<TileScheduler::BLOCK_M><<<1, block_size, 0, stream>>>(
                      reinterpret_cast<const uint32_t*>(grouped_layout), kNumGroups, reinterpret_cast<uint32_t*>(block_m_info));
                    layout_info = block_m_info;
                }

                max_blocks_per_cu = compute_occupancy_for_kernel<GemmKernel>();
                dim3 const block = GemmKernel::get_block_shape();
                dim3 const grid(num_sms * max_blocks_per_cu, 1, 1);
                smem_size_kernel = GemmKernel::SharedStorageSize;
                warp_num = block.x / 32; // used for print

                char *pEnv_params = std::getenv("show_log");
                if (pEnv_params && isdigit(*pEnv_params)) {
                    hggcFuncAttributes attr;
                    hggcFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>);

                    printf("[GemmGrouped-FP4-DynamicTile:]\n");
                    printf("group:%d, problem:[%d, %d, %d], expected_m:%d, gemm_type:%s, kIsNoPadPreprocessLayout:%d, dynamic_tile_id:%d\n",
                        kNumGroups, shape_m, ShapeN, ShapeK, expected_m, GemmTypeS[static_cast<int>(kGemmType)], TileScheduler::kIsNoPadPreprocessLayout, DynamicTildId);

                    printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
                        BlockM, BlockN, BlockK, WarpM, WarpN, BlockK, kNumStages);

                    printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d, warp_num:%d\n", num_sms, max_blocks_per_cu, grid.x, warp_num);

                    printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
                }

                typename GemmKernel::Arguments arguments {
                    reinterpret_cast<ElementA const*>(a_ptr), stride_A,
                    reinterpret_cast<ElementB const*>(b_ptr), stride_B,
                    reinterpret_cast<ElementSFA*>(scale_a), stride_SFA,
                    reinterpret_cast<ElementSFB*>(scale_b), stride_SFB,
                    params_epilogue, shape_m, layout_info
                };
                auto params = GemmKernel::to_underlying_arguments(arguments, nullptr);


                DgProfParam dg_prof_params;
                if (ProfilingInterface::Instance().get_op_info()){
                    dg_prof_params.set_params(
                        kGemmType, false, std::string("fp4"), kNumGroups, shape_m, ShapeN, ShapeK, expected_m,
                        grouped_layout, stream
                    );
                }
                ProfilingInterface::Instance().instrument(true, dg_prof_params);
                cutlass::device_kernel<GemmKernel><<<grid, block, smem_size_kernel, stream>>>(params);
                ProfilingInterface::Instance().instrument(false, dg_prof_params);
            };

            using GemmKernelDynamic = cutlass::gemm::kernel::Fp4DeepGemmDynamicTile<
                kGemmType, ElementA, ElementB, ElementC, ElementD, ElementAccumulator, ElementCompute,
                ShapeN, ShapeK, kNumGroups, kDynamicTileId>;
            launch_dynamic_tile_kernel(GemmKernelDynamic{});
        } else {
          // Core kernel configurations
          using ElementAccumulator  = float;                                          // Element type for internal accumulation
          using ElementCompute      = float;                                          // Element type for epilogue computation
          using ElementBias         = float;                                          // Element type for bias addition
          using ElementScalar    = ElementCompute;

          using WarpOnM = Int<BlockM / WarpM>;
          using WarpOnN = Int<BlockN / WarpN>;
          static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
          static constexpr bool TransA = false;
          static constexpr bool TransB = false;

          using DispatchPolicy = cutlass::gemm::MainloopWithScalePPU0015Aiu<kNumStages>;
          using GemmOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementA, TransA, Int<BlockM>, Int<BlockK>, false>;
          using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementB, TransB, Int<BlockN>, Int<BlockK>, true>;

          using TransformA = cute::identity;
          using TransformB = cute::identity;

          // Use Aiu for SFA
          using ElementSFA = uint16_t;
          using LayoutSFA = cutlass::layout::ColumnMajor;
          constexpr bool TransSFA = true;
          constexpr int MinAiuContElemSize = 32 / (sizeof_bits<ElementSFA>::value / 8);      // 16

          constexpr int ScaleGranularityK = 32;
          constexpr int ScaleMsPerTile = BlockM;
          constexpr int ScaleKsPerTile = BlockK / ScaleGranularityK; // BlockK must divideable by 32

          constexpr int SFATileM = TransSFA ? cute::max(ScaleMsPerTile, MinAiuContElemSize) : ScaleMsPerTile;
          constexpr int SFATileK = TransSFA ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);

          constexpr bool swap = true;
          constexpr int StageStride = 0;
          constexpr bool swzl = false;
          using GemmOperandSFA = cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementSFA, TransSFA, Int<SFATileM>, Int<SFATileK>, swap, StageStride, swzl>;

          // Use Aiu for SFB
          using ElementSFB = uint16_t;
          using LayoutSFB = cutlass::layout::RowMajor;
          constexpr bool TransSFB = true;
          constexpr int MinAiuContElemSizeSFB = 32 / (sizeof_bits<ElementSFB>::value / 8);      // 16

          constexpr int ScaleNsPerTile = BlockN;
          constexpr int ScaleKsPerTileSFB = ScaleKsPerTile;

          constexpr int SFBTileN = TransSFB ? cute::max(ScaleNsPerTile, MinAiuContElemSizeSFB) : ScaleNsPerTile;
          constexpr int SFBTileK = TransSFB ? ScaleKsPerTileSFB : cute::max(ScaleKsPerTileSFB, MinAiuContElemSizeSFB);

          using GemmOperandSFB = cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementSFB, TransSFB, Int<SFBTileN>, Int<SFBTileK>, swap, StageStride, swzl>;

          using MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;

          using TiledMma = cute::TiledMMA<
          cute::MMA_Atom<MmaInst>,
          cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>;

          using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMmaScaleFp4<
            DispatchPolicy, Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>,
            ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
            ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
            TiledMma,
            typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
            typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB,
            ElementSFA, cutlass::detail::TagToStrideA_t<LayoutSFA>, typename GemmOperandSFA::GmemTiledCopy, typename GemmOperandSFA::SmemLayoutAtom,
            ElementSFB, cutlass::detail::TagToStrideB_t<LayoutSFB>, typename GemmOperandSFB::GmemTiledCopy, typename GemmOperandSFB::SmemLayoutAtom>;

          using EpilogueOutputOp = typename cutlass::platform::conditional<
              hasBias,
              cutlass::epilogue::thread::LinearCombinationBiasElementwise<ElementD, ElementAccumulator, ElementCompute, ElementD, ElementD, AlignmentD, cutlass::epilogue::thread::Identity<float>, cutlass::plus<ElementCompute>, false, ElementBias>,
              cutlass::epilogue::thread::LinearCombination<ElementD, 2, ElementAccumulator, ElementCompute, cutlass::epilogue::thread::ScaleType::Nothing, cutlass::FloatRoundStyle::round_to_nearest, ElementC>,
            >::type;

          using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<AlignmentC * sizeof(ElementC) * 8>;
          using GemmEpilogueConfiguration = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, float, AlignmentC, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;
          static constexpr bool IsAligedN = ShapeN % BlockN == 0 ? true : false;

          using CollectiveEpilogueWithTsm = typename cutlass::epilogue::collective::Epilogue<
            cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
            cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
            EpilogueOutputOp,
            typename GemmEpilogueConfiguration::SmemLayoutO,
            Copy_Atom<EpilogueCopyInst,float>,
            typename GemmEpilogueConfiguration::GmemTiledCopyO,
            Copy_Atom<EpilogueCopyInst,ElementC>
          >;

          using CollectiveEpilogueNoTsm = typename cutlass::epilogue::collective::DefaultEpilogueNoTsm<
            cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
            cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
            EpilogueOutputOp,
            cutlass::gemm::EpilogueDefault,
            IsAligedN
          >;

          // CollectiveEpilogueNoTsm requires that N is divideable by 2
          using CollectiveEpilogue = typename cutlass::platform::conditional<
            hasBias || (ShapeN % 2 != 0),
            CollectiveEpilogueWithTsm,
            CollectiveEpilogueNoTsm
          >::type;

          using TileScheduler = DeepGemmScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN * N_EXPAND, kNumGroups>;
          using GemmKernel = typename deep_gemm::DeepGemmUniversal<
              Shape<int,int,int,int>,
              CollectiveMainloop,
              CollectiveEpilogue,
              TileScheduler,
              hasBias,
              N_EXPAND,
              kEnableSboOverlap,
          >;

          using StrideA = typename GemmKernel::StrideA;
          using StrideB = typename GemmKernel::StrideB;
          using StrideC = typename GemmKernel::StrideC;
          using StrideD = typename GemmKernel::StrideD;
          using StrideBias  = Stride<_0,_1,int64_t>;
          using StrideSFA = typename GemmKernel::StrideSFA;
          using StrideSFB = typename GemmKernel::StrideSFB;

          int* layout_info = grouped_layout;
          // compute block_m_info
          if (TileScheduler::kIsNoPadPreprocessLayout) {
              uint32_t block_size = max(32, next_power_of_two(kNumGroups));
              computeBlockInfoKernel<BlockM><<<1, block_size, 0, stream>>>(reinterpret_cast<const uint32_t*>(grouped_layout), kNumGroups, reinterpret_cast<uint32_t*>(block_m_info));
              layout_info = block_m_info;
          }

          StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(shape_m, ShapeK, 1));
          StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(ShapeN, ShapeK, 1));
          // SFA is M-Major in uint16_t
          StrideSFA stride_meta_A = cutlass::make_cute_packed_stride(StrideSFA{}, cute::make_shape(shape_m, cute::ceil_div(ShapeK, 32), 1));
          StrideSFB stride_meta_B = cutlass::make_cute_packed_stride(StrideSFB{}, cute::make_shape(ShapeN, cute::ceil_div(ShapeK, 32), 1));
          StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(shape_m, 0, 1));
          StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(shape_m, ShapeN, 1));
          StrideBias stride_Bias = {};

          int max_blocks_per_cu = compute_occupancy_for_kernel<GemmKernel>();
          cutlass::KernelHardwareInfo hw_info;
          hw_info.device_id = 0;
          hw_info.cu_count = KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id) * max_blocks_per_cu;
          cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(d_ptr);

          typename CollectiveEpilogue::Arguments epilogue_arguments = [&]() -> auto {
              if constexpr (hasBias) {
                return typename CollectiveEpilogueWithTsm::Arguments{{1.0, 0.0, nullptr, nullptr, c_ptr, stride_Bias}, nullptr, stride_C, converted_output, stride_D};
              } else {
                return typename CollectiveEpilogueNoTsm::Arguments{{1.0f, 0.0f}, c_ptr, stride_C, converted_output, stride_D};
              }
          }();

          typename GemmKernel::Arguments arguments{
              cutlass::gemm::GemmUniversalMode::kGemm,
              {shape_m, ShapeN, ShapeK, 1},
              {
                {shape_m, ShapeN, ShapeK},
                reinterpret_cast<ElementA const*>(a_ptr), stride_A,
                reinterpret_cast<ElementB const*>(b_ptr), stride_B,
                reinterpret_cast<ElementSFA*>(scale_a), stride_meta_A,
                reinterpret_cast<ElementSFB*>(scale_b), stride_meta_B
              },
              epilogue_arguments,
              hw_info, {}, signal
          };

          size_t workspace_size = GemmKernel::get_workspace_size(arguments);
          cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
          typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(arguments, workspace.get(), layout_info);
          dim3 const block = GemmKernel::get_block_shape();
          dim3 const grid = GemmKernel::get_grid_shape(params);
          int sharemem_size = GemmKernel::SharedStorageSize;

          char *pEnv_params = std::getenv("show_log");
          if (pEnv_params && isdigit(*pEnv_params)) {
              hggcFuncAttributes attr;
              hggcFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>);

              printf("[GemmGrouped-FP4:]\n");
              printf("group:%d, problem:[%d, %d, %d], expected_m:%d, gemm_type:%s, kIsNoPadPreprocessLayout:%d\n",
                  kNumGroups, shape_m, ShapeN, ShapeK, expected_m, GemmTypeS[static_cast<int>(kGemmType)], TileScheduler::kIsNoPadPreprocessLayout);

              printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
                  BlockM, BlockN, BlockK, WarpM, WarpN, BlockK, kNumStages);

              printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms, max_blocks_per_cu, grid.x);

              printf("smem_size:%d, vreg:%d, stack:%d\n", sharemem_size, int(attr.numRegs), int(attr.localSizeBytes));
          }

          DgProfParam dg_prof_params;
          if (ProfilingInterface::Instance().get_op_info()) {
              dg_prof_params.set_params(
                  kGemmType, false, std::string("fp4"), kNumGroups, shape_m, ShapeN, ShapeK, expected_m,
                  grouped_layout, stream
              );
          }

          ProfilingInterface::Instance().instrument(true, dg_prof_params);
          cutlass::device_kernel<GemmKernel><<<grid, block, sharemem_size, stream>>>(params);
          ProfilingInterface::Instance().instrument(false, dg_prof_params);
      }
    }
  };

};  // namespace deep_gemm
