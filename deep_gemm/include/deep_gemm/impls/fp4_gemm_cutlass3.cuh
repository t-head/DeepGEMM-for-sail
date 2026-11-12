#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#include <cutlass/cutlass.h>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/epilogue/epilogue_traits.cuh>

#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/workspace.h>
#include <cutlass/fast_math.h>
#include <cutlass/epilogue/collective/detail.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/tensor_predicate.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/tensor.hpp>
#include <cute/ppu_util.hpp>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/gemm/gemm.h>
#include <ppu_include.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/trace.h>
#include <cutlass/gemm/config/gemm_configs.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cutlass/numeric_conversion.h>
#include <tools/util/include/cutlass/util/packed_stride.hpp>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>
#include <deep_gemm/impls/fp4_gemm_cutlass3_dynamic.cuh>
#include <deep_gemm/mma/fp4_mma.cuh>
#include <deep_gemm/epilogue/fp4_epilogue_silu_and_mul_post_quant.cuh>
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
  bool kEnableSboOverlap = false,
  EpilogueType kEpilogueType = EpilogueType::Default
>
class DeepGemmUniversal;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_,
  bool hasBias_,
  int nExpand,
  EpilogueType kEpilogueType
>
class DeepGemmUniversal <
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  hasBias_,
  nExpand,
  false,
  kEpilogueType
> {
  public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  static_assert(kEpilogueType == EpilogueType::Default, "NExpand mode only supports EpilogueType::Default because TSM overlap in the Epilogue");

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
    if constexpr (TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    }
    if constexpr (TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad) {
      // each warp prefetch one cacheline
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx << 5)));
      }
    }

    int lane_idx = threadIdx.x % 32;
    int aiu_warp_group_thread_idx = warp_idx * 32;
    int warp_m_idx = warp_idx % warp_on_m;
    int warp_n_idx = warp_idx / warp_on_m;

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
  bool hasBias_,
  EpilogueType kEpilogueType
>
class DeepGemmUniversal <
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  hasBias_,
  1,
  false,
  kEpilogueType
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
      TiledMma tiled_mma;
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
      auto residue_mnk = CollectiveMainloop::compute_tile(
          update_params, problem_shape_MNKL, blk_coord_mnkl, accumulators, thread_idx, smem_buf);

      // Update the pointer to C and D of MXFP4 in the epilogue parameters
      auto params_epilogue_local = params.epilogue;
      params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c();
      if constexpr (hasBias_) {
        params_epilogue_local.ptr_Bias += deep_scheduler.curr_offset_mxfp4_c();
      }
      if constexpr (EpilogueTraits<kEpilogueType>::needs_sfd_offset) {
        params_epilogue_local.ptr_SFD += deep_scheduler.curr_offset_c_scales();
      }
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

#include <deep_gemm/impls/fp4_gemm_cutlass3_overlap_prologue.cuh>
