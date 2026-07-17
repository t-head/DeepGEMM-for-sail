#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

// #include <iostream>
#include <hggc_fp8.h>
#ifndef FP8_HGRTC
    #include "profiling_interface.hpp"
    #include "tools/util/include/cutlass/util/host_tensor.h"
#endif
// #include "utils.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cute/ppu_util.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/config/gemm_configs.hpp"
#include "cutlass/epilogue/fusion/ppu_callbacks.hpp"
#include "tools/util/include/cutlass/util/packed_stride.hpp"
#include "scheduler_cutlass3.cuh"
#include "utils_cutlass3.h"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage_with_scale.hpp"
namespace deep_gemm {
using namespace cute;
using cutlass::KernelHardwareInfo;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_,
  bool kEnableSboOverlap = false,
  bool kUseNStageKernel = false
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
  false
> {
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
  using ElementScale  = typename CollectiveMainloop::ElementScale;
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

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

  using LayoutSFA  =  typename CollectiveMainloop::LayoutSFA;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
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
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
    int32_t* signal{nullptr};
  };


  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace, int* grouped_layout) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");
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

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    void* scheduler_workspace = workspace_ptr;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = cutlass::round_nearest(workspace_offset,  cutlass::MinWorkspaceAlignment);
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = cutlass::round_nearest(workspace_offset,  cutlass::MinWorkspaceAlignment);
    void* mainloop_workspace = nullptr;
    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
    // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
    // subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = 1; //CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(grouped_layout,
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);
    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      scheduler,
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
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hggcStream_t stream = nullptr,
    cutlass::HostAdapter* host_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  // Computes the kernel launch grid shape based on runtime parameters
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
    if (TileScheduler::GEMM_TYPE == GemmType::GroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    } else if (TileScheduler::GEMM_TYPE == GemmType::GroupedNoPad) {
      // each warp prefetch one cacheline
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx<<5)));
      }
    }

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
      // update actual global ptr offset
      MainloopParams update_params = {
        ptr_A, params.mainloop.dA, ptr_B, params.mainloop.dB, 4,
        ptr_scale_A, layout_SFA,
        ptr_scale_B, params.mainloop.layout_SFB
      };
      problem_shape_MNKL = ProblemShape{M, N, K, L};
      auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);
      static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

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
      collective_mainloop(
        update_params,
        load_inputs,
        accumulators,
        k_tile_iter, k_tile_count,
        thread_idx,
        smem_buf
      );
      // update params.epilogue for ptrC and ptrD
      auto params_epilogue_local = params.epilogue;
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
  true
> {
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
  using ElementScale  = typename CollectiveMainloop::ElementScale;
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

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

  using LayoutSFA  =  typename CollectiveMainloop::LayoutSFA;
  using LayoutSFB  =  typename CollectiveMainloop::LayoutSFB;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));

  static constexpr int ScaleGranularityM = CollectiveMainloop::ScaleGranularityM;
  static constexpr int ScaleGranularityN = CollectiveMainloop::ScaleGranularityN;
  static constexpr int ScaleGranularityK = CollectiveMainloop::ScaleGranularityK;
  static constexpr int BlockM = size<0>(TileShape{});
  static constexpr int BlockN = size<1>(TileShape{});
  static constexpr int BlockK = size<2>(TileShape{});

  static constexpr int ScaleMsPerTile = CollectiveMainloop::ScaleMsPerTile;
  static constexpr int ScaleNsPerTile = CollectiveMainloop::ScaleNsPerTile;
  static constexpr int ScaleKsPerTile = CollectiveMainloop::ScaleKsPerTile;

  static constexpr uint32_t MinBlocksPerMultiprocessor = ((BlockM == 64) && (BlockN == 128) && (BlockK == 128)) ? 4 : 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr uint32_t N_PREFETCH_CACHELINE = cute::ceil_div(TileScheduler::kNumGroups, 32); // numGroups * sizeof(int) / 128 Byte = cacheline
  static constexpr int N_EXPAND = KernelAiuMultistageOnN::N_EXPAND;
  static constexpr int Stages = TileScheduler::SHAPE_K < BlockK * DispatchPolicy::Stages ? TileScheduler::SHAPE_K / BlockK : DispatchPolicy::Stages;

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
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
    int32_t* signal{nullptr};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace, int* grouped_layout) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

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

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* scheduler_workspace = workspace_ptr;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = cutlass::round_nearest(workspace_offset,  cutlass::MinWorkspaceAlignment);

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = cutlass::round_nearest(workspace_offset,  cutlass::MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
    // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
    // subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = 1; //CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(grouped_layout,
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      scheduler,
      workspace,
      args.signal
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hggcStream_t stream = nullptr,
    cutlass::HostAdapter* host_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  // Computes the kernel launch grid shape based on runtime parameters
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
    if (TileScheduler::GEMM_TYPE == GemmType::GroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    } else if (TileScheduler::GEMM_TYPE == GemmType::GroupedNoPad) {
      // each warp prefetch one cacheline
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx<<5)));
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
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)
    auto strides = make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, get<0>(params.problem_shape)));

    uint32_t m_block_idx, n_block_idx;
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx * N_EXPAND;
      auto l_coord = 0;
      auto M = deep_scheduler.curr_problem_m();
      auto problem_shape_MNKL = ProblemShape{M, N, K, L};

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
      // update actual global ptr offset
      MainloopParams update_params = {
        ptr_A, params.mainloop.dA, ptr_B, params.mainloop.dB, 4,
        ptr_scale_A, layout_SFA,
        ptr_scale_B, params.mainloop.layout_SFB
      };

      auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);

      static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

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

      int k_tile_iter  = 0;
      int k_tile_count = size<2>(gA);

      using FrgTensorC = decltype(accumulators);
      FrgTensorC& accum = accumulators;

      using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
      using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
      using GmemTiledCopyA = typename CollectiveMainloop::GmemTiledCopyA;
      using GmemTiledCopyB = typename CollectiveMainloop::GmemTiledCopyB;
      using GmemTiledCopySFA = typename CollectiveMainloop::GmemTiledCopySFA;
      using SmemCopyAtomA = typename CollectiveMainloop::SmemCopyAtomA;
      using SmemCopyAtomB = typename CollectiveMainloop::SmemCopyAtomB;
      using TransformA = typename CollectiveMainloop::TransformA;
      using TransformB = typename CollectiveMainloop::TransformB;

      using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;

      GmemTiledCopyA& gmem_tiled_copy_A = collective_mainloop.gmem_tiled_copy_A;
      GmemTiledCopyB& gmem_tiled_copy_B = collective_mainloop.gmem_tiled_copy_B;
      GmemTiledCopySFA& gmem_tiled_copy_SFA = collective_mainloop.gmem_tiled_copy_SFA;

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

      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
      Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,_0{}));                     // (MMA,MMA_M,MMA_K)
      Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,_0{}));                     // (MMA,MMA_N,MMA_K)

      CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
      CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
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

      using SmemLayoutSFA = typename CollectiveMainloop::SmemLayoutSFA;
      using RealSmemLayoutSFA = typename CollectiveMainloop::RealSmemLayoutSFA;
      using CopyAtomSFA = typename CollectiveMainloop::CopyAtomSFA;   // use async copy
      auto SFA_shape = shape(layout_SFA);

      constexpr int IsAiuLoadSFA = CollectiveMainloop::IsAiuLoadSFA;
      Tensor gSFA = get<2, 0>(load_inputs);
      auto sSFA = make_tensor(cute::make_smem_ptr(storage.smem_SFA.data()), RealSmemLayoutSFA{});

      auto SFA_load_tuple = [&]() {
        if constexpr (IsAiuLoadSFA) {
          auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_slice(thread_idx);
          Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
          Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);
          return cute::make_tuple(0, tSFAgSFA, tSFAsSFA, 0, 0);
        } else {
          Tensor cSFA = get<2, 1>(load_inputs);
          TiledCopy scale_copy_a = make_tiled_copy(CopyAtomSFA{}, Layout<Shape<Int<ScaleMsPerTile>>>{}, Layout<Shape<_1>>{});
          ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(thread_idx - 64);   // use warp 2 - warp n
          Tensor tSFAgSFA = thr_scale_copy_a.partition_S(gSFA);
          Tensor tSFAsSFA = thr_scale_copy_a.partition_D(sSFA);
          Tensor tSFAcSFA = thr_scale_copy_a.partition_S(cSFA);
          Tensor tSFApSFA = make_tensor<bool>(shape(filter_zeros(tSFAsSFA(_,_,_,_0{}))));
          return cute::make_tuple(scale_copy_a, tSFAgSFA, tSFAsSFA, tSFApSFA, tSFAcSFA);
        }
      }();
      auto scale_copy_a = get<0>(SFA_load_tuple);
      auto tSFAgSFA = get<1>(SFA_load_tuple);
      auto tSFAsSFA = get<2>(SFA_load_tuple);
      auto tSFApSFA = get<3>(SFA_load_tuple);
      auto tSFAcSFA = get<4>(SFA_load_tuple);

      Tensor mSFB_nkl = make_tensor(make_gmem_ptr(update_params.ptr_scale_B), update_params.layout_SFB);          // (scale_n,k,l)
      Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N,BLK_K,n,k,l)
      Tensor gSFB = gSFB_nkl(_,_,n_coord,_,l_coord);

      constexpr int K_TILE_COUNT = (K + BlockK - 1) / BlockK;
      auto k_tile_iter_reset = k_tile_iter;

      constexpr int warp_sfa = cute::ceil_div(Int<ScaleMsPerTile>{}, _32{});
      auto copy_to_tsm = [&](int k_pipe_write, int k_idx, int warp_idx) {
        if (warp_idx == 0) {
          copy(gmem_tiled_copy_A, tAgA(_,_,_,k_idx), tAsA(_,_,_,k_pipe_write));
        } else if (warp_idx == 1) {
          copy(gmem_tiled_copy_B, tBgB(_,_,_,k_idx), tBsB(_,_,_,k_pipe_write));
        }
        if constexpr (IsAiuLoadSFA) {
          if (warp_idx == 2) {
            copy(gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,k_idx), tSFAsSFA(_,_,_,k_pipe_write));
          }
        } else {
          Tensor tSFAcSFA_compact = filter_zeros(tSFAcSFA(_,_,_,k_idx));
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tSFApSFA); ++i) {
            tSFApSFA(i) = elem_less(get<0>(tSFAcSFA_compact(i)), get<0>(SFA_shape));
          }
          if (warp_idx > 1 && warp_idx <= 1 + warp_sfa) {
            copy_if(scale_copy_a, tSFApSFA, filter_zeros(tSFAgSFA(_,_,_,k_idx)), filter_zeros(tSFAsSFA(_,_,_,k_pipe_write)));
          }
        }
      };

      // Prologue, Start async loads for all pipes but the last
      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < Stages; ++k_pipe) {
        if (k_tile_count > 0) {
          copy_to_tsm(k_pipe, k_tile_iter, warp_idx);
          ++k_tile_iter;
        }
        cp_async_fence();
        --k_tile_count;
      }
      auto k_tile_count_reset = k_tile_count;

    // Block scaling layout
    auto layout_sSFA_copy = [&]() {
      if constexpr(IsAiuLoadSFA) {
        return make_layout(
          make_shape(
            make_shape(_1{}, shape<0>(RealSmemLayoutSFA{})),
            get<1>(TileShape{}),
            make_shape(
              shape<1>(SmemLayoutSFA{}),
              shape<2>(SmemLayoutSFA{}))
          ),
          make_stride(
            make_stride(_0{}, stride<0>(RealSmemLayoutSFA{})),
            _0{},
            make_stride(
              make_stride(_0{}, stride<1>(RealSmemLayoutSFA{})),
              stride<2>(RealSmemLayoutSFA{}))
          )
        );
      } else {
        return make_layout(
          make_shape(
            shape<0>(SmemLayoutSFA{}),
            get<1>(TileShape{}),
            make_shape(
              shape<1>(SmemLayoutSFA{}),
              shape<2>(SmemLayoutSFA{}))
          ),
          make_stride(
            stride<0>(SmemLayoutSFA{}),
            _0{},
            make_stride(
              stride<1>(SmemLayoutSFA{}),
              stride<2>(SmemLayoutSFA{}))
          )
        );
      }
    }();

      auto layout_gSFB_copy = make_layout(
        make_shape(get<0>(TileShape{}), shape<0>(gSFB), make_shape(shape<1>(gSFB), shape<2>(gSFB))),
        make_stride(_0{}, stride<0>(gSFB), make_stride(stride<1>(gSFB), stride<2>(gSFB)))
      );

      Tensor sSFA_copy = make_tensor(cute::make_smem_ptr(storage.smem_SFA.data()), layout_sSFA_copy);
      Tensor gSFB_copy = make_tensor(cute::make_gmem_ptr(gSFB.data()), layout_gSFB_copy);

      Tensor tCsSFA = tiled_mma.get_slice(thread_idx).partition_C(sSFA_copy);
      Tensor tCgSFB = tiled_mma.get_slice(thread_idx).partition_C(gSFB_copy);

      Tensor tCrSFA = make_fragment_like<ElementScale>(tCsSFA(_, _, _, _0{}));
      Tensor tCrSFB = make_fragment_like<ElementScale>(tCgSFB(_, _, _, _0{}));

      __ppu_prefetch_KSD((void*)(raw_pointer_cast(tCgSFB.data())));

      using EngineAccum =  typename FrgTensorC::engine_type;
      using LayoutAccum =  typename FrgTensorC::layout_type;
      cutlass::gemm::collective::MixAccumulation<EngineAccum, LayoutAccum, ElementAccumulator> accumulation(
        accum, CollectiveMainloop::ScalePromotionInterval, size<2>(tCrA));
      // Current pipe index in smem to read from
      int smem_pipe_read  = 0;
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
        cp_async_wait<Stages-1>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
        copy(tCsSFA(_,_,_,make_coord(_0{}, _0{})), tCrSFA);
      }
      for (int n_iter = 0; n_iter < N_EXPAND; n_iter++) {
        clear(accumulators);
        auto blk_coord_mnkl = make_coord(m_coord, n_coord + n_iter, _, l_coord);
        Tensor gSFB = gSFB_nkl(_,_,n_coord + n_iter,_,l_coord);
        Tensor gSFB_copy = make_tensor(cute::make_gmem_ptr(gSFB.data()), layout_gSFB_copy);
        Tensor tCgSFB = tiled_mma.get_slice(thread_idx).partition_C(gSFB_copy);
        int k_iter = 0;
        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > -(Stages)) {
          copy(tCgSFB(_,_,_,k_iter * BlockK), tCrSFB);
          ++k_iter;
          clear(accumulation());
          if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
            tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
          }
          if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
            ElementScale scale_b = tCrSFB(_0{});
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
              filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
            }
          }
          if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
            ElementScale scale_a = tCrSFA(_0{});
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
              filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
            }
          }
          // Pipeline the outer products with a static for loop.
          //
          // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
          for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
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
              cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), accumulation());
            }

            if (k_block == K_BLOCK_MAX - 2) {
              // Commit the smem for smem_pipe_read
              if constexpr (Stages > 1) {
                cp_async_wait<Stages-2>();
              } else {
                cp_async_wait<Stages-1>();
              }
              __syncthreads();
              if (k_tile_count > 0 || n_iter < N_EXPAND - 1) {
                if (k_tile_iter >= K_TILE_COUNT) {
                  if (n_iter < N_EXPAND - 1) {
                    // load for next n_iter, avoid invalid page
                    tBgB.data() = tBgB.data() + K * BlockN;
                  }
                  k_tile_iter = k_tile_iter_reset;
                }
                copy_to_tsm(smem_pipe_write, k_tile_iter, warp_idx);
              }
              cp_async_fence();
              --k_tile_count;
              ++k_tile_iter;
              // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
              ++smem_pipe_read;
              smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
              smem_pipe_write = smem_pipe_read;
            }
          }); // for_each
          // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
          if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
            ElementScale scale_ab = tCrSFA(_0{});
            accumulation.scale(scale_ab);
          }
          if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
            accumulation.scale(tCrSFA);
          }
          if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
            accumulation.scale(tCrSFB);
          }
          if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
            accumulation.scale(tCrSFA, tCrSFB);
          }
          copy(tCsSFA(_,_,_,make_coord(_0{}, smem_pipe_read)), tCrSFA);
        } // while k_tile_count
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
        if (n_iter < N_EXPAND - 1) {
          k_tile_count = k_tile_count_reset;
          cp_async_wait<Stages-1>();
          __syncthreads();
          // Prefetch the first rmem from the first k-tile
          copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
          copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
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

template <int32_t SHAPE_N, int32_t SHAPE_K,
          int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K,
          int32_t WARP_M, int32_t WARP_N,
          int32_t BLOCK_N_PADDING,
          int32_t kSwizzleDMode,
          int32_t kNumGroups, int32_t kNumStages,
          GemmType kGemmType,
          bool kEnableSboOverlap = false>
class Fp8Gemm {

public:
    Fp8Gemm() = default;

    static constexpr bool UseAIU = true;
    // A matrix configuration
    using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
    using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    static constexpr int AlignmentA  = UseAIU ? 1 : 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
    using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    static constexpr int AlignmentB  = UseAIU ? 1 : 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // D matrix configuration
    using         ElementD    = cutlass::bfloat16_t;
    using         LayoutD     = cutlass::layout::RowMajor;
    static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

    // C matrix configuration
    using         ElementC    = ElementD;
    using         LayoutC     = LayoutD;
    static constexpr int AlignmentC  = AlignmentD;

    using ArchTag = cutlass::arch::PPU0015;

    // Core kernel configurations
    using ElementAccumulator  = float;                                          // Element type for internal accumulation
    using ElementCompute      = float;                                          // Element type for epilogue computation
    using ElementScalar    = ElementCompute;

    static constexpr int Stage = kNumStages;
    using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
    using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;

    using ScaleGranularityShape = Shape<_1,_128,_128>;
    using ScaleConfig         = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(ScaleGranularityShape{}));
    using LayoutSFA           = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
    using LayoutSFB           = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand

    static constexpr bool kUseNStageKernel = SHAPE_K <= 512 && (SHAPE_N % (BLOCK_N * KernelAiuMultistageOnN::N_EXPAND) == 0)
                                              && (BLOCK_K == 128) && (SHAPE_K % BLOCK_K == 0) && kNumStages == 2;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
      ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
      ElementAccumulator,
      TileShape, WarpShape,
      Int<Stage>,
      cutlass::gemm::KernelAiuMultistageWithBlockWiseScale
    >::CollectiveOp;

    using EpilogueDispatchPolicy = cutlass::epilogue::EpilogueSimtVectorized;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    using CollectiveEpilogueWithTsm = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp,
        TileShape, WarpShape,
        EpilogueTileType,
        ElementCompute, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueDispatchPolicy
      >::CollectiveOp;

    // Epilogue
    static constexpr bool IsAligedN = SHAPE_N % BLOCK_N == 0 ? true : false;
    // reduce vreg to use ScaleType::Nothing for alpha=1 & beta=0
    using CollectiveEpilogueNoTsm = typename cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        cutlass::detail::TagToStrideA_t<LayoutC>,
        cutlass::detail::TagToStrideA_t<LayoutC>,
        cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
        cutlass::gemm::EpilogueDefault,
        IsAligedN>;
    static constexpr bool EpilogueWithTsm = false;
    using CollectiveEpilogue = typename cutlass::platform::conditional<
        EpilogueWithTsm,
        CollectiveEpilogueWithTsm,
        CollectiveEpilogueNoTsm
    >::type;

    static void run(__ppu_bfloat16* gmem_d,
                    __hg_fp8_e4m3* input_a,
                    __hg_fp8_e4m3* input_b,
                    float* scales_a,
                    float* scales_b,
                    int* grouped_layout, int* block_m_info,
                    int32_t shape_m, uint32_t expected_m,
                    hggcStream_t stream,
                    int num_sms, uint32_t smem_size, int32_t* signal = nullptr) {
        constexpr int N_EXPAND = kUseNStageKernel ? KernelAiuMultistageOnN::N_EXPAND : 1;
        using TileScheduler = DeepGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, kNumGroups>;
        
        using GemmKernel = typename deep_gemm::DeepGemmUniversal<
          Shape<int,int,int,int>,
          CollectiveMainloop,
          CollectiveEpilogue,
          TileScheduler,
          kEnableSboOverlap,
          kUseNStageKernel
        >;

        using StrideA = typename GemmKernel::StrideA;
        using StrideB = typename GemmKernel::StrideB;
        using StrideC = typename GemmKernel::StrideC;
        using StrideD = typename GemmKernel::StrideD;

        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(shape_m, SHAPE_K, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(SHAPE_N, SHAPE_K, 1));
        StrideD stride_D;
        if constexpr (kGemmType == GemmType::BatchGemm) {
          // BHD output: D[b,h,d] has offset b*H*D + h*D + d
          // In the kernel's (batch=h, M=b, N=d) model:
          //   row stride = H*D = kNumGroups * SHAPE_N
          //   col stride = 1
          //   batch stride = D = SHAPE_N
          stride_D = StrideD{int64_t(kNumGroups) * SHAPE_N, cute::Int<1>{}, int64_t(SHAPE_N)};
        } else {
          stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(shape_m, SHAPE_N, 1));
        }
        LayoutSFA layout_SFA;
        LayoutSFB layout_SFB;
        auto ScaleGranularityN = size<1>(ScaleGranularityShape{});
        auto ScaleGranularityK = size<2>(ScaleGranularityShape{});
        auto scale_k = (SHAPE_K + ScaleGranularityK - 1) / ScaleGranularityK;
        auto scale_n = (SHAPE_N + ScaleGranularityN - 1) / ScaleGranularityN;
        layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(shape_m, SHAPE_N, SHAPE_K, 1));
        layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(shape_m, SHAPE_N, SHAPE_K, 1));

        int* layout_info = grouped_layout;
        // compute block_m_info
        if (TileScheduler::kIsNoPadPreprocessLayout) {
            uint32_t block_size = max(32, next_power_of_two(kNumGroups));
            computeBlockInfoKernel<BLOCK_M><<<1, block_size, 0, stream>>>(reinterpret_cast<const uint32_t*>(grouped_layout), kNumGroups, reinterpret_cast<uint32_t*>(block_m_info));
            layout_info = block_m_info;
        }

        cutlass::float_e4m3_t* converted_input_b = reinterpret_cast<cutlass::float_e4m3_t*>(input_b);
        cutlass::float_e4m3_t* converted_input_a = reinterpret_cast<cutlass::float_e4m3_t*>(input_a);
        cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(gmem_d);
        int max_blocks_per_cu = compute_occupancy_for_kernel<GemmKernel>();
        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.cu_count = num_sms * max_blocks_per_cu;
        typename GemmKernel::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {shape_m, SHAPE_N, SHAPE_K, 1},
          {converted_input_a, stride_A, converted_input_b, stride_B, 4,
           scales_a, layout_SFA, scales_b, layout_SFB},
          {
            {1, 0},
            nullptr, stride_D,
            converted_output, stride_D
          },
          hw_info, {}, signal
        };
        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size = GemmKernel::get_workspace_size(arguments);

        // Allocate workspace memory
        // // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        // cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

        // evt realization must construct evt params on device, can't use GemmUniversalAdapter
        typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(arguments, nullptr, layout_info);

        dim3 const block = GemmKernel::get_block_shape();
        dim3 const grid = GemmKernel::get_grid_shape(params);
        int sharemem_size = GemmKernel::SharedStorageSize;
        // std::cout << "block = " << block << std::endl;
        // std::cout << "grid = " << grid << std::endl;
        // std::cout << "smem_size_kernel = " << sharemem_size << std::endl;

        int max_active_tb_num = max_blocks_per_cu;
        const int threadblock_count = num_sms * max_active_tb_num;

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>);

            printf("[GemmGrouped-FP8:]\n");
            printf("group:%d, problem:[%d, %d, %d], expected_m:%d, gemm_type:%s, kIsNoPadPreprocessLayout:%d, kUseNStageKernel:%d\n",
                kNumGroups, shape_m, SHAPE_N, SHAPE_K, expected_m, GemmTypeS[static_cast<int>(kGemmType)], TileScheduler::kIsNoPadPreprocessLayout, kUseNStageKernel);

            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
                BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K, kNumStages);

            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms, max_active_tb_num, threadblock_count);

            printf("smem_size:%d, vreg:%d, stack:%d\n", sharemem_size, int(attr.numRegs), int(attr.localSizeBytes));
        }

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()){
            dg_prof_params.set_params(
                kGemmType, false, std::string("fp8"), kNumGroups, shape_m, SHAPE_N, SHAPE_K, expected_m,
                grouped_layout, stream
            );
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        cutlass::device_kernel<GemmKernel><<<grid, block, sharemem_size, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);

  }
};

};  // namespace deep_gemm
#pragma clang diagnostic pop
