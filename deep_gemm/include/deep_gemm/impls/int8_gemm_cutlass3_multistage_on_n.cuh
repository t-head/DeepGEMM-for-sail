#pragma once

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
  cute::enable_if_t<cute::is_base_of_v<KernelAiuMultistageOnN, typename CollectiveMainloop_::DispatchPolicy::Schedule>>, false> {
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
  static constexpr int N_EXPAND = KernelAiuMultistageOnN::N_EXPAND;

  constexpr static uint32_t CTA_M = shape<0>(TileShape{});
  static_assert(cute::is_same_v<ArchTag, cutlass::arch::PPU0015>, "MultistageOnN is PPU1.5 only");
  static_assert(TileScheduler::BLOCK_N == CollectiveMainloop::CTA_N * N_EXPAND);

  static constexpr int KTileCount = K / CollectiveMainloop::CTA_K;
  using SmemLayoutA = decltype(tile_to_shape(
      typename CollectiveMainloop::SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<KTileCount>{})));
  using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
  using ElementScale = typename CollectiveMainloop::ElementScale;

  struct SharedStorage {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    ElementScale channel_scales[CTA_M + 2 * CollectiveMainloop::CTA_N];
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize == KernelAiuMultistageOnN::smem_size(
      K, CTA_M, CollectiveMainloop::CTA_N, CollectiveMainloop::CTA_K, DispatchPolicy::Stages),
      "The host TSM size must match SharedStorage");

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

  template<int BarrierId = 2>
  CUTLASS_DEVICE
  static void pipeline_barrier() {
    // Complete the warp's TSM accesses before handing off the slot.
    // Stores to D can stay in flight while the CTA advances.
    __ppu_wait_tsmcnt(0);
    __ppu_barrier_sync(BarrierId, MaxThreadsPerBlock, 0);
  }

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

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.cu_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, nullptr),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
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
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx < N_PREFETCH_CACHELINE)
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx << 5)));

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    constexpr int BlockM = get<0>(TileShape{})();
    constexpr int BlockN = get<1>(TileShape{})();
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)

    TileScheduler deep_scheduler(params.scheduler);

    uint32_t m_block_idx, n_block_idx;
    constexpr uint32_t L = 1;
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx * N_EXPAND;
      constexpr int tail_tiles = (N / BlockN) % N_EXPAND;
      int panel_tiles = N_EXPAND;
      if constexpr (tail_tiles != 0) {
        if (n_block_idx == (N / BlockN) / N_EXPAND)
          panel_tiles = tail_tiles;
      }

      uint32_t M = deep_scheduler.curr_problem_m();

      auto problem_shape_MNKL = ProblemShape{M, N, K, L};

      const uint32_t m_max_coord = M - BlockM * m_coord;
      TiledMma tiled_mma;
      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
      Tensor mma_acc = make_fragment_like<ElementAccumulator>(
          partition_fragment_C(tiled_mma, take<0,2>(blk_shape)));

      CollectiveMainloop collective_mma(params.mainloop, take<0,3>(problem_shape_MNKL));
      using SmemLayoutScaleA = typename CollectiveMainloop::SmemLayoutScaleA;
      using SmemLayoutScaleB = decltype(tile_to_shape(
          typename CollectiveMainloop::SmemLayoutAtomScale{}, Shape<Int<CollectiveMainloop::CTA_N>, _1, _2>{}));
      using GmemTiledCopyScaleB = typename CollectiveMainloop::GmemTiledCopyScaleB;
      using SmemCopyAtomScale = Copy_Atom<
          cute::AutoVectorizingCopyWithAssumedAlignment<cutlass::sizeof_bits<ElementScale>::value>, ElementScale>;
      constexpr int CTA_N = CollectiveMainloop::CTA_N;

      static_assert(cute::is_same_v<typename CollectiveMainloop::TransformA, cute::identity> &&
                    cute::is_same_v<typename CollectiveMainloop::TransformB, cute::identity>);

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, 0);
      auto const& mainloop_params = params.mainloop;
      const auto offset_m = deep_scheduler.curr_offset_m();
      const auto offset_a = deep_scheduler.curr_offset_a();
      const auto offset_b = deep_scheduler.curr_offset_b();
      const ElementA* ptr_A = mainloop_params.ptr_A + offset_a;
      const ElementB* ptr_B = reinterpret_cast<const ElementB*>(
          reinterpret_cast<const uint8_t*>(mainloop_params.ptr_B) +
          offset_b * cutlass::sizeof_bits<ElementB>::value / 8);
      auto load_inputs = collective_mma.load_init(
          problem_shape_MNKL, blk_coord_mnkl, mainloop_params, M, offset_m, offset_b, ptr_A, ptr_B);
      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);
      Tensor gScaleA = get<2>(load_inputs);

      auto scale_b = make_tensor(make_gmem_ptr(mainloop_params.ptr_scale_B + offset_b / K), make_shape(N, 1));
      auto gScaleB = local_tile(make_mix_tensor_like(scale_b), Shape<Int<CTA_N>, _1>{},
                                make_coord(_, Int<0>{}));

      typename CollectiveMainloop::GmemTiledCopyA gmem_tiled_copy_A = collective_mma.gmem_tiled_copy_A;
      typename CollectiveMainloop::GmemTiledCopyB gmem_tiled_copy_B = collective_mma.gmem_tiled_copy_B;
      auto& gmem_tiled_copy_scaleA = collective_mma.gmem_tiled_copy_scaleA;
      GmemTiledCopyScaleB gmem_tiled_copy_scaleB = collective_mma.gmem_tiled_copy_scaleB;

      // Construct shared memory tiles
      SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
      Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,K_TILE)
      Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
      // Partition the copying of A and B tiles across the threads
      auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
      auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

      Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
      Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,k)
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
      Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

      auto smem_tiled_copy_A = make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomA{}, tiled_mma);
      auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
      Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,k)

      auto smem_tiled_copy_B = make_tiled_copy_B(typename CollectiveMainloop::SmemCopyAtomB{}, tiled_mma);
      auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
      Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)

      // A scales stay resident; B scales alternate between the current and next N tile.
      Tensor sSA = make_tensor(make_smem_ptr(storage.channel_scales), SmemLayoutScaleA{});
      Tensor sSB = make_tensor(make_smem_ptr(storage.channel_scales + CTA_M), SmemLayoutScaleB{});

      auto gmem_thr_copy_scaleA = gmem_tiled_copy_scaleA.get_slice(thread_idx);
      auto gmem_thr_copy_scaleB = gmem_tiled_copy_scaleB.get_slice(thread_idx);

      Tensor tSgSA = gmem_thr_copy_scaleA.partition_S(gScaleA);
      Tensor tSsSA = gmem_thr_copy_scaleA.partition_D(sSA);
      Tensor tSgSB = gmem_thr_copy_scaleB.partition_S(gScaleB);
      Tensor tSsSB = gmem_thr_copy_scaleB.partition_D(sSB);

      // Both scale TiledCopy atoms use the linear AIU load.
      copy_aiu(gmem_tiled_copy_scaleA, tSgSA(_,_,_,0), tSsSA(_,_,_,0),
               gmem_tiled_copy_scaleB, tSgSB(_,_,_,n_coord), tSsSB(_,_,_,0), warp_idx);

      // Split waits need all A ready initially. A deeper ring keeps the
      // original staged publication; A still loads only once per N group.
      CUTLASS_PRAGMA_UNROLL
      for (int k_tile = DispatchPolicy::Stages == 2 ? 0 : DispatchPolicy::Stages;
           k_tile < KTileCount; ++k_tile) {
        copy_aiu(gmem_tiled_copy_A, tAgA(_,_,_,k_tile), tAsA(_,_,_,k_tile), warp_idx);
      }

      // Fill the B ring; each K tile retires and refills its consumed slot.
      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages; ++k_pipe) {
        if constexpr (DispatchPolicy::Stages == 2) {
          if (warp_idx == 1)
            copy(gmem_tiled_copy_B, tBgB(_,_,_,k_pipe), tBsB(_,_,_,k_pipe));
        } else {
          copy_aiu(gmem_tiled_copy_A, tAgA(_,_,_,k_pipe), tAsA(_,_,_,k_pipe),
                   gmem_tiled_copy_B, tBgB(_,_,_,k_pipe), tBsB(_,_,_,k_pipe), warp_idx);
        }
        cp_async_fence();
      }

      // ScaleB is loaded along N; the epilogue views it as (1, N, slot).
      auto sSB_copy = make_tensor(sSB.data(), select<1, 0, 2>(sSB.layout()));

      auto smem_tiled_copy_scale = make_tiled_copy_C(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_scale = smem_tiled_copy_scale.get_thread_slice(thread_idx);
      Tensor tCsSA = smem_thr_copy_scale.partition_S(sSA);
      Tensor tCsSB = smem_thr_copy_scale.partition_S(sSB_copy);

      Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
      Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
      Tensor tCrSA = make_fragment_like<ElementScale>(thr_mma.partition_fragment_C(sSA(_,_,Int<0>{})));
      Tensor tCrSB = make_fragment_like<ElementScale>(thr_mma.partition_fragment_C(sSB_copy(_,_,Int<0>{})));

      auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
      auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
      auto tCrSA_copy_view = smem_thr_copy_scale.retile_D(tCrSA);
      auto tCrSB_copy_view = smem_thr_copy_scale.retile_D(tCrSB);

      CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(mma_acc));
      CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(mma_acc));
      CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
      CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
      CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));
      CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
      CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));
      constexpr int KBlocks = decltype(size<2>(tCrA_copy_view))::value;
      constexpr int AtomsPerBlock = decltype(size<2>(tCrA) / size<2>(tCrA_copy_view))::value;
      CUTE_STATIC_ASSERT_V(size<2>(tCrA_copy_view) > Int<1>{});

      clear(mma_acc);
      cp_async_wait<DispatchPolicy::Stages-1>();
      pipeline_barrier();
      copy(smem_tiled_copy_A, tCsA(_,_,Int<0>{},Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB(_,_,Int<0>{},Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_scale, tCsSA(_,_,Int<0>{},Int<0>{}), tCrSA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_scale, tCsSB(_,Int<0>{},_,Int<0>{}), tCrSB_copy_view(_,Int<0>{},_));

      for (int n_iter = 0; n_iter < panel_tiles; ++n_iter) {
        const bool next_n = n_iter + 1 < panel_tiles;
        for_each(make_int_sequence<KTileCount>{}, [&](auto tile) {
          constexpr int k = decltype(tile)::value;
          constexpr int future_k = (k + DispatchPolicy::Stages) % KTileCount;
          const bool next_k = k + 1 < KTileCount || next_n;
          const bool future = k + DispatchPolicy::Stages < KTileCount || next_n;
          const int slot = (n_iter * KTileCount + k) % DispatchPolicy::Stages;
          const int next_slot = (slot + 1) % DispatchPolicy::Stages;
          auto a_current = tCsA(_,_,_,k);
          auto a_next = tCsA(_,_,_,(k + 1) % KTileCount);

          // Only the producers run this refill. ScaleB travels with B's K0
          // publication, so its destination is ready before the next epilogue.
          auto refill = [&](int destination) {
            if (warp_idx == 1) {
              if constexpr (future_k == 0) {
                tBgB.data() = tBgB.data() + K * BlockN;
                copy(gmem_tiled_copy_scaleB, tSgSB(_,_,_,n_coord + n_iter + 1),
                     tSsSB(_,_,_,(n_iter + 1) & 1));
              }
              copy(gmem_tiled_copy_B, tBgB(_,_,_,Int<future_k>{}), tBsB(_,_,_,destination));
            }
            cp_async_fence();
          };
          const bool producer = warp_idx == 1;
          for_each(make_int_sequence<KBlocks>{}, [&](auto block) {
            constexpr int j = decltype(block)::value;
            if constexpr (j + 1 < KBlocks) {
              copy(smem_tiled_copy_A, a_current(_,_,Int<j + 1>{}), tCrA_copy_view(_,_,Int<j + 1>{}));
              copy(smem_tiled_copy_B, tCsB(_,_,Int<j + 1>{},slot), tCrB_copy_view(_,_,Int<j + 1>{}));
            } else {
              if (next_k) {
                copy(smem_tiled_copy_A, a_next(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
                if constexpr (DispatchPolicy::Stages > 2)
                  copy(smem_tiled_copy_B, tCsB(_,_,Int<0>{},next_slot), tCrB_copy_view(_,_,Int<0>{}));
              }
            }

            for_each(make_int_sequence<AtomsPerBlock>{}, [&](auto subatom) {
              constexpr int atom = j * AtomsPerBlock + decltype(subatom)::value;
              if (k != 0 || atom != 0 || n_iter == 0)
                cute::gemm(tiled_mma, mma_acc, tCrA(_,_,Int<atom>{}), tCrB(_,_,Int<atom>{}), mma_acc);
            });

            if constexpr (j == KBlocks - 2) {
              if constexpr (DispatchPolicy::Stages == 2) {
                // With two slots, refill overlaps the consumers' last MMA.
                __ppu_wait_tsmcnt(0);
                if (producer) {
                  __ppu_barrier_sync(2, MaxThreadsPerBlock, 0);
                  if (future)
                    refill(slot);
                } else {
                  __ppu_barrier_arrive(2, MaxThreadsPerBlock, 0);
                }
              } else {
                // A deeper ring already hides the load; keep one rendezvous
                // for both the consumed slot and the next ready tile.
                cp_async_wait<DispatchPolicy::Stages - 2>();
                pipeline_barrier();
                if (producer && future)
                  refill(slot);
                else
                  cp_async_fence();  // Advance both producers' groups, including drain.
              }
            }
            if constexpr (DispatchPolicy::Stages == 2 && j + 1 == KBlocks) {
              if (producer) {
                if (future)
                  cp_async_wait<DispatchPolicy::Stages - 1>();
                else
                  cp_async_wait<0>();
              }
              // Publish the next B tile only after its AIU load completes.
              pipeline_barrier<3>();
              if (next_k) {
                copy(smem_tiled_copy_B, tCsB(_,_,Int<0>{},next_slot), tCrB_copy_view(_,_,Int<0>{}));
              }
            }
          });
        });

        using ThreadOp = typename CollectiveEpilogue::ThreadEpilogueOp;
        using FragmentAccumulator = typename CollectiveEpilogue::FragmentAccumulator;
        using OutputPair = typename CollectiveEpilogue::VectorType;
        using StorePair = typename CollectiveEpilogue::CopyAtomR2G;
        constexpr int OutputAlignment = CollectiveEpilogue::kOutputAlignment;
        CUTE_STATIC_ASSERT_V(size<0>(mma_acc) == Int<8>{});
        CUTE_STATIC_ASSERT_V(size<0>(shape<0>(mma_acc)) == Int<OutputAlignment>{});

        auto stride_d = [&] {
          if constexpr (TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad) {
            // NoPad requires packed row-major D; fold fragment store offsets.
            return make_stride(Int<N>{}, Int<1>{}, Int<0>{});
          } else {
            return cutlass::epilogue::collective::detail::get_epilogue_stride<
                typename CollectiveEpilogue::EpilogueSchedule>(params.epilogue.dD);
          }
        }();
        auto out = make_tensor(make_gmem_ptr(params.epilogue.ptr_D + deep_scheduler.curr_offset_c()),
                               make_shape(M, Int<N>{}, Int<L>{}), stride_d);
        auto out_tiles = local_tile(out, blk_shape, make_coord(_,_,_), Step<_1,_1,X>{});
        auto out_tile = out_tiles(_,_,m_coord,n_coord + n_iter,0);
        auto tD = thr_mma.partition_C(out_tile);
        auto tCoord = thr_mma.partition_C(make_identity_tensor(take<0,2>(blk_shape)));
        ThreadOp convert(params.epilogue.thread);

        auto store_row = [&](auto m, auto r) {
          for_each(make_int_sequence<size<2>(mma_acc)>{}, [&](auto n) {
            for_each(make_int_sequence<size<2>(shape<0>(mma_acc))>{}, [&](auto c) {
              FragmentAccumulator values;
              for_each(make_int_sequence<OutputAlignment>{}, [&](auto i) {
                auto p = make_coord(i, r, c);
                values[i] = mma_acc(p,m,n) *
                            tCrSA(p,m,Int<0>{}) * tCrSB(p,Int<0>{},n);
              });
              auto converted = convert(values);
              auto src = recast<OutputPair>(make_tensor(make_rmem_ptr(converted.data()), Int<OutputAlignment>{}));
              auto dst = recast<OutputPair>(tD(make_coord(_, r, c), m, n));
              copy(StorePair{}, src, dst);
            });
          });
        };

        // Drain both rows in each M fragment before seeding the next N tile.
        auto drain = [&](auto start_next, auto full_m) {
          for_each(make_int_sequence<size<1>(mma_acc)>{}, [&](auto m) {
            for_each(make_int_sequence<size<1>(shape<0>(mma_acc))>{}, [&](auto r) {
              if constexpr (decltype(full_m)::value) {
                store_row(m, r);
              } else {
                auto row = get<0>(tCoord(make_coord(Int<0>{}, r, Int<0>{}), m, Int<0>{}));
                if (row < m_max_coord)
                  store_row(m, r);
              }
            });
            // All lanes reconverge after predicated stores. The MMA
            // always runs for the full warp, including M-tail tiles.
            if constexpr (decltype(start_next)::value) {
              for_each(make_int_sequence<size<2>(mma_acc)>{}, [&](auto n) {
                auto fragment = mma_acc(_,m,n);
                clear(fragment);
                cute::gemm(tiled_mma, tCrA(_,m,Int<0>{}),
                           tCrB(_,n,Int<0>{}), fragment);
              });
            }
          });
        };
        if (next_n) {
          if (m_max_coord >= CTA_M)
            drain(cute::true_type{}, cute::true_type{});
          else
            drain(cute::true_type{}, cute::false_type{});
        } else {
          if (m_max_coord >= CTA_M)
            drain(cute::false_type{}, cute::true_type{});
          else
            drain(cute::false_type{}, cute::false_type{});
        }

        if (next_n) {
          copy(smem_tiled_copy_scale, tCsSB(_,Int<0>{},_,(n_iter + 1) & 1),
               tCrSB_copy_view(_,Int<0>{},_));
        }
      } // for n_iter
    } // Scheduler work fetch loop
  }

};

} // namespace cutlass::gemm::kernel
