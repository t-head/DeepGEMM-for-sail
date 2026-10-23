#pragma once
#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
#include <deep_gemm/impls/moe_gemm_scatter.cuh>

// Opt-in FP8 channelwise GEMM + scatter implementation.
// The existing channelwise CollectiveMma ABI and common MoE transport
// primitives are composed above; the fused epilogue and kernel live here.
// The generic KernelAiuMultistage path may reuse completed mainloop storage.
// OverlapPrologue is integrated through a dedicated epilogue wrapper whose
// BF16 scratch is physically after the entire mainloop shared-memory region.
// This is required because the next tile's prologue is already writing the
// mainloop region before the current tile reaches its epilogue.

namespace deep_gemm {

struct NoopChannelwiseGemmScatter {
  static constexpr bool kEnabled = false;
  static constexpr bool kPipelinePreviousTile = false;
  static constexpr bool kSharedFromAccumulators = false;
  static constexpr bool kPairedNSharedEpilogue = false;
  static constexpr bool kPairedNRegisterEpilogue = false;
  static constexpr bool kPairedNEpilogue = false;
  static constexpr int kPairedNSharedScratchBytes = 0;
  static constexpr bool kScatterLocalPairWork = false;
  static constexpr bool kRemotePairScheduler = false;
  static constexpr bool kExactFullGroupM = false;
  static constexpr bool kDrainRemoteAtLocalTransition = false;
  struct Arguments {};
  using Params = Arguments;

  static Params to_underlying_arguments(Arguments const& args) {
    return args;
  }
};

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_,
    bool kEnableSboOverlap = false,
    class GemmScatter_ = NoopChannelwiseGemmScatter,
    bool kUseBm256S4AccumulatorPairStaging = false>
class ChannelwiseFp8DeepGemmScatterUniversal
    : public cutlass::gemm::kernel::DeepGemmUniversal<
          ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, kEnableSboOverlap> {
public:
  using BaseUniversal = cutlass::gemm::kernel::DeepGemmUniversal<
      ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, kEnableSboOverlap>;
  // The base owns the ordinary GEMM types, parameter ABI and launch geometry.
  // This adapter adds routed stores and replica operands to that contract.
  using typename BaseUniversal::ProblemShape;
  using typename BaseUniversal::CollectiveMainloop;
  using typename BaseUniversal::TileShape;
  using typename BaseUniversal::TiledMma;
  using typename BaseUniversal::ElementA;
  using typename BaseUniversal::StrideA;
  using typename BaseUniversal::ElementB;
  using typename BaseUniversal::StrideB;
  using typename BaseUniversal::ElementAccumulator;
  using typename BaseUniversal::MainloopArguments;
  using typename BaseUniversal::MainloopParams;
  using typename BaseUniversal::CollectiveEpilogue;
  using typename BaseUniversal::ElementC;
  using typename BaseUniversal::StrideC;
  using typename BaseUniversal::ElementD;
  using typename BaseUniversal::StrideD;
  using typename BaseUniversal::ElementCompute;
  using typename BaseUniversal::EpilogueArguments;
  using typename BaseUniversal::EpilogueParams;
  using typename BaseUniversal::TileScheduler;
  using typename BaseUniversal::TileSchedulerArguments;
  using typename BaseUniversal::TileSchedulerParams;
  using GemmScatter = GemmScatter_;
  using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
  using BaseUniversal::MaxThreadsPerBlock;
  using BaseUniversal::MinBlocksPerMultiprocessor;
  using BaseUniversal::NumMmaWarpGroups;
  using BaseUniversal::N;
  using BaseUniversal::K;
  static constexpr uint32_t NPrefetchCachelines = BaseUniversal::N_PREFETCH_CACHELINE;

  static constexpr uint32_t BlockM =
      CUTE_STATIC_V(cute::size<0>(TileShape{}));
  static constexpr uint32_t BlockN =
      CUTE_STATIC_V(cute::size<1>(TileShape{}));
  static constexpr uint32_t SharedBf16TileBytes =
      BlockM * BlockN * sizeof(cutlass::bfloat16_t);

  static_assert(
      !GemmScatter::kEnabled ||
          TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad,
      "channelwise fused scatter currently supports GroupedNoPad only");
  static_assert(
      !GemmScatter::kRemotePairScheduler,
      "channelwise remote-pair scheduling is not implemented");
  static_assert(
      !GemmScatter::kPipelinePreviousTile,
      "channelwise previous-tile scatter is not implemented");
  static_assert(
      !GemmScatter::kExactRemoteFirstSegmentSchedule ||
          (TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad &&
           N == 6144 && K == 2048 &&
           TileScheduler::kNumGroups == 32 &&
           BlockM == 256 && BlockN == 256),
      "channelwise exact remote-first scheduling requires the "
      "GLM-5.2 EP8 T4096 BM256/BN256 layout");
  static_assert(
      !GemmScatter::kExactFullGroupM ||
          (GemmScatter::kExactGroupM == 1024 &&
           TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad &&
           N == 6144 && K == 2048 &&
           TileScheduler::kNumGroups == 32 &&
           BlockM == 256 && BlockN == 256),
      "channelwise full-tile staging requires the exact GLM-5.2 EP8 "
      "T4096 BM256/BN256 layout");
  static_assert(
      !GemmScatter::kExactRemoteFirstSegmentSchedule ||
          GemmScatter::kStoresLocalInD,
      "channelwise exact remote-first scheduling requires local-D "
      "mixed-source reduction");

  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;

  using SharedStorage = typename BaseUniversal::SharedStorage;

  static_assert(
      !GemmScatter::kSharedFromAccumulators ||
          sizeof(MainloopSharedStorage) >= SharedBf16TileBytes,
      "generic channelwise shared scatter requires the completed mainloop "
      "storage to hold one exact BF16 output tile");
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments : BaseUniversal::Arguments {
    typename GemmScatter::Arguments scatter{};
    ElementB const* rhs_replicas{nullptr};
    float const* rhs_replica_scales{nullptr};
    uint32_t num_master_groups{TileScheduler::kNumGroups};
  };

  struct Params : BaseUniversal::Params {
    typename GemmScatter::Params scatter{};
    ElementB const* rhs_replicas{nullptr};
    float const* rhs_replica_scales{nullptr};
    uint32_t num_master_groups{TileScheduler::kNumGroups};
  };

  static Params to_underlying_arguments(
      Arguments const& args, void* workspace) {
    return {
        BaseUniversal::to_underlying_arguments(args, workspace),
        GemmScatter::to_underlying_arguments(args.scatter),
        args.rhs_replicas,
        args.rhs_replica_scales,
        args.num_master_groups};
  }

  static bool can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemm;
  }

  using BaseUniversal::get_workspace_size;
  using BaseUniversal::initialize_workspace;
  using BaseUniversal::get_grid_shape;
  using BaseUniversal::get_block_shape;

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;

    const int warp_idx = cutlass::canonical_warp_idx_sync();
    if constexpr (TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad) {
      if constexpr (GemmScatter::kExactRemoteFirstSegmentSchedule) {
        const uint32_t phase_cachelines =
            (params.scatter.segment_count + 31u) / 32u;
        if (static_cast<uint32_t>(warp_idx) < phase_cachelines) {
          __ppu_prefetch_KSD(
              static_cast<void*>(
                  const_cast<int32_t*>(
                      params.scatter.segment_order +
                      params.scatter.segment_begin +
                      (warp_idx << 5))));
        }
      } else if (warp_idx <
                 static_cast<int>(NPrefetchCachelines)) {
        __ppu_prefetch_KSD(
            static_cast<void*>(
                const_cast<int32_t*>(
                    params.scheduler.grouped_layout +
                    (warp_idx << 5))));
      }
    }

    // Bound the persistent loop by the grouped M tiles. The preprocessed
    // scheduler already carries this bound and checks it in fetch_next_work.
    uint32_t m_tiles = 0;
    if constexpr (!TileScheduler::kIsNoPadPreprocessLayout) {
      const uint32_t lane = threadIdx.x % 32;
      for (uint32_t e = lane; e < TileScheduler::kNumGroups; e += 32)
        m_tiles += math::ceil_div(
            static_cast<uint32_t>(__ldg(params.scheduler.grouped_layout + e)),
            uint32_t(BlockM));
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        m_tiles += __shfl_xor_sync(0xffffffffu, m_tiles, off);
    }
    const uint32_t work_tiles = m_tiles * math::ceil_div(N, BlockN);
    TileScheduler scheduler(params.scheduler);
    SharedStorage& shared_storage =
        *reinterpret_cast<SharedStorage*>(smem_buf);
    const int thread_idx = static_cast<int>(threadIdx.x);
    auto block_shape = TileShape{};

    uint32_t m_block_idx;
    uint32_t n_block_idx;
    while (TileScheduler::kIsNoPadPreprocessLayout ||
           scheduler.current_iter * gridDim.x + blockIdx.x < work_tiles) {
      bool has_work;
      if constexpr (
          GemmScatter::kExactRemoteFirstSegmentSchedule) {
        static_assert(N % BlockN == 0);
        constexpr uint32_t kRowsPerGroup = 1024;
        constexpr uint32_t kMBlocksPerGroup =
            kRowsPerGroup / BlockM;
        constexpr uint32_t kMaxNumSegments =
            TileScheduler::kNumGroups * kMBlocksPerGroup;
        constexpr uint32_t kNumNBlocks = N / BlockN;
        const uint32_t total_tasks =
            params.scatter.segment_count * kNumNBlocks;
        const uint32_t linear_work_idx =
            static_cast<uint32_t>(scheduler.current_iter++) *
                static_cast<uint32_t>(gridDim.x) +
            static_cast<uint32_t>(blockIdx.x);
        has_work = linear_work_idx < total_tasks;
        if (has_work) {
          const uint32_t phase_segment_pos =
              linear_work_idx / kNumNBlocks;
          const uint32_t ordered_segment_pos =
              params.scatter.segment_begin + phase_segment_pos;
          n_block_idx =
              linear_work_idx -
              phase_segment_pos * kNumNBlocks;
          const uint32_t descriptor =
              static_cast<uint32_t>(
                  __ldg(
                      params.scatter.segment_order +
                      ordered_segment_pos));
          static_assert(kMaxNumSegments == 128);
          if (descriptor >= kMaxNumSegments) {
            // Fail closed before descriptor-derived scheduler/mainloop
            // addressing. The checked host path additionally requires a
            // complete [0, 128) permutation.
            continue;
          }
          const uint32_t group_idx =
              descriptor / kMBlocksPerGroup;
          m_block_idx =
              descriptor -
              group_idx * kMBlocksPerGroup;
          scheduler.curr_group_idx = group_idx;
          scheduler.curr_group_m = kRowsPerGroup;
          scheduler.curr_cumsum_m =
              group_idx * kRowsPerGroup;
        }
      } else {
        has_work =
            scheduler.fetch_next_work(
                m_block_idx, n_block_idx);
      }
      if (!has_work) {
        break;
      }
      uint32_t problem_m;
      const uint32_t row_start = m_block_idx * BlockM;
      if constexpr (GemmScatter::kExactFullGroupM) {
        problem_m = GemmScatter::kExactGroupM;
      } else {
        problem_m = scheduler.curr_problem_m();
        if constexpr (GemmScatter::kEnabled) {
          problem_m = GemmScatter::get_valid_m(
              params.scatter, scheduler.curr_group_idx, problem_m);
          // The scheduler intentionally follows physical padded rows.  A
          // valid prefix may end before a later physical tile begins; skip
          // the whole tile before forming an unsigned residue in that case.
          if constexpr (GemmScatter::kUsesPaddedValidRows) {
            if (row_start >= problem_m) {
              continue;
            }
          }
        }
      }

      constexpr uint32_t L = 1;
      auto problem_shape_mnkl =
          ProblemShape{problem_m, N, K, L};
      auto block_coord_mnkl =
          make_coord(m_block_idx, n_block_idx, _, 0);

      TiledMma tiled_mma;
      auto accumulators = make_fragment_like<ElementCompute>(
          partition_fragment_C(tiled_mma, take<0, 2>(block_shape)));
      clear(accumulators);
      auto residue_mnk =
          make_tuple(uint32_t{0}, uint32_t{0}, uint32_t{0});

      // Keep all prologue/mainloop-only objects out of the epilogue live
      // range.  BM256/S4 is at the PPU0015 register ceiling, so retaining
      // offsets, operand pointers, load tensors, and the K iterator across
      // staged scatter causes otherwise avoidable private spills.
      {
        const int offset_m =
            static_cast<int>(scheduler.curr_offset_m());
        const int64_t offset_a = scheduler.curr_offset_a();
        const uint32_t expert_id = scheduler.curr_group_idx;
        const bool use_replica =
            params.rhs_replicas != nullptr &&
            expert_id >= params.num_master_groups;
        const uint32_t rhs_group = use_replica
            ? expert_id - params.num_master_groups
            : expert_id;
        const int64_t offset_b = use_replica
            ? static_cast<int64_t>(rhs_group) * N * K
            : scheduler.curr_offset_b(m_block_idx);
        auto mainloop_params = params.mainloop;
        if (use_replica) {
          mainloop_params.ptr_B = params.rhs_replicas;
          mainloop_params.ptr_scale_B = params.rhs_replica_scales;
        }
        const ElementA* ptr_a =
            reinterpret_cast<const ElementA*>(
                mainloop_params.ptr_A) +
            offset_a;
        const ElementB* ptr_b =
            reinterpret_cast<const ElementB*>(
                mainloop_params.ptr_B) +
            offset_b;

        const auto tile_residue = deep_gemm::compute_channelwise_tile<CollectiveMainloop, N, false>(
            problem_shape_mnkl, block_coord_mnkl, mainloop_params, problem_m,
            offset_m, offset_b, ptr_a, ptr_b, accumulators,
            thread_idx, warp_idx, smem_buf);
        residue_mnk = make_tuple(
            static_cast<uint32_t>(get<0>(tile_residue)),
            static_cast<uint32_t>(get<1>(tile_residue)),
            static_cast<uint32_t>(get<2>(tile_residue)));
      }

      auto epilogue_params = params.epilogue;
      epilogue_params.ptr_C += scheduler.curr_offset_c();
      epilogue_params.ptr_D += scheduler.curr_offset_c();
      const uint32_t tile_pool_row_base =
          scheduler.curr_cumsum_m + row_start;
      bool store_tile_in_local_d = false;
      if constexpr (GemmScatter::kStoresLocalInD) {
        store_tile_in_local_d =
            GemmScatter::is_local_tile(
                params.scatter, tile_pool_row_base);
      }

      if constexpr (!GemmScatter::kSharedFromAccumulators) {
        CollectiveEpilogue epilogue{
            epilogue_params, shared_storage.tensors.epilogue};
        epilogue(
            problem_shape_mnkl, block_shape, block_coord_mnkl,
            accumulators, tiled_mma, residue_mnk, thread_idx,
            reinterpret_cast<char*>(
                &shared_storage.tensors.epilogue));
      } else if constexpr (GemmScatter::kStoresLocalInD) {
        // The exact layout guarantees one destination rank for the whole
        // BM256 segment. Same-rank segments take the native contiguous-D
        // epilogue and never enter the routed shared-scatter path.
        if (store_tile_in_local_d) {
          CollectiveEpilogue epilogue{
              epilogue_params, shared_storage.tensors.epilogue};
          epilogue(
              problem_shape_mnkl, block_shape, block_coord_mnkl,
              accumulators, tiled_mma, residue_mnk, thread_idx,
              reinterpret_cast<char*>(
                  &shared_storage.tensors.epilogue));
        }
      }

      if constexpr (GemmScatter::kEnabled) {
        uint32_t valid_m;
        if constexpr (GemmScatter::kExactFullGroupM) {
          valid_m = BlockM;
        } else {
          const uint32_t group_valid_m = GemmScatter::get_valid_m(
              params.scatter, scheduler.curr_group_idx,
              static_cast<uint32_t>(scheduler.curr_group_m));
          const uint32_t remaining_m =
              group_valid_m > row_start
                  ? group_valid_m - row_start
                  : 0;
          valid_m = cute::min(BlockM, remaining_m);
        }
        const uint32_t pool_row_base = tile_pool_row_base;
        const uint32_t n_start = n_block_idx * BlockN;
        auto* tile_d =
            reinterpret_cast<const __ppu_bfloat16*>(
                epilogue_params.ptr_D +
                m_block_idx * BlockM * N);

        if constexpr (GemmScatter::kSharedFromAccumulators) {
          // Local exact-rank segments have already materialized D through
          // the native epilogue above. Only remote segments stage BF16 and
          // issue peer NCP stores.
          if (!store_tile_in_local_d) {
          auto* shared_tile =
              reinterpret_cast<__ppu_bfloat16*>(smem_buf);
          uint4* exact_remote_vec_base = nullptr;
          if constexpr (
              GemmScatter::kExactRemoteFirstSegmentSchedule &&
              GemmScatter::kUsesScalarizedExactRoute) {
            // Read the actual rank from the current device metadata. This is
            // intentionally independent of any host tensor pointer/version
            // cache. Hoist only the 64-bit peer+N base across accumulator
            // staging; all per-row low words remain post-barrier scalar loads.
            uint32_t actual_rank;
#if __HGGC_ARCH__ == 150
            const auto* packed_words =
                reinterpret_cast<const uint32_t*>(
                    params.scatter.packed_src_info);
            actual_rank = __ld_smem(
                packed_words + (pool_row_base << 1) + 1);
#else
            const uint64_t packed =
                static_cast<uint64_t>(
                    params.scatter.packed_src_info[pool_row_base]);
            actual_rank = static_cast<uint32_t>(packed >> 32);
#endif
            const int64_t remote_n_base =
                reinterpret_cast<int64_t>(
                    params.scatter.combine_base) +
                GemmScatter::peer_offset(
                    params.scatter, actual_rank) +
                static_cast<int64_t>(n_start) *
                    static_cast<int64_t>(sizeof(__ppu_bfloat16));
            exact_remote_vec_base =
                reinterpret_cast<uint4*>(remote_n_base);
          }
          auto thread_mma =
              tiled_mma.get_thread_slice(thread_idx);
          auto identity = make_identity_tensor(
              make_shape(Int<BlockM>{}, Int<BlockN>{}));
          auto accumulator_coords =
              thread_mma.partition_C(identity);
          CUTE_STATIC_ASSERT_V(
              size(accumulator_coords) == size(accumulators));

          if constexpr (kUseBm256S4AccumulatorPairStaging) {
            // This pair mapping is deliberately restricted to the exact
            // BM256/BN256/BK128/W64x64/S4 Default TiledMMA selected below.
            // Its innermost accumulator fragment contains two adjacent N
            // values in one row, so convert and store both with one 4-byte
            // transaction. Other channelwise layouts retain the generic
            // exact-coordinate path because their pair may cross rows.
            static_assert(
                BlockM == 256 && BlockN == 256,
                "paired channelwise staging requires BM256/BN256");
            static_assert(
                MaxThreadsPerBlock == 512,
                "paired channelwise staging requires 512 threads");
            static_assert(
                decltype(size<1>(accumulator_coords))::value == 4,
                "paired channelwise staging requires four repeat-M groups");
            static_assert(
                decltype(size<2>(accumulator_coords))::value == 4,
                "paired channelwise staging requires four repeat-N groups");
            static_assert(
                decltype(size<0>(shape<0>(accumulator_coords)))::value == 2,
                "paired channelwise staging requires two adjacent values");
            static_assert(
                decltype(size<1>(shape<0>(accumulator_coords)))::value == 2,
                "paired channelwise staging requires two MMA-M atoms");
            static_assert(
                decltype(size<2>(shape<0>(accumulator_coords)))::value == 2,
                "paired channelwise staging requires two MMA-N atoms");
            static_assert(
                cute::is_same_v<
                    typename decltype(accumulators)::value_type,
                    float>,
                "paired channelwise staging requires FP32 accumulators");

            // Lock every layout property used when one coordinate is shared
            // by a two-float register fragment.  These probes are constexpr
            // only and intentionally generate no runtime instructions.
            using ExpectedAtomLayoutC = Layout<
                Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;
            static_assert(
                cute::is_same_v<
                    typename TiledMma::AtomLayoutC_TV,
                    ExpectedAtomLayoutC>,
                "paired channelwise staging requires the PPU0015 "
                "16x16 row-major accumulator layout");
            constexpr auto probe_identity =
                make_identity_tensor(make_shape(_256{}, _256{}));
            constexpr auto probe_coords =
                TiledMma{}
                    .get_thread_slice(_0{})
                    .partition_C(probe_identity);
            constexpr auto probe_fragment =
                TiledMma::make_fragment_C(probe_coords);
            static_assert(
                cute::is_same_v<
                    typename decltype(probe_fragment)::value_type,
                    float>,
                "paired channelwise fragment values must be FP32");
            CUTE_STATIC_ASSERT_V(
                (stride<0, 0>(probe_fragment) == _1{}));

            constexpr auto pair_0 = probe_coords(
                make_coord(_0{}, _0{}, _0{}), _0{}, _0{});
            constexpr auto pair_1 = probe_coords(
                make_coord(_1{}, _0{}, _0{}), _0{}, _0{});
            constexpr auto mma_n_1 = probe_coords(
                make_coord(_0{}, _0{}, _1{}), _0{}, _0{});
            constexpr auto repeat_n_1 = probe_coords(
                make_coord(_0{}, _0{}, _0{}), _0{}, _1{});
            constexpr auto last_n_pair = probe_coords(
                make_coord(_0{}, _0{}, _1{}), _0{}, _3{});
            CUTE_STATIC_ASSERT_V(
                get<0>(pair_0) == get<0>(pair_1));
            CUTE_STATIC_ASSERT_V(
                get<1>(pair_1) == get<1>(pair_0) + _1{});
            CUTE_STATIC_ASSERT_V(
                get<0>(mma_n_1) == get<0>(pair_0));
            CUTE_STATIC_ASSERT_V(
                get<0>(repeat_n_1) == get<0>(pair_0));
            CUTE_STATIC_ASSERT_V(
                get<1>(pair_0) % _2{} == _0{});
            CUTE_STATIC_ASSERT_V(
                get<1>(mma_n_1) % _2{} == _0{});
            CUTE_STATIC_ASSERT_V(
                get<1>(repeat_n_1) % _2{} == _0{});
            CUTE_STATIC_ASSERT_V(
                get<1>(last_n_pair) + _1{} < Int<BlockN>{});

            using FragmentAccumulator = cutlass::Array<float, 2>;
            union PackedBf16x2 {
              __ppu_bfloat162 bf16;
              uint32_t bits;
            };

            if constexpr (GemmScatter::kAssumesFullScatterTile) {
              // Integral-constant indices are required on PPU0015: a normal
              // unrolled `int` loop still materializes CuTe coordinate
              // algebra at every store site.  The host-gated exact GLM path
              // has four complete BM256 tiles per 1024-row expert, so neither
              // the coordinate construction nor the store needs valid-M
              // predication.
              cute::for_each(
                  cute::make_seq<
                      CUTE_STATIC_V(size<1>(accumulator_coords))>{},
                  [&](auto repeat_m) {
                cute::for_each(
                    cute::make_seq<CUTE_STATIC_V(
                        size<1>(shape<0>(accumulator_coords)))>{},
                    [&](auto mma_m) {
                  const auto row_coord = accumulator_coords(
                      make_coord(Int<0>{}, mma_m, Int<0>{}),
                      repeat_m, Int<0>{});
                  const uint32_t row =
                      static_cast<uint32_t>(get<0>(row_coord));

                  cute::for_each(
                      cute::make_seq<
                          CUTE_STATIC_V(size<2>(accumulator_coords))>{},
                      [&](auto repeat_n) {
                    cute::for_each(
                        cute::make_seq<CUTE_STATIC_V(
                            size<2>(shape<0>(accumulator_coords)))>{},
                        [&](auto mma_n) {
                      const auto coord = accumulator_coords(
                          make_coord(Int<0>{}, mma_m, mma_n),
                          repeat_m, repeat_n);
                      const uint32_t col =
                          static_cast<uint32_t>(get<1>(coord));
                      const FragmentAccumulator input =
                          *reinterpret_cast<const FragmentAccumulator*>(
                              accumulators(
                                  make_coord(_, mma_m, mma_n),
                                  repeat_m, repeat_n)
                                  .data());
                      PackedBf16x2 value;
                      value.bf16 = __floats2bfloat162_rn(
                          input[0], input[1]);
                      cutlass::arch::shared_store<4>(
                          cutlass::arch::cutlass_get_smem_pointer(
                              shared_tile + row * BlockN + col),
                          &value.bits);
                    });
                  });
                });
              });
            } else {
              // Preserve the generic and padded-row implementation exactly:
              // runtime valid-M predication remains mandatory for tail tiles.
              CUTLASS_PRAGMA_UNROLL
              for (int repeat_m = 0;
                   repeat_m < size<1>(accumulator_coords);
                   ++repeat_m) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_m = 0;
                     mma_m < size<1>(shape<0>(accumulator_coords));
                     ++mma_m) {
                  auto row_coord = accumulator_coords(
                      make_coord(Int<0>{}, mma_m, Int<0>{}),
                      repeat_m, Int<0>{});
                  const uint32_t row =
                      static_cast<uint32_t>(get<0>(row_coord));
                  if (row >= valid_m) continue;

                  CUTLASS_PRAGMA_UNROLL
                  for (int repeat_n = 0;
                       repeat_n < size<2>(accumulator_coords);
                       ++repeat_n) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mma_n = 0;
                         mma_n <
                             size<2>(shape<0>(accumulator_coords));
                         ++mma_n) {
                      auto coord = accumulator_coords(
                          make_coord(Int<0>{}, mma_m, mma_n),
                          repeat_m, repeat_n);
                      const uint32_t col =
                          static_cast<uint32_t>(get<1>(coord));
                      FragmentAccumulator input =
                          *reinterpret_cast<
                              const FragmentAccumulator*>(
                              accumulators(
                                  make_coord(_, mma_m, mma_n),
                                  repeat_m, repeat_n)
                                  .data());
                      PackedBf16x2 value;
                      value.bf16 = __floats2bfloat162_rn(
                          input[0], input[1]);
                      cutlass::arch::shared_store<4>(
                          cutlass::arch::cutlass_get_smem_pointer(
                              shared_tile + row * BlockN + col),
                          &value.bits);
                    }
                  }
                }
              }
            }
          } else {
            // Generic channelwise configs must stage every accumulator at
            // its exact coordinate; for example, W16x16 can place the two
            // values of one fragment in different rows.
            auto shared_output = make_tensor(
                make_smem_ptr(shared_tile),
                make_layout(
                    make_shape(Int<BlockM>{}, Int<BlockN>{}),
                    make_stride(Int<BlockN>{}, _1{})));
            auto thread_shared_output =
                thread_mma.partition_C(shared_output);
            CUTE_STATIC_ASSERT_V(
                size(thread_shared_output) == size(accumulators));

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(accumulators); ++i) {
              const auto coord = accumulator_coords(i);
              if (elem_less(
                      coord,
                      make_coord(
                          get<0>(residue_mnk),
                          get<1>(residue_mnk)))) {
                const __ppu_bfloat16 value =
                    __float2bfloat16_rn(
                        static_cast<float>(accumulators(i)));
                thread_shared_output(i) = value;
              }
            }
          }
#if __HGGC_ARCH__ == 150
          __ppu_barrier_sync_nocnt(0, 1);
#else
          __syncthreads();
#endif
          uint32_t exact_segment_rank = 0;
          if constexpr (
              GemmScatter::kExactRemoteFirstSegmentSchedule &&
              !GemmScatter::kUsesScalarizedExactRoute) {
            // Re-read the actual tile rank only after accumulator staging.
            // It remains correct when an external kernel refills the route
            // tensor in place without advancing PyTorch's version counter.
            const uint64_t packed =
                static_cast<uint64_t>(
                    __ldg(
                        params.scatter.packed_src_info +
                        pool_row_base));
            exact_segment_rank =
                static_cast<uint32_t>(packed >> 32);
          }
          GemmScatter::template scatter_staged_bf16<
              BlockM, BlockN, MaxThreadsPerBlock>(
              params.scatter, shared_tile,
              GemmScatter::kAssumesFullScatterTile
                  ? BlockM
                  : valid_m,
              n_start, pool_row_base, exact_segment_rank,
              exact_remote_vec_base);
          // Every CTA thread must finish consuming this tile before any warp
          // advances the persistent scheduler and lets the next mainloop
          // overwrite the shared-storage union.  PPU0015 needs only a control
          // rendezvous here; producer store queues drain once at kernel exit.
#if __HGGC_ARCH__ == 150
          __ppu_barrier_sync_nocnt(0, 1);
#else
          __syncthreads();
#endif
          }
        } else {
          GemmScatter::template scatter<
              BlockM, BlockN, MaxThreadsPerBlock>(
              params.scatter, tile_d, valid_m, n_start, N,
              pool_row_base);
        }
      }

      if constexpr (
          kEnableSboOverlap &&
          TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
        cp_async_wait<0>();
        __syncthreads();
        if (threadIdx.x == 0) {
          atomic_add_release_global(
              params.signal +
                  scheduler.curr_group_idx *
                      math::ceil_div(
                          scheduler.params.shape_m,
                          TileScheduler::BLOCK_M) +
                  m_block_idx,
              1);
        }
      }
    }

    if constexpr (GemmScatter::kEnabled) {
      GemmScatter::complete_stores();
    }
  }
};

template <
    typename ElementAB, typename ElementAcc,
    uint32_t ShapeN, uint32_t ShapeK,
    uint32_t BlockM, uint32_t BlockN, uint32_t BlockK,
    uint32_t WarpM, uint32_t WarpN,
    uint32_t NumGroups, uint32_t NumStages,
    GemmType GemmTypeValue,
    bool EnableSboOverlap = false,
    KernelStrategy KernelStrategyValue = KernelStrategy::Default,
    class GemmScatter = NoopChannelwiseGemmScatter,
    bool StageOnlyDebug = false,
    bool EnableBm128OverlapPairStaging = false>
class ChannelwiseFp8GemmWithScatter {
public:
  using ScatterArguments = typename GemmScatter::Arguments;

  static_assert(
      KernelStrategyValue == KernelStrategy::Default ||
          KernelStrategyValue == KernelStrategy::OverlapPrologue,
      "channelwise fused scatter supports Default and OverlapPrologue");
  static_assert(
      !EnableBm128OverlapPairStaging ||
          (KernelStrategyValue == KernelStrategy::OverlapPrologue &&
           ShapeN == 6144 && ShapeK == 2048 && NumGroups == 32 &&
           BlockM == 128 && BlockN == 256 && BlockK == 128 &&
           WarpM == 32 && WarpN == 64 && NumStages == 4),
      "BM128 overlap pair staging requires exact GLM T512 geometry");
  static_assert(
      cute::is_same_v<ElementAB, cutlass::float_e4m3_t>,
      "channelwise FP8 scatter requires E4M3 inputs");
  static_assert(
      cute::is_same_v<ElementAcc, float>,
      "channelwise FP8 scatter requires FP32 accumulators");
  static_assert(
      !GemmScatter::kEnabled || BlockN == 256,
      "channelwise BF16 scatter requires BlockN=256: the PPU0015 "
      "shared/global bulk transaction is a full 32-lane BF16x8 "
      "collective");
  static_assert(
      !GemmScatter::kEnabled || ShapeN % BlockN == 0,
      "channelwise BF16 scatter requires ShapeN to be divisible by "
      "BlockN so every full-warp store remains in bounds");

    using ElementA = ElementAB;
    using ElementB = ElementAB;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = ElementC;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using ElementCompute = float;
    using ArchTag = cutlass::arch::PPU0015;

    using TileShape =
        Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
    static constexpr int WarpOnM = BlockM / WarpM;
    static constexpr int WarpOnN = BlockN / WarpN;
    using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<
        ArchTag, ElementAB, ElementAB, ElementAcc>::type;
    using TiledMma = TiledMMA<
        MMA_Atom<MmaInst>,
        Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,
        Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _32>>;

    static constexpr bool TransA = false;
    static constexpr bool TransB = false;
    static constexpr bool EnableOverlapPrologue =
        KernelStrategyValue == KernelStrategy::OverlapPrologue;
    static constexpr int SmemLayoutStageStrideA =
        EnableOverlapPrologue
            ? (BlockM + BlockN) * BlockK
            : BlockM * BlockK;
    static constexpr int SmemLayoutStageStrideB =
        EnableOverlapPrologue
            ? (BlockM + BlockN) * BlockK
            : BlockN * BlockK;
    using DefaultOperandA =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<
            ArchTag, ElementA, TransA, Int<BlockM>, Int<BlockK>,
            false, SmemLayoutStageStrideA>;
    using DefaultOperandB =
        cutlass::gemm::config::DefaultGemm_AIU_Operand<
            ArchTag, ElementB, TransB, Int<BlockN>, Int<BlockK>,
            true, SmemLayoutStageStrideB>;
    using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
    using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
    using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
    using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
    using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
    using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

    using KernelSchedule = cute::conditional_t<
        EnableOverlapPrologue,
        ::KernelAiuMultistageOverlapPrologue,
        cutlass::gemm::KernelAiuMultistage>;
    using DispatchPolicy = cute::conditional_t<
        EnableOverlapPrologue,
        cutlass::gemm::MainloopPPUAiuA8W8OverlapPrologue<
            NumStages, KernelSchedule>,
        cutlass::gemm::MainloopPPUAiuA8W8<
            NumStages, KernelSchedule>>;
    using CollectiveMainloop =
        cutlass::gemm::collective::CollectiveMma<
            ArchTag, DispatchPolicy, TileShape,
            ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
            ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
            TiledMma,
            GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA,
            cute::identity,
            GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB,
            cute::identity>;

    static constexpr bool IsAlignedN = ShapeN % BlockN == 0;
    using BaseCollectiveEpilogue =
        cutlass::epilogue::collective::DefaultEpilogueNoTsm<
            cutlass::detail::TagToStrideA_t<LayoutC>,
            cutlass::detail::TagToStrideA_t<LayoutD>,
            cutlass::epilogue::thread::LinearCombination<
                ElementC, 2, float, float,
                cutlass::epilogue::thread::ScaleType::Nothing>,
            cutlass::gemm::EpilogueDefault,
            IsAlignedN>;

    using TileScheduler = DeepGemmScheduler<
        GemmTypeValue, ShapeN, ShapeK,
        BlockM, BlockN, NumGroups, math::ceil_div(ShapeN, BlockN), 2,
        EpilogueType::Default>;
    static constexpr bool UseBm256S4AccumulatorPairStaging =
        KernelStrategyValue == KernelStrategy::Default &&
        BlockM == 256 && BlockN == 256 && BlockK == 128 &&
        WarpM == 64 && WarpN == 64 && NumStages == 4 &&
        ShapeN % BlockN == 0;
    static constexpr bool UseBm128S4OverlapAccumulatorPairStaging =
        EnableBm128OverlapPairStaging &&
        KernelStrategyValue == KernelStrategy::OverlapPrologue &&
        BlockM == 128 && BlockN == 256 && BlockK == 128 &&
        WarpM == 32 && WarpN == 64 && NumStages == 4 &&
        ShapeN % BlockN == 0 &&
        GemmScatter::kSharedFromAccumulators;
    using OverlapScatterEpilogue =
        mega_moe::detail::MoeOverlapScatterEpilogue<
            BaseCollectiveEpilogue,
            typename CollectiveMainloop::SharedStorage,
            GemmScatter,
            TileShape,
            CUTE_STATIC_V(cute::size(TiledMma{})),
            ShapeN,
            StageOnlyDebug,
            UseBm128S4OverlapAccumulatorPairStaging>;
    using DefaultGemmKernel = ChannelwiseFp8DeepGemmScatterUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        BaseCollectiveEpilogue,
        TileScheduler,
        EnableSboOverlap,
        GemmScatter,
        UseBm256S4AccumulatorPairStaging>;
    using OverlapGemmKernel =
        cutlass::gemm::kernel::DeepGemmUniversal<
            Shape<int, int, int, int>,
            CollectiveMainloop,
            OverlapScatterEpilogue,
            TileScheduler,
            EnableSboOverlap, void, true>;
    using GemmKernel = cute::conditional_t<
        EnableOverlapPrologue,
        OverlapGemmKernel,
        DefaultGemmKernel>;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;

};

}  // namespace deep_gemm
