#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/epilogue_traits.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/dispatch_policy.hpp>

#include <ppu_include.hpp>

#include <deep_gemm/mma/fp4_mma.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>
#include <deep_gemm/mma/fp4_mma_overlap_prologue.cuh>
#include <deep_gemm/epilogue/fp4_epilogue_silu_and_mul_post_quant.cuh>

namespace cutlass::gemm {

constexpr static int Fp4TileLength = 5;
constexpr static int Fp4ElementsPerTile = 6; // BLOCK_M, BLOCK_N, warpM, warpN, blockK, stages

struct KernelAiuFp4DynamicTileLargeEM {
  constexpr static deep_gemm::FP4DynamicTileId DynamicTildId = deep_gemm::FP4DynamicTileId::LargeEM;
  constexpr static int TileConfigList[Fp4TileLength][Fp4ElementsPerTile] = {
    {64,  256, 32, 32, 128, 4},
    {64,  256, 32, 32, 128, 4},
    {128, 256, 32, 64, 128, 4},
    {192, 256, 48, 64, 128, 3},
    {256, 256, 64, 64, 128, 3}
  };
}; // max

struct KernelAiuFp4DynamicTileLargeK {
  constexpr static deep_gemm::FP4DynamicTileId DynamicTildId = deep_gemm::FP4DynamicTileId::LargeK;
  constexpr static int TileConfigList[Fp4TileLength][Fp4ElementsPerTile] = {
    {32, 256, 32, 32, 128, 3},
    {64, 256, 32, 64, 128, 3},
    {128, 128, 32, 64, 128, 3},
    {192, 128, 48, 64, 128, 3},
    {256, 128, 64, 64, 64,  3}
  };
};

struct KernelAiuFp4DynamicTileLargeK_G2 {
  constexpr static deep_gemm::FP4DynamicTileId DynamicTildId = deep_gemm::FP4DynamicTileId::LargeK_G2;
  constexpr static int TileConfigList[Fp4TileLength][Fp4ElementsPerTile] = {
    { 32, 256, 32, 32, 128, 3},
    { 64, 256, 32, 64, 128, 3},
    { 64, 256, 32, 64, 128, 3},
    {128, 256, 64, 64,  64, 3},
    {128, 256, 64, 64,  64, 3}
  };
};

struct KernelAiuFp4DynamicTileSmallEM {
  constexpr static deep_gemm::FP4DynamicTileId DynamicTildId = deep_gemm::FP4DynamicTileId::SmallEM;
  constexpr static int TileConfigList[Fp4TileLength][Fp4ElementsPerTile] = {
    {16, 128, 16, 32, 128, 3},
    {32, 128, 32, 32, 128, 4},
    {64, 128, 32, 64, 128, 3},
    {64, 128, 32, 64, 128, 3},
    {128, 128, 64, 64, 64, 3}
  };
};

// Select FP4 dynamic-tile variant by FP4DynamicTileId
template <deep_gemm::FP4DynamicTileId kDynamicTileId> struct Fp4DynamicTileSelector;
template <> struct Fp4DynamicTileSelector<deep_gemm::FP4DynamicTileId::LargeEM>   { using type = KernelAiuFp4DynamicTileLargeEM; };
template <> struct Fp4DynamicTileSelector<deep_gemm::FP4DynamicTileId::LargeK>    { using type = KernelAiuFp4DynamicTileLargeK; };
template <> struct Fp4DynamicTileSelector<deep_gemm::FP4DynamicTileId::LargeK_G2> { using type = KernelAiuFp4DynamicTileLargeK_G2; };
template <> struct Fp4DynamicTileSelector<deep_gemm::FP4DynamicTileId::SmallEM>   { using type = KernelAiuFp4DynamicTileSmallEM; };

}

namespace cutlass::gemm::kernel {

using deep_gemm::GemmType;

template <
  GemmType kGemmType,
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementD,
  typename ElementAcc,
  typename ElementCompute,
  int SHAPE_N,
  int SHAPE_K,
  int kNumGroups,
  typename KernelAiuFp4DynamicTile,
  int TileId,
  bool hasBias = false,
  int N_EXPAND = 1,
  deep_gemm::EpilogueType kEpilogueType = deep_gemm::EpilogueType::Default,
  bool kApplySwigluLimit = false
>
struct Fp4TypeBuilder {
  using LayoutA     = cutlass::layout::RowMajor;
  using LayoutB     = cutlass::layout::ColumnMajor;
  using LayoutC     = cutlass::layout::RowMajor;
  using StrideA     = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideB     = cutlass::detail::TagToStrideB_t<LayoutB>;
  using StrideC     = cutlass::detail::TagToStrideC_t<LayoutC>;
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ElementBias         = float;                                          // Element type for bias addition
  constexpr static int AlignmentC  = 1;
  constexpr static int AlignmentD  = AlignmentC;
  using ArchTag = cutlass::arch::PPU0015;
  static constexpr bool TransA = false;
  static constexpr bool TransB = false;

  using ProblemShape = Shape<int,int,int,int>;

  static constexpr int BLOCK_M = KernelAiuFp4DynamicTile::TileConfigList[TileId][0];
  static constexpr int BLOCK_N = KernelAiuFp4DynamicTile::TileConfigList[TileId][1];
  static constexpr int WARP_M = KernelAiuFp4DynamicTile::TileConfigList[TileId][2];
  static constexpr int WARP_N = KernelAiuFp4DynamicTile::TileConfigList[TileId][3];
  static constexpr int BLOCK_K = KernelAiuFp4DynamicTile::TileConfigList[TileId][4];
  static constexpr int kNumStages = KernelAiuFp4DynamicTile::TileConfigList[TileId][5];
  static_assert(deep_gemm::EpilogueTraits<kEpilogueType>::is_valid_config(SHAPE_N, BLOCK_N, N_EXPAND, hasBias),
                "SiluAndMulPostQuantFp4 Epilogue only support BLOCK_N >= 64, SHAPE_N % 64 == 0, "
                "N_EXPAND == 1 and hasBias == false.");

  using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
  using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
  using WarpOnM = Int<BLOCK_M / WARP_M>;
  using WarpOnN = Int<BLOCK_N / WARP_N>;
  static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;

  using MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;
  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>;
      
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));

  using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false>;
  using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true>;
  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  using TransformA = cute::identity;
  using TransformB = cute::identity;

  // Use Aiu for SFA
  using ElementSFA = uint16_t;
  using LayoutSFA = cutlass::layout::ColumnMajor;
  static constexpr bool TransSFA = true;
  static constexpr int MinAiuContElemSize = 32 / (sizeof_bits<ElementSFA>::value / 8);      // 16

  static constexpr int ScaleGranularityK = 32;
  static constexpr int ScaleMsPerTile = BLOCK_M;
  static constexpr int ScaleKsPerTile = BLOCK_K / ScaleGranularityK; // BlockK must divideable by 32

  static constexpr int SFATileM = TransSFA ? cute::max(ScaleMsPerTile, MinAiuContElemSize) : ScaleMsPerTile;
  static constexpr int SFATileK = TransSFA ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);

  static constexpr bool swap = true;
  static constexpr int StageStride = 0;
  static constexpr bool swzl = false;
  using GemmOperandSFA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementSFA, TransSFA, Int<SFATileM>, Int<SFATileK>, swap, StageStride, swzl>;

  // Use Aiu for SFB
  using ElementSFB = uint16_t;
  using LayoutSFB = cutlass::layout::RowMajor;
  static constexpr bool TransSFB = true;
  static constexpr int MinAiuContElemSizeSFB = 32 / (sizeof_bits<ElementSFB>::value / 8);      // 16

  static constexpr int ScaleNsPerTile = BLOCK_N;
  static constexpr int ScaleKsPerTileSFB = ScaleKsPerTile;

  static constexpr int SFBTileN = TransSFB ? cute::max(ScaleNsPerTile, MinAiuContElemSizeSFB) : ScaleNsPerTile;
  static constexpr int SFBTileK = TransSFB ? ScaleKsPerTileSFB : cute::max(ScaleKsPerTileSFB, MinAiuContElemSizeSFB);

  using GemmOperandSFB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementSFB, TransSFB, Int<SFBTileN>, Int<SFBTileK>, swap, StageStride, swzl>;
  using GemmOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false>;
  using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true>;

  using DispatchPolicy = cutlass::gemm::MainloopWithScalePPU0015Aiu<kNumStages>;

  using CollectiveMainloopBase = typename cutlass::gemm::collective::CollectiveMmaScaleFp4<
      DispatchPolicy, TileShape,
      ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
      ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
      TiledMma,
      typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
      typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB,
      ElementSFA, cutlass::detail::TagToStrideA_t<LayoutSFA>, typename GemmOperandSFA::GmemTiledCopy, typename GemmOperandSFA::SmemLayoutAtom,
      ElementSFB, cutlass::detail::TagToStrideB_t<LayoutSFB>, typename GemmOperandSFB::GmemTiledCopy, typename GemmOperandSFB::SmemLayoutAtom, KernelAiuFp4DynamicTile::DynamicTildId>;

  // All 11 builder/family combinations that select_builder() can reach were measured with the
  // prefetch on (FP4_TFU_HANDOFF E55: masked, 3 paired reps, 10 faster by 2.5-11.2%, 1 flat), so
  // the dynamic path needs no performance gate here -- only the protocol conditions. Re-measure if
  // TileConfigList changes.
  static constexpr bool kUseOverlapPrologue =
      kEpilogueType == deep_gemm::EpilogueType::Default && !hasBias &&
      N_EXPAND == 1 && SHAPE_N % 2 == 0 &&
      cutlass::gemm::collective::CollectiveMmaScaleFp4OverlapPrologue<CollectiveMainloopBase>::template use_overlap_prologue<SHAPE_K>();

  // OverlapPrologue lives outside the base mainloop; disabled builders keep the
  // untouched base collective.
  using CollectiveMainloop = cute::conditional_t<
      kUseOverlapPrologue,
      cutlass::gemm::collective::CollectiveMmaScaleFp4OverlapPrologue<CollectiveMainloopBase>,
      CollectiveMainloopBase>;

  // Epilogue
  using EpilogueOutputOp = typename cutlass::platform::conditional<
    hasBias,
    cutlass::epilogue::thread::LinearCombinationBiasElementwise<ElementD, ElementAccumulator, ElementCompute, ElementD, ElementD, AlignmentD, cutlass::epilogue::thread::Identity<float>, cutlass::plus<ElementCompute>, false, ElementBias>,
    cutlass::epilogue::thread::LinearCombination<ElementD, 2, ElementAccumulator, ElementCompute, cutlass::epilogue::thread::ScaleType::Nothing, cutlass::FloatRoundStyle::round_to_nearest, ElementC>,
  >::type;

  using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<AlignmentC * sizeof(ElementC) * 8>;
  using GemmEpilogueConfiguration = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, float, AlignmentC, Int<BLOCK_M>, Int<BLOCK_N>, WarpOnM, ThreadNum>;
  static constexpr bool IsAligedN = SHAPE_N % BLOCK_N == 0 ? true : false;

  using CollectiveEpilogueNoTsm = typename cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOutputOp,
    cutlass::gemm::EpilogueDefault,
    IsAligedN
  >;

  using CollectiveEpilogueSiluAndMulPostQuant = typename cutlass::epilogue::collective::EpilogueSiluAndMulPostQuant<
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOutputOp,
    cutlass::gemm::EpilogueDefault,
    IsAligedN,
    kApplySwigluLimit,
    kEpilogueType
  >;

  using CollectiveEpilogue = typename cutlass::platform::conditional<
    is_act_and_quant_fused(kEpilogueType),
    CollectiveEpilogueSiluAndMulPostQuant,
    CollectiveEpilogueNoTsm
  >::type;

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

  // next_work() returns (M, m_coord, n_coord, offset_a, offset_b, offset_scalea, offset_scaleb)
  // of the scheduler's next tile; its loads are only issued when has_next is true.
  template<typename Params, typename NextWork>
  static CUTLASS_DEVICE void run(Params params, char* smem_buf, int M, int m_coord, int n_coord,
                          int64_t offset_a, int64_t offset_b, int64_t offset_m, int64_t offset_c,
                          int64_t offset_scalea, int64_t offset_scaleb, int64_t offset_scalec,
                          NextWork const& next_work, int& pipe_slot, bool& preloaded, bool has_next) {
    if constexpr (is_act_and_quant_fused(kEpilogueType)) {
      // Epilogue reuse the shared storage comes from mainloop.
      // Make sure that the size of epilogue SharedStorage is less than Mainloop.
      static_assert(CollectiveEpilogueSiluAndMulPostQuant::get_shared_storage_size(BLOCK_M, BLOCK_N)
                    <= SharedStorageSize,
                    "fused epilogue act tile exceeds the kernel shared storage");
    }

    using namespace cute;
    int thread_idx = int(threadIdx.x);
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto N = Int<SHAPE_N>{};
    auto K = Int<SHAPE_K>{};
    constexpr int L = 1;
    auto problem_shape_MNKL = ProblemShape{M, N, K, L};
    auto blk_shape = TileShape{};

    auto mainloop_args = [&](int m, int64_t a, int64_t b, int64_t scale_a, int64_t scale_b)
        __attribute__((always_inline)) {
      return typename CollectiveMainloop::Arguments{
        {m, N, K},
        reinterpret_cast<const ElementA*>(params.ptr_A) + a, params.stride_A,
        reinterpret_cast<const ElementB*>(params.ptr_B) + b, params.stride_B,
        reinterpret_cast<const typename CollectiveMainloop::ElementSFA*>(params.ptr_scale_A) + scale_a, params.stride_SFA,
        reinterpret_cast<const typename CollectiveMainloop::ElementSFB*>(params.ptr_scale_B) + scale_b, params.stride_SFB
      };
    };
    auto args_mainloop = mainloop_args(M, offset_a, offset_b, offset_scalea, offset_scaleb);

    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, 0);
    TiledMma tiled_mma;
    Tensor accumulators = make_fragment_like<ElementCompute>(partition_fragment_C(tiled_mma, take<0,2>(blk_shape)));
    auto residue_mnk = [&]() __attribute__((always_inline)) {
      if constexpr (kUseOverlapPrologue) {
        typename CollectiveMainloop::PipelineState pipeline_state{pipe_slot, preloaded};
        auto next_fn = [&]() __attribute__((always_inline)) {
          auto [next_m, next_m_coord, next_n_coord, next_a, next_b, next_scale_a, next_scale_b] = next_work();
          return make_tuple(mainloop_args(next_m, next_a, next_b, next_scale_a, next_scale_b),
                            ProblemShape{next_m, N, K, L}, make_coord(next_m_coord, next_n_coord, _, 0));
        };
        auto residue = CollectiveMainloop::compute_tile(
          args_mainloop, problem_shape_MNKL, blk_coord_mnkl, accumulators, thread_idx, smem_buf,
          &pipeline_state, next_fn, has_next
        );
        pipe_slot = has_next ? pipeline_state.smem_pipe_read : 0;
        preloaded = has_next && pipeline_state.preloaded;
        return residue;
      } else {
        pipe_slot = 0;
        preloaded = false;
        return CollectiveMainloop::compute_tile(
          args_mainloop, problem_shape_MNKL, blk_coord_mnkl, accumulators, thread_idx, smem_buf
        );
      }
    }();

    // update params.epilogue for ptrC and ptrD
    auto params_epilogue_local = *(reinterpret_cast<typename CollectiveEpilogue::Params*>(&(params.epi_params)));
    params_epilogue_local.ptr_D += offset_c;
    if constexpr (deep_gemm::EpilogueTraits<kEpilogueType>::needs_sfd_offset) {
      params_epilogue_local.ptr_SFD += offset_scalec;
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

};

template <
  GemmType kGemmType,
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementD,
  typename ElementAcc,
  typename ElementCompute,
  int SHAPE_N,
  int SHAPE_K,
  int kNumGroups,
  deep_gemm::FP4DynamicTileId kDynamicTileId,
  deep_gemm::EpilogueType kEpilogueType,
  bool kApplySwigluLimit
>
struct Fp4DeepGemmDynamicTile {

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  using ProblemShape = Shape<int,int,int,int>;

  static constexpr bool IsEP = (kGemmType == GemmType::MGroupedMasked);

  using KernelAiuFp4DynamicTile = typename Fp4DynamicTileSelector<kDynamicTileId>::type;
  using Builder0 = Fp4TypeBuilder<kGemmType, ElementA, ElementB, ElementC, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuFp4DynamicTile, 0,
                                  false /*has_bias*/, 1 /*n_expand*/, kEpilogueType, kApplySwigluLimit>;
  using Builder1 = Fp4TypeBuilder<kGemmType, ElementA, ElementB, ElementC, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuFp4DynamicTile, 1,
                                  false /*has_bias*/, 1 /*n_expand*/, kEpilogueType, kApplySwigluLimit>;
  using Builder2 = Fp4TypeBuilder<kGemmType, ElementA, ElementB, ElementC, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuFp4DynamicTile, 2,
                                  false /*has_bias*/, 1 /*n_expand*/, kEpilogueType, kApplySwigluLimit>;
  using Builder3 = Fp4TypeBuilder<kGemmType, ElementA, ElementB, ElementC, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuFp4DynamicTile, 3,
                                  false /*has_bias*/, 1 /*n_expand*/, kEpilogueType, kApplySwigluLimit>;
  using Builder4 = Fp4TypeBuilder<kGemmType, ElementA, ElementB, ElementC, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuFp4DynamicTile, 4,
                                  false /*has_bias*/, 1 /*n_expand*/, kEpilogueType, kApplySwigluLimit>;
  static constexpr uint32_t MaxThreadsPerBlock = Builder4::MaxThreadsPerBlock;

  using CollectiveEpilogue = typename Builder4::CollectiveEpilogue;
  using ElementSFA         = typename Builder4::CollectiveMainloop::ElementSFA;
  using ElementSFB         = typename Builder4::CollectiveMainloop::ElementSFB;

  // Dynamic tile builder config: aggregates all builders' BLOCK_M/BLOCK_N
  using BuilderConfig = ::deep_gemm::DynamicTileBuilderConfig<
      Builder0::BLOCK_M, Builder1::BLOCK_M, Builder2::BLOCK_M, Builder3::BLOCK_M, Builder4::BLOCK_M,
      Builder0::BLOCK_N, Builder1::BLOCK_N, Builder2::BLOCK_N, Builder3::BLOCK_N, Builder4::BLOCK_N>;
  using TileScheduler = ::deep_gemm::DynamicTileScheduler<
      kGemmType, SHAPE_N, SHAPE_K, BuilderConfig, kNumGroups, 2 /*kNum1DBlocksPerGroup*/, kEpilogueType>;
  using StrideA            = typename Builder4::CollectiveMainloop::StrideA;
  using StrideB            = typename Builder4::CollectiveMainloop::StrideB;
  using StrideC            = typename Builder4::CollectiveEpilogue::StrideC;
  using StrideD            = typename Builder4::CollectiveEpilogue::StrideD;
  using StrideSFA          = typename Builder4::CollectiveMainloop::StrideSFA;
  using StrideSFB          = typename Builder4::CollectiveMainloop::StrideSFB;

  struct Arguments {
    // arguments for mainloop
    ElementA const* ptr_A;
    StrideA stride_A;
    ElementB const* ptr_B;
    StrideB stride_B;
    ElementSFA const* ptr_scale_A;
    StrideSFA stride_SFA;
    ElementSFB const* ptr_scale_B;
    StrideSFB stride_SFB;
    // arguments for epilogue
    typename CollectiveEpilogue::Params epi_params;
    // arguments for scheduler
    uint32_t shape_m;
    int* grouped_layout;
  };

  using Params = Arguments;
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return args;
  }
  struct SharedStorage {
    // Mainloop and epilogue don't use smem concurrently since kernel is non-persistent, so we can use a union
    union SharedTensorStorage {
      typename Builder0::SharedStorage tensor_0;
      typename Builder1::SharedStorage tensor_1;
      typename Builder2::SharedStorage tensor_2;
      typename Builder3::SharedStorage tensor_3;
      typename Builder4::SharedStorage tensor_4;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }
  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    TileScheduler deep_scheduler({params.shape_m, params.grouped_layout});

    uint32_t m_block_idx = 0, n_block_idx = 0;
    int builder_idx = 4;
    int pipe_slot = 0;
    bool preloaded = false;
    constexpr bool kAnyOverlapPrologue = Builder0::kUseOverlapPrologue || Builder1::kUseOverlapPrologue ||
                                  Builder2::kUseOverlapPrologue || Builder3::kUseOverlapPrologue || Builder4::kUseOverlapPrologue;
    bool has_work = false;
    if constexpr (kAnyOverlapPrologue) {
      has_work = deep_scheduler.fetch_next_work(m_block_idx, n_block_idx, builder_idx);
    }
    #pragma clang loop licm(disable)
    while ([&]() __attribute__((always_inline)) {
      if constexpr (kAnyOverlapPrologue) {
        return has_work;
      } else {
        return deep_scheduler.fetch_next_work(m_block_idx, n_block_idx, builder_idx);
      }
    }()) {
      uint32_t M = deep_scheduler.curr_problem_m();
      auto offset_m = deep_scheduler.curr_offset_m();
      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx);
      auto offset_c = deep_scheduler.curr_offset_c();
      auto offset_scalea = deep_scheduler.curr_offset_mxfp4_scalea();
      auto offset_scaleb = deep_scheduler.curr_offset_mxfp4_scaleb(m_block_idx);
      int64_t offset_scalec = 0;
      if constexpr (deep_gemm::EpilogueTraits<kEpilogueType>::needs_sfd_offset) {
        offset_scalec = deep_scheduler.curr_offset_c_scales();
      }

      uint32_t next_m_block_idx = 0, next_n_block_idx = 0;
      int next_builder_idx = 4;
      bool has_next_work = false;
      bool same_builder = false;
      if constexpr (kAnyOverlapPrologue) {
        has_next_work = deep_scheduler.fetch_next_work(next_m_block_idx, next_n_block_idx, next_builder_idx);
        same_builder = has_next_work && builder_idx == next_builder_idx;
      }
      // Read after the lookahead, so the scheduler already describes the next tile.
      auto next_work = [&]() __attribute__((always_inline)) {
        return cute::make_tuple(int(deep_scheduler.curr_problem_m()), int(next_m_block_idx), int(next_n_block_idx),
                          deep_scheduler.curr_offset_a(), deep_scheduler.curr_offset_b(next_m_block_idx),
                          deep_scheduler.curr_offset_mxfp4_scalea(),
                          deep_scheduler.curr_offset_mxfp4_scaleb(next_m_block_idx));
      };

      // Dispatch to the selected builder
      switch (builder_idx) {
        case 0:
          Builder0::run(params, smem_buf, M, m_block_idx, n_block_idx,
            offset_a, offset_b, offset_m, offset_c, offset_scalea, offset_scaleb, offset_scalec,
            next_work, pipe_slot, preloaded, same_builder && Builder0::kUseOverlapPrologue);
          break;
        case 1:
          Builder1::run(params, smem_buf, M, m_block_idx, n_block_idx,
            offset_a, offset_b, offset_m, offset_c, offset_scalea, offset_scaleb, offset_scalec,
            next_work, pipe_slot, preloaded, same_builder && Builder1::kUseOverlapPrologue);
          break;
        case 2:
          Builder2::run(params, smem_buf, M, m_block_idx, n_block_idx,
            offset_a, offset_b, offset_m, offset_c, offset_scalea, offset_scaleb, offset_scalec,
            next_work, pipe_slot, preloaded, same_builder && Builder2::kUseOverlapPrologue);
          break;
        case 3:
          Builder3::run(params, smem_buf, M, m_block_idx, n_block_idx,
            offset_a, offset_b, offset_m, offset_c, offset_scalea, offset_scaleb, offset_scalec,
            next_work, pipe_slot, preloaded, same_builder && Builder3::kUseOverlapPrologue);
          break;
        default:
          Builder4::run(params, smem_buf, M, m_block_idx, n_block_idx,
            offset_a, offset_b, offset_m, offset_c, offset_scalea, offset_scaleb, offset_scalec,
            next_work, pipe_slot, preloaded, same_builder && Builder4::kUseOverlapPrologue);
          break;
      }
      if constexpr (kAnyOverlapPrologue) {
        has_work = has_next_work;
        m_block_idx = next_m_block_idx;
        n_block_idx = next_n_block_idx;
        builder_idx = next_builder_idx;
      }
    }
  }

};

} // end of cutlass::gemm::kernel
