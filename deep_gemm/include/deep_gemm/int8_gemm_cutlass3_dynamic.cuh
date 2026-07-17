#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "ppu_include.hpp"


namespace cutlass::gemm {

constexpr static int TileLength = 5;
constexpr static int ElementsPerTile = 6; // blockM, blockN, warpM, warpN, blockK, stages

struct KernelAiuDynamicTileLargeK {
  constexpr static int DynamicTildId = 0;
  constexpr static int TileConfigList[TileLength][ElementsPerTile] = {
    {16, 128, 16, 16, 256, 3},
    {32, 128, 16, 32, 256, 3},
    {64, 256, 32, 64, 128, 3},
    {128, 128, 32, 64, 128, 3},
    {192, 128, 48, 64, 128, 3}
  };
};

struct KernelAiuDynamicTileSmallK {
  constexpr static int DynamicTildId = 1;
  constexpr static int TileConfigList[TileLength][ElementsPerTile] = {
    {16, 256, 16, 32, 128, 3},
    {32, 256, 32, 32, 128, 3},
    {64, 256, 32, 64, 128, 3},
    {128, 128, 32, 64, 128, 3},
    {192, 128, 48, 64, 128, 3}
  };
};

struct KernelAiuDynamicTileLargeEM {
  constexpr static int DynamicTildId = 2;
  constexpr static int TileConfigList[TileLength][ElementsPerTile] = {
    {128, 256, 32, 64, 128, 5},
    {192, 256, 48, 64, 128, 4},
    {256, 256, 64, 64, 128, 4},
    {256, 256, 64, 64, 128, 4},
    {256, 256, 64, 64, 128, 4}
  };
};

struct KernelAiuDynamicTileTPLargeK {
  constexpr static int DynamicTildId = 10;
  constexpr static int TileConfigList[TileLength][ElementsPerTile] = {
    {16, 256, 16, 32, 128, 3},
    {32, 256, 32, 32, 128, 3},
    {64, 256, 32, 64, 128, 3},
    {96, 256, 48, 64, 128, 2},
    {128, 256, 64, 64, 128, 2}
  };
};

struct KernelAiuDynamicTileTPSmallK {
  constexpr static int DynamicTildId = 11;
  constexpr static int TileConfigList[TileLength][ElementsPerTile] = {
    {16, 128, 16, 32, 128, 3},
    {32, 128, 32, 32, 128, 3},
    {64, 128, 32, 64, 128, 2},
    {96, 128, 48, 64, 128, 2},
    {128, 128, 64, 64, 128, 2}
  };
};

}

namespace cutlass::gemm::kernel {


template <
  GemmType kGemmType,
  typename ElementA,
  typename ElementB,
  typename ElementD,
  typename ElementAcc,
  typename ElementCompute,
  int SHAPE_N,
  int SHAPE_K,
  int kNumGroups,
  typename KernelAiuDynamicTile,
  int TileId
>
struct PPUTypeBuilder {
  using ElementC = ElementD;
  using LayoutA     = cutlass::layout::RowMajor;
  using LayoutB     = cutlass::layout::ColumnMajor;
  using LayoutC     = cutlass::layout::RowMajor;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  static constexpr bool TransA = false;
  static constexpr bool TransB = false;
  using ArchTag = cutlass::arch::PPU0015;

  using ProblemShape = Shape<int,int,int,int>;

  static constexpr int BLOCK_M = KernelAiuDynamicTile::TileConfigList[TileId][0];
  static constexpr int BLOCK_N = KernelAiuDynamicTile::TileConfigList[TileId][1];
  static constexpr int WARP_M = KernelAiuDynamicTile::TileConfigList[TileId][2];
  static constexpr int WARP_N = KernelAiuDynamicTile::TileConfigList[TileId][3];
  static constexpr int BLOCK_K = KernelAiuDynamicTile::TileConfigList[TileId][4];
  static constexpr int kNumStages = KernelAiuDynamicTile::TileConfigList[TileId][5];

  using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
  using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
  static constexpr int WarpOnM = BLOCK_M / WARP_M;
  static constexpr int WarpOnN = BLOCK_N / WARP_N;

  using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementA,ElementB,ElementAcc>::type;
  using TiledMma = TiledMMA<
      MMA_Atom<MmaInst>,
      Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
      Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _32>>;       // 1x1x1 value group

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

  using KernelSchedule = cutlass::gemm::KernelAiuMultistage;
  using DispatchPolicy = cutlass::gemm::MainloopPPUAiuA8W8<kNumStages, KernelSchedule>;

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
  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
      cutlass::detail::TagToStrideA_t<LayoutC>,
      cutlass::detail::TagToStrideA_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
      cutlass::gemm::EpilogueDefault,
      IsAligedN>;

  using TileScheduler = ::deep_gemm::DeepGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups>;

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

  template<
  typename Params
  >
  static CUTLASS_DEVICE void run(Params params, char* smem_buf, int M, int m_coord, int n_coord,
                          int64_t offset_a, int64_t offset_b, int64_t offset_m, int64_t offset_c) {
    int thread_idx = int(threadIdx.x);
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto problem_shape_MNKL = ProblemShape{M, SHAPE_N, SHAPE_K, 1};
    auto blk_shape = TileShape{};

    typename CollectiveMainloop::Arguments args_mainloop{params.ptr_A, params.stride_A, params.ptr_B, params.stride_B,
                                                          params.ptr_scale_A, params.ptr_scale_B};

    const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.ptr_A) + offset_a;
    const ElementB* ptr_B = reinterpret_cast<const ElementB*>(params.ptr_B) + offset_b;

    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, 0);
    CollectiveMainloop collective_mma(args_mainloop, take<0, 3>(problem_shape_MNKL));

    auto load_inputs = collective_mma.load_init(problem_shape_MNKL, blk_coord_mnkl, args_mainloop,
                                                M, offset_m, offset_b, ptr_A, ptr_B);
    // Extract out partitioned A and B.
    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    // Compute tile residues for predication
    auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
    auto n_max_coord = SHAPE_N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
    auto k_residue   = SHAPE_K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
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
    auto params_epilogue_local = *(reinterpret_cast<typename CollectiveEpilogue::Params*>(&(params.epi_params)));
    params_epilogue_local.ptr_D += offset_c;

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
  typename ElementD,
  typename ElementAcc,
  typename ElementCompute,
  int SHAPE_N,
  int SHAPE_K,
  int kNumGroups,
  bool kLargeEM
>
struct DeepGemmDynamicTile {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
  using StrideD = cutlass::detail::TagToStrideC_t<LayoutD>;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  using ElementScale = float;
  using ProblemShape = Shape<int,int,int,int>;

  static constexpr bool IsEP = (kGemmType == GemmType::GroupedMasked);

  using KernelAiuDynamicTile = typename cutlass::platform::conditional<
          IsEP,
          typename cutlass::platform::conditional<
            kLargeEM,
            KernelAiuDynamicTileLargeEM,
            typename cutlass::platform::conditional<
              (SHAPE_K > 2048),
              KernelAiuDynamicTileLargeK,
              KernelAiuDynamicTileTPSmallK>::type
            >::type,
          typename cutlass::platform::conditional<
            (SHAPE_K > 384),
            KernelAiuDynamicTileTPLargeK,
            KernelAiuDynamicTileTPSmallK>::type
          >::type;

  using Builder0 = PPUTypeBuilder<kGemmType, ElementA, ElementB, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuDynamicTile, 0>;
  using Builder1 = PPUTypeBuilder<kGemmType, ElementA, ElementB, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuDynamicTile, 1>;
  using Builder2 = PPUTypeBuilder<kGemmType, ElementA, ElementB, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuDynamicTile, 2>;
  using Builder3 = PPUTypeBuilder<kGemmType, ElementA, ElementB, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuDynamicTile, 3>;
  using Builder4 = PPUTypeBuilder<kGemmType, ElementA, ElementB, ElementD, ElementAcc, ElementCompute,
                                  SHAPE_N, SHAPE_K, kNumGroups, KernelAiuDynamicTile, 4>;
  static constexpr bool kEnableNExpand = !(Builder4::BLOCK_N == Builder0::BLOCK_N
                                           && Builder4::BLOCK_N == Builder1::BLOCK_N
                                           && Builder4::BLOCK_N == Builder2::BLOCK_N
                                           && Builder4::BLOCK_N == Builder3::BLOCK_N);
  static constexpr uint32_t MaxThreadsPerBlock = Builder4::MaxThreadsPerBlock;

  using CollectiveEpilogue = typename Builder4::CollectiveEpilogue;
  using TileScheduler = typename Builder4::TileScheduler;

  struct Arguments {
    // arguments for mainloop
    ElementA const* ptr_A;
    StrideA stride_A;
    ElementB const* ptr_B;
    StrideB stride_B;
    ElementScale const* ptr_scale_A;
    ElementScale const* ptr_scale_B;
    // arguments for epilogue
    typename CollectiveEpilogue::Params epi_params;
    // arguments for scheduler
    uint32_t shape_m;
    int* grouped_layout;
  };

  using Params = Arguments;

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
    int thread_idx = int(threadIdx.x);
    TileScheduler deep_scheduler({params.shape_m, params.grouped_layout});

    uint32_t m_block_idx, n_block_idx;
    while (deep_scheduler.template fetch_next_work_dynamic_tile<kEnableNExpand>(m_block_idx, n_block_idx)) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx;

      uint32_t M = deep_scheduler.curr_problem_m();
      auto offset_m = deep_scheduler.curr_offset_m();
      auto expert_id = deep_scheduler.problem_index();
      auto offset_a = deep_scheduler.curr_offset_a();
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx);
      auto offset_c = deep_scheduler.curr_offset_c();
      if (M <= Builder0::BLOCK_M) {
        Builder0::run(params, smem_buf, M, m_coord, n_coord, offset_a, offset_b, offset_m, offset_c);
      } else if (M <= Builder1::BLOCK_M) {
        Builder1::run(params, smem_buf, M, m_coord, n_coord, offset_a, offset_b, offset_m, offset_c);
      } else if (M <= Builder2::BLOCK_M) {
        Builder2::run(params, smem_buf, M, m_coord, n_coord, offset_a, offset_b, offset_m, offset_c);
      } else if (M <= Builder3::BLOCK_M){
        Builder3::run(params, smem_buf, M, m_coord, n_coord, offset_a, offset_b, offset_m, offset_c);
      } else {
        Builder4::run(params, smem_buf, M, m_coord, n_coord, offset_a, offset_b, offset_m, offset_c);
      }

    }

  }

};

} // end of cutlass::gemm::kernel
