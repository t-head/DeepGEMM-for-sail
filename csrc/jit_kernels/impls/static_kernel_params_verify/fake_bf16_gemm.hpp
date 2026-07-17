namespace deep_gemm_bf16_common {
#define FP8_HGRTC
#include <../../../deep_gemm/include/deep_gemm/bf16_gemm_cutlass3.cuh>
namespace deep_gemm {
using namespace cute;
using cutlass::KernelHardwareInfo;
static size_t get_bf16_tample_params_size() {
  constexpr int SHAPE_N = 256;
  constexpr int SHAPE_K = 256;
  constexpr int BLOCK_M = 64;
  constexpr int BLOCK_N = 64;
  constexpr int BLOCK_K = 64;
  constexpr int NUM_GROUPS = 1;
  constexpr int WARP_M = 32;
  constexpr int WARP_N = 32;
  constexpr int STAGES = 2;
  static constexpr KernelType kKernelType = KernelType::Default;
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
  using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
  static constexpr int WarpOnM = BLOCK_M / WARP_M;
  static constexpr int WarpOnN = BLOCK_N / WARP_N;
  static constexpr bool kEnableSboOverlap = false;

  using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, cutlass::bfloat16_t, cutlass::bfloat16_t, float>::type;
  using TiledMma = TiledMMA<
      MMA_Atom<MmaInst>,
      Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
      Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _16>>;       // 1x1x1 value group

  static constexpr int N_EXPAND = 1;//kKernelType == KernelType::MultistageOnN && (SHAPE_N % (BLOCK_N) == 0) ? KernelAiuMultistageOnN::N_EXPAND : 1;
  using KernelSchedule = cute::conditional_t<
      kKernelType == KernelType::OverlapMainloop,
      KernelAiuMultistageOverlapMainloop,
      cute::conditional_t<
        kKernelType == KernelType::OverlapPrologue,
        KernelAiuMultistageOverlapPrologue,
        cute::conditional_t<
          kKernelType == KernelType::MultistageOnN,
          KernelAiuMultistageOnN,
          cutlass::gemm::KernelAiuMultistage>>>;
  using DispatchPolicy = cute::conditional_t<
      kKernelType == KernelType::OverlapMainloop,
      cutlass::gemm::MainloopPPUOverlapMainloop<STAGES, KernelSchedule>,
      cute::conditional_t<
        kKernelType == KernelType::OverlapPrologue,
        cutlass::gemm::MainloopPPUOverlapPrologue<STAGES, KernelSchedule>,
        cutlass::gemm::MainloopPPUAiuOpt<STAGES, KernelSchedule>>>;

  static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  static constexpr int TSM_LD_NUM = BLOCK_M == 8 ? 2 : 4;

  static constexpr int SmemLayoutStageStrideA = kKernelType == KernelType::OverlapMainloop || kKernelType == KernelType::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_M * BLOCK_K;
  static constexpr int SmemLayoutStageStrideB = kKernelType == KernelType::OverlapMainloop || kKernelType == KernelType::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_N * BLOCK_K;
  using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
  using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;
  // using t1 = DefaultOperandB::xhzhao;
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

  using TileScheduler = DeepGemmScheduler<GemmType::DenseGemm, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, NUM_GROUPS>;
  using GemmKernel = cutlass::gemm::kernel::DeepGemmUniversal<
      cutlass::bfloat16_t,
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler,
      kEnableSboOverlap>;

  return sizeof(GemmKernel::Params);
  
  }
}
}
