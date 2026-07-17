#define FP8_HGRTC
#include <../../../deep_gemm/include/deep_gemm/fp8_gemm.cuh>
namespace deep_gemm {
using namespace cute;
using cutlass::KernelHardwareInfo;
static size_t get_fp8_tample_params_size() {
  constexpr int SHAPE_N = 256;
  constexpr int SHAPE_K = 256;
  constexpr int BLOCK_M = 64;
  constexpr int BLOCK_N = 64;
  constexpr int BLOCK_K = 64;
  constexpr int NUM_GROUPS = 1;
  constexpr int WARP_M = 32;
  constexpr int WARP_N = 32;
  constexpr int STAGES = 2;
  using ScaleGranularityShape = cute::Shape<cute::_1,cute::_128,cute::_128>;
  using ScaleConfig         = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(ScaleGranularityShape{}));
  using LayoutSFA           = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
  using LayoutSFB           = decltype(ScaleConfig::deduce_layoutSFB());
  static constexpr bool kEnableMultistageOnN = false;
  static constexpr bool kUseNStageKernel = SHAPE_K <= 512 && (SHAPE_N % (BLOCK_N * KernelAiuMultistageOnN::N_EXPAND) == 0) && (BLOCK_K == 128) && STAGES == 2;

  constexpr int N_EXPAND = kUseNStageKernel ? 4 : 1;

  using TileScheduler = DeepGemmScheduler<
    GemmType::DenseGemm, 
    SHAPE_N, 
    SHAPE_K, 
    BLOCK_M, 
    BLOCK_N * N_EXPAND, 
    NUM_GROUPS
  >;

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

  struct Debug_CollectiveMainloop; 
  using ArchTag = cutlass::arch::PPU0015;

  // Core kernel configurations
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ElementCompute      = float;                                          // Element type for epilogue computation
  using ElementScalar    = ElementCompute;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,  // ElementAccumulator
    Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>,
    Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>,
    Int<STAGES>,
    cutlass::gemm::KernelAiuMultistageWithBlockWiseScale
  >::CollectiveOp;

  using EpilogueDispatchPolicy = cutlass::epilogue::EpilogueSimtVectorized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using CollectiveEpilogueWithTsm = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>, 
      Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>,
      EpilogueTileType,
      ElementCompute, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueDispatchPolicy
    >::CollectiveOp;

  // Epilogue
  static constexpr bool kEnableSboOverlap = false;
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

  using GemmKernel = DeepGemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler,
    kEnableSboOverlap,
    kUseNStageKernel
  >;

  return sizeof(GemmKernel::Params);
  
  }
}
