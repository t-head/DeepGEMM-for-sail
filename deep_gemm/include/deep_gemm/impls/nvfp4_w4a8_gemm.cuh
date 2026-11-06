#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wcuda-compat"
#pragma clang diagnostic ignored "-Wswitch"
#include <cute/ppu_tensor_mix.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/atom/mma_traits_ppu0015.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0015_aiu.hpp>
#include <cute/algorithm/ppu_copy.hpp>
#include <cutlass/fast_numeric_conversion_for_mix_gemm.h>
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/scheduler/fused_scheduler.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/nvfp4_decode.cuh>
#include <deep_gemm/common/nvfp4_gemm_traits.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/tix/cvt.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>

using namespace cute;

namespace deep_gemm::nvfp4::detail {
// NVFP4 warp-K tensor reshaping, epilogue layout and BF16 output conversion.
template <int MmaK, int BlockK, int WarpsOnK, int mode, typename Tensor>
CUTLASS_DEVICE constexpr auto w4_reshape_warpk(Tensor& t) {
  using namespace cute;
  static_assert(mode == 0 || mode == 1);
  if constexpr (mode == 0) {
    constexpr int k = decltype(size<mode>(t))::value;
    constexpr int ko = k / WarpsOnK;
    auto atom_tiler = make_layout(make_shape(Int<WarpsOnK>{}, Int<ko>{}), make_stride(Int<ko>{}, _1{}));
    auto tiler = make_shape(atom_tiler, size<1>(t));
    return flat_divide(t, tiler)(_,_,0,0,_);
  } else {
    constexpr int ko = BlockK / MmaK / WarpsOnK;
    auto atom_tiler = make_layout(make_shape(Int<MmaK>{}, make_shape(Int<WarpsOnK>{}, Int<ko>{})),
                                 make_stride(_1{}, make_stride(Int<ko * MmaK>{}, Int<MmaK>{})));
    auto tiler = make_shape(size<0>(t), atom_tiler);
    return flat_divide(t, tiler)(_,_,0,0,_);
  }
}

template <typename ElementD, int BlockM, int BlockN, int WarpM, int Threads>
struct W4EpilogueLayout {
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using CopyOp = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
  using Config = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<
      CopyOp, ElementD, AlignmentD, cute::Int<BlockM>, cute::Int<BlockN>, cute::Int<BlockM / WarpM>, Threads>;
  using SmemLayout = typename Config::SmemLayoutO;
  using CopyAtomR2S = cute::Copy_Atom<CopyOp, ElementD>;
  using TiledCopyS2R = typename Config::GmemTiledCopyO;
  using CopyAtomR2G = cute::Copy_Atom<CopyOp, ElementD>;
};
} // namespace deep_gemm::nvfp4::detail

namespace cutlass::gemm::kernel {

using deep_gemm::GemmType;
using deep_gemm::EpilogueType;

using deep_gemm::nvfp4::detail::kNvfp4E4m3ScaleLut;

// Unit-scale E2M1 expansion for the exact K16 partial-sum path.
struct E2m1ToE4m3Converter {
  using result_type = cutlass::Array<cutlass::float_e4m3_t, 8>;
  using source_type = cutlass::Array<int4_t, 8>;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    // Magnitudes retain four-nibble PRMT selector layout. Sign bits are
    // cross-packed offline: weights 0..3 occupy byte MSBs and weights 4..7
    // occupy bits [3,11,19,27]. This halves decoder PRMT issue while keeping
    // exactly four storage bits per E2M1 value.
    result_type result;
    uint32_t* dst = reinterpret_cast<uint32_t*>(&result);
    const uint32_t packed = reinterpret_cast<uint32_t const&>(source);
    constexpr uint64_t scale_lut = 0x4c4844403c383000ull;
    const uint32_t scaled_lut0 = uint32_t(scale_lut);
    const uint32_t scaled_lut1 = uint32_t(scale_lut >> 32);
    dst[0] = __byte_perm(
        scaled_lut0, scaled_lut1, packed & 0x7777u) |
        (packed & 0x80808080u);
    dst[1] = __byte_perm(
        scaled_lut0, scaled_lut1, (packed >> 16) & 0x7777u) |
        ((packed & 0x08080808u) << 4);
    return result;
  }
};

// Adjacent K16 halves have independent E4M3 scales. Decode each half
// directly with its rounded magnitude table, retaining the offline sign layout.
struct Nvfp4E2m1ToE4m3Converter {
  using result_type = E2m1ToE4m3Converter::result_type;
  using source_type = E2m1ToE4m3Converter::source_type;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source, uint64_t lut_lo, uint64_t lut_hi) {
    const uint32_t packed = reinterpret_cast<uint32_t const&>(source);
    result_type result;
    auto dst = reinterpret_cast<uint32_t*>(&result);
    dst[0] = __byte_perm(uint32_t(lut_lo), uint32_t(lut_lo >> 32), packed & 0x7777u)
             | (packed & 0x80808080u);
    dst[1] = __byte_perm(uint32_t(lut_hi), uint32_t(lut_hi >> 32), (packed >> 16) & 0x7777u)
             | ((packed & 0x08080808u) << 4);
    return result;
  }

  CUTLASS_DEVICE
  static result_type convert(source_type const& source, uint8_t scale_lo,
                             uint8_t scale_hi, uint64_t const* scale_lut_table) {
    // Preprocessing validates nonnegative finite scales; -0 is valid too.
    return convert(source, scale_lut_table[scale_lo & 0x7fu],
                   scale_lut_table[scale_hi & 0x7fu]);
  }
};

template <int ShapeN, int ShapeK,
          int BlockM, int BlockN, int BlockK,
          int WarpM, int WarpN, int WarpK,
          int kNumGroups, int kNumStages, GemmType kGemmType,
          int kGroupSize = 16, int N_EXPAND = 1,
          int kAScaleGroup = 0,
          bool kHasAScale = false, bool kHasWeightGlobalScale = false,
          bool kHasBScale = false, int kBScaleBlock = 0>
class W4A8GEMM {
public:
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = int4_t;
  using ElementScale = uint8_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAcc = float;

  // basic
  static constexpr bool Fused = (kGemmType == GemmType::MGroupedFused);
  static constexpr int N = ShapeN;
  static constexpr int K = ShapeK;
  static constexpr int L = kNumGroups;
  static_assert(ShapeN % 64 == 0, "W4A8 requires N to be a multiple of 64.");
  static_assert(ShapeK % 32 == 0, "NVFP4 requires K to be a multiple of 32.");
  // Now only support WarpK >= groupsize, TODO: group_size > WarpK
  static_assert(WarpK >= kGroupSize && WarpK % kGroupSize == 0);
  static constexpr int GroupsPerBlock = BlockK / kGroupSize;
  static constexpr int GroupsPerWarp = WarpK / kGroupSize;
  static_assert(kGroupSize == 16, "Invalid FP4 scale group size.");
  static_assert(kAScaleGroup == 0 || kAScaleGroup == 16 || kAScaleGroup % 32 == 0);
  static_assert(kAScaleGroup == 0 || kHasAScale, "Blockwise activation scales are required.");
  static constexpr bool GroupAScale = kAScaleGroup >= 32;
  static_assert(kBScaleBlock == 0 || kBScaleBlock == 128);
  static_assert(kBScaleBlock == 0 || kHasBScale, "Blockwise weight scales are required.");
  static constexpr bool BlockBScale = kBScaleBlock > 0;
  static constexpr bool GroupScale = GroupAScale || BlockBScale;
  static constexpr bool Nvfp8Decode = BlockBScale;
  // Each full K128 warp tile has one scale per row and per N128 block.
  // Keep those loads and accumulator promotion outside the MMA fragments.
  static constexpr bool FastBlockScale = deep_gemm::nvfp4::detail::use_fast_block_scales(
      K, BlockK, WarpK, N_EXPAND, kAScaleGroup, kBScaleBlock);
  static constexpr bool StageBlockScale = FastBlockScale && !Fused &&
      deep_gemm::nvfp4::detail::can_stage_block_scales(BlockM, BlockN, BlockK, WarpM);
  // Reserve a 128-byte row for B scales so the following AIU buffers keep
  // their alignment. A scales have one entry per CTA row and K128 stage.
  static constexpr int OuterScaleBytes = StageBlockScale
      ? (BlockM + 32) * kNumStages * sizeof(float) : 0;
  // M warp groups share each decoded FP8 B fragment. Keep this
  // specialization separate from fused routing and the legacy FP4 paths.
  static constexpr bool ShareDecodedB = StageBlockScale &&
      deep_gemm::nvfp4::detail::is_shared_decode_tile(BlockM, BlockN, BlockK, WarpM, WarpN, WarpK);
  // Consume one K32 operand fragment before replacing its registers. Shared
  // decoding retains its existing cross-warp prefetch schedule.
  static constexpr bool RollingOperands = FastBlockScale && !ShareDecodedB && !Fused &&
      BlockK == 128 && WarpK == 128 && BlockM == WarpM && WarpM <= 64;
  static constexpr bool PrefetchFinalScales = N_EXPAND == 1 && !ShareDecodedB && !RollingOperands &&
      (K + BlockK - 1) / BlockK > kNumStages;
  static constexpr int DecodedBytes = ShareDecodedB ? BlockN * BlockK : 0;
  static constexpr bool SplitMma = (!Nvfp8Decode) || kAScaleGroup == 16;
  static constexpr int ScaleBytesPerStage = GroupsPerBlock * BlockN;
  static constexpr int ScaleGroupsPerMma = 32 / kGroupSize;
  using TileShape = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
  static constexpr int N_Blocks = deep_gemm::math::ceil_div(ShapeN, BlockN * N_EXPAND);
  static constexpr int scheduler_swizzle = (BlockM == 16) ? 1: 2;
  using TileScheduler = cute::conditional_t<
    Fused,
    deep_gemm::FusedGemmScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN * N_EXPAND, kNumGroups, N_Blocks, scheduler_swizzle>,
    deep_gemm::DeepGemmScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN * N_EXPAND, kNumGroups, N_Blocks, scheduler_swizzle, EpilogueType::Default>
  >;
#if __HGGC_ARCH__ == 100
    using ArchTag = cutlass::arch::PPU0010;
#else
    using ArchTag = cutlass::arch::PPU0015;
#endif

  // mma
  using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementA, ElementA, ElementAcc>::type;
  static constexpr int MmaM = get<0>(typename MMA_Atom<MmaInst>::Shape_MNK{});
  static constexpr int WarpsOnM = BlockM / WarpM;
  static constexpr int WarpsOnN = BlockN / WarpN;
  static constexpr int MmaNPerWarp = WarpN / 16;
  static constexpr int WarpsOnK = BlockK / WarpK;
  static_assert((WarpsOnK & (WarpsOnK - 1)) == 0,
                "K-warp reduction requires a power of two number of warp groups.");
  using PermutationMNK = Tile<
    Int<WarpsOnM * MmaM>,
    Layout<Shape<_16, Int<WarpsOnN>, Int<MmaNPerWarp>>, Stride<_1, Int<WarpN>, _16>>,
    // E4M3 x E4M3 MMA consumes K=32 per instruction (BF16 uses K=16).
    Int<WarpsOnK * 32>,
  >;
  using TiledMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<Int<WarpsOnM>, Int<WarpsOnN>, Int<WarpsOnK>>>, PermutationMNK>;
  // M warps interleave one MMA fragment each before moving to the next.
  // Tail dispatch retains every shared-decode and barrier participant.
  static constexpr bool TrimM = ShareDecodedB;
  static constexpr int MFragmentsPerWarp = WarpM / MmaM;
  static constexpr int RowsPerMFragment = WarpsOnM * MmaM;
  static constexpr int MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr int MinBlocksPerMultiprocessor = ShareDecodedB ? 2 : 1;
  static constexpr bool SplitAIU = (MaxThreadsPerBlock / NumThreadsPerWarp) > 1;

  // sA
  static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
  using ACopyInst = cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t>;
  using DefaultOperandA = cute::conditional_t<
    Fused,
    cutlass::gemm::config::DefaultGemm_TensorOpPPU_Operand<ElementA, false, AlignA, Int<BlockK>, MaxThreadsPerBlock, ACopyInst>,
    cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, false, Int<BlockM>, Int<BlockK>, false, 0, true>
  >;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
  using TilerA = typename GmemTiledCopyA::Tiler_MN;
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<kNumStages>{})));

  // sB
  using TileB = Shape<Int<BlockK / 32>, Int<BlockN * 4>>;
  using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, int, false, decltype(get<0>(TileB{})), decltype(get<1>(TileB{})), false, 0, false>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(get<0>(TileB{}), get<1>(TileB{}), Int<kNumStages>{})));
  // custom s2r copy B
  using BCopyOp = UniversalCopy<cute::uint128_t>;
  using BCopyTraits = Copy_Traits<BCopyOp>;
  using SmemCopyAtomB = Copy_Atom<BCopyTraits, int>;
  using SmemThrLayoutB64 = Layout<
    Shape<Int<WarpsOnK>, Shape<_32, Int<WarpsOnN>>>,
    Stride<Int<32 * WarpsOnN>, Stride<_1, _32>>
  >;
  // A pair of WarpN32 warps split the eight packed words that one WarpN64
  // warp owns. Logical order is (subwarp, lane, 64-column warp pair), while
  // physical thread IDs remain lane + 32 * subwarp + 64 * pair.
  using SmemThrLayoutB32 = Layout<
    Shape<Int<WarpsOnK>, Shape<_2, _32, Int<WarpsOnN / 2>>>,
    Stride<Int<32 * WarpsOnN>, Stride<_32, _1, _64>>
  >;
  using SmemThrLayoutB = cute::conditional_t<
      WarpN == 64, SmemThrLayoutB64, SmemThrLayoutB32>;
  using SmemValLayoutB = Layout<Shape<_1, Int<WarpN / 8>>>;
  using SmemTiledCopyB = decltype(make_tiled_copy(SmemCopyAtomB{}, SmemThrLayoutB{}, SmemValLayoutB{}));

  // epilogue cta reduce
  static_assert(WarpsOnM == 1 || WarpsOnK == 1); // TODO: support WarpsOnM >= 1 reduce.
  using SmemLayoutReduce = decltype(make_layout(make_shape(Int<16 * 64>{}, Int<WarpsOnN>{}, Int<WarpsOnK>{}), LayoutLeft{}));

  // Shared epilogue geometry; arithmetic/store policies remain dtype-specific.
  using EpilogueLayout = deep_gemm::nvfp4::detail::W4EpilogueLayout<ElementD, BlockM, BlockN, WarpM, MaxThreadsPerBlock / WarpsOnK>;
  static constexpr int AlignmentD = EpilogueLayout::AlignmentD;
  using EpilogueCopyOp = typename EpilogueLayout::CopyOp;
  using EpilogueConfig = typename EpilogueLayout::Config;
  using SmemLayoutC = typename EpilogueLayout::SmemLayout;
  using CopyAtomR2S = typename EpilogueLayout::CopyAtomR2S;
  using TiledCopyS2R = typename EpilogueLayout::TiledCopyS2R;
  using CopyAtomR2G = typename EpilogueLayout::CopyAtomR2G;

  struct SharedStorage {
    // Keep magnitude tables CTA-local when lanes select different scale
    // rows. Pad their storage to preserve 128-byte AIU destination alignment.
    cute::array_aligned<uint64_t, (Nvfp8Decode ? 128 : 32) + (OuterScaleBytes + DecodedBytes) / 8> smem_lut;
    union {
      struct {
        cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
        cute::array_aligned<int, cute::cosize_v<SmemLayoutB>> smem_b;
        cute::array_aligned<ElementScale, ScaleBytesPerStage * kNumStages>
            smem_scale;
      }; // mainloop
      cute::array_aligned<ElementAcc, cute::cosize_v<SmemLayoutReduce>> smem_reduce; // epilogue cta reduce
      cute::array_aligned<ElementD, cute::cosize_v<SmemLayoutC>> smem_c; // epilogue
    };
  };

  struct Params {
    const ElementA* ptr_a;
    const int* ptr_b;
    const ElementScale* ptr_scale;
    ElementD* ptr_d;
    int num_token;
    int topk;
    const int* sorted_token_ids;
    typename TileScheduler::Params scheduler;
    const float* ptr_a_scale;
    const float* ptr_weight_global_scale;
    int weight_global_scale_stride;
    const float* ptr_b_scale;

    CUTLASS_HOST
    Params(const ElementA* ptr_a_,
          const int* ptr_b_,
          const ElementScale* ptr_scale_,
          ElementD* ptr_d_)
        : ptr_a(ptr_a_),
          ptr_b(ptr_b_),
          ptr_scale(ptr_scale_),
          ptr_d(ptr_d_) {}
  };

  // for constexpr patial initialize
  struct NoTensor {};

  template <int Fragments, typename Function>
  CUTLASS_DEVICE __forceinline__ static void dispatch_m_fragments(int rows, Function& function) {
    if constexpr (Fragments == 1) function(Int<1>{});
    else if (rows > (Fragments - 1) * RowsPerMFragment) function(Int<Fragments>{});
    else dispatch_m_fragments<Fragments - 1>(rows, function);
  }

  template<int mode=0, typename Tensor>
  CUTLASS_DEVICE constexpr auto reshape_warpk(Tensor& t) {
    return deep_gemm::nvfp4::detail::w4_reshape_warpk<32, BlockK, WarpsOnK, mode>(t);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    int thread_idx = threadIdx.x;
    int warp_idx = canonical_warp_idx_sync();
    int warpm_idx = warp_idx % WarpsOnM;
    int warpn_idx = (warp_idx / WarpsOnM) % WarpsOnN;
    int warpk_idx = (warp_idx / WarpsOnM) / WarpsOnN;
    int lane_idx = thread_idx % NumThreadsPerWarp;
    TileScheduler deep_scheduler{params.scheduler};
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto staged_a_scales = [&]() {
      return reinterpret_cast<float*>(storage.smem_lut.data() + (Nvfp8Decode ? 128 : 32));
    };
    auto staged_b_scales = [&]() {
      return staged_a_scales() + BlockM * kNumStages;
    };
    auto decoded_b_buffer = [&]() {
      return reinterpret_cast<ElementA*>(storage.smem_lut.data() +
          (Nvfp8Decode ? 128 : 32) + OuterScaleBytes / 8);
    };
    if constexpr (Nvfp8Decode) {
      for (int i = thread_idx; i < 127; i += MaxThreadsPerBlock)
        storage.smem_lut[i] = kNvfp4E4m3ScaleLut[i];
    }
    __syncthreads();
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_K/32,BLK_N*4,PIPE)
    Tensor sReduce = make_tensor(make_smem_ptr<ElementAcc>(storage.smem_reduce.data()), SmemLayoutReduce{}); // (16*64,WARPS_ON_N,WARPS_ON_K)
    auto sReduce_tile = local_tile(sReduce, make_tile(_8{}), make_coord(_)); // (8,128,WARPS_ON_N,WARPS_ON_K)
    Tensor sC = make_tensor(make_smem_ptr(storage.smem_c.data()), SmemLayoutC{}); // (16*WARPS_ON_M,BLK_N)
    auto sC_tile = make_shape(size<0>(sC), size<1>(sC)); // (16*WARPS_ON_M,BLK_N)
    static_assert(size<1>(sC) == BlockN);

    // Global element offsets can exceed 32 bits even when every dimension fits int.
    auto strideA = make_stride(C<int64_t(K)>{}, _1{});
    auto strideB = make_stride(C<int64_t(N) * 4>{}, _1{}, C<int64_t(K) * N / 8>{});
    auto strideD = make_stride(C<int64_t(N)>{}, _1{});
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    gmem_tiled_copy_B.desc_.template init<int, false, get<0>(TileB{}), get<1>(TileB{})>(nullptr, K / 32, N * 4, strideB);
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto fragment_k = [](auto index) {
      if constexpr (RollingOperands) return Int<0>{};
      else return index;
    };
    auto tCrA = [&]() {
      auto full = thr_mma.partition_fragment_A(sA(_,_,0));
      if constexpr (RollingOperands)
        return make_tensor<typename decltype(full)::value_type>(make_shape(shape<0>(full), shape<1>(full), _1{}));
      else return full;
    }();
    auto tCrB = [&]() {
      auto full = make_tensor<typename TiledMma::FrgTypeB>(partition_shape_B(tiled_mma, take<1,3>(TileShape{})));
      if constexpr (RollingOperands)
        return make_tensor<typename decltype(full)::value_type>(make_shape(shape<0>(full), shape<1>(full), _1{}));
      else return full;
    }();
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    static_assert(decltype(size<1>(accum))::value == MFragmentsPerWarp,
                  "Unexpected M fragment layout.");
    auto scale_accum = [&]() {
      if constexpr (GroupScale) return make_tensor_like(accum);
      else return NoTensor{};
    }();

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto [smem_thr_copy_A, tCsA] = [&]() {
      if constexpr(Fused) {
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
        auto sA_r2s = reshape_warpk<1>(sA);
        auto tCsA = smem_thr_copy_A.partition_S(sA_r2s);
        return cute::make_tuple(smem_thr_copy_A, tCsA);
      } else {
        smem_tiled_copy_A.smem_base_ = storage.smem_a.data();
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
        auto mix_sA = make_mix_tensor_like(sA);
        auto sA_r2s = reshape_warpk<1>(mix_sA);
        auto tCsA = smem_thr_copy_A.partition_S(sA_r2s);
        return cute::make_tuple(smem_thr_copy_A, tCsA);
      }
    }();
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA); // (CPY,CPY_M,CPY_K)
    constexpr int K_BLOCK_MAX = WarpK / 32;
    constexpr int RegisterKBlocks = RollingOperands ? 1 : K_BLOCK_MAX;
    constexpr int K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);
    static_assert(K_ATOM_PER_COPY == 1, "K_ATOM_PER_COPY should be 1.");

    SmemTiledCopyB smem_tiled_copy_B;
    int threadidx_nk = (warp_idx / WarpsOnM) * NumThreadsPerWarp + lane_idx;
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(threadidx_nk);
    Tensor sB_r2s = reshape_warpk<0>(sB);
    Tensor tCsB = smem_thr_copy_B.partition_S(sB_r2s); // (4,CPY_K,CPY_N,PIPE) CPY_N=1
    auto tCrB_copy_view = [&]() {
      auto full = make_tensor_like(tCsB(_,_,_,0));
      if constexpr (RollingOperands)
        return make_tensor<typename decltype(full)::value_type>(make_shape(shape<0>(full), _1{}, shape<2>(full)));
      else return full;
    }(); // (4,CPY_K,CPY_N)
    Tensor tCrB_i32 = make_tensor(
      tCrB_copy_view.data(),
      Layout<Shape<_2, Int<MmaNPerWarp>, Int<RegisterKBlocks>>, Stride<_1, _2, Int<2 * MmaNPerWarp>>>{}
    ); // (two packed words, N fragments owned by this warp, CPY_K)
    uint32_t m_block_idx, n_block_idx;
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {

      int coord_l = kGemmType == GemmType::Normal ? 0 : deep_scheduler.curr_group_idx;
      const int64_t scale_offset_l = int64_t(coord_l) * (K / kGroupSize) * N;
      n_block_idx *= N_EXPAND;

      Tensor mB_nkl = make_tensor(make_gmem_ptr(params.ptr_b), make_shape(Int<K / 32>{}, Int<N * 4>{}, Int<L>{}), strideB); // (k/32,n*4,l)
      Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,coord_l)); // (k/32,n*4)
      Tensor gB = local_tile(mB_nk, TileB{}, make_coord(_, n_block_idx)); // (BLK_K/32,BLK_N*4,blocks_k)
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

      Tensor cD = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
      Tensor cAccum = thr_mma.partition_C(cD);
      auto [blk_token_base, tAgA, residual_m, gD] = [&]() {
        if constexpr(!Fused) {
          int M = kGemmType == GemmType::Normal ? params.num_token : deep_scheduler.curr_group_m;
          gmem_tiled_copy_A.desc_.template init<ElementA, false, BlockM, BlockK>(nullptr, M, K, strideA);
          Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_a + deep_scheduler.curr_offset_a()), make_shape(M, Int<K>{}), strideA); // (m,k)
          Tensor mA_mk = make_mix_tensor_like(mA_mkl); // (m,k)
          Tensor gA = local_tile(mA_mk, TileShape{}, make_coord(m_block_idx, n_block_idx, _), Step<_1, X,_1>{}); // (BLK_M,BLK_K,blocks_k)
          Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
          Tensor mD_mn = make_tensor(make_gmem_ptr(params.ptr_d + deep_scheduler.curr_offset_c()), make_shape(M, Int<N>{}), strideD); // (m,n)
          Tensor gD = local_tile(mD_mn, TileShape{}, make_coord(m_block_idx, n_block_idx, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N)
          int residual_m = M - BlockM * m_block_idx;

          return cute::make_tuple(
              nullptr, tAgA, residual_m, gD);
        } else {
          const int* blk_token_base = params.sorted_token_ids + int64_t(deep_scheduler.cumsum_m_block_idx) * BlockM;
          return cute::make_tuple(
              blk_token_base, NoTensor{}, 0, NoTensor{});
        }
      }();

      // mainloop
      clear(accum);
      int k_tile_iter = 0;
      int k_tile_count = size<2>(gB);
      int smem_pipe_read  = 0;
      Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
      Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);
      int compute_k_tile = 0;
      const int first_n_block_idx = n_block_idx;
      int scale_n_block_idx = n_block_idx;

      auto g2s_copy_A = [&](int k_tile_iter, int k_pipe) {
        if constexpr(!Fused) { // aiu copy
          if (warp_idx == 0) {
            copy(gmem_tiled_copy_A, tAgA(_,_,_,k_tile_iter), tAsA(_,_,_,k_pipe));
          }
        } else { // async copy
          constexpr int tiler_m = get<0>(TilerA{});
          constexpr int tiler_k = get<1>(TilerA{});
          constexpr int NumThreads_CPY_K = tiler_k / AlignA;
          constexpr int M_ITER = deep_gemm::math::ceil_div(BlockM, tiler_m);
          constexpr int K_ITER = BlockK / tiler_k;
          static_assert(BlockK % tiler_k == 0);

          int tid_k = thread_idx % NumThreads_CPY_K;
          int tid_m = thread_idx / NumThreads_CPY_K;
          if (tid_m < BlockM) {
            CUTLASS_PRAGMA_UNROLL
            for(int m_iter = 0; m_iter < M_ITER; ++m_iter) {
              CUTLASS_PRAGMA_UNROLL
              for(int k_iter = 0; k_iter < K_ITER; ++k_iter) {
                int64_t m_offset = __ldg(blk_token_base + (m_iter * tiler_m) + tid_m);
                int k_offset = k_iter * tiler_k + tid_k * AlignA + k_tile_iter * BlockK;
                const cutlass::uint128_t& src = *reinterpret_cast<const cutlass::uint128_t*>(params.ptr_a + m_offset * K + k_offset);
                cutlass::uint128_t& dst = *reinterpret_cast<cutlass::uint128_t*>(raw_pointer_cast(tAsA(_,m_iter,k_iter,k_pipe).data()));
                bool valid = m_offset < params.num_token;
                if constexpr (K % BlockK != 0) valid = valid && k_offset < K;
                ACopyInst::copy(src, dst, valid);
              }
            }
          }
        }
      };

      auto g2s_copy_B = [&](int k_tile_iter, int k_pipe) {
        constexpr int copy_warp_idx = SplitAIU ? 1 : 0;
        if (warp_idx == copy_warp_idx) {
          copy(gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,k_pipe));
          if constexpr (N_EXPAND == 1) {
            constexpr int VectorsPerScaleRow = BlockN / 16;
            constexpr int ScaleVectorsPerStage =
                GroupsPerBlock * VectorsPerScaleRow;
            CUTLASS_PRAGMA_UNROLL
            for (int vector_idx = lane_idx;
                 vector_idx < ScaleVectorsPerStage; vector_idx += 32) {
              const int group = vector_idx / VectorsPerScaleRow;
              const int vector_n = vector_idx % VectorsPerScaleRow;
              const int64_t src_offset =
                  scale_offset_l +
                  int64_t(k_tile_iter * GroupsPerBlock + group) * N +
                  n_block_idx * BlockN + vector_n * 16;
              const cutlass::uint128_t& src =
                  *reinterpret_cast<const cutlass::uint128_t*>(
                      params.ptr_scale + src_offset);
              cutlass::uint128_t& dst =
                  *reinterpret_cast<cutlass::uint128_t*>(
                      storage.smem_scale.data() +
                      k_pipe * ScaleBytesPerStage + vector_idx * 16);
              ACopyInst::copy(src, dst,
                  k_tile_iter * GroupsPerBlock + group < K / kGroupSize &&
                  n_block_idx * BlockN + vector_n * 16 < N);
            }
          }
        }
      };

      auto g2s = [&](int k_tile_iter, int k_pipe) {
        g2s_copy_A(k_tile_iter, k_pipe);
        g2s_copy_B(k_tile_iter, k_pipe);
        if constexpr (StageBlockScale) {
          using ScaleCopy = cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<uint32_t>;
          CUTLASS_PRAGMA_UNROLL
          for (int chunk = 0; chunk < deep_gemm::math::ceil_div(BlockM, MaxThreadsPerBlock); ++chunk) {
            const int local_row = thread_idx + chunk * MaxThreadsPerBlock;
            if (local_row < BlockM) {
              const int64_t row = deep_scheduler.curr_offset_a() / K + m_block_idx * BlockM + local_row;
              const auto* src = params.ptr_a_scale + row * (K / 128) + k_tile_iter;
              auto* dst = staged_a_scales() + k_pipe * BlockM + local_row;
              ScaleCopy::copy(*reinterpret_cast<const uint32_t*>(src),
                              *reinterpret_cast<uint32_t*>(dst), local_row < residual_m);
            }
          }
          if (thread_idx < deep_gemm::math::ceil_div(BlockN, 128)) {
            constexpr int n_blocks = deep_gemm::math::ceil_div(N, 128);
            const int n = n_block_idx * BlockN / 128 + thread_idx;
            const auto* src = params.ptr_b_scale + (int64_t(coord_l) * n_blocks + n) * (K / 128) + k_tile_iter;
            auto* dst = staged_b_scales() + k_pipe * 32 + thread_idx;
            ScaleCopy::copy(*reinterpret_cast<const uint32_t*>(src),
                            *reinterpret_cast<uint32_t*>(dst), n < n_blocks);
          }
        }
      };

      auto s2r_prologue_impl = [&](auto active_m_fragments) {
        if constexpr (TrimM) {
          for_each(make_int_sequence<decltype(active_m_fragments)::value>{}, [&](auto mi) {
            copy(smem_tiled_copy_A, tCsA_p(_,mi,0), tCrA_copy_view(_,mi,0));
          });
        } else {
          copy(smem_tiled_copy_A, tCsA_p(_,_,0), tCrA_copy_view(_,_,0));
        }
        if constexpr (!ShareDecodedB) copy(smem_tiled_copy_B, tCsB_p(_,0,_), tCrB_copy_view(_,0,_));
      };

      auto s2r_prologue = [&]() {
        s2r_prologue_impl(Int<MFragmentsPerWarp>{});
      };

      auto update_s2r_source = [&](int smem_pipe_read) {
        tCsA_p = tCsA(_,_,_,smem_pipe_read);
        tCsB_p = tCsB(_,_,_,smem_pipe_read);
      };

      auto s2r_next_impl = [&](int k_block, auto active_m_fragments) {
        auto k_block_next = (k_block + 1) % K_BLOCK_MAX;
        if constexpr (TrimM) {
          for_each(make_int_sequence<decltype(active_m_fragments)::value>{}, [&](auto mi) {
            copy(smem_tiled_copy_A, tCsA_p(_,mi,k_block_next), tCrA_copy_view(_,mi,k_block_next));
          });
        } else {
          copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        }
        if constexpr (!ShareDecodedB) copy(smem_tiled_copy_B, tCsB_p(_,k_block_next,_), tCrB_copy_view(_,k_block_next,_));
      };

      auto s2r_next = [&](int k_block) {
        s2r_next_impl(k_block, Int<MFragmentsPerWarp>{});
      };

      // Keep the input pointer stable across output stores. Otherwise the
      // compiler reloads this Params field for each pair of output values.
      const float* __restrict__ a_scale_ptr = params.ptr_a_scale;
      auto activation_scale = [&](int row, int global_k) {
        if constexpr (!kHasAScale) return 1.0f;
        else {
          int64_t global_row;
          if constexpr (Fused) {
            if (row >= deep_scheduler.valid_m_in_block) return 1.0f;
            global_row = __ldg(blk_token_base + row);
            if (global_row >= params.num_token) return 1.0f;
          } else {
            if (row >= residual_m) return 1.0f;
            global_row = deep_scheduler.curr_offset_a() / K + m_block_idx * BlockM + row;
          }
          constexpr int columns = kAScaleGroup > 0 ? (K + kAScaleGroup - 1) / kAScaleGroup : 1;
          const int column = kAScaleGroup > 0 ? global_k / (kAScaleGroup > 0 ? kAScaleGroup : 1) : 0;
          return a_scale_ptr[global_row * columns + column];
        }
      };
      float weight_global_scale = 1.0f;
      if constexpr (kHasWeightGlobalScale) {
        weight_global_scale = params.ptr_weight_global_scale[coord_l * params.weight_global_scale_stride];
      }
      auto output_scale = [&](int row) {
        float scale = weight_global_scale;
        if constexpr (kHasAScale && kAScaleGroup == 0) scale *= activation_scale(row, 0);
        return scale;
      };
      // These are outer FP8 channel scales, independent of FP4 K-block scales.
      const float* __restrict__ b_scale_ptr = params.ptr_b_scale;
      auto channel_scale = [&](int col) {
        if constexpr (kHasBScale && !BlockBScale)
          return b_scale_ptr[int64_t(coord_l) * N + n_block_idx * BlockN + col];
        else
          return 1.0f;
      };
      auto block_weight_scale = [&](int col, int global_k) {
        if constexpr (BlockBScale) {
          const int global_n = scale_n_block_idx * BlockN + col;
          if (global_n >= N) return 1.0f;
          constexpr int n_blocks = (N + kBScaleBlock - 1) / kBScaleBlock;
          constexpr int k_blocks = (K + kBScaleBlock - 1) / kBScaleBlock;
          return b_scale_ptr[(int64_t(coord_l) * n_blocks + global_n / kBScaleBlock) * k_blocks
                             + global_k / kBScaleBlock];
        } else {
          return 1.0f;
        }
      };

      // Within each MMA fragment, a lane owns rows spaced eight apart.
      // Reuse their scales across all N fragments of this warp.
      constexpr auto scale_ref = TiledMma{}.get_thread_slice(_0{}).partition_C(
          make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{})));
      auto block_row_scales = [&]() {
        if constexpr (FastBlockScale) return cutlass::Array<float, BlockM / 8>{};
        else return NoTensor{};
      }();
      auto preload_block_scales = [&](int global_k, auto active_m_fragments) {
        if constexpr (FastBlockScale) {
          const int pipe = (global_k / 128) % kNumStages;
          const float sb = [&]() {
            if constexpr (StageBlockScale)
              return staged_b_scales()[pipe * 32 + warpn_idx * WarpN / 128];
            else
              return block_weight_scale(warpn_idx * WarpN, global_k);
          }();
          for_each(make_int_sequence<decltype(size(accum))::value>{}, [&](auto first) {
            constexpr int row_id = get<0>(scale_ref(first));
            constexpr bool first_in_row = [&]() {
              for (int j = 0; j < decltype(first)::value; ++j)
                if (get<0>(scale_ref(j)) == row_id) return false;
              return true;
            }();
            if constexpr (first_in_row && row_id < int(active_m_fragments) * RowsPerMFragment) {
              const int row = get<0>(cAccum(first));
              if constexpr (StageBlockScale)
                block_row_scales[row_id / 8] = sb * staged_a_scales()[pipe * BlockM + row];
              else
                block_row_scales[row_id / 8] = sb * activation_scale(row, global_k);
            }
          });
        }
      };
      auto promote_block_scales = [&](auto active_m_fragments) {
        if constexpr (FastBlockScale) {
          for_each(make_int_sequence<decltype(size(accum))::value>{}, [&](auto i) {
            constexpr int row_id = get<0>(scale_ref(i));
            if constexpr (row_id < int(active_m_fragments) * RowsPerMFragment)
              accum(i) += scale_accum(i) * block_row_scales[row_id / 8];
          });
        }
      };

      auto decoded_offset = [&](auto kb, auto ni) {
        return (((int(kb) * WarpsOnN + warpn_idx) * MmaNPerWarp + int(ni))
                * NumThreadsPerWarp + lane_idx) * 16;
      };
      auto decode_shared_tile = [&]() {
        if constexpr (ShareDecodedB) {
          // Each M warp group produces disjoint K32 fragments for all
          // consumers. The previous stage's last shared read precedes the
          // existing pipeline barrier, so this buffer is safe to overwrite.
          auto decode_fragment = [&](auto kb) {
              copy(smem_tiled_copy_B, tCsB_p(_,kb,_), tCrB_copy_view(_,kb,_));
              constexpr int ScalesPerLane = WarpN / 8;
              constexpr int WarpsPerScaleBlock = 64 / WarpN;
              const int scale_n = (warpn_idx / WarpsPerScaleBlock) * 64 +
                  8 * (lane_idx / 4) + (warpn_idx % WarpsPerScaleBlock) * ScalesPerLane;
              const auto* scales = storage.smem_scale.data() +
                  (compute_k_tile % kNumStages) * ScaleBytesPerStage +
                  int(kb) * ScaleGroupsPerMma * BlockN + scale_n;
              using PackedScales = cute::conditional_t<WarpN == 64, uint64_t, uint32_t>;
              const auto lo = *reinterpret_cast<const PackedScales*>(scales);
              const auto hi = *reinterpret_cast<const PackedScales*>(scales + BlockN);
              for_each(make_int_sequence<MmaNPerWarp>{}, [&](auto ni) {
                cutlass::Array<ElementA, 16> decoded;
                CUTLASS_PRAGMA_UNROLL
                for (int word = 0; word < 2; ++word) {
                  const auto& src = *reinterpret_cast<const cutlass::Array<ElementB, 8>*>(
                      &tCrB_i32(word, ni, kb));
                  auto& dst = *reinterpret_cast<cutlass::Array<ElementA, 8>*>(
                      reinterpret_cast<uint32_t*>(&decoded) + 2 * word);
                  const int shift = 8 * (2 * int(ni) + word);
                  dst = Nvfp4E2m1ToE4m3Converter::convert(src,
                      static_cast<uint8_t>(lo >> shift), static_cast<uint8_t>(hi >> shift),
                      storage.smem_lut.data());
                }
                *reinterpret_cast<cute::uint128_t*>(decoded_b_buffer() + decoded_offset(kb, ni)) =
                    *reinterpret_cast<const cute::uint128_t*>(&decoded);
              });
          };
          // Dispatch each producer once, then decode its K32 fragments
          // consecutively. This avoids alternating producer branches for
          // every fragment while preserving the shared-buffer layout.
          for_each(make_int_sequence<WarpsOnM>{}, [&](auto producer) {
            if (warpm_idx == int(producer)) {
              for_each(make_int_sequence<(K_BLOCK_MAX + WarpsOnM - 1) / WarpsOnM>{}, [&](auto j) {
                constexpr int kb = int(producer) + int(j) * WarpsOnM;
                if constexpr (kb < K_BLOCK_MAX) decode_fragment(Int<kb>{});
              });
            }
          });
          __syncthreads();
        }
      };
      auto load_decoded = [&](auto kb) {
        if constexpr (ShareDecodedB) {
          for_each(make_int_sequence<MmaNPerWarp>{}, [&](auto ni) {
            auto dst = tCrB(_,ni,kb);
            *reinterpret_cast<cute::uint128_t*>(dst.data()) =
                *reinterpret_cast<const cute::uint128_t*>(decoded_b_buffer() + decoded_offset(kb, ni));
          });
        }
      };

      auto load_final_scales = [&](auto k_block) {
        if constexpr (!PrefetchFinalScales || decltype(k_block)::value != K_BLOCK_MAX - 1) {
          return NoTensor{};
        } else {
          const auto* scales = storage.smem_scale.data() +
              (compute_k_tile % kNumStages) * ScaleBytesPerStage;
          const int group = warpk_idx * GroupsPerWarp + k_block * ScaleGroupsPerMma;
          if constexpr (Nvfp8Decode) {
            using PackedScales = cute::conditional_t<WarpN == 64, uint64_t, uint32_t>;
            constexpr int WarpsPerScaleBlock = 64 / WarpN;
            const int col = (warpn_idx / WarpsPerScaleBlock) * 64 + 8 * (lane_idx / 4) +
                (warpn_idx % WarpsPerScaleBlock) * (WarpN / 8);
            return cute::make_tuple(
                *reinterpret_cast<const PackedScales*>(scales + group * BlockN + col),
                *reinterpret_cast<const PackedScales*>(scales + (group + 1) * BlockN + col));
          } else {
            // Pack four E4M3 scale bytes per register, including both K16 halves.
            constexpr int Values = decltype(size<0>(accum))::value;
            cutlass::Array<uint32_t, (2 * Values * MmaNPerWarp + 3) / 4> packed;
            packed.fill(0);
            for_each(make_int_sequence<MmaNPerWarp>{}, [&](auto ni) {
              for_each(make_int_sequence<Values>{}, [&](auto i) {
                const int col = get<1>(cAccum(i,0,ni));
                const int swizzled_col = col / 64 * 64 + (col % 8) * 8 + (col % 64) / 8;
                for_each(make_int_sequence<2>{}, [&](auto half) {
                  constexpr int index = 2 * (int(ni) * Values + int(i)) + int(half);
                  const uint32_t bits = scales[(group + int(half)) * BlockN + swizzled_col];
                  packed[index / 4] |= bits << (8 * (index % 4));
                });
              });
            });
            return packed;
          }
        }
      };

      auto mix_gemm_impl = [&](auto k_block, auto active_m_fragments, const auto& final_scales) {
        if constexpr (FastBlockScale) {
          if constexpr (decltype(k_block)::value == 0) {
            for_each(make_int_sequence<decltype(active_m_fragments)::value>{}, [&](auto mi) {
              clear(scale_accum(_,mi,_));
            });
            preload_block_scales(compute_k_tile * BlockK +
                                 (WarpsOnK > 1 ? warpk_idx * WarpK : 0), active_m_fragments);
          }
        }

        // The 8x8 offline swizzle makes this warp lane's eight E4M3 scale bytes
        // contiguous.  Fetch them once and consume them with the corresponding
        // packed E2M1 words during register-only dequantization.
        const int local_scale_group =
            warpk_idx * GroupsPerWarp + k_block * ScaleGroupsPerMma;
        constexpr int ScalesPerLaneGroup = WarpN / 8;
        constexpr int WarpsPerScaleBlock = 64 / WarpN;
        const int scale_warp_block = warpn_idx / WarpsPerScaleBlock;
        const int scale_subwarp = warpn_idx % WarpsPerScaleBlock;
        const int local_scale_n =
            scale_warp_block * 64 + 8 * (lane_idx / 4) +
            scale_subwarp * ScalesPerLaneGroup;
        using PackedScales = cute::conditional_t<WarpN == 64, uint64_t, uint32_t>;
        PackedScales packed_scales = 0;
        PackedScales packed_scales_hi = 0;
        if constexpr (N_EXPAND == 1) {
          if constexpr (Nvfp8Decode && PrefetchFinalScales &&
                        decltype(k_block)::value == K_BLOCK_MAX - 1) {
            packed_scales = get<0>(final_scales);
            packed_scales_hi = get<1>(final_scales);
          } else {
            const int scale_pipe = compute_k_tile % kNumStages;
            packed_scales = *reinterpret_cast<const PackedScales*>(
                storage.smem_scale.data() + scale_pipe * ScaleBytesPerStage +
                local_scale_group * BlockN + local_scale_n);
            if constexpr (Nvfp8Decode) {
              packed_scales_hi = *reinterpret_cast<const PackedScales*>(
                  storage.smem_scale.data() + scale_pipe * ScaleBytesPerStage +
                  (local_scale_group + 1) * BlockN + local_scale_n);
            }
          }
        } else {
          const int scale_group =
              compute_k_tile * GroupsPerBlock + local_scale_group;
          const int scale_n_base =
              scale_n_block_idx * BlockN + local_scale_n;
          if (scale_n_base < N) {
            packed_scales = __ldg(reinterpret_cast<const PackedScales*>(
                params.ptr_scale + scale_offset_l +
                int64_t(scale_group) * N + scale_n_base));
            if constexpr (Nvfp8Decode) {
              packed_scales_hi = __ldg(reinterpret_cast<const PackedScales*>(
                  params.ptr_scale + scale_offset_l +
                  int64_t(scale_group + 1) * N + scale_n_base));
            }
          }
        }

        const int mma_global_k = compute_k_tile * BlockK + warpk_idx * WarpK + k_block * 32;
        float mma_weight_scale = 1.0f;
        if constexpr (Nvfp8Decode && !FastBlockScale) {
          bool flush = k_block == K_BLOCK_MAX - 1 || mma_global_k + 32 >= K
                       || (mma_global_k + 32) % kBScaleBlock == 0;
          if constexpr (GroupAScale) flush |= (mma_global_k + 32) % kAScaleGroup == 0;
          if (flush && mma_global_k < K)
            mma_weight_scale = block_weight_scale(warpn_idx * WarpN, mma_global_k);
        }

        // Issue independent table reads for both N fragments before MMA.
        // Keeping them in registers hides lookup latency without decoding B twice.
        constexpr bool PrefetchLut = FastBlockScale && !ShareDecodedB &&
            BlockM == 64 && BlockK == 128 && WarpN == 32 &&
            kGemmType == GemmType::MGroupedNoPad;
        auto prefetched_luts = [&]() {
          if constexpr (PrefetchLut) return cutlass::Array<uint64_t, 4 * MmaNPerWarp>{};
          else return NoTensor{};
        }();
        if constexpr (PrefetchLut) {
          for_each(make_int_sequence<MmaNPerWarp>{}, [&](auto ni) {
            CUTLASS_PRAGMA_UNROLL
            for (int word = 0; word < 2; ++word) {
              const int shift = 8 * (2 * int(ni) + word);
              prefetched_luts[4 * int(ni) + 2 * word] =
                  storage.smem_lut[uint8_t(packed_scales >> shift) & 0x7fu];
              prefetched_luts[4 * int(ni) + 2 * word + 1] =
                  storage.smem_lut[uint8_t(packed_scales_hi >> shift) & 0x7fu];
            }
          });
        }

        for_each(make_int_sequence<MmaNPerWarp>{}, [&](auto n_idx) {
          constexpr int NIdx = decltype(n_idx)::value;
          constexpr int InterleavedElems = 16;
          using DstArray = cutlass::Array<ElementA, InterleavedElems>;
          using SrcHalf = cutlass::Array<ElementB, 8>;
          using DstHalf = cutlass::Array<ElementA, 8>;
          DstArray decoded_b;
          if constexpr (ShareDecodedB) {
            auto src = tCrB(_,n_idx,fragment_k(k_block));
            *reinterpret_cast<cute::uint128_t*>(&decoded_b) =
                *reinterpret_cast<const cute::uint128_t*>(src.data());
          } else {
          CUTLASS_PRAGMA_UNROLL
          for (int word = 0; word < 2; ++word) {
            const SrcHalf& src_half = *reinterpret_cast<const SrcHalf*>(
                &tCrB_i32(word, n_idx, fragment_k(k_block)));
            DstHalf& dst_half = *reinterpret_cast<DstHalf*>(
                reinterpret_cast<uint32_t*>(&decoded_b) + 2 * word);
            const auto scale_lo = static_cast<uint8_t>(packed_scales >> (8 * (2 * NIdx + word)));
            if constexpr (Nvfp8Decode) {
              const auto scale_hi = static_cast<uint8_t>(packed_scales_hi >> (8 * (2 * NIdx + word)));
              if constexpr (PrefetchLut) {
                dst_half = Nvfp4E2m1ToE4m3Converter::convert(
                    src_half, prefetched_luts[4 * NIdx + 2 * word],
                    prefetched_luts[4 * NIdx + 2 * word + 1]);
              } else {
                dst_half = Nvfp4E2m1ToE4m3Converter::convert(
                    src_half, scale_lo, scale_hi, storage.smem_lut.data());
              }
            } else {
              dst_half = E2m1ToE4m3Converter::convert(
                  src_half);
            }
          }
          }
          auto decoded_b_fragment = make_tensor(
              decoded_b.data(), tCrB(_,n_idx,fragment_k(k_block)).layout());
          CUTLASS_PRAGMA_UNROLL
          for (int m_idx = 0; m_idx < int(active_m_fragments); m_idx++) {
            if constexpr (FastBlockScale) {
              cute::gemm(tiled_mma, tCrA(_,m_idx,fragment_k(k_block)), decoded_b_fragment,
                         scale_accum(_,m_idx,n_idx));
            } else {
              if (mma_global_k >= K) continue;
              auto target = [&]() {
                if constexpr (GroupScale) return scale_accum(_,m_idx,n_idx);
                else return accum(_,m_idx,n_idx);
              }();
              if constexpr (GroupScale) {
                // Partial sums cannot cross either activation or weight scale
                // boundaries. Warp/tile boundaries may also split a scale block.
                bool begin = k_block == 0;
                if constexpr (GroupAScale) begin |= mma_global_k % kAScaleGroup == 0;
                if constexpr (BlockBScale) begin |= mma_global_k % kBScaleBlock == 0;
                if (begin) clear(target);
              }
              if constexpr (!SplitMma) {
                cute::gemm(tiled_mma, tCrA(_,m_idx,fragment_k(k_block)), decoded_b_fragment, target);
              } else {
                // Preserve the two independent K=16 weight/activation scales.
                CUTLASS_PRAGMA_UNROLL
                for (int half = 0; half < 2; ++half) {
                  DstArray fragment = decoded_b;
                  auto words = reinterpret_cast<uint32_t*>(&fragment);
                  words[1 - half] = 0;
                  words[3 - half] = 0;
                  auto fragment_tensor = make_tensor(fragment.data(), decoded_b_fragment.layout());
                  auto partial = make_tensor_like(accum(_,m_idx,n_idx));
                  clear(partial);
                  cute::gemm(tiled_mma, tCrA(_,m_idx,fragment_k(k_block)), fragment_tensor, partial);
                  for_each(make_int_sequence<decltype(size(partial))::value>{}, [&](auto i) {
                    const int row = get<0>(cAccum(i,m_idx,n_idx));
                    const int col = get<1>(cAccum(i,m_idx,n_idx));
                    float scale = 1.0f;
                    if constexpr (!Nvfp8Decode) {
                      const int group = local_scale_group + half;
                      const int swizzled_col = col / 64 * 64 + (col % 8) * 8 + (col % 64) / 8;
                      uint8_t bits;
                      if constexpr (N_EXPAND == 1) {
                        if constexpr (PrefetchFinalScales && decltype(k_block)::value == K_BLOCK_MAX - 1) {
                          constexpr int index = 2 * (int(n_idx) * decltype(size<0>(accum))::value + int(i));
                          bits = uint8_t(final_scales[index / 4] >> (8 * (index % 4 + half)));
                        } else {
                          bits = storage.smem_scale[(compute_k_tile % kNumStages) * ScaleBytesPerStage
                                                   + group * BlockN + swizzled_col];
                        }
                      } else {
                        const int gn = scale_n_block_idx * BlockN + swizzled_col;
                        bits = gn < N ? params.ptr_scale[scale_offset_l
                           + int64_t(compute_k_tile * GroupsPerBlock + group) * N + gn] : 0;
                      }
                      cutlass::float_e4m3_t s;
                      reinterpret_cast<uint8_t&>(s) = bits;
                      scale = float(s);
                    }
                    if constexpr (kAScaleGroup == 16) {
                      scale *= activation_scale(row, mma_global_k + half * 16);
                    }
                    target(i) += partial(i) * scale;
                  });
                }
              }
              if constexpr (GroupScale) {
                bool flush = k_block == K_BLOCK_MAX - 1 || mma_global_k + 32 >= K;
                if constexpr (GroupAScale) flush |= (mma_global_k + 32) % kAScaleGroup == 0;
                if constexpr (BlockBScale) flush |= (mma_global_k + 32) % kBScaleBlock == 0;
                if (flush) {
                  if constexpr (Nvfp8Decode) {
                    // A warp's N tile lies within one N128 weight scale block.
                    // Group C fragments by row so all columns share one A load.
                    constexpr auto ref = TiledMma{}.get_thread_slice(_0{}).partition_C(
                        make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{})));
                    constexpr int count = decltype(size(target))::value;
                    for_each(make_int_sequence<count>{}, [&](auto first) {
                      constexpr int row_id = get<0>(ref(first,0,0));
                      constexpr bool first_in_row = [&]() {
                        for (int j = 0; j < decltype(first)::value; ++j)
                          if (get<0>(ref(j,0,0)) == row_id) return false;
                        return true;
                      }();
                      if constexpr (first_in_row) {
                        float scale = mma_weight_scale;
                        if constexpr (GroupAScale)
                          scale *= activation_scale(get<0>(cAccum(first,m_idx,n_idx)), mma_global_k);
                        for_each(make_int_sequence<count>{}, [&](auto i) {
                          if constexpr (get<0>(ref(i,0,0)) == row_id)
                            accum(i,m_idx,n_idx) += target(i) * scale;
                        });
                      }
                    });
                  } else {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < size(target); ++i) {
                      const int row = get<0>(cAccum(i,m_idx,n_idx));
                      const int col = get<1>(cAccum(i,m_idx,n_idx));
                      float scale = block_weight_scale(col, mma_global_k);
                      if constexpr (GroupAScale) scale *= activation_scale(row, mma_global_k);
                      accum(i,m_idx,n_idx) += target(i) * scale;
                    }
                  }
                }
              }
            }
          }
        });
        if constexpr (FastBlockScale && decltype(k_block)::value == 3) {
          promote_block_scales(active_m_fragments);
        }
      };

      auto mix_gemm = [&](auto k_block) {
        mix_gemm_impl(k_block, Int<MFragmentsPerWarp>{}, NoTensor{});
      };

      auto epilogue_cta_reduce = [&]() {
        if constexpr(WarpsOnK == 1) return;

        #pragma unroll
        for (int m_idx = 0; m_idx < size<1>(accum); m_idx++) {
          #pragma unroll
          for (int warp_offset = WarpsOnK / 2; warp_offset > 0; warp_offset >>= 1) {
            if (warp_offset <= warpk_idx && warpk_idx < 2 * warp_offset) {
              #pragma unroll
              for (int n_idx = 0; n_idx < size<2>(accum); n_idx++) {
                int r2s_idx = lane_idx + n_idx * NumThreadsPerWarp;
                if (warp_offset < WarpsOnK / 2) {
                  auto partial_sum0 = make_tensor_like(accum(_, m_idx, n_idx));
                  auto partial_sum1 = make_tensor_like(partial_sum0);
                  copy_aligned(sReduce_tile(_, r2s_idx, warpn_idx, 2 * warpk_idx), partial_sum0);
                  copy_aligned(sReduce_tile(_, r2s_idx, warpn_idx, 2 * warpk_idx + 1), partial_sum1);
                  #pragma unroll
                  for (int fragc_idx = 0; fragc_idx < size(partial_sum0); fragc_idx++) {
                    accum(fragc_idx, m_idx, n_idx) += partial_sum0(fragc_idx) + partial_sum1(fragc_idx);
                  }
                }
                copy_aligned(accum(_, m_idx, n_idx), sReduce_tile(_, r2s_idx, warpn_idx, warpk_idx));
              }
            }
            __syncthreads();
          }
          if (warpk_idx == 0) {
            #pragma unroll
            for (int n_idx = 0; n_idx < size<2>(accum); n_idx++) {
              int r2s_idx = lane_idx + n_idx * NumThreadsPerWarp;
              auto partial_sum = make_tensor_like(accum(_, m_idx, n_idx));
              copy_aligned(sReduce_tile(_, r2s_idx, warpn_idx, _1{}), partial_sum);
              #pragma unroll
              for (int fragc_idx = 0; fragc_idx < size(partial_sum); fragc_idx++) {
                accum(fragc_idx, m_idx, n_idx) += partial_sum(fragc_idx);
              }
            }
          }
          __syncthreads();
        }
      };

      auto epilogue_no_tsm = [&]() {
        if constexpr(WarpsOnK > 1) {
          if (warpk_idx > 0) return;
        }

        #if __HGGC_ARCH__ == 150
          constexpr int align_elems = 2;
        #else
          constexpr int align_elems = 1;
        #endif
        Tensor tDcD = thr_mma.partition_C(cD);
        auto tDgD = [&]() {
          if constexpr(!Fused) {
            return thr_mma.partition_C(gD);
          } else {
            return NoTensor{};
          }
        }();
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tDcD); i += align_elems) {
          ElementD* dst_ptr;
          bool cond;
          if constexpr(Fused) {
            int local_m = get<0>(tDcD(i));
            int64_t m_offset = int64_t(deep_scheduler.curr_block_m_offset) + local_m;
            cond = local_m < deep_scheduler.valid_m_in_block;
            dst_ptr = params.ptr_d + m_offset * N + n_block_idx * BlockN + get<1>(tDcD(i));
          } else {
            cond = get<0>(tDcD(i)) < residual_m;
            dst_ptr = &tDgD(i);
          }
          if constexpr(N % BlockN) {
            cond = cond && get<1>(tDcD(i)) < (N - n_block_idx * BlockN);
          }
          if (!cond) continue;
          #if __HGGC_ARCH__ == 150
            uint32_t d;
            if constexpr (kHasBScale || kHasWeightGlobalScale || (kHasAScale && kAScaleGroup == 0)) {
              d = deep_gemm::tix::cvt_rte_bf16x2_f32(
                  accum(i+1) * (output_scale(get<0>(tDcD(i+1))) * channel_scale(get<1>(tDcD(i+1)))),
                  accum(i) * (output_scale(get<0>(tDcD(i))) * channel_scale(get<1>(tDcD(i)))));
            } else {
              d = deep_gemm::tix::cvt_rte_bf16x2_f32(accum(i+1), accum(i));
            }
            *reinterpret_cast<uint32_t*>(dst_ptr) = d;
          #else
            *dst_ptr = ElementD(accum(i) * (output_scale(get<0>(tDcD(i))) * channel_scale(get<1>(tDcD(i)))));
          #endif
        }
      };

      auto epilogue_with_tsm = [&]() {
        if constexpr(WarpsOnK > 1) {
          if (warpk_idx > 0) return;
        }

        auto smem_tiled_copy_C = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
        auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(thread_idx);
        Tensor tCrC = smem_thr_copy_C.retile_S(accum);
        Tensor tCsC = smem_thr_copy_C.partition_D(sC);

        auto smem_tiled_copy_s2r = TiledCopyS2R{};
        auto smem_thr_copy_s2r = smem_tiled_copy_s2r.get_thread_slice(thread_idx);
        Tensor tDsD = smem_thr_copy_s2r.partition_S(sC);

        Tensor cDt  = flat_divide(cD, sC_tile);
        Tensor tDcD = smem_thr_copy_s2r.partition_D(cDt);
        Tensor tDrD = make_tensor<ElementD>(take<0,3>(shape(tDcD)));

        auto tDgD = [&]() {
          if constexpr(!Fused) {
            Tensor gDt = flat_divide(gD, sC_tile);
            Tensor tDgD = smem_thr_copy_s2r.partition_D(gDt);
            return tDgD;
          } else {
            return NoTensor{};
          }
        }();

        auto convert_b16 = [&](int mma_m, int mma_n) {
          constexpr int mma_elems = decltype(size<0>(accum))::value;
          constexpr int epilogue_mma_elems = decltype(size<0>(tCsC))::value;
          constexpr int epilogue_num_mma = epilogue_mma_elems / mma_elems;
          using SrcArray = cutlass::Array<ElementAcc, mma_elems>;
          using DstArray = cutlass::Array<ElementD, mma_elems>;
          cutlass::NumericArrayConverter<ElementD, ElementAcc, mma_elems, FloatRoundStyle::round_to_nearest> converter;

          cutlass::Array<ElementD, epilogue_mma_elems> convert_buf;
          Tensor tCrC_convert = make_tensor(
            convert_buf.data(),
            make_layout(make_shape(Int<mma_elems>{}, Int<epilogue_num_mma>{}), make_stride(_1{}, Int<mma_elems>{}))
          );
          CUTLASS_PRAGMA_UNROLL
          for (int mma_k = 0; mma_k < epilogue_num_mma; mma_k++) {
            const SrcArray& src = *reinterpret_cast<const SrcArray *>(&tCrC(mma_k*mma_elems,mma_m,mma_n));
            DstArray& dst = *reinterpret_cast<DstArray *>(tCrC_convert(_,mma_k).data());
            if constexpr (kHasBScale || kHasWeightGlobalScale || (kHasAScale && kAScaleGroup == 0)) {
              SrcArray scaled = src;
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < mma_elems; ++i) {
                const auto coord = cAccum(mma_k*mma_elems+i,mma_m,mma_n);
                scaled[i] *= output_scale(get<0>(coord)) * channel_scale(get<1>(coord));
              }
              dst = converter(scaled);
            } else {
              dst = converter(src);
            }
          }
          return tCrC_convert;
        };

        static_assert(size<3>(cDt) == 1);
        CUTLASS_PRAGMA_UNROLL
        for (int step_m = 0; step_m < size<2>(cDt); ++step_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
              int mma_m = step_m * size<1>(tCsC) + pipe_m;
              int mma_n = pipe_n;
              copy(smem_tiled_copy_C, convert_b16(mma_m, mma_n), tCsC(_,pipe_m,pipe_n));
            }
          }

          __ppu_barrier_sync(3, MaxThreadsPerBlock / WarpsOnK);

          copy(smem_tiled_copy_s2r, tDsD, tDrD);
          Tensor tDcDmn = tDcD(_,_,_,step_m,0);
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tDcDmn); ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tDcDmn); ++n) {
              if constexpr(Fused) {
                int local_m = get<0>(tDcDmn(0,m,n));
                int64_t m_offset = int64_t(deep_scheduler.curr_block_m_offset) + local_m;
                int n_offset = n_block_idx * BlockN + get<1>(tDcDmn(0,m,n));
                bool cond = local_m < deep_scheduler.valid_m_in_block;
                if constexpr(N % BlockN) {
                  cond = cond && n_offset < N;
                }
                if (cond) {
                  uint128_t* dst_ptr = (uint128_t*)(params.ptr_d + m_offset * N + n_offset);
                  uint128_t* acc_ptr = (uint128_t*)(raw_pointer_cast(tDrD(_,m,n).data()));
                  *dst_ptr = *acc_ptr;
                }
              } else {
                Tensor tDgDmn = tDgD(_,_,_,step_m,0);
                bool cond = get<0>(tDcDmn(0,m,n)) < residual_m;
                if constexpr(N % BlockN) {
                  cond = cond && get<1>(tDcDmn(0,m,n)) < (N - n_block_idx * BlockN);
                }
                if (cond) {
                  copy(CopyAtomR2G{}, tDrD(_,m,n), tDgDmn(_,m,n));
                }
              }
            }
          }
        }
      };

      auto finish_work = [&]() {
        cp_async_wait<0>();
        __syncthreads();
        epilogue_cta_reduce();
        #if __HGGC_ARCH__ == 150
          epilogue_no_tsm();
        #else
          epilogue_with_tsm();
        #endif
        __syncthreads();
      };

      if constexpr(N_EXPAND > 1) {
        static_assert(N % (BlockN * N_EXPAND) == 0);
        static_assert(WarpsOnK == 1, "N_EXPAND is incompatible with WarpsOnK > 1.");
        constexpr int K_TILE_COUNT = size<2>(gB);
        static_assert(K_TILE_COUNT >= kNumStages,
                      "N expansion requires enough K tiles to fill the pipeline.");
        auto k_tile_iter_reset = k_tile_iter;
        int num_n_copy_done = 0;

        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < kNumStages; ++k_pipe) {
          if (k_tile_count > 0) {
            g2s(k_tile_iter, k_pipe);
            ++k_tile_iter;
          }
          cp_async_fence();
          --k_tile_count;
        }
        auto k_tile_count_reset = k_tile_count;

        cp_async_wait<kNumStages - 1>();
        __syncthreads();
        if constexpr(K_BLOCK_MAX > 1) {
          s2r_prologue();
        }

        for (int n_iter = 0; n_iter < N_EXPAND; n_iter++) {
          compute_k_tile = 0;
          scale_n_block_idx = first_n_block_idx + n_iter;
          auto process_kblock_iterations = [&](auto k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
              update_s2r_source(smem_pipe_read);
            }

            s2r_next(k_block);

            if (K_BLOCK_MAX == 1 || k_block == K_BLOCK_MAX - 2) {
              cp_async_wait<kNumStages-2>();
              __syncthreads();
              if (k_tile_count > 0 || n_iter < N_EXPAND - 1) {
                if (k_tile_iter == K_TILE_COUNT) {
                  num_n_copy_done++;
                  if (num_n_copy_done < N_EXPAND) {
                    tBgB.data() = tBgB.data() + 4 * BlockN;
                  }
                  k_tile_iter = k_tile_iter_reset;
                }

                g2s(k_tile_iter, smem_pipe_read);
                ++k_tile_iter;
              }
              cp_async_fence();
              --k_tile_count;

              smem_pipe_read = (smem_pipe_read + 1) % kNumStages;
            }

            mix_gemm(k_block);
            if (k_block == K_BLOCK_MAX - 1) {
              ++compute_k_tile;
            }
          };

          #if __HGGC_ARCH__ == 150
          if constexpr (BlockM == 16) {
            for_each(make_int_sequence<K_BLOCK_MAX >{}, [&] (auto k_block) {
              process_kblock_iterations(k_block);
            });
          }
          #endif

          CUTLASS_PRAGMA_NO_UNROLL
          while (k_tile_count > -kNumStages) {
            for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
              process_kblock_iterations(k_block);
            });
          }

          // epilogue
          if (n_iter > 0) {
            n_block_idx++;
            if constexpr(!Fused) {
              gD.data() = gD.data() + BlockN;
            }
          }
          epilogue_no_tsm();

          if (n_iter < N_EXPAND - 1) {
            clear(accum);
            k_tile_count = k_tile_count_reset;

            cp_async_wait<kNumStages-1>();
            __syncthreads();

            s2r_prologue();
          }
        }
        continue;
      } else {
        int smem_pipe_write = kNumStages-1;
        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < kNumStages-1; ++k_pipe) {
          if (k_tile_count > 0) {
            g2s(k_tile_iter, k_pipe);
            ++k_tile_iter;
          }
          cp_async_fence();
          --k_tile_count;
        }

        auto run_mainloop = [&](auto active_m_fragments) {
          cp_async_wait<kNumStages-2>();
          __syncthreads();
          if constexpr(K_BLOCK_MAX > 1 && !RollingOperands) {
            s2r_prologue_impl(active_m_fragments);
          }

          auto prefetch_next_tile = [&]() {
            if (k_tile_count > 0) {
              g2s(k_tile_iter, smem_pipe_write);
              ++k_tile_iter;
            }
            cp_async_fence();
            --k_tile_count;
            smem_pipe_write = smem_pipe_read;
            smem_pipe_read = (smem_pipe_read + 1) % kNumStages;
          };

          auto process_kblock_iterations = [&](auto k_block) {
            if constexpr (RollingOperands) {
              for_each(make_int_sequence<decltype(active_m_fragments)::value>{}, [&](auto mi) {
                copy(smem_tiled_copy_A, tCsA_p(_,mi,k_block), tCrA_copy_view(_,mi,Int<0>{}));
              });
              copy(smem_tiled_copy_B, tCsB_p(_,k_block,_), tCrB_copy_view(_,Int<0>{},_));
              if constexpr (decltype(k_block)::value == 0) prefetch_next_tile();
              mix_gemm_impl(k_block, active_m_fragments, NoTensor{});
              if constexpr (decltype(k_block)::value == K_BLOCK_MAX - 1) {
                // All native-scale and operand reads finish before releasing
                // this stage to the next tile's asynchronous producer.
                update_s2r_source(smem_pipe_read);
                cp_async_wait<kNumStages-2>();
                __syncthreads();
                ++compute_k_tile;
              }
            } else {
              if constexpr (decltype(k_block)::value == 0) {
                decode_shared_tile();
                load_decoded(Int<0>{});
              }
              // Read the next B fragment before this fragment's MMA. In particular,
              // the final shared read completes before the existing stage barrier
              // below; the next decode therefore needs only its publish barrier.
              if constexpr (decltype(k_block)::value < K_BLOCK_MAX - 1)
                load_decoded(Int<decltype(k_block)::value + 1>{});
              // Read final-fragment scales before the barrier releases their slot.
              const auto final_scales = load_final_scales(k_block);
              if constexpr (K_BLOCK_MAX == 1) {
                s2r_next_impl(k_block, active_m_fragments);
                prefetch_next_tile();
              }
              if (k_block == K_BLOCK_MAX - 1) {
                update_s2r_source(smem_pipe_read);
                cp_async_wait<kNumStages-2>();
                __syncthreads();
              }

              if constexpr (K_BLOCK_MAX > 1) {
                s2r_next_impl(k_block, active_m_fragments);
                if (k_block == 0) prefetch_next_tile();
              }

              mix_gemm_impl(k_block, active_m_fragments, final_scales);
              if (k_block == K_BLOCK_MAX - 1) {
                ++compute_k_tile;
              }
            }
          };

          #if __HGGC_ARCH__ == 150
            if constexpr (BlockM == 16) {
              for_each(make_int_sequence<K_BLOCK_MAX >{}, [&] (auto k_block) {
              process_kblock_iterations(k_block);
            });
            }
          #endif

          CUTLASS_PRAGMA_NO_UNROLL
          while (k_tile_count > -(kNumStages-1)) {
            for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
              process_kblock_iterations(k_block);
            });
          }
        };
        if constexpr (RollingOperands || TrimM) {
          // Complete each exclusive tail before joining the next work item;
          // its accumulators need not remain live across a shared epilogue.
          auto run_active = [&](auto active_m_fragments) {
            run_mainloop(active_m_fragments);
            finish_work();
          };
          dispatch_m_fragments<MFragmentsPerWarp>(residual_m, run_active);
          continue;
        } else {
          run_mainloop(Int<MFragmentsPerWarp>{});
        }
      }

      finish_work();
    }
  }
};

} // namespace cutlass::gemm::kernel

#pragma clang diagnostic pop
