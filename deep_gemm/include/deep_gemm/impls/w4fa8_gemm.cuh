#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wswitch"

#include <cute/ppu_tensor_mix.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/atom/mma_traits_ppu0015.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0015_aiu.hpp>
#include <cute/algorithm/ppu_copy.hpp>
#include <deep_gemm/impls/dequant_w4a16.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/scheduler/w4a8_small_m_scheduler.cuh>
#include <deep_gemm/scheduler/w4a8_routed_scheduler.cuh>
#include <deep_gemm/scheduler/fused_scheduler.cuh>
#include <deep_gemm/impls/fused_gemm_util.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/tix/cvt.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>

using namespace cute;

namespace cutlass::gemm::kernel {

using deep_gemm::GemmType;

template <int ShapeN, int ShapeK, int BlockM, int BlockN, int BlockK, int WarpM, int WarpN, int WarpK, int kNumGroups,
          int kNumStages, GemmType kGemmType, int N_EXPAND = 1, bool ApplyInvScale = false,
          deep_gemm::w4fa8::NativeStrategy Strategy = deep_gemm::w4fa8::NativeStrategy::Default,
          class SchedulerAdapter = void>
class W4A8GEMM_MMA {
public:
  static constexpr bool Routed = Strategy == deep_gemm::w4fa8::NativeStrategy::Routed;
  static constexpr bool Prefix = Strategy == deep_gemm::w4fa8::NativeStrategy::Compact;
  static constexpr bool SharedPrefix = Prefix && BlockM == 32;
  static constexpr bool Compact = Routed || (Prefix && BlockM != 16);
  static constexpr int DecodeN = Routed ? 1 : SharedPrefix ? 2 : WarpN / 16;
  // A narrow ring bounds live A operands for wide NoPad tiles.
  static constexpr int ADistance = Routed && kGemmType == GemmType::MGroupedNoPad ? 1 : -1;
  static_assert(!Compact || (N_EXPAND == 1 && BlockK == WarpK));
  static_assert(WarpN == 16 || WarpN == 32 || WarpN == 48 || WarpN == 64);
  static_assert(N_EXPAND >= 1 && N_EXPAND <= 4);
  static_assert(N_EXPAND == 1 || (kGemmType == GemmType::MGroupedNoPad && kNumStages == 2 && BlockK == WarpK));
  static_assert(N_EXPAND == 1 || ShapeN % (BlockN * N_EXPAND) == 0);
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float4_t; // fp4
  using ElementD = cutlass::bfloat16_t;
  using ElementAcc = float;
  using ElementScale = uint8_t; // e8m0 scale

  // A fourth slot reduces CTA residency for the default N256 tile.
  static constexpr bool ResidentA = N_EXPAND > 1 && WarpM == 64 && ShapeK <= 384;
  static constexpr int ASlots = ResidentA && ShapeK / BlockK > kNumStages ? ShapeK / BlockK : kNumStages;
  static constexpr bool CacheRowScale = N_EXPAND > 1 && WarpM == 64 && ShapeK <= 512;
  // basic
  static constexpr bool Fused = (kGemmType == GemmType::MGroupedFused);
  // A packed K64 slice feeds two K32 MMA steps for every M tile and grouped kind.
  static constexpr bool BatchDequant = Routed || WarpK == 128;
  static constexpr bool E8M0_scale = true;
  static constexpr bool scale_use_ld_matrix = true;
  static constexpr int N = ShapeN;
  static constexpr int K = ShapeK;
  static constexpr int K2 = K / 2;
  static constexpr int L = kNumGroups;
  static constexpr int BlockK2 = BlockK / 2;
  static constexpr int WarpK2 = WarpK / 2;
  static constexpr int kGroupSize = 32;
  static constexpr int GroupsPerBlock = BlockK / kGroupSize;
  static constexpr int GroupsPerWarp = WarpK / kGroupSize;
  static constexpr int MmaPerGroup = kGroupSize / 16;
  using TileShape = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
  static constexpr int N_Blocks = deep_gemm::math::ceil_div(ShapeN, BlockN* N_EXPAND);
  static constexpr int scheduler_swizzle = (BlockM == 16) ? 1 : 2;
  using DefaultScheduler =
      cute::conditional_t<Fused,
                          deep_gemm::FusedGemmScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN * N_EXPAND,
                                                        kNumGroups, N_Blocks, scheduler_swizzle>,
                          deep_gemm::DeepGemmScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN * N_EXPAND, kNumGroups,
                                                       N_Blocks, scheduler_swizzle>>;
  using PrefixScheduler = cute::conditional_t<
      BlockM == 16,
      deep_gemm::WarpPrefixPairScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN, kNumGroups, N_Blocks, 2, true>,
      cute::conditional_t<
          SharedPrefix,
          deep_gemm::SharedWarpPrefixScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN, kNumGroups, N_Blocks, 2>,
          deep_gemm::WarpPrefixScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN, kNumGroups, N_Blocks, 2>>>;
  using TileScheduler =
      cute::conditional_t<Routed, SchedulerAdapter, cute::conditional_t<Prefix, PrefixScheduler, DefaultScheduler>>;
  using ArchTag = cutlass::arch::PPU0015;
  static constexpr int WarpsOnM = BlockM / WarpM;
  static constexpr int WarpsOnN = BlockN / WarpN; // WarpN = 64
  static constexpr int WarpsOnK = BlockK / WarpK;

  // fp4 mma => dequant
  using FP4MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;
  using FP4PermutationMNK =
      Tile<Layout<Shape<_16, Int<WarpsOnN>, Int<WarpN / 16>>, Stride<_1, Int<WarpN>, _16>>, _16, Int<WarpsOnK * 32>>;
  using FP4TiledMma =
      TiledMMA<MMA_Atom<FP4MmaInst>, Layout<Shape<Int<WarpsOnN>, _1, Int<WarpsOnK>>>, FP4PermutationMNK>;

  // Mixed E4M3 activation x E5M2 dequantized weight MMA
  using MmaInst = PPU0015_16x16x32_F32E4M3E5M2F32_TN;
  using PermutationMNK =
      Tile<Int<WarpsOnM * 16>, Layout<Shape<_16, Int<WarpsOnN>, Int<WarpN / 16>>, Stride<_1, Int<WarpN>, _16>>,
           Int<WarpsOnK * 32>>;
  using TiledMma =
      TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<Int<WarpsOnM>, Int<WarpsOnN>, Int<WarpsOnK>>>, PermutationMNK>;
  static constexpr int MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr bool SplitAIU = (MaxThreadsPerBlock / NumThreadsPerWarp) > 1;

  // sA
  static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
  using ACopyInst = cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t>;
  using FusedOperandA = cutlass::gemm::config::Gemm_Hybrid_Operand<ArchTag, ElementA, false, AlignA, Int<BlockK>,
                                                                   MaxThreadsPerBlock, ACopyInst, Int<BlockM>>;
  using DefaultOperandA =
      cute::conditional_t<Fused, FusedOperandA,
                          cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, false, Int<BlockM>,
                                                                         Int<BlockK>, false, 0, true>>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
  using TilerA = typename GmemTiledCopyA::Tiler_MN;
  using CopyAConfig = deep_gemm::CopyAToTsmConfig<ElementA, TilerA, BlockM, BlockK, MaxThreadsPerBlock>;
  using SmemLayoutA =
      decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<ASlots>{})));

  // sW => fp4 weights
  using TilerB = Shape<Int<BlockN>, Int<BlockK2>>;
  using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, uint8_t, false, Int<BlockN>,
                                                                         Int<BlockK2>, false, 0, true>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;
  using SmemLayoutB =
      decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<BlockN>{}, Int<BlockK2>{}, Int<kNumStages>{})));

  // sScale
  using TilerScale = Shape<Int<BlockN / 64>, Int<BlockK * 2>>;
  using DefaultOperandScale =
      cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementScale, false, Int<BlockN / 64>, Int<BlockK * 2>,
                                                     false, 0, false>;
  using SmemLayoutAtomScale = typename DefaultOperandScale::SmemLayoutAtom;
  using GmemTiledCopyScale = typename DefaultOperandScale::GmemTiledCopy;
  using SmemLayoutScale = decltype(tile_to_shape(SmemLayoutAtomScale{},
                                                 make_shape(Int<BlockN / 64>{}, Int<BlockK * 2>{}, Int<kNumStages>{})));
  // custom s2r copy scale
  using ScaleCopyOp = UniversalCopy<cute::uint32_t>;
  using ScaleCopyTraits = Copy_Traits<ScaleCopyOp>;
  using SmemCopyAtomScale = Copy_Atom<ScaleCopyTraits, ElementScale>;
  using SmemThrLayoutScale =
      Layout<Shape<Int<BlockN / 64>, Shape<_32, Int<WarpsOnK>>>, Stride<Int<32 * WarpsOnK>, Stride<_1, _32>>>;
  using SmemValLayoutScale = Layout<Shape<_1, _4>>;
  using SmemTiledCopyScale = decltype(make_tiled_copy(SmemCopyAtomScale{}, SmemThrLayoutScale{}, SmemValLayoutScale{}));

  // epilogue cta reduce
  static_assert(WarpsOnM == 1 || WarpsOnK == 1);
  using SmemLayoutReduce =
      decltype(make_layout(make_shape(Int<16 * 64>{}, Int<WarpsOnN>{}, Int<WarpsOnK>{}), LayoutLeft{}));

  struct MainloopStorage {
    union {
      struct {
        cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
        cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutB>> smem_b;
        cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScale>> smem_scale;
      }; // mainloop
      cute::array_aligned<ElementAcc, cute::cosize_v<SmemLayoutReduce>> smem_reduce; // epilogue cta reduce
    };
  };

  using SchedulerStorage = deep_gemm::SmallMSchedulerStorage<MainloopStorage, TileScheduler, SharedPrefix>;
  using SharedStorage = deep_gemm::w4fa8::RowScaleStorage<SchedulerStorage, Routed ? BlockM : 0>;

  struct NormalArguments {
    int* block_m_info;
  };
  struct FusedArguments {
    const int* expert_ids_and_cumsum;
    const int* sorted_token_ids;
    const int* aligned_num_m_blocks;
    int topk;
  };
  using Arguments = cute::conditional_t<Fused, FusedArguments, NormalArguments>;

  struct Params {
    const ElementA* ptr_a;
    const uint8_t* ptr_b;
    const ElementScale* ptr_scale;
    ElementD* ptr_d;
    const float* ptr_a_scale;
    int num_token;
    float inv_scale;
    const int* sorted_token_ids;
    typename TileScheduler::Params scheduler;
  };

  // for constexpr patial initialize
  struct NoTensor {};

  // Each column permutation gathers four consecutive K values into one lane.
  // Return an owning tensor: a view of a local C array would dangle.
  CUTLASS_DEVICE auto init_identity(int tid) {
    auto id = make_tensor<uint32_t>(Shape<_4, _4>{});
    clear(id);
    const unsigned g = unsigned(tid) / 4u, q = unsigned(tid) % 4u;
    const unsigned sigma = (g & 1u) + 4u * ((g >> 1u) & 3u);
    auto word = [&](unsigned h, unsigned upper) {
      const unsigned pos = 16u * h + sigma + 2u * upper - 8u * q;
      return pos < 8u ? (2u << (4u * (pos & 7u))) : 0u;
    };
    const uint32_t a = word(0, 0), b = word(0, 1), c = word(1, 0), d = word(1, 1);
    id(0, 0) = a;
    id(1, 2) = a;
    id(2, 0) = b;
    id(3, 2) = b;
    id(0, 1) = c;
    id(1, 3) = c;
    id(2, 1) = d;
    id(3, 3) = d;
    return id;
  }

  template <int mode = 0, typename Tensor> CUTLASS_DEVICE constexpr auto reshape_warpk(Tensor& t) {
    static_assert(mode == 0 || mode == 1);
    if constexpr (mode == 0) { // sB: (k_tiles, ..., stage), sScale: (k_tiles / MmaPerGroup, ..., stage)
      constexpr int k = decltype(size<mode>(t))::value;
      constexpr int ko = k / WarpsOnK;
      auto atom_tiler = make_layout(make_shape(Int<WarpsOnK>{}, Int<ko>{}), make_stride(Int<ko>{}, _1{}));
      auto tiler = make_shape(atom_tiler, size<1>(t));
      return flat_divide(t, tiler)(_, _, 0, 0, _);
    } else { // sA: (..., blockk, stage) where blockk = 32 * k_tiles.
      constexpr int k = BlockK / 32;
      constexpr int ko = k / WarpsOnK;
      auto atom_tiler = make_layout(make_shape(_32{}, make_shape(Int<WarpsOnK>{}, Int<ko>{})),
                                    make_stride(_1{}, make_stride(Int<ko * 32>{}, _32{})));
      auto tiler = make_shape(size<0>(t), atom_tiler);
      return flat_divide(t, tiler)(_, _, 0, 0, _);
    }
  }

  template <int Atoms = 1, class Process> CUTLASS_DEVICE static void dispatch_tail(int rows, Process const& process) {
    if constexpr (Atoms == WarpM / 16)
      process(Int<Atoms>{});
    else if (rows <= Atoms * 16)
      process(Int<Atoms>{});
    else
      dispatch_tail<Atoms + 1>(rows, process);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    int thread_idx = threadIdx.x;
    int warp_idx = canonical_warp_idx_sync();
    int warpm_idx = warp_idx % WarpsOnM;
    int warpn_idx = (warp_idx / WarpsOnM) % WarpsOnN;
    int warpk_idx = (warp_idx / WarpsOnM) / WarpsOnN;
    int lane_idx = thread_idx % NumThreadsPerWarp;
    Tensor tCrID = init_identity(lane_idx);
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto deep_scheduler = [&]() {
      if constexpr (SharedPrefix)
        return TileScheduler{params.scheduler, storage.scheduler_state};
      else
        return TileScheduler{params.scheduler};
    }();
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});             // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});             // (BLK_N,BLK_K2,PIPE)
    Tensor sScale = make_tensor(make_smem_ptr(storage.smem_scale.data()), SmemLayoutScale{}); // (BLN/64,BLK*2,PIPE)
    Tensor sReduce = make_tensor(make_smem_ptr<ElementAcc>(storage.smem_reduce.data()),
                                 SmemLayoutReduce{});                        // (16*64,WARPS_ON_N,WARPS_ON_K)
    auto sReduce_tile = local_tile(sReduce, make_tile(_8{}), make_coord(_)); // (8,128,WARPS_ON_N,WARPS_ON_K)

    auto strideA = make_stride(Int<K>{}, _1{});
    auto strideB = make_stride(Int<K2>{}, _1{}, Int<N * K2>{});
    auto strideScale = make_stride(Int<K * 2>{}, _1{}, Int<K * N / 32>{});
    auto strideD = make_stride(Int<N>{}, _1{});
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    GmemTiledCopyScale gmem_tiled_copy_scale;
    gmem_tiled_copy_B.desc_.template init<uint8_t, false, BlockN, BlockK2>(nullptr, N, K2, strideB);
    gmem_tiled_copy_scale.desc_.template init<ElementScale, false, BlockN / 64, BlockK * 2>(nullptr, N / 64, K * 2,
                                                                                            strideScale);
    uint32_t thread_idx_A = Fused ? CopyAConfig::logical_thread_idx(thread_idx) : thread_idx;
    int thread_idx_B = (warp_idx / WarpsOnM) * NumThreadsPerWarp + lane_idx;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx_A);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    auto gmem_thr_copy_scale = gmem_tiled_copy_scale.get_slice(thread_idx);
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);
    Tensor tSsS = gmem_thr_copy_scale.partition_D(sScale);

    FP4TiledMma fp4_tiled_mma;
    auto fp4_thr_mma = fp4_tiled_mma.get_thread_slice(thread_idx_B);
    auto fragment_k = [](int index) {
      if constexpr (Compact)
        return Int<0>{};
      else
        return index;
    };
    auto fp4_tCrB = [&]() {
      auto full = fp4_thr_mma.partition_fragment_A(sB(_, _, 0));
      if constexpr (Compact)
        return make_tensor<typename decltype(full)::value_type>(make_shape(shape<0>(full), shape<1>(full), _1{}));
      else
        return full;
    }();
    auto fp4_accum = make_tensor<ElementAcc>(Shape<_8, Int<DecodeN>, Int<Compact && !Routed ? 2 : 1>>{});
    auto decode_fragment = [&](int n, int half) {
      if constexpr (Routed)
        return fp4_accum(_, Int<0>{}, Int<0>{});
      else if constexpr (Compact)
        return fp4_accum(_, n, half);
      else
        return fp4_accum(_, n, Int<0>{});
    };

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto tCrA = [&]() {
      auto full = thr_mma.partition_fragment_A(sA(_, _, 0));
      if constexpr (!Compact)
        return full;
      else if constexpr (ADistance >= 0)
        return make_tensor<typename decltype(full)::value_type>(make_shape(shape<0>(full), Int<ADistance + 1>{}, _1{}));
      else
        return make_tensor<typename decltype(full)::value_type>(make_shape(shape<0>(full), shape<1>(full), _1{}));
    }();
    auto tCrB = [&]() {
      auto full = make_tensor<typename TiledMma::FrgTypeB>(partition_shape_B(tiled_mma, take<1, 3>(TileShape{})));
      if constexpr (Compact)
        return make_tensor<typename decltype(full)::value_type>(make_shape(shape<0>(full), shape<1>(full), _1{}));
      else
        return full;
    }();
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto [smem_thr_copy_A, tCsA] = [&]() {
      if constexpr (Fused) {
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
        auto mix_sA = make_mix_tensor_like(sA);
        auto sA_r2s = reshape_warpk<1>(mix_sA);
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
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    constexpr int K_BLOCK_MAX = WarpK / 32;

    auto smem_tiled_copy_B = make_tiled_copy_A(SmemCopyAtomB{}, fp4_tiled_mma); // fp4 B is matrixA in fp4 gemm
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice((warp_idx / WarpsOnM) * NumThreadsPerWarp);
    auto mix_sB = make_mix_tensor_like(sB);
    Tensor tCsB = smem_thr_copy_B.partition_S(mix_sB);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(fp4_tCrB);

    // Native scale packing is organized in N64 units. Adjacent narrow
    // N warps load the same scale word and select their N16 sub-fragment.
    const int scale_group = WarpN == 48 ? warpn_idx * 3 / 4 : warpn_idx / (64 / WarpN);
    int thread_idx_nk = (scale_group * WarpsOnK + warpk_idx) * NumThreadsPerWarp;
    if constexpr (not scale_use_ld_matrix) {
      thread_idx_nk += lane_idx;
    } else {
      thread_idx_nk += lane_idx * 4; // thread0-7 get the row address
    }
    SmemTiledCopyScale smem_tiled_copy_scale;
    auto smem_thr_copy_scale = smem_tiled_copy_scale.get_thread_slice(thread_idx_nk);
    Tensor tCsS = smem_thr_copy_scale.partition_S(sScale);
    auto scale_alt_thr = smem_tiled_copy_scale.get_thread_slice(
        (min(scale_group + 1, BlockN / 64 - 1) * WarpsOnK + warpk_idx) * 32 + lane_idx * 4);
    Tensor tCsS_alt = scale_alt_thr.partition_S(sScale);
    Tensor tCrS_copy_view = make_tensor_like(tCsS(_, _, _, 0));
    Tensor tCrS = recast<uint32_t>(tCrS_copy_view);

    uint32_t m_block_idx, n_block_idx;
    int work_serial = 0;
#pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {

      const int scale_buffer = work_serial++ & 1;
      int coord_l = deep_scheduler.curr_group_idx;
      n_block_idx *= N_EXPAND;

      const uint8_t* ptr_b_l = params.ptr_b + int64_t(coord_l) * N * K2;
      Tensor mB_nk = make_tensor(make_gmem_ptr(ptr_b_l), make_shape(Int<N>{}, Int<K2>{}), make_stride(Int<K2>{}, _1{}));
      Tensor mB_nk_mix = make_mix_tensor_like(mB_nk);
      Tensor gB = local_tile(mB_nk_mix, TilerB{}, make_coord(n_block_idx, _));
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

      Tensor mScale_nkl =
          make_tensor(make_gmem_ptr(params.ptr_scale), make_shape(Int<N / 64>{}, Int<K * 2>{}, Int<L>{}), strideScale);
      Tensor mScale_nk = make_mix_tensor_like(mScale_nkl(_, _, coord_l));
      Tensor gScale = local_tile(mScale_nk, TilerScale{}, make_coord(n_block_idx, _));
      Tensor tSgS = gmem_thr_copy_scale.partition_S(gScale);

      Tensor cD = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
      uint32_t token_offsets[CopyAConfig::M_ITER];
      auto [tAgA, residual_m, gD] = [&]() {
        if constexpr (!Fused) {
          int M = deep_scheduler.curr_group_m;
          gmem_tiled_copy_A.desc_.template init<ElementA, false, BlockM, BlockK>(nullptr, M, K, strideA);
          Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_a + deep_scheduler.curr_offset_a()),
                                      make_shape(M, Int<K>{}), strideA); // (m,k)
          Tensor mA_mk = make_mix_tensor_like(mA_mkl);                   // (m,k)
          Tensor gA = local_tile(mA_mk, TileShape{}, make_coord(m_block_idx, n_block_idx, _),
                                 Step<_1, X, _1>{}); // (BLK_M,BLK_K,blocks_k)
          Tensor tAgA = gmem_thr_copy_A.partition_S(gA);

          Tensor mD_mn = make_tensor(make_gmem_ptr(params.ptr_d + deep_scheduler.curr_offset_c()),
                                     make_shape(M, Int<N>{}), strideD); // (m,n)
          Tensor gD = local_tile(mD_mn, TileShape{}, make_coord(m_block_idx, n_block_idx, _),
                                 Step<_1, _1, X>{}); // (BLK_M,BLK_N)
          int residual_m = M - BlockM * m_block_idx;

          return cute::make_tuple(tAgA, residual_m, gD);
        } else {
          const int* blk_token_base = params.sorted_token_ids + [&]() {
            if constexpr (Routed)
              return deep_scheduler.token_base;
            else
              return deep_scheduler.cumsum_m_block_idx * BlockM;
          }();
          if constexpr (Routed) {
            // Internal tiles can extend beyond the original expert padding.
            // Never read token metadata for those rows.
            const int logical = CopyAConfig::logical_thread_idx(thread_idx);
            const int tm = logical / CopyAConfig::NumThreads_CPY_K;
#pragma unroll
            for (int mi = 0; mi < CopyAConfig::M_ITER; ++mi) {
              const int row = mi * CopyAConfig::TILER_M + tm;
              token_offsets[mi] =
                  row < deep_scheduler.valid_m_in_block ? __ldg(blk_token_base + row) : params.num_token;
            }
          } else {
            deep_gemm::prefetch_A_token_offsets<ElementA, TilerA, BlockM, BlockK, MaxThreadsPerBlock>(
                token_offsets, blk_token_base, params.num_token, thread_idx);
            deep_gemm::clear_invalid_A_to_tsm<ElementA, TilerA, BlockM, BlockK, K, kNumStages>(
                tAsA, params.ptr_a, token_offsets, params.num_token, thread_idx, Int<MaxThreadsPerBlock>{});
          }
          Tensor mD_mn = make_tensor(make_gmem_ptr(params.ptr_d + deep_scheduler.curr_block_m_offset * N),
                                     make_shape(Int<BlockM>{}, Int<N>{}), strideD); // (m,n)
          Tensor gD =
              local_tile(mD_mn, TileShape{}, make_coord(_0{}, n_block_idx, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N)
          int residual_m = deep_scheduler.valid_m_in_block;
          return cute::make_tuple(NoTensor{}, residual_m, gD);
        }
      }();

      // The prologue commit/wait covers these 4-byte asynchronous copies too.
      // A buffer is reused only after every warp has crossed the intervening
      // work item's prologue CTA barrier, completing its old epilogue reads.
      if constexpr (Routed)
        if (thread_idx < BlockM) {
          const bool valid = thread_idx < residual_m;
          int64_t scale_row = 0;
          if (valid) {
            if constexpr (Fused)
              scale_row = params.sorted_token_ids[deep_scheduler.token_base + thread_idx];
            else
              scale_row = deep_scheduler.curr_offset_a() / K + m_block_idx * BlockM + thread_idx;
          }
          cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(params.ptr_a_scale[scale_row],
                                                            storage.row_scales[scale_buffer][thread_idx], valid);
        }

      auto g2s_copy_A = [&](int k_tile_iter, int k_pipe) {
        if constexpr (!Fused) { // aiu copy
          if (warp_idx == 0) {
            copy(gmem_tiled_copy_A, tAgA(_, _, _, k_tile_iter), tAsA(_, _, _, k_pipe));
          }
        } else { // async copy
          const uint32_t logical = CopyAConfig::logical_thread_idx(thread_idx);
          const uint32_t cohort = CopyAConfig::Rebalance ? thread_idx / CopyAConfig::ThreadsPerKIter : 0;
          const uint32_t tm = logical / CopyAConfig::NumThreads_CPY_K;
          const uint32_t tk = logical % CopyAConfig::NumThreads_CPY_K;
          if (cohort < CopyAConfig::KCohorts) {
#pragma unroll
            for (int mi = 0; mi < CopyAConfig::M_ITER; ++mi) {
              if (mi * CopyAConfig::TILER_M + tm >= BlockM)
                continue;
              const uint32_t token = token_offsets[mi];
#pragma unroll
              for (int ki = cohort; ki < CopyAConfig::K_ITER; ki += CopyAConfig::KCohorts) {
                const auto* src = reinterpret_cast<const cutlass::uint128_t*>(
                    params.ptr_a + uint64_t(token) * K + ki * CopyAConfig::TILER_N + tk * CopyAConfig::KPerThread +
                    BlockK * k_tile_iter);
                auto* dst = reinterpret_cast<cutlass::uint128_t*>(raw_pointer_cast(tAsA(_, mi, ki, k_pipe).data()));
                if constexpr (Routed)
                  ACopyInst::copy(*src, *dst, token < uint32_t(params.num_token));
                else if (token < uint32_t(params.num_token))
                  cute::PPU_CP_ASYNC_CACHEALWAYS<cutlass::uint128_t>::copy(*src, *dst);
              }
            }
          }
        }
      };

      auto g2s_copy_B_and_scale = [&](int k_tile_iter, int k_pipe) {
        constexpr int copy_warp = SplitAIU ? 1 : 0;
        constexpr int scale_warp = Routed && MaxThreadsPerBlock >= 96 ? 2 : copy_warp;
        if (warp_idx == copy_warp)
          copy(gmem_tiled_copy_B, tBgB(_, _, _, k_tile_iter), tBsB(_, _, _, k_pipe));
        if (warp_idx == scale_warp)
          copy(gmem_tiled_copy_scale, tSgS(_, _, _, k_tile_iter), tSsS(_, _, _, k_pipe));
      };

      // Retain the complete short-K A tile in dedicated shared slots across N.
      int a_tiles_loaded = 0;
      auto g2s = [&](int k_tile_iter, int k_pipe) {
        if constexpr (ResidentA) {
          if (a_tiles_loaded < K / BlockK) {
            g2s_copy_A(k_tile_iter, k_tile_iter);
            ++a_tiles_loaded;
          }
        } else
          g2s_copy_A(k_tile_iter, k_pipe);
        g2s_copy_B_and_scale(k_tile_iter, k_pipe);
      };

      auto s2r_copy_A = [&](int stage_idx, int k_block) {
        if constexpr (ADistance < 0)
          copy(smem_tiled_copy_A, tCsA(_, _, k_block, stage_idx), tCrA_copy_view(_, _, fragment_k(k_block)));
      };

      auto s2r_copy_B_and_scale = [&](int stage_idx) {
        if constexpr (!BatchDequant) {
          copy(smem_tiled_copy_B, tCsB(_, _, _, stage_idx), tCrB_copy_view(_, _, _));
          if constexpr (not scale_use_ld_matrix) {
            copy(smem_tiled_copy_scale, tCsS(_, _, _, stage_idx), tCrS_copy_view(_, _, _));
          } else {
#pragma unroll
            for (int fp4_k_block = 0; fp4_k_block < size<2>(tCsS); fp4_k_block++) {
              uint32_t d;
              uint32_t addr =
                  cute::cast_smem_ptr_to_uint(cute::raw_pointer_cast(tCsS(_, _, fp4_k_block, stage_idx).data()));
              asm volatile("ppu.tc02.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];" : "=r"(d) : "r"(addr));
              if constexpr (WarpN == 48) {
                uint32_t alt;
                uint32_t addr_alt =
                    cast_smem_ptr_to_uint(raw_pointer_cast(tCsS_alt(_, _, fp4_k_block, stage_idx).data()));
                asm volatile("ppu.tc02.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];"
                             : "=r"(alt)
                             : "r"(addr_alt));
                const int shift = (warpn_idx * 3) & 3, q = lane_idx & 3;
                const int source_lane = (lane_idx & ~3) + ((q + shift) & 3);
                const auto a = __shfl_sync(0xffffffffu, d, source_lane);
                const auto b = __shfl_sync(0xffffffffu, alt, source_lane);
                d = q + shift < 4 ? a : b;
              } else if constexpr (WarpN < 64) {
                const int source_lane =
                    (lane_idx & ~3) + (lane_idx & (WarpN / 16 - 1)) + (warpn_idx % (64 / WarpN)) * (WarpN / 16);
                d = __shfl_sync(0xffffffffu, d, source_lane);
              }
              tCrS(0, 0, fp4_k_block) = d;
            }
          }
        }
      };

      // One packed K64 slice supplies two FP8 K32 steps. Delay its load to
      // avoid retaining both packed slices alongside the converted B fragment.
      auto s2r_copy_B_and_scale_slice = [&](int stage_idx, int fp4_k_block) {
        copy(smem_tiled_copy_B, tCsB(_, _, fp4_k_block, stage_idx), tCrB_copy_view(_, _, fragment_k(fp4_k_block)));
        if constexpr (not scale_use_ld_matrix) {
          copy(smem_tiled_copy_scale, tCsS(_, _, fp4_k_block, stage_idx), tCrS_copy_view(_, _, fp4_k_block));
        } else {
          {
            uint32_t d;
            uint32_t addr =
                cute::cast_smem_ptr_to_uint(cute::raw_pointer_cast(tCsS(_, _, fp4_k_block, stage_idx).data()));
            asm volatile("ppu.tc02.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];" : "=r"(d) : "r"(addr));
            if constexpr (WarpN == 48) {
              uint32_t alt;
              uint32_t addr_alt =
                  cast_smem_ptr_to_uint(raw_pointer_cast(tCsS_alt(_, _, fp4_k_block, stage_idx).data()));
              asm volatile("ppu.tc02.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];" : "=r"(alt) : "r"(addr_alt));
              const int shift = (warpn_idx * 3) & 3, q = lane_idx & 3;
              const int source_lane = (lane_idx & ~3) + ((q + shift) & 3);
              const auto a = __shfl_sync(0xffffffffu, d, source_lane);
              const auto b = __shfl_sync(0xffffffffu, alt, source_lane);
              d = q + shift < 4 ? a : b;
            } else if constexpr (WarpN < 64) {
              const int source_lane =
                  (lane_idx & ~3) + (lane_idx & (WarpN / 16 - 1)) + (warpn_idx % (64 / WarpN)) * (WarpN / 16);
              d = __shfl_sync(0xffffffffu, d, source_lane);
            }
            tCrS(0, 0, fp4_k_block) = d;
          }
        }
      };

      auto fp4_mma_dequant = [&](int n_block, int k_block) {
        Tensor s = make_tensor<uint32_t>(Int<4>{});
        s(0) = tCrS(0, 0, k_block / 4); // scale W
        s(1) = 0x7f7f7f7fu;             // 1.0 E8M0 for identity matrix
        s(2) = n_block;                 // a-psel
        s(3) = 0;                       // b-psel

        Tensor d = decode_fragment(n_block, k_block & 1);
        Tensor a = fp4_tCrB(_, n_block, fragment_k(k_block / 4));
        Tensor b = tCrID(_, k_block % 4);
        Tensor c = d;
        cute::mma_unpack(MMA_Atom<FP4MmaInst>{}, d, a, b, c, s);
      };

      auto quant_fp32C_to_e5m2B = [&](int n_block, int k_block, int half) {
        Tensor src = decode_fragment(n_block, half);
        Tensor dst = tCrB(_, n_block, fragment_k(k_block));
        uint32_t* words = reinterpret_cast<uint32_t*>(dst.data());
        CUTE_UNROLL
        for (int row = 0; row < 2; ++row) {
          // Pack hi first, then merge it into the second CVT through its third source (example SASS):
          // v.pcnvt.e5m2x2.f32x2.rtte vreg167, vreg206, vreg207, c0x0
          // v.pcnvt.e5m2x2.f32x2.rtte vreg226, vreg202, vreg203, vreg167
          asm volatile("{\n"
                       ".reg .b16 lo, hi;\n"
                       "ppu.cvt.rtte.satfinite.e5m2x2.f32 lo, %1, %2;\n"
                       "ppu.cvt.rtte.satfinite.e5m2x2.f32 hi, %3, %4;\n"
                       "ppu.mov.b32 %0, {lo, hi};\n"
                       "}"
                       : "=r"(words[2 * row + half])
                       : "f"(src(2 * row + 1)), "f"(src(2 * row)), "f"(src(2 * row + 5)), "f"(src(2 * row + 4)));
        }
      };

      auto epilogue_cta_reduce = [&]() {
        if constexpr (WarpsOnK == 1)
          return;
        __syncthreads();

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

      auto apply_inv_scale = [&](float scale) {
        if constexpr (ApplyInvScale)
          return scale * params.inv_scale;
        else
          return scale;
      };
      // Reuse each lane's row scales across all N_EXPAND outputs of this work item.
      auto cached_row_scale = make_tensor<float>(make_shape(size<1>(accum), Int<2>{}));
      // Borrow the last A stage until the prologue has loaded row scales into registers.
      auto smem_row_scale = reinterpret_cast<float*>(sA(_, _, Int<kNumStages - 1>{}).data().get());
      auto prefetch_row_scale = [&]() {
        if constexpr (CacheRowScale) {
          static_assert(BlockK >= sizeof(float));
          for (int row = thread_idx; row < BlockM; row += MaxThreadsPerBlock) {
            const bool valid = row < residual_m;
            const int64_t scale_row = deep_scheduler.curr_offset_a() / K + m_block_idx * BlockM + (valid ? row : 0);
            cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(params.ptr_a_scale[scale_row], smem_row_scale[row],
                                                              valid);
          }
        }
      };
      auto load_cached_row_scale = [&]() {
        if constexpr (CacheRowScale) {
          Tensor row_coords = thr_mma.partition_C(cD);
          CUTE_UNROLL
          for (int m = 0; m < size<1>(accum); ++m) {
            CUTE_UNROLL
            for (int row = 0; row < 2; ++row) {
              const int local_row = get<0>(row_coords(2 * row, m, 0));
              if (local_row < residual_m) {
                cached_row_scale(m, row) = apply_inv_scale(smem_row_scale[local_row]);
              }
            }
          }
        }
      };

      auto epilogue_no_tsm = [&](auto active_m) {
        constexpr int UsedM = decltype(active_m)::value;
        if constexpr (WarpsOnK > 1) {
          if (warpk_idx > 0)
            return;
        }

        Tensor tDcD = thr_mma.partition_C(cD);
        auto tDgD = thr_mma.partition_C(gD);
        // Each m16n16 accumulator has two rows per lane (v=0/1/4/5
        // and v=2/3/6/7). Reuse each input row scale across N.
        static_assert(size<0>(accum) == 8);
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < UsedM; ++m) {
          CUTLASS_PRAGMA_UNROLL
          for (int row = 0; row < 2; ++row) {
            const int local_row = get<0>(tDcD(2 * row, m, 0));
            if (local_row >= residual_m)
              continue;
            int64_t scale_row;
            if constexpr (Fused) {
              if constexpr (Routed)
                scale_row = params.sorted_token_ids[deep_scheduler.token_base + local_row];
              else
                scale_row = params.sorted_token_ids[deep_scheduler.cumsum_m_block_idx * BlockM + local_row];
            } else
              scale_row = deep_scheduler.curr_offset_a() / K + m_block_idx * BlockM + local_row;
            const float a_scale = [&]() {
              if constexpr (Routed)
                return apply_inv_scale(storage.row_scales[scale_buffer][local_row]);
              else
                return CacheRowScale ? cached_row_scale(m, row) : apply_inv_scale(params.ptr_a_scale[scale_row]);
            }();
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(accum); ++n) {
              CUTLASS_PRAGMA_UNROLL
              for (int col_half = 0; col_half < 2; ++col_half) {
                const int v = 2 * row + 4 * col_half;
                if constexpr (N % BlockN) {
                  if (get<1>(tDcD(v, m, n)) >= N - n_block_idx * BlockN)
                    continue;
                }
                const uint32_t d =
                    deep_gemm::tix::cvt_rte_bf16x2_f32(accum(v + 1, m, n) * a_scale, accum(v, m, n) * a_scale);
                *reinterpret_cast<uint32_t*>(&tDgD(v, m, n)) = d;
              }
            }
          }
        }
      };

      prefetch_row_scale();

      // mainloop
      constexpr int k_tile_count = size<2>(gB);
      uint32_t current_stage = 0;
      // Prologue
      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < kNumStages - 1; k_pipe++) {
        if (k_pipe < k_tile_count) {
          g2s(k_pipe, k_pipe);
        }
        current_stage++;
        cp_async_fence();
      }
      cp_async_wait<kNumStages - 2>();
      __syncthreads();
      load_cached_row_scale();
      if constexpr (CacheRowScale)
        __syncthreads();

#pragma unroll
      for (int n_iter = 0; n_iter < N_EXPAND; n_iter++) {
        auto process_active_rows = [&](auto active_m) {
          constexpr int UsedM = decltype(active_m)::value;
          clear(accum);
          auto process_k_tile_iteration = [&](int k_tile_idx) {
            int stage_idx = (k_tile_idx + n_iter * k_tile_count) % kNumStages;
            int next_k_tile_idx = k_tile_idx + kNumStages - 1;
            if constexpr (N_EXPAND > 1) {
              if (next_k_tile_idx == k_tile_count) {
                tBgB.data() = tBgB.data() + BlockN * K2;
                tSgS.data() = tSgS.data() + BlockN * K / 32;
              }
              if (next_k_tile_idx < k_tile_count || n_iter < N_EXPAND - 1) {
                g2s(next_k_tile_idx % k_tile_count, current_stage);
              }
            } else {
              if (next_k_tile_idx < k_tile_count) {
                g2s(next_k_tile_idx, current_stage);
              }
            }
            cp_async_fence();
            current_stage = (current_stage + 1) % kNumStages;

            if constexpr (!BatchDequant)
              s2r_copy_B_and_scale(stage_idx);

            if constexpr (Compact) {
              for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto ki) {
                constexpr int kb = decltype(ki)::value;
                if constexpr ((Routed && Fused && BlockK == 128) || kb % 2 == 0)
                  s2r_copy_B_and_scale_slice(stage_idx, kb / 2);
                if constexpr (ADistance > 0 && BlockM == 160) {
#pragma unroll
                  for (int mb = 0; mb < ADistance && mb < UsedM; ++mb)
                    copy(smem_tiled_copy_A, tCsA(_, mb, kb, stage_idx), tCrA_copy_view(_, mb, Int<0>{}));
                }
                if constexpr (Routed) {
#pragma unroll
                  for (int nb = 0; nb < WarpN / 16; ++nb) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                      clear(fp4_accum);
                      fp4_mma_dequant(nb, 2 * kb + half);
                      quant_fp32C_to_e5m2B(nb, kb, half);
                    }
                  }
                } else {
                  for_each(make_int_sequence<WarpN / (16 * DecodeN)>{}, [&](auto ng) {
                    constexpr int n0 = decltype(ng)::value * DecodeN;
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
#pragma unroll
                      for (int j = 0; j < DecodeN; ++j) {
                        clear(decode_fragment(j, half));
                        fp4_mma_dequant(n0 + j, 2 * kb + half);
                      }
                    }
#pragma unroll
                    for (int j = 0; j < DecodeN; ++j) {
#pragma unroll
                      for (int half = 0; half < 2; ++half)
                        quant_fp32C_to_e5m2B(n0 + j, kb, half);
                    }
                  });
                }
                if constexpr (ADistance < 0) {
                  s2r_copy_A(ResidentA ? k_tile_idx : stage_idx, kb);
#pragma unroll
                  for (int nb = 0; nb < WarpN / 16; ++nb) {
#pragma unroll
                    for (int mb = 0; mb < UsedM; ++mb)
                      cute::gemm(tiled_mma, tCrA(_, mb, Int<0>{}), tCrB(_, nb, Int<0>{}), accum(_, mb, nb));
                  }
                } else {
                  auto load_a = [&](int mb) {
                    copy(smem_tiled_copy_A, tCsA(_, mb, kb, stage_idx),
                         tCrA_copy_view(_, mb % (ADistance + 1), Int<0>{}));
                  };
#pragma unroll
                  for (int mb = 0; mb < ADistance && mb < UsedM; ++mb)
                    if constexpr (BlockM != 160)
                      load_a(mb);
#pragma unroll
                  for (int mb = 0; mb < UsedM; ++mb) {
                    if (mb + ADistance < UsedM)
                      load_a(mb + ADistance);
#pragma unroll
                    for (int nb = 0; nb < WarpN / 16; ++nb)
                      cute::gemm(tiled_mma, tCrA(_, mb % (ADistance + 1), Int<0>{}), tCrB(_, nb, Int<0>{}),
                                 accum(_, mb, nb));
                  }
                }
              });
            } else {
              for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block_ic) {
                constexpr int k_block = decltype(k_block_ic)::value;
                if constexpr (BatchDequant && k_block % 2 == 0)
                  s2r_copy_B_and_scale_slice(stage_idx, k_block / 2);
                // Wide-M A fragments must not overlap the FP4 dequantization
                // scratch registers; load A after the B fragment is converted.
                if constexpr (WarpM <= 64 || !BatchDequant)
                  s2r_copy_A(ResidentA ? k_tile_idx : stage_idx, k_block);

#pragma unroll
                for (int n_block = 0; n_block < WarpN / 16; n_block++) {
                  CUTE_UNROLL
                  for (int half = 0; half < 2; ++half) {
                    auto dequant_accum = decode_fragment(n_block, half);
                    clear(dequant_accum);
                    fp4_mma_dequant(n_block, 2 * k_block + half);
                    quant_fp32C_to_e5m2B(n_block, k_block, half);
                  }

                  if constexpr (!BatchDequant) {
#pragma unroll
                    for (int m_block = 0; m_block < UsedM; m_block++) {
                      cute::gemm(tiled_mma, tCrA(_, m_block, k_block), tCrB(_, n_block, k_block),
                                 accum(_, m_block, n_block));
                    }
                  }
                }
                if constexpr (BatchDequant) {
                  if constexpr (WarpM > 64)
                    s2r_copy_A(ResidentA ? k_tile_idx : stage_idx, k_block);
// Convert the whole warp N tile before consuming it, giving the
// compiler independent FP4 operations to hide conversion latency.
#pragma unroll
                  for (int n_block = 0; n_block < WarpN / 16; ++n_block) {
#pragma unroll
                    for (int m_block = 0; m_block < UsedM; ++m_block) {
                      cute::gemm(tiled_mma, tCrA(_, m_block, k_block), tCrB(_, n_block, k_block),
                                 accum(_, m_block, n_block));
                    }
                  }
                }
              });
            }

            cp_async_wait<kNumStages - 2>();
            __syncthreads();
          };

          if constexpr (k_tile_count <= 4) {
#pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx++) {
              process_k_tile_iteration(k_tile_idx);
            }
          } else {
            if constexpr (!Routed)
              process_k_tile_iteration(0);
#pragma unroll 1
            for (int k_tile_idx = Routed ? 0 : 1; k_tile_idx < k_tile_count; k_tile_idx++) {
              process_k_tile_iteration(k_tile_idx);
            }
          }

          if (n_iter > 0) {
            n_block_idx++;
            gD.data() = gD.data() + BlockN;
          }
          epilogue_cta_reduce();
          epilogue_no_tsm(active_m);
        };
        if constexpr (Routed) {
          // Mutually exclusive branches keep each tail's accumulators local.
          // Independent predicates can merge their live ranges and spill.
          dispatch_tail((residual_m + WarpsOnM - 1) / WarpsOnM, process_active_rows);
        } else
          process_active_rows(Int<WarpM / 16>{});
      }
    }
  }
};

} // namespace cutlass::gemm::kernel
#pragma clang diagnostic pop
