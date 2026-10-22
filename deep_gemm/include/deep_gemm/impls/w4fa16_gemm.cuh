#pragma once
#include <deep_gemm/scheduler/mega_moe_work_queue.cuh>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wswitch"

#ifndef W4A16_HGRTC
    #include <deep_gemm/common/profiling_interface.cuh>
#endif

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/math.cuh>

#include <deep_gemm/tix/cvt.cuh>
#include <cute/ppu_tensor_mix.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/atom/mma_traits_ppu0015.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0015_aiu.hpp>
#include <cute/algorithm/ppu_copy.hpp>
#include <deep_gemm/impls/dequant_w4a16.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/scheduler/fused_scheduler.cuh>
#include <deep_gemm/impls/fused_gemm_util.cuh>
#include <deep_gemm/common/utils_cutlass3.cuh>

using namespace cute;

namespace cutlass::gemm::kernel {

using deep_gemm::GemmType;

template <int ShapeN, int ShapeK,
          int BlockM, int BlockN, int BlockK,
          int WarpM, int WarpN, int WarpK,
          int kNumGroups, int kNumStages, GemmType kGemmType,
          int N_EXPAND = 1, bool kMoeScatter = false, bool kFusedSitu = false,
          bool kUseReplicaWeights = false, bool kDynamicWork = false>
class W4A16GEMM_MMA {
public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::float4_t; // fp4
  using ElementD = ElementA;
  using ElementAcc = float;
  using ElementScale = uint8_t; // e8m0 scale

  // basic
  static constexpr bool Fused = (kGemmType == GemmType::MGroupedFused);
  static constexpr bool E8M0_scale = true;
  static constexpr bool kExtendedParams = kMoeScatter || kFusedSitu || kUseReplicaWeights;
  static_assert(!kMoeScatter || kGemmType == GemmType::MGroupedNoPad, "Scatter requires grouped no-pad");
  static_assert(!kFusedSitu || kGemmType == GemmType::MGroupedNoPad, "SiTU requires grouped no-pad");
  static_assert(!(kMoeScatter && kFusedSitu), "Scatter and SiTU are separate epilogues");

  static constexpr bool scale_use_ld_matrix = true;
  static constexpr int N = ShapeN;
  static constexpr int K = ShapeK;
  static constexpr int K2 = K / 2;
  // MegaMoE pads pairs of K=32 scale groups to 128 bytes per 64 columns.
  // Ordinary GEMM accepts the compact [E, N/64, K*2] scale layout.
  static constexpr int ScaleStride = kExtendedParams ? ((K + 63) / 64) * 128 : K * 2;
  static constexpr int L = kNumGroups;
  static constexpr int BlockK2 = BlockK / 2;
  static constexpr int WarpK2 = WarpK / 2;
  static constexpr int kGroupSize = 32;
  static constexpr int GroupsPerBlock = BlockK / kGroupSize;
  static constexpr int GroupsPerWarp = WarpK / kGroupSize;
  static constexpr int MmaPerGroup = kGroupSize / 16;
  using TileShape = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
  static constexpr int N_Blocks = ceil_div(ShapeN, BlockN * N_EXPAND);
  static constexpr int scheduler_swizzle = (BlockM == 16) ? 1: 2;
  using TileScheduler = cute::conditional_t<
    Fused,
    deep_gemm::FusedGemmScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN * N_EXPAND, kNumGroups, N_Blocks, scheduler_swizzle>,
    deep_gemm::DeepGemmScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN * N_EXPAND, kNumGroups, N_Blocks, scheduler_swizzle>
  >;
  using ArchTag = cutlass::arch::PPU0015;
  static constexpr int WarpsOnM = BlockM / WarpM;
  static constexpr int WarpsOnN = BlockN / WarpN;
  static_assert(WarpN == 64 || (kFusedSitu && BlockM == 128 && WarpM == 128 && WarpN == 32),
                "32-column warps require the BM128 SiTU layout");
  static constexpr int WarpsOnK = BlockK / WarpK;

  // fp4 mma => dequant
  using FP4MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;
  using FP4PermutationMNK = Tile<Layout<Shape<_16, Int<WarpsOnN>, Int<WarpN / 16>>, Stride<_1, Int<WarpN>, _16>>, _16, Int<WarpsOnK * 32>>;
  using FP4TiledMma = TiledMMA<MMA_Atom<FP4MmaInst>, Layout<Shape<Int<WarpsOnN>, _1, Int<WarpsOnK>>>, FP4PermutationMNK>;

  // bf16 mma
  using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementA, ElementA, ElementAcc>::type;
  using PermutationMNK = Tile<
    Int<WarpsOnM * 16>,
    Layout<Shape<_16, Int<WarpsOnN>, Int<WarpN / 16>>, Stride<_1, Int<WarpN>, _16>>,
    Int<WarpsOnK * 16>,
  >;
  using TiledMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<Int<WarpsOnM>, Int<WarpsOnN>, Int<WarpsOnK>>>, PermutationMNK>;
  static constexpr int MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr bool SplitAIU = (MaxThreadsPerBlock / NumThreadsPerWarp) > 1;

  // sA
  static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
  using ACopyInst = cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t>;
  using FusedOperandA = cutlass::gemm::config::Gemm_Hybrid_Operand<
    ArchTag, ElementA, false, AlignA, Int<BlockK>, MaxThreadsPerBlock,
    ACopyInst, Int<BlockM>>;
  using DefaultOperandA = cute::conditional_t<
    Fused,
    FusedOperandA,
    cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, false, Int<BlockM>, Int<BlockK>, false, 0, true>
  >;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
  using TilerA = typename GmemTiledCopyA::Tiler_MN;
  using CopyAConfig = deep_gemm::CopyAToTsmConfig<ElementA, TilerA, BlockM, BlockK, MaxThreadsPerBlock>;
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<kNumStages>{})));

  // sW => fp4 weights
  using TilerB = Shape<Int<BlockN>, Int<BlockK2>>;
  using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, uint8_t, false, Int<BlockN>, Int<BlockK2>, false, 0, true>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<BlockN>{}, Int<BlockK2>{}, Int<kNumStages>{})));

  // sScale
  using TilerScale = Shape<Int<BlockN / 64>, Int<BlockK * 2>>;
  using DefaultOperandScale = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementScale, false, Int<BlockN / 64>, Int<BlockK * 2>, false, 0, false>;
  using SmemLayoutAtomScale = typename DefaultOperandScale::SmemLayoutAtom;
  using GmemTiledCopyScale = typename DefaultOperandScale::GmemTiledCopy;
  using SmemLayoutScale = decltype(tile_to_shape(SmemLayoutAtomScale{}, make_shape(Int<BlockN / 64>{}, Int<BlockK * 2>{}, Int<kNumStages>{})));
  // custom s2r copy scale
  using ScaleCopyOp = UniversalCopy<cute::uint32_t>;
  using ScaleCopyTraits = Copy_Traits<ScaleCopyOp>;
  using SmemCopyAtomScale = Copy_Atom<ScaleCopyTraits, ElementScale>;
  using SmemThrLayoutScale = Layout<
    Shape<Int<BlockN / 64>, Shape<_32, Int<WarpsOnK>>>,
    Stride<Int<32 * WarpsOnK>, Stride<_1, _32>>
  >;
  using SmemValLayoutScale = Layout<Shape<_1, _4>>;
  using SmemTiledCopyScale = decltype(make_tiled_copy(SmemCopyAtomScale{}, SmemThrLayoutScale{}, SmemValLayoutScale{}));

  // epilogue cta reduce
  static_assert(WarpsOnM == 1 || WarpsOnK == 1);
  using SmemLayoutReduce = decltype(make_layout(make_shape(Int<16 * 64>{}, Int<WarpsOnN>{}, Int<WarpsOnK>{}), LayoutLeft{}));

  struct SharedStorage {
    union {
      struct {
        cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
        cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutB>> smem_b;
        cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScale>> smem_scale;
      }; // mainloop
      cute::array_aligned<ElementAcc, cute::cosize_v<SmemLayoutReduce>> smem_reduce; // epilogue cta reduce
      cute::array_aligned<ElementD, kMoeScatter ? BlockM * BlockN : 1> smem_moe_scatter;
    };
  };

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

  struct StandardParams {
    const ElementA* ptr_a;
    const uint8_t* ptr_b;
    const ElementScale* ptr_scale;
    ElementD* ptr_d;
    int num_token;
    const int* sorted_token_ids;
    typename TileScheduler::Params scheduler;

    CUTLASS_HOST
    StandardParams(const ElementA* ptr_a_,
          const uint8_t* ptr_b_,
          const ElementScale* ptr_scale_,
          ElementD* ptr_d_)
        : ptr_a(ptr_a_),
          ptr_b(ptr_b_),
          ptr_scale(ptr_scale_),
          ptr_d(ptr_d_) {}
  };

  struct ExtendedParams {
    const ElementA* ptr_a;
    const uint8_t* ptr_b;
    const ElementScale* ptr_scale;
    const uint8_t* ptr_b_replicas = nullptr;
    const ElementScale* ptr_scale_replicas = nullptr;
    uint32_t num_master_groups = L;
    ElementD* ptr_d;
    int num_token;
    int topk;
    const int* sorted_token_ids;
    const int64_t* packed_src_info = nullptr;
    const int64_t* sym_offsets = nullptr;
    const int* valid_m_rows = nullptr;
    int rank = 0;
    int max_tokens = 0;
    float situ_beta = 4.0f;
    float situ_linear_beta = 25.0f;
    typename TileScheduler::Params scheduler;

    CUTLASS_HOST_DEVICE
    ExtendedParams(const ElementA* ptr_a_,
          const uint8_t* ptr_b_,
          const ElementScale* ptr_scale_,
          ElementD* ptr_d_)
        : ptr_a(ptr_a_),
          ptr_b(ptr_b_),
          ptr_scale(ptr_scale_),
          ptr_d(ptr_d_) {}
  };
  struct QueueParams : ExtendedParams {
    using ExtendedParams::ExtendedParams;
    uint32_t* queue_counter = nullptr;
    uint32_t* arrival_counts = nullptr;
    uint32_t dispatch_block_m = 0;
  };
  using Params = cute::conditional_t<kDynamicWork, QueueParams,
      cute::conditional_t<kExtendedParams, ExtendedParams, StandardParams>>;


  // for constexpr patial initialize
  struct NoTensor {};

  CUTLASS_DEVICE auto init_identity(int tid) {
    int group = tid >> 2;
    int tid_in_group = tid & 3;

    const int bit = (group >> 1) * 8 + ((group & 1) ? 5 : 1);
    const uint32_t fp4_one = 1u << bit;
    uint32_t b_identity[4][4] = {};
    switch (tid_in_group) {
        case 0:
            b_identity[0][0] = fp4_one;
            b_identity[2][1] = fp4_one;
            break;
        case 1:
            b_identity[0][2] = fp4_one;
            b_identity[2][3] = fp4_one;
            break;
        case 2:
            b_identity[1][0] = fp4_one;
            b_identity[3][1] = fp4_one;
            break;
        case 3:
            b_identity[1][2] = fp4_one;
            b_identity[3][3] = fp4_one;
            break;
    }

    Tensor tCrID = make_tensor(&b_identity[0][0], Shape<Int<4>, Int<4>>{});
    return tCrID;
  }

  template<int mode=0, typename Tensor>
  CUTLASS_DEVICE constexpr
  auto reshape_warpk(Tensor& t) {
    static_assert(mode == 0 || mode == 1);
    if constexpr(mode == 0) { // sB: (k_tiles, ..., stage), sScale: (k_tiles / MmaPerGroup, ..., stage)
      constexpr int k = decltype(size<mode>(t))::value;
      constexpr int ko = k / WarpsOnK;
      auto atom_tiler = make_layout(make_shape(Int<WarpsOnK>{}, Int<ko>{}), make_stride(Int<ko>{}, _1{}));
      auto tiler = make_shape(atom_tiler, size<1>(t));
      return flat_divide(t, tiler)(_,_,0,0,_);
    } else { // sA: (..., blockk, stage) where blockk = 16 * k_tiles.
      constexpr int k = BlockK / 16;
      constexpr int ko = k / WarpsOnK;
      auto atom_tiler = make_layout(
        make_shape(_16{}, make_shape(Int<WarpsOnK>{}, Int<ko>{})),
        make_stride(_1{}, make_stride(Int<ko * 16>{}, _16{}))
      );
      auto tiler = make_shape(size<0>(t), atom_tiler);
      return flat_divide(t, tiler)(_,_,0,0,_);
    }
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
    Tensor tCrID = init_identity(lane_idx);
    TileScheduler deep_scheduler{params.scheduler};
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K2,PIPE)
    Tensor sScale = make_tensor(make_smem_ptr(storage.smem_scale.data()), SmemLayoutScale{}); // (BLN/64,BLK*2,PIPE)
    Tensor sReduce = make_tensor(make_smem_ptr<ElementAcc>(storage.smem_reduce.data()), SmemLayoutReduce{}); // (16*64,WARPS_ON_N,WARPS_ON_K)
    auto sReduce_tile = local_tile(sReduce, make_tile(_8{}), make_coord(_)); // (8,128,WARPS_ON_N,WARPS_ON_K)

    auto strideA = make_stride(Int<K>{}, _1{});
    auto strideB = make_stride(Int<K2>{}, _1{}, Int<N * K2>{});
    auto strideScale = make_stride(Int<ScaleStride>{}, _1{}, Int<(N / 64) * ScaleStride>{});
    auto strideD = make_stride(Int<N>{}, _1{});
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    GmemTiledCopyScale gmem_tiled_copy_scale;
    gmem_tiled_copy_B.desc_.template init<uint8_t, false, BlockN, BlockK2>(nullptr, N, K2, strideB);
    gmem_tiled_copy_scale.desc_.template init<ElementScale, false, BlockN / 64, BlockK * 2>(nullptr, N / 64, ScaleStride, strideScale);
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
    Tensor fp4_tCrB = fp4_thr_mma.partition_fragment_A(sB(_,_,0));
    Tensor fp4_accum = partition_fragment_C(fp4_tiled_mma, Shape<Int<BlockN>, Int<BlockK>>{})(_,_,0);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
    Tensor tCrB = make_tensor<typename TiledMma::FrgTypeB>(partition_shape_B(tiled_mma, take<1,3>(TileShape{})));
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto [smem_thr_copy_A, tCsA] = [&]() {
      if constexpr(Fused) {
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
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);
    constexpr int K_BLOCK_MAX = size<2>(tCrA_copy_view);

    auto smem_tiled_copy_B = make_tiled_copy_A(SmemCopyAtomB{}, fp4_tiled_mma); // fp4 B is matrixA in fp4 gemm
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx_B);
    auto mix_sB = make_mix_tensor_like(sB);
    Tensor tCsB = smem_thr_copy_B.partition_S(mix_sB);
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(fp4_tCrB);

    int thread_idx_nk = ((warpn_idx / (64 / WarpN)) * WarpsOnK + warpk_idx) * NumThreadsPerWarp;
    if constexpr(not scale_use_ld_matrix) {
      thread_idx_nk += lane_idx;
    } else {
      thread_idx_nk += lane_idx * 4; // thread0-7 get the row address
    }
    SmemTiledCopyScale smem_tiled_copy_scale;
    auto smem_thr_copy_scale = smem_tiled_copy_scale.get_thread_slice(thread_idx_nk);
    Tensor tCsS = smem_thr_copy_scale.partition_S(sScale);
    Tensor tCrS_copy_view = make_tensor_like(tCsS(_,_,_,0));
    Tensor tCrS = recast<uint32_t>(tCrS_copy_view);

    uint32_t m_block_idx, n_block_idx;
    __shared__ uint32_t queue_ticket;
    const auto fetch_work = [&]() {
      if constexpr (!kDynamicWork) return deep_scheduler.fetch_next_work(m_block_idx, n_block_idx);
      else {
        static_assert(kFusedSitu && kNumGroups < 128 && N_EXPAND == 1);
        const deep_gemm::mega_moe::WorkQueue queue{params.queue_counter, &queue_ticket};
        queue.prime(thread_idx);
        const uint32_t work = queue.current();
        while (deep_scheduler.curr_group_idx < L) {
          deep_scheduler.curr_group_m = __ldg(params.scheduler.grouped_layout + deep_scheduler.curr_group_idx);
          const uint32_t mb = deep_gemm::math::ceil_div(deep_scheduler.curr_group_m, uint32_t(BlockM));
          if (work < (deep_scheduler.curr_cumsum + mb) * N_Blocks) {
            deep_scheduler.get_swizzled_block_idx(mb, work - deep_scheduler.curr_cumsum * N_Blocks, m_block_idx, n_block_idx);
            const uint32_t row = m_block_idx * BlockM;
            const uint32_t valid = __ldg(params.valid_m_rows + deep_scheduler.curr_group_idx);
            if (row < valid) {
              const uint32_t first = (deep_scheduler.curr_cumsum_m + row) / params.dispatch_block_m;
              const uint32_t last = (deep_scheduler.curr_cumsum_m + cute::min(row + uint32_t(BlockM), valid) - 1) / params.dispatch_block_m;
              for (uint32_t g = first; g <= last; ++g)
                while (deep_gemm::tix::ld_acq(params.arrival_counts + g) == 0) {}
            }
            return true;
          }
          deep_scheduler.curr_cumsum += mb;
          deep_scheduler.curr_cumsum_m += deep_scheduler.curr_group_m;
          ++deep_scheduler.curr_group_idx;
        }
        return false;
      }
    };
    #pragma clang loop licm(disable)
    while (fetch_work()) {

      int coord_l = deep_scheduler.curr_group_idx;
      n_block_idx *= N_EXPAND;
      int activation_rows = 0;
      if constexpr (!Fused) activation_rows = deep_scheduler.curr_group_m;
      if constexpr (kFusedSitu || kMoeScatter) {
        if (params.valid_m_rows != nullptr)
          activation_rows = __ldg(params.valid_m_rows + coord_l);
        // Keep padded expert offsets, but neither load nor compute empty tiles.
        if (m_block_idx * BlockM >= activation_rows) {
          // A skipped tile never enters the mainloop's CTA barriers. Retire
          // every ticket read before the leader can overwrite shared_ticket.
          if constexpr (kDynamicWork) __syncthreads();
          continue;
        }
      }

      int weight_group = coord_l;
      const uint8_t* weight_base = params.ptr_b;
      const ElementScale* scale_base = params.ptr_scale;
      if constexpr (kExtendedParams) {
        if (params.ptr_b_replicas != nullptr && coord_l >= static_cast<int>(params.num_master_groups)) {
          weight_group -= params.num_master_groups;
          weight_base = params.ptr_b_replicas;
          scale_base = params.ptr_scale_replicas;
        }
      }
      const uint8_t* ptr_b_l = weight_base + int64_t(weight_group) * N * K2;
      Tensor mB_nk = make_tensor(make_gmem_ptr(ptr_b_l), make_shape(Int<N>{}, Int<K2>{}), make_stride(Int<K2>{}, _1{}));
      Tensor mB_nk_mix = make_mix_tensor_like(mB_nk);
      Tensor gB = local_tile(mB_nk_mix, TilerB{}, make_coord(n_block_idx,_));
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

      Tensor mScale_nkl = make_tensor(make_gmem_ptr(scale_base), make_shape(Int<N / 64>{}, Int<ScaleStride>{}, Int<L>{}), strideScale);
      Tensor mScale_nk = make_mix_tensor_like(mScale_nkl(_,_,weight_group));
      Tensor gScale = local_tile(mScale_nk, TilerScale{}, make_coord(n_block_idx,_));
      Tensor tSgS = gmem_thr_copy_scale.partition_S(gScale);

      Tensor cD = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
      uint32_t token_offsets[CopyAConfig::M_ITER];
      auto [tAgA, residual_m, gD] = [&]() {
        if constexpr(!Fused) {
          int M = activation_rows;
          gmem_tiled_copy_A.desc_.template init<ElementA, false, BlockM, BlockK>(nullptr, M, K, strideA);
          Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_a + deep_scheduler.curr_offset_a()), make_shape(M, Int<K>{}), strideA); // (m,k)
          Tensor mA_mk = make_mix_tensor_like(mA_mkl); // (m,k)
          Tensor gA = local_tile(mA_mk, TileShape{}, make_coord(m_block_idx, n_block_idx, _), Step<_1, X,_1>{}); // (BLK_M,BLK_K,blocks_k)
          Tensor tAgA = gmem_thr_copy_A.partition_S(gA);

          Tensor mD_mn = make_tensor(make_gmem_ptr(params.ptr_d + deep_scheduler.curr_offset_c()), make_shape(M, Int<N>{}), strideD); // (m,n)
          Tensor gD = local_tile(mD_mn, TileShape{}, make_coord(m_block_idx, n_block_idx, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N)
          int residual_m = M - BlockM * m_block_idx;

          return cute::make_tuple(tAgA, residual_m, gD);
        } else {
          const int* blk_token_base = params.sorted_token_ids + deep_scheduler.cumsum_m_block_idx * BlockM;
          deep_gemm::prefetch_A_token_offsets<ElementA, TilerA, BlockM, BlockK, MaxThreadsPerBlock>(
              token_offsets, blk_token_base, params.num_token, thread_idx);
          Tensor mD_mn = make_tensor(make_gmem_ptr(params.ptr_d + deep_scheduler.curr_block_m_offset * N), make_shape(Int<BlockM>{}, Int<N>{}), strideD); // (m,n)
          Tensor gD = local_tile(mD_mn, TileShape{}, make_coord(_0{}, n_block_idx, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N)
          int residual_m = deep_scheduler.valid_m_in_block;
          return cute::make_tuple(NoTensor{}, residual_m, gD);
        }
      }();

      auto g2s_copy_A = [&](int k_tile_iter, int k_pipe) {
        if constexpr(!Fused) { // aiu copy
          if (warp_idx == 0) {
            copy(gmem_tiled_copy_A, tAgA(_,_,_,k_tile_iter), tAsA(_,_,_,k_pipe));
          }
        } else { // async copy
          deep_gemm::copy_A_to_tsm<ElementA, ACopyInst, TilerA, BlockM, BlockK, K>(
              tAsA(_,_,_,k_pipe), params.ptr_a, token_offsets,
              BlockK * k_tile_iter, params.num_token, thread_idx,
              Int<MaxThreadsPerBlock>{});
        }
      };

      auto g2s_copy_B_and_scale = [&](int k_tile_iter, int k_pipe) {
        constexpr int copy_warp_idx = SplitAIU ? 1 : 0;
        if (warp_idx == copy_warp_idx) {
          copy(gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,k_pipe));
          copy(gmem_tiled_copy_scale, tSgS(_,_,_,k_tile_iter), tSsS(_,_,_,k_pipe));
        }
      };

      auto g2s = [&](int k_tile_iter, int k_pipe) {
        g2s_copy_A(k_tile_iter, k_pipe);
        g2s_copy_B_and_scale(k_tile_iter, k_pipe);
      };

      auto s2r_copy_A = [&](int stage_idx, int k_block) {
        copy(smem_tiled_copy_A, tCsA(_,_,k_block,stage_idx), tCrA_copy_view(_,_,k_block));
      };

      auto s2r_copy_B_and_scale = [&](int stage_idx) {
        copy(smem_tiled_copy_B, tCsB(_,_,_,stage_idx), tCrB_copy_view(_,_,_));
        if constexpr(not scale_use_ld_matrix) {
          copy(smem_tiled_copy_scale, tCsS(_,_,_,stage_idx), tCrS_copy_view(_,_,_));
        } else {
          #pragma unroll
          for (int fp4_k_block = 0; fp4_k_block < size<2>(tCsS); fp4_k_block++) {
            uint32_t d;
            uint32_t addr = cute::cast_smem_ptr_to_uint(cute::raw_pointer_cast(tCsS(_,_,fp4_k_block,stage_idx).data()));
            asm volatile("ppu.tc02.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];" : "=r"(d) : "r"(addr));
            // Scale groups cover 64 columns; each 32-column warp selects its half.
            if constexpr (WarpN == 32)
              d = __shfl_xor_sync(0xffffffffu, d, (warpn_idx & 1) * 2);
            tCrS(0,0,fp4_k_block) = d;
          }
        }
      };

      auto fp4_mma_dequant = [&](int n_block, int k_block) {
        Tensor s = make_tensor<uint32_t>(Int<4>{});
        s(0) = tCrS(0,0,k_block / 4);  // scale W
        s(1) = 0x7f7f7f7fu;  // 1.0 E8M0 for identity matrix
        s(2) = n_block;      // a-psel
        s(3) = 0;            // b-psel

        Tensor d = fp4_accum(_,n_block);
        Tensor a = fp4_tCrB(_,n_block,k_block / 4);
        Tensor b = tCrID(_,k_block % 4);
        Tensor c = fp4_accum(_,n_block);
        cute::mma_unpack(MMA_Atom<FP4MmaInst>{}, d, a, b, c, s);
      };

      auto quant_fp32C_to_bf16B = [&](int n_block, int k_block) {
        Tensor src = fp4_accum(_,n_block);
        Tensor dst = tCrB(_,n_block,k_block);
        auto fp32_to_bf16_helper = [&](int dst_idx, int src_idx) {
          ElementA* dst_ptr = &dst(dst_idx);
          uint32_t d;
          d = deep_gemm::tix::cvt_rte_bf16x2_f32(src(src_idx+1), src(src_idx));
          *reinterpret_cast<uint32_t*>(dst_ptr) = d;
        };
        fp32_to_bf16_helper(0, 0);
        fp32_to_bf16_helper(2, 4);
        fp32_to_bf16_helper(4, 2);
        fp32_to_bf16_helper(6, 6);
      };

      auto epilogue_cta_reduce = [&]() {
        if constexpr(WarpsOnK == 1) return;
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

      auto epilogue_no_tsm = [&]() {
        if constexpr(WarpsOnK > 1) {
          if (warpk_idx > 0) return;
        }

        constexpr int align_elems = 2;
        Tensor tDcD = thr_mma.partition_C(cD);
        if constexpr(kFusedSitu) {
#if defined(__HGGC_ARCH__)
          static_assert(__HGGC_ARCH__ == 150,
                        "W4A16 fused SiTU requires PPU0015.");
#endif
          static_assert(WarpsOnK == 1,
                        "W4A16 fused SiTU requires one K warp.");
          static_assert(BlockN % 16 == 0,
                        "W4A16 fused SiTU requires complete G=8 pairs.");

          // The PPU0015 m16n16 accumulator assigns one thread's G=8 physical
          // gate/up pair as [gate row0 x2, gate row1 x2, up row0 x2,
          // up row1 x2]. Pair these register fragments directly. This keeps
          // the required FP32 -> BF16 activation boundary while eliminating
          // the dense shared tile and both CTA barriers from the first fused
          // implementation.
          static_assert(size(tDcD) % 8 == 0,
                        "SiTU accumulator groups must contain gate/up pairs.");
          const float inv_beta = __ppu_rcpf(params.situ_beta);
          const float inv_linear_beta =
              __ppu_rcpf(params.situ_linear_beta);
          CUTLASS_PRAGMA_UNROLL
          for (int group = 0; group < size(tDcD); group += 8) {
            CUTLASS_PRAGMA_UNROLL
            for (int row_in_group = 0; row_in_group < 2; ++row_in_group) {
              const int gate_idx = group + row_in_group * 2;
              const int up_idx = gate_idx + 4;
              const int local_m = static_cast<int>(get<0>(tDcD(gate_idx)));
              if (local_m >= residual_m) continue;
              const int physical_gate_col =
                  static_cast<int>(get<1>(tDcD(gate_idx)));

              uint32_t gate_bits;
              uint32_t up_bits;
              asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n"
                           : "=r"(gate_bits)
                           : "f"(accum(gate_idx + 1)),
                             "f"(accum(gate_idx)));
              asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n"
                           : "=r"(up_bits)
                           : "f"(accum(up_idx + 1)),
                             "f"(accum(up_idx)));
              const auto gate_pair =
                  *reinterpret_cast<const __ppu_bfloat162*>(&gate_bits);
              const auto up_pair =
                  *reinterpret_cast<const __ppu_bfloat162*>(&up_bits);
              const float2 gate = __bfloat1622float2(gate_pair);
              const float2 up = __bfloat1622float2(up_pair);

              const float gate_out0 = params.situ_beta
                  * cutlass::fast_tanh(gate.x * inv_beta)
                  * __ppu_sgmdf(gate.x);
              const float gate_out1 = params.situ_beta
                  * cutlass::fast_tanh(gate.y * inv_beta)
                  * __ppu_sgmdf(gate.y);
              const float up_out0 = params.situ_linear_beta
                  * cutlass::fast_tanh(up.x * inv_linear_beta);
              const float up_out1 = params.situ_linear_beta
                  * cutlass::fast_tanh(up.y * inv_linear_beta);

              uint32_t out_bits;
              asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n"
                           : "=r"(out_bits)
                           : "f"(gate_out1 * up_out1),
                             "f"(gate_out0 * up_out0));
              const int64_t global_m =
                  static_cast<int64_t>(deep_scheduler.curr_cumsum_m)
                  + static_cast<int64_t>(m_block_idx) * BlockM + local_m;
              const int compact_n =
                  static_cast<int>(n_block_idx) * (BlockN / 2)
                  + (physical_gate_col / 16) * 8
                  + physical_gate_col % 8;
              *reinterpret_cast<uint32_t*>(
                  params.ptr_d + global_m * (N / 2) + compact_n) = out_bits;
            }
          }
          return;
        }

        if constexpr(kMoeScatter) {
#if defined(__HGGC_ARCH__)
          static_assert(__HGGC_ARCH__ == 150,
                        "W4A16 direct MoE scatter requires PPU0015.");
#endif
          static_assert(WarpsOnK == 1,
                        "W4A16 direct MoE scatter requires one K warp.");
          static_assert(BlockN % 256 == 0,
                        "W4A16 direct MoE scatter requires full warp stores (256 columns).");
          static_assert(N % BlockN == 0,
                        "W4A16 direct MoE scatter requires full N tiles.");
          static_assert(
              BlockM * BlockN * sizeof(ElementD) <= 262144,
              "W4A16 direct MoE scatter tile exceeds PPU0015 shared memory.");

          // Reuse the mainloop/epilogue union after GEMM. The accumulator is
          // distributed across the CTA, so transpose BF16 pairs into a dense
          // row-major shared tile before assigning complete rows to warps.
          auto* tile_smem = reinterpret_cast<uint32_t*>(
              storage.smem_moe_scatter.data());
          // Mainloop and scatter reuse the same dynamic shared-memory arena.
          __syncthreads();
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tDcD); i += 2) {
            const int row = static_cast<int>(get<0>(tDcD(i)));
            const int col = static_cast<int>(get<1>(tDcD(i)));
            uint32_t pair;
            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n"
                         : "=r"(pair) : "f"(accum(i + 1)), "f"(accum(i)));
            tile_smem[row * (BlockN / 2) + col / 2] = pair;
          }
          __syncthreads();

          const int row_start = static_cast<int>(m_block_idx) * BlockM;
          const int group_valid = __ldg(params.valid_m_rows + coord_l);
          const int valid_m = group_valid > row_start
              ? cute::min(BlockM, group_valid - row_start) : 0;
          const int pool_row_base =
              static_cast<int>(deep_scheduler.curr_cumsum_m) + row_start;
          const int n_start = static_cast<int>(n_block_idx) * BlockN;
          constexpr int kNumWarps = MaxThreadsPerBlock / NumThreadsPerWarp;
          const int scatter_warp_idx = thread_idx / NumThreadsPerWarp;

          for (int row = scatter_warp_idx; row < valid_m;
               row += kNumWarps) {
            const int64_t packed = __ldg(
                params.packed_src_info + pool_row_base + row);
            const uint32_t packed_low = static_cast<uint32_t>(packed);
            const uint32_t dst_rank = static_cast<uint32_t>(packed >> 32);
            const uint32_t dst_token = packed_low & 0x00ffffffu;
            const uint32_t dst_topk = packed_low >> 24;
            auto* local_dst_row = params.ptr_d +
                (static_cast<int64_t>(dst_topk) * params.max_tokens +
                 dst_token) * N;
            auto* dst_row = reinterpret_cast<ElementD*>(
                params.sym_offsets[dst_rank] +
                reinterpret_cast<int64_t>(local_dst_row));
            const auto* src_vectors = reinterpret_cast<const uint4*>(
                tile_smem + row * (BlockN / 2));
            auto* dst_vectors = reinterpret_cast<uint4*>(dst_row + n_start);
            CUTLASS_PRAGMA_UNROLL
            for (int vector_base = 0; vector_base < BlockN / 8;
                 vector_base += NumThreadsPerWarp) {
              const int vector = vector_base + lane_idx;
              const uint4 payload = __ppu_shared_load_bulk_b32x4(
                  src_vectors + vector);
              if (dst_rank != static_cast<uint32_t>(params.rank)) {
                __ppu_remote_store_bulk_ncp_b32x4(
                    reinterpret_cast<void*>(dst_vectors + vector), payload);
              } else {
                __ppu_global_store_bulk_b32x4(
                    reinterpret_cast<void*>(dst_vectors + vector), payload);
              }
            }
          }
          // Persistent CTAs may immediately reuse this union for another tile.
          __syncthreads();
          return;
        }


        auto tDgD = thr_mma.partition_C(gD);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tDcD); i += align_elems) {
          bool cond = get<0>(tDcD(i)) < residual_m;
          ElementD* dst_ptr = &tDgD(i);
          if constexpr(N % BlockN) {
            cond = cond && get<1>(tDcD(i)) < (N - n_block_idx * BlockN);
          }
          if (!cond) continue;
          uint32_t d;
          d = deep_gemm::tix::cvt_rte_bf16x2_f32(accum(i+1), accum(i));
          *reinterpret_cast<uint32_t*>(dst_ptr) = d;
        }
      };

      // Choose the valid M fragments outside the K loop; preserve MMA order.
      auto run_m_tile = [&](auto m_fragments) {
        // mainloop
        constexpr int k_tile_count = size<2>(gB);
        int current_stage = 0;
        // Prologue
        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < kNumStages-1; k_pipe++) {
          if (k_pipe < k_tile_count) {
            g2s(k_pipe, k_pipe);
          }
          current_stage++;
          cp_async_fence();
        }
        cp_async_wait<kNumStages-2>();
        __syncthreads();

        #pragma unroll
        for (int n_iter = 0; n_iter < N_EXPAND; n_iter++) {
          clear(accum);
          auto process_k_tile_iteration = [&](int k_tile_idx) {
            int stage_idx = (k_tile_idx + n_iter * k_tile_count) % kNumStages;
            int next_k_tile_idx = k_tile_idx + kNumStages - 1;
            if constexpr(N_EXPAND > 1) {
              if (next_k_tile_idx == k_tile_count) {
                tBgB.data() = tBgB.data() + BlockN * K2;
                tSgS.data() = tSgS.data() + (BlockN / 64) * ScaleStride;
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

            s2r_copy_B_and_scale(stage_idx);

            for_each(make_int_sequence<K_BLOCK_MAX >{}, [&] (auto k_block_ic) {
              constexpr int k_block = decltype(k_block_ic)::value;
              s2r_copy_A(stage_idx, k_block);

              clear(fp4_accum);
              #pragma unroll
              for (int n_block = 0; n_block < WarpN / 16; n_block++) {
                fp4_mma_dequant(n_block, k_block);
                quant_fp32C_to_bf16B(n_block, k_block);

                #pragma unroll
                for (int m_block = 0; m_block < decltype(m_fragments)::value; m_block++) {
                  cute::gemm(tiled_mma, tCrA(_,m_block,k_block), tCrB(_,n_block,k_block), accum(_,m_block,n_block));
                }
              }
            });

            cp_async_wait<kNumStages - 2>();
            __syncthreads();
          };

          if constexpr(k_tile_count <= 4) {
            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx++) {
              process_k_tile_iteration(k_tile_idx);
            }
          } else {
            #pragma unroll 1
            for (int k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx++) {
              process_k_tile_iteration(k_tile_idx);
            }
          }

          if (n_iter > 0) {
            n_block_idx++;
            gD.data() = gD.data() + BlockN;
          }
          epilogue_cta_reduce();
          epilogue_no_tsm();
        }
      };
      if constexpr ((kFusedSitu || kMoeScatter) && BlockM == 128 &&
                    (WarpM == 64 || (kFusedSitu && WarpM == 128 && WarpN == 32))) {
        constexpr int rows_per_fragment = WarpsOnM * 16;
        if (kFusedSitu && rows_per_fragment == 16 && residual_m <= 16) run_m_tile(cute::Int<1>{});
        else if (residual_m <= 32) run_m_tile(cute::Int<32 / rows_per_fragment>{});
        else if (residual_m <= 64) run_m_tile(cute::Int<64 / rows_per_fragment>{});
        else if (residual_m <= 96) run_m_tile(cute::Int<96 / rows_per_fragment>{});
        else run_m_tile(cute::Int<size<1>(accum)>{});
      } else {
        run_m_tile(cute::Int<size<1>(accum)>{});
      }
    }
    if constexpr (kMoeScatter) {
#if __HGGC_ARCH__ == 150
      // Drain once after all tiles, on every local/remote store producer.
      __ppu_wait_remote_vstcnt(0);
      __ppu_wait_vstcnt(0);
#endif
    }
  }
};

} // namespace cutlass::gemm::kernel
