#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wswitch"

#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/atom/copy_traits_ppu0015_aiu.hpp"
#include "cute/algorithm/ppu_copy.hpp"
#include "dequant_w4a16.cuh"
#include "profiling_interface.hpp"
#include "scheduler_cutlass3.cuh"
#include "fused_scheduler.cuh"
#include "fused_gemm_util.cuh"
#include "utils.cuh"
#include "utils_cutlass3.h"

using namespace cute;

namespace cutlass::gemm::kernel {

template <int ShapeN, int ShapeK,
          int BlockM, int BlockN, int BlockK,
          int WarpM, int WarpN, int WarpK,
          int kNumGroups, int kNumStages, GemmType kGemmType,
          int N_EXPAND = 1>
class W4A16GEMM_MMA {
public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::float4_t; // fp4
  using ElementD = ElementA;
  using ElementAcc = float;
  using ElementScale = uint8_t; // e8m0 scale

  // basic
  static constexpr bool Fused = (kGemmType == GemmType::GroupedFused);
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
  static constexpr int N_Blocks = ceil_div(ShapeN, BlockN * N_EXPAND);
  static constexpr int scheduler_swizzle = (BlockM == 16) ? 1: 2;
  using TileScheduler = cute::conditional_t<
    Fused,
    deep_gemm::FusedGemmScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN * N_EXPAND, kNumGroups, N_Blocks, scheduler_swizzle>,
    deep_gemm::DeepGemmScheduler<kGemmType, ShapeN, ShapeK, BlockM, BlockN * N_EXPAND, kNumGroups, N_Blocks, scheduler_swizzle>
  >;
  using ArchTag = cutlass::arch::PPU0015;
  static constexpr int WarpsOnM = BlockM / WarpM;
  static constexpr int WarpsOnN = BlockN / WarpN; // WarpN = 64
  static constexpr int WarpsOnK = BlockK / WarpK;

  // fp4 mma => dequant
  using FP4MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;
  using FP4PermutationMNK = Tile<Layout<Shape<_16, Int<WarpsOnN>, _4>, Stride<_1, Int<WarpN>, _16>>, _16, Int<WarpsOnK * 32>>;
  using FP4TiledMma = TiledMMA<MMA_Atom<FP4MmaInst>, Layout<Shape<Int<WarpsOnN>, _1, Int<WarpsOnK>>>, FP4PermutationMNK>;

  // bf16 mma
  using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementA, ElementA, ElementAcc>::type;
  using PermutationMNK = Tile<
    Int<WarpsOnM * 16>,
    Layout<Shape<_16, Int<WarpsOnN>, _4>, Stride<_1, _64, _16>>,
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
    Shape<Int<WarpsOnN>, Shape<_32, Int<WarpsOnK>>>,
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

  struct Params {
    const ElementA* ptr_a;
    const uint8_t* ptr_b;
    const ElementScale* ptr_scale;
    ElementD* ptr_d;
    int num_token;
    const int* sorted_token_ids;
    typename TileScheduler::Params scheduler;

    CUTLASS_HOST
    Params(const ElementA* ptr_a_,
          const uint8_t* ptr_b_,
          const ElementScale* ptr_scale_,
          ElementD* ptr_d_)
        : ptr_a(ptr_a_),
          ptr_b(ptr_b_),
          ptr_scale(ptr_scale_),
          ptr_d(ptr_d_) {}
  };

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
    auto strideScale = make_stride(Int<K * 2>{}, _1{}, Int<K * N / 32>{});
    auto strideD = make_stride(Int<N>{}, _1{});
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    GmemTiledCopyScale gmem_tiled_copy_scale;
    gmem_tiled_copy_B.desc_.template init<uint8_t, false, BlockN, BlockK2>(nullptr, N, K2, strideB);
    gmem_tiled_copy_scale.desc_.template init<ElementScale, false, BlockN / 64, BlockK * 2>(nullptr, N / 64, K * 2, strideScale);
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

    int thread_idx_nk = (warpn_idx * WarpsOnK + warpk_idx) * NumThreadsPerWarp;
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
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {

      int coord_l = deep_scheduler.curr_group_idx;
      n_block_idx *= N_EXPAND;

      const uint8_t* ptr_b_l = params.ptr_b + int64_t(coord_l) * N * K2;
      Tensor mB_nk = make_tensor(make_gmem_ptr(ptr_b_l), make_shape(Int<N>{}, Int<K2>{}), make_stride(Int<K2>{}, _1{}));
      Tensor mB_nk_mix = make_mix_tensor_like(mB_nk);
      Tensor gB = local_tile(mB_nk_mix, TilerB{}, make_coord(n_block_idx,_));
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

      Tensor mScale_nkl = make_tensor(make_gmem_ptr(params.ptr_scale), make_shape(Int<N / 64>{}, Int<K * 2>{}, Int<L>{}), strideScale);
      Tensor mScale_nk = make_mix_tensor_like(mScale_nkl(_,_,coord_l));
      Tensor gScale = local_tile(mScale_nk, TilerScale{}, make_coord(n_block_idx,_));
      Tensor tSgS = gmem_thr_copy_scale.partition_S(gScale);

      Tensor cD = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
      uint32_t token_offsets[CopyAConfig::M_ITER];
      auto [tAgA, residual_m, gD] = [&]() {
        if constexpr(!Fused) {
          int M = deep_scheduler.curr_group_m;
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
          asm volatile("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(src(src_idx+1)), "f"(src(src_idx)));
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
          asm volatile("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(accum(i+1)), "f"(accum(i)));
          *reinterpret_cast<uint32_t*>(dst_ptr) = d;
        }
      };

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

          s2r_copy_B_and_scale(stage_idx);

          for_each(make_int_sequence<K_BLOCK_MAX >{}, [&] (auto k_block_ic) {
            constexpr int k_block = decltype(k_block_ic)::value;
            s2r_copy_A(stage_idx, k_block);

            clear(fp4_accum);
            #pragma unroll
            for (int n_block = 0; n_block < 4; n_block++) {
              fp4_mma_dequant(n_block, k_block);
              quant_fp32C_to_bf16B(n_block, k_block);

              #pragma unroll
              for (int m_block = 0; m_block < size<1>(accum); m_block++) {
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
    }
  }
};

} // namespace cutlass::gemm::kernel