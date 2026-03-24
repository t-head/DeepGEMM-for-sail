#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wcuda-compat"
#pragma clang diagnostic ignored "-Wswitch"

#include "ppu/cute/tensor_mix.hpp"
#include "ppu/gemm/config/gemm_operands.hpp"
#include "ppu/cute/atom/mma_traits_acompute10000.hpp"
#include "ppu/cute/atom/mma_traits_acompute10500.hpp"
#include "ppu/cute/atom/copy_traits_acompute10000_aiu.hpp"
#include "ppu/cute/atom/copy_traits_acompute10500_aiu.hpp"
#include "ppu/cute/algorithm/copy.hpp"
#include "ppu/cutlass/fast_numeric_conversion_for_mix_gemm.h"
#include "profiling_interface.hpp"
#include "scheduler_cutlass3.cuh"
#include "utils.cuh"
#include "utils_cutlass3.h"

using namespace cute;

namespace cutlass::gemm::kernel {

template <int ShapeN, int ShapeK,
          int BlockM, int BlockN, int BlockK,
          int WarpM, int WarpN, int WarpK,
          int kNumGroups, int kNumStages, GemmType kGemmType,
          int kGroupSize = 32>
class W4A16GEMM {
public:
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::int4_t;
  using ElementScale = ElementA;
  using ElementD = ElementA;
  using ElementAcc = float;

  // basic
  static constexpr int N = ShapeN;
  static constexpr int K = ShapeK;
  static constexpr int L = kNumGroups;
  // Now only support WarpK >= groupsize, TODO: group_size > WarpK
  static_assert(WarpK >= kGroupSize && WarpK % kGroupSize == 0);
  static constexpr int GroupsPerBlock = BlockK / kGroupSize;
  static constexpr int GroupsPerWarp = WarpK / kGroupSize;
  static constexpr int MmaPerGroup = kGroupSize / 16;
  using TileShape = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
  using TileScheduler = deep_gemm::DeepGemmScheduler<GemmType::GroupedNoPad, ShapeN, ShapeK, BlockM, BlockN, kNumGroups>;
  static constexpr bool FusedLoadA = (kGemmType == GemmType::GroupedFused);

  // mma
  using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ElementA, ElementA, ElementAcc>::type;
  static constexpr int WarpsOnM = BlockM / WarpM;
  static constexpr int WarpsOnN = BlockN / 64; // WarpN = 64
  static constexpr int WarpsOnK = BlockK / WarpK;
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
  using ACopyInst = cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t>;
  using DefaultOperandA = cute::conditional_t<
    FusedLoadA,
    cutlass::gemm::config::DefaultGemm_TensorOpSm80_Operand<ElementA, false, AlignA, Int<BlockK>, MaxThreadsPerBlock, ACopyInst>,
    cutlass::gemm::config::DefaultGemm_AIU_Operand<ElementA, false, Int<BlockM>, Int<BlockK>, false, 0, true>
  >;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
  using TilerA = typename GmemTiledCopyA::Tiler_MN;
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<kNumStages>{})));

  // sB
  using TileB = Shape<Int<BlockK / 16>, Int<BlockN * 2>>;
  using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<int, false, decltype(get<0>(TileB{})), decltype(get<1>(TileB{})), false, 0, false>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(get<0>(TileB{}), get<1>(TileB{}), Int<kNumStages>{})));
  // custom s2r copy B
  using CopyOp = UniversalCopy<cute::uint128_t>;
  using CopyTraits = Copy_Traits<CopyOp>;
  using SmemCopyAtomB = Copy_Atom<CopyTraits, int>;
  using SmemThrLayoutB = Layout<
    Shape<Int<WarpsOnK>, Shape<_32, Int<WarpsOnN>>>,
    Stride<Int<32 * WarpsOnN>, Stride<_1, _32>>
  >;
  using SmemValLayoutB = Layout<Shape<_1, _4>>;
  using SmemTiledCopyB = decltype(make_tiled_copy(SmemCopyAtomB{}, SmemThrLayoutB{}, SmemValLayoutB{}));

  // sScale
  using TileScale = Shape<Int<GroupsPerBlock>, Int<BlockN>>;
  using DefaultOperandScale = cutlass::gemm::config::DefaultGemm_AIU_Operand<ElementScale, false, Int<GroupsPerBlock>, Int<BlockN>, false, 0, false>;
  using SmemLayoutAtomScale = typename DefaultOperandScale::SmemLayoutAtom;
  using GmemTiledCopyScale = typename DefaultOperandScale::GmemTiledCopy;
  using SmemLayoutScale = decltype(tile_to_shape(SmemLayoutAtomScale{}, make_shape(Int<GroupsPerBlock>{}, Int<BlockN>{}, Int<kNumStages>{})));
  // custom s2r copy scale
  using SmemCopyAtomScale = Copy_Atom<CopyTraits, ElementScale>;
  using SmemThrLayoutScale = Layout<
    Shape<Int<WarpsOnK>, Shape<_8, Int<WarpsOnN>>>,
    Stride<Int<8 * WarpsOnN>, Stride<_1, _8>>
  >;
  using SmemValLayoutScale = Layout<Shape<_1, _8>>;
  using SmemTiledCopyScale = decltype(make_tiled_copy(SmemCopyAtomScale{}, SmemThrLayoutScale{}, SmemValLayoutScale{}));

  // epilogue cta reduce
  static_assert(WarpsOnM == 1 || WarpsOnK == 1); // TODO: support WarpsOnM >= 1 reduce.
  using SmemLayoutReduce = decltype(make_layout(make_shape(Int<16 * 64>{}, Int<WarpsOnN>{}, Int<WarpsOnK>{}), LayoutLeft{}));

  // epilogue
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using EpilogueCopyOp = AutoVectorizingCopyWithAssumedAlignment<128>;
  using EpilogueConfig = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<
    EpilogueCopyOp, ElementD, AlignmentD, Int<BlockM>, Int<BlockN>, Int<BlockM / WarpM>, MaxThreadsPerBlock / WarpsOnK>;
  using SmemLayoutC = typename EpilogueConfig::SmemLayoutO;
  using CopyAtomR2S = Copy_Atom<EpilogueCopyOp, ElementD>;
  using TiledCopyS2R = typename EpilogueConfig::GmemTiledCopyO;
  using CopyAtomR2G = Copy_Atom<EpilogueCopyOp, ElementD>;

  struct SharedStorage {
    union {
      struct {
        cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
        cute::array_aligned<int, cute::cosize_v<SmemLayoutB>> smem_b;
        cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScale>> smem_scale;
      }; // mainloop
      cute::array_aligned<ElementAcc, cute::cosize_v<SmemLayoutReduce>> smem_reduce; // epilogue cta reduce
      cute::array_aligned<ElementD, cute::cosize_v<SmemLayoutC>> smem_c; // epilogue
    };
  };

  struct Arguments {
    const ElementA* ptr_a;
    const int* ptr_b;
    const ElementScale* ptr_scale;
    ElementD* ptr_d;
    int *sorted_token_ids;
    typename TileScheduler::Params scheduler;
  };
  using Params = Arguments;

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
    TileScheduler deep_scheduler{params.scheduler};
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_K/16,BLK_N*2,PIPE)
    Tensor sScale = make_tensor(make_smem_ptr(storage.smem_scale.data()), SmemLayoutScale{}); // (BLK_K/GROUP_SIZE,BLK_N,PIPE)
    Tensor sReduce = make_tensor(make_smem_ptr<ElementAcc>(storage.smem_reduce.data()), SmemLayoutReduce{}); // (16*64,WARPS_ON_N,WARPS_ON_K)
    auto sReduce_tile = local_tile(sReduce, make_tile(_8{}), make_coord(_)); // (8,128,WARPS_ON_N,WARPS_ON_K)
    Tensor sC = make_tensor(make_smem_ptr(storage.smem_c.data()), SmemLayoutC{}); // (16*WARPS_ON_M,BLK_N)
    auto sC_tile = make_shape(size<0>(sC), size<1>(sC)); // (16*WARPS_ON_M,BLK_N)
    static_assert(size<1>(sC) == BlockN);

    auto strideA = make_stride(Int<K>{}, _1{});
    auto strideB = make_stride(Int<N * 2>{}, _1{}, Int<K * N / 8>{});
    auto strideScale = make_stride(Int<N>{}, _1{}, Int<K * N / kGroupSize>{});
    auto strideD = make_stride(Int<N>{}, _1{});
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    GmemTiledCopyScale gmem_tiled_copy_scale;
    gmem_tiled_copy_B.desc_.template init<int, false, get<0>(TileB{}), get<1>(TileB{})>(nullptr, K / 16, N * 2, strideB);
    gmem_tiled_copy_scale.desc_.template init<ElementScale, false, GroupsPerBlock, BlockN>(nullptr, K / kGroupSize, N, strideScale);
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    auto gmem_thr_copy_scale = gmem_tiled_copy_scale.get_slice(thread_idx);
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);
    Tensor tSsS = gmem_thr_copy_scale.partition_D(sScale);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
    Tensor tCrB = make_tensor<typename TiledMma::FrgTypeB>(partition_shape_B(tiled_mma, take<1,3>(TileShape{}))); // manual partition_fragment_B
    Tensor tCrB_dequant_view = make_tensor(tCrB.data(), Layout<Shape<_4, _2, _4, decltype(size<2>(tCrB))>, Stride<_1, _4, _8, _32>>{});
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    if constexpr(!FusedLoadA) {
      smem_tiled_copy_A.smem_base_ = storage.smem_a.data();
    }
    auto smem_thr_copy_A = [&]() {
      if constexpr(FusedLoadA) {
        return smem_tiled_copy_A.get_thread_slice(thread_idx);
      } else {
        return smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
      }
    }();
    auto tCsA = [&]() {
      if constexpr(FusedLoadA) {
        auto sA_r2s = reshape_warpk<1>(sA);
        return smem_thr_copy_A.partition_S(sA_r2s);
      } else {
        auto mix_sA = make_mix_tensor_like(sA);
        auto sA_r2s = reshape_warpk<1>(mix_sA);
        return smem_thr_copy_A.partition_S(sA_r2s);
      }
    }();
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA); // (CPY,CPY_M,CPY_K)

    SmemTiledCopyB smem_tiled_copy_B;
    int threadidx_nk = (warp_idx / WarpsOnM) * NumThreadsPerWarp + lane_idx;
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(threadidx_nk);
    Tensor sB_r2s = reshape_warpk<0>(sB);
    Tensor tCsB = smem_thr_copy_B.partition_S(sB_r2s); // (4,CPY_K,CPY_N,PIPE) CPY_N=1
    Tensor tCrB_copy_view = make_tensor_like(tCsB(_,_,_,0)); // (4,CPY_K,CPY_N)
    Tensor tCrB_i32 = composition(tCrB_copy_view, select<0, 2, 1>(tCrB_copy_view.layout())); // (4,CPY_N,CPY_K)

    SmemTiledCopyScale smem_tiled_copy_scale;
    auto smem_thr_copy_scale = smem_tiled_copy_scale.get_thread_slice(threadidx_nk / 4); // thread 0-3 share the same scale
    Tensor sScale_r2s = reshape_warpk<0>(sScale);
    Tensor tCsS = smem_thr_copy_scale.partition_S(sScale_r2s); // (8,CPY_K,CPY_N,PIPE) CPY_N=1
    Tensor tCrS_copy_view = make_tensor_like(tCsS(_,_,_,0)); // (8,CPY_K,CPY_N)
    Tensor tCrS = composition(tCrS_copy_view, select<0, 2, 1>(tCrS_copy_view.layout())); // (8,CPY_N,CPY_K)
    static_assert(size<2>(tCrS) == GroupsPerWarp);
    Tensor tCrS_dequant_view = make_tensor(tCrS.data(), Layout<Shape<_4, _2, _4, Int<GroupsPerWarp>>, Stride<_0, _1, _2, _8>>{});

    uint32_t m_block_idx, n_block_idx;
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      int M = deep_scheduler.curr_problem_m();
      if constexpr(!FusedLoadA) {
        gmem_tiled_copy_A.desc_.template init<ElementA, false, BlockM, BlockK>(nullptr, M, K, strideA);
      }
      int coord_l = deep_scheduler.problem_index();

      Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_a + deep_scheduler.curr_offset_a()), make_shape(M, Int<K>{}), strideA); // (m,k)
      Tensor mA_mk = make_mix_tensor_like(mA_mkl); // (m,k)
      Tensor gA = local_tile(mA_mk, TileShape{}, make_coord(m_block_idx, n_block_idx, _), Step<_1, X,_1>{}); // (BLK_M,BLK_K,blocks_k)

      Tensor mB_nkl = make_tensor(make_gmem_ptr(params.ptr_b), make_shape(Int<K / 16>{}, Int<N * 2>{}, Int<L>{}), strideB); // (k/16,n*2,l)
      Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,coord_l)); // (k/16,n*2)
      Tensor gB = local_tile(mB_nk, TileB{}, make_coord(_, n_block_idx)); // (BLK_K/16,BLK_N*2,blocks_k)

      Tensor mScale_nkl = make_tensor(make_gmem_ptr(params.ptr_scale), make_shape(Int<K / kGroupSize>{}, Int<N>{}, Int<L>{}), strideScale); // (k/group_size,n,l)
      Tensor mScale_nk = make_mix_tensor_like(mScale_nkl(_,_,coord_l)); // (k/group_size,n)
      Tensor gScale = local_tile(mScale_nk, TileScale{}, make_coord(_, n_block_idx)); // (BLK_K/group_size,BLOCK_N,blocks_k)

      Tensor mD_mn = make_tensor(make_gmem_ptr(params.ptr_d + deep_scheduler.curr_offset_c()), make_shape(M, Int<N>{}), strideD); // (m,n)
      Tensor gD = local_tile(mD_mn, TileShape{}, make_coord(m_block_idx, n_block_idx, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N)
      Tensor cD   = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
      auto residual_mn = make_coord(M - BlockM * m_block_idx,  N - BlockN * n_block_idx);

      Tensor tAgA = gmem_thr_copy_A.partition_S(gA);
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);
      Tensor tSgS = gmem_thr_copy_scale.partition_S(gScale);

      auto g2s_copy_A = [&](int k_tile_iter, int k_pipe) {
        if constexpr(!FusedLoadA) { // aiu copy
          if (warp_idx == 0) {
            copy(gmem_tiled_copy_A, tAgA(_,_,_,k_tile_iter), tAsA(_,_,_,k_pipe));
          }
          return;
        }
        // FusedLoadA: async copy
        // TODO: solve problems when tiler_m > block_m
        constexpr int tiler_m = get<0>(TilerA{});
        constexpr int tiler_k = get<1>(TilerA{});
        constexpr int NumThreads_CPY_K = tiler_k / AlignA;
        constexpr int M_ITER = BlockM / tiler_m;
        constexpr int K_ITER = BlockK / tiler_k;
        CUTLASS_PRAGMA_UNROLL
        for(int m_iter = 0; m_iter < M_ITER; ++m_iter) {
          CUTLASS_PRAGMA_UNROLL
          for(int k_iter = 0; k_iter < K_ITER; ++k_iter) {
            int tid_k = thread_idx % NumThreads_CPY_K;
            int tid_m = thread_idx / NumThreads_CPY_K;
            int m_offset = __ldg(params.sorted_token_ids + (m_block_idx + deep_scheduler.curr_cumsum) * BlockM + (m_iter * tiler_m) + tid_m);
            int k_offset = k_iter * tiler_k + tid_k * AlignA + k_tile_iter * BlockK;
            int token_offset = m_offset * K + k_offset;
            const cutlass::uint128_t& src = *reinterpret_cast<const cutlass::uint128_t*>(params.ptr_a + token_offset);
            cutlass::uint128_t& dst = *reinterpret_cast<cutlass::uint128_t*>(raw_pointer_cast(tAsA(_,m_iter,k_iter,k_pipe).data()));
            ACopyInst::copy(src, dst, m_offset < params.scheduler.shape_m);
          }
        }
      };

      auto g2s_copy_B_and_scale = [&](int k_tile_iter, int k_pipe) {
        constexpr int copy_warp_idx = SplitAIU ? 1 : 0;
        if (warp_idx == copy_warp_idx) {
          copy(gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,k_pipe));
          copy(gmem_tiled_copy_scale, tSgS(_,_,_,k_tile_iter), tSsS(_,_,_,k_pipe));
        }
      };

      auto dequant = [&](int n_idx, int k_block) {
        constexpr int InterleavedElems = 8;
        using Converter = MixGemmNumericArrayConverter<ElementA, ElementB, InterleavedElems>;
        using SrcArray = cutlass::Array<ElementB, InterleavedElems>;
        using DstArray = cutlass::Array<ElementA, InterleavedElems>;

        const SrcArray& src = *reinterpret_cast<const SrcArray*>(tCrB_i32(n_idx,_,k_block).data());
        DstArray& dst = *reinterpret_cast<DstArray*>(tCrB(_,n_idx,k_block).data());
        dst = Converter::convert(src);
        cute::transform(
          tCrB_dequant_view(_,_,n_idx,k_block),
          tCrS_dequant_view(_,_,n_idx,k_block/MmaPerGroup),
          tCrB_dequant_view(_,_,n_idx,k_block),
          cute::multiplies{}
        );
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
        if (warpk_idx != 0) return;

        Tensor tDgD = thr_mma.partition_C(gD);
        Tensor tDcD = thr_mma.partition_C(cD);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tDgD); i += 2) {
          if (elem_less(tDcD(i), residual_mn)) {
            ElementAcc* acc_ptr = raw_pointer_cast(accum.data()) + accum.layout()(i);
            uint32_t* dst_ptr = (uint32_t*)((raw_pointer_cast(tDgD.data()) + tDgD.layout()(i)));
            uint32_t d;
            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(acc_ptr[1]), "f"(acc_ptr[0]));
            *dst_ptr = d;
          }
        }
      };

      auto epilogue_with_tsm = [&]() {
        if (WarpsOnK == 1 || warpk_idx == 0) {
          Tensor gDt = flat_divide(gD, sC_tile);
          Tensor cDt  = flat_divide(cD, sC_tile);

          auto smem_tiled_copy_C = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
          auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(thread_idx);
          Tensor tCrC = smem_thr_copy_C.retile_S(accum);
          Tensor tCsC = smem_thr_copy_C.partition_D(sC);

          auto smem_tiled_copy_s2r = TiledCopyS2R{};
          auto smem_thr_copy_s2r = smem_tiled_copy_s2r.get_thread_slice(thread_idx);
          Tensor tDsD = smem_thr_copy_s2r.partition_S(sC);

          Tensor tDgD = smem_thr_copy_s2r.partition_D(gDt);
          Tensor tDrD = make_tensor<ElementD>(take<0,3>(shape(tDgD)));
          Tensor tDcD = smem_thr_copy_s2r.partition_D(cDt);

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
              dst = converter(src);
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

            __syncthreads();

            copy(smem_tiled_copy_s2r, tDsD, tDrD);
            Tensor tDgDmn = tDgD(_,_,_,step_m,0);
            Tensor tDcDmn = tDcD(_,_,_,step_m,0);
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<1>(tDgDmn); ++m) {
              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < size<2>(tDgDmn); ++n) {
                if (elem_less(tDcDmn(0,m,n), residual_mn)) {
                  copy(CopyAtomR2G{}, tDrD(_,m,n), tDgDmn(_,m,n));
                }
              }
            }
          }
        } else {
          constexpr int msteps = BlockM / size<0>(sC);
          for (int step_m = 0; step_m < msteps; ++step_m) {
            __syncthreads();
          }
        }
      };

      // mainloop
      clear(accum);
      int k_tile_iter = 0;
      int k_tile_count = size<2>(gA);
      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < kNumStages-1; ++k_pipe) {
        if (k_tile_count > 0) {
          g2s_copy_A(k_tile_iter, k_pipe);
          g2s_copy_B_and_scale(k_tile_iter, k_pipe);
          ++k_tile_iter;
        }
        cp_async_fence();
        --k_tile_count;
      }

      int smem_pipe_read  = 0;
      int smem_pipe_write = kNumStages-1;
      Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
      Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);
      Tensor tCsS_p = tCsS(_,_,_,smem_pipe_read);
      constexpr int K_BLOCK_MAX = size<2>(tCrA_copy_view);
      constexpr int K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);
      static_assert(K_ATOM_PER_COPY == 1, "K_ATOM_PER_COPY should be 1.");
      if constexpr(K_BLOCK_MAX > 1) {
        cp_async_wait<kNumStages-2>();
        __syncthreads();

        copy(smem_tiled_copy_A, tCsA_p(_,_,0), tCrA_copy_view(_,_,0));
        copy(smem_tiled_copy_B, tCsB_p(_,0,_), tCrB_copy_view(_,0,_));
        if constexpr(GroupsPerWarp > 1) {
          copy(smem_tiled_copy_scale, tCsS_p(_,0,_), tCrS_copy_view(_,0,_));
        }
      }

      auto process_kblock_iterations = [&](int k_block) {
        if (k_block == K_BLOCK_MAX - 1) {
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);
          tCsS_p = tCsS(_,_,_,smem_pipe_read);
          cp_async_wait<kNumStages-2>();
          __syncthreads();
        }

        auto k_block_next = (k_block + 1) % K_BLOCK_MAX;
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,k_block_next,_), tCrB_copy_view(_,k_block_next,_));
        if ((GroupsPerWarp > 1 && k_block_next % MmaPerGroup == 0) || (GroupsPerWarp == 1 && k_block == 0)) {
          copy(smem_tiled_copy_scale, tCsS_p(_,k_block_next/MmaPerGroup,_), tCrS_copy_view(_,k_block_next/MmaPerGroup,_));
        }

        if (k_block == 0) {
          if (k_tile_count > 0) {
            g2s_copy_A(k_tile_iter, smem_pipe_write);
            g2s_copy_B_and_scale(k_tile_iter, smem_pipe_write);
            ++k_tile_iter;
          }
          cp_async_fence();
          --k_tile_count;

          smem_pipe_write = smem_pipe_read;
          smem_pipe_read = (smem_pipe_read + 1) % kNumStages;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int n_idx = 0; n_idx < 4; n_idx++) {
          dequant(n_idx, k_block);
          CUTLASS_PRAGMA_UNROLL
          for (int m_idx = 0; m_idx < size<1>(accum); m_idx++) {
            cute::gemm(tiled_mma, tCrA(_,m_idx,k_block), tCrB(_,n_idx,k_block), accum(_,m_idx,n_idx));
          }
        }
      };

      for_each(make_int_sequence<K_BLOCK_MAX >{}, [&] (auto k_block) {
        process_kblock_iterations(k_block);
      });

      CUTLASS_PRAGMA_NO_UNROLL
      while (k_tile_count > -(kNumStages-1)) {
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
          process_kblock_iterations(k_block);
        });
      }

      cp_async_wait<0>();
      __syncthreads();

      epilogue_cta_reduce();
      #if __HGGC_ARCH__ == 150
        epilogue_no_tsm();
      #else
        epilogue_with_tsm();
      #endif
      __syncthreads();
    }
  }
};

} // namespace cutlass::gemm::kernel

namespace deep_gemm {
using cutlass::KernelHardwareInfo;

template <int ShapeN, int ShapeK,
          int BlockM, int BlockN, int BlockK,
          int WarpM, int WarpN, int WarpK,
          int kNumGroups, int kNumStages, GemmType kGemmType,
          int kGroupSize = 32>
class W4A16Gemm {
  static_assert((BlockM == 16) || (BlockM == 32) || (BlockM == 64) || (BlockM == 128) || (BlockM == 256), "BlockM should only be in [16, 32, 64, 128, 256].");
  static_assert((BlockN == 16) || (BlockN == 32) || (BlockN == 64) || (BlockN == 128) || (BlockN == 256), "BlockM should only be in [16, 32, 64, 128, 256].");
  static_assert((WarpM % 16 == 0), "WarpM must be divideable by 16.");
  static_assert((WarpN == 64), "WarpN must be 64.");
  static_assert((BlockK >= kGroupSize && BlockK % kGroupSize == 0), "BlockK must be multiple of group_size.");
  static_assert((kGroupSize >= 16 && kGroupSize % 16 == 0), "Group_size must be multiple of 16 (mma_k).");

public:
    W4A16Gemm() = default;

    static void run(const cutlass::bfloat16_t *a_ptr, const int *b_ptr, const cutlass::bfloat16_t *scale_b,
                    cutlass::bfloat16_t *d_ptr, int shape_m, int *grouped_layout, int *block_m_info, int *sorted_token_ids,
                    uint32_t expected_m, cudaStream_t stream, int num_sms) {
      using Kernel = cutlass::gemm::kernel::W4A16GEMM<ShapeN, ShapeK, BlockM, BlockN, BlockK, WarpM, WarpN, WarpK, kNumGroups, kNumStages, kGemmType, kGroupSize>;

      int num_threads = Kernel::MaxThreadsPerBlock;
      int tb_per_sm = min(8, compute_occupancy_for_kernel<Kernel>());
      int threadblock_count = num_sms < 20 ? num_sms : num_sms * tb_per_sm;

      dim3 block(num_threads, 1, 1);
      dim3 grid(threadblock_count, 1, 1);
      int smem_size = sizeof(typename Kernel::SharedStorage);

      int* layout_info = grouped_layout;
      if constexpr(Kernel::TileScheduler::kIsNoPadPreprocessLayout) {
        uint32_t block_size = max(32, next_power_of_two(kNumGroups));
        computeBlockInfoKernel<BlockM><<<1, block_size, 0, stream>>>(reinterpret_cast<const uint32_t*>(grouped_layout), kNumGroups, reinterpret_cast<uint32_t*>(block_m_info));
        layout_info = block_m_info;
      }

      typename Kernel::Params params{
        a_ptr, b_ptr, scale_b, d_ptr, sorted_token_ids,
        {shape_m, layout_info}
      };

      DgProfParam dg_prof_params;
      if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(
            kGemmType, false, std::string("w4a16"), kNumGroups, shape_m, ShapeN, ShapeK, expected_m,
            grouped_layout, stream
        );
        dg_prof_params.add_params(std::string("quant_type"), std::string("group"));
        dg_prof_params.add_params(std::string("group_size"), kGroupSize);
      }
      ProfilingInterface::Instance().instrument(true, dg_prof_params);
      cutlass::device_kernel<Kernel><<<grid, block, smem_size, stream>>>(params);
      ProfilingInterface::Instance().instrument(false, dg_prof_params);

      char *pEnv_params = std::getenv("show_log");
      if (pEnv_params && isdigit(*pEnv_params)) {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, cutlass::device_kernel<Kernel>);

        printf("[GemmGrouped-W4A16:]\n");
        printf("group:%d, problem:[%d, %d, %d], expected_m:%d, gemm_type:%s, kIsNoPadPreprocessLayout: %d\n",
            kNumGroups, shape_m, ShapeN, ShapeK, expected_m, GemmTypeS[static_cast<int>(kGemmType)], Kernel::TileScheduler::kIsNoPadPreprocessLayout);

        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
            BlockM, BlockN, BlockK, WarpM, WarpN, WarpK, kNumStages);

        printf("num_sms:%d, tb_per_sm:%d, threadblock_count:%d, num_threads: %d\n", num_sms, tb_per_sm, threadblock_count, num_threads);

        printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size, int(attr.numRegs), int(attr.localSizeBytes));
      }

    }
  };

};  // namespace deep_gemm