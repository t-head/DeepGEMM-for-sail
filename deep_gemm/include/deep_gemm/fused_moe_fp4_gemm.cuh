#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include "utils.cuh"
#include "profiling_interface.hpp"

#include "cute/ppu_tensor_mix.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"

#include "cute/atom/mma_traits_ppu0015.hpp"
#include "cute/atom/copy_traits_ppu0015_aiu.hpp"
#include "cute/algorithm/ppu_copy.hpp"
#include "fp4_gemm_cutlass3.cuh"

#include "fused_scheduler.cuh"
#include "fused_gemm_util.cuh"
#include "utils_cutlass3.h"

using namespace cute;

namespace deep_gemm {

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int kNumStages>
class MockCollectiveMmaScaleFp4 {
public:
  using DispatchPolicy = cutlass::gemm::MainloopWithScalePPU0015Aiu<kNumStages>;
  using ElementA = cutlass::float4_t;
  using ElementB = cutlass::float4_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementSFA = uint16_t;
  using ElementSFB = uint16_t;
  using LayoutSFA = cutlass::layout::ColumnMajor;
  using LayoutSFB = cutlass::layout::RowMajor;
  using ArchTag = cutlass::arch::PPU0015;

  using WarpOnM = Int<BLOCK_M / WARP_M>;
  using WarpOnN = Int<BLOCK_N / WARP_N>;

  using MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;
  using TiledMma = cute::TiledMMA<
    cute::MMA_Atom<MmaInst>,
    cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>;

  using GemmOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, false, Int<BLOCK_M>, Int<BLOCK_K>, false>;
  using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, false, Int<BLOCK_N>, Int<BLOCK_K>, true>;

  using TransformA = cute::identity;
  using TransformB = cute::identity;

  // Use Aiu for SFA
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
  static constexpr bool TransSFB = true;
  static constexpr int MinAiuContElemSizeSFB = 32 / (sizeof_bits<ElementSFB>::value / 8);      // 16

  static constexpr int ScaleNsPerTile = BLOCK_N;
  static constexpr int ScaleKsPerTileSFB = ScaleKsPerTile;

  static constexpr int SFBTileN = TransSFB ? cute::max(ScaleNsPerTile, MinAiuContElemSizeSFB) : ScaleNsPerTile;
  static constexpr int SFBTileK = TransSFB ? ScaleKsPerTileSFB : cute::max(ScaleKsPerTileSFB, MinAiuContElemSizeSFB);

  using GemmOperandSFB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementSFB, TransSFB, Int<SFBTileN>, Int<SFBTileK>, swap, StageStride, swzl>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMmaScaleFp4<
    DispatchPolicy, Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
    typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB,
    ElementSFA, cutlass::detail::TagToStrideA_t<LayoutSFA>, typename GemmOperandSFA::GmemTiledCopy, typename GemmOperandSFA::SmemLayoutAtom,
    ElementSFB, cutlass::detail::TagToStrideB_t<LayoutSFB>, typename GemmOperandSFB::GmemTiledCopy, typename GemmOperandSFB::SmemLayoutAtom>;
};

template <GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N,
          uint32_t BLOCK_SIZE, int kNumStages, typename MockMainloopFp4>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void
fp4_gemm_fused_moe_kernel(const QuantGemmArgs args) {
    static constexpr uint32_t GROUP_K = 32; // view as uint16
    static constexpr uint32_t ScaleMsPerTile = BLOCK_M;
    static constexpr uint32_t ScaleNsPerTile = BLOCK_N;
    static constexpr uint32_t ScaleKsPerTile = cute::ceil_div(Int<BLOCK_K>{}, Int<GROUP_K>{});
    static constexpr uint32_t ShapeScaleKs = cute::ceil_div(Int<SHAPE_K>{}, Int<GROUP_K>{});
    static constexpr bool kAligned = (ShapeScaleKs % ScaleKsPerTile) == 0;

    constexpr uint32_t STRIDE_AM = SHAPE_K;
    constexpr uint32_t STRIDE_BE = SHAPE_N * SHAPE_K;
    constexpr uint32_t STRIDE_CM = SHAPE_N;
    constexpr uint32_t STRIDE_BSE = SHAPE_N * ShapeScaleKs;

    using SrcT = typename MockMainloopFp4::ElementA;
    using SrcSFT = typename MockMainloopFp4::ElementSFA;
    using DstT = cutlass::bfloat16_t;
    using AccT = float;
    using TileShape = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>, cute::Int<BLOCK_K>>;
    using WarpShape = cute::Shape<cute::Int<WARP_M>, cute::Int<WARP_N>, cute::Int<BLOCK_K>>;
    using ArchTag = cutlass::arch::PPU0015;

    using TiledMma = typename MockMainloopFp4::TiledMma;

    using TileScheduler = FusedGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, kNumGroups>;

    // Shared memory
    using TsmCfg = GemmSmemConfigFp4<kNumStages, BLOCK_M, BLOCK_N, BLOCK_K, MockMainloopFp4>;
    extern __shared__ __align__(128) uint8_t smem_buffer[];
    SrcT* smem_a = reinterpret_cast<SrcT*>(smem_buffer);
    SrcT* smem_b = reinterpret_cast<SrcT*>(smem_buffer + TsmCfg::kSmemASize);
    SrcSFT* smem_sfa = reinterpret_cast<SrcSFT*>(smem_buffer + TsmCfg::kSmemASize + TsmCfg::kSmemBSize);
    SrcSFT* smem_sfb = reinterpret_cast<SrcSFT*>(smem_buffer + TsmCfg::kSmemASize + TsmCfg::kSmemBSize + TsmCfg::kSmemSFASize);

    uint32_t thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    constexpr int warp_on_m = BLOCK_M / WARP_M;
    int warp_m_idx = warp_idx % warp_on_m;
    int warp_n_idx = warp_idx / warp_on_m;
    constexpr int WARP_NUM = warp_on_m * (BLOCK_N / WARP_N);
    static_assert(((WARP_NUM & 1) == 0) && WARP_NUM > 1, "the number of warp for fp4 fused_moe must be even and greater than 1.");

    uint32_t shape_m = args.shape_m;

    // 1. load A from hbm to tsm: use async copy.
    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<SrcT>::value;
    using ACopyInst = cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t>;
    using GemmOperandA = cutlass::gemm::config::Gemm_Hybrid_Operand<
            ArchTag, SrcT, false, AlignmentA, cute::Int<BLOCK_K>, BLOCK_SIZE,
            ACopyInst, cute::Int<BLOCK_M>>;

    using TilerA = typename GemmOperandA::GmemTiledCopy::Tiler_MN;
    using SmemLayoutA = decltype(tile_to_shape(typename GemmOperandA::SmemLayoutAtom{},
            Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_K>, cute::Int<kNumStages>>{}));

    typename GemmOperandA::GmemTiledCopy gmem_tiled_copy_A;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    Tensor sA = cute::make_tensor(cute::make_smem_ptr(smem_a), SmemLayoutA{});
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);

    // 2. load B from hbm to tsm: use aiu load.
    using SmemLayoutB = typename MockMainloopFp4::SmemLayoutB;

    // 2.1 init aiu desc for B
    typename MockMainloopFp4::GmemTiledCopyB gmem_tiled_copy_B;
    using TilerB = typename MockMainloopFp4::GmemTiledCopyB::Tiler_MN;
    auto shape_B = cute::make_shape(Int<SHAPE_N>{}, Int<SHAPE_K>{});
    auto stride_B = cute::make_shape((int)SHAPE_K, _1{});
    gmem_tiled_copy_B.desc_.template init<SrcT, false, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, SHAPE_N, SHAPE_K, stride_B);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor sB = cute::make_tensor(cute::make_smem_ptr(smem_b), SmemLayoutB{});
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);

    // --------------------------------------- scale operand -------------------------------------//
    // 3. Use cp.async for SFA: SFA is K-Major for fused_moe
    // thread -> (m, k) mapping of the sfa gmem->tsm copy:
    //  *(lane-on-k): `ScaleKsPerTile` lanes share one row, their b16 requests are contiguous and
    //    collapsed into one gmem request, at the cost of a `ScaleKsPerTile`-way tsm write bank conflict
    constexpr uint32_t NumThreads_Needed = ScaleMsPerTile * ScaleKsPerTile;
    constexpr uint32_t ACP_SFA_ITER = ceil_div(NumThreads_Needed, BLOCK_SIZE);
    static_assert(32 % ScaleKsPerTile == 0,
                  "ScaleKsPerTile must divide the warp size for the lane-on-k sfa g2s mapping.");
    // one warp group (2 warps, 64 threads) covers `64 / ScaleKsPerTile` rows of m,
    // the 2 warps in a group interleave on m (even/odd) to avoid tsm bank conflict of contiguous 2B in the same bank.
    uint32_t parity = warp_idx & 1;
    uint32_t warp_group = warp_idx / 2;
    uint32_t tid_k = lane_idx % ScaleKsPerTile;
    constexpr uint32_t tid_m_per_warp_group = 64 / ScaleKsPerTile;
    uint32_t tid_m_base = warp_group * tid_m_per_warp_group + (lane_idx / ScaleKsPerTile) * 2 + parity;
    constexpr uint32_t tid_m_block_offset = BLOCK_SIZE / ScaleKsPerTile;

    using SmemLayoutSFA = typename MockMainloopFp4::SmemLayoutSFA;
    auto copy_SFA_to_tsm = [&](int pipe_write, uint32_t k_idx, const int* blk_token_base, const uint32_t* token_offsets = nullptr) {
      bool k_valid = true;
      if constexpr (!kAligned) {
        k_valid = ((k_idx * ScaleKsPerTile + tid_k) < ShapeScaleKs);
      }
      CUTLASS_PRAGMA_UNROLL
      for(uint32_t idx = 0; idx < ACP_SFA_ITER; idx++) {
        uint32_t tid_m = tid_m_base + idx * tid_m_block_offset;
        uint32_t token_offset = token_offsets[idx];
        bool token_mask = token_offset < shape_m && k_valid;
        SrcSFT* src_ptr = (SrcSFT*)args.scale_a_ptr + (tid_k + k_idx * ScaleKsPerTile) + token_offset * ShapeScaleKs;
        SrcSFT* dst_ptr = (SrcSFT*)smem_sfa + SmemLayoutSFA{}(make_coord(make_coord(tid_m % 64, tid_m / 64), tid_k, pipe_write));
        if (token_mask) {
          // acp b16 does not have zfill field.
          asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], 2;" : : "r"(dst_ptr), "l"(src_ptr));
        }
      }
    };
    Tensor sSFA = make_tensor(make_smem_ptr(smem_sfa), SmemLayoutSFA{});
    // epilogue will not store m rows out of m_predicate.
    if constexpr (!kAligned) {
      // clear(sSFA) in case of 255 in the uninitialized smem; BlockM is divideable by 16;
      constexpr uint32_t SFA_SIZE_IN_BYTE = TsmCfg::kSmemSFASize;
      constexpr uint32_t SFA_SIZE_IN_INST = SFA_SIZE_IN_BYTE / sizeof(uint128_t);
      uint128_t* smem_sfa_clear = reinterpret_cast<uint128_t*>(smem_sfa);
      for (uint32_t i = threadIdx.x; i < SFA_SIZE_IN_INST; i += blockDim.x) {
          smem_sfa_clear[i] = 0;
      }
      __syncthreads();
    }


    // 4. Use Aiu for SFB
    using SmemLayoutSFB = typename MockMainloopFp4::SmemLayoutSFB;
    auto mSFB_layout = make_layout(make_shape(SHAPE_N, cute::ceil_div(SHAPE_K, 16 * 2)), LayoutLeft{}); // uint16_t
    // init SFB aiu desc
    typename MockMainloopFp4::GmemTiledCopySFB gmem_tiled_copy_SFB;
    using TilerSFB = typename MockMainloopFp4::GmemTiledCopySFB::Tiler_MN;

    // 4.1 init aiu desc
    auto SFK = cute::ceil_div(SHAPE_K, Int<32>{});
    auto stride_sfb = cute::make_stride(_1{}, SHAPE_N);
    gmem_tiled_copy_SFB.desc_.template init<SrcSFT, true, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(nullptr, SHAPE_N, SFK, stride_sfb);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor sSFB = make_tensor(make_smem_ptr(smem_sfb), SmemLayoutSFB{});
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{})); // (MMA,MMA_M,MMA_N)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //
    using SmemCopyAtomA = typename GemmOperandA::SmemCopyAtom;
    using SmemCopyAtomB = typename MockMainloopFp4::SmemCopyAtomB;

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // ========= fp4 scale smem tiledCopy =========
    // create rSFA refer to tCsA/tCrA
    constexpr auto warp_iter_num_sfa = size<1>(tCsA);
    constexpr auto block_iter_num_sfa = size<2>(tCsA);
    constexpr auto warp_iter_stride_sfa = (BLOCK_M / WARP_M) * 16;
    constexpr int ldmat_iter_num_sfa = cute::ceil_div(warp_iter_num_sfa, Int<4>{});
    Tensor tCrSFA = make_tensor<uint32_t>(Shape<Int<1>, Int<ldmat_iter_num_sfa>, Int<block_iter_num_sfa>>{});
    static_assert(warp_iter_num_sfa <= 4, "warp_iter_num_sfa > 4 is not valid now.");
    // create rSFB refer to tCsB/tCrB
    constexpr auto warp_iter_num_sfb = size<1>(tCsB);
    constexpr auto block_iter_num_sfb = size<2>(tCsB);
    constexpr auto warp_iter_stride_sfb = [&]() constexpr {
      if constexpr (cute::is_scaled_basis<decltype(stride<1>(tCsB))>::value) {
        return stride<1>(tCsB).value();
      } else {
        return stride<1>(tCsB);
      }
    }();
    constexpr int ldmat_iter_num_sfb = cute::ceil_div(warp_iter_num_sfb, Int<4>{});
    Tensor tCrSFB = make_tensor<uint32_t>(Shape<Int<1>, Int<ldmat_iter_num_sfb>, Int<block_iter_num_sfb>>{});
    static_assert(warp_iter_num_sfb <= 4, "warp_iter_num_sfb > 4 is not valid now.");

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    TileScheduler deep_scheduler(args.aligned_num_m_blocks, args.expert_ids_and_cumsum);

    uint32_t token_offsets[ACP_SFA_ITER];
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto blk_coord_mnkl = make_coord(m_block_idx, n_block_idx, _, _1{});
      uint32_t blk_n_offset = n_block_idx * BLOCK_N;
      const int* blk_token_base = args.sorted_token_ids + deep_scheduler.cumsum_m_block_idx * BLOCK_M;

      // prefetch token_offset into vreg to solve memory deps
      CUTLASS_PRAGMA_UNROLL
      for(uint32_t idx = 0; idx < ACP_SFA_ITER; idx++) {
        uint32_t tid_m = tid_m_base + idx * tid_m_block_offset;
        // M block predicate: `NumThreads_Needed` may not be a multiple of BLOCK_SIZE, guard the tail iteration
        token_offsets[idx] = tid_m < ScaleMsPerTile ? __ldg(blk_token_base + tid_m) : shape_m;
      }

      // gmem_b in block
      SrcT* gmem_b = (SrcT*)args.b_ptr + deep_scheduler.curr_group_idx * STRIDE_BE;
      Tensor mB_nk = cute::make_tensor(cute::make_gmem_ptr(gmem_b), shape_B, stride_B);
      Tensor gB = cute::local_tile(cute::make_mix_tensor_like(mB_nk), TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)
      Tensor tBgB = gmem_thr_copy_B.partition_S(gB);

      // gmem_scale b in block
      SrcSFT* gmem_scale_b = (SrcSFT*)args.scale_b_ptr + deep_scheduler.curr_group_idx * STRIDE_BSE;
      Tensor mSFB_nk = make_tensor(make_gmem_ptr(gmem_scale_b), mSFB_layout);
      Tensor gSFB = local_tile(make_mix_tensor_like(mSFB_nk), make_shape(Int<BLOCK_N>{}, Int<ScaleKsPerTile>{}), make_coord(n_block_idx, _)); // (BLK_N,BLK_K,k)
      Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);

      int k_tile_iter  = 0;
      int k_tile_count = size<2>(gB);

      CUTLASS_PRAGMA_UNROLL
      for (int k_pipe = 0; k_pipe < kNumStages; ++k_pipe) {
        if (k_tile_count > 0) {
          copy_A_to_tsm<SrcT, ACopyInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM>(
              tAsA(_,_,_,k_pipe), args.a_ptr,
              blk_token_base, BLOCK_K * k_tile_iter, shape_m, thread_idx);
          copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,k_pipe), warp_idx);
          copy_SFA_to_tsm(k_pipe, k_tile_iter, blk_token_base, token_offsets);
          if (warp_idx == 1) {
            copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_tile_iter), tSFBsSFB(_,_,_,k_pipe));
          }
          ++k_tile_iter;
        }
        cp_async_fence();
        --k_tile_count;
      }

      clear(accum);

      //
      // PIPELINED MAIN LOOP
      //

      // Current pipe index in smem to read from
      int smem_pipe_read  = 0;
      // Current pipe index in smem to write to
      int smem_pipe_write = 0;

      Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
      Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

      // Size of the register pipeline
      auto K_BLOCK_MAX = size<2>(tCrA);
      static_assert(K_BLOCK_MAX > 1, "K_BLOCK_MAX should be large than 1.");

      int base_offset_sfa = 0;
      int base_offset_sfb = 0;
      MockMainloopFp4::cal_ldmat_offset(base_offset_sfa, SmemLayoutSFA{}, lane_idx, warp_m_idx, Int<warp_iter_num_sfa>{}, Int<warp_iter_stride_sfa>{});
      MockMainloopFp4::cal_ldmat_offset(base_offset_sfb, SmemLayoutSFB{}, lane_idx, warp_n_idx, Int<warp_iter_num_sfb>{}, Int<warp_iter_stride_sfb>{});

      // PREFETCH register pipeline
      if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<kNumStages-1>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
        MockMainloopFp4::sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, 0);
        MockMainloopFp4::sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, 0);
      }

      auto process_kblock_iterations = [&](auto k_block) {
        // Load A, B shmem->regs for k_block+1
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == K_BLOCK_MAX - 1) {
          // Commit the smem for smem_pipe_read
          if (k_tile_count > 0) {
            copy_A_to_tsm<SrcT, ACopyInst, TilerA, BLOCK_M, BLOCK_K, STRIDE_AM>(
              tAsA(_,_,_,smem_pipe_write), args.a_ptr,
              blk_token_base, BLOCK_K * k_tile_iter, shape_m, thread_idx);
            copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,k_tile_iter), tBsB(_,_,_,smem_pipe_write), warp_idx);
            copy_SFA_to_tsm(smem_pipe_write, k_tile_iter, blk_token_base, token_offsets);
            if (warp_idx == 1) {
              copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_tile_iter), tSFBsSFB(_,_,_,smem_pipe_write));
            }
            ++k_tile_iter;
          }
          cp_async_fence();
          // Advance the tile
          --k_tile_count;

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == kNumStages) ? 0 : smem_pipe_read;
          smem_pipe_write = smem_pipe_read;

          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          cp_async_wait<kNumStages-1>();
          __syncthreads();
        }
        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        MockMainloopFp4::sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<k_block_next>{}, smem_pipe_read);
        MockMainloopFp4::sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<k_block_next>{}, smem_pipe_read);

        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrSFA(_,_,k_block), tCrB(_,_,k_block), tCrSFB(_,_,k_block), accum);
      };

      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      //
      // Split out the first loop iteration to facilitate the use of mm instructions
      for_each(make_int_sequence<K_BLOCK_MAX >{}, [&] (auto k_block) {
        process_kblock_iterations(k_block);
      }); // for_each

      CUTLASS_PRAGMA_NO_UNROLL
      while (k_tile_count > -(kNumStages)) {
        // Pipeline the outer products with a static for loop.
        //
        // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
          process_kblock_iterations(k_block);
        }); // for_each
      }
      cp_async_wait<0>();
      __syncthreads();

      // acc write back
      auto blk_mn_shape = Shape<Int<BLOCK_M>, Int<BLOCK_N>>{};
      Tensor cC = make_identity_tensor(blk_mn_shape);
      Tensor tCcC = thr_mma.partition_C(cC);
      CUTE_STATIC_ASSERT_V(size(tCcC) == size(accum),
          "Accumulator count must have the same destination element count.");

      // acc write back
      epilogue_no_tsm<AccT, DstT, SHAPE_N, BLOCK_N, STRIDE_CM>(accum, tCcC, args.c_ptr,
          deep_scheduler.curr_block_m_offset, deep_scheduler.valid_m_in_block, blk_n_offset);
    }
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t WARP_M, uint32_t WARP_N, int32_t kNumStages,
          GemmType kGemmType, bool kEnableSboOverlap = false,
          KernelType kKernelType = KernelType::Default>
class Fp4FusedMoeGemm {
    static_assert((SHAPE_K % 16 == 0), "SHAPE_K must be divideable by 16.");
    static_assert((BLOCK_K % 32 == 0), "BlockK must be divideable by 32.");
    static_assert((WARP_M <= 64) && (WARP_M % 16 == 0), "WarpM must be divideable by 16 and less than 64.");
    static_assert((WARP_N <= 64) && (WARP_N % 16 == 0), "WarpN must be divideable by 16 and less than 64.");

    using SrcT = uint8_t;
    using DstT = __ppu_bfloat16;
    using SrcSFT = uint16_t;
    using MockMainloopFp4 = typename MockCollectiveMmaScaleFp4<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumStages>::CollectiveMainloop;

public:
    Fp4FusedMoeGemm() = default;

    static void run(DstT* gmem_d, SrcT* gmem_a, SrcT* gmem_b, SrcSFT* gmem_sfa, SrcSFT* gmem_sfb,
                    int* m_rows, int* expert_ids_and_cumsum, int* sorted_token_ids,
                    int* aligned_num_m_blocks, uint32_t shape_m, uint32_t topk,
                    hggcStream_t stream, int num_cus) {

        QuantGemmArgs args;

        args.a_ptr = (void *)gmem_a;
        args.b_ptr = (void *)gmem_b;
        args.c_ptr = (void *)gmem_d;
        args.scale_a_ptr = (void *)gmem_sfa;
        args.scale_b_ptr = (void *)gmem_sfb;

        args.expert_ids_and_cumsum = expert_ids_and_cumsum;
        args.sorted_token_ids = sorted_token_ids;
        args.aligned_num_m_blocks = aligned_num_m_blocks;
        args.shape_m = shape_m;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()){
            // check src type
            std::string data_type = "fp4";
            dg_prof_params.set_params(
                GemmType::GroupedFused, false, data_type, kNumGroups, shape_m, SHAPE_N, SHAPE_K, 1,
                m_rows, stream
            );
        }
        // dispatch and launch kernel
        constexpr int BlockSize = BLOCK_M / WARP_M * BLOCK_N / WARP_N * 32;

        auto device_func = fp4_gemm_fused_moe_kernel<kGemmType, SHAPE_N, SHAPE_K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BlockSize, kNumStages, MockMainloopFp4>;
        constexpr uint32_t smem_size = GemmSmemConfigFp4<kNumStages, BLOCK_M, BLOCK_N, BLOCK_K, MockMainloopFp4>::kTotalSize;
        CHECK_HGGC(hggcFuncSetAttribute(device_func, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int max_blocks_per_cu = -1;
        CHECK_HGGC(hggcOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu, device_func, BlockSize, smem_size));

        int cu_count = num_cus * max_blocks_per_cu;
        dim3 grid(cu_count, 1, 1);

        const char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && std::atoi(pEnv_params) == 1) {
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, device_func);

            printf("[FusedMoeGemm-FP4:]\n");
            printf("group:%d, problem:[%d, %d, %d], gemm_type:%s, kernel_type:%s\n",
                kNumGroups, shape_m, SHAPE_N, SHAPE_K, GemmTypeS[static_cast<int>(kGemmType)], KernelTypeS[static_cast<int>(kKernelType)]);

            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], kNumStages:%d\n",
                BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K, kNumStages);

            printf("grid:%d, vreg:%d, smem_size: %d, tb_per_cu:%d, stack:%d\n",
                cu_count, int(attr.numRegs), smem_size, max_blocks_per_cu, int(attr.localSizeBytes));
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        device_func<<<grid, BlockSize, smem_size, stream>>>(args);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};

} // namespace deep_gemm

#pragma clang diagnostic pop