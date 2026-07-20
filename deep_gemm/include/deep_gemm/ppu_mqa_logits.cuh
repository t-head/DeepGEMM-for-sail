#pragma once
#include "tools/util/include/cutlass/util/packed_stride.hpp"
#include "ppu_include.hpp"
#include "cute_tie.cuh"
#include "utils.cuh"
#include "utils_cutlass3.h"
#include "profiling_interface.hpp"

#define ENABLE_WARP_CONTIG_LAYOUT 1

__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__  __forceinline__ float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ppu.ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

namespace cutlass::gemm::kernel {

template <typename ElementQK, typename ElementAcc, typename ElementLogits,
          uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_QH, uint32_t BLOCK_KV,
          uint32_t WARP_QH, uint32_t WARP_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          typename StrideKType = uint32_t>
class PPUMqaLogits {
  static_assert(std::is_same_v<StrideKType, uint32_t> || std::is_same_v<StrideKType, uint64_t>,
                "StrideKType must be uint32_t or uint64_t");
public:
  using ElementC            = float;
  using LayoutA             = cutlass::layout::RowMajor;
  using LayoutB             = cutlass::layout::ColumnMajor;
  using LayoutC             = cutlass::layout::RowMajor;
  using ElementD            = ElementC;
  using LayoutD             = cutlass::layout::RowMajor;
  using ElementCompute      = float;
  using ElementScale        = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  static constexpr int BLOCK_M = BLOCK_KV;
  static constexpr int BLOCK_N = BLOCK_QH;
  static constexpr int BLOCK_K = kHeadDim;
  static constexpr int WARP_M = WARP_KV;
  static constexpr int WARP_N = WARP_QH;
  static constexpr int BLOCK_Q = BLOCK_QH / kNumHeads;
  static constexpr int WARP_Q = WARP_QH / kNumHeads;

  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
  using ProblemShape_MNKL = Shape<int,int,int,int>;

  using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
  using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
  static constexpr int WarpOnM = BLOCK_M / WARP_M;
  static constexpr int WarpOnN = BLOCK_N / WARP_N;
#if __HGGC_ARCH__ == 100
    using ArchTag = cutlass::arch::PPU0010;
#else
    using ArchTag = cutlass::arch::PPU0015;
#endif

  using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementQK,ElementQK,ElementAcc>::type;
  using MmaK_type = typename cutlass::platform::conditional<sizeof(ElementQK) == 2, _16, _32 >::type;

#if ENABLE_WARP_CONTIG_LAYOUT
  static constexpr int InstM = 16;
  static constexpr int InstN = 16;
  using WarpIterM = Int<BLOCK_M / WARP_M>;
  using WarpIterN = Int<BLOCK_N / WARP_N>;
  using MmaIterM = Int<WARP_M / InstM>;
  using MmaIterN = Int<WARP_N / InstN>;
  using PermutationMNK = Tile<
      Layout<Shape<Int<InstM>, WarpIterM, MmaIterM>, Stride<_1, Int<WARP_M>, Int<InstM>> >,
      Layout<Shape<Int<InstN>, WarpIterN, MmaIterN>, Stride<_1, Int<WARP_N>, Int<InstN>> >,
      MmaK_type
    >;
  using TiledMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<WarpIterM, WarpIterN, _1>>, PermutationMNK>;
#else
  using TiledMma = TiledMMA<
      MMA_Atom<MmaInst>,
      Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
      Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, MmaK_type
      >>;       // 1x1x1 value group
#endif

  static constexpr int NumThreadsPerCTA = size(TiledMma{});
  // WarpInterleaving is enabled only when NumThreadsPerCTA is 512, which satisfy the condition that 2 warp group partitioned onto separate WEs.
  static constexpr bool WarpInterleaving = (NumThreadsPerCTA == 512);

  static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementQK, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false>;
  using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementQK, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true>;
  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kNumKVStages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kNumQStages>{})));

  constexpr static uint32_t CTA_M = shape<0>(TileShape{});
  constexpr static uint32_t CTA_N = shape<1>(TileShape{});
  constexpr static uint32_t CTA_K = shape<2>(TileShape{});

  using ScaleCopyAtomWidth = cute::uint32_t;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  constexpr static uint32_t ScaleGranularity = sizeof(ScaleCopyAtomWidth) / sizeof(float);
  static constexpr int ScaleMsPerThread = cute::ceil_div(size<0>(TileShape{}), Int<MaxThreadsPerBlock * ScaleGranularity>{});
  static constexpr int ScaleNsPerThread = cute::ceil_div(size<1>(TileShape{}), Int<MaxThreadsPerBlock * ScaleGranularity>{});
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // ScaleA
  using GmemTiledCopyScaleA = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_M / ScaleMsPerThread>, _1>>{},
                    Layout<Shape <Int<ScaleMsPerThread>,_1>>{}));

  // ScaleB
  using GmemTiledCopyScaleB = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_N / ScaleNsPerThread>, _1>>{},
                    Layout<Shape <Int<ScaleNsPerThread>,_1>>{}));

  using SmemLayoutAtomScale = Layout<Shape<Int<ScaleGranularity>, _1>>;
  using SmemLayoutScaleA = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(Int<CTA_M>{}, Int<1>{}, Int<kNumKVStages>{})));   // assert CTA_SCALE_K = 1, gs >= cta_k
  using SmemLayoutScaleB = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(Int<CTA_N>{}, Int<1>{}, Int<kNumQStages>{})));   // assert CTA_SCALE_K = 1, gs >= cta_k
  static constexpr int WAPR_LIMIT_SFA = CTA_M / (32 * ScaleMsPerThread);
  static constexpr int WAPR_LIMIT_SFB = CTA_N / (32 * ScaleNsPerThread);
  static_assert(WAPR_LIMIT_SFA > 0 && WAPR_LIMIT_SFA <= (WarpOnM * WarpOnN / 2), "WAPR_LIMIT_SFA must > 0 and < half warps");
  static_assert(WAPR_LIMIT_SFB > 0 && WAPR_LIMIT_SFB <= (WarpOnM * WarpOnN / 2), "WAPR_LIMIT_SFB must > 0 and < half warps");

  // Kernel level shared memory storage
  struct SharedStorage {
    cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>> smem_k;
    cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>> smem_q;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleA>> smem_k_scales;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleB>> smem_weight;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    const ElementQK * ptr_q;
    const ElementQK * ptr_k;
    const float * k_scales;
    const float * weights;
    uint32_t* cu_seq_len_k_start;
    uint32_t* cu_seq_len_k_end;
    ElementLogits* logits;
    const uint32_t seq_len_q;
    const uint32_t seq_len_k;
    const StrideKType stride_k;
    StrideA dA;
    StrideB dB;
    KernelHardwareInfo hw_info{};
    // TileSchedulerArguments scheduler{};
    CUTLASS_DEVICE void
    print() const {
        printf("templates, kNumHeads=%d, kHeadDim=%d, tile=(%d, %d, %d, %d, %d, %d), BLOCK_Q=%d\n",
                kNumHeads, kHeadDim, BLOCK_KV, BLOCK_QH, WARP_KV, WARP_QH, kNumQStages, kNumKVStages, BLOCK_Q);
        printf("arguments, ptr_q=%p, ptr_k=%p, k_scales=%p, weights=%p, cu_seq_len_k_start=%p, cu_seq_len_k_end=%p, logits=%p\n",
                ptr_q, ptr_k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits);
        printf("arguments, seq_len_q=%u, seq_len_k=%u, stride_k=%llu\n",
                seq_len_q, seq_len_k, static_cast<unsigned long long>(stride_k));
    }
  };

  // Kernel entry point API
  using Params = Arguments;

  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;

  GmemTiledCopyScaleA gmem_tiled_copy_scaleA;
  GmemTiledCopyScaleB gmem_tiled_copy_scaleB;

  // // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    return dim3(params.hw_info.cu_count, 1, 1);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE void
  init_aiu_copy(ProblemShape_MNKL const& problem_shape_mnkl, Params params) {
    auto [M,N,K,L] = problem_shape_mnkl;
    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;

    gmem_tiled_copy_A.desc_.template init<ElementQK, TransA, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, CTA_M, CTA_K, params.dA);
    gmem_tiled_copy_B.desc_.template init<ElementQK, TransB, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, CTA_N, CTA_K, params.dB);

    gmem_tiled_copy_scaleA = make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_M / ScaleMsPerThread>, _1>>{},
                    Layout<Shape < Int<ScaleMsPerThread>,_1>>{});

    gmem_tiled_copy_scaleB = make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<ScaleCopyAtomWidth>, ElementScale>{},
                    Layout<Shape <Int<CTA_N / ScaleNsPerThread>, _1>>{},
                    Layout<Shape < Int<ScaleNsPerThread>,_1>>{});
  };

  template <class BlockCoord_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_mnkl, BlockCoord_MNKL const& blk_coord_mnkl, Params const& params) {
    auto [M,N,K,L] = problem_shape_mnkl;
    auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;
    // load init A
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_k), make_shape(M,K,L), params.dA);   // (m,k,l)
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,l_coord));                                 // (m,k)
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)

    // load init B
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.ptr_q), make_shape(N,K,L), params.dB);   //(n,k,l)
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));                                 // (n,k)
    Tensor gB = local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

    // // load init scale A/B
    Tensor mSFA_mk = make_tensor(make_gmem_ptr(params.k_scales), make_shape(M,1));      // (n,scale_k,l)
    auto sfa_shape = make_shape(Int<CTA_M>{}, Int<1>{});
    Tensor gSFA = local_tile(mSFA_mk, sfa_shape, make_coord(m_coord, _));    // (BLK_N, 1, scale_k)
    Tensor iSFA_mk = make_identity_tensor(sfa_shape);
    Tensor cSFA = local_tile(iSFA_mk, sfa_shape, make_coord(m_coord, _));

    Tensor mSFB_nk = make_tensor(make_gmem_ptr(params.weights), make_shape(N,1));           // (n,scale_k,l)
    auto sfb_shape = make_shape(Int<CTA_N>{}, Int<1>{});
    Tensor gSFB = local_tile(mSFB_nk, sfb_shape, make_coord(n_coord, _));    // (BLK_N, 1, scale_k)
    Tensor iSFB_nk = make_identity_tensor(sfb_shape);
    Tensor cSFB = local_tile(iSFB_nk, sfb_shape, make_coord(n_coord, _));

    return cute::make_tuple(gA, gB, cute::make_tuple(gSFA, cSFA), cute::make_tuple(gSFB, cSFB));
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    // printf("run ppu aiu deepgemm persistent!!!");
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    int warp_idx = canonical_warp_idx_sync();
    int thread_idx = int(threadIdx.x);

    // if (thread0()) {
    //   printf("EpilogueSharedStorage size = %d\n", sizeof(CollectiveEpilogue::SharedStorage));
    // }

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    const auto& num_q_blocks = ceil_div(params.seq_len_q, BLOCK_Q);
    auto M = params.seq_len_k;
    auto N = params.seq_len_q * kNumHeads;
    auto K = kHeadDim;
    auto L = 1;
    auto problem_shape_mnkl = ProblemShape_MNKL{M, N, K, L};
    auto blk_coord_mnkl = make_coord(0, 0, _, 0);

    // init aiu copy and async copy
    init_aiu_copy(problem_shape_mnkl, params);

    // init input tensors
    auto load_inputs = load_init(problem_shape_mnkl, blk_coord_mnkl, params);
    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);
    Tensor gSFA = get<2, 0>(load_inputs);
    Tensor gSFB = get<3, 0>(load_inputs);
    Tensor cSFA = get<2, 1>(load_inputs);
    Tensor cSFB = get<3, 1>(load_inputs);

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    Tensor sSFA = make_tensor(make_smem_ptr(shared_storage.smem_k_scales.data()), SmemLayoutScaleA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_weight.data()), SmemLayoutScaleB{});

    auto gmem_thr_copy_scaleA = gmem_tiled_copy_scaleA.get_slice(thread_idx);
    auto gmem_thr_copy_scaleB = gmem_tiled_copy_scaleB.get_slice(thread_idx);

    Tensor tSFAgSFA = gmem_thr_copy_scaleA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_scaleA.partition_D(sSFA);
    Tensor tSFAcSFA = gmem_thr_copy_scaleA.partition_S(cSFA);

    Tensor tSFBgSFB = gmem_thr_copy_scaleB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_scaleB.partition_D(sSFB);
    Tensor tSFBcSFB = gmem_thr_copy_scaleB.partition_S(cSFB);

    Tensor tSFApSFA = make_tensor<bool>(shape(tSFAsSFA));
    Tensor tSFBpSFB = make_tensor<bool>(shape(tSFBsSFB));

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                  // MMA_K

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    constexpr bool enable_print = false;

    if (enable_print) {
        params.print();
        print("gA: "); print(gA); print("\n");
        print("gB: "); print(gB); print("\n");
        print("sA: "); print(sA); print("\n");
        print("sB: "); print(sB); print("\n");

        print("tCsA: "); print(tCsA); print("\n");
        print("tCsB: "); print(tCsB); print("\n");

        print("gSFA: "); print(gSFA); print("\n");
        print("gSFB: "); print(gSFB); print("\n");

        print("sSFA: "); print(sSFA); print("\n");
        print("sSFB: "); print(sSFB); print("\n");

        print("tSFAsSFA: "); print(tSFAsSFA); print("\n");
        print("tSFBsSFB: "); print(tSFBsSFB); print("\n");
    }

    // Block scheduler
    uint32_t block_q_idx = blockIdx.x, q_iter_idx = 0;
    const auto& get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
      return {block_q_idx + gridDim.x, q_iter_idx + 1};
    };
    const auto& load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto& q_idx = min(block_q_idx * BLOCK_Q + i, params.seq_len_q - 1);
            start = min(start, min(__ldg(params.cu_seq_len_k_start + q_idx), params.seq_len_k));
            end = max(end, min(__ldg(params.cu_seq_len_k_end + q_idx), params.seq_len_k));
        }
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, end, ceil_div(end - start, BLOCK_KV)};          // Task info
    };

    ElementLogits weights[kNumHeads / 4];

    clear(accum);
    auto tKgK = tAgA;
    auto tSFKgSFK = tSFAgSFA;
    auto tQgQ = tBgB;
    auto tSFQgSFQ = tSFBgSFB;

    const auto& lane_idx = get_lane_idx();
    const auto& warp_offset = (warp_idx % WarpOnM) * WARP_M;
    const auto& v_0_offset = lane_idx / 4 + 0;
    const auto& v_1_offset = lane_idx / 4 + 8;
    uint32_t warp_q_idx = warp_idx / WarpOnM;
    int warp_group_id = warp_idx / 8;

    static constexpr uint32_t kMmaIterM = MmaIterM{};
    float scale_kv_array[kMmaIterM * 2];

    auto load_q_g2s = [&](uint32_t block_q_idx) {
        for (int i = 0; i < size(tSFBpSFB); ++i) {
            tSFBpSFB(i) = (get<0>(tSFBcSFB(i)) + block_q_idx * BLOCK_Q * kNumHeads) < N;
        }
        gmem_tiled_copy_B.desc_.dim_h = (params.seq_len_q - block_q_idx * BLOCK_Q) * kNumHeads;
        copy_aiu(gmem_tiled_copy_B, tBgB(_,_,_,0), tBsB(_,_,_,0), warp_idx);
        if (warp_idx < WAPR_LIMIT_SFB) {
            copy_if(gmem_tiled_copy_scaleB, tSFBpSFB(_,_,_,0), tSFBgSFB(_,_,_,0), tSFBsSFB(_,_,_,0));
        }
    };

    auto load_kv_g2s = [&](uint32_t kv_start, uint32_t kv_end, uint32_t kv_block_idx, uint32_t smem_stage) {
        for (int i = 0; i < size(tSFApSFA); ++i) {
            tSFApSFA(i) = (get<0>(tSFAcSFA(i)) + kv_start + kv_block_idx * BLOCK_KV) < kv_end;
        }
        gmem_tiled_copy_A.desc_.dim_h = kv_end - (kv_start + kv_block_idx * BLOCK_KV);
        if (enable_print) {
            printf("    kv_block_idx = %d, dim_h = %d\n", kv_block_idx, kv_end - (kv_start + kv_block_idx * BLOCK_KV));
            print("        tAsA(_,_,_,smem_stage): "); print(tAsA(_,_,_,smem_stage)); print("\n");
        }
        copy_aiu(gmem_tiled_copy_A, tAgA(_,_,_,0), tAsA(_,_,_,smem_stage), warp_idx);
        if (sizeof(ElementQK) == 1 && warp_idx < WAPR_LIMIT_SFA) {
            copy_if(gmem_tiled_copy_scaleA, tSFApSFA(_,_,_,0), tSFAgSFA(_,_,_,0), tSFAsSFA(_,_,_,smem_stage));
        }
        tAgA.data() = tAgA.data() + BLOCK_KV * kHeadDim;
        tSFAgSFA.data() = tSFAgSFA.data() + BLOCK_KV;
    };

    auto load_q_s2r = [&]() {
        copy(smem_tiled_copy_B, tCsB(_,_,_,0), tCrB_copy_view);
        float * smem_weights_staged = sSFB(_,_,0).data().get() + warp_q_idx * kNumHeads;
        if constexpr (std::is_same_v<ElementLogits, float>) {
            #pragma unroll
            for (uint32_t j = 0; j < kNumHeads / 4; ++ j) {
#if __HGGC_ARCH__ == 150
                weights[j] = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
#else
                weights[j] = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) * 4 + lane_idx % 4);
#endif
            }
        } else {
            #pragma unroll
            for (uint32_t j = 0; j < kNumHeads / 4; j += 2) {
                float w0, w1;
#if __HGGC_ARCH__ == 150
                w0 = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                w1 = ld_shared(smem_weights_staged + ((j + 1) / 2) * 8 + ((j + 1) & 1) + (lane_idx % 4) * 2);
#else
                w0 = ld_shared(smem_weights_staged + (j / 2) * 8 + (j & 1) * 4 + lane_idx % 4);
                w1 = ld_shared(smem_weights_staged + ((j + 1) / 2) * 8 + ((j + 1) & 1) * 4 + lane_idx % 4);
#endif
                uint32_t d;
                asm volatile("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;\n"
                             : "=r"(d)
                             : "f"(w1), "f"(w0));
                *reinterpret_cast<uint32_t*>(&weights[j]) = d;
            }
        }
    };

    auto load_kv_scale_s2r = [&](uint32_t kv_stage_idx) {
        float * smem_kv_scales = sSFA(_,_,kv_stage_idx).data().get();
        for (int m = 0; m < kMmaIterM; m++) {
            uint32_t mma_offset = m * InstM;
            scale_kv_array[m * 2    ] = (sizeof(ElementQK) == 2) ? 1 : ld_shared(smem_kv_scales + warp_offset + mma_offset + v_0_offset);
            scale_kv_array[m * 2 + 1] = (sizeof(ElementQK) == 2) ? 1 : ld_shared(smem_kv_scales + warp_offset + mma_offset + v_1_offset);
        }
    };

    auto epilogue = [&](uint32_t block_q_idx, uint32_t kv_start, uint32_t kv_block_idx) {
        const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
        static constexpr uint32_t kNumAccumPerMma = 8;
        static constexpr uint32_t kAccumStrideN16 = kNumAccumPerMma * kMmaIterM;
        CUTE_STATIC_ASSERT(kNumHeads % 8 == 0);
        CUTE_STATIC_ASSERT(WARP_Q == 1);
        for (int m = 0; m < kMmaIterM; m++) {
            uint32_t mma_offset = m * InstM;
            float scale_kv_0 = scale_kv_array[m * 2];
            float scale_kv_1 = scale_kv_array[m * 2 + 1];

            if constexpr (std::is_same_v<ElementLogits, float>) {
                // Float epilogue: reduce over heads, scale by per-row KV scale and store
                auto shifted_accum = accum.data() + m * kNumAccumPerMma;
                const auto& transform = [&](const uint32_t& j, const uint32_t& n = 0) {
#if __HGGC_ARCH__ == 150
                    return fmaxf(shifted_accum[n * kAccumStrideN16 + j], 0) * weights[n * 4 + (j / 4) * 2 + (j & 1)];
#else
                    return fmaxf(shifted_accum[n * kAccumStrideN16 + j], 0) * weights[n * 4 + j % 4];
#endif
                };

                // Intra-thread reduction
                float sum[8] = {transform(0), transform(1), transform(2), transform(3),
                                transform(4), transform(5), transform(6), transform(7)};
                #pragma unroll
                for (uint32_t n = 1; n < kNumHeads / InstN; ++ n) {
                    #pragma unroll
                    for (uint32_t k = 0; k < kNumAccumPerMma; k ++)
                        sum[k] += transform(k, n);
                }
#if __HGGC_ARCH__ == 150
                float v_0 = (sum[0] + sum[1] + sum[4] + sum[5]) * scale_kv_0;
                float v_1 = (sum[2] + sum[3] + sum[6] + sum[7]) * scale_kv_1;
#else
                float v_0 = (sum[0] + sum[1] + sum[2] + sum[3]) * scale_kv_0;
                float v_1 = (sum[4] + sum[5] + sum[6] + sum[7]) * scale_kv_1;
#endif

                // Inter-thread reduction
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const auto& offset = static_cast<int>(1u << j);
                    v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                    v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                }

                // Store into the global memory
                const uint32_t& q_idx = block_q_idx * BLOCK_Q + warp_q_idx;
                params.logits[q_idx * params.stride_k + kv_offset + mma_offset + v_0_offset] = v_0;
                params.logits[q_idx * params.stride_k + kv_offset + mma_offset + v_1_offset] = v_1;
            } else {
                // BF16 vectorized epilogue: separate cvt and fma2 phases with __ppu_sched_bound()
                constexpr int kTotalTransforms = 4 * (kNumHeads / InstN);
                __ppu_bfloat162 sum_0 = {0, 0}, sum_1 = {0, 0};
                uint32_t cvt_buf[kTotalTransforms];

                // cvt phase: issue all ppu.cvt.rtte.bf16x2.f32.relu
                #pragma unroll
                for (int idx = 0; idx < kTotalTransforms; idx++) {
                    int n = idx / 4;
                    int sub = idx % 4;
                    int j = sub * 2;
                    asm volatile("ppu.cvt.rtte.bf16x2.f32.relu %0, %1, %2;\n"
                                 : "=r"(cvt_buf[idx])
                                 : "f"(accum(j + 1, m, n)), "f"(accum(j, m, n)));
                }
                __ppu_sched_bound();
                // fma2 phase: issue all __hfma2
                #pragma unroll
                for (int idx = 0; idx < kTotalTransforms; idx++) {
                    int n = idx / 4;
                    int sub = idx % 4;
                    int j = sub * 2;
                    __ppu_bfloat162 a = reinterpret_cast<__ppu_bfloat162&>(cvt_buf[idx]);
#if __HGGC_ARCH__ == 150
                    __ppu_bfloat162 b = *reinterpret_cast<__ppu_bfloat162*>(&weights[n * 4 + (j / 4) * 2]);
                    if (sub % 2 == 0) {
                        sum_0 = __hfma2(a, b, sum_0);
                    } else {
                        sum_1 = __hfma2(a, b, sum_1);
                    }
#else
                    __ppu_bfloat162 b = *reinterpret_cast<__ppu_bfloat162*>(&weights[n * 4 + ((j % 4) / 2) * 2]);
                    if (sub < 2) {
                        sum_0 = __hfma2(a, b, sum_0);
                    } else {
                        sum_1 = __hfma2(a, b, sum_1);
                    }
#endif
                }
                __ppu_sched_bound();
                __ppu_bfloat16 v_0_bf = __hadd(__low2bfloat16(sum_0), __high2bfloat16(sum_0));
                __ppu_bfloat16 v_1_bf = __hadd(__low2bfloat16(sum_1), __high2bfloat16(sum_1));

                // Apply per-token KV scale
                v_0_bf = __hmul(v_0_bf, (__ppu_bfloat16)scale_kv_0);
                v_1_bf = __hmul(v_1_bf, (__ppu_bfloat16)scale_kv_1);

                // Packed cross-lane reduction
                __ppu_bfloat162 packed = {v_0_bf, v_1_bf};
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    uint32_t bits = reinterpret_cast<uint32_t&>(packed);
                    uint32_t received_bits = __shfl_xor_sync(0xffffffffu, bits, 1u << j);
                    __ppu_bfloat162 received = reinterpret_cast<__ppu_bfloat162&>(received_bits);
                    packed = __hadd2(packed, received);
                }

                // Store
                const uint32_t& q_idx = block_q_idx * BLOCK_Q + warp_q_idx;
                params.logits[q_idx * params.stride_k + kv_offset + mma_offset + v_0_offset] = __low2bfloat16(packed);
                params.logits[q_idx * params.stride_k + kv_offset + mma_offset + v_1_offset] = __high2bfloat16(packed);
            }
        }
    };

    while (block_q_idx < num_q_blocks) {
        CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, kv_end, num_kv_blocks);
        tAgA.data() = tKgK.data() + kv_start * kHeadDim;
        tSFAgSFA.data() = tSFKgSFK.data() + kv_start;
        tBgB.data() = tQgQ.data() + block_q_idx * BLOCK_Q * kNumHeads * kHeadDim;
        tSFBgSFB.data() = tSFQgSFQ.data() + block_q_idx * BLOCK_Q * kNumHeads;

        if (enable_print) {
            printf("block_q_idx = %d, kv_start = %d, kv_end=%d, num_kv_blocks = %d\n",
                block_q_idx, kv_start, kv_end, num_kv_blocks);
        }

        uint32_t current_stage_kv = 0;
        if (num_kv_blocks > 0) {
            // Issue AIU Q
            load_q_g2s(block_q_idx);

            // Issue AIU K in prologue
            for (int kv_pipe = 0; kv_pipe < kNumKVStages - 1; kv_pipe++) {
                if (kv_pipe < num_kv_blocks) {
                    load_kv_g2s(kv_start, kv_end, kv_pipe, kv_pipe);
                }
                current_stage_kv++;
                cp_async_fence();
            }

            // wait AIU Q and first K
            cp_async_wait<kNumKVStages - 2>();
            __syncthreads();

            // load Q + weights to vreg
            load_q_s2r();
        }

        if constexpr (WarpInterleaving) {
            if (warp_group_id == 1) {
                __ppu_barrier_arrive(5, NumThreadsPerCTA, 0);
            }
        }

        // Compute over KV blocks
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
            uint32_t kv_stage_idx = kv_block_idx % kNumKVStages;

            if constexpr (WarpInterleaving) {
                __ppu_barrier_sync(5 + warp_group_id, NumThreadsPerCTA); // group 0 wait bar 5, group 1 wait bar 6
            }

            int kv_block_idx_copy = kv_block_idx + kNumKVStages - 1;
            if (kv_block_idx_copy < num_kv_blocks) {
                load_kv_g2s(kv_start, kv_end, kv_block_idx_copy, current_stage_kv);
            }
            cp_async_fence();
            current_stage_kv = (current_stage_kv + 1) % kNumKVStages;

            if (enable_print) {
                printf("    block_q_idx = %d, kv_block_idx = %d, kv_stage_idx = %d\n",
                    block_q_idx, kv_block_idx, kv_stage_idx);
                if (kv_stage_idx < 0) {
                    print("error error error, kv_stage_idx < 0, value =  ", kv_stage_idx);
                }
            }
            // load V to vreg
            Tensor tCsA_p = tCsA(_,_,_,kv_stage_idx);
            copy(smem_tiled_copy_A, tCsA_p, tCrA_copy_view);

            // compute
            cute::gemm(tiled_mma, accum, tCrA, tCrB, accum);

            // load scale from tsm to vreg before __ppu_barrier_arrive
            load_kv_scale_s2r(kv_stage_idx);

            if constexpr (WarpInterleaving) {
                __ppu_barrier_arrive(6 - warp_group_id, NumThreadsPerCTA, 0);
            }

            // Reduce + store logits
            epilogue(block_q_idx, kv_start, kv_block_idx);

            // wait for next K, load next K to vreg
            cp_async_wait<kNumKVStages - 2>();
            if constexpr (!WarpInterleaving) {
                __syncthreads();
            }

            clear(accum);
        }

        if constexpr (WarpInterleaving) {
            if (warp_group_id == 0) {
                __ppu_barrier_sync(5, NumThreadsPerCTA);
            }
        }

        cp_async_wait<0>();
        __syncthreads();

        // Jump to the next block
        CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
    }

  }

};

} // namespace cutlass::gemm::kernel


namespace deep_gemm {

template <typename ElementQK, typename ElementAcc, typename ElementLogits,
          uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_QH, uint32_t BLOCK_KV,
          uint32_t WARP_QH, uint32_t WARP_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          typename StrideKType = uint32_t>
class Attention {

public:
    Attention() = default;

    static uint32_t generate_id() {
        static uint32_t id = 0;
        return ++id;
    }

    static void run(const ElementQK * ptr_q,
                    const ElementQK * ptr_k,
                    const float * k_scales,
                    const float * weights,
                    uint32_t* cu_seq_len_k_start,
                    uint32_t* cu_seq_len_k_end,
                    ElementLogits* logits,
                    const uint32_t seq_len_q, const uint32_t seq_len_k, const StrideKType stride_k,
                    hggcStream_t stream, int num_sms) {

        using AttnKernel = cutlass::gemm::kernel::PPUMqaLogits<ElementQK, ElementAcc, ElementLogits, kNumHeads, kHeadDim, BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages, StrideKType>;

        using StrideA = typename AttnKernel::StrideA;
        using StrideB = typename AttnKernel::StrideB;

        auto SHAPE_M = seq_len_k;
        auto SHAPE_N = seq_len_q * kNumHeads;
        auto SHAPE_K = kHeadDim;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape((int)SHAPE_M, (int)SHAPE_K, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape((int)SHAPE_N, (int)SHAPE_K, 1));
        int max_blocks_per_cu = compute_occupancy_for_kernel<AttnKernel>();

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.cu_count = num_sms * max_blocks_per_cu;

        typename AttnKernel::Arguments arguments{ptr_q, ptr_k, k_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits,
                                                 seq_len_q, seq_len_k, stride_k, stride_A, stride_B, hw_info};
        auto params = arguments;
        dim3 const block = AttnKernel::get_block_shape();
        dim3 const grid = AttnKernel::get_grid_shape(params);
        int smem_size_kernel = AttnKernel::SharedStorageSize;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()){
            std::string data_type_str = "unknown";
            if (std::is_same_v<ElementQK, cutlass::bfloat16_t>) {
                data_type_str = "bf16";
            } else if (std::is_same_v<ElementQK, cutlass::float_e4m3_t>) {
                data_type_str = "fp8";
            } else if (std::is_same_v<ElementQK, int8_t>) {
                data_type_str = "int8";
            }

            dg_prof_params.set_mqa_logits_params(data_type_str, seq_len_q, seq_len_k, kNumHeads, kHeadDim, stream);
            if constexpr (std::is_same_v<ElementLogits, __ppu_bfloat16>) {
                dg_prof_params.add_params("logits_dtype", std::string("bf16"));
            }
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        cutlass::device_kernel<AttnKernel><<<grid, block, smem_size_kernel, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        int max_active_tb_num = max_blocks_per_cu;
        const int threadblock_count = num_sms * max_active_tb_num;
        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, cutlass::device_kernel<AttnKernel>);

            printf("[mqa_logits:]\n");
            printf("kNumHeads:%u, kHeadDim:%u, seq_len_q:%u, seq_len_k:%u, stride_k:%llu\n",
                kNumHeads, kHeadDim, seq_len_q, seq_len_k, static_cast<unsigned long long>(stride_k));

            printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n",
                BLOCK_QH, BLOCK_KV, WARP_QH, WARP_KV, kNumQStages, kNumKVStages);

            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms, max_active_tb_num, threadblock_count);
            printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
            std::cout << "block = " << block << std::endl;
            std::cout << "grid = " << grid << std::endl;
        }
    }
};

};  // namespace deep_gemm