#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include "cutlass/cutlass.h"
#include "utils.cuh"
#include <iostream>
#include "profiling_interface.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/collective/ppu_epilogue_vectorized.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "ppu_include.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_conversion.h"
#include "tools/util/include/cutlass/util/host_tensor.h"
#include "tools/util/include/cutlass/util/packed_stride.hpp"
#include "scheduler_cutlass3.cuh"
#include "utils_cutlass3.h"
#include <math.h>


namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  class DispatchPolicy_,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_,
  class ElementSFA_,
  class StrideSFA_,
  class GmemTiledCopySFA_,
  class SmemLayoutAtomSFA_,
  class ElementSFB_,
  class StrideSFB_,
  class GmemTiledCopySFB_,
  class SmemLayoutAtomSFB_,
  FP4DynamicTileId kDynamicTileId = FP4DynamicTileId::Disabled>
struct CollectiveMmaScaleFp4
{
  //
  // Type Aliases
  //
  using DispatchPolicy = DispatchPolicy_;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using GmemTiledCopySFA = GmemTiledCopySFA_;
  using SmemLayoutAtomSFA = SmemLayoutAtomSFA_;
  using GmemTiledCopySFB = GmemTiledCopySFB_;
  using SmemLayoutAtomSFB = SmemLayoutAtomSFB_;
  using ElementSFA = ElementSFA_;
  using StrideSFA = StrideSFA_;
  using ElementSFB = ElementSFB_;
  using StrideSFB = StrideSFB_;
  using ThrLayoutVMNK = typename TiledMma::ThrLayoutVMNK;
  static constexpr int warp_on_m = size<1>(ThrLayoutVMNK{});
  static constexpr int warp_on_n = size<2>(ThrLayoutVMNK{});

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  static constexpr int BlockKSF = size<2>(TileShape{}) / 32;  // must ensure BlockK is divideable by 32
  using SmemLayoutSFA_ = decltype(tile_to_shape(
      SmemLayoutAtomSFA{},
      make_shape(shape<0>(TileShape{}), Int<BlockKSF>{}, Int<DispatchPolicy::Stages>{})));
  using SmemLayoutSFB_ = decltype(tile_to_shape(
      SmemLayoutAtomSFB{},
      make_shape(shape<1>(TileShape{}), Int<BlockKSF>{}, Int<DispatchPolicy::Stages>{})));

  template<int block_mn_sf>
  static constexpr int deduce_smem_swizzle_padding_sf(cute::Int<block_mn_sf>) {
    // element type is uint16_t, when BlockM > 64, need to pad 16 rows in case of bankconflict.
    if constexpr (block_mn_sf > 64) return 16;
    else return 0;
  }

  template<typename SmemLayoutSF, int SwizzlePaddingSF>
  static constexpr auto deduce_smem_layout_sf(SmemLayoutSF smem_layout_sf, cute::Int<SwizzlePaddingSF>) {
    constexpr auto smem_layout_sf_shape = shape(smem_layout_sf);
    constexpr auto smem_layout_sf_stride = stride(smem_layout_sf);

    if constexpr (rank<0>(smem_layout_sf) == 1) {
      constexpr auto shape_m = get<0>(smem_layout_sf_shape);
      constexpr auto shape_k = get<1>(smem_layout_sf_shape);
      constexpr auto shape_stage = get<2>(smem_layout_sf_shape);
      constexpr auto stride_m = get<0>(smem_layout_sf_stride);
      constexpr auto stride_k = get<1>(smem_layout_sf_stride);
      constexpr auto stride_stage = get<2>(smem_layout_sf_stride);

      // if the BlockM is 16, 32, 64, swizzle padding is no need;
      constexpr auto sfa_shape = make_shape(make_shape(shape_m, Int<1>{}), shape_k, shape_stage);
      constexpr auto sfa_stride = make_stride(make_stride(stride_m, Int<0>{}), stride_k, stride_stage);

      return Layout<decltype(sfa_shape), decltype(sfa_stride)>{};
    } else {
      constexpr auto shape_atom_k = get<0, 1>(smem_layout_sf_shape); // must be <0, 1> rather than <1>
      constexpr auto stride_atom_m = get<0, 0>(smem_layout_sf_stride);
      constexpr auto stride_k_block = get<0, 1>(smem_layout_sf_stride) + Int<SwizzlePaddingSF>{};
      constexpr auto stride_outer_m = get<1>(smem_layout_sf_stride);
      constexpr auto stride_stage = get<2>(smem_layout_sf_stride) + Int<SwizzlePaddingSF>{} * Int<shape_atom_k>{};

      constexpr auto sfa_stride = make_stride(make_stride(stride_atom_m, stride_k_block), stride_outer_m, stride_stage);

      return Layout<decltype(smem_layout_sf_shape), decltype(sfa_stride)>{};
    }
  }

  static constexpr int SwizzlePaddingSFA = deduce_smem_swizzle_padding_sf(shape<0>(TileShape{}));
  static constexpr auto smem_layout_sfa = deduce_smem_layout_sf(SmemLayoutSFA_{}, Int<SwizzlePaddingSFA>{});
  using SmemLayoutSFA = decltype(smem_layout_sfa);

  static constexpr int SwizzlePaddingSFB = deduce_smem_swizzle_padding_sf(shape<1>(TileShape{}));
  static constexpr auto smem_layout_sfb = deduce_smem_layout_sf(SmemLayoutSFB_{}, Int<SwizzlePaddingSFB>{});
  using SmemLayoutSFB = decltype(smem_layout_sfb);

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  // warp_num == 1, use sigle warp to issue AIU LOAD
  static constexpr bool SplitAIU = (warp_on_m * warp_on_n) > 1 && (size<0>(TileShape{}) != 256 || size<1>(TileShape{}) != 256);
  static constexpr int AIU_SFA_WARP = (SplitAIU && (warp_on_m * warp_on_n) > 2) ? 2 : 0;
  static constexpr int AIU_SFB_WARP = (SplitAIU && (warp_on_m * warp_on_n) > 3) ? 3 : 0;

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    cute::array_aligned<ElementSFA, cute::cosize_v<SmemLayoutSFA>> smem_sfa;
    cute::array_aligned<ElementSFB, cute::cosize_v<SmemLayoutSFB>> smem_sfb;
  };

  // Host side kernel arguments
  struct Arguments {
    Shape<int,int,int> problem_shape;
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementSFA const* ptr_scale_A;
    StrideSFA dSFA;
    ElementSFB const* ptr_scale_B;
    StrideSFB dSFB;
  };

  // Device side kernel params
  using Params = Arguments;
  // store params to const scale tensor in mainloop
  // avoid modify kernel file for fp4
  Params params_;

  // sm90 realization put TMA_A into params directly
  // put gmem_tiled_copy here and copy desc in kernel to simplify rtc usage
  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;
  GmemTiledCopySFA gmem_tiled_copy_SFA;
  GmemTiledCopySFB gmem_tiled_copy_SFB;

  //
  // Methods
  //

  template <class ProblemShape>
  CUTLASS_DEVICE
  CollectiveMmaScaleFp4(Params params, ProblemShape problem_shape_MNK, int batch_idx = 0) {
    params_ = params;

    auto M = get<0>(problem_shape_MNK);
    auto N = get<1>(problem_shape_MNK);
    auto K = get<2>(problem_shape_MNK);
    auto SFK = cute::ceil_div(K, Int<32>{});

    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;

    gmem_tiled_copy_A.desc_.template init<ElementA, false, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, M, K, params.dA);
    gmem_tiled_copy_B.desc_.template init<ElementB, false, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, N, K, params.dB);

    // update desc.ptr to current batch, use y_offset as batch will cause smaller bit wide
    gmem_tiled_copy_A.desc_.gmem_ptr += batch_idx * get<2>(params.dA) * sizeof(ElementA);
    gmem_tiled_copy_B.desc_.gmem_ptr += batch_idx * get<2>(params.dB) * sizeof(ElementB);

    using TilerSFA = typename GmemTiledCopySFA::Tiler_MN;
    constexpr bool TransSFA = true;
    gmem_tiled_copy_SFA.desc_.template init<ElementSFA, TransSFA, get<0>(TilerSFA{}), get<1>(TilerSFA{})>(nullptr, M, SFK, params.dSFA);

    using TilerSFB = typename GmemTiledCopySFB::Tiler_MN;
    constexpr bool TransSFB = true;
    gmem_tiled_copy_SFB.desc_.template init<ElementSFB, TransSFB, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(nullptr, N, SFK, params.dSFB);
  };

  template <class ProblemShape_MNKL, class BlockCoord_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, BlockCoord_MNKL const& blk_coord_mnkl, Params const& params) {
    using X = Underscore;

    auto [M,N,K,L] = problem_shape_MNKL;
    auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;

    // load init A
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_A), make_shape(M,K,L), params.dA);   // (m,k,l)
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,l_coord));                                 // (m,k)
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)

    // load init B
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.ptr_B), make_shape(N,K,L), params.dB);   //(n,k,l)
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));                                 // (n,k)
    Tensor gB = local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

    // load init scale A/B
    auto mSFA_layout = make_layout(make_shape(M, cute::ceil_div(K, 16 * 2)), LayoutLeft{}); // uint16_t
    Tensor mSFA_mk = make_tensor(make_gmem_ptr(params_.ptr_scale_A), mSFA_layout);
    Tensor gSFA = local_tile(make_mix_tensor_like(mSFA_mk), make_shape(shape<0>(TileShape{}), Int<BlockKSF>{}), make_coord(m_coord, _)); // (BLK_M,BLK_K,k)

    auto mSFB_layout = make_layout(make_shape(N, cute::ceil_div(K, 16 * 2)), LayoutLeft{}); // uint16_t
    Tensor mSFB_nk = make_tensor(make_gmem_ptr(params_.ptr_scale_B), mSFB_layout);
    Tensor gSFB = local_tile(make_mix_tensor_like(mSFB_nk), make_shape(shape<1>(TileShape{}), Int<BlockKSF>{}), make_coord(n_coord, _)); // (BLK_N,BLK_K,k)

    return cute::make_tuple(gA, gB, gSFA, gSFB);
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  static CUTLASS_DEVICE void ld_mat_m8n8_x1_b16(uint128_t const& smem_src, uint32_t& dst0) {
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    asm volatile ("ppu.tc01.ex.ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
      : "=r"(dst0)
      : "r"(smem_int_ptr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
    asm volatile ("ppu.tc02.ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
      : "=r"(dst0)
      : "r"(smem_int_ptr));
#else
    assert(0);
#endif
  }

  template <typename SmemLayoutSF, int warp_iter_num, int warp_iter_stride>
  static CUTLASS_DEVICE void cal_ldmat_offset(int& base_offset, SmemLayoutSF& smem_layout_sf, const int lane_idx, const int warp_mn_idx, cute::Int<warp_iter_num>, cute::Int<warp_iter_stride>) {
    constexpr int inner_shape = size<0, 0>(cute::shape(SmemLayoutSF{}));
    constexpr int inner_stride = size<0, 1>(cute::stride(SmemLayoutSF{}));

    constexpr int valid_lane_id = warp_iter_num * 2;
    if (lane_idx < valid_lane_id) {
      // each 2 lane consist a group.
      int group_idx = lane_idx / 2;
      int is_down = lane_idx % 2;
      int warp_offset = warp_mn_idx * 16;
      int group_offset = group_idx * warp_iter_stride;
      int down_offset = is_down * 8;
      int offset_ = warp_offset + group_offset + down_offset;
      base_offset = (offset_ % inner_shape) + (offset_ / inner_shape) * inner_stride;
    }
  }

  template <typename TensorSF, typename TensortCrSF, int k_block>
  static CUTLASS_DEVICE void sf_s2r(TensorSF& sSF, TensortCrSF& tCrSF, int base_offset, const int lane_idx, cute::Int<k_block>, const int stage) {
    static constexpr int k_block_stride = size<1>(cute::stride(TensorSF{}));
    static constexpr int stage_stride = size<2>(cute::stride(TensorSF{}));

    // for SFA & SFB
    auto& vreg = tCrSF(Int<0>{}, Int<0>{}, k_block);
    auto smem_start_ptr_sf = cute::raw_pointer_cast(sSF.data());

    constexpr int k_block_offset = k_block * k_block_stride;
    auto smem_ptr_lane_sf = smem_start_ptr_sf + base_offset + k_block_offset + stage * stage_stride;
    ld_mat_m8n8_x1_b16(*reinterpret_cast<uint128_t const*>(smem_ptr_lane_sf), vreg);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class... Ts,
    class FrgTensorD,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    int aiu_warp_group_thread_idx = warp_idx * 32;

    int warp_m_idx = warp_idx % warp_on_m;
    int warp_n_idx = warp_idx / warp_on_m;

    auto M = get<0>(params_.problem_shape);
    auto N = get<1>(params_.problem_shape);
    auto K = get<2>(params_.problem_shape);

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    Tensor gSFA = get<2>(load_inputs);
    Tensor gSFB = get<3>(load_inputs);

    Tensor sSFA = make_tensor(make_smem_ptr(storage.smem_sfa.data()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(storage.smem_sfb.data()), SmemLayoutSFB{});

    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_thread_slice(thread_idx);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_thread_slice(thread_idx);

    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    auto copy_to_tsm = [&](int k_pipe_write, int k_idx, int warp_idx) {
      copy_aiu<SplitAIU>(
        gmem_tiled_copy_A, tAgA(_,_,_,k_idx), tAsA(_,_,_,k_pipe_write),
        gmem_tiled_copy_B, tBgB(_,_,_,k_idx), tBsB(_,_,_,k_pipe_write),
        warp_idx
      );
      if (warp_idx == AIU_SFA_WARP) {
        copy(gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,k_idx), tSFAsSFA(_,_,_,k_pipe_write));
      }
      if (warp_idx == AIU_SFB_WARP) {
        copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_idx), tSFBsSFB(_,_,_,k_pipe_write));
      }
    };

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages; ++k_pipe) {
      if (k_tile_count > 0){
        copy_to_tsm(k_pipe, *k_tile_iter, warp_idx);
        ++k_tile_iter;
      }
      cp_async_fence();
      --k_tile_count;
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //

    // tsm ld swzl needn't distinguish params inner warp
    // use original smem_tiled_copy to avoid use specific tailed_layout_tv like below, use warp_idx*32 to get scaler layout of current warp
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    smem_tiled_copy_A.smem_base_ = storage.smem_a.data();
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                 // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    smem_tiled_copy_B.smem_base_ = storage.smem_b.data();
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // create rSFA refer to tCsA/tCrA
    constexpr auto warp_iter_num_sfa = size<1>(tCsA);
    constexpr auto block_iter_num_sfa = size<2>(tCsA);
    constexpr auto warp_iter_stride_sfa = [&]() constexpr {
      if constexpr (cute::is_scaled_basis<decltype(stride<1>(tCsA))>::value) {
        return stride<1>(tCsA).value();
      } else {
        return stride<1>(tCsA);
      }
    }();
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
    cal_ldmat_offset(base_offset_sfa, SmemLayoutSFA{}, lane_idx, warp_m_idx, Int<warp_iter_num_sfa>{}, Int<warp_iter_stride_sfa>{});
    cal_ldmat_offset(base_offset_sfb, SmemLayoutSFB{}, lane_idx, warp_n_idx, Int<warp_iter_num_sfb>{}, Int<warp_iter_stride_sfb>{});
    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-1>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
      sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, 0);
      sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, 0);
    }

    auto process_kblock_iterations = [&](auto k_block) {
      if constexpr (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tCsA_p = tCsA(_,_,_,smem_pipe_read);
        tCsB_p = tCsB(_,_,_,smem_pipe_read);
        copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
        sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, Int<0>{}, smem_pipe_read);
        sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, Int<0>{}, smem_pipe_read);
      } else {
        auto k_block_next = k_block + Int<1>{};  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        sf_s2r(sSFA, tCrSFA, base_offset_sfa, lane_idx, k_block_next, smem_pipe_read);
        sf_s2r(sSFB, tCrSFB, base_offset_sfb, lane_idx, k_block_next, smem_pipe_read);
      }
      // Transform before compute
      cute::transform(tCrA(_,_,k_block), TransformA{});
      cute::transform(tCrB(_,_,k_block), TransformB{});
      cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrSFA(_,_,k_block), tCrB(_,_,k_block), tCrSFB(_,_,k_block), src_accum);
      if constexpr (k_block == K_BLOCK_MAX - 2) {
        // Commit the smem for smem_pipe_read
        cp_async_wait<DispatchPolicy::Stages-2>();
        __syncthreads();

        if (k_tile_count > 0){
          copy_to_tsm(smem_pipe_write, *k_tile_iter, warp_idx);
          ++k_tile_iter;
        }
        cp_async_fence();

        // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;

          smem_pipe_write = smem_pipe_read;
        }
  };
    if constexpr (kDynamicTileId != FP4DynamicTileId::LargeEM) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        process_kblock_iterations(k_block);
      });
      --k_tile_count;
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages); --k_tile_count)
    {
      // Pipeline the outer products with a static for loop.
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        process_kblock_iterations(k_block);
      });
    }

    // TODO: original cutlass3 miss this sync
    cp_async_wait<0>();
    __syncthreads();
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective