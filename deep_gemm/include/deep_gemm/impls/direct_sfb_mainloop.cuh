/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
// Derived from Actlize v1.0.0:
// include/cutlass/gemm/collective/ppu_mma_aiu_multistage_with_scale.hpp.
// The BN192 specialization maps internal N64 loader coordinates to the
// original W128x128 scales without repacking weights or changing the MMA loop.

#include <cutlass/cutlass.h>

#include <cute/algorithm/functional.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/tensor_predicate.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>

#include <cutlass/gemm/collective/collective_mma_decl.hpp>

#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/gemm/collective/ppu_promotion_with_scale_accumulation.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<class Native> struct DirectSfbMainloop;

template <
  typename Arch,
  int Stages_,
  class TileShape_,
  class ElementA_,
  class StridePairA_,
  class ElementB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomPairA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomPairB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct DirectSfbMainloop<CollectiveMma<
  Arch,
  MainloopWithScalePPUAiu<Stages_, KernelAiuMultistageWithBlockWiseScale>,
  TileShape_,
  ElementA_,
  StridePairA_,
  ElementB_,
  StridePairB_,
  TiledMma_,
  GmemTiledCopyPairA_,
  SmemLayoutAtomPairA_,
  SmemCopyAtomA_,
  TransformA_,
  GmemTiledCopyPairB_,
  SmemLayoutAtomPairB_,
  SmemCopyAtomB_,
  TransformB_>>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopWithScalePPUAiu<Stages_, KernelAiuMultistageWithBlockWiseScale>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = cute::tuple_element_t<0,StridePairA_>;
  using ElementB = ElementB_;
  using StrideB = cute::tuple_element_t<0,StridePairB_>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = cute::tuple_element_t<0, GmemTiledCopyPairA_>;
  using GmemTiledCopyB = cute::tuple_element_t<0, GmemTiledCopyPairB_>;
  using SmemLayoutAtomA = cute::tuple_element_t<0, SmemLayoutAtomPairA_>;
  using SmemLayoutAtomB = cute::tuple_element_t<0, SmemLayoutAtomPairB_>;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ElementScale = float;
  using ArchTag = Arch;

  static constexpr int NumThreadsPerCTA = size(TiledMma{});

  // WarpInterleaving is enabled only when NumThreadsPerCTA is 512, which satisfy the condition that 2 warp group partitioned onto separate WEs.
  static constexpr bool WarpInterleaving = (NumThreadsPerCTA == 512);

  using LayoutSFA = cute::tuple_element_t<1,StridePairA_>;
  using LayoutSFB = cute::tuple_element_t<1,StridePairB_>;

  static constexpr int ScaleGranularityM = size<0,0>(LayoutSFA{});
  static constexpr int ScaleGranularityN = size<0,0>(LayoutSFB{});
  static constexpr int ScaleGranularityK = size<1,0>(LayoutSFA{});
  static constexpr int blockK = size<2>(TileShape{});

  static_assert(blockK > ScaleGranularityK ? (blockK % ScaleGranularityK) == 0 : (ScaleGranularityK % blockK) == 0,
              "block scaling granularity must evenly divide tile shape along K.");
  static_assert(ScaleGranularityK % size<2>(typename TiledMma::AtomShape_MNK{}) == 0);

  static constexpr int ScalePromotionInterval = ScaleGranularityK / size<2>(typename TiledMma::AtomShape_MNK{});
  static_assert(ScalePromotionInterval % 4 == 0, "ScalePromotionInterval must be a multiple of 4.");
  static constexpr int ScaleMsPerTile = cute::ceil_div(size<0>(TileShape{}), Int<ScaleGranularityM>{});
  static constexpr int ScaleNsPerTile = cute::ceil_div(size<1>(TileShape{}), Int<ScaleGranularityN>{});
  static constexpr int ScaleKsPerTile = cute::ceil_div(size<2>(TileShape{}), Int<ScaleGranularityK>{});

  static constexpr int StageNumPerBlock = cute::ceil_div(Int<ScaleGranularityK>{}, size<2>(TileShape{}));
  static constexpr int ScaleStages = StageNumPerBlock > 1 ? (DispatchPolicy::Stages / StageNumPerBlock + 1) : DispatchPolicy::Stages;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static constexpr bool TransSFA = is_static<decltype(stride<1>(LayoutSFA{}))>::value ? false : true;
  static constexpr bool TransSFB = is_static<decltype(stride<1>(LayoutSFB{}))>::value ? false : true;

  using ScaleConfig = ::cutlass::detail::PPUBlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
  using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(TileShape{}));
  using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(TileShape{}));

  using SmemLayoutA = decltype(tile_to_shape(
          SmemLayoutAtomA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
          SmemLayoutAtomB{}, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  using CopyAtomSFA = Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS_ZFILL<ElementScale>, ElementScale>;
  using CopyAtomSFB = Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS_ZFILL<ElementScale>, ElementScale>;

  // Block scaling smem layout
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  // for aiu ld
  static constexpr int warp_sfa = cute::ceil_div(Int<ScaleMsPerTile>{}, _32{});

  static constexpr bool IsAiuLoadSFA = !(shape<0>(TileShape{}) > 32 && shape<0>(TileShape{}) % 32 == 0 && (warp_sfa + 2) <= cute::min(8, NumThreadsPerCTA / 32));
  static constexpr bool IsAiuLoadSFB = true;

  static constexpr int MaxAiuContElemSize = 128 / (sizeof_bits<ElementScale>::value / 8);     // 32
  static constexpr int MinAiuContElemSize = 32 / (sizeof_bits<ElementScale>::value / 8);      // 8
#if 1
  static constexpr int SFATileM = TransSFA ? cute::max(ScaleMsPerTile, MinAiuContElemSize) : ScaleMsPerTile;
  static constexpr int SFATileK = TransSFA ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);
#else
  static constexpr int SFATileM = MaxAiuContElemSize;
  static constexpr int SFATileK = ScaleMsPerTile * ScaleKsPerTile / MaxAiuContElemSize;
#endif

  static constexpr int SFBTileN = TransSFB ? cute::max(ScaleNsPerTile, MinAiuContElemSize) : ScaleNsPerTile;
  static constexpr int SFBTileK = TransSFB ? ScaleKsPerTile : cute::max(ScaleKsPerTile, MinAiuContElemSize);

  using GmemTiledCopySFA = cute::tuple_element_t<1, GmemTiledCopyPairA_>;
  using GmemTiledCopySFB = cute::tuple_element_t<1, GmemTiledCopyPairB_>;

  using SmemLayoutAtomSFA_ = cute::conditional_t<TransSFA,
            cute::tuple_element_t<1, SmemLayoutAtomPairA_>,
            Layout<Shape<Int<SFATileM>, Int<SFATileK>>, Stride<Int<SFATileK>, _1>>>;
  using SmemLayoutAtomSFB_ = cute::conditional_t<TransSFB,
            cute::tuple_element_t<1, SmemLayoutAtomPairB_>,
            Layout<Shape<Int<SFBTileN>, Int<SFBTileK>>, Stride<Int<SFBTileK>, _1>>>;

  using SmemLayoutSFA_ = decltype(tile_to_shape(
      SmemLayoutAtomSFA_{},
      make_shape(Int<SFATileM>{}, Int<SFATileK>{}, Int<ScaleStages>{})));

  using SmemLayoutSFB_ = decltype(tile_to_shape(
      SmemLayoutAtomSFB_{},
      make_shape(Int<SFBTileN>{}, Int<SFBTileK>{}, Int<ScaleStages>{})));

  using RealSmemLayoutSFA = cute::conditional_t<IsAiuLoadSFA, SmemLayoutSFA_, SmemLayoutSFA>;
  using RealSmemLayoutSFB = cute::conditional_t<IsAiuLoadSFB, SmemLayoutSFB_, SmemLayoutSFB>;

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    CUTE_ALIGNAS(128) cute::array<ElementScale, cute::cosize_v<RealSmemLayoutSFA>> smem_SFA; // ScaleMsPerTile x PIPE_K
    CUTE_ALIGNAS(128) cute::array<ElementScale, cute::cosize_v<RealSmemLayoutSFB>> smem_SFB; // ScaleNsPerTile x PIPE_K
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    uint32_t mma_promotion_interval = 4;
    ElementScale const* ptr_scale_A;
    LayoutSFA layout_SFA;
    ElementScale const* ptr_scale_B;
    LayoutSFB layout_SFB;
  };

  // Device side kernel params
  using Params = Arguments;

  // put gmem_tiled_copy here and copy desc in kernel to simplify rtc usage
  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;
  GmemTiledCopySFA gmem_tiled_copy_SFA;
  GmemTiledCopySFB gmem_tiled_copy_SFB;
  // Per-tile local object state; the kernel Params/Arguments ABI is unchanged.
  int sfb_n_origin64 = 0;
  struct SfbCopyPredicate {
    bool valid;
    CUTLASS_DEVICE bool operator()() const { return valid; }
  };
  static_assert(ScaleGranularityN == 64 && ScaleGranularityK == 128 && blockK == 128);
  static_assert(!TransSFB && ScaleNsPerTile == 3, "direct SFB requires BN192/K128 N64 shared layout");


  //
  // Methods
  //
  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;
    bool implementable = true;
    // We expect full tiles in K
    if (K % size<2>(TileShape{}) != 0) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size K is incompatible with tile size.\n");
    }
    return implementable;
  }

  template <class ProblemShape_MNKL, class BlockCoord_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, BlockCoord_MNKL const& blk_coord_mnkl, Params const& params) {
    using X = Underscore;

    auto [M,N,K,L] = problem_shape_MNKL;
    auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;

    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;

    static constexpr bool TransA = is_static<decltype(get<1>(params.dA))>::value ? false : true;
    static constexpr bool TransB = is_static<decltype(get<1>(params.dB))>::value ? false : true;

    gmem_tiled_copy_A.desc_.template init<ElementA, TransA, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, M, K, params.dA);
    gmem_tiled_copy_B.desc_.template init<ElementB, TransB, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, N, K, params.dB);

    // update desc.ptr to current batch, use y_offset as batch will cause smaller bit wide
    // gmem_tiled_copy_A.desc_.gmem_ptr += batch_idx * get<2>(params.dA) * sizeof(ElementA);
    // gmem_tiled_copy_B.desc_.gmem_ptr += batch_idx * get<2>(params.dB) * sizeof(ElementB);

    // load init A
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_A), make_shape(M,K,L), params.dA);   // (m,k,l)
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,l_coord));                                 // (m,k)
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)

    // load init B
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.ptr_B), make_shape(N,K,L), params.dB);   // (n,k,l)
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));                                 // (n,k)
    Tensor gB = local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

    // load init scale A/B
    auto gSFA = [&]() {
      auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;
      if constexpr (IsAiuLoadSFA) {
        auto scale_m = shape<0,1>(params.layout_SFA);
        auto scale_k = shape<1,1>(params.layout_SFA);
        static constexpr int m_factor = cute::ceil_div(Int<ScaleGranularityM>{}, size<0>(TileShape{}));
        using TilerSFA = typename GmemTiledCopySFA::Tiler_MN;
#if 1
        auto stride_SFA = cute::make_stride(stride<0,1>(params.layout_SFA), stride<1,1>(params.layout_SFA));
        auto layout_SFA = make_layout(cute::make_shape(scale_m, scale_k), stride_SFA);
        Tensor mSFA_mk = make_tensor(make_gmem_ptr(params.ptr_scale_A), layout_SFA);
        Tensor gSFA = local_tile(make_mix_tensor_like(mSFA_mk), make_tile(Int<ScaleMsPerTile>{}, Int<ScaleKsPerTile>{}),
            make_coord(m_coord / m_factor, _));
        gmem_tiled_copy_SFA.desc_.template init<ElementScale, TransSFA, get<0>(TilerSFA{}), get<1>(TilerSFA{})>(
          nullptr, scale_m, scale_k, stride_SFA);
#else
        // fold
        auto layout_SFA = cute::make_layout(
          cute::make_shape(
            cute::make_shape(Int<SFATileM>{}, scale_m / ScaleMsPerTile),
            cute::make_shape(Int<SFATileK>{}, scale_k)
          ),
          cute::make_stride(
            cute::make_stride(_1{}, Int<ScaleMsPerTile>{}),
            cute::make_stride(Int<SFATileM>{}, scale_m)
          )
        );
        Tensor mSFA_mk = make_tensor(make_gmem_ptr(params.ptr_scale_A), layout_SFA);
        auto layout_counting = make_layout(mSFA_mk.shape(), make_stride(
          make_stride(ScaledBasis<_0, 1>{}, ScaledBasis<Int<SFATileK>, 0>{}),
          make_stride(ScaledBasis<_0, 0>{}, ScaledBasis<int, 0>{scale_m / ScaleMsPerTile * SFATileK})
        ));
        Tensor mA_nk_counting = make_counting_tensor(layout_counting);
        Tensor gSFA = local_tile(mA_nk_counting, make_tile(Int<ScaleMsPerTile / SFATileK>{}, Int<SFATileK>{}), make_coord(m_coord,_));
        auto stride_SFA = cute::make_stride(stride<0,1>(params.layout_SFA), SFATileM);
        gmem_tiled_copy_SFA.desc_.template init<ElementScale, TransSFA, get<0>(TilerSFA{}), get<1>(TilerSFA{})>(
          (uint8_t*)(raw_pointer_cast(mSFA_mk.data())), SFATileM, scale_m * scale_k / SFATileM, stride_SFA);
#endif
        return cute::make_tuple(gSFA, 0);
      }
      else {
        Tensor mSFA_mkl = make_tensor(make_gmem_ptr(params.ptr_scale_A), params.layout_SFA);          // (scale_m,k,l)
        Tensor gSFA_mkl = local_tile(mSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M,BLK_K,m,k,l)
        Tensor gSFA = gSFA_mkl(_,_,m_coord,_,l_coord);
        Tensor iSFA_mkl = make_identity_tensor(shape(params.layout_SFA));
        Tensor cSFA_mkl = local_tile(iSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M,BLK_K,m,k,l)
        Tensor cSFA = cSFA_mkl(_,_,m_coord,_,l_coord);
        return cute::make_tuple(gSFA, cSFA);
      }
    }();

    auto gSFB = [&]() {
      auto [m_coord, n_coord, _, l_coord] = blk_coord_mnkl;
      if constexpr (IsAiuLoadSFB) {
        auto scale_n = shape<0,1>(params.layout_SFB);
        auto scale_k = shape<1,1>(params.layout_SFB);
        auto stride_SFB = cute::make_stride(stride<0,1>(params.layout_SFB), stride<1,1>(params.layout_SFB));
        auto layout_SFB = make_layout(cute::make_shape(scale_n, scale_k), stride_SFB);

        static constexpr int n_factor = cute::ceil_div(Int<ScaleGranularityN>{}, size<1>(TileShape{}));
        Tensor mSFB_nk = make_tensor(make_gmem_ptr(params.ptr_scale_B), layout_SFB);
        Tensor gSFB = local_tile(make_mix_tensor_like(mSFB_nk), make_tile(Int<ScaleNsPerTile>{}, Int<ScaleKsPerTile>{}),
                                 make_coord(n_coord / n_factor,_));
        using TilerSFB = typename GmemTiledCopySFB::Tiler_MN;
        gmem_tiled_copy_SFB.desc_.template init<ElementScale, TransSFB, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(
          nullptr, scale_n, scale_k, stride_SFB);

        return cute::make_tuple(gSFB, 0);
      }
      else {
        Tensor mSFB_nkl = make_tensor(make_gmem_ptr(params.ptr_scale_B), params.layout_SFB);          // (scale_n,k,l)
        Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N,BLK_K,n,k,l)
        Tensor gSFB = gSFB_nkl(_,_,n_coord,_,l_coord);
        Tensor iSFB_nkl = make_identity_tensor(shape(params.layout_SFB));
        Tensor cSFB_nkl = local_tile(iSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N,BLK_K,n,k,l)
        Tensor cSFB = cSFB_nkl(_,_,n_coord,_,l_coord);
        return cute::make_tuple(gSFB, cSFB);
      }
    }();

    return cute::make_tuple(gA, gB, gSFA, gSFB);
  }

  template <
    class... Ts,
    class FrgTensorC,
    class KTileIterator
  >
  CUTLASS_DEVICE void
  operator() (
      Params const& mainloop_params,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC &accum,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    // static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();
    int aiu_warp_group_thread_idx = warp_idx * 32;
    int warp_group_id = warp_idx / 8;

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

    Tensor gSFA = get<2, 0>(load_inputs);
    Tensor gSFB = get<3, 0>(load_inputs);

    auto sSFA = make_tensor(cute::make_smem_ptr(storage.smem_SFA.data()), RealSmemLayoutSFA{});
    auto sSFB = make_tensor(cute::make_smem_ptr(storage.smem_SFB.data()), RealSmemLayoutSFB{});

    auto SFA_load_tuple = [&]() {
      if constexpr (IsAiuLoadSFA) {
        auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_slice(thread_idx);
        Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
        Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);
        return cute::make_tuple(0, tSFAgSFA, tSFAsSFA, 0, 0);
      } else {
        Tensor cSFA = get<2, 1>(load_inputs);
        TiledCopy scale_copy_a = make_tiled_copy(CopyAtomSFA{}, Layout<Shape<Int<ScaleMsPerTile>>>{}, Layout<Shape<_1>>{});
        ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(thread_idx - 64);   // use warp 2 - warp n
        Tensor tSFAgSFA = thr_scale_copy_a.partition_S(gSFA);
        Tensor tSFAsSFA = thr_scale_copy_a.partition_D(sSFA);
        Tensor tSFAcSFA = thr_scale_copy_a.partition_S(cSFA);
        Tensor tSFApSFA = make_tensor<bool>(shape(filter_zeros(tSFAsSFA(_,_,_,_0{}))));
        return cute::make_tuple(scale_copy_a, tSFAgSFA, tSFAsSFA, tSFApSFA, tSFAcSFA);
      }
    }();
    auto SFB_load_tuple = [&]() {
      if constexpr (IsAiuLoadSFB) {
        auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_slice(thread_idx);
        Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
        Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);
        return cute::make_tuple(0, tSFBgSFB, tSFBsSFB, 0, 0);
      } else {
        Tensor cSFB = get<3, 1>(load_inputs);
        TiledCopy scale_copy_b = make_tiled_copy(CopyAtomSFB{}, Layout<Shape<_32>>{}, Layout<Shape<_1>>{});
        ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(thread_idx - 32);   // use warp 1
        Tensor tSFBgSFB = thr_scale_copy_b.partition_S(gSFB);
        Tensor tSFBsSFB = thr_scale_copy_b.partition_D(sSFB);
        Tensor tSFBcSFB = thr_scale_copy_b.partition_S(cSFB);
        Tensor tSFBpSFB = make_tensor<bool>(shape(filter_zeros(tSFBsSFB(_,_,_,_0{}))));
        return cute::make_tuple(scale_copy_b, tSFBgSFB, tSFBsSFB, tSFBpSFB, tSFBcSFB);
      }
    }();

    auto scale_copy_a = get<0>(SFA_load_tuple);
    auto tSFAgSFA = get<1>(SFA_load_tuple);
    auto tSFAsSFA = get<2>(SFA_load_tuple);
    auto tSFApSFA = get<3>(SFA_load_tuple);
    auto tSFAcSFA = get<4>(SFA_load_tuple);

    auto scale_copy_b = get<0>(SFB_load_tuple);
    auto tSFBgSFB = get<1>(SFB_load_tuple);
    auto tSFBsSFB = get<2>(SFB_load_tuple);
    auto tSFBpSFB = get<3>(SFB_load_tuple);
    auto tSFBcSFB = get<4>(SFB_load_tuple);

    auto SFA_shape = shape(mainloop_params.layout_SFA);
    auto SFB_shape = shape(mainloop_params.layout_SFB);

    auto copy_to_tsm = [&](int k_pipe_write, int k_scale_pipe_write, int k_idx, int warp_idx, bool scale_flag) {
      if (warp_idx == 0) {
        copy(gmem_tiled_copy_A, tAgA(_,_,_,k_idx), tAsA(_,_,_,k_pipe_write));
        copy(gmem_tiled_copy_B, tBgB(_,_,_,k_idx), tBsB(_,_,_,k_pipe_write));
      }
      if constexpr (StageNumPerBlock > 1) {
        if (scale_flag == false) {
          return;
        }
      }
      int k_scale_idx = k_idx / StageNumPerBlock;
      if constexpr (IsAiuLoadSFA) {
        if (warp_idx == 2) {
          copy(gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,k_scale_idx), tSFAsSFA(_,_,_,k_scale_pipe_write));
        }
      } else {
        Tensor tSFAcSFA_compact = filter_zeros(tSFAcSFA(_,_,_,k_scale_idx));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSFApSFA); ++i) {
          tSFApSFA(i) = elem_less(get<0>(tSFAcSFA_compact(i)), get<0>(SFA_shape));
        }
        if (warp_idx > 1 && warp_idx <= 1 + warp_sfa) {
          copy_if(scale_copy_a, tSFApSFA, filter_zeros(tSFAgSFA(_,_,_,k_scale_idx)), filter_zeros(tSFAsSFA(_,_,_,k_scale_pipe_write)));
        }
      }
      if constexpr (IsAiuLoadSFB) {
        if (warp_idx == 1) {
          const int lane = thread_idx % 32;
          if (lane < ScaleNsPerTile) {
            const int global_n64 = sfb_n_origin64 + lane;
            const int kg = shape<1,1>(mainloop_params.layout_SFB);
            const bool valid = global_n64 < shape<0,1>(mainloop_params.layout_SFB) &&
                               k_scale_idx >= 0 && k_scale_idx < kg;
            // Original 128x128 scale values, mapped to the internal N64 tile.
            // A safe source address plus the native ZFILL predicate also covers
            // the final BN192 tile, which has only one valid N64 scale.
            const ElementScale* src_ptr = mainloop_params.ptr_scale_B +
                (valid ? (global_n64 / 2) * kg + k_scale_idx : 0);
            ElementScale* dst_ptr = &sSFB(lane, 0, k_scale_pipe_write);
            auto src_one = make_tensor(make_gmem_ptr(src_ptr), Layout<Shape<_1>>{});
            auto dst_one = make_tensor(make_smem_ptr(dst_ptr), Layout<Shape<_1>>{});
            copy_if(CopyAtomSFB{}, SfbCopyPredicate{valid}, src_one, dst_one);
          }
        }
      } else {
        Tensor tSFBcSFB_compact = filter_zeros(tSFBcSFB(_,_,_,k_scale_idx));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSFBpSFB); ++i) {
          tSFBpSFB(i) = elem_less(get<0>(tSFBcSFB_compact(i)), get<0>(SFB_shape));
        }
        if (warp_idx == 1) {
          copy_if(scale_copy_b, tSFBpSFB, filter_zeros(tSFBgSFB(_,_,_,k_scale_idx)), filter_zeros(tSFBsSFB(_,_,_,k_scale_pipe_write)));
        }
      }
    };

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      if (k_tile_count > 0) {
        copy_to_tsm(k_pipe,  k_pipe / StageNumPerBlock, *k_tile_iter, warp_idx, k_pipe % StageNumPerBlock == 0);
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
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,_0{}));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,_0{}));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    using EngineAccum =  typename FrgTensorC::engine_type;
    using LayoutAccum =  typename FrgTensorC::layout_type;
    MixAccumulation<EngineAccum, LayoutAccum, ElementAccumulator> accumulation(accum, ScalePromotionInterval, size<2>(tCrA));

    //
    // Copy Atom retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,PIPE)

    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // Block scaling
    auto layout_sSFA_copy = [&]() {
      if constexpr(IsAiuLoadSFA) {
        return make_layout(
          make_shape(
            make_shape(_1{}, shape<0>(RealSmemLayoutSFA{})),
            get<1>(TileShape{}),
            make_shape(
              shape<1>(SmemLayoutSFA{}),
              shape<2>(SmemLayoutSFA{}))
          ),
          make_stride(
            make_stride(_0{}, stride<0>(RealSmemLayoutSFA{})),
            _0{},
            make_stride(
              make_stride(_0{}, stride<1>(RealSmemLayoutSFA{})),
              stride<2>(RealSmemLayoutSFA{}))
          )
        );
      } else {
        return make_layout(
          make_shape(
            shape<0>(SmemLayoutSFA{}),
            get<1>(TileShape{}),
            make_shape(
              shape<1>(SmemLayoutSFA{}),
              shape<2>(SmemLayoutSFA{}))
          ),
          make_stride(
            stride<0>(SmemLayoutSFA{}),
            _0{},
            make_stride(
              stride<1>(SmemLayoutSFA{}),
              stride<2>(SmemLayoutSFA{}))
          )
        );
      }
    }();

    auto sfb_n_stride = [&]() {
      if constexpr(IsAiuLoadSFB) {
        return make_stride(_0{}, get<0>(stride(RealSmemLayoutSFB{})));
      } else {
        return get<0>(stride(RealSmemLayoutSFB{}));
      }
    }();
    auto layout_sSFB_copy = make_layout(
      make_shape(
        get<0>(TileShape{}),
        get<0>(shape(SmemLayoutSFB{})),       // n-broadcast
        make_shape(
          get<1>(shape(SmemLayoutSFB{})),     // k-broadcast
          get<2>(shape(SmemLayoutSFB{})))
      ),
      make_stride(
        _0{},
        sfb_n_stride,
        make_stride(
          get<1>(stride(SmemLayoutSFB{})),
          get<2>(stride(RealSmemLayoutSFB{})))
      )
    );

    Tensor sSFA_copy = make_tensor(cute::make_smem_ptr(storage.smem_SFA.data()), layout_sSFA_copy);
    Tensor sSFB_copy = make_tensor(cute::make_smem_ptr(storage.smem_SFB.data()), layout_sSFB_copy);

    Tensor tCsSFA = tiled_mma.get_slice(thread_idx).partition_C(sSFA_copy);
    Tensor tCsSFB = tiled_mma.get_slice(thread_idx).partition_C(sSFB_copy);
    Tensor tCrSFA = make_fragment_like<ElementScale>(tCsSFA(_, _, _, _0{}));
    Tensor tCrSFB = make_fragment_like<ElementScale>(tCsSFB(_, _, _, _0{}));

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    int pipe_stage_step = 0;

    int smem_scale_pipe_read = smem_pipe_read;
    int smem_scale_pipe_write = ScaleStages - 1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
    auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);
    auto K_BLOCK_GROUP = K_BLOCK_MAX / Int<ScaleKsPerTile>{};

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
      copy(tCsSFA(_,_,_,make_coord(_0{}, _0{})), tCrSFA);
      copy(tCsSFB(_,_,_,make_coord(_0{}, _0{})), tCrSFB);
    }
    if constexpr (WarpInterleaving) {
      if (warp_group_id == 1) {
        __ppu_barrier_arrive(5, NumThreadsPerCTA, 0);
      }
    }

    auto preprocess_scale = [&]() {
      if constexpr (WarpInterleaving) {
        __ppu_barrier_sync(5 + warp_group_id, NumThreadsPerCTA); // group 0 wait bar 5, group 1 wait bar 6
      }
      clear(accumulation());
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementScale scale_b = tCrSFB(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
          filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementScale scale_a = tCrSFA(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
          filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
        }
      }
    };

    auto process_scale_and_copy = [&](int k_pipe_read, int k_idx = _0{}) {
      if constexpr (WarpInterleaving) {
        __ppu_barrier_arrive(6 - warp_group_id, NumThreadsPerCTA, 0);
      }
      // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementScale scale_ab = tCrSFA(_0{});
        accumulation.scale(scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        accumulation.scale(tCrSFA);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        accumulation.scale(tCrSFB);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        accumulation.scale(tCrSFA, tCrSFB);
      }

      copy(tCsSFB(_,_,_,make_coord(k_idx, k_pipe_read)), tCrSFB);
      copy(tCsSFA(_,_,_,make_coord(k_idx, k_pipe_read)), tCrSFA);
    };
    constexpr int ScaleLoadStep = int(DispatchPolicy::Stages % 2 == 0);

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -(DispatchPolicy::Stages-1)) {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        if (k_block == K_BLOCK_MAX - 1) {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          if constexpr (WarpInterleaving) {
            __ppu_barrier_sync(3 + warp_group_id, NumThreadsPerCTA / 2);   // warp group sync
          } else {
            __syncthreads();
          }
        }
        if constexpr (StageNumPerBlock > 1) {
          if ((k_block % K_BLOCK_GROUP == 0) && pipe_stage_step == 0) {
            preprocess_scale();
          }
        } else {
          if (k_block % K_BLOCK_GROUP == 0) {
            preprocess_scale();
          }
        }
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0) {
          if constexpr (StageNumPerBlock > 1) {
            copy_to_tsm(smem_pipe_write, smem_scale_pipe_write, *k_tile_iter, warp_idx, pipe_stage_step == ScaleLoadStep);
            if (pipe_stage_step == ScaleLoadStep) {
              smem_scale_pipe_write = smem_scale_pipe_read;
              smem_scale_pipe_read = (smem_scale_pipe_read + 1) % ScaleStages;
            }
          } else {
            copy_to_tsm(smem_pipe_write, smem_pipe_write, *k_tile_iter, warp_idx, true);
          }
          cp_async_fence();

          // Advance the tile
          --k_tile_count;
          if (k_tile_count > 0) { ++k_tile_iter; }

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }
        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

        CUTLASS_PRAGMA_UNROLL
        for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
          auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
          // Transform before compute
          cute::transform(tCrA(_,_,atom_idx), TransformA{});
          cute::transform(tCrB(_,_,atom_idx), TransformB{});
          // gemm for one tiled_mma atom on K
          cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), accumulation());
        }
        if constexpr (StageNumPerBlock > 1) {         // ctaK = 64
          if ((k_block_next == 0) && (pipe_stage_step == 1)) {
            process_scale_and_copy(smem_scale_pipe_read);
          }
        } else if constexpr(ScaleKsPerTile > 1) {
          if (k_block_next % K_BLOCK_GROUP == 0) {    // ctaK = 256
            process_scale_and_copy(smem_scale_pipe_read, k_block_next / K_BLOCK_GROUP * ScaleGranularityK);
            smem_scale_pipe_read = smem_pipe_read;
          }
        } else {                                      // ctaK = 128
          if (k_block_next == 0) {
            process_scale_and_copy(smem_pipe_read);
          }
        }
      }); // for_each
      if constexpr (StageNumPerBlock > 1) {
        ++pipe_stage_step;
        pipe_stage_step = (pipe_stage_step == StageNumPerBlock) ? 0 : pipe_stage_step;
      }
    }
    if constexpr (WarpInterleaving) {
      if (warp_group_id == 0) {
        __ppu_barrier_sync(5, NumThreadsPerCTA);
      }
    }

    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
