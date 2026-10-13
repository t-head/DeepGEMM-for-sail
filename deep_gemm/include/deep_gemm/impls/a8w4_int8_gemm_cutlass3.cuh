#pragma once

#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
#include <cutlass/gemm/collective/ppu_mma_aiu_multistage_mixed_input.hpp>
#include <cutlass/fast_numeric_conversion_for_mix_gemm.h>

namespace cutlass {

template <>
struct MixGemmNumericArrayConverter<int8_t, int4b_t, 8> {
  using result_type = Array<int8_t, 8>;
  using source_type = Array<int4b_t, 8>;

  CUTLASS_DEVICE static result_type convert(source_type const& source) {
    using v2i16 = __attribute__((__vector_size__(4))) short;
    result_type result;
    const uint32_t packed = reinterpret_cast<uint32_t const&>(source);
    auto* dst = reinterpret_cast<uint32_t*>(&result);
    // Move nibble pairs {1,0}/{3,2} and {5,4}/{7,6} into the high byte of each 16-bit lane
    const v2i16 lo_pairs = reinterpret_cast<v2i16>(__byte_perm(packed, packed, 0x1000));
    const v2i16 hi_pairs = reinterpret_cast<v2i16>(__byte_perm(packed, packed, 0x3020));
    // Sign-extend odd and even nibbles with 16x2 shifts
    const v2i16 lo_odd = lo_pairs >> 12, lo_even = (lo_pairs << 4) >> 12;
    const v2i16 hi_odd = hi_pairs >> 12, hi_even = (hi_pairs << 4) >> 12;
    // Interleave back to {n3, n2, n1, n0} and {n7, n6, n5, n4}
    dst[0] = __byte_perm(reinterpret_cast<uint32_t>(lo_even), reinterpret_cast<uint32_t>(lo_odd), 0x6240);
    dst[1] = __byte_perm(reinterpret_cast<uint32_t>(hi_even), reinterpret_cast<uint32_t>(hi_odd), 0x6240);
    return result;
  }

  CUTLASS_DEVICE result_type operator()(source_type const& source) const {
    return convert(source);
  }
};

template <>
struct MixGemmNumericArrayConverter<int8_t, int4b_t, 32> {
  using result_type = Array<int8_t, 32>;
  using source_type = Array<int4b_t, 32>;

  CUTLASS_DEVICE static result_type convert(source_type const& source) {
    result_type result;
    using VecResult = Array<int8_t, 8>;
    using VecSource = Array<int4b_t, 8>;
    auto* result_ptr = reinterpret_cast<VecResult*>(&result);
    auto const* source_ptr = reinterpret_cast<VecSource const*>(&source);
    MixGemmNumericArrayConverter<int8_t, int4b_t, 8> converter;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      const int dst = (i % 2) * 2 + i / 2;
      result_ptr[dst] = converter(source_ptr[i]);
    }
    return result;
  }

  CUTLASS_DEVICE result_type operator()(source_type const& source) const {
    return convert(source);
  }
};

} // namespace cutlass

namespace deep_gemm {

// A8W4 consumes signed INT4 weights packed along K: the low nibble stores k=2*i
// and the high nibble stores k=2*i+1. K rows use the int8-MMA permutation.
template <typename Arch, int BlockN, int BlockK>
struct A8W4PackedOperandB {
  static_assert(cute::is_same_v<Arch, cutlass::arch::PPU0010>, "A8W4 is only supported on PPU0010");
  static_assert(BlockK % 64 == 0, "A8W4 requires BLOCK_K to be a multiple of 64");

  static constexpr int BlockBytes = BlockK / 2;
  static constexpr int AiuContBytes = BlockBytes > 128 ? 128 : BlockBytes;
  static constexpr int AiuContElements = AiuContBytes * 2;
  static constexpr int InstNum = BlockK / AiuContElements;
  static constexpr int BitsPerAiu = BlockN * AiuContBytes * 8;

  using CopyInst = cute::PPU0010_AIU_LOAD<cute::C<BitsPerAiu>, cutlass::int4b_t, false>;
  using GmemTiledCopy = decltype(cute::make_tiled_copy(
      cute::Copy_Atom<CopyInst, cutlass::int4b_t>{},
      cute::Layout<cute::Shape<cute::_1, cute::_1>, cute::Stride<cute::_1, cute::_1>>{},
      cute::Layout<cute::Shape<cute::Int<BlockN>, cute::Int<AiuContElements>>>{}));

  using SmemCopyOp = cute::PPU0010_TSM_LD_SWZL<
      int8_t, BlockN, AiuContElements / 2, true, false, InstNum>;
  using SmemCopyAtom = cute::Copy_Atom<SmemCopyOp, int8_t>;
  using SmemLayoutAtom = cute::Layout<
      cute::Shape<cute::_8, cute::Int<AiuContElements>>,
      cute::Stride<cute::Int<AiuContElements>, cute::_1>>;
};

} // namespace deep_gemm

namespace cutlass::gemm {

template <int Stages_, typename Schedule_ = KernelAiuMultistage>
struct MainloopPPUAiuA8W4 {
  static constexpr int Stages = Stages_;
  using ArchTag = arch::PPU0010;
  using Schedule = Schedule_;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
};

} // namespace cutlass::gemm

namespace cutlass::gemm::collective {

template <
    int Stages, typename KernelSchedule,
    typename TileShape_, typename StrideA_, typename StrideB_,
    typename TiledMma_, typename GmemTiledCopyA_, typename SmemLayoutAtomA_, typename SmemCopyAtomA_,
    typename TransformA_, typename GmemTiledCopyB_, typename SmemLayoutAtomB_, typename SmemCopyAtomB_,
    typename TransformB_>
struct CollectiveMma<
    cutlass::arch::PPU0010, MainloopPPUAiuA8W4<Stages, KernelSchedule>, TileShape_,
    int8_t, StrideA_, cutlass::int4b_t, StrideB_, TiledMma_,
    GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_,
    GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_> {
public:
  using DispatchPolicy = MainloopPPUAiuA8W4<Stages, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = int8_t;
  using StrideA = StrideA_;
  using ElementB = cutlass::int4b_t;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using ElementScale = float;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = cutlass::arch::PPU0010;

  static_assert(Stages >= 2, "A8W4 requires at least two pipeline stages");

  static constexpr int BlockM = cute::size<0>(TileShape{});
  static constexpr int BlockN = cute::size<1>(TileShape{});
  static constexpr int BlockK = cute::size<2>(TileShape{});
  static constexpr int WarpOnM = cute::size<1>(typename TiledMma::ThrLayoutVMNK{});
  static constexpr int WarpOnN = cute::size<2>(typename TiledMma::ThrLayoutVMNK{});

private:
  // Reuse actlize's mixed-input collective only for descriptor construction, shared-memory layouts,
  // copy atoms, and load_init(). A8W4 does not call its operator(); mma_mainloop() below is custom.
  using MixedInputBase = cutlass::gemm::collective::CollectiveMma<
      ArchTag,
      cutlass::gemm::MainloopPPUAiuMixedInput<
          Stages, cute::Int<1>, cutlass::gemm::KernelAiuMultistageMixedInput>,
      cute::tuple<TileShape>,
      ElementA, StrideA,
      cute::tuple<ElementB>, StrideB,
      TiledMma,
      GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, TransformA,
      GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, TransformB>;
  using MixedInputBaseParams = typename MixedInputBase::Params;

public:
  using SharedStorage = typename MixedInputBase::SharedStorage;

  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScale const* ptr_scale_A;
    ElementScale const* ptr_scale_B;
  };
  using Params = Arguments;

private:
  MixedInputBase base_{};
  MixedInputBaseParams base_params_{};
  Params params_;
  ElementScale const* tile_scale_a_ = nullptr;
  ElementScale const* tile_scale_b_ = nullptr;

public:
  template <typename ProblemShape>
  CUTLASS_DEVICE CollectiveMma(Params params, ProblemShape) : params_(params) {
    base_params_.ptr_A = params.ptr_A;
    base_params_.dA = params.dA;
    base_params_.ptr_B = params.ptr_B;
    base_params_.dB = params.dB;
  }

  template <class ProblemShape_MNKL, class BlockCoord_MNKL>
  CUTLASS_DEVICE auto load_init(
      ProblemShape_MNKL const& problem_shape,
      BlockCoord_MNKL const& block_coord,
      Params const&,
      int m, int offset_m, int64_t offset_b,
      ElementA const* ptr_a, ElementB const* ptr_b) {
    auto local_params = base_params_;
    local_params.ptr_A = ptr_a;
    local_params.ptr_B = ptr_b;
    auto inputs = base_.load_init(problem_shape, block_coord, local_params);

    auto m_coord = cute::get<0>(block_coord);
    auto n_coord = cute::get<1>(block_coord);
    tile_scale_a_ = params_.ptr_scale_A + offset_m + m_coord * BlockM;
    tile_scale_b_ = params_.ptr_scale_B + offset_b / cute::get<2>(problem_shape) + n_coord * BlockN;
    return inputs;
  }

  template <class... Ts, class FrgTensorD, class FrgTensorC, class KTileIterator, class ResidueMNK>
  CUTLASS_DEVICE void operator()(
      FrgTensorD& accum, cute::tuple<Ts...> const& inputs, FrgTensorC const&,
      KTileIterator k_tile_iter, int k_tile_count, ResidueMNK residue,
      int thread_idx, int, char* smem_buf) {
    auto mma_acc = cute::make_fragment_like<ElementAccumulator>(accum);

    // Each thread owns 2 rows x 4 cols per 16x16 accumulator atom (value i: row 8 * (i / 4), col 4 * (i % 4)
    // relative to its first element); atoms repeat every WarpOnM * 16 rows and WarpOnN * 16 cols.
    // Load these scales once before the mainloop so their latency overlaps the MMAs.
    static_assert(cute::size<0>(FrgTensorD{}) == 8, "Unexpected PPU0010 int8 accumulator fragment");
    constexpr int kMmaM = cute::size<1>(FrgTensorD{});
    constexpr int kMmaN = cute::size<2>(FrgTensorD{});
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    auto coords = thread_mma.partition_C(
        cute::make_identity_tensor(cute::make_shape(cute::Int<BlockM>{}, cute::Int<BlockN>{})));
    const int row0 = cute::get<0>(coords(0, 0, 0));
    const int col0 = cute::get<1>(coords(0, 0, 0));
    // A single base pointer per operand keeps every scale load an immediate-offset access.
    ElementScale const* scale_a_ptr = tile_scale_a_ + row0;
    ElementScale const* scale_b_ptr = tile_scale_b_ + col0;
    // The N residue is static when N is a multiple of BLOCK_N, so the column predicate folds away.
    constexpr bool kAlignedN = cute::is_static_v<cute::remove_cvref_t<decltype(cute::get<1>(residue))>>;
    float scale_a[kMmaM][2], scale_b[kMmaN][4];
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < kMmaM; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        const int offset = m * WarpOnM * 16 + r * 8;
        scale_a[m][r] = row0 + offset < cute::get<0>(residue) ? __ldg(scale_a_ptr + offset) : 0.0f;
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kMmaN; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < 4; ++c) {
        const int offset = n * WarpOnN * 16 + c * 4;
        scale_b[n][c] = (kAlignedN || col0 + offset < cute::get<1>(residue)) ? __ldg(scale_b_ptr + offset) : 0.0f;
      }
    }

    mma_mainloop(mma_acc, cute::get<0>(inputs), cute::get<1>(inputs), k_tile_iter, k_tile_count, thread_idx, smem_buf);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kMmaN; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < kMmaM; ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
          accum(i, m, n) = mma_acc(i, m, n) * scale_a[m][i / 4] * scale_b[n][i % 4];
        }
      }
    }
  }

private:
  // Same partitioning as the actlize mixed-input mainloop, but every stage is filled up front and the
  // tail never reloads a k-tile, so short-K decode tiles keep Stages loads in flight.
  template <class FrgTensorAcc, class TensorA, class TensorB, class KTileIterator>
  CUTLASS_DEVICE void mma_mainloop(
      FrgTensorAcc& accum, TensorA const& gA, TensorB const& gB,
      KTileIterator k_tile_iter, int k_tile_count, int thread_idx, char* smem_buf) {
    using namespace cute;
    clear(accum);
    using SmemLayoutA = typename MixedInputBase::SmemLayoutA;
    using SmemLayoutB = typename MixedInputBase::SmemLayoutB;
    constexpr int kStages = Stages;

    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto k_iter_shape = shape<2>(gB);
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.begin()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.begin()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

    auto& gmem_tiled_copy_A = base_.gmem_tiled_copy_A;
    auto& gmem_tiled_copy_B = base_.gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);  // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);  // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);  // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);  // (BCPY,BCPY_N,BCPY_K,PIPE)

    const int num_k_tiles = k_tile_count;
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < kStages; ++k_pipe) {
      if (k_tile_count > 0) {
        copy_aiu(gmem_tiled_copy_A, tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, k_pipe),
                 gmem_tiled_copy_B, tBgB(_, _, _, idx2crd(*k_tile_iter, k_iter_shape)), tBsB(_, _, _, k_pipe),
                 warp_idx);
        ++k_tile_iter;
      }
      cp_async_fence();
      --k_tile_count;
    }

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));      // (MMA,MMA_M,MMA_K)
    Tensor tCrB_mma = thr_mma.partition_fragment_B(sB(_, _, 0));  // (MMA,MMA_N,MMA_K)

    // Recasting packed INT4 smem as int8 halves its logical K extent, so the load partition uses a
    // K32 value tile for the same physical K64 data covered by the compute TiledMma. This helper only
    // defines the register/load layout; all MMA computation still uses the template TiledMma.
    static_assert(size<2>(typename TiledMma::AtomShape_MNK{}) == 32, "A8W4 requires a K32 MMA atom");
    using LoadTiledMma = TiledMMA<typename TiledMma::Atom,
                                  Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,
                                  Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _32>>;
    LoadTiledMma load_tiled_mma;
    auto load_thr_mma = load_tiled_mma.get_thread_slice(thread_idx);

    auto smem_tiled_copy_A = make_tiled_copy_A(typename MixedInputBase::SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
    Tensor tCsA = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);                // (CPY,CPY_M,CPY_K)

    auto sB_s8 = recast<int8_t>(sB);
    Tensor tCrB_load = load_thr_mma.partition_fragment_B(sB_s8(_, _, 0));
    auto smem_tiled_copy_B = make_tiled_copy_B(typename MixedInputBase::SmemCopyAtomB{}, load_tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(warp_idx * 32);
    Tensor tCsB = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB_s8));  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB_load);             // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));

    constexpr int K_BLOCK_MAX = size<2>(decltype(tCrB_copy_view){});
    constexpr int K_ATOM_PER_COPY = size<2>(decltype(tCrB_mma){}) / K_BLOCK_MAX;
    using Converter = cutlass::MixGemmNumericArrayConverter<int8_t, ElementB, 32>;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
      // Tile k_tile is commit group k_tile; one refill group is committed per iteration after the first.
      if (k_tile == 0) {
        cp_async_wait<kStages - 1>();
      } else {
        cp_async_wait<kStages - 2>();
      }
      __syncthreads();

      // Every warp has finished reading the previous stage, so refill it with the next pending k-tile.
      const int smem_pipe_read = k_tile % kStages;
      if (k_tile > 0) {
        const int smem_pipe_write = (k_tile - 1) % kStages;
        if (k_tile_count > 0) {
          copy_aiu(gmem_tiled_copy_A, tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, smem_pipe_write),
                   gmem_tiled_copy_B, tBgB(_, _, _, idx2crd(*k_tile_iter, k_iter_shape)), tBsB(_, _, _, smem_pipe_write),
                   warp_idx);
          ++k_tile_iter;
        }
        cp_async_fence();
        --k_tile_count;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, smem_pipe_read), tCrB_copy_view(_, _, k_block));
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, smem_pipe_read), tCrA_copy_view(_, _, k_block));

        Tensor cvt_in = recast<ElementB>(tCrB_load(_, _, k_block));
        Tensor cvt_out = make_tensor(tCrB_mma(_, _, k_block * K_ATOM_PER_COPY).data(), cvt_in.layout());
        static_assert(size(decltype(cvt_in.layout()){}) % 32 == 0, "Unexpected packed B fragment size");
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(decltype(cvt_in.layout()){}) / 32; ++i) {
          auto const* src = reinterpret_cast<typename Converter::source_type const*>(raw_pointer_cast(cvt_in(_, i).data()));
          auto* dst = reinterpret_cast<typename Converter::result_type*>(raw_pointer_cast(cvt_out(_, i).data()));
          *dst = Converter::convert(*src);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int k_atom = 0; k_atom < K_ATOM_PER_COPY; ++k_atom) {
          const int atom_idx = k_block * K_ATOM_PER_COPY + k_atom;
          cute::gemm(tiled_mma, accum,
                     tCrA(_, _, atom_idx), tCrB_mma(_, _, atom_idx), accum);
        }
      }
    }

    cp_async_wait<0>();
    __syncthreads();
  }

public:
  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
};

} // namespace cutlass::gemm::collective
