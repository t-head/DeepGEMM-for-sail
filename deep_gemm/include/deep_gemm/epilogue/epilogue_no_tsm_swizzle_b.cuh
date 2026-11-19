#pragma once

// Included from the tail of the kernel header (e.g. fp4_gemm_cutlass3.cuh).
// Requires: m16n16 MMA atom, bf16 output, fp32 accumulator. Type-agnostic (FP4/INT8/BF16/FP8).

namespace cutlass::epilogue::collective {

using namespace cute;

// Epilogue for offline B-swizzled GEMM. B rows (and scales) are permuted per 16-col MMA atom
// so that each lane's fragment positions hold contiguous output columns. This epilogue writes
// 4 bf16 (b64) per lane per atom instead of the base's 2x b32, halving store requests.
// Output remains in natural column order; the permutation is fully absorbed by store addressing.
template <class BaseEpilogue_>
struct CollectiveEpilogueNoTsmBSwizzled {
  using Base = BaseEpilogue_;
  using Params = typename Base::Params;
  using Arguments = typename Base::Arguments;
  using SharedStorage = typename Base::SharedStorage;
  using TensorStorage = typename Base::TensorStorage;
  using DispatchPolicy = typename Base::DispatchPolicy;
  using EpilogueSchedule = typename Base::EpilogueSchedule;
  using ThreadEpilogueOp = typename Base::ThreadEpilogueOp;
  using ElementOutput = typename Base::ElementOutput;
  using FragmentAccumulator = typename Base::FragmentAccumulator;
  using ElementAccumulator = typename Base::ElementAccumulator;
  using ElementCompute = typename Base::ElementCompute;
  using ElementScalar = typename Base::ElementScalar;
  using ElementC = typename Base::ElementC;
  using StrideC = typename Base::StrideC;
  using ElementD = typename Base::ElementD;
  using StrideD = typename Base::StrideD;
  using GmemTiledCopyC = typename Base::GmemTiledCopyC;
  using GmemTiledCopyD = typename Base::GmemTiledCopyD;

  static constexpr bool kIsAlignedN = Base::kIsAlignedN;
  static constexpr size_t SharedStorageSize = 0;

  CUTLASS_HOST_DEVICE
  CollectiveEpilogueNoTsmBSwizzled(Params const& params, SharedStorage const& = SharedStorage())
      : params_(params), epilogue_op_(params.thread) {}

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      [[maybe_unused]] char* smem_buf)
  {
    using X = Underscore;

    static_assert(cute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(cute::rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(cute::rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 4");

    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    auto stride_d = detail::get_epilogue_stride<DispatchPolicy>(params_.dD);
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params_.ptr_D), make_shape(M, N, L), stride_d);
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gD = gD_mnl(_, _, m_coord, n_coord, l_coord);

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    auto cD = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
    Tensor tCcD = thr_mma.partition_C(cD);
    static_assert(size<2>(shape<0>(decltype(tCcD){})) == 2,
                  "B-swizzle epilogue requires 16-col MMA atoms (MMA_N=2, i.e. m16n16)");

    const int lane_in_quad = thread_idx % 4;        // 4 lanes in a mma are in one row.
    const int quad_col_offset = 4 * lane_in_quad;   // each lane store 4 b16 sequentially
    const int valid_m = get<0>(residue_mnk);
    const int valid_n = get<1>(residue_mnk);

    // Base PPU0015 loop structure: ITER_M -> MMA_M -> [M predicate] -> ITER_N.
    // Pack both mma_n halves (h=0, h=1) into one b64 store at the permuted column address.
    // Address strategy: compute row base pointer ONCE per row (first atom's permuted position),
    // then store subsequent atoms at compile-time constant byte offsets (mirrors base's immediate offsets).
    CUTLASS_PRAGMA_UNROLL
    for (int warp_m = 0; warp_m < size<1>(tCcD); ++warp_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < size<1>(shape<0>(tCcD)); ++mma_m) {
        if constexpr (kIsAlignedN) {
          auto condition = elem_less(tCcD(make_coord(Int<0>{}, mma_m, Int<0>{}), warp_m, Int<0>{}),
                                     make_coord(valid_m, valid_n));
          if (!condition) continue;
        }
        const int row = get<0>(tCcD(make_coord(Int<0>{}, mma_m, Int<0>{}), warp_m, Int<0>{}));
        const int kc0 = get<1>(tCcD(make_coord(Int<0>{}, mma_m, Int<0>{}), warp_m, Int<0>{}));
        const int kc0_atom = kc0 & ~15;   // (kc0 / 16) * 16
        char* store_base = reinterpret_cast<char*>(&gD(row, kc0_atom + quad_col_offset));

        CUTLASS_PRAGMA_UNROLL
        for (int warp_n = 0; warp_n < size<2>(tCcD); ++warp_n) {
          if constexpr (!kIsAlignedN) {
            auto condition = elem_less(tCcD(make_coord(Int<0>{}, mma_m, Int<0>{}), warp_m, warp_n),
                                       make_coord(valid_m, valid_n));
            if (!condition) continue;
          }
          const int kc = get<1>(tCcD(make_coord(Int<0>{}, mma_m, Int<0>{}), warp_m, warp_n));
          const int byte_offset = ((kc & ~15) - kc0_atom) * int(sizeof(ElementD));

          FragmentAccumulator f0 = *reinterpret_cast<const FragmentAccumulator*>(
              accumulators(make_coord(_, mma_m, Int<0>{}), warp_m, warp_n).data());
          FragmentAccumulator f1 = *reinterpret_cast<const FragmentAccumulator*>(
              accumulators(make_coord(_, mma_m, Int<1>{}), warp_m, warp_n).data());
          typename Base::VectorType converted[2] = {epilogue_op_(f0), epilogue_op_(f1)};

          *reinterpret_cast<uint2*>(store_base + byte_offset) = *reinterpret_cast<uint2*>(converted);
        }
      }
    }
  }

private:
  Params params_;
  ThreadEpilogueOp epilogue_op_;
};

} // namespace cutlass::epilogue::collective
