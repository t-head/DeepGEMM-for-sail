#pragma once
#include <cstdint>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/numeric_conversion.h>

namespace deep_gemm {

// Fused accumulator + C; no-C kernels keep the original epilogue unchanged.
template <class StrideC_, class StrideD_, class ThreadEpilogueOp_,
          class EpilogueSchedule_, bool IsAlignedN = false, bool VectorAligned = false,
          bool RegisterC = false>
class Bf16AddCEpilogueNoTsm {
    using Base = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        StrideC_, StrideD_, ThreadEpilogueOp_, EpilogueSchedule_, IsAlignedN>;
public:
    using EpilogueSchedule = EpilogueSchedule_;
    using ThreadEpilogueOp = ThreadEpilogueOp_;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using ElementCompute = typename Base::ElementCompute;
    using ElementC = typename Base::ElementC;
    using ElementD = typename Base::ElementD;
    using StrideC = StrideC_;
    using StrideD = StrideD_;
    using FragmentAccumulator = typename Base::FragmentAccumulator;
    static constexpr bool kRegisterC = RegisterC;
    using SharedStorage = typename Base::SharedStorage;
    static_assert(!RegisterC || VectorAligned, "register C requires aligned BF16 pairs");
    // Exact original types preserve the host/device Params ABI.
    using Arguments = typename Base::Arguments;
    using Params = typename Base::Params;
    static_assert(cutlass::platform::is_same<ElementC, cutlass::bfloat16_t>::value &&
                  cutlass::platform::is_same<ElementD, cutlass::bfloat16_t>::value);
    static_assert(cutlass::platform::is_same<ElementAccumulator, float>::value &&
                  cutlass::platform::is_same<ElementCompute, float>::value);
    static_assert(ThreadEpilogueOp::kCount == 2);

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return args;
    }
    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
    CUTLASS_HOST_DEVICE
    Bf16AddCEpilogueNoTsm(Params const& params_, SharedStorage const& = SharedStorage()) : params(params_) {}

    // Each lane loads its own packed output pairs while the last A/B tiles drain.
    template <class Problem, class Block, class Coord, class TiledMma, class Residue, int Pairs>
    CUTLASS_DEVICE static void load_register_c(Params const& args, Problem problem, Block block, Coord coord,
                                               TiledMma tiled_mma, Residue residue, int thread_idx,
                                               cutlass::Array<uint32_t, Pairs>& packed) {
        using namespace cute;
        auto thr = tiled_mma.get_thread_slice(thread_idx);
        auto tC = thr.partition_C(output_tile(args.ptr_C, args.dC, problem, block, coord));
        auto coords = thr.partition_C(make_identity_tensor(take<0, 2>(block)));
        auto bounds = make_coord(get<0>(residue), get<1>(residue));
        CUTE_STATIC_ASSERT_V(size<0>(shape<0>(tC)) == Int<2>{});
        CUTE_STATIC_ASSERT_V(size(tC) == Int<2 * Pairs>{});
        CUTLASS_PRAGMA_UNROLL
        for (int warp_m = 0; warp_m < size<1>(tC); ++warp_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < size<1>(shape<0>(tC)); ++mma_m) {
                const bool row_valid = !IsAlignedN || elem_less(
                    coords(make_coord(Int<0>{}, mma_m, Int<0>{}), warp_m, Int<0>{}), bounds);
                CUTLASS_PRAGMA_UNROLL
                for (int warp_n = 0; warp_n < size<2>(tC); ++warp_n) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mma_n = 0; mma_n < size<2>(shape<0>(tC)); ++mma_n) {
                        // Match the row/pair order consumed by the epilogue.
                        const int pair = ((warp_m * size<1>(shape<0>(tC)) + mma_m) * size<2>(tC)
                                          + warp_n) * size<2>(shape<0>(tC)) + mma_n;
                        const bool valid = IsAlignedN ? row_valid : elem_less(
                            coords(make_coord(Int<1>{}, mma_m, mma_n), warp_m, warp_n), bounds);
                        if (valid) {
                            auto ptr = tC(make_coord(_, mma_m, mma_n), warp_m, warp_n).data().get();
                            packed[pair] = *reinterpret_cast<const uint32_t*>(ptr);
                        } else {
                            packed[pair] = 0;
                        }
                    }
                }
            }
        }
    }

    template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL,
              class FrgEngine, class FrgLayout, class TiledMma, class ResidueMNK, class Packed = decltype(nullptr)>
    CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl,
                                  BlockShapeMNK blk_shape_MNK, BlockCoordMNKL blk_coord_mnkl,
                                  cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
                                  TiledMma tiled_mma, ResidueMNK residue_mnk,
                                  int thread_idx, [[maybe_unused]] char* smem_buf, Packed const& packed = nullptr) {
        using namespace cute;
        Tensor gC = output_tile(params.ptr_C, params.dC, problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl);
        Tensor gD = output_tile(params.ptr_D, params.dD, problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl);
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCgC = thr_mma.partition_C(gC);
        Tensor tCgD = thr_mma.partition_C(gD);
        Tensor coords = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
        Tensor tCcD = thr_mma.partition_C(coords);
        auto bounds = make_coord(get<0>(residue_mnk), get<1>(residue_mnk));
        CUTE_STATIC_ASSERT_V(size(tCgC) == size(tCgD));
        CUTE_STATIC_ASSERT_V(size(tCgD) == size(accumulators));

#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ != 100
        if constexpr (VectorAligned) {
            // Same PPU1.5 fragment layout and 32-bit store width as the old path.
            CUTE_STATIC_ASSERT_V(size<0>(shape<0>(tCgD)) == Int<2>{});
            CUTLASS_PRAGMA_UNROLL
            for (int warp_m = 0; warp_m < size<1>(tCgD); ++warp_m) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_m = 0; mma_m < size<1>(shape<0>(tCgD)); ++mma_m) {
                    // With aligned N, predicate the row once, as in the original epilogue.
                    const bool row_valid = !IsAlignedN || elem_less(
                        tCcD(make_coord(Int<0>{}, mma_m, Int<0>{}), warp_m, Int<0>{}), bounds);
                    if (row_valid) {
                        constexpr int kMmaN = decltype(size<2>(shape<0>(tCgD)))::value;
                        constexpr int kWarpN = decltype(size<2>(tCgD))::value;
                        constexpr int kPairs = kMmaN * kWarpN;
                        constexpr int kPrefetchPairs = 4;
                        CUTLASS_PRAGMA_UNROLL
                        for (int chunk = 0; chunk < kPairs; chunk += kPrefetchPairs) {
                            cutlass::Array<uint32_t, kPrefetchPairs> source_bits;
                            if constexpr (!RegisterC) {
                                // Issue independent C reads before their dependent adds/stores.
                                // The bounded fragment costs only four 32-bit registers.
                                CUTLASS_PRAGMA_UNROLL
                                for (int slot = 0; slot < kPrefetchPairs; ++slot) {
                                    int pair = chunk + slot;
                                    if (pair < kPairs) {
                                        int warp_n = pair / kMmaN;
                                        int mma_n = pair % kMmaN;
                                        auto second = make_coord(Int<1>{}, mma_m, mma_n);
                                        if (IsAlignedN || elem_less(tCcD(second, warp_m, warp_n), bounds)) {
                                            auto source = tCgC(make_coord(_, mma_m, mma_n), warp_m, warp_n).data().get();
                                            source_bits[slot] = *reinterpret_cast<uint32_t const*>(source);
                                        }
                                    }
                                }
                            }
                            CUTLASS_PRAGMA_UNROLL
                            for (int slot = 0; slot < kPrefetchPairs; ++slot) {
                                int pair = chunk + slot;
                                if (pair < kPairs) {
                                    int warp_n = pair / kMmaN;
                                    int mma_n = pair % kMmaN;
                                    auto first = make_coord(Int<0>{}, mma_m, mma_n);
                                    auto second = make_coord(Int<1>{}, mma_m, mma_n);
                                    if (IsAlignedN || elem_less(tCcD(first, warp_m, warp_n), bounds)) {
                                        if (IsAlignedN || elem_less(tCcD(second, warp_m, warp_n), bounds)) {
                                            FragmentAccumulator input_array = *reinterpret_cast<const FragmentAccumulator*>(
                                                accumulators(make_coord(_, mma_m, mma_n), warp_m, warp_n).data());
                                            auto destination = tCgD(make_coord(_, mma_m, mma_n), warp_m, warp_n).data().get();
                                            if constexpr (RegisterC) {
                                                const int row = warp_m * size<1>(shape<0>(tCgC)) + mma_m;
                                                store_pair(packed[row * kPairs + pair], destination, input_array[0], input_array[1]);
                                            } else {
                                                store_pair(source_bits[slot], destination, input_array[0], input_array[1]);
                                            }
                                        } else {
                                            tCgD(first, warp_m, warp_n) = add_scalar(
                                                accumulators(first, warp_m, warp_n), tCgC(first, warp_m, warp_n));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else
#endif
        {
            // Predicated scalar accesses cover unaligned storage and odd N.
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(accumulators); ++i) {
                if (elem_less(tCcD(i), bounds))
                    tCgD(i) = add_scalar(accumulators(i), tCgC(i));
            }
        }
    }

private:
    template <class Element, class Stride, class Problem, class Block, class Coord>
    CUTLASS_DEVICE static auto output_tile(Element* ptr, Stride stride, Problem problem, Block block, Coord coord) {
        using namespace cute;
        auto [m, n, k, l] = problem;
        auto [mi, ni, ki, li] = coord;
        auto d = cutlass::epilogue::collective::detail::get_epilogue_stride<EpilogueSchedule>(stride);
        auto matrix = make_tensor(make_gmem_ptr(ptr), make_shape(m, n, l), d);
        auto tiles = local_tile(matrix, block, make_coord(_, _, _), Step<_1, _1, Underscore>{});
        return tiles(_, _, mi, ni, li);
    }

    CUTLASS_DEVICE static ElementD add_scalar(float accumulator, ElementC source) {
        cutlass::NumericConverter<float, ElementC> to_float;
        cutlass::NumericConverter<ElementD, float, cutlass::FloatRoundStyle::round_to_nearest> to_bf16;
        return to_bf16(accumulator + to_float(source));
    }

    CUTLASS_DEVICE static void store_pair(uint32_t source_bits, ElementD* destination,
                                        float accumulator0, float accumulator1) {
        using SourceFragment = cutlass::Array<ElementC, 2>;
        using ComputeFragment = cutlass::Array<float, 2>;
        // Source pairs are already loaded, including when C == D.
        auto old_values = cutlass::platform::bit_cast<SourceFragment>(source_bits);
        auto values = cutlass::NumericArrayConverter<float, ElementC, 2>{}(old_values);
        ComputeFragment sum;
        sum[0] = accumulator0 + values[0];
        sum[1] = accumulator1 + values[1];
        auto output = cutlass::NumericArrayConverter<ElementD, float, 2,
            cutlass::FloatRoundStyle::round_to_nearest>{}(sum);
        *reinterpret_cast<uint32_t*>(destination) = cutlass::platform::bit_cast<uint32_t>(output);
    }

    Params params;
};

} // namespace deep_gemm
