#pragma once
#include <cstdint>
#include <cute/arch/copy_ppu.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/numeric_conversion.h>

namespace deep_gemm {

// Fused accumulator + C; no-C kernels keep the original epilogue unchanged.
template <class StrideC_, class StrideD_, class ThreadEpilogueOp_,
          class EpilogueSchedule_, bool IsAlignedN = false, bool VectorAligned = false,
          bool WidePrefetch = false, int PrefetchPairs = 8, bool StageC = false, int StagedM = 256, int StagedN = 256>
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
    struct alignas(16) StagedSharedStorage { ElementC source[StagedM * StagedN]; };
    using SharedStorage = cute::conditional_t<StageC, StagedSharedStorage, typename Base::SharedStorage>;
    static constexpr bool kPreloadC = StageC;
    // Exact original types preserve the host/device Params ABI.
    using Arguments = typename Base::Arguments;
    using Params = typename Base::Params;
    static_assert(cutlass::platform::is_same<ElementC, cutlass::bfloat16_t>::value &&
                  cutlass::platform::is_same<ElementD, cutlass::bfloat16_t>::value);
    static_assert(cutlass::platform::is_same<ElementAccumulator, float>::value &&
                  cutlass::platform::is_same<ElementCompute, float>::value);
    static_assert(ThreadEpilogueOp::kCount == 2);
    static_assert(PrefetchPairs > 0, "C prefetch must contain at least one pair");
    static_assert(!StageC || (StagedM == 128 || StagedM == 256), "unsupported staged C tile height");
    static_assert(!StageC || (StagedN == 128 || StagedN == 256 || StagedN == 512),
                  "unsupported staged C tile width");
    static_assert(!StageC || (VectorAligned && WidePrefetch && PrefetchPairs == 8),
                  "staged C requires the vectorized flat-8 epilogue");

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return args;
    }
    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
    CUTLASS_HOST_DEVICE
    Bf16AddCEpilogueNoTsm(Params const& params_, SharedStorage const& = SharedStorage()) : params(params_) {}

    template <int NumThreads>
    CUTLASS_DEVICE static void preload(Params const& args, int m_coord, int n_coord, int n,
                                       int thread_idx, char* smem_buf) {
        using namespace cute;
        static_assert(StageC && (NumThreads == StagedN || NumThreads == 2 * StagedN));
        constexpr int kVectorsPerRow = StagedN / 8;
        constexpr int kRowsPerIssue = NumThreads / kVectorsPerRow;
        static_assert(kRowsPerIssue == 8 || kRowsPerIssue == 16);
        static_assert(StagedM % (4 * kRowsPerIssue) == 0);
        auto* storage = reinterpret_cast<SharedStorage*>(smem_buf);
        // The completed mainloop wait<0> + CTA barrier protects aliased AB storage.
        const int64_t row_stride = get<0>(args.dC);
        int row = thread_idx / kVectorsPerRow;
        int col = (thread_idx % kVectorsPerRow) * 8;
        const int64_t global_col = int64_t(n_coord) * StagedN + col;
        // The host requires N % 8 == 0 and 16B-aligned C/D. Therefore each
        // 16-byte vector is wholly in bounds or wholly invalid, never partial.
        const bool column_valid = IsAlignedN || global_col + 8 <= int64_t(n);
        // Keep even a zero-fill source reference within its row: the native
        // masked copy takes a C++ reference, although false predicates read 0B.
        const ElementC* input = args.ptr_C + int64_t(m_coord * StagedM + row) * row_stride +
                                (column_valid ? global_col : 0);
        auto* output = reinterpret_cast<uint4*>(storage->source) + (thread_idx ^ (row & 7));
        // Native 16-byte global-to-shared copies avoid payload registers.
        // The row swizzle repeats every eight rows; both supported CTA sizes
        // advance by a multiple of eight rows and preserve the swizzle phase.
        using AsyncCopy = cute::PPU_CP_ASYNC_CACHEALWAYS<uint4>;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int chunk = 0; chunk < StagedM / (4 * kRowsPerIssue); ++chunk) {
            if constexpr (IsAlignedN) {
                // Preserve the original unpredicated loader for aligned N.
                AsyncCopy::copy(*reinterpret_cast<uint4 const*>(input), output[0]);
                AsyncCopy::copy(*reinterpret_cast<uint4 const*>(input + kRowsPerIssue * row_stride), output[NumThreads]);
                AsyncCopy::copy(*reinterpret_cast<uint4 const*>(input + 2 * kRowsPerIssue * row_stride), output[2 * NumThreads]);
                AsyncCopy::copy(*reinterpret_cast<uint4 const*>(input + 3 * kRowsPerIssue * row_stride), output[3 * NumThreads]);
            } else {
                // Native src_size=0 copies initialize invalid shared vectors
                // without reading beyond the logical C row, including C == D.
                using MaskedCopy = cute::PPU_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>;
                MaskedCopy::copy(*reinterpret_cast<uint4 const*>(input), output[0], column_valid);
                MaskedCopy::copy(*reinterpret_cast<uint4 const*>(input + kRowsPerIssue * row_stride), output[NumThreads], column_valid);
                MaskedCopy::copy(*reinterpret_cast<uint4 const*>(input + 2 * kRowsPerIssue * row_stride), output[2 * NumThreads], column_valid);
                MaskedCopy::copy(*reinterpret_cast<uint4 const*>(input + 3 * kRowsPerIssue * row_stride), output[3 * NumThreads], column_valid);
            }
            cute::cp_async_fence();
            input += 4 * kRowsPerIssue * row_stride;
            output += 4 * NumThreads;
        }
        // The mainloop's groups were drained before entry. Complete all
        // new groups before any thread consumes another thread's shared C.
        cute::cp_async_wait<0>();
        __syncthreads();
    }

    template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL,
              class FrgEngine, class FrgLayout, class TiledMma, class ResidueMNK>
    CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl,
                                  BlockShapeMNK blk_shape_MNK, BlockCoordMNKL blk_coord_mnkl,
                                  cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
                                  TiledMma tiled_mma, ResidueMNK residue_mnk,
                                  int thread_idx, [[maybe_unused]] char* smem_buf) {
        using namespace cute;
        using X = Underscore;
        auto M = get<0>(problem_shape_mnkl);
        auto N = get<1>(problem_shape_mnkl);
        auto L = get<3>(problem_shape_mnkl);
        auto stride_c = cutlass::epilogue::collective::detail::get_epilogue_stride<EpilogueSchedule>(params.dC);
        auto stride_d = cutlass::epilogue::collective::detail::get_epilogue_stride<EpilogueSchedule>(params.dD);
        auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
        Tensor gC = [&]() {
            if constexpr (StageC) {
                CUTE_STATIC_ASSERT_V(size<0>(blk_shape_MNK) == Int<StagedM>{});
                CUTE_STATIC_ASSERT_V(size<1>(blk_shape_MNK) == Int<StagedN>{});
                auto* storage = reinterpret_cast<SharedStorage*>(smem_buf);
                // A row advances log2(StagedN) BF16 elements. Match the
                // preload's vector-column XOR with the row's low three bits.
                constexpr int kRowSwizzleShift = StagedN == 512 ? 6 : (StagedN == 256 ? 5 : 4);
                auto layout = composition(Swizzle<3,3,kRowSwizzleShift>{}, make_layout(
                    make_shape(Int<StagedM>{}, Int<StagedN>{}), make_stride(Int<StagedN>{}, Int<1>{})));
                return make_tensor(make_smem_ptr(storage->source), layout);
            } else {
                Tensor mC = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M, N, L), stride_c);
                Tensor tiledC = local_tile(mC, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});
                return tiledC(_, _, m_coord, n_coord, l_coord);
            }
        }();
        Tensor mD = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M, N, L), stride_d);
        Tensor tiledD = local_tile(mD, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});
        Tensor gD = tiledD(_, _, m_coord, n_coord, l_coord);
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCgC = thr_mma.partition_C(gC);
        Tensor tCgD = thr_mma.partition_C(gD);
        Tensor coords = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
        Tensor tCcD = thr_mma.partition_C(coords);
        auto bounds = make_coord(get<0>(residue_mnk), get<1>(residue_mnk));
        CUTE_STATIC_ASSERT_V(size(tCgC) == size(tCgD));
        CUTE_STATIC_ASSERT_V(size(tCgD) == size(accumulators));

#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ != 100
        if constexpr (VectorAligned && WidePrefetch) {
            // Same PPU1.5 fragment layout and 32-bit store width as the old path.
            CUTE_STATIC_ASSERT_V(size<0>(shape<0>(tCgD)) == Int<2>{});
            constexpr int kMmaM = decltype(size<1>(shape<0>(tCgD)))::value;
            constexpr int kMmaN = decltype(size<2>(shape<0>(tCgD)))::value;
            constexpr int kWarpM = decltype(size<1>(tCgD))::value;
            constexpr int kWarpN = decltype(size<2>(tCgD))::value;
            constexpr int kPairsPerRow = kMmaN * kWarpN;
            constexpr int kPairs = kWarpM * kMmaM * kPairsPerRow;
            constexpr int kPrefetchPairs = PrefetchPairs;
            constexpr int kChunks = (kPairs + kPrefetchPairs - 1) / kPrefetchPairs;
            // Batch across M-fragment rows as well as N, while
            // retaining the old warp_m/mma_m/warp_n/mma_n traversal order.
            // Static indices keep accumulator fragments in registers even when
            // the compiler would otherwise leave an integer loop unrolled late.
            for_each(make_int_sequence<kChunks>{}, [&](auto chunk_index) {
                constexpr int kChunk = decltype(chunk_index)::value * kPrefetchPairs;
                cutlass::Array<uint32_t, kPrefetchPairs> source_bits;
                for_each(make_int_sequence<kPrefetchPairs>{}, [&](auto slot_index) {
                    constexpr int kSlot = decltype(slot_index)::value;
                    constexpr int kPair = kChunk + kSlot;
                    if constexpr (kPair < kPairs) {
                        constexpr int kRow = kPair / kPairsPerRow;
                        constexpr int kWarpMIndex = kRow / kMmaM;
                        constexpr int kMmaMIndex = kRow % kMmaM;
                        constexpr int kWarpNIndex = (kPair / kMmaN) % kWarpN;
                        constexpr int kMmaNIndex = kPair % kMmaN;
                        auto row_first = make_coord(Int<0>{}, Int<kMmaMIndex>{}, Int<0>{});
                        auto second = make_coord(Int<1>{}, Int<kMmaMIndex>{}, Int<kMmaNIndex>{});
                        const bool row_valid = !IsAlignedN || elem_less(
                            tCcD(row_first, Int<kWarpMIndex>{}, Int<0>{}), bounds);
                        if (row_valid && (IsAlignedN || elem_less(
                                tCcD(second, Int<kWarpMIndex>{}, Int<kWarpNIndex>{}), bounds))) {
                            auto source = tCgC(make_coord(_, Int<kMmaMIndex>{}, Int<kMmaNIndex>{}),
                                              Int<kWarpMIndex>{}, Int<kWarpNIndex>{}).data().get();
                            source_bits[kSlot] = *reinterpret_cast<uint32_t const*>(source);
                        }
                    }
                });
                // Every valid pair in this batch is loaded before any write,
                // including when C and D are the same tensor.
                for_each(make_int_sequence<kPrefetchPairs>{}, [&](auto slot_index) {
                    constexpr int kSlot = decltype(slot_index)::value;
                    constexpr int kPair = kChunk + kSlot;
                    if constexpr (kPair < kPairs) {
                        constexpr int kRow = kPair / kPairsPerRow;
                        constexpr int kWarpMIndex = kRow / kMmaM;
                        constexpr int kMmaMIndex = kRow % kMmaM;
                        constexpr int kWarpNIndex = (kPair / kMmaN) % kWarpN;
                        constexpr int kMmaNIndex = kPair % kMmaN;
                        auto row_first = make_coord(Int<0>{}, Int<kMmaMIndex>{}, Int<0>{});
                        auto first = make_coord(Int<0>{}, Int<kMmaMIndex>{}, Int<kMmaNIndex>{});
                        auto second = make_coord(Int<1>{}, Int<kMmaMIndex>{}, Int<kMmaNIndex>{});
                        const bool row_valid = !IsAlignedN || elem_less(
                            tCcD(row_first, Int<kWarpMIndex>{}, Int<0>{}), bounds);
                        if (row_valid && (IsAlignedN || elem_less(
                                tCcD(first, Int<kWarpMIndex>{}, Int<kWarpNIndex>{}), bounds))) {
                            if (IsAlignedN || elem_less(
                                    tCcD(second, Int<kWarpMIndex>{}, Int<kWarpNIndex>{}), bounds)) {
                                FragmentAccumulator input_array = *reinterpret_cast<const FragmentAccumulator*>(
                                    accumulators(make_coord(_, Int<kMmaMIndex>{}, Int<kMmaNIndex>{}),
                                                 Int<kWarpMIndex>{}, Int<kWarpNIndex>{}).data());
                                auto destination = tCgD(make_coord(_, Int<kMmaMIndex>{}, Int<kMmaNIndex>{}),
                                                        Int<kWarpMIndex>{}, Int<kWarpNIndex>{}).data().get();
                                store_pair(source_bits[kSlot], destination, input_array[0], input_array[1]);
                            } else {
                                tCgD(first, Int<kWarpMIndex>{}, Int<kWarpNIndex>{}) = add_scalar(
                                    accumulators(first, Int<kWarpMIndex>{}, Int<kWarpNIndex>{}),
                                    tCgC(first, Int<kWarpMIndex>{}, Int<kWarpNIndex>{}));
                            }
                        }
                    }
                });
            });
        } else if constexpr (VectorAligned) {
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
                                            store_pair(source_bits[slot], destination, input_array[0], input_array[1]);
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
