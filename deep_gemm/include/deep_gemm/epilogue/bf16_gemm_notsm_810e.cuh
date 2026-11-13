#pragma once

#include <cstdint>
#include <cute/tensor.hpp>
#include <deep_gemm/cute_free/epilogue/epilogue_traits.cuh>

namespace deep_gemm {

// Store TC01 accumulators directly from registers, preserving the EVT Params ABI.
// This epilogue implements alpha=1, beta=0. The public API handles C accumulation
// in FP32 before the final output conversion.
template <typename BaseEpilogue, bool UsePackedTailStores, bool IsFullTileM,
          bool IsFullTileN, bool IsAlignedOutput>
class BF16GemmNoTsmEpilogue810E {
public:
    using Params = typename BaseEpilogue::Params;
    using Arguments = typename BaseEpilogue::Arguments;
    using ElementC = typename BaseEpilogue::ElementC;
    using StrideC = typename BaseEpilogue::StrideC;
    using ElementD = typename BaseEpilogue::ElementD;
    using StrideD = typename BaseEpilogue::StrideD;
    using ElementCompute = typename BaseEpilogue::ElementCompute;
    struct SharedStorage {};

    CUTLASS_DEVICE
    BF16GemmNoTsmEpilogue810E(Params const& params, SharedStorage const&) : params_(params) {}

    template <typename ProblemShape>
    static constexpr Params to_underlying_arguments(
        ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
        return BaseEpilogue::to_underlying_arguments(problem_shape, args, workspace);
    }

    template <typename ProblemShape>
    static constexpr size_t get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <typename ProblemShape, typename TileShape, typename TileCoord,
              typename AccumulatorEngine, typename AccumulatorLayout, typename TiledMma, typename ResidueMN>
    CUTLASS_DEVICE void operator()(
        ProblemShape problem_shape, TileShape tile_shape, TileCoord tile_coord,
        cute::Tensor<AccumulatorEngine, AccumulatorLayout> const& accumulators,
        TiledMma tiled_mma, ResidueMN, int thread_idx, char*) {
        if constexpr (UsePackedTailStores) {
            store_fragment<true, IsFullTileM, IsFullTileN>(
                problem_shape, tile_shape, tile_coord, accumulators, tiled_mma, thread_idx);
        } else {
            // One uniform decision per CTA removes bounds checks from complete
            // tiles. Scalar stores avoid shuffles on the remaining edge tiles.
            const bool full_m = IsFullTileM ||
                (cute::get<0>(tile_coord) + 1) * cute::size<0>(tile_shape) <= cute::get<0>(problem_shape);
            const bool full_n = IsFullTileN ||
                (cute::get<1>(tile_coord) + 1) * cute::size<1>(tile_shape) <= cute::get<1>(problem_shape);
            if (full_m && full_n) {
                store_fragment<true, true, true>(
                    problem_shape, tile_shape, tile_coord, accumulators, tiled_mma, thread_idx);
            } else if (full_n) {
                store_fragment<false, IsFullTileM, true>(
                    problem_shape, tile_shape, tile_coord, accumulators, tiled_mma, thread_idx);
            } else if (full_m) {
                store_fragment<false, true, IsFullTileN>(
                    problem_shape, tile_shape, tile_coord, accumulators, tiled_mma, thread_idx);
            } else {
                store_fragment<false, IsFullTileM, IsFullTileN>(
                    problem_shape, tile_shape, tile_coord, accumulators, tiled_mma, thread_idx);
            }
        }
    }

private:
    using OutputTraits = epilogue::EpilogueElementTraits<ElementD>;
    Params params_;

    // TC01 gives lane 0 columns {0,4,8,12}, and lane 1 {1,5,9,13}.
    // Exchange FP32 values before conversion to form adjacent output pairs.
    CUTLASS_DEVICE static void shuffle_adjacent_columns(
        float first, float second, float& lower, float& upper, int lane_idx) {
        const bool odd_lane = lane_idx & 1;
        const float peer = __shfl_xor_sync(0xffffffffu, odd_lane ? first : second, 1);
        lower = odd_lane ? peer : first;
        upper = odd_lane ? second : peer;
    }

    template <bool FullN>
    CUTLASS_DEVICE static void store_pair(ElementD* output, float lower, float upper, int col, int n) {
        if ((FullN || col + 1 < n) &&
            (IsAlignedOutput || reinterpret_cast<uintptr_t>(output) % (2 * sizeof(ElementD)) == 0)) {
            *reinterpret_cast<typename OutputTraits::PackedType*>(output) = OutputTraits::cvt_pack(upper, lower);
        } else {
            output[0] = ElementD(lower);
            if (FullN || col + 1 < n) {
                output[1] = ElementD(upper);
            }
        }
    }

    template <bool PackedStores, bool FullM, bool FullN, typename ProblemShape, typename TileShape,
              typename TileCoord, typename AccumulatorEngine, typename AccumulatorLayout, typename TiledMma>
    CUTLASS_DEVICE void store_fragment(
        ProblemShape problem_shape, TileShape, TileCoord tile_coord,
        cute::Tensor<AccumulatorEngine, AccumulatorLayout> const& accumulators,
        TiledMma tiled_mma, int thread_idx) {
        using namespace cute;
        CUTE_STATIC_ASSERT_V(size<0>(accumulators) == Int<8>{});
        constexpr int BLOCK_M = size<0>(TileShape{}), BLOCK_N = size<1>(TileShape{});
        const int m = get<0>(problem_shape), n = get<1>(problem_shape), lane_idx = thread_idx & 31;
        auto accumulator_coords = tiled_mma.get_slice(thread_idx).partition_C(
            make_identity_tensor(make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{})));
        auto* batch_output = params_.ptr_D + int64_t(get<3>(tile_coord)) * get<2>(params_.dD);
        CUTE_UNROLL
        for (int mma_m = 0; mma_m < size<1>(accumulators); ++mma_m) {
            CUTE_UNROLL
            for (int mma_n = 0; mma_n < size<2>(accumulators); ++mma_n) {
                CUTE_UNROLL
                for (int row_half = 0; row_half < 2; ++row_half) {
                    const auto coord = accumulator_coords(row_half * 4, mma_m, mma_n);
                    const int row = get<0>(tile_coord) * BLOCK_M + get<0>(coord);
                    const int first_col = get<1>(tile_coord) * BLOCK_N + get<1>(coord);
                    auto* row_output = batch_output + int64_t(row) * get<0>(params_.dD);
                    if constexpr (!PackedStores) {
                        CUTE_UNROLL
                        for (int element = 0; element < 4; ++element) {
                            const int col = first_col + element * 4;
                            if ((FullM || row < m) && (FullN || col < n)) {
                                row_output[col] = ElementD(accumulators(row_half * 4 + element, mma_m, mma_n));
                            }
                        }
                    } else {
                        CUTE_UNROLL
                        for (int pair = 0; pair < 2; ++pair) {
                            float lower, upper;
                            shuffle_adjacent_columns(
                                accumulators(row_half * 4 + pair * 2, mma_m, mma_n),
                                accumulators(row_half * 4 + pair * 2 + 1, mma_m, mma_n), lower, upper, lane_idx);
                            const int col = first_col + pair * 8 + (lane_idx & 1) * 3;
                            // Test bounds after the shuffle so every lane participates.
                            if ((FullM || row < m) && (FullN || col < n)) {
                                store_pair<FullN>(row_output + col, lower, upper, col, n);
                            }
                        }
                    }
                }
            }
        }
    }
};

} // namespace deep_gemm
