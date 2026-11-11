#pragma once

#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/numeric_conversion.h>

namespace deep_gemm {

// Unaligned rows, narrow tails and kernels without available staging storage
// can still add C in FP32 prior to the single rounded BF16 output conversion.
template <class Base>
struct Bf16AddCEpilogueNoTsm : Base {
    using CStage = void;
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;

    CUTLASS_HOST_DEVICE Bf16AddCEpilogueNoTsm(Params const& p, SharedStorage const& s)
        : Base(p, s), params(p) {}

    template <class Problem, class Tile, class Coord, class Fragment, class Mma, class Residue>
    CUTLASS_DEVICE void operator()(Problem problem, Tile tile, Coord coord, Fragment const& accum,
                                   Mma mma, Residue residue, int tid, char*) {
        using namespace cute;
        auto thread = mma.get_thread_slice(tid);
        auto output_tile = [&](auto ptr, auto stride) {
            auto matrix = make_tensor(make_gmem_ptr(ptr), make_shape(get<0>(problem), get<1>(problem), get<3>(problem)), stride);
            auto tiles = local_tile(matrix, tile, make_coord(_, _, _), Step<_1, _1, Underscore>{});
            return thread.partition_C(tiles(_, _, get<0>(coord), get<1>(coord), get<3>(coord)));
        };

        auto src = output_tile(params.ptr_C, params.dC);
        auto dst = output_tile(params.ptr_D, params.dD);
        auto coords = thread.partition_C(make_identity_tensor(take<0, 2>(tile)));
        auto bounds = make_coord(get<0>(residue), get<1>(residue));
        cutlass::NumericConverter<float, cutlass::bfloat16_t> to_float;
        cutlass::NumericConverter<cutlass::bfloat16_t, float, cutlass::FloatRoundStyle::round_to_nearest> to_bf16;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
            if (elem_less(coords(i), bounds))
                dst(i) = to_bf16(accum(i) + to_float(src(i)));
        }
    }

private:
    Params const& params;
};

} // namespace deep_gemm
