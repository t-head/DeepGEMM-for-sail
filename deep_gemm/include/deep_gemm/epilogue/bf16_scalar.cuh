#pragma once

#include <cutlass/epilogue/collective/default_epilogue.hpp>

namespace deep_gemm {

// The no-C epilogue for odd N predicates each BF16 store separately.
template <class Base>
struct Bf16ScalarEpilogueNoTsm : Base {
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;

    CUTLASS_HOST_DEVICE Bf16ScalarEpilogueNoTsm(Params const& p, SharedStorage const& s)
        : Base(p, s), params(p) {}

    template <class Problem, class Tile, class Coord, class Fragment, class Mma, class Residue>
    CUTLASS_DEVICE void operator()(Problem problem, Tile tile, Coord coord, Fragment const& accum,
                                   Mma mma, Residue residue, int tid, char*) {
        using namespace cute;
        auto thread = mma.get_thread_slice(tid);
        auto matrix = make_tensor(make_gmem_ptr(params.ptr_D),
                                  make_shape(get<0>(problem), get<1>(problem), get<3>(problem)), params.dD);
        auto tiles = local_tile(matrix, tile, make_coord(_, _, _), Step<_1, _1, Underscore>{});
        auto dst = thread.partition_C(tiles(_, _, get<0>(coord), get<1>(coord), get<3>(coord)));
        auto coords = thread.partition_C(make_identity_tensor(take<0, 2>(tile)));
        auto bounds = make_coord(get<0>(residue), get<1>(residue));
        typename Base::ThreadEpilogueOp epilogue_op(params.thread);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
            if (elem_less(coords(i), bounds))
                dst(i) = epilogue_op(accum(i));
        }
    }

private:
    Params const& params;
};

} // namespace deep_gemm
