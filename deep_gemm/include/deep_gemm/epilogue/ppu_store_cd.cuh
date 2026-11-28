#pragma once

#include <cooperative_groups.h>
#include <deep_gemm/common/epilogue_args.cuh>
#include <deep_gemm/tix/ld_st.cuh>
#include <deep_gemm/epilogue/activation_quant_ops.cuh>
#include <cute/tensor.hpp>
#include <deep_gemm/epilogue/operators.cuh>

namespace deep_gemm {

// Blockwise BMM quantizes independent 32-column groups with packed UE8M0 scales.
template <class BaseEpilogue, int BlockM, int BlockN,
          class EpilogueOperator = epilogue::operators::CastToFP8WithClamp<>>
class PPUStoreCD : public BaseEpilogue {
public:
    static_assert(BlockN % 32 == 0);
    using ElementD = cutlass::float_e4m3_t;
    using Arguments = PPUStoreCDArguments<typename BaseEpilogue::ThreadEpilogueOp::Params>;
    using Params = Arguments;
    struct SharedStorage { cutlass::bfloat16_t values[BlockM * BlockN]; };
    using TensorStorage = SharedStorage;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(const ProblemShape&, const Arguments& args, void*) {
        return args;
    }
    template <class ProblemShape>
    static size_t get_workspace_size(const ProblemShape&, const Arguments&) { return 0; }

    CUTLASS_DEVICE PPUStoreCD(const Params& args, SharedStorage& storage)
        : BaseEpilogue(typename BaseEpilogue::Params{}), args(args), storage(storage) {}

    template <class ProblemShape, class BlockShape, class BlockCoord,
              class Engine, class Layout, class TiledMma, class Residue>
    CUTLASS_DEVICE void operator()(const ProblemShape& problem, const BlockShape& shape,
                                  const BlockCoord& block, const cute::Tensor<Engine, Layout>& acc,
                                  TiledMma mma, const Residue& residue, int tid, char* smem) {
        using namespace cute;
        const auto coords = mma.get_thread_slice(tid).partition_C(
            make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{})));
        // The mainloop and epilogue share storage. Every warp must finish its
        // operand reads before this tile is overwritten, and its stores before reuse.
        cute::cp_async_wait<0>();
        __syncthreads();
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc); ++i) {
            storage.values[get<0>(coords(i)) * BlockN + get<1>(coords(i))] = cutlass::bfloat16_t(acc(i));
        }
        __syncthreads();
        EpilogueOperator epilogue_operator(args.epilogue_args);
        constexpr int Groups = BlockM * (BlockN / 32);
        const int lane = tid % 8;
        for (int group = tid / 8; group < Groups; group += blockDim.x / 8) {
            const int row = group / (BlockN / 32);
            const int col = group % (BlockN / 32) * 32;
            const int global_row = get<0>(block) * BlockM + row;
            const int global_col = get<1>(block) * BlockN + col;
            const auto* values = reinterpret_cast<const __ppu_bfloat162*>(
                storage.values + row * BlockN + col + lane * 4);
            const auto lo = values[0], hi = values[1];
            const auto exponent = epilogue_operator.prepare_fp8(lo, hi,
                global_row < get<0>(problem) && global_col < get<1>(problem));
            const uint32_t packed = epilogue_operator.cast_fp8(lo, hi, exponent);
            if (global_row < get<0>(problem) && global_col < get<1>(problem)) {
                *reinterpret_cast<uint32_t*>(args.ptr_D + int64_t(global_row) * get<0>(args.dD)
                                              + global_col + lane * 4) = packed;
                if (lane == 0)
                    epilogue_operator.store_sf(global_row, global_col, args.batch_idx, exponent);
            }
        }
        epilogue_operator.finalize();
        __syncthreads();
    }
protected:
    const Params& args;
    SharedStorage& storage;
};

template <class Base, int BM, int BN, int RowTiles,
          class EpilogueOperator = epilogue::operators::CastToFP8PerToken>
struct EinsumUnbufferedEpilogue : PPUStoreCD<Base, BM, BN> {
    using Parent = PPUStoreCD<Base, BM, BN>;
    using Parent::Parent;

    uint32_t* output_base = nullptr;

    template <class Problem, class Shape, class Block, class E, class L, class MMA, class Residue>
    CUTLASS_DEVICE void operator()(const Problem& problem, const Shape&, const Block& block,
                                  const cute::Tensor<E,L>& acc, MMA mma, const Residue&, int tid, char*) {
        using namespace cute;
        const auto coords = mma.get_thread_slice(tid).partition_C(
            make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));
        const int first = get<0>(block) * BM;
        const int col0 = get<1>(block) * BN;
        const int cols = min(BN, get<1>(problem) - col0);
        const int rows = min(BM, get<0>(problem) - first);

        if (this->args.phase != RowQuantPhase::Store) {
            EpilogueOperator::template reduce_and_publish<Base::kIsAlignedN, BM, RowTiles>(
                acc, coords, mma, this->storage.values, this->args.epilogue_args.row_scales,
                first, rows, cols, tid);
            if (this->args.phase == RowQuantPhase::Reduce) return;
            if constexpr (RowTiles > 1) {
                if (tid == 0) {
                    auto* arrivals = output_base + int64_t(first) * get<0>(this->args.dD) / 4;
                    tix::atomic_add_rel(arrivals, 1);
                    while (tix::ld_acq(arrivals) < RowTiles) {}
                    // Once all waiters have read the counter, its owner can overwrite it with FP8.
                    tix::atomic_add_rel(arrivals, 1);
                    if (col0 == 0 && reinterpret_cast<uint32_t*>(this->args.ptr_D) == output_base)
                        while (tix::ld_acq(arrivals) < 2 * RowTiles) {}
                }
                __syncthreads();
            }
        }
        {
            // In a native 16x16 MMA, four lanes cover columns [0,8) and [8,16) of a row.
            static_assert(cute::is_same_v<typename MMA::LayoutC_TV, cute::PPU0015_16x16_Row>);
            // Reuse the existing shared tile for packed output followed by one inverse per row.
            auto* packed = reinterpret_cast<uint8_t*>(this->storage.values);
            auto* scales = reinterpret_cast<float*>(packed + BM * BN);
            cp_async_wait<0>();
            __syncthreads();
            for (int row = tid; row < rows; row += blockDim.x) {
                auto* scale = this->args.epilogue_args.row_scales + first + row;
                scales[row] = 1.f / *scale;
            }
            __syncthreads();
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<1>(acc); ++m) {
                CUTLASS_PRAGMA_UNROLL
                for (int mi = 0; mi < size<1>(shape<0>(acc)); ++mi) {
                    const int r = get<0>(coords(make_coord(0, mi, 0), m, 0));
                    const float inverse = r < rows ? scales[r] : 0.f;
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < size<2>(acc); ++k) {
                        const uint32_t value = EpilogueOperator::cast_fp8(
                            acc(make_coord(0, mi, 0), m, k), acc(make_coord(1, mi, 0), m, k),
                            acc(make_coord(0, mi, 1), m, k), acc(make_coord(1, mi, 1), m, k), inverse);
                        // Store each lane's two FP8 pairs together, then reorder at vector writeback.
                        if (r < rows) {
                            const int c = get<1>(coords(make_coord(0, mi, 0), m, k));
                            const int column = ((c & ~15) + 2 * (c & 7)) ^ ((r % min(8, BN / 16)) * 16);
                            *reinterpret_cast<uint32_t*>(packed + r * BN + column) = value;
                        }
                    }
                }
            }
            __syncthreads();
            constexpr int vectors_per_row = BN / 16;
            for (int i = tid; i < rows * vectors_per_row; i += blockDim.x) {
                const int row = i / vectors_per_row;
                const int c = (i % vectors_per_row) * 16;
                if (!Base::kIsAlignedN && c >= cols) continue;
                auto* dst = this->args.ptr_D + int64_t(first + row) * get<0>(this->args.dD) + col0 + c;
                const auto* src = packed + row * BN + (c ^ ((row % min(8, BN / 16)) * 16));
                const uint4 value = *reinterpret_cast<const uint4*>(src);
                const uint4 output{__byte_perm(value.x, value.y, 0x5410), __byte_perm(value.z, value.w, 0x5410),
                                   __byte_perm(value.x, value.y, 0x7632), __byte_perm(value.z, value.w, 0x7632)};
                if (reinterpret_cast<uintptr_t>(dst) % 16 == 0 && (Base::kIsAlignedN || c + 16 <= cols)) {
                    *reinterpret_cast<uint4*>(dst) = output;
                } else {
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 16; ++j) {
                        const uint32_t word = j < 4 ? output.x : j < 8 ? output.y : j < 12 ? output.z : output.w;
                        if (c + j < cols) reinterpret_cast<uint8_t*>(dst)[j] = uint8_t(word >> ((j % 4) * 8));
                    }
                }
            }
            __syncthreads();
        }

    }
};
template <class Base, int BM, int BN, int RowTiles, class Op>
struct UsesOutputCounter<EinsumUnbufferedEpilogue<Base, BM, BN, RowTiles, Op>>
    : cute::bool_constant<(RowTiles > 1)> {};

} // namespace deep_gemm
