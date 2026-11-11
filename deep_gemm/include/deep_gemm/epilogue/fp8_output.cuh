#pragma once

#include <deep_gemm/common/fp8_output_args.cuh>
#include <cute/tensor.hpp>

namespace deep_gemm {

// PPU accumulator coordinates differ from SM100 TMEM. Reorder the rounded BF16
// tile in shared memory, then quantize each row in independent 32-column groups.
template <class BaseEpilogue, int BlockM, int BlockN>
class FP8OutputEpilogue : public BaseEpilogue {
public:
    static_assert(BlockN % 32 == 0);
    using ElementD = cutlass::float_e4m3_t;
    using Arguments = FP8OutputArguments<typename BaseEpilogue::ThreadEpilogueOp::Params>;
    using Params = Arguments;
    struct SharedStorage { cutlass::bfloat16_t values[BlockM * BlockN]; };
    using TensorStorage = SharedStorage;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(const ProblemShape&, const Arguments& args, void*) {
        return args;
    }
    template <class ProblemShape>
    static size_t get_workspace_size(const ProblemShape&, const Arguments&) { return 0; }

    CUTLASS_DEVICE FP8OutputEpilogue(const Params& args, SharedStorage& storage)
        : BaseEpilogue(typename BaseEpilogue::Params{}), args(args), storage(storage) {}

    template <class ProblemShape, class BlockShape, class BlockCoord,
              class Engine, class Layout, class TiledMma, class Residue>
    CUTLASS_DEVICE void operator()(const ProblemShape& problem, const BlockShape&,
                                  const BlockCoord& block, const cute::Tensor<Engine, Layout>& acc,
                                  TiledMma mma, const Residue&, int tid, char*) {
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
            const auto maximum = __hmax2(__habs2(lo), __habs2(hi));
            uint32_t amax = max(uint32_t(__bfloat16_as_ushort(maximum.x)),
                                uint32_t(__bfloat16_as_ushort(maximum.y)));
            CUTLASS_PRAGMA_UNROLL
            for (int shift = 1; shift < 8; shift *= 2)
                amax = max(amax, __shfl_xor_sync(0xffffffff, amax, shift));
            // Upstream get_ue8m0_sf_exp<e4m3>: ceil(amax / 448) to a
            // power of two, with the same 2^-22 floor (amax clamp 1e-4).
            const uint32_t exponent = max((amax + 31u) >> 7, 113u) - 8u;
            const auto inverse = __bfloat162bfloat162(__ushort_as_bfloat16((254u - exponent) << 7));
            const auto scaled_lo = __bfloat1622float2(__hmul2(lo, inverse));
            const auto scaled_hi = __bfloat1622float2(__hmul2(hi, inverse));
            const uint32_t packed = __hg_fp8x4_e4m3(
                make_float4(scaled_lo.x, scaled_lo.y, scaled_hi.x, scaled_hi.y)).__x;
            if (global_row < get<0>(problem) && global_col < get<1>(problem)) {
                *reinterpret_cast<uint32_t*>(args.ptr_D + int64_t(global_row) * get<0>(args.dD)
                                              + global_col + lane * 4) = packed;
                if (lane == 0) {
                    const int sf_idx = (args.batch_idx * get<1>(problem) + global_col) / 32;
                    auto* sf = reinterpret_cast<uint8_t*>(args.sfd + (sf_idx / 4) * args.sfd_stride + global_row);
                    sf[sf_idx % 4] = exponent;
                }
            }
        }
        __syncthreads();
    }
private:
    const Params& args;
    SharedStorage& storage;
};

} // namespace deep_gemm
