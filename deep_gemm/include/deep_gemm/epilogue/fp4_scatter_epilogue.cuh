#pragma once

#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cute/tensor.hpp>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <hggc_bf16.h>

namespace deep_gemm::epilogue {

// FP4 GEMM2 epilogue used by the two-stage MegaMoE path. It converts the
// accumulator tile to BF16 and scatters each valid row directly into the
// source token rank's symmetric scratch buffer.
template <
    class StrideC_, class StrideD_, class SmemLayout_,
    class CopyAtomR2S_, class ElementAccumulator_, class ElementD_,
    class ThreadEpilogueOp_,
    int kWorldSize = 2,
    bool kWideScatterOffset = false,
    bool kWarpPipeline_ = false>
class Fp4ScatterEpilogue {
public:
    using StrideC = StrideC_;
    using StrideD = StrideD_;
    using SmemLayout = SmemLayout_;
    using CopyAtomR2S = CopyAtomR2S_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementC = ElementD_;
    using ElementD = ElementD_;
    using ThreadEpilogueOp = ThreadEpilogueOp_;
    static constexpr bool kWarpPipeline = kWarpPipeline_;
    static constexpr int kPipelineSlots = 3;
    static constexpr int kReadyBarrier = 4;
    static constexpr int kFreeBarrier = kReadyBarrier + kPipelineSlots;
    using SharedElement = std::conditional_t<kWarpPipeline, ElementD, ElementAccumulator>;
    static_assert(!kWarpPipeline ||
                  (cute::size<0>(SmemLayout{}) == 32 && cute::size<1>(SmemLayout{}) == 256),
                  "warp pipeline uses complete 32x256 BF16 subtiles");

    struct SharedStorage {
        cute::array_aligned<SharedElement,
                            (kWarpPipeline ? kPipelineSlots : 2) * cute::cosize_v<SmemLayout>> accumulator;
    };

    struct Arguments {
        typename ThreadEpilogueOp::Params thread{};
        layout::TokenSrcMetadata const* src_info = nullptr;
        int32_t const* masked_m = nullptr;
        uint32_t token_capacity = 0;
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
            ProblemShape const&, Arguments const& args, void*) {
        return args;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    CUTLASS_HOST_DEVICE explicit Fp4ScatterEpilogue(Params const& params)
        : params_(params), epilogue_op_(params.thread) {}

#if defined(__HGGCCC__)
    CUTLASS_HOST_DEVICE Fp4ScatterEpilogue(Params const& params, SharedStorage&)
        : params_(params), epilogue_op_(params.thread) {}
#endif

    CUTLASS_DEVICE bool is_source_needed() {
        return epilogue_op_.is_source_needed();
    }

    template <
        class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL,
        class FrgEngine, class FrgLayout, class TiledMma, class ResidueMNK>
    CUTLASS_DEVICE void operator()(
            ProblemShapeMNKL problem_shape_mnkl,
            BlockShapeMNK block_shape_mnk,
            BlockCoordMNKL block_coord_mnkl,
            cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
            TiledMma tiled_mma, ResidueMNK, int thread_idx, char* smem,
            uint32_t expert, uint32_t offset_m,
            layout::SymBuffer<kWorldSize> const& scatter_buffer) {
        static_assert(!kWarpPipeline, "use produce/consume for warp pipeline");
        using namespace cute;

        const auto N = get<1>(problem_shape_mnkl);
        auto [m_coord, n_coord, k_coord, l_coord] = block_coord_mnkl;
        (void)k_coord;
        (void)l_coord;

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
        constexpr int kSmemStageElements = cute::cosize_v<SmemLayout>;
        Tensor shared_accumulator = make_tensor(
            make_smem_ptr(storage.accumulator.data()), SmemLayout{});

        auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
        auto thread_r2s = tiled_r2s.get_thread_slice(thread_idx);
        Tensor register_accumulator = thread_r2s.retile_S(accumulators);

        auto subtile_shape = make_shape(
            size<0>(shared_accumulator), size<1>(shared_accumulator));
        Tensor identity = make_identity_tensor(
            make_shape(get<0>(block_shape_mnk), get<1>(block_shape_mnk)));
        Tensor divided_identity = flat_divide(identity, subtile_shape);

        constexpr int kSmemM = size<0>(shared_accumulator);
        constexpr int kSmemN = size<1>(shared_accumulator);
        constexpr int kNumThreads = CUTE_STATIC_V(size(TiledMma{}));
        const uint32_t tile_row = m_coord * get<0>(block_shape_mnk);
        const uint32_t block_row = offset_m + tile_row;
        const uint32_t valid_rows = get<0>(problem_shape_mnkl);

        CUTLASS_PRAGMA_UNROLL
        for (int step_m = 0; step_m < size<2>(divided_identity); ++step_m) {
            const uint32_t subtile_m_begin = tile_row + step_m * kSmemM;
            if (subtile_m_begin >= valid_rows) {
                break;
            }

            CUTLASS_PRAGMA_UNROLL
            for (int step_n = 0; step_n < size<3>(divided_identity); ++step_n) {
                const int smem_stage =
                    (step_m * int(size<3>(divided_identity)) + step_n) & 1;
                Tensor shared_accumulator_stage = make_tensor(
                    make_smem_ptr(storage.accumulator.data() +
                                  smem_stage * kSmemStageElements),
                    SmemLayout{});
                Tensor shared_r2s =
                    thread_r2s.partition_D(shared_accumulator_stage);
                const uint32_t n_base =
                    n_coord * get<1>(block_shape_mnk) + step_n * kSmemN;

                CUTLASS_PRAGMA_UNROLL
                for (int pipe_m = 0; pipe_m < size<1>(shared_r2s); ++pipe_m) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int pipe_n = 0; pipe_n < size<2>(shared_r2s); ++pipe_n) {
                        const int mma_m =
                            step_m * size<1>(shared_r2s) + pipe_m;
                        const int mma_n =
                            step_n * size<2>(shared_r2s) + pipe_n;
                        copy(tiled_r2s,
                             register_accumulator(_, mma_m, mma_n),
                             shared_r2s(_, pipe_m, pipe_n));
                    }
                }
                __syncthreads();

                scatter_subtile<kNumThreads>(shared_accumulator_stage, N,
                    thread_idx, block_row + step_m * kSmemM, subtile_m_begin,
                    valid_rows, n_base, scatter_buffer);

            }
        }
        // Alternating buffers remove subtile read/write hazards, but the next
        // persistent GEMM tile still reuses this storage for its mainloop.
        __syncthreads();
    }

    template <class BlockShape, class FrgEngine, class FrgLayout, class TiledMma>
    CUTLASS_DEVICE void produce(
            BlockShape block_shape_mnk, uint32_t m_coord,
            cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
            TiledMma tiled_mma, int thread_idx, char* smem,
            uint32_t valid_rows, uint32_t& sequence) {
        static_assert(kWarpPipeline);
        using namespace cute;

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
        constexpr int kSmemStageElements = cute::cosize_v<SmemLayout>;
        Tensor shared_accumulator = make_tensor(
            make_smem_ptr(storage.accumulator.data()), SmemLayout{});

        auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
        auto thread_r2s = tiled_r2s.get_thread_slice(thread_idx);
        Tensor register_accumulator = thread_r2s.retile_S(accumulators);

        auto subtile_shape = make_shape(
            size<0>(shared_accumulator), size<1>(shared_accumulator));
        Tensor identity = make_identity_tensor(
            make_shape(get<0>(block_shape_mnk), get<1>(block_shape_mnk)));
        Tensor divided_identity = flat_divide(identity, subtile_shape);

        constexpr int kSmemM = size<0>(shared_accumulator);
        const uint32_t tile_row = m_coord * get<0>(block_shape_mnk);

        CUTLASS_PRAGMA_UNROLL
        for (int step_m = 0; step_m < size<2>(divided_identity); ++step_m) {
            const uint32_t subtile_m_begin = tile_row + step_m * kSmemM;
            if (subtile_m_begin >= valid_rows) {
                break;
            }

            CUTLASS_PRAGMA_UNROLL
            for (int step_n = 0; step_n < size<3>(divided_identity); ++step_n) {
                const int smem_stage = sequence % kPipelineSlots;
                __ppu_barrier_sync(kFreeBarrier + smem_stage, scatter_pipeline::kCtaThreads);
                Tensor shared_accumulator_stage = make_tensor(
                    make_smem_ptr(storage.accumulator.data() +
                                  smem_stage * kSmemStageElements),
                    SmemLayout{});
                Tensor shared_r2s =
                    thread_r2s.partition_D(shared_accumulator_stage);

                CUTLASS_PRAGMA_UNROLL
                for (int pipe_m = 0; pipe_m < size<1>(shared_r2s); ++pipe_m) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int pipe_n = 0; pipe_n < size<2>(shared_r2s); ++pipe_n) {
                        const int mma_m =
                            step_m * size<1>(shared_r2s) + pipe_m;
                        const int mma_n =
                            step_n * size<2>(shared_r2s) + pipe_n;
                        auto fragment = register_accumulator(_, mma_m, mma_n);
                        auto converted = make_tensor<ElementD>(shape(fragment));
                        cutlass::NumericConverter<ElementD, ElementAccumulator,
                            cutlass::FloatRoundStyle::round_to_nearest> convert;
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < size(fragment); ++i)
                            converted(i) = convert(fragment(i));
                        copy(converted, shared_r2s(_, pipe_m, pipe_n));
                    }
                }

                __ppu_barrier_arrive(kReadyBarrier + smem_stage, scatter_pipeline::kCtaThreads, 0);
                ++sequence;
            }
        }
    }

    template <class ProblemShape, class BlockShape, class BlockCoord>
    CUTLASS_DEVICE void consume(ProblemShape problem_shape_mnkl,
            BlockShape block_shape_mnk, BlockCoord block_coord_mnkl,
            int thread_idx, char* smem, uint32_t offset_m,
            layout::SymBuffer<kWorldSize> const& scatter_buffer,
            uint32_t& sequence) {
        using namespace cute;
        static_assert(kWarpPipeline);
        constexpr int kSmemM = size<0>(SmemLayout{});
        constexpr int kSmemN = size<1>(SmemLayout{});
        const auto N = get<1>(problem_shape_mnkl);
        auto [m_coord, n_coord, k_coord, l_coord] = block_coord_mnkl;
        const uint32_t tile_row = m_coord * get<0>(block_shape_mnk);
        const uint32_t valid_rows = get<0>(problem_shape_mnkl);
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
        CUTLASS_PRAGMA_UNROLL
        for (int step_m = 0; step_m < get<0>(BlockShape{}) / kSmemM; ++step_m) {
            const uint32_t subtile_m_begin = tile_row + step_m * kSmemM;
            if (subtile_m_begin >= valid_rows) break;
            CUTLASS_PRAGMA_UNROLL
            for (int step_n = 0; step_n < get<1>(BlockShape{}) / kSmemN; ++step_n) {
                const int smem_stage = sequence % kPipelineSlots;
                __ppu_barrier_sync(kReadyBarrier + smem_stage, scatter_pipeline::kCtaThreads);
                Tensor shared_accumulator_stage = make_tensor(
                    make_smem_ptr(storage.accumulator.data() +
                        smem_stage * cosize_v<SmemLayout>), SmemLayout{});
                const uint32_t n_base = n_coord * get<1>(block_shape_mnk) + step_n * kSmemN;
                scatter_subtile<scatter_pipeline::kScatterThreads>(shared_accumulator_stage, N, thread_idx,
                    offset_m + subtile_m_begin, subtile_m_begin, valid_rows, n_base, scatter_buffer);
                // Free releases only local shared reuse. The final system fence
                // and owner notification still publish remote data to reduce.
                __ppu_barrier_arrive(kFreeBarrier + smem_stage, scatter_pipeline::kCtaThreads, 0);
                ++sequence;
            }
        }
    }

private:

    template <int kNumThreads, class SharedTensor>
    CUTLASS_DEVICE void scatter_subtile(SharedTensor shared_accumulator_stage,
            int N, int thread_idx, uint32_t global_row, uint32_t subtile_m_begin,
            uint32_t valid_rows, uint32_t n_base,
            layout::SymBuffer<kWorldSize> const& scatter_buffer) {
        using namespace cute;
        constexpr int kSmemM = size<0>(SmemLayout{});
        constexpr int kSmemN = size<1>(SmemLayout{});
        constexpr int kVector = kSmemN / 32;
        constexpr int kRowsPerIteration = kNumThreads / 32;
        constexpr int kRowsPerSubtile = kSmemM / kRowsPerIteration;
        static_assert(kNumThreads % 32 == 0 && kSmemM % kRowsPerIteration == 0);
        static_assert(kRowsPerSubtile % 2 == 0 && (kVector == 4 || kVector == 8));
        const int row_lane = thread_idx / 32;
        const int column = (thread_idx % 32) * kVector;
        auto* local_tile_base = scatter_buffer.template get_base_ptr<__ppu_bfloat16*>() + n_base;
        layout::TokenSrcMetadata metadata[kRowsPerSubtile];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kRowsPerSubtile; ++i) {
            const uint32_t row = row_lane + i * kRowsPerIteration;
            // Compact metadata has no padding after the last expert.
            // Mask the load itself, not only the later scatter store.
            metadata[i] = {};
            if (subtile_m_begin + row < valid_rows) {
                metadata[i] = params_.src_info[
                    global_row + row];
            }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int row_base = 0; row_base < kSmemM;
             row_base += 2 * kRowsPerIteration) {
            const int rows[2] = {
                row_lane + row_base,
                row_lane + row_base + kRowsPerIteration};
            const layout::TokenSrcMetadata infos[2] = {
                metadata[row_base / kRowsPerIteration],
                metadata[(row_base + kRowsPerIteration) /
                         kRowsPerIteration]};

            CUTLASS_PRAGMA_UNROLL
            for (int pair = 0; pair < 2; ++pair) {
                const int row = rows[pair];
                const layout::TokenSrcMetadata token_info = infos[pair];
                const uint32_t expert_row = subtile_m_begin + row;
                const uint32_t destination_rank = token_info.rank_idx;
                if (row < kSmemM && expert_row < valid_rows) {
                    __ppu_bfloat16* destination = scatter_buffer.map(
                        local_tile_base, destination_rank);
                    if constexpr (kWideScatterOffset) {
                        destination +=
                            (static_cast<uint64_t>(token_info.topk_idx) *
                                 params_.token_capacity +
                             token_info.token_idx) * N + column;
                    } else {
                        destination +=
                            (token_info.topk_idx *
                                 params_.token_capacity +
                             token_info.token_idx) * N + column;
                    }

                    const auto* source = &shared_accumulator_stage(row, column);
                    if constexpr (kVector == 8) {
                        union {
                            __ppu_bfloat162 bf16[4];
                            int4 packed;
                        } value;
                        if constexpr (kWarpPipeline) {
                            value.packed = *reinterpret_cast<const int4*>(source);
                        } else {
                            const float4 low = reinterpret_cast<const float4*>(source)[0];
                            const float4 high = reinterpret_cast<const float4*>(source)[1];
                            value.bf16[0] = __floats2bfloat162_rn(low.x, low.y);
                            value.bf16[1] = __floats2bfloat162_rn(low.z, low.w);
                            value.bf16[2] = __floats2bfloat162_rn(high.x, high.y);
                            value.bf16[3] = __floats2bfloat162_rn(high.z, high.w);
                        }
                        if (destination_rank == scatter_buffer.rank_idx) {
                            asm volatile(
                                "ppu.st.global.bl.v4.b32 [%0], {%1, %2, %3, %4};"
                                :: "l"(destination), "r"(value.packed.x),
                                   "r"(value.packed.y), "r"(value.packed.z),
                                   "r"(value.packed.w) : "memory");
                        } else {
                            __ppu_remote_store_bulk_volatile_b32x4(
                                reinterpret_cast<volatile void*>(destination),
                                *reinterpret_cast<uint4*>(&value.packed));
                        }
                    } else {
                        const float4 values = *reinterpret_cast<const float4*>(source);
                        const __ppu_bfloat162 low =
                            __floats2bfloat162_rn(values.x, values.y);
                        const __ppu_bfloat162 high =
                            __floats2bfloat162_rn(values.z, values.w);
                        const int2 packed = {
                            *reinterpret_cast<const int32_t*>(&low),
                            *reinterpret_cast<const int32_t*>(&high)};
                        asm volatile(
                            "ppu.st.global.bl.v2.b32 [%0], {%1, %2};"
                            :: "l"(destination), "r"(packed.x),
                               "r"(packed.y) : "memory");
                    }
                }
            }
        }
    }

    Params params_;
    ThreadEpilogueOp epilogue_op_;
};

}  // namespace deep_gemm::epilogue
