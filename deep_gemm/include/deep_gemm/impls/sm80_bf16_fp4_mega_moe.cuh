#pragma once
#include <deep_gemm/impls/w4a16_gemm_cutlass3.cuh>
#include <deep_gemm/impls/int8_mega_moe_dispatch.cuh>
#include <deep_gemm/scheduler/sm80_mega_moe.cuh>
#include <deep_gemm/epilogue/mega_moe_l2_epilogue.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>

namespace deep_gemm::sm80 {
template <bool First, int N, int BM, int BN, int WM, int Ranks, int Capacity, int TopK,
          int TilesPerFlag, GatedActivation Activation, bool FastMath>
struct W4Output {
    const layout::SymBuffer<Ranks>& scatter;
    layout::TokenSrcMetadata* sources;
    cutlass::bfloat16_t* activated;
    const int* replica;
    const uint8_t* replica_scale;
    int masters;
    float limit, alpha, beta, linear_beta;

    CUTLASS_DEVICE void select_bank(int& expert, const int*& w, const uint8_t*& sf) const {
        if (expert >= masters) {
            expert -= masters;
            w = replica;
            sf = replica_scale;
        }
    }
    template <class Scheduler>
    CUTLASS_DEVICE void wait_for_input(const Scheduler& sched, int k) const {
        if constexpr (First)
            if (k % TilesPerFlag == 0)
                comm::wait_copy_stripe(sched.params.copy_ready_flags + sched.curr_global_block_m_idx,
                                       sched.round_generation, k / TilesPerFlag + 1);
    }
    template <class Acc, class Coord, class TiledMma>
    CUTLASS_DEVICE void store(Acc acc, Coord coord, TiledMma mma, int k_warp, int base, int mb, int nb,
                              int rows, char* smem) const {
        using namespace cute;
        if constexpr (First) {
            auto tile = make_tensor(make_smem_ptr(reinterpret_cast<float*>(smem)),
                                    make_layout(make_shape(Int<BM>{}, Int<BN>{}), LayoutRight{}));
            if (!k_warp) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(acc); ++i)
                    tile(get<0>(coord(i)), get<1>(coord(i))) = acc(i);
            }
            __syncthreads();
            for (int j = threadIdx.x; j < min(int(BM), rows) * (BN / 2); j += blockDim.x) {
                const int row = j / (BN / 2), col = j % (BN / 2);
                const int gate = col / 8 * 16 + col % 8;
                const float value = routed_gated_activation<Activation, FastMath>(
                    tile(row, gate), tile(row, gate + 8), limit, alpha, beta, linear_beta);
                activated[int64_t(base + mb * BM + row) * (N / 2) + nb * (BN / 2) + col] =
                    cutlass::bfloat16_t(value);
            }
        } else {
            using Stride = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
            using Op = cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>;
            using SmemLayout = Layout<Shape<Int<(BM / WM) * 16>, Int<BN>>, cute::Stride<Int<BN>, _1>>;
            using Scatter = epilogue::MegaMoEScatterEpilogue<Stride, Stride, SmemLayout,
                Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>, float,
                cutlass::bfloat16_t, Op, Ranks, (int64_t(TopK) * Capacity * N > UINT32_MAX)>;
            typename Scatter::Params p{{}, sources, nullptr, Capacity};
            Scatter epilogue{p};
            constexpr int kWarpsOnK = size<3>(typename TiledMma::ThrLayoutVMNK{});
            epilogue.template operator()<kWarpsOnK>(
                make_shape(rows + mb * BM, N, 1, 1), make_shape(Int<BM>{}, Int<BN>{}, _1{}),
                make_coord(mb, nb, 0, 0), acc, mma, make_shape(0, 0, 0),
                threadIdx.x, smem, 0, base, scatter);
        }
    }
};

// Both phases use the native grouped W4A16 MMA and offline-packed E8M0 weights.
template <bool First, int N, int K, int BM, int BN, int BK, int WM, int WK,
          int Groups, int Stages, int Ranks, int Capacity, int TopK, int TilesPerFlag,
          GatedActivation Activation, bool FastMath, bool OutputAligned,
          int ColumnSplits, int ReduceGroupThreads>
__device__ void w4_mega_gemm(const sched::Int8MegaMoESchedulerArguments& routing,
        const layout::SymBuffer<Ranks>& scatter, const cutlass::bfloat16_t* a,
        const int* w, const uint8_t* sf, const int* replica, const uint8_t* replica_sf,
        int masters, cutlass::bfloat16_t* d, float limit, float alpha, float beta, float linear_beta,
        char* smem, const int64_t* ids, const float* weights, __ppu_bfloat16* output, int tokens) {
    using Dispatch = Int8FusedDispatch<BK, true>;
    using Scheduler = sched::SM80MegaMoEScheduler<N, K, BM, BN, Groups,
        First ? MegaMoEPhaseKind::Linear1 : MegaMoEPhaseKind::Linear2,
        Ranks, TilesPerFlag, Capacity, TopK, Dispatch>;
    using Kernel = cutlass::gemm::kernel::W4A16GEMM<cutlass::float4_t, uint8_t, N, K,
        BM, BN, BK, WM, 64, WK, Groups, Stages, GemmType::MGroupedNoPad, 32, 1, Scheduler>;
    typename Kernel::Params p{a, w, sf, d};
    if constexpr (First) {
        Dispatch::template prepare<Scheduler, Kernel::MaxThreadsPerBlock>(routing, smem);
        p.scheduler = routing;
    } else {
        p.scheduler = TileSchedulerArguments(routing.shape_m, routing.grouped_layout);
    }
    Kernel{}(p, smem, W4Output<First, N, BM, BN, WM, Ranks, Capacity, TopK, TilesPerFlag, Activation, FastMath>{
        scatter, routing.src_info, d, replica, replica_sf, masters, limit, alpha, beta, linear_beta});
    if constexpr (!First) {
        auto* signals = routing.sym_buffer.get_base_ptr<layout::MegaMoESignals*>();
        const auto generation = __ldbl(signals->preprocess_ready) & ~kPreprocessMetadataReadyBit;
        comm::scatter_release<Ranks>(*signals, scatter, gridDim.x, blockIdx.x, threadIdx.x, generation, BN == 256);
        // The GEMM has retired its shared tile before it becomes the rank writer mask.
        auto* writer_mask = reinterpret_cast<uint32_t*>(smem);
        comm::rank_barrier<Ranks, true>(signals->scatter_arrivals, generation, writer_mask);
        moe_reduce_dispatch<OutputAligned, TopK, N, Kernel::MaxThreadsPerBlock,
                            ColumnSplits, ReduceGroupThreads, /*kSkipZeroWeight=*/true>(
            scatter.template get_base_ptr<const __ppu_bfloat16*>(), ids, weights, output,
            Capacity, Groups, scatter.rank_idx, tokens, *reinterpret_cast<const uint64_t*>(writer_mask));
    }
}
}  // namespace deep_gemm::sm80
