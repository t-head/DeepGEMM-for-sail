#pragma once

#include <deep_gemm/gemm_comm/reduce_scatter.cuh>
#define __syncthreads() deep_gemm::gemm_comm::mainloop_sync()
#include <deep_gemm/impls/int8_densegemm_cutlass3.cuh>
#undef __syncthreads

namespace deep_gemm::gemm_comm {
template <class Mainloop, bool Pipeline> struct ChannelwiseRSMainloop {
    using type = Mainloop;
};
template <class Arch, class DP, class Tile, class EA, class SA, class EB, class SB, class Mma, class CA, class SLA,
          class SCA, class TA, class CB, class SLB, class SCB, class TB>
struct ChannelwiseRSMainloop<
    cutlass::gemm::collective::CollectiveMma<Arch, DP, Tile, EA, SA, EB, SB, Mma, CA, SLA, SCA, TA, CB, SLB, SCB, TB>, true> {

    static_assert(DP::Stages >= 3, "GEMM/RS sender requires at least three input stages");
    static constexpr int BlockM = cute::size<0>(Tile{});
    static constexpr bool OwnerRows = can_align_owner_rows(DG_GEMM_RS_M, DG_GEMM_RS_WORLD, BlockM);
    using type = cutlass::gemm::collective::CollectiveMma<Arch, DP, Tile, EA, SA, EB, SB, Mma,
                                                          GemmRSSendCopy<CA, Mma, Tile, DG_GEMM_RS_SHARED_PACKETS, DG_GEMM_RS_N,
                                                              cute::size(Mma{}),
                                                              !OwnerRows && DG_GEMM_RS_M % (DG_GEMM_RS_WORLD * BlockM) != 0
                                                                  ? DG_GEMM_RS_M / DG_GEMM_RS_WORLD : 0,
                                                              DG_GEMM_RS_N % cute::size<1>(Tile{}) != 0>, SLA,
                                                          SCA, TA, CB, SLB, SCB, TB>;
};

// Keep the native input mainloop and parameter ABI for both output paths.
template <class Problem, class Mainloop, class Epilogue, class Scheduler>
class Fp8ChannelwiseGemmRS
    : public cutlass::gemm::kernel::DenseGemmKernel<
          Problem, typename ChannelwiseRSMainloop<Mainloop, (DG_GEMM_RS_SHARED_PACKETS > 0)>::type, Epilogue,
          Scheduler> {
  public:
    static constexpr bool Pipeline = DG_GEMM_RS_SHARED_PACKETS > 0;
    using OriginalBase = cutlass::gemm::kernel::DenseGemmKernel<Problem, Mainloop, Epilogue, Scheduler>;
    using CollectiveMainloop = typename ChannelwiseRSMainloop<Mainloop, Pipeline>::type;
    using Base = cutlass::gemm::kernel::DenseGemmKernel<Problem, CollectiveMainloop, Epilogue, Scheduler>;
    static_assert(sizeof(typename Base::Params) == sizeof(typename OriginalBase::Params));
    static_assert(alignof(typename Base::Params) == alignof(typename OriginalBase::Params));
    static_assert(sizeof(typename CollectiveMainloop::SharedStorage) == sizeof(typename Mainloop::SharedStorage));
    static_assert(Base::WarpOnK == 1, "channelwise GEMM/RS requires complete K accumulators");
    using Params = typename Base::Params;
    using TileShape = typename Base::TileShape;
    using TiledMma = typename Base::TiledMma;
    using ElementCompute = typename Base::ElementCompute;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    static_assert(DG_GEMM_RS_SHARED_BUDGET >= Base::SharedStorageSize);
    static constexpr int SharedStorageSize = DG_GEMM_RS_SHARED_BUDGET;
    static constexpr int M = DG_GEMM_RS_M, N = DG_GEMM_RS_N, World = DG_GEMM_RS_WORLD, Rank = DG_GEMM_RS_RANK;
    static constexpr int BM = cute::size<0>(TileShape{}), BN = cute::size<1>(TileShape{});
    static constexpr int LM = M / World, OwnerTiles = (LM + BM - 1) / BM;
    static constexpr bool OwnerRows = can_align_owner_rows(M, World, BM);

    template <class Acc, class Shape>
    __device__ void store_owned_tile(const Acc& acc, Shape shape, GemmRSState state, int epoch,
                                     int owner, int mt, int nt) {
        using namespace cute;
        const auto phase = int64_t((epoch - 1) & 1) * state.phase_elements() / 2 +
                           int64_t(Rank) * GemmRSPackets<LM>::StorageRows * N / 2;
        const int first_atom = mt * (BM / 16) * (N / 16) + nt * (BN / 16);
        auto* destination = reinterpret_cast<uint32_t*>(state.peer(owner)) + phase + first_atom * 128;
        const auto coords = TiledMma{}.get_slice((threadIdx.x / 32) * 32).partition_C(
            make_identity_tensor(take<0, 2>(shape)));
        for_each(make_int_sequence<size(Acc{}) / 8>{}, [&](auto packet) {
            constexpr int first = decltype(packet)::value * 8;
            const int col = get<1>(coords(first));
            if (N % BN == 0 || nt * BN + col < N) {
                const int atom = get<0>(coords(first)) / 16 * (N / 16) + col / 16;
                const auto rounded = pack_accumulator_packet(acc, first);
                if (owner == Rank)
                    __ppu_global_store_bulk_b32x4(destination + atom * 128, rounded);
                else
                    __ppu_remote_store_bulk_ncp_b32x4(destination + atom * 128, rounded);
            }
        });
    }

    template <class Acc, class Shape, class Transport>
    __device__ void stage_remote(const Acc& acc, Shape shape, GemmRSState state, int epoch, int mt, int nt,
                                 Transport& transport) {
        using namespace cute;
        constexpr int M = DG_GEMM_RS_M, N = DG_GEMM_RS_N;
        constexpr int World = DG_GEMM_RS_WORLD, Rank = DG_GEMM_RS_RANK, LM = M / World;
        constexpr int BM = size<0>(Shape{}), BN = size<1>(Shape{});
        constexpr bool SplitOwner = !OwnerRows && M % (World * BM) != 0;
        int owner = 0;
        if constexpr (OwnerRows) {
            owner = mt / OwnerTiles;
            mt %= OwnerTiles;
            if (owner == Rank) {
                store_owned_tile(acc, shape, state, epoch, owner, mt, nt);
                return;
            }
        } else if constexpr (!SplitOwner) {
            auto coords = TiledMma{}.get_slice((threadIdx.x / 32) * 32).partition_C(
                make_identity_tensor(take<0, 2>(shape)));
            owner = (mt * BM + get<0>(coords(0))) / LM;
            if (owner == Rank) {
                store_packets<M, N, World, Rank>(state, epoch, acc, TiledMma{}, shape, mt, nt);
                return;
            }
        }
        if (SplitOwner || owner != Rank) {
            auto stage = const_cast<uint4*>(transport.staging);
            for_each(make_int_sequence<size(Acc{}) / 8>{}, [&](auto packet) {
                constexpr int p = decltype(packet)::value;
                const auto value = pack_accumulator_packet(acc, p * 8);
                if constexpr (p < Transport::SharedPacketCount)
                    stage[(((p / 2) * Transport::Warps + threadIdx.x / 32) * 2 + p % 2) * 32 + threadIdx.x % 32] = value;
                else
                    transport.cache[p - Transport::SharedPacketCount] = value;
            });
        }
        mainloop_sync();
        const auto phase = int64_t((epoch - 1) & 1) * state.word(PhaseElements) / 2;
        if constexpr (SplitOwner) {
            transport.destination = reinterpret_cast<uint32_t*>(state.ptr<int64_t>(Peers));
            transport.base_offset = phase + int64_t(Rank) * GemmRSPackets<LM>::StorageRows * N / 2;
            if constexpr (LM % 16 == 0)
                transport.base_offset += nt * BN / 16 * 128;
            transport.own_rank = Rank;
            if constexpr (Transport::ReuseOwnerAddress) {
                const auto coordinate = TiledMma{}.get_slice((threadIdx.x / 32) * 32).partition_C(
                    make_identity_tensor(take<0, 2>(shape)));
                const int first_row = mt * BM + get<0>(coordinate(0));
                const int last_row = mt * BM + get<0>(coordinate(size(Acc{}) - 8)) + 15;
                const int first_owner = first_row / LM;
                if (last_row < M && first_owner != Rank && last_row / LM == first_owner) {
                    const int row_atom = mt * (BM / 16) - first_owner * LM / 16;
                    const int atom = row_atom * (N / 16) + nt * (BN / 16);
                    transport.destination = reinterpret_cast<uint32_t*>(state.peer(first_owner)) +
                                            transport.base_offset + atom * 128;
                    transport.own_rank = -1;
                }
            }
        } else {
            const int first = (OwnerRows ? mt * BM : (mt * BM) % LM) / 16 * (N / 16) + nt * BN / 16;
            transport.destination = owner == Rank ? nullptr :
                reinterpret_cast<uint32_t*>(state.ptr<int64_t>(Peers)[owner]) + phase +
                int64_t(Rank) * GemmRSPackets<LM>::StorageRows * N / 2 + first * 128;
        }
        transport.m_offset = mt * BM;
        transport.n_remaining = N - nt * BN;
        transport.calls = 0;
    }

    template <class Index>
    __device__ __forceinline__ auto compute_tile(const Params& params, CollectiveMainloop& mainloop, TiledMma mma,
                                                Index mt, Index nt, char* smem_buf) {
        using namespace cute;
        constexpr int M = DG_GEMM_RS_M, N = DG_GEMM_RS_N, K = DG_GEMM_RS_K;
        constexpr int BN = size<1>(TileShape{});
        auto shape = TileShape{};
        const int offset_m = OwnerRows ? mt / OwnerTiles * LM : 0;
        const auto tile_m = OwnerRows ? mt % OwnerTiles : mt;
        // Padding rows are discarded by the owner reduction. Read a full input
        // tile wherever those rows remain inside the original activation.
        const int rows = OwnerRows ? min(OwnerTiles * BM, M - offset_m) : M;
        auto inputs = mainloop.load_init(Problem{rows, N, K, 1}, make_coord(tile_m, nt, _, 0),
                                         params.mainloop, rows, offset_m, 0,
                                         reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + int64_t(offset_m) * K,
                                         reinterpret_cast<const ElementB*>(params.mainloop.ptr_B));
        Tensor gA = get<0>(inputs);
        Tensor gB = get<1>(inputs);
        const auto residue_n = [&] {
            if constexpr (!Pipeline && N % BN == 0)
                return BN;
            else
                return N - size<0>(gB) * nt;
        }();
        auto residue = make_tuple(rows - size<0>(gA) * tile_m, residue_n, K - size<1>(gA) * size<2>(gA));
        Tensor acc = make_fragment_like<ElementCompute>(partition_fragment_C(mma, take<0, 2>(shape)));
        clear(acc);
        auto k_iter = make_coord_iterator(cute::shape<2>(gA));
        int k_count = size<2>(gA);
        mainloop(acc, inputs, acc, k_iter, k_count, residue, int(threadIdx.x), smem_buf);
        return acc;
    }

    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        using namespace cute;
        constexpr int M = DG_GEMM_RS_M, N = DG_GEMM_RS_N, K = DG_GEMM_RS_K;
        constexpr int World = DG_GEMM_RS_WORLD, Rank = DG_GEMM_RS_RANK, LM = M / World;
        constexpr int BM = size<0>(TileShape{}), BN = size<1>(TileShape{});
        GemmRSState state{static_cast<const int64_t*>(params.workspace)};
        const int epoch = Pipeline ? begin(state, smem_buf) : begin_warp(state);
        auto output = reinterpret_cast<__ppu_bfloat16*>(params.epilogue.ptr_D);
        CollectiveMainloop mainloop{params.mainloop, make_shape(OwnerRows ? LM : M, N, K)};
        TiledMma mma;
        auto shape = TileShape{};
        if constexpr (Pipeline) {
            static_assert(M >= BM && LM >= 16);
            auto& transport = mainloop.gmem_tiled_copy_A;
            using Transport = std::remove_reference_t<decltype(transport)>;
            transport.staging = reinterpret_cast<const uint4*>(smem_buf + OriginalBase::SharedStorageSize);
            static_assert(OriginalBase::SharedStorageSize + Transport::SharedPacketCount * size(TiledMma{}) * 16 <=
                          DG_GEMM_RS_SHARED_BUDGET);
        }
        auto scheduler_params = params.scheduler;
        scheduler_params.shape_m = M;
        scheduler_params.shape_n = N;
        scheduler_params.shape_k = K;
        Scheduler scheduler{scheduler_params};
        uint32_t mt, nt;
        while (scheduler.fetch_next_work(mt, nt)) {
            mt = owner_balanced_m_tile<M, World, Rank, BM>(mt);
            auto acc = compute_tile(params, mainloop, mma, mt, nt, smem_buf);
            if constexpr (Pipeline) {
                auto& transport = mainloop.gmem_tiled_copy_A;
                transport.flush();
                transport.clear_pending();
                stage_remote(acc, shape, state, epoch, int(mt), int(nt), transport);
            } else if constexpr (OwnerRows) {
                store_owned_tile(acc, shape, state, epoch, int(mt) / OwnerTiles, int(mt) % OwnerTiles, int(nt));
            } else {
#if __HGGC_ARCH__ == 150
                constexpr int OutputOffset = DG_GEMM_RS_SHARED_BUDGET - BM * BN * 2;
                if constexpr (M <= BM && is_single_owner_packet(World, LM, BN) &&
                              OutputOffset >= OriginalBase::SharedStorageSize) {
                    // This storage was never read by the input pipeline.
                    auto scratch = smem_buf + OutputOffset;
                    compact_stage<M, N, BM, BN>(scratch, acc, mma, shape, int(mt), int(nt));
                    compact_emit<M, N, World, Rank, BM, BN>(state, epoch, scratch, int(mt), int(nt));
                } else
#endif
                store_packets<M, N, World, Rank>(state, epoch, acc, mma, shape, int(mt), int(nt), smem_buf);
            }
        }
        if constexpr (Pipeline)
            mainloop.gmem_tiled_copy_A.flush();
#if __HGGC_ARCH__ == 150
        if constexpr (!Pipeline && single_packet_scratch<M, N, World, BM, BN>()) {
            constexpr int OutputOffset = DG_GEMM_RS_SHARED_BUDGET - BM * BN * 2;
            auto scratch = smem_buf + (OutputOffset >= OriginalBase::SharedStorageSize ? OutputOffset : 0);
            finish_column_tail<M, N, World, Rank, BM, BN, true>(state, epoch, output, scratch);
            return;
        }
#endif
        if constexpr (OwnerRows)
            finish<M, N, World, Rank, BM, BN, false, true>(state, epoch, output);
        else
            finish_tiles<M, N, World, Rank, BM, BN>(state, epoch, output);
    }
};
} // namespace deep_gemm::gemm_comm
