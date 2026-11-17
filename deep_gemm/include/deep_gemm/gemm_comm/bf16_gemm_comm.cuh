#pragma once

namespace deep_gemm::gemm_comm {

// Shared host/device launch payload; dimensions and contiguous strides are
// already part of the fused runtime's compile key and generated definitions.
struct Bf16GemmRSParams {
    const void* ptr_A;
    const void* ptr_B;
    void* ptr_D;
    void* workspace;
};
} // namespace deep_gemm::gemm_comm

#if defined(__HGGC_ARCH__) && defined(DG_GEMM_RS_M)

#include <deep_gemm/gemm_comm/reduce_scatter.cuh>

#if __HGGC_ARCH__ == 150
#include <cutlass/gemm/config/gemm_configs.hpp>
#define __syncthreads() deep_gemm::gemm_comm::mainloop_sync()
#include <deep_gemm/impls/bf16_gemm_cute_free.cuh>
#undef __syncthreads
#else
#include <deep_gemm/impls/bf16_gemm_cutlass3.cuh>
#endif

namespace deep_gemm::gemm_comm {

#if __HGGC_ARCH__ == 150
// Native CuteFree warps own contiguous WM/WN rectangles. Packet coordinates
// are expressed one MMA atom at a time, independently of the Cutlass warp tiler.
using Bf16PacketAtom = typename cutlass::gemm::config::GetAiuMmaInst<cutlass::arch::PPU0015, cutlass::bfloat16_t,
                                                                     cutlass::bfloat16_t, float>::type;
struct Bf16PacketMma {
    CUTLASS_DEVICE auto get_slice(int) const {
        return cute::make_tiled_mma(cute::MMA_Atom<Bf16PacketAtom>{}).get_slice(0);
    }
};
#endif

#if __HGGC_ARCH__ == 150
// Expose the native warp rectangle through the common packet reducer view.
template <int WM, int WN, int WarpsN> struct Bf16RectangleMma {
    struct Coordinates {
        int row, col;
        CUTLASS_DEVICE auto operator()(int value) const {
            const int packet = value / 8;
            return cute::make_coord(row + packet / (WN / 16) * 16, col + packet % (WN / 16) * 16);
        }
    };
    struct Slice {
        int warp;
        template <class Tensor> CUTLASS_DEVICE auto partition_C(Tensor const&) const {
            return Coordinates{warp / WarpsN * WM, warp % WarpsN * WN};
        }
    };
    CUTLASS_DEVICE auto get_slice(int thread) const { return Slice{thread / 32}; }
};

struct Bf16NoSend {
    CUTLASS_DEVICE void progress() const {}
    CUTLASS_DEVICE void flush() const {}
};
#endif

// A complete native warp rectangle has one row owner. Cache only its
// immutable destination and phase stride for packet emission, after MMA.
struct Bf16PacketStorage {
    int64_t destination, phase_stride;
    CUTLASS_DEVICE int64_t peer(int) const { return destination; }
    CUTLASS_DEVICE int64_t phase_elements() const { return phase_stride; }
};

// Reuse the native BF16 mainloops. Communication replaces only
// the output epilogue; each source rounds its complete partial to BF16 first.
#if __HGGC_ARCH__ == 150
template <int BM, int BN, int BK, int WM, int WN, int WK, int Stages,
          bool DenseS2Opt, bool InputOverlap, bool AlignedN, bool PackedStore,
          class GemmTypeTag, class ElementOutput = cutlass::bfloat16_t,
          int BCachePolicy = 1, bool NonPersistent = false>
#else
template <class ProblemShape_, class Mainloop_, class Epilogue_, class Scheduler_, bool Overlap = false>
#endif
class Bf16GemmRS {
  public:
#if __HGGC_ARCH__ == 150
    using Base = deep_gemm::BF16GemmCuteFreeKernel<BM, BN, BK, WM, WN, WK, Stages,
        DenseS2Opt, InputOverlap, AlignedN, PackedStore, GemmTypeTag, ElementOutput, BCachePolicy, NonPersistent>;
    static_assert(std::is_same_v<ElementOutput, cutlass::bfloat16_t> && !NonPersistent);
    using Params = Bf16GemmRSParams;
    using TileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<BK>>;
#else
    using Base = cutlass::gemm::kernel::DeepGemmUniversal<ProblemShape_, Mainloop_, Epilogue_, Scheduler_, Overlap>;
    using Params = typename Base::Params;
    using TileShape = typename Base::TileShape;
    using TiledMma = typename Base::TiledMma;
    using CollectiveMainloop = Mainloop_;
    using ElementCompute = typename Epilogue_::ElementCompute;
#endif
    using SharedStorage = typename Base::SharedStorage;
    static constexpr int SharedStorageSize = Base::SharedStorageSize;
    static constexpr int MaxThreadsPerBlock = Base::MaxThreadsPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = Base::MinBlocksPerMultiprocessor;

    CUTLASS_DEVICE void operator()(Params const &params, char *smem) {
        using namespace cute;
        constexpr int M = DG_GEMM_RS_M, N = DG_GEMM_RS_N, K = DG_GEMM_RS_K;
        constexpr int World = DG_GEMM_RS_WORLD, Rank = DG_GEMM_RS_RANK;
        const int thread = threadIdx.x, warp = cutlass::canonical_warp_idx_sync();
        GemmRSState state{static_cast<const int64_t *>(params.workspace)};
        const int epoch = begin_epoch(state, smem);
#if __HGGC_ARCH__ == 150
        DenseGemmScheduler<BM, BN> scheduler{DenseGemmTileSchedulerArguments{M, N, K, nullptr}};
        using Native = Base;
        using Mainloop = typename Native::Mainloop;
        constexpr bool Compact = World > 2 && !GemmRSPackets<M / World>::Aligned && M <= BM && M >= GemmRSCompactMinM;
        // Stage zero of A/B may be prefetched while retired later stages
        // temporarily hold compact output. Preserve the native overlap window.
        constexpr int AStageBytes = BM * BK * 2, BStageBytes = BN * BK * 2;
        constexpr int OutputBytes = BM * BN * 2;
        constexpr bool FitsA = OutputBytes <= (Stages - 1) * AStageBytes;
        constexpr bool FitsB = OutputBytes <= (Stages - 1) * BStageBytes;
        constexpr bool CompactPrefetch = InputOverlap && Compact && (FitsA || FitsB);
        constexpr int OutputOffset = !CompactPrefetch ? 0 : (FitsA ? AStageBytes : Stages * AStageBytes + BStageBytes);
        char* output_smem = smem + OutputOffset;
        static_assert(!Compact || OutputOffset + OutputBytes <= DG_GEMM_RS_SHARED_BUDGET);
        constexpr int SharedPackets = DG_BF16_RS_SHARED_PACKETS;
        constexpr bool SendPipeline = M % BM == 0 && (M / World) % WM == 0 &&
                                      M / World >= 16 && SharedPackets > 0;
        using Sender = std::conditional_t<SendPipeline,
            GemmRSSendCopy<Bf16NoSend, Bf16RectangleMma<WM, WN, Native::kWarpsN>, TileShape,
                           SharedPackets, N, Native::kWarpsMN * 32, (M % (World * BM) == 0 ? 0 : M / World), N % BN != 0>, Bf16NoSend>;
        Sender sender;
        if constexpr (SendPipeline)
            sender.staging = reinterpret_cast<uint4*>(smem + Base::SharedStorageSize);
        typename Mainloop::Params ml{static_cast<const cutlass::bfloat16_t*>(params.ptr_A),
                                     static_cast<const cutlass::bfloat16_t*>(params.ptr_B), M, N, K, K, K};
        const int warp_mn = warp % Native::kWarpsMN, warp_k = warp / Native::kWarpsMN;
        const int warp_row = warp_mn / Native::kWarpsN, warp_col = warp_mn % Native::kWarpsN;
        uint32_t mt = 0, nt = 0;
        bool tile_valid = false;
        if constexpr (InputOverlap) {
            tile_valid = scheduler.fetch_next_work(mt, nt);
            if (tile_valid)
                Mainloop::issue_prologue_stage0(smem, warp, ml, int(owner_balanced_m_tile<M, World, Rank, BM>(mt)) * BM,
                                                int(nt) * BN);
        }
        while (true) {
            if constexpr (InputOverlap) {
                if (!tile_valid)
                    break;
            } else if (!scheduler.fetch_next_work(mt, nt)) {
                break;
            }
            const auto row = owner_balanced_m_tile<M, World, Rank, BM>(mt);
            const auto col = nt;
            float accumulator[Native::kMmasPerWarpM][Native::kMmasPerWarpN][8] = {};
            Mainloop::run(ml, smem, warp, warp_row, warp_col, warp_k, int(row) * BM, int(col) * BN, accumulator, sender);
            Native::WarpKReduce::run(accumulator, reinterpret_cast<float *>(smem), warp_mn, warp_k, thread % 32);
            auto prefetch_next = [&] {
                if constexpr (InputOverlap) {
                    tile_valid = scheduler.fetch_next_work(mt, nt);
                    if (tile_valid)
                        Mainloop::issue_prologue_stage0(
                            smem, warp, ml, int(owner_balanced_m_tile<M, World, Rank, BM>(mt)) * BM, int(nt) * BN);
                }
            };
            if constexpr (!Compact)
                prefetch_next();
            else {
                compact_begin();
                if constexpr (CompactPrefetch)
                    prefetch_next();
            }
            if (warp_k == 0) {
                if constexpr (SendPipeline) {
                    const int owner = (int(row) * BM + warp_row * WM) / (M / World);
                    if (owner != Rank) {
                        sender.flush();
                        constexpr int Packets = WM * WN / 256, Warps = Native::kWarpsMN;
                        auto staging = const_cast<uint4*>(sender.staging);
                        auto acc = make_tensor(make_rmem_ptr(&accumulator[0][0][0]),
                                               make_layout(make_shape(Int<8>{}, Int<WN / 16>{}, Int<WM / 16>{})));
                        for_each(make_int_sequence<Packets>{}, [&](auto packet) {
                            constexpr int p = decltype(packet)::value;
                            uint4 value = pack_accumulator_packet(acc, p * 8);
                            if constexpr (p < SharedPackets)
                                staging[(((p / 2) * Warps + warp) * 2 + p % 2) * 32 + thread % 32] = value;
                            else
                                sender.cache[p - SharedPackets] = value;
                        });
                        sender.destination = reinterpret_cast<uint32_t*>(state.peer(owner)) +
                            int64_t((epoch - 1) & 1) * state.phase_elements() / 2 + Rank * (M / World) * N / 2 +
                            int(col) * (BN / 16) * 128;
                        if constexpr (M % (World * BM) == 0)
                            sender.destination += ((int(row) * BM) % (M / World)) / 16 * (N / 16) * 128;
                        else
                            sender.m_offset = int(row) * BM;
                        if constexpr (N % BN != 0)
                            sender.n_remaining = N - int(col) * BN;
                        sender.calls = 0;
                        continue;
                    }
                }
                auto send = [&](auto storage) {
#pragma unroll
                    for (int im = 0; im < Native::kMmasPerWarpM; ++im) {
#pragma unroll
                        for (int in = 0; in < Native::kMmasPerWarpN; ++in) {
                            auto acc = make_tensor(make_rmem_ptr(&accumulator[im][in][0]), make_layout(Int<8>{}));
                            if constexpr (Compact)
                                compact_stage<M, N, BM, BN>(output_smem, acc, Bf16PacketMma{}, make_shape(Int<16>{}, Int<16>{}, Int<16>{}),
                                                           int(row), int(col), warp_row * WM + im * 16, warp_col * WN + in * 16);
                            else
                                store_packets<M, N, World, Rank>(storage, epoch, acc, Bf16PacketMma{},
                                                             make_shape(Int<16>{}, Int<16>{}, Int<16>{}),
                                                             int(row) * (BM / 16) + warp_row * (WM / 16) + im,
                                                             int(col) * (BN / 16) + warp_col * (WN / 16) + in);
                        }
                    }
                };
                if constexpr (Compact) {
                    send(state);
                } else if constexpr (M % BM == 0 && (M / World) % WM == 0 && M / World >= 16) {
                    const int owner = (int(row) * BM + warp_row * WM) / (M / World);
                    Bf16PacketStorage storage{state.peer(owner), state.phase_elements()};
                    send(storage);
                } else {
                    send(state);
                }
            }
            if constexpr (Compact) {
                compact_emit<M, N, World, Rank, BM, BN>(state, epoch, output_smem, int(row), int(col));
                if constexpr (!CompactPrefetch)
                    prefetch_next();
            }
        }
        sender.flush();
#else
        Scheduler_ scheduler{params.scheduler};
        const auto blk_shape = TileShape{};
        const auto problem = ProblemShape_{M, N, K, 1};
        CollectiveMainloop mainloop{params.mainloop, make_shape(M, N, K)};
        uint32_t mt, nt;
        while (scheduler.fetch_next_work(mt, nt)) {
            const auto row = owner_balanced_m_tile<M, World, Rank, size<0>(TileShape{})>(mt);
            const auto *a = params.mainloop.ptr_A + scheduler.curr_offset_a();
            const auto *b = params.mainloop.ptr_B + scheduler.curr_offset_b(mt);
            auto coord = make_coord(row, nt, _, 0);
            auto inputs = mainloop.load_init(problem, coord, params.mainloop, scheduler.curr_offset_m(), 0, a, b);
            auto ga = get<0>(inputs);
            auto gb = get<1>(inputs);
            auto residue = make_tuple(M - size<0>(ga) * row, N - size<0>(gb) * nt, K - size<1>(ga) * size<2>(ga));
            TiledMma mma;
            auto acc = make_fragment_like<ElementCompute>(partition_fragment_C(mma, take<0, 2>(blk_shape)));
            clear(acc);
            mainloop(acc, inputs, acc, make_coord_iterator(shape<2>(ga)), size<2>(ga), residue, thread, warp, smem);
            store_packets<M, N, World, Rank>(state, epoch, acc, mma, blk_shape, int(row), int(nt), smem);
        }
#endif
#if __HGGC_ARCH__ == 150
        auto* output = static_cast<__ppu_bfloat16*>(params.ptr_D);
#else
        auto* output = reinterpret_cast<__ppu_bfloat16*>(params.epilogue.ptr_D);
#endif
        finish_tiles<M, N, World, Rank, size<0>(TileShape{}), size<1>(TileShape{})>(state, epoch, output);
    }
};
} // namespace deep_gemm::gemm_comm

#endif // JIT device implementation
