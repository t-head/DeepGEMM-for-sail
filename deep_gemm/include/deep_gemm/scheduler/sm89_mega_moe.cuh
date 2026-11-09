#pragma once

#include <type_traits>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>

namespace deep_gemm::sched {

// Tile ordering is independent of operand type and the expert-row layout.
template <uint32_t NumNBlocks, uint32_t NumRanks, uint32_t GroupSpan = 2>
struct MegaMoETileOrder {
#if defined(__HGGC__)
    CUTLASS_DEVICE static void get_block_idx(
        uint32_t num_m_blocks, uint32_t block_idx,
        uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t blocks_per_group = NumNBlocks * GroupSpan;
        const uint32_t first = block_idx / blocks_per_group * GroupSpan;
        const uint32_t in_group = block_idx % blocks_per_group;
        const uint32_t rows = min(GroupSpan, num_m_blocks - first);
        m_block_idx = first + in_group % rows;
        n_block_idx = in_group / rows;
        if constexpr (NumRanks == 8) {
            if (num_m_blocks >= 16) {
                const uint32_t bits = 32u - __clz(num_m_blocks - 1u);
                const uint32_t input = bits >= 2 && num_m_blocks == (3u << (bits - 2))
                    ? m_block_idx + m_block_idx / 3u : m_block_idx;
                const uint32_t permuted = __brev(input) >> (32u - bits);
                if (permuted < num_m_blocks)
                    m_block_idx = permuted;
            }
        }
    }
#endif
};

// The caller owns ticket publication and retirement: ordinary fetch retires
// operand storage first, while fused epilogues may prefetch the next ticket.
struct MegaMoEWorkQueue {
    uint32_t* counter;
    uint32_t* shared_ticket;

#if defined(__HGGC__)
    CUTLASS_DEVICE void claim(uint32_t thread, uint32_t count = 1) const {
        if (thread == 0) *shared_ticket = atomicAdd(counter, count);
    }
    CUTLASS_DEVICE void prime(uint32_t thread, uint32_t count = 1) const {
        claim(thread, count);
        __syncthreads();
    }
    CUTLASS_DEVICE uint32_t current() const { return *shared_ticket; }
#endif
};

struct IndexedMegaMoEBlocks {
    static constexpr bool kCacheLastExpert = false;
    const int* layout;
    uint32_t num_m_blocks;

#if defined(__HGGC__)
    template <class State>
    CUTLASS_DEVICE bool get_block(uint32_t block, State&, uint4& entry) const {
        if (block >= num_m_blocks) return false;
        entry = (reinterpret_cast<const uint4*>(layout) + 1)[block];
        return true;
    }
#endif
};

// Each warp owns a distributed histogram. Pull and compute use the same
// expert-to-pool mapping without allocating another block metadata table.
template <uint32_t Experts, uint32_t BlockM>
struct DistributedMegaMoEBlocks {
    static constexpr bool kCacheLastExpert = true;
    static constexpr uint32_t kChunks = math::ceil_div(Experts, 32u);
    uint32_t rows[kChunks], begin[kChunks], end[kChunks];
    uint32_t total_blocks = 0;

#if defined(__HGGC__)
    CUTLASS_DEVICE DistributedMegaMoEBlocks(const uint64_t* counts, uint32_t lane) {
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
            const uint32_t expert = chunk * 32 + lane;
            rows[chunk] = expert < Experts ? uint32_t(counts[expert]) : 0;
            const uint32_t blocks = math::ceil_div(rows[chunk], BlockM);
            uint32_t prefix = blocks;
            #pragma unroll
            for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                const uint32_t value = __shfl_up_sync(0xffffffffu, prefix, offset);
                if (lane >= offset) prefix += value;
            }
            begin[chunk] = total_blocks + prefix - blocks;
            end[chunk] = total_blocks + prefix;
            total_blocks += __shfl_sync(0xffffffffu, prefix, 31);
        }
    }

    CUTLASS_DEVICE bool get_block(uint32_t block, uint4& entry) const {
        if (block >= total_blocks) return false;
        const uint32_t lane = threadIdx.x % 32;
        uint32_t expert = Experts;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
            const uint32_t mask = __ballot_sync(0xffffffffu,
                chunk * 32 + lane < Experts && block >= begin[chunk] && block < end[chunk]);
            if (expert == Experts && mask)
                expert = chunk * 32 + uint32_t(__ffs(mask) - 1);
        }
        uint32_t expert_rows = 0, first = 0;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
            if (chunk == expert / 32) {
                expert_rows = rows[chunk];
                first = begin[chunk];
            }
        }
        expert_rows = __shfl_sync(0xffffffffu, expert_rows, expert % 32);
        first = __shfl_sync(0xffffffffu, first, expert % 32);
        entry = make_uint4(expert, expert_rows, first, first * BlockM);
        return true;
    }

    template <class State>
    CUTLASS_DEVICE bool get_block(uint32_t block, State&, uint4& entry) const {
        return get_block(block, entry);
    }
#endif
};

// Expert rows may already be padded, or may require separate transport
// padding. The monotonic scheduler cursor owns both logical and physical
// prefixes, including repeated offsets for empty experts.
template <uint32_t Experts, uint32_t BlockM, uint32_t PoolAlignment = 1, class Row = int32_t>
struct RowCountMegaMoEBlocks {
    static constexpr bool kCacheLastExpert = true;
    const Row* rows;

#if defined(__HGGC__)
    template <class State>
    CUTLASS_DEVICE bool get_block(uint32_t block, State& state, uint4& entry) const {
        while (state.curr_group_idx < Experts) {
            const uint32_t count = uint32_t(__ldg(rows + state.curr_group_idx));
            const uint32_t blocks = math::ceil_div(count, BlockM);
            if (block < state.curr_cumsum + blocks) {
                entry = make_uint4(state.curr_group_idx, count, state.curr_cumsum, state.curr_cumsum_m);
                return true;
            }
            ++state.curr_group_idx;
            state.curr_cumsum += blocks;
            state.curr_cumsum_m += math::ceil_div(count, PoolAlignment) * PoolAlignment;
        }
        return false;
    }
#endif
};

// Shared GEMM1 task decoding. Metadata supplies
// {expert, expert rows, first M-block, physical first row}; transport readiness
// and MMA/epilogue execution stay with the precision backend.
struct MegaMoEGemm1Scheduler {
    uint32_t curr_group_idx = 0, curr_cumsum = 0, curr_group_m = 0, curr_cumsum_m = 0;
    uint32_t curr_global_block_m_idx = 0, curr_group_m_end = 0;

#if defined(__HGGC__)
    template <uint32_t BlockM, uint32_t NumNBlocks, uint32_t GroupSpan = 1, class Metadata>
    CUTLASS_DEVICE bool select_work(uint32_t work, uint32_t& m, uint32_t& n, Metadata& metadata) {
        const uint32_t global_m = work / NumNBlocks;
        uint4 entry;
        if constexpr (Metadata::kCacheLastExpert) {
            // Queue tickets are monotonic within each CTA.
            if (global_m < curr_group_m_end)
                entry = make_uint4(curr_group_idx, curr_group_m, curr_cumsum, curr_cumsum_m);
            else if (!metadata.get_block(global_m, *this, entry))
                return false;
        } else if (!metadata.get_block(global_m, *this, entry)) {
            return false;
        }
        curr_group_idx = entry.x;
        curr_group_m = entry.y;
        curr_cumsum = entry.z;
        curr_cumsum_m = entry.w;
        if constexpr (Metadata::kCacheLastExpert)
            curr_group_m_end = entry.z + math::ceil_div(entry.y, BlockM);
        if constexpr (GroupSpan == 1) {
            m = global_m - entry.z;
            n = work - global_m * NumNBlocks;
            curr_global_block_m_idx = global_m;
        } else {
            MegaMoETileOrder<NumNBlocks, 1, GroupSpan>::get_block_idx(
                math::ceil_div(entry.y, BlockM), work - entry.z * NumNBlocks, m, n);
            curr_global_block_m_idx = entry.z + m;
        }
        return true;
    }
#endif
};

template <uint32_t N, uint32_t K, uint32_t BlockM, uint32_t BlockN, uint32_t Groups,
          uint32_t GroupSpan = 1>
struct RoutedMegaMoEGemm1Scheduler : MegaMoEGemm1Scheduler {
    using Params = TileSchedulerArguments;
    const Params& params;
#if defined(__HGGC__)
    CUTLASS_DEVICE explicit RoutedMegaMoEGemm1Scheduler(const Params& params_) : params(params_) {}
    CUTLASS_DEVICE bool select_work(uint32_t work, uint32_t& m, uint32_t& n) {
        RowCountMegaMoEBlocks<Groups, BlockM> metadata{params.grouped_layout};
        return MegaMoEGemm1Scheduler::select_work<BlockM, math::ceil_div(N, BlockN), GroupSpan>(
            work, m, n, metadata);
    }
    CUTLASS_DEVICE int64_t curr_offset_a() const { return int64_t(curr_cumsum_m) * K; }
    CUTLASS_DEVICE int64_t curr_offset_c() const { return int64_t(curr_cumsum_m) * N; }
#endif
};

struct MegaMoESchedulerArguments {
    int* grouped_layout = nullptr;
    // Copy and compute CTAs share the same M-block metadata.
    int* copy_grouped_layout = nullptr;
    uint32_t shape_m = 0;
    const uint32_t* expert_rank_prefix = nullptr;
    uint8_t* local_fp4_buf = nullptr;
    uint16_t* local_sfa_buf = nullptr;
    volatile uint32_t* copy_ready_flags = nullptr;
    uint32_t num_copy_blocks = 0;
    layout::SymBuffer<> sym_buffer{};
    layout::MegaMoEStaging staging{};
    const uint8_t* dispatch_fp4 = nullptr;
    const uint32_t* dispatch_token_topk = nullptr;
    uint32_t* expert_base = nullptr;
    layout::TokenSrcMetadata* src_info = nullptr;
    int32_t* cumulative_local_expert_recv_stats = nullptr;

    CUTLASS_HOST_DEVICE
    MegaMoESchedulerArguments(uint32_t shape_m = 0, int* grouped_layout_ptr = nullptr)
        : grouped_layout(grouped_layout_ptr), shape_m(shape_m) {}
};

// Two-stage PPU scheduler. Keep the GEMM1 tile queue and scatter traversal
// separate from the ordinary grouped GEMM scheduler.
template <uint32_t SHAPE_N_, uint32_t SHAPE_K_,
          uint32_t BLOCK_M_, uint32_t BLOCK_N_, uint32_t kNumGroups_,
          MegaMoEPhaseKind Stage, uint32_t kNumRanks_,
          uint32_t kNumCopyBlocks_ = 0, uint32_t kKTilesPerFlag_ = 0,
          uint32_t kSourceCapacity_ = 1, uint32_t kNumTopK_ = 1,
          uint32_t kGroupSpan = 2>
struct SM89MegaMoEScheduler : MegaMoEGemm1Scheduler {
    static constexpr uint32_t SHAPE_N = SHAPE_N_;
    static constexpr uint32_t SHAPE_K = SHAPE_K_;
    static constexpr uint32_t BLOCK_M = BLOCK_M_;
    static constexpr uint32_t BLOCK_N = BLOCK_N_;
    static constexpr uint32_t kNumGroups = kNumGroups_;
    static constexpr uint32_t kNumRanks = kNumRanks_;
    static constexpr uint32_t kNumCopyBlocks = kNumCopyBlocks_;
    static constexpr uint32_t kKTilesPerFlag = kKTilesPerFlag_;
    static constexpr uint32_t kSourceCapacity = kSourceCapacity_;
    static constexpr uint32_t kNumTopK = kNumTopK_;
    static constexpr GemmType GEMM_TYPE = GemmType::MegaMoE;
    static constexpr bool kIsMegaMoE = true;
    static constexpr bool kIsFusedDispatch = Stage == MegaMoEPhaseKind::Linear1;
    static constexpr bool kIsMaskedLayout = kIsFusedDispatch;
    static constexpr bool kIsNoPadLayout = !kIsFusedDispatch;
    static constexpr bool kIsNoPadPreprocessLayout = false;
    static constexpr uint32_t kNumNBlocks = math::ceil_div(SHAPE_N, BLOCK_N);
    static constexpr uint32_t SHAPE_N_OUT = EpilogueTraits<
        kIsFusedDispatch ? EpilogueType::SiluAndMulPostQuantFp4 : EpilogueType::Default
    >::get_shape_n_out(SHAPE_N);
    static constexpr uint32_t kNum1DBlocksPerGroup = !kIsFusedDispatch && kNumRanks == 8 ? 16 : 2;

    using Arguments = std::conditional_t<kIsFusedDispatch,
                                         MegaMoESchedulerArguments, TileSchedulerArguments>;
    using Params = Arguments;
    Params const& params;
    int current_iter = 0;
    uint32_t num_aligned_m_blocks;
    uint32_t num_blocks;

    template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
    static Params to_underlying_arguments(
        int* groups_layout, ProblemShapeMNKL problem_shape_mnkl,
        TileShape, ClusterShape, KernelHardwareInfo const&, Arguments const& arguments,
        void* = nullptr, const uint32_t = 1, uint32_t = 1u) {
        static_assert(cute::is_static<TileShape>::value);
        static_assert(cute::is_static<ClusterShape>::value);
        const auto problem_shape = cutlass::gemm::to_gemm_coord(problem_shape_mnkl);
        Params params = arguments;
        params.grouped_layout = groups_layout;
        params.shape_m = static_cast<uint32_t>(problem_shape.m());
        return params;
    }

    template <class ProblemShape, class ElementAccumulator>
    static size_t get_workspace_size(
        Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t,
        const uint32_t = 1, uint32_t = 1) {
        return 0;
    }

#if defined(__HGGC__)
    CUTLASS_DEVICE explicit SM89MegaMoEScheduler(Params const& params_, const int warp_group_id = 0)
        : params(params_), current_iter(warp_group_id) {
        num_aligned_m_blocks = math::ceil_div(params_.shape_m, BLOCK_M);
        if constexpr (kIsFusedDispatch) {
            // Reuse the block-granular metadata already produced for copy CTAs.
            num_aligned_m_blocks = params_.copy_grouped_layout[0];
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        }
    }

    CUTLASS_DEVICE void get_swizzled_block_idx(
        const uint32_t num_m_blocks, int block_idx,
        uint32_t& m_block_idx, uint32_t& n_block_idx) {
        MegaMoETileOrder<kNumNBlocks, kIsFusedDispatch ? 1 : kNumRanks, kGroupSpan>::get_block_idx(
            num_m_blocks, block_idx, m_block_idx, n_block_idx);
    }

    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m_block_idx, uint32_t& n_block_idx,
                                        uint32_t* ticket_storage = nullptr) {
        uint32_t next_block_idx;
        if constexpr (kIsFusedDispatch && kNumCopyBlocks > 0) {
            // Finish the previous tile before reusing GEMM shared storage for a ticket.
            __syncthreads();
            const MegaMoEWorkQueue queue{
                reinterpret_cast<uint32_t*>(params.copy_grouped_layout + 1), ticket_storage};
            queue.claim(threadIdx.x);
            __syncthreads();
            next_block_idx = *reinterpret_cast<volatile uint32_t*>(ticket_storage);
            __syncthreads();
        } else {
            next_block_idx = current_iter++ * gridDim.x + blockIdx.x;
        }
        if constexpr (kIsFusedDispatch) {
            IndexedMegaMoEBlocks metadata{params.copy_grouped_layout, num_aligned_m_blocks};
            if (!select_work<BLOCK_M, kNumNBlocks>(next_block_idx, m_block_idx, n_block_idx, metadata)) {
                m_block_idx = num_aligned_m_blocks;
                n_block_idx = kNumNBlocks;
                return false;
            }
        } else {
            uint32_t num_m_blocks;
            while (true) {
                if (curr_group_idx == kNumGroups) {
                    m_block_idx = num_aligned_m_blocks;
                    n_block_idx = kNumNBlocks;
                    return false;
                }
                curr_group_m = static_cast<uint32_t>(__ldg(params.grouped_layout + curr_group_idx));
                num_m_blocks = math::ceil_div(curr_group_m, BLOCK_M);
                const auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                    break;
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
                curr_cumsum_m += curr_group_m;
            }
            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks,
                                  m_block_idx, n_block_idx);
        }
        return true;
    }

    __device__ __forceinline__ int32_t curr_problem_m() const { return curr_group_m; }
    __device__ __forceinline__ int32_t problem_index() const { return curr_group_idx; }
    __device__ __forceinline__ int64_t curr_offset_a() const {
        return int64_t(curr_cumsum_m) * SHAPE_K;
    }
    __device__ __forceinline__ int64_t curr_offset_mxfp4_scalea() const { return int64_t(curr_cumsum_m); }
    __device__ __forceinline__ int64_t curr_offset_m() const { return curr_cumsum_m; }
    __device__ __forceinline__ int64_t curr_offset_b(const int = 0) const {
        return int64_t(curr_group_idx) * SHAPE_N * SHAPE_K;
    }
    __device__ __forceinline__ int64_t curr_offset_mxfp4_scaleb(const int = 0) const {
        const uint32_t shape_k_scale = math::ceil_div(SHAPE_K, uint32_t(32));
        return int64_t(curr_group_idx) * SHAPE_N * shape_k_scale;
    }
    __device__ __forceinline__ int64_t curr_offset_mxfp4_c() const {
        return int64_t(curr_group_idx) * SHAPE_N;
    }
    __device__ __forceinline__ int64_t curr_offset_c() const {
        return int64_t(curr_cumsum_m) * SHAPE_N_OUT;
    }
    __device__ __forceinline__ int64_t curr_offset_c_scales() const { return int64_t(curr_cumsum_m); }
#endif
};

// The routed precisions retain their physical landing-pool offsets and
// preprocessed large-expert layout, while sharing MegaMoE's G2 tile order.
template <uint32_t N, uint32_t K, uint32_t BlockM, uint32_t BlockN,
          uint32_t Groups, uint32_t Ranks, uint32_t GroupSpan = 2>
struct RoutedMegaMoEScheduler
    : SM89MegaMoEScheduler<N, K, BlockM, BlockN, Groups,
                          MegaMoEPhaseKind::Linear2, Ranks, 0, 0, 1, 1, GroupSpan> {
    using Base = SM89MegaMoEScheduler<N, K, BlockM, BlockN, Groups,
                                    MegaMoEPhaseKind::Linear2, Ranks, 0, 0, 1, 1, GroupSpan>;
    using Params = typename Base::Params;
    static constexpr GemmType GEMM_TYPE = GemmType::MGroupedNoPad;
    static constexpr bool kIsMegaMoE = false;  // Uses the routed transport, not FP4's fused copy protocol.
    static constexpr bool kIsNoPadPreprocessLayout = Groups >= 128;
    static constexpr bool kIsTMAMulticastOnA = false;

#if defined(__HGGC__)
    CUTLASS_DEVICE explicit RoutedMegaMoEScheduler(const Params& params, int warp_group = 0)
        : Base(params, warp_group) {
        if constexpr (kIsNoPadPreprocessLayout) {
            this->num_aligned_m_blocks = params.grouped_layout[0];
            this->num_blocks = this->num_aligned_m_blocks * Base::kNumNBlocks;
        }
    }

    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m, uint32_t& n) {
        if constexpr (kIsNoPadPreprocessLayout) {
            const uint32_t work = this->current_iter++ * gridDim.x + blockIdx.x;
            if (work >= this->num_blocks)
                return false;
            const uint4 info = (reinterpret_cast<const uint4*>(this->params.grouped_layout) + 1)
                [work / Base::kNumNBlocks];
            this->curr_group_idx = info.x;
            this->curr_group_m = info.y;
            this->curr_cumsum_m = info.w;
            this->get_swizzled_block_idx(math::ceil_div(info.y, BlockM),
                work - info.z * Base::kNumNBlocks, m, n);
            return true;
        } else {
            return Base::fetch_next_work(m, n);
        }
    }

    CUTLASS_DEVICE int64_t curr_offset_a(int64_t = -1) const { return Base::curr_offset_a(); }
    CUTLASS_DEVICE int64_t curr_offset_scalea() const { return Base::curr_offset_m(); }
    CUTLASS_DEVICE int64_t curr_offset_b(int = 0, int64_t = int64_t(N) * K) const {
        return Base::curr_offset_b();
    }
    CUTLASS_DEVICE int64_t curr_offset_c(int64_t = 0) const { return Base::curr_offset_c(); }
#endif
};

}  // namespace deep_gemm::sched
