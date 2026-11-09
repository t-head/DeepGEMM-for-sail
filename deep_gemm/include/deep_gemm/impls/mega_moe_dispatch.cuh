#pragma once

#include <deep_gemm/impls/mega_moe_pre_dispatch.cuh>
#include <deep_gemm/scheduler/mega_moe_common.cuh>

namespace deep_gemm {

// Copy and scale reshape use the same rank prefixes and token slots.
template <uint32_t NumRanks, uint32_t SourceCapacity, uint32_t TopK>
__device__ __forceinline__ layout::TokenSrcMetadata load_route_source(
    const uint32_t* tokens, const uint32_t* rank_prefix, uint32_t expert, uint32_t row) {
    static_assert(is_supported_megamoe_rank_count(NumRanks));
    uint32_t rank = 0;
    for (uint32_t step = NumRanks / 2; step; step /= 2) {
        const uint32_t next = rank + step;
        if (row >= rank_prefix[next - 1]) rank = next;
    }
    const uint32_t begin = rank ? rank_prefix[rank - 1] : 0;
    const uint64_t slot = (uint64_t(expert) * NumRanks + rank) * SourceCapacity + row - begin;
    const uint32_t route = __ldbl(tokens + slot);
    return {rank, route / TopK, route % TopK};
}

template <class Scheduler>
struct Stage1RouteLayout {
    using Arguments = typename Scheduler::Arguments;
    static constexpr bool kInt8 = std::is_same_v<typename Arguments::ScaleType, float>;
    static constexpr uint32_t kHidden = Scheduler::SHAPE_K * (kInt8 ? 1 : 2);
    using ScaleFormat = MegaMoEScaleFormat<kInt8>;
    using Vector = typename ScaleFormat::Vector;
    static constexpr uint32_t kVectors = ScaleFormat::vectors(kHidden);
    using Buffers = layout::MegaMoEBufferLayout<kInt8>;
};

// The owner builds rank prefixes and expert layouts for copy and compute.
template <class Scheduler, uint32_t NumThreads>
__device__ __forceinline__ void prepare_stage1_metadata(
    const typename Scheduler::Arguments& sched, layout::MegaMoESignals* signals,
    const typename Stage1RouteLayout<Scheduler>::Buffers& buffers,
    char* smem, uint32_t generation, uint32_t num_compute_ctas = 0) {
    constexpr uint32_t BlockM = Scheduler::BLOCK_M, NumRanks = Scheduler::kNumRanks;
    constexpr uint32_t NumLocalExperts = Scheduler::kNumGroups, SourceCapacity = Scheduler::kSourceCapacity;
    constexpr bool ReshapeSfa = !Stage1RouteLayout<Scheduler>::kInt8;
    auto* smem_workspace = reinterpret_cast<uint32_t*>(smem);
    auto* __restrict__ local_pushed_counts =
        layout::bank_data<uint32_t>(buffers.staging_counts, signals, generation & 1u);

    const auto* arrival_slots = signals->dispatch_arrivals;
    auto* __restrict__ copy_grouped_layout = sched.copy_grouped_layout;
    auto* __restrict__ expert_rank_prefix = const_cast<uint32_t*>(sched.expert_rank_prefix);
    auto* __restrict__ expert_base = sched.expert_base;
    auto* __restrict__ masked_m = reinterpret_cast<uint32_t*>(sched.grouped_layout);
    auto* cumulative_local_expert_recv_stats = sched.cumulative_local_expert_recv_stats;
    auto* sfa_ready_flags = signals->preprocess_ready + 1;

    // Generation tags avoid clearing every round. Reset all allocation aliases
    // before the first metadata publication, including FP4's K-partition flags.
    if (generation == 2u) {
        const uint32_t num_copy_ready_flags = sched.staging.tokens / (16 * sizeof(uint32_t));
        for (uint32_t i = threadIdx.x; i < num_copy_ready_flags; i += NumThreads)
            __stbl(const_cast<uint32_t*>(sched.copy_ready_flags + i), 0u);
        constexpr uint32_t kNumPreprocessReadyFlags =
            layout::MegaMoESignals::kNumMaxExperts * FusedSfaPartitionLayout::kNumPartitions;
        for (uint32_t i = threadIdx.x; i < kNumPreprocessReadyFlags; i += NumThreads)
            __stbl(sfa_ready_flags + i, 0u);
    }

    constexpr uint32_t total_pairs = NumLocalExperts * NumRanks;
    constexpr uint32_t kSfaPartitions = FusedSfaPartitionLayout::kNumPartitions;
    uint32_t* pair_counts = smem_workspace;
    uint32_t* expert_m_blocks = pair_counts + total_pairs;
    uint32_t* expert_block_offsets = expert_m_blocks + NumLocalExperts;
    uint32_t* expert_row_offsets = expert_block_offsets + NumLocalExperts;
    uint32_t* expert_counts = expert_row_offsets + NumLocalExperts;

    __syncthreads();
    // All routing paths leave counts in source-rank order.
    #pragma unroll
    for (uint32_t tid = threadIdx.x; tid < total_pairs; tid += NumThreads)
        pair_counts[tid] = comm::Stage1Count<SourceCapacity>::read(
            local_pushed_counts + tid, generation);
    __syncthreads();

    // Write up to four rank prefixes at a time, including EP1 and EP2.
    constexpr uint32_t kCountVecElements = NumRanks < 4 ? NumRanks : 4;
    struct alignas(kCountVecElements * sizeof(uint32_t)) CountVec {
        uint32_t data[kCountVecElements];
    };
    for (uint32_t e = threadIdx.x; e < NumLocalExperts; e += NumThreads) {
        uint32_t total = 0;
        auto* prefix = reinterpret_cast<CountVec*>(expert_rank_prefix + e * NumRanks);
        const auto* counts = pair_counts + e;
        #pragma unroll
        for (uint32_t rank = 0; rank < NumRanks; rank += kCountVecElements) {
            CountVec values;
            #pragma unroll
            for (uint32_t r = 0; r < kCountVecElements; ++r) {
                total += counts[(rank + r) * NumLocalExperts];
                values.data[r] = total;
            }
            prefix[rank / kCountVecElements] = values;
        }
        masked_m[e] = total;
        expert_counts[e] = total;
        expert_m_blocks[e] = (total + BlockM - 1) / BlockM;
        if (cumulative_local_expert_recv_stats)
            atomicAdd(cumulative_local_expert_recv_stats + e, static_cast<int32_t>(total));
    }
    __syncthreads();

    uint32_t& share_rows = smem_workspace[0];
    // Scan experts a warp at a time, keeping both block and row offsets.
    if (threadIdx.x < 32) {
        uint32_t lane = threadIdx.x;
        uint32_t block_prefix = 0;
        uint32_t row_prefix = 0;
        uint32_t max_rows = 0;
        for (uint32_t base = 0; base < NumLocalExperts; base += 32) {
            uint32_t e = base + lane;
            uint32_t blocks = (e < NumLocalExperts)
                ? expert_m_blocks[e] : 0u;
            uint32_t rows = (e < NumLocalExperts)
                ? expert_counts[e] : 0u;
            if constexpr (ReshapeSfa) max_rows = max(max_rows, rows);
            uint32_t block_scan = blocks;
            uint32_t row_scan = rows;
            #pragma unroll
            for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                uint32_t prior_blocks = __shfl_up_sync(0xffffffff, block_scan, offset);
                uint32_t prior_rows = __shfl_up_sync(0xffffffff, row_scan, offset);
                if (lane >= offset) {
                    block_scan += prior_blocks;
                    row_scan += prior_rows;
                }
            }
            if (e < NumLocalExperts) {
                expert_block_offsets[e] = block_prefix + block_scan - blocks;
                expert_row_offsets[e] = row_prefix + row_scan - rows;
            }
            block_prefix += __shfl_sync(0xffffffff, block_scan, 31);
            row_prefix += __shfl_sync(0xffffffff, row_scan, 31);
        }
        const bool multiple_m_blocks = block_prefix > NumLocalExperts;
        if (ReshapeSfa && multiple_m_blocks) {
            #pragma unroll
            for (uint32_t delta = 16; delta > 0; delta >>= 1)
                max_rows = max(max_rows, __shfl_down_sync(0xffffffffu, max_rows, delta));
        }
        if (lane == 0) {
            // Share rows only when a K partition exceeds twice the ideal work per CTA.
            share_rows = ReshapeSfa && multiple_m_blocks &&
                uint64_t(max_rows) * num_compute_ctas > uint64_t(2) * kSfaPartitions * row_prefix;
            copy_grouped_layout[0] = block_prefix;
            if constexpr (ReshapeSfa) copy_grouped_layout[2] = share_rows;
        }
    }
    __syncthreads();

    if (ReshapeSfa && share_rows) {
        // The last completed row chunk brings the first partition tag to generation.
        for (uint32_t work = threadIdx.x; work < NumLocalExperts * kSfaPartitions; work += NumThreads)
            __stbl(sfa_ready_flags + work, generation - expert_m_blocks[work / kSfaPartitions]);
    }

    for (uint32_t e = threadIdx.x; e < NumLocalExperts;
         e += NumThreads) {
        // Entry: expert, token count, first M-block, first row.
        uint4 entry = {e, expert_counts[e], expert_block_offsets[e], expert_row_offsets[e]};
        expert_base[e] = entry.w;
        if (e + 1 == NumLocalExperts)
            expert_base[NumLocalExperts] = entry.w + entry.y;
        for (uint32_t mb = 0; mb < expert_m_blocks[e]; ++mb)
            reinterpret_cast<uint4*>(copy_grouped_layout + 4)[entry.z + mb] = entry;
    }

    // FP4 compute CTAs consume a shared tile queue; INT8 traverses expert rows.
    if constexpr (ReshapeSfa)
        if (threadIdx.x == 0) copy_grouped_layout[1] = 0;
    // Counts can arrive early; payload and token metadata still need every source release.
    comm::rank_barrier<NumRanks>(arrival_slots, generation);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        __stbl(signals->preprocess_ready, kPreprocessMetadataReadyBit | generation);
}

__device__ __forceinline__ void fence_route_writes() {
    __syncwarp();
    if (threadIdx.x % 32 == 0) __ppu_threadfence_system_uncache();
    __syncthreads();
}

template <uint32_t Threads>
__device__ __forceinline__ void clear_route_counts(uint32_t* counts, uint32_t first, uint32_t end) {
    for (uint32_t e = first + threadIdx.x; e < end; e += Threads) counts[e] = 0;
    __syncthreads();
}

template <uint32_t Threads, uint32_t Capacity>
__device__ __forceinline__ void write_route_counts(
    uint32_t* dst, const uint32_t* counts, uint32_t first, uint32_t end, uint32_t generation) {
    for (uint32_t e = first + threadIdx.x; e < end; e += Threads)
        __stbl(dst + e, comm::Stage1Count<Capacity>::encode(counts[e], generation));
}

template <uint32_t Ranks, uint32_t Capacity, uint32_t Staged>
__device__ __forceinline__ void append_peer_route(
    uint32_t* dst, uint32_t source, uint32_t expert, uint32_t route,
    uint32_t* counts, uint32_t* routes) {
    const uint32_t slot = atomicAdd(counts + expert, 1u);
    if (slot < Staged)
        routes[expert * Staged + slot] = route;
    else if (slot < Capacity)
        __stbl(dst + (uint64_t(expert) * Ranks + source) * Capacity + slot, route);
}

template <uint32_t Ranks, uint32_t Capacity, uint32_t Staged, uint32_t Threads>
__device__ __forceinline__ void flush_peer_routes(
    uint32_t* dst, uint32_t source, uint32_t first_expert, uint32_t end_expert,
    const uint32_t* counts, const uint32_t* routes) {
    // Each warp writes one expert's slots in order.
    for (uint32_t e = first_expert + threadIdx.x / 32; e < end_expert; e += Threads / 32) {
        auto* slots = dst + (uint64_t(e) * Ranks + source) * Capacity;
        const uint32_t count = min(counts[e], Staged);
        for (uint32_t i = threadIdx.x % 32; i < count; i += 32)
            __stbl(slots + i, routes[e * Staged + i]);
    }
}

// Each peer's CTAs split experts and scale rows in the same pass.
template <class Scheduler, uint32_t Threads, uint32_t SharedBytes, uint32_t MetadataCTAs>
__device__ __forceinline__ void publish_routes_by_peer(
    const typename Scheduler::Arguments& sched, char* smem, uint32_t generation) {
    constexpr uint32_t Ranks = Scheduler::kNumRanks, Groups = Scheduler::kNumGroups;
    constexpr uint32_t TopK = Scheduler::kNumTopK, Capacity = Scheduler::kSourceCapacity;
    using RouteLayout = Stage1RouteLayout<Scheduler>;
    using Vector = typename RouteLayout::Vector;
    constexpr uint32_t Hidden = RouteLayout::kHidden, Vectors = RouteLayout::kVectors;
    constexpr uint32_t VectorLanes = Vectors < 32 ? Vectors : 32;
    constexpr uint32_t RowLanes = math::next_power_of_two(VectorLanes > TopK ? VectorLanes : TopK);
    static_assert(TopK <= 32 && Threads % 32 == 0 && SharedBytes / sizeof(uint32_t) >= Groups);
    constexpr uint32_t room = (SharedBytes / sizeof(uint32_t) - Groups) / Groups;
    constexpr uint32_t Staged = room < Capacity ? room : Capacity;
    const uint32_t n = sched.num_source_tokens, worker = blockIdx.x - MetadataCTAs;
    const uint32_t route_ctas = gridDim.x - MetadataCTAs;
    const uint32_t parts = max(1u, min(route_ctas / Ranks, math::ceil_div(n, Threads / RowLanes)));
    const uint32_t source = sched.sym_buffer.rank_idx, parity = generation & 1u;
    const typename RouteLayout::Buffers buffers(Ranks, Ranks * Groups, Capacity, TopK, Hidden,
                                               Scheduler::SHAPE_N / 2, sched.staging);
    auto* base = sched.sym_buffer.get_base_ptr();
    const auto* ids = buffers.topk_idx.template data<int64_t>(base);
    const auto* scales = RouteLayout::ScaleFormat::source_scales(buffers, base, source, Capacity);
    auto* histogram = reinterpret_cast<uint32_t*>(smem);
    auto* routes = histogram + Groups;
    const uint32_t lane = threadIdx.x % RowLanes;
    const uint32_t mask = (0xffffffffu >> (32 - RowLanes)) << (threadIdx.x % 32 - lane);
    for (uint32_t job = worker; job < Ranks * parts; job += route_ctas) {
        const uint32_t peer = (source + job) % Ranks, part = job / Ranks;
        const uint32_t first = part * Groups / parts, end = (part + 1) * Groups / parts;
        const uint32_t row_count = math::ceil_div(n, parts);
        const uint32_t first_row = part * row_count, end_row = min(n, first_row + row_count);
        auto* tokens = sched.sym_buffer.map(layout::bank_data<uint32_t>(buffers.staging_tokens, base, parity), peer);
        auto* dst_scales = sched.sym_buffer.map(buffers.staging_scales[1].template data<Vector>(base), peer)
                           + uint64_t(source) * Capacity * Vectors;
        clear_route_counts<Threads>(histogram, first, end);
        for (uint32_t row = 0; row < n; row += Threads / RowLanes) {
            const uint32_t token = row + threadIdx.x / RowLanes, route = token * TopK + lane;
            const uint32_t id = token < n && lane < TopK ? uint32_t(__ldg(ids + route)) : ~0u;
            const uint32_t expert = id - peer * Groups;
            const uint32_t selected = __ballot_sync(0xffffffffu, expert < Groups) & mask;
            if (token >= first_row && token < end_row && selected) {
                for (uint32_t v = lane; v < Vectors; v += RowLanes) {
                    const uint64_t offset = uint64_t(token) * Vectors + v;
                    __stbl(dst_scales + offset, __ldbl(scales + offset));
                }
            }
            if (expert >= first && expert < end)
                append_peer_route<Ranks, Capacity, Staged>(tokens, source, expert, route, histogram, routes);
        }
        __syncthreads();
        auto* dst_counts = sched.sym_buffer.map(layout::bank_data<uint32_t>(buffers.staging_counts, base, parity), peer)
                           + source * Groups;
        write_route_counts<Threads, Capacity>(dst_counts, histogram, first, end, generation);
        flush_peer_routes<Ranks, Capacity, Staged, Threads>(tokens, source, first, end, histogram, routes);
        fence_route_writes();
        if (threadIdx.x == 0) {
            auto* retired = layout::bank_data<uint32_t>(buffers.dispatch_counts, base, parity) + peer;
            if (atomicInc(retired, parts - 1) == parts - 1)
                __stbl(sched.sym_buffer.map(reinterpret_cast<layout::MegaMoESignals*>(base)->dispatch_arrivals, peer)
                       + source, generation);
        }
        __syncthreads();
    }
}

// A warp assigns one token's slots and pushes its scales once per destination.
template <class Scheduler, uint32_t Threads, uint32_t MetadataCTAs>
__device__ __forceinline__ void publish_routes_by_token(
    const typename Scheduler::Arguments& sched, char* smem, uint32_t generation) {
    constexpr uint32_t Ranks = Scheduler::kNumRanks, Groups = Scheduler::kNumGroups;
    constexpr uint32_t TopK = Scheduler::kNumTopK, Capacity = Scheduler::kSourceCapacity;
    using RouteLayout = Stage1RouteLayout<Scheduler>;
    using Vector = typename RouteLayout::Vector;
    constexpr uint32_t Hidden = RouteLayout::kHidden, Vectors = RouteLayout::kVectors, Warps = Threads / 32;
    static_assert(TopK <= 32 && Threads % 32 == 0);
    const uint32_t n = sched.num_source_tokens, worker = blockIdx.x - MetadataCTAs;
    const uint32_t route_ctas = gridDim.x - MetadataCTAs;
    const uint32_t token_budget = stage1_token_cta_budget(route_ctas, Ranks);
    const uint32_t workers = min(token_budget, max(1u, math::ceil_div(n, Warps)));
    // Avoid adding a second wave of counts work behind the token writers.
    const uint32_t count_workers = route_ctas - token_budget;
    const uint32_t participants = workers + count_workers;
    if (worker >= participants) return;
    const typename RouteLayout::Buffers buffers(Ranks, Ranks * Groups, Capacity, TopK, Hidden,
                                        Scheduler::SHAPE_N / 2, sched.staging);
    auto* base = sched.sym_buffer.get_base_ptr();
    const uint32_t parity = generation & 1u, source = sched.sym_buffer.rank_idx;
    const auto* ids = buffers.topk_idx.template data<int64_t>(base);
    auto* counters = layout::bank_data<uint32_t>(buffers.dispatch_counts, base, parity);
    auto* counts = layout::bank_data<uint32_t>(buffers.staging_counts, base, parity);
    auto* tokens = layout::bank_data<uint32_t>(buffers.staging_tokens, base, parity);
    const auto* scales = RouteLayout::ScaleFormat::source_scales(buffers, base, source, Capacity);
    auto* received = buffers.staging_scales[1].template data<Vector>(base) + uint64_t(source) * Capacity * Vectors;
    auto* arrivals = reinterpret_cast<layout::MegaMoESignals*>(base)->dispatch_arrivals;
    auto* last = reinterpret_cast<uint32_t*>(smem);
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (worker >= workers) {
        // Spare CTAs count routes while the token warps write their rows.
        const uint32_t peer = (source + worker - workers) % Ranks;
        auto* histogram = reinterpret_cast<uint32_t*>(smem);
        clear_route_counts<Threads>(histogram, 0, Groups);
        for (uint32_t route = threadIdx.x; route < n * TopK; route += Threads) {
            const uint32_t expert = uint32_t(__ldg(ids + route)) - peer * Groups;
            if (expert < Groups) atomicAdd(histogram + expert, 1u);
        }
        __syncthreads();
        auto* dst = sched.sym_buffer.map(counts, peer) + source * Groups;
        write_route_counts<Threads, Capacity>(dst, histogram, 0, Groups, generation);
    } else {
        for (uint32_t t = worker * Warps + warp; t < n; t += workers * Warps) {
            const uint32_t route = t * TopK + lane;
            const uint32_t id = lane < TopK ? uint32_t(__ldg(ids + route)) : ~0u;
            const bool valid = id < Ranks * Groups;
            const uint32_t owner = valid ? id / Groups : Ranks;
            const uint32_t same = __match_any_sync(0xffffffffu, owner);
            const uint32_t owners = __ballot_sync(0xffffffffu, valid && !(same & __ppu_read_lanemask_lt()));
            if constexpr (Vectors == 1) {
                const Vector value = __ldbl(scales + t);
                if (owners & (1u << lane))
                    __stbl(sched.sym_buffer.map(received, owner) + t, value);
            } else {
                for (uint32_t first = 0; first < Vectors; first += 32) {
                    const uint32_t v = first + lane;
                    const uint64_t offset = uint64_t(t) * Vectors + v;
                    Vector value;
                    if (v < Vectors) value = __ldbl(scales + offset);
                    for (uint32_t mask = owners; mask; mask &= mask - 1) {
                        const uint32_t peer = __shfl_sync(0xffffffffu, owner, __ffs(mask) - 1);
                        if (v < Vectors) __stbl(sched.sym_buffer.map(received, peer) + offset, value);
                    }
                }
            }
            if (valid) {
                const uint32_t slot = atomicAdd(counters + id, 1u);
                if (slot < Capacity) {
                    const uint64_t row = (uint64_t(id % Groups) * Ranks + source) * Capacity + slot;
                    __stbl(sched.sym_buffer.map(tokens, owner) + row, route);
                }
            }
        }
    }
    fence_route_writes();
    if (!threadIdx.x) {
        auto* retire = buffers.dispatch_retire.template data<uint32_t>(base) + parity;
        *last = atomicInc(retire, participants - 1) == participants - 1;
    }
    __syncthreads();
    if (*last) {
        // Clear slot counters; publish counts here only when no CTA counted them.
        for (uint32_t id = threadIdx.x; id < Ranks * Groups; id += Threads) {
            if (count_workers) {
                __stbl(counters + id, 0u);
            } else {
                const uint32_t count = atomicExch(counters + id, 0u);
                auto* dst = sched.sym_buffer.map(counts, id / Groups) + source * Groups;
                __stbl(dst + id % Groups, comm::Stage1Count<Capacity>::encode(count, generation));
            }
        }
        if (!count_workers) fence_route_writes();
        for (uint32_t peer = threadIdx.x; peer < Ranks; peer += Threads)
            __stbl(sched.sym_buffer.map(arrivals, peer) + source, generation);
    }
    __syncthreads();
}

template <class Scheduler, uint32_t Threads, uint32_t SharedBytes, Stage1RoutePolicy Routes,
          bool MetadataInThisGrid>
__device__ __forceinline__ void publish_stage1_routes(
    const typename Scheduler::Arguments& sched, char* smem, uint32_t generation) {
    constexpr uint32_t MetadataCTAs = MetadataInThisGrid ? 1 : 0;
    if (blockIdx.x < MetadataCTAs) return;
    if constexpr (Routes == Stage1RoutePolicy::PerPeer)
        publish_routes_by_peer<Scheduler, Threads, SharedBytes, MetadataCTAs>(sched, smem, generation);
    else
        publish_routes_by_token<Scheduler, Threads, MetadataCTAs>(sched, smem, generation);
}

template <bool Int8, int Hidden, int Intermediate, int BlockM, int Groups, int Ranks,
          int Capacity, int TopK, int Threads, int SharedBytes, Stage1RoutePolicy Routes>
struct MegaMoEDispatchPrepare {
    using Arguments = sched::MegaMoESchedulerArgumentsT<std::conditional_t<Int8, float, uint16_t>>;
    static constexpr int kNumRanks = Ranks, kNumGroups = Groups;
    static constexpr int kSourceCapacity = Capacity, kNumTopK = TopK;
    static constexpr int SHAPE_N = Intermediate * 2, SHAPE_K = Int8 ? Hidden : Hidden / 2;
    static constexpr int BLOCK_M = BlockM;
    static_assert(SharedBytes >= expert_metadata_smem_bytes(Groups, Ranks));

    __device__ static void run(const Arguments& sched, char* smem, uint32_t num_compute_ctas) {
        auto* signals = sched.sym_buffer.template get_base_ptr<layout::MegaMoESignals*>();
        const uint32_t generation = comm::resolve_fused_round_generation(*signals);
        publish_stage1_routes<MegaMoEDispatchPrepare, Threads, SharedBytes, Routes, false>(sched, smem, generation);
        fence_route_writes();
        auto* last = reinterpret_cast<uint32_t*>(smem);
        if (threadIdx.x == 0)
            *last = atomicInc(signals->dispatch_retire + 2 + (generation & 1u), gridDim.x - 1) == gridDim.x - 1;
        __syncthreads();
        // Only the last publisher waits, so pending CTAs can always run.
        if (*last) {
            const layout::MegaMoEBufferLayout<Int8> buffers(
                Ranks, Ranks * Groups, Capacity, TopK, Hidden, Intermediate, sched.staging);
            prepare_stage1_metadata<MegaMoEDispatchPrepare, Threads>(
                sched, signals, buffers, smem, generation, num_compute_ctas);
        }
    }
};

}  // namespace deep_gemm
