#pragma once

namespace deep_gemm::gemm_comm {

// Device-resident immutable pointer table. Epochs/counters live separately.

enum GemmRSStateIndex { Peers, FlagPeers, Completed, PhaseElements, Rank, ChunkCapacity, GemmRSStateWords };
} // namespace deep_gemm::gemm_comm

#if defined(__HGGC__)
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/ppu_copy.hpp>
#include <cutlass/numeric_conversion.h>

#if __HGGC_ARCH__ == 150
extern "C" __device__ uint32_t __ppumath_hadd2_2bf16(uint32_t, uint32_t);
#endif

namespace deep_gemm::gemm_comm {
#if defined(DG_GEMM_RS_COMPACT_MIN_M)
constexpr int GemmRSCompactMinM = DG_GEMM_RS_COMPACT_MIN_M;
#else
// Bare-header and old compile fixtures do not provide launch geometry.
constexpr int GemmRSCompactMinM = 0;
#endif
__device__ __forceinline__ void mainloop_sync() {
    // Input readiness is established by the native pipeline's waits. Keep
    // unrelated remote output stores independent from this shared-memory join.
    __ppu_barrier_sync_nocnt(0, 1);
}

struct GemmRSState {
    const int64_t* words;
    __device__ int64_t word(int i) const { return __ldg(words + i); }
    template <class T> __device__ T* ptr(int i) const { return reinterpret_cast<T*>(word(i)); }
    __device__ int64_t phase_elements() const { return word(PhaseElements); }
    __device__ int64_t peer(int owner) const { return __ldg(ptr<int64_t>(Peers) + owner); }
};

// Resolve immutable addresses before waiting for incoming payload publication.
__device__ __forceinline__ const __ppu_bfloat16* reduction_slots(GemmRSState state, int epoch, int rank) {
    return reinterpret_cast<const __ppu_bfloat16*>(__ldcg(state.ptr<int64_t>(Peers) + rank)) +
           int64_t((epoch - 1) & 1) * state.phase_elements();
}

__device__ inline void system_fence() { asm volatile("ppu.fence.acq_rel.sys;" ::: "memory"); }
__device__ inline void publish_fence() { asm volatile("ppu.fence.release.sys;" ::: "memory"); }
__device__ inline void acquire_fence() { asm volatile("ppu.fence.acquire.sys;" ::: "memory"); }

// Every warp snapshots before the CTA's finish barrier and completion increment.
// Thus no CTA can advance the high epoch until all warp snapshots are taken.
__device__ inline int begin_warp(GemmRSState s) {
    int epoch = 0;
    if (threadIdx.x % 32 == 0) {
        unsigned long long completed;
        asm volatile("ppu.ld.relaxed.gpu.b64 %0,[%1];"
                     : "=l"(completed)
                     : "l"(s.ptr<unsigned long long>(Completed))
                     : "memory");
        epoch = int(completed >> 32) + 1;
    }
    return __shfl_sync(0xffffffffu, epoch, 0);
}

__device__ inline int begin(GemmRSState s, char* scratch) {
    auto word = reinterpret_cast<volatile int*>(scratch);
    if (threadIdx.x == 0) {
        // High32 counts completed calls; low32 counts this call's producers.
        // The epoch cannot advance until every CTA has entered and finished,
        // so this snapshot is uniform even when consecutive grids differ.
        unsigned long long completed;
        asm volatile("ppu.ld.relaxed.gpu.b64 %0,[%1];"
                     : "=l"(completed)
                     : "l"(s.ptr<unsigned long long>(Completed))
                     : "memory");
        *word = int(completed >> 32) + 1;
    }
    __syncthreads();
    const int epoch = *word;
    __syncthreads();
    return epoch;
}

__device__ inline int begin_epoch(GemmRSState state, char* scratch) {
#if __HGGC_ARCH__ == 150
    return begin_warp(state);
#else
    return begin(state, scratch);
#endif
}

// Called by lane0 after the CTA has retired its stores. All CTAs must
// snapshot the epoch before the last arrival can advance the next call.
__device__ __forceinline__ bool complete_cta(GemmRSState state, int epoch) {
    auto completed = state.ptr<unsigned long long>(Completed);
    const auto previous = atomicAdd(completed, 1ULL);
    const bool last = uint32_t(previous) + 1 == gridDim.x;
    if (last)
        atomicExch(completed, static_cast<unsigned long long>(epoch) << 32);
    return last;
}

// Stagger row-owner groups across ranks so concurrent epilogues do not
// concentrate every source on the same destination. Keep complete M tiles.
template <int M, int World, int Rank, int BlockM>
__device__ __forceinline__ uint32_t owner_balanced_m_tile(uint32_t tile) {
    if constexpr (World > 1) {
        constexpr uint32_t tiles = (M + BlockM - 1) / BlockM;
        return (tile + ((Rank + 1) % World) * tiles / World) % tiles;
    } else {
        return tile;
    }
}

// Share only packets crossing owner boundaries and crop their prefix.
// Aligned shards retain the native 4/8/16-row packet representation.
template <int LocalM> struct GemmRSPackets {
    static constexpr int Rows = LocalM < 8 ? 4 : (LocalM < 16 ? 8 : 16);
    static constexpr bool Aligned = LocalM % Rows == 0 || LocalM < 3;
    static constexpr int StorageRows = Aligned ? (LocalM < 4 ? 4 : LocalM)
                                               : (LocalM + 2 * Rows - 2) / Rows * Rows;
    // Independently published N tiles start on a 128-byte bulk-word group.
    static constexpr int compact_tile_elements(int columns) {
        return (LocalM * columns + 63) / 64 * 64;
    }
};

template <int Words>
__device__ __forceinline__ void transpose_quad(uint32_t (&rows)[Words]) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
        const int distance = 1 << stage;
#pragma unroll
        for (int r = 0; r < Words; ++r) {
            if ((r & distance) == 0) {
                const uint32_t swapped =
                    __shfl_xor_sync(0xffffffffu, (lane & distance) ? rows[r] : rows[r + distance], distance);
                rows[r] = (lane & distance) ? swapped : rows[r];
                rows[r + distance] = (lane & distance) ? rows[r + distance] : swapped;
            }
        }
    }
}

// A warp reduces two adjacent 16x16 packets, then transposes the eight pair
// registers into complete 64-byte row segments before vector stores.
template <int World>
__device__ __forceinline__ void sum_rank_packets(const uint4 (&values)[World], uint32_t (&packed)[4]);

template <int World, int M, int N, int Rank, int ColumnWidth = 0>
__device__ __forceinline__ void packed_sum_rows(const __ppu_bfloat16* slots, __ppu_bfloat16* output, int column = 0) {
    using Packets = GemmRSPackets<M / World>;
    constexpr int lm = M / World, plm = Packets::StorageRows;
    constexpr int row_offset = Packets::Aligned ? 0 : Rank * lm % 16;
    static_assert(Packets::Rows == 16 && N % 16 == 0);
    const int width = ColumnWidth ? min(ColumnWidth, N - column) : N;
    const int n_pairs = (N % 32 == 0 && ColumnWidth % 32 == 0) ? width / 32 : (width + 31) / 32;
    const int lane = threadIdx.x % 32;
    for (int pair = (ColumnWidth ? 0 : blockIdx.x * (blockDim.x / 32)) + threadIdx.x / 32; pair < ((lm + row_offset + 15) / 16) * n_pairs;
         pair += (ColumnWidth ? 1 : gridDim.x) * (blockDim.x / 32)) {
        const int first_atom = column / 16 + (pair / n_pairs) * (N / 16) + (pair % n_pairs) * 2;
        uint32_t packed[2][4];
#pragma unroll
        for (int atom = 0; atom < 2; ++atom) {
            uint4 values[World];
#pragma unroll
            for (int src = 0; src < World; ++src) {
                if constexpr (N % 32 != 0 || ColumnWidth % 32 != 0) {
                    if ((pair % n_pairs) * 32 + atom * 16 >= width) {
                        values[src] = make_uint4(0, 0, 0, 0);
                        continue;
                    }
                }
                const auto packet = slots + src * plm * N + (first_atom + atom) * 256;
#if __HGGC_ARCH__ == 150
                if (src == Rank && !(World == 2 && ColumnWidth > 0))
                    values[src] = __ppu_global_ldcg_bulk_b32x4(packet);
                else
#endif
                    values[src] = __ppu_global_ldcv_bulk_b32x4(packet);
            }
            sum_rank_packets(values, packed[atom]);
        }
        uint32_t rows[8] = {packed[0][0], packed[0][2], packed[1][0], packed[1][2], packed[0][1], packed[0][3], packed[1][1], packed[1][3]};
        transpose_quad(rows);
        const int row = (pair / n_pairs) * 16 + lane / 4 - row_offset;
        const int col = column + (pair % n_pairs) * 32 + (lane % 4) * 8;
        if ((Packets::Aligned || (row >= 0 && row < lm)) && ((N % 32 == 0 && ColumnWidth % 32 == 0) || col < column + width))
            *reinterpret_cast<uint4*>(output + row * N + col) = make_uint4(rows[0], rows[1], rows[2], rows[3]);
        if ((Packets::Aligned || (row + 8 >= 0 && row + 8 < lm)) && ((N % 32 == 0 && ColumnWidth % 32 == 0) || col < column + width))
            *reinterpret_cast<uint4*>(output + (row + 8) * N + col) = make_uint4(rows[4], rows[5], rows[6], rows[7]);
    }
}

// A short N span reads only its valid 128-byte words. Full groups retain
// the native 512-byte load; unused words stay zero through the lane transpose.
__device__ __forceinline__ uint4 load_packet_words(const __ppu_bfloat16* p, int words, bool local) {
    if (words == 4)
        return local ? __ppu_global_ldcg_bulk_b32x4(p) : __ppu_global_ldcv_bulk_b32x4(p);
    uint4 value = make_uint4(0, 0, 0, 0);
    if (words >= 2) {
        const auto pair = local ? __ppu_global_ldcg_bulk_b32x2(p) : __ppu_global_ldcv_bulk_b32x2(p);
        value.x = pair.x;
        value.y = pair.y;
    } else {
        value.x = local ? __ppu_global_ldcg_bulk_b32(p) : __ppu_global_ldcv_bulk_b32(p);
    }
    if (words == 3)
        value.z = local ? __ppu_global_ldcg_bulk_b32(p + 128) : __ppu_global_ldcv_bulk_b32(p + 128);
    return value;
}

// Complete native packets retain their native lane stores. Boundary and
// padded packets share wide reads and transpose into complete row segments.
template <int World, int M, int N, int Rank, int R, int ColumnWidth = 0>
__device__ __forceinline__ void packed_sum_narrow_rows(const __ppu_bfloat16* slots, __ppu_bfloat16* output,
                                                       int column = 0) {
    using Packets = GemmRSPackets<M / World>;
    constexpr int lm = M / World, plm = Packets::StorageRows;
    constexpr bool NativePacket = Packets::Aligned && lm >= R;
    constexpr int cols = NativePacket ? 16 : 256 / R;
    constexpr int prefix = Packets::Aligned ? 0 : Rank * lm % R;
    constexpr int groups = (lm + prefix + R - 1) / R;
    static_assert((R == 4 || R == 8) && N % 16 == 0);
    const int width = ColumnWidth ? min(ColumnWidth, N - column) : N;
    const int n_groups = (N % cols == 0 && ColumnWidth % cols == 0) ? width / cols : (width + cols - 1) / cols;
    const int lane = threadIdx.x % 32;
    for (int group = (ColumnWidth ? 0 : blockIdx.x * (blockDim.x / 32)) + threadIdx.x / 32; group < groups * n_groups;
         group += (ColumnWidth ? 1 : gridDim.x) * (blockDim.x / 32)) {
        const int atom = column / 16 + (group / n_groups) * (N / 16) + (group % n_groups) * (cols / 16);
        const int words = (N % cols == 0 && ColumnWidth % cols == 0) ? cols * R / 64 : min(cols, width - (group % n_groups) * cols) * R / 64;
        uint4 values[World];
#pragma unroll
        for (int source = 0; source < World; ++source) {
            const auto p = slots + source * plm * N + atom * R * 16;
#if __HGGC_ARCH__ == 150
            values[source] = load_packet_words(p, words, source == Rank);
#else
            values[source] = load_packet_words(p, words, false);
#endif
        }
        uint32_t rows[4];
        sum_rank_packets(values, rows);
        if constexpr (NativePacket) {
            const int row = (group / n_groups) * R + (R == 4 ? (lane % 16) / 4 : lane / 4);
#pragma unroll
            for (int word = 0; word < R / 4; ++word) {
                const int col = column + (group % n_groups) * 16 + 2 * (lane % 4) +
                                (R == 8 ? word * 8 : (lane / 16) * 8);
                reinterpret_cast<uint32_t*>(output)[(row * N + col) / 2] = rows[word];
            }
            continue;
        }
        transpose_quad(rows);
        if constexpr (R == 4) {
            const int from = (lane % 2) * 16 + (lane / 8) * 4 + (lane % 8) / 2;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                rows[i] = __shfl_sync(0xffffffffu, rows[i], from);
        }
        const int row = (group / n_groups) * R + lane / (cols / 8) - prefix;
        const int col = column + (group % n_groups) * cols + (lane % (cols / 8)) * 8;
        if (row >= 0 && row < lm && ((N % cols == 0 && ColumnWidth % cols == 0) || col < column + width))
            *reinterpret_cast<uint4*>(output + row * N + col) = make_uint4(rows[0], rows[1], rows[2], rows[3]);
    }
}

#if __HGGC_ARCH__ == 150
// Each N tile contains contiguous owner rows; a warp can consume across rows.
template <int World, int M, int N, int Rank, int ColumnWidth, int TileN>
__device__ __forceinline__ void compact_reduce(const __ppu_bfloat16* slots, __ppu_bfloat16* output, int column) {
    constexpr int LM = M / World, BN = ColumnWidth ? ColumnWidth : TileN;
    static_assert(BN > 0);
    constexpr int chunks = (LM * (BN < N ? BN : N) + 255) / 256;
    const int total = ColumnWidth ? chunks : ((N + BN - 1) / BN) * chunks;
    const int warps = blockDim.x / 32, lane = threadIdx.x % 32;
    for (int group = (ColumnWidth ? 0 : blockIdx.x * warps) + threadIdx.x / 32; group < total;
         group += (ColumnWidth ? 1 : gridDim.x) * warps) {
        const int col = ColumnWidth ? column : (group / chunks) * BN;
        const int width = min(BN, N - col), start = (group % chunks) * 256;
        const int count = min(256, LM * width - start);
        if (count <= 0) continue;
        uint4 values[World];
#pragma unroll
        for (int source = 0; source < World; ++source) {
            const auto ptr = slots + source * GemmRSPackets<LM>::StorageRows * N +
                             col * LM + (ColumnWidth ? column / BN : group / chunks) *
                                 (GemmRSPackets<LM>::compact_tile_elements(BN) - LM * BN) + start;
            if (count == 256)
                values[source] = source == Rank ? __ppu_global_ldcg_bulk_b32x4(ptr) : __ppu_global_ldcv_bulk_b32x4(ptr);
            else {
                values[source] = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int word = 0; word < 4; ++word)
                    if (word * 64 + lane * 2 < count)
                        reinterpret_cast<uint32_t*>(&values[source])[word] = source == Rank ?
                            __ppu_global_ldcg_bulk_b32(ptr + word * 64) : __ppu_global_ldcv_bulk_b32(ptr + word * 64);
            }
        }
        uint32_t result[4];
        sum_rank_packets(values, result);
        auto store = [&](auto tile_width) {
            constexpr int Width = decltype(tile_width)::value;
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                const int index = start + word * 64 + lane * 2;
                if (index < LM * Width)
                    reinterpret_cast<uint32_t*>(output)[(index / Width) * (N / 2) + (col + index % Width) / 2] = result[word];
            }
        };
        if constexpr (N % BN == 0)
            store(cute::Int<BN>{});
        else if (col + BN <= N)
            store(cute::Int<BN>{});
        else
            store(cute::Int<N % BN>{});
    }
}

#endif

// Keep a multi-tile owner's reduction live ranges out of the GEMM mainloop.
template <int World, int M, int N, int Rank>
__device__ __noinline__ void packed_sum_rows_outlined(const __ppu_bfloat16* slots, __ppu_bfloat16* output) {
    packed_sum_rows<World, M, N, Rank>(slots, output);
}

template <int World, int R, int StaticM = 0, int StaticN = 0, int StaticRank = -1, int BlockM = 0, int ColumnWidth = 0, int TileN = 0>
__device__ __forceinline__ void local_reduce(const __ppu_bfloat16* slots, __ppu_bfloat16* output, int column = 0) {
#if __HGGC_ARCH__ == 150
    if constexpr (World > 2 && !GemmRSPackets<StaticM / World>::Aligned && StaticM <= BlockM && StaticM >= GemmRSCompactMinM) {
        compact_reduce<World, StaticM, StaticN, StaticRank, ColumnWidth, TileN>(slots, output, column);
    } else
#endif
    if constexpr (StaticM > 0 && (R == 4 || R == 8)) {
        packed_sum_narrow_rows<World, StaticM, StaticN, StaticRank, R, ColumnWidth>(slots, output, column);
    } else {
        if constexpr (ColumnWidth == 0 && BlockM > 0 && StaticM / World > BlockM)
            packed_sum_rows_outlined<World, StaticM, StaticN, StaticRank>(slots, output);
        else
            packed_sum_rows<World, StaticM, StaticN, StaticRank, ColumnWidth>(slots, output, column);
    }
}

template <int World, int Skip = -1>
__device__ __forceinline__ void wait_flags(const int* source_flags, int epoch, int stride = 32) {
    auto flags = reinterpret_cast<const volatile int*>(source_flags);
    if constexpr (World == 2 && Skip >= 0) {
        if (threadIdx.x == 0) {
            while (flags[(1 - Skip) * stride] < epoch) {}
            acquire_fence();
        }
        __syncthreads();
        return;
    }
    if (threadIdx.x < World && threadIdx.x != Skip)
        while (flags[threadIdx.x * stride] < epoch) {
        }
#if __HGGC_ARCH__ == 150
    __ppu_barrier_sync_nocnt(0, 1);
#else
    __syncthreads();
#endif
    acquire_fence();
}

template <int World, int StaticRank> __device__ inline void wait_sources(GemmRSState s, int epoch) {
    wait_flags<World>(reinterpret_cast<const int*>(s.ptr<int64_t>(FlagPeers)[StaticRank]), epoch);
}

template <int M, int N, int World, int Rank, int BlockM, int BlockN>
__device__ void finish(GemmRSState s, int epoch, __ppu_bfloat16* output) {
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
    publish_fence();
    __syncthreads();
    const auto slots = reduction_slots(s, epoch, Rank);
    if (threadIdx.x < 32) {
        int last = 0;
        if (threadIdx.x == 0)
            last = complete_cta(s, epoch);
        last = __shfl_sync(0xffffffffu, last, 0);
        if (last) {
#pragma unroll
            for (int owner = 0; owner < World; ++owner) {
                auto flag = reinterpret_cast<int*>(s.ptr<int64_t>(FlagPeers)[owner]) + Rank * 32;
                if (owner == Rank)
                    reinterpret_cast<volatile int*>(flag)[threadIdx.x] = epoch;
                else
                    __ppu_remote_store_bulk_ncp_b32(flag, epoch);
            }
#if __HGGC_ARCH__ != 150
            __ppu_wait_remote_vstcnt(0);
            __ppu_wait_vstcnt(0);
            system_fence();
#endif
        }
    }
    wait_sources<World, Rank>(s, epoch);
    constexpr int lm = M / World;
    constexpr int rows = GemmRSPackets<lm>::Rows;
    local_reduce<World, rows, M, N, Rank, BlockM, 0, BlockN>(slots, output);
#if __HGGC_ARCH__ == 150
    // Retire payload before publication; overlap only notification ACKs.
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
#endif
}

__device__ __forceinline__ uint32_t native_bf16_pair(float x, float y) {
    cutlass::Array<float, 2> input;
    input[0] = x;
    input[1] = y;
    auto converted = cutlass::NumericArrayConverter<cutlass::bfloat16_t, float, 2>{}(input);
    static_assert(sizeof(converted) == sizeof(uint32_t));
    return *reinterpret_cast<const uint32_t*>(&converted);
}

template <class Acc> __device__ __forceinline__ uint4 pack_accumulator_packet(const Acc& acc, int first) {
    return make_uint4(native_bf16_pair(float(acc(first)), float(acc(first + 1))),
                      native_bf16_pair(float(acc(first + 2)), float(acc(first + 3))),
                      native_bf16_pair(float(acc(first + 4)), float(acc(first + 5))),
                      native_bf16_pair(float(acc(first + 6)), float(acc(first + 7))));
}

template <int World>
__device__ __forceinline__ void sum_rank_packets(const uint4 (&values)[World], uint32_t (&packed)[4]) {
#pragma unroll
    for (int word = 0; word < 4; ++word) {
#if __HGGC_ARCH__ == 150
        if constexpr (World == 2) {
            const auto a = reinterpret_cast<const uint32_t*>(&values[0])[word];
            const auto b = reinterpret_cast<const uint32_t*>(&values[1])[word];
            // Match the positive-zero initial FP32 accumulator, including
            // the -0 + -0 case. More than two sources still require FP32.
            packed[word] = __ppumath_hadd2_2bf16(__ppumath_hadd2_2bf16(a, b), 0);
            continue;
        }
#endif
        float x = 0, y = 0;
#pragma unroll
        for (int rank = 0; rank < World; ++rank) {
            union Pair {
                uint32_t u;
                __ppu_bfloat16 b[2];
            } value;
            value.u = reinterpret_cast<const uint32_t*>(&values[rank])[word];
            x += __bfloat162float(value.b[0]);
            y += __bfloat162float(value.b[1]);
        }
        packed[word] = native_bf16_pair(x, y);
    }
}

// Route one native MMA packet; at most two owners intersect its rows.
template <int M, int N, int World, int Rank, class Storage, class Values>
__device__ __forceinline__ void store_owner_packet(Storage s, int64_t phase_offset, int row, int col, Values values) {
    using Packets = GemmRSPackets<M / World>;
    constexpr int lm = M / World, r = Packets::Rows;
    static_assert(Packets::Aligned || r <= 2 * lm);
    const auto store = [&](int owner) {
        const int packet_row = Packets::Aligned ? (row % lm) / r : row / r - owner * lm / r;
        const int atom = packet_row * (N / 16) + col / 16;
        auto dst = reinterpret_cast<uint32_t*>(s.peer(owner)) + phase_offset + atom * r * 8;
        if (owner == Rank) {
#if __HGGC_ARCH__ == 150
            if constexpr (r == 16)
                __ppu_global_store_bulk_b32x4(dst, values);
            else if constexpr (r == 8)
                __ppu_global_store_bulk_b32x2(dst, values);
            else
                __ppu_global_store_bulk_b32(dst, values);
#else
            const int lane = threadIdx.x % 32;
#pragma unroll
            for (int word = 0; word < r / 4; ++word)
                dst[lane + word * 32] = reinterpret_cast<const uint32_t*>(&values)[word];
#endif
        } else {
            if constexpr (r == 16)
                __ppu_remote_store_bulk_ncp_b32x4(dst, values);
            else if constexpr (r == 8)
                __ppu_remote_store_bulk_ncp_b32x2(dst, values);
            else
                __ppu_remote_store_bulk_ncp_b32(dst, values);
        }
    };
    const int first = row / lm;
    store(first);
    if constexpr (!Packets::Aligned) {
        const int last = (row + r - 1 < M ? row + r - 1 : M - 1) / lm;
        if (last != first)
            store(last);
    }
}

// A temporary row-major view reuses retired native input storage.
#if __HGGC_ARCH__ == 150
__device__ __forceinline__ void compact_begin() {
    __ppu_wait_vldcnt(0);
    __ppu_wait_sldcnt(0);
    __ppu_wait_tsmcnt(0);
    __ppu_barrier_sync_nocnt(0, 1);
}

template <int M, int N, int BM, int BN, class Acc, class Mma, class Shape>
__device__ __forceinline__ void compact_stage(char* scratch, const Acc& acc, const Mma& mma, Shape shape,
                                             int mt, int nt, int row_offset = 0, int col_offset = 0) {
    using namespace cute;
    static_assert(BM * BN * 2 <= DG_GEMM_RS_SHARED_BUDGET, "compact output exceeds native shared storage");
    auto coords = mma.get_slice((threadIdx.x / 32) * 32).partition_C(make_identity_tensor(take<0, 2>(shape)));
    auto shared = reinterpret_cast<uint32_t*>(scratch);
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int pair = 0; pair < size(Acc{}) / 2; pair += 4) {
        const int br = get<0>(coords(2 * pair)) / 16 * 16 + row_offset;
        const int bc = get<1>(coords(2 * pair)) / 16 * 16 + col_offset;
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int row = br + lane / 4 + (word % 2) * 8;
            const int col = bc + (lane % 4) * 2 + (word / 2) * 8;
            // The complete native atom fits the retired scratch tile. Its
            // global tail is masked when emitting to the row owner.
            if ((size<0>(Shape{}) == 16 && size<1>(Shape{}) == 16) ||
                    (mt * BM + row < M && nt * BN + col < N))
                shared[(row * BN + col) / 2] = native_bf16_pair(float(acc(2 * (pair + word))), float(acc(2 * (pair + word) + 1)));
        }
    }
}

template <int M, int N, int World, int Rank, int BM, int BN, int Width>
__device__ __forceinline__ void compact_emit_width(GemmRSState s, int epoch, char* scratch, int mt, int nt) {
    constexpr int LM = M / World, width = Width;
    static_assert(M <= BM, "compact emission owns one complete M tile");
    const int lane = threadIdx.x % 32;
    const int64_t phase = int64_t((epoch - 1) & 1) * s.phase_elements() / 2 +
                          int64_t(Rank) * GemmRSPackets<LM>::StorageRows * N / 2;
    constexpr int elements = LM * width, chunks = (elements + 255) / 256;
    const int warp = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
    // Interleave owners and stagger the first destination across sources.
    for (int task = warp; task < World * chunks; task += blockDim.x / 32) {
        const int owner = (task % World + Rank + 1) % World;
        const int start = (task / World) * 256;
        auto dst = reinterpret_cast<uint32_t*>(s.peer(owner)) + phase +
                   nt * GemmRSPackets<LM>::compact_tile_elements(BN) / 2;
        auto shared = reinterpret_cast<uint32_t*>(scratch) + owner * LM * BN / 2;
        const int count = min(256, elements - start);
            uint4 value = make_uint4(0, 0, 0, 0);
            if (width == BN && count == 256)
                value = __ppu_shared_load_bulk_b32x4(shared + start / 2);
            else {
#pragma unroll
                for (int word = 0; word < 4; ++word) {
                    const int index = start + word * 64 + lane * 2;
                    if (index < elements)
                        reinterpret_cast<uint32_t*>(&value)[word] = shared[(index / width) * (BN / 2) + (index % width) / 2];
                }
            }
            if (count == 256) {
                if (owner == Rank) __ppu_global_store_bulk_b32x4(dst + start / 2, value);
                else __ppu_remote_store_bulk_ncp_b32x4(dst + start / 2, value);
            } else {
#pragma unroll
                for (int word = 0; word < 4; ++word)
                    if (word * 64 + lane * 2 < count) {
                        const auto v = reinterpret_cast<uint32_t*>(&value)[word];
                        if (owner == Rank) __ppu_global_store_bulk_b32(dst + start / 2 + word * 32, v);
                        else __ppu_remote_store_bulk_ncp_b32(dst + start / 2 + word * 32, v);
                    }
            }
    }
}

template <int M, int N, int World, int Rank, int BM, int BN>
__device__ __forceinline__ void compact_emit(GemmRSState s, int epoch, char* scratch, int mt, int nt) {
    __ppu_wait_tsmcnt(0);
    __ppu_barrier_sync_nocnt(0, 1);
    if constexpr (N % BN == 0)
        compact_emit_width<M, N, World, Rank, BM, BN, BN>(s, epoch, scratch, mt, nt);
    else if ((nt + 1) * BN <= N)
        compact_emit_width<M, N, World, Rank, BM, BN, BN>(s, epoch, scratch, mt, nt);
    else
        compact_emit_width<M, N, World, Rank, BM, BN, N % BN>(s, epoch, scratch, mt, nt);
    // The finish path already joins the CTA. A separate scratch-lifetime
    // join is needed only when a successor tile can reuse this storage.
    if (gridDim.x < ((M + BM - 1) / BM) * ((N + BN - 1) / BN)) {
        __ppu_wait_tsmcnt(0);
        __ppu_barrier_sync_nocnt(0, 1);
    }
}

#endif

// Both GEMM mainloops produce final FP32 accumulators. Convert exactly once
// and send native MMA packets to their row owner.
template <int M, int N, int World, int Rank, class Acc, class Mma, class Shape, class Storage>
__device__ void store_packets(Storage s, int epoch, Acc const& acc, Mma const& mma, Shape shape, int m_tile,
                              int n_tile, char* scratch = nullptr) {
    using namespace cute;
#if __HGGC_ARCH__ == 150
    if constexpr (World > 2 && !GemmRSPackets<M / World>::Aligned && M <= size<0>(Shape{}) && M >= GemmRSCompactMinM) {
        constexpr int BM = size<0>(Shape{}), BN = size<1>(Shape{});
        compact_begin();
        compact_stage<M, N, BM, BN>(scratch, acc, mma, shape, m_tile, n_tile);
        compact_emit<M, N, World, Rank, BM, BN>(s, epoch, scratch, m_tile, n_tile);
        return;
    }
#endif
    using Packets = GemmRSPackets<M / World>;
    constexpr int lm = M / World, plm = Packets::StorageRows;
    constexpr int r = Packets::Rows;
    const int64_t phase_offset = int64_t((epoch - 1) & 1) * s.phase_elements() / 2 + Rank * plm * N / 2;
    const int lane = threadIdx.x % 32;
#if __HGGC_ARCH__ == 150
    auto coords =
        mma.get_slice((threadIdx.x / 32) * 32).partition_C(make_identity_tensor(take<0, 2>(shape)));
#else
    auto coords = mma.get_slice(threadIdx.x).partition_C(make_identity_tensor(take<0, 2>(shape)));
#endif
    constexpr int pairs = size(Acc{}) / 2;
    CUTE_STATIC_ASSERT_V(size<0>(acc) == Int<8>{});
    static_assert(pairs % 4 == 0);
#pragma unroll
    for (int pair = 0; pair < pairs; pair += 4) {
        uint4 values;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#if __HGGC_ARCH__ == 150
            reinterpret_cast<uint32_t*>(&values)[j] =
                native_bf16_pair(float(acc(2 * (pair + j))), float(acc(2 * (pair + j) + 1)));
#else
            union Pair {
                uint32_t u;
                __ppu_bfloat16 v[2];
            } q;
            q.v[0] = __float2bfloat16_rn(float(acc(2 * (pair + j))));
            q.v[1] = __float2bfloat16_rn(float(acc(2 * (pair + j) + 1)));
            reinterpret_cast<uint32_t*>(&values)[j] = q.u;
#endif
        }
        const int row = get<0>(coords(2 * pair)) + m_tile * size<0>(shape);
        const int col = get<1>(coords(2 * pair)) + n_tile * size<1>(shape);
        const int base_row = row / 16 * 16, base_col = col / 16 * 16;
        if (base_col >= N)
            continue; // Uniform for a complete16-column atom.
        if constexpr (r == 16) {
            if (base_row >= M)
                continue;
            store_owner_packet<M, N, World, Rank>(s, phase_offset, row, col, values);
        } else if constexpr (r == 8) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row8 = base_row + half * 8;
                if (row8 >= M)
                    continue;
                const uint2 v = half == 0 ? make_uint2(values.x, values.z) : make_uint2(values.y, values.w);
                store_owner_packet<M, N, World, Rank>(s, phase_offset, row8, base_col, v);
            }
        } else {
            constexpr int rows = Packets::Aligned && lm < 4 ? lm : 4;
#pragma unroll
            for (int group = 0; group < 16 / rows; ++group) {
                const int row4 = base_row + group * rows;
                if (row4 >= M)
                    continue;
                int from = (lane % 16) + (group * rows % 8) * 4;
                uint32_t lo = __shfl_sync(0xffffffffu, group * rows < 8 ? values.x : values.y, from);
                uint32_t hi = __shfl_sync(0xffffffffu, group * rows < 8 ? values.z : values.w, from);
                uint32_t v = lane < 16 ? lo : hi;
                if ((lane % 16) / 4 >= rows)
                    v = 0;
                if constexpr (!Packets::Aligned)
                    if (row4 + (lane % 16) / 4 >= M)
                        v = 0;
                store_owner_packet<M, N, World, Rank>(s, phase_offset, row4, base_col, v);
            }
        }
    }
}

template <int M, int N, int World, int Rank, int BM, int BN>
__device__ void finish_column_tail(GemmRSState state, int epoch, __ppu_bfloat16* output) {
    static_assert(M <= BM && BN % 16 == 0);
    constexpr int columns = (N + BN - 1) / BN;
    constexpr int rows = GemmRSPackets<M / World>::Rows;
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
    publish_fence();
    __syncthreads();
    const int capacity = int(state.word(ChunkCapacity));
    if (threadIdx.x < 32) {
        // Each complete M tile publishes the N tiles produced by this CTA.
        for (int column = blockIdx.x; column < columns; column += gridDim.x) {
#pragma unroll
            for (int owner = 0; owner < World; ++owner) {
                if (owner != Rank) {
                    auto flag = reinterpret_cast<int*>(state.ptr<int64_t>(FlagPeers)[owner]) +
                                (World + Rank * capacity + column) * 32;
                    __ppu_remote_store_bulk_ncp_b32(flag, epoch);
                }
            }
        }
    }
    auto flags = reinterpret_cast<const int*>(state.ptr<int64_t>(FlagPeers)[Rank]);
    const auto slots = reduction_slots(state, epoch, Rank);
    for (int column = blockIdx.x; column < columns; column += gridDim.x) {
        wait_flags<World, Rank>(flags + (World + column) * 32, epoch, capacity * 32);
        local_reduce<World, rows, M, N, Rank, BM, BN>(slots, output, column * BN);
    }
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
    __syncthreads();
    if (threadIdx.x == 0)
        complete_cta(state, epoch);
}

// Complete M tiles can consume columns with whole reduction groups.
template <int M, int N, int World, int Rank, int BM, int BN>
__device__ void finish_tiles(GemmRSState state, int epoch, __ppu_bfloat16* output) {
#if __HGGC_ARCH__ == 150
    // R16 reductions pair adjacent 16-column packets. Keep full pairs in
    // each independent column; boundary packets keep their existing path.
    constexpr int columns_per_group = GemmRSPackets<M / World>::Rows == 16 ? 32 : 16;
    if constexpr (M <= BM && BN % 16 == 0 &&
                  (World == 2 || BN % columns_per_group == 0 || !GemmRSPackets<M / World>::Aligned)) {
        finish_column_tail<M, N, World, Rank, BM, BN>(state, epoch, output);
        return;
    }
#endif
    finish<M, N, World, Rank, BM, BN>(state, epoch, output);
}

} // namespace deep_gemm::gemm_comm


#if defined(DG_GEMM_RS_M) && defined(DG_GEMM_RS_WORLD) && defined(DG_GEMM_RS_RANK)
namespace deep_gemm::gemm_comm {
template <class Copy, class Mma, class Shape, int SharedPackets, int N, int Threads = cute::size(Mma{}),
          int LocalM = 0, bool MaskN = false>
struct GemmRSSendCopy : Copy {
    const uint4* staging = nullptr;
    uint32_t* destination = nullptr;
    int m_offset = 0, n_remaining = N;
    // Direct destinations retain -1. Split-owner tiles supply a rank and
    // resolve the peer table per packet rather than assuming one warp owner.
    int own_rank = -1;
    int64_t base_offset = 0;
    static constexpr int Packets = cute::size<0>(Shape{}) * cute::size<1>(Shape{}) / Threads / 8;
    static constexpr int SharedPacketCount = SharedPackets;
    static constexpr int Warps = Threads / 32;
    static_assert(SharedPackets % 2 == 0);
    static_assert(SharedPackets > 0 && SharedPackets <= Packets);
    mutable uint4 cache[Packets > SharedPackets ? Packets - SharedPackets : 1];
    mutable int calls = 0;

    __device__ int64_t peer(int owner) const {
        return __ldg(reinterpret_cast<const int64_t*>(destination) + owner);
    }
    __device__ bool has_pending() const { return destination != nullptr; }
    __device__ void clear_pending() {
        // End cached-payload lifetimes before staging the next tile.
        destination = nullptr;
        calls = 0;
#pragma unroll
        for (int i = 0; i < Packets - SharedPackets; ++i)
            cache[i] = make_uint4(0, 0, 0, 0);
    }
    __device__ void send_packet(int packet) const {
        uint4 payload;
        if (packet < SharedPackets) {
            payload = staging[(((packet / 2) * Warps + threadIdx.x / 32) * 2 + packet % 2) * 32 + threadIdx.x % 32];
        } else {
            payload = cache[0];
#pragma unroll
            for (int i = 0; i < Packets - SharedPackets - 1; ++i)
                cache[i] = cache[i + 1];
        }
        using namespace cute;
        auto coords = Mma{}.get_slice((threadIdx.x / 32) * 32).partition_C(make_identity_tensor(take<0, 2>(Shape{})));
        const int row = get<0>(coords(packet * 8)), col = get<1>(coords(packet * 8));
        if constexpr (MaskN)
            if (col >= n_remaining)
                return;
        if constexpr (LocalM > 0) {
            if constexpr (DG_GEMM_RS_M % cute::size<0>(Shape{}) != 0)
                if (row + m_offset >= DG_GEMM_RS_M)
                    return;
            if constexpr (LocalM % 16 != 0) {
                store_owner_packet<DG_GEMM_RS_M, N, DG_GEMM_RS_WORLD, DG_GEMM_RS_RANK>(
                    *this, base_offset, row + m_offset, N - n_remaining + col, payload);
                return;
            }
        }
        int local_row = row;
        if constexpr (LocalM > 0)
            local_row = (row + m_offset) % LocalM;
        if constexpr (MaskN)
            if (col >= n_remaining)
                return;
        const int atom = local_row / 16 * (N / 16) + col / 16;
        if constexpr (LocalM > 0) {
            if (own_rank >= 0) {
                const int owner = (row + m_offset) / LocalM;
                const auto peer = __ldg(reinterpret_cast<const int64_t*>(destination) + owner);
                auto address = reinterpret_cast<uint32_t*>(peer) + base_offset + atom * 128;
                if (owner == own_rank)
                    __ppu_global_store_bulk_b32x4(address, payload);
                else
                    __ppu_remote_store_bulk_ncp_b32x4(address, payload);
                return;
            }
        }
        __ppu_remote_store_bulk_ncp_b32x4(destination + atom * 128, payload);
    }
    __device__ void progress() const {
        if (has_pending() && calls < Packets)
            send_packet(calls++);
    }
    __device__ void flush() const {
        if (has_pending()) {
            for (int packet = calls; packet < Packets; ++packet)
                send_packet(packet);
            calls = Packets;
        }
    }
};
template <bool SplitAIU = true, class CA, class Mma, class Shape, int SharedPackets, int N,
          int Threads, int LocalM, bool MaskN, class SEa, class SLa, class DEa,
          class DLa, class CB, class SEb, class SLb, class DEb, class DLb>
__device__ __forceinline__ void copy_aiu(const GemmRSSendCopy<CA, Mma, Shape, SharedPackets, N, Threads, LocalM, MaskN>& a,
                                         const cute::Tensor<SEa, SLa>& sa, cute::Tensor<DEa, DLa>&& da, const CB& b,
                                         const cute::Tensor<SEb, SLb>& sb, cute::Tensor<DEb, DLb>&& db, int& warp) {
    cute::copy_aiu<SplitAIU>(static_cast<const CA&>(a), sa, static_cast<cute::Tensor<DEa, DLa>&&>(da), b, sb,
                             static_cast<cute::Tensor<DEb, DLb>&&>(db), warp);
    a.progress();
}

} // namespace deep_gemm::gemm_comm
#endif // GEMM/RS launch geometry

#endif // defined(__HGGC__)
