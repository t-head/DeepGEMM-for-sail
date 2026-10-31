#pragma once

namespace deep_gemm::gemm_comm {

// Device-resident immutable pointer table. Epochs/counters live separately.
enum GemmRSStateIndex {
    Peers,
    FlagPeers,
    Completed,
    ShapeM,
    ShapeN,
    Rank,
    GemmRSStateWords
};
} // namespace deep_gemm::gemm_comm

#if defined(__HGGC__)
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <cute/tensor.hpp>

namespace deep_gemm::gemm_comm {
struct GemmRSState {
    const int64_t* words;
    template <class T>
    __device__ T* ptr(int i) const {
        return reinterpret_cast<T*>(words[i]);
    }
};

__device__ inline void system_fence() {
    asm volatile("ppu.fence.acq_rel.sys;" ::: "memory");
}

__device__ inline int begin(GemmRSState s, char* scratch) {
    auto word = reinterpret_cast<volatile int*>(scratch);
    if (threadIdx.x == 0) {
        // High32 counts completed calls; low32 counts this call's producers.
        // The epoch cannot advance until every CTA has entered and finished,
        // so this snapshot is uniform even when consecutive grids differ.
        unsigned long long completed;
        asm volatile("ppu.ld.acquire.gpu.b64 %0,[%1];" : "=l"(completed)
                     : "l"(s.ptr<unsigned long long>(Completed)) : "memory");
        *word = int(completed >> 32) + 1;
    }
    __syncthreads();
    const int epoch = *word;
    __syncthreads();
    return epoch;
}

// Both quantizations store the same owner-local MMA packets. Preserve source
// order, accumulate in FP32, and round to BF16 after the final source.
template <int World, int R, int StaticM = 0, int StaticN = 0>
__device__ __forceinline__ void packed_sum(const __ppu_bfloat16* slots, __ppu_bfloat16* output, int shape_m, int shape_n,
                                           int epoch) {
    const int m = StaticM ? StaticM : shape_m, n = StaticN ? StaticN : shape_n;
    const int lm = m / World, plm = lm < 4 ? 4 : lm, nt = n / 16;
    slots += int64_t((epoch - 1) & 1) * World * plm * n;
    const int lane = threadIdx.x % 32;
    for (int atom = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; atom < (plm / R) * nt;
         atom += gridDim.x * (blockDim.x / 32)) {
        uint4 values[World];
#pragma unroll
        for (int src = 0; src < World; ++src) {
            auto p = slots + src * plm * n + atom * R * 16;
            if constexpr (R == 16)
                values[src] = __ppu_global_ldcv_bulk_b32x4(p);
            else if constexpr (R == 8) {
                auto v = __ppu_global_ldcv_bulk_b32x2(p);
                values[src].x = v.x;
                values[src].y = v.y;
            } else
                values[src].x = __ppu_global_ldcv_bulk_b32(p);
        }
#pragma unroll
        for (int j = 0; j < R / 4; ++j) {
            float a = 0, b = 0;
#pragma unroll
            for (int src = 0; src < World; ++src) {
                union Pair {
                    uint32_t u;
                    __ppu_bfloat16 v[2];
                } p;
                p.u = reinterpret_cast<uint32_t*>(&values[src])[j];
                a += __bfloat162float(p.v[0]);
                b += __bfloat162float(p.v[1]);
            }
            int row = (atom / nt) * R + (R == 4 ? (lane % 16) / 4 : lane / 4) + (R == 16 ? (j % 2) * 8 : 0);
            int col = (atom % nt) * 16 + 2 * (lane % 4) + (R == 16 ? (j / 2) * 8 : (R == 8 ? j * 8 : (lane / 16) * 8));
            if (row >= lm)
                continue;
            union Pair {
                uint32_t u;
                __ppu_bfloat16 v[2];
            } q;
            q.v[0] = __float2bfloat16_rn(a);
            q.v[1] = __float2bfloat16_rn(b);
            reinterpret_cast<uint32_t*>(output)[(row * n + col) / 2] = q.u;
        }
    }
}

template <int World, int R, int StaticM = 0, int StaticN = 0, int StaticRank = -1>
__device__ __forceinline__ void local_reduce(GemmRSState s, int epoch, __ppu_bfloat16* output) {
    auto slots =
        reinterpret_cast<const __ppu_bfloat16*>(s.ptr<int64_t>(Peers)[StaticRank < 0 ? s.words[Rank] : StaticRank]);
    packed_sum<World, R, StaticM, StaticN>(slots, output, s.words[ShapeM], s.words[ShapeN],
                                           epoch);
}

template <int World>
__device__ __forceinline__ void wait_flags(const int* source_flags, int epoch) {
    auto flags = reinterpret_cast<const volatile int*>(source_flags);
    if (threadIdx.x < World)
        while (flags[threadIdx.x * 32] < epoch) {
        }
    __syncthreads();
    system_fence();
}

template <int World>
__device__ inline void wait_sources(GemmRSState s, int epoch) {
    wait_flags<World>(reinterpret_cast<const int*>(s.ptr<int64_t>(FlagPeers)[s.words[Rank]]), epoch);
}

template <int M, int N, int World, int Rank, bool StaticShape = false>
__device__ void finish(GemmRSState s, int epoch, __ppu_bfloat16* output) {
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
    system_fence();
    __syncthreads();
    if (threadIdx.x < 32) {
        int last = 0;
        if (threadIdx.x == 0) {
            auto completed = s.ptr<unsigned long long>(Completed);
            const auto previous = atomicAdd(completed, 1ULL);
            last = uint32_t(previous) + 1 == gridDim.x;
            if (last)
                atomicExch(completed, static_cast<unsigned long long>(epoch) << 32);
        }
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
            __ppu_wait_remote_vstcnt(0);
            __ppu_wait_vstcnt(0);
            system_fence();
        }
    }
    wait_sources<World>(s, epoch);
    constexpr int lm = M / World;
    constexpr int rows = lm < 8 ? 4 : (lm < 16 ? 8 : 16);
    if constexpr (StaticShape)
        local_reduce<World, rows, M, N, Rank>(s, epoch, output);
    else
        local_reduce<World, rows>(s, epoch, output);
}

// Both GEMM mainloops produce final FP32 accumulators. Convert exactly once
// and send native MMA packets to their row owner.
template <int M, int N, int World, int Rank, class Acc, class Mma, class Shape>
__device__ void store_packets(GemmRSState s, int epoch, Acc const& acc, Mma const& mma, Shape shape, int m_tile, int n_tile,
                              int n_origin = 0, int valid_begin = 0, int valid_end = N) {
    using namespace cute;
    constexpr int lm = M / World, plm = lm < 4 ? 4 : lm;
    constexpr int r = lm < 8 ? 4 : (lm < 16 ? 8 : 16);
    const int lane = threadIdx.x % 32;
    auto coords = mma.get_slice(threadIdx.x).partition_C(make_identity_tensor(take<0, 2>(shape)));
    constexpr int pairs = size(Acc{}) / 2;
    CUTE_STATIC_ASSERT_V(size<0>(acc) == Int<8>{});
    static_assert(pairs % 4 == 0);
#pragma unroll
    for (int pair = 0; pair < pairs; pair += 4) {
        uint4 values;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            union Pair {
                uint32_t u;
                __ppu_bfloat16 v[2];
            } q;
            q.v[0] = __float2bfloat16_rn(float(acc(2 * (pair + j))));
            q.v[1] = __float2bfloat16_rn(float(acc(2 * (pair + j) + 1)));
            reinterpret_cast<uint32_t*>(&values)[j] = q.u;
        }
        const int row = get<0>(coords(2 * pair)) + m_tile * size<0>(shape);
        const int col = get<1>(coords(2 * pair)) + n_tile * size<1>(shape) + n_origin;
        const int base_row = row / 16 * 16, base_col = col / 16 * 16;
        if (base_col >= N || base_col < valid_begin || base_col >= valid_end)
            continue; // Uniform for a complete16-column atom.
        if constexpr (r == 16) {
            if (base_row >= M)
                continue;
            const int owner = row / lm;
            const int atom = ((row % lm) / 16) * (N / 16) + col / 16;
            auto dst = reinterpret_cast<uint32_t*>(s.ptr<int64_t>(Peers)[owner]) +
                       int64_t((epoch - 1) & 1) * World * plm * N / 2 + Rank * plm * N / 2 + atom * 128;
            if (owner == Rank) {
                dst[lane] = values.x;
                dst[lane + 32] = values.y;
                dst[lane + 64] = values.z;
                dst[lane + 96] = values.w;
            } else
                __ppu_remote_store_bulk_ncp_b32x4(dst, values);
        } else if constexpr (r == 8) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row8 = base_row + half * 8;
                if (row8 >= M)
                    continue;
                const int owner = row8 / lm, atom = (row8 % lm) / 8 * (N / 16) + base_col / 16;
                auto dst = reinterpret_cast<uint32_t*>(s.ptr<int64_t>(Peers)[owner]) +
                           int64_t((epoch - 1) & 1) * World * plm * N / 2 + Rank * plm * N / 2 + atom * 64;
                uint2 v = half == 0 ? make_uint2(values.x, values.z) : make_uint2(values.y, values.w);
                if (owner == Rank) {
                    dst[lane] = v.x;
                    dst[lane + 32] = v.y;
                } else
                    __ppu_remote_store_bulk_ncp_b32x2(dst, v);
            }
        } else {
            constexpr int rows = lm < 4 ? lm : 4;
#pragma unroll
            for (int group = 0; group < 16 / rows; ++group) {
                const int row4 = base_row + group * rows;
                if (row4 >= M)
                    continue;
                int owner = row4 / lm, atom = (row4 % lm) / 4 * (N / 16) + base_col / 16;
                int from = (lane % 16) + (group * rows % 8) * 4;
                uint32_t lo = __shfl_sync(0xffffffffu, group * rows < 8 ? values.x : values.y, from);
                uint32_t hi = __shfl_sync(0xffffffffu, group * rows < 8 ? values.z : values.w, from);
                uint32_t v = lane < 16 ? lo : hi;
                if ((lane % 16) / 4 >= rows)
                    v = 0;
                auto dst = reinterpret_cast<uint32_t*>(s.ptr<int64_t>(Peers)[owner]) +
                           int64_t((epoch - 1) & 1) * World * plm * N / 2 + Rank * plm * N / 2 + atom * 32;
                if (owner == Rank)
                    dst[lane] = v;
                else
                    __ppu_remote_store_bulk_ncp_b32(dst, v);
            }
        }
    }
}

} // namespace deep_gemm::gemm_comm

#endif // defined(__HGGC__)
