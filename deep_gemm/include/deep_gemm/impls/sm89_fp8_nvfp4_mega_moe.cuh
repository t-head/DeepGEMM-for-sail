#pragma once

#include <algorithm>
#include <deep_gemm/impls/nvfp4_w4a8_gemm.cuh>
#include <deep_gemm/impls/fp8_blockwise_mega_moe_dispatch_gemm1.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>
#include <deep_gemm/tix/ld_st.cuh>

namespace deep_gemm::mega_moe::nvfp4 {

template <int N, int K, int M, int BlockN, int BlockK, int WarpM, int WarpN, int WarpK, int Stages>
struct Gemm1 {
    using Kernel = cutlass::gemm::kernel::W4A8GEMM<
        N, K, M, BlockN, BlockK, WarpM, WarpN, WarpK, 1, Stages,
        GemmType::MGroupedNoPad, 16, 1, 128, true, false, true, 128>;
    static constexpr bool ScaleColumnMajor = false;
    static constexpr uint32_t WorkBatch = 1;
    static constexpr int ShapeN = N, ShapeK = K, BlockM = M;
    static_assert((BlockN == 128 || BlockN == 256) && M <= 128);
    static constexpr int Parts = 256 / BlockN;
    static constexpr int GroupThreads = Kernel::MaxThreadsPerBlock;
    static constexpr int kNumThreads = Parts * GroupThreads;
    static_assert(kNumThreads <= 1024);
    static constexpr int64_t WeightBytes = int64_t(N) * K / 2;
    static constexpr uint32_t TileOffset = offsetof(typename Kernel::SharedStorage, smem_a);
    static constexpr uint32_t SlotBytes = std::max(uint32_t(sizeof(typename Kernel::SharedStorage)),
        uint32_t(TileOffset + M * (BlockN + 2) * sizeof(float)));
    template <int Experts>
    static constexpr uint32_t shared_memory_bytes() {
        return std::max({Parts * SlotBytes, uint32_t(kNumThreads / 32 * K), uint32_t(Experts * 4)});
    }
    // Each subgroup runs the ordinary grouped-GEMM mainloop independently.
    // The surrounding CTA barriers publish all gate/up fragments before quantization.
    struct ThreadGroup {
        CUTLASS_DEVICE int thread_index() const { return threadIdx.x % GroupThreads; }
        CUTLASS_DEVICE int warp_index() const { return thread_index() / 32; }
        CUTLASS_DEVICE void sync() const {
            if constexpr (Parts == 1) {
                __syncthreads();
            } else {
                const int group = threadIdx.x / GroupThreads;
                if (group == 0) __ppu_barrier_sync(1, GroupThreads);
                else if (group == 1) __ppu_barrier_sync(2, GroupThreads);
            }
        }
    };
    struct Epilogue {
        float* tile;
        int valid;
        CUTLASS_DEVICE const int* weight(const int* p, int, int64_t) const { return p; }
        CUTLASS_DEVICE const uint8_t* scale(const uint8_t* p, int, int64_t) const { return p; }
        CUTLASS_DEVICE const float* outer_scale(const float* p, int, int64_t) const { return p; }
        template <class Accum, class Coords, class Scheduler>
        CUTLASS_DEVICE void store(Accum& accum, const Coords& coords, const Scheduler&,
                                 int, int, int, char*) const {
            using namespace cute;
            ThreadGroup{}.sync();
            CUTE_UNROLL
            for (int i = 0; i < size(accum); ++i) {
                const auto coord = coords(i);
                if (ThreadGroup{}.thread_index() < (M / WarpM) * (BlockN / WarpN) * 32 && get<0>(coord) < valid)
                    tile[get<0>(coord) * (BlockN + 2) + get<1>(coord)] = accum(i);
            }
            ThreadGroup{}.sync();
        }
    };
    template <bool Clamp, GatedActivation Activation>
    struct Values {
        const char* smem;
        float limit, alpha, beta, linear_beta;
        CUTLASS_DEVICE float2 pair(int row, int column) const {
            const auto* tile = reinterpret_cast<const float*>(smem + (column / BlockN) * SlotBytes + TileOffset);
            return *reinterpret_cast<const float2*>(tile + row * (BlockN + 2) + column % BlockN);
        }
        CUTLASS_DEVICE float2 operator()(int row, int column) const {
            const float2 g = pair(row, column), u = pair(row, column + 128);
            return make_float2(
                routed_gated_activation<Activation, bool(DG_MOE_FAST_MATH)>(
                    g.x, u.x, Clamp ? limit : -1.f, alpha, beta, linear_beta),
                routed_gated_activation<Activation, bool(DG_MOE_FAST_MATH)>(
                    g.y, u.y, Clamp ? limit : -1.f, alpha, beta, linear_beta));
        }
    };
    template <bool Clamp, GatedActivation Activation>
    CUTLASS_DEVICE static void run(const uint8_t* a, const float* sa, const uint8_t* b,
        const float* sb, const uint8_t* native, uint8_t* out, float* out_scales,
        int pool_rows, int valid, int nb, int, char* smem, float limit,
        float alpha, float beta, float linear_beta, int writable) {
        typename Kernel::Params p(reinterpret_cast<const cutlass::float_e4m3_t*>(a),
            reinterpret_cast<const int*>(b), native, reinterpret_cast<cutlass::bfloat16_t*>(out));
        p.num_token = pool_rows;
        p.ptr_a_scale = sa;
        p.ptr_b_scale = sb;
        const float clamp = Clamp ? __bfloat162float(__float2bfloat16(limit)) : -1.f;
        const int group = threadIdx.x / GroupThreads;
        char* group_smem = smem + group * SlotBytes;
        sched::SingleTileScheduler scheduler{valid, nb * Parts + group};
        auto* tile = reinterpret_cast<float*>(group_smem + TileOffset);
        Kernel{}.run(p, group_smem, scheduler, Epilogue{tile, valid}, ThreadGroup{});
        __syncthreads();
        fp8_blockwise::quantize_activated_tile<N, kNumThreads, ScaleColumnMajor>(
            Values<Clamp, Activation>{smem, clamp, alpha, beta, linear_beta},
            out, out_scales, pool_rows, valid, nb, threadIdx.x, valid);
    }
};

template <int N, int K, int M, int BlockN, int StoreWarps>
struct Scatter {
    static_assert(BlockN == 64 || BlockN == 128 || BlockN == 256);
    static constexpr int Stride = BlockN / 2 + 4;
    static constexpr uint32_t SharedBytes = M * Stride * sizeof(uint32_t);
    __ppu_bfloat16* output;
    const int64_t *routes, *peer_offsets;
    const int *replica_weights;
    const uint8_t* replica_native_scales;
    const float* replica_outer_scales;
    int masters, capacity, self_rank;
    CUTLASS_DEVICE const int* weight(const int* p, int e, int64_t size) const {
        return (e < masters ? p : replica_weights) + int64_t(e < masters ? e : e - masters) * size;
    }
    CUTLASS_DEVICE const uint8_t* scale(const uint8_t* p, int e, int64_t size) const {
        return (e < masters ? p : replica_native_scales) + int64_t(e < masters ? e : e - masters) * size;
    }
    CUTLASS_DEVICE const float* outer_scale(const float* p, int e, int64_t size) const {
        return (e < masters ? p : replica_outer_scales) + int64_t(e < masters ? e : e - masters) * size;
    }
    template <class Accum, class Coords, class Scheduler>
    CUTLASS_DEVICE void store(Accum& accum, const Coords& coords, const Scheduler& scheduler,
                             int mb, int nb, int valid, char* smem) const {
        using namespace cute;
        auto* tile = reinterpret_cast<uint32_t*>(smem);
        __syncthreads();
        CUTE_UNROLL
        for (int i = 0; i < size(accum); i += 2) {
            const auto coord = coords(i);
            // K reduction leaves complete accumulators in the first K group.
            if (threadIdx.x < StoreWarps * 32 && get<0>(coord) < valid)
                tile[get<0>(coord) * Stride + get<1>(coord) / 2] =
                    tix::cvt_rte_bf16x2_f32(accum(i + 1), accum(i));
        }
        __syncthreads();
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        for (int row = warp; row < valid; row += blockDim.x / 32) {
            const int64_t pool_row = scheduler.curr_offset_a() / K + mb * M + row;
            const uint64_t route = routes[pool_row];
            const uint32_t rank = route >> 32, slot = (route >> 24) & 255, token = route & 0xffffff;
            auto* dst = reinterpret_cast<__ppu_bfloat16*>(reinterpret_cast<char*>(output) + peer_offsets[rank])
                + (int64_t(slot) * capacity + token) * N + nb * BlockN;
            // Use all 32 lanes for contiguous 64/128/256-column writes.
            if constexpr (BlockN == 256) {
                const uint4 payload = __ppu_shared_load_bulk_b32x4(
                    reinterpret_cast<const uint4*>(tile + row * Stride) + lane);
                if (rank == self_rank)
                    __ppu_global_store_bulk_b32x4(reinterpret_cast<uint4*>(dst) + lane, payload);
                else
                    __ppu_remote_store_bulk_ncp_b32x4(reinterpret_cast<uint4*>(dst) + lane, payload);
            } else if constexpr (BlockN == 128) {
                const uint2 payload = __ppu_shared_load_bulk_b32x2(
                    reinterpret_cast<const uint2*>(tile + row * Stride) + lane);
                if (rank == self_rank)
                    __ppu_global_store_bulk_b32x2(reinterpret_cast<uint2*>(dst) + lane, payload);
                else
                    __ppu_remote_store_bulk_ncp_b32x2(reinterpret_cast<uint2*>(dst) + lane, payload);
            } else {
                const uint32_t payload = __ppu_shared_load_bulk_b32(tile + row * Stride + lane);
                if (rank == self_rank)
                    __ppu_global_store_bulk_b32(reinterpret_cast<uint32_t*>(dst) + lane, payload);
                else
                    __ppu_remote_store_bulk_ncp_b32(reinterpret_cast<uint32_t*>(dst) + lane, payload);
            }
        }
        __syncthreads();
    }
};

} // namespace deep_gemm::mega_moe::nvfp4

namespace deep_gemm {

template <class Backend, int Ranks, int CTAs, int Experts, int Topk,
          int Pulls, int Masters, int G2M, bool Clamp, GatedActivation Activation,
          uint64_t ITI, uint64_t IT, uint64_t ISF, uint64_t PT, uint64_t PSF,
          uint64_t META, int Capacity, uint64_t Routing>
struct MegaMoEFp8Nvfp4Gemm1Kernel {
    static constexpr uint32_t shared_memory_bytes() {
        return Backend::template shared_memory_bytes<Experts>();
    }
    static constexpr int kNumThreads = Backend::kNumThreads;
    // Paired mainloops keep routing and FP32 epilogue state live together.
    // Preserve the native GEMM register budget; shared memory alone can force spills.
    // Single-group tiles retain their higher-residency target.
    static constexpr int kMinBlocks = std::min(
        Backend::Parts > 1 ? Backend::Kernel::MinBlocksPerMultiprocessor : 8,
        int(256 * 1024 / Backend::template shared_memory_bytes<Experts>()));
    // Forward the kernel argument without a thread-local copy of its mapping table.
    static CUTLASS_DEVICE void run(const layout::SymBuffer<>& sb, uint8_t* local_pool,
        int tokens, const int* weights, const uint8_t* native, const float* scales,
        const int* replica, const uint8_t* replica_native, const float* replica_scales,
        uint8_t* output, float* output_scales, float limit, float alpha, float beta, float linear_beta,
        uint4* block_info) {
        auto p = [&](uint64_t off) -> uint8_t* {
            return off >= PT ? local_pool + (off - PT) : reinterpret_cast<uint8_t*>(sb.base + off);
        };
        mega_moe::fp8_blockwise::dispatch_gemm1_kernel_device<
            Ranks, CTAs, Experts, Topk, Backend::ShapeK, Backend::ShapeN, Pulls, Clamp, Masters,
            Activation, Backend::BlockM, G2M, Backend>(
            p(0), sb, reinterpret_cast<int64_t*>(p(ITI)), p(IT), reinterpret_cast<float*>(p(ISF)),
            p(PT), reinterpret_cast<float*>(p(PSF)), reinterpret_cast<int32_t*>(p(META)), block_info,
            reinterpret_cast<const uint8_t*>(weights), scales, output, output_scales,
            tokens, Capacity, limit, alpha, beta, linear_beta, reinterpret_cast<const uint8_t*>(replica),
            replica_scales, Routing, native, replica_native);
    }
};

template <class Builder>
__global__ __launch_bounds__(Builder::kNumThreads, Builder::kMinBlocks) void sm89_fp8_nvfp4_mega_moe_l1(
    layout::SymBuffer<> sym_buffer, uint8_t* local_pool, int num_tokens,
    const int* weights, const uint8_t* native_scales, const float* outer_scales,
    const int* replica_weights, const uint8_t* replica_native_scales, const float* replica_outer_scales,
    uint8_t* output, float* output_scales, float limit, float alpha, float beta, float linear_beta,
    uint4* block_info) {
    Builder::run(sym_buffer, local_pool, num_tokens, weights, native_scales, outer_scales,
        replica_weights, replica_native_scales, replica_outer_scales,
        output, output_scales, limit, alpha, beta, linear_beta, block_info);
}

template <int N, int K, int M, int BlockN, int BlockK, int WarpM, int WarpN, int WarpK,
          int Groups, int Stages, int Ranks>
struct MegaMoEFp8Nvfp4Gemm2Kernel {
    // Scatter keeps routing state live with the accumulators. Consume each
    // LUT entry on demand to shorten its register lifetime.
    using Kernel = cutlass::gemm::kernel::W4A8GEMM<N, K, M, BlockN, BlockK, WarpM, WarpN, WarpK,
        Groups, Stages, GemmType::MGroupedNoPad, 16, 1, 128, true, false, true, 128, false>;
    using Epilogue = mega_moe::nvfp4::Scatter<N, K, M, BlockN, (M / WarpM) * (BlockN / WarpN)>;
    using Base = sched::RoutedMegaMoEScheduler<N, K, M, BlockN, Groups, Ranks>;
    struct Scheduler : Base {
        const int32_t* offsets;
        CUTLASS_DEVICE Scheduler(const TileSchedulerArguments& args, const int32_t* offsets)
            : Base(args), offsets(offsets) {}
        CUTLASS_DEVICE int64_t curr_offset_a() const {
            return int64_t(offsets[this->curr_group_idx]) * K;
        }
    };
    static constexpr uint32_t shared_memory_bytes() {
        return std::max(uint32_t(sizeof(typename Kernel::SharedStorage)),
            uint32_t(offsetof(typename Kernel::SharedStorage, smem_a)) + Epilogue::SharedBytes);
    }
    static constexpr int kNumThreads = Kernel::MaxThreadsPerBlock;
};

template <class Builder>
__global__ __launch_bounds__(Builder::kNumThreads) void sm89_fp8_nvfp4_mega_moe_l2(
    const uint8_t* acts, const float* act_scales,
    const int* weights, const uint8_t* native_scales, const float* outer_scales,
    const int* replica_weights, const uint8_t* replica_native_scales, const float* replica_outer_scales,
    int masters, int32_t* rows, const int32_t* offsets, const int64_t* routes, const int64_t* peer_offsets,
    __ppu_bfloat16* output, int capacity, int pool_rows, int rank) {
    using Kernel = typename Builder::Kernel;
    extern __shared__ char smem[];
    typename Kernel::Params p(reinterpret_cast<const cutlass::float_e4m3_t*>(acts),
        weights, native_scales, reinterpret_cast<cutlass::bfloat16_t*>(output));
    p.num_token = pool_rows; p.ptr_a_scale = act_scales; p.ptr_b_scale = outer_scales;
    const TileSchedulerArguments scheduler_args(pool_rows, rows);
    typename Builder::Scheduler scheduler(scheduler_args, offsets);
    const typename Builder::Epilogue epilogue{output, routes, peer_offsets, replica_weights, replica_native_scales,
        replica_outer_scales, masters, capacity, rank};
    Kernel{}.run(p, smem, scheduler, epilogue);
#if __HGGC_ARCH__ == 150
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
#endif
}

} // namespace deep_gemm
