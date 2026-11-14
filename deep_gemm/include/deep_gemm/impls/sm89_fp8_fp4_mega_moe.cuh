#pragma once

#include <deep_gemm/impls/w4fa8_gemm.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>
#include <deep_gemm/epilogue/activation_quant_ops.cuh>
#include <deep_gemm/impls/mega_moe_io.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/impls/fp8_channelwise_mega_moe_dispatch_gemm1.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>

namespace deep_gemm::mega_moe::w4a8 {

struct SingleTile {
    int curr_group_m;
    int curr_group_idx = 0;
    bool pending = true;
    CUTLASS_DEVICE int64_t curr_offset_a() const { return 0; }
    CUTLASS_DEVICE int64_t curr_offset_c() const { return 0; }
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m, uint32_t& n) {
        m = n = 0;
        const bool result = pending;
        pending = false;
        return result;
    }
};

template <int BlockM, int BlockN, int K, int WarpM, int WarpN, int Stages, GatedActivation Activation,
          w4fa8::FragmentPolicy Fragment, w4fa8::ALoadPolicy ALoad>
struct Gemm1 {
    using Kernel = cutlass::gemm::kernel::W4A8GEMM_MMA<
        BlockN, K, BlockM, BlockN, 128, WarpM, WarpN, 128,
        1, Stages, GemmType::MGroupedNoPad, 1, false, Fragment, ALoad>;

    struct Epilogue {
        __ppu_bfloat16* output;
        const float* row_scales;
        int stride;
        float limit, alpha, inv_scale, situ_beta, situ_linear_beta;
        uint32_t *queue_counter, *next_ticket;
        CUTLASS_DEVICE const uint8_t* weight(const uint8_t* p, int, int) const { return p; }
        CUTLASS_DEVICE const uint8_t* scale(const uint8_t* p, int, int) const { return p; }
        CUTLASS_DEVICE int valid_rows(int, int m) const { return m; }

        template <class Accum, class Coords, class Scheduler>
        CUTLASS_DEVICE void store(Accum& acc, const Coords& coords, const Scheduler&,
                                  int, int, int valid) const {
            using namespace cute;
            // The last mainloop barrier retired operand reads. Claim the next
            // tile while the current activation epilogue stores its result.
            if (queue_counter && threadIdx.x == 0)
                *next_ticket = atomicAdd(queue_counter, 1u);
            // The W4A8 MMA fragment pairs gate/up at v and v+4 (eight columns).
            static_assert(decltype(size<0>(acc))::value == 8);
            CUTE_UNROLL
            for (int m = 0; m < size<1>(acc); ++m) {
                CUTE_UNROLL
                for (int r = 0; r < 2; ++r) {
                    const int row = get<0>(coords(2 * r, m, 0));
                    if (row >= valid) continue;
                    const float sf = row_scales[row] * inv_scale;
                    CUTE_UNROLL
                    for (int n = 0; n < size<2>(acc); ++n) {
                        const int col = get<1>(coords(2 * r, m, n));
                        const int dst = (col / 16) * 8 + col % 8;
                        CUTE_UNROLL
                        for (int v = 0; v < 2; ++v) {
                            float gate = acc(2*r+v, m, n) * sf;
                            float up = acc(2*r+v+4, m, n) * sf;
                            const float value = routed_gated_activation<Activation, bool(DG_MOE_FAST_MATH)>(
                                gate, up, limit, alpha, situ_beta, situ_linear_beta);
                            output[int64_t(row) * stride + dst + v] = epilogue::quant::bf16(value);
                        }
                    }
                }
            }
        }
    };

    CUTLASS_DEVICE static void run(const uint8_t* a, const float* a_scale,
        const uint8_t* b, const uint8_t* b_scale, __ppu_bfloat16* out,
        int stride, int valid_m, char* smem, float limit, float alpha, float inv_scale,
        float situ_beta, float situ_linear_beta, uint32_t* queue_counter, uint32_t* next_ticket) {
        typename Kernel::Params params{};
        params.ptr_a = reinterpret_cast<const cutlass::float_e4m3_t*>(a);
        params.ptr_b = b;
        params.ptr_scale = b_scale;
        params.ptr_d = reinterpret_cast<cutlass::bfloat16_t*>(out);
        params.ptr_a_scale = a_scale;
        SingleTile scheduler{valid_m};
        Kernel{}.run(params, smem, scheduler,
                     Epilogue{out, a_scale, stride, limit, alpha, inv_scale, situ_beta, situ_linear_beta,
                              queue_counter, next_ticket});
        __syncthreads();
    }
};

template <int N, int K, int BlockM, int BlockN>
struct Scatter {
    // A bulk store covers 256 BF16 columns with all 32 lanes on one peer.
    static_assert(N % 256 == 0 && BlockN % 256 == 0);
    static constexpr int Stride = BlockN / 2 + 4;
    static constexpr uint32_t SharedBytes = BlockM * Stride * sizeof(uint32_t);
    __ppu_bfloat16* output;
    const float* row_scales;
    const int32_t* valid;
    const int64_t* routes;
    const int64_t* rank_offsets;
    const uint8_t *replica_weights, *replica_scales;
    int masters, capacity;
    float inv_scale, replica_inv_scale;
    char* smem;
    int self_rank;

    CUTLASS_DEVICE const uint8_t* weight(const uint8_t* p, int e, int64_t size) const {
        return (e < masters ? p : replica_weights) + int64_t(e < masters ? e : e - masters) * size;
    }
    CUTLASS_DEVICE const uint8_t* scale(const uint8_t* p, int e, int64_t size) const {
        return (e < masters ? p : replica_scales) + int64_t(e < masters ? e : e - masters) * size;
    }
    CUTLASS_DEVICE int valid_rows(int e, int) const { return valid[e]; }

    template <class Accum, class Coords, class Scheduler>
    CUTLASS_DEVICE void store(Accum& acc, const Coords& coords, const Scheduler& scheduler,
                              int m_block, int n_block, int residual_m) const {
        using namespace cute;
        auto* tile = reinterpret_cast<uint32_t*>(smem);
        // The W4A8 mainloop ends every K iteration with a CTA barrier.
        // N_EXPAND=1 and WarpK=BlockK leave no shared operands in use here.
        __syncthreads();
        CUTE_UNROLL
        for (int m = 0; m < size<1>(acc); ++m) {
            CUTE_UNROLL
            for (int r = 0; r < 2; ++r) {
                const int row = get<0>(coords(2*r, m, 0));
                if (row >= residual_m) continue;
                const int64_t pool_row = scheduler.curr_offset_a()/K + m_block*BlockM + row;
                const float sf = row_scales[pool_row] *
                    (scheduler.curr_group_idx < masters ? inv_scale : replica_inv_scale);
                CUTE_UNROLL
                for (int n = 0; n < size<2>(acc); ++n) {
                    CUTE_UNROLL
                    for (int half = 0; half < 2; ++half) {
                        const int v = 2*r + 4*half;
                        const uint32_t col = get<1>(coords(v, m, n));
                        tile[row*Stride + col/2] = tix::cvt_rte_bf16x2_f32(acc(v+1,m,n)*sf, acc(v,m,n)*sf);
                    }
                }
            }
        }
        __syncthreads();
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        for (int row = warp; row < min(BlockM, residual_m); row += blockDim.x/32) {
            const int64_t pool_row = scheduler.curr_offset_a()/K + m_block*BlockM + row;
            const uint64_t route = routes[pool_row];
            const uint32_t rank = route >> 32, slot = (route >> 24) & 255, token = route & 0xffffff;
            auto* dst = reinterpret_cast<__ppu_bfloat16*>(reinterpret_cast<char*>(output) + rank_offsets[rank])
                + (int64_t(slot)*capacity+token)*N + n_block*BlockN;
            for (int vec = 0; vec < BlockN/8; vec += 32) {
                if (n_block*BlockN + vec*8 >= N) break;
                const uint4 payload = __ppu_shared_load_bulk_b32x4(
                    reinterpret_cast<const uint4*>(tile+row*Stride) + vec + lane);
                auto* address = reinterpret_cast<uint4*>(dst) + vec + lane;
                if (rank == self_rank) __ppu_global_store_bulk_b32x4(address, payload);
                else __ppu_remote_store_bulk_ncp_b32x4(address, payload);
            }
        }
        __syncthreads();
    }
};

} // namespace deep_gemm::mega_moe::w4a8


namespace deep_gemm {

template <int BLOCK_M_, int GEMM1_BLOCK_N_, int TOKEN_BYTES_, int GEMM1_WARP_M_, int GEMM1_WARP_N_,
          int GEMM1_STAGES_, GatedActivation ACTIVATION_, w4fa8::FragmentPolicy FRAGMENT_POLICY_,
          w4fa8::ALoadPolicy A_LOAD_POLICY_, int NUM_EXPERTS_, int NUM_THREADS_, int NUM_RANKS_, int NUM_CTAS_,
          int NUM_EXPERTS_PER_RANK_, int NUM_TOPK_, int GEMM1_N_, int PULL_CTAS_, int PULL_GROUP_SIZE_,
          int GEMM2_BLOCK_M_, bool GEMM2_VALID_ROWS_, uint64_t LAYOUT_OFF_ITI_, uint64_t LAYOUT_OFF_IT_,
          uint64_t LAYOUT_OFF_ISF_, uint64_t LAYOUT_OFF_PT_, uint64_t LAYOUT_OFF_PSF_,
          uint64_t LAYOUT_OFF_GEMM2_META_, int NMTPR_, uint64_t StagingCounts_, uint64_t StagingTokens_>
struct MegaMoEFp8Fp4Gemm1Kernel {
    static constexpr layout::RoutedMoEStaging staging{StagingCounts_, StagingTokens_};
    static constexpr int BLOCK_M = BLOCK_M_, GEMM1_BLOCK_N = GEMM1_BLOCK_N_, TOKEN_BYTES = TOKEN_BYTES_;
    static constexpr int GEMM1_WARP_M = GEMM1_WARP_M_, GEMM1_WARP_N = GEMM1_WARP_N_, GEMM1_STAGES = GEMM1_STAGES_;
    static constexpr GatedActivation ACTIVATION = ACTIVATION_;
    static constexpr w4fa8::FragmentPolicy FRAGMENT_POLICY = FRAGMENT_POLICY_;
    static constexpr w4fa8::ALoadPolicy A_LOAD_POLICY = A_LOAD_POLICY_;
    static constexpr int NUM_EXPERTS = NUM_EXPERTS_, NUM_THREADS = NUM_THREADS_, NUM_RANKS = NUM_RANKS_;
    static constexpr int NUM_CTAS = NUM_CTAS_, NUM_EXPERTS_PER_RANK = NUM_EXPERTS_PER_RANK_, NUM_TOPK = NUM_TOPK_;
    static constexpr int GEMM1_N = GEMM1_N_, PULL_CTAS = PULL_CTAS_, PULL_GROUP_SIZE = PULL_GROUP_SIZE_;
    static constexpr int GEMM2_BLOCK_M = GEMM2_BLOCK_M_;
    static constexpr bool GEMM2_VALID_ROWS = GEMM2_VALID_ROWS_;
    static constexpr uint64_t LAYOUT_OFF_ITI = LAYOUT_OFF_ITI_, LAYOUT_OFF_IT = LAYOUT_OFF_IT_;
    static constexpr uint64_t LAYOUT_OFF_ISF = LAYOUT_OFF_ISF_, LAYOUT_OFF_PT = LAYOUT_OFF_PT_;
    static constexpr uint64_t LAYOUT_OFF_PSF = LAYOUT_OFF_PSF_, LAYOUT_OFF_GEMM2_META = LAYOUT_OFF_GEMM2_META_;
    static constexpr int NMTPR = NMTPR_;

    using Backend =
        mega_moe::w4a8::Gemm1<BLOCK_M, GEMM1_BLOCK_N, TOKEN_BYTES, GEMM1_WARP_M, GEMM1_WARP_N,
                              GEMM1_STAGES, ACTIVATION, FRAGMENT_POLICY, A_LOAD_POLICY>;
    static constexpr uint32_t compute_bytes = sizeof(typename Backend::Kernel::SharedStorage);
    static constexpr uint32_t dispatch_bytes =
        ((NUM_EXPERTS * 4 + 15) & ~15u) + NUM_THREADS / 32 * TOKEN_BYTES;
    static constexpr uint32_t shared_memory_bytes() {
        return compute_bytes > dispatch_bytes ? compute_bytes : dispatch_bytes;
    }
    static constexpr int kNumThreads = NUM_THREADS;
    static constexpr int kMinBlocks = 1;
};

template <typename Builder>
__global__ __launch_bounds__(Builder::kNumThreads, Builder::kMinBlocks) void sm89_fp8_fp4_mega_moe_l1(
    layout::SymBuffer<> sb, int tokens, const uint8_t* w, const uint8_t* s, float inv, const uint8_t* rw,
    const uint8_t* rs, float rinv, int masters, __ppu_bfloat16* out, float limit, float alpha,
    float situ_beta, float situ_linear_beta, uint4* gemm2_block_info) {
    using Backend = typename Builder::Backend;
    auto p = [&](int64_t off) { return reinterpret_cast<uint8_t*>(sb.base + off); };
    mega_moe::fp8_channelwise::dispatch_gemm1_device<
        Builder::NUM_RANKS, Builder::NUM_CTAS, Builder::NUM_THREADS, Builder::NUM_EXPERTS, Builder::NUM_EXPERTS_PER_RANK,
            Builder::NUM_TOPK, Builder::BLOCK_M,
        Builder::TOKEN_BYTES, Builder::GEMM1_N, Builder::GEMM1_BLOCK_N, 128, Builder::GEMM1_WARP_M, Builder::GEMM1_WARP_N,
            Builder::GEMM1_STAGES,
        Builder::PULL_CTAS, Builder::ACTIVATION, Backend, Builder::PULL_GROUP_SIZE, Builder::GEMM2_BLOCK_M,
            Builder::GEMM2_VALID_ROWS>(
        p(0), sb, reinterpret_cast<int64_t*>(p(Builder::LAYOUT_OFF_ITI)), p(Builder::LAYOUT_OFF_IT),
        reinterpret_cast<float*>(p(Builder::LAYOUT_OFF_ISF)), p(Builder::LAYOUT_OFF_PT),
        reinterpret_cast<float*>(p(Builder::LAYOUT_OFF_PSF)), reinterpret_cast<int32_t*>(p(Builder::LAYOUT_OFF_GEMM2_META)),
        gemm2_block_info, reinterpret_cast<const cutlass::float_e4m3_t*>(w), reinterpret_cast<const float*>(s),
        reinterpret_cast<const cutlass::float_e4m3_t*>(rw), reinterpret_cast<const float*>(rs), masters, out, limit,
        alpha, tokens, Builder::NMTPR, inv, rinv, situ_beta, situ_linear_beta, Builder::staging);
}

template <int N_, int K_, int BLOCK_M_, int BLOCK_N_, int WARP_M_, int WARP_N_, int NUM_GROUPS_, int NUM_STAGES_,
          w4fa8::FragmentPolicy FRAGMENT_POLICY_, w4fa8::ALoadPolicy A_LOAD_POLICY_, int NUM_THREADS_,
          int NUM_RANKS_>
struct MegaMoEFp8Fp4Gemm2Kernel {
    static constexpr int N = N_, K = K_, BLOCK_M = BLOCK_M_, BLOCK_N = BLOCK_N_, WARP_M = WARP_M_;
    static constexpr int WARP_N = WARP_N_, NUM_GROUPS = NUM_GROUPS_, NUM_STAGES = NUM_STAGES_;
    static constexpr w4fa8::FragmentPolicy FRAGMENT_POLICY = FRAGMENT_POLICY_;
    static constexpr w4fa8::ALoadPolicy A_LOAD_POLICY = A_LOAD_POLICY_;
    static constexpr int NUM_THREADS = NUM_THREADS_, NUM_RANKS = NUM_RANKS_;

    using Kernel = cutlass::gemm::kernel::W4A8GEMM_MMA<N, K, BLOCK_M, BLOCK_N, 128, WARP_M, WARP_N,
                                                       128, NUM_GROUPS, NUM_STAGES, GemmType::MGroupedNoPad, 1,
                                                       false, FRAGMENT_POLICY, A_LOAD_POLICY>;
    using Epilogue = mega_moe::w4a8::Scatter<N, K, BLOCK_M, BLOCK_N>;

    // Keep the thread bound without a minimum-residency hint: HGGC's two-argument
    // launch bound can miscompile this shared-memory scatter epilogue.
    static constexpr uint32_t shared_memory_bytes() {
        return Epilogue::SharedBytes > sizeof(typename Kernel::SharedStorage) ? Epilogue::SharedBytes
                                                                              : sizeof(typename Kernel::SharedStorage);
    }
    static constexpr int kNumThreads = NUM_THREADS;
};

template <typename Builder>
__global__ __launch_bounds__(Builder::kNumThreads) void sm89_fp8_fp4_mega_moe_l2(
    const cutlass::float_e4m3_t* a, const float* a_scale, const uint8_t* w, const uint8_t* s, float inv,
    const uint8_t* rw, const uint8_t* rs, float rinv, int masters, int32_t* rows, const int32_t* valid,
    const int64_t* routes, __ppu_bfloat16* output, const int64_t* offsets, int capacity, int pool, int self_rank) {
    using Kernel = typename Builder::Kernel;
    using Epilogue = typename Builder::Epilogue;
    extern __shared__ char smem[];
    typename Kernel::Params params{};
    params.ptr_a = a;
    params.ptr_b = w;
    params.ptr_scale = s;
    params.ptr_d = reinterpret_cast<cutlass::bfloat16_t*>(output);
    params.ptr_a_scale = a_scale;
    params.scheduler = TileSchedulerArguments(pool, rows);
    sched::RoutedMegaMoEScheduler<Builder::N, Builder::K, Builder::BLOCK_M, Builder::BLOCK_N, Builder::NUM_GROUPS,
        Builder::NUM_RANKS,
                                  (Builder::BLOCK_M == 16 ? 1 : 2)>
        scheduler(params.scheduler);
    const Epilogue epilogue{output,  a_scale,  valid, routes, offsets, rw,       rs,
                            masters, capacity, inv,   rinv,   smem,    self_rank};
    Kernel{}.run(params, smem, scheduler, epilogue);
#if __HGGC_ARCH__ == 150
    // Complete every producer's stores before combine publishes this rank.
    __ppu_wait_remote_vstcnt(0);
    __ppu_wait_vstcnt(0);
#endif
}

} // namespace deep_gemm
