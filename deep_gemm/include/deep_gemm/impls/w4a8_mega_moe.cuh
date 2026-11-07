#pragma once

#include <deep_gemm/impls/w4fa8_gemm.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/impls/mega_moe_io.cuh>

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

template <int BlockM, int BlockN, int K, int WarpM, int WarpN, int Stages, bool OAI, bool SiTU,
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
                            float gate = __bfloat162float(__float2bfloat16(acc(2*r+v, m, n) * sf));
                            float up = __bfloat162float(__float2bfloat16(acc(2*r+v+4, m, n) * sf));
                            float value;
                            if constexpr (SiTU) {
                                const float soft_gate = situ_beta *
                                    detail::moe_tanh<DG_MOE_FAST_MATH>(gate * detail::moe_reciprocal<DG_MOE_FAST_MATH>(situ_beta));
                                const float soft_up = situ_linear_beta > 0 ? situ_linear_beta *
                                    detail::moe_tanh<DG_MOE_FAST_MATH>(up * detail::moe_reciprocal<DG_MOE_FAST_MATH>(situ_linear_beta)) : up;
                                value = soft_gate * detail::moe_sigmoid<DG_MOE_FAST_MATH>(gate) * soft_up;
                            } else {
                                if (limit >= 0) {
                                    const float bound = __bfloat162float(__float2bfloat16(limit));
                                    gate = fminf(gate, bound);
                                    up = fminf(fmaxf(up, -bound), bound);
                                }
                                value = gate * detail::moe_sigmoid<DG_MOE_FAST_MATH>(OAI ? alpha * gate : gate)
                                            * (OAI ? up + 1.0f : up);
                            }
                            output[int64_t(row) * stride + dst + v] = __float2bfloat16(value);
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
                        const int col = get<1>(coords(v, m, n));
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
