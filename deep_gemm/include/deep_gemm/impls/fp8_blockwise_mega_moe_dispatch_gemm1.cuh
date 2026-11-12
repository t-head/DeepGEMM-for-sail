#pragma once
#include <tools/util/include/cutlass/util/packed_stride.hpp>
#include <deep_gemm/impls/mega_moe_io.cuh>
#include <deep_gemm/scheduler/mega_moe_work_queue.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>

// Native block-FP8 fused dispatch/GEMM1.
#include <deep_gemm/impls/mega_moe_transport.cuh>
#include <deep_gemm/impls/fp8_gemm.cuh>

namespace deep_gemm::mega_moe::fp8_blockwise {

// Gate/up retain native 128-row blockwise weight-scale groups.
template <int kN, int kK, int kComputeM = 128>
struct Gemm1 {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using TileShape = cute::Shape<cute::Int<kComputeM>, cute::_256, cute::_128>;
    using WarpShape = cute::Shape<cute::Int<(kComputeM < 64 ? 16 : kComputeM / 4)>, cute::_64, cute::_128>;
    using ScaleShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<
        ScaleShape, false, true>(ScaleShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::PPU0015, cutlass::arch::OpClassTensorOp,
        ElementA, cute::tuple<cutlass::layout::RowMajor, LayoutSFA>, 1,
        ElementB, cute::tuple<cutlass::layout::ColumnMajor, LayoutSFB>, 1,
        float, TileShape, WarpShape, cute::_4,
        cutlass::gemm::KernelAiuMultistageWithBlockWiseScale>::CollectiveOp;
    static constexpr int kNumThreads = cute::size(typename CollectiveMainloop::TiledMma{});

    template <int kExperts>
    static constexpr uint32_t shared_memory_bytes() {
        // Expert counts, pull staging and GEMM execute in separate phases and
        // reuse the same allocation. GEMM1's epilogue also fits its mainloop storage.
        constexpr uint32_t gemm = sizeof(typename CollectiveMainloop::SharedStorage);
        constexpr uint32_t pull = (kNumThreads / 32) * kK;
        constexpr uint32_t counts = kExperts * sizeof(uint32_t);
        constexpr uint32_t data = gemm > pull ? gemm : pull;
        return data > counts ? data : counts;
    }
};

template <int kN, int kNumThreads, bool kClamp, bool kSwigluOai, class Accum, class Coordinates>
__device__ __forceinline__ void swiglu_quant_epilogue(
        Accum& accum, const Coordinates& coords, uint8_t* output, float* scales,
        int pool_rows, int valid_rows, int n_block, int thread, char* smem,
        float swiglu_limit, float activation_alpha, int writable_rows) {
    using namespace cute;
    // Match the BF16 clamp boundary used by the other FP8 activation backends.
    if constexpr (kClamp)
        swiglu_limit = __bfloat162float(__float2bfloat16(swiglu_limit));
    // One FP32 word of padding avoids subwarp shared-memory bank aliasing.
    constexpr int kStride = 258;
    auto* tile = reinterpret_cast<__ppu_bfloat16*>(smem);
    __syncthreads();
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < size<1>(coords); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < size<1>(shape<0>(coords)); ++mm) {
            const int base_m = get<0>(coords(make_coord(Int<0>{}, mm, Int<0>{}), wm, Int<0>{}));
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < size<2>(coords); ++wn) {
                CUTLASS_PRAGMA_UNROLL
                for (int mn = 0; mn < size<2>(shape<0>(coords)); ++mn) {
                    auto frag = accum(make_coord(_, mm, mn), wm, wn);
                    const int col = get<1>(coords(make_coord(Int<0>{}, mm, mn), wm, wn)) - n_block * 256;
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < size(frag) / 2; ++i) {
                        const int row = base_m + i * 8;
                        if (row < valid_rows)
                            *reinterpret_cast<__ppu_bfloat162*>(tile + row * kStride + col) =
                                __floats2bfloat162_rn(frag(2 * i), frag(2 * i + 1));
                    }
                }
            }
        }
    }
    __syncthreads();
    const int lane = thread % 8;
    for (int row = thread / 8; row < writable_rows; row += kNumThreads / 8) {
        float values[16];
        float amax = 1e-10f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
            float2 activated = make_float2(0.0f, 0.0f);
            if (row < valid_rows) {
                float2 g = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(
                    tile + row * kStride + lane * 16 + i * 2));
                float2 u = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(
                    tile + row * kStride + 128 + lane * 16 + i * 2));
                if constexpr (kClamp) {
                    g.x = fminf(g.x, swiglu_limit);
                    g.y = fminf(g.y, swiglu_limit);
                    u.x = fminf(fmaxf(u.x, -swiglu_limit), swiglu_limit);
                    u.y = fminf(fmaxf(u.y, -swiglu_limit), swiglu_limit);
                }
                if constexpr (kSwigluOai) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
                    // Honor fast_math without changing the BF16 rounding boundaries.
                    activated = make_float2(
                        (g.x * deep_gemm::mega_moe::detail::moe_reciprocal<DG_MOE_FAST_MATH>(1.0f + deep_gemm::mega_moe::detail::moe_exp<DG_MOE_FAST_MATH>(-activation_alpha * g.x))) * (u.x + 1.0f),
                        (g.y * deep_gemm::mega_moe::detail::moe_reciprocal<DG_MOE_FAST_MATH>(1.0f + deep_gemm::mega_moe::detail::moe_exp<DG_MOE_FAST_MATH>(-activation_alpha * g.y))) * (u.y + 1.0f));
#else
                    activated = make_float2(
                        (g.x / (1.0f + deep_gemm::mega_moe::detail::moe_exp<DG_MOE_FAST_MATH>(-activation_alpha * g.x))) * (u.x + 1.0f),
                        (g.y / (1.0f + deep_gemm::mega_moe::detail::moe_exp<DG_MOE_FAST_MATH>(-activation_alpha * g.y))) * (u.y + 1.0f));
#endif
                } else {
                    activated = make_float2(
                        (g.x / (1.0f + deep_gemm::mega_moe::detail::moe_exp<DG_MOE_FAST_MATH>(-g.x))) * u.x,
                        (g.y / (1.0f + deep_gemm::mega_moe::detail::moe_exp<DG_MOE_FAST_MATH>(-g.y))) * u.y);
                }
                // Round the activation to BF16 before FP8 quantization.
                activated = __bfloat1622float2(__float22bfloat162_rn(activated));
            }
            values[2 * i] = activated.x;
            values[2 * i + 1] = activated.y;
            amax = fmaxf(amax, fmaxf(fabsf(activated.x), fabsf(activated.y)));
        }
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 4; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 8));
        if (lane == 0)
            scales[int64_t(n_block) * pool_rows + row] = amax * (1.0f / 448.0f);
        const float inv = 448.0f / amax;
        __align__(16) __hg_fp8x2_e4m3 packed[8];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i)
            packed[i] = __hg_fp8x2_e4m3(make_float2(values[2 * i] * inv, values[2 * i + 1] * inv));
        *reinterpret_cast<uint4*>(output + int64_t(row) * (kN / 2) + n_block * 128 + lane * 16) =
            *reinterpret_cast<uint4*>(packed);
    }
    __syncthreads();
}

template <int kN, int kK, bool kClamp, bool kSwigluOai = false, int kComputeM = 128>
__device__ __forceinline__ void gemm1_block(
        const uint8_t* a, const float* sa, const uint8_t* b, const float* sb,
        uint8_t* output, float* output_scales, int pool_rows, int valid_rows, int n_block,
        int thread, char* smem, float swiglu_limit, float activation_alpha, int writable_rows) {
    using namespace cute;
    using G = Gemm1<kN, kK, kComputeM>;
    using Mainloop = typename G::CollectiveMainloop;
    using Tile = typename G::TileShape;
    using Mma = typename Mainloop::TiledMma;
    using StrideA = typename Mainloop::StrideA;
    using StrideB = typename Mainloop::StrideB;
    auto problem = make_shape(valid_rows, kN, kK, 1);
    typename Mainloop::Params params{
        reinterpret_cast<const typename G::ElementA*>(a),
        cutlass::make_cute_packed_stride(StrideA{}, make_shape(valid_rows, kK, 1)),
        reinterpret_cast<const typename G::ElementB*>(b),
        cutlass::make_cute_packed_stride(StrideB{}, make_shape(kN, kK, 1)), 4,
        sa, G::ScaleConfig::tile_atom_to_shape_SFA(make_shape(pool_rows, kN, kK, 1)),
        sb, G::ScaleConfig::tile_atom_to_shape_SFB(make_shape(pool_rows, kN, kK, 1))};
    Mainloop mainloop;
    auto block_coord = make_coord(0, n_block, _, 0);
    auto inputs = mainloop.load_init(problem, block_coord, params);
    auto gA = get<0>(inputs);
    Mma mma;
    auto accum = partition_fragment_C(mma, take<0, 2>(Tile{}));
    clear(accum);
    auto k_iter = make_coord_iterator(shape<2>(gA));
    mainloop(params, inputs, accum, k_iter, int(size<2>(gA)), thread, smem);

    auto coordinates = make_identity_tensor(make_shape(valid_rows, kN));
    auto c_tile = local_tile(coordinates, take<0, 2>(Tile{}), make_coord(0, n_block));
    auto t_coord = mma.get_thread_slice(thread).partition_C(c_tile);
    static_assert(kComputeM * 258 * sizeof(__ppu_bfloat16) <= sizeof(typename Mainloop::SharedStorage));
    swiglu_quant_epilogue<kN, G::kNumThreads, kClamp, kSwigluOai>(accum, t_coord, output,
        output_scales, pool_rows, valid_rows, n_block, thread, smem, swiglu_limit, activation_alpha, writable_rows);
}

template <uint32_t kRanks, uint32_t kCTAs, uint32_t kExperts,
          uint32_t kTopk, uint32_t kHidden, uint32_t kN, uint32_t kPullCTAs,
          bool kClamp = false,
          uint32_t kMasterExperts = kExperts / kRanks, bool kSwigluOai = false,
          uint32_t kComputeM = 128, uint32_t kGemm2BlockM = 0>
CUTLASS_DEVICE void dispatch_gemm1_kernel_device(
        void* workspace_base, const layout::SymBuffer<>& sym_buffer,
        int64_t* ids, const uint8_t* input,
        const float* scales, uint8_t* pool, float* pool_scales,
        int32_t* gemm2_metadata, uint4* gemm2_block_info, const uint8_t* w1, const float* s1,
        uint8_t* output, float* output_scales,
        uint32_t num_tokens, uint32_t cap, float swiglu_limit = 0.0f, float activation_alpha = 1.0f,
        const uint8_t* w1_replicas = nullptr, const float* s1_replicas = nullptr) {
    static_assert(kRanks > 0 && kRanks <= 64 && kExperts % kRanks == 0);
    constexpr uint32_t kEpr = kExperts / kRanks;
    constexpr uint32_t kBlockM = 128, kBlockN = 256;
    constexpr uint32_t kThreads = Gemm1<kN, kHidden, kComputeM>::kNumThreads;
    constexpr uint32_t kWarps = kThreads / 32, kScales = kHidden / 128;
    constexpr uint32_t kNBlocks = kN / kBlockN;
    static_assert(kHidden % 128 == 0 && kN % kBlockN == 0);
    static_assert(kEpr > 0 && kEpr <= 128);
    static_assert(kMasterExperts > 0 && kMasterExperts <= kEpr);
    static_assert(kPullCTAs > 0 && kPullCTAs < kCTAs);
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t sm = blockIdx.x, thread = threadIdx.x;
    const uint32_t warp = thread / 32, lane = thread % 32;
    const layout::Workspace ws(workspace_base, kRanks, kExperts, cap, kTopk);
    constexpr uint32_t kWorkBatch = 2;
    if (sm == 0 && thread == 0)
        *ws.get_gemm1_work_count_ptr() = (kCTAs - kPullCTAs) * kWorkBatch;
    for (uint32_t g = sm * kThreads + thread;
         g < ws.num_max_pool_tokens / kBlockM; g += kCTAs * kThreads)
        *ws.get_l1_arrival_count_ptr(g) = 0;
    mega_moe::detail::dispatch_handshake<kRanks, kCTAs, kThreads, kExperts, kEpr, kTopk, true>(
        ws, sym_buffer, ids, reinterpret_cast<uint32_t*>(smem), num_tokens, sm, thread);

    const sched::DistributedMegaMoEBlocks<kEpr, kBlockM> blocks(ws.get_expert_recv_count_sum_ptr(), lane);
    const uint32_t total_blocks = blocks.total_blocks;
    const layout::Gemm2Metadata gemm2_meta(gemm2_metadata, kEpr);
    if (sm == 0 && warp == 0)
        gemm2_meta.store_expert_layout<kEpr, kBlockM, kGemm2BlockM>(blocks, lane, gemm2_block_info);
    static_assert(kComputeM == 128 ||
        (kComputeM == 32 || kComputeM == 192));
    auto* work_counter = ws.get_gemm1_work_count_ptr();
    // Reuse the MMA arena between tiles. The epilogue retires its reads
    // before prime(); ArrivalCache::wait retires ticket reads before MMA.
    const mega_moe::WorkQueue queue{work_counter, reinterpret_cast<uint32_t*>(smem)};

    struct TileInfo { uint32_t expert, begin, valid, local_begin; };
    const auto resolve = [&](uint32_t g) {
        uint4 info;
        blocks.get_block(g, info);
        const uint32_t expert = info.x, rows = info.y, begin = info.z;
        const uint32_t local_begin = (g - begin) * kBlockM;
        return TileInfo{expert, g * kBlockM, mega_moe::detail::umin(kBlockM, rows - local_begin), local_begin};
    };

    if (sm < kPullCTAs) {
        auto* scratch = smem + warp * kHidden;
        // Each producer owns complete M tiles. Different producers can pull
        // subsequent tiles while consumer CTAs compute already-published ones.
        for (uint32_t g = sm; g < total_blocks; g += kPullCTAs) {
            const auto tile = resolve(g);
            const mega_moe::detail::WarpSourceRanges<kRanks> sources(ws, tile.expert, lane);
            for (uint32_t row = warp; row < tile.valid; row += kWarps) {
                const uint32_t in_expert = tile.local_begin + row;
                const auto source = sources.resolve(in_expert, lane);
                const uint32_t src = source.rank, slot = source.slot;
                uint32_t route = lane == 0 ? *ws.get_src_token_topk_idx_ptr(tile.expert, src, slot) : 0;
                route = __shfl_sync(0xffffffffu, route, 0);
                const uint32_t token = route / kTopk, topk_slot = route % kTopk;
                const uint32_t dst = tile.begin + row;
                mega_moe::detail::warp_async_copy_global_to_shared(scratch,
                    sym_buffer.map(input + uint64_t(token) * kHidden, src), kHidden, lane);
                for (uint32_t sf = lane; sf < kScales; sf += 32)
                    pool_scales[uint64_t(sf) * ws.num_max_pool_tokens + dst] =
                        *sym_buffer.map(scales + uint64_t(token) * kScales + sf, src);
                if (lane == 0) {
                    *ws.get_token_src_metadata_ptr(dst) = {src, token, topk_slot};
                    gemm2_meta.store_route(dst, src, token, topk_slot);
                }
                cute::cp_async_wait<0>();
                __syncwarp();
                mega_moe::detail::warp_copy_bytes(pool + uint64_t(dst) * kHidden, scratch, kHidden, lane);
                __syncwarp();
            }
#if __HGGC_ARCH__ == 150
            __ppu_wait_vstcnt(0);
#endif
            __threadfence();
            __syncthreads();
            if (thread == 0)
                tix::red_add_rel(ws.get_l1_arrival_count_ptr(g), 1u);
        }
    }
    {
        mega_moe::ArrivalCache ready;
        uint32_t work = (sm - kPullCTAs) * kWorkBatch;
        if (sm < kPullCTAs) {
            queue.prime(thread, kWorkBatch);
            work = queue.current();
        }
        uint32_t batch_end = work + kWorkBatch;
        sched::MegaMoEGemm1Scheduler scheduler;
        sched::RowCountMegaMoEBlocks<kEpr, kComputeM, kBlockM, uint64_t> metadata{
            ws.get_expert_recv_count_sum_ptr()};
        uint32_t m, n;
        #pragma clang loop licm(disable)
        while (scheduler.select_work<kComputeM, kNBlocks>(work, m, n, metadata)) {
            const uint32_t expert = scheduler.curr_group_idx, rows = scheduler.curr_group_m;
            const uint32_t dispatch_blocks = math::ceil_div(rows, kBlockM);
            const uint32_t local_begin = m * kComputeM;
            const uint32_t valid = mega_moe::detail::umin(kComputeM, rows - local_begin);
            const uint32_t base = scheduler.curr_cumsum_m + local_begin;
            const uint32_t first = base / kBlockM, last = (base + valid - 1) / kBlockM;
            ready.wait(ws.get_l1_arrival_count_ptr(0), first, last, total_blocks);
            asm volatile("" ::: "memory");
            uint32_t weight_expert = expert;
            const uint8_t* weight_base = w1;
            const float* scale_base = s1;
            if constexpr (kMasterExperts < kEpr) {
                if (weight_expert >= kMasterExperts) {
                    weight_expert -= kMasterExperts;
                    weight_base = w1_replicas;
                    scale_base = s1_replicas;
                }
            }
            const uint32_t writable = local_begin + kComputeM >= rows
                ? dispatch_blocks * kBlockM - local_begin : kComputeM;
            auto* tile_output = output + uint64_t(base) * (kN / 2);
            gemm1_block<kN, kHidden, kClamp, kSwigluOai, kComputeM>(
                pool + uint64_t(base) * kHidden, pool_scales + base,
                weight_base + uint64_t(weight_expert) * kN * kHidden,
                scale_base + uint64_t(weight_expert) * (kN / 128) * kScales,
                tile_output, output_scales + base,
                ws.num_max_pool_tokens, valid, n, thread, reinterpret_cast<char*>(smem),
                swiglu_limit, activation_alpha, writable);
            if (++work == batch_end) {
                queue.prime(thread, kWorkBatch);
                work = queue.current();
                batch_end = work + kWorkBatch;
            }
        }
    }
}

} // namespace deep_gemm::mega_moe::fp8_blockwise
