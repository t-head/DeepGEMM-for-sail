#pragma once
#include <deep_gemm/scheduler/mega_moe_work_queue.cuh>
#include <deep_gemm/layout/mega_moe_types.cuh>

// Native block-FP8 fused dispatch/GEMM1.
#include <deep_gemm/impls/fp8_mega_moe_transport.cuh>
#include <deep_gemm/impls/fp8_gemm.cuh>

namespace deep_gemm::mega_moe::fp8_blockwise {

// Gate/up retain native 128-row blockwise weight-scale groups.
template <int kN, int kK, Gemm1Epilogue kEpilogue = Gemm1Epilogue::IdentityBf16, int kComputeM = 128>
struct Gemm1 : Fp8Gemm<kN, kK, kComputeM, (kEpilogue == Gemm1Epilogue::SwiGluFp8) ? 256 : 128, 128,
                      kComputeM / 4, (kEpilogue == Gemm1Epilogue::SwiGluFp8) ? 64 : 32, 0, 128, 1, 4, GemmType::Normal> {
    using Base = Fp8Gemm<kN, kK, kComputeM, (kEpilogue == Gemm1Epilogue::SwiGluFp8) ? 256 : 128, 128,
                         kComputeM / 4, (kEpilogue == Gemm1Epilogue::SwiGluFp8) ? 64 : 32, 0, 128, 1, 4, GemmType::Normal>;
    using ScaleShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<
        ScaleShape, false, true>(ScaleShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        typename Base::ArchTag, cutlass::arch::OpClassTensorOp,
        typename Base::ElementA, cute::tuple<typename Base::LayoutA, LayoutSFA>, Base::AlignmentA,
        typename Base::ElementB, cute::tuple<typename Base::LayoutB, LayoutSFB>, Base::AlignmentB,
        float, typename Base::TileShape, typename Base::WarpShape, cute::Int<Base::Stage>,
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
        float swiglu_limit, float swiglu_alpha, int writable_rows) {
    using namespace cute;
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
                    // Use the native PPU reciprocal for the sigmoid denominator.
                    // Gate/up and activation retain their BF16 rounding boundaries.
                    activated = make_float2(
                        (g.x * __ppu_rcpf(1.0f + __expf(-swiglu_alpha * g.x))) * (u.x + 1.0f),
                        (g.y * __ppu_rcpf(1.0f + __expf(-swiglu_alpha * g.y))) * (u.y + 1.0f));
#else
                    activated = make_float2(
                        (g.x / (1.0f + __expf(-swiglu_alpha * g.x))) * (u.x + 1.0f),
                        (g.y / (1.0f + __expf(-swiglu_alpha * g.y))) * (u.y + 1.0f));
#endif
                } else {
                    activated = make_float2(
                        (g.x / (1.0f + __expf(-g.x))) * u.x,
                        (g.y / (1.0f + __expf(-g.y))) * u.y);
                }
                // Preserve the existing BF16 activation rounding boundary.
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

template <int kN, int kK, Gemm1Epilogue kEpilogue, bool kClamp, bool kSwigluOai = false, int kComputeM = 128>
__device__ __forceinline__ void gemm1_block(
        const uint8_t* a, const float* sa, const uint8_t* b, const float* sb,
        void* output, float* output_scales, int pool_rows, int valid_rows, int n_block,
        int thread, char* smem, float swiglu_limit, float swiglu_alpha, int writable_rows) {
    using namespace cute;
    using G = Gemm1<kN, kK, kEpilogue, kComputeM>;
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
    if constexpr ((kEpilogue == Gemm1Epilogue::SwiGluFp8)) {
        static_assert(kComputeM * 258 * sizeof(__ppu_bfloat16) <= sizeof(typename Mainloop::SharedStorage));
        swiglu_quant_epilogue<kN, G::kNumThreads, kClamp, kSwigluOai>(accum, t_coord, static_cast<uint8_t*>(output),
            output_scales, pool_rows, valid_rows, n_block, thread, smem, swiglu_limit, swiglu_alpha, writable_rows);
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size(accum); ++j) {
            auto coord = t_coord(j);
            const int row = get<0>(coord), col = get<1>(coord);
            if (row < valid_rows && col < kN)
                static_cast<__ppu_bfloat16*>(output)[int64_t(row) * kN + col] =
                    __float2bfloat16_rn(accum(j));
        }
    }
    // All warps must finish collective shared-memory reads before this CTA
    // reuses its multistage storage for the next expert tile.
    __syncthreads();
}

template <uint32_t kRanks, uint32_t kCTAs, uint32_t kExperts,
          uint32_t kTopk, uint32_t kHidden, uint32_t kN, uint32_t kPullCTAs,
          Gemm1Epilogue kEpilogue = Gemm1Epilogue::IdentityBf16, bool kClamp = false,
          uint32_t kMasterExperts = kExperts / kRanks, bool kSwigluOai = false,
          uint32_t kComputeM = 128, bool kDynamicWork = false>
CUTLASS_DEVICE void dispatch_gemm1_kernel_device(
        void* workspace_base, const layout::SymBuffer<kRanks>& sym_buffer,
        const int64_t* ids, const float* weights, const uint8_t* input,
        const float* scales, uint8_t* pool, float* pool_scales,
        float* pool_weights, const uint8_t* w1, const float* s1,
        void* output, float* output_scales,
        uint32_t num_tokens, uint32_t cap, float swiglu_limit = 0.0f, float swiglu_alpha = 1.0f,
        const uint8_t* w1_replicas = nullptr, const float* s1_replicas = nullptr) {
    static_assert(kRanks > 0 && kRanks <= 64 && kExperts % kRanks == 0);
    static_assert(!kClamp || (kEpilogue == Gemm1Epilogue::SwiGluFp8));
    constexpr uint32_t kEpr = kExperts / kRanks;
    constexpr uint32_t kBlockM = 128, kBlockN = (kEpilogue == Gemm1Epilogue::SwiGluFp8) ? 256 : 128;
    constexpr uint32_t kThreads = Gemm1<kN, kHidden, kEpilogue, kComputeM>::kNumThreads;
    constexpr uint32_t kWarps = kThreads / 32, kScales = kHidden / 128;
    constexpr uint32_t kChunks = (kEpr + 31) / 32;
    constexpr uint32_t kNBlocks = kN / kBlockN;
    static_assert(kHidden % 128 == 0 && kN % kBlockN == 0);
    static_assert(kEpr > 0 && kEpr <= 128);
    static_assert(kMasterExperts > 0 && kMasterExperts <= kEpr);
    static_assert(kPullCTAs > 0 && kPullCTAs < kCTAs);
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t sm = blockIdx.x, thread = threadIdx.x;
    const uint32_t warp = thread / 32, lane = thread % 32;
    const layout::Workspace ws(workspace_base, kRanks, kExperts, cap, kTopk);
    const auto block_sync = [=]() { __syncthreads(); };
    mega_moe::detail::dispatch_handshake<kRanks, kCTAs, kThreads, kExperts, kEpr, kTopk>(
        ws, sym_buffer, ids, reinterpret_cast<uint32_t*>(smem), num_tokens, sm, thread);

    // Four warp-distributed expert chunks cover Qwen's full EP4 EPR=128.
    uint32_t expert_rows[kChunks], block_begin[kChunks], block_end[kChunks];
    uint32_t total_blocks = 0;
    #pragma unroll
    for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
        const uint32_t e = chunk * 32 + lane;
        expert_rows[chunk] = e < kEpr ? uint32_t(*ws.get_expert_recv_count_sum_ptr(e)) : 0;
        const uint32_t blocks = math::ceil_div(expert_rows[chunk], kBlockM);
        uint32_t prefix = blocks;
        #pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            const uint32_t v = __shfl_up_sync(0xffffffffu, prefix, offset);
            if (lane >= offset) prefix += v;
        }
        block_begin[chunk] = total_blocks + prefix - blocks;
        block_end[chunk] = total_blocks + prefix;
        total_blocks += __shfl_sync(0xffffffffu, prefix, 31);
    }
    const layout::Gemm2Metadata gemm2_meta(pool_weights + ws.num_max_pool_tokens, kEpr);
    if (sm == 0 && warp == 0) {
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
            const uint32_t expert = chunk * 32 + lane;
            if (expert < kEpr) {
                const uint32_t padded = (block_end[chunk] - block_begin[chunk]) * kBlockM;
                const uint32_t offset = block_begin[chunk] * kBlockM;
                gemm2_meta.store_rows(expert, expert_rows[chunk], padded, offset, padded, offset);
            }
        }
    }
    static_assert(kComputeM == 128 || (kComputeM == 192 && kEpilogue == Gemm1Epilogue::SwiGluFp8));
    constexpr uint32_t kWorkBatch = 2;
    auto* work_counter = ws.get_gemm1_work_count_ptr();
    __shared__ uint32_t cta_work;
    const mega_moe::WorkQueue queue{work_counter, &cta_work};
    if constexpr (kDynamicWork) {
        if (sm == 0 && thread == 0) *work_counter = (kCTAs - kPullCTAs) * kWorkBatch;
    }
    for (uint32_t g = sm * kThreads + thread; g < total_blocks; g += kCTAs * kThreads)
        *ws.get_l1_arrival_count_ptr(g) = 0;
    comm::grid_sync<kCTAs, 2>(ws, sm, thread, block_sync);

    struct TileInfo { uint32_t expert, begin, valid, local_begin; };
    const auto resolve = [&](uint32_t g) {
        uint32_t expert = kEpr, rows = 0, begin = 0;
        #pragma unroll
        for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
            const uint32_t mask = __ballot_sync(0xffffffffu,
                chunk * 32 + lane < kEpr && g >= block_begin[chunk] && g < block_end[chunk]);
            if (mask) {
                const uint32_t owner = uint32_t(__ffs(mask) - 1);
                expert = chunk * 32 + owner;
                rows = __shfl_sync(0xffffffffu, expert_rows[chunk], owner);
                begin = __shfl_sync(0xffffffffu, block_begin[chunk], owner);
            }
        }
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
                    pool_weights[dst] = *sym_buffer.map(weights + route, src);
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
    if (sm >= kPullCTAs || kDynamicWork) {
        uint32_t work = (sm - kPullCTAs) * (kDynamicWork ? kWorkBatch : 1);
        if constexpr (kDynamicWork) {
            if (sm < kPullCTAs) {
                queue.prime(thread, kWorkBatch);
                work = queue.current();
            }
        }
        uint32_t batch_end = work + kWorkBatch;
        uint32_t first_work = 0, pool_base = 0;
        // Both static strides and queue tickets are monotonic. Keep an expert
        // cursor instead of scanning all preceding experts for every tile.
        #pragma unroll 1
        for (uint32_t expert = 0; expert < kEpr; ++expert) {
            const uint32_t rows = uint32_t(__ldg(ws.get_expert_recv_count_sum_ptr(expert)));
            const uint32_t dispatch_blocks = math::ceil_div(rows, kBlockM);
            const uint32_t jobs = math::ceil_div(rows, kComputeM) * kNBlocks;
            // Keep mainloop descriptors inside the tile loop to bound register
            // lifetimes in BM192 specializations.
            #pragma clang loop licm(disable)
            while (work < first_work + jobs) {
                const uint32_t local_work = work - first_work;
                const uint32_t local_begin = (local_work / kNBlocks) * kComputeM;
                const uint32_t n = local_work % kNBlocks;
                const uint32_t valid = mega_moe::detail::umin(kComputeM, rows - local_begin);
                const uint32_t base = pool_base + local_begin;
                const uint32_t first = base / kBlockM, last = (base + valid - 1) / kBlockM;
                for (uint32_t ready = first; ready <= last; ++ready)
                    while (tix::ld_acq(ws.get_l1_arrival_count_ptr(ready)) == 0) {}
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
                auto* tile_output = static_cast<uint8_t*>(output) + uint64_t(base) *
                    (kEpilogue == Gemm1Epilogue::SwiGluFp8 ? kN / 2 : kN * 2);
                gemm1_block<kN, kHidden, kEpilogue, kClamp, kSwigluOai, kComputeM>(
                    pool + uint64_t(base) * kHidden, pool_scales + base,
                    weight_base + uint64_t(weight_expert) * kN * kHidden,
                    scale_base + uint64_t(weight_expert) * (kN / 128) * kScales,
                    tile_output, kEpilogue == Gemm1Epilogue::SwiGluFp8 ? output_scales + base : nullptr,
                    ws.num_max_pool_tokens, valid, n, thread, reinterpret_cast<char*>(smem),
                    swiglu_limit, swiglu_alpha, writable);
                if constexpr (kDynamicWork) {
                    if (++work == batch_end) {
                        queue.prime(thread, kWorkBatch);
                        work = queue.current();
                        batch_end = work + kWorkBatch;
                    }
                } else {
                    work += kCTAs - kPullCTAs;
                }
            }
            first_work += jobs;
            pool_base += dispatch_blocks * kBlockM;
        }
    }
    comm::grid_sync<kCTAs, 1>(ws, sm, thread, block_sync);
}

} // namespace deep_gemm::mega_moe::fp8_blockwise
