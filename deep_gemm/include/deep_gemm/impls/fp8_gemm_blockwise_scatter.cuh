#pragma once
#include <deep_gemm/impls/fp8_gemm.cuh>
#include <deep_gemm/impls/moe_gemm_scatter.cuh>

namespace deep_gemm::mega_moe::fp8_blockwise {

// Reuse the common BF16 scatter implementation, including packed routes,
// shared-memory vector loads and local/remote NCP stores. Only the mapping from
// a padded pool row to its expert differs from that path.
struct BlockwiseScatter : mega_moe::detail::Bf16MoeScatter<true, true> {
    using Base = mega_moe::detail::Bf16MoeScatter<true, true>;
    struct Arguments : Base::Arguments {
        const int32_t* row_offsets{nullptr};
    };
    using Params = Arguments;
    static Params to_underlying_arguments(const Arguments& args) { return args; }

    CUTLASS_DEVICE static uint32_t get_overlap_valid_m(
            const Params& params, uint32_t pool_row_base,
            uint32_t row_start, uint32_t fallback_m) {
        // upper_bound handles repeated offsets from empty experts. A scheduled
        // tile always belongs to the last nonempty interval containing its row.
        uint32_t lo = 0, hi = params.num_groups;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (uint32_t(__ldg(params.row_offsets + mid)) <= pool_row_base)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo == 0) return 0;
        const int valid = __ldg(params.valid_m_rows + lo - 1);
        return valid > int(row_start) ? cute::min(fallback_m, uint32_t(valid) - row_start) : 0;
    }
};

// Each producer must finish its own remote stores before the later IPC
// barrier publishes readiness. Do not drain per tile: NCP writes may remain
// outstanding while this CTA computes its next GEMM tile.
template <class NativeKernel>
struct CompletedScatterKernel : NativeKernel {
    CUTLASS_DEVICE void operator()(const typename NativeKernel::Params& params, char* smem) {
        NativeKernel::operator()(params, smem);
#if __HGGC_ARCH__ == 150
        __ppu_wait_remote_vstcnt(0);
        __ppu_wait_vstcnt(0);
#endif
        __threadfence_system();
        __syncthreads();
    }
};

template <int N, int K, int Groups, int MasterGroups = Groups>
struct Gemm2Scatter {
    // Native blockwise mainloop and GroupedNoPad scheduler, with the existing
    // shared MoE epilogue adapter. The checkpoint scales remain 128x128.
    // Three stages plus the disjoint BF16 scatter tile fit in 256 KiB.
    static constexpr int Stages = (N == 4096 && K == 2048 && MasterGroups == 72) ? 3
        : (N == 2560 && K == 640 && Groups == 128) ? 3 : 2;
    using G = Fp8Gemm<N, K, 128, 256, 128, 32, 64, 0, 0, Groups, Stages,
                     GemmType::GroupedNoPad>;
    using Mainloop = typename G::CollectiveMainloop;
    using Epilogue = mega_moe::detail::MoeOverlapScatterEpilogue<
        typename G::CollectiveEpilogue, typename Mainloop::SharedStorage,
        BlockwiseScatter, typename G::TileShape, 512, N, false, true>;
    static_assert(Epilogue::kUsesDisjointBf16Scratch,
                  "blockwise fused scatter must not materialize global D");
    using Scheduler = DeepGemmScheduler<GemmType::GroupedNoPad, N, K, 128, 256, Groups,
        ceil_div(N, 256), 2, EpilogueType::Default, true, true>;
    using Kernel = CompletedScatterKernel<DeepGemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue, Scheduler, false, false, true>>;

};

}  // namespace deep_gemm::mega_moe::fp8_blockwise
