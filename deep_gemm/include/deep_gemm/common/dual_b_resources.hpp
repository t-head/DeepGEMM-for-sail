#pragma once

namespace deep_gemm {
namespace dual_b {

// Compile-time scheduling choices, local to each kernel specialization.
template <bool EarlyG2S_ = false, bool MmaS2ROverlap_ = false,
          bool StaticStages_ = false, int KOrder_ = 0>
struct PipelinePolicy {
    static constexpr bool EarlyG2S = EarlyG2S_;
    static constexpr bool MmaS2ROverlap = MmaS2ROverlap_;
    static constexpr bool StaticStages = StaticStages_;
    static constexpr int KOrder = KOrder_;
    static_assert(!(EarlyG2S && MmaS2ROverlap), "incompatible copy schedules");
    static_assert(KOrder >= 0 && KOrder <= 6, "unsupported K tile order");
};

template <typename Pipeline_ = PipelinePolicy<>,
          bool OverlapPrologue_ = false, bool ValidEpilogue_ = false,
          bool PackedN_ = false, int StaticM_ = 0, int StaticN_ = 0, int StaticK_ = 0,
          int DecodeSplitK_ = 1, bool PackedStore_ = false,
          bool CtaK_ = false, bool OneTile_ = false>
struct KernelPolicy {
    using Pipeline = Pipeline_;
    static constexpr bool OverlapPrologue = OverlapPrologue_;
    static constexpr bool ValidEpilogue = ValidEpilogue_;
    static constexpr bool PackedN = PackedN_;
    static constexpr int StaticM = StaticM_;
    static constexpr int StaticN = StaticN_;
    static constexpr int StaticK = StaticK_;
    static constexpr int DecodeSplitK = DecodeSplitK_;
    // PackedStore retains the eligibility gate for the validated BF16 variants.
    static constexpr bool PackedStore = PackedStore_;
    static constexpr bool CtaK = CtaK_;
    static constexpr bool OneTile = OneTile_;
    static_assert((StaticM == 0 && StaticN == 0 && StaticK == 0) ||
                  (StaticM > 0 && StaticN > 0 && StaticK > 0), "incomplete static shape");
    static_assert(PackedStore || (!CtaK && !OneTile && Pipeline::KOrder == 0),
                  "BF16 variant options require packed-store eligibility");
    static_assert(DecodeSplitK == 1 || (PackedN && StaticM > 0 &&
                  (DecodeSplitK == 2 || DecodeSplitK == 4 || DecodeSplitK == 8)),
                  "decode partitions require a static packed-N kernel");
    static_assert(!Pipeline::StaticStages || (StaticM > 0 && !PackedN),
                  "static stages require a static ordinary kernel");
    static_assert(!(OverlapPrologue || ValidEpilogue) ||
                  (!Pipeline::EarlyG2S && !Pipeline::MmaS2ROverlap),
                  "incompatible epilogue overlap schedule");
    static_assert(!PackedN || (!CtaK && !Pipeline::EarlyG2S && !Pipeline::MmaS2ROverlap &&
                  !OverlapPrologue && !ValidEpilogue), "incompatible packed-N schedule");
};

constexpr int shared_storage_bytes(int bm, int bn, int bk, int wm, int wn, int wk,
                                   int stages, bool packed_n, int partitions,
                                   bool cta_k, int element_bytes = 2) {
    const int operands = partitions * stages * (bm + (packed_n ? 1 : 2) * bn) * bk * element_bytes;
    const int threads = (bm / wm) * (bn / wn) * (bk / wk) * partitions * 32;
    const int partials = cta_k ? threads * (wm * wn / 32) * 2 * 4 : 0;
    return ((operands > partials ? operands : partials) + 127) & ~127;
}

} // namespace dual_b
} // namespace deep_gemm
