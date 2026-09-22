#pragma once

#include <cstdint>

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm {

enum class EpilogueType {
    Default,
    SiluAndMulPostQuantFp4,
    SiluAndMulStepPostQuantFp4,    // STEP Models
};

inline constexpr const char* EpilogueTypeS[] = { "Default", "SiluAndMulPostQuantFp4", "SiluAndMulStepPostQuantFp4"};

template <EpilogueType T>
struct EpilogueTraits {
    using ElementD = cutlass::bfloat16_t;
    using ElementDPtr = __ppu_bfloat16;
    static constexpr bool needs_sfd_offset = false;
    static constexpr int  kOutElemsPerByte    = 1;

    static constexpr bool is_valid_config(const int shape_n, const int block_n, const int n_expand, const bool has_bias) { return true; }

    static constexpr int get_shape_n_out(const int shape_n) { return shape_n; }

    static constexpr int get_shape_sfn_out(const int shape_n) = delete;
};

template <>
struct EpilogueTraits<EpilogueType::SiluAndMulPostQuantFp4> {
    using ElementD = uint8_t;
    using ElementDPtr = uint8_t;
    static constexpr bool needs_sfd_offset = true;
    static constexpr int  kOutElemsPerByte    = 4;
    static constexpr int kQuantGroupSize = 32;  // 64 values share 2*e8m0 scale

    static constexpr bool is_valid_config(const int shape_n, const int block_n, const int n_expand, const bool has_bias) {
        return ((block_n >= 64 && block_n % 64 == 0 && shape_n % 64 == 0) && n_expand == 1 && has_bias == false);
    }

    static constexpr int get_shape_n_out(const int shape_n) {
        return (shape_n / kOutElemsPerByte);
    }

    static constexpr int get_shape_sfn_out(const int shape_n) {
        return math::ceil_div(shape_n / kOutElemsPerByte, kQuantGroupSize);
    }
};

template <>
struct EpilogueTraits<EpilogueType::SiluAndMulStepPostQuantFp4>: EpilogueTraits<EpilogueType::SiluAndMulPostQuantFp4> {};

// The fused activation + mxfp4 post-quant epilogue is shared by every activation variant
// (currently plain swiglu and the STEP silu-then-clamp variant, and any future activation types).
// Any host/device code that selects the fused epilogue path must treat all of them alike -- only
// the in-epilogue activation math differs
__device__ __host__ constexpr bool is_act_and_quant_fused(EpilogueType t) {
    return t == EpilogueType::SiluAndMulPostQuantFp4
        || t == EpilogueType::SiluAndMulStepPostQuantFp4;
}

} // namespace deep_gemm
