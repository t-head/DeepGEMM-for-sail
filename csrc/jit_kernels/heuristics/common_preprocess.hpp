#pragma once
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
using namespace deep_gemm;
namespace deep_gemm_preprocess_common {

/// Weight quantization format, one entry per mode DeepGemm supports (mirrors the W4A16Type
/// family plus the standalone fp4/fp8/int8/bf16 GEMM precisions).
enum class PreprocessMode {
    MXFP4,            // w4a4 mxfp4 weight with E8M0 block scales
    W4A16_MXFP4,      // w4a16 fp4 weight with E8M0 block scales
    W4A16_MXFP4_MMA,  // w4a16 fp4 weight, E8M0 scales, mma dequant kernel
    W4A16_INT4,       // w4a16 int4 weight with bf16 scales
    FP8_CHANNEL,      // fp8 weight with per-channel scales
    FP8_BLOCK,        // fp8 weight with block scales
    INT8,             // int8 weight with per-channel scales
    BF16,             // unquantized bf16 weight
    W4A8_MXFP4_MMA,   // E4M3 activation, FP4 MMA dequant with E8M0 scales
    W4A8_INT8,        // w4a8 int4 weight with per-channel scales
};

struct PreprocessedWeights {
    torch::Tensor weight;
    torch::Tensor scale;
    // Only used in W4A8_MXFP4_MMA: Fold into the activation row scale
    float inv_scale = 1.0f;
};

inline const char* preprocess_mode_name(const PreprocessMode& mode) {
    switch (mode) {
        case PreprocessMode::MXFP4:           return "mxfp4";
        case PreprocessMode::W4A16_MXFP4:     return "w4a16-mxfp4";
        case PreprocessMode::W4A16_MXFP4_MMA: return "w4a16-mxfp4-mma";
        case PreprocessMode::W4A8_MXFP4_MMA: return "w4a8-mxfp4-mma";
        case PreprocessMode::W4A16_INT4:      return "w4a16-int4";
        case PreprocessMode::FP8_CHANNEL:     return "fp8-channel";
        case PreprocessMode::FP8_BLOCK:       return "fp8-block";
        case PreprocessMode::INT8:            return "int8";
        case PreprocessMode::BF16:            return "bf16";
        case PreprocessMode::W4A8_INT8:       return "w4a8-int8";
    }
    return "unknown";
}

/// Prepare weights and scales for the selected kernel layout. MXFP4 activation fusion
/// interleaves gate/up rows; W4FA8 and W4FA16 MMA keep the natural packed weight order.
/// `mode` selects the weight format (PreprocessMode) and `fuse_activation` requests interleaving
/// for the activation-fused epilogue; both are mandatory with no defaults. Weights may be
/// omitted for MXFP4/W4A8_MXFP4_MMA without fusion; scales remain required. Supported: MXFP4 with or without activation
/// fusion, and W4A16_INT4/W4A16_MXFP4/W4A16_MXFP4_MMA/W4A8_MXFP4_MMA without it.
inline PreprocessedWeights preprocess_weights_and_scales(
        const PreprocessMode& mode, bool fuse_activation,
        const torch::Tensor& weight, const std::optional<const torch::Tensor>& weight_scale) {
    const bool scale_only = !fuse_activation &&
        (mode == PreprocessMode::MXFP4 || mode == PreprocessMode::W4A8_MXFP4_MMA);
    TORCH_CHECK(weight.defined() || scale_only,
                "preprocess mode '", preprocess_mode_name(mode), "' with fuse_activation=", fuse_activation,
                " requires weights");
    if ((mode == PreprocessMode::W4A16_INT4 || mode == PreprocessMode::W4A16_MXFP4)
            && !fuse_activation) {
        const bool int4 = mode == PreprocessMode::W4A16_INT4;
        TORCH_CHECK(weight.dim() == 3 && weight.scalar_type() == torch::kUInt8,
                    "W4A16 packed weights must be uint8 [E,N,K/2]");
        TORCH_CHECK(weight_scale.has_value(), "W4A16 requires weight scales");
        const auto& scales = *weight_scale;
        const int64_t e = weight.size(0), n = weight.size(1), k = weight.size(2) * 2;
        TORCH_CHECK(e > 0 && n > 0 && n % 64 == 0 && k > 0 && k % 32 == 0,
                    "W4A16 requires E>0, N%64=0 and K%32=0");
        TORCH_CHECK(scales.scalar_type() == (int4 ? torch::kBFloat16 : torch::kUInt8) &&
                    scales.sizes() == std::vector<int64_t>({e, n, k / 32}),
                    "W4A16 scales must be [E,N,K/32], BF16 for INT4 or uint8 E8M0 for MXFP4");
        TORCH_CHECK(weight.device() == scales.device(), "weights and scales must share a device");

        // Reproduce the 16x16 weight tiles and the MMA register ordering.
        std::vector<int64_t> weight_perm;
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int block = 0; block < 2; ++block) {
                    for (int row : {2 * (i % 4), 2 * (i % 4) + 1,
                                    2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1}) {
                        weight_perm.push_back(16 * row + i / 4 + 8 * block + 256 * j);
                    }
                }
            }
        }
        std::vector<int64_t> interleaved_perm;
        for (int i = 0; i < 1024; i += 8)
            for (int j : {0, 2, 4, 6, 1, 3, 5, 7})
                interleaved_perm.push_back(weight_perm[i + j]);
        const auto options = at::TensorOptions().dtype(at::kLong);
        auto indices = at::tensor(interleaved_perm, options).to(weight.device());
        auto codes = at::stack({at::bitwise_and(weight, 15), at::bitwise_right_shift(weight, 4)}, -1)
                         .reshape({e, n, k}).transpose(1, 2).contiguous();
        codes = codes.reshape({e, k / 16, 16, n / 16, 16}).permute({0, 1, 3, 2, 4})
                     .reshape({-1, 1024}).index_select(1, indices).reshape({e, k / 16, n * 16})
                     .to(torch::kInt32);
        auto packed = at::zeros({e, k / 16, n * 2}, weight.options().dtype(torch::kInt32));
        for (int j = 0; j < 8; ++j)
            packed.bitwise_or_(at::bitwise_left_shift(codes.slice(2, j, n * 16, 8), 4 * j));

        std::vector<int64_t> scale_perm;
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                scale_perm.push_back(i + 8 * j);
        if (!int4) {
            auto original = scale_perm;
            for (int i = 0; i < 16; ++i) {
                const int order[] = {0, 2, 1, 3};
                for (int j = 0; j < 4; ++j)
                    scale_perm[4 * i + j] = original[4 * i + order[j]];
            }
        }
        indices = at::tensor(scale_perm, options).to(scales.device());
        auto rearranged = scales.transpose(1, 2).contiguous().reshape({-1, 64})
                              .index_select(1, indices).reshape({e, k / 32, n}).contiguous();
        return {packed, rearranged};
    }
    if ((mode == PreprocessMode::W4A8_MXFP4_MMA || mode == PreprocessMode::W4A16_MXFP4_MMA)
            && !fuse_activation) {
        const bool w4a8 = mode == PreprocessMode::W4A8_MXFP4_MMA;
        const char* name = w4a8 ? "W4FA8" : "W4FA16 MMA";
        TORCH_CHECK(weight_scale.has_value(), name, " requires weight scales");
        const auto& packed = weight;
        const auto& scales = *weight_scale;
        TORCH_CHECK(scales.dim() == 3 && scales.scalar_type() == torch::kUInt8,
                    name, " raw E8M0 scales must be uint8 [E,N,K/32]");
        const int64_t e = scales.size(0), n = scales.size(1), k = scales.size(2) * 32;
        TORCH_CHECK(e>0 && n>0 && n%64==0 && k>0 && k%(w4a8 ? 128 : 64)==0,
                    name, " requires E>0, N a multiple of 64 and K a multiple of ", w4a8 ? 128 : 64);
        if (packed.defined()) {
            TORCH_CHECK(packed.scalar_type() == torch::kUInt8 &&
                        packed.sizes() == std::vector<int64_t>({e, n, k / 2}),
                        name, " packed weights must be uint8 [E,N,K/2] matching scales");
            TORCH_CHECK(packed.device() == scales.device(), "weights and scales must share a device");
        }
        int exponent_shift = 0;
        torch::Tensor adjusted_scales = scales;
        if (w4a8) {
            const int min_scale = scales.min().item<int>();
            const int max_scale = scales.max().item<int>();
            TORCH_CHECK(min_scale > 0 && max_scale < 255 && max_scale - min_scale <= 26,
                        "W4FA8 E8M0 scale range [", min_scale, ",", max_scale,
                        "] cannot fit [114,140]: 0/255 are invalid and spread must be <=26");
            if (min_scale < 114)
                exponent_shift = 114 - min_scale;
            else if (max_scale > 140)
                exponent_shift = 140 - max_scale;
            if (exponent_shift != 0)
                adjusted_scales = scales.to(torch::kInt16).add(exponent_shift).to(torch::kUInt8);
        }
        std::vector<int64_t> perm;
        for (int i=0;i<8;++i) for(int j=0;j<8;++j) {
            perm.push_back(2*i+16*j);
            perm.push_back(2*i+16*j+1);
        }
        const auto indices=at::tensor(perm,at::TensorOptions().dtype(at::kLong)).to(scales.device());
        auto rearranged=adjusted_scales.reshape({e,n/64,64,k/64,2}).permute({0,1,3,2,4}).contiguous();
        rearranged=rearranged.reshape({-1,128}).index_select(1,indices).reshape({e,n/64,k*2}).contiguous();
        return {packed.defined() ? packed.contiguous() : packed, rearranged, std::ldexp(1.0f, -exponent_shift)};
    }
    if (mode == PreprocessMode::MXFP4) {
        TORCH_CHECK(weight_scale.has_value(), "MXFP4 requires scales");
        auto weight_out = weight;
        auto scale_out = *weight_scale;
        if (fuse_activation) {
            DG_HOST_ASSERT(weight.dim() == 3);
            DG_HOST_ASSERT(weight.dtype() == torch::kUInt8);
            DG_HOST_ASSERT(weight.is_contiguous());
            DG_HOST_ASSERT(weight_scale.has_value());
            DG_HOST_ASSERT(weight_scale->dim() == 3);
            DG_HOST_ASSERT(weight_scale->dtype() == torch::kUInt8);
            DG_HOST_ASSERT(weight_scale->is_contiguous());

            // do interleaving to make up and gate be adjacent.
            const int64_t num_groups = weight.size(0), n = weight.size(1), k = weight.size(2);
            DG_HOST_ASSERT(n % 2 == 0);  // silu_and_mul splits N in half
            const int64_t half_n = n / 2;
            const auto gate = weight.slice(1, 0, half_n);
            const auto up = weight.slice(1, half_n, n);
            // `at::stack` writes a fresh contiguous tensor, so the following `view` is always legal.
            weight_out = at::stack({gate, up}, 2).view({num_groups, n, k});

            const int64_t sfn = weight_scale->size(1), sfk = weight_scale->size(2);
            // `weight` is uint8, so one byte holds two fp4 values: sfk == ceil_div(2 * k, 32).
            DG_HOST_ASSERT(weight_scale->size(0) == num_groups && sfn == n && sfk == ceil_div<int64_t>(k, 16));
            const auto gate_scale = weight_scale->slice(1, 0, half_n);
            const auto up_scale = weight_scale->slice(1, half_n, n);
            scale_out = at::stack({gate_scale, up_scale}, 2).view({num_groups, sfn, sfk});
        }
        // Keep the packed weight unchanged unless gate/up interleaving was requested.
        // The legacy scale-only API uses an undefined weight and returns only the scale.
        auto scales = scale_out.contiguous();
        if (scales.scalar_type() == torch::kUInt16) scales = scales.view(torch::kUInt8);
        DG_HOST_ASSERT(scales.scalar_type() == torch::kUInt8);
        DG_HOST_ASSERT(scales.dim() == 2 || scales.dim() == 3);
        if (scales.size(-1) % 2 != 0)
            scales = at::constant_pad_nd(scales, {0, 1}, 0);
        scales = scales.view(torch::kUInt16);
        if (scales.dim() == 2)
            scales = scales.t().contiguous().t();
        else
            scales = scales.permute({0, 2, 1}).contiguous().permute({0, 2, 1});
        return {weight_out, scales};
    }
    // ============================================================================
    // A8W4 INT8 Device Weight Preprocess
    // ============================================================================
    // The model stores weights in the portable packed signed-INT4 format. This device-aware transform
    // only reorders packed bytes into the layout expected by the PPU A8W4 kernels.

    /// Reorder portable packed signed-INT4 MoE weights into the PPU A8W4 kernel layout.
    /// `weight` is contiguous uint8 with shape [..., N, K / 2]. Byte i stores logical k=2*i in the low
    /// nibble and k=2*i+1 in the high nibble, both in two's-complement signed-INT4 representation.
    // First transpose 2-byte units inside every 16-byte (32-element K) group so one TSM_LD hands each
    // lane the K values of its two K32 INT8-MMA atoms. Then transpose each eight-nibble converter group
    // from {a,b,c,d,e,f,g,h} to packed bytes {a,e}, {b,f}, {c,g}, {d,h}. Finally toggle every nibble's
    // sign bit, changing two's-complement INT4 into the unsigned, zero-point-8 encoding consumed by the
    // seven-instruction register converter.
    if (mode == PreprocessMode::W4A8_INT8) {
        TORCH_CHECK(!fuse_activation, "Fuse activation and quant functions into A8W4-INT8 path is not supported.");
        auto weight_out = weight;
        auto scale_out = weight_scale.value_or(torch::Tensor{});
        TORCH_CHECK(weight.dtype() == torch::kUInt8,
                    "preprocess_weights_and_scales[w4a8-int8]: weight must use portable packed signed-INT4 uint8 storage");
        TORCH_CHECK(weight.is_contiguous(), "preprocess_weights_and_scales[w4a8-int8]: weight must be contiguous");
        TORCH_CHECK(weight.dim() >= 2, "preprocess_weights_and_scales[w4a8-int8]: weight must have shape [..., N, K / 2]");
        const int64_t k_bytes = weight.size(-1);
        TORCH_CHECK(k_bytes % 16 == 0,
                    "preprocess_weights_and_scales[w4a8-int8]: K must be a multiple of 32 (K / 2 = ", k_bytes, ")");

        // [..., K / 32, j1 = 2, j2 = 4, bytes = 2] 2-byte units
        // -> [..., K / 32, j2 = 4, j1 = 2, bytes = 2], so one TSM_LD hands each lane the K values of
        // its two K32 INT8-MMA atoms. Keep the transform in uint8 so a contiguous input with an odd
        // storage offset does not require int16 alignment.
        std::vector<int64_t> grouped_shape = weight.sizes().vec();
        grouped_shape.back() = k_bytes / 16;  // K32 groups: 16 packed bytes each
        grouped_shape.push_back(2);           // j1: two halves of the K32 group
        grouped_shape.push_back(4);           // j2: four 2-byte units per half
        grouped_shape.push_back(2);           // bytes: the two bytes of one unit
        auto reordered = weight.view(grouped_shape).transpose(-3, -2).contiguous().flatten(-4);

        // [..., K / 8, bytes = 4]: one converter group of eight nibbles {a,b,c,d,e,f,g,h},
        // stored as bytes {a,b}, {c,d}, {e,f}, {g,h} (low nibble first).
        std::vector<int64_t> converter_shape = reordered.sizes().vec();
        converter_shape.back() = k_bytes / 4;  // converter groups: 4 packed bytes each
        converter_shape.push_back(4);          // bytes: the four bytes of one group
        auto blocks = reordered.view(converter_shape);
        auto low = blocks.bitwise_and(0x0f);        // per byte: a, c, e, g
        auto high = blocks.bitwise_right_shift(4);  // per byte: b, d, f, h
        auto pack_nibbles = [](const torch::Tensor& lower, const torch::Tensor& upper) {
            return lower.bitwise_or(upper.bitwise_left_shift(4));
        };
        // {a,b,c,d,e,f,g,h} -> bytes {a,e}, {b,f}, {c,g}, {d,h}, so the converter reads
        // a..d from the low nibbles and e..h from the high nibbles of one uint32.
        auto transposed = torch::stack({
            pack_nibbles(low.select(-1, 0), low.select(-1, 2)),    // {a,e}
            pack_nibbles(high.select(-1, 0), high.select(-1, 2)),  // {b,f}
            pack_nibbles(low.select(-1, 1), low.select(-1, 3)),    // {c,g}
            pack_nibbles(high.select(-1, 1), high.select(-1, 3)),  // {d,h}
        }, -1);
        // Flip each nibble's sign bit: two's-complement INT4 v -> unsigned v + 8.
        weight_out = transposed.bitwise_xor(0x88).flatten(-2);
        return {weight_out, scale_out};
    }
    DG_HOST_UNREACHABLE(fmt::format("preprocess mode '{}' with fuse_activation={} is not implemented",
                                    preprocess_mode_name(mode), fuse_activation));
}

/// Deprecated weight-only alias; preserve the original dtype, padding and stride behavior.
inline torch::Tensor preprocess_a8w4_weight(const torch::Tensor& weight) {
    return preprocess_weights_and_scales(PreprocessMode::W4A8_INT8, /*fuse_activation=*/false, weight, torch::Tensor{}).weight;
}

/// Deprecated scale-only alias; preserve the original dtype, padding and stride behavior.
inline torch::Tensor preprocess_mxfp4_scales(const torch::Tensor& scale) {
    return preprocess_weights_and_scales(PreprocessMode::MXFP4, /*fuse_activation=*/false, torch::Tensor{}, scale).scale;
}

/// Deprecated alias kept for forward compatibility with the pre-rename API: it exposed exactly
/// the MXFP4 + fuse_activation interleaving, so forward it instead of duplicating the logic.
inline std::pair<torch::Tensor, torch::Tensor> preprocess_mxfp4_weight_for_act_and_quant_fusing(
        const torch::Tensor& weight, const torch::Tensor& weight_scale) {
    auto result = preprocess_weights_and_scales(PreprocessMode::MXFP4, /*fuse_activation=*/true, weight, weight_scale);
    return {result.weight, result.scale};
}

} // namespace deep_gemm_preprocess_common
