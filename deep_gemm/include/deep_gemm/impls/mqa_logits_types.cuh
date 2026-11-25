#pragma once

#include <cstdint>

// MQA types shared by host configuration, dump/replay and device kernels.
namespace deep_gemm {

// ============================================================
// Scale mode: selects what the kernel's `weights` argument slot carries
// and how it is applied in the epilogue
// ============================================================

// kScaleModeWeights: per-(q, head) weights, logits = sum_h(relu(qk) * w) * k_scale
// kScaleModeQRow:    avg variant, per-Q-row float scale in the weights slot,
//                    logits = sum_h(relu(qk)) * scale_q * k_scale / sqrt(head_dim)
// kScaleModeQScalar: avg variant, broadcast scalar float scale in the weights slot
//                    (same formula, scale_q shared by every Q row)
// kScaleModeUnity:   avg variant without any Q-row scale (scale_q = 1)
enum QScaleMode : uint32_t {
    kScaleModeWeights = 0,
    kScaleModeQRow = 1,
    kScaleModeQScalar = 2,
    kScaleModeUnity = 3,
    // INT8 AVG: FP32 quantization scales per (Q, head), in the weights slot.
    kScaleModeQHead = 4,
};

struct alignas(8) FP4VarlenCTAStart {
    uint32_t source_q, kv_split;
};
struct alignas(16) FP4VarlenQBlock {
    uint32_t source_q, valid_q, context_len, table_row;
};
static_assert(sizeof(FP4VarlenCTAStart) == 8 && sizeof(FP4VarlenQBlock) == 16);

} // namespace deep_gemm
