#pragma once

#include <cmath>
#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#if defined(__HGGC__) || defined(__HGGCCC__) || defined(__HGGCCC_RTC__)
#include <cutlass/epilogue/thread/activation.h>
#else
// The host launcher only instantiates the activation parameter types.
namespace cutlass::epilogue::thread {
template <typename T> struct Sigmoid;
}
#endif

namespace deep_gemm {

enum class GatedActivation { SwiGLU, SiTU, SwiGLUStep, SwiGLUOAI };

template <bool FastMath, typename T>
CUTLASS_DEVICE T gated_sigmoid(T value) {
    if constexpr (FastMath)
        return cutlass::epilogue::thread::Sigmoid<T>{}(value);
    return T(1) / (T(1) + expf(-value));
}

// Non-SiTU epilogues run sigmoid, gate_times_sigmoid and multiply_up in separate fragment-wide passes.
template <GatedActivation Kind, bool ApplyClamp = false, bool kHasLinearBeta = true,
          bool FastMath = true>
struct GatedActivationOp;

template <bool ApplyClamp, bool FastMath>
struct GatedActivationOp<GatedActivation::SwiGLU, ApplyClamp, true, FastMath> {
    static constexpr auto kKind = GatedActivation::SwiGLU;
    // Preserve the existing grouped FP4 epilogue's single-float parameter ABI.
    using Params = float;

    template <typename T>
    CUTLASS_DEVICE static T prepare_gate(T gate, Params limit) {
        if constexpr (ApplyClamp) return min(gate, limit);
        return gate;
    }

    template <typename T>
    CUTLASS_DEVICE static T sigmoid(T gate, Params) {
        return gated_sigmoid<FastMath>(gate);
    }

    template <typename T>
    CUTLASS_DEVICE static T gate_times_sigmoid(T gate, T sigmoid, Params) {
        return cutlass::multiplies<T>{}(gate, sigmoid);
    }

    template <typename T>
    CUTLASS_DEVICE static T multiply_up(T gate, T up, Params limit) {
        if constexpr (ApplyClamp) up = max(min(up, limit), -limit);
        return cutlass::multiplies<T>{}(gate, up);
    }
};

// STEP clips SiLU(gate), whereas SwiGLU clips gate before sigmoid.
template <bool ApplyClamp, bool FastMath>
struct GatedActivationOp<GatedActivation::SwiGLUStep, ApplyClamp, true, FastMath>
    : GatedActivationOp<GatedActivation::SwiGLU, ApplyClamp, true, FastMath> {
    static constexpr auto kKind = GatedActivation::SwiGLUStep;

    template <typename T>
    CUTLASS_DEVICE static T prepare_gate(T gate, float) {
        return gate;
    }

    template <typename T>
    CUTLASS_DEVICE static T gate_times_sigmoid(T gate, T sigmoid, float limit) {
        T value = cutlass::multiplies<T>{}(gate, sigmoid);
        if constexpr (ApplyClamp)
            value = min(value, limit);
        return value;
    }
};

template <bool ApplyClamp, bool kHasLinearBeta, bool FastMath>
struct GatedActivationOp<GatedActivation::SiTU, ApplyClamp, kHasLinearBeta, FastMath> {
    static constexpr auto kKind = GatedActivation::SiTU;
    static_assert(!ApplyClamp, "SiTU does not support activation_clamp");
    struct Params {
        float beta;
        float linear_beta;
    };

    template <typename T>
    CUTLASS_DEVICE static T softcap(T value, float beta) {
        if constexpr (!FastMath)
            return beta * tanhf(value / beta);
#if DG_MEGAMOE_SITU_USE_RCP
        const T scaled = value * __ppu_rcpf(beta);
#else
        const T scaled = value / beta;
#endif
        return beta * __ppu_tanhf(scaled);
    }

    template <typename T>
    CUTLASS_DEVICE static T apply(T gate, T up, const Params& p) {
        const T gated = gated_sigmoid<FastMath>(gate) * softcap(gate, p.beta);
        if constexpr (kHasLinearBeta)
            up = softcap(up, p.linear_beta);
        return gated * up;
    }
};

template <bool ApplyClamp, bool FastMath>
struct GatedActivationOp<GatedActivation::SwiGLUOAI, ApplyClamp, true, FastMath> {
    static constexpr auto kKind = GatedActivation::SwiGLUOAI;
    struct Params {
        float limit;
        float alpha;
    };

    template <typename T>
    CUTLASS_DEVICE static T prepare_gate(T gate, const Params& p) {
        if constexpr (ApplyClamp) return min(gate, p.limit);
        return gate;
    }

    template <typename T>
    CUTLASS_DEVICE static T sigmoid(T gate, const Params& p) {
        return gated_sigmoid<FastMath>(gate * p.alpha);
    }

    template <typename T>
    CUTLASS_DEVICE static T gate_times_sigmoid(T gate, T sigmoid, const Params&) {
        return cutlass::multiplies<T>{}(gate, sigmoid);
    }

    template <typename T>
    CUTLASS_DEVICE static T multiply_up(T value, T up, const Params& p) {
        if constexpr (ApplyClamp) up = max(min(up, p.limit), -p.limit);
        return value * (up + T(1));
    }
};

}  // namespace deep_gemm
