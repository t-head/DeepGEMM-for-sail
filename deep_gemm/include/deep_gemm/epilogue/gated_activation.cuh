#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <hggc_bf16.h>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm {

enum class GatedActivation { SwiGLU, SiTU, SwiGLUStep, SwiGLUOAI };

template <class Op, typename T>
CUTLASS_DEVICE T compose_gated_activation(T gate, T up, const typename Op::Params& params) {
    gate = Op::prepare_gate(gate, params);
    const T sigmoid = Op::sigmoid(gate, params);
    return Op::multiply_up(Op::gate_times_sigmoid(gate, sigmoid, params), up, params);
}

// Non-SiTU epilogues run sigmoid, gate_times_sigmoid and multiply_up in separate fragment-wide passes.
template <GatedActivation Kind, bool ApplyClamp = false, bool kHasLinearBeta = true, bool FastMath = false>
struct GatedActivationOp;

template <bool ApplyClamp, bool FastMath>
struct GatedActivationOp<GatedActivation::SwiGLU, ApplyClamp, true, FastMath> {
    static constexpr auto kKind = GatedActivation::SwiGLU;
    // Preserve the existing grouped FP4 epilogue's single-float parameter ABI.
    using Params = float;

    template <typename T>
    CUTLASS_DEVICE static T apply(T gate, T up, Params params) {
        return compose_gated_activation<GatedActivationOp>(gate, up, params);
    }

    template <typename T>
    CUTLASS_DEVICE static T prepare_gate(T gate, Params limit) {
        if constexpr (ApplyClamp) return min(gate, limit);
        return gate;
    }

    template <typename T>
    CUTLASS_DEVICE static T sigmoid(T gate, Params) {
        return __ppu_sgmdf(gate);
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
    CUTLASS_DEVICE static T apply(T gate, T up, float params) {
        return compose_gated_activation<GatedActivationOp>(gate, up, params);
    }

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
        return beta * __ppu_tanhf(math::fp32_divide<FastMath>(value, T(beta)));
    }

    template <typename T>
    CUTLASS_DEVICE static T apply(T gate, T up, const Params& p) {
        return compose_gated_activation<GatedActivationOp>(gate, up, p);
    }

    template <typename T>
    CUTLASS_DEVICE static T prepare_gate(T gate, const Params&) { return gate; }

    template <typename T>
    CUTLASS_DEVICE static T sigmoid(T gate, const Params&) { return __ppu_sgmdf(gate); }

    template <typename T>
    CUTLASS_DEVICE static T gate_times_sigmoid(T gate, T sigmoid, const Params& p) {
        return sigmoid * softcap(gate, p.beta);
    }

    template <typename T>
    CUTLASS_DEVICE static T multiply_up(T gated, T up, const Params& p) {
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
    CUTLASS_DEVICE static T apply(T gate, T up, const Params& p) {
        return compose_gated_activation<GatedActivationOp>(gate, up, p);
    }

    template <typename T>
    CUTLASS_DEVICE static T prepare_gate(T gate, const Params& p) {
        if constexpr (ApplyClamp) return min(gate, p.limit);
        return gate;
    }

    template <typename T>
    CUTLASS_DEVICE static T sigmoid(T gate, const Params& p) {
        return __ppu_sgmdf(gate * p.alpha);
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

// Routed backends share formulas but retain their BF16 clamp-boundary contract.
// FP4's fragment-wide epilogue uses GatedActivationOp directly with FP32 limits.
template <GatedActivation Kind, bool FastMath>
CUTLASS_DEVICE float routed_gated_activation(float gate, float up, float limit, float alpha,
                                            float beta, float linear_beta) {
    if constexpr (Kind == GatedActivation::SiTU) {
        if (linear_beta > 0)
            return GatedActivationOp<Kind, false, true, FastMath>::apply(gate, up, {beta, linear_beta});
        return GatedActivationOp<Kind, false, false, FastMath>::apply(gate, up, {beta, linear_beta});
    } else {
        const float bound = __bfloat162float(__float2bfloat16(limit));
        if constexpr (Kind == GatedActivation::SwiGLUOAI) {
            if (limit >= 0) return GatedActivationOp<Kind, true, true, FastMath>::apply(gate, up, {bound, alpha});
            return GatedActivationOp<Kind, false, true, FastMath>::apply(gate, up, {bound, alpha});
        } else {
            if (limit >= 0) return GatedActivationOp<Kind, true, true, FastMath>::apply(gate, up, bound);
            return GatedActivationOp<Kind, false, true, FastMath>::apply(gate, up, bound);
        }
    }
}

}  // namespace deep_gemm
