#pragma once

#include <cstdint>
#include <type_traits>
#include <deep_gemm/tix/cvt.cuh>

#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <deep_gemm/epilogue/gated_activation.cuh>
#include <deep_gemm/epilogue/epilogue_traits.cuh>

namespace deep_gemm::epilogue {
template <bool ApplyClamp, bool FastSigmoid = false>
__device__ __forceinline__ float int8_swiglu_value(float gate, float up, float limit) {
    using Activation = GatedActivationOp<GatedActivation::SwiGLU, ApplyClamp>;
    // Preserve Legacy's BF16 GEMM output boundary even inside the epilogue.
    gate = __bfloat162float(__float2bfloat16(gate));
    up = __bfloat162float(__float2bfloat16(up));
    gate = Activation::prepare_gate(gate, limit);
    float activated;
    if constexpr (FastSigmoid) {
        // Reuse FP4's Sigmoid primitive, which maps to native PPU sigmoid.
        // The precise policy below retains expf and division.
        activated = Activation::gate_times_sigmoid(
            gate, Activation::sigmoid(gate, limit), limit);
    } else {
        activated = gate / (1.0f + expf(-gate));
    }
    activated = __bfloat162float(__float2bfloat16(activated));
    return __bfloat162float(__float2bfloat16(Activation::multiply_up(activated, up, limit)));
}

// The additional public activations keep FP4's formula and the BF16 GEMM/product boundaries.
template <GatedActivation Kind, bool ApplyClamp, bool HasLinearBeta, bool FastMath = true>
__device__ __forceinline__ float int8_gated_value(
        float gate, float up, float limit, float beta, float linear_beta, float alpha) {
    using Activation = GatedActivationOp<Kind, ApplyClamp, HasLinearBeta, FastMath>;
    gate = __bfloat162float(__float2bfloat16(gate));
    up = __bfloat162float(__float2bfloat16(up));
    float value;
    if constexpr (Kind == GatedActivation::SiTU) {
        value = Activation::apply(gate, up, {beta, linear_beta});
    } else {
        const typename Activation::Params params = [&] {
            if constexpr (Kind == GatedActivation::SwiGLUOAI)
                return typename Activation::Params{limit, alpha};
            else return limit;
        }();
        gate = Activation::prepare_gate(gate, params);
        const float activated = Activation::gate_times_sigmoid(
            gate, Activation::sigmoid(gate, params), params);
        value = Activation::multiply_up(activated, up, params);
    }
    return __bfloat162float(__float2bfloat16(value));
}

enum class Int8GatedFragment { GateUpFloat, GateUpPackedBf16 };

// INT8 keeps its per-token FP32 scale contract. Gated activation is performed here;
// the subsequent row reduction/INT8 conversion consumes the compact BF16 result.
template <class StrideC, class StrideD, class ThreadOp, bool Clamp, int GateGroup,
          uint32_t LimitBits = 0xffffffffu, Int8GatedFragment Fragment = Int8GatedFragment::GateUpFloat,
          GatedActivation Kind = GatedActivation::SwiGLU, uint32_t BetaBits = 0,
          uint32_t LinearBetaBits = 0, uint32_t AlphaBits = 0, bool FastMath = true>
class Int8GatedEpilogue : public cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        StrideC, StrideD, ThreadOp, cutlass::gemm::EpilogueDefault, true> {
    using Base = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        StrideC, StrideD, ThreadOp, cutlass::gemm::EpilogueDefault, true>;
public:
    using typename Base::SharedStorage;
    using typename Base::ElementD;
    static constexpr bool kHasPairTransform = Fragment != Int8GatedFragment::GateUpFloat;
    using AccumulatorStorage = std::conditional_t<Fragment == Int8GatedFragment::GateUpPackedBf16, uint32_t, float>;
    CUTLASS_DEVICE static float resolved_limit(float runtime) {
        if constexpr (LimitBits == 0xffffffffu) return runtime;
        else return __uint_as_float(LimitBits);
    }
    CUTLASS_DEVICE static float activate(float gate, float up, float runtime) {
        if constexpr (Kind == GatedActivation::SwiGLU)
            return int8_swiglu_value<Clamp, FastMath>(gate, up, resolved_limit(runtime));
        else
            return int8_gated_value<Kind, Clamp, Kind != GatedActivation::SiTU || LinearBetaBits != 0, FastMath>(
                gate, up, resolved_limit(runtime), __uint_as_float(BetaBits),
                __uint_as_float(LinearBetaBits), __uint_as_float(AlphaBits));
    }
    struct PairTransform {
        CUTLASS_DEVICE uint32_t operator()(float gate, float up) const {
            return deep_gemm::tix::cvt_rte_bf16x2_f32(up, gate);
        }
    };

    struct Arguments : Base::Arguments { float activation_limit = 0.0f; };
    using Params = Arguments;

    template <class Shape>
    static constexpr Params to_underlying_arguments(const Shape&, const Arguments& args, void*) {
        return args;
    }

    CUTLASS_HOST_DEVICE Int8GatedEpilogue(const Params& args,
            const SharedStorage& storage = SharedStorage()) : Base(args, storage), params(args) {}

    template <class Problem, class Block, class Coord, class Engine, class Layout,
              class Mma, class Residue>
    CUTLASS_DEVICE void operator()(Problem shape, Block block, Coord tile,
            const cute::Tensor<Engine, Layout>& accumulators, Mma mma, Residue residue,
            int thread, char*) {
        using namespace cute;
        auto identity = make_identity_tensor(take<0, 2>(block));
        auto coordinates = mma.get_thread_slice(thread).partition_C(identity);
        CUTE_STATIC_ASSERT_V(size(accumulators) % Int<2>{} == Int<0>{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accumulators); i += 2) {
            auto coordinate = coordinates(i);
            if (elem_less(coordinate, make_coord(get<0>(residue), get<1>(residue)))) {
                const int row = get<0>(tile) * get<0>(block) + get<0>(coordinate);
                const int physical_column = get<1>(tile) * get<1>(block) + get<1>(coordinate);
                const int column = physical_column / (GateGroup * 2) * GateGroup + physical_column % GateGroup;
                float value;
                if constexpr (Fragment == Int8GatedFragment::GateUpPackedBf16) {
                    const uint32_t pair = accumulators(i);
                    value = activate(__uint_as_float(pair << 16), __uint_as_float(pair & 0xffff0000u), params.activation_limit);
                } else {
                    value = activate(accumulators(i), accumulators(i + 1), params.activation_limit);
                }
                params.ptr_D[row * get<0>(params.dD) + column] = ElementD(value);
            }
        }
    }
private:
    Params params;
};

} // namespace deep_gemm::epilogue
