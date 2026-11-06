#pragma once

#include <cstdint>
#include <type_traits>
#include <deep_gemm/tix/cvt.cuh>

#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <deep_gemm/impls/int8_mega_moe_activation.cuh>
#include <deep_gemm/epilogue/epilogue_traits.cuh>

namespace deep_gemm::epilogue {
enum class Int8GatedFragment { GateUpFloat, GateUpPackedBf16, ActivatedFloat };

// INT8 keeps its per-token FP32 scale contract. Gated activation is performed here;
// the subsequent row reduction/INT8 conversion consumes the compact BF16 result.
template <class StrideC, class StrideD, class ThreadOp, bool Normal, bool Clamp, int GateGroup,
          uint32_t LimitBits = 0xffffffffu, Int8GatedFragment Fragment = Int8GatedFragment::GateUpFloat,
          GatedActivation Kind = GatedActivation::SwiGLU, uint32_t BetaBits = 0,
          uint32_t LinearBetaBits = 0, uint32_t AlphaBits = 0>
class Int8GatedEpilogue : public cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        StrideC, StrideD, ThreadOp, cutlass::gemm::EpilogueDefault, true> {
    using Base = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        StrideC, StrideD, ThreadOp, cutlass::gemm::EpilogueDefault, true>;
public:
    using typename Base::SharedStorage;
    using typename Base::ElementD;
    static constexpr int kOutputColumnDivisor = 2;
    static constexpr bool kHasPairTransform = Fragment != Int8GatedFragment::GateUpFloat;
    using AccumulatorStorage = std::conditional_t<Fragment == Int8GatedFragment::GateUpPackedBf16, uint32_t, float>;
    CUTLASS_DEVICE static float resolved_limit(float runtime) {
        if constexpr (LimitBits == 0xffffffffu) return runtime;
        else return __uint_as_float(LimitBits);
    }
    CUTLASS_DEVICE static float activate(float gate, float up, float runtime) {
        if constexpr (Kind == GatedActivation::SwiGLU)
            return int8_swiglu_value<Normal, Clamp, true>(gate, up, resolved_limit(runtime));
        else
            return int8_gated_value<Kind, Clamp, Kind != GatedActivation::SiTU || LinearBetaBits != 0>(
                gate, up, resolved_limit(runtime), __uint_as_float(BetaBits),
                __uint_as_float(LinearBetaBits), __uint_as_float(AlphaBits));
    }
    struct PairTransform {
        float limit;
        static constexpr bool kCopyScalesLate = Fragment == Int8GatedFragment::ActivatedFloat;
        CUTLASS_DEVICE auto operator()(float gate, float up) const {
            if constexpr (Fragment == Int8GatedFragment::GateUpPackedBf16)
                return deep_gemm::tix::cvt_rte_bf16x2_f32(up, gate);
            else return Int8GatedEpilogue::activate(gate, up, limit);
        }
    };
    CUTLASS_DEVICE static PairTransform make_pair_transform(float limit) { return {limit}; }

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
                if constexpr (Fragment == Int8GatedFragment::ActivatedFloat) {
                    value = accumulators(i);
                } else if constexpr (Fragment == Int8GatedFragment::GateUpPackedBf16) {
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
