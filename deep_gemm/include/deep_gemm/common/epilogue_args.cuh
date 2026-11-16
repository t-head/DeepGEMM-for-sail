#pragma once

#include <type_traits>
#include <deep_gemm/common/types.cuh>
#include <cute/int_tuple.hpp>
#include <cutlass/numeric_types.h>

namespace deep_gemm {

template <class Epilogue> struct UsesOutputCounter : std::false_type {};

// Blockwise output uses packed UE8M0 SFD; channelwise output uses FP32 row scales.
enum class RowQuantPhase { Fused, Reduce, Store };

template <class ThreadParams>
struct PPUStoreCDArguments {
    ThreadParams callback{};
    const cutlass::bfloat16_t* ptr_C = nullptr;
    cute::Stride<int64_t, cute::Int<1>, int64_t> dC{};
    cutlass::float_e4m3_t* ptr_D = nullptr;
    cute::Stride<int64_t, cute::Int<1>, int64_t> dD{};
    EpilogueOperatorArgs epilogue_args{};
    int batch_idx = 0;
    RowQuantPhase phase = RowQuantPhase::Fused;
};

} // namespace deep_gemm
