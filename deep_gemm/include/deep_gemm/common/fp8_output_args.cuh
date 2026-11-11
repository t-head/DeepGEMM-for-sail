#pragma once

#include <cstdint>
#include <cute/int_tuple.hpp>
#include <cutlass/numeric_types.h>

namespace deep_gemm {

// Matches the unclamped upstream (FP8 D, packed UE8M0 SFD) epilogue contract.
template <class ThreadParams>
struct FP8OutputArguments {
    ThreadParams callback{};
    const cutlass::bfloat16_t* ptr_C = nullptr;
    cute::Stride<int64_t, cute::Int<1>, int64_t> dC{};
    cutlass::float_e4m3_t* ptr_D = nullptr;
    cute::Stride<int64_t, cute::Int<1>, int64_t> dD{};
    uint32_t* sfd = nullptr;
    int64_t sfd_stride = 0;
    int batch_idx = 0;
};

} // namespace deep_gemm
