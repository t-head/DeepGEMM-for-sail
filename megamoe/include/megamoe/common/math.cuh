#pragma once

#include <megamoe/common/exception.cuh>
#include <cuda/std/cstdint>

namespace megamoe::math {

template <typename dtype_t = void>
DG_HOST_DEVICE dtype_t* advance_ptr(void* ptr, const uint64_t num_bytes) {
    return reinterpret_cast<dtype_t*>(static_cast<uint8_t*>(ptr) + num_bytes);
}

template <typename T>
DG_HOST_DEVICE T ceil_div(T numerator, T denominator) {
    return (numerator + denominator - 1) / denominator;
}

template <typename T>
DG_HOST_DEVICE constexpr T constexpr_ceil_div(T numerator, T denominator) {
    return (numerator + denominator - 1) / denominator;
}

template <typename T>
DG_HOST_DEVICE T align_up(T value, T alignment) {
    return ceil_div(value, alignment) * alignment;
}

template <typename T>
DG_HOST_DEVICE T align_down(T value, T alignment) {
    return (value / alignment) * alignment;
}

template <typename T>
DG_HOST_DEVICE constexpr T constexpr_align_up(T value, T alignment) {
    return constexpr_ceil_div(value, alignment) * alignment;
}

}  // namespace megamoe::math
