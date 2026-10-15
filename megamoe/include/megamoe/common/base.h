#pragma once

// Base definitions shared by every kernel header: execution-space macros
// and the small scalar/tuple helpers the kernels use.

#include <cuda_fp8.h>

#define DG_DEVICE __device__ __forceinline__
#define DG_HOST_DEVICE __host__ __device__ __forceinline__
#define DG_GLOBAL __global__

namespace megamoe {

template <typename T>
DG_HOST_DEVICE constexpr T min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
DG_HOST_DEVICE constexpr T max(T a, T b) {
    return a < b ? b : a;
}

// 4-element tuple for the scheduler's get_next_block() return type.
template <typename T0, typename T1, typename T2, typename T3>
struct tuple {
    T0 _0;
    T1 _1;
    T2 _2;
    T3 _3;
};

template <size_t I, typename T0, typename T1, typename T2, typename T3>
DG_HOST_DEVICE constexpr auto& get(tuple<T0, T1, T2, T3>& value) {
    if constexpr (I == 0)
        return value._0;
    else if constexpr (I == 1)
        return value._1;
    else if constexpr (I == 2)
        return value._2;
    else if constexpr (I == 3)
        return value._3;
}

template <size_t I, typename T0, typename T1, typename T2, typename T3>
DG_HOST_DEVICE constexpr const auto& get(tuple<T0, T1, T2, T3> const& value) {
    if constexpr (I == 0)
        return value._0;
    else if constexpr (I == 1)
        return value._1;
    else if constexpr (I == 2)
        return value._2;
    else if constexpr (I == 3)
        return value._3;
}

}  // namespace megamoe
