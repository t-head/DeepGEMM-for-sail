#pragma once

#include <torch/python.h>

#include "exception.hpp"
#include "../../deep_gemm/include/deep_gemm/utils_rtc.cuh"

namespace deep_gemm {

template <typename T>
static constexpr T align(const T& a, const T& b) {
    return ceil_div(a, b) * b;
}

int gcd(int a, int b) { while (b != 0) { int temp = b; b = a % b; a = temp; } return a; }

static int get_tma_aligned_size(const int& x, const int& element_size) {
    constexpr int kNumTMAAlignmentBytes = 16;
    DG_HOST_ASSERT(kNumTMAAlignmentBytes % element_size == 0);
    return align(x, kNumTMAAlignmentBytes / element_size);
}

} // namespace deep_gemm
