#pragma once

#include <cstdint>
#include <cute/int_tuple.hpp>

namespace deep_gemm {

// BatchGemm alone extends the shared mainloop arguments with channel-scale strides.
template <class MainloopArguments>
struct BatchGemmMainloopArguments : public MainloopArguments {
    cute::Stride<int64_t, int64_t> stride_scale_A{1, 0};
    cute::Stride<int64_t, int64_t> stride_scale_B{1, 0};
};

} // namespace deep_gemm
