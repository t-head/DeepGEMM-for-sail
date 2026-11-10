#pragma once

#include "../jit_kernels/impls/fp8_blockwise_gemm_comm.hpp"
#include "../jit_kernels/impls/fp8_channelwise_gemm_comm.hpp"

namespace deep_gemm::gemm_comm {
inline void register_apis(pybind11::module_& m) {
    register_fp8_blockwise_gemm_comm_apis(m);
    register_fp8_channelwise_gemm_comm_apis(m);
}
} // namespace deep_gemm::gemm_comm
