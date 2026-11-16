#pragma once

#include "../jit_kernels/impls/bf16_gemm_comm.hpp"
#include "../jit_kernels/impls/fp8_blockwise_gemm_comm.hpp"
#include "../jit_kernels/impls/fp8_channelwise_gemm_comm.hpp"

namespace deep_gemm::gemm_comm {
inline void register_apis(pybind11::module_& m) {
    m.def("_gemm_comm_bf16_gemm_rs_config", &get_bf16_gemm_rs_config, pybind11::arg("m"), pybind11::arg("n"),
          pybind11::arg("k"), pybind11::arg("world"), pybind11::arg("configs") = std::nullopt,
          pybind11::arg("tile_policy") = "rs_aware");
    m.def("_gemm_comm_bf16_gemm_rs", &launch_bf16_gemm_rs, pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("out"),
          pybind11::arg("state"), pybind11::arg("m"), pybind11::arg("rank"), pybind11::arg("world"),
          pybind11::arg("config"), pybind11::arg("launch") = true);
    m.def("_gemm_comm_fp8_blockwise_gemm_rs_config", &get_blockwise_gemm_rs_config, pybind11::arg("m"),
          pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("world"), pybind11::arg("configs") = pybind11::none(),
          pybind11::arg("tile_policy") = "rs_aware");
    m.def("_gemm_comm_fp8_blockwise_gemm_rs", &launch_blockwise_gemm_rs, pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("out"), pybind11::arg("state"), pybind11::arg("m"), pybind11::arg("rank"),
          pybind11::arg("world"), pybind11::arg("config"), pybind11::arg("launch") = true);
    m.def("_gemm_comm_fp8_channelwise_gemm_rs_config", &get_channelwise_gemm_rs_config, pybind11::arg("m"),
          pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("world"), pybind11::arg("configs") = pybind11::none(),
          pybind11::arg("tile_policy") = "rs_aware");
    m.def("_gemm_comm_fp8_channelwise_gemm_rs", &launch_channelwise_gemm_rs, pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("out"), pybind11::arg("state"), pybind11::arg("m"), pybind11::arg("rank"),
          pybind11::arg("world"), pybind11::arg("config"), pybind11::arg("launch") = true);
}
} // namespace deep_gemm::gemm_comm
