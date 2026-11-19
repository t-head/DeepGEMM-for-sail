"""Native PPU host-codegen regression; requires the source checkout and PPU SDK.

Run on an idle PPU: python tests/test_jit_compile_keys.py -v
The helper compiles host C++ only. It never compiles or launches generated GEMMs.
"""
import os
from pathlib import Path
import shutil
import unittest

import torch
from torch.utils.cpp_extension import load_inline


# Keep the native checks with their Python runner; no production test bindings.
CPP_SOURCE = r'''
// Exercise the actual host code generators, without compiling or launching GEMMs.
#include "jit_kernels/impls/bf16_gemm.hpp"
#include "jit_kernels/impls/fp4_gemm.hpp"
#include "jit_kernels/impls/fp8_gemm.hpp"

#include <stdexcept>

namespace {
using namespace deep_gemm;

template <typename Runtime, typename Change>
int check_change(const typename Runtime::Args& base, Change change,
                 const char* field, bool changes_source = true) {
    auto other = base;
    change(other);
    const bool same_source = Runtime::generate_impl(base) == Runtime::generate_impl(other);
    const bool same_key = Runtime::compile_key(base) == Runtime::compile_key(other);
    if (same_source == changes_source || same_key == changes_source)
        throw std::runtime_error(std::string(field) + ": expected " +
                                 (changes_source ? "distinct source and keys" : "the same source and key"));
    return 1;
}

template <typename Runtime>
typename Runtime::Args dense_args() {
    typename Runtime::Args args{};
    auto& info = args.launch_info;
    info.block_m = 128; info.block_n = 128; info.block_k = 64;
    info.warp_m = 32; info.warp_n = 64; info.num_stages = 4;
    info.gemm_type = "Normal"; info.kernel_name = "test_gemm";
    args.kernel_params.problem_shape = cute::make_tuple(64, 512, 512, 1);
    return args;
}

// Shared fields still exercise each runtime's own compile_key and generate_impl.
template <typename Runtime>
int tile_checks(const typename Runtime::Args& args) {
    int count = 0;
#define CHECK_TILE(field, value) \
    count += check_change<Runtime>(args, [](auto& a) { a.launch_info.field = value; }, #field)
    CHECK_TILE(block_m, 64);
    CHECK_TILE(block_n, 256);
    CHECK_TILE(block_k, 128);
    CHECK_TILE(warp_m, 64);
    CHECK_TILE(warp_n, 32);
    CHECK_TILE(num_stages, 3);
    CHECK_TILE(kernel_name, "another_entry");
#undef CHECK_TILE
    return count;
}

template <typename Runtime>
int dynamic_checks(const typename Runtime::Args& args) {
    int count = 0;
    count += check_change<Runtime>(args, [](auto& a) {
        a.launch_args.grid_dim = dim3(19, 2, 1);
        a.launch_args.block_dim = dim3(256, 1, 1);
    }, "launch dimensions", false);
    count += check_change<Runtime>(args, [](auto& a) {
        cute::get<0>(a.kernel_params.problem_shape) = 96;
        a.kernel_params.hw_info.cu_count = 19;
        auto& ptr_a = a.kernel_params.collective_mainloop_params.ptr_A;
        ptr_a = reinterpret_cast<std::remove_reference_t<decltype(ptr_a)>>(uintptr_t(0x1000));
        auto& ptr_d = a.kernel_params.collective_epilogue_params.ptr_D;
        ptr_d = reinterpret_cast<std::remove_reference_t<decltype(ptr_d)>>(uintptr_t(0x2000));
        a.kernel_params.workspace = reinterpret_cast<void*>(uintptr_t(0x1000));
    }, "runtime M/CU/workspace", false);
    return count;
}

template <typename Runtime>
int bf16_checks() {
    auto args = dense_args<Runtime>();
    args.launch_info.warp_k = 64;
    int count = tile_checks<Runtime>(args) + dynamic_checks<Runtime>(args);
#define CHECK_BF16(field, value) \
    count += check_change<Runtime>(args, [](auto& a) { a.launch_info.field = value; }, #field)
    CHECK_BF16(warp_k, 32);
    CHECK_BF16(dense_s2_opt, true);
    CHECK_BF16(overlap_prologue, true);
    CHECK_BF16(b_cache_policy, 3);
    CHECK_BF16(nonpersistent, true);
    CHECK_BF16(gemm_type, "MGroupedContiguous");
#undef CHECK_BF16
    count += check_change<Runtime>(args, [](auto& a) {
        cute::get<1>(a.kernel_params.problem_shape) = 514;
    }, "N alignment");
    count += check_change<Runtime>(args, [](auto& a) {
        cute::get<1>(a.kernel_params.problem_shape) = 768;
        cute::get<2>(a.kernel_params.problem_shape) = 1024;
    }, "aligned N and runtime K", false);
    auto unaligned_n = args;
    cute::get<1>(unaligned_n.kernel_params.problem_shape) = 514;
    count += check_change<Runtime>(unaligned_n, [](auto& a) {
        cute::get<1>(a.kernel_params.problem_shape) = 515;
    }, "odd N changes packed-store eligibility");
    count += check_change<Runtime>(args, [](auto& a) {
        auto& ptr_d = a.kernel_params.collective_epilogue_params.ptr_D;
        using Element = std::remove_pointer_t<std::remove_reference_t<decltype(ptr_d)>>;
        ptr_d = reinterpret_cast<std::remove_reference_t<decltype(ptr_d)>>(uintptr_t(sizeof(Element)));
    }, "output pointer alignment changes packed-store eligibility");
    return count;
}

template <typename Output>
int bf16_tsm_store_checks() {
    using Runtime = BF16GemmCutlass3RuntimeT<Output>;
    typename Runtime::Args args{};
    args.launch_info = {64, 64, 32, 32, 32, 1, 2, "Normal", "Default", "test_tsm", false};
    constexpr int alignment = 16 / sizeof(Output);
    const bool with_tsm = !is_ppu1v5_device();
    auto params = typename Runtime::template GemmKernelParamsT<typename Runtime::EpilogueParamWithTsm>{};
    params.problem_shape = cute::make_tuple(3, 64, 32, 1);
    params.collective_epilogue_params.ptr_D = reinterpret_cast<Output*>(uintptr_t(0x1000));
    params.collective_epilogue_params.stride_D = {64, cute::Int<1>{}, 256};
    if (with_tsm) args.kernel_params.with_tsm = params;
    else {
        args.kernel_params.no_tsm = {};
        args.kernel_params.no_tsm.problem_shape = params.problem_shape;
        args.kernel_params.no_tsm.collective_epilogue_params.ptr_D = params.collective_epilogue_params.ptr_D;
        args.kernel_params.no_tsm.collective_epilogue_params.stride_D = params.collective_epilogue_params.stride_D;
    }
    const auto base_source = Runtime::generate_impl(args);
    if (Runtime::requires_safe_store(args) || base_source.find("kEnableSafeStore = false;") == std::string::npos)
        throw std::runtime_error("aligned outputs must preserve unchecked stores");
    int count = 1;
    auto check_policy = [&](const auto& other, bool expected, bool changes_source) {
        const auto source = Runtime::generate_impl(other);
        if (Runtime::requires_safe_store(other) != expected ||
            source.find(std::string("kEnableSafeStore = ") + (expected ? "true;" : "false;")) == std::string::npos ||
            (source != base_source) != changes_source ||
            (get_hex_digest(source) != get_hex_digest(base_source)) != changes_source)
            throw std::runtime_error("TSM store policy/source-cache identity mismatch");
        ++count;
    };
    for (int offset = 0; offset < alignment; ++offset) {
        auto other = args;
        auto ptr = reinterpret_cast<Output*>(uintptr_t(0x2000 + offset * sizeof(Output)));
        if (with_tsm) other.kernel_params.with_tsm.collective_epilogue_params.ptr_D = ptr;
        else other.kernel_params.no_tsm.collective_epilogue_params.ptr_D = ptr;
        check_policy(other, with_tsm && offset != 0, with_tsm && offset != 0);
    }
    if (!with_tsm) return count;
    auto row_stride = args;
    cute::get<0>(row_stride.kernel_params.with_tsm.collective_epilogue_params.stride_D) = 65;
    check_policy(row_stride, true, true);
    auto tail = args;
    cute::get<1>(tail.kernel_params.with_tsm.problem_shape) = 62;
    cute::get<0>(tail.kernel_params.with_tsm.collective_epilogue_params.stride_D) = 64;
    check_policy(tail, true, true);
    auto unused_batch_stride = args;
    cute::get<2>(unused_batch_stride.kernel_params.with_tsm.collective_epilogue_params.stride_D) = 257;
    check_policy(unused_batch_stride, false, false);
    auto batched = unused_batch_stride;
    cute::get<3>(batched.kernel_params.with_tsm.problem_shape) = 3;
    check_policy(batched, true, true);
    if (Runtime::generate_impl(args, true).find("kEnableSafeStore = true;") == std::string::npos)
        throw std::runtime_error("explicit upper-level enable must be forwarded");
    typename Model810eTsmRuntime<Output, false>::Args model{};
    static_cast<typename Runtime::Args&>(model) = tail;
    if (Model810eTsmRuntime<Output, false>::generate_impl(model).find("kEnableSafeStore = true;") == std::string::npos)
        throw std::runtime_error("dummy scheduler N must not hide the real output tail");
    return count + 2;
}

int int8_checks() {
    using Runtime = DenseINT8GemmCutlass3Runtime;
    auto args = dense_args<Runtime>();
    args.type_info = "int8_t"; args.launch_info.warp_k = 64;
    int count = tile_checks<Runtime>(args) + dynamic_checks<Runtime>(args);
    count += check_change<Runtime>(args, [](auto& a) {
        a.type_info = "cutlass::float_e4m3_t";
    }, "INT8/FP8 channel element type");
    count += check_change<Runtime>(args, [](auto& a) { a.launch_info.warp_k = 32; }, "warp_k");
    count += check_change<Runtime>(args, [](auto& a) { a.launch_info.dense_s2_opt = true; }, "dense_s2_opt");
    count += check_change<Runtime>(args, [](auto& a) {
        cute::get<1>(a.kernel_params.problem_shape) = 514;
    }, "N alignment");
    count += check_change<Runtime>(args, [](auto& a) {
        cute::get<1>(a.kernel_params.problem_shape) = 768;
        cute::get<2>(a.kernel_params.problem_shape) = 1024;
    }, "aligned N and runtime K", false);
    return count;
}

int fp8_checks() {
    using Runtime = FP8GemmRuntime;
    auto args = dense_args<Runtime>();
    args.launch_info.num_groups = 1;
    args.launch_info.kernel_strategy = "Default";
    int count = tile_checks<Runtime>(args) + dynamic_checks<Runtime>(args);
#define CHECK_FP8(field, value) \
    count += check_change<Runtime>(args, [](auto& a) { a.launch_info.field = value; }, #field)
    CHECK_FP8(num_groups, 2);
    CHECK_FP8(gemm_type, "MGroupedContiguous");
    CHECK_FP8(kernel_strategy, "OtherStrategy");
    CHECK_FP8(enable_sbo_overlap, true);
#undef CHECK_FP8
    count += check_change<Runtime>(args, [](auto& a) {
        cute::get<1>(a.kernel_params.problem_shape) = 768;
    }, "literal N");
    count += check_change<Runtime>(args, [](auto& a) {
        cute::get<2>(a.kernel_params.problem_shape) = 1024;
    }, "literal K");
    return count;
}

int fp4_checks() {
    using Runtime = FP4GemmRuntime;
    Runtime::Args args{};
    auto& info = args.launch_info;
    info.block_m = 128; info.block_n = 128; info.block_k = 64;
    info.warp_m = 32; info.warp_n = 64; info.num_stages = 4;
    info.num_groups = 1; info.n = 512; info.k = 512; info.n_expand = 1;
    info.gemm_type = "Normal"; info.kernel_name = "test_fp4";
    args.launch_args.smem_size = 32768;
    int count = tile_checks<Runtime>(args);
#define CHECK_FP4(field, value) \
    count += check_change<Runtime>(args, [](auto& a) { a.launch_info.field = value; }, #field)
    CHECK_FP4(n, 514);
    CHECK_FP4(k, 1024);
    CHECK_FP4(num_groups, 2);
    CHECK_FP4(gemm_type, "MGroupedContiguous");
    CHECK_FP4(enable_sbo_overlap, true);
    CHECK_FP4(hasBias, true);
    CHECK_FP4(n_expand, 2);
    CHECK_FP4(epilogue_type, "SiluAndMulPostQuantFp4");
    CHECK_FP4(apply_swiglu_limit, true);
#undef CHECK_FP4
    count += check_change<Runtime>(args, [](auto& a) { a.launch_args.smem_size += 1024; }, "smem static_assert");
    count += check_change<Runtime>(args, [](auto& a) {
        a.launch_args.grid_dim = dim3(19, 1, 1);
        a.launch_args.block_dim = dim3(256, 1, 1);
    }, "launch dimensions", false);
    return count;
}

int gemv_checks() {
    using Runtime = DenseGemvRuntime;
    int count = 0;
    for (int m : {1, 2}) {
        Runtime::Args args{};
        args.launch_info = {32, 4, 8, m, 1, "test_gemv", false};
#define CHECK_GEMV(field, value) \
        count += check_change<Runtime>(args, [](auto& a) { a.launch_info.field = value; }, #field)
        CHECK_GEMV(block_x, 64);
        CHECK_GEMV(block_y, 8);
        CHECK_GEMV(k_per_thread, 16);
        CHECK_GEMV(kernel_name, "another_entry");
        CHECK_GEMV(output_fp32, true);
        if (m == 2) { CHECK_GEMV(npt, 2); }
#undef CHECK_GEMV
        count += check_change<Runtime>(args, [m](auto& a) { a.launch_info.m = 3 - m; }, "M1/M2 entry");
        count += check_change<Runtime>(args, [](auto& a) {
            a.kernel_params.N = 514; a.kernel_params.K = 1024;
            a.kernel_params.stride_wn = 2048;
            a.kernel_params.x_ptr = reinterpret_cast<void*>(uintptr_t(0x1000));
            a.kernel_params.w_ptr = reinterpret_cast<void*>(uintptr_t(0x2000));
            a.kernel_params.y_ptr = reinterpret_cast<void*>(uintptr_t(0x3000));
            a.launch_args.grid_dim = dim3(5, 1, 1);
        }, "GEMV runtime dimensions, strides and addresses", false);
    }
    return count;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_cute_free", [] { return bf16_checks<deep_gemm::DenseBF16GemmCuteFreeRuntimeT<cutlass::bfloat16_t>>()
                                    + bf16_checks<deep_gemm::DenseBF16GemmCuteFreeRuntimeT<float>>(); });
    m.def("bf16_tsm_stores", [] { return bf16_tsm_store_checks<cutlass::bfloat16_t>()
                                       + bf16_tsm_store_checks<float>(); });
    m.def("int8_and_fp8_channel", int8_checks);
    m.def("fp8_block", fp8_checks);
    m.def("fp4", fp4_checks);
    m.def("gemv", gemv_checks);
}
'''


class CompileKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires a PPU-enabled PyTorch environment")
        root = Path(__file__).resolve().parents[1]
        sdk = Path(os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK')
        if not (sdk / 'bin/hgcc').is_file() and shutil.which('hgcc'):
            sdk = Path(shutil.which('hgcc')).resolve().parents[1]
        if not (sdk / 'bin/hgcc').is_file():
            raise unittest.SkipTest("requires the native PPU SDK")
        actlize = root / 'third-party/actlize_v1.0.0'
        includes = [sdk / 'targets/x86_64-linux/include', sdk / 'targets/aarch64-linux/include',
                    sdk / 'include', sdk / 'include/cccl', root / 'csrc', root / 'deep_gemm/include',
                    actlize, actlize / 'include', actlize / 'tools', root / 'third-party/fmt/include']
        library_dirs = [sdk / part for part in ('lib', 'lib64', 'lib64/stubs')]
        cls.native = load_inline(
            name='deep_gemm_compile_key_tests',
            cpp_sources=CPP_SOURCE,
            extra_include_paths=[str(path) for path in includes if path.is_dir()],
            extra_cflags=['-O0', '-std=c++17', '-DUSE_HGGC'],
            extra_ldflags=[f'-L{path}' for path in library_dirs if path.is_dir()]
                          + ['-lhggc', '-lhggcrt1', '-lhgrtc', '-lacblasLt', '-lc10_cuda', '-ltorch_cuda'],
            with_cuda=False)

    def test_bf16_cute_free_both_output_types(self):
        self.assertGreater(self.native.bf16_cute_free(), 0)

    def test_bf16_tsm_store_policy_and_source_cache(self):
        self.assertGreater(self.native.bf16_tsm_stores(), 0)

    def test_int8_and_fp8_channel(self):
        self.assertGreater(self.native.int8_and_fp8_channel(), 0)

    def test_fp8_block(self):
        self.assertGreater(self.native.fp8_block(), 0)

    def test_fp4(self):
        self.assertGreater(self.native.fp4(), 0)

    def test_gemv_both_m_and_output_types(self):
        self.assertGreater(self.native.gemv(), 0)


if __name__ == '__main__':
    unittest.main()
