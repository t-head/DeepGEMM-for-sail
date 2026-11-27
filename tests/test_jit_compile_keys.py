"""Native PPU host-codegen regression; requires the source checkout and PPU SDK.

Run on an idle PPU: python tests/test_jit_compile_keys.py -v
Host checks exercise codegen and keys. The coexistence check compiles generated
GEMMs in one translation unit without launching them.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch
from torch.utils.cpp_extension import load_inline


# Keep the native checks with their Python runner; no production test bindings.
CPP_SOURCE = r'''
// Exercise the actual host code generators, without compiling or launching GEMMs.
#include "jit_kernels/impls/bf16_gemm.hpp"
#include "jit_kernels/impls/bf16xfp32_gemm.hpp"
#include "jit_kernels/impls/fp4_gemm.hpp"
#include "jit_kernels/impls/fp8_gemm.hpp"

#include <stdexcept>
#include <vector>

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
int dual_b_checks() {
    using Runtime = DenseBf16xFp32GemmCuteFreeRuntimeT<Output>;
    typename Runtime::Args args{};
    auto& i = args.launch_info;
    i.block_m = 128; i.block_n = 128; i.block_k = 64;
    i.warp_m = 32; i.warp_n = 64; i.warp_k = 64; i.num_stages = 4;
    i.gemm_type = "Normal"; i.kernel_name = "test_dual_b";
    i.decode_split_k = 1;
    i.packed_store = true;
    args.kernel_params.M = 256; args.kernel_params.N = 384; args.kernel_params.K = 7168;
    int count = tile_checks<Runtime>(args);
#define CHECK_DUAL(field, value) \
    count += check_change<Runtime>(args, [](auto& a) { a.launch_info.field = value; }, #field)
    CHECK_DUAL(warp_k, 32);
    CHECK_DUAL(gemm_type, "MGroupedContiguous");
    CHECK_DUAL(early_g2s, true);
    CHECK_DUAL(mma_s2r_overlap, true);
    CHECK_DUAL(overlap_prologue, true);
    CHECK_DUAL(valid_epilogue, true);
    CHECK_DUAL(packed_dual_n, true);
    CHECK_DUAL(static_router, true);
    CHECK_DUAL(static_stages, true);
    CHECK_DUAL(decode_split_k, 2);
    CHECK_DUAL(packed_store, false);
    CHECK_DUAL(cta_k, true);
    CHECK_DUAL(one_tile, true);
    CHECK_DUAL(k_order, 6);
#undef CHECK_DUAL
    count += check_change<Runtime>(args, [](auto& a) { a.kernel_params.N = 386; }, "N alignment");
    count += check_change<Runtime>(args, [](auto& a) {
        a.kernel_params.M = 512; a.kernel_params.N = 512; a.kernel_params.K = 8192;
        a.kernel_params.low_scale = 0.5f; a.kernel_params.lda = 8192;
        a.kernel_params.ptr_A = reinterpret_cast<const cutlass::bfloat16_t*>(uintptr_t(0x1000));
        a.kernel_params.ptr_D = reinterpret_cast<Output*>(uintptr_t(0x2000));
        a.launch_args.grid_dim = dim3(20); a.launch_args.smem_size = 65536;
    }, "dynamic dimensions, scale, strides, addresses and launch", false);
    args.launch_info.static_router = true;
    count += check_change<Runtime>(args, [](auto& a) { a.kernel_params.M += 1; }, "static M");
    count += check_change<Runtime>(args, [](auto& a) { a.kernel_params.N += 128; }, "static N");
    count += check_change<Runtime>(args, [](auto& a) { a.kernel_params.K += 64; }, "static K");
    const auto source = Runtime::generate_impl(args);
    for (const char* fixed : {
        "using Pipeline = dual_b::PipelinePolicy<0, 0, 0, 0>;",
        "using Policy = dual_b::KernelPolicy<Pipeline, 0, 0, 0, 256, 384, 7168, 1, 1, 0, 0>;",
        ", Policy>;", "host/device Params ABI mismatch", "host/device shared storage mismatch",
        "__launch_bounds__(deep_gemm::Kernel::MaxThreadsPerBlock, 1)",
        "BF16xFp32GemmCuteFreeKernel<128, 128, 64, 32, 64, 64, 4, true, true,"}) {
        if (source.find(fixed) == std::string::npos)
            throw std::runtime_error("fixed dual-B codegen default mismatch");
        ++count;
    }
    for (const char* prefix : {"DG_DUAL_B_", "DG_BFOUT_"}) {
        if (source.find(prefix) != std::string::npos)
            throw std::runtime_error("dual-B codegen leaked a configuration macro");
        ++count;
    }
    return count;
}

int dual_b_lut_checks(std::vector<std::string>* sources = nullptr) {
    using namespace dual_b;
    int count = 0;
    struct Expected {
        int m, n, k;
        at::ScalarType dtype;
        int bm, bn, bk, wm, wn, wk, stages, fixed_grid, grid, smem;
        unsigned pipeline;
        int split_k, k_order;
    };
    const Expected cases[] = {
        {8,384,7168,at::kBFloat16,16,16,128,16,16,128,4,48,48,65536,PackedStore|StaticShape|PackedN,2,0},
        {16,384,7168,at::kBFloat16,16,16,128,16,16,128,4,48,48,65536,PackedStore|StaticShape|PackedN,2,0},
        {128,384,7168,at::kBFloat16,48,32,128,16,16,128,5,36,36,143360,PackedStore|StaticShape|OneTile,1,0},
        {256,384,7168,at::kBFloat16,96,32,128,48,16,64,6,36,36,245760,PackedStore|StaticShape|OneTile|CtaK,1,0},
        {512,384,7168,at::kBFloat16,96,64,64,48,16,64,5,36,36,143360,PackedStore|StaticShape|OneTile|MmaS2ROverlap,1,0},
        {1024,384,7168,at::kBFloat16,192,64,64,48,32,64,4,36,36,163840,PackedStore|StaticShape|OneTile|MmaS2ROverlap,1,0},
        {2048,384,7168,at::kBFloat16,256,96,64,32,48,64,4,32,32,229376,PackedStore|StaticShape|OverlapPrologue,1,0},
        {4096,384,7168,at::kBFloat16,320,128,64,80,32,64,3,39,39,221184,PackedStore,1,0},
        {8192,384,7168,at::kBFloat16,320,128,64,80,32,64,3,39,39,221184,PackedStore|StaticShape|OverlapPrologue|ValidEpilogue,1,1},
        {16384,384,7168,at::kBFloat16,256,128,64,64,32,64,4,39,39,262144,PackedStore|StaticShape|OverlapPrologue|ValidEpilogue|StaticStages,1,6},
        {32768,384,7168,at::kBFloat16,256,128,64,64,16,64,4,0,39,262144,0,1,0},
        {8,384,7168,at::kFloat,16,16,128,16,16,128,4,0,48,65536,StaticShape|PackedN,2,0},
        {16,384,7168,at::kFloat,16,16,128,16,16,128,4,0,48,65536,StaticShape|PackedN,2,0},
        {128,384,7168,at::kFloat,64,32,128,16,16,128,4,0,24,131072,StaticShape|EarlyG2S,1,0},
        {256,384,7168,at::kFloat,96,32,64,32,16,64,7,0,36,143360,StaticShape|EarlyG2S,1,0},
        {512,384,7168,at::kFloat,128,64,64,32,16,64,4,0,24,131072,StaticShape,1,0},
        {1024,384,7168,at::kFloat,192,64,64,32,16,64,4,0,36,163840,0,1,0},
        {2048,384,7168,at::kFloat,128,192,64,32,32,64,4,0,32,262144,0,1,0},
        {4096,384,7168,at::kFloat,320,128,64,80,32,64,2,0,39,147456,StaticShape|ValidEpilogue,1,0},
        {8192,384,7168,at::kFloat,320,128,64,80,32,64,3,0,39,221184,0,1,0},
        {16384,384,7168,at::kFloat,256,128,64,64,32,64,4,0,39,262144,StaticShape|OverlapPrologue|ValidEpilogue,1,0},
        {32768,384,7168,at::kFloat,256,128,64,64,16,64,4,0,39,262144,0,1,0},
        {512,2048,4096,at::kBFloat16,128,256,64,64,32,64,3,0,32,245760,0,1,0},
        {512,2048,4096,at::kFloat,128,256,64,64,32,64,3,0,32,245760,0,1,0},
    };
    for (const auto& e : cases) {
        const ExactKey key{e.m,e.n,e.k,e.dtype,AC_PPU0015,39};
        const auto* cfg = exact_config(key,false,false);
        if (!cfg || !(cfg->key == key))
            throw std::runtime_error("exact LUT key mismatch");
        const auto geometry = std::make_tuple(cfg->bm,cfg->bn,cfg->bk,cfg->wm,cfg->wn,cfg->wk,cfg->stages);
        const auto wanted = std::make_tuple(e.bm,e.bn,e.bk,e.wm,e.wn,e.wk,e.stages);
        if (geometry != wanted || cfg->grid != e.fixed_grid || cfg->pipeline != e.pipeline
            || cfg->decode_split_k != e.split_k || cfg->k_order != e.k_order)
            throw std::runtime_error("LUT geometry or pipeline mismatch");
        const int smem = shared_storage_bytes(cfg->bm,cfg->bn,cfg->bk,cfg->wm,cfg->wn,cfg->wk,
            cfg->stages,cfg->has(PackedN),cfg->decode_split_k,cfg->has(CtaK));
        if (smem != e.smem || smem > 256 * 1024)
            throw std::runtime_error("LUT shared storage mismatch");
        const int output_tiles = ceil_div(key.m,cfg->bm) * ceil_div(key.n,cfg->has(PackedN) ? cfg->bn/2 : cfg->bn);
        const int grid = cfg->grid > 0 ? cfg->grid : std::min((256 * 1024 / smem) * key.num_sms,output_tiles);
        if (grid != e.grid || grid > output_tiles || (cfg->has(OneTile) && grid != output_tiles))
            throw std::runtime_error("LUT launch grid mismatch");
        ++count;
        const ExactKey misses[] = {
            {key.m+1,key.n,key.k,key.output_dtype,key.soc,key.num_sms},
            {key.m,key.n+2,key.k,key.output_dtype,key.soc,key.num_sms},
            {key.m,key.n,key.k+64,key.output_dtype,key.soc,key.num_sms},
            {key.m,key.n,key.k,at::kHalf,key.soc,key.num_sms},
            {key.m,key.n,key.k,key.output_dtype,AC_PPU0010,key.num_sms},
            {key.m,key.n,key.k,key.output_dtype,AC_PPU_MAX_ARCH,key.num_sms},
            {key.m,key.n,key.k,key.output_dtype,key.soc,20},
            {key.m,key.n,key.k,key.output_dtype,key.soc,38},
            {key.m,key.n,key.k,key.output_dtype,key.soc,40},
            {key.m,key.n,key.k,key.output_dtype,key.soc,64},
        };
        for (const auto& other : misses) {
            if (key == other || other == key || exact_config(other,false,false) || exact_config(other,false,true))
                throw std::runtime_error("non-exact LUT key hit");
            ++count;
        }
        auto other_dtype = key;
        other_dtype.output_dtype = key.output_dtype == at::kFloat ? at::kBFloat16 : at::kFloat;
        const auto* other_cfg = exact_config(other_dtype,false,false);
        bool other_expected = false;
        for (const auto& candidate : cases)
            other_expected |= candidate.m == key.m && candidate.n == key.n && candidate.k == key.k
                && candidate.dtype == other_dtype.output_dtype;
        if (key == other_dtype || other_dtype == key || bool(other_cfg) != other_expected
            || (other_cfg && (other_cfg == cfg || !(other_cfg->key == other_dtype))))
            throw std::runtime_error("LUT output dtype mismatch");
        ++count;
        if (exact_config(key,true,false) || exact_config(key,true,true))
            throw std::runtime_error("explicit config priority bypassed");
        const auto* rtc_cfg = exact_config(key,false,true);
        if (rtc_cfg != (cfg->has(PackedStore) ? nullptr : cfg))
            throw std::runtime_error("LUT backend eligibility mismatch");
        ++count;
        auto check_codegen = [&](auto output) {
            using Runtime = DenseBf16xFp32GemmCuteFreeRuntimeT<decltype(output)>;
            typename Runtime::Args args{};
            args.launch_info = {cfg->bm,cfg->bn,cfg->bk,cfg->wm,cfg->wn,cfg->wk,cfg->stages,
                "Normal","test_dual_b_lut",cfg->has(EarlyG2S),cfg->has(MmaS2ROverlap),
                cfg->has(OverlapPrologue),cfg->has(ValidEpilogue),cfg->has(PackedN),cfg->has(StaticShape),
                cfg->has(StaticStages),cfg->decode_split_k,cfg->has(PackedStore),cfg->has(CtaK),
                cfg->has(OneTile),cfg->k_order};
            args.kernel_params.M = key.m; args.kernel_params.N = key.n; args.kernel_params.K = key.k;
            if (sources) args.launch_info.kernel_name += std::to_string(sources->size());
            const auto source = Runtime::generate_impl(args);
            const auto pipeline = fmt::format("dual_b::PipelinePolicy<{}, {}, {}, {}>",
                int(cfg->has(EarlyG2S)), int(cfg->has(MmaS2ROverlap)),
                int(cfg->has(StaticStages)), cfg->has(PackedStore) ? cfg->k_order : 0);
            const auto policy = fmt::format("dual_b::KernelPolicy<Pipeline, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                int(cfg->has(OverlapPrologue)), int(cfg->has(ValidEpilogue)), int(cfg->has(PackedN)),
                cfg->has(StaticShape) ? key.m : 0, cfg->has(StaticShape) ? key.n : 0,
                cfg->has(StaticShape) ? key.k : 0, cfg->decode_split_k, int(cfg->has(PackedStore)),
                int(cfg->has(PackedStore) && cfg->has(CtaK)), int(cfg->has(PackedStore) && cfg->has(OneTile)));
            if (source.find(pipeline) == std::string::npos || source.find(policy) == std::string::npos
                || source.find("DG_DUAL_B_") != std::string::npos || source.find("DG_BFOUT_") != std::string::npos
                || source.find(fmt::format("Kernel::SharedStorageSize == {}",e.smem)) == std::string::npos)
                throw std::runtime_error("LUT template policy or storage mismatch");
            if (sources) sources->push_back(source);
            ++count;
        };
        if (key.output_dtype == at::kFloat) check_codegen(float{});
        else check_codegen(cutlass::bfloat16_t{});
    }
    return count;
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
    info.kernel_strategy = "Default";
    args.launch_args.smem_size = 32768;
    int count = tile_checks<Runtime>(args);
#define CHECK_FP4(field, value) \
    count += check_change<Runtime>(args, [](auto& a) { a.launch_info.field = value; }, #field)
    CHECK_FP4(n, 514);
    CHECK_FP4(k, 1024);
    CHECK_FP4(num_groups, 2);
    CHECK_FP4(gemm_type, "MGroupedContiguous");
    CHECK_FP4(kernel_strategy, "OverlapPrologue");
    CHECK_FP4(enable_sbo_overlap, true);
    CHECK_FP4(hasBias, true);
    CHECK_FP4(n_expand, 2);
    CHECK_FP4(epilogue_type, "SiluAndMulPostQuantFp4");
    CHECK_FP4(apply_swiglu_limit, true);
    CHECK_FP4(swizzle_b, true);
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
        CHECK_GEMV(staticK, true);
        CHECK_GEMV(pairLoad, true);
        CHECK_GEMV(persistent, true);
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
        args.launch_info.staticK = true;
        args.kernel_params.K = 64;
        count += check_change<Runtime>(args, [](auto& a) {
            a.kernel_params.K = 96;
        }, "GEMV static K and row stride");
    }
    return count;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_cute_free", [] { return bf16_checks<deep_gemm::DenseBF16GemmCuteFreeRuntimeT<cutlass::bfloat16_t>>()
                                    + bf16_checks<deep_gemm::DenseBF16GemmCuteFreeRuntimeT<float>>(); });
    m.def("dual_b", [] { return dual_b_checks<cutlass::bfloat16_t>() + dual_b_checks<float>() + dual_b_lut_checks(); });
    m.def("dual_b_sources", [] {
        std::vector<std::string> sources;
        dual_b_lut_checks(&sources);
        return sources;
    });
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
                    sdk / 'include', sdk / 'include/cccl',
                    # Mirrors setup.py: Torch's c10 CUDA headers need `cuda_runtime_api.h`,
                    # and this tree must stay after the native ones so that the native
                    # `hggc_fp16.h` definitions win over its `cuda_fp16.h` copies.
                    sdk / 'CUDA_SDK/include',
                    root / 'csrc', root / 'deep_gemm/include',
                    actlize, actlize / 'include', actlize / 'tools', root / 'third-party/fmt/include']
        cls.sdk = sdk
        cls.device_includes = [root / 'deep_gemm/include', actlize / 'include', actlize / 'tools/util/include']
        library_dirs = [sdk / part for part in ('lib', 'lib64', 'lib64/stubs')]
        cls.native = load_inline(
            name='deep_gemm_compile_key_tests',
            cpp_sources=CPP_SOURCE,
            extra_include_paths=[str(path) for path in includes if path.is_dir()],
            extra_cflags=['-O0', '-std=c++17', '-DUSE_HGGC'],
            extra_ldflags=[f'-L{path}' for path in library_dirs if path.is_dir()]
                          # `-lacblas` provides `acblasGetStatusString`; see the note next to
                          # `build_libraries` in `setup.py`.
                          + ['-lhggc', '-lhggcrt1', '-lhgrtc', '-lacblasLt', '-lacblas',
                             '-lc10_cuda', '-ltorch_cuda'],
            with_cuda=False)

    def test_bf16_cute_free_both_output_types(self):
        self.assertGreater(self.native.bf16_cute_free(), 0)

    def test_dual_b_lut_and_codegen_keys(self):
        self.assertGreater(self.native.dual_b(), 0)

    def test_dual_b_template_coexistence(self):
        if torch.cuda.get_device_capability() != (8, 9):
            self.skipTest('dual-B device kernels require PPU1.5')
        sources = self.native.dual_b_sources()
        self.assertEqual(len(sources), 24)
        units = []
        for index, source in enumerate(sources):
            source = source.replace('namespace deep_gemm {', f'namespace deep_gemm {{ namespace config_{index} {{')
            source = source.replace('} // namespace deep_gemm', '}} // namespace deep_gemm')
            source = source.replace('deep_gemm::Kernel', f'deep_gemm::config_{index}::Kernel')
            units.append(source)
        units.append(r'''
using Tag = std::integral_constant<deep_gemm::GemmType, deep_gemm::GemmType::Normal>;
using DefaultKernel = deep_gemm::BF16xFp32GemmCuteFreeKernel<128, 128, 64, 32, 64, 64, 4, true, true, Tag>;
using ExplicitDefault = deep_gemm::BF16xFp32GemmCuteFreeKernel<128, 128, 64, 32, 64, 64, 4, true, true, Tag,
    cutlass::bfloat16_t, deep_gemm::dual_b::KernelPolicy<>>;
static_assert(std::is_same_v<DefaultKernel, ExplicitDefault>);
static_assert(DefaultKernel::SharedStorageSize == 196608);
static_assert(!std::is_same_v<deep_gemm::config_0::Kernel, deep_gemm::config_1::Kernel>);
static_assert(!std::is_same_v<deep_gemm::config_0::Kernel, deep_gemm::config_11::Kernel>);
''')
        with tempfile.TemporaryDirectory(prefix='dual_b_templates_') as directory:
            source = Path(directory) / 'coexist.cu'
            source.write_text('\n'.join(units))
            command = [str(self.sdk / 'bin/hgcc'), '-std=c++17', '-O3', '-arch=ppu_15', '-dopt', '3',
                       '-ftemplate-depth=8192', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                       '-x', 'hg', '-hgbin', str(source), '-o', str(Path(directory) / 'coexist.hgbin')]
            command.extend(f'-I{path}' for path in self.device_includes)
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout)

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
