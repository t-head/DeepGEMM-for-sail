#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_bf16.hpp"
#include <deep_gemm/common/fused_gemm_common.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm {

class Bf16KGroupedRuntime final : public LaunchRuntime<Bf16KGroupedRuntime> {
public:
    struct LaunchInfo {
        int shape_m, shape_n, num_groups;
        int block_m, block_n, block_k;
        int warp_m, warp_n;
        int num_stages;
        int warp_map;
        bool aiu_before_tsm;
        std::string kernel_name;
        bool cta_probe = false;
        bool output_tile_range = false;
        int b_padding_bytes = 0;
        int tsm_load_order = 0;
        bool accumulate=true;
        bool first_nonempty=false;
        bool use_psum_layout=false;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KGroupedGemmArgs kernel_params;
        void* cta_probe_ptr = nullptr;
        uint2 output_tile_range{};
        const int32_t* first_flags=nullptr;
    };

    static std::string generate_impl(const Args& args) {
        auto source=fmt::format(
            R"(
#include <deep_gemm/impls/k_grouped_bf16_gemm.cuh>

namespace deep_gemm {{

constexpr uint32_t SHAPE_M = {};
constexpr uint32_t SHAPE_N = {};
constexpr uint32_t NUM_GROUPS = {};
constexpr uint32_t BLOCK_M = {};
constexpr uint32_t BLOCK_N = {};
constexpr uint32_t BLOCK_K = {};
constexpr uint32_t WARP_M = {};
constexpr uint32_t WARP_N = {};
constexpr uint32_t NUM_STAGES = {};
constexpr uint32_t WARP_MAP = {};
constexpr bool AIU_BEFORE_TSMLD = {};
constexpr uint32_t BLOCK_SIZE = 32 * (BLOCK_M / WARP_M) * (BLOCK_N / WARP_N);

extern "C"
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void {}(const KGroupedGemmArgs args{}) {{
    {}
    bf16_k_grouped_gemm_impl<
        cutlass::bfloat16_t, float, GemmType::{},
        SHAPE_M, SHAPE_N, NUM_GROUPS,
        BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_SIZE, NUM_STAGES, true,
        WARP_MAP, AIU_BEFORE_TSMLD{}
    >(args{});
    {}
}}

}}  // namespace deep_gemm
)",
            args.launch_info.shape_m,
            args.launch_info.shape_n,
            args.launch_info.num_groups,
            args.launch_info.block_m,
            args.launch_info.block_n,
            args.launch_info.block_k,
            args.launch_info.warp_m,
            args.launch_info.warp_n,
            args.launch_info.num_stages,
            args.launch_info.warp_map,
            args.launch_info.aiu_before_tsm,
            args.launch_info.kernel_name,
            args.launch_info.first_nonempty ? ", const int32_t* first_flags" :
            args.launch_info.cta_probe ? ", unsigned long long* cta_probe"
                : args.launch_info.output_tile_range ? ", const uint2 tile_range" : "",
            args.launch_info.cta_probe ? "const unsigned long long cta_begin = __ppu_read_clock64();" : "",
            args.launch_info.use_psum_layout ? "KGroupedContiguousWithPsumLayout" : "KGroupedContiguous",
            args.launch_info.first_nonempty
                ? fmt::format(", {}, {}, false, false, true",args.launch_info.b_padding_bytes,args.launch_info.tsm_load_order)
                : args.launch_info.tsm_load_order
                ? fmt::format(", {}, {}", args.launch_info.b_padding_bytes, args.launch_info.tsm_load_order)
                : args.launch_info.b_padding_bytes ? fmt::format(", {}", args.launch_info.b_padding_bytes) : "",
            args.launch_info.first_nonempty ? ", 0u, ~0u, nullptr, nullptr, nullptr, first_flags" :
            args.launch_info.output_tile_range ? ", tile_range.x, tile_range.y" : "",
            args.launch_info.cta_probe ?
                "__syncthreads(); if (threadIdx.x == 0) { "
                "cta_probe[2 * blockIdx.x] = cta_begin; "
                "cta_probe[2 * blockIdx.x + 1] = __ppu_read_clock64(); }" : "");
        if(!args.launch_info.accumulate) {
            const std::string from="BLOCK_SIZE, NUM_STAGES, true,";
            const auto pos=source.find(from);
            DG_HOST_ASSERT(pos!=std::string::npos && source.find(from,pos+from.size())==std::string::npos);
            source.replace(pos,from.size(),"BLOCK_SIZE, NUM_STAGES, false,");
        }
        return source;
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if(args.launch_info.first_nonempty)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,args.kernel_params,args.first_flags));
        else if (args.launch_info.output_tile_range)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params, args.output_tile_range));
        else if (args.launch_info.cta_probe)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params, args.cta_probe_ptr));
        else
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

static void k_grouped_bf16_gemm_tn_contiguous_impl(
    const torch::Tensor& a, const torch::Tensor& b,
    const std::optional<torch::Tensor>& c, const torch::Tensor& d,
    const int& m, const int& n, const int& sum_k,
    const std::optional<std::vector<int>>& ks_cpu, const torch::Tensor& grouped_layout,
    const bool device_lengths = false,
    const std::optional<torch::Tensor>& cta_probe = std::nullopt,
    const int compute_cus_override = 0, const bool accumulate=true, const int32_t* first_flags=nullptr,
    const bool use_psum_layout=false) {
    const int num_groups = static_cast<int>(d.size(0));
    const bool has_host_ks=ks_cpu.has_value() && !ks_cpu->empty();
    DG_HOST_ASSERT(!device_lengths || has_host_ks);
    const int tsm_load_order = device_lengths ? get_env<int>("DG_BF16_WGRAD_TSM_LOAD_ORDER", 0) : 0;
    DG_HOST_ASSERT(tsm_load_order >= 0 && tsm_load_order <= 2);
    const int b_padding_bytes = device_lengths ? get_env<int>("DG_BF16_WGRAD_B_PAD_BYTES", 0) : 0;
    DG_HOST_ASSERT(b_padding_bytes >= 0 && b_padding_bytes <= 16384 && b_padding_bytes % 1024 == 0);
    const int tile_chunk = device_lengths ? get_env<int>("DG_BF16_WGRAD_TILE_CHUNK", 0) : 0;
    DG_HOST_ASSERT(tile_chunk >= 0 && tile_chunk <= 1 << 20);
    // A single probe buffer cannot represent several launches. Reject before
    // any output mutation; the original diagnostic remains unchanged at0.
    DG_HOST_ASSERT(tile_chunk == 0 || !cta_probe.has_value());
    // Follow upstream: choose the config using the largest group's K.
    const int max_k = has_host_ks ? *std::max_element(ks_cpu->begin(),ks_cpu->end())
        : int(std::min(a.size(0),b.size(0)));
    const auto [num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, _smem_config] =
        device_lengths
            ? deep_gemm_bf16_common::get_best_configs(m, n, max_k, num_groups, get_num_sms(),
                                                      GemmType::KGroupedContiguous)
            : has_host_ks ? deep_gemm_bf16_common::get_k_grouped_configs(m,n,sum_k,max_k,*ks_cpu,get_num_sms())
            : deep_gemm_bf16_common::get_k_grouped_device_configs(m,n,num_groups,get_num_sms());
    const int warp_map = device_lengths ? get_env<int>("DG_BF16_WGRAD_WARP_MAP", 0) : 0;
    const int aiu_before_tsm = device_lengths ? get_env<int>("DG_BF16_WGRAD_AIU_BEFORE_TSM", 1) : 1;
    DG_HOST_ASSERT(warp_map >= 0 && warp_map <= 3);
    DG_HOST_ASSERT(aiu_before_tsm == 0 || aiu_before_tsm == 1);
    const int warp_rows = block_m / warp_m, warp_cols = block_n / warp_n;
    if (warp_map == 2) DG_HOST_ASSERT((warp_cols & (warp_cols - 1)) == 0);
    if (warp_map == 3) DG_HOST_ASSERT((warp_rows & (warp_rows - 1)) == 0);
    const int block_size = 32 * (block_m / warp_m) * (block_n / warp_n);
    const int smem_size = static_cast<int>(gemm_smem_total_size(
        2, num_stages, block_m, block_n, block_k)) + b_padding_bytes;
    DG_HOST_ASSERT(smem_size <= 256 * 1024);
    std::string kernel_name = tile_chunk ? "bf16_k_grouped_gemm_tile_range"
        : cta_probe.has_value() ? "bf16_k_grouped_gemm_cta_probe" : "bf16_k_grouped_gemm";
    if (!accumulate) kernel_name += "_overwrite";
    if(first_flags) {DG_HOST_ASSERT(accumulate && !cta_probe && tile_chunk==0);kernel_name+="_first_nonempty";}
    if (b_padding_bytes) kernel_name += fmt::format("_bpad{}", b_padding_bytes);
    if (tsm_load_order) kernel_name += fmt::format("_tsmorder_v2_{}", tsm_load_order);

    auto args = Bf16KGroupedRuntime::Args{
        .launch_info = {m, n, num_groups, block_m, block_n, block_k,
                        warp_m, warp_n, num_stages, warp_map, bool(aiu_before_tsm), kernel_name, cta_probe.has_value(), tile_chunk != 0, b_padding_bytes, tsm_load_order,accumulate,first_flags!=nullptr},
        .launch_args = {get_grid_shape(num_sms), dim3(block_size), smem_size},
        .kernel_params = {
            .a_ptr = a.data_ptr(),
            .b_ptr = b.data_ptr(),
            .c_ptr = d.data_ptr(),
            .grouped_layout = grouped_layout.data_ptr<int32_t>(),
        }
    };

    args.first_flags=first_flags;
    args.launch_info.use_psum_layout=use_psum_layout;
    const auto& code = Bf16KGroupedRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    const auto& kernel = runtime->kernel;

    int blocks_per_cu = 0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, kernel, block_size, smem_size));
#else
    DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, kernel, block_size, smem_size));
#endif
    DG_HOST_ASSERT(blocks_per_cu > 0 and "K-grouped tile exceeds the device's launch resources");
    args.launch_args.grid_dim.x *= blocks_per_cu;
    DG_HOST_ASSERT(compute_cus_override >= 0 && compute_cus_override <= get_num_sms());
    const int compute_cus = compute_cus_override ? compute_cus_override
        : device_lengths ? get_env<int>("DG_BF16_WGRAD_COMPUTE_CUS", 0) : 0;
    DG_HOST_ASSERT(compute_cus >= 0 && compute_cus <= get_num_sms());
    if (compute_cus)
        args.launch_args.grid_dim.x = std::min(args.launch_args.grid_dim.x,
                                              uint32_t(compute_cus * blocks_per_cu));
    if (cta_probe.has_value()) {
        DG_HOST_ASSERT(cta_probe->numel() >= 2 * args.launch_args.grid_dim.x);
        args.cta_probe_ptr = cta_probe->data_ptr();
    }

    DgProfParam dg_prof_params;
    if (deep_gemm::profiling_enabled() && !device_lengths) {
        if(has_host_ks)dg_prof_params.set_k_grouped_params(m,n,sum_k,*ks_cpu,
            c->data_ptr()!=d.data_ptr(),current_stream());
        else {
            dg_prof_params.set_params(GemmType::KGroupedContiguousWithPsumLayout,false,"bf16",
                num_groups,m,n,0,0,nullptr,current_stream());
            dg_prof_params.add_params("ks_cpu","None");
        }
        if(use_psum_layout)dg_prof_params.add_params("use_psum_layout",1);
    }
    if (device_lengths) {
        // The internal profiler's CPU K list cannot describe a dynamic device
        // layout. External profilers still observe the actual kernel normally.
        if (tile_chunk) {
            const uint64_t tiles = uint64_t(num_groups) * math::ceil_div(m, block_m) * math::ceil_div(n, block_n);
            DG_HOST_ASSERT(tiles <= UINT32_MAX);
            for (uint64_t first = 0; first < tiles; first += tile_chunk) {
                args.output_tile_range = make_uint2(uint32_t(first), uint32_t(std::min(tiles, first + tile_chunk)));
                Bf16KGroupedRuntime::launch(runtime, args);
            }
        } else {
            Bf16KGroupedRuntime::launch(runtime, args);
        }
    } else {
        auto profiling_guard = deep_gemm::show_gemm_ut(dg_prof_params);
        Bf16KGroupedRuntime::launch(runtime, args);
    }

    const char* p_env_params = std::getenv("show_log");
    if (p_env_params && std::isdigit(static_cast<unsigned char>(*p_env_params))) {
        int num_regs = 0, local_size = 0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
        hggcFuncAttributes attributes;
        DG_HGGC_RUNTIME_CHECK(hggcFuncGetAttributes(&attributes, kernel));
        num_regs = attributes.numRegs;
        local_size = static_cast<int>(attributes.localSizeBytes);
#else
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&num_regs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel));
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&local_size, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel));
#endif

        std::printf("[%s_BF16:]\n", device_lengths ? "KGroupedDeviceLengths" : "KGroupedContiguous");
        if (device_lengths)
            std::printf("groups:%d, m:%d, n:%d, capacity_rows:%d, tile_bound_k:%d\n",
                        num_groups, m, n, sum_k, max_k);
        else
            std::printf("groups:%d, m:%d, n:%d, %s:%d, max_k_upper_bound:%d, use_psum_layout:%d\n",
                num_groups,m,n,has_host_ks ? "sum_k" : "k_capacity",sum_k,max_k,use_psum_layout);
        std::printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%u, threads:%d\n",
                    num_sms, blocks_per_cu, args.launch_args.grid_dim.x, block_size);
        std::printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n",
                    block_m, block_n, block_k, warp_m, warp_n, block_k, num_stages);
        std::printf("dynamic_smem_bytes:%d, vreg:%d, local_size_bytes:%d\n", smem_size, num_regs, local_size);
        std::printf("warp_map:%d, aiu_before_tsm:%d\n", warp_map, aiu_before_tsm);
    }
}

}  // namespace deep_gemm


#include <deep_gemm/common/mega_moe_training.cuh>
namespace deep_gemm {
class Bf16KGroupedSegmentedRuntime final : public LaunchRuntime<Bf16KGroupedSegmentedRuntime> {
public:
    struct LaunchInfo {
        int m,n,groups,bm,bn,bk,wm,wn,stages,warp_map,aiu_first,pad,load_order,segments,max_continuous_k;
        bool accumulate,first_nonempty;
        std::string name;
    };
    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KGroupedGemmArgs kernel_params;
        KGroupedSegmentArgs segments;
        const int32_t* first_flags=nullptr;
    };
    static std::string generate_impl(const Args& a) {
        const auto& c=a.launch_info;
        return fmt::format(R"(
#include <deep_gemm/impls/mega_moe_training/k_grouped_bf16_segmented.cuh>
namespace deep_gemm {{
extern "C" __launch_bounds__({},1) __global__ void {}(
        const KGroupedGemmArgs args,const KGroupedSegmentArgs segments,const int32_t* first_flags) {{
    bf16_k_grouped_segmented_impl<cutlass::bfloat16_t,float,GemmType::KGroupedContiguous,
        {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},false,false,{},{},{}>(args,segments,0u,~0u,nullptr,nullptr,nullptr,first_flags);
}}
}}
)",32*(c.bm/c.wm)*(c.bn/c.wn),c.name,c.m,c.n,c.groups,c.bm,c.bn,c.bk,c.wm,c.wn,
    32*(c.bm/c.wm)*(c.bn/c.wn),c.stages,c.accumulate,c.warp_map,bool(c.aiu_first),c.pad,c.load_order,c.segments,c.max_continuous_k,c.first_nonempty);
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.kernel_params,a.segments,a.first_flags));
    }
};

static void k_grouped_bf16_segmented_impl_host(
        const std::vector<torch::Tensor>& a,const std::vector<torch::Tensor>& b,
        const std::vector<torch::Tensor>& offsets,const torch::Tensor& d,
        int config_k,bool accumulate,int max_continuous_k=0,const int32_t* first_flags=nullptr) {
    const int count=int(a.size()),g=int(d.size(0)),m=int(d.size(1)),n=int(d.size(2));
    const auto [sms,bm,bn,bk,wm,wn,stages,unused_smem]=deep_gemm_bf16_common::get_best_configs(
        m,n,config_k,g,get_num_sms(),GemmType::KGroupedContiguous);
    const int warp_map=get_env<int>("DG_BF16_WGRAD_WARP_MAP",0);
    const int aiu_first=get_env<int>("DG_BF16_WGRAD_AIU_BEFORE_TSM",1);
    const int pad=get_env<int>("DG_BF16_WGRAD_B_PAD_BYTES",0);
    const int load_order=get_env<int>("DG_BF16_WGRAD_TSM_LOAD_ORDER",0);
    DG_HOST_ASSERT(warp_map>=0 && warp_map<=3 && (aiu_first==0 || aiu_first==1));
    DG_HOST_ASSERT(pad>=0 && pad<=16384 && pad%1024==0 && load_order>=0 && load_order<=2);
    DG_HOST_ASSERT(get_env<int>("DG_BF16_WGRAD_TILE_CHUNK",0)==0);
    const int threads=32*(bm/wm)*(bn/wn);
    const int smem=int(gemm_smem_total_size(2,stages,bm,bn,bk))+pad;
    DG_HOST_ASSERT(smem<=256*1024);
    const std::string name=fmt::format("bf16_k_grouped_gemm_segments{}_{}_cap{}_first{}",count,accumulate?"add":"overwrite",max_continuous_k,int(first_flags!=nullptr));
    Bf16KGroupedSegmentedRuntime::Args args{
        .launch_info={m,n,g,bm,bn,bk,wm,wn,stages,warp_map,aiu_first,pad,load_order,count,max_continuous_k,accumulate,first_flags!=nullptr,name},
        .launch_args={get_grid_shape(sms),dim3(threads),smem},
        .kernel_params={.a_ptr=a[0].data_ptr(),.b_ptr=b[0].data_ptr(),.c_ptr=d.data_ptr(),
                        .grouped_layout=offsets[0].data_ptr<int32_t>()},
        .segments={},
        .first_flags=first_flags
    };
    for(int s=0;s<count;++s) {
        args.segments.a[s]=a[s].data_ptr();args.segments.b[s]=b[s].data_ptr();
        args.segments.offsets[s]=offsets[s].data_ptr<int32_t>();
    }
    const auto& code=Bf16KGroupedSegmentedRuntime::generate(args);
    const auto& runtime=compiler->build(name,code);int blocks=0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,runtime->kernel,threads,smem));
#else
    DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,runtime->kernel,threads,smem));
#endif
    DG_HOST_ASSERT(blocks>0);args.launch_args.grid_dim.x*=blocks;
    const int cus=get_env<int>("DG_BF16_WGRAD_COMPUTE_CUS",0);DG_HOST_ASSERT(cus>=0 && cus<=get_num_sms());
    if(cus)args.launch_args.grid_dim.x=std::min(args.launch_args.grid_dim.x,uint32_t(cus*blocks));
    Bf16KGroupedSegmentedRuntime::launch(runtime,args);
}
} // namespace deep_gemm
