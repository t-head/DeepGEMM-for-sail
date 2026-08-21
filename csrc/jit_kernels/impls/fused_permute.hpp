#pragma once

#include <torch/python.h>
#include <cstdint>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"

namespace deep_gemm {

class FusedPermuteRuntime final : public LaunchRuntime<FusedPermuteRuntime> {
public:
    struct PermuteArgs {
        const char* src_a;
        char* dst_a;
        const char* src_sfa;
        char* dst_sfa;
        int dim0;
        int dim2_a;
        long long stride0_a_bytes;
        long long stride1_a_bytes;
        long long stride0_sfa_bytes;
        long long stride1_sfa_bytes;
        long long stride2_sfa_bytes;
    };

    struct TemplateParams {
        int vec_size = 16;
        int threads_per_block;
        int rows_per_iter;
        int dim1;
        int num_d2_vecs_a;
        int tail_bytes_a;
        int dim2_sfa_bytes;
        int d1_tile;
        bool use_fast_path;
        bool sfa_d2_contiguous;
    };

    struct Args {
        PermuteArgs permute_args;
        LaunchArgs launch_args;
        TemplateParams template_params;
        std::string kernel_name;
    };

    static std::string generate_impl(const Args& args) {
        const auto& p = args.template_params;
        return fmt::format(
            R"(
#include <deep_gemm/impls/fused_permute.cuh>

extern "C"
__global__ __launch_bounds__({})
void {}(
    const char* src_a, char* dst_a,
    const char* src_sfa, char* dst_sfa,
    int dim0, int dim2_a,
    long long stride0_a_bytes, long long stride1_a_bytes,
    long long stride0_sfa_bytes, long long stride1_sfa_bytes, long long stride2_sfa_bytes)
{{
    deep_gemm::fused_permute_kernel_impl<{},{},{},{},{},{},{},{},{},{}>(
        src_a, dst_a, src_sfa, dst_sfa,
        dim0, dim2_a,
        stride0_a_bytes, stride1_a_bytes,
        stride0_sfa_bytes, stride1_sfa_bytes, stride2_sfa_bytes);
}}
)",
            p.threads_per_block, args.kernel_name,
            p.vec_size, p.threads_per_block, p.rows_per_iter, p.dim1,
            p.num_d2_vecs_a, p.tail_bytes_a, p.dim2_sfa_bytes, p.d1_tile,
            p.use_fast_path ? "true" : "false",
            p.sfa_d2_contiguous ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        const auto& a = args.permute_args;
        DG_HGGC_CHECK(launch_kernel(kernel, config,
            a.src_a, a.dst_a, a.src_sfa, a.dst_sfa,
            a.dim0, a.dim2_a,
            a.stride0_a_bytes, a.stride1_a_bytes,
            a.stride0_sfa_bytes, a.stride1_sfa_bytes, a.stride2_sfa_bytes));
    }
};

static std::pair<torch::Tensor, torch::Tensor>
fused_permute(const torch::Tensor& a, const torch::Tensor& sfa) {
    DG_HOST_ASSERT(a.dim() == 3);
    DG_HOST_ASSERT(sfa.dim() == 3);
    TORCH_CHECK(a.stride(-1) == 1, "fused_permute: tensor a must have stride(-1) == 1 for vectorized loads");

    int64_t dim0 = a.size(0), dim1 = a.size(1), dim2_a = a.size(2);
    int64_t dim0_s = sfa.size(0), dim1_s = sfa.size(1), dim2_sfa = sfa.size(2);
    DG_HOST_ASSERT(dim0 == dim0_s && dim1 == dim1_s);

    if (dim0 == 0 || dim1 == 0 || dim2_a == 0 || dim2_sfa == 0) {
        auto out_a = torch::empty({dim1, dim0, dim2_a}, a.options());
        auto out_sfa = torch::empty({dim1, dim0, dim2_sfa}, sfa.options());
        return {out_a, out_sfa};
    }

    auto out_a = torch::empty({dim1, dim0, dim2_a}, a.options());
    auto out_sfa = torch::empty({dim1, dim0, dim2_sfa}, sfa.options());

    int elem_size_a = a.element_size();
    int64_t dim2_a_bytes = dim2_a * elem_size_a;
    int elem_size_sfa = sfa.element_size();
    int64_t dim2_sfa_bytes = dim2_sfa * elem_size_sfa;

    int num_d2_vecs_a = dim2_a_bytes / 16;
    int tail_bytes_a = dim2_a_bytes % 16;
    const int max_smem = 65536;

    int64_t total_a_bytes = dim0 * dim1 * dim2_a_bytes;
    static constexpr int64_t FAST_PATH_MAX_BYTES = 512 * 1024;
    bool use_fast_path = total_a_bytes <= FAST_PATH_MAX_BYTES;

    int threads_per_block, rows_per_iter, d1_tile;
    if (use_fast_path) {
        rows_per_iter = 1;
        d1_tile = 1;
        int d2_blocks_64 = (num_d2_vecs_a + 63) / 64;
        int grid_x_64 = (int)dim1 * d2_blocks_64;
        int grid_y_fast = (int)dim0;
        int total_blocks_64 = grid_x_64 * grid_y_fast;
        if (total_blocks_64 <= 256 && grid_y_fast >= 8) {
            threads_per_block = 32;
        } else {
            threads_per_block = 64;
        }
    } else {
        if (dim0 <= 64) {
            threads_per_block = 128;
        } else if (dim0 <= 256) {
            threads_per_block = 256;
        } else {
            threads_per_block = 512;
        }

        if (dim0 >= 256) {
            rows_per_iter = 4;
        } else if (dim0 >= 64) {
            rows_per_iter = 2;
        } else {
            int max_rows_for_smem = max_smem / ((int)dim1 * num_d2_vecs_a * 16);
            rows_per_iter = std::min((int)dim0, std::max(1, max_rows_for_smem));
            rows_per_iter = 1 << (rows_per_iter == 0 ? 0 : (31 - __builtin_clz((unsigned)rows_per_iter)));
            rows_per_iter = std::min(rows_per_iter, 16);
        }

        int grid_y = ((int)dim0 + rows_per_iter - 1) / rows_per_iter;
        int target_blocks = 128;
        int min_d1_groups = std::max(1, (target_blocks + grid_y - 1) / grid_y);
        int max_d1_tile_by_parallelism = std::max(1, (int)dim1 / min_d1_groups);
        int max_d1_tile_by_smem = std::max(1, max_smem / (rows_per_iter * num_d2_vecs_a * 16));
        d1_tile = std::min((int)dim1, std::min(max_d1_tile_by_smem, max_d1_tile_by_parallelism));
        d1_tile = std::max(1, d1_tile);
        while ((int)dim1 % d1_tile != 0) {
            d1_tile--;
        }
    }

    long long stride0_a_bytes = (long long)a.stride(0) * elem_size_a;
    long long stride1_a_bytes = (long long)a.stride(1) * elem_size_a;
    long long stride0_sfa_bytes = (long long)sfa.stride(0) * elem_size_sfa;
    long long stride1_sfa_bytes = (long long)sfa.stride(1) * elem_size_sfa;
    long long stride2_sfa_bytes = (long long)sfa.stride(2) * elem_size_sfa;
    bool sfa_d2_contiguous = (sfa.stride(-1) == 1);

    int smem_size;
    if (use_fast_path) {
        smem_size = 0;
    } else {
        smem_size = rows_per_iter * d1_tile * num_d2_vecs_a * 16;
    }

    int grid_x, grid_y;
    if (use_fast_path) {
        int d2_blocks = (num_d2_vecs_a + threads_per_block - 1) / threads_per_block;
        grid_x = (int)dim1 * d2_blocks;
    } else {
        grid_x = (int)dim1 / d1_tile;
    }
    grid_y = ((int)dim0 + rows_per_iter - 1) / rows_per_iter;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(threads_per_block, 1, 1);

    std::string element_a = (a.dtype() == torch::kFloat8_e4m3fn) ? "fp8" : "int8";
    std::string kernel_name = "fused_permute_" + element_a;

    auto args = FusedPermuteRuntime::Args{
        .permute_args = {
            (const char*)a.data_ptr(), (char*)out_a.data_ptr(),
            (const char*)sfa.data_ptr(), (char*)out_sfa.data_ptr(),
            (int)dim0, (int)dim2_a_bytes,
            stride0_a_bytes, stride1_a_bytes,
            stride0_sfa_bytes, stride1_sfa_bytes, stride2_sfa_bytes
        },
        .launch_args = {grid, block, smem_size},
        .template_params = {
            16, threads_per_block, rows_per_iter, (int)dim1,
            num_d2_vecs_a, tail_bytes_a, (int)dim2_sfa_bytes, d1_tile,
            use_fast_path, sfa_d2_contiguous
        },
        .kernel_name = kernel_name
    };

    const auto& code = FusedPermuteRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code, threads_per_block, smem_size);
    FusedPermuteRuntime::launch(runtime, args);

    return {out_a, out_sfa};
}

} // namespace deep_gemm
