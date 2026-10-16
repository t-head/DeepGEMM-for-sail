#pragma once

#include "mega_moe.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega_moe_aux {

struct AuxKernelTag {};

inline void run_aux(const std::string& suffix, const std::string& signature, const std::string& body,
                    dim3 grid, int threads, std::vector<NativeKernelArgument> arguments) {
    const std::string name = "ppu_moe_aux_" + suffix;
    NativeKernelArgs args{"#include <deep_gemm/impls/mega_moe_combine.cuh>\n"
                          "extern \"C\" __global__ void __launch_bounds__(" +
                              std::to_string(threads) + ") " + name + "(" + signature +
                              ") {\nusing namespace deep_gemm::mega_moe::detail;\n" + body + "\n}\n",
                          {grid, dim3(threads), 0},
                          std::move(arguments)};
    run_native_kernel<AuxKernelTag>(name, args);
}

inline void reset_done(const torch::Tensor& done) {
    TORCH_CHECK(done.is_cuda() && done.scalar_type() == torch::kInt32 && done.numel() > 0);
    if (get_compile_mode() == static_cast<int>(CompileMode::COMPILE_AND_RUN) &&
        get_env<std::string>("HGGC_WARM_UP").empty())
        DG_HGGC_RUNTIME_CHECK(hggcMemsetAsync(done.data_ptr(), 0, sizeof(int32_t), current_stream()));
}

inline void barrier_reduce_impl(const torch::Tensor& buf, const torch::Tensor& weights, int tokens,
                                int hidden, int topk, int capacity, const torch::Tensor& pointers,
                                int rank, int world, const torch::Tensor& done,
                                const std::optional<torch::Tensor>& scales, int scale_block_n,
                                const torch::Tensor& out) {
    reset_done(done);
    const bool fp8 = buf.scalar_type() == torch::kFloat8_e4m3fn;
    const bool scaled = scales.has_value();
    const int chunk = mega_moe_config::chunked_reduce(tokens, fp8 && !scaled) ? 256 : 0;
    const int threads = chunk ? chunk : 512;
    const int scale_tiles = scaled ? ceil_div(hidden, scale_block_n) : 0;
    const int shift = fp8 && !scaled && !mega_moe_config::fp8_unit_scale()
                          ? get_env<int>("DG_MOE_SCATTER_FP8_EXP_SHIFT", 0) : 0;
    const auto body = fmt::format(
        "barrier_reduce_kernel<{}, {}, {}, {}, {}>(b, s, w, y, t, h, c, st, sn, shift, p, r, ws, d);",
        topk, fp8, scaled, chunk, threads);
    run_aux("barrier_reduce",
            "const void* b, const uint64_t* s, const float* w, __ppu_bfloat16* y, int t, int h, int c, "
            "int st, int sn, int shift, int32_t** p, int r, int ws, int32_t* d",
            body, dim3(std::max(tokens, 1), chunk ? ceil_div(hidden / 8, chunk) : 1), threads,
            {buf.data_ptr(), scaled ? scales->data_ptr() : nullptr, weights.data_ptr(), out.data_ptr(),
             int32_t(tokens), int32_t(hidden), int32_t(capacity), int32_t(scale_tiles), int32_t(scale_block_n),
             int32_t(shift), pointers.data_ptr(), int32_t(rank), int32_t(world), done.data_ptr()});
}

inline void prepare_rows_impl(const torch::Tensor& counts, const torch::Tensor& rows,
                              const torch::Tensor& valid, const torch::Tensor& offsets, int block_m,
                              const std::optional<torch::Tensor>& meta,
                              const std::optional<torch::Tensor>& packed,
                              const std::optional<torch::Tensor>& src_rows,
                              const std::optional<torch::Tensor>& src_offsets, int src_block_m) {
    const int groups = counts.numel();
    if (groups == 0)
        return;
    if (!meta) {
        run_aux("prepare_rows", "const uint64_t* c, int32_t* r, int32_t* v, int32_t* o, int g, int bm",
                "prepare_moe_gemm2_rows_kernel(c, r, v, o, g, bm);", dim3(1), 1,
                {counts.data_ptr(), rows.data_ptr(), valid.data_ptr(), offsets.data_ptr(), int32_t(groups),
                 int32_t(block_m)});
        return;
    }
    run_aux("prepare_rows_meta",
            "const uint64_t* c, const int32_t* m, int32_t* dr, int32_t* v, int32_t* doff, int32_t* sr, "
            "int32_t* soff, int64_t* p, int g, int db, int sb",
            "prepare_moe_gemm2_rows_and_pack_meta_kernel(c, m, dr, v, doff, sr, soff, p, g, db, sb);",
            dim3(1), 256,
            {counts.data_ptr(), meta->data_ptr(), rows.data_ptr(), valid.data_ptr(),
             offsets.data_ptr(), src_rows->data_ptr(), src_offsets->data_ptr(), packed->data_ptr(),
             int32_t(groups), int32_t(block_m), int32_t(src_block_m)});
}

} // namespace deep_gemm::mega_moe_aux
