#pragma once

#include "sm80_mega_moe_copy.hpp"
#include "../heuristics/common_w4a16.hpp"

namespace deep_gemm::mega {

using W4Bank = std::tuple<torch::Tensor, torch::Tensor>;

struct SM80W4Runtime : LaunchRuntime<SM80W4Runtime> {
    struct Config {
        deep_gemm_w4a16_common::W4A16Config tile;
        deep_gemm_w4a16_common::W4A16Type weight_type;
        bool first;
        int n, k, groups, ranks, capacity, topk, tiles_per_flag;
        GatedActivation activation;
        bool fast_math;
        int threads;
        bool output_aligned;
        ReduceConfig reduce;
    };
    static std::string generate_impl(const Config& c) {
        const auto& t = c.tile;
        return fmt::format(R"(
#define DG_MEGAMOE_OWNER_MODE 1
#define DG_MEGAMOE_SPLIT_SYNC 1
#include <deep_gemm/impls/sm80_bf16_fp4_mega_moe.cuh>
using namespace deep_gemm;
using ElementB = {};
using ElementScale = {};
extern "C" __global__ __launch_bounds__({}) void {}(
        sched::Int8MegaMoESchedulerArguments routing, layout::SymBuffer<{}> scatter,
        const cutlass::bfloat16_t* a, const int* w, const ElementScale* sf,
        const int* replica, const ElementScale* replica_sf, int masters, cutlass::bfloat16_t* d,
        float limit, float alpha, float beta, float linear_beta,
        const int64_t* ids, const float* weights, __ppu_bfloat16* output, int tokens) {{
    extern __shared__ char smem[];
    sm80::w4_mega_gemm<ElementB, ElementScale, {},{},{},{},{},{},{},{},{},{},{},{},{},{},static_cast<GatedActivation>({}),{},{},{},{}>(
        routing, scatter, a, w, sf, replica, replica_sf, masters, d, limit, alpha, beta, linear_beta, smem, ids, weights, output, tokens);
}}
)", get_element_b(c.weight_type), get_element_scale(c.weight_type), c.threads, c.first ? "megamoe_w4a16_gemm1" : "megamoe_w4a16_scatter", c.ranks,
            c.first, c.n, c.k, t.block_m, t.block_n, t.block_k, t.warp_m, t.warp_k,
            c.groups, t.num_stages, c.ranks, c.capacity, c.topk, c.tiles_per_flag,
            static_cast<int>(c.activation), c.fast_math, c.output_aligned,
            c.reduce.column_splits, c.reduce.group_threads);
    }
};

static void sm80_w4_mega_moe(const torch::Tensor& y, const W4Bank& w1, const W4Bank& w2,
        const W4Bank& replica1, const W4Bank& replica2, uint8_t* base, const layout::Bf16MegaMoEBuffer& b,
        const std::vector<int64_t>& pointers, int rank, int capacity, int topk, int expected_m,
        CopyStreamResources& streams, int32_t* stats, GatedActivation activation,
        float limit, float alpha, float beta, float linear_beta, bool fast_math) {
    using namespace deep_gemm_w4a16_common;
    const int ranks = pointers.size(), groups = b.groups, h = y.size(1), i = std::get<0>(w1).size(2) / 4;
    const int sms = std::min(device_runtime->get_num_sms(), device_runtime->get_tensor_cu_count());
    const int masters = std::get<0>(w1).size(0), tokens = y.size(0);
    const auto& scales = std::get<1>(w1);
    const auto weight_type = get_w4a16_type(std::get<0>(w1).scalar_type(), scales.scalar_type());
    auto phase = [&](bool first) {
        const int n = first ? 2 * i : h, k = first ? h : i;
        auto t = get_best_configs(expected_m, n, k, groups, sms, GemmType::MegaMoE,
                                  weight_type);
        if (k % t.block_k) {
            // GEMM2 also accepts intermediate widths divisible by 128 but not 256.
            t.block_k = t.warp_k = 64;
        }
        const int threads = get_num_threads(t, false);
        // Keep each stage's JIT key independent of options consumed only by its peer.
        return SM80W4Runtime::Config{t, weight_type, first, n, k, groups, ranks, capacity, topk,
            first ? aligned_copy_ktpf(2 * k, 2 * t.block_k, 0) : 0,
            first ? activation : GatedActivation::SwiGLU, first && fast_math, threads,
            !first && is_bf16x8_aligned(y.data_ptr()),
            first ? ReduceConfig{} : select_reduce_config(tokens, h, threads, sms)};
    };
    const auto l1 = phase(true), l2 = phase(false);
    auto compute = compiler->build("megamoe_w4a16_gemm1", SM80W4Runtime::generate(l1));
    auto scatter = compiler->build("megamoe_w4a16_scatter", SM80W4Runtime::generate(l2));
    auto routing = make_mega_moe_scheduler<float>(b, base, ranks * capacity);
    routing.sym_buffer = layout::SymBuffer<>(pointers, rank);
    routing.cumulative_local_expert_recv_stats = stats;
    routing.num_copy_blocks = std::min(4, sms);
    const int bm = l1.tile.block_m;
    using Copy = SM80MegaMoECopyRuntime;
    const int copy_smem = Copy::shared_bytes(bm, ranks, groups);
    Copy::Args copy_args{2 * i, h, bm, l1.tile.block_k, groups, ranks, l1.tiles_per_flag, capacity, topk, 256,
        {dim3(routing.num_copy_blocks), dim3(256), copy_smem}, routing, true, true};
    auto copy = compiler->build("megamoe_int8_dispatch_copy", Copy::generate(copy_args));
    const auto shared_bytes = [&scales](const auto& c) {
        const int gemm = get_smem_config(c.tile, false, scales.element_size(), 32);
        if (c.first)
            return c.tile.block_k == c.tile.warp_k ? gemm : std::max(gemm, c.tile.block_m * c.tile.block_n * 4);
        const int scatter = 2 * (c.tile.block_m / c.tile.warp_m) * 16 * c.tile.block_n * 4;
        return std::max(gemm, scatter);
    };
    // Match grouped W4A16's persistent grid policy. Query the actual binary
    // for each phase; the Tensor CU budget also bounds the resident scatter grid.
    const auto grid_for = [&](const auto& runtime, const auto& c) {
        if (sms < 20) return sms;
        return sms * std::min(8, get_kernel_occupancy(runtime, c.threads, shared_bytes(c)));
    };
    const int compute_grid = grid_for(compute, l1);
    const int scatter_grid = grid_for(scatter, l2);
    launch_sync("mega_moe_dispatch_sync", ranks, reinterpret_cast<layout::MegaMoESignals*>(base));
    if (is_compile_only()) return;
    const auto caller = current_stream();
    for_ranks(ranks, [&](auto r) {
        layout::SymBuffer<decltype(r)::value> peer(pointers, rank);
        peer.base += b.scatter_storage.offset;
        auto launch_phase = [&](const auto& runtime, const auto& c, const W4Bank& w, const W4Bank& replica,
                                void* a, void* d, int grid) {
            auto cfg = construct_launch_config(runtime->kernel, caller, shared_bytes(c), dim3(grid), dim3(c.threads));
            DG_HGGC_UNIFIED_CHECK(launch_kernel(runtime->kernel, cfg, routing, peer, a, std::get<0>(w).data_ptr(),
                std::get<1>(w).data_ptr(), std::get<0>(replica).data_ptr(), std::get<1>(replica).data_ptr(),
                masters, d, limit, alpha, beta, linear_beta,
                b.topk_idx.data<int64_t>(base), b.topk_weights.data<float>(base), y.data_ptr(), tokens));
        };
        DG_HGGC_RUNTIME_CHECK(hggcEventRecord(streams.fork, caller));
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(streams.copy, streams.fork, 0));
        Copy::launch_on_stream(copy, copy_args, streams.copy);
        launch_phase(compute, l1, w1, replica1, b.local_data.data(base), b.l2_acts.data(base), compute_grid);
        DG_HGGC_RUNTIME_CHECK(hggcEventRecord(streams.done, streams.copy));
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(caller, streams.done, 0));
        routing.shape_m = b.compact;
        launch_phase(scatter, l2, w2, replica2, b.l2_acts.data(base), b.scatter_storage.data(base), scatter_grid);
    });
}
} // namespace deep_gemm::mega
