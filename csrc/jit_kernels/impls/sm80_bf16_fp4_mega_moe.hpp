#pragma once

#include "sm80_int8_int8_mega_moe.hpp"
#include "../heuristics/common_w4a16.hpp"

namespace deep_gemm::mega {

using W4Bank = std::tuple<torch::Tensor, torch::Tensor>;

struct SM80W4Runtime : LaunchRuntime<SM80W4Runtime> {
    struct Config {
        deep_gemm_w4a16_common::W4A16Config tile;
        bool first;
        int n, k, groups, ranks, capacity, topk, tiles_per_flag;
        GatedActivation activation;
        bool fast_math;
        int threads;
        bool output_aligned;
        ReduceConfig reduce;
        int head_m = 0;
    };
    static std::string generate_impl(const Config& c) {
        const auto& t = c.tile;
        return fmt::format(R"(
#define DG_MEGAMOE_OWNER_MODE 1
#define DG_MEGAMOE_SPLIT_SYNC 1
#include <deep_gemm/impls/sm80_bf16_fp4_mega_moe.cuh>
using namespace deep_gemm;
extern "C" __global__ __launch_bounds__({}) void {}(
        sched::Int8MegaMoESchedulerArguments routing, layout::SymBuffer<{}> scatter,
        const cutlass::bfloat16_t* a, const int* w, const uint8_t* sf,
        const int* replica, const uint8_t* replica_sf, int masters, cutlass::bfloat16_t* d,
        float limit, float alpha, float beta, float linear_beta,
        const int64_t* ids, const float* weights, __ppu_bfloat16* output, int tokens) {{
    extern __shared__ char smem[];
    sm80::w4_mega_gemm<{},{},{},{},{},{},{},{},{},{},{},{},{},{},static_cast<GatedActivation>({}),{},{},{},{},{}>(
        routing, scatter, a, w, sf, replica, replica_sf, masters, d, limit, alpha, beta, linear_beta, smem, ids, weights, output, tokens);
}}
)", c.threads, c.first ? "megamoe_w4a16_gemm1" : "megamoe_w4a16_scatter", c.ranks,
            c.first, c.n, c.k, t.block_m, t.block_n, t.block_k, t.warp_m, t.warp_k,
            c.groups, t.num_stages, c.ranks, c.capacity, c.topk, c.tiles_per_flag,
            static_cast<int>(c.activation), c.fast_math, c.output_aligned,
            c.reduce.column_splits, c.reduce.group_threads, c.head_m);
    }
};

static void sm80_w4_mega_moe(const torch::Tensor& y, const W4Bank& w1, const W4Bank& w2,
        const W4Bank& replica1, const W4Bank& replica2, uint8_t* base, const layout::Bf16MegaMoEBuffer& b,
        const std::vector<int64_t>& pointers, int rank, int capacity, int topk, int expected_m,
        mega_int8::CopyStreamResources& streams, int32_t* stats, GatedActivation activation,
        float limit, float alpha, float beta, float linear_beta, bool fast_math) {
    using namespace deep_gemm_w4a16_common;
    const int ranks = pointers.size(), groups = b.groups, h = y.size(1), i = std::get<0>(w1).size(2) / 4;
    const int sms = device_runtime->get_num_sms(), masters = std::get<0>(w1).size(0), tokens = y.size(0);
    auto phase = [&](bool first) {
        const int n = first ? 2*i : h, k = first ? h : i;
        auto t = get_best_configs(expected_m, n, k, groups, sms, GemmType::MGroupedNoPad,
                                  W4A16Type::mxfp4_e8m0);
        t.n_expand = 1;
        if (k % t.block_k) {
            if (first) t.block_n = 256;
            t.block_k = t.warp_k = 64;
        }
        const int threads = get_num_threads(t, false);
        return SM80W4Runtime::Config{t, first, n, k, groups, ranks, capacity, topk,
            first ? aligned_copy_ktpf(2*k, 2*t.block_k, 0) : 0,
            activation, fast_math, threads, is_bf16x8_aligned(y.data_ptr()),
            first ? ReduceConfig{} : select_reduce_config(tokens, h, threads, sms)};
    };
    auto l1 = phase(true);
    const auto l2 = phase(false);
    const int mode = get_env<int>("DG_W4_HEAD_MODE");
    const int count = std::min(sms, get_env<int>("DG_W4_HEAD_NCB", 4));
    const int head = ceil_div(sms, ceil_div(l1.n, l1.tile.block_n));
    const int alignment = 512 / std::gcd(2 * l1.tile.block_k, 512);
    if (mode == 2 || mode == 3) {
        const int tiles = l1.k / l1.tile.block_k;
        const int half = std::max(ceil_div(l1.tiles_per_flag, 2), ceil_div(tiles, 7));
        l1.tiles_per_flag = ceil_div(half, alignment) * alignment;
    }
    const int stripes = ceil_div(l1.k / l1.tile.block_k, l1.tiles_per_flag);
    const int workers = std::min(stripes, count / head);
    if ((mode == 1 || mode == 2 || mode == 4) && workers >= 2)
        l1.head_m = head;
    const int grid = mode == 4 && l1.head_m ? head * workers : count;
    auto compute = compiler->build("megamoe_w4a16_gemm1", SM80W4Runtime::generate(l1));
    auto scatter = compiler->build("megamoe_w4a16_scatter", SM80W4Runtime::generate(l2));
    auto routing = make_mega_moe_scheduler<float>(b, base, ranks * capacity);
    routing.sym_buffer = layout::SymBuffer<>(pointers, rank);
    routing.cumulative_local_expert_recv_stats = stats;
    routing.num_copy_blocks = grid;
    const int bm = l1.tile.block_m;
    const int copy_smem = std::max(ceil_div(4*ranks + 8*bm, 16)*16, 4*groups*(ranks+4));
    using Copy = mega_int8::Int8MegaMoECopyRuntime;
    Copy::Args ca{2*i,h,bm,l1.tile.block_k,groups,ranks,l1.tiles_per_flag,capacity,topk,256,
        {dim3(routing.num_copy_blocks),dim3(256),copy_smem}, routing,true,true,l1.head_m,l1.head_m ? workers : 1};
    auto copy = compiler->build("megamoe_int8_dispatch_copy", Copy::generate(ca));
    const auto shared_bytes = [](const auto& c) {
        const int epilogue = c.first ? c.tile.block_m * c.tile.block_n * 4 :
            2 * (c.tile.block_m / c.tile.warp_m) * 16 * c.tile.block_n * 4;
        return std::max(get_smem_config(c.tile, false, 1, 32), epilogue);
    };
    // Scatter releases through a grid barrier before the cross-rank wait.
    // Every GEMM2 CTA must be resident to let the last writer notify its peers.
    const int resident = get_kernel_occupancy(scatter, l2.threads, shared_bytes(l2)) *
                         device_runtime->get_prop()->multiProcessorCount;
    const int scatter_grid = std::min(sms, resident);
    if (is_compile_only()) return;
    auto* signals = reinterpret_cast<layout::MegaMoESignals*>(base);
    const auto caller = current_stream();
    for_ranks(ranks, [&](auto r) {
        layout::SymBuffer<decltype(r)::value> peer(pointers, rank);
        peer.base += b.scatter_storage.offset;
        auto launch_phase = [&](const auto& runtime, const auto& c, const W4Bank& w, const W4Bank& replica,
                                void* a, void* d) {
            const int grid = c.first ? sms : scatter_grid;
            auto cfg = construct_launch_config(runtime->kernel, caller, shared_bytes(c), dim3(grid), dim3(c.threads));
            DG_HGGC_UNIFIED_CHECK(launch_kernel(runtime->kernel, cfg, routing, peer, a, std::get<0>(w).data_ptr(),
                std::get<1>(w).data_ptr(), std::get<0>(replica).data_ptr(), std::get<1>(replica).data_ptr(),
                masters, d, limit, alpha, beta, linear_beta,
                b.topk_idx.data<int64_t>(base), b.topk_weights.data<float>(base), y.data_ptr(), tokens));
        };
        launch_sync("mega_moe_dispatch_sync", ranks, signals);
        DG_HGGC_RUNTIME_CHECK(hggcEventRecord(streams.fork, caller));
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(streams.copy, streams.fork, 0));
        Copy::launch_on_stream(copy, ca, streams.copy);
        launch_phase(compute, l1, w1, replica1, b.local_data.data(base), b.l2_acts.data(base));
        DG_HGGC_RUNTIME_CHECK(hggcEventRecord(streams.done, streams.copy));
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(caller, streams.done, 0));
        routing.shape_m = b.compact;
        launch_phase(scatter, l2, w2, replica2, b.l2_acts.data(base), b.scatter_storage.data(base));
    });
}
} // namespace deep_gemm::mega
