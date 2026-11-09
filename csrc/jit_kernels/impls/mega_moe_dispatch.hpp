#pragma once

#include "mega_moe_runtime.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega {

class MegaMoERecvStatsRuntime final : public LaunchRuntime<MegaMoERecvStatsRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        const int32_t* valid_rows;
        int32_t* cumulative;
        int groups;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_gemm/impls/mega_moe_transport.cuh>
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mega_moe::detail::mega_moe_update_recv_stats<int32_t>);
}
)";
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.valid_rows, args.cumulative, args.groups));
    }
};

inline void update_recv_stats(uint8_t* base, const layout::RoutedMoEBuffer& buffer, int groups,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    if (!cumulative_local_expert_recv_stats)
        return;
    const layout::Gemm2Metadata meta(base + buffer.gemm2_metadata, groups);
    const MegaMoERecvStatsRuntime::Args args{
        {dim3(ceil_div(groups, 256)), dim3(256), 0}, meta.valid_rows,
        cumulative_local_expert_recv_stats->data_ptr<int32_t>(), groups};
    const auto runtime = compiler->build("mega_moe_update_recv_stats", MegaMoERecvStatsRuntime::generate(args));
    MegaMoERecvStatsRuntime::launch(runtime, args);
}

} // namespace deep_gemm::mega
