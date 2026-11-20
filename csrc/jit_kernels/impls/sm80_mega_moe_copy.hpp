#pragma once

#include <hggc_ad.h>
#include <deep_gemm/scheduler/sm80_mega_moe.cuh>
#include "mega_moe_runtime.hpp"

namespace deep_gemm::mega {

DECL_LAZY_HGGC_DRIVER_FUNCTION(hgStreamSetAttributeAD);

// One workspace owns one copy stream and its events. Workspace aliases share
// these resources and must execute serially, including Graph replays.
struct CopyStreamResources {
    int device;
    hggcStream_t copy = nullptr;
    hggcEvent_t fork = nullptr, done = nullptr, launched = nullptr;

    explicit CopyStreamResources(int device_) : device(device_) {
        try {
            DG_HGGC_RUNTIME_CHECK(hggcStreamCreateWithFlags(&copy, hggcStreamNonBlocking));
            // The dedicated stream always runs copy work on the vector-CU pool.
            HGstreamAttrValueAD attr{};
            attr.yieldMultiprocessor = HGAD_YIELD_MULTIPROCESSOR_INTENSIVE;
            DG_HGGC_DRIVER_CHECK(lazy_hgStreamSetAttributeAD(
                copy, HGAD_STREAM_ATTRIBUTE_YIELD_MULTIPROCESSOR, &attr));
            DG_HGGC_RUNTIME_CHECK(hggcEventCreateWithFlags(&fork, hggcEventDisableTiming));
            DG_HGGC_RUNTIME_CHECK(hggcEventCreateWithFlags(&done, hggcEventDisableTiming));
        } catch (...) {
            if (done) hggcEventDestroy(done);
            if (fork) hggcEventDestroy(fork);
            if (copy) hggcStreamDestroy(copy);
            throw;
        }
    }
    CopyStreamResources(const CopyStreamResources&) = delete;
    CopyStreamResources& operator=(const CopyStreamResources&) = delete;
    ~CopyStreamResources() {
        int previous = device;
        if (hggcGetDevice(&previous) != hggcSuccess || hggcSetDevice(device) != hggcSuccess)
            return;
        if (launched) hggcEventDestroy(launched);
        hggcEventDestroy(done);
        hggcEventDestroy(fork);
        hggcStreamDestroy(copy);
        if (previous != device) hggcSetDevice(previous);
    }
};

class SM80MegaMoECopyRuntime final : public LaunchRuntime<SM80MegaMoECopyRuntime> {
public:
    static int shared_bytes(int block_m, int ranks, int groups) {
        const int row_map = ceil_div(4 * ranks + 8 * block_m, 16) * 16;
        return std::max(row_map, 4 * groups * (ranks + 4));
    }

    struct Args {
        int n, k, block_m, block_k, groups, ranks, k_tiles_per_flag, capacity, topk, threads;
        LaunchArgs launch_args;
        sched::Int8MegaMoESchedulerArguments scheduler;
        bool split_sync = false;
        bool bf16_payload = false;
    };
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#define DG_MEGAMOE_SPLIT_SYNC {}
#include <deep_gemm/impls/int8_mega_moe_dispatch.cuh>
using Kernel = deep_gemm::sm80::MegaMoECopyKernel<{},{},{},{},{},{},{},{},{},{},{}>;
static_assert(Kernel::SharedStorageSize == {});
static_assert(sizeof(Kernel::Arguments) == {});
extern "C" __global__ __launch_bounds__({}) void megamoe_int8_dispatch_copy(
        Kernel::Arguments scheduler) {{
    extern __shared__ char smem[];
    Kernel::run(scheduler, smem);
}}
)", static_cast<int>(a.split_sync), a.n, a.k, a.block_m, a.block_k, a.groups, a.ranks,
            a.k_tiles_per_flag, a.capacity, a.topk, a.threads, a.bf16_payload,
            a.launch_args.smem_size, sizeof(a.scheduler), a.threads);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.scheduler));
    }

    static void launch_on_stream(const std::shared_ptr<KernelRuntime>& runtime, const Args& args,
                                 hggcStream_t stream, hggcEvent_t launch_event = nullptr) {
        if (is_compile_only())
            return;
        const auto& launch = args.launch_args;
        auto config = construct_launch_config(runtime->kernel, stream, launch.smem_size,
                                              launch.grid_dim, launch.block_dim);
        LaunchAttrHandle attr{};
        if (launch_event) {
#if HGGCRT_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
            attr.id = hggcLaunchAttributeLaunchCompletionEvent;
            attr.val.launchCompletionEvent.event = launch_event;
#else
            attr.id = HG_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
            attr.value.launchCompletionEvent.event = reinterpret_cast<HGevent>(launch_event);
#endif
            config.attrs = &attr;
            config.numAttrs = 1;
        }
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Launch kernel with {%d, %d} x %d  %d, shared memory: %d bytes, stream: %ld\n",
                   launch.grid_dim.x, launch.grid_dim.y, launch.block_dim.x, launch.block_dim.y,
                   launch.smem_size, (uintptr_t)stream);
        launch_impl(runtime->kernel, config, args);
    }
};

} // namespace deep_gemm::mega
