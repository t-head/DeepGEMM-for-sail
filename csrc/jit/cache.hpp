#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_gemm {

class KernelRuntimeCache {
    struct CacheEntry {
        std::mutex mutex;
        std::shared_ptr<KernelRuntime> runtime;
    };
    std::mutex mutex;
    std::unordered_map<HGcontext, std::unordered_map<std::string, CacheEntry>> cache;

public:
    // TODO: consider cache capacity
    KernelRuntimeCache() = default;

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        HGcontext context;
        DG_HGGC_DRIVER_CHECK(lazy_hgCtxGetCurrent(&context));
        CacheEntry* entry;
        {
            std::lock_guard<std::mutex> lock(mutex);
            entry = &cache[context][dir_path.native()];
            if (entry->runtime != nullptr)
                return entry->runtime;
        }
        // Entries are never erased; rehashing preserves their addresses.
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->runtime != nullptr)
            return entry->runtime;

        if (not KernelRuntime::check_validity(dir_path))
            return nullptr;
        auto runtime = std::make_shared<KernelRuntime>(dir_path);
        {
            std::lock_guard<std::mutex> cache_lock(mutex);
            entry->runtime = runtime;
        }
        return runtime;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_gemm
