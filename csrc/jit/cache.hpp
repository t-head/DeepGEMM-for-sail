#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_gemm {

class KernelRuntimeCache {
    std::unordered_map<HGcontext, std::unordered_map<std::string, std::shared_ptr<KernelRuntime>>> cache;

public:
    // TODO: consider cache capacity
    KernelRuntimeCache() = default;

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        HGcontext context;
        DG_HGGC_DRIVER_CHECK(lazy_hgCtxGetCurrent(&context));
        auto& context_cache = cache[context];
        // Hit the runtime cache
        if (const auto iterator = context_cache.find(dir_path); iterator != context_cache.end())
            return iterator->second;

        if (KernelRuntime::check_validity(dir_path))
            return context_cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);
        return nullptr;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_gemm
