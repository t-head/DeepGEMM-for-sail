#pragma once

#include <torch/extension.h>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "mega_moe_runtime.hpp"
#include <deep_gemm/layout/routed_moe.cuh>

namespace deep_gemm {

static_assert(sizeof(layout::SymBuffer<>) == 528, "MegaMoE symmetric buffer ABI changed");

using SourceFields = std::vector<std::pair<std::string, std::string>>;

inline std::string source_float_literal(float value) {
    TORCH_CHECK(std::isfinite(value), "MegaMoE specialization must be finite");
    return fmt::format("{:.9e}f", value);
}

inline void replace_source_field(std::string& code, const std::string& key, const std::string& text) {
    const auto marker = "{" + key + "}";
    for (size_t pos = code.find(marker); pos != std::string::npos; pos = code.find(marker, pos + text.size()))
        code.replace(pos, marker.size(), text);
}

template <typename Runtime>
inline std::string rendered_source(const std::string& name, std::string source, const SourceFields& fields) {
    std::string key = name;
    for (const auto& [field, value] : fields)
        key += ":" + field + "=" + value;
    static std::mutex mutex;
    static std::unordered_map<std::string, std::string> sources;
    const std::lock_guard<std::mutex> guard(mutex);
    const auto found = sources.find(key);
    if (found != sources.end())
        return found->second;
    const std::string marker = "PPU_KERNEL_ENTRY";
    for (size_t pos = source.find(marker); pos != std::string::npos;
         pos = source.find(marker, pos + name.size()))
        source.replace(pos, marker.size(), name);
    for (const auto& [field, value] : fields)
        replace_source_field(source, field, value);
    return sources.emplace(key, std::move(source)).first->second;
}

template <typename Runtime, typename Config>
inline std::string rendered_source(const std::string& name, std::string source, const Config& config) {
    return rendered_source<Runtime>(name, std::move(source), source_fields(config));
}

inline void* device_pointer(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "MegaMoE expects CUDA device tensors");
    return tensor.data_ptr();
}

template <typename T> inline T* device_pointer(const torch::Tensor& tensor) {
    return reinterpret_cast<T*>(device_pointer(tensor));
}

inline layout::SymBuffer<> make_sym_buffer(const std::vector<int64_t>& pointers, int rank_idx, int num_ranks) {
    TORCH_CHECK(int(pointers.size()) == num_ranks && num_ranks > 0 && num_ranks <= layout::kNumMaxRanks &&
                rank_idx >= 0 && rank_idx < num_ranks, "invalid symmetric pointer table or rank");
    for (const auto pointer : pointers)
        TORCH_CHECK(pointer != 0 && pointer % 128 == 0, "symmetric buffer bases must be 128-byte aligned");
    return layout::SymBuffer<>(pointers, rank_idx);
}

} // namespace deep_gemm
