#pragma once

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <deep_gemm/common/types.cuh>
#include "../../jit/device_runtime.hpp"

namespace deep_gemm_lut {

using deep_gemm::GemmType;

using Tile = std::tuple<int, int, int, int, int, int>;
enum class RuntimeLutKind { FP8Blockwise, FP8Channelwise, INT8, A8W4INT8, BF16, FP4 };
using RuntimeKey = std::tuple<int, int, int, int>; // (n, k, num_groups, m); grouped GEMMs use lower_bound on m
using RuntimeTable = std::map<RuntimeKey, Tile>;
using TableId = std::pair<RuntimeLutKind, GemmType>;

inline const char* get_runtime_lut_kind_name(RuntimeLutKind kind) {
    switch (kind) {
        case RuntimeLutKind::FP8Blockwise: return "fp8_blockwise";
        case RuntimeLutKind::FP8Channelwise: return "fp8_channelwise";
        case RuntimeLutKind::INT8: return "int8";
        case RuntimeLutKind::A8W4INT8: return "a8w4_int8";
        case RuntimeLutKind::BF16: return "bf16";
        case RuntimeLutKind::FP4: return "fp4";
    }
    throw std::runtime_error("Unsupported runtime LUT kind");
}

inline const char* get_gemm_type_name(GemmType gemm_type) {
    switch (gemm_type) {
        case GemmType::Normal: return "densegemm";
        case GemmType::MGroupedContiguous: return "groupedcontiguous";
        case GemmType::MGroupedMasked: return "groupedmasked";
        case GemmType::MGroupedNoPad: return "groupednopad";
        case GemmType::MGroupedFused: return "groupedfused";
        case GemmType::Batched: return "batchgemm";
    }
    throw std::runtime_error("Unsupported GEMM type");
}

inline std::string get_runtime_lut_filename(const std::string& device_name,
                                            RuntimeLutKind kind, GemmType gemm_type) {
    return device_name + "_" + get_runtime_lut_kind_name(kind) + "_" +
           get_gemm_type_name(gemm_type) + ".lut";
}

struct RuntimeTableState {
    std::once_flag once;
    RuntimeTable table; // Immutable after call_once completes.
};

struct RuntimeTables {
    std::mutex mutex;
    std::map<TableId, RuntimeTableState> tables;
};

inline RuntimeTables& get_runtime_tables() {
    static RuntimeTables state;
    return state;
}

inline RuntimeTableState& get_runtime_table_state(RuntimeLutKind kind, GemmType gemm_type) {
    // Cache stable map-node addresses; only the first access per thread needs the registry lock.
    thread_local std::map<TableId, RuntimeTableState*> cache;
    const TableId id{kind, gemm_type};
    const auto cached = cache.find(id);
    if (cached != cache.end()) return *cached->second;
    auto& state = get_runtime_tables();
    RuntimeTableState* table;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        table = &state.tables[id];
    }
    cache.emplace(id, table);
    return *table;
}

inline bool supports_runtime_lut(GemmType gemm_type) {
    return gemm_type == GemmType::Normal ||
           gemm_type == GemmType::MGroupedNoPad ||
           gemm_type == GemmType::MGroupedMasked;
}

inline RuntimeTable load_runtime_lut(const std::string& path, GemmType gemm_type) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open runtime LUT file: " + path);
    }

    RuntimeTable parsed_table;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }

        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }
        for (char& c : line) {
            if (c == '{' || c == '}' || c == ',') {
                c = ' ';
            }
        }
        const bool grouped = gemm_type != GemmType::Normal;
        const size_t expected = grouped ? 10 : 9;

        std::istringstream input(line);
        std::vector<int> values;
        values.reserve(expected);
        for (std::string token; input >> token;) {
            int value;
            const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
            if (error != std::errc{} || end != token.data() + token.size()) {
                throw std::runtime_error("Invalid runtime LUT entry at " + path + ":" +
                                         std::to_string(line_number));
            }
            values.push_back(value);
        }

        if (values.size() != expected) {
            throw std::runtime_error("Invalid runtime LUT entry at " + path + ":" +
                                     std::to_string(line_number) + ": expected " +
                                     std::to_string(expected) + " integers, got " +
                                     std::to_string(values.size()));
        }

        const int num_groups = grouped ? values[0] : 1;
        const int offset = grouped ? 1 : 0;
        const int m = values[offset], n = values[offset + 1], k = values[offset + 2];
        const int block_m = values[offset + 3], block_n = values[offset + 4];
        const int block_k = values[offset + 5], warp_m = values[offset + 6];
        const int warp_n = values[offset + 7], stages = values[offset + 8];
        if (std::any_of(values.begin(), values.end(), [](int value) { return value <= 0; }) ||
            (gemm_type == GemmType::Normal && m % 16 != 0) ||
            warp_m > block_m || warp_n > block_n ||
            block_m % warp_m != 0 || block_n % warp_n != 0) {
            throw std::runtime_error("Invalid runtime LUT dimensions at " + path + ":" +
                                     std::to_string(line_number));
        }
        const auto key = std::make_tuple(n, k, num_groups, m);
        const auto config = std::make_tuple(block_m, block_n, block_k, warp_m, warp_n, stages);
        const bool inserted = parsed_table.emplace(key, config).second;
        if (!inserted) {
            std::cerr << "[DeepGEMM] duplicate runtime LUT key at " << path << ':' << line_number
                      << ", keeping the first entry and ignoring this one\n";
        }
    }
    return parsed_table;
}

inline void ensure_runtime_lut_loaded(RuntimeTableState& state, RuntimeLutKind kind,
                                      GemmType gemm_type, const char* path);

inline std::optional<Tile> get_runtime_config(int m, int n, int k, int num_groups,
                                              GemmType gemm_type, RuntimeLutKind kind) {
    if (!supports_runtime_lut(gemm_type)) return std::nullopt;
    static const std::string path = [] {
        const char* value = std::getenv("SAIL_DEEPGEMM_LUT_PATH");
        return value ? std::string(value) : std::string{};
    }();
    if (path.empty()) return std::nullopt;
    auto& state = get_runtime_table_state(kind, gemm_type);
    ensure_runtime_lut_loaded(state, kind, gemm_type, path.c_str());
    const int lookup_m = gemm_type == GemmType::Normal ? ((m + 15) / 16) * 16 : m;
    const auto key = std::make_tuple(n, k, gemm_type == GemmType::Normal ? 1 : num_groups, lookup_m);
    const auto& table = state.table;
    const auto it = gemm_type == GemmType::Normal ? table.find(key) : table.lower_bound(key);
    return it == table.end() || std::get<0>(it->first) != n || std::get<1>(it->first) != k ||
           std::get<2>(it->first) != std::get<2>(key)
               ? std::nullopt : std::optional<Tile>(it->second);
}

inline std::filesystem::path resolve_runtime_lut_path(
        const std::string& configured_path,
        RuntimeLutKind kind,
        GemmType gemm_type) {
    const std::filesystem::path path(configured_path);
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error("LUT path must be a directory: " + configured_path);
    }

    static const std::string device_name(deep_gemm::device_runtime->get_prop()->name);
    return path / get_runtime_lut_filename(device_name, kind, gemm_type);
}

inline void ensure_runtime_lut_loaded(RuntimeTableState& state, RuntimeLutKind kind,
                                      GemmType gemm_type, const char* path) {
    std::call_once(state.once, [&state, kind, gemm_type, path] {
        try {
            const auto resolved_path = resolve_runtime_lut_path(path, kind, gemm_type);
            if (!std::filesystem::is_regular_file(resolved_path)) {
                std::cerr << "[DeepGEMM] runtime LUT file not found: " << resolved_path
                          << "; falling back to built-in LUT\n";
                return;
            }
            state.table = load_runtime_lut(resolved_path.string(), gemm_type);
        } catch (const std::exception& e) {
            std::cerr << "[DeepGEMM] failed to load runtime LUT " << path
                      << ": " << e.what() << "; falling back to built-in LUT\n";
        }
    });
}

} // namespace deep_gemm_lut
