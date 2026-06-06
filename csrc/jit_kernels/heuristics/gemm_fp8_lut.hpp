#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace deep_gemm_fp8_lut {

// 6-tuple tile: (block_m, block_n, block_k, warp_m, warp_n, num_stages)
using Tile = std::tuple<int, int, int, int, int, int>;

// Entry used to bulk-build MNKDict: (m, n, k, tile)
using MNKEntry = std::tuple<int, int, int, Tile>;

// Dictionary keyed by (n, k); for each key stores a sorted list of m values
// and the corresponding tile. `query` returns the tile of the smallest m
// such that m >= m_query (equivalent to Python's bisect_left lookup).
class MNKDict {
public:
    MNKDict() = default;

    explicit MNKDict(const std::vector<MNKEntry>& data) {
        build_from_data(data);
    }

    std::optional<Tile> query(int m_query, int n, int k) const {
        const auto key = std::make_pair(n, k);
        const auto it_index = index_.find(key);
        if (it_index == index_.end()) {
            return std::nullopt;
        }

        const auto& m_list = it_index->second;
        auto pos = std::lower_bound(m_list.begin(), m_list.end(), m_query);
        if (pos == m_list.end()) {
            return std::nullopt;
        }

        const int best_m = *pos;
        const auto& tile_dict = tiles_.at(key);
        return tile_dict.at(best_m);
    }

private:
    void build_from_data(const std::vector<MNKEntry>& data) {
        // Group entries by (n, k)
        std::map<std::pair<int, int>, std::vector<std::pair<int, Tile>>> groups;
        for (const auto& entry : data) {
            int m, n, k;
            Tile tile;
            std::tie(m, n, k, tile) = entry;
            groups[std::make_pair(n, k)].emplace_back(m, tile);
        }

        // Sort each group by m and split into m_list / tile_map
        for (auto& kv : groups) {
            auto& items = kv.second;
            std::sort(items.begin(), items.end(),
                      [](const std::pair<int, Tile>& a, const std::pair<int, Tile>& b) {
                          return a.first < b.first;
                      });

            std::vector<int> m_list;
            std::map<int, Tile> tile_dict;
            m_list.reserve(items.size());
            for (const auto& item : items) {
                m_list.push_back(item.first);
                tile_dict.emplace(item.first, item.second);
            }
            index_[kv.first] = std::move(m_list);
            tiles_[kv.first] = std::move(tile_dict);
        }
    }

    std::map<std::pair<int, int>, std::vector<int>> index_;        // (n, k) -> sorted list of m
    std::map<std::pair<int, int>, std::map<int, Tile>> tiles_;     // (n, k) -> {m: tile}
};

// Static dense lookup table: keyed by (m_aligned_to_16, n, k).
inline const std::map<std::tuple<int, int, int>, Tile>& get_fp8_dense_list() {
    static const std::map<std::tuple<int, int, int>, Tile> table = {
        {{ 64, 2304, 4096}, { 32,  64, 128, 16, 32, 5}},
        {{ 64, 4096, 2048}, { 64,  64, 128, 32, 32, 4}},
        {{128, 2304, 4096}, { 64,  64, 128, 32, 32, 4}},
        {{128, 4096, 2048}, {128, 128, 128, 32, 32, 3}},
        {{144, 2304, 4096}, { 64,  64, 128, 32, 32, 3}},
        {{144, 4096, 2048}, {192, 128, 128, 48, 32, 5}},
        {{160, 2304, 4096}, { 64,  64, 128, 32, 32, 3}},
        {{160, 4096, 2048}, {192, 128, 128, 48, 32, 5}},
        {{192, 2304, 4096}, {128, 128, 128, 32, 32, 4}},
        {{224, 2304, 4096}, { 64, 256, 128, 32, 32, 3}},
        {{256, 2304, 4096}, { 64, 256, 128, 32, 32, 3}},
    };
    return table;
}

// Static nopad lookup table (built once on first use).
inline const MNKDict& get_fp8_nopad_list() {
    static const MNKDict table(std::vector<MNKEntry>{
        {  4,  768, 4096, { 32,  64, 128, 16, 32, 3}},
        {  1, 2048,  128, { 16, 256, 128, 16, 64, 2}},
        { 12, 2048,  128, { 16, 128, 128, 16, 32, 2}},
        { 32, 2048,  128, { 32, 128, 128, 16, 64, 2}},
        {  1,  256, 2048, { 16, 128, 128, 16, 32, 4}},
        {  3,  256, 2048, { 16,  64, 128, 16, 16, 3}},
        { 13,  256, 2048, { 16,  64, 128, 16, 16, 4}},
        { 29,  256, 2048, { 32, 128, 128, 16, 64, 3}},
        { 32,  256, 2048, { 64, 256, 128, 32, 64, 3}},
    });
    return table;
}

// Look up the best config tile from the static LUTs.
// Returns std::nullopt when no entry matches (equivalent to Python returning None).
inline std::optional<Tile> get_best_configs_from_lut(int m, int n, int k,
                                                     int groups,
                                                     bool is_grouped_contiguous,
                                                     bool is_grouped_masked) {
    if (!is_grouped_contiguous && !is_grouped_masked && groups > 1) {
        return get_fp8_nopad_list().query(m, n, k);
    } else if (!is_grouped_contiguous && !is_grouped_masked && groups == 1) {
        const int m_aligned = ((m + 15) / 16) * 16;
        const auto key = std::make_tuple(m_aligned, n, k);
        const auto& dense = get_fp8_dense_list();
        const auto it = dense.find(key);
        if (it == dense.end()) {
            return std::nullopt;
        }
        return it->second;
    }
    return std::nullopt;
}

} // namespace deep_gemm_fp8_lut
