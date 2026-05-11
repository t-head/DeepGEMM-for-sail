#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <sstream>
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"
#include "../../utils/utils.hpp"

inline int align(int x, int y) {
    return ceil_div(x, y) * y;
}

class MatmulHeuristicsTile {
public:
    static const bool MAX_TOPK = false;
    static const bool MIN_TOPK = true;

    struct HWMetric {
        int cu_count;
        int mmad_cal_bpp;
        int share_mem_size;
        int max_warps_per_cu;
        int min_stages;
        int min_warp_per_block;
        int max_warp_per_block;
        int max_warp_per_cu;
        int max_register_per_block;
        int max_register_per_cu;
        int max_register_per_thread;
        int max_register_per_warp;
        int max_block_per_cu;
        int semem_alloc_alignment;
        int reg_alignment_size;
        int reg_unit_size;
        int cu_per_ce;
        int l2_size;
        int llc_size;
        int max_warp_tile;
        int min_warp_tile;
        int tc_inst_fp16_m;
        int tc_inst_fp16_n;
        int tc_inst_fp16_k;
        double max_reg_utils;

        HWMetric() : cu_count(::deep_gemm::get_num_sms()) {
            mmad_cal_bpp = 4;
            share_mem_size = 256 * 1024;
            max_warps_per_cu = 64;
            min_stages = 2;
            min_warp_per_block = 4;
            max_warp_per_block = 32;
            max_warp_per_cu = 64;
            max_register_per_block = 131072;
            max_register_per_cu = 131072;
            max_register_per_thread = 256;
            max_register_per_warp = 256 * 32;
            max_block_per_cu = 64;
            semem_alloc_alignment = 8;
            reg_alignment_size = 64;
            reg_unit_size = 4;
            cu_per_ce = 4;
            l2_size = 1024 * 1024;
            llc_size = 1024 * 1024 * 64;
            max_warp_tile = 128;
            min_warp_tile = 16;
            tc_inst_fp16_m = 16;
            tc_inst_fp16_n = 16;
            tc_inst_fp16_k = 16;
            max_reg_utils = 0.99;
        }
    };

    enum IDX {
        M,
        N,
        K,
        BM,
        BN,
        BK,
        WM,
        WN,
        WK,
        STAGES,
        CU_COUNT,
        BLOCK_RLOADSIZE,
        WAVE,
        BLOCK_UTILS,
        LAST_WAVE_BLOCK,
        OCC,
        WARPS_PER_BLOCK,
        WARPS_PER_CU,
        REGS_PER_BLOCK,
        REGS_RLOADSIZE_PER_CU,
        LLC_UTILS,
        SHAREMEM_UTILS,
        L2_UTILS,
        REG_UTILS,
        OCC_WAVE
    };

private:
    std::vector<int> _shape;
    int _bpp;
    int reg_db;
    std::vector<std::vector<int>> _config_tile;
    std::unordered_map<int, bool> _filter_metric;
    HWMetric _hw_metric;

public:
    MatmulHeuristicsTile(const std::vector<int>& shape, int bpp, const std::vector<std::vector<int>>& config_tile = {})
        : _shape(shape), _bpp(bpp), reg_db(2), _config_tile(config_tile) {
        _filter_metric[BLOCK_RLOADSIZE] = MIN_TOPK;
        _filter_metric[BLOCK_UTILS] = MAX_TOPK;
        _filter_metric[WAVE] = MIN_TOPK;
        _filter_metric[LAST_WAVE_BLOCK] = MAX_TOPK;
        _filter_metric[WARPS_PER_CU] = MIN_TOPK;
        _filter_metric[L2_UTILS] = MAX_TOPK;
        _filter_metric[SHAREMEM_UTILS] = MAX_TOPK;
        _filter_metric[REGS_RLOADSIZE_PER_CU] = MIN_TOPK;
        _filter_metric[REG_UTILS] = MAX_TOPK;
    }

    std::vector<std::vector<int>> _add_candidate_tile(const std::vector<std::vector<int>>& candidate_tile,
                                                      const std::vector<int>& new_tile, int compare_index = -1,
                                                      int top_k = 10, bool ascend_order = true) {
        // merge duplicate item
        std::unordered_map<int, std::vector<std::vector<int>>> tile_map;
        std::vector<std::vector<int>> tile_result;

        for (const auto& item : candidate_tile) {
            int key = item[compare_index];
            if (tile_map.find(key) != tile_map.end()) {
                tile_map[key].push_back(item);
            } else {
                tile_map[key] = {item};
            }
        }

        if (candidate_tile.empty()) {
            return {new_tile};
        }

        if (tile_map.find(new_tile[compare_index]) != tile_map.end()) {
            std::vector<std::vector<int>> candidate_tile_copy = candidate_tile;
            auto it = std::find(candidate_tile_copy.begin(), candidate_tile_copy.end(),
                                tile_map[new_tile[compare_index]].back());
            if (it != candidate_tile_copy.end()) {
                candidate_tile_copy.insert(it + 1, new_tile);
                return candidate_tile_copy;
            }
        }

        int ascend_flag = ascend_order ? 1 : -1;
        auto cmp = [ascend_flag](const std::pair<int, std::vector<std::vector<int>>>& a,
                                 const std::pair<int, std::vector<std::vector<int>>>& b) {
            return a.first > b.first;
        };

        std::priority_queue<std::pair<int, std::vector<std::vector<int>>>,
                            std::vector<std::pair<int, std::vector<std::vector<int>>>>, decltype(cmp)>
            heap(cmp);

        for (const auto& pair : tile_map) {
            heap.push({pair.first * ascend_flag, pair.second});
        }

        if (tile_map.size() < static_cast<size_t>(top_k)) {
            heap.push({new_tile[compare_index] * ascend_flag, {new_tile}});
        } else if (new_tile[compare_index] * ascend_flag < heap.top().first) {
            heap.pop();
            heap.push({new_tile[compare_index] * ascend_flag, {new_tile}});
        }

        // sort tile_list according to compare_index
        std::vector<std::pair<int, std::vector<std::vector<int>>>> sorted_items;
        while (!heap.empty()) {
            sorted_items.push_back(heap.top());
            heap.pop();
        }

        std::sort(sorted_items.begin(), sorted_items.end());

        for (const auto& pair : sorted_items) {
            tile_result.insert(tile_result.end(), pair.second.begin(), pair.second.end());
        }

        return tile_result;
    }

    std::vector<std::vector<int>> _get_collapsed_topk(const std::vector<std::vector<int>>& candidate_tile,
                                                      int compare_index = -1, int top_k = 10,
                                                      bool ascend_order = true) {
        // merge duplicate item
        std::unordered_map<int, std::vector<std::vector<int>>> tile_map;
        std::vector<std::vector<int>> tile_result;
        int ascend_flag = ascend_order ? 1 : -1;

        for (const auto& item : candidate_tile) {
            int key = item[compare_index] * ascend_flag;
            if (tile_map.find(key) != tile_map.end()) {
                tile_map[key].push_back(item);
            } else {
                tile_map[key] = {item};
            }
        }

        if (candidate_tile.empty()) {
            return {};
        }

        std::vector<std::pair<int, std::vector<std::vector<int>>>> sorted_list;
        for (const auto& pair : tile_map) {
            sorted_list.push_back({pair.first, pair.second});
        }

        std::sort(sorted_list.begin(), sorted_list.end());

        int cur_num = 0;
        for (const auto& pair : sorted_list) {
            cur_num++;
            tile_result.insert(tile_result.end(), pair.second.begin(), pair.second.end());
            if (cur_num >= top_k) {
                break;
            }
        }

        return tile_result;
    }

    int _cal_smem_size(int bm, int bn, int bk, int stages) {
        int sharemem_utils = std::max((bm * bk + bn * bk) * _bpp * stages, bm * bn * _hw_metric.mmad_cal_bpp);
        return sharemem_utils;
    }

    int _get_max_stage(int bm, int bn, int bk) {
        int bk_mem = (bm * bk + bn * bk) * _bpp;
        return _hw_metric.share_mem_size / bk_mem;
    }

    std::vector<int> _get_candidate_factor(int m) {
        if (m >= 4096) {
            return {512, 256, 128};
        } else {
            return {};
        }
    }

    std::vector<int> _get_candidate_factor_ld(int m) {
        if (m >= 4096) {
            return {64, 128};
        } else {
            return {};
        }
    }

    std::vector<int> _get_candidate_stages(int m, int n, int k) {
        if (m >= 4096 && n >= 4096) {
            return {3, 2};
        } else {
            return {};
        }
    }

    std::vector<int> _get_candidate_warp_factor(int m) {
        std::vector<int> candidate_factors = {m / 2, m / 4, m / 8, m / 16, m};
        std::vector<int> valid_factors;

        for (int x = _hw_metric.min_warp_tile; x <= _hw_metric.max_warp_tile; x += 16) {
            if (x <= m) {
                valid_factors.push_back(x);
            }
        }

        int n = _shape[1];
        if (m >= 4096 && n >= 4096) {
            std::vector<int> filtered_factors;
            for (int factor : candidate_factors) {
                if (factor > _hw_metric.min_warp_tile) {
                    filtered_factors.push_back(factor);
                }
            }
            candidate_factors = filtered_factors;
        }

        // intersection of two lists
        std::vector<int> result;
        for (int factor : candidate_factors) {
            if (std::find(valid_factors.begin(), valid_factors.end(), factor) != valid_factors.end()) {
                result.push_back(factor);
            }
        }

        return result;
    }

    int _get_register_occupies(int bm, int bn, int wm, int wn, int wk) {
        int reg_u_size = _hw_metric.reg_unit_size;
        int reg_align_size = _hw_metric.reg_alignment_size;
        int warp_num = (bm / wm) * (bn / wn);

        return (align(wm * _hw_metric.tc_inst_fp16_k * _bpp / reg_u_size, reg_align_size) +
                align(wn * _hw_metric.tc_inst_fp16_k * _bpp / reg_u_size, reg_align_size)) *
                   reg_db * warp_num +
               align(bm * bn * _hw_metric.mmad_cal_bpp / reg_u_size, reg_align_size);
    }

    double _get_l2_usage(int bm, int bn, int bk, int stages, int occ) {
        return ((bm * bk + bn * bk) * _bpp * stages * occ * _hw_metric.cu_per_ce) /
               static_cast<double>(_hw_metric.l2_size);
    }

    double _get_sharemem_usage(int bm, int bn, int bk, int stages, int occ) {
        int mem_use = _cal_smem_size(bm, bn, bk, stages);
        return (mem_use * occ) / static_cast<double>(_hw_metric.share_mem_size);
    }

    double _get_llc_usage(int bm, int bn, int bk, int stages, int occ) {
        return (((bm * bk + bn * bk) * _bpp * stages + bm * bn * _bpp) * occ * _hw_metric.cu_count) /
               static_cast<double>(_hw_metric.llc_size);
    }

    int _get_occ(int bm, int bn, int bk, int wm, int wn, int num_stages) {
        int mem_size = _cal_smem_size(bm, bn, bk, num_stages);
        int reg_utils = _get_register_occupies(bm, bn, wm, wn, bk);
        return std::min(_hw_metric.share_mem_size / mem_size, _hw_metric.max_register_per_cu / reg_utils);
    }

    double _get_reg_usage(int bm, int bn, int bk, int wm, int wn, int occ) {
        int reg_utils = _get_register_occupies(bm, bn, wm, wn, bk) * occ;
        return reg_utils / static_cast<double>(_hw_metric.max_register_per_cu);
    }

    std::vector<std::vector<int>> _get_mem_usage(const std::vector<std::vector<int>>& tile_list) {
        std::vector<std::vector<int>> new_tile_candidate;

        for (auto tile : tile_list) {
            if (tile.size() <= static_cast<size_t>(REG_UTILS)) {
                tile.resize(REG_UTILS + 1, 0);
            }

            int bm = tile[BM], bn = tile[BN], bk = tile[BK], wm = tile[WM], wn = tile[WN];
            int stages = tile[STAGES], occ = tile[OCC], wave = tile[WAVE];

            double reg_utils = _get_reg_usage(bm, bn, bk, wm, wn, occ);
            if (reg_utils >= _hw_metric.max_reg_utils) {
                continue;
            }

            double l2_utils = _get_l2_usage(bm, bn, bk, stages, occ);
            double sharemem_utils = _get_sharemem_usage(bm, bn, bk, stages, occ);
            double llc_utils = _get_llc_usage(bm, bn, bk, stages, occ);

            tile[OCC_WAVE] = static_cast<int>(wave / static_cast<double>(occ));
            int cu_count = _get_cu_count();

            tile[LLC_UTILS] = static_cast<int>(llc_utils * 1000); // Scale for integer storage
            tile[SHAREMEM_UTILS] = static_cast<int>(sharemem_utils * 1000);
            tile[L2_UTILS] = static_cast<int>(l2_utils * 1000);
            tile[REG_UTILS] = static_cast<int>(reg_utils * 1000);
            tile[CU_COUNT] = cu_count;

            new_tile_candidate.push_back(tile);
        }

        return new_tile_candidate;
    }

    int _get_block_repeat_loadsize(int m, int n, int k, int bm, int bn) {
        int Asize = m * k * _bpp;
        int Bsize = n * k * _bpp;
        return Asize * ceil_div(n, bn) + Bsize * ceil_div(m, bm);
    }

    int _get_num_waves(int m, int n, int bm, int bn) {
        return ceil_div(ceil_div(m, bm) * ceil_div(n, bn), _hw_metric.cu_count);
    }

    double _get_block_utils(int m, int n, int bm, int bn) {
        return ((m / static_cast<double>(bm)) * (n / static_cast<double>(bn))) / (ceil_div(m, bm) * ceil_div(n, bn));
    }

    int _get_last_wave_util(int m, int n, int bm, int bn) {
        auto fix_wave_saturate = [this](int x) -> int {
            return x == 0 ? this->_hw_metric.cu_count : x;
        };
        return fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn)) % _hw_metric.cu_count);
    }

    std::vector<std::vector<int>> _get_block_tile(int m, int n, int k, const std::vector<std::vector<int>>& tile_list,
                                                  int top_k = 50) {
        std::vector<std::vector<int>> new_tile_candidate;

        for (auto tile : tile_list) {
            if (tile.size() <= static_cast<size_t>(LAST_WAVE_BLOCK)) {
                tile.resize(LAST_WAVE_BLOCK + 1, 0);
            }

            int bm = tile[BM], bn = tile[BN], bk = tile[BK];
            if (_cal_smem_size(bm, bn, bk, _hw_metric.min_stages) > _hw_metric.share_mem_size || (k % bk != 0)) {
                continue;
            }

            int reload_size = _get_block_repeat_loadsize(m, n, k, bm, bn);
            int num_waves = _get_num_waves(m, n, bm, bn);
            double block_util = _get_block_utils(m, n, bm, bn);
            int last_waves = _get_last_wave_util(m, n, bm, bn);

            tile[BLOCK_RLOADSIZE] = reload_size;
            tile[WAVE] = num_waves;
            tile[BLOCK_UTILS] = static_cast<int>(block_util * 1000); // Scale for integer storage
            tile[LAST_WAVE_BLOCK] = last_waves;

            new_tile_candidate = _add_candidate_tile(new_tile_candidate, tile, BLOCK_RLOADSIZE, top_k);
        }

        return new_tile_candidate;
    }

    std::vector<std::vector<int>> _get_stage_tile(const std::vector<int>& stage_candidates,
                                                  const std::vector<std::vector<int>>& tile_list, int topk = 5,
                                                  bool ascend_order = false) {
        std::vector<std::vector<int>> new_tile_candidate;

        for (const auto& tile : tile_list) {
            int bm = tile[BM], bn = tile[BN], bk = tile[BK];
            int max_stage = _get_max_stage(bm, bn, bk);

            for (int num_stages : stage_candidates) {
                if (num_stages > max_stage) {
                    continue;
                }

                std::vector<int> new_tile = tile;
                if (new_tile.size() <= static_cast<size_t>(STAGES)) {
                    new_tile.resize(STAGES + 1, 0);
                }
                new_tile[STAGES] = num_stages;
                new_tile_candidate = _add_candidate_tile(new_tile_candidate, new_tile, STAGES, topk, ascend_order);
            }
        }

        return new_tile_candidate;
    }

    int _get_warps_per_block(int bm, int bn, int wm, int wn) {
        return ((bm / wm) * (bn / wn));
    }

    int _get_warps_per_cu(int bm, int bn, int bk, int wm, int wn, int stages) {
        int warps_per_block = _get_warps_per_block(bm, bn, wm, wn);
        return warps_per_block * _get_occ(bm, bn, bk, wm, wn, stages);
    }

    int _get_cu_count() {
        return _hw_metric.cu_count;
    }

    double _get_regs_rloadsize_per_cu(int bm, int bn, int bk, int wm, int wn) {
        int m = _shape[0], n = _shape[1];
        double rload_data_per_block =
            (wm * bk + wn * bk) * _bpp * (bm / wm) * (bn / wn) / static_cast<double>((bm * bk + bn * bk) * _bpp);
        double block_per_cu = (m / static_cast<double>(bm)) * (n / static_cast<double>(bn)) / _hw_metric.cu_count;
        return rload_data_per_block * block_per_cu;
    }

    std::vector<std::vector<int>> _get_warp_tile(const std::vector<std::vector<int>>& tile_list, int topk = 50) {
        std::vector<std::vector<int>> new_tile_candidate;

        for (auto tile : tile_list) {
            int bm = tile[BM], bn = tile[BN], bk = tile[BK], stages = tile[STAGES];
            std::vector<int> wm_factors = _get_candidate_warp_factor(bm);
            std::vector<int> wn_factors = _get_candidate_warp_factor(bn);

            for (int wm : wm_factors) {
                for (int wn : wn_factors) {
                    int warps_per_block = _get_warps_per_block(bm, bn, wm, wn);
                    int warps_per_cu = _get_warps_per_cu(bm, bn, bk, wm, wn, stages);
                    int regs_per_block = _get_register_occupies(bm, bn, wm, wn, bk);
                    int available_regs_per_block = _hw_metric.max_register_per_warp * warps_per_block;
                    double regs_rload = _get_regs_rloadsize_per_cu(bm, bn, bk, wm, wn);

                    if (!(_hw_metric.min_warp_per_block <= warps_per_block &&
                          warps_per_block <= _hw_metric.max_warp_per_block)) {
                        continue;
                    }

                    if (warps_per_cu > _hw_metric.max_warps_per_cu ||
                        regs_per_block > _hw_metric.max_register_per_block ||
                        regs_per_block > available_regs_per_block) {
                        continue;
                    }

                    std::vector<int> new_tile = tile;
                    if (new_tile.size() <= static_cast<size_t>(OCC_WAVE)) {
                        new_tile.resize(OCC_WAVE + 1, 0);
                    }

                    int occ = _get_occ(bm, bn, bk, wm, wn, stages);
                    new_tile[WARPS_PER_BLOCK] = warps_per_block;
                    new_tile[WARPS_PER_CU] = warps_per_cu;
                    new_tile[REGS_PER_BLOCK] = regs_per_block;
                    new_tile[REGS_RLOADSIZE_PER_CU] = static_cast<int>(regs_rload * 1000); // Scale for integer storage
                    new_tile[OCC] = occ;
                    new_tile[WM] = wm;
                    new_tile[WN] = wn;
                    new_tile[WK] = bk;

                    new_tile_candidate = _add_candidate_tile(new_tile_candidate, new_tile, REGS_RLOADSIZE_PER_CU, topk);
                }
            }
        }

        return new_tile_candidate;
    }

    std::vector<std::vector<int>> _get_wave_from_occ(const std::vector<std::vector<int>>& tile_list, int topk = 10) {
        std::vector<std::vector<int>> new_tile_candidate;

        for (auto tile : tile_list) {
            if (tile.size() <= static_cast<size_t>(OCC_WAVE)) {
                tile.resize(OCC_WAVE + 1, 0);
            }

            int wave = tile[WAVE], occ = tile[OCC];
            tile[OCC_WAVE] = static_cast<int>(wave / static_cast<double>(occ));
            new_tile_candidate = _add_candidate_tile(new_tile_candidate, tile, OCC_WAVE, topk);
        }

        return new_tile_candidate;
    }

    std::vector<std::vector<int>> _get_candidate_block_tile(int m, int n, int k) {
        std::vector<int> candidate_m = _get_candidate_factor(m);
        std::vector<int> candidate_n = _get_candidate_factor(n);
        std::vector<int> candidate_k = _get_candidate_factor_ld(k);

        std::vector<std::vector<int>> candidate_tile_list;
        std::vector<std::vector<int>> new_tile_candidate;

        for (int x : candidate_m) {
            for (int y : candidate_n) {
                for (int z : candidate_k) {
                    new_tile_candidate.push_back({x, y, z});
                }
            }
        }

        for (const auto& tile : new_tile_candidate) {
            std::vector<int> new_tile(IDX::OCC_WAVE + 1, 0);
            new_tile[M] = m;
            new_tile[N] = n;
            new_tile[K] = k;
            new_tile[BM] = tile[0];
            new_tile[BN] = tile[1];
            new_tile[BK] = tile[2];
            candidate_tile_list.push_back(new_tile);
        }

        return candidate_tile_list;
    }

    std::vector<std::vector<int>> _add_config_tile(std::vector<std::vector<int>> candidate_tile_list) {
        int m = _shape[0], n = _shape[1], k = _shape[2];

        // Create set of existing tiles
        std::set<std::vector<int>> cand_tile_set;
        for (const auto& item : candidate_tile_list) {
            std::vector<int> key = {item[BM], item[BN], item[BK], item[WM], item[WN], item[WK], item[STAGES]};
            cand_tile_set.insert(key);
        }

        for (const auto& tile : _config_tile) {
            std::vector<int> key = {tile[0], tile[1], tile[2], tile[3], tile[4], tile[5], tile[6]};
            if (cand_tile_set.find(key) != cand_tile_set.end()) {
                continue;
            }

            int bm = tile[0], bn = tile[1], bk = tile[2], wm = tile[3], wn = tile[4], wk = tile[5], stages = tile[6];

            std::vector<int> new_tile(IDX::OCC_WAVE + 1, 0);
            new_tile[M] = m;
            new_tile[N] = n;
            new_tile[K] = k;
            new_tile[BM] = bm;
            new_tile[BN] = bn;
            new_tile[BK] = bk;
            new_tile[WM] = wm;
            new_tile[WN] = wn;
            new_tile[WK] = wk;
            new_tile[STAGES] = stages;
            new_tile[BLOCK_RLOADSIZE] = _get_block_repeat_loadsize(m, n, k, bm, bn);
            new_tile[WAVE] = _get_num_waves(m, n, bm, bn);
            new_tile[BLOCK_UTILS] = static_cast<int>(_get_block_utils(m, n, bm, bn) * 1000);
            new_tile[LAST_WAVE_BLOCK] = _get_last_wave_util(m, n, bm, bn);

            int occ = _get_occ(bm, bn, bk, wm, wn, stages);
            new_tile[OCC] = occ;
            new_tile[WARPS_PER_BLOCK] = _get_warps_per_block(bm, bn, wm, wn);
            new_tile[WARPS_PER_CU] = _get_warps_per_cu(bm, bn, bk, wm, wn, stages);
            new_tile[REGS_PER_BLOCK] = _get_register_occupies(bm, bn, wm, wn, bk);
            new_tile[REGS_RLOADSIZE_PER_CU] = static_cast<int>(_get_regs_rloadsize_per_cu(bm, bn, bk, wm, wn) * 1000);
            new_tile[LLC_UTILS] = static_cast<int>(_get_llc_usage(bm, bn, bk, stages, occ) * 1000);
            new_tile[SHAREMEM_UTILS] = static_cast<int>(_get_sharemem_usage(bm, bn, bk, stages, occ) * 1000);
            new_tile[L2_UTILS] = static_cast<int>(_get_l2_usage(bm, bn, bk, stages, occ) * 1000);
            new_tile[REG_UTILS] = static_cast<int>(_get_reg_usage(bm, bn, bk, wm, wn, occ) * 1000);
            new_tile[CU_COUNT] = _get_cu_count();
            new_tile[OCC_WAVE] = static_cast<int>(new_tile[WAVE] / static_cast<double>(new_tile[OCC]));

            candidate_tile_list.push_back(new_tile);
        }

        return candidate_tile_list;
    }

    bool _check_poor_candidate(const std::vector<int>& tile) {
        // 1.1 remove warps_per_cu <= 4
        if (tile[WARPS_PER_CU] <= 4) {
            return true;
        }
        // 1.2 register overflow
        std::vector<std::vector<int>> reg_overflow_tile = {{256, 256, 128, 128, 32, 128},
                                                           {256, 256, 128, 32, 128, 128}};

        std::vector<int> tile_slice = {tile[BM], tile[BN], tile[BK], tile[WM], tile[WN], tile[WK]};
        for (const auto& overflow_tile : reg_overflow_tile) {
            if (tile_slice == overflow_tile) {
                return true;
            }
        }
        return false;
    }

    std::vector<std::vector<int>> _multi_phase_topk(std::vector<std::vector<int>> new_tile_candidate,
                                                    int num_candidate) {
        if (new_tile_candidate.size() <= static_cast<size_t>(num_candidate)) {
            return new_tile_candidate;
        }

        auto tile_list = new_tile_candidate;
        int phase_topk = std::max(static_cast<int>(tile_list.size()) / 2, num_candidate);

        while (tile_list.size() > static_cast<size_t>(num_candidate)) {
            for (const auto& pair : _filter_metric) {
                int mc = pair.first;
                bool ascend = pair.second;
                tile_list = _get_collapsed_topk(tile_list, mc, phase_topk, ascend);
                if (tile_list.size() < static_cast<size_t>(num_candidate)) {
                    break;
                }
            }

            if (phase_topk == 1) {
                break;
            }

            if (phase_topk <= 5) {
                phase_topk = std::max(phase_topk - 1, 1);
            } else {
                phase_topk = std::max(phase_topk / 2, 1);
            }
        }

        if (num_candidate == 1 && tile_list.size() > 1) {
            return {tile_list[0]};
        } else {
            return tile_list;
        }
    }

    std::vector<std::vector<int>> _get_knowledge_base() {
        std::vector<std::vector<int>> shape_knowledge = {
            {4096, 4608, 7168}, {4096, 36864, 7168}, {4096, 7168, 16384}, {4096, 7168, 18432}, {4096, 4096, 7168}};

        for (const auto& shape : shape_knowledge) {
            if (_shape == shape) {
                std::vector<int> new_tile(IDX::OCC_WAVE + 1, 0);
                for (int i = 0; i < 3; i++) {
                    new_tile[i] = _shape[i];
                }
                std::vector<int> config = {256, 128, 64, 32, 128, 64, 2, 72};
                for (int i = 0; i < 8; i++) {
                    new_tile[3 + i] = config[i];
                }
                return {new_tile};
            }
        }

        return {{}};
    }

    std::vector<std::vector<int>> get_candidate_tile(int num_candidate = 5) {
        int m = _shape[0], n = _shape[1], k = _shape[2];

        // 0. hit shape case
        std::vector<std::vector<int>> new_tile_candidate = _get_knowledge_base();
        if (new_tile_candidate != std::vector<std::vector<int>>{{}}) {
            return new_tile_candidate;
        }

        // 1. get block_tile
        // 1.1 Get top 50 candidate according to repeat load
        new_tile_candidate = _get_candidate_block_tile(m, n, k);
        new_tile_candidate = _get_block_tile(m, n, k, new_tile_candidate, 50);

        // 1.2 Get top 30 according to min wave
        new_tile_candidate = _get_collapsed_topk(new_tile_candidate, WAVE, 30);

        // 1.3 Get top 20 according to block_utils and last wave block
        new_tile_candidate = _get_collapsed_topk(new_tile_candidate, BLOCK_UTILS, 20, false);
        new_tile_candidate = _get_collapsed_topk(new_tile_candidate, LAST_WAVE_BLOCK, 20, false);

        // 2. get stages
        std::vector<int> stage_candidates = _get_candidate_stages(m, n, k);
        new_tile_candidate = _get_stage_tile(stage_candidates, new_tile_candidate, 5);

        // 3. get warps
        new_tile_candidate = _get_warp_tile(new_tile_candidate, 20);

        // 4. calculate memory utilization of each layer
        new_tile_candidate = _get_mem_usage(new_tile_candidate);

        // 5. filter by poor candidate
        std::vector<std::vector<int>> filtered_tiles;
        for (const auto& tile : new_tile_candidate) {
            if (!_check_poor_candidate(tile)) {
                filtered_tiles.push_back(tile);
            }
        }
        new_tile_candidate = filtered_tiles;

        // 6. Add pre-configured optimal tiling
        new_tile_candidate = _add_config_tile(new_tile_candidate);

        // 7. select tiling according to num_candidate
        new_tile_candidate = _multi_phase_topk(new_tile_candidate, num_candidate);

        return new_tile_candidate;
    }

    void output_csv(const std::vector<std::vector<int>>& candidate_tile_list,
                    const std::string& file_name = "tiling_result") {
        std::string result_file = file_name + ".csv";
        std::vector<std::string> header = {"m",
                                           "n",
                                           "k",
                                           "bm",
                                           "bn",
                                           "bk",
                                           "wm",
                                           "wn",
                                           "wk",
                                           "stages",
                                           "cu_count",
                                           "block_rloadsize",
                                           "wave",
                                           "block_utils",
                                           "last_wave_block",
                                           "OCC",
                                           "warps_per_block",
                                           "warps_per_cu",
                                           "regs_per_block",
                                           "regs_rload",
                                           "LLC_UTILS",
                                           "SHAREMEM_UTILS",
                                           " L2_UTILS",
                                           " REG_UTILS",
                                           "occ_wave"};

        std::ofstream file(result_file);
        if (!file.is_open()) {
            return;
        }

        // Write header
        for (size_t i = 0; i < header.size(); ++i) {
            file << header[i];
            if (i < header.size() - 1) {
                file << ",";
            }
        }
        file << "\n";

        // Write data
        for (const auto& record : candidate_tile_list) {
            for (size_t i = 0; i < record.size() && i < header.size(); ++i) {
                file << record[i];
                if (i < header.size() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }

        file.close();
    }
};