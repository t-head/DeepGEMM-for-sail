import heapq
from enum import IntEnum
import csv
from .utils import get_num_sms, ceil_div

def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y

class MatmulHeuristicsTile:
    MAX_TOPK = False
    MIN_TOPK = True
    class HWMetric:
        def __init__(self,hw=None):
            self.cu_count = get_num_sms()
            self.mmad_cal_bpp = 4 #fp32 of C in share memory
            self.share_mem_size = 256*1024
            self.max_warps_per_cu = 64
            self.min_stages = 2
            self.min_warp_per_block = 4
            self.max_warp_per_block = 32
            self.max_warp_per_cu = 64
            self.max_register_per_block = 131072
            self.max_register_per_cu = 131072
            self.max_register_per_thread = 256
            self.max_register_per_warp = 256*32
            self.max_block_per_cu = 64
            self.semem_alloc_alignment = 8 # unit:B
            self.reg_alignment_size = 64 # unit:B
            self.reg_unit_size = 4 # unit:B
            self.cu_per_ce = 4
            self.l2_size = 1024 * 1024 # unit:B
            self.llc_size = 1024 * 1024 * 64 # unit:B
            self.max_warp_tile = 128
            self.min_warp_tile = 16
            self.tc_inst_fp16_m = 16
            self.tc_inst_fp16_n = 16
            self.tc_inst_fp16_k = 16
            self.max_reg_utils = 0.99 # reserve(1310 reg from 131072) for compile and avoid register overflow
    # candidate_tile_list:
    # ["m","n","k","bm","bn","bk","wm","wn","wk","stages","cu_count"
    # "block_rloadsize","wave","block_utils","last_wave_block","occ",
    # "warps_per_block","warps_per_cu","regs_per_block","regs_rload",
    # "llc_utils","sharemem_utils"," l2_utils"," reg_utils","occ_wave"]
    class IDX(IntEnum):
        M,N,K = range(3)
        BM,BN,BK,WM,WN,WK,STAGES,CU_COUNT\
        = [3 + i for i in range(8)]
        BLOCK_RLOADSIZE,WAVE,BLOCK_UTILS,LAST_WAVE_BLOCK,OCC,WARPS_PER_BLOCK\
        = [11 + i for i in range(6)]
        WARPS_PER_CU,REGS_PER_BLOCK,REGS_RLOADSIZE_PER_CU,LLC_UTILS,SHAREMEM_UTILS, L2_UTILS, REG_UTILS,OCC_WAVE\
        = [17 + i for i in range(8)]

    def __init__(self, shape: list[int], bpp: int, config_tile=[]):
        self._shape = shape
        self._bpp = bpp
        self.reg_db = 2 #double buffer on register
        self._config_tile = config_tile
        self._filter_metric = {
            self.IDX.BLOCK_RLOADSIZE:self.MIN_TOPK, self.IDX.BLOCK_UTILS:self.MAX_TOPK, self.IDX.WAVE:self.MIN_TOPK,\
            self.IDX.LAST_WAVE_BLOCK:self.MAX_TOPK, self.IDX.WARPS_PER_CU:self.MIN_TOPK,\
            self.IDX.L2_UTILS:self.MAX_TOPK, self.IDX.SHAREMEM_UTILS:self.MAX_TOPK,\
            self.IDX.REGS_RLOADSIZE_PER_CU:self.MIN_TOPK, self.IDX.REG_UTILS:self.MAX_TOPK
        }
        self._hw_metric = self.HWMetric()

    def _add_candidate_tile(self, candidate_tile, new_tile, compare_index=-1, top_k=10, ascend_order = True):
        # merge duplicate item
        tile_map = {}
        tile_result = []
        for item in candidate_tile:
            if item[compare_index] in tile_map:
                tile_map[item[compare_index]] += [item]
            else:
                tile_map[item[compare_index]] = [item]
        if len(candidate_tile) == 0:
            return [new_tile]

        if new_tile[compare_index] in tile_map:
            tile = tile_map[new_tile[compare_index]][-1]
            index = candidate_tile.index(tile)
            candidate_tile.insert(index+1,new_tile)
            return candidate_tile

        # keep topK elements
        ascend_flag = 1 if ascend_order == True else -1
        heap = [(ascend_flag * key, value) for key, value in tile_map.items()]
        heapq.heapify(heap)
        if len(list(tile_map)) < top_k:
            heapq.heappush(heap, (new_tile[compare_index] * ascend_flag, [new_tile]))
        elif new_tile[compare_index] * ascend_flag < heap[0][0]:
            heapq.heapreplace(heap, (new_tile[compare_index]*ascend_flag, [new_tile]))

        # sort tile_list according to compare_index
        for _, item in sorted(heap):
            tile_result += item
        return tile_result

    def _get_collapsed_topk(self, candidate_tile, compare_index=-1, top_k=10, ascend_order = True):
        # merge duplicate item
        tile_map = {}
        tile_result = []
        ascend_flag = 1 if ascend_order == True else -1
        for item in candidate_tile:
            key = item[compare_index] * ascend_flag
            if key in tile_map:
                tile_map[key] += [item]
            else:
                tile_map[key] = [item]
        if len(candidate_tile) == 0:
            return []

        sorted_list = {k: tile_map[k] for k in sorted(tile_map)}
        cur_num = 0
        for _, item in sorted_list.items():
            cur_num =  cur_num + 1
            tile_result += item
            if cur_num >= top_k:
                break
        return tile_result

    def _cal_smem_size(self, bm: int, bn: int, bk: int, stages: int):
        sharemem_utils = max((bm * bk + bn * bk) * self._bpp * stages, bm * bn * self._hw_metric.mmad_cal_bpp)
        return sharemem_utils

    def _get_max_stage(self, bm: int, bn: int, bk:int):
        bk_mem =  (bm * bk + bn * bk) * self._bpp
        return self._hw_metric.share_mem_size // bk_mem

    def _get_candidate_factor(self, m:int):
        if m>=4096:
            return [512,256,128]
        else:
            return []

    def _get_candidate_factor_ld(self, m:int):
        if m>=4096:
            return [64,128]
        else:
            return []

    def _get_candidate_stages(self, m, n, k):
        if(m>=4096 and n>=4096):
            return [3, 2]
        else:
            return []

    def _get_candidate_warp_factor(self, m):
        candidate_factors = [m//2, m//4, m//8, m//16, m]# max warps per block:32
        valid_factors = [x for x in list(range(self._hw_metric.min_warp_tile, self._hw_metric.max_warp_tile+1,16))\
                if x <= m]
        m, n = self._shape[0],self._shape[1]
        if m >= 4096 and n>=4096:
            candidate_factors = [factor for factor in candidate_factors if factor > self._hw_metric.min_warp_tile]
        return [factor for factor in candidate_factors if factor in valid_factors]#intersection of two list

    def _get_register_occupies(self, bm, bn, wm, wn, wk):
        reg_u_size = self._hw_metric.reg_unit_size
        reg_align_size = self._hw_metric.reg_alignment_size
        warp_num = (bm//wm) * (bn//wn)
        return (align(wm*self._hw_metric.tc_inst_fp16_k*self._bpp / reg_u_size, reg_align_size)\
                + align(wn*self._hw_metric.tc_inst_fp16_k*self._bpp / reg_u_size, reg_align_size))\
                *self.reg_db * warp_num\
                + align(bm*bn*self._hw_metric.mmad_cal_bpp / reg_u_size, reg_align_size)

    def _get_l2_usage(self, bm, bn, bk, stages, occ):
        return ((bm*bk+bn*bk)*self._bpp * stages * occ * self._hw_metric.cu_per_ce) / self._hw_metric.l2_size

    def _get_sharemem_usage(self, bm, bn, bk, stages, occ):
        mem_use = self._cal_smem_size(bm,bn,bk,stages)
        return (mem_use * occ) / self._hw_metric.share_mem_size

    def _get_llc_usage(self, bm,bn,bk,stages,occ):
        return (((bm*bk+bn*bk)*self._bpp*stages + bm*bn*self._bpp) * occ * self._hw_metric.cu_count) / self._hw_metric.llc_size

    def _get_occ(self, bm, bn, bk, wm, wn, num_stages):
        mem_size = self._cal_smem_size(bm, bn, bk, num_stages)
        reg_utils = self._get_register_occupies(bm, bn, wm, wn, bk)
        return min(self._hw_metric.share_mem_size // mem_size, self._hw_metric.max_register_per_cu // reg_utils)

    def _get_reg_usage(self, bm, bn, bk, wm, wn, occ):
        reg_utils = self._get_register_occupies(bm, bn, wm, wn, bk) * occ
        return reg_utils / self._hw_metric.max_register_per_cu

    def _get_mem_usage(self, tile_list):
        new_tile_candidate = []
        for tile in tile_list:
            bm,bn,bk,wm,wn,= tile[self.IDX.BM],tile[self.IDX.BN],tile[self.IDX.BK],\
                tile[self.IDX.WM],tile[self.IDX.WN]
            stages,occ,wave = tile[self.IDX.STAGES],tile[self.IDX.OCC],tile[self.IDX.WAVE]
            reg_utils = self._get_reg_usage(bm, bn, bk, wm, wn, occ)
            if reg_utils >= self._hw_metric.max_reg_utils:
               continue
            l2_utils = self._get_l2_usage(bm,bn,bk,stages,occ)
            sharemem_utils = self._get_sharemem_usage(bm,bn,bk,stages,occ)
            llc_utils = self._get_llc_usage(bm,bn,bk,stages,occ)
            tile[self.IDX.OCC_WAVE] = wave/occ
            cu_count = self._get_cu_count()
            tile[self.IDX.LLC_UTILS],tile[self.IDX.SHAREMEM_UTILS],tile[self.IDX.L2_UTILS],tile[self.IDX.REG_UTILS],tile[self.IDX.CU_COUNT]=\
            llc_utils, sharemem_utils, l2_utils, reg_utils, cu_count
            new_tile_candidate += [tile]
        return new_tile_candidate

    def _get_block_repeat_loadsize(self,m,n,k,bm,bn):
        Asize = m * k * self._bpp
        Bsize = n * k * self._bpp
        return Asize * ceil_div(n, bn) + Bsize * ceil_div(m, bm)

    def _get_num_waves(self,m,n,bm,bn):
        return ceil_div(ceil_div(m, bm) * ceil_div(n, bn), self._hw_metric.cu_count)

    def _get_block_utils(self,m,n,bm,bn):
        return ((m/bm) * (n/bn)) / (ceil_div(m,bm) * ceil_div(n,bn))

    def _get_last_wave_util(self,m,n,bm,bn):
        fix_wave_saturate = lambda x: self._hw_metric.cu_count if x == 0 else x
        return fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn)) % self._hw_metric.cu_count)

    def _get_block_tile(self, m, n, k, tile_list, top_k=50):
        new_tile_candidate = []
        for tile in tile_list:
            bm,bn,bk = tile[self.IDX.BM],tile[self.IDX.BN],tile[self.IDX.BK]
            if self._cal_smem_size(bm, bn, bk, self._hw_metric.min_stages) > self._hw_metric.share_mem_size or (k%bk!=0):
                continue
            reload_size = self._get_block_repeat_loadsize(m,n,k,bm,bn)
            num_waves = self._get_num_waves(m,n,bm,bn)
            block_util = self._get_block_utils(m,n,bm,bn)
            last_waves = self._get_last_wave_util(m,n,bm,bn)
            tile[self.IDX.BLOCK_RLOADSIZE],tile[self.IDX.WAVE],tile[self.IDX.BLOCK_UTILS],tile[self.IDX.LAST_WAVE_BLOCK]\
            = reload_size,num_waves,block_util,last_waves
            new_tile_candidate = self._add_candidate_tile(new_tile_candidate, tile, self.IDX.BLOCK_RLOADSIZE, top_k)

        return new_tile_candidate

    def _get_stage_tile(self, stage_candidates, tile_list, topk=5, ascend_order = False):
        new_tile_candidate = []
        for tile in tile_list:
            bm,bn,bk = tile[self.IDX.BM],tile[self.IDX.BN],tile[self.IDX.BK]
            max_stage =  self._get_max_stage(bm, bn, bk)
            for num_stages in stage_candidates:
                if num_stages > max_stage:
                    continue
                new_tile = tile.copy()
                new_tile[self.IDX.STAGES] = num_stages
                new_tile_candidate = self._add_candidate_tile(new_tile_candidate, new_tile, self.IDX.STAGES, topk, ascend_order)
        return new_tile_candidate

    def _get_warps_per_block(self,bm,bn,wm,wn):
        return ((bm // wm) * (bn // wn))

    def _get_warps_per_cu(self,bm,bn,bk,wm,wn,stages):
        warps_per_block = self._get_warps_per_block(bm,bn,wm,wn)
        return warps_per_block * self._get_occ(bm, bn, bk, wm, wn, stages)

    def _get_cu_count(self):
        return self._hw_metric.cu_count

    def _get_regs_rloadsize_per_cu(self,bm,bn,bk,wm,wn):
        m,n = self._shape[0],self._shape[1]
        rload_data_per_block = (wm*bk + wn*bk)*self._bpp*(bm//wm)*(bn//wn) / ((bm*bk+bn*bk)*self._bpp)
        block_per_cu = (m//bm)*(n//bn) / self._hw_metric.cu_count
        return rload_data_per_block * block_per_cu

    def _get_warp_tile(self, tile_list, topk=50):
        new_tile_candidate = []
        for tile in tile_list:
            bm, bn, bk, stages = tile[self.IDX.BM],tile[self.IDX.BN],tile[self.IDX.BK],tile[self.IDX.STAGES]
            wm_factors = self._get_candidate_warp_factor(bm)
            wn_factors = self._get_candidate_warp_factor(bn)
            warp_tile_list = [[x, y] for x in wm_factors for y in wn_factors]
            for warp_tile in warp_tile_list:
                wm, wn = warp_tile
                warps_per_block = self._get_warps_per_block(bm,bn,wm,wn)
                warps_per_cu = self._get_warps_per_cu(bm,bn,bk,wm,wn,stages)
                regs_per_block = self._get_register_occupies(bm,bn,wm,wn,bk)
                available_regs_per_block = self._hw_metric.max_register_per_warp * warps_per_block
                regs_rload = self._get_regs_rloadsize_per_cu(bm,bn,bk,wm,wn)
                if not (self._hw_metric.min_warp_per_block <= warps_per_block <= self._hw_metric.max_warp_per_block):
                    continue
                if warps_per_cu > self._hw_metric.max_warps_per_cu or \
                    regs_per_block > self._hw_metric.max_register_per_block or\
                    regs_per_block > available_regs_per_block:
                    continue
                new_tile = tile.copy()
                occ = self._get_occ(bm, bn, bk, wm, wn, stages)
                new_tile[self.IDX.WARPS_PER_BLOCK],new_tile[self.IDX.WARPS_PER_CU],new_tile[self.IDX.REGS_PER_BLOCK],\
                    new_tile[self.IDX.REGS_RLOADSIZE_PER_CU],new_tile[self.IDX.OCC]\
                = warps_per_block, warps_per_cu, regs_per_block, regs_rload, occ
                new_tile[self.IDX.WM],new_tile[self.IDX.WN],new_tile[self.IDX.WK] = wm,wn,bk
                new_tile_candidate = self._add_candidate_tile(new_tile_candidate, new_tile, self.IDX.REGS_RLOADSIZE_PER_CU, topk)

        return new_tile_candidate

    def _get_wave_from_occ(self, tile_list, topk=10):
        new_tile_candidate = []
        for tile in tile_list:
            wave,occ = tile[self.IDX.WAVE],tile[self.IDX.OCC]
            tile[self.IDX.OCC_WAVE] = wave/occ
            new_tile_candidate = self._add_candidate_tile(new_tile_candidate, tile, self.IDX.OCC_WAVE, topk)
        return new_tile_candidate

    def _get_candidate_block_tile(self, m, n, k):
        candidate_m = self._get_candidate_factor(m)
        candidate_n = self._get_candidate_factor(n)
        candidate_k = self._get_candidate_factor_ld(k)
        candidate_tile_list = []
        new_tile_candidate = [[x, y, z] for x in candidate_m for y in candidate_n for z in candidate_k]
        for tile in new_tile_candidate:
            new_tile = [0] * len(list(self.IDX))
            new_tile[self.IDX.M],new_tile[self.IDX.N],new_tile[self.IDX.K] = m,n,k
            new_tile[self.IDX.BM],new_tile[self.IDX.BN],new_tile[self.IDX.BK] = tile[0],tile[1],tile[2]
            candidate_tile_list += [new_tile]
        return candidate_tile_list

    def _add_config_tile(self, candidate_tile_list):
        m,n,k = self._shape[0],self._shape[1],self._shape[2]
        cand_tile = set(tuple(item[self.IDX.BM:self.IDX.STAGES+1]) for item in candidate_tile_list)
        for tile in self._config_tile:
            if tuple(tile) in cand_tile:
                continue
            bm,bn,bk,wm,wn,wk,stages = tile
            new_tile = [0] * len(list(self.IDX))
            new_tile[self.IDX.M],new_tile[self.IDX.N],new_tile[self.IDX.K] = m,n,k
            new_tile[self.IDX.BM],new_tile[self.IDX.BN],new_tile[self.IDX.BK] = bm,bn,bk
            new_tile[self.IDX.WM],new_tile[self.IDX.WN],new_tile[self.IDX.WK] = wm,wn,wk
            new_tile[self.IDX.STAGES] =  stages
            new_tile[self.IDX.BLOCK_RLOADSIZE] = self._get_block_repeat_loadsize(m,n,k,bm,bn)
            new_tile[self.IDX.WAVE] = self._get_num_waves(m,n,bm,bn)
            new_tile[self.IDX.BLOCK_UTILS] = self._get_block_utils(m,n,bm,bn)
            new_tile[self.IDX.LAST_WAVE_BLOCK] = self._get_last_wave_util(m,n,bm,bn)
            occ = self._get_occ(bm, bn, bk, wm, wn, stages)
            new_tile[self.IDX.OCC] = occ
            new_tile[self.IDX.WARPS_PER_BLOCK] = self._get_warps_per_block(bm,bn,wm,wn)
            new_tile[self.IDX.WARPS_PER_CU] = self._get_warps_per_cu(bm,bn,bk,wm,wn,stages)
            new_tile[self.IDX.REGS_PER_BLOCK] = self._get_register_occupies(bm,bn,wm,wn,bk)
            new_tile[self.IDX.REGS_RLOADSIZE_PER_CU] = self._get_regs_rloadsize_per_cu(bm,bn,bk,wm,wn)
            new_tile[self.IDX.LLC_UTILS] = self._get_llc_usage(bm,bn,bk,stages,occ)
            new_tile[self.IDX.SHAREMEM_UTILS] = self._get_sharemem_usage(bm,bn,bk,stages,occ)
            new_tile[self.IDX.L2_UTILS] = self._get_l2_usage(bm,bn,bk,stages,occ)
            new_tile[self.IDX.REG_UTILS] = self._get_reg_usage(bm, bn, bk, wm, wn, occ)
            new_tile[self.IDX.CU_COUNT] = self._get_cu_count()
            new_tile[self.IDX.OCC_WAVE] = new_tile[self.IDX.WAVE]/new_tile[self.IDX.OCC]
            candidate_tile_list += [new_tile]
        return candidate_tile_list

    def _check_poor_candidate(self,tile):
        #1.1 remove warps_per_cu <= 4
        if tile[self.IDX.WARPS_PER_CU] <= 4:
            return True
        #1.2 register overflow
        reg_overflow_tile = [[256,256,128,128,32,128],\
                            [256,256,128,32,128,128]]
        if tile[self.IDX.BM:self.IDX.STAGES] in reg_overflow_tile:
            return True

        return False

    def _multi_phase_topk(self, new_tile_candidate, num_candidate):
        if len(new_tile_candidate) <= num_candidate:
            return new_tile_candidate

        tile_list = new_tile_candidate
        phase_topk = max(len(tile_list)//2, num_candidate)
        while len(tile_list) > num_candidate:
            for mc,ascend in self._filter_metric.items():
                tile_list = self._get_collapsed_topk(tile_list, mc, phase_topk, ascend)
                if (len(tile_list) < num_candidate):
                    break
            if phase_topk==1:
                break
            if phase_topk<=5:
                phase_topk = max(phase_topk-1,1)
            else:
                phase_topk = max(phase_topk//2, 1)

        if num_candidate==1 and len(tile_list)>1:
            return [tile_list[0]]
        else:
            return tile_list

    def _get_knowledge_base(self):
        shape_knowledge = [[4096, 4608, 7168],\
                        [4096, 36864, 7168],\
                        [4096, 7168, 16384],\
                        [4096, 7168, 18432],\
                        [4096, 4096, 7168]\
                       ]
        if self._shape in shape_knowledge:
            new_tile = [0] * len(list(self.IDX))
            new_tile[0:3] = self._shape[0:3]
            new_tile[3:11] = [256, 128, 64, 32, 128, 64, 2, 72]
            return [new_tile]
        else:
            return [[]]

    def get_candidate_tile(self, num_candidate: int = 5):
        m, n, k = self._shape

        #0. hit shape case
        new_tile_candidate = self._get_knowledge_base()
        if new_tile_candidate != [[]]:
            return new_tile_candidate

        #1. get block_tile
        #1.1 Get top 50 candidate according to repeat load
        new_tile_candidate = self._get_candidate_block_tile(m, n, k)
        new_tile_candidate = self._get_block_tile(m, m, k, new_tile_candidate, top_k=50)

        #1.2 Get top 30 according to min wave
        new_tile_candidate = self._get_collapsed_topk(new_tile_candidate, self.IDX.WAVE, top_k=30)

        #1.3 Get top 20 according to block_utils and last wave block
        new_tile_candidate = self._get_collapsed_topk(new_tile_candidate, self.IDX.BLOCK_UTILS, top_k=20, ascend_order = False)
        new_tile_candidate = self._get_collapsed_topk(new_tile_candidate, self.IDX.LAST_WAVE_BLOCK, top_k=20, ascend_order = False)

        #2. get stages
        stage_candidates = self._get_candidate_stages(m,n,k)
        new_tile_candidate = self._get_stage_tile(stage_candidates, new_tile_candidate, topk=5)

        #3. get warps
        new_tile_candidate = self._get_warp_tile(new_tile_candidate, topk=20)

        #4. calculate memory utilization of each layer
        new_tile_candidate = self._get_mem_usage(new_tile_candidate)

        #5. filter by poor candidate
        new_tile_candidate =  [tile for tile in new_tile_candidate if not self._check_poor_candidate(tile)]

        #6. Add pre-configured optimal tiling
        new_tile_candidate = self._add_config_tile(new_tile_candidate)

        #7. select tiling according to num_candidate
        new_tile_candidate = self._multi_phase_topk(new_tile_candidate, num_candidate)

        return new_tile_candidate

    def output_csv(self, candidate_tile_list,file_name = "tiling_result"):
        result_file = file_name + ".csv"
        header = ["m","n","k","bm","bn","bk","wm","wn","wk","stages","cu_count","block_rloadsize","wave","block_utils","last_wave_block","OCC","warps_per_block","warps_per_cu","regs_per_block","regs_rload","LLC_UTILS","SHAREMEM_UTILS"," L2_UTILS"," REG_UTILS","occ_wave"]
        with open(result_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for record in candidate_tile_list:
                writer.writerow(record)