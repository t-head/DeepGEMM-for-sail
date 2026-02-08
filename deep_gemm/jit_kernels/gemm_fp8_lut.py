from typing import Tuple
from functools import lru_cache

from collections import defaultdict
import bisect

class MNKDict:
    def __init__(self, data=None):
        self.index = defaultdict(list)      # (n, k) -> sorted list of m
        self.tiles = defaultdict(dict)     # (n, k) -> {m: tile}

        if data is not None:
            self._build_from_data(data)

    def _build_from_data(self, data):
        """
        批量构建内部结构
        """
        # 临时存储：按 (n, k) 分组
        groups = defaultdict(list)

        for m, n, k, tile in data:
            groups[(n, k)].append((m, tile))

        # 对每组按 m 排序，并分离出 m_list 和 tile_map
        for (n, k), items in groups.items():
            # 按 m 升序排序
            sorted_items = sorted(items, key=lambda x: x[0])
            m_list = [item[0] for item in sorted_items]
            tile_dict = {item[0]: item[1] for item in sorted_items}

            self.index[(n, k)] = m_list
            self.tiles[(n, k)] = tile_dict

    def query(self, m_query: int, n: int, k: int):
        """
        查询 (n,k) 下 m >= m_query 的最大 m 对应的 tile
        """
        key = (n, k)
        if key not in self.index:
            return None

        m_list = self.index[key]
        pos = bisect.bisect_left(m_list, m_query)
        if pos >= len(m_list):
            return None

        best_m = m_list[pos]
        return self.tiles[key][best_m]

@lru_cache(maxsize=None)
def get_best_configs_from_lut(m: int, n: int, k: int, groups: int, is_grouped_contiguous: bool, is_grouped_masked: bool) -> \
    Tuple[int, int, int, int, int, int]:
    fp8_dense_list = {
        ( 64, 2304, 4096): ( 32,  64, 128, 16, 32, 5),
        ( 64, 4096, 2048): ( 64,  64, 128, 32, 32, 4),
        (128, 2304, 4096): ( 64,  64, 128, 32, 32, 4),
        (128, 4096, 2048): (128, 128, 128, 32, 32, 3),
        (144, 2304, 4096): ( 64,  64, 128, 32, 32, 3),
        (144, 4096, 2048): (192, 128, 128, 48, 32, 5),
        (160, 2304, 4096): ( 64,  64, 128, 32, 32, 3),
        (160, 4096, 2048): (192, 128, 128, 48, 32, 5),
        (192, 2304, 4096): (128, 128, 128, 32, 32, 4),
        (224, 2304, 4096): ( 64, 256, 128, 32, 32, 3),
        (256, 2304, 4096): ( 64, 256, 128, 32, 32, 3),
    }

    fp8_nopad_list = MNKDict([
        (  4,  768, 4096, ( 32,  64, 128, 16, 32, 3)),
        (   1,  2048,   128, ( 16, 256, 128, 16, 64, 2)),
        (  12,  2048,   128, ( 16, 128, 128, 16, 32, 2)),
        (  32,  2048,   128, ( 32, 128, 128, 16, 64, 2)),
        (   1,   256,  2048, ( 16, 128, 128, 16, 32, 4)),
        (   3,   256,  2048, ( 16,  64, 128, 16, 16, 3)),
        (  13,   256,  2048, ( 16,  64, 128, 16, 16, 4)),
        (  29,   256,  2048, ( 32, 128, 128, 16, 64, 3)),
        (  32,   256,  2048, ( 64, 256, 128, 32, 64, 3)),
    ])

    if is_grouped_contiguous == False and is_grouped_masked == False and groups > 1:
        return fp8_nopad_list.query(m, n, k)
    elif is_grouped_contiguous == False and is_grouped_masked == False and groups == 1:
        m_aligned = ((m + 15) // 16) * 16
        key = (m_aligned, n, k)
        if key in fp8_dense_list.keys():
            return fp8_dense_list[key]
        else:
            return None
    else:
        return None
