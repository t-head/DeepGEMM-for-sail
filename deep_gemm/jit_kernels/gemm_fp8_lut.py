from typing import Tuple
from functools import lru_cache


@lru_cache(maxsize=None)
def get_best_configs_from_lut(m: int, n: int, k: int, groups: int, is_grouped_contiguous: bool, is_grouped_masked: bool) -> \
    Tuple[int, int, int, int, int, int]:
    fp8_grouped_contiguous_list = {
        ( 768, 4096): (128, 256, 128, 32, 64, 5),
        (4096,  384): (128, 256, 128, 32, 64, 5)
    }

    fp8_dense_list = {
        ( 64, 2304, 4096): ( 32,  64, 128, 16, 32, 5),
        ( 64, 4096, 2048): ( 64,  64, 128, 32, 32, 4),
        (144, 2304, 4096): (128, 128, 128, 32, 32, 4),
        (160, 2304, 4096): (128, 128, 128, 32, 32, 4),
        (192, 2304, 4096): (128, 128, 128, 32, 32, 4),
        (224, 2304, 4096): ( 64, 256, 128, 32, 32, 3),
        (256, 2304, 4096): ( 64, 256, 128, 32, 32, 3),
    }

    fp8_nopad_list = {
        (  4,  768, 4096):  ( 32,  64, 128, 16, 32, 3),
        (  4, 4096,  384):  ( 16, 128, 128, 16, 32, 2),
        (  8,  768, 4096):  ( 16, 128, 128, 16, 32, 2),
        (  8, 4096,  384):  ( 16, 128, 128, 16, 32, 2),
        (275, 4096,  384):  ( 64, 128, 128, 32, 32, 2),
        (272, 4096,  384):  ( 64, 128, 128, 32, 32, 2),
    }


    if is_grouped_contiguous == False and is_grouped_masked == False and groups > 1:
        key = (m, n, k)
        if key in fp8_nopad_list.keys():
            return fp8_nopad_list[key]
        else:
            return None
    elif is_grouped_contiguous == False and is_grouped_masked == False and groups == 1:
        m_aligned = ((m + 15) // 16) * 16
        key = (m_aligned, n, k)
        if key in fp8_dense_list.keys():
            return fp8_dense_list[key]
        else:
            return None
    else:
        return None
