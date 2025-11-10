from typing import Tuple
from functools import lru_cache


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

    fp8_nopad_list = {
        (  4,  768, 4096):  ( 32,  64, 128, 16, 32, 3),
        (  4, 4096,  384):  ( 16, 128, 128, 16, 32, 2),
        (  8,  768, 4096):  ( 16, 128, 128, 16, 32, 2),
        (  8, 4096,  384):  ( 16, 128, 128, 16, 32, 2),
        ( 16,  768, 4096):  ( 16, 128, 128, 16, 32, 2),
        ( 16, 4096,  384):  ( 16, 128, 128, 16, 32, 2),
        ( 32,  768, 4096):  ( 32,  64, 128, 16, 32, 4),
        ( 32, 4096,  384):  ( 32,  64, 128, 16, 32, 2),
        (224, 4096,  384):  ( 64, 128, 128, 32, 32, 2),
        (240, 4096,  384):  ( 64, 128, 128, 32, 32, 2),
        (256, 4096,  384):  ( 64, 128, 128, 32, 32, 2),
        (272, 4096,  384):  ( 64, 128, 128, 32, 32, 2),
        (288, 4096,  384):  ( 64,  64, 128, 32, 32, 2),
        (304, 4096,  384):  ( 64,  64, 128, 32, 32, 2),
        (320, 4096,  384):  ( 64,  64, 128, 32, 32, 2),
        (336, 4096,  384):  ( 64,  64, 128, 32, 32, 2),
        (512, 4096,  384):  ( 64, 128, 128, 32, 32, 2),
    }


    if is_grouped_contiguous == False and is_grouped_masked == False and groups > 1:
        m_aligned = ((m + 15) // 16) * 16
        if m <= 8:
            key = (m, n, k)
        else:
            key = (m_aligned, n, k)
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
