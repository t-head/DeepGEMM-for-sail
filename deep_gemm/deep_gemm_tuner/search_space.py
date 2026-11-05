import copy
import torch

def get_search_space(d: torch.dtype, gemm_type: str, m: int = 0, n: int = 0, k: int = 0) -> list:
    assert gemm_type in ('nopad', 'masked', 'dense', 'contiguous')

    block_k = 64 if d == torch.bfloat16 else 128
    tile_list = [
        [16, 64, 16, 16, block_k, 2],
        [16, 64, 16, 16, block_k, 3],
        [16, 64, 16, 16, block_k, 4],
        [16, 64, 16, 16, block_k, 5],
        [16, 64, 16, 16, block_k * 2, 2],
        [16, 64, 16, 16, block_k * 2, 3],
        [16, 64, 16, 16, block_k * 2, 4],
        [16, 64, 16, 16, block_k * 2, 5],

        [16, 128, 16, 32, block_k, 2],
        [16, 128, 16, 32, block_k, 3],
        [16, 128, 16, 32, block_k, 4],
        [16, 128, 16, 32, block_k * 2, 2],
        [16, 128, 16, 32, block_k * 2, 3],
        [16, 128, 16, 32, block_k * 2, 4],
        [16, 128, 16, 16, block_k, 2],
        [16, 128, 16, 16, block_k, 3],
        [16, 128, 16, 16, block_k, 4],
        [16, 128, 16, 16, block_k * 2, 2],
        [16, 128, 16, 16, block_k * 2, 3],
        [16, 128, 16, 16, block_k * 2, 4],

        [16, 256, 16, 64, block_k, 2],
        [16, 256, 16, 64, block_k, 3],
        [16, 256, 16, 64, block_k * 2, 2],
        [16, 256, 16, 64, block_k * 2, 3],

        [32, 64, 16, 32, block_k, 2],
        [32, 64, 16, 32, block_k, 3],
        [32, 64, 16, 32, block_k, 4],
        [32, 64, 16, 32, block_k, 5],
        [32, 64, 16, 32, block_k * 2, 2],
        [32, 64, 16, 32, block_k * 2, 3],
        [32, 64, 16, 32, block_k * 2, 4],
        [32, 64, 16, 16, block_k, 2],
        [32, 64, 16, 16, block_k, 3],
        [32, 64, 16, 16, block_k * 2, 2],
        [32, 64, 16, 16, block_k * 2, 3],

        [32, 128, 16, 64, block_k, 2],
        [32, 128, 16, 64, block_k, 3],
        [32, 128, 16, 64, block_k, 4],
        [32, 128, 16, 64, block_k * 2, 2],
        [32, 128, 16, 64, block_k * 2, 3],
        [32, 128, 16, 32, block_k, 2],
        [32, 128, 16, 32, block_k, 3],
        [32, 128, 16, 32, block_k * 2, 2],
        [32, 128, 16, 32, block_k * 2, 3],

        [32, 256, 16, 64, block_k, 2],
        [32, 256, 16, 64, block_k, 3],
        [32, 256, 16, 64, block_k * 2, 2],
        [32, 256, 16, 64, block_k * 2, 3],

        [48, 64, 16, 16, block_k, 2],
        [48, 64, 16, 16, block_k * 2, 2],
        [48, 64, 16, 16, block_k, 3],
        [48, 64, 16, 16, block_k * 2, 3],
        [48, 64, 16, 32, block_k, 2],
        [48, 64, 16, 32, block_k * 2, 2],
        [48, 64, 16, 32, block_k, 3],
        [48, 64, 16, 32, block_k * 2, 3],

        [48, 128, 48, 32, block_k, 2],
        [48, 128, 48, 32, block_k * 2, 2],
        [48, 128, 48, 32, block_k, 3],
        [48, 128, 48, 32, block_k * 2, 3],

        [64, 64, 32, 32, block_k, 2],
        [64, 64, 32, 32, block_k, 3],
        [64, 64, 32, 32, block_k, 4],
        [64, 64, 32, 32, block_k * 2, 2],
        [64, 64, 32, 32, block_k * 2, 3],
        [64, 64, 32, 32, block_k * 2, 4],

        [64, 128, 32, 64, block_k, 2],
        [64, 128, 32, 64, block_k, 3],
        [64, 128, 32, 32, block_k, 2],
        [64, 128, 32, 32, block_k, 3],
        [64, 128, 32, 64, block_k * 2, 2],
        [64, 128, 32, 64, block_k * 2, 3],
        [64, 128, 32, 32, block_k * 2, 2],
        [64, 128, 32, 32, block_k * 2, 3],

        [64, 256, 32, 64, block_k, 2],
        [64, 256, 32, 64, block_k, 3],
        [64, 256, 32, 64, block_k * 2, 2],
        [64, 256, 32, 64, block_k * 2, 3]
    ]

    if 'dense' in gemm_type:
        tile_list.extend([
            [128, 128, 64, 64, block_k, 2],
            [128, 128, 64, 64, block_k, 3],
            [128, 128, 64, 64, block_k, 4],
            [128, 128, 64, 64, block_k * 2, 3],
            [128, 256, 64, 64, block_k, 2],
            [128, 256, 64, 64, block_k, 3],

            [160, 128, 80, 64, block_k, 2],
            [160, 128, 80, 64, block_k, 3],
            [160, 128, 80, 64, block_k, 4],
            [160, 128, 80, 64, block_k * 2, 3],
            [160, 256, 80, 64, block_k, 2],
            [160, 256, 80, 64, block_k, 4],

            [192, 128, 48, 64, block_k, 2],
            [192, 128, 48, 64, block_k, 3],
            [192, 128, 48, 64, block_k, 4],
            [192, 128, 48, 64, block_k * 2, 3],
            [192, 256, 48, 64, block_k, 2],
            [192, 256, 48, 64, block_k, 4],

            [256, 64, 32, 64, block_k, 2],
            [256, 64, 32, 64, block_k, 3],
            [256, 64, 32, 64, block_k * 2, 2],
            [256, 64, 32, 64, block_k * 2, 3],
            [256, 128, 64, 64, block_k, 2],
            [256, 128, 64, 64, block_k, 3],
            [256, 128, 64, 64, block_k, 4],
            [256, 256, 64, 64, block_k, 4],

            [320, 128, 80, 64, block_k, 2],
            [320, 128, 80, 64, block_k, 3],
            [320, 128, 80, 64, block_k, 4],
            [320, 256, 80, 64, block_k, 2],
            [320, 256, 80, 64, block_k, 3],

            [128, 128, 64, 64, int(block_k / 2), 2],
            [128, 256, 64, 64, int(block_k / 2), 2],
            [256, 128, 64, 64, int(block_k / 2), 2],
        ])

    tile_list_rtn = []
    if d == torch.float8_e4m3fn:
        for tile in tile_list:
            if tile[0] in (16, 32, 64, 128, 192) and tile[4] == block_k:
                tile_list_rtn.append(tile)
        return tile_list_rtn

    if k != 0 and k <= 512:
        for tile in tile_list:
            tile_list_rtn.append(tile)
            if tile[4] == block_k:
                tile_copy = copy.deepcopy(tile)
                tile_copy[4] = int(block_k / 2)
                tile_list_rtn.append(tile_copy)
    else:
        tile_list_rtn = tile_list

    return tile_list_rtn
