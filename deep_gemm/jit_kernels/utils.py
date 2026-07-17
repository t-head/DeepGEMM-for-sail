import torch
import os
import functools
from enum import Enum
import copy

_num_sms = None

class GemmType(Enum):
    """Matches C++ enum GemmType in utils_rtc.cuh"""
    DenseGemm = 0
    GroupedContiguous = 1
    GroupedMasked = 2
    GroupedNoPad = 3
    GroupedFused = 4
    BatchGemm = 5

class CompuleMode(Enum):
    COMPILE_AND_RUN = 0
    #ONLY_COMPILE must be 1 to align with sglang deepgemm usage
    ONLY_COMPILE = 1

compile_mode = CompuleMode.COMPILE_AND_RUN

def set_compile_mode(mode):
    global compile_mode
    compile_mode = mode

def get_compile_mode():
    global compile_mode
    return compile_mode

def set_num_sms(num_sms: int) -> None:
    """
    Set the maximum SM count for all GEMM kernels to use.

    Arguments:
        num_sms: the desired maximum SM count for all GEMM kernels to use.
    """
    global _num_sms
    assert 0 < num_sms <= torch.cuda.get_device_properties(device='cuda').multi_processor_count
    _num_sms = num_sms


def get_num_sms() -> int:
    """
    Get the current maximum limit of SM count for all GEMM kernels to use.
    If the count is never specified, the function will return the number of device SMs.

    Returns:
        Current maximum limit of SM count for all GEMM kernels to use.
    """
    global _num_sms
    if _num_sms is None:
        device_props = torch.cuda.get_device_properties(device='cuda')
        print(f'device_props.name:{device_props.name}')
        if "ZW810E" in device_props.name or "ZW610E" in device_props.name:
            _num_sms = 20
        else:
            _num_sms = device_props.multi_processor_count
    return _num_sms


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def get_m_alignment_for_contiguous_layout():
    """
    When we do a grouped GEMM in contiguous format, LHS are grouped into several batches along the M axis.
    Since we deal with exactly one sub-matrix of RHS for each GEMM block, batch sizes above should align well
        with GEMM block shape.

    Returns:
        Group-level alignment requirement for grouped contiguous layout, which is always 128.
    """
    return 128

def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Returns TMA-aligned transposed format of the input tensor. `torch.transpose` will be called if necessary.
    If the input tensor is already column-major layout and 16-byte aligned along the M axis
        (thus meets the requirement of LHS scaling tensor in DeepGEMM), this function will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in a device kernel
    assert x.dim() in (2, 3)
    remove_dim = False
    m, n = x.shape[-2], x.shape[-1]
    # aligned_m = get_tma_aligned_size(m, x.element_size())
    aligned_m = m
    if x.dim() == 2:
        if x.stride(0) == 1 and x.stride(1) == aligned_m:
            return x
        x, remove_dim = x.unsqueeze(0), True

    b = x.shape[0]

    # The last kernel gives a column-major TMA aligned layout
    if x.stride(0) == aligned_m * n and x.stride(1) == 1 and x.stride(2) == aligned_m:
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = torch.transpose(torch.empty((b, n, aligned_m), device=x.device, dtype=x.dtype), 1, 2)
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x

def get_col_major_tensor(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() in (2, 3), "Only 2-D or 3-D tensors supported"

    squeeze_dim = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_dim = True

    B, M, N = x.shape

    col_major = torch.empty((B, N, M), dtype=x.dtype, device=x.device)
    col_major = col_major.transpose(-2, -1)

    col_major[...] = x

    if squeeze_dim:
        col_major = col_major.squeeze(0)

    return col_major


def get_case_id() -> int:
    if not hasattr(get_case_id, 'case_id'):
        get_case_id.case_id = 0
    get_case_id.case_id += 1
    return get_case_id.case_id


@functools.lru_cache(maxsize=None)
def is_ppu1v5_device():
    device_prop = torch.cuda.get_device_properties(device='cuda')
    if device_prop.major == 8 and device_prop.minor == 9:
        return True
    else:
        return False

@functools.lru_cache(maxsize=None)
def get_sm_count():
    device_prop = torch.cuda.get_device_properties(device='cuda')
    return device_prop.multi_processor_count

@functools.lru_cache(maxsize=None)
def get_extra_info(m=0, n=0, k=0, dtype=torch.int8, api_type="dense") -> dict:
    extra_info = {}
    use_cutlass3 = False
    use_multistage_on_N = False
    use_moe_dynamic_tile = False

    if is_ppu1v5_device():
        use_cutlass3 = True

    if 'DG_USE_CUTLASS3' in os.environ:
        use_cutlass3 = int(os.getenv('DG_USE_CUTLASS3'))
    extra_info['use_cutlass3'] = use_cutlass3

    if 'DG_USE_MULTISTAGE_ON_N' in os.environ:
        use_multistage_on_N = int(os.getenv('DG_USE_MULTISTAGE_ON_N'))
    extra_info['use_multistage_on_N'] = use_multistage_on_N

    if 'DG_USE_MOE_DYNAMIC_TILE' in os.environ:
        use_moe_dynamic_tile = int(os.getenv('DG_USE_MOE_DYNAMIC_TILE'))
    extra_info['use_moe_dynamic_tile'] = use_moe_dynamic_tile
    return extra_info

def get_search_space(d: torch.dtype, gemm_type : str, m:int=0, n:int=0, k:int=0) -> list:
    """
    Returns search space according input gemm type

    Arguments:
        gemm_type: nopad, masked, dense

    Returns:
        The tile list:{block_m, block_n, warp_m, warp_n, stage}
    """
    assert gemm_type in ('nopad', 'masked', 'dense','contiguous')

    block_k = 64 if d == torch.bfloat16 else 128
    tile_list = [
        # blockM = 16
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


        # blockM = 32
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


        # blockM = 48
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


        # blockM = 64
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
            # blockM = 128
            [128, 128, 64, 64, block_k    , 2],
            [128, 128, 64, 64, block_k    , 3],
            [128, 128, 64, 64, block_k    , 4],
            [128, 128, 64, 64, block_k * 2, 3],
            [128, 256, 64, 64, block_k    , 2],
            [128, 256, 64, 64, block_k    , 3],

            # blockM = 160
            [160, 128, 80, 64, block_k    , 2],
            [160, 128, 80, 64, block_k    , 3],
            [160, 128, 80, 64, block_k    , 4],
            [160, 128, 80, 64, block_k * 2, 3],
            [160, 256, 80, 64, block_k    , 2],
            [160, 256, 80, 64, block_k    , 4],

            # blockM = 192
            [192, 128, 48, 64, block_k    , 2],
            [192, 128, 48, 64, block_k    , 3],
            [192, 128, 48, 64, block_k    , 4],
            [192, 128, 48, 64, block_k * 2, 3],
            [192, 256, 48, 64, block_k    , 2],
            [192, 256, 48, 64, block_k    , 4],

            # blockM = 256
            [256, 64, 32, 64, block_k,      2],
            [256, 64, 32, 64, block_k,      3],
            [256, 64, 32, 64, block_k * 2,  2],
            [256, 64, 32, 64, block_k * 2,  3],
            [256, 128, 64, 64, block_k    , 2],
            [256, 128, 64, 64, block_k    , 3],
            [256, 128, 64, 64, block_k    , 4],
            [256, 256, 64, 64, block_k,     4],

            # blockM = 320
            [320, 128, 80, 64, block_k    , 2],
            [320, 128, 80, 64, block_k    , 3],
            [320, 128, 80, 64, block_k    , 4],
            [320, 256, 80, 64, block_k    , 2],
            [320, 256, 80, 64, block_k    , 3],

            # blockK = 64B
            [128, 128, 64, 64, int(block_k / 2), 2],
            [128, 256, 64, 64, int(block_k / 2), 2],
            [256, 128, 64, 64, int(block_k / 2), 2],
        ])

    tile_list_rtn = []
    if d == torch.float8_e4m3fn:
        for tile in tile_list:
            if tile[0] in (16, 32, 64, 128, 192) and tile[4] == block_k:
                tile_list_rtn.append(tile)
                # if tile[4] == block_k and tile[0] != 192:
                #     tile_copy = copy.deepcopy(tile)
                #     tile_copy[4] = int(block_k / 2)
                #     tile_list_rtn.append(tile_copy)
        return tile_list_rtn

    # add block_k / 2 tile for k <256
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

@functools.lru_cache(maxsize=4096)
def get_paged_mqa_logits_tile(next_n, split_kv, num_heads, head_dim, datasize):
    def get_smem_tb_per_sm(stage_q, stage_k):
        block_m = split_kv
        block_n = next_n * num_heads
        block_k = head_dim
        smem_q = block_n * block_k * stage_q * datasize
        smem_k = block_m * block_k * stage_k * datasize
        smem_k_scale = block_m * stage_k * 4
        smem_weight = block_n * stage_q * 4
        smem = smem_q + smem_k + smem_k_scale + smem_weight
        smem_tb_per_sm = 262144 // smem
        return smem_tb_per_sm

    def get_vreg_tb_per_sm():
        assert next_n == 1 and split_kv == 64 and head_dim == 128 and (num_heads == 32 or num_heads == 64)
        vreg_base = 24
        if num_heads == 32 and datasize == 1:
            vreg_weights, vreg_acc, vreg_A, vreg_B = 8, 16, 16, 32
        elif num_heads == 64 and datasize == 1:
            vreg_weights, vreg_acc, vreg_A, vreg_B = 16, 32, 16, 64
        elif num_heads == 32 and datasize == 2:
            vreg_weights, vreg_acc, vreg_A, vreg_B = 8, 16, 32, 64
        elif num_heads == 64 and datasize == 2:
            vreg_weights, vreg_acc, vreg_A, vreg_B = 16, 32, 32, 128
        vreg = vreg_base + vreg_weights + vreg_acc + vreg_A + vreg_B
        num_threads = 128
        warps_per_we = 512 // vreg
        num_warps = 8 * warps_per_we
        vreg_tb_per_sm = num_warps // (num_threads // 32)
        return vreg_tb_per_sm

    if datasize == 1 and head_dim == 64: # fp4 packed head_dim = 64
        if num_heads == 32 and split_kv == 64:
            return (2, 3, 14)
        elif num_heads == 32 and split_kv == 256: # warp-interleave
            return (2, 3, 3)
        elif num_heads == 64 and split_kv == 64:
            return (2, 3, 8)
        elif num_heads == 64 and split_kv == 256: # warp-interleave
            return (2, 3, 2)
        else:
            raise ValueError
    tile_list = [(2, 3, get_smem_tb_per_sm(2, 3))]
    if next_n == 1 and split_kv == 64 and head_dim == 128 and (num_heads == 32 or num_heads == 64):
        tile_list.append((1, 3, min(get_smem_tb_per_sm(1, 3), get_vreg_tb_per_sm())))
    tile = max(tile_list, key=lambda x: (x[2], x[0]))
    # print(tile_list, tile)
    return tile
