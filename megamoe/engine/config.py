import os
from dataclasses import dataclass

_W4A4_BLOCK_N = 512
_W4A4_BLOCK_K = 256

_PREFILL_TOKEN_THRESHOLD = 512
_DECODE_TILED_TOKEN_THRESHOLD = 320

_DISPATCH_MAX_TOPK = 10


def _read_env_int(var_name: str, default: int) -> int:
    value = os.getenv(var_name)
    return default if value is None else int(value)


def _is_env_set(var_name: str) -> bool:
    return os.environ.get(var_name) is not None


_WORKSPACE_BLOCK_M_CANDIDATES = (32, 64, 96, 128)


@dataclass(frozen=True)
class _Ppu890KernelConfig:
    block_m: int
    num_threads: int


_WARPS_PER_BLOCK = 16
_GEMM_WARP_N = 8


def _get_kernel_config(m_tiles_per_warp: int) -> _Ppu890KernelConfig:
    if m_tiles_per_warp <= 0:
        raise ValueError("DG_M_TILES_PER_WARP must be positive")
    block_m = _WARPS_PER_BLOCK // _GEMM_WARP_N * 16 * m_tiles_per_warp
    if block_m not in _WORKSPACE_BLOCK_M_CANDIDATES:
        raise ValueError(f"DG_M_TILES_PER_WARP must select a valid BLOCK_M in "
                         f"{_WORKSPACE_BLOCK_M_CANDIDATES} (got {block_m})")
    return _Ppu890KernelConfig(block_m=block_m, num_threads=_WARPS_PER_BLOCK * 32)


def _get_topk_weight_site() -> int:
    topk_weight_site = _read_env_int("DG_TOPK_WEIGHT_SITE", 1)
    assert topk_weight_site in (1, 2), f"DG_TOPK_WEIGHT_SITE must be 1 (L2 epilogue) or 2 (combine) (got {topk_weight_site})"
    return topk_weight_site


def _uses_prefill_gemm(declared_max_tokens_per_rank: int, intermediate_hidden: int) -> bool:
    return declared_max_tokens_per_rank > _PREFILL_TOKEN_THRESHOLD and intermediate_hidden >= 768 and intermediate_hidden % 256 == 0


def _uses_tiled_gemm(declared_max_tokens_per_rank: int, intermediate_hidden: int, num_ranks: int, num_tokens: int) -> bool:
    return (intermediate_hidden >= 768 and intermediate_hidden % 256 == 0 and
            (declared_max_tokens_per_rank > _PREFILL_TOKEN_THRESHOLD or (num_ranks == 16 and num_tokens >= _DECODE_TILED_TOKEN_THRESHOLD)))
