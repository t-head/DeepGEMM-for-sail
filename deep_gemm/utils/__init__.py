import torch

from . import math, layout
from .layout import *
from .math import *
from .dist import init_dist, uneven_all_gather


def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: tuple[int, int, int],
    num_groups: int | None = None,
    is_sfa: bool = False,
    disable_ue8m0_cast: bool = False,
) -> torch.Tensor:
    """
    Fake implementation intended to emulate upstream PPU0010 (FP32, 128, 128) path:
    no transform, just return `sf` (optionally validating basic expectations).
    All parameters are accepted to match the original signature.
    PPU DeepGemm get col-major input and row-major weights.
    """
    return sf.contiguous()


# Prefer C++ transform_sf_into_required_layout over the Python NO-OP version
try:
    from ..deep_gemm_cpp import transform_sf_into_required_layout  # noqa: F811
except ImportError:
    pass
