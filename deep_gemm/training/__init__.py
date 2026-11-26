"""Routed MoE training on PPU0015; see docs/mega_moe_training.md."""

from .mega_moe import MegaMoEPlan
from ._common import mega_moe, to_g8_weight, from_g8_weight
from .deepep import DeepEPMoEPlan, CompactDeepEPMoEPlan, plan_compact_route_capacity

__all__ = [
    "mega_moe",
    "MegaMoEPlan",
    "DeepEPMoEPlan",
    "CompactDeepEPMoEPlan",
    "plan_compact_route_capacity",
    "to_g8_weight",
    "from_g8_weight",
]
