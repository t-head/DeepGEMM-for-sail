"""MegaMoE fused-kernel launchers (PPU890 W4A4 MoE kernel).

Public API re-exported from the sub-modules for convenient single-import access.
"""

from .launch import (
    compile_fp4_fp4_megamoe,
    launch_fp4_fp4_megamoe,
)
from .weights import transform_weights_for_megamoe
from .workspace import (
    MegamoeWorkspace,
    MegamoeWorkspaceLayout,
    get_megamoe_workspace_layout,
    get_symm_buffer_for_megamoe,
)