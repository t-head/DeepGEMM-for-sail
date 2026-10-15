"""MegaMoE: PPU890 fused W4A4 (FP4 activation x FP4 weight) MoE kernel.

An independent, self-contained project: it carries its own JIT compiler and the
fused kernel. DeepEP and DeepGEMM are OPTIONAL baseline dependencies, imported under
try/except by the benchmark's baseline paths (megamoe/bench/engine.py; the test's
availability probe imports deep_ep as well), so the fused path runs with neither
installed. The benchmark's reference path does need triton, which importing
megamoe.bench.engine pulls in rather than the package root.

Naming:
  Python package ............ megamoe
  C++ namespace / include ... megamoe
"""

from . import jit
from .engine import (
    MegamoeWorkspace,
    MegamoeWorkspaceLayout,
    compile_fp4_fp4_megamoe,
    launch_fp4_fp4_megamoe,
    get_megamoe_workspace_layout,
    get_symm_buffer_for_megamoe,
    transform_weights_for_megamoe,
)
from .bench.runner import (
    bench_kineto,
    calc_diff,
)
