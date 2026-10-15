"""The torch reference's inputs, its forward, and the cache that keeps it off the critical path.

The reference is the harness's ground truth, and building it is expensive, so it is cached on disk
per (semantics, point, rank). The key covers exactly what can change the reference's numbers:

  * this package's own code -- the inputs, the forward, the fp4 helpers and the distributed
    gather. Those files are single-purpose, so hashing them whole picks up nothing unrelated.
  * the production pieces the reference has to reproduce, which live outside this package: the
    activation quantizer it re-runs (`bench/bf16_to_mxfp4.py` -- its python fixes the shapes the
    reference consumes and its kernel source the numbers), and the routed-weight site, taken as a
    value: what can change the output is the site the reference reads, not the code that computes it.
  * the point itself (`config`, rank) and the actual per-rank token counts.

Nothing else. The baselines, the SM topology, the harness and the workspace layout cannot change
these numbers; and the JIT layer that compiles the quantizer is left out deliberately, because a
change there re-compiles the kernels as well -- a reference left over from the previous compiler
then shows up as a loud calc_diff rather than as a silent pass.

The inputs mirror the benchmark's own construction bit for bit (shapes, RNG order, the seeded
re-randomization, the FP4 quantization): the benchmark assigns the
returned tensors onto itself, so the fused kernel and the reference see the same data.

Because the harness itself is not in the key, every value that shapes these inputs has to reach
`load_cached_reference` through its `config` argument -- that is what makes a new input-shaping flag safe to
add without touching this module, and what makes it wrong to forget.
"""

import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist

from megamoe.jit.compiler import get_cache_dir

from .dist import dist_print
from .fp4 import per_token_cast_to_fp4
from .reference import quantize_grouped_fp4_weights, compute_reference_output

# This file sits at <repo>/megamoe/bench/ref/, so three dirnames reach the package and a fourth
# the repo root; `megamoe.*` imports resolve against the repo root, relative ones against the
# importing file's own package.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)
_REF_DIR = os.path.dirname(os.path.abspath(__file__))
_FROM_IMPORT_RE = re.compile(r'^\s*from\s+([.\w]+)\s+import\s', re.MULTILINE)
# The activation quantizer the reference re-runs: hashed whole, because both its python (the shapes)
# and its kernel source (the numbers) are what the reference consumes.
_PRODUCTION_SOURCES = (os.path.join(_REPO_ROOT, 'megamoe', 'bench', 'bf16_to_mxfp4.py'), )


def _resolve_import(module: str, pkg_dir: str):
    """The local .py file a from-import names, or None for stdlib/third-party."""
    if module.startswith('.'):
        level = len(module) - len(module.lstrip('.'))
        parts = [part for part in module[level:].split('.') if part]
        base = pkg_dir
        for _ in range(level - 1):
            base = os.path.dirname(base)
    elif module == 'megamoe' or module.startswith('megamoe.'):
        parts = module.split('.')
        base = _REPO_ROOT
    else:
        return None
    path = os.path.join(base, *parts) if parts else base
    path = os.path.join(path, '__init__.py') if os.path.isdir(path) else path + '.py'
    return path if os.path.exists(path) else None


def _collect_own_sources(entry: str) -> Tuple[str, ...]:
    """This package's own files, reached through from-imports, transitively. Imports that leave the
    package are not followed: what the reference takes from outside is hashed explicitly or read as
    a value (see the module docstring)."""
    seen, pending = set(), [entry]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        with open(path) as f:
            source = f.read()
        for module in _FROM_IMPORT_RE.findall(source):
            target = _resolve_import(module, os.path.dirname(path))
            if target is None or not target.startswith(_REF_DIR + os.sep) or target in seen:
                continue
            pending.append(target)
    return tuple(sorted(seen))


# Hashed as (relative path, bytes) so a rename invalidates the key as well as an edit.
REF_CACHE_SOURCES = _collect_own_sources(os.path.abspath(__file__)) + _PRODUCTION_SOURCES


def collect_production_values() -> tuple:
    """The production knobs the reference reads as values rather than as code."""
    from megamoe.engine.config import _get_topk_weight_site
    return (int(_get_topk_weight_site()), )


@dataclass(frozen=True)
class ReferenceInputs:
    """The tensors this module builds for the reference (and the benchmark keeps on itself)."""

    x_bf16: torch.Tensor
    x_fp4: Tuple[torch.Tensor, torch.Tensor]
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    l1_weights: Tuple[torch.Tensor, torch.Tensor]
    l2_weights: Tuple[torch.Tensor, torch.Tensor]


# The routed-weight quantization recipe both weight matrices take: pow2 scales, k-granularity 32,
# one n-group per row, with the preprocessed SFB the kernels consume.
_W4A4_WEIGHT_QUANT_KWARGS = dict(pow2=True, k_granularity=32, n_granularity=1, use_preprocessed_sfb=True, store_n_granularity=1)


def build_reference_inputs(*, rank_idx: int, num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, hidden: int,
                           intermediate_hidden: int, masked_ratio: float) -> ReferenceInputs:
    """Build the activations, the routing and the quantized weights the reference reads."""
    experts_per_rank = num_experts // num_ranks
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    l1_weights_bf16 = torch.randn((experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device="cuda")
    l2_weights_bf16 = torch.randn((experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device="cuda")

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_weights = topk_weights.contiguous()

    if masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    seed = (rank_idx << 32) + num_tokens
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    x_bf16.data.copy_(torch.randn_like(x_bf16))
    l1_weights_bf16.data.copy_(torch.randn_like(l1_weights_bf16))
    l2_weights_bf16.data.copy_(torch.randn_like(l2_weights_bf16))
    torch.random.set_rng_state(rng_state)

    x_fp4 = per_token_cast_to_fp4(x_bf16, use_ue8m0=True, k_granularity=32, use_packed_ue8m0=True)
    l1_weights = quantize_grouped_fp4_weights(l1_weights_bf16, **_W4A4_WEIGHT_QUANT_KWARGS)
    l2_weights = quantize_grouped_fp4_weights(l2_weights_bf16, **_W4A4_WEIGHT_QUANT_KWARGS)
    return ReferenceInputs(x_bf16=x_bf16,
                           x_fp4=x_fp4,
                           topk_idx=topk_idx,
                           topk_weights=topk_weights,
                           l1_weights=l1_weights,
                           l2_weights=l2_weights)


def build_reference(inputs: ReferenceInputs, *, rank_idx: int, num_ranks: int, group, num_experts: int, num_topk: int, hidden: int,
                    intermediate_hidden: int, activation_clamp: float) -> torch.Tensor:
    """This rank's reference output for the given inputs."""
    return compute_reference_output(inputs.topk_idx,
                                    inputs.topk_weights,
                                    inputs.l1_weights[0],
                                    inputs.l1_weights[1],
                                    inputs.l2_weights[0],
                                    inputs.l2_weights[1],
                                    rank_idx,
                                    num_ranks,
                                    group,
                                    num_experts,
                                    num_topk,
                                    hidden,
                                    intermediate_hidden,
                                    activation_clamp,
                                    x_bf16=inputs.x_bf16)


def compute_ref_cache_key(config: tuple, rank_idx: int) -> str:
    digest = hashlib.sha256()
    for path in REF_CACHE_SOURCES:
        digest.update(os.path.relpath(path, _REPO_ROOT).encode())
        with open(path, 'rb') as f:
            digest.update(f.read())
    digest.update(repr((torch.__version__, config, rank_idx, collect_production_values())).encode())
    return digest.hexdigest()[:16]


def load_cached_reference(build, *, config: tuple, rank_idx: int, expected_shape) -> torch.Tensor:
    """Return the reference for this (source semantics, point, rank), from disk when it is there."""
    enabled = os.environ.get('DG_REF_CACHE_ENABLED', '1') != '0'
    local_token_count = torch.tensor([int(expected_shape[0])], dtype=torch.long, device='cuda')
    all_rank_token_counts = [torch.empty_like(local_token_count) for _ in range(dist.get_world_size())]
    dist.all_gather(all_rank_token_counts, local_token_count)
    token_counts_per_rank = tuple(int(size.item()) for size in all_rank_token_counts)
    key = compute_ref_cache_key(config + (token_counts_per_rank, ), rank_idx)
    path = f'{get_cache_dir()}/ref.{key}.pt'
    blob, load_error = None, None
    if enabled and os.path.exists(path):
        try:
            candidate = torch.load(path, map_location='cuda')
            if candidate.get('key') == key and tuple(candidate['y_ref'].shape) == tuple(expected_shape):
                blob = candidate
        except Exception as ex:
            load_error = ex
    all_hit = torch.tensor([1 if blob is not None else 0], dtype=torch.int64, device='cuda')
    dist.all_reduce(all_hit, op=dist.ReduceOp.MIN)
    if int(all_hit.item()) == 1:
        dist_print(f'[ref-cache] hit  {key}', once_in_node=True)
        return blob['y_ref']
    started = time.time()
    y_ref = build()
    if enabled:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'key': key, 'y_ref': y_ref.cpu()}, path)
    detail = f' (load failed: {type(load_error).__name__}: {load_error})' if load_error else ''
    dist_print(f'[ref-cache] {"saved" if enabled else "bypass"} {key} in {time.time() - started:.1f}s{detail}', once_in_node=True)
    return y_ref
