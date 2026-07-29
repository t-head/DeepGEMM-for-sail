"""Shared adaptive tile-selection strategy for DenseGemm on PPU1.5 (890P).

This module hosts the wave-aware, register-model-driven adaptive tile
selector (previously embedded in gemm.py) so that all DenseGemm dtypes
share a single strategy implementation:

  - bf16 (`gemm.py`): uses the returned tile as-is.
  - int8 (`gemm_int8.py`): reuses the bf16 tile and doubles block_k/warp_k
    (int8 has bpp=1, so the doubled-K tile fits the same SMEM budget),
    then recomputes smem_config with its own int8 get_smem_config.

The selector is pure tile geometry — it returns:
    (num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages)
and each caller computes its own dtype-specific smem_config.

Env knobs:
  DG_TILE_REG_MODEL=1  model-only register spill check (default: empirical
                       cap table with model fallback)
  DG_COMPUTE_TILE=0    disable the compute-bound 16-warp tile override
  DG_BF16_ADAPTIVE=0   (in gemm.py) bypass this selector entirely
"""
import math
import os
from functools import lru_cache

from .utils import ceil_div

_USE_REG_MODEL_ONLY = os.environ.get('DG_TILE_REG_MODEL', '1') == '1'
_COMPUTE_TILE_ENABLED = os.environ.get('DG_COMPUTE_TILE', '1') != '0'

# int8 adaptive tile selector gate using shared DenseGemm strategy
#   DG_INT8_ADAPTIVE=0: force disable adaptive tile
#   DG_INT8_ADAPTIVE=1: force enable adaptive tile
#   unset: use is_int8_adaptive_shape() default logic
_INT8_ADAPTIVE_ENV = os.environ.get('DG_INT8_ADAPTIVE', None)


def is_int8_adaptive_enabled(m, n, k):
    # 0: force disable, 1: force enable, other values: use default
    if _INT8_ADAPTIVE_ENV == '0':
        return False
    if _INT8_ADAPTIVE_ENV == '1':
        return True
    return is_int8_adaptive_shape(m, n, k)


def is_int8_adaptive_shape(m: int, n: int, k: int) -> bool:
    """Whether (m, n, k) falls within the validated INT8 adaptive tile shape range.

    This is the single source of truth for which shapes are allowed to enter the
    INT8 adaptive tile-selector path. Add newly validated shapes here.
    """
    return (m <= 160 and (
        (n >= 10240 and k >= 1024) or
        (n == 8192 and k == 16384)
    ))

def is_bf16_adaptive_shape(m: int, n: int, k: int) -> bool:
    """Whether (m, n, k) falls within the BF16 adaptive tile shape range.
    """
    return (m <= 160 and n >= 5120 and k >= 1024)

def is_bf16_adaptive_enabled(m, n, k):
    # 0: force disable, 1: force enable, other values: use default
    env_val = os.environ.get('DG_BF16_ADAPTIVE')
    if env_val == '0':
        return False
    if env_val == '1':
        return True
    return is_bf16_adaptive_shape(m, n, k)

_SMEM_SIZE = 256 * 1024
_BASE_BLOCK_K = 64
_MAX_BLOCK_K = 512
_BLOCK_N_MAX = 992
_STAGE_OPTIONS = (2, 3, 4)
_BLOCKM_CANDIDATES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 256, 320, 384]

_REG_FILE_SIZE = 131072
_NATURAL_REGS_PER_THREAD = 168
_THREADS_PER_WARP = 32
_MAX_WARPS_PER_BLOCK = _REG_FILE_SIZE // (_NATURAL_REGS_PER_THREAD * _THREADS_PER_WARP)
_MISC_REGS = 20  # misc overhead: address gen, loop counters, predicates


def _compute_compiler_limit(warps_per_block):
    """Per-warp register limit given total warps in a block.

    Hardware: 8 WE/CU, 512 warp-regs/WE.
    limit = 512 / ceil(warps / 8), capped at 256.
    Special: 3 warps/WE -> 168 (compiler convention).
    """
    warps_per_WE = math.ceil(warps_per_block / 8)
    if warps_per_WE == 0:
        return 256
    raw = 512 // warps_per_WE
    if warps_per_WE == 3:
        raw = 168
    return min(raw, 256)


def _estimate_minimum_regs(wm, wn):
    """Lower bound on per-thread registers for a (WM, WN) tile.

    acc  = WM*WN/32  (FP32 accumulators)
    ab   = (WM+WN)/2 (A+B fragments, double-buffered)
    misc = loop vars, addresses, predicates
    """
    return (wm * wn) // 32 + (wm + wn) // 2 + _MISC_REGS


def _is_model_spill_free(wm, wn, total_warps):
    """Model-based spill-free check: minimum_needed <= compiler_limit."""
    return _estimate_minimum_regs(wm, wn) <= _compute_compiler_limit(total_warps)


def _tile_spill_ok(wm, wn, total_warps):
    """Whether (WM, WN) at total_warps passes the register spill check.

    DG_TILE_REG_MODEL=1 (model-only): pure model check.
    Default (empirical):              validated _SPILL_FREE_WN_CAP when
                                      available, model fallback for
                                      unmeasured WM.
    """
    if _USE_REG_MODEL_ONLY:
        return _is_model_spill_free(wm, wn, total_warps)
    empirical = _SPILL_FREE_WN_CAP.get(wm)
    if empirical is not None:
        return wn <= empirical
    return _is_model_spill_free(wm, wn, total_warps)


# Empirical spill-free WN ceiling per WM from hgobjdump scan of 721
# (WM, WN, BK, S, BM, BN) configs on the bf16 cutlass3 template (see
# _spill_sweep_mp.py / spill_table.csv). For each WM the value is the largest
# WN such that STACK SIZE == 0 at EVERY probed BK/S/BN. This is the true
# physical spill ceiling, used as a hard safety bound on top of the model.
#
# Comparison to register model floor (24-warp conservative):
#   WM    model  empirical
#    16     128        224
#    32      80        128
#    48      48        128
#    64      32         96
#    80      32         64
#    96      16         32
#   112      16         32   <- model under-estimates; promotion is a win
#   144,192  16          -   (unmeasured; fallback to model)
#
# Model matches or under-shoots the empirical ceiling everywhere except
# WM=112, where promoting WN 16->32 is measured to win (see memory
# `project_deepgemm_bm112_wn32_promotion`). For other WMs the empirical
# headroom is real but un-A/B-tested; using it would halve total_warps and
# typically regresses (see `feedback_generalization_not_search`).
_SPILL_FREE_WN_CAP = {
    16:  224,
    32:  128,
    48:  128,
    64:   96,
    80:   64,
    96:   32,
    112:  32,
}


def _model_wn_cap(wm, total_warps=None):
    """Model-based max spill-free WN, aligned to 16.

    Solves: WM*WN/32 + (WM+WN)/2 + MISC <= compiler_limit(total_warps).
    Default: total_warps = _MAX_WARPS_PER_BLOCK (conservative floor for
    formula-vs-empirical boundary detection).
    """
    if total_warps is None:
        total_warps = _MAX_WARPS_PER_BLOCK
    limit = _compute_compiler_limit(total_warps)
    denom = wm / 32.0 + 0.5
    if denom <= 0:
        return 256
    max_wn = (limit - wm / 2.0 - _MISC_REGS) / denom
    return max(16, int(max_wn) // 16 * 16)


def _wn_per_warp_cap(warp_m):
    """Max wn-per-warp that passes the register spill check.

    Model-only mode: conservative _model_wn_cap.
    Empirical mode (default): validated _SPILL_FREE_WN_CAP when available
    (allows tiles the model might reject but that perform well in practice),
    model fallback for unmeasured WM.
    """
    if _USE_REG_MODEL_ONLY:
        return _model_wn_cap(warp_m)
    empirical = _SPILL_FREE_WN_CAP.get(warp_m)
    if empirical is not None:
        return empirical
    return _model_wn_cap(warp_m)


def _select_blockm(cutlass_m):
    for c in _BLOCKM_CANDIDATES:
        if cutlass_m <= c:
            return c
    return 256


_WM_CANDIDATES = (16, 32, 48, 64, 96, 192)


def _pick_wm_wn(block_m, block_n):
    if block_n < 16 or block_n % 16 != 0:
        return None, None
    candidates = []
    for wm in _WM_CANDIDATES:
        if wm > block_m or block_m % wm != 0:
            continue
        warp_on_m = block_m // wm
        for wn in range(16, block_n + 1, 16):
            if block_n % wn != 0:
                continue
            wxn = block_n // wn
            total = warp_on_m * wxn
            if not _tile_spill_ok(wm, wn, total):
                continue
            if total > _MAX_WARPS_PER_BLOCK:
                continue
            candidates.append((wm, wn, total))
    if not candidates:
        return None, None
    wide = [c for c in candidates if c[1] >= 32]
    pool = wide if wide else candidates
    tier1 = [c for c in pool
             if c[2] % 8 == 0 and (block_m // c[0]) <= 4]
    if tier1:
        def _key(c):
            wm, wn, total = c
            return (_we_balance_rank(total),
                    abs((block_m // wm) - (block_n // wn)), -wn, -wm)
        best = min(tier1, key=_key)
    else:
        def _key(c):
            wm, wn, total = c
            return (abs(wm - 64), _we_balance_rank(total),
                    abs((block_m // wm) - (block_n // wn)), -wn, -wm)
        best = min(pool, key=_key)
    return best[0], best[1]


def _get_warp_m(block_m, block_n=None):
    if block_m > 160:
        if block_n is not None:
            wm, _ = _pick_wm_wn(block_m, block_n)
            if wm is not None:
                return wm
        return 64
    # BM <= 160: identity for BM<=112, half for BM>=128
    warp_m = block_m if block_m <= 112 else block_m // 2
    return warp_m


def _max_warps_on_n_for(block_m):
    warp_m = _get_warp_m(block_m)
    warp_on_m = max(1, block_m // warp_m)
    budget = _MAX_WARPS_PER_BLOCK if block_m > 160 else 32
    return max(1, budget // warp_on_m)


def _valid_wn_candidates(block_m, block_n):
    """Return list of (wn, total_warps) pairs valid for (BM, BN), WN descending."""
    if block_m > 160 or block_n < 16 or block_n % 16 != 0:
        return []
    warp_m = _get_warp_m(block_m)
    warp_on_m = max(1, block_m // warp_m)
    max_warp_on_n = _max_warps_on_n_for(block_m)
    wn_max_regs = _wn_per_warp_cap(warp_m)
    wn_cap = min(wn_max_regs, block_n) // 16 * 16
    min_warp_on_n = 1
    if warp_on_m == 1 and any(
        block_n % w == 0 and block_n // w >= 2 for w in range(16, block_n, 16)
    ):
        min_warp_on_n = 2
    results = []
    for wn in range(wn_cap, 0, -16):
        if block_n % wn == 0 and min_warp_on_n <= (block_n // wn) <= max_warp_on_n:
            tw = warp_on_m * (block_n // wn)
            if tw <= _MAX_WARPS_PER_BLOCK:
                results.append((wn, tw))
    return results


def _valid_warp_n(block_m, block_n):
    if block_m > 160:
        _, wn = _pick_wm_wn(block_m, block_n)
        return wn
    warp_m = _get_warp_m(block_m)
    warp_on_m = max(1, block_m // warp_m)
    model_floor = _model_wn_cap(warp_m)
    for wn, tw in _valid_wn_candidates(block_m, block_n):
        if wn > model_floor and warp_on_m * (block_n // wn) < 5:
            continue
        return wn
    return None


def _get_warp_n(block_m, block_n):
    wn = _valid_warp_n(block_m, block_n)
    if wn is not None:
        return wn
    raise AssertionError(
        f"_get_warp_n: no TSM-safe WN for BM={block_m} BN={block_n}. "
        f"BN must be in the set produced by _snap_bn_to_valid."
    )


def _snap_bn_to_valid(block_m, target_bn, max_bn):
    if 16 <= target_bn <= max_bn and _valid_warp_n(block_m, target_bn) is not None:
        return target_bn
    radius = 16
    max_radius = max(target_bn, max_bn)
    while radius <= max_radius:
        for cand in (target_bn + radius, target_bn - radius):
            if 16 <= cand <= max_bn and _valid_warp_n(block_m, cand) is not None:
                return cand
        radius += 16
    return 16


def _max_bn_smem(block_m, stage):
    return _SMEM_SIZE // (_BASE_BLOCK_K * 2 * stage) - block_m


def _max_bn_for_bm(block_m):
    """Max BN for BM>160 based on _get_max_warp_on_n across all valid (WM, WN)."""
    max_bn = 0
    for wm in _WM_CANDIDATES:
        if wm > block_m or block_m % wm != 0:
            continue
        for wn in range(16, 257, 16):
            won = _get_max_warp_on_n(block_m, wm, wn)
            if won >= 1:
                max_bn = max(max_bn, won * wn)
    return max_bn


_WE_PER_CU = 8


def _warp_grid_total(block_m, block_n):
    if block_m > 160:
        wm, wn = _pick_wm_wn(block_m, block_n)
        if wm is None:
            return _MAX_WARPS_PER_BLOCK + 1
        return (block_m // wm) * (block_n // wn)
    warp_m = _get_warp_m(block_m)
    warp_on_m = max(1, block_m // warp_m)
    warp_n = _get_warp_n(block_m, block_n)
    wxn = max(1, math.ceil(block_n / warp_n))
    return warp_on_m * wxn


def _we_balance_rank(total_warps):
    if total_warps % _WE_PER_CU != 0:
        return 100 + (_MAX_WARPS_PER_BLOCK - total_warps)
    if total_warps == 16: return 0
    if total_warps == 24: return 1
    if total_warps == 8:  return 2
    return 3


def _estimate_warpOnN(block_m, warp_m, warp_n, acc_size=4, input_size=2):
    """Estimate max WarpOnN from hardware constraints (SMEM + warp budget + VREG).

    Returns (warpOnN_spill_free, warpOnN_spill_but_ok).
    - acc_size:   accumulator element size in bytes (4 for fp32).
    - input_size: input fragment element size in bytes (2 for bf16/fp16).
    """
    warp_on_m = block_m // warp_m

    # --- Constraint 1: warp budget (32 warps/block max) ---
    warp_cap = 32 // warp_on_m

    # --- Constraint 2: SMEM capacity ---
    # (BM + BN) * BK_bytes * num_stages <= 256 KB
    # With BK=128B and stages=2: (BM + BN) * 128 * 2 = 256*1024 → BM + BN <= 1024
    smem_cap = (1024 - block_m) // warp_n

    hw_max = min(warp_cap, smem_cap)

    # --- Constraint 3: VREG pressure (register spill estimation) ---
    # Accumulator regs: WM * WN elements / 32 threads, each acc_size bytes / 4 bytes per vreg
    acc_vreg = warp_m * warp_n * acc_size // (32 * 4)
    # Input fragment regs per K-iteration: (WM + WN) * K16 * input_size / (32 threads * 4B/vreg)
    input_vreg = (warp_m + warp_n) * 16 * input_size // (32 * 4)
    # Double-buffered (normal) vs single-buffered (extreme packing)
    basic_vreg = acc_vreg + input_vreg * 2
    extreme_vreg = acc_vreg + input_vreg

    # Per-warp register budget at each occupancy level:
    #   CU has 8 WEs, each WE has 512 regs.
    #   4 warps/WE (32 total): 512/4 = 128 regs/warp
    #   3 warps/WE (24 total): align_down(512/3, 8) = 168 regs/warp
    #   2 warps/WE (16 total): 512/2 = 256 regs/warp
    VREG_PER_WARP = (128, 168, 256)  # indexed by [4w/WE, 3w/WE, 2w/WE]
    TOTAL_WARPS   = ( 32,  24,  16)  # corresponding total warps in block

    # sf: determined by basic_vreg (double-buffered, zero-spill guarantee)
    if basic_vreg <= VREG_PER_WARP[0]:
        sf = TOTAL_WARPS[0] // warp_on_m
    elif basic_vreg <= VREG_PER_WARP[1]:
        sf = TOTAL_WARPS[1] // warp_on_m
    elif basic_vreg <= VREG_PER_WARP[2]:
        sf = TOTAL_WARPS[2] // warp_on_m
    else:
        sf = 0

    # ok: determined by extreme_vreg (single-buffered, some spill acceptable)
    if extreme_vreg <= VREG_PER_WARP[0]:
        ok = TOTAL_WARPS[0] // warp_on_m
    elif extreme_vreg <= VREG_PER_WARP[1]:
        ok = TOTAL_WARPS[1] // warp_on_m
    elif extreme_vreg <= VREG_PER_WARP[2]:
        ok = TOTAL_WARPS[2] // warp_on_m
    else:
        ok = 0

    return min(sf, hw_max), min(ok, hw_max)

_WN_CANDIDATES_SMALLBM = (16, 32, 48, 64)

# Empirical WarpOnN caps from warp64_pivot_perf.csv (2026-06-27).
# Only entries that DIFFER from _estimate_warpOnN() theory are listed.
# Two-value tuple: (spill_free, spill_but_ok).
#   spill_free:   max WoN where STACK SIZE == 0 (zero spill).
#   spill_but_ok: max WoN where spill exists but perf >= 95% of spill_free
#                 baseline (benign spill zone, validated by checkin).
# Default mode uses spill_but_ok; DG_TILE_REG_MODEL=1 uses spill_free.
# Absent keys defer to _estimate_warpOnN() theoretical prediction.
_REGISTER_WARPONN_CAP = {
    # (BM, WM, WN): (spill_free, spill_but_ok)
    # Only entries where empirical != theory from _estimate_warpOnN.

    # --- BM <= 160: memory-bound path ---
    ( 80,  80, 32): (29, 29),  # theory=(24,29), sf higher than predicted
    ( 96,  48, 64): (14, 14),  # theory=(12,14), compiler register reuse
    ( 96,  96, 16): (31, 32),  # theory=(32,32), sf lower (stack=8B, negligible)
    ( 96,  96, 32): (24, 24),  # theory=(24,29), ok lower than predicted
    ( 96,  96, 64): (14, 14),  # theory=( 0,14), sf better (compiler reuse)
    (128,  64, 48): (12, 12),  # theory=(12,16), ok lower than predicted
    (144, 144, 16): (31, 32),  # theory=(24,32), sf higher than predicted
    (160,  80, 16): (15, 16),  # theory=(16,16), sf lower (stack=8B, negligible)
    (192,  96, 16): (13, 16),  # theory=(16,16), sf lower (stack=8B, negligible)
    (192,  96, 32): (12, 12),  # theory=(12,16), ok lower
    (192,  96, 48): ( 8,  8),  # theory=(16,16), empirical max=8 from warp64_pivot_perf.csv
    (192,  96, 64): ( 8,  8),  # theory=( 0, 8), sf better (compiler reuse)

    # --- BM > 160: compute-bound path ---
    (256,  64, 32): ( 7,  8),  # theory=( 8, 8), sf lower (stack=8B, negligible)
    (256,  64, 48): ( 6,  6),  # theory=( 6, 8), ok lower
    (256, 128, 16): (15, 16),  # theory=(12,16), sf higher
    (256, 128, 32): ( 8,  8),  # theory=( 8,12), ok lower
    (256, 128, 48): ( 1,  8),  # theory=( 0, 8), sf better (36B spill OK)
    (384,  96, 16): ( 6,  8),  # theory=( 8, 8), sf lower (stack=8B, negligible)
    (384,  96, 32): ( 6,  6),  # theory=( 6, 8), ok lower
    (384,  96, 64): ( 4,  4),  # theory=( 0, 4), sf better (compiler reuse)
    (512,  64, 48): ( 3,  3),  # theory=( 3, 4), ok lower
    (512, 128, 16): ( 7,  8),  # theory=( 6, 8), sf higher
    (512, 128, 32): ( 4,  4),  # theory=( 4, 6), ok lower
    (512, 128, 48): ( 1,  4),  # theory=( 0, 4), sf better (36B spill OK)
}


def _get_max_warp_on_n(block_m, warp_m, warp_n):
    """Effective max WarpOnN: empirical override if present, else theory."""
    cap = _REGISTER_WARPONN_CAP.get((block_m, warp_m, warp_n))
    if cap is not None:
        sf, ok = cap
    else:
        sf, ok = _estimate_warpOnN(block_m, warp_m, warp_n)
    return sf if _USE_REG_MODEL_ONLY else ok


def _select_tile_memory_bound(cutlass_n, cutlass_m, cutlass_k, num_sms, block_m, warp_m):
    """Memory-bound path (BM<=160): pick (BN, WN, BK, stages, warp_k) together.

    WN candidates: {16, 32, 48, 64}. For each WN, achievable BN is
    _get_max_warp_on_n(BM,WM,WN)*WN (already bounded by SMEM + regs + warps).
    Pipeline:
      1. plan_max_bn = max achievable BN across all WN candidates.
      2. wave_upper  = ceil(total_tiles / num_sms).
      3. bn_wave_even = ideal BN per SM per wave for even distribution.
      4. warp_n = largest WN whose achievable BN >= bn_wave_even AND
         WarpOnN * WarpOnM >= 4 (minimum occupancy).
      5. block_n = ceil(bn_wave_even / warp_n) * warp_n.
      6. block_k from SMEM budget (double BK to keep stages 2-3);
         stages capped by K-iterations; warp_k from warp grid.

    Invariants (implicit, no explicit clamp needed):
      - block_n <= achievable_bn(warp_n) <= smem cap.
      - WarpOnN = block_n / warp_n <= _get_max_warp_on_n(BM, WM, warp_n).
    """
    def achievable_bn(wn):
        return _get_max_warp_on_n(block_m, warp_m, wn) * wn

    # For BM>=80, WN=16 over-shrinks the warp tile (0/46 WIN across
    # K=1024/5120 fullgen, mostly REGR). Drop WN=16 from the candidate set;
    # BM<=64 keeps WN=16 as the small-N escape hatch.
    wn_candidates = (tuple(wn for wn in _WN_CANDIDATES_SMALLBM if wn >= 32)
                     if block_m >= 128 else _WN_CANDIDATES_SMALLBM)

    # Step 1: plan_max_bn = max achievable BN (SMEM + regs + warps bounded).
    plan_max_bn = max(achievable_bn(wn) for wn in wn_candidates)

    # Step 2: wave count from plan_max_bn.
    m_tiles    = math.ceil(cutlass_m / block_m)
    tile_max   = m_tiles * math.ceil(cutlass_n / plan_max_bn)
    wave_upper = math.ceil(tile_max / num_sms)

    # Step 3: per-wave BN target for even distribution.
    bn_wave_even = math.ceil(m_tiles * cutlass_n / (num_sms * wave_upper))
    # Step 4: pick largest WN (for TC utilization) whose capacity covers
    # bn_wave_even AND gives total warps (WarpOnN * WarpOnM) >= 4.
    warp_on_m = block_m // warp_m
    warp_n = 32 # BN < 16 * 4， (DG_N < ~2496): default WN=32 for balanced occupancy.
    #for wn in reversed(wn_candidates):
    for wn in wn_candidates:
        if achievable_bn(wn) >= bn_wave_even:
            warp_n = wn
            break
            #warp_on_n = math.ceil(bn_wave_even / wn)
            #if warp_on_n * warp_on_m >= 4:
            #    warp_n = wn
            #    break

    # Step 5: block_n = ceil-align bn_wave_even to warp_n.
    block_n = math.ceil(bn_wave_even / warp_n) * warp_n

    # Step 6: BK, stages, warp_k — memory-bound policy.
    # Start BK=64, double BK while stages > 3, up to BK=512.
    # Memory-bound keeps stages in {2, 3}.
    block_k = _BASE_BLOCK_K
    while True:
        max_stages = _SMEM_SIZE // ((block_m + block_n) * block_k * 2)
        if max_stages <= 3 or block_k >= _MAX_BLOCK_K:
            break
        block_k *= 2

    # Cap stages at 3; also bounded by K-iterations.
    num_stages = min(max_stages, 3, math.ceil(cutlass_k / block_k))
    num_stages = max(num_stages, 2)
    warp_k = get_warp_k(block_m, block_n, block_k, warp_m, warp_n, num_stages)
    return block_n, warp_n, block_k, num_stages, warp_k


def _compute_adaptive_blockn(cutlass_n, cu_num, block_m, cutlass_m):
    # BM<=160 has its own (BN, WN) selector; this entry point handles BM>160.
    max_warp_on_n = _max_warps_on_n_for(block_m)
    max_bn_smem_val = _max_bn_smem(block_m, 2)
    max_bn_regs = _max_bn_for_bm(block_m)
    max_bn = min(max_bn_smem_val, _BLOCK_N_MAX, max_bn_regs)
    plan_max_bn = max_bn

    m_tiles = math.ceil(cutlass_m / block_m)
    target_wave = math.ceil(m_tiles * cutlass_n / (cu_num * plan_max_bn))
    ideal_bn = m_tiles * cutlass_n / (target_wave * cu_num)
    warp_n = max(16, math.ceil(ideal_bn / max_warp_on_n / 16) * 16)
    bn_baseline = min(plan_max_bn, math.ceil(ideal_bn / warp_n) * warp_n)
    bn_baseline = _snap_bn_to_valid(block_m, bn_baseline, max_bn)

    def _grid_healthy(bn):
        wm, wn = _pick_wm_wn(block_m, bn)
        if wm is None:
            return False
        total_warps = (block_m // wm) * (bn // wn)
        return (block_m // wm) <= 4 and wn >= 32 and total_warps >= 8

    def _realized_waves(bn):
        return math.ceil(m_tiles * math.ceil(cutlass_n / bn) / cu_num)

    base_waves = _realized_waves(bn_baseline)
    fewer_healthy = [bn for bn in range(bn_baseline + 16, max_bn + 1, 16)
                     if _grid_healthy(bn) and _realized_waves(bn) < base_waves]
    if fewer_healthy:
        min_w = min(_realized_waves(bn) for bn in fewer_healthy)
        bn_baseline = min(bn for bn in fewer_healthy
                          if _realized_waves(bn) == min_w)

    if _grid_healthy(bn_baseline):
        return bn_baseline

    cands = []
    for bn in range(max(16, bn_baseline - 64),
                    min(max_bn, bn_baseline + 64) + 1, 16):
        if bn == bn_baseline or not _grid_healthy(bn):
            continue
        cands.append(bn)
    if not cands:
        return bn_baseline

    def _key(bn):
        return (abs(bn - bn_baseline),
                _we_balance_rank(_warp_grid_total(block_m, bn)), -bn)
    snapped = min(cands, key=_key)

    def _waves(bn):
        return math.ceil(math.ceil(cutlass_m / block_m)
                         * math.ceil(cutlass_n / bn) / cu_num)
    if _waves(bn_baseline) == 1 and _waves(snapped) >= 2:
        return bn_baseline
    return snapped


# Compute-bound BM>=192 strategy (gated by DG_COMPUTE_TILE)
# Targets 16-warp WE-ideal tiles (~9% SOL boost). Empirical guards (890P/39SM):
#   - baseline_wave == 1: never override (1-wave is sacred).
#   - baseline_wave <= 2 or BM == 192: same-wave only (extra wave regresses).
#   - baseline_wave >= 3 AND BM >= 256: allow +1 wave (but never +2).
def _compute_bound_tile(block_m, m, n, k, num_sms):
    if block_m not in (192, 256, 320, 384):
        return None
    if k < 2048 or n < 12288:
        return None
    baseline_bn = _compute_adaptive_blockn(n, num_sms, block_m, m)
    m_tiles = math.ceil(m / block_m)
    baseline_wave = math.ceil(m_tiles * math.ceil(n / baseline_bn) / num_sms)
    if baseline_wave == 1:
        return None
    max_wave = baseline_wave
    if baseline_wave >= 3 and block_m >= 256:
        max_wave = baseline_wave + 1
    candidates = []
    for wm in list(_WM_CANDIDATES) + [80]:
        if wm > block_m or block_m % wm != 0:
            continue
        warp_on_m = block_m // wm
        if warp_on_m > 4 or 16 % warp_on_m != 0:
            continue
        warp_on_n = 16 // warp_on_m
        for wn in (32, 48, 64):
            if not _tile_spill_ok(wm, wn, warp_on_m * warp_on_n):
                continue
            bn = warp_on_n * wn
            if bn < 16 or bn > _BLOCK_N_MAX or bn > _max_bn_smem(block_m, 2):
                continue
            prop_wave = math.ceil(m_tiles * math.ceil(n / bn) / num_sms)
            above_formula = wn > _model_wn_cap(wm)
            if above_formula and (prop_wave > baseline_wave or
                                  baseline_bn < bn - 32):
                continue
            if not above_formula and prop_wave > max_wave:
                continue
            candidates.append((wm, wn, bn, warp_on_m, warp_on_n, prop_wave))
    if not candidates:
        return None
    def _key(c):
        wm, wn, bn, wom, won, pw = c
        # 1. fewest waves first (always preferable)
        # 2. square-ish warp grid (|WoM-WoN| small) for L2 reuse
        # 3. larger BN (better N-reuse)
        # 4. smaller acc (more reg headroom)
        return (pw, abs(wom - won), -bn, (wm * wn))
    wm, wn, bn, _, _, _ = min(candidates, key=_key)
    return bn, wm, wn


def _select_adaptive_smem(block_m, block_n):
    bound = _SMEM_SIZE / ((block_m + block_n) * _BASE_BLOCK_K * 2.0)
    best_bk, best_s = None, None
    best_gap = float('inf')
    for s in _STAGE_OPTIONS:
        bk = _BASE_BLOCK_K
        while bk <= _MAX_BLOCK_K:
            val = s * (bk // _BASE_BLOCK_K)
            if val <= bound:
                gap = bound - val
                if best_bk is None or gap < best_gap:
                    update = True
                elif gap == best_gap:
                    update = (s > best_s) or (s == best_s and bk < best_bk)
                else:
                    update = False
                if update:
                    best_bk, best_s = bk, s
                    best_gap = gap
            else:
                break
            bk *= 2
    return best_bk, best_s


def get_warp_k(block_m, block_n, block_k, warp_m, warp_n, num_stages):
    """Return the WARP_K tile size (= block_k // WarpOnK).

    WarpOnK=1 -> warp_k = block_k (no K-split)
    WarpOnK=2 -> warp_k = block_k // 2
    """
    warp_on_m = max(1, block_m // warp_m)
    warp_on_n = max(1, block_n // warp_n)
    base_warps = warp_on_m * warp_on_n
    warp_on_k_max = max(1, 32 // base_warps)
    warp_on_k = block_k // 128

    if block_k in (256, 512) and warp_on_k <= warp_on_k_max:
        return 128
    else:
        return block_k

@lru_cache(maxsize=None)
def get_adaptive_configs(m: int, n: int, k: int, num_sms: int):
    # M/N swap wrapper. The adaptive selector is tuned assuming the short side
    # is M (m <= n). When n < m, swap the inputs so the short side becomes M,
    # run the selector, then swap the resulting tile back to the real (m, n)
    # orientation: BM<->BN and WM<->WN. block_k / warp_k / num_stages are
    # orientation-invariant; num_tiles is recomputed for the real shape.
    if n < m:
        (_, block_m, block_n, block_k, warp_m, warp_n, warp_k,
         num_stages) = _get_adaptive_configs_impl(n, m, k, num_sms)
        block_m, block_n = block_n, block_m
        warp_m, warp_n = warp_n, warp_m
        num_tiles = ceil_div(m, block_m) * ceil_div(n, block_n)
        return (min(num_tiles, num_sms), block_m, block_n, block_k,
                warp_m, warp_n, warp_k, num_stages)
    return _get_adaptive_configs_impl(m, n, k, num_sms)


def _get_adaptive_configs_impl(m: int, n: int, k: int, num_sms: int):
    # m > 512: compute-bound large-M, not covered by the wave-fit generalization.
    # Hard-pin to 256x256x64/WM64/WN64/S4. The overlay otherwise snaps BN down
    # to non-power-of-2 (e.g. m=527/n=5120/k=13824 -> BN=240/WN=48) which costs
    # ~12% vs the natural square tile (active cycles -20%).
    if m > 512 and n > 512:
        block_m, block_n, block_k = 256, 256, 64
        warp_m, warp_n = 64, 64
        num_stages = 4
        warp_k = block_k
        num_tiles = ceil_div(m, block_m) * ceil_div(n, block_n)
        return min(num_tiles, num_sms), block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages

    block_m = _select_blockm(m)

    if block_m <= 160:
        # BM = WM = 144，WN = 16 ｜｜ 32, stack issue need to fix by compiler
        if block_m == 144:
            block_m = 160

        # Memory-bound path: WM is a property of BM alone, then (BN, WN, BK,
        # stages, warp_k) are picked together. See `_select_tile_memory_bound`.
        warp_m = _get_warp_m(block_m)
        block_n, warp_n, block_k, num_stages, warp_k = _select_tile_memory_bound(n, m, k, num_sms, block_m, warp_m)
    else:
        # Compute-bound override (gated): try 16-warp WE-ideal first.
        ct = (_compute_bound_tile(block_m, m, n, k, num_sms)
              if _COMPUTE_TILE_ENABLED else None)
        if ct is not None:
            block_n, warp_m, warp_n = ct
        else:
            block_n = _compute_adaptive_blockn(n, num_sms, block_m, m)
            warp_m = _get_warp_m(block_m, block_n)
            warp_n = _get_warp_n(block_m, block_n)
        block_k, num_stages = _select_adaptive_smem(block_m, block_n)
        warp_k = get_warp_k(block_m, block_n, block_k, warp_m, warp_n, num_stages)

    num_tiles = ceil_div(m, block_m) * ceil_div(n, block_n)
    return min(num_tiles, num_sms), block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages
