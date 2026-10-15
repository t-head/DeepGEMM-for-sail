"""SM topology: count, dead-SM probe, push-SM selection, push-SM budget."""

from typing import Dict, Optional

import torch

from ..jit import build_runtime

from .config import _read_env_int

# -- SM count ------------------------------------------------------------------


def _get_num_sms(num_sms: Optional[int]) -> int:
    available = torch.cuda.get_device_properties('cuda').multi_processor_count
    if num_sms is None:
        num_sms = _read_env_int("DG_NUM_SMS", available)
    if num_sms <= 0 or num_sms > available:
        raise ValueError(f"num_sms must be positive and <= {available}; got {num_sms}")
    return num_sms


# -- Dead-SM probe / push-SM selection -----------------------------------------

_TOTAL_PHYSICAL_SMS = 40
_SMS_PER_DIE = 20
_dead_sm_runtime = None
_dead_sm_cache: Dict[int, int] = {}
_push_smids_cache: Dict[int, list] = {}


def _build_dead_sm_runtime():
    global _dead_sm_runtime
    if _dead_sm_runtime is not None:
        return _dead_sm_runtime
    src = f"""
#include <cstdint>
#include <cuda_runtime.h>

#define DG_DEAD_SM_SLOTS {_TOTAL_PHYSICAL_SMS}

extern "C" __global__ void find_dead_sm_kernel(int* out) {{
    if (threadIdx.x == 0) {{
        uint32_t smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        if (smid < DG_DEAD_SM_SLOTS) out[smid] = 1;
    }}
}}

extern "C" void launch(int* out, int device_index, void* __raw_stream, int& __return_code) {{
    auto stream = reinterpret_cast<cudaStream_t>(__raw_stream);
    cudaMemsetAsync(out, 0, DG_DEAD_SM_SLOTS * sizeof(int), stream);
    int num_sms = 0;
    // The probe covers every physical SM slot, so the count has to come from the probed device.
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_index);
    find_dead_sm_kernel<<<num_sms * 32, 1, 0, stream>>>(out);
    __return_code = 0;
}}
"""
    _dead_sm_runtime = build_runtime('find_dead_sm_megamoe', (('out', torch.int), ('device_index', int), ('stream', torch.cuda.Stream)),
                                     src)
    return _dead_sm_runtime


def _detect_dead_sm(device: torch.device) -> int:
    """Physical smid that never responds, or _TOTAL_PHYSICAL_SMS when there is no gap.

    The launch maps a physical smid to a blockIdx with `s if s < dead_smid else s - 1`, so returning
    _TOTAL_PHYSICAL_SMS (one past every smid) makes that the identity -- the right answer when all
    physical SMs respond. A probe that cannot decide returns the same sentinel rather than guessing,
    because a wrong index silently moves the push SMs off their intended dies on that rank.
    """
    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    if device_idx in _dead_sm_cache:
        return _dead_sm_cache[device_idx]
    dead_sm_runtime = _build_dead_sm_runtime()
    prev_device_idx = torch.cuda.current_device()
    torch.cuda.set_device(device_idx)
    try:
        sm_present_flags = torch.zeros(_TOTAL_PHYSICAL_SMS, dtype=torch.int32, device=device)
        dead_sm_runtime(sm_present_flags, device_idx, torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        missing_smids = [i for i, seen in enumerate(sm_present_flags.tolist()) if not seen]
    finally:
        torch.cuda.set_device(prev_device_idx)
    dead_smid = missing_smids[0] if len(missing_smids) == 1 else _TOTAL_PHYSICAL_SMS
    _dead_sm_cache[device_idx] = dead_smid
    return dead_smid


def _select_push_smids(device: torch.device, num_push_sms: int) -> list:
    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    if device_idx in _push_smids_cache and len(_push_smids_cache[device_idx]) == num_push_sms:
        return _push_smids_cache[device_idx]
    dead_smid = _detect_dead_sm(device)
    push_sms_per_die = num_push_sms // 2
    die0_smids = [s for s in range(_SMS_PER_DIE) if s != dead_smid][:push_sms_per_die]
    die1_smids = [s for s in range(_SMS_PER_DIE, 2 * _SMS_PER_DIE) if s != dead_smid][:push_sms_per_die]
    smids = die0_smids + die1_smids
    assert len(smids) == num_push_sms, f"need {num_push_sms} push smids, got {len(smids)}"
    _push_smids_cache[device_idx] = smids
    return smids


def _get_live_smids(dead_smid) -> list:
    """Physical smids a launch actually runs on, ascending.

    `dead_smid` is the probe's answer: the lowest smid that never responded, or _TOTAL_PHYSICAL_SMS when
    every smid responded. The fused kernels are launched with one block per live smid."""
    return [s for s in range(_TOTAL_PHYSICAL_SMS) if s != dead_smid]


def _choose_idle_sm(dead_smid) -> int:
    """The single SM left out of the quant set.

    Taken from the die that holds more live SMs (its highest smid), so the quant SMs end up split
    evenly across the two dies: the quant loop is a plain grid stride, and an unbalanced split would
    leave one die's memory pipe idle. This is also the SM the count pass will run on."""
    live_smids = _get_live_smids(dead_smid)
    die0_smids = [s for s in live_smids if s < _SMS_PER_DIE]
    die1_smids = [s for s in live_smids if s >= _SMS_PER_DIE]
    return (die0_smids if len(die0_smids) > len(die1_smids) else die1_smids)[-1]


def _choose_push_pool_size(num_sms: int, num_ranks: int) -> int:
    """Push pool size for the push: DG_PUSH_POOL_SIZE when set, otherwise the 16-SM default.

    Each token is owned by exactly one push SM: the pool splits the token range into contiguous
    per-push-SM slices (cache-friendly), and each push SM parses its tokens' topk, builds the 35-line
    payload once in registers and emits it to the deduped destination ranks. The budget keeps
    num_ranks SMs for the consume side (pool + num_ranks <= num_sms), and the pool is even so it
    splits evenly across the two dies. Overridable via DG_PUSH_POOL_SIZE (even, within budget).
    """
    # Two ceilings: the SM budget above (num_ranks SMs stay on the consume side), and the physical
    # smid range -- the die-balanced selection takes half the pool from each die, so a pool larger
    # than 2 * _SMS_PER_DIE would run past the last physical smid. The role masks are 64-bit, so
    # they hold any smid the selection can reach.
    pool_cap = min((num_sms - num_ranks) & ~1, 2 * _SMS_PER_DIE)
    if pool_cap < 2:
        raise ValueError(f"num_sms={num_sms} leaves no room for a push pool: the consume side needs "
                         f"{num_ranks} SMs and the push pool is even and at least 2")
    num_push_sms = _read_env_int("DG_PUSH_POOL_SIZE", 16)
    if num_push_sms < 2 or num_push_sms % 2 != 0 or num_push_sms > pool_cap:
        raise ValueError(f"DG_PUSH_POOL_SIZE must be an even push-pool size in 2..{pool_cap} (got {num_push_sms})")
    return num_push_sms
