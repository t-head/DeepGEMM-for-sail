"""Compatibility for the pre-v1.1 FP4 weighted logits calling convention.

Calls without legacy keywords retain the native v1.1 API. Legacy paged calls
must obtain their schedule through this module's metadata entry point. MTP
queries are expanded into independent next_n=1 rows; no query is dropped.
"""

import torch

from . import _C


def fp8_fp4_mqa_logits(*args, **kwargs):
    if "clean_logits" not in kwargs and "logits_dtype" not in kwargs:
        return _C.fp8_fp4_mqa_logits(*args, **kwargs)
    return _C._fp4_mqa_logits_legacy(*args, **kwargs)


_UNSET = object()


def fp8_fp4_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    indices=None,
    *,
    clean_logits=_UNSET,
    logits_dtype=_UNSET,
):
    if clean_logits is _UNSET and logits_dtype is _UNSET:
        return _C.fp8_fp4_paged_mqa_logits(
            q, kv_cache, weights, context_lens, block_table,
            schedule_meta, max_context_len, indices=indices,
        )
    clean_logits = False if clean_logits is _UNSET else clean_logits
    logits_dtype = torch.float32 if logits_dtype is _UNSET else logits_dtype
    if q[0].dim() != 4 or context_lens.dim() != 2:
        raise ValueError("legacy FP4 paged Q/context_lens must have rank 4/2")
    batch_size, next_n, heads, packed_dim = q[0].shape
    if not 1 <= next_n <= 6 or context_lens.shape != (batch_size, next_n):
        raise ValueError("context_lens must match Q's [batch_size, next_n], next_n=1..6")
    if next_n != 1:
        if indices is not None:
            raise ValueError("explicit request indices require next_n=1")
        q = (
            q[0].reshape(batch_size * next_n, 1, heads, packed_dim),
            q[1].reshape(batch_size * next_n, 1, heads),
        )
        context_lens = context_lens.reshape(batch_size * next_n, 1)
        block_table = block_table.repeat_interleave(next_n, dim=0)
    if indices is None:
        # Match metadata generation: each expanded query has its own schedule.
        indices = torch.arange(
            context_lens.shape[0], device=context_lens.device, dtype=torch.int32
        )
    return _C._fp4_paged_mqa_logits_legacy(
        q, kv_cache, weights, context_lens, block_table,
        schedule_meta, max_context_len, clean_logits=clean_logits,
        logits_dtype=logits_dtype, indices=indices,
    )


def get_paged_mqa_logits_metadata(
    context_lens, block_kv, num_sms, indices=None, metadata_extra=None
):
    if metadata_extra is not None:
        next_n, heads, head_dim, dtype = metadata_extra
        if type(dtype) is int:
            # Legacy tuples carried packed D and element size, rather than dtype.
            if dtype == 1 and head_dim == 64:
                raise ValueError(
                    "ambiguous packed/FP8 metadata: pass logical head_dim and torch dtype"
                )
            if dtype == 1:
                dtype = torch.float8_e4m3fn
            elif dtype == 2:
                dtype = torch.bfloat16
            else:
                raise ValueError("unsupported legacy metadata element size")
        if dtype == torch.int8 and heads in (32, 64) and indices is None:
            if context_lens.dim() != 2 or context_lens.shape[1] != next_n:
                raise ValueError("metadata next_n must match 2D context_lens")
            if not 1 <= next_n <= 6:
                raise ValueError("legacy FP4 paged next_n must be in 1..6")
            context_lens = context_lens.reshape(-1, 1)
            indices = torch.arange(
                context_lens.shape[0], device=context_lens.device, dtype=torch.int32
            )
            next_n = 1
        metadata_extra = (next_n, heads, head_dim, dtype)
    return _C.get_paged_mqa_logits_metadata(
        context_lens, block_kv, num_sms, indices=indices,
        metadata_extra=metadata_extra,
    )
