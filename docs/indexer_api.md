# PPU indexer API

The weighted and AVG policies are intentionally separate. Ordinary FP8 stays on
legacy entry points; `fp8_fp4_*` weighted entry points support MXFP4 only on
PPU1.5. They do not silently quantize FP8 inputs. BF16 Q/K APIs have been removed.

| Family | Weights | Logits | Paged next_n |
| --- | --- | --- | --- |
| Legacy FP8 | FP32 | FP32 | 1 |
| Legacy INT8 (810e) | FP32 or BF16 | Same dtype as weights | 1 |
| MXFP4 weighted | BF16 | BF16 | 1 |
| FP8/FP4 AVG | none | BF16 by default; optional FP32 (`logits_dtype`) | 1–6 |

Legacy `clean_logits` parameters default to False and reject True.
All AVG APIs omit `clean_logits`; nonpaged AVG requires `max_seqlen_k > 0`.
New weighted MXFP4 APIs have no `clean_logits` or `logits_dtype` parameter.
Logits outside the per-query valid window are unspecified: mask or supply
valid lengths to top-k. The library does not call DeepSelect internally.

## Complete Python API signatures

The following signatures match the current pybind exports in
`csrc/apis/attention.hpp` and are available under `deep_gemm`.
All functions except `get_mqa_logits_metadata` return `torch.Tensor`.
`get_mqa_logits_metadata` is a compatibility stub returning `None`. Other
metadata tensors are opaque and must be produced and consumed by the same
backend/version. Parameter order and
defaults below are part of the Python API. `next_n` is encoded in the paged Q
shape rather than passed as a separate argument.

## Non-paged DSA indexer

```python
def get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads) -> None: ...
def fp8_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=False, max_seqlen_k=0) -> torch.Tensor: ...
def int8_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=False, max_seqlen_k=0) -> torch.Tensor: ...
def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, schedule_meta=None) -> torch.Tensor: ...
```

Legacy `max_seqlen_k=0` returns `[Q,K]` with global K columns; positive values
return `[Q,max_seqlen_k]` with columns relative to each query's start. MXFP4
requires a positive maximum. MXFP4 Q is `(int8[Q,H,64], int32[Q,H])`, KV is
`(int8[K,64], int32[K])`, for logical D128. Ordinary FP8 Q is a tensor and
KV is `(e4m3[K,D], float32[K])`. INT8 uses actual INT8 arithmetic, unlike the
packed MXFP4 storage container.

Weights have shape `[Q,H]`, unit column stride and 16-byte aligned row stride.
Padded row views are accepted and packed before the existing PPU kernel.
Current dense MXFP4 supports H32/H64 and logical D128. INT8 Q is
`int8[Q,H,D]`, KV is `(int8[K,D], float32[K])`; fold per-query/head Q
quantization scales into weights before calling the INT8 API. Return row stride is 1024-byte aligned;
logical shape does not include allocation padding.

`get_mqa_logits_metadata(ks,ke,num_kv_tokens,num_heads)` is a compatibility
stub returning `None`. It performs no metadata generation or argument
validation. `fp8_fp4_mqa_logits` accepts but ignores `schedule_meta` and always
uses the explicitly supplied `cu_seq_len_k_start` / `cu_seq_len_k_end`.

## Paged DSA indexer

Q values have shape `[Q,1,H,D_storage]`, context lengths are contiguous
int32 `[Q,1]`, and block tables have one int32 row per query with unit column
stride. Supported page sizes are 32/64/128. Logits have logical shape
`[Q,max_context_len]`. INT8 and legacy FP8 take Q values directly; MXFP4
takes `(packed_values, int32_scales[Q,1,H])`.

```python
def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None, metadata_extra=None) -> torch.Tensor: ...
def fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits=False, indices=None) -> torch.Tensor: ...
def int8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits=False, indices=None) -> torch.Tensor: ...
def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, indices=None) -> torch.Tensor: ...
```

`get_paged_mqa_logits_metadata(context_lens,block_kv,num_sms,indices=None,
metadata_extra=None)` retains the PPU shape hint for AVG and legacy callers.
For MXFP4, supply request indices `[Q]` with int32 dtype and next_n=1. If
metadata_extra is absent, this selects the H32/H64 packed-D64 MXFP4 scheduling
policy. Both head counts share that tile. Pass the same current SM count to
metadata generation and execution.

New paged weighted MXFP4 requires `indices`; legacy FP8 and INT8 reject non-None
indices. Each query has its own context-length and block-table row. The current
PPU scheduler visits expanded rows independently, including rows belonging to
the same request; it does not pair query rows for cross-row KV reuse. Thus IDs
are checked but do not index a request-level block table. A repeated request ID
still requires an expanded per-query block table. Request IDs do not enable cross-query KV reuse in this implementation.

## Sparse indexer

```python
def get_sparse_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, sparse_kv_block_indices, qk_dtype, sparse_block_kv, use_unaligned_ks=False) -> torch.Tensor: ...
def fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv, use_unaligned_ks=False) -> torch.Tensor: ...
def get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv, sparse_kv_block_indices, qk_dtype, sparse_block_kv) -> torch.Tensor: ...
def fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv) -> torch.Tensor: ...
```

Sparse MXFP4 uses BF16 weights/output, H32 and logical D128.
`sparse_block_kv` must be 8 or 16. `num_max_sparse_blocks` must be positive,
divisible by 4 and at most 4096. Paged sparse Q has N1; its page size must be
positive and divisible by `sparse_block_kv`. Non-paged metadata generation
and computation must use the same `use_unaligned_ks` setting.
Both sparse APIs expose logical output
shape `[Q,num_max_sparse_blocks*sparse_block_kv]`, retaining aligned row stride.
MXFP4 callers must pass BF16 weights and provide valid candidate lengths to
top-k. For compressed dense output, use each query's `ke-ks`; for sparse
output, derive valid candidates from the selected blocks and context bounds.
After top-k, map candidate slots through the gather mapping or page table
to the coordinates expected by the downstream attention operation. Legacy FP8 remains
FP32. INT8 callers select BF16 or FP32 logits through the weights dtype; neither
INT8 API accepts `logits_dtype`. BF16 weights produce BF16 logits and FP32
weights produce FP32 logits.

## AVG indexer

`fp8_fp4_paged_mqa_avg_logits` accepts MXFP4 only (packed int8 values and
required int32 Q scales); ordinary FP8 and missing Q scales are rejected.
Use `fp8_paged_mqa_avg_logits` for ordinary FP8 paged AVG. The nonpaged
`fp8_fp4_mqa_avg_logits` continues to accept both FP8 and MXFP4.

```python
def fp8_mqa_avg_logits(q, kv_s, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, q_scale=None, logits_dtype=torch.bfloat16) -> torch.Tensor: ...
def fp8_fp4_mqa_avg_logits(q, kv, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, q_scale=None, logits_dtype=torch.bfloat16) -> torch.Tensor: ...
def fp8_paged_mqa_avg_logits(q, fused_kv_cache, context_lens, block_table, schedule_meta, max_context_len, indices=None, logits_dtype=torch.bfloat16) -> torch.Tensor: ...
def fp8_fp4_paged_mqa_avg_logits(q, fused_kv_cache, context_lens, block_table, schedule_meta, max_context_len, indices=None, logits_dtype=torch.bfloat16) -> torch.Tensor: ...
```

All four AVG functions default to BF16 output and allow explicit FP32 output.
The FP8-only APIs take Q as a tensor. The `fp8_fp4_*` APIs take Q as a tuple:
`(fp8_values, None)` for non-paged ordinary FP8, or
`(packed_fp4_values, int32_mx_scales)` for MXFP4. The paged tuple API accepts
only the latter. All non-paged AVG inputs require compressed output
(`max_seqlen_k > 0`);
`max_seqlen_k` is mandatory for both FP8 and FP4. Nonpaged `q_scale` remains
optional (FP32 scalar or `[Q]`); paged AVG has no extra q_scale. AVG does not
support request indices. All AVG variants omit the `clean_logits` parameter.

Paged Q is `[B,N,H,Dstorage]`, N=1–6 for AVG and N=1 for weighted calls.
Context lengths are `[B,N]`. Build AVG metadata with the actual N and Q storage
shape: `metadata_extra=(N,4,64,1)` for packed FP4 D128, `(N,4,D,1)` for FP8.
Callers must keep the hint consistent with Q. The implementation checks N
and selected scheduling constraints, but does not detect every incorrect
H/D/element-size hint; a successful call does not prove the hint is correct. Multiple speculative tokens remain
supported for AVG; weighted callers flatten them into query rows.

Paged quantized caches are byte views `[pages,P,1,value_bytes+4]`: each page
contains all P data rows followed by all P four-byte scales, with optional
page-stride padding. This is not an interleaved data+scale row layout.

AVG integration must omit `clean_logits` and supply a positive
`max_seqlen_k` for non-paged calls. For paged calls, keep N within 1–6
in both metadata generation and execution, including shape probing and
graph preparation. Callers must use a separate supported execution path
for requests outside this range.

## Recommended PPU integration for high performance

These are recommended integration settings. Validate accuracy and performance
with the actual model on the target device.

### Device and dtype selection

| Device / workload | Recommended API | Q/K | Weights | Logits |
| --- | --- | --- | --- | --- |
| 810E / PPU1.0 non-paged DSA | `int8_mqa_logits` | INT8 + FP32 KV scales | BF16 | BF16, inferred from weights dtype |
| 810E / PPU1.0 paged DSA | `int8_paged_mqa_logits` | INT8 + FP32 scales in the fused KV cache | BF16 | BF16, inferred from weights dtype |
| M890 / PPU1.5 non-paged DSA | `fp8_fp4_mqa_logits` | MXFP4 + packed int32 MX scales | BF16 | Fixed BF16 |
| M890 / PPU1.5 paged DSA | `fp8_fp4_paged_mqa_logits` | MXFP4 + int32 MX scales in the fused KV cache | BF16 | Fixed BF16 |
| M890 / PPU1.5 sparse | `fp8_fp4_sparse_mqa_logits` / `fp8_fp4_paged_sparse_mqa_logits` | MXFP4 | BF16 | Fixed BF16 |
| M890 / PPU1.5 AVG | Non-paged / paged `fp8_fp4_*avg_logits` | MXFP4 | None | BF16 by default |

INT8 APIs no longer accept `logits_dtype`: callers must supply BF16 weights
to obtain BF16 logits. If Q's INT8 quantization scales must be folded into
weights, perform that multiplication at the reference computation's precision,
then cast the final weights to BF16. Include this rounding in accuracy checks.
FP32 weights produce FP32 logits. Legacy FP8 remains a compatibility path
with fixed FP32 weights and logits.

### Paged scheduling: pass metadata_extra explicitly

When generating paged metadata, explicitly pass the keyword argument
`metadata_extra=(N, H, D_storage, element_size)`. Metadata generation and
execution must use matching Q shapes, page sizes and the current SM count.
This tuple describes the **physical storage shape of Q values**, rather than
the logical head dimension. It does not encode weights or logits dtype:

- `N`: `q_values.shape[1]`; fixed to 1 for DSA, and 1–6 for AVG.
- `H`: `q_values.shape[2]`; the actual head count for DSA, and 4 for AVG.
- `D_storage`: `q_values.shape[3]`; half the logical D for packed MXFP4.
- `element_size`: `q_values.element_size()`; currently 1 byte for INT8, FP8 and packed MXFP4.

| Input type / workload | metadata_extra | Example for logical D128 |
| --- | --- | --- |
| INT8 DSA | `(1, H, D, 1)` | `(1, 64, 128, 1)` |
| Legacy ordinary FP8 DSA | `(1, H, D, 1)` | `(1, 64, 128, 1)` |
| MXFP4 DSA | `(1, H, D // 2, 1)` | `(1, 64, 64, 1)` |
| Ordinary FP8 AVG | `(N, 4, D, 1)` | `(N, 4, 128, 1)` |
| MXFP4 AVG | `(N, 4, D // 2, 1)` | `(N, 4, 64, 1)` |

```python
# Quantized Q values: q[0] for an MXFP4 tuple, or q itself for INT8 / legacy FP8.
metadata_extra = (q_values.size(1), q_values.size(2), q_values.size(3), q_values.element_size())
schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, page_kv, deep_gemm.get_num_sms(), indices=indices, metadata_extra=metadata_extra)
```

Use `indices=None` for INT8, legacy FP8 and AVG. New MXFP4 paged DSA requires
contiguous int32 `indices[Q]`; pass the same indices to the compute API.
Use `fp8_paged_mqa_avg_logits` for ordinary FP8 paged AVG;
`fp8_fp4_paged_mqa_avg_logits` accepts MXFP4 only.

`metadata_extra` controls PPU tile and resource configuration selection.
Omitting it can select a compatibility tile for legacy/AVG callers. Although
the new MXFP4 path with indices can infer a default configuration, explicitly
passing the actual storage shape is recommended. Do not pass the tuple as
the fourth positional argument: that position belongs to indices. This is a
scheduling shape hint; it does not quantize inputs or convert their dtypes.
Sparse paged APIs use the separate `get_paged_sparse_mqa_logits_metadata`,
which has no metadata_extra argument.

### Non-paged: use compressed logits to reduce memory usage

For INT8 and legacy FP8, explicitly pass a positive `max_seqlen_k` to enable
compressed logits. New MXFP4 DSA and all non-paged AVG APIs already require
a positive value. Supply an upper bound covering every query's valid window:

```text
max_seqlen_k >= max_i(ke[i] - ks[i]), and max_seqlen_k > 0
```

The output changes from global columns `[Q,K]` to relative window columns
`[Q,max_seqlen_k]`. When valid windows are substantially shorter than the
total KV length, this reduces logits allocation and the range scanned by
downstream top-k. Actual allocation still includes row-stride alignment.
BF16 logits also use half the per-element storage of FP32. Compressed mode
alone does not guarantee substantial memory savings when windows are not
shorter than the full context.

Top-k must mask invalid columns using each row's `ke-ks`; do not rely on
zero or negative-infinity initialization. For non-paged output, convert a
relative column `j` to the global KV index using `ks[i]+j`. Sparse, paged or
gathered candidates require their respective slot/page mappings; do not
apply that formula directly to those layouts. Use a PPU top-k implementation with BF16 and appropriate masking support,
such as PPU DeepSelect when available. A valid-length prefix is sufficient
only when all valid candidates form a contiguous prefix; sparse candidate
slots may require an explicit mask or compaction before top-k. DeepGemm does not run top-k
internally.

To avoid synchronization in the decode hot path, use a window upper bound
already known to the caller. Avoid computing it through a GPU reduction
followed by `.item()` on every call. PPU1.5 `get_mqa_logits_metadata`
currently returns None and provides no scheduling speedup. Paged metadata
must still be generated correctly for the current contexts and scheduling
configuration.
