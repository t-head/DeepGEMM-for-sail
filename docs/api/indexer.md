# PPU indexer API

The weighted and AVG policies are intentionally separate. Ordinary FP8 stays on
legacy entry points; `fp8_fp4_*` weighted entry points support MXFP4 only on
PPU1.5 (M890). INT8 indexer entry points support only PPU1.0 (810E);
M890 rejects both non-paged and paged INT8 indexer calls. FP8 and MXFP4
compute calls are rejected on 810E before kernel dispatch, including AVG
and sparse; sparse metadata generation is also M890-only.
The weighted MXFP4 entry points do not silently quantize FP8 inputs.

| Family | Weights | Logits | Paged next_n |
| --- | --- | --- | --- |
| Legacy FP8 | FP32 | FP32 | 1 |
| Legacy INT8 (810E only) | FP32 or BF16 | Same dtype as weights | 1 |
| MXFP4 weighted | BF16 | BF16 | 1 |
| FP8/FP4 AVG | none | BF16 by default; optional FP32 (`logits_dtype`) | 1–6 |

FP8 on M890 and INT8 on 810E support logical head dimensions 64 and 128.
Logical D32 is unsupported and rejected by the non-paged/paged compute
entries and explicit paged metadata configuration, including FP8 AVG.
MXFP4 DSA, AVG and sparse remain restricted to logical D128 (packed D64).

Legacy `clean_logits` parameters default to False and reject True.
All AVG APIs omit `clean_logits`; nonpaged AVG requires `max_seqlen_k > 0`.
New weighted MXFP4 APIs have no `clean_logits` or `logits_dtype` parameter.
Logits outside the per-query valid window are unspecified: mask or supply
valid lengths to top-k. The library does not call DeepSelect internally.

## Supported head dimensions and head counts

All D values below are **logical `head_dim`**. FP8/INT8 accept only D64 or
D128; MXFP4 accepts only D128, represented by a packed tensor dimension of
64. All other head dimensions are unsupported and rejected by host checks.
`num_heads` refers to Q heads, not the single KV head in a paged cache.

| API | Format / device | Logical head_dim | num_heads |
| --- | --- | --- | --- |
| `get_mqa_logits_metadata` | Compatibility stub | Not inspected | Ignored |
| `fp8_mqa_logits` | FP8 / M890 | 64, 128 | 32, 64 |
| `int8_mqa_logits` | INT8 / 810E only | 64, 128 | 32, 64 |
| `fp8_fp4_mqa_logits` | MXFP4 / M890 | 128 | 32, 64 |
| `get_paged_mqa_logits_metadata` | FP8 / M890; INT8 / 810E | 64, 128 | DSA: 32, 64; AVG: 4 (FP8 only) |
| `get_paged_mqa_logits_metadata` | MXFP4 / M890 | 128 | DSA: 32, 64; AVG: 4 |
| `fp8_paged_mqa_logits` | FP8 / M890 | 64, 128 | 32, 64 |
| `int8_paged_mqa_logits` | INT8 / 810E only | 64, 128 | 32, 64 |
| `fp8_fp4_paged_mqa_logits` | MXFP4 / M890 | 128 | 32, 64 |
| `get_sparse_mqa_logits_metadata` | MXFP4 / M890 | Fixed 128; no D argument | Fixed 32; no H argument |
| `fp8_fp4_sparse_mqa_logits` | MXFP4 / M890 | 128 | 32 |
| `get_paged_sparse_mqa_logits_metadata` | MXFP4 / M890 | Fixed 128; no D argument | Fixed 32; no H argument |
| `fp8_fp4_paged_sparse_mqa_logits` | MXFP4 / M890 | 128 | 32 |
| `fp8_mqa_avg_logits` | FP8 / M890 | 64, 128 | 4 |
| `fp8_fp4_mqa_avg_logits` | FP8 / M890 | 64, 128 | 4 |
| `fp8_fp4_mqa_avg_logits` | MXFP4 / M890 | 128 | 4 |
| `fp8_paged_mqa_avg_logits` | FP8 / M890 | 64, 128 | 4 |
| `fp8_fp4_paged_mqa_avg_logits` | MXFP4 / M890 | 128 | 4 |

Host checks enforce the table's head-count limits: DSA accepts only H32/H64,
AVG only H4, and sparse only H32. All other head counts are rejected before
kernel dispatch. Explicit paged metadata accepts H32/H64 for DSA and H4 for
AVG on M890; H4 is rejected on 810E, which has no AVG indexer support.
The dense metadata stub performs no validation. Sparse metadata has no D/H
arguments and targets its fixed-layout compute APIs.

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
stride. M890 supports page sizes 32/64/128; 810E supports only 64/128.
810E page32 is rejected by metadata generation and compute due to an
actlize TSM cross-cube addressing issue. Logits have logical shape
`[Q,max_context_len]`. INT8 and legacy FP8 take Q values directly; MXFP4
takes `(packed_values, int32_scales[Q,1,H])`.

For every paged DSA call, `max_context_len` must be at least the largest
value in `context_lens`. It is output capacity, not a clipping limit:
scheduling follows the actual context lengths, and insufficient capacity
can cause out-of-bounds writes. The same requirement applies to paged AVG.
Use a host-known upper bound to avoid a device reduction and synchronization.

```python
def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None, metadata_extra=None) -> torch.Tensor: ...
def fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits=False, indices=None) -> torch.Tensor: ...
def int8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits=False, indices=None) -> torch.Tensor: ...
def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, indices=None) -> torch.Tensor: ...
```

`get_paged_mqa_logits_metadata(context_lens,block_kv,num_sms,indices=None,
metadata_extra=None)` requires an explicit `metadata_extra` tuple on both
PPU0010 (810E) and PPU0015 (M890), for DSA and AVG. Omitting it or passing
None raises an error, including when `indices` is supplied. The Python
signature retains the None default for argument ordering only; there is no
compatibility fallback or automatic shape inference. Pass the same current
SM count to metadata generation and execution.

New paged weighted MXFP4 requires `indices`; legacy FP8 and INT8 reject non-None
indices. Each query still has its own context-length and block-table row.
Pass the same request grouping to metadata generation and computation, as
shown below. Metadata is opaque; do not construct or modify its internal format.

### Request indices and MTP / ragged integration

`indices` is a contiguous GPU int32 tensor of shape `[Q]`, on the same device
as `context_lens`. **`indices[i]` identifies the real request that owns query
row `i`.** It is not the query-row number, KV page number, selected Top-K
index, context length, or number of draft tokens. IDs may be local request
numbers or stable request-pool IDs; distinct requests in a call must have
distinct IDs. An ID is a grouping label, not an index into `block_table`.

Use real request ownership even when a backend schedules query rows
independently. Keeping the same request-major representation lets a backend
adopt request-aware scheduling without requiring callers to change their IDs
or tensor layouts.

**Current PPU behavior:** metadata generation and compute validate the indices
tensor's layout, but scheduling still visits each expanded query independently.
The current backend does not use ID values to group work or enable cross-query
KV reuse. The integration conventions below are valid with this implementation.

Flatten query tokens in request order, keeping each request's rows consecutive.
Apply the same row order to packed Q, Q scales, weights, context lengths, block
tables and indices. The public weighted API always uses `next_n=1`, even for
MTP: Q is `(int8[Q,1,H,64], int32[Q,1,H])`, weights are BF16 `[Q,H]`, and
context lengths are int32 `[Q,1]`. Do not put the MTP count in Q's second
dimension or in `metadata_extra[0]`.

Consecutive equal IDs describe a request run. Separate requests must never
be combined just because their context lengths match. Nonconsecutive
occurrences of the same ID form separate runs;
the backend does not sort rows. For request-major integration, expand each
request's KV page mapping to all of its query rows. Keep the largest context
in the run's last query, while retaining every query's actual length; earlier
lengths may differ and need not be monotonic. This ordering prepares inputs
for request-aware backends; the current per-query PPU scheduler uses each
row's own length. Each output row is limited by that query's context length.
The block table remains **per query**, `[Q,max_pages]`, even with repeated IDs.

The following examples assume `request_block_table` contains one row per real
request and `device` is the Q/context device. Q, scales, weights and per-token
context lengths must already follow the indicated flattened order.

**No MTP: one query per request.** Here `Q=B`, so an arange is a valid real
request grouping:

```python
B = request_block_table.shape[0]
indices = torch.arange(B, device=device, dtype=torch.int32)
block_table = request_block_table  # [B, max_pages]
# Three requests: indices = [0, 1, 2].
```

If the caller already has unique request-pool IDs, use those IDs instead of
arange. Arange is appropriate here because each real request has exactly one
query, not merely because the public interface has `next_n=1`.

**Fixed MTP: N queries per request.** For request-major flattened rows,
`Q=B*N`; repeat each real request ID N times:

```python
B = request_block_table.shape[0]
request_ids = torch.arange(B, device=device, dtype=torch.int32)
indices = torch.repeat_interleave(request_ids, repeats=N).contiguous()
block_table = torch.repeat_interleave(request_block_table, repeats=N, dim=0)
# B=2, N=3: indices = [0, 0, 0, 1, 1, 1].
```

Flatten Q/scales/weights from `[B,N,...]` to query rows, retaining the singleton
`next_n` dimension for Q/scales. Supply each token's actual context length;
do not replace those lengths by a single request maximum. All context lengths
must use the KV cache's position units, including any sequence compression.

**Ragged verify: a different verify_lens for each request.** Set
`Q=sum(verify_lens)` and repeat each request ID by its actual verify count.
Do not use a fixed maximum draft count or reset every query to an independent
request:

```python
# Host-known lengths from the batch plan; query rows are request-major.
verify_lens_cpu = [3, 1, 2]
B = len(verify_lens_cpu)
Q = sum(verify_lens_cpu)
request_ids = torch.arange(B, device=device, dtype=torch.int32)
verify_lens = torch.tensor(verify_lens_cpu, device=device, dtype=torch.int32)
indices = torch.repeat_interleave(
    request_ids, verify_lens, output_size=Q,
).contiguous()
block_table = torch.repeat_interleave(
    request_block_table, verify_lens, dim=0, output_size=Q,
)
# indices = [0, 0, 0, 1, 2, 2], NOT [0, 1, 2, 3, 4, 5].
```

The same formula handles uniform verify counts and ordinary decode counts of
one. A request with zero queries contributes no rows; skip the compute call
when the entire batch has `Q=0`. If a framework uses a fixed-size Graph buffer,
keep real rows correctly grouped; padding rows must have zero context and
separate dummy request IDs, with valid allocated Q/scale/weight/table rows.
Do not label padding as an unrelated real request.

**Common metadata and compute calls for all three scenarios:**

```python
metadata_extra = (1, num_heads, 128, torch.int8)  # M890 MXFP4, logical D128
schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
    context_lens, page_kv, deep_gemm.get_num_sms(),
    indices=indices, metadata_extra=metadata_extra,
)
logits = deep_gemm.fp8_fp4_paged_mqa_logits(
    (q_packed, q_scales), kv_cache, weights,
    context_lens, block_table, schedule_meta, max_context_len,
    indices=indices,
)
```

Always pass the same current request IDs to both calls, including on backends
that currently ignore ID values for scheduling. As a portable integration
rule, regenerate metadata when contexts or grouping change. For CUDA Graph
replay with mutable inputs, capture metadata generation before compute so
replay rebuilds the schedule from the current GPU tensors. On the current
PPU backend, context changes affect scheduling and ID changes do not. Derive
`Q`/`output_size` from host-known batch planning information when available;
avoid GPU `.item()` reads in the hot path. Mask invalid logits using each
query's own context length before Top-K.

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
Context lengths are `[B,N]`. Build AVG metadata with the actual N, logical
head dimension and dtype: `metadata_extra=(N,4,128,torch.int8)` for MXFP4 D128 on
M890, `(N,4,D,torch.float8_e4m3fn)` for FP8. D is always logical.
For each row, the last context length must be the row maximum:
`context_lens[b,N-1] == max(context_lens[b,:])`. The scheduler uses that
last entry to determine the KV range for all N query tokens. Increasing
prefix lengths satisfy this condition; an ordering such as `[257,1]` does
not and can leave valid logits unwritten. Also ensure
`max_context_len >= max(context_lens)` across all rows and tokens.
Callers must keep the hint consistent with Q. The implementation checks N
and selected scheduling constraints, but does not detect every incorrect
H/D/dtype hint; a successful call does not prove the hint is correct. Multiple speculative tokens remain
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
| 810E non-paged DSA | `int8_mqa_logits` | INT8 + FP32 KV scales | BF16 | BF16, inferred from weights dtype |
| 810E paged DSA | `int8_paged_mqa_logits` | INT8 + FP32 scales in the fused KV cache | BF16 | BF16, inferred from weights dtype |
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
`metadata_extra=(N, H, D, qk_dtype)`. **D is the logical head dimension for
all formats, including MXFP4. Do not pass packed D // 2.** Metadata generation
and execution must use matching query configurations, page sizes and SM counts.
The tuple can be constructed from runtime configuration before Q exists:

- `N`: future Q's next-token count; fixed to 1 for DSA, and 1–6 for AVG.
- `H`: the actual head count for DSA, and 4 for AVG.
- `D`: logical head dimension; MXFP4 D128 uses 128, even though its packed
  Q tensor has last dimension 64.
- `qk_dtype`: a `torch.dtype`, interpreted together with the current device
  architecture. It does not describe weights or logits dtype.

**The meaning of `torch.int8` depends on the device.** On M890 (PPU1.5), it
means packed MXFP4, matching the int8 storage container used by the MXFP4
compute APIs and sparse metadata APIs. On 810E (PPU1.0), it means true INT8
arithmetic. M890 does not support true INT8 indexer computation: both
`int8_mqa_logits` and `int8_paged_mqa_logits` explicitly reject calls on M890.
Passing `torch.int8` to metadata on M890 does not enable INT8 computation.

| Device / workload | qk_dtype | metadata_extra for logical D128 | Internal storage D |
| --- | --- | --- | --- |
| 810E INT8 DSA | `torch.int8` | `(1,H,128,torch.int8)` | 128 |
| M890 legacy FP8 DSA | `torch.float8_e4m3fn` | `(1,H,128,torch.float8_e4m3fn)` | 128 |
| M890 MXFP4 DSA | `torch.int8` | `(1,H,128,torch.int8)` | 64 |
| M890 FP8 AVG | `torch.float8_e4m3fn` | `(N,4,128,torch.float8_e4m3fn)` | 128 |
| M890 MXFP4 AVG | `torch.int8` | `(N,4,128,torch.int8)` | 64 |

Logical D must be 64 or 128; D32 is rejected.
810E metadata accepts only `torch.int8`. M890 metadata accepts
`torch.int8` or `torch.float8_e4m3fn`; MXFP4 requires logical D128.
Use `torch.int8` for MXFP4, not `torch.float4_e2m1fn_x2` or `torch.uint8`.

```python
# M890 MXFP4: configure metadata before Q exists, using logical D128.
metadata_extra = (1, num_heads, 128, torch.int8)
schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, page_kv, deep_gemm.get_num_sms(), indices=indices, metadata_extra=metadata_extra)
```

The library converts logical D to the storage dimension internally before
selecting `split_kv`: M890 MXFP4 uses D // 2, while FP8 and 810E INT8 use D.
Callers must still match the compute workload's N, H, format and logical D;
metadata tensor shape alone does not guarantee a compatible schedule.

The fourth tuple entry must be a `torch.dtype`; integer byte sizes are not
accepted. MXFP4 metadata uses logical D128, while its Q/K tensors use packed
D64.

Use `indices=None` for INT8, legacy FP8 and AVG. New MXFP4 paged DSA requires
contiguous int32 `indices[Q]`; pass the same indices to the compute API.
Use `fp8_paged_mqa_avg_logits` for ordinary FP8 paged AVG;
`fp8_fp4_paged_mqa_avg_logits` accepts MXFP4 only.

`metadata_extra` controls PPU tile and resource configuration selection and
is mandatory for every paged DSA/AVG metadata call on PPU0010/PPU0015.
The library rejects missing hints instead of silently selecting a potentially
severely slower compatibility schedule. Supplying `indices` does not remove
this requirement. Pass the tuple by keyword: the fourth positional argument
belongs to indices. The hint does not quantize inputs or convert their dtypes.
Sparse paged APIs use the separate `get_paged_sparse_mqa_logits_metadata`,
which has no metadata_extra argument.

### Non-paged: use compressed logits to reduce memory usage

For INT8 and legacy FP8, explicitly pass a positive `max_seqlen_k` to enable
compressed logits. New MXFP4 DSA and all non-paged AVG APIs already require
a positive value. Supply an upper bound covering every query's valid window:

```text
max_seqlen_k >= max_i(ke[i] - ks[i]), and max_seqlen_k > 0
```

For integration, derive this bound from the runtime maximum model length
and the KV sequence compression ratio, rather than from the total number of
KV tokens concatenated across requests:

```python
# Both values are host-side runtime configuration, available before Q exists.
# Exact division is sufficient when runtime_max_model_len is divisible by the ratio.
max_seqlen_k = (runtime_max_model_len + kv_compress_ratio - 1) // kv_compress_ratio
```

This is `runtime_max_model_len / kv_compress_ratio`, rounded up when needed.
`kv_compress_ratio` describes compression along the KV sequence axis, not
FP4 bit packing or the Q/K head dimension; use 1 for an uncompressed KV
sequence. The resulting bound must use the same KV-position units as
`ks` and `ke`. Account for any additional runtime positions or special KV
entries if they can make a valid window exceed this bound. Pass this positive
bound to every non-paged API requiring `max_seqlen_k`; legacy FP8/INT8 can
use the same convention to enable compressed logits. The bound determines
output capacity, while each row's `ke-ks` determines its valid length.

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
