# UT Functional Test
Use `run_deep_gemm.py` with --format to specify input parameters.
## Usage
run_deep_gemm.py [-h] [--format FORMAT] [--caselist CASELIST] [--case_idx CASE_IDX] [--disable_acc] [--benchmark] [--ref_backend {host,device}]

options: \
  -h, --help            show this help message and exit \
  --format FORMAT       Case cmd to describe problem size. \
  --caselist CASELIST   the folder of DG cases \
  --case_idx CASE_IDX   the line index(1~line) of case in caselist file \
  --disable_acc         if or not open accuracy check \
  --benchmark           specify if run benchmark \
  --ref_backend {host,device}
                        specify the backend used to compute ref output \
## Example
### Accuracy test: compare deepgemm with pytorch cpu implementation
[recommended for cmodel environment] \
python ./run_deep_gemm.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12 --ref_backend=host

### Accuracy test: compare deepgemm with pytorch device kernel implementation
python ./run_deep_gemm.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12

### Run deepgemm kernel only without correctness check
python ./run_deep_gemm.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12 --disable_acc


# UT Performance Test
Use `run_deep_gemm_perf.py` with --format to specify input parameters and --mode to specify mode.
## Usage
run_deep_gemm_perf.py [-h] [--caselist CASELIST] [--case_idx CASE_IDX] [--format FORMAT] [--output OUTPUT] [--mode {metrics,full,show_log,umd_perf}]
                             [--device DEVICE] [--disable_acc] [--ref_backend {host,device}]

options: \
  -h, --help            show this help message and exit \
  --caselist CASELIST   the folder of DG cases \
  --case_idx CASE_IDX   the line index(1~line) of case in caselist file \
  --format FORMAT       Case cmd to describe problem size. \
  --output OUTPUT       the output storing cycles of DG cases \
  --mode {metrics,full,show_log,umd_perf}
                        run perf mode \
  --device DEVICE       devices index to run cases, 0 means gpu0. 0,3 means gpu0,1,2,3 \
  --disable_acc         if or not open accuracy check \
  --ref_backend {host,device}
                        specify the backend used to compute ref output \

## Example
### Run a single case with parameters specified by format
python ./run_deep_gemm_perf.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12,distribution:[45,200,30,18,13,21,31,3,5,17,17,16,18,15,18,13]

### Run a batch of cases (accuracy + performance by default)
python ./run_deep_gemm_perf.py --caselist=./deepgemm.caselist

# Checkin Test
## Test fp8
python ./test_fp8_core.py

## Test int8 and bf16
python ./test_core.py

test_core.py options:
  -h, --help            show this help message and exit
  --func {DenseGemm,GroupedContiguous,GroupedMasked,DenseGemm}  target test func
# Routed MoE correctness tests

The MoE tests use pytest and one process-group fixture. Select precision by file
and model by pytest ID; model dimensions, activation, routing limits and EP sizes
are declared in each precision file. Add a model to those tables instead of
copying a test script.

| File | Coverage |
| --- | --- |
| `test_moe_w4a4.py` | Runnable correctness/benchmark example; DeepSeek V4 Pro / Flash, GLM5.2, Kimi K3 and Qwen3.8 2.4T; SwiGLU / STEP / SiTU; FP4 pre-dispatch → fused GEMM1/activation/quantization → GEMM2/BF16 scatter → combine; replica weights |
| `test_moe_w4a16.py` | Small / Kimi K3 shapes, fused dispatch/G1/SiTU → GEMM2/BF16 scatter → combine; replica weights, complete EP8/16 pipelines |
| `test_moe_w8a8.py` | Channelwise GLM5.2 / GLM5.3 Flash / Qwen3.8 Flash / MiniMax M3, plus blockwise smoke; independent stage and pipeline references |

`moe_test_utils.py` contains process-group setup, common runtime helpers and
independent Torch references, including FP4 round-to-nearest-even quantization.
All-zero references have a defined zero error.

## Symmetric workspace

All precisions use one allocation API, called once on every rank of the group:

```python
buffer = dg.get_symm_buffer_for_mega_moe(
    group,
    num_experts=num_experts,
    num_max_tokens_per_rank=capacity,
    num_topk=topk,
    hidden=hidden,
    precision="w4a4",
    intermediate_hidden=2 * intermediate,
)
```

`precision` accepts `w4a4`, `w4a16`, `fp8_channelwise` and `fp8_blockwise`,
matching the compute APIs. W4 buffers require `intermediate_hidden` including
both gate and up projections. FP8 buffers describe transport only and reject
this argument; their scale layout is selected by `precision`.

Tile selection is handled internally. W4A4 accepts
`gemm1_block_n`, `gemm1_block_k` and `config_override`; W4A16 and FP8 accept
`block_m`. Omitting these options preserves the backend defaults and allows
measured tuning hints to apply. Explicit W4A16 `block_m` takes precedence over
the tuning table. Unsupported combinations fail before collective allocation.
The buffer owns a fixed-capacity arena and can be reused across token counts,
including calls with zero local tokens. Release it with `buffer.destroy()`
after all ranks have finished using it.

## Measured configurations and fallback

The model pipelines query a read-only C++ tuning configuration once per input
shape, before graph capture, and pass it to GEMM1, GEMM2/scatter and combine:

```python
tuning = dg.get_mega_moe_tuning_config(
    num_ranks=8, num_experts=384, num_topk=6, hidden=7168,
    intermediate_hidden=3072, num_tokens=1024,
    precision="w4a4", activation="swiglu")
# Pass tuning=tuning to mega_moe_dispatch_gemm1,
# mega_moe_gemm2_scatter and mega_moe_combine.
print(tuning.profile)  # deepseek_v4_pro; empty string means fallback
```

Here `intermediate_hidden` is the expert intermediate width, before doubling
for the gate/up projection. The table matches precision, activation, EP size,
expert count, top-k, hidden/intermediate dimensions and tokens per rank exactly.
It contains the measured W4A4, W4A16 and FP8 channelwise model configurations
at 1024, 2048 and 4096 tokens on ZW-M890P with 39 SMs. No profile is selected
for other token counts or devices. FP8 blockwise uses its existing heuristics.
Profiles cover quantized GEMM1 output (BF16 for W4A16) and BF16 scatter.

Profiles provide only measured overrides; unset fields use the existing
precision-specific heuristics. Explicit API configuration and environment
overrides take precedence. The read-only properties expose table hints, not
the fully resolved configuration after those overrides. Set
`DG_MOE_USE_TUNED_CONFIGS=0` before querying to disable the table. Calls without
`tuning` use the default selectors without measured overrides. Re-query
when the token count or model configuration changes; mismatched configurations
are rejected before launch. The query does not read routing tensors or copy
their contents to the CPU.

The table lives in `csrc/jit_kernels/heuristics/mega_moe_tuning.hpp`;
precision-specific fallback, override handling and validation remain in
`csrc/jit_kernels/heuristics/mega_moe.hpp`. Launchers consume the resulting
configuration without maintaining a separate model table.

## Four-stage compute interface

All precision examples call the same native functions:

```python
mega_moe_pre_dispatch(hidden_states, topk_ids, topk_weights, buffer, num_tokens,
                      precision=...)
mega_moe_dispatch_gemm1(buffer, rhs, out, rank, num_tokens,
                       precision=..., activation=..., output_quantization=...)
mega_moe_gemm2_scatter(lhs, rhs, token_meta, combine_buf, sym_offsets,
                      m_rows, valid_m_rows, rank, max_tokens, topk, precision=...)
mega_moe_combine(combine_buf, weights, num_tokens, hidden, topk, max_tokens,
                 flag_ptrs, rank, world_size, done_flag)
```

The `precision` selector is required for the first three stages. It must match
the workspace and preprocessed weights. `rhs` is always `(weight, scales)`;
GEMM2 `lhs` is a BF16 tensor for W4A16 and `(tensor, scales)` otherwise.
GEMM1 writes the caller-provided `out` and returns the dispatch row block size.

All precisions produce GEMM2 row layouts and packed routes inside dispatch/GEMM1.
Obtain the persistent views once with `buffer.get_gemm2_metadata()`; no separate
prepare call or caller-owned metadata output tensors are needed. W4A4 selects
GEMM2 padding internally, independently of the GEMM1 tile. Pass the same tuning
object to both stages so they use the same token count and selection policy:

```python
meta = buffer.get_gemm2_metadata()
tuning = dg.get_mega_moe_tuning_config(
    world_size, num_experts, topk, hidden, intermediate, num_tokens,
    precision="w4a4", activation="swiglu")
mega_moe_dispatch_gemm1(
    buffer, rhs, fp4_out, rank, num_tokens, precision="w4a4",
    activation="swiglu", output_quantization="mxfp4", out_scales=fp4_scales,
    tuning=tuning)
mega_moe_gemm2_scatter(
    (fp4_out, fp4_scales), rhs2, meta.packed_src_info, combine_buf, sym_offsets,
    meta.m_rows, meta.valid_m_rows, rank, capacity, topk, precision="w4a4",
    row_offsets=meta.src_m_offsets, tuning=tuning)
```

The supported W4A4 GEMM2 M tiles are 16, 32, 64, 128 and 256. For MXFP4 GEMM1
output, omitted or zero `gemm2_block_m` selects the GEMM2 tile automatically;
without `tuning`, both stages select it from workspace capacity. BF16 GEMM1
output retains the GEMM1 padding. Explicit GEMM2 `configs` or `expected_m`
overrides must be paired with a matching GEMM1 `gemm2_block_m`. BF16 and FP8
scatter use the same default padding. Other precisions use their dispatch
row layout. Metadata is recomputed for current routing on every launch and
Graph replay, including empty ranks. Activation, rounding and router-weight
placement retain each precision's numerical contract.
Only `valid_m_rows` rows per expert contain defined activation and route data;
padding must not be consumed as a routed row.


| `precision` | GEMM1 `activation` | `output_quantization` | GEMM2 scatter output |
| --- | --- | --- | --- |
| `w4a4` | `None`, `swiglu`, `step`, `situ` | `None` (BF16); `mxfp4` requires an activation | BF16 or FP8 |
| `w4a16` | `situ` | `None` (BF16) | BF16 |
| `fp8_channelwise` | `swiglu`, `swiglu_oai` | `None` (BF16) or `fp8` (per-token scales) | BF16 |
| `fp8_blockwise` | `None` | `None` (raw BF16 projection) | BF16 |
| `fp8_blockwise` | `swiglu` | `fp8` (128-element block scales) | BF16 |

Unsupported combinations raise before dispatch; the selector does not silently
change activation or quantization. Optional `activation_limit` is a positive
finite clamp; use `None` to disable it. `activation_alpha` is accepted only for
`swiglu_oai` (default 1.702). SiTU uses `situ_beta=4` and
`situ_linear_beta=25`. FP8 blockwise does not support SwiGLU-OAI.

Quantized GEMM1 output requires `out_scales`. The channelwise path also requires
`bf16_scratch`, preserving the existing fused/separate-quantization scheduling
without a hidden large allocation. W4A16 GEMM1 currently does not accept split
replica weights; the other GEMM1 paths and all GEMM2 paths
accept `rhs_replicas=(weight, scales)`. W4A4 GEMM1 takes only the tile-packed
uint16 scale layout, prepared once as shown in `_pack_gemm1_scales`.

For example, channelwise FP8 with SwiGLU-OAI and quantized GEMM1 output uses:

```python
dg.mega_moe_dispatch_gemm1(
    buffer, (w13, w13_scales), quant_output, rank, num_tokens,
    precision="fp8_channelwise", activation="swiglu_oai",
    output_quantization="fp8", out_scales=quant_scales,
    bf16_scratch=activated, activation_limit=7.0, activation_alpha=1.702)
```

GEMM2 requires `m_indices` for channelwise FP8 and `row_offsets` for blockwise
FP8. W4A4 uses `row_offsets` when GEMM1 and GEMM2 have different row padding.
`configs` retains the existing backend tile tuple; blockwise FP8 uses its built-in
tile. Options specific to a different backend are rejected. See each precision
file for the workspace allocation and complete call chain. W4A16 references use
Torch tensors and ordinary grouped GEMM; its pipeline uses only public APIs.
Blockwise FP8 retains a private dispatch-only helper for its staged GEMM1 reference.

Workspace layout and scatter tile queries are internal implementation details.
Use `get_symm_buffer_for_mega_moe` for allocation and
`get_mega_moe_tuning_config` for measured workload presets with heuristic fallback.

The metadata views contain `m_rows`, `valid_m_rows`, `m_offsets`, `src_m_rows`,
`src_m_offsets` and `packed_src_info`. When W4A4 source and destination padding
differ, pass `meta.src_m_offsets` as GEMM2 `row_offsets`. Activation tensors stay
in their source layout. W4A16 and FP8 callers use the same metadata accessor;
GEMM1 no longer accepts separate `m_rows` or `packed_src` output parameters.

The primary model matrix uses **1024, 2048 and 4096 input tokens per rank**,
defined by `MODEL_TOKENS` in `moe_test_utils.py`. These are generated MoE layer
inputs/weights with model dimensions, not checkpoint-based full-model tests.

| Model / case ID | GEMM precision | EP | Activation | Workspace capacity per rank |
| --- | --- | --- | --- | --- |
| DeepSeek V4 Pro / `deepseek_v4_pro` | W4A4 (MXFP4) | 8 | SwiGLU, limit 10 | 8192 |
| DeepSeek V4 Flash / `deepseek_v4_flash` | W4A4 (MXFP4) | 4 | SwiGLU, limit 10 | 4096 |
| GLM5.2 / `glm52` | W4A4 (MXFP4) | 8 | SwiGLU | 4096 |
| GLM5.2 / `glm52` | W8A8 FP8 channelwise | 8 | SwiGLU | 4096 |
| GLM5.3 Flash / `glm53_flash` | W8A8 FP8 channelwise | 4 | SwiGLU, limit 10 | 4096 |
| Kimi K3 / `kimi_k3` | W4A4 (MXFP4) | 16 | SiTU | 4096 |
| Kimi K3 / `kimi_k3_compute_*` | W4A16 (FP4 weights, BF16 activations) | 16 | SiTU | 4096 |
| Qwen3.8 2.4T / `qwen38_2_4t` | W4A4 (MXFP4) | 16 | SwiGLU | 4096 |
| Qwen3.8 Flash / `qwen38_flash` | W8A8 FP8 channelwise | 4 | SwiGLU | 8192 |
| MiniMax M3 / `minimax_m3` | W8A8 FP8 channelwise | 8 | SwiGLU-OAI, alpha 1.702, limit 7 | 4096 |

MiniMax M3 follows the checkpoint
`MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken`: its weights use static
per-channel FP8 scales, its activations use dynamic per-token FP8 scales, and
`hidden_act=swigluoai`. The model case preserves this quantization and activation
contract. Blockwise FP8 remains covered by the generic smoke pipeline.

All rows run the three primary token counts. Capacity is the buffer limit, not
the number of tokens passed to each forward. Small/empty/masked-route cases
remain in the smoke tests. W4A16 validates routing and landed token bytes inside
its public fused pipeline; there is no separate transport-only case. This table
describes configured coverage, not a record that every case has passed on hardware.

Run from the repository root in a PPU runtime, after assigning free devices:

```bash
export PYTHONPATH="$PWD:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
# Single-device tests: no fixed rendezvous port is needed.
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_moe_w4a4.py tests/test_moe_w4a16.py tests/test_moe_w8a8.py -k smoke -q
# See the full matrix without allocating distributed buffers.
python -m pytest tests/test_moe_w8a8.py --collect-only -q
# EP2 W4A4 and blockwise W8A8 pipelines, including remote BF16 scatter.
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29876 \
  -m pytest tests/test_moe_w4a4.py tests/test_moe_w8a8.py \
  -k 'smoke and (test_pipeline or blockwise_pipeline)' -q
# EP8 channelwise MiniMax M3 pipeline at 1024/2048/4096 tokens per rank.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
  --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29876 \
  -m pytest tests/test_moe_w8a8.py -k 'channelwise_pipeline and minimax_m3' -q
# EP4 channelwise GLM5.3 at the same three token counts.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29876 \
  -m pytest tests/test_moe_w8a8.py -k 'channelwise_pipeline and glm53_flash' -q
```

This suite targets BF16 scatter/combine. FP8 scatter/combine remains implemented
but is outside this test matrix for now. W4A4 Kimi covers BF16 combine with top-k 16.
Scatter and combine are checked within the precision pipelines, without separate
component test entrypoints.

Launch the same pytest selection on every rank. Tests whose model requires a
different EP size are skipped explicitly. Kimi W4A4/W4A16 and Qwen3.8 2.4T W4A4
model cases require a two-node EP16 launch. The single-rank W4A16 `situ_large_shape` case uses four
experts/top-k 2 for an additional shape check; it is not a Kimi model case.
W4A4 smoke H is at least 512 because dispatch copies
scale data in 16-byte units. Large model cases can allocate substantial weight
and routing buffers: select `smoke` for a quick check.

W4A16 and W8A8 currently expose pytest correctness entrypoints. W4A4 also has a
standalone runner for correctness and performance, sharing the same `forward()`
with pytest. Existing dense GEMM and attention entrypoints are unchanged.

## W4A4 end-to-end example and benchmark

`W4A4MoE.forward()` in `test_moe_w4a4.py` shows the complete call sequence:

```text
BF16 hidden states + top-k IDs/weights
  -> FP4 pre-dispatch
  -> dispatch + GEMM1 + activation + weighted FP4 quantization
  -> GEMM2 row/metadata preparation
  -> GEMM2 + BF16 scatter
  -> dispatch count cleanup
  -> mega_moe_combine (cross-rank barrier + weighted reduce)
  -> BF16 output
```

All precision paths use `deep_gemm.mega_moe_combine` after GEMM2/scatter on the
same stream. The scatter kernels finish their local/remote stores; only
`mega_moe_combine` waits for peer ranks and reduces the local combine buffer.
There is no separate `ipc_barrier` or local-only reduce entrypoint.

The former standalone IPC-barrier call sites are covered in the precision tests:

| Test | Coverage |
| --- | --- |
| `test_moe_w4a16.py::test_pipeline` | EP1, H/I=256/128 and 3584/3072, four tokens, fused GEMM2/scatter immediately followed by reduce, replica weights, poisoned destinations and changed-input graph replay |
| `test_moe_w8a8.py::test_channelwise_scatter_combine[default_ep2]` | EP2, M/N/K=257/512/2048, partial BM256 tile, local/remote BF16 scatter and weighted reduce |
| `test_moe_w8a8.py::test_channelwise_scatter_combine[overlap_padded_ep2]` | EP2, M/N/K=512/512/2048, four groups with 128 physical rows and valid prefixes 127/65/63/1, BM64 `OverlapPrologue`, zero padding metadata |

Both EP2 cases check every received element, untouched rows, allocation guards,
and barrier generations over eight changing-input/routing rounds, including
graph replay. The device consumer runs directly after scatter; host barriers
only protect buffer reuse between rounds.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_moe_w4a16.py::test_pipeline
CUDA_VISIBLE_DEVICES=0,1 torchrun --master-addr=127.0.0.1 --master-port=29876 \
  --nproc-per-node=2 -m pytest -q tests/test_moe_w8a8.py::test_channelwise_scatter_combine
```

The combine buffer is `[topk, max_tokens, hidden]` (BF16 or E4M3 FP8); weights
are contiguous FP32 `[num_tokens, topk]`. Invalid routes must have zero weights;
the reducer skips those slots without reading them. FP8 may additionally pass
`combine_scales` (packed BF16 scale in the low 16 bits of each int64) and
`scale_block_n`. Without side scales, the FP8 exponent shift must match scatter.
For W4A4 FP8 scatter, set `output_quantization="fp8"` on `mega_moe_gemm2_scatter`
and pass a `torch.float8_e4m3fn` combine buffer to both scatter and combine.
BF16 and FP8 share the synchronization and reduction implementation and return
BF16. All scatter producers and combine use this slot-major layout; the former
`DG_MOE_COMBINE_TOKEN_MAJOR` override has been removed. W4A4/W4A16 scatter requires
complete N tiles with `BLOCK_N` divisible by 256 because BF16 bulk stores involve
a full warp. FP8 side scales use the supplied grouping without device-side
clamping: `scale_block_n` must be at least 64, aligned to 8, divide `BLOCK_N`, and
produce at most four groups per tile. Explicit scale tensors are honored even
if legacy environment overrides disagree.

`test_moe_w4a4.py::test_fp8_combine_pipeline` covers fixed-scale FP8 and static /
dynamic side scales with 64/128/256-column groups, invalid groups and tiles,
poisoned scale storage, adjacent scatter/combine and graph replay. These optional
FP8 checks share the W4A4 end-to-end setup; model benchmarks still use BF16 scatter.

`flag_ptrs` is a device int64 table of one peer-visible int32 counter per rank,
initialized to zero once. `done_flag` is one local int32, reset by the launcher
on the same stream for each invocation (also during graph replay). The tests'
`CombineBarrier` helper owns these buffers. Every rank must invoke the collective
in the same order, including `num_tokens=0`; each concurrent MoE invocation needs
its own state. Keep buffers alive and do not reset counters while calls are in
flight. Before reusing a combine buffer, all readers of the previous iteration
must finish: the next MegaMoE dispatch supplies this synchronization in the full
pipeline. Callers using only scatter/reduce must arrange that reuse boundary.

The router and shared-expert branch are outside this routed MoE example. The
workspace and preprocessed local expert weights are reused across invocations.
The `W4A4MoE` class is an example in this test file, not an additional library API.
To reuse it with your tensors after initializing a process group:

```python
from test_moe_w4a4 import W4A4MoE

# w1/w2: local logical expert weights, each (packed FP4, raw E8M0 scales).
# w1 rows are [gate; up]. Inputs are contiguous CUDA tensors with dtypes
# BF16, int64 and float32 for hidden_states, topk_ids and topk_weights.
moe = W4A4MoE(case, group, w1, w2)
moe.set_inputs(hidden_states, topk_ids, topk_weights)
moe.compile()  # Finish JIT on every rank before cooperative kernels run.
output = moe.forward()
# Reuse forward(); after binding new inputs, call compile() again on all ranks.
torch.cuda.synchronize()
moe.close()
```

Run from the built repository root in a PPU runtime. Direct script execution
loads this checkout. Keep the distributed environment above for torchrun.

```bash
# Single PPU: complete pipeline checked against independent Torch references.
CUDA_VISIBLE_DEVICES=0 python tests/test_moe_w4a4.py \
  --case smoke_swiglu --tokens 33

# Two PPUs: empty rank/all-empty calls, masked slots and split replica weights.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tests/test_moe_w4a4.py --case smoke_situ --tokens 1 0 33 \
  --edge-cases --replicas

# Eight PPUs: all tokens checked at the three primary sizes, then graph timing.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 \
  tests/test_moe_w4a4.py --case deepseek_v4_pro --tokens 1024 2048 4096 --benchmark \
  --warmup 10 --iterations 100 --repeats 5

# Eager timing, or performance only after correctness has been established.
CUDA_VISIBLE_DEVICES=0 python tests/test_moe_w4a4.py \
  --case smoke_step --tokens 33 --benchmark --mode eager
CUDA_VISIBLE_DEVICES=0 python tests/test_moe_w4a4.py \
  --case smoke_swiglu --tokens 33 --benchmark --no-check
```

The CLI checks every requested token by default. `--reference-tokens N` explicitly
limits numerical reference checks to the first N tokens per rank; route counts
and metadata are still checked for all routes. Pytest keeps bounded references
for large model cases and checks every smoke token. Both use the independent FP4
full-pipeline reference and a GEMM2/scatter/combine reference based on actual
GEMM1 output, so activation quantization error cannot hide scatter/reduce errors.

Benchmark output is one JSON object per token count on rank 0. It records model
shape, EP size, actual token counts on all ranks, workspace capacity, correctness
status/errors, execution mode, timing samples and global token throughput.
`benchmark.max_rank_us` reports min/median/max across samples; **each sample is
the maximum across ranks of their mean forward latency**, in microseconds.
Throughput uses the sum of actual tokens across ranks and that median latency.

JIT, weight/input generation, CPU layout selection, reference computation,
validation and host barriers are outside timing. Every timed invocation includes
all kernels from pre-dispatch through combine, including device synchronization.
Graph mode captures a batch of `--iterations` complete forwards; eager mode calls
the same forward from Python and its device-event time includes launch gaps.
All ranks use identical warmup/iteration/repeat counts. Defaults use round-robin
routing without injected edge cases; `--edge-cases` and `--replicas` are explicit
and recorded in the result. `--capacity` changes the workspace capacity without
changing the token count, so keep it fixed when comparing measurements.

MoE warmup uses `compile_kernels` from `moe_test_utils.py` to compile and load
native kernels serially without launching them. The helper restores the previous
compile mode and propagates compiler failures. All ranks finish warmup before
entering cooperative device barriers.
