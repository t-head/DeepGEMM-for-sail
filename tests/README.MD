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
# Routed MoE correctness and performance tests

The MoE tests share one end-to-end runner between pytest and the command line.
Select precision by file and model by `--case` or pytest ID; model dimensions, activation, routing limits and EP sizes
are declared in each precision file. Add a model to those tables instead of
copying a test script.

| File | Coverage |
| --- | --- |
| `test_moe_w4a4.py` | Runnable correctness/benchmark example; DeepSeek V4 Pro / Flash, GLM5.2, Kimi K3, Qwen3.8 2.4T and MiMo V2.6 Pro; SwiGLU / STEP / SiTU; FP4 pre-dispatch → fused GEMM1/activation/quantization → GEMM2/BF16 scatter → combine; replica weights |
| `test_moe_w4a16.py` | Small / Kimi K3 shapes, fused dispatch/G1/SiTU → GEMM2/BF16 scatter → combine; replica weights, EP8/16/32 and EP32 decode cases |
| `test_moe_w8a8.py` | Channelwise GLM5.2 / GLM5.3 Flash / Qwen3.8 Flash / MiniMax M3; blockwise SwiGLU smoke and MiniMax SwiGLU-OAI shape at EP8/16/32/64; complete pipelines with independent Torch references |

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
For FP8 channelwise, `block_m` selects both dispatch granularity and the GEMM1
tile (128 or 256). FP8 blockwise keeps dispatch/padding at 128 rows and selects
the GEMM1 compute tile independently (128 or 192 for FP8 output).
The buffer owns a fixed-capacity arena and can be reused across token counts,
including calls with zero local tokens. Release it with `buffer.destroy()`
after all ranks have finished using it.

Peer-visible allocations use `torch.distributed._symmetric_memory` directly.
Blockwise FP8 keeps its expert landing pools in local Torch allocations;
input staging and cross-rank control data remain symmetric.

For W4A16 decode, `num_tokens_per_rank` optionally selects a common workload
bucket at allocation time. Every rank must supply the same bucket, even when
their actual input counts differ. The Kimi K3 EP32 table matches exactly
4, 8, 16, 32, 64 and 128 tokens, with 896 physical experts or 928 with replicas.
These profiles use buffer BM16, except the 32-token profile uses BM32.
Unmatched or disabled profiles use BM64; explicit `block_m` overrides this
choice. Capacity remains the upper bound on input length. The W4A16 test
selects allocation buckets from the table without model-name branches.
The decode profiles launch dispatch and GEMM1 separately so they can
use independent thread counts and resident CTA counts. Row and route preparation
remains fused into dispatch, and the public pipeline calls stay the same.

W4A16 pre-dispatch copies BF16 tokens, routing IDs and weights in one launch.
Upstream producers may instead write directly to the three tensors returned by
`buffer.input_views(num_tokens)` and pass those views to `mega_moe_pre_dispatch`.
An input that already aliases its destination is not copied. Keep the buffer
alive and order writes and the pipeline on the same stream. W4A16 resets
consumed dispatch counters internally; read valid rows from
`buffer.get_gemm2_metadata().valid_m_rows`, rather than retaining receive counters
after dispatch/GEMM1.

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
at 1024, 2048 and 4096 tokens on ZW-M890P with 39 SMs. MiMo V2.6 Pro has an
additional W4A4 EP8 profile at exactly 8192 tokens per rank, with GEMM2 tile
`128,256,64,64,64,3`; its other stages use the default selectors. Each profile
declares its token counts explicitly, including the six K3 EP32 decode counts.
Unlisted counts or devices fall back to
the heuristics. FP8 blockwise uses its existing heuristics.
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

For decode, `buffer_block_m` describes collective allocation;
`gemm1_tile`, `gemm1_separate_dispatch` and `dispatch_threads` describe the
compute and communication launches. `gemm2_tile` describes the scatter launch.
For W4A4, `gemm2_m_tiles_per_group` controls the existing GEMM2 scheduler:
`1` traverses all N tiles of one M tile before the next; `None` retains the
default interleaving of two M tiles. Only measured profiles override this field.
Combine always reduces one complete token row per CTA, including a synchronization
CTA on empty ranks. It uses 256 threads for at least 512 local tokens, 1024 threads
for at most 64 local tokens with hidden size above 4096, and 512 threads otherwise.
This policy uses the actual combine token count, independently of the tuning table.
`gemm1_dynamic_queue=None` selects the backend's automatic policy;
`True` and `False` explicitly enable and disable it. In particular, blockwise
FP8 cannot override an explicit `False` with its automatic queue heuristic.
The automatic BM128/BM192 and queue thresholds are empirical fallback choices,
not hardware requirements.

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

All precisions produce GEMM2 row layouts and routing metadata inside dispatch/GEMM1.
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
    (fp4_out, fp4_scales), rhs2, meta.token_meta, combine_buf, sym_offsets,
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
Graph replay, including empty ranks. All precisions apply routing weights
after GEMM2, during combine.
Only `valid_m_rows` rows per expert contain defined activation and route data;
padding must not be consumed as a routed row.


| `precision` | GEMM1 `activation` | `output_quantization` | GEMM2 scatter output |
| --- | --- | --- | --- |
| `w4a4` | `None`, `swiglu`, `step`, `situ` | `None` (BF16); `mxfp4` requires an activation | BF16 or FP8 |
| `w4a16` | `situ` | `None` (BF16) | BF16 |
| `fp8_channelwise` | `swiglu`, `swiglu_oai` | `None` (BF16) or `fp8` (per-token scales) | BF16 |
| `fp8_blockwise` | `None` | `None` (raw BF16 projection) | BF16 |
| `fp8_blockwise` | `swiglu`, `swiglu_oai` | `fp8` (128-element block scales) | BF16 |

Unsupported combinations raise before dispatch; the selector does not silently
change activation or quantization. Optional `activation_limit` is a positive
finite clamp; use `None` to disable it. `activation_alpha` is accepted only for
`swiglu_oai` (default 1.702). SiTU uses `situ_beta=4` and
`situ_linear_beta=25`.

Quantized GEMM1 output requires `out_scales`. The channelwise path also requires
`bf16_scratch`, preserving the existing fused/separate-quantization scheduling
without a hidden large allocation. All GEMM1 and GEMM2 paths accept
`rhs_replicas=(weight, scales)`. Physical expert slots are ordered as masters
followed by replicas on each rank; their total must match the workspace. W4A4 GEMM1 takes only the tile-packed
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
FP8. W4A4 requires `row_offsets=meta.src_m_offsets` for the activation and
scale inputs, which retain GEMM1 row padding. Packed routes follow GEMM2 padding.
`configs` retains the existing backend tile tuple; blockwise FP8 selects BM128
or BM192 internally, using `tuning.gemm2_tile` when supplied by a profile.
Blockwise GEMM1 weights must be prepared with
`interleave_fp8_blockwise_mega_moe_weights`: it interleaves 128 gate rows and
128 up rows and retains native `[experts, N/128, K/128]` scales. Previously
cached G8 weights and expanded `[experts, N/8, K/128]` scales must be regenerated
from logical weights and native scales; the old scale shape is rejected.
Options specific to a different backend are rejected. See each precision
file for the workspace allocation and complete call chain. Each pipeline uses
public APIs. References compute the entire MoE from source inputs, routing and
logical expert weights with Torch, without consuming kernel intermediates.

Workspace layout and scatter tile queries are internal implementation details.
Use `get_symm_buffer_for_mega_moe` for allocation and
`get_mega_moe_tuning_config` for measured workload presets with heuristic fallback.

The metadata views contain `m_rows`, `valid_m_rows`, `m_offsets`, `src_m_rows`,
`src_m_offsets` and `token_meta`. All precisions use packed int64 `[pool]`
routes in GEMM2 row order: rank in bits 63:32, top-k slot in bits 31:24,
and token in bits 23:0. Dispatch/GEMM1 writes these routes together with the
row metadata; `get_gemm2_metadata()` only returns views and launches no kernel.
Activation tensors stay in their source layout. All precisions use the same
metadata accessor; GEMM1 no longer accepts separate `m_rows` or `packed_src`
output parameters.

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
| Kimi K3 / `kimi_k3` | W4A4 (MXFP4) | 8, 16 | SiTU | 4096 |
| Kimi K3 / `kimi_k3` | W4A16 (FP4 weights, BF16 activations) | 8, 16, 32 | SiTU | 4096 |
| Qwen3.8 Max (2.4T) / `qwen38_2_4t` | W4A4 (MXFP4) | 8, 16 | SwiGLU | 4096 |
| Qwen3.8 Flash / `qwen38_flash` | W8A8 FP8 channelwise | 4 | SwiGLU | 8192 |
| MiniMax M3 / `minimax_m3` | W8A8 FP8 channelwise | 8 | SwiGLU-OAI, alpha 1.702, limit 7 | 4096 |
| MiniMax blockwise shape / `minimax_m3_blockwise` | W8A8 FP8 blockwise | 8, 16, 32, 64 | SwiGLU-OAI, alpha 1.702, limit 7 | 4096 |
| MiMo V2.6 Pro / `mimo_v26_pro` | W4A4 (MXFP4) | 8 | SwiGLU | 8192 |

MiniMax M3 follows the checkpoint
`MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken`: its weights use static
per-channel FP8 scales, its activations use dynamic per-token FP8 scales, and
`hidden_act=swigluoai`. The model case preserves this quantization and activation
contract. The separate `minimax_m3_blockwise` case uses H7168, I5120, 384
experts and top-k 4; it is not a precision toggle for the channelwise case.
Its token counts include 64 plus the primary matrix. The W4A16 `kimi_k3`
case includes 4, 8, 16, 32, 64 and 128 tokens per rank in addition to prefill.

All model rows run the three primary token counts; MiMo also runs 8192 tokens
per rank to cover its tuned profile. Capacity is the buffer limit, not the
number of tokens passed to each forward. The small smoke cases use
1, 0, 33 and 8 tokens to cover empty ranks, all-empty calls and partial tiles.
This table describes configured coverage, not a record that every case has
passed on hardware. These tests exercise BF16 scatter/combine; optional FP8
scatter/combine remains implemented but is outside this test matrix.

## Running correctness and performance tests

All three precision files accept the same command-line options. The default
runs the complete pipeline and checks its BF16 output against an independent
Torch reference. Add `--benchmark` to measure that same pipeline.

Run from the built repository root after selecting free devices:

```bash
export PYTHONPATH="$PWD:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0

# Small complete pipelines, including graph replay and separate replica weights.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m pytest tests/test_moe_w4a4.py tests/test_moe_w4a16.py tests/test_moe_w8a8.py \
  -k smoke -q
# Channelwise FP8 smoke requires EP4 or EP8.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  tests/test_moe_w8a8.py --case smoke_channelwise --tokens 1 0 33 8 \
  --edge-cases --replicas --benchmark

# DeepSeek V4 Pro: correctness and full-pipeline graph timing.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc-per-node=8 tests/test_moe_w4a4.py \
  --case deepseek_v4_pro --tokens 1024 2048 4096 \
  --benchmark --warmup 10 --iterations 100 --repeats 5

# MiMo V2.6 Pro: fallback and measured GEMM2 profile.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc-per-node=8 tests/test_moe_w4a4.py \
  --case mimo_v26_pro --tokens 1024 2048 4096 8192 \
  --benchmark --warmup 10 --iterations 100 --repeats 5

# FP8 uses the same interface.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  tests/test_moe_w8a8.py --case glm53_flash --tokens 1024 2048 4096 --benchmark

# W4A16 single-device smoke; add --replicas to exercise separate weight buffers.
CUDA_VISIBLE_DEVICES=0 python tests/test_moe_w4a16.py \
  --case smoke --tokens 1 0 33 8 --edge-cases --replicas --benchmark

# Inspect configured cases without creating a process group.
python -m pytest tests/test_moe_w4a4.py tests/test_moe_w4a16.py tests/test_moe_w8a8.py \
  --collect-only -q
```

Kimi K3 and Qwen3.8 Max support EP8 on one eight-device node with
`--nproc-per-node=8`. For EP16, on each of two eight-device nodes,
use the same command and set `NODE_RANK` to 0 or 1 and `MASTER_ADDR` to the
first node's reachable address:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29876 \
  tests/test_moe_w4a16.py --case kimi_k3 --tokens 1024 2048 4096 --benchmark
```

Use the communication environment appropriate to the machine topology on both
nodes. Pytest skips model cases whose required EP does not match the launch;
the CLI reports an error for an incompatible `--case`.

For Kimi K3 decode, use four eight-device nodes with `--nnodes=4` and
`--node-rank=0..3`, then select
`--case kimi_k3 --tokens 4 8 16 32 64 128 --benchmark`.
For MiniMax blockwise, use `tests/test_moe_w8a8.py --case minimax_m3_blockwise`;
two, four or eight nodes provide EP16, EP32 or EP64. All participating devices
must support Torch symmetric-memory peer access within the same fabric domain.

All precisions apply routing weights in `mega_moe_combine`, after GEMM2.
Pass the FP32 routing weights with invalid routing IDs masked to zero.
GEMM1 activation and output quantization operate on unweighted values.
The W4A4 example uses `buffer.input_topk_weights[:num_tokens]` for combine-stage
weighting: pre-dispatch refreshes this view and zeros invalid routes on every
launch. This keeps changing router weights visible during graph replay without
an additional masking kernel.

The CLI checks every requested token by default. `--reference-tokens N` limits
the numerical comparison to the first N tokens on each rank while still running
the complete workload. Pytest uses bounded references for large model cases and
checks every smoke token. Correctness also checks finite output, a poisoned
combine buffer and repeated graph replay with changed inputs.

`--edge-cases` injects masked routes, zero routing weights and an empty final
rank when `--tokens` includes 1. `--replicas` places the final local physical
expert in a separate allocation for both GEMMs. For W4A16 Kimi K3, it retains
896 masters globally and adds one replica slot per rank (912 physical experts
at EP16 or 928 at EP32). Other cases split their existing physical slots. Pytest exercises both
ordinary and replica weights where the case has enough local experts.

Performance results are JSON objects printed by rank 0, one per token count.
They include precision, shape, EP, actual tokens per rank, replica/edge options,
correctness errors and timing samples. `benchmark.max_rank_us` gives
min/median/max across samples; each sample is the maximum across ranks of their
mean forward latency in microseconds. Throughput uses the sum of actual tokens
and the median latency.

Timing includes all device work from pre-dispatch through combine, including
cross-rank synchronization. JIT, weight/input generation, reference computation,
validation and host barriers are outside timing. Graph mode captures
`--iterations` complete forwards; `--mode eager` measures the same calls with
Python launch gaps. `--no-check` is allowed only with `--benchmark`, for timing
after correctness has already been established. Keep `--capacity`, routing,
EP, token counts and timing mode fixed when comparing results.

## Pipeline and synchronization

Every precision example uses the public pre-dispatch, dispatch/GEMM1,
GEMM2/scatter and combine APIs in sequence on one stream. GEMM1 also prepares
GEMM2 metadata. Scatter completes its local/remote stores; combine performs
the cross-rank barrier and reduction. There is no separate IPC barrier call.
All precisions multiply each valid route by its routing weight during combine.

The BF16 combine buffer is `[topk, capacity, hidden]`. Invalid routes must have
zero combine weights so the reducer skips unwritten slots. `CombineBarrier`
owns one peer-visible generation counter per rank and one local completion
flag. All ranks, including empty ranks, invoke the pipeline in the same order.
Keep these buffers alive across graph replay. The next dispatch synchronizes
buffer reuse; each concurrent MoE invocation needs its own state.

`compile_kernels` compiles and loads native kernels without launching them,
then all ranks synchronize before entering cooperative device barriers.
The router and shared-expert branch are outside these routed MoE examples.

## Reusing the W4A4 example

`W4A4MoE` in `test_moe_w4a4.py` is a reusable example workspace, not another
library API. It preprocesses logical GEMM1 weights and raw scales into groups
of eight gate rows followed by eight up rows: `g0,...,g7,u0,...,u7,g8,...`.
SwiGLU, STEP and SiTU use this G8 layout for BF16 and MXFP4 output.
Gate/up projections and their clamps are rounded to BF16; activation arithmetic
uses FP32, then rounds to BF16 before MXFP4 quantization. Routing weights are
applied during combine. Library callers must apply the same G8
row permutation to weights and raw E8M0 scales before packing scales into
GEMM1 N tiles; see `interleave_gate_up` and `_pack_gemm1_scales` in the example.
Previously prepared G1 weights and scales must be rebuilt in G8 order.

After initializing a process group:

```python
from test_moe_w4a4 import W4A4MoE

# w1/w2: (packed FP4, raw E8M0 scales) for local logical expert weights.
# w1 rows are [gate; up]. Input dtypes are BF16, int64 and float32.
moe = W4A4MoE(case, group, w1, w2)
moe.set_inputs(hidden_states, topk_ids, topk_weights)
moe.compile()
output = moe.forward()
# Call compile() on all ranks again after binding new input shapes.
torch.cuda.synchronize()
moe.close()
```
