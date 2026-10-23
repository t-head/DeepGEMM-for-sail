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

MegaMoE tests target ZW-M890P. Device checks use the
`is_ppu1v5_device()` architecture predicate. On other architectures, pytest
skips the MoE tests and direct CLI execution reports an error before process-group
initialization. CI runs these tests only in the ZW-M890P suites; the OAM-810E
suite runs shared-operator regressions.

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

Run the shared end-to-end smoke cases on two idle ZW-M890P devices:

```bash
torchrun --standalone --nproc-per-node=2 -m pytest -q -s -k smoke \
  tests/test_moe_w4a4.py tests/test_moe_w4a16.py tests/test_moe_w8a8.py
```

Use `--nproc-per-node=8` for EP8. Each smoke case reuses its workspace across
1, 0, 33, 129 and 8 tokens, including an empty last rank in the one-token case.
The same precision-specific weight preparation and independent Torch reference
serve the CLI, pytest and synchronization checks; every smoke token is checked.
The shared replay helper changes inputs, routes and routing weights, including
next-rank and last-rank destinations, random routes, hot experts and fully inactive routes.
Before each checked replay or eager call, receive and output buffers are poisoned
with NaN and one rotating rank starts late. This checks that an empty source rank
still computes remote expert work. No host barrier is inserted between scatter
and combine. Inactive routes must produce exact zeros.

The normalized squared-error thresholds are 0.003 for W4A4, 0.0005 for W4A16
and 0.001 for FP8, matching the existing FP8 grouped contiguous/masked GEMM
tests. W4A16 also requires relative L2 below 0.005. Replay outputs are saved
before eager execution and compared bitwise with it, then checked against the
independent reference.

CI runs these cases in the separate `ZW-M890P-EP2` suite with two devices in one
worker pod. The launcher checks device count and architecture before testing.
Single-device board suites retain their existing commands and separate reports.

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
    intermediate_hidden=intermediate,
)
```

`precision` accepts `w4a4`, `w4a16`, `fp8_channelwise` and `fp8_blockwise`,
matching the compute APIs. Every precision requires `intermediate_hidden=I`,
the activated intermediate width; GEMM1 width 2I is derived internally. The
workspace owns both stages' scratch, route metadata, combine and barrier state.

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

The complete MoE entries select and cache tuning configurations internally
from the input dimensions, precision, activation and actual token count.
Callers do not construct or pass tuning objects.

Here `intermediate_hidden` is the expert intermediate width, before doubling
for the gate/up projection. The table matches precision, activation, EP size,
expert count, top-k, hidden/intermediate dimensions and tokens per rank exactly.
It contains the measured W4A4, W4A16 and FP8 channelwise model configurations
at 1024, 2048 and 4096 tokens on ZW-M890P with 39 SMs. MiMo V2.6 Pro has an
additional W4A4 EP8 profile at exactly 8192 tokens per rank, with GEMM2 tile
`128,256,64,64,64,3`; its other stages use the default selectors. Each profile
declares its token counts explicitly, including the six K3 EP32 decode counts.
Unlisted counts or devices fall back to
the heuristics. FP8 blockwise uses its backend heuristics.
Profiles cover quantized GEMM1 output (BF16 for W4A16) and BF16 scatter.

Profiles provide only measured overrides; unset fields use the
precision-specific heuristics. Explicit workspace options and supported
environment overrides take precedence. Set `DG_MOE_USE_TUNED_CONFIGS=0`
before workspace allocation to disable measured profiles. Internal selection
does not read routing tensors or copy their contents to the CPU.

Decode profiles specify the collective buffer tile, GEMM1 tile and dispatch
launch geometry, plus GEMM2 and combine settings. W4A4 profiles can traverse
all N tiles of one M tile before moving to the next; the fallback interleaves
two M tiles. Only measured profiles override this order.
Combine always reduces one complete token row per CTA, including a synchronization
CTA on empty ranks. It uses 256 threads for at least 512 local tokens, 1024 threads
for at most 64 local tokens with hidden size above 4096, and 512 threads otherwise.
This policy uses the actual combine token count, independently of the tuning table.
Profiles can enable or disable the GEMM1 dynamic queue internally; an unset
policy uses the backend heuristic. Blockwise FP8 respects an explicit profile
choice before applying its automatic queue heuristic.
The automatic BM128/BM192 and queue thresholds are empirical fallback choices,
not hardware requirements.

The table lives in `csrc/jit_kernels/heuristics/mega_moe_tuning.hpp`;
precision-specific fallback, override handling and validation remain in
`csrc/jit_kernels/heuristics/mega_moe.hpp`. Launchers consume the resulting
configuration without maintaining a separate model table.

## Complete compute interface

All precision examples allocate one reusable library workspace, transform both
weight pairs, then call pre-dispatch and one complete output-first entry:

```python
w1, w2 = dg.transform_weights_for_mega_moe(w1_raw, w2_raw, precision="w4a4")
buffer = dg.get_symm_buffer_for_mega_moe(
    group, num_experts, capacity, topk, hidden,
    precision="w4a4", intermediate_hidden=intermediate)
y = torch.empty_like(hidden_states)

dg.mega_moe_pre_dispatch(hidden_states, topk_ids, topk_weights, buffer)
dg.fp4_fp4_mega_moe(y, w1, w2, buffer, activation="swiglu")
```

Use `bf16_fp4_mega_moe` for W4A16 SiTU, and `fp8_fp8_mega_moe` for FP8
channelwise or blockwise. The workspace precision selects the FP8 backend.
The complete entry owns metadata, scratch, tuning selection, GEMM2/scatter,
count recycling and combine. It writes directly into `y`.
[The public calling contract](../README.md#megamoe) documents weight layouts,
activation support and workspace reuse.

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
1, 0, 33, 129 and 8 tokens to cover empty ranks, all-empty calls and partial tiles.
This table describes configured coverage, not a record that every case has
passed on hardware. These tests exercise BF16 scatter/combine; optional FP8
scatter/combine remains implemented but is outside this test matrix.

## Accuracy checks and comparison methodology

The test runners use independent Torch references built from source tensors and
logical quantized weights. They model each precision's activation quantization
and BF16 rounding. The normalized squared-error metric is
`sum((actual-reference)^2) / max(sum(actual^2+reference^2), 1e-30)`.
Its threshold is 0.003 for W4A4, 0.0005 for W4A16 and 0.001 for FP8. The FP8
threshold matches the existing grouped contiguous/masked GEMM tests. Relative
L2 is reported for every precision; W4A16 also requires it below 0.005. Graph
and eager outputs must match bitwise. Changed inputs, routes and routing weights
are checked against the independent reference with the same thresholds.

`--reference-tokens 0` checks every token; a positive value samples that many
tokens per rank and reports the sample size. `--edge-cases` includes a zero-token
rank at the 1-token test point, invalid IDs and zero weights. Replay checks
concentrated routes, no valid routes, tiny/nonuniform/zero weights, and return
to the original inputs; combine storage is poisoned with NaNs between calls.
`--replicas` uses separate master/replica weight allocations.

Graph latency is the median of per-repeat maximum-rank means. For A/B
comparisons, keep model, precision, routing distribution, seeds, capacity,
replicas, EP and timing mode identical. Include input staging/quantization,
communication, both GEMMs, activation, scatter/combine and counter cleanup;
exclude one-time weight transformation, JIT and reference evaluation.
The commands below exercise these checks; device and CI results must be
recorded for the revision being tested.

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
# Channelwise FP8 smoke supports EP1, EP2, EP4 and EP8; this example uses EP4.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  tests/test_moe_w8a8.py --case smoke_channelwise --tokens 1 0 33 8 \
  --edge-cases --replicas --benchmark

# DeepSeek V4 Pro: correctness and full-pipeline graph timing.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc-per-node=8 tests/test_moe_w4a4.py \
  --case deepseek_v4_pro --tokens 1024 2048 4096 --routing random \
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

All precisions apply original routing weights once, during combine after GEMM2.
Pass them only to `mega_moe_pre_dispatch`; the library masks invalid routes and
refreshes the weight view on every graph replay. Quantization scales remain
separate from routing weights.

The CLI checks every requested token by default. `--reference-tokens N` limits
the numerical comparison to the first N tokens on each rank while still running
the complete workload. Pytest uses bounded references for large model cases and
checks every smoke token. Correctness also checks finite output, a poisoned
combine buffer and repeated graph replay with changed inputs. The changed
inputs/routes/weights are checked against the independent Torch reference with
the same numerical thresholds, as well as bitwise against eager execution.

`--routing round_robin` selects a balanced distribution.
`--routing random` chooses distinct random experts per token;
`--routing skewed` sends 80% of tokens to the same top-k experts. All three
precision runners support these options and record the choice in their JSON.

`--edge-cases` injects masked routes, zero routing weights and an empty final
rank when `--tokens` includes 1. With multiple ranks it also exercises targeted
remote routing and staggered launches in the shared replay checks. `--replicas` places the final local physical
expert in a separate allocation for both GEMMs. W4A16 Kimi K3 uses
896 masters globally and one replica slot per rank (912 physical experts
at EP16 or 928 at EP32). Other cases split their physical slots. Pytest exercises both
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

Each example allocates a library workspace and transforms logical weights once.
Its forward function calls `mega_moe_pre_dispatch` and the corresponding
complete entry point (`fp4_fp4_mega_moe`, `bf16_fp4_mega_moe`, or
`fp8_fp8_mega_moe`), writing into a preallocated BF16 output.
See the [public API contract](../README.md#megamoe) for a standalone example.
Normal forward code does not allocate communication buffers or access metadata.

The workspace owns the generation barrier, local completion flag, scratch,
route metadata and BF16 combine storage. All ranks, including empty ones,
invoke the pipeline in the same order on one stream. Keep the workspace and
weights alive across graph replay; overlapping invocations need separate
workspaces. Counters are recycled by the library before the next invocation.

Tests poison private combine storage with NaNs exclusively for fault injection.
Graph checks change inputs, routes (including all-invalid routes) and weights
between consecutive calls and compare eager and captured results exactly.
Every changed workload is also checked against the independent numerical
reference, including tiny/nonuniform weights and all-zero weights. Smoke runs
check unsupported public factory/entry combinations before cooperative compute.
`compile_kernels` compiles native kernels without launching them, then ranks
synchronize before entering cooperative device barriers. The router and
shared-expert branch are outside these routed MoE examples.
