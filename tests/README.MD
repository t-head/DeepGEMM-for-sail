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

## Native JIT compile-key regression (PPU)

From a source checkout with initialized submodules, PPU PyTorch and the PPU SDK:

```bash
CUDA_VISIBLE_DEVICES=0 MAX_JOBS=1 python tests/test_jit_compile_keys.py -v
```

This uses `load_inline` to build the host C++ checks embedded in the Python test
against the checkout's actual code generators.
It checks that changing each codegen input changes both source and cache key,
and that runtime-only arguments preserve both. Coverage includes BF16 CuteFree
and CUTLASS with BF16/FP32 output, GEMV M1/M2, FP8 block/channel, INT8 and FP4.
The helper queries hardware constants but never compiles or launches generated
GEMMs. Public API accuracy and launch behavior remain covered by the GEMM UTs.

## Test fp8
python ./test_fp8_core.py

## Test int8 and bf16
python ./test_core.py

test_core.py options:
  -h, --help            show this help message and exit
  --func {DenseGemm,GroupedContiguous,GroupedMasked,DenseGemm}  target test func
# Routed MoE correctness and performance tests

MegaMoE supports W4A4 on PPU1.5, validated on ZW-M890P.
`test_mega_moe.py` contains public-entry correctness and regression tests;
`bench_mega_moe.py` contains full-pipeline timing and optional DeepEP comparisons.
The benchmark reuses the correctness runner's case table and input/reference
helpers. Both use the public complete call.
Other architecture pytest runs skip these tests; CLI use reports an
unsupported-device error.

`test_mega_moe_precisions.py` exercises FP8 channelwise/blockwise, W4A8 and W4A16
through their public complete calls. Its default model cases are listed below;
all token counts are per rank. CLI options can select other counts within capacity.

| Case | Precision | EP | Default tokens/rank |
|---|---|---|---|
| `glm52` | FP8 channelwise | 8 | 32, 1024, 2048, 4096 |
| `glm53_flash` | FP8 channelwise | 4 | 1024, 2048, 4096 |
| `qwen38_flash` | FP8 channelwise | 4 | 1024, 2048, 4096 |
| `minimax_m3` (open-source model) | FP8 channelwise | 8 | 1024, 2048, 4096 |
| `minimax_m3_blockwise` (next-generation model) | FP8 blockwise | 8, 16, 32, 64 | 64, 1024, 2048, 4096 |
| `kimi_k3` | W4A16 | 8, 16 | 1024, 2048, 4096 |
| `kimi_k3_decode` | W4A16 | 32 | 4, 8, 16, 32, 64, 128 |
| `deepseek_v4_pro_w4a8` | W4A8 | 8 | 32, 1024, 2048, 4096 |
| `kimi_k3_w4a8` | W4A8 | 8, 16, 32 | 32, 64, 128, 1024, 2048, 4096 |

Small distributed cases cover separate replicas, both math modes, OAI/clamp,
empty ranks, consecutive changing routes, statistics and graph replay.
Distributed worklist and tile regressions use the common workspace's 64-rank
capacity rather than precision-specific EP limits.
The complete-call and replica pytest smoke cases use four experts per rank
and accept EP1 through EP64. Other smoke cases keep their fixed global expert
counts and run at EP1/2/4/8. The model EP sizes above describe deployment
workloads, not precision-specific API limits.
The API cases also check malformed weights, odd GEMM1 widths, int32/int64
routing IDs and rejected mismatched token counts. Slice input prefixes at the
call site: `num_tokens` must equal the number of rows passed to pre-dispatch.
All precisions use the common workspace entry point for complete-pipeline checks,
including compatible `SymmBuffer` base aliases.
Activation-reuse cases switch supported activations and SiTU coefficients on
one allocation, then replay earlier graphs to check that later calls do not
change their behavior. Unsupported activations remain rejected.
Accuracy uses `deep_gemm.testing.calc_diff < 0.001`, matching grouped FP8
GEMM; W4A8 retains its existing 0.0015 threshold. Relative L2 is reported on
failure. References run outside timing. The CLI examples use
`--reference-tokens 8` to sample eight tokens per rank; the CLI default of 0
checks every token.

```bash
# No private C++ bindings or test-side counter resets are needed.
python -m pytest -q -s --disable-warnings -x tests/test_mega_moe_precisions.py -k smoke
torchrun --standalone --nproc-per-node=2 -m pytest -q -s --disable-warnings -x \
  tests/test_mega_moe_precisions.py -k smoke
torchrun --standalone --nproc-per-node=8 tests/test_mega_moe_precisions.py \
  --case kimi_k3 --tokens 1024 2048 4096 --routing random --reference-tokens 8 \
  --benchmark --warmup 10 --iterations 100 --repeats 5
# The same benchmark can be invoked directly; --precision selects its case table.
torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
  --precision w4a16 --case kimi_k3 --tokens 1024 2048 4096 --routing random \
  --reference-tokens 8 --warmup 10 --iterations 100 --repeats 5
```

The precision CLI forwards `--benchmark` to the same benchmark driver as W4A4.
All precisions accept the same token/capacity, routing, math, reference and timing
options. `--mode graph` (the default) times the complete graph, including BF16 input preparation,
communication, both GEMMs, activation/quantization, scatter, combine and recycling.
`--mode eager` times the same complete call without graph capture. Both report
the slowest rank and median/min/max over repeats. DeepEP comparisons currently
cover W4A4 only; requesting that baseline for another precision is rejected.

### W4A4 cases

| Case | EP | H | I | Experts | Top-k | Activation | Capacity |
|---|---|---:|---:|---:|---:|---|---:|
| `deepseek_v4_pro` | 8 | 7168 | 3072 | 384 | 6 | SwiGLU, limit 10 | 8192 |
| `deepseek_v4_flash` | 4 | 4096 | 2048 | 256 | 6 | SwiGLU, limit 10 | 4096 |
| `deepseek_v41_flash` | 8,16 | 5120 | 2304 | 384 | 6 | SwiGLU, limit 10 | 4096 |
| `glm52` | 8 | 6144 | 2048 | 256 | 8 | SwiGLU | 4096 |
| `kimi_k3` | 8,16 | 3584 | 3072 | 896 | 16 | SiTU | 4096 |
| `qwen38_2_4t` | 8,16 | 8192 | 2048 | 512 | 10 | SwiGLU | 4096 |
| `mimo_v26_pro` | 8 | 6144 | 2048 | 384 | 8 | SwiGLU | 8192 |

I is the activated intermediate width; GEMM1 has 2I output rows. Primary model
tokens/rank are 1024, 2048 and 4096; MiMo also includes 8192. CLI overrides can
select 32 tokens or other counts within capacity. `--num-tokens` selects
actual tokens and `--num-max-tokens-per-rank` selects workspace capacity;
`--tokens` and `--capacity` remain CLI aliases. `--routing` supports
`round_robin` (default), `random` and `skewed`. Use random routing explicitly
for comparisons with random-route benchmark reports.
`--fast-math 1` is the default and passes `fast_math=True` to the public call;
`--fast-math 0` selects the regular activation math path. The three smoke
activations cover both modes and graph replay. Replica tests duplicate physical
expert slots within the same rank-local weight bank.

```bash
# Single-device smoke tests, including all three activations and API/workspace regressions.
python -m pytest -q -s --disable-warnings -x -k smoke tests/test_mega_moe.py

# Manual multi-device regression on two idle devices.
torchrun --standalone --nproc-per-node=2 -m pytest -q -s -k smoke \
  tests/test_mega_moe.py

# Correctness for DeepSeek V4 Pro.
torchrun --standalone --nproc-per-node=8 tests/test_mega_moe.py \
  --case deepseek_v4_pro --num-tokens 32 1024 4096 --routing random \
  --reference-tokens 8

# Full-pipeline graph latency without a baseline.
torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
  --case deepseek_v4_pro --num-tokens 32 1024 4096 --routing random \
  --reference-tokens 8 --warmup 10 --iterations 100 --repeats 7

# Prefill against DeepEP normal, with eager timing for both paths.
torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
  --case deepseek_v4_pro --num-tokens 1024 2048 4096 --routing random \
  --reference-tokens 8 --baseline normal --warmup 10 --iterations 100 --repeats 7

# Decode against DeepEP low latency, with graph timing for both paths.
torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
  --case deepseek_v4_pro --num-tokens 32 64 128 --routing random \
  --reference-tokens 8 --baseline low_latency --warmup 10 --iterations 100 --repeats 7

# SiTU correctness and performance with duplicated physical expert slots.
torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
  --case kimi_k3 --num-tokens 1024 2048 4096 --routing random --replicas \
  --reference-tokens 8
```

EP16 uses two nodes of eight devices: set `--nnodes=2`, `--node-rank=0/1`,
`--master-addr` and `--master-port` consistently. Enable the appropriate
symmetric-memory fabric transport and use idle devices on every rank.
For single-node containers whose hostname is not resolvable, replace
`--standalone` with `--nnodes=1 --node-rank=0 --master-addr=127.0.0.1
--master-port=29500` (choose an unused port).
The multi-rank stateful test requires at least two ranks and is skipped in
single-device runs. Use the two-device regression command above to run it.

Single-device smoke regression is also available through
`python tests/checkin/run_checkin.py --suite ZW-M890P-MegaMoE` and runs in the
M890P CI job. Multi-device tests remain manual; see the [check-in guide](checkin/README.md#megamoe).

Smoke SwiGLU, Step and SiTU cases use 1, 0, 33, 129 and 8 tokens. The 129-token
case crosses a 128-row tile boundary. Every smoke token is checked against an
independent quantize/dequantize and expert-matmul reference. References apply
original routing weights after GEMM2. The shared `deep_gemm.testing.calc_diff`
measures normalized squared error, mathematically equivalent to
`sum((actual-reference)^2) / sum(actual^2+reference^2)`, and returns zero when
both tensors are all zero; W4A4 requires error below 0.003.
Relative L2 is also reported. `--reference-tokens N` samples N tokens per rank
for large-model mathematical reference checks; omit it to check all tokens.
The correctness/benchmark JSON also reports `activation_quantization_error`
against FP32 activations and matmuls with the same dequantized MXFP4 weights.
This diagnostic omits input/intermediate FP4 quantization and intermediate
BF16 rounding; it is not a full-model accuracy measurement or an unquantized
weight reference. The 0.003 pass threshold applies only to the quantization-aware
reference, not to this diagnostic. Both references run outside benchmark timing.

Model tests check complete outputs against independent references and basic graph
replay; `--replicas` additionally exercises duplicated physical experts. Shared API and state-reuse
checks are run once with the small SwiGLU case in the same test file:

- `test_mega_moe_smoke_api` checks unsupported MMA/activation/recipe options,
  scale layouts, local expert limits, removed public options and malformed tensors.
- `test_mega_moe_smoke_stateful` requires at least two ranks and checks public
  pre-dispatch, SM90-style positional calls, offset views, an empty rank,
  `-1` route masks and zero weights.
  Consecutive calls and graph replays change inputs, routes and weights and
  compare against fresh references. Dispatch counts must be recycled; INT32
  cumulative receive statistics must match independent routing histograms.
  Capture must not increment statistics; each replay must.
- `test_mega_moe_smoke_workspace` checks allocation/registration counts, EP1,
  positional compatibility and `base` aliases without resetting protocol state.
- `test_mega_moe_smoke_clamp` checks None, zero, positive and +inf thresholds,
  independent numerical references and graph replay.
- `test_mega_moe_smoke_nan_slots` produces NaN combine slots through public
  calls with NaN scale codes, then verifies masked routes on reuse and replay.
- `test_mega_moe_smoke_grouped_workspace` covers buffer-owned scheduler scratch
  with 128 or 256 local physical experts.

`-k smoke` includes these tests. The CLI and benchmark `--edge-cases`
option explicitly enable extended empty-rank, masked-route and state-reuse
checks for a requested model; ordinary runs use the basic replay check.

Timing starts with the same BF16 inputs and includes required input preparation,
quantization, dispatch, GEMM1/activation/output quantization, metadata,
GEMM2/scatter, state recycling and combine. One-time workspace allocation,
weight transforms, JIT, reference checks and harness barriers are outside the
timed region. DeepEP normal rebuilds its dispatch layout and expert permutation
on every forward; these operations and any required host/device transfers are
included. Original routing weights are applied once, after GEMM2, in both paths.
Both backends use the same raw quantized weights, inputs, routing IDs and weights.

Each sample is the slowest rank's mean latency; JSON reports contain all samples
and their median/min/max. `baseline.speedup` is DeepEP latency divided by MegaMoE
latency. `--baseline none` (default) and `--baseline low_latency` default to graph
timing. DeepEP normal returns CPU expert counts and requires eager execution;
`--baseline normal` therefore selects eager for both paths and rejects
`--mode graph`. Eager timing includes host launch gaps. Graph and eager results
must not be mixed in the same speedup calculation.

DeepEP and Triton are optional and are imported only for a requested baseline.
If either is absent on any rank, all ranks skip the comparison and report the
reason while still benchmarking MegaMoE. An installed backend's import or
execution errors fail the run. The baseline implements the following W4A4 activations:

| Activation | DeepEP expert computation |
|---|---|
| SwiGLU | Grouped GEMM1 with fused activation and MXFP4 quantization |
| Step | Grouped GEMM1 with fused Step activation and MXFP4 quantization |
| SiTU | Grouped GEMM1, then `activation_and_mul` / `activation_and_mul_masked` with `ActivationMode.SITU` and `QuantMode.MXFP4` |

The standalone SiTU kernel is included in baseline timing; the ordinary grouped
GEMM currently has no fused SiTU epilogue. JSON records `activation_impl` so this
difference is visible. MegaMoE uses the configured case capacity; DeepEP LL sizes
its per-expert receive storage for the largest requested token count, which is
also recorded as `baseline.num_max_tokens_per_rank`.
DeepEP has its own shape restrictions. Use a supported model case for LL comparisons;
backend errors are reported rather than changing the requested shape.

Correctness and basic graph replay with unchanged inputs/routes run before timing
by default, using the existing W4A4 numerical threshold. Add `--edge-cases` to
check changing inputs, routes and routing weights across replays. DeepEP is also
checked against the same independent reference. `bench_mega_moe.py --no-check`
skips correctness and replay checks.
The old `test_mega_moe.py ... --benchmark` command forwards to the benchmark
entry and preserves the existing token/capacity aliases and timing arguments.

Tile and copy-CTA selection lives in `csrc/jit_kernels/heuristics/mega_moe.hpp`.
Tile selection is internal; workspace constructors do not expose GEMM1/GEMM2
tile overrides. Normal benchmark commands require no tuning overrides.
The public calling example and workspace reuse contract are in the main README.

# W4A8 correctness and performance tests

On M890, [test_w4a8.py](test_w4a8.py) covers MXFP4 and
[test_w4a8_nvfp4.py](test_w4a8_nvfp4.py) covers NVFP4. The NVFP4 calling contract,
scale layouts and execution modes are in the
[NVFP4 W4A8 API guide](../docs/api/nvfp4_w4a8.md).

## Correctness

Run from the repository root after building the checkout:

```bash
python tests/test_w4a8.py
python tests/run_deep_gemm.py \
    --format='GroupedNoPad,data_type:w4a8_nvfp4,groups:4,m:128,n:256,k:1024,execution:fused'
python tests/run_deep_gemm.py \
    --format='GroupedNoPad,data_type:w4a8_nvfp4,groups:4,m:128,n:256,k:1024,execution:dequantize'
python -m pytest tests/test_w4a8_nvfp4.py -q -s
# Standalone alternative for the same NVFP4 tests:
python tests/test_w4a8_nvfp4.py
```

`data_type:w4a8_nvfp4` uses the common dense, nopad, masked and fused-routing
test functions. Both execution modes are also included in
`checkin/checkin_890p.list`. `execution` defaults to `fused`; `dequantize`
allocates an FP8 workspace once per case and refreshes it on each call.
`quant_type:block` (default) uses 1x128 activation scales and 128x128 outer
weight scales; fused execution also accepts `quant_type:channel` for
per-token activations and per-channel weights. `lhs_scale_group_size:0`
selects per-token activation scales independently; fused execution also accepts
`lhs_scale_group_size:16` for K16 activation scales. Native NVFP4 scales are
always K16 E4M3, with a separate FP32 global scale per expert.

Dimensions, `distribution`, `em`, `padding_m`, `num_token` and `topk` follow
the common runner's conventions: nopad `m` is the total number of rows,
masked distributions specify valid rows per expert, and fused routing uses
`num_token * topk` output rows. No requested dimensions are silently changed;
decode-to-FP8 fused routing requires N/K to be multiples of 128.
`--ref_backend=host` computes independent references on the CPU, and
`--disable_acc` skips reference calculation and output checks.

NVFP4 tests target M890. Pytest skips unsupported devices; the standalone
runner prints the hardware requirement. The three test entry points are:

| Test | Coverage |
| :--- | :--- |
| `test_nvfp4_decode` | Fused dense/grouped boundary cases and model-size EP32/EP64 nopad GEMMs |
| `test_nvfp4_prefill` | Per-call decode plus FP8 for the same cases; exact workspace checks |
| `test_nvfp4_offline_scale` | Offline scale absorption versus direct FP8 casts, in both execution modes |

Decode/prefill boundary cases use batch sizes 1, 17, 128, 257 and 4096, with N/K=192/160
for small cases and 256/1024 otherwise; decode-to-FP8 fused routing uses
256/256 to meet its alignment requirement. Additional M=1, N=1024, K=64/128
cases cover disabled/enabled N expansion and are included in the check-in list.
K=192 cases use distinct K64 scale ranges and repeated graph replay to cover scale-slot reuse.
Fused execution also uses M=17, N/K=256/192 with distinct adjacent K16
activation scales in all four layouts, with channel and block outer weight
scales. These cases verify the two partial sums within each K32 MMA.
Batch is the activation row count,
or per-expert buffer capacity for masked GEMMs; fused routing uses top-k=2.
These tests include empty experts, N/K tails, native/global/outer scales,
a shared-decode tile, invalid K-warp configs, routing config selection and
overrides, rejection of routing metadata with a mismatched M tile, and graph replay after weight
and scale updates. Independent FP32 references include the specified
weight conversion and scales. Checks require squared relative L2 error
below `2e-5` against BF16-rounded references and elementwise agreement with
`rtol=0.02, atol=0.25`.

Both tests also use the 384-expert, top-k=4 NVFP4 MoE model dimensions:
gate/up N/K=10240/7168 and down N/K=7168/5120, at EP32
(12 local experts) and EP64 (6). B is 1, 128 or 4096 tokens per rank before
routing; local M=4*B rows are balanced across the local experts. These nopad
cases use 1x128 activation scales and 128x128 outer weight scales, and the
same independent FP32 references and tolerances as the boundary tests.
Check-in includes all four model configurations in both execution modes
at B=128 (M=512). Model dimensions are retained in full; weights, scales,
activations and routing counts are synthetic, not checkpoint data or a
recorded router trace. Small boundary cases remain separate coverage.

The offline example uses three 256x256 expert weights, dense/nopad layouts
and batches 1, 128 and 4096. Its synthetic E4M3 K16 scales are uniform in
[0.25,1.25] (`small`), uniform in [128,448] (`large`), or log-uniform in
[2^-9,448] (`wide`). Large/wide cases also force a scale of 448 and FP4
value 6 in each row to exercise saturation. Kernel outputs are checked
against FP32 GEMMs using each method's converted FP8 weights, with relative
L2 below `0.006`. Offline absorption must improve weight error in the
large/wide profiles, with restored-weight error below `0.05` and GEMM
output error below `0.06` against the original NVFP4 references.

Printed errors use `norm(actual - reference) / norm(reference)`; multiply
by 100 for a percentage. The NVFP4 reference includes original K16 and
global scales; the FP8 reference includes conversion and restoring scales:

| Field | Comparison |
| :--- | :--- |
| `weight_vs_nvfp4_rel_l2` | Restored FP8 weights vs original NVFP4 weights |
| `fp8_reference_vs_nvfp4_rel_l2` | FP32 GEMM with converted FP8 weights vs FP32 GEMM with original NVFP4 weights |
| `fused_vs_nvfp4_rel_l2`, `dequantize_vs_nvfp4_rel_l2` | Kernel output vs original NVFP4 reference |
| `*_vs_fp8_reference_rel_l2` | Kernel output vs the converted FP8 reference |
| `fused_vs_dequantize_rel_l2` | Agreement between execution modes |

These errors measure conversion and arithmetic after NVFP4 quantization;
they do not evaluate model accuracy against pre-quantization weights.

## Performance

The common performance entry accepts the same NVFP4 case strings:

```bash
python tests/run_deep_gemm.py --benchmark \
    --format='GroupedNoPad,data_type:w4a8_nvfp4,groups:6,m:512,n:10240,k:7168,execution:dequantize'
python tests/run_deep_gemm_perf.py --device 0 --output nvfp4-perf \
    --format='GroupedNoPad,data_type:w4a8_nvfp4,groups:6,m:512,n:10240,k:7168,execution:dequantize'
```

Use `execution:fused` to measure fused W4A8. The `dequantize` timing includes
both the per-call decoder and FP8 GEMM; preparation and workspace allocation
are outside timing. The profiler's Tensor Cell/HBM percentages describe the
GEMM, while time and cycles cover both kernels. In the default `metrics`
mode, `run_deep_gemm_perf.py` profiles without accuracy checks, then runs a
separate correctness pass unless `--disable_acc` is supplied.

For FP8 comparisons, use `data_type:fp8,quant_type:block` with the same
GEMM layout, M/N/K, group count and row distribution in the common runner.
For example, these commands measure the EP64 gate/up configuration at
B=128 tokens/rank (6 local experts, M=4*B=512):

```bash
python tests/run_deep_gemm.py --benchmark \
    --format='GroupedNoPad,data_type:fp8,quant_type:block,groups:6,m:512,n:10240,k:7168,distribution:[86,86,85,85,85,85]'
python tests/run_deep_gemm.py --benchmark \
    --format='GroupedNoPad,data_type:w4a8_nvfp4,groups:6,m:512,n:10240,k:7168,distribution:[86,86,85,85,85,85],execution:fused'
python tests/run_deep_gemm.py --benchmark \
    --format='GroupedNoPad,data_type:w4a8_nvfp4,groups:6,m:512,n:10240,k:7168,distribution:[86,86,85,85,85,85],execution:dequantize'
```

The same case strings work with `run_deep_gemm_perf.py --format` or in a
`--caselist`; profiler results are written to CSV using the common runner's
options. Each case generates its own synthetic inputs and checks its own
reference. Kernel timing follows the same method as other GEMM dtypes.
Use EP32/groups=12 or EP64/groups=6, and up N/K=10240/7168 or down
N/K=7168/5120. For another pre-routing batch B, set M=4*B and distribute
those rows across the local experts.
