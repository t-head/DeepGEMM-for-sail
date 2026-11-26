# MegaMoE training on ZW-M890P

This module implements first-order SwiGLU expert forward/backward on PPU0015,
with BF16 or blockwise E4M3 FP8 GEMMs. Selected FP8 recipes use FP8 operands
for DW as well as forward/dGrad; DW accumulation and outputs are FP32.
The supported setup is PPU SDK 2.2.0 and native PPU Torch 2.11.0. CUDA API
names below refer to the PPU Torch runtime.

## Execution paths

| Plan | Communication and expert execution |
| --- | --- |
| `MegaMoEPlan` | Optimized training path: direct-peer source/ready pipeline with dispatch/quantization/GEMM overlap and expert return stores; selected bidirectional fusion. |
| `DeepEPMoEPlan` | DeepEP dispatch/combine around bounded expert windows; windowed control. |
| `CompactDeepEPMoEPlan` | The same DeepEP transport, compact expert permutation, grouped GEMMs and active weighted unpermutation; compact control. |

Use `from deep_gemm.training import MegaMoEPlan` for the optimized training
plan and `mega_moe(..., plan=plan)` for its first-order autograd adapter. Compact
DeepEP is the primary dispatch/combine baseline; windowed DeepEP isolates the
layout effect. Internal `peer` transport and kernel names describe the direct
memory-access mechanism used by MegaMoE.

The selected peer path requires one physical direct-access supernode. There are
eight PPUs per host and eight hosts per supernode: EP32 uses four hosts and EP64
uses eight. Process-group membership alone does not prove peer accessibility.
DeepEP controls must report exactly one scale-out group; the benchmark rejects
another topology before warmup. Cross-supernode scaling needs separate validation.

The compact layout follows MoK's
[`bench_deepep_te.py`](https://github.com/cursor/mixture-of-kittens/blob/f3a8bd2432457c5a2a2ea13915f0dcd0224581e4/benchmarks/bench_deepep_te.py)
workflow: dispatch, expert permutation, MLP, weighted unpermutation and combine,
with the corresponding reverse path. The PPU implementation uses native GEMMs
and PPU blockwise FP8 instead of Transformer Engine/NVIDIA MXFP8. MoK dispatches
BF16 activations even for FP8 compute. The selected PPU FP8 EP32 control dispatches
source-quantized FP8 in forward and redispatches BF16 X for backward; the EP64
control dispatches BF16 and quantizes received rows. Reported
results compare complete PPU implementations, rather than reproducing MoK's
published GPU speedups.

## API and correctness contracts

BF16 DW uses `k_grouped_bf16_gemm_tn_contiguous` for both metadata sources.
The original positional call with `ks_cpu=[...]` and FP32 `c` is preserved.
Use `ks_cpu=None, use_psum_layout=False` to read current group lengths directly from the int32
device `grouped_layout`; optional `max_group_k` defaults to input capacity
and selects tuning without reading device values on the host. In length mode,
`c=None` returns fresh gradients, `c=out` accumulates in place, and a separate
`c` supplies initial FP32 values. Device-only scheduling controls are keyword
arguments. The producer must supply nonnegative 128-aligned lengths whose sum
fits both input capacities. Upstream's `use_psum_layout=True` also accepts
device-only inclusive prefixes with `ks_cpu=None`, retaining required FP32 `c`;
training tuning keywords apply to length mode. Host metadata and device prefixes
accept unequal or padded A/B capacities when every active row fits both buffers.
`k_grouped_bf16_gemm_tn_segmented` accepts multiple
A/B buffers and device prefix offsets without concatenating their storage.

`W1=[local_experts,2*I,H]`, `W2=[local_experts,H,I]`, and `X=[tokens,H]` use BF16.
Expert IDs are `int64[tokens,topk]` with global expert numbering; `-1` masks a
route. Scores are FP32. A device int32 scalar bounds the valid token prefix.
The chosen recipes require H divisible by 256 and I divisible by 128.

```python
from deep_gemm.training import CompactDeepEPMoEPlan

plan = CompactDeepEPMoEPlan(H, I, local_experts, topk, tokens,
                          chunk_tokens=chunk, ring_rows=reference_window,
                          compact_rows=reservation, group=group, device=device,
                          compute="fp8", check_errors=False)
plan.prepare(w1, w2)  # on the execution stream, before payload/capture
plan.reset_errors()
y = plan.forward(x, ids, scores, w1, w2, valid_tokens)
dx, dw1, dw2, dscores = plan.backward(dy, x, ids, scores, w1, w2, valid_tokens)
plan.check()         # also required after graph replay before accepting results
plan.close()
```

Backward accepts an optional pair of caller-owned FP32 DW accumulators and adds
the current gradients to their initial values. Without that pair it returns
fresh logical FP32 DW outputs. Autograd wrappers support first-order gradients.
Pair forward/backward before reusing activation storage; one plan must not serve
concurrent independent steps. DeepEP plans provide `close()`; peer
plans follow their Torch tensor and symmetric-memory lifetimes. Synchronize
before releasing them. `prepare` compiles local kernels and agrees on
preparation failures before entering payload collectives.

Expert-side gradient plans omit the unused symmetric top-k DY matrix. This
saves `chunk_tokens * topk * H * 2` bytes per PPU: 320 MiB for Qwen3.5-397B-A17B
and 2.5 GiB for Qwen3.8-2.4T-A95B. Preparation warms only the selected DW precision.
The forward tile pipeline reserves a bounded total of producer and polling
consumer CTAs; configurations exceeding the device CU count are rejected.

The compact planner reserves aligned expert rows from representative routing
with 25% headroom. This is setup, outside steady-state timing. Every replay
rebuilds current routing metadata; it does not reuse a fixed permutation.
Overflow reports an error and invalidates all outputs of that step. Replan and
recapture, use a windowed plan, or supply routes fitting the reservation.
`reset_errors()` clears status for a new attempt; it does not increase capacity.

For a padded active route count R and reference window W, backward recomputes
the prefix ending at `((R-1)//W)*W` when R is positive. Compact execution retains
this last-active-window policy and pins GEMM selection using the reference
capacity. BF16 DW retains the reference `config_k` hint. Thus the layout change
preserves FC1/FC2/dHidden/DX/DW work, precision, and live-weight preparation.
FP8 DW uses K128 scales, left-operand scale extent 64, native Tensor Cell
computation and FP32 accumulation/output. The selected DW schedule is serial
relative to competing expert GEMMs because concurrent DW increased contention.

## Reproduction

Only two training test/benchmark files are needed:

- `tests/benchmark_mega_moe_training.py`: the 24 selected performance recipes.
- `tests/test_mega_moe_training.py`: CPU contracts and device/distributed acceptance.

[Recipe definitions](ppu_mega_moe_training_recipes.json) cover EP32/EP64, Qwen3.5-397B-A17B/Qwen3.8-2.4T-A95B
shapes, BF16/FP8, and peer/windowed/compact execution. Each profile embeds its
exact DW LUT bytes and SHA256; the launcher materializes them under ignored
`build/mega_moe_training_luts/` before importing DeepGEMM. Profiles are kept separate
so removing auxiliary LUT files does not change automatic kernel selection.

```bash
# Inspect without importing Torch, initializing a device, or writing LUT files.
python tests/benchmark_mega_moe_training.py --ep 32 --model Qwen3.5-397B-A17B --precision fp8 --dry-run
python tests/test_mega_moe_training.py --mode recipes

# On SDK 2.2 installations needing native DeepEP JIT header compatibility:
python tests/benchmark_mega_moe_training.py --prepare-deepep-compat
export PYTHONPATH="$PWD/build/campaign/deepep-native-scope:$PWD:${PYTHONPATH:-}"
export CPATH="$PWD/build/campaign/compat/deepep:/usr/local/pccl/include:/usr/local/pccl/nccl_wrapper/include:/usr/local/PPU_SDK/include:${CPATH:-}"

# Run on EVERY participating host with its own NODE_RANK.
# EP32: NNODES=4; EP64: NNODES=8, all within one physical supernode.
torchrun --nnodes="$NNODES" --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  tests/benchmark_mega_moe_training.py --ep 32 --model Qwen3.5-397B-A17B --precision fp8 \
  --backend deepep --deepep-layout compact
```

Choose `--backend megamoe` for fusion or `--backend deepep --deepep-layout windowed`
for the other control. Set `--ep 64` alongside `NNODES=8` for EP64. Both precisions
and both models use the same CLI. Use idle hosts, check physical topology, and
reject timings if another compute job starts during the run.

Acceptance groups use the same `torchrun` prefix. `recipes` needs only Python;
`capacity` uses CPU tensors with the installed PPU extension. `dw` and `jit` need one PPU;
`cold`, `overflow`, and toy `compact`/`transport` use eight local ranks.

```bash
python tests/test_mega_moe_training.py --mode capacity
python tests/test_mega_moe_training.py --mode dw
python tests/test_mega_moe_training.py --mode prepare
for precision in bf16 fp8; do
  DG_JIT_CACHE_DIR="/tmp/mega-moe-cold-$precision" \
    torchrun --standalone --nproc-per-node=8 \
      tests/test_mega_moe_training.py --mode cold --precision "$precision"
done
# Also run --mode overflow, --mode compact, and transport for each precision.
torchrun --standalone --nproc-per-node=8 tests/test_mega_moe_training.py \
  --mode transport --precision fp8
# On the actual EP32/64 cohort, replace the benchmark command with:
# tests/test_mega_moe_training.py --mode compact --model Qwen3.5-397B-A17B --precision fp8
# tests/test_mega_moe_training.py --mode transport --model Qwen3.8-2.4T-A95B --precision bf16
```

Start the cold test with an empty dedicated JIT directory. It verifies that
prepare leaves no new DeepGEMM modules for the first complete forward/backward.
Use a node-local filesystem for the JIT cache. The serial compiler's immediate
read after atomic publication can fail on a shared filesystem when many ranks
compile the same missing specialization concurrently.
It covers both precisions of the windowed, compact and MegaMoE plans, including
first-nonempty and segmented DW variants before entering peer synchronization.
Preparation compiles the selected pointwise and GEMM specializations in order
before payload publication and graph capture. Zero-length DW warmups retain
one temporary gradient buffer at a time.
Compact tests compare every element of Y/dX/DW1/DW2/dScores, changed weights and
routes, equal GEMM/DW work, and nonzero external FP32 accumulation. Toy coverage
also includes G8 weights, empty/partial/skewed routes. Transport tests compare
fused paths with duplicate scalar/bulk controls; DW tests use an independent
long-K CPU FP64 oracle, zero groups, tail guards, repeated beta1 accumulation,
and aliased-output rejection. Overflow tests reject an overfull captured step
and recover the same graph with fitting routes.

## Measured performance

The latest complete paired comparisons show **1.26–1.53× speedup over Compact
DeepEP**, including **1.48–1.53× for FP8**, across the model shapes below.

| Shape | H | I | Tokens/rank | Global experts | Top-k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-397B-A17B | 4096 | 1024 | 4096 | 512 | 10 |
| Qwen3.8-2.4T-A95B | 8192 | 2048 | 16384 | 512 | 10 |

The Qwen3.5-397B-A17B routed dimensions match its
[official config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json).
The Qwen3.8-2.4T-A95B routed dimensions match the supplied checkpoint's `config.json`
(SHA256 `8d8e95c4ca3f834bb069963f73ae56ba949a6df1997aee372c57c692b5302728`):
H8192, expert I2048, 512 experts and top-k10. Its architecture is
`Qwen3_5MoeForCausalLM`, with 92 layers and shared-expert intermediate size 2048.
These operator tests use generated BF16/E4M3 FP8 operands, without loading the
checkpoint's MXFP4 weights. Tokens/rank is a benchmark workload choice.
Fixed tokens/rank make EP64 weak
scaling with twice the EP32 global token count.

| EP | Shape | Precision | Windowed DeepEP | Compact DeepEP | MegaMoE | Compact/MegaMoE |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 32 | Qwen3.5-397B-A17B | BF16 | 28.101 ms | 23.995 ms | 19.057 ms | 1.259x |
| 32 | Qwen3.5-397B-A17B | FP8 | 26.895 ms | 22.672 ms | 14.906 ms | 1.521x |
| 32 | Qwen3.8-2.4T-A95B | BF16 | 295.325 ms | 263.112 ms | 198.028 ms | 1.329x |
| 32 | Qwen3.8-2.4T-A95B | FP8 | 252.271 ms | 218.436 ms | 147.938 ms | 1.477x |
| 64 | Qwen3.5-397B-A17B | BF16 | 28.672 ms | 23.899 ms | 18.675 ms | 1.280x |
| 64 | Qwen3.5-397B-A17B | FP8 | 27.363 ms | 22.357 ms | 14.632 ms | 1.528x |
| 64 | Qwen3.8-2.4T-A95B | BF16 | 301.704 ms | 264.823 ms | 204.013 ms | 1.298x |
| 64 | Qwen3.8-2.4T-A95B | FP8 | 259.595 ms | 219.758 ms | 146.228 ms | 1.503x |

EP32 pins source `5c9a872d04363a92a712880b59f70d5390514afb` and native SHA256
`4ed4082f904cf1b5a3ff321c96024b095af33b8a7485ed76c94c657f6625f565` on
hosts 213–216. EP64 pins source
`6574c77aa728fefeba31fb1064d7daaeee6ae46e` and native SHA256
`4ed4082f904cf1b5a3ff321c96024b095af33b8a7485ed76c94c657f6625f565` on
hosts 213–220. Both include the upstream BF16 API merge and passed all 20
validation/timing phases within their physical supernode. Both include the
symmetric-payload cleanup and source ownership across BF16 Qwen N tiles.
Performance evidence remains tied to these revisions. The submission uses the
original serial compiler; these historical measurements do not validate the
changed preparation implementation.

Separately, post-consolidation source
`3ba3f71b4255977e1476505575e3cf75c8170cb4` (native SHA256
`aa4faef2254586d809917cd809c80e3e3b109ef5555c381e793af4e486cce9fa`)
passes the following fresh EP32 fusion-only retest on hosts 213–216:

| Shape | Precision | MegaMoE |
| --- | --- | ---: |
| Qwen3.5-397B-A17B | BF16 | 19.051 ms |
| Qwen3.5-397B-A17B | FP8 | 14.949 ms |
| Qwen3.8-2.4T-A95B | BF16 | 197.973 ms |
| Qwen3.8-2.4T-A95B | FP8 | 144.768 ms |

These are new fusion measurements; the complete paired DeepEP comparisons above
retain their earlier source identities. The latest EP64 fusion retest at
`b8323a5485df51f2ac11e0fc494e173ce9a09633` passes Qwen3.5-397B-A17B BF16/FP8 at
18.660/14.570 ms and Qwen3.8-2.4T-A95B BF16 at 203.947 ms. Its Qwen3.8-2.4T-A95B FP8 first forward stalls
after all ranks finish preparation; it produces no accepted timing. The current
submission restores the earlier pointwise launch grid. A fresh serial-source
`7a43f2d8646320ce783db08dbb54f110f4c67a66` retest on all eight hosts passes
EP64 Qwen3.8-2.4T-A95B FP8 at **147.036 ms**, including the first complete
forward/backward and the unchanged 5/20/5 graph benchmark. Its node-local caches
are preseeded with validated binaries; every rank and host audit passes. This
does not establish the original stall's cause by a single-factor experiment.
The current serial BF16 confirmation is pending. The 146.228 ms FP8 value in
the complete paired table retains its historical source905 identity.

Preparation source `912c95acea52e7099c4a738552599f1a34b163be` additionally fixes
seven late JIT specializations: selected vector activations, owner scores,
raw-gradient DW packing and local FC2 recomputation now compile before peer
payloads. Its full unit suite passes. A fresh eight-PPU cold run with the
Qwen3.8-2.4T-A95B EP64 per-rank geometry (E8, H8192/I2048/T16384) confirms
zero new JIT binaries during the first forward and backward on every rank.
This is a preparation regression check with fewer global experts. The separate
64-rank serial FP8 retest above establishes current EP64 progress and timing.

Timing covers the routed layer's forward and complete backward, live weight
preparation, quantization, required recomputation, and fresh FP32 DW. It excludes
optimizer updates, attention, router projection and shared experts. There are
five eager warmups, capture, five graph warmups, then five samples of 20 graph
replays. Each sample takes maximum-rank event latency; the table shows its
median. Profiling is disabled. Graph-pool allocation occurs during capture;
logical DW initialization and computation run on every replay.

MegaMoE and DeepEP may use different native GEMM/FP8 scale strategies. This is a
comparison of complete implementations. Windowed versus compact separately
preserves GEMM policy and exact recomputation work. The measured gain is
1.26–1.53x; a complete-step 2–3x gain has not been demonstrated. The tested
DeepEP wrapper's 32-bit indexing bounds Qwen chunks to 5504 tokens at EP32 and
3968 at EP64, versus 16384 for MegaMoE. Chunk counts and launch schedules differ;
these results do not isolate communication overlap alone.

## Compilation and small-model validation

Consolidated source955 (`3ba3f71b4255977e1476505575e3cf75c8170cb4`, native
SHA256 `aa4faef2254586d809917cd809c80e3e3b109ef5555c381e793af4e486cce9fa`)
passes its complete DW/JIT/recipe unit suite and three paired small-model runs.
Cold three-path results are retained separately with their actual source IDs.
The source removes fixed-off Python policy branches and an always-returning DW
helper, and retains the established four-CTA-per-CU pointwise launch default.

Parallel preparation reuses an existing loaded module instead of unloading a
runtime referenced by a captured graph. The public `--mode prepare` includes the
two-thread graph regression. A separate forced four-publisher test covers
private scratch cleanup. Source952 (`ee11384fafc4e76f91787089e5992a4136f60a7c`,
native SHA256 `556a0ae406ae761ce2d41be8201a0025b875ff1ff911fe07967c12e07c479c2b`)
passes the default M890P check-in: 257 compilation, 257 accuracy and six
cold/disk/graph cases. Those historical receipts retain their original JIT
implementation and are not acceptance of the current serial preparation path.
Legacy inference at source954 passes 19 MegaMoE and five MegaGate cases;
21 architecture/multi-rank skips and 13 deselected cases are not claimed as
verified coverage.

A 57.8M-parameter full MoE character model (four layers, H512/I1024, E8/top-k2,
attention, router and shared experts) trains for 2000 AdamW steps on pinned
TinyShakespeare text with three matched BF16/FP8 seeds. Routing and expert
weights remain dynamic; FP8 uses native FP8 DW. Source955 evaluates all 111539
heldout next-character targets:

| Seed | BF16 validation nats | FP8 validation nats |
| ---: | ---: | ---: |
| 4827 | 1.597039 | 1.589729 |
| 4891 | 1.590040 | 1.588477 |
| 6113 | 1.599737 | 1.597689 |

This is small-model evidence. BF16 trajectories vary across reruns; it establishes
neither statistical noninferiority nor large-model pretraining convergence.
The pinned model harness, corpus hash, checkpoints and rank logs are retained
outside the minimal operator check-in. Shared-path 810E coverage remains unrun;
the authorized pool contains M890P boards. Distributed validation uses one
process per PPU.
