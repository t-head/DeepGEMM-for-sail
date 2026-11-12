# M890 BF16 HBM read performance reproduction

This document describes the measurement protocol used to compare
`deep_gemm.bf16_gemm_nt` on M890 against a cuBLAS reference, and the execution
policies of the default BF16 CuteFree path. `tests/bench_bf16_hbm_read.py`
implements the protocol. It is an opt-in performance tool, outside check-in and
daily UT.

## Execution policies

The default path selects a tile first, then resolves its cache policy and
scheduling mode. Let `Tm = ceil(M/BM)`, `Tn = ceil(N/BN)` and
`C = scheduled_sms * blocks_per_cu`. Both policies support BF16 and FP32 output.

| Policy | Selection rule and purpose |
| --- | --- |
| B read-through | Enabled when `Tm == 1`, equivalently `0 < M <= BM`. Each B tile is consumed by one output CTA, so there is no B reuse across M tiles to cache for. There is no B-size or L2-size threshold. Multiple-M-tile problems keep the default policy. |
| Non-persistent dispatch | One CTA per output tile. Enabled through one wave (`Tm*Tn <= C`); single-M-tile problems may extend to two waves (`Tm*Tn <= 2*C`). |
| CU budget | A non-persistent grid spreads over the device, so it is used only when the CTA count fits the caller's budget, or when the caller requested the full physical CU count. |
| Tile mapping | The non-persistent grid retains the persistent scheduler's M-pair swizzle, including odd M-tile counts. |

The policies preserve K traversal and accumulation order. They are selected
rules, not a per-shape performance guarantee: removing cross-call cache reuse
does not have to be a win for every shape.

`select_dense_bf16_tile` holds four measured tiles ahead of the shared adaptive
selector. They apply only to ZW-M890P with 39 physical CUs and the full 39-CU
scheduling budget; other devices, other CU counts and reduced budgets use the
general selector. Shapes are M,N,K.

| M,N,K | Block | Warp | Stages |
| --- | --- | --- | ---: |
| 32,7168,12288 | 32×192×64 | 16×32×64 | 6 |
| 32,7168,8192 | 32×192×64 | 32×16×64 | 6 |
| 64,7168,8192 | 64×192×64 | 32×32×64 | 7 |
| 64,12288,1536 | 64×320×64 | 64×32×64 | 4 |

These entries use deeper pipelines than the general selector chooses. A different
tile changes floating-point summation order, so reference-error checks do not
imply bitwise equality across tiles.

GEMV keeps dispatch priority. Explicit configurations and external runtime LUT
entries keep their existing paths and take precedence over automatic selection.
`DG_BF16_MEMORY_TUNING=0` disables the execution policies while retaining adaptive
tile selection; `DG_BF16_ADAPTIVE=0` also bypasses adaptive selection, including
the four measured entries.

## Workload and metric

- BF16 contiguous A[M,K], B[N,K], BF16 D[M,N], computing A × Bᵀ.
- M = 1, 2, 4, 8, 16, 32, 64 for each N/K pair:
  49152/7168, 7168/12288, 7168/8192, 12288/1536, 2112/7168. 35 shapes total.
- M890: installed `deep_gemm.bf16_gemm_nt` with its normal automatic selector.
- NVIDIA: `torch.nn.functional.linear`. Loaded cuBLAS/cuBLASLt paths, version
  integers and SHA-256 fingerprints are recorded after initialization.
  Algorithm selection belongs to that environment; this is not an offline
  cuBLAS algorithm sweep.
- Default inputs: seeded GPU BF16 normal random A divided by sqrt(K) and normal
  random B. Every output is checked against a complete CPU FP32 reference with
  normalized squared error `sum((D-ref)^2) / max(sum(D^2)+sum(ref^2), 1e-30) < 1e-5`.
  `--input-mode integer` instead uses BF16 integers in [-2,2] and exact BF16
  output equality; record and match the input mode across compared runs.
  Allocation, references and 64 warm-up calls happen before profiling. PPU
  outputs also carry guard regions, and TF32 is disabled for both backends.
- Exactly one complete API call is enclosed by profiler start/stop per capture.
  Sum **all** its kernel durations, including auxiliary and reduction kernels.
- Use `--profile-from-start off --cache-control all --clock-control none` and
  collect duration only. The protocol uses no manual memset eviction and no
  CUDA-event timing.
- Take the median of complete-call kernel sums across seeds and samples. The
  peak is the maximum utilization among these **shape medians**.

```text
logical_read_bytes = 2 × (M×K + N×K)
logical_read_TB/s  = logical_read_bytes / (median_kernel_us × 1e6)
logical_read_util = logical_read_TB/s / reference_bandwidth_TB/s × 100%
speedup           = NVIDIA_median_kernel_us / PPU_median_kernel_us
```

These are logical input-read metrics. They exclude output writes and do not
substitute physical DRAM counter bytes, so the percentage is normalized to the
explicit reference denominator and is not ACU's physical HBM throughput metric.
Kernel-duration sums exclude CPU selection and launch work and inter-kernel
gaps; they are not application latency. The profiler may flush before each
kernel replay, including within a multi-kernel API, so keep this boundary
identical for compared runs.

## Environment

Use an idle device and the same released operator build, SDK/driver, device SKU,
CU budget and relevant environment flags as the result being reproduced. An
older installed wheel can select different kernels even when the benchmark
source is identical. Compare `summary.json` metadata and library fingerprints
with the release record before comparing numbers; the run records the exact
build identifiers.

M890 requires the vendor PyTorch build for M890, installed DeepGEMM and working
Asight Compute (`acu`). NVIDIA requires CUDA PyTorch and Nsight Compute (`ncu`).
Set tool paths according to the installed SDK. The NVIDIA entry and the
aggregation step import no accelerator-specific dependency.

The M890 product specification uses **2.664 TB/s** and the H200 comparison uses
**4.8 TB/s**. Keep these values as the utilization denominator;
`--hbm-tb-s` supplies the specification bandwidth for the device under test and
the script does not infer it from a product label.

Record device state before and after a run with `ppu-smi` / `nvidia-smi`,
including other processes, temperature, power and clocks where available.
Schedule an idle device; the script changes no clocks, power, CU masks or
exclusive mode.

## Reference results

24 cold samples per shape for both versions, 1,680 API captures in total, each
with full-output and guard checks. Every capture contained exactly one GEMM
kernel, verified from the raw report, and its duration was the complete API
kernel sum.

| Operator | Peak shape M,N,K | Shape median | Read TB/s | Read / 2.664 TB/s |
| --- | --- | ---: | ---: | ---: |
| Historical report | 1,49152,7168 | 320.98 us | 2.195 | 82.41% |
| Archived algorithm, rebuilt | 1,49152,7168 | 317.21 us | 2.221 | 83.39% |
| Same rebuilt algorithm, customer script | 1,49152,7168 | 318.45 us | 2.213 | 83.06% |
| Current, paired phaseclean regression | 1,49152,7168 | 319.43 us | 2.206 | 82.81% |

The customer-script row uses this repository's `collect` command with three seeds
and eight samples per seed: 24 independent profiler processes, complete
CPU-reference checks and no benchmark-specific operator code.

The paired regression compared the phaseclean baseline (production without
K-phase) against the current source, both rebuilt in one environment, sharing the
same inputs and alternating versions. M=1 was 317.87 us for the baseline and
319.43 us for current. Across seven M values the current-minus-baseline latency
difference was -0.06% to 0.53%, median 0.26%.

K-phase is not part of the current kernel. It staggers K traversal by
`blockIdx.x`, which changes accumulation order and can make identical inputs
produce different values in different output regions; a reference-error tolerance
does not establish region consistency. The archived report build that used it is
historical evidence, not a baseline for this branch.

## 1. Verify correctness

Run from the repository root with the interpreter belonging to each installed
environment. The benchmark records the loaded module path and binary hashes.

```bash
# On M890:
DG_JIT_USE_HGRTC=0 CUDA_VISIBLE_DEVICES=0 python tests/bench_bf16_hbm_read.py check \
  --backend ppu --output results/ppu-check

# On the NVIDIA comparison server:
CUDA_VISIBLE_DEVICES=0 python tests/bench_bf16_hbm_read.py check \
  --backend cublas --output results/cublas-check
```

Both check all 35 shapes. Use `--shape 64,7168,8192` for a diagnostic subset.
Output directories must be new to prevent mixing stale samples.

## 2. Collect cold-cache kernel times

M890 uses 24 samples per shape: three seeds with eight samples each. NVIDIA
uses five samples per shape.

```bash
# M890: 35 × 3 × 8 = 840 independent captures.
DG_JIT_USE_HGRTC=0 CUDA_VISIBLE_DEVICES=0 python tests/bench_bf16_hbm_read.py collect \
  --backend ppu --hbm-tb-s 2.664 \
  --seeds 2701 2702 2703 --samples 8 --output results/ppu

# NVIDIA/H200: 35 × 1 × 5 = 175 independent captures.
CUDA_VISIBLE_DEVICES=0 python tests/bench_bf16_hbm_read.py collect \
  --backend cublas --hbm-tb-s 4.8 \
  --seeds 2701 --samples 5 --output results/cublas
```

Confirm profiler availability first with
`--shape 1,49152,7168 --seeds 2701 --samples 1`. Full runs take tens of minutes
or longer: each sample starts a fresh profiler process, validates the output and
warms the operator. This isolation gives every CSV exactly one API call and
avoids splitting multiple kernels across samples.

The generated commands are retained in `*.command.json`:

```text
acu --profile-from-start off --cache-control all --clock-control none \
    --metrics ppu__time_duration.sum --csv --csv-file SAMPLE.csv ...

ncu --profile-from-start off --cache-control all --clock-control none \
    --metrics gpu__time_duration.sum --csv --page raw --print-units base \
    --log-file SAMPLE.csv ...
```

`--timeout` sets the per-process timeout (default 180 seconds). Profiling
errors, unavailable metrics, missing or zero timings, duplicate launch rows,
incomplete captures and changed environments fail the run. A failed run retains
its logs and an incomplete plan without accepting a performance summary, and no
event-based result is substituted for a missing profiler capture. A successful
correctness run does not certify profiler availability, so validate the
profiler on the target host with a native report before comparing numbers.

## 3. Aggregate and compare

Collection summarizes automatically after every planned capture succeeds.
Summaries can be rebuilt later, including on a CPU server without Torch:

```bash
python tests/bench_bf16_hbm_read.py summarize results/ppu
python tests/bench_bf16_hbm_read.py summarize results/cublas

python tests/bench_bf16_hbm_read.py compare \
  results/ppu/summary.json results/cublas/summary.json \
  --output results/ppu-vs-cublas.csv
```

Outputs:

- `plan.json`: complete shape and sample plan, profiler version, harness hash.
- `case-*.command.json`, `.log`, `.csv`: exact command and raw profiler output.
- `case-*.json`: correctness, operator binary and JIT-header fingerprints,
  compiler version, loaded cuBLAS versions and selection/precision flags.
- `summary.json`: per-kernel times, complete-call sums, all samples, medians,
  bandwidth and the peak of shape medians. `summary.csv` is the compact table.
- Comparison CSV: matched shapes, both latencies, their ratio and read bandwidths.

The reader accepts ACU wide duration columns and NCU raw and metric-row formats
with explicit units. If another tool version changes the format or unit
convention, preserve the raw data and validate the reader before publishing
numbers.

For different cuBLAS versions, run in their respective independently prepared
environments and separate result directories. The loaded library versions and
hashes in each summary establish which version was tested.

## Diagnostic counters

Physical DRAM counters confirm the cold-read condition but are not the reported
metric. Preserve a native report and import it rather than relying on
`--csv-file`, which may contain duration only even when more metrics were
requested. Do not treat absent columns as zero traffic.

```bash
mkdir -p results/diagnostic
METRICS=ppu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum
METRICS=$METRICS,ce__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second
DG_JIT_USE_HGRTC=0 CUDA_VISIBLE_DEVICES=0 acu \
  --profile-from-start off --cache-control all --clock-control none \
  --metrics "$METRICS" \
  --check-exit-code yes -o results/diagnostic/sample \
  python tests/bench_bf16_hbm_read.py _sample --backend ppu \
  --shape 1,49152,7168 --seed 2701 --output results/diagnostic/sample.json
acu --import results/diagnostic/sample.acurep --csv --page raw \
  > results/diagnostic/sample.csv
```

Inspect `dram__bytes_read.sum`, `ce__cycles_elapsed.avg.per_second` and
`dram__cycles_elapsed.avg.per_second`. Also sample N=2112, K=7168, whose B fits
in the 32 MiB L2, so a large working set alone cannot explain its DRAM reads.

## Timing references

NCU's [CLI options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
define capture ranges, metric CSV output, cache and clock controls.
Its [cache-control documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)
describes flushing before each replay. Check ACU options with the installed
`acu --help`; timestamp implementation details are tool-specific.
PyTorch exposes the active cuBLAS handle through
[`current_blas_handle`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.current_blas_handle.html).
Version APIs are queried from libraries already loaded by that process.