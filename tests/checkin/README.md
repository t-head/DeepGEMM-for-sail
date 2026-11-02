# Functional check-in

Check-in keeps representative cases for supported operators and independent execution paths. Full shape/distribution/tile sweeps belong in daily functional lists; performance measurements belong in perf lists. Board suites run their functional caselist; the separate single-device MegaMoE suite is described below. The test scripts and tile scans in `tests/daily/daily_810e.list` / `tests/daily/daily_890p.list` run through the explicit daily entry in `tests/daily/run_daily.py`; they are not appended to default check-in. The retired `DG_USE_CUTE=0` fallback is excluded.

```bash
python tests/checkin/run_checkin.py --suite ZW-810E
python tests/checkin/run_checkin.py --suite ZW-M890P
# Fewer workers can reduce process startup overhead with an already warm JIT cache:
python tests/checkin/run_checkin.py --suite ZW-M890P --compile-workers 4
# Inspect a board's list on a CPU host, without importing Torch:
python tests/checkin/run_checkin.py --suite ZW-M890P --list
```

Each non-comment line is one `run_deep_gemm.py --format` specification, not a shell command. Nested `.list` paths are relative to their containing file. `--caselist PATH` selects a custom list instead of a board suite. The defaults are sixteen compiler workers and a 1200-second timeout per case per phase; reduce workers on memory-constrained machines.

The scheduler uses one independent reusable subprocess per compiler worker with `HGGC_WARM_UP=1`. Workers import Torch once per phase. It waits for every compile result before running any accuracy case, then runs accuracy cases serially in a separate reusable subprocess. A compile failure prevents the accuracy phase. Accuracy failures are collected while remaining cases continue. Both phases require each selected case to be supported on the device; unsupported cases cannot silently pass. `HGGC_WARM_UP` is removed from accuracy subprocesses, including an inherited value of `0` (any nonempty value enables warm-up).

Warm-up disables accuracy, metadata output checks, signal checks and benchmarking, but retains the output formats and kernel variants needed for the real run. Sparse cases also precompile the dense JIT reference kernels. BF16 Dense CuteFree honors the same warm-up check as the common launch runtime. Data preparation through Torch still runs; warm-up is not a driver-wide prohibition on device work.

For W4 weights, warm-up prepares one synthetic expert through the production preprocessor and repeats its final contiguous layout to the full expert bank. This avoids full reference quantization/decoding and the preprocessor's full-bank INT32 temporary tensors. GEMM compilation retains the original dimensions, types and layouts. Accuracy and ordinary `--disable_acc` benchmark runs continue to prepare real quantized inputs. Sixteen compiler workers passed both boards' full lists with cold and warm caches. The default and CI use sixteen workers to shorten cold-cache check-in compilation. With an already warm JIT cache, explicitly selecting four workers can reduce process startup overhead.

`test-results.xml` distinguishes `DeepGEMM.compile` and `DeepGEMM.run` results. Complete per-case output is in `checkin-logs/compile-NNNN.log` and `run-NNNN.log`. CI copies the XML, environment information and logs to the board's existing NAS result directory. Each case resets its random seeds and accuracy/benchmark controls. Timeout handling terminates the whole child process group, including the compiler; the next case starts a new process after a timeout or crash. Ordinary UT exceptions are logged and collected without restarting Torch. Worker startup/teardown output is retained in `worker-*.log`.

Default check-in does not run the full `test_core.py`, `test_fp4_core.py`, `test_fp8_core.py`, `test_attention.py` shape loops or either tile sweep. Required functionality stays in the operator cases: small-M/N/K tails, quantization layouts, masked overlap/signals, attention regression/metadata checks and preprocessing contracts. FP4 default-epilogue NoPad/Fused checks retain the old strict `0.001` accuracy boundary; fused activation/quantization paths retain their existing thresholds.

For full shape and explicit-tile coverage, use the daily entry. It first runs the functional caselist, then the board-specific daily list (`tests/daily/daily_810e.list` / `tests/daily/daily_890p.list`) directly, retaining per-command timeouts and failure-output detection. Both tile scans keep `--thread_count=32`. Daily scripts run after accuracy even if accuracy fails, and never after a compile failure. `HGGC_WARM_UP` is removed for real checks. Results appear as `DeepGEMM.script` alongside compile/run results. Raw script output is in `checkin-logs/script-*.log`.

```bash
python tests/daily/run_daily.py --suite ZW-810E
python tests/daily/run_daily.py --suite ZW-M890P
python tests/daily/run_daily.py --suite ZW-M890P --skip-checkin
python tests/daily/run_daily.py --suite ZW-M890P --list-daily-scripts
```

The daily entry runs the checkin functional list by default. Add `--skip-checkin` to run only the board-specific daily list while still generating reports and retaining failure logs. This option requires `--suite` and cannot be combined with `--caselist` or `--phase compile`. It does not affect the standalone checkin entry.

The daily-only sweep logic (sweep lists, per-command timeouts and failure-output detection) lives in `tests/daily/run_daily.py`; `run_checkin.py` keeps only shared scheduling and reporting and has no sweep options. Full cold-cache shape sweeps can exceed the original script timeouts; daily is not the check-in latency target. `--phase compile` remains compilation-only, including for the daily entry.

The shared CI workflow uses the functional check-in with sixteen compiler workers in both `deepgemm` and `actlize1.0.0` source modes. The actlize mode clones DeepGEMM main and replaces its actlize submodule before building; both modes require `tests/checkin/run_checkin.py`.

Case logs include peak Torch allocated and reserved device memory, reset before each case and measured before releasing its cached allocations.

For manual debugging, `--phase compile` only precompiles, and `--phase run` runs correctness serially. A custom `--caselist` does not select daily script checks. Compilation-only results are not accuracy results. Keep the wheel, device and compilation-related environment identical between the phases. If overriding cache roots, use the same `DG_JIT_CACHE_DIR` (C++) and `DG_CACHE_DIR` (Python) in both phases and all workers.

A single case can also be replayed directly:

```bash
python tests/run_deep_gemm.py --caselist tests/checkin/checkin_890p.list --case_idx 1 --require_supported
```

`--case_idx` counts executable cases, excluding comments and blank lines. Warm-up is recognized by this entry point as well.

For masked GEMM, the `test_m_grouped_gemm_masked->test_func:` log line contains a replayable `--format` specification with the generated expert counts, padding, scheduler hint, epilogue, clipping limit and bias setting. Pass that specification to `tests/run_deep_gemm.py --format` to repeat the same case configuration.

## MegaMoE

The single-device suite runs the public-entry smoke tests, including all three
activations, separate replica weights, invalid inputs and workspace reuse:

```bash
python tests/checkin/run_checkin.py --suite ZW-M890P-MegaMoE
```

It compiles and executes the pipeline together, without `HGGC_WARM_UP` or the
legacy `testcase.json`. `--phase compile` is not supported for this suite.
Output and failures are retained in `checkin-logs/script-0001.log` and the usual
JUnit report. Multi-card tests are disabled in automatic check-in because a
runner may have only one device. On a host with two free PPU1.5 devices, run:

```bash
NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
torchrun --standalone --nproc-per-node=2 --max-restarts=0 \
  -m pytest -q -s --disable-warnings -x -k smoke tests/test_mega_moe.py
```

## Additional functional specifications

The unified entry point calls `test_activation.py` and `test_fp4_core.py` directly for the corresponding regressions. Their warm-up and accuracy phases share the same test functions.

- `BatchGemm`: `groups=h`, `m=b`, `n=d`, `k=r`; supports complete `expr:bhr,hdr->bhd` and `expr:bhr,hdr->hbd` expressions. INT8 uses channel quantization; FP8 has block/channel cases on 890P.
- `Preprocess,data_type:fp4`: preserves the independent MXFP4 preprocessing contract checks, including packed scales, scale-only calls, odd group counts, strides and aliases.
- `Activation,data_type:bf16,activation:silu,quant_mode:BF16`: 890P standalone SiLU/SiTU smoke tests cover BF16, FP8 channel/block, MXFP8 and both MXFP4 scale layouts. Each case checks dense and masked output, including an empty expert and an undersized scheduling hint. `data_type:bf16` describes the input, while `quant_mode` selects the output.
- `DenseGemm,data_type:bf16_bf16_fp32`: BF16 inputs with real FP32 outputs, preserving main's strict planted-element check against BF16 rounding. Both boards include main's dense, tail and GEMV cases, using the default kernel path.
- `GroupedMasked` / `GroupedNoPad,data_type:w4a8_int8`: 810E packed INT4 weights with INT8 activations. Cases from main retain large shapes, K=32/K=160 tails, and both small-M GEMV shapes; weight packing/preprocessing is exercised by the original generators.
- `MqaAvgLogits,data_type:fp4`: 890P covers all three Q-scale modes, both output dtypes and compressed/full output, plus partial Q/KV tiles, KV balancing and multi-MMA-group execution. Full Cartesian sweeps remain in the retained attention script.
- `with_bias:1`: FP4 Dense/GroupedNoPad/GroupedMasked Default epilogues; bias and fused activation epilogues are tested separately.

The added representatives also preserve INT8 contiguous, masked overlap/signal checks, FP8 channel contiguous/fused, BF16 paged attention, FP4 dense, and 890P BF16 dense M=2/grouped small-M GEMV. Existing colleague-authored cases are retained. FP4 masked overlap is currently unimplemented (the device template only specializes overlap=false); it is excluded rather than treated as a supported positive case.

CPU checks for parsing, scheduling, process environment, timeout/reporting and failure exit status:

```bash
python -m unittest discover -s tests/checkin -p test_checkin_runner.py -v
```

Actual HGCC compilation, device accuracy, cache reuse and timing must be validated on both PPU boards.

Device checks comparing W4 warm-up shapes, types and strides with real quantization, and ensuring `--disable_acc` alone keeps real inputs:

```bash
python tests/checkin/test_warmup_inputs.py -v
```
