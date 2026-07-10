# Hints

## Project-specific constraints

- **JIT-compiled kernel**: `.cuh` files in `solution/` are copied to
  `deep_gemm/include/deep_gemm/` before each bench run. The JIT compiler
  auto-detects file changes via MD5 hash — no manual cache clearing needed.
- **Multi-GPU required**: bench uses `torchrun --nproc_per_node=N`. Minimum 2 GPUs.
- **PPU hardware**: ZW-M890P with 39 SMs, ICN8 full-connected NVLink.
- **Check GPU availability** (`ppu-smi`) before running — avoid occupied GPUs.
- **Compile flags**: must use `-gencode=arch=compute_89,code=sm_89`, not `-arch=ppu0015`.
- **Profiling**: use `acu` (not `ncu`) for kernel profiling.

## Optimization focus (2026-07-09: shifted to PRE-GEMM overhead per user task)

- **Primary target**: pre-GEMM overhead (quant + expert_preprocess kernels) on the
  2-GPU pipeline. Baseline ~0.88x non-fused; ~54µs of quant+preprocess is exposed
  (bc_pipeline − bc_kernel_only): quant ~30µs / preprocess ~24µs marginal.
- **Key finding**: pre-GEMM cost is LAUNCH/latency-bound, not compute/BW-bound.
  quant moves ~10MB (~10µs at BW) but takes 34-56µs; preprocess arrival barrier is
  only ~5µs while its launch overhead is ~31µs. Lever = launch count / per-launch cost.
- **Tried, NEUTRAL (don't repeat)**: merge prepare+finalize (MERGED_PREPROCESS),
  fold arrival_push into quant tail (FUSED_ARRIVAL_IN_QUANT) — each removes one launch
  but pays it back (fence / kernel-boundary visibility). See ITERATIONS.md 07-08/07-09.
- **Prior focus (deferred)**: copy/GEMM "persistent blocks" (copy SMs idle after copy,
  4-GPU 0.80x). Revisit after pre-GEMM.

## Environment

- Python env: system python with torch + deep_gemm installed in-place
- ncu equivalent: `ppu-ncu` (may or may not be available — fallback to analytical)
