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

## Optimization focus

- **Primary target**: 4-GPU performance (currently 0.80x non-fused, biggest room for improvement)
- **Secondary target**: 2-GPU (currently 0.97x, near optimal)
- **Key bottleneck**: 8 copy SMs idle after copy completes (20% SM waste)
- **Top optimization direction**: Persistent blocks — copy blocks transition to GEMM after copy

## Environment

- Python env: system python with torch + deep_gemm installed in-place
- ncu equivalent: `ppu-ncu` (may or may not be available — fallback to analytical)
