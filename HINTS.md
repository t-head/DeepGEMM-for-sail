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

## Optimization focus (2026-07-11: 新仓库 DeepGemm-block-copy-fusedopt, 聚焦 8卡 fused copy block GEMM kernel)

- **本次任务**: 在新拷贝仓库优化 fused copy block GEMM 性能, 目标 8卡 (prod, world_size=8)。
- **容器**: `sglang0512.lxh`, 路径 `/DeepGemm_workspace/codebase/DeepGemm-block-copy-fusedopt`。
- **8卡基线 (d3e85f0)**: pipeline 0.291ms/0.87x (ncb=3); kernel-only BC=0.249 / all-local=0.231 / NF=0.194ms; P2P +0.018ms; pre-GEMM (quant+preproc)=0.043ms。
- **主杠杆假设**: fused GEMM all-local 比 NF 慢 0.037ms, 疑似 ncb 个 copy block 做完 copy 后 idle (损失 ncb/39≈7.7% 算力)。P2P 已很小。
- **迭代节奏**: signal=8卡 perf-only (SKIP_ISOLATION 可选, NCB_SWEEP 精简 2,3,4); verdict=8卡 FULL_CORRECTNESS 必须 bit-exact。
- **测量纪律 (用户要求 2026-07-11)**:
  1. 每次运行前先 `ppu-smi` 确认机器空闲 (No running processes), 避免抢占/被抢占。
  2. 每个优化至少在空闲时**跑 3 次**取中位/看方差, 确认性能稳定 (小改动易被噪声淹没, 见 memory feedback_measure_small_kernel_changes)。

- **Tried NEUTRAL/负: copy loop 换通用 PPU bulk-DMA (swizzled)** — 2026-07-11。
  bulk load/store 是 warp-collective shuffle 对, 必须配对+满 warp-wave (见 ITERATIONS 同日)。
  正确写法 (满 wave bulk + 尾部 plain) 4卡 nr4 bit-exact, 但 **P2P exposure 不变 (+0.020), kernel-only +0.005**
  → 通用 bulk 没加速 remote 读; copy 已 coalesced 8-wide MLP 近最优。opt-in `DG_BULK_COPY=1`(默认关)。
- **★ Tried WIN: remote 专用 bulk load (DG_BULK_REMOTE)** — 2026-07-12。warp-aligned body 里 remote rank 用
  `__ppu_remote_load_bulk_b32x4` (local rank 用通用 bulk load), 都由 `st_bulk_global`(.cg) un-swizzle; 尾部 plain。
  local/remote 判定 = `r==rank_idx` (已把 rank_idx 从 .py 经 sched 传进 copy loop)。**8卡 nr8 pipeline ~0.345→~0.339
  (-2%, 稳定, bit-exact), 首个稳定 beat baseline 的变体** (加速 88% remote 读)。⚠️ 必须 `SKIP_ISOLATION=1`
  (all-local isolation 会让 remote 指令打 local 地址 → illegal crash)。opt-in `DG_BULK_REMOTE=1`(默认关)。

## 关键瓶颈分析 (2026-07-11, 代码+算术推理)

- GEMM tile 数 = (M/block_m)×(N/block_n) = (1536/128)×(6144/256) = 12×24 = **288 tiles**。
- fused GEMM 用 36 SM (39 - ncb=3 copy blocks) → 288/36 = **8 整 wave**。
- **回收 idle copy SM 无效**: ceil(288/39)=8 wave 仍是 8, SM 36→39 不减 wave。这解释 iter1/iter10 为何失败——瓶颈不是 SM 数。
- NF 用 block_m=256 → (1536/256)×(6144/256)=6×24=144 tiles/39 = **4 wave**。fused 因 block_m=128 被迫 8 wave (每 wave 有 epilogue/流水线固定开销), 这是结构性慢因。
- fused **不能**用 block_m=256 (per-expert M=128 会 2x padding 浪费), 是 masked-grouped 死结; 根治需 GroupedContiguous (大改)。
- **可试杠杆**: block_n 256→512 把 N_blocks 减半 → 12×12=144 tiles/36=4 wave, 计算量不变但 wave 减半。smem 需 <~210KB (NF 用 209600 可行)。copy block 只搬 A (与 block_n 无关), 应透明。

## (历史) Optimization focus (2026-07-09: shifted to PRE-GEMM overhead per user task)

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
