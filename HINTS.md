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

## Optimization focus (2026-07-15: pre-GEMM preprocess overlap, 承接 SFA push 到 0.81x)

- **来源: 用户 prompt (2026-07-15)**。SFA push 方案已到 0.81x 严格正确 (commit bc3a3f9, 见 SFA_OPT_NOTES.md)。
- **本次任务**: 优化 fused GEMM 前的 pre-GEMM 流水 (quant / expert_preprocess merged prepare+finalize / SFA reshape)。
  1. 用 clock64 拆解各阶段开销, 找最重的。
  2. **验证预期 overlap 的阶段是否真的 overlap 了** (当前 pipeline 单 stream 串行 quant→preprocess→reshape→GEMM, 怀疑没 overlap)。
  3. 允许把本地转置/reshape 等逻辑与 fused GEMM overlap。
- **生产默认**: `DG_SFA_PUSH=1 FUSED_ARRIVAL_IN_QUANT=1` (含 grid-stride/stbl 单uncache/atomic前置/__ldbl)。
- **已知 (勿重复)**: pre-GEMM 是 launch/latency-bound; MERGED_PREPROCESS + FUSED_ARRIVAL_IN_QUANT 已合并进默认; dist.barrier 对齐无效; metadata counts push 净零 (DG_SFA_PUSH_COUNTS 默认关)。

### 用户追加目标（2026-07-15，本轮 prompt）

- DeepEP dispatch 发送 FP4 数据约 **50us**，而 fused pre-GEMM preprocess 也约
  **50us**；用户认为该开销过大，要求优先推进 preprocess 降时。
- 本轮必须拆分 quant、expert prepare/finalize、SFA reshape 的独立与边际暴露，
  优先减少 launch 数或把本地 reshape 融入已有 kernel；不能只调 GEMM 参数。

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

- For expert-preprocess profiling, collect three independent `clock64` runs and
  report arrival polling, CTA synchronization/setup, remote count reads, and
  finalize/layout generation separately. State explicitly which host/device
synchronization is inside each measured interval.

- Before further expert-preprocess rewrites, use `hgobjdump` when available to
  inspect the generated kernel for register pressure/local-memory spills and
  load/ALU/store dependency chains. Prefer changes that expose independent work
and overlap latency without increasing spills.

- While the user is away, continue the quant-last-CTA preprocess-fusion campaign
  autonomously. Do not pause for non-blocking questions; benchmark, log, commit,
  and revert failed candidates before moving to the next evidence-backed direction.

## 用户追加方向（2026-07-16：quant store overlap）

- 尝试让 quant Phase 2 中的 FP4 local scatter 与 SFA peer push 显式 overlap。
- 优先使用 quant/slot atomic 完成后空闲的专职 warp 发射 SFA push，避免同一
  topk warp 先完成全部 FP4 store、再串行发射 SFA store。
- 以 full pipeline 为最终标准，同时观察 quant event 与 P2 clock64；必须保留
  arrival 前的 CTA barrier/system fence 和现有正确性门禁。

## 用户追加测量纪律（2026-07-16：每次汇总对照路径）

- 来源：用户 prompt（2026-07-16）。每次性能运行都要实际测量并汇总：
  1. fused full pipeline；
  2. non-fused full pipeline 与 non-fused GEMM-only；
  3. fused normal-P2P kernel-only；
  4. 将所有 rank 地址映射成本地副本的 all-local kernel-only；
  5. normal-P2P 与 all-local 的差值（P2P exposure）。
- 禁止把 `SKIP_ISOLATION=1` 产生的 `0.000` 当作 kernel 时间。除非某个已知
  remote-only 指令变体无法合法访问 local 地址，否则 benchmark 默认必须启用
  isolation；若因该限制跳过，结果中必须明确标记为 N/A。

## 用户追加协作规则（2026-07-17：远端 Claude 交叉 review）

- 来源：用户 prompt（2026-07-17）。允许调用 swu246 上的 Claude Code 协助
  GPU kernel 分析或提出优化候选；Claude 默认只读工作，Codex 必须独立检查
  源码、profile 与 benchmark 证据，review 其结论，并可通过追问进行交叉讨论。
- Claude 的建议不能直接作为优化结论或代码落地依据；任何实现仍需遵守完整的
  correctness gate、full-pipeline 最终指标和既有迭代记录/提交纪律。

## 用户追加结构方向（2026-07-17：pre-GEMM 大 kernel）

- 来源：用户 prompt（2026-07-17）。SFA overlap/reshape 属于 preprocess，优先
  放入 `expert_preprocess.cuh`，避免 `fp4_gemm_cutlass3.cuh` 反向 include
  preprocess 并污染所有 GEMM JIT。
- 将 count/prefix、metadata pack、SFA reshape 写成可组合 device helper，当前先
  组合为一个 pre-GEMM overlap kernel；为后续把 pre-GEMM 阶段合成一个大 kernel
  保留清晰的 CTA 角色与同步边界。

## 用户追加测量纪律（2026-07-17：锁频门禁）

- 来源：用户 prompt（2026-07-17）。性能 A/B 前必须确认 8 卡 CU/显存频率已经
  锁定，并记录 `ppu-smi` 状态；未锁频数据只能用于同状态方向判断，不能与锁频
  后绝对延迟混用。
- 当前锁频状态为 CU 1300 MHz、memory 1600 MHz，空闲功耗约 225–240 W；此前
  未锁频慢档空闲功耗约 155–160 W，GEMM 路径整体慢约 35–41%。
