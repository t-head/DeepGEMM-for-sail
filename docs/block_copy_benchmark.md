# Block-Level Copy + GEMM Fused Dispatch Benchmark

## Hardware
- PPU ZW-M890P × 8 (39 SMs, ICN8 full-connected)
- Config: 13 experts/rank, hidden=7168, N=6144, max_tokens=256, topk=6

## Pre-Optimization Baseline

Date: 2026-06-30

使用加固后的测试脚本（逐元素 CPU reference 比较、median 计时、P2P remote 修正）。

### 正确性发现

**Swizzle 竞态 bug 确认**：多 M-block experts 在 `fetch_next_work` 的 swizzle 映射下存在竞态条件。
- 单 M-block experts: 全部 PASS（elem diff = 0.000000）
- 多 M-block experts: 随机 FAIL（elem diff 0.002 ~ 0.22，取决于 ncb 和时序）
- ncb=1: Expert 9,11,12 FAIL; ncb=8: Expert 1 FAIL（不同的 expert 在不同时序下出问题）
- 根因：`get_swizzled_block_idx` 交错映射跨 M-block，但 `copy_ready_flags` 检查未考虑 swizzle 后的映射
- 将在 Task 2 (N-major + skip swizzle) 中修复

### 2-GPU Baseline (GPU 4,6 / 6,7), M=1648, block_m=128

3 次测试，数据非常稳定（<1% 波动）。

| Run | GPU pair | Pipeline (ms) | Kernel-only (ms) | Non-fused (ms) | vs NF |
|-----|----------|--------------|------------------|----------------|-------|
| 1 | 4,6 | 0.462 | 0.442 | 0.384 | 0.83x |
| 2 | 4,6 | 0.463 | 0.444 | 0.384 | 0.83x |
| 3 | 6,7 | 0.463 | 0.438 | 0.382 | 0.83x |

**NCB Sweep (典型值)**:

| ncb | Pipeline (ms) | vs NF |
|-----|--------------|-------|
| 1 | 1.617 | 0.24x |
| 2 | 0.893 | 0.43x |
| 4 | 0.587 | 0.65x |
| **8** | **0.462** | **0.83x** |

### 4-GPU Baseline (GPU 2,3,6,7), M=1688, block_m=256

3 次测试。Block-copy pipeline 极稳定（0.450-0.451ms），Non-fused Run 1 偏高（cold start）。

| Run | Pipeline (ms) | Kernel-only (ms) | Non-fused (ms) | vs NF |
|-----|--------------|------------------|----------------|-------|
| 1 | 0.451 | 0.426 | 0.444* | 0.98x* |
| 2 | 0.451 | 0.427 | 0.353 | 0.78x |
| 3 | 0.450 | 0.426 | 0.355 | 0.79x |

*Run 1 non-fused 异常，疑 cold start

**NCB Sweep (典型值)**:

| ncb | Pipeline (ms) | vs NF |
|-----|--------------|-------|
| 1 | 1.265 | 0.28x |
| 2 | 0.791 | 0.45x |
| 4 | 0.621 | 0.57x |
| **8** | **0.451** | **0.78x** |

### Baseline Summary

| Config | Pipeline (ms) | vs NF | 待优化空间 |
|--------|--------------|-------|-----------|
| 2-GPU (ncb=8) | 0.462 | 0.83x | 0.078ms (20%) |
| 4-GPU (ncb=8) | 0.451 | 0.78x | 0.097ms (28%) |

**Kernel-only 分析**（2-GPU）：
- Pipeline overhead (quant+preprocess) ≈ 0.020ms — 很小
- Kernel-only (copy+GEMM) = 0.442ms — 性能瓶颈
- Non-fused GEMM-only = 0.248ms
- 差值 0.194ms 来自: copy blocks 占 SM (8/39=20% 算力减少) + GEMM spin-wait + MC contention

---

## Task 2: Swizzle 竞态修复（M-major 无 swizzle）

Date: 2026-06-30

### 修改内容

`scheduler_cutlass3.cuh` `fetch_next_work()`: 对 `FusedDispatch` 路径，跳过 `get_swizzled_block_idx`，
改为直接 M-major 分解 `(m_block = idx / N_tiles, n_block = idx % N_tiles)`。

**Bug 根因**：`get_swizzled_block_idx` 的 `kNum1DBlocksPerGroup=2` 交错映射使得同一个 `block_m_idx`
的 work items 映射到不同的局部 M-block。GEMM 的 `copy_ready_flags` 只检查全局 M-block 索引，
不考虑 swizzle 后的映射，导致 GEMM 读取未拷贝完的数据。

**修复方案**：去掉 swizzle，直接使用 M-major 线性分解，确保 `curr_global_block_m_idx` 与实际数据位置一一对应。

**同时验证了 N-major 方案**：N-major 修复了竞态但导致性能回退 8%（0.494ms vs 0.462ms），
因为 N-major 让 GEMM 立即需要所有 M-blocks 的数据，与 copy 顺序不匹配。最终采用 M-major 无 swizzle。

### 正确性

ncb=1（最易触发竞态）下 ALL 13 experts PASSED（elem diff = 0.000000），包括之前 FAIL 的多 M-block experts。

### 2-GPU Results (GPU 6,7), M=1648, block_m=128

| Run | Pipeline (ms) | Kernel-only (ms) | Non-fused (ms) | vs NF |
|-----|--------------|------------------|----------------|-------|
| 1 | 0.477 | 0.461 | 0.382 | 0.80x |
| 2 | 0.476 | 0.458 | 0.383 | 0.81x |
| 3 | 0.476 | 0.460 | 0.382 | 0.80x |

vs Baseline: 0.476ms vs 0.462ms = **-3% (去掉 swizzle L2 优化)**

### 4-GPU Results (GPU 2,3,6,7), M=1688, block_m=256

| Run | Pipeline (ms) | Kernel-only (ms) | Non-fused (ms) | vs NF |
|-----|--------------|------------------|----------------|-------|
| 1 | 0.451 | 0.427 | 0.352 | 0.78x |
| 2 | 0.452 | 0.425 | 0.356 | 0.79x |
| 3 | 0.451 | 0.425 | 0.352 | 0.78x |

vs Baseline: 0.451ms vs 0.451ms = **无变化**（4-GPU block_m=256, 每 expert 仅 1 M-block, swizzle 无效果）

### Analysis

2-GPU 的 3% 回退来自 swizzle 在 block_m=128 时提供的 L2 B-matrix 复用（同 expert 相邻 M-blocks 交错
处理同一 N-tile）。4-GPU 下 block_m=256 导致大多 expert 只有 1 M-block，swizzle 无意义，所以无回退。
此回退是修复正确性 bug 的必要代价。

---

## Task 5: 去除 8-alignment Padding + Config 修复

Date: 2026-06-30

### 修改内容

**1. 去除 8-alignment padding**（`dispatch_preprocess.cuh`）

去掉 `dispatch_expert_preprocess` 中 3 处 `(c + 7) & ~7u` padding：
- Line 386: `padded_total += (c + 7) & ~7u` → `padded_total += c`
- Line 448: `smem_row += (take + 7) & ~7u` → `smem_row += take`
- Line 493: `smem_row += (take + 7) & ~7u` → `smem_row += take`

效果：每个 expert-rank pair 的 token 数不再强制 8 对齐，减少 shape_m 约 6%。

**2. Config 选择修复**（`dispatch_fused_gemm.py` + `test_block_copy_gemm1_multi_gpu.py`）

去除 padding 后 4-GPU shape_m 从 1688→1547，`expected_m = ceil_div(1547, 13) = 119`。
由于 119 < 128，`ceil_div(119, 128) = 1 = ceil_div(119, 256)`，config selector 因 m_util 优势
（93% vs 47%）选择 block_m=128（block_k=64），导致 4-GPU 性能回退 10%。

**根因**：block_m=128 和 block_m=256 在 expected_m ≤ 128 时产生相同数量的 M-tiles，但 config selector
不考虑 block_k 差异（256x256 tile 获得 block_k=128，compute AI 更高）。

**修复**：`expected_m = max(expected_m, 129)`。当 expected_m ≤ 128 时，强制为 129 使得
`ceil_div(129, 128) = 2 > ceil_div(129, 256) = 1`，让 selector 正确权衡 wave count 而选择 block_m=256。

### 正确性

2-GPU 和 4-GPU 均 PASSED（逐元素 CPU reference 比较）。

### 2-GPU Results (GPU 2,4), M=1562, block_m=256

| Run | Pipeline (ms) | Kernel-only (ms) | Non-fused (ms) | vs NF |
|-----|--------------|------------------|----------------|-------|
| 1 | 0.402* | 0.371 | 3.741* | — |
| 2 | 0.395 | 0.372 | 0.384 | 0.97x |
| 3 | 0.397 | 0.373 | 0.384 | 0.97x |

*Run 1 non-fused 异常（JIT cold start）

vs Task 2 baseline: **0.476ms → 0.396ms = +17% 提升**（0.80x → 0.97x）

### 4-GPU Results (GPU 2,4,6,7), M=1547, block_m=256

| Run | Pipeline (ms) | Kernel-only (ms) | Non-fused (ms) | vs NF |
|-----|--------------|------------------|----------------|-------|
| 1 | 0.448* | 0.422 | 2.679* | — |
| 2 | 0.447 | 0.422 | 0.353 | 0.79x |
| 3 | 0.448 | 0.423 | 0.359 | 0.80x |

*Run 1 non-fused 异常（JIT cold start）

vs Task 2 baseline: **0.451ms → 0.447ms = +1% 提升**（0.78x → 0.80x）

### Analysis

**2-GPU 大幅提升的原因**：
- 去 padding 减少 shape_m（1648→1562，减少 86 tokens 的无效计算）
- Config 修复使 block_m 从 128→256：block_k 从 64→128，K-loop 迭代次数减半（112→56），
  pipeline 效率更高。虽然 M-utilization 降低（93%→47%），但 tile 数不变（每 group 仍是 1 个 M-tile），
  总 FLOPs 相同，pipeline 效率的提升远大于 utilization 的损失。

**4-GPU 改善有限**：
- Task 2 baseline 已经使用 block_m=256（因为 expected_m=130 > 128），config 未变
- 仅有去 padding 的 6% shape_m 缩减带来的微小收益

**累计优化效果（vs Pre-Optimization Baseline）**：

| Config | Baseline | Task 2 | Task 5 | 累计提升 |
|--------|----------|--------|--------|----------|
| 2-GPU | 0.462ms (0.83x) | 0.476ms (0.80x) | 0.396ms (0.97x) | **+14%** |
| 4-GPU | 0.451ms (0.78x) | 0.451ms (0.78x) | 0.447ms (0.80x) | **+1%** |

---

## Task 3: 集中拷贝（Cooperative Copy）— 已验证无效

Date: 2026-06-30

### 方案

将 copy blocks 从 round-robin 分配改为协作模式：所有 copy blocks 同时处理同一个 M-block，
通过 `atomicAdd` 计数完成、global_tid 跨 block 分配线程。

### 结果

| Config | Round-robin (baseline) | Cooperative | Delta |
|--------|----------------------|-------------|-------|
| 2-GPU ncb=8 | 0.397ms (0.97x) | 0.411ms (0.93x) | **-3.5%** |
| 4-GPU ncb=8 | 0.447ms (0.80x) | 0.482ms (0.73x) | **-7.8%** |

### 分析

Cooperative copy 性能退化，根因是 **破坏了 copy/GEMM pipeline overlap**：

- **Round-robin**：8 copy blocks 同时处理不同 M-blocks（block 0 → M-block 0, block 1 → M-block 1, ...），
  多个 M-blocks 并行拷贝，GEMM blocks 完成 M-block 0 后 M-block 1 已经或即将就绪。
  
- **Cooperative**：所有 copy blocks 顺序处理 M-blocks（先 M-block 0，再 M-block 1, ...），
  单个 M-block 完成更快（8x 线程），但后续 M-blocks 未开始。GEMM blocks 完成 M-block 0 的计算后
  需等待 M-block 1 拷贝完成，造成更多 spin-wait。

M-major GEMM 调度使 GEMM blocks 快速消费 M-blocks，而 cooperative copy 的顺序处理
无法提供足够的 M-block 预取深度。Round-robin 的并行预取策略更适合此工作负载。

**结论**：保留原始 round-robin 分配，不采纳 cooperative copy。

---

## Task 6: PPU Bulk Load/Store — 验证无效果

Date: 2026-06-30

### 方案

将 copy block 的 `__ldg` 替换为 PPU 专用 bulk load/store 指令对
（`__ppu_global_load_bulk_volatile_b32x4` + `__ppu_global_store_bulk_volatile_b32x4`），
参考 DeepEP `utils.cuh` 中的 warp-level bulk copy 模式。

### 实验

1. **Per-thread bulk**（每线程独立调用 bulk 指令）：
   - ncb=8: 0.917ms（vs __ldg 0.397ms）— **-131% 退化**
   - 原因：bulk 指令是 warp-collective 操作，per-thread 调用产生巨大开销
   - ncb 增加时性能反而恶化（资源争抢）

2. **Warp-level bulk**（参考 DeepEP `warp_copy_bulk` 模式，32 int4s/warp/iter）：
   - ncb=8: 0.396ms（vs __ldg 0.397ms）— **持平**
   - 正确性 PASSED，性能无变化
   - Remainder fallback 用 `__ldg`

### 分析

Bulk load/store 对 copy 性能无提升，说明 **copy 指令本身不是瓶颈**。

当前性能分解（2-GPU, ncb=8）：
- Non-fused GEMM (39 SMs): 0.229ms
- 理论 GEMM (31 SMs): 0.229 × 39/31 = 0.288ms
- 实际 Kernel-only: 0.372ms
- Exposed copy overhead: 0.372 - 0.288 = 0.084ms

**结论**：保留 `__ldg`，不采纳 bulk load/store。Copy 指令的吞吐量不是瓶颈，
但 P2P link 本身是（见下方 P2P 隔离实验）。

---

## P2P 隔离实验（Local-to-Local）

Date: 2026-07-01

将所有 `rank_addr_a` 远程地址重定向到本地 all-gather 的数据副本，copy blocks 仍然运行
但读写全部在本地 HBM，不经过 NVLink。对比 kernel 时间差来分离 P2P link 开销。

### 2-GPU Results (GPU 6,7), M=1562, block_m=256, ncb=8

| Metric | Normal P2P | All-local | Delta |
|--------|-----------|-----------|-------|
| Kernel-only (ms) | 0.370 (std=0.007) | 0.310 (std=0.006) | **+0.060ms (+16.2%)** |
| Pipeline (ms) | 0.393 | — | — |
| Non-fused GEMM-only (ms) | — | — | 0.229 |

### 4-GPU Results (GPU 3,5,6,7), M=1547, block_m=256, ncb=8

| Metric | Normal P2P | All-local | Delta |
|--------|-----------|-----------|-------|
| Kernel-only (ms) | 0.423 (std=0.007) | 0.323 (std=0.007) | **+0.100ms (+23.7%)** |
| Pipeline (ms) | 0.449 | — | — |
| Non-fused GEMM-only (ms) | — | — | 0.248 |

### 性能分解

**结论：P2P link 是显著瓶颈**，之前的 "MC contention 为主" 推断不成立。

| 因素 | 2-GPU | 4-GPU | 说明 |
|------|-------|-------|------|
| GEMM 算力损失 (31/39 SMs) | ~0.059ms | ~0.064ms | 0.229×(39/31-1), 0.248×(39/31-1) |
| **P2P link overhead** | **0.060ms** | **0.100ms** | normal - local, NVLink 带宽瓶颈 |
| 其他 (spin-wait + MC) | ~0.022ms | ~0.011ms | local - 理论GEMM(31SMs) |

4-GPU P2P 开销更大（0.100 vs 0.060ms），因为 75% 数据需远程读取（vs 50%）。

### 优化方向更新

P2P link 是最大单一瓶颈，优化重点应放在减少 NVLink 传输量或提高利用率：

1. **TMA-based copy**：用 TMA 替代 __ldg，可能更高效利用 NVLink 带宽
2. **压缩传输**：减少需要远程传输的数据量
3. **Prefetch overlap**：将 P2P copy 与其他操作重叠（如 SFA prefetch）
4. **减少 ncb + 接受更长 copy**：释放 SM 给 GEMM，如果 P2P 带宽已饱和则无意义
