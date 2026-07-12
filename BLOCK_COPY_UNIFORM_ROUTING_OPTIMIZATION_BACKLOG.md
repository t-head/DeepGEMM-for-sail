# Block-Copy Remote-Pull：均匀 Routing 待优化清单

## 1. 目标与范围

本文整理 block-copy fused dispatch GEMM1 在**均匀 routing** 下仍值得推进的优化项，目标是降低完整 pipeline 和 `copy + GEMM` kernel 的 wall-clock，而不是只优化局部指标。

当前生产形状参考：

```text
tokens/rank = 256
topk        = 6
experts/rank= 12
N           = 6144
K           = 7168
```

性能测试使用：

```python
(token_idx * topk + j) % num_total_experts
```

因此每个源 rank 到每个 expert 的 token 数完全相等：

| 卡数 | total experts | count / (src rank, expert) | owner expert 总 M |
|---:|---:|---:|---:|
| 2 | 24 | 64 | 128 |
| 4 | 48 | 32 | 128 |
| 8 | 96 | 16 | 128 |

这意味着 `run_copy_block()` 和 `run_copy_block_kstripe()` 中的 `min_n == max_n`，变长 rank 条件尾部不会进入性能关键路径。EMASK/predicated tail 优化保留为后续非均匀 routing 项，不作为本文首要目标。

## 2. 当前性能认识

- Copy CTA 使用静态 round-robin M-block 分配是当前正确基线；cooperative copy 已被 2/4/8 卡 A/B 证伪。
- P2P 暴露主要集中在 wave 0；后续 wave 基本可被 GEMM 掩盖。
- `__threadfence()` 在 PPU 上会降低为整 SM cache write-back/invalidate，次数比单线程发射方式更重要。
- 8-rank 公共路径已能连续发出 8 路远端 load；2/4-rank 路径仍有提高 memory-level parallelism 的空间。
- GPU-side SFA copy 展平和 16-byte 向量化后已接近 SFA-off 下界，不宜继续做脱离关键路径的局部微调。

## 3. 优先级清单

### P0：移除 NoPad 的 `expected_m <= 128 -> 129` legacy clamp

#### 现状

均匀 routing 下每个 expert 的实际 M 正好为 128。NoPad 路径仍把 tuning expected M 提升到 129，导致配置选择偏向 `BLOCK_M=256`，每个 expert 实际只有 128 行却使用 256 高 tile。

涉及位置：

- `deep_gemm/jit_kernels/dispatch_fused_gemm.py`
  - `fused_dispatch_block_copy_gemm1_fp4()`
  - `BlockCopyDispatchContext._select_configs()`
- `tests/test_block_copy_gemm1_multi_gpu.py`
  - `get_gemm_configs()`

#### 已有证据

历史公平 A/B 中，NoPad 强制 `BLOCK_M=128` 与 Masked `BLOCK_M=128` 基本一致：

| 卡数 | NoPad BM256 pipeline | NoPad BM128 pipeline | 收益 |
|---:|---:|---:|---:|
| 2 | 约 0.302 ms | 约 0.267 ms | 约 35 us |
| 8 | 约 0.309 ms | 约 0.273 ms | 约 36 us |

#### 实施建议

1. 删除三处 legacy clamp，保证 preprocess 与 GEMM 使用同一配置选择逻辑。
2. 保留 `FORCE_EXPECTED_M` 仅用于诊断，不作为生产修复。
3. 覆盖 expert M 为 127、128、129，以及第二个 M-block 的 correctness。

#### 验收

- 均匀 prod 配置自动选择 `BLOCK_M=128`。
- 2/4/8 卡 FULL_CORRECTNESS 通过。
- pipeline 和 kernel-only 不劣于 Masked BM128 对照，目标收益大于 20 us。

---

### P0：将 exact-grid、NCB、KTPF 从实验开关收敛为配置策略

#### 现状

- `BlockCopyDispatchContext.run()` 默认 `num_copy_blocks=1`、`k_tiles_per_flag=0`。
- `FUSED_EXACT_GRID` 默认关闭，生产逻辑仍依赖环境变量。
- 主性能 sweep 默认 NCB 为 `4,8,12`，没有覆盖历史上 exact-grid 的稳定点 NCB=3。

#### 已有证据

均匀 routing 历史 sweep 中：

- exact-grid 通常优于超发 grid，尤其能消除小 NCB 的启动 bubble。
- exact-grid + round-robin + NCB=3 是稳定候选。
- coarse K-stripe（KTPF=7/14）有约 5–8 us 的潜在收益，但幅度接近噪声，需要与当前代码重新 A/B。
- cooperative `copy_mode=1/2` 是明确负优化，生产保持 `copy_mode=0`。

#### 实施建议

1. 先建立 shape-keyed 配置表，避免未经验证的全局默认：

   ```text
   (num_ranks, M, N, K, groups) -> (exact_grid, NCB, KTPF)
   ```

2. prod 均匀形状优先复测：

   ```text
   exact=1, NCB in {2,3,4}, KTPF in {0,7,14}
   ```

3. 选择以 pipeline wall-clock 为主，kernel-only、wave0 wait 和 copy-only BW 为解释指标。

#### 验收

- 默认公开 API 不再落到未经调优的 NCB=1。
- 2/4/8 卡至少三轮，报告 median 与 min-of-clean-runs。
- 新默认在每个卡数上不比当前最佳 sweep 慢超过 2%。

---

### P1：增加 2-rank / 4-rank 固定宽度 copy primitive

#### 现状

FP4 copy 公共区间中：

- 8-rank 路径每轮连续发出 8 个远端 `int4` load。
- 4-rank 路径每轮只有 4 个跨-rank load。
- 2-rank 会进入逐 rank 路径，主要依靠单 rank 内四次展开，跨-rank MLP 不足。

均匀 routing 不进入条件 tail，因此这是直接命中当前性能路径的 remote-pull 优化。

#### 实施建议

1. 封装编译期固定宽度的 `copy2`、`copy4`、`copy8` primitive。
2. 4-rank 路径可展开两个 `i`，争取连续发出 8 个 load 后再 store。
3. 2-rank 路径交错两个 rank 和多个 `i`，避免完整复制 rank 0 后才处理 rank 1。
4. 同时覆盖 whole-M-block 与 K-stripe 两条路径。
5. dump 真实 fused specialization ISA，确认 load 区间没有过早的 `s.wait vldcnt(0)`。

#### 风险

- 更多 `int4` 临时值会增加 vector register 压力。
- fused kernel 的寄存器上限由 GEMM 与 copy 分支共同决定，需要检查 occupancy 和编译资源。
- 若链路已经饱和，copy-only 可能改善但完整 kernel 不一定改善。

#### 验收

- 2/4 卡 copy-only BW 或 wave0 ready time 有可重复改善。
- 8 卡路径零回退。
- 真实 ISA 保持目标数量的 outstanding VMEM load。
- 以完整 pipeline 至少 3–5 us 收益作为保留门槛。

---

### P1：降低每个 M-block 的 device fence 成本

#### 现状

每个 M-block 完成 FP4 与 SFA copy 后执行：

```cpp
__syncthreads();
__threadfence();
copy_ready_flags[mb] = 1;
```

PPU 上 device fence 会触发全 SM cache write-back/invalidate。当前 round-robin 的 fence 次数约等于 M-block 数；K-stripe 则约等于 `M-block × stripe` 数。

#### 方向 A：后续 M-block 批量发布

首批 M-block 继续逐块 fence/flag，保证 wave0 尽早启动。后续已能被 GEMM 掩盖的 M-block，可由每个 copy CTA 连续复制两个，再执行一次 fence 并发布两个 flag。

示意：

```text
first owned M-block: copy -> fence -> flag
later pair:          copy + copy -> fence -> flag + flag
```

这保持每个 M-block 由单 CTA 所有，不引入 cooperative straggler。

#### 方向 B：write-through/cache-bypass store

探索 PPU 是否支持目标 HBM 的 write-through/cache-bypass vector store。若 copy 数据不滞留在本 SM cache，可尝试使用仅排序/等待的 release 代替全 cache `wbinv`。

该方向必须先做独立 producer/consumer 跨 SM 可见性验证，不能只凭单 CTA correctness 推断。

#### 验收

- FULL_CORRECTNESS 包含跨 CTA 消费与连续 generation。
- wave0 ready time不回退。
- fence/wbinv 动态次数或 ISA 明确下降。
- pipeline 收益超过测量噪声；若只改善局部 fence 时间但 wall 不变，则不合入。

---

### P1：用 generation-stamped ready flag 去掉每轮 `zero_()`

#### 现状

每次 fused GEMM 前都会调用：

```python
copy_ready_flags.zero_()
```

buffer 只有约百字节，成本主要来自额外的异步 memset/launch，而不是带宽。

#### 实施建议

- KTPF=0：producer 写 `ready_generation[mb] = generation`，consumer 等待本轮 generation。
- KTPF>0：使用 `(generation, stripe)` 编码，或拆为 epoch 与 stripe 两个字段。
- 默认 round-robin 不需要 cooperative done-counter；实验模式可保留单独清零逻辑。
- generation wraparound 需要明确比较语义，优先考虑 64-bit epoch 或不依赖大小关系的精确相等。

#### 验收

- profile 中不再出现每轮 flag memset。
- 连续 generation 1/2/3、parity 1/0/1 correctness 通过。
- KTPF=0 与 KTPF>0 都不会误接受旧 stripe。
- 完整 pipeline 获得稳定的数微秒收益。

---

### P2：K-stripe 下只预拷当前 stripe 所需的 SFA

#### 现状

`run_copy_block_kstripe()` 在复制第一条 FP4 stripe 前先复制整个 M-block 的 SFA，然后发布第一条 stripe flag。GEMM mainloop 实际按 K tile 消费 SFA，因此第一条 flag 理论上只要求对应 K 范围的 SFA 已就绪。

#### 实施建议

- 将 `copy_mblock_sfa()` 扩展为 K-scale row range 版本。
- 每条 FP4 stripe 同时复制对应 SFA rows，再复用该 stripe 的 fence/flag。
- KTPF=0 保持当前一次性 SFA copy。

#### 判断

SFA 向量化后完整路径与 SFA-off 下界只差约 1 us，因此该项主要价值是缩短第一条 stripe 的就绪延迟，优先级低于 BM、grid、MLP 与 fence。

---

### P2：K-stripe metadata 与地址计算整理

当前每条 stripe 会重复读取同一 M-block 的 rank address、split 和 count，并执行 pitched row 的除法/取模映射。

可实验：

- 将只读 metadata 提到 stripe 循环外，但必须观察寄存器 live range。
- 对常用 `stripe_int4s` 提供 power-of-two 快速映射。
- 使用 row/column 嵌套遍历替换 flatten 后的重复 div/mod。

该项仅在 ISA 显示 scalar/address pipeline 成为瓶颈时推进，不应仅凭源码复杂度判断收益。

## 4. 已证伪或暂不优先的方向

### Cooperative copy

所有 NCB 合搬同一个 M-block 会带来：

- `M-block × NCB` 次 device fence；
- 最慢 CTA straggler；
- 多卡下显著回退。

保留实验代码即可，生产继续使用 `copy_mode=0`。

### 均匀性能测试上的 EMASK tail 优化

当前固定 routing 下 `min_n == max_n`，条件 tail 为零。EMASK primitive 已有独立 ISA/correctness 依据，但必须在随机、hot/cold 或真实 router replay 下评价，不能用当前均匀 perf 结果决定取舍。

### 继续单独优化 SFA 总吞吐

现有 CTA 展平与 16-byte 向量化已经消除了主要回退。除 K-stripe 首发顺序外，SFA 不是当前最优先瓶颈。

## 5. 推荐实施顺序

```text
Step 1  删除 NoPad clamp，固定 BM128 correctness/perf
Step 2  复测 exact-grid × NCB × coarse KTPF，并形成默认配置表
Step 3  实现 copy2/copy4 MLP A/B，检查真实 ISA 与寄存器
Step 4  generation-stamped flag，移除每轮 zero_
Step 5  后续 M-block fence batching
Step 6  独立验证 write-through/cache-bypass 的跨 SM 发布协议
Step 7  视结果推进 stripe-SFA 与地址计算微调
```

前两步已有历史数据支撑，预计收益最大且风险最低；第三至第六步才是 remote-pull kernel 的主要新增工程空间。

## 6. 统一测试矩阵

### Correctness

- 2/4/8 卡 FULL_CORRECTNESS。
- generation 1/2/3，覆盖 parity 1/0/1。
- NoPad 与 Masked。
- KTPF 0/7/14。
- expert M 边界 127/128/129，以及超过一个 M-block 的场景。

### Performance

- 固定均匀 routing，避免与 tail 优化混淆。
- exact-grid `{0,1}`。
- NCB `{2,3,4,8}`。
- KTPF `{0,7,14}`。
- 至少 3 轮，GPU 独占；报告 median 和 clean-run min。

### 必看指标

1. Full pipeline wall-clock。
2. Kernel-only `copy + GEMM`。
3. Local-only GEMM 下界。
4. Wave0 wait 与 wave1+ wait。
5. Copy-only BW。
6. 寄存器数、occupancy、真实 ISA outstanding-load/fence 序列。

局部指标改善但完整 pipeline 无收益的方案不进入生产默认。

## 7. 相关文件

- `deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh`
- `deep_gemm/include/deep_gemm/scheduler_cutlass3.cuh`
- `deep_gemm/jit_kernels/dispatch_fused_gemm.py`
- `deep_gemm/include/deep_gemm/expert_preprocess.cuh`
- `tests/test_block_copy_gemm1_multi_gpu.py`
- `BLOCK_COPY_SFA_P0_PERF_AB.md`
- `ITERATIONS_block_copy_p2p.md`
- `BLOCK_COPY_TAIL_OPTIMIZATION_HANDOFF.md`

