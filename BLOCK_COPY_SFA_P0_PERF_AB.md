# Block-copy GPU-side SFA P0 性能 A/B 记录

## 结论

2026-07-06 在同一台 2卡 ZW-M890P 节点上对比：

| 版本 | SFA 路径 | 最佳 NCB | Fused full pipeline | Non-fused | Fused / non-fused |
|---|---|---:|---:|---:|---:|
| `c1ff49b` | 计时前 host `all_gather + build_merged_sfa` | 4 | 0.306 ms | 0.266 ms | 0.87x |
| `b263633` | GPU-side SFA copy/repack，成本包含在 fused pipeline | 12 | 0.510 ms | 0.267 ms | 0.52x |
| `b263633` + SFA copy 优化 | 同样的 GPU-side 功能口径 | 8 | 0.312 ms | 0.267 ms | 0.85x |
| `b263633` + 禁用 SFA copy（诊断用） | 同一 kernel 的性能下界 | 4 | 0.311 ms | 0.266 ms | 0.86x |

`b263633` 的最佳 fused pipeline 相比 `c1ff49b` 增加 0.204 ms，约慢 67%。Non-fused 基线基本不变（0.266/0.267 ms），说明差异主要位于 block-copy fused 路径。

专项诊断确认回退几乎全部来自 SFA copy：在同一 `b263633` kernel 中临时禁用 SFA copy 后为 0.311 ms。将 copy 改为 CTA 级展平并在对齐时使用 16-byte 向量交易后，完整功能路径为 0.312 ms，与该下界只差 0.001 ms。相比未优化 P0 的 0.510 ms，最佳耗时降低 0.198 ms（约 38.8%）。

但两者不是完全等价的端到端对比：`c1ff49b` 在计时前完成 host SFA 预拼接，0.306 ms 没有包含动态 routing 所需的 SFA 更新/搬运/重排；`b263633` 才是完整功能口径。因此 pre-P0 数字可作为「不含 SFA 生产成本」的性能上界，不应直接作为可回退的生产实现。

## 测试环境

- 节点：`30.21.206.25`
- 容器：`sglang.lxh`
- 设备：ZW-M890P，`CUDA_VISIBLE_DEVICES=0,1`
- 配置：prod，2 ranks，12 experts/rank，`M=1536, N=6144, K=7168`
- 共同参数：`PERF_VERBOSE=1 SKIP_ISOLATION=1`
- fused grouping：`nopad`
- 计时：NCB sweep，每个 NCB 取 20 iterations 中位数
- 卡 0/1 测试前均为 Default 模式且无其他计算进程

`b263633` 命令：

```bash
FULL_CORRECTNESS=1 PERF_VERBOSE=1 SKIP_ISOLATION=1 \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29641 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose
```

`c1ff49b` 使用 detached worktree 执行：

```bash
SKIP_CORRECTNESS=1 PERF_VERBOSE=1 SKIP_ISOLATION=1 \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29644 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose
```

## 详细数据

### `c1ff49b` （pre-P0）

| NCB | Fused pipeline | 相对 non-fused |
|---:|---:|---:|
| 4 | 0.306 ms | 0.87x |
| 8 | 0.309 ms | 0.86x |
| 12 | 0.316 ms | 0.84x |

其它拆解：

- quantize + symmetric-buffer zero：0.049 ms
- expert preprocess：0.030 ms
- DeepEP dispatch：0.065 ms
- non-fused full pipeline：0.266 ms

### `b263633` （GPU-side SFA P0）

| NCB | Fused pipeline | 相对 non-fused |
|---:|---:|---:|
| 4 | 0.959 ms | 0.28x |
| 8 | 0.717 ms | 0.37x |
| 12 | 0.510 ms | 0.52x |

其它拆解：

- quantize + symmetric-buffer zero：0.025 ms
- expert preprocess：0.054 ms
- DeepEP dispatch：0.067 ms
- non-fused full pipeline：0.267 ms

### GPU-side SFA copy 优化后（最终生产路径）

2 卡 `FULL_CORRECTNESS=1`：

| NCB | Fused pipeline | 相对 non-fused |
|---:|---:|---:|
| 4 | 0.324 ms | 0.82x |
| 8 | 0.312 ms | 0.85x |
| 12 | 0.323 ms | 0.83x |

- correctness：所有 ranks 和 experts 通过，expert diff 为 0
- quantize + symmetric-buffer zero：0.046 ms
- expert preprocess：0.031 ms
- non-fused full pipeline：0.267 ms

4 卡 `FULL_CORRECTNESS=1`（GPU 0/1/5/6）：

| NCB | Fused pipeline | 相对 non-fused |
|---:|---:|---:|
| 4 | 0.381 ms | 0.70x |
| 8 | 0.336 ms | 0.79x |
| 12 | 0.350 ms | 0.76x |

- correctness：所有 ranks 通过
- non-fused full pipeline：0.267 ms

### SFA-off 定位实验（非生产配置）

| NCB | Fused pipeline |
|---:|---:|
| 4 | 0.311 ms |
| 8 | 0.317 ms |
| 12 | 0.322 ms |

该实验只用于隔离 SFA copy 成本，临时开关未保留在生产代码中。

## 根因与优化

1. 旧实现的循环顺序是 `k-scale block` 在外、token 在内。生产 case 中 `cnt=64`，256-thread copy CTA 每轮只有 64 个线程工作，每个活跃线程又串行处理 112 个 16-bit scale。
2. 优化后把 `k_scale_blocks × token count` 展平到整个 CTA，使全部 copy threads 参与。当源和目标行起点均为 16-byte 对齐时，每次用 `int4` 搬运 8 个 `uint16_t` scale。
3. 任意 routing 可能导致 rank 边界不对齐，因此保留 scalar fallback；token count 非 8 的倍数时由显式 scalar tail 处理，不改变功能语义。
4. 两个 commit 的 GEMM 配置均为 fused `block_m=256`，non-fused `block_m=128`，该已知 config 分叉在两侧都存在，不是本次 P0 A/B 差异的来源。

## 验证结论

- SFA-off A/B 和完整路径只差 0.001 ms，原 P0 回退已解决。
- 2 卡、4 卡 `FULL_CORRECTNESS=1` 均通过。
- 生产代码中不保留 SFA-off 开关或 profiling 分支。
- 后续可继续 generation/parity/context 重构，不需要回退到 host `build_merged_sfa`。

## 原始日志

- `b263633`：远程容器 `/tmp/deepgemm_b263633_baseline_2gpu.log`
- `c1ff49b`：远程容器 `/tmp/deepgemm_c1ff49b_perf_2gpu.log`
- SFA-off 定位：远程容器 `/tmp/deepgemm_b263633_sfa_off_perf_2gpu.log`
- 优化后 2 卡最终生产路径：远程容器 `/tmp/deepgemm_sfa_vectorized_final_2gpu.log`
- 优化后 4 卡 correctness + performance：远程容器 `/tmp/deepgemm_sfa_vectorized_correctness_4gpu.log`
