# Block-Copy Generation、Parity 与 SFA Review 总结

## 背景与目标

当前 block-copy fused dispatch GEMM 使用 symmetric buffer 在多个 rank 之间传递量化后的 FP4 激活和 SFA，并通过 generation 实现跨 rank arrival 同步和双缓冲。

本次 review 的目标是确认：

- generation/arrival 协议是否能保证每轮读取正确的数据；
- parity 0/1 双缓冲是否被功能测试完整覆盖；
- SFA 是否走了真实的生产数据路径；
- 性能测试统计的范围是否等同于完整 fused pipeline。

相关实现主要位于：

- `deep_gemm/include/deep_gemm/dispatch_layout.cuh`
- `deep_gemm/include/deep_gemm/mxfp4_quant.cuh`
- `deep_gemm/include/deep_gemm/dispatch_preprocess.cuh`
- `deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh`
- `deep_gemm/include/deep_gemm/scheduler_cutlass3.cuh`
- `deep_gemm/jit_kernels/dispatch_fused_gemm.py`
- `tests/test_block_copy_gemm1_multi_gpu.py`
- `tests/remote_snapshots/test_block_copy_gemm1_multi_gpu.remote.py`

## 1. Generation 协议与 API

generation 当前承担两个不同但都必要的职责：

```text
parity = generation & 1       选择本轮使用的 symmetric data buffer
arrival_slot = generation     标识 producer 已经完成的具体迭代
```

generation 不能退化成内部 0/1 翻转。假设 producer 在第 1 轮留下 `flag=1`，较快的 consumer 在 producer 尚未完成第 3 轮时进入第 3 轮；如果只比较 parity，它会把第 1 轮的 `1` 误认为第 3 轮已经完成。这是典型的 ABA 问题。

完整 generation 可以避免该问题：

```text
slot = 1, expected = 3  =>  继续等待
slot = 3, expected = 3  =>  本轮 producer 已完成
```

### API 建议

- 生产高层 API 不应要求用户手工维护 generation。
- generation 应由绑定 communication group、symmetric buffer 和 workspace 的 context 管理。
- quantize 与 preprocess 必须共享同一轮的 generation。
- 如果两个操作必须保持为独立异步 API，可由 context 返回 opaque iteration ticket，而不是让用户填写整数。
- 低层 kernel API 可以保留显式 generation，供单元测试、调试和协议验证使用。
- 不应使用进程级全局计数器；多个 group、buffer、stream 或并发 pipeline 会互相干扰。

## 2. `generation=0` 的真实语义

`generation=0` 不是合法的多 rank legacy correctness path。

当前代码在 generation 为 0 时会跳过：

- producer 的 arrival push；
- arrival push 前的 `__threadfence_system()` 发布；
- preprocess 对所有 producer arrival slot 的等待。

因此，在没有额外 `torch.cuda.synchronize()` 和 `dist.barrier()` 的情况下，preprocess 可能读取尚未完成或尚未全局可见的数据。

更准确的定义是：

```text
generation > 0：多 rank 生产功能路径
generation = 0：数据已预先准备并显式同步后的 benchmark/debug bypass
```

建议：

- 生产 API 要求 generation 从 1 开始，并由内部 context 单调递增。
- 取消 production API 中 `generation=0` 的默认行为。
- preprocess 性能拆解若需要排除 arrival barrier，应使用明确命名的内部选项，例如 `wait_for_arrival=False`，不要复用特殊 generation 值表达该语义。
- generation 0 仅用于已经显式完成跨 rank 同步的性能隔离测试。

## 3. Parity 寻址问题

每个 rank 的 symmetric buffer 数据区为：

```text
base
├── parity 0: [metadata | FP4 | SFA]
├── parity 1: [metadata | FP4 | SFA]
└── arrival slots
```

本轮所有数据必须统一基于：

```text
data_base = base + (generation & 1) * data_region_size
```

量化 kernel 和 preprocess 已通过 `parity_base()` 选择本轮 data buffer，但测试辅助代码中的以下逻辑固定从 symmetric buffer 原始基址读取：

- `get_expert_token_counts()`；
- `build_merged_sfa()`。

因此它们只能正确读取 parity 0。

### `BC_GEN=2` 的含义

`BC_GEN=2` 只是针对上述缺陷的 workaround：

```text
2 > 0       => 启用 arrival barrier
2 & 1 == 0  => metadata、FP4、SFA 仍写入 parity 0
```

它虽然进入了 generation 大于 0 的同步路径，却没有覆盖 parity 1，也没有覆盖 parity buffer 的再次复用。

需要让所有测试 helper 接收 generation 或 parity，并使 metadata、FP4、SFA 的地址全部从同一个 `data_base` 计算。修复后不应再需要 `BC_GEN=2` 配置。

## 4. 功能测试覆盖不足

现有测试在一次运行中固定 `x` 和 `topk_ids`，并复用提前构建的 merged SFA。即使读取了旧 parity 的 SFA，因为内容恰好相同，结果也可能继续通过。

主功能测试应在同一个 symmetric buffer 上至少连续执行三轮：

```text
generation=1: input A + topk A  => parity 1
generation=2: input B + topk B  => parity 0
generation=3: input C + topk C  => parity 1
```

三轮必须使用不同的输入和 routing，并分别与独立 reference 比较。这样才能覆盖：

- parity 0 和 parity 1；
- parity 1 的再次复用；
- 动态 expert token count；
- 动态 SFA 内容及行布局；
- 第 3 轮错误读取第 1 轮数据的问题。

还应增加 rank-skew 测试：人为延迟一个 producer rank，确认其他 rank 的 preprocess 会等待当前 generation，而不是接受旧 arrival slot。

不应通过在每轮 quantize 与 preprocess 之间加入 `dist.barrier()` 规避该测试；arrival generation 协议本身就是要替代更重的 host/NCCL barrier。

## 5. `build_merged_sfa` 暴露的生产数据路径缺口

量化 kernel 已经将 SFA 写入每个 producer rank 的 symmetric buffer。理论上，owner rank 上运行的 fused GEMM 可以通过 P2P 直接读取这些 SFA；`build_merged_sfa` 不是算法本身要求的步骤。

它目前存在，是因为 GEMM 数据接口与多 rank SFA 的实际布局不匹配。

对于同一个 expert，各 producer rank 分别持有：

```text
rank 0 SFA: [Kscale, max_tokens]，有效 M 为 c0
rank 1 SFA: [Kscale, max_tokens]，有效 M 为 c1
rank 2 SFA: [Kscale, max_tokens]，有效 M 为 c2
```

当前 GEMM mainloop 期望每个 expert 只有一个基址和固定 stride：

```text
merged SFA: [Kscale, c0 + c1 + c2]
             rank0 | rank1 | rank2
```

当前代码状态是：

- preprocess 已经生成每个 M-block、每个 rank 的 `rank_addr_sfa`；
- block-copy kernel 实际只使用 `rank_addr_a` 搬运 FP4；
- `rank_addr_sfa` 没有被 copy block 消费；
- GEMM mainloop 使用每个 expert 一个地址的 `remote_addr_sfa`，测试实际向它传入 `merged_sfa_addrs`；
- 测试通过 `all_gather + build_merged_sfa` 在 GEMM 前预先拼接 SFA。

因此，`build_merged_sfa` 本质上是在测试侧补齐尚未实现的 SFA copy/repack 路径。

### 推荐实现（P0，最高优先级）

GPU 侧 SFA 数据路径应先于 generation/parity 测试改造完成。只要测试仍复用计时前构建的固定 merged SFA，即使 metadata 和 FP4 正确切换了 parity，也不能证明 GEMM 使用的是同一 generation 的完整输入。动态 parity、动态 topk 和动态输入的可靠验证都依赖本轮 SFA 能随本轮 FP4 一起更新。

优先让 dedicated copy block 同时搬运 FP4 和 SFA：

```text
remote FP4  -> local_fp4_buf
remote SFA  -> local_sfa_buf
两者完成    -> 设置 copy-ready flag
```

SFA 是 column-major，必须根据 `rank_counts`、`rank_split_m`、源端 `max_tokens` stride 和目标 expert M stride进行重排，不能简单做一次整块 memcpy。

另一个方案是修改 GEMM mainloop，使其按 M-block 使用 `rank_addr_sfa` 直接读取远端分段 SFA。但该方案需要处理一个 M tile 跨多个 rank、源端不同 stride、指针选择以及 P2P 访存开销，复杂度更高。

生产路径不应依赖 Python 测试侧的 `all_gather + build_merged_sfa`。

## 6. 性能测试口径问题

主性能循环使用递增 generation，因此会交替覆盖 parity 并启用 arrival barrier。例如：

```text
9401 -> parity 1
9402 -> parity 0
9403 -> parity 1
```

但是当前性能测试：

- 固定 `x` 和 `topk_ids`；
- 在计时前构建一次 merged SFA；
- 每轮复用该 merged SFA；
- 不计入 `all_gather + build_merged_sfa`。

所以当前所谓 full pipeline 实际只统计：

```text
quantize -> arrival/preprocess -> FP4 block-copy -> GEMM
```

它没有统计动态输入和动态 routing 下必需的 SFA 更新、搬运或重排成本。

建议拆分为两个清楚命名的测试：

1. 固定 routing 的 kernel microbenchmark，用于稳定比较 kernel 和调度性能。
2. 每轮更新 SFA 且包含 SFA copy/repack 的端到端 benchmark。

如果 copy block 实现了 SFA 搬运，该成本应自然包含在 fused pipeline event 计时范围内，并删除计时前的测试侧 SFA 预拼接。

## 7. 推荐实施顺序

1. **P0：使用已有 `rank_addr_sfa` 实现 GPU 侧 SFA copy/repack。** FP4 和 SFA 必须属于同一 generation，并在两者均完成后发布 copy-ready flag。
2. **P0：让 fused GEMM 消费本轮 GPU 侧生成的 local SFA。** 删除正式功能和性能路径对固定 `merged_sfa_addrs` 的依赖。
3. **P0：删除 `all_gather + build_merged_sfa` 生产路径。** `build_merged_sfa` 最多暂时保留为 reference/debug helper，之后可删除。
4. 修复仍需保留的 reference helper 的 parity 寻址，确保调试结果能读取指定 generation。
5. 删除依赖 `BC_GEN=2` 才能覆盖标准路径的测试配置。
6. 增加 `generation=1,2,3` 且每轮改变输入和 topk 的功能测试，验证 metadata、FP4、SFA 同步切换 parity。
7. 增加 rank-skew/ABA 功能测试。
8. 让高层 context/workspace 内部管理 generation；低层保留显式 generation。
9. 将 `generation=0` 改为明确的测试内部 barrier-bypass 机制。
10. 拆分 kernel microbenchmark 与真实端到端 benchmark。

## 8. 验收标准

- 多 rank production path 始终使用大于 0、单调递增的 generation。
- 用户无需手工选择 generation 数值或 parity。
- 同一 symmetric buffer 连续执行 generation 1、2、3，动态输入和动态 topk 均正确。
- metadata、FP4 和 SFA 始终来自同一 generation 对应的 parity。
- rank 发生执行偏斜时不会读取旧 generation，也不会死锁。
- fused GEMM 使用本轮 routing 对应的 SFA，而不是计时前缓存的固定 SFA。
- 生产路径不依赖测试侧 `build_merged_sfa`。
- 端到端性能统计包含 SFA 搬运或重排成本。
- preprocess barrier-bypass 只存在于明确的性能隔离测试中，不作为 correctness API。
