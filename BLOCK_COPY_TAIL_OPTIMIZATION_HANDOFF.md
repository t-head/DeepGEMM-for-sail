# Block-copy 变长 rank 尾部优化：方案与实验交接

## 1. 问题背景

目标代码位于：

- `deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh`
- `run_copy_block()` 的 8-rank/4-rank 尾部
- `run_copy_block_kstripe()` 的 8-rank/4-rank 尾部

当前 8-rank 快速路径先处理所有 rank 都有效的公共区间：

```cpp
for (; i < min_n; i += stride) {
    int4 v0 = ld_nc_global(s0 + addr);
    // ... v1~v7
    d0[addr] = v0;
    // ... d1~d7
}
```

随后用条件 load/store 处理 `[min_n,max_n)`：

```cpp
for (; i < max_n; i += stride) {
    int4 v0, v1, v2, v3, v4, v5, v6, v7;
    if (i < n0) v0 = ld_nc_global(s0 + addr);
    // ... n1~n7
    if (i < n0) d0[addr] = v0;
    // ... n1~n7
}
```

`hgobjdump` 显示普通 C++ `if` 被编译为多个条件 CFG。每一路条件 load 前出现
`s.wait vldcnt(0)`，使不同 rank 的 P2P load 基本串行。公共 `[0,min_n)` 路径则能先
连续发出 8 个 load，再以 `vldcnt(7)...vldcnt(0)` 逐步消费。

## 2. 真实负载为何需要优化尾部

生产示例：256 tokens/rank、topk=6、13 experts/rank、32 ranks。

```text
每源 rank 的路由项 = 256 × 6 = 1536
全局 expert 数      = 13 × 32 = 416
单源 rank→单 expert 的平均 count = 1536 / 416 ≈ 3.69
```

用 Poisson(3.69) 近似一个 8-rank 小组：

- 至少一个 count=0 的概率约 18%。
- `E[min]≈1.26`，`E[max]≈6.61`。
- 约 34% 的实际数据落在公共快速路径。
- 约 66% 的实际数据落在条件尾部。

真实 router 的 hot/cold expert 往往比理想均匀模型更偏斜，因此条件尾部可能是主要路径。

## 3. Wave-uniform 条件

K-stripe 中：

```cpp
stripe_int4s = K_TILES_PER_FLAG * BLOCK_K / 16;
n_r = rank_counts[r] * stripe_int4s;
```

当前常用配置：

```text
wave size = 32
BLOCK_K = 128
K_TILES_PER_FLAG = 4
stripe_int4s = 32
```

此时 `n_r` 总是 32 的倍数，所以 `i<n_r` 对同一 wave 是 uniform 的：整个 wave
要么执行，要么不执行。它不是整个 thread block uniform；不同 wave 对应不同 token index。

若 `K_TILES_PER_FLAG=1/2`，`stripe_int4s=8/16`，则可能出现 wave 内部分 lane active。
任何只支持 wave-uniform 的优化都应加入：

```cpp
static_assert(stripe_int4s % 32 == 0);
```

或在编译期回退到通用路径。

## 4. 方案一：PTX predicate / fake-predication

### 4.1 核心思路

在一个 inline PTX block 中为 8 个 rank 分别建立 predicate，然后连续发出 8 个
predicated vector load，再发出 8 个 predicated store：

```ptx
.reg .pred p0, p1, ..., p7;
setp.lt.u32 p0, i, n0;
// ... p1~p7

@p0 ld.global.nc.v4.u32 {a0,a1,a2,a3}, [s0];
// ... 8 loads
@p0 st.global.v4.u32 [d0], {a0,a1,a2,a3};
// ... 8 stores
```

PPU 1.x 的 VMEM 指令没有显式 predicate 字段。后端把 PTX `@p` 降低成 fake predicate：

```asm
v.cmp.gt.u32  predicate_mask, n, lane
s.lop.emsk    saved_emsk, predicate_mask # 0x8
vmem.ld.b32x4 ...
s.or.b32      emsk, emsk, saved_emsk
```

概念语义：

```text
new_emsk = old_emsk & predicate_mask
saved    = old_emsk & ~predicate_mask
执行 VMEM
old_emsk = new_emsk | saved
```

zero-emask VMEM 不产生实际访存请求。

### 4.2 已完成实验

实验源码：

- `tests/test_emask_outstanding_loads.cu`

完整 PPU 1.5 ISA：

- `tests/isa/test_emask_outstanding_loads.ppu15.isa`

远端编译运行环境：

```text
host: 30.21.206.25
container: sglang.lxh
arch: sm_89 / PPU ISA ppu1.5
```

编译与运行：

```bash
nvcc -w -std=c++17 -arch=sm_89 -lineinfo -O3 \
  test_emask_outstanding_loads.cu -o test_emask_outstanding_loads
CUDA_VISIBLE_DEVICES=4 ./test_emask_outstanding_loads
```

结果：

```text
predicated emask test: PASS (0 errors)
```

测试还执行了：

```cpp
predicated_load_store<<<1, 32>>>(
    nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0, 0);
```

该调用通过，证明全零 emask 能抑制无效地址的 load/store。

反汇编的关键结果：

```asm
s.lop.emsk
vmem.ld.b32x4
s.or.b32 emsk, emsk, saved
// 重复 8 次；8 个 load 之间没有 vldcnt wait

s.lop.emsk
s.wait vldcnt(7)
vmem.st.b32x4
// ...
s.wait vldcnt(0)
vmem.st.b32x4
```

因此已验证：

1. 修改/restoring emsk 时可以保留 outstanding VMEM load。
2. PTX `@p` 不产生 `s.cbr` 条件 CFG。
3. 8 个 load 可以连续发出。
4. store 按 load 完成顺序使用递减 wait。

### 4.3 优点

- 已通过正确性和 ISA 实验。
- 不修改 preprocess 或 rank 元数据布局。
- 稀疏和稠密 count 使用同一条路径。
- zero-emask load 不产生 P2P transaction。
- 能保留最多 8 路跨-rank memory-level parallelism。

### 4.4 风险

- 大型 inline asm 使用 8 predicates、32 个 u32 value registers 和大量位置参数，维护较 tricky。
- 8 个 `int4` 至少占 32 个 vector registers；需要检查融合 GEMM kernel 的寄存器数/occupancy。
- 依赖 PPU 后端的 fake-pred lowering 和调度形态，SDK 升级后需要 ISA regression check。
- 当前实验是独立小 kernel，尚未接入真正 fused kernel，也未测 2/4/8 卡性能。

### 4.5 建议实现方式

不要把大段 asm 直接放进 `run_copy_block_kstripe()`。封装成独立 primitive：

```cpp
__device__ __forceinline__
void predicated_copy8_int4(
    uint32_t i,
    const int4* s0, ..., const int4* s7,
    int4* d0, ..., int4* d7,
    uint32_t n0, ..., uint32_t n7);
```

增加：

- `static_assert(sizeof(int4)==16)`。
- 8-rank 和 4-rank 两个固定宽度版本。
- 编译宏/环境变量控制 old-if 与 predicated-asm A/B。
- `hgobjdump` 检查 load 区间不存在 `s.wait vldcnt(0)` 或 `s.cbr`。

## 5. 方案二：运行期 active-rank compaction + 静态 copy8/4/1

### 5.1 核心思路

先根据当前 wave 对应的 token index 统计 active ranks：

```cpp
active(r) = wave_token_index < rank_counts[r];
```

把有效 rank ID 压缩成列表或 bitmap，然后只对 active ranks 发 load。动态选择与静态 load
宽度分离：

```cpp
int pos = 0;
for (; pos + 8 <= active_count; pos += 8)
    copy_8_ranks(active_ids[pos + 0], ..., active_ids[pos + 7]);
for (; pos + 4 <= active_count; pos += 4)
    copy_4_ranks(active_ids[pos + 0], ..., active_ids[pos + 3]);
for (; pos < active_count; ++pos)
    copy_1_rank(active_ids[pos]);
```

`copy_8_ranks` 必须编译期静态展开：

```cpp
int4 v0 = ld(src[id0]);
// ... v1~v7
store(dst[id0], v0);
// ...
```

这样运行期 active count 不会迫使编译器只复用一组 value register，仍可保留 8 路 load。

### 5.2 active list 的实现选择

不建议直接使用动态索引的本地数组，可能 spill 到 local memory。可优先尝试 32-bit bitmap：

```cpp
uint32_t active_mask = ...;
int r0 = pop_first(active_mask);
// ... 一次取 8 个 set bit
copy_8_ranks(r0, ..., r7);
```

另一种长期方案是在 preprocess 中把以下 tuple 按 count 降序排列：

```text
(rank_count, rank_addr_a, rank_split_m)
```

排序后某个 wave 的 active ranks 天然构成前缀，只需求 `active_count`，无需压缩 ID。
`rank_split_m` 跟随 tuple 重排，所以目标 row 布局不变。

### 5.3 优点

- 热循环完全不发 inactive rank 的 VMEM 指令。
- copy8/4/1 可以使用普通 C++，维护性好于大型 inline asm。
- 若 active ranks 极少，可能显著减少指令发射和地址计算。
- 静态 helper 内仍有跨-rank ILP。

### 5.4 风险

- 每个 wave 的 active set 不同，必须按 `wave_token_index<count` 计算，不能只判断 `count>0`。
- compaction、`ffs/popcount`、动态 rank metadata gather 都有开销。
- `active_ids[]` 可能 spill；动态索引可能让 source/destination pointer load 变慢。
- 后端未必能识别 active-count 分支是 wave-uniform，可能再次生成 emsk/CFG。
- `copy8` 前必须先完成 8 路地址/count metadata load，避免 `sldcnt` 插入 P2P load 序列。
- 仅适用于满足 wave-uniform 的配置；KTPF=1/2 需要回退。
- 此方案目前只有设计分析，尚未写实验代码、尚无 ISA 或性能结果。

## 6. 两种方案的关系

两者是同一问题的替代实现，生产中不应在同一次 copy 上叠加：

- predicated asm：固定扫描 rank，以 emsk 抑制无效 lane。
- active compaction：先删除无效 rank，再调用无条件静态 copy helper。

开发阶段建议与旧路径三者共存做 A/B：

```text
0 = original C++ if
1 = PTX predicated asm
2 = active compaction + copy8/4/1
```

测试稳定后选择一条作为生产默认；另一实验路径可短期保留，最终删除以降低维护成本。

## 7. 推荐实验顺序

### 第一步：实现 predicated asm 到真实 kernel

1. 只替换 `[min_n,max_n)`，保留现有 `[0,min_n)` 无条件快速路径。
2. 同时覆盖普通 block-copy 与 kstripe 的 8-rank/4-rank路径。
3. 用宏或 JIT template 参数保留 old-if。
4. 先跑 FULL_CORRECTNESS，再跑性能。
5. dump 真实 fused specialization ISA，确认目标序列。

正确性命令模板：

```bash
FULL_CORRECTNESS=1 CUDA_VISIBLE_DEVICES=... \
torchrun --nproc_per_node=2 tests/test_block_copy_gemm1_multi_gpu.py --verbose
```

主要阈值：逐元素 CPU FP4 reference `calc_diff < 0.002`。

性能命令模板：

```bash
SKIP_CORRECTNESS=1 NCB_SWEEP=2 FUSED_EXACT_GRID=1 \
K_TILES_PER_FLAG=4 CUDA_VISIBLE_DEVICES=... \
torchrun --nproc_per_node=... tests/test_block_copy_gemm1_multi_gpu.py --verbose
```

重点看：

- `Kernel-only (copy+GEMM)`
- `Pipeline (full)`
- 2/4/8 卡多轮最小值
- register count / occupancy
- 真实 ISA 中 8 个 P2P load 是否连续

### 第二步：实现 active compaction 原型

1. 先只支持 `stripe_int4s % 32 == 0`。
2. 用 bitmap 而非 local array。
3. 提供静态 `copy8/4/1` helper。
4. dump ISA 确认 helper 内 8 load 连续，active-count CFG 不把 load 序列切碎。
5. 与 predicated asm 在相同 routing、相同 NCB、相同 binary config 下 A/B。

### 第三步：补真实稀疏路由

当前 performance test 使用 round-robin 均匀路由，不能充分体现 hot/cold expert。至少增加：

- 当前均匀路由。
- correctness 中的随机路由。
- 从真实 router count 采样/回放的路由。
- 人工 cold-expert 场景（大量 rank count=0）。

记录每个 8-rank group 的：

```text
min_count, max_count, active_count_per_wave,
tail_bytes / total_bytes, zero-mask rank ratio
```

## 8. 当前推荐

优先实现方案一，原因：

- 已通过独立 correctness 和 null-address zero-emask 验证。
- 已得到预期 PPU ISA：8 load 无中间 wait。
- 不需要 preprocess 改动。
- zero-emask VMEM 不产生实际 P2P transaction。

方案二作为可维护性更强的对照原型。它是否更快取决于“跳过 inactive 指令”的收益能否覆盖
active compaction 和动态 metadata gather 开销，必须以真实 fused kernel A/B 决定。

## 9. 相关文件

- 实际 fused copy：`deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh`
- preprocess：`deep_gemm/include/deep_gemm/dispatch_preprocess.cuh`
- multi-GPU test：`tests/test_block_copy_gemm1_multi_gpu.py`
- 远端 test 快照：`tests/remote_snapshots/test_block_copy_gemm1_multi_gpu.remote.py`
- 远端 test 导读：`tests/remote_snapshots/test_block_copy_gemm1_multi_gpu.remote.walkthrough.zh-CN.md`
- emask 实验：`tests/test_emask_outstanding_loads.cu`
- emask 完整 ISA：`tests/isa/test_emask_outstanding_loads.ppu15.isa`
- 历史背景与性能记录：`ITERATIONS_block_copy_p2p.md`
