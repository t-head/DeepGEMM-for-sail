# Fused Dispatch Block-Copy GEMM1 代码逐行解析

本文档详细解析 Block-Copy Fused Dispatch GEMM1 的完整实现：从 Python API 入口到 CUDA kernel 的每个阶段。

## 目录

1. [整体架构](#1-整体架构)
2. [Python API 层](#2-python-api-层dispatch_fused_gemmpygemm_fp4py)
3. [Preprocess 阶段](#3-preprocess-阶段dispatch_preprocesscuh)
4. [GEMM Kernel 主体](#4-gemm-kernel-主体fp4_gemm_cutlass3cuh)
5. [Tile Scheduler](#5-tile-schedulerscheduler_cutlass3cuh)
6. [数据流总览](#6-数据流总览)
7. [优化点总结](#7-优化点总结)

---

## 1. 整体架构

Block-Copy Fused Dispatch GEMM1 是 DeepSeek MoE 推理中的 GEMM1（up-projection）实现。它将多 GPU 间的数据分发（dispatch）与 FP4 GEMM 计算融合在单个 kernel 中，通过 **专用 copy blocks** 做 P2P 数据搬运，**GEMM blocks** 做矩阵乘法。

### 执行流水线

```
Python API
    │
    ├── Phase 1: MXFP4 Quantization (mxfp4_quantize_to_sym_buffer)
    │   BF16 activations → FP4 packed into symmetric buffer
    │
    ├── Phase 2: Expert Preprocess (dispatch_expert_preprocess)
    │   读取路由信息 → 生成 grouped_layout + per-rank split metadata
    │
    └── Phase 3: Fused GEMM Kernel (fused_dispatch_block_copy_gemm1_fp4)
        ├── Copy Blocks [0..ncb-1]: P2P remote → local HBM
        └── GEMM Blocks [ncb..num_sms-1]: FP4 GEMM from local HBM
```

### 关键维度

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 6144 | FFN intermediate hidden dim |
| K | 7168 | model hidden dim |
| M | ~1500-1700 | 所有 expert 的 token 总数（动态） |
| num_local_experts | 13 | 每个 rank 的 expert 数 |
| max_tokens_per_expert | 256 | 每 expert 最大 token 数 |
| BLOCK_M | 128 或 256 | M-tile 大小（config 自动选择） |
| BLOCK_N | 256 | N-tile 大小 |
| BLOCK_K | 64 或 128 | K-tile 大小（取决于 BLOCK_M） |

---

## 2. Python API 层（dispatch_fused_gemm.py, gemm_fp4.py）

### 2.1 fused_dispatch_block_copy_gemm1_fp4()

**文件**: `deep_gemm/jit_kernels/dispatch_fused_gemm.py:397-490`

这是 Python 侧的主入口函数。

```python
def fused_dispatch_block_copy_gemm1_fp4(
    rhs_: Tuple[torch.Tensor, torch.Tensor],  # (B_weight, B_scales)
    out: torch.Tensor,                          # 输出
    grouped_layout: torch.Tensor,               # preprocess 产出的 M-block 调度表
    rank_addr_a, rank_addr_sfa,                 # per-(M-block × rank) 的远程数据地址
    rank_split_m, rank_counts,                  # per-(M-block × rank) 的 SMEM 偏移和 token 数
    shape_m: int,                               # 所有 expert 的 token 总和
    ...
    num_copy_blocks: int = 1,                   # 分配给 copy 的 block 数
)
```

#### Config 选择（L426-438）

```python
expected_m = ceil_div(shape_m, num_groups)  # 每 expert 平均 token 数
if expected_m <= 128:
    expected_m = 129  # 强制 ≥129，确保 config 选择 block_m=256
```

**为什么要 clamp expected_m？** `get_best_configs` 的 voting system 在 `expected_m ≤ 128` 时，
`ceil_div(expected_m, 128) = ceil_div(expected_m, 256) = 1`，两种 block_m 产生相同数量的
M-tiles，但 block_m=128 的 m_util 更高（93% vs 47%），导致 voting 倾向 block_m=128。
然而 block_m=256 搭配 block_k=128（K 迭代减半），pipeline 效率远高于 block_m=128 + block_k=64。
Clamp 到 129 使 `ceil_div(129, 128) = 2 > ceil_div(129, 256) = 1`，wave count 正确偏好 block_m=256。

#### Config 选择器：get_best_configs()

**文件**: `deep_gemm/jit_kernels/gemm_fp4.py:283-439`

对 `GemmType.GroupedNoPad`（FusedDispatch 也走此路径），config 选择使用 4-metric voting system：

```python
block_ms = (256, 128, 64, 32, 16)  # 从大到小遍历
block_ns = (256, 128, 64, 32)

for block_m in block_ms:
    for block_n in block_ns:
        if m < 512:  # 小 M 场景：3 选 2 voting
            valid_wave = (occ_wave / best_occ_wave) <= 1     # wave 数不多于当前最优
            valid_util = (num_utils / best_num_utils) >= 1   # tile 利用率不低于当前最优
            valid_ai   = (ai_util / best_ai_util) >= 1       # 计算强度不低于当前最优
            success = (valid_wave + valid_util + valid_ai) >= 2  # 3 选 2
```

Block_K 的确定（L428-433）：

```python
block_k = 64  # 默认
if best_block_m == 256 and best_block_n == 256:
    block_k = 128  # 256×256 tile 用 block_k=128，K 迭代次数减半
```

#### JIT 编译与启动（L454-490）

通过 `jit_tuner.compile_and_tune` 编译 C++ 模板，模板参数包括 BLOCK_M, BLOCK_N, BLOCK_K,
NUM_GROUPS, NUM_STAGES, NUM_RANKS, NUM_COPY_BLOCKS。编译后缓存在 `~/.deep_gemm/cache/` 下。

---

## 3. Preprocess 阶段（dispatch_preprocess.cuh）

### 3.1 dispatch_expert_preprocess_device()

**文件**: `deep_gemm/include/deep_gemm/dispatch_preprocess.cuh:302-498`

这是 expert-level 的 preprocess。它将所有 rank 的 token 按 expert 聚合，生成 GEMM kernel 需要的调度元数据。

#### Phase 1: SymBuffer 初始化（L324-335）

```cpp
__shared__ SymBuffer smem_sym;
if (threadIdx.x == 0) {
    smem_sym.rank_idx = rank_idx;      // 当前 rank ID
    smem_sym.num_ranks = num_ranks;
    smem_sym.base = sym_buf_addrs[rank_idx];  // 本 rank 的 symmetric buffer 基址
}
if (threadIdx.x < kNumMaxRanks) {
    smem_sym.offsets[threadIdx.x] = sym_buf_addrs[threadIdx.x] - sym_buf_addrs[rank_idx];
}
```

`SymBuffer` 维护每个 rank 的 symmetric buffer 地址偏移。`sym_buffer.map(local_ptr, rank)` 将本地指针映射到远程 rank 的对应位置（通过加偏移实现 P2P 访问）。

#### Phase 2: Flag Polling（L342-350）

```cpp
if (generation > 0 && threadIdx.x < num_ranks) {
    volatile uint32_t* remote_flag = sym_buffer.map(local_flag, threadIdx.x);
    while (*remote_flag < generation) { }  // 自旋等待远程 rank 完成量化
}
```

多 rank 间的同步：等待所有远程 rank 的量化完成后才能读取路由信息。`generation` 是一个单调递增的 epoch 号。

#### Phase 3: 读取 pair token counts（L352-373）

```cpp
for (uint32_t tid = threadIdx.x; tid < total_pairs; tid += num_threads) {
    uint32_t src_rank = tid / num_local_experts;
    uint32_t local_expert = tid % num_local_experts;
    uint32_t global_expert = local_expert_start + local_expert;
    
    uint32_t* remote_counts_ptr = sym_buffer.map(local_counts_ptr, src_rank);
    pair_token_counts[tid] = __ldg(remote_counts_ptr + global_expert);
}
```

从每个远程 rank 的 symmetric buffer 中读取每个 expert 收到的 token 数量。结果存入 SMEM 的 `pair_token_counts` 数组（按 `[rank][expert]` 索引）。

#### Phase 4: Expert-level 聚合 + 前缀和（L376-406）

```cpp
if (threadIdx.x < num_local_experts) {
    uint32_t padded_total = 0;
    for (uint32_t r = 0; r < num_ranks; ++r) {
        uint32_t c = pair_token_counts[r * num_local_experts + e];
        padded_total += c;  // 直接累加，不做 8-alignment padding
    }
    expert_total_count[e] = padded_total;
    expert_m_blocks_arr[e] = ceil_div(padded_total, BLOCK_M);
}
```

每个线程处理一个 expert，汇总所有 rank 的 token 数。之前有 `(c + 7) & ~7u` 的 8-alignment padding，已在 Task 5 中去除。

前缀和计算所有 expert 的 cumulative M-block 数和 cumulative token 数：

```cpp
if (threadIdx.x == 0) {
    for (uint32_t e = 0; e < num_local_experts; ++e) {
        expert_cumsum_blocks[e] = cumsum_b;
        expert_cumsum_m[e] = cumsum_m;
        cumsum_b += expert_m_blocks_arr[e];
        cumsum_m += expert_total_count[e];
    }
    grouped_layout[0] = cumsum_b;  // 总 M-block 数
}
```

`grouped_layout[0]` 是总 M-block 数，GEMM kernel 的 scheduler 和 copy blocks 都需要读取。

#### Phase 5: Greedy Packing — per-rank split metadata（L408-498）

这是最核心的部分：为每个 M-block 生成 per-rank 的数据地址和偏移信息。

```cpp
for (uint32_t mb = 0; mb < num_mblocks; ++mb) {
    // grouped_layout entry: 被 scheduler 和 GEMM kernel 使用
    uint4 entry;
    entry.x = e;            // expert index (local)
    entry.y = total_count;  // expert 的总 token 数（所有 M-blocks 共享此值）
    entry.z = base_block;   // expert 的起始 M-block 全局索引
    entry.w = base_m;       // expert 的起始 token 全局偏移
    reinterpret_cast<uint4*>(grouped_layout + 4)[block_idx] = entry;
```

`grouped_layout` 的内存布局：
- `grouped_layout[0]`: 总 M-block 数（uint32_t）
- `grouped_layout[4..4+total_m_blocks*4]`: 每个 M-block 的 uint4 entry（从偏移 4 开始，每个 entry 占 4 个 int）

每个 M-block 内部，token 按 rank 顺序紧密排列（greedy packing）：

```cpp
for (uint32_t r = 0; r < num_ranks; ++r) {
    uint32_t available = BLOCK_M - smem_row;   // 剩余空间
    uint32_t take = min(remaining[r], available);  // 尽可能多放
    
    rank_addr_a[idx] = remote_fp4 + rank_offset[r] * k_half;  // FP4 源地址
    rank_split_m[idx] = smem_row;   // 在 M-block 内的行偏移
    rank_counts[idx] = take;         // 实际 token 数
    
    smem_row += take;         // 不再有 8-alignment padding
    remaining[r] -= take;
    rank_offset[r] += take;
}
```

数据可以跨 M-block 分割：如果 rank 0 有 200 tokens，BLOCK_M=256，则 rank 0 的前 200 tokens 全在 M-block 0，rank 1 的 tokens 紧接其后填充。如果超出 BLOCK_M，溢出到 M-block 1。

输出数组的索引是 `[block_idx * num_ranks + r]`：每个 M-block 有 `num_ranks` 个 entry。

---

## 4. GEMM Kernel 主体（fp4_gemm_cutlass3.cuh）

### 4.1 Kernel 启动（run_fused_dispatch）

**文件**: `fp4_gemm_cutlass3.cuh:1901-2097`

#### Grid 大小

```cpp
dim3 grid = GemmKernel::get_grid_shape(params);  // = num_sms (39)
grid.x += num_copy_blocks;                       // 39 + 8 = 47
```

总共启动 `num_sms + num_copy_blocks` 个 blocks。前 `num_copy_blocks` 个 block 做 P2P copy，
后面的做 GEMM。硬件有 39 SMs，所以 47 blocks 中约 31 个同时运行（8 copy + 23 GEMM 第一波），
剩余 GEMM blocks 依次调度到空闲 SM。

#### Stride 设置

```cpp
// A 的 stride 用 max_tokens_per_expert 而非 shape_m，因为 A 数据在 local staging buffer 中
// 按 expert 独立存储，每个 expert 的行数最多 max_tokens_per_expert
StrideA stride_A = make_cute_packed_stride(StrideA{}, {max_tokens_per_expert, ShapeK, 1});

// B 的 stride 标准：N×K per group
StrideB stride_B = make_cute_packed_stride(StrideB{}, {ShapeN, ShapeK, 1});
```

#### TileSchedulerArguments 初始化

```cpp
TileSchedulerArguments sched_args(
    shape_m, layout_info,                    // 总 M、grouped_layout 指针
    rank_addr_a, rank_addr_sfa,              // per-(M-block × rank) 数据地址
    rank_split_m, rank_counts, num_ranks,    // per-(M-block × rank) 偏移/计数
    remote_addr_sfa,                         // per-expert 的 merged SFA 地址
    local_fp4_buf, k_half, max_tokens_per_expert,  // local staging buffer 信息
    copy_ready_flags, num_copy_blocks);      // copy 同步 flags
```

### 4.2 Copy Block 逻辑

**文件**: `fp4_gemm_cutlass3.cuh:51-108`

#### Block specialization（operator() 入口）

```cpp
if constexpr (TileScheduler::GEMM_TYPE == GemmType::FusedDispatch) {
    if (blockIdx.x < params.scheduler.num_copy_blocks) {
        uint32_t total_m_blocks = params.scheduler.grouped_layout[0];
        run_copy_block<BlockM>(params.scheduler, total_m_blocks);
        return;  // copy block 完成后直接退出
    }
}
```

Block ID < num_copy_blocks 的是 copy blocks，它们执行完 copy 就退出，不参与 GEMM。

#### run_copy_block()

```cpp
template <uint32_t BLOCK_M>
__device__ void run_copy_block(const TileSchedulerArguments& sched, uint32_t total_m_blocks) {
    uint32_t num_copy = sched.num_copy_blocks;
    
    // Round-robin: block 0 处理 M-block 0,8,16,...; block 1 处理 M-block 1,9,17,...
    for (uint32_t mb = blockIdx.x; mb < total_m_blocks; mb += num_copy) {
```

Round-robin 分配使多个 M-blocks 被并行拷贝。这比"所有 copy blocks 协作同一 M-block"更优，
因为 GEMM 的 M-major 调度需要多个 M-blocks 同时就绪（Task 3 中已验证）。

```cpp
        // 读取 grouped_layout entry
        uint4 gl = (reinterpret_cast<const uint4*>(sched.grouped_layout) + 1)[mb];
        uint32_t expert_local = gl.x;       // expert index
        uint32_t base_bidx = gl.z;           // expert 的起始 M-block 索引
        uint32_t m_block_in_expert = mb - base_bidx;  // 在 expert 内的 M-block 序号
        
        // 读取 per-rank metadata
        for (uint32_t r = 0; r < nr; ++r) {
            uint32_t idx = mb * nr + r;
            pf_addr_a[r] = __ldg(sched.rank_addr_a + idx);    // FP4 远程源地址
            pf_split_m[r] = __ldg(sched.rank_split_m + idx);  // SMEM 行偏移
            pf_counts[r] = __ldg(sched.rank_counts + idx);    // token 数
        }
```

计算本地 staging buffer 的目标地址：

```cpp
        uint8_t* local_a_base = sched.local_fp4_buf +
            expert_local * max_tok * k_half +          // expert 偏移
            m_block_in_expert * BLOCK_M * k_half;      // M-block 内偏移
```

Local staging buffer 布局：`[expert_idx][token_row][k_half]`，每 expert 最多 `max_tokens_per_expert` 行。

#### cooperative_remote_prefetch_fp4()

```cpp
__device__ void cooperative_remote_prefetch_fp4(
    const uint64_t* rank_addrs, const uint32_t* rank_counts,
    const uint32_t* rank_split_m, uint8_t* local_dst,
    uint32_t num_ranks, uint32_t k_half)
{
    for (uint32_t r = 0; r < num_ranks; ++r) {
        if (rank_counts[r] == 0) continue;
        const int4* src = reinterpret_cast<const int4*>(rank_addrs[r]);
        int4* dst = reinterpret_cast<int4*>(local_dst + rank_split_m[r] * k_half);
        uint32_t total_int4s = rank_counts[r] * k_half / 16;  // 每 int4 = 16 bytes
        
        for (uint32_t i = threadIdx.x; i < total_int4s; i += blockDim.x) {
            dst[i] = ld_nc_global(src + i);  // __ldg: non-coherent, cached in L2
        }
    }
}
```

每个线程以 stride=blockDim.x（128 threads）处理不同的 int4 元素。`ld_nc_global` 使用 `__ldg`（non-coherent global load，走 L2 cache 但跳过 L1）。

对于远程 rank 的数据，`src` 实际指向远程 GPU 的 symmetric buffer，通过 NVLink/ICN8 完成 P2P 读取。对于本地 rank 的数据，这是同一 GPU 内的 HBM-to-HBM 拷贝。

#### Flag 通知

```cpp
        __syncthreads();    // 确保所有线程完成写入
        __threadfence();    // 确保写入对其他 block 可见（device-wide fence）
        if (threadIdx.x == 0) {
            sched.copy_ready_flags[mb] = 1;  // 通知 GEMM blocks: M-block mb 已就绪
        }
```

`volatile uint32_t* copy_ready_flags` 数组每个 M-block 一个 flag。Copy block 写 1 表示数据就绪，GEMM blocks 自旋等待此 flag。

### 4.3 GEMM Block 逻辑

#### Spin-wait on copy_ready_flags（L426-431 / L1010-1015）

```cpp
if constexpr (TileScheduler::GEMM_TYPE == GemmType::FusedDispatch) {
    uint32_t global_m_blk = deep_scheduler.curr_global_block_m_idx;
    volatile uint32_t* flag = &params.scheduler.copy_ready_flags[global_m_blk];
    while (*flag == 0) { }   // 自旋等待 copy block 完成
    __threadfence();         // 确保看到 copy block 写入的数据
```

GEMM block 在处理每个 tile 前，先检查对应 M-block 的 copy_ready_flag。如果数据未就绪，自旋等待。`__threadfence()` 确保在 flag=1 可见之后，后续读取的 FP4 数据也已经可见。

#### A 矩阵指针重定向

```cpp
    // 从 local staging buffer 而非远程 symmetric buffer 读 A
    uint32_t expert_local = deep_scheduler.problem_index();
    ptr_A = reinterpret_cast<const ElementA*>(
        params.scheduler.local_fp4_buf +
        expert_local * max_tok * k_half);
```

关键设计：GEMM blocks 读的是 **local staging buffer** 中的 FP4 数据，不是远程地址。
Copy blocks 已经将远程数据搬运到本地 HBM，GEMM 直接从本地读取，避免 NVLink 延迟。

#### SFA scale stride 覆盖

```cpp
    auto dSFA_local = params.mainloop.dSFA;
    if constexpr (TileScheduler::GEMM_TYPE == GemmType::FusedDispatch) {
        get<1>(dSFA_local) = static_cast<int64_t>(M);
    }
```

SFA（Scale Factor A）是 MXFP4 的行级 scale。在标准 GEMM 中 SFA 的 K-dim stride = max_tokens_per_expert，但在 fused dispatch 中每个 expert 的实际 token 数 M ≠ max_tokens_per_expert，因此需要覆盖 stride 为实际的 M。SFA 数据通过 `remote_addr_sfa[expert_local]`（merged SFA 地址）直接从远程读取，不经过 local staging buffer。

#### GEMM 计算（CuTe pipeline）

GEMM 使用 CuTe 的标准 multi-stage pipeline：

1. **Prologue**（L602-617）：预填充 `Stages` 个 K-tile 到 SMEM（A、B、SFA、SFB 各自的 SMEM bank）
2. **Mainloop**（L637-692）：
   - 外层循环：N-expand（通常=1）
   - 中层循环：K-tiles（从 0 到 K_TILE_COUNT）
   - 内层循环：K_BLOCK_MAX sub-tiles（SMEM-to-register 的分块）
   - 每个 sub-tile：`copy` SMEM→register, `gemm` MMA 指令, 异步 `copy_to_tsm` 预取下一个 K-tile
3. **Epilogue**（L693-709）：accumulator → 输出矩阵 D（BF16）

```cpp
// MMA instruction: PPU0015 16×16×64 F32=F4×F4+F32
cute::gemm(tiled_mma, accum, tCrA, tCrSFA, tCrB, tCrSFB, accum);
```

每个 MMA instruction 计算 16×16 的 output tile，输入是 FP4 的 A 和 B，加上各自的 FP16 scale factor，累加到 FP32 accumulator。

---

## 5. Tile Scheduler（scheduler_cutlass3.cuh）

### 5.1 DeepGemmScheduler 结构

**文件**: `scheduler_cutlass3.cuh:119-287`

```cpp
template <GemmType kGemmType, uint32_t SHAPE_N_, uint32_t SHAPE_K_,
          uint32_t BLOCK_M_, uint32_t BLOCK_N_, uint32_t kNumGroups_,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N_, BLOCK_N_),  // = 6144/256 = 24
          uint32_t kNum1DBlocksPerGroup = 2>
struct DeepGemmScheduler {
```

Scheduler 模板参数在编译时确定。`kNumNBlocks = 24` 表示 N 维度有 24 个 tile。

#### 构造函数（L153-169）

```cpp
CUTLASS_DEVICE explicit DeepGemmScheduler(Params const& params_) : params(params_) {
    // FusedDispatch + kIsNoPadPreprocessLayout:
    num_aligned_m_blocks = params_.grouped_layout[0];  // preprocess 输出的总 M-block 数
    num_blocks = num_aligned_m_blocks * num_n_blocks;  // 总 work items = M-blocks × N-tiles
}
```

`kIsNoPadPreprocessLayout` 对 FusedDispatch 总是 true（L136），意味着 grouped_layout 使用 preprocess 生成的紧凑格式而非原始的 per-group-m 数组。

### 5.2 fetch_next_work()

**文件**: `scheduler_cutlass3.cuh:207-287`

这是 tile scheduler 的核心函数，GEMM kernel 的 while 循环每次调用一次来获取下一个 (m_block, n_block) tile。

#### Effective grid 和 block ID

```cpp
uint32_t eff_grid = gridDim.x;
uint32_t eff_bidx = blockIdx.x;
if (params.num_copy_blocks > 0) {
    eff_grid = gridDim.x - params.num_copy_blocks;   // 47 - 8 = 39
    eff_bidx = blockIdx.x - params.num_copy_blocks;   // GEMM blocks 的逻辑 ID
}
const auto next_block_idx = (current_iter++) * eff_grid + eff_bidx;
```

GEMM blocks 的逻辑 ID 从 0 开始（跳过 copy blocks）。Work item 分配是静态 round-robin：
第 `i` 次迭代中，GEMM block `b` 处理 work item `i * eff_grid + b`。

#### FusedDispatch 的 M-major 无 swizzle 路径（L221-231）

```cpp
if constexpr (kGemmType == GemmType::FusedDispatch) {
    int block_m_idx = next_block_idx / kNumNBlocks;   // M-major: 前 24 项 → M-block 0
    n_block_idx = next_block_idx % kNumNBlocks;        // N-tile 索引
    curr_global_block_m_idx = block_m_idx;
    
    uint4 data = (((const uint4*)params.grouped_layout) + 1)[block_m_idx];
    curr_group_idx = data.x;    // expert index
    curr_group_m = data.y;      // expert total token count
    curr_cumsum_m = data.w;     // expert cumulative token offset
    m_block_idx = block_m_idx - data.z;  // M-block 在 expert 内的 local index
}
```

**M-major 顺序**（`block_m_idx = idx / kNumNBlocks`）：work items 按 M-block 分组，
同一 M-block 的所有 N-tiles 连续编号。这与 copy block 的 round-robin M-block 顺序匹配：
copy block 先完成 M-block 0，GEMM blocks 也先处理 M-block 0 的所有 N-tiles。

**无 swizzle**：FusedDispatch 路径直接使用 `block_m_idx` 和 `n_block_idx`，不调用 `get_swizzled_block_idx`。Swizzle 会将同一 expert 的相邻 M-blocks 交错排列以优化 L2 B-matrix 复用，但在 copy 模式下会导致 `curr_global_block_m_idx` 与实际 M-block 不一致，产生 copy_ready_flags 竞态（Task 2 中修复）。

#### 非 FusedDispatch 路径（L232-246）

其他 grouped GEMM 类型（GroupedNoPad, GroupedFused）使用原来的 M-major + swizzle 路径。这些不涉及 copy blocks，没有 flag 竞态问题。

### 5.3 Offset 计算函数

Scheduler 提供多个 `curr_offset_*` 函数用于计算当前 tile 的各矩阵指针偏移。FusedDispatch 主要使用：

- `curr_offset_c()` → `curr_cumsum_m * SHAPE_N`：输出矩阵 D 的行偏移
- `curr_offset_mxfp4_c()` → `curr_group_idx * SHAPE_N`：bias 的 expert 偏移
- `problem_index()` → `curr_group_idx`：当前 expert index
- `curr_problem_m()` → `curr_group_m`：当前 expert 的总 token 数

注意 FusedDispatch 不使用 `curr_offset_a()` 和 `curr_offset_mxfp4_scalea()`，因为 A 矩阵指针在 operator() 中直接从 local_fp4_buf 计算。

---

## 6. 数据流总览

```
Remote Rank r                          Local Rank (current)
┌─────────────────┐                   ┌─────────────────────────────────────────┐
│  Symmetric Buffer│                   │                                         │
│  ┌─────────────┐│   P2P (NVLink)     │  Local Staging Buffer (local_fp4_buf)   │
│  │ FP4 data    ││ ───────────────►   │  ┌───────────────────────────────────┐  │
│  │ (per expert)││  copy blocks       │  │ Expert 0: [token_0..token_n][K/2] │  │
│  └─────────────┘│  (__ldg)           │  │ Expert 1: [token_0..token_m][K/2] │  │
│  ┌─────────────┐│                    │  │ ...                               │  │
│  │ SFA scales  ││ ──(direct read)──► │  └───────────────────────────────────┘  │
│  │ (per expert)││  GEMM blocks       │                                         │
│  └─────────────┘│                    │  Weight B: [expert][N][K]               │
│  ┌─────────────┐│                    │  Scale B:  [expert][N][K/32]            │
│  │ Token counts││ ──(preprocess)──►  │                                         │
│  │ Ready flag  ││                    │  Output D: [total_tokens][N]            │
│  └─────────────┘│                    └─────────────────────────────────────────┘
└─────────────────┘

Copy Block 数据流:
  remote FP4 data ─── __ldg (P2P) ───► local_fp4_buf
  设置 copy_ready_flags[mb] = 1

GEMM Block 数据流:
  等待 copy_ready_flags[mb] == 1
  A:   local_fp4_buf[expert][block_m * BLOCK_M][K/2]  (local HBM read)
  SFA: remote_addr_sfa[expert]                         (remote read via P2P)
  B:   weight_b[expert][n_tile * BLOCK_N][K]           (local HBM read)
  SFB: scale_b[expert][n_tile * BLOCK_N][K/32]         (local HBM read)
  ──► MMA: FP4 × FP4 + scale → FP32 accum → BF16 output
```

**SFA 不经过 local staging buffer** 的原因：SFA 数据量远小于 FP4 数据（每 32 个 FP4 元素共享 1 个 FP16 scale），
直接 P2P 读取的开销可忽略。Copy blocks 只搬运 FP4 data（占数据量 >99%）。

---

## 7. 优化点总结

### 已实施的优化

| 优化 | 位置 | 效果 |
|------|------|------|
| **M-major 无 swizzle**（Task 2）| scheduler fetch_next_work | 修复 copy_ready_flags 竞态 bug，2-GPU -3%（L2 tradeoff） |
| **去除 8-align padding**（Task 5）| dispatch_preprocess | 减少 ~6% 无效 token 计算 |
| **Config 选择修复**（Task 5）| dispatch_fused_gemm.py | 2-GPU: block_m 128→256, block_k 64→128, +17% |

### 验证后放弃的优化

| 方案 | 原因 |
|------|------|
| **Cooperative copy**（Task 3）| 破坏 copy/GEMM pipeline overlap，性能退化 3-8% |
| **PPU Bulk Load/Store**（Task 6）| Copy 指令不是瓶颈，持平 __ldg |

### 当前性能瓶颈

2-GPU (ncb=8): 0.396ms (0.97x non-fused)
4-GPU (ncb=8): 0.447ms (0.80x non-fused)

P2P 隔离实验（将 rank_addr_a 全部重定向到本地数据副本）表明：

| 因素 | 2-GPU | 4-GPU |
|------|-------|-------|
| **P2P link overhead** | 0.060ms (16%) | 0.100ms (24%) |
| SM 分配损失 (31/39 SMs) | ~0.059ms | ~0.064ms |
| 其他 (spin-wait + MC) | ~0.022ms | ~0.011ms |

**P2P link 是最大单一瓶颈**，NVLink 带宽限制了 copy block 吞吐量。
4-GPU 开销更大因为 75% 数据需远程读取（vs 2-GPU 的 50%）。

---

## 8. 测试与运行

### 测试脚本

`tests/test_block_copy_gemm1_multi_gpu.py` — 多 GPU 正确性 + 性能测试。

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TEST_CONFIG` | 配置选择: `small`(2 experts), `prod`(13 experts), `prod12`(12 experts) | `prod` |
| `SKIP_CORRECTNESS` | 设为 `1` 跳过正确性测试，只跑性能 | `0` |
| `FULL_CORRECTNESS` | 设为 `1` 开启逐 expert Python reference 对比 | `0` |
| `PERF_VERBOSE` | 设为 `1` 输出详细性能分解（P2P isolation 等） | `0` |

### 执行命令

需在 Docker 容器 (`deepgemm.lxh`) 内执行，项目根目录为 `/DeepGemm_workspace/codebase/DeepGemm-block-copy`。

```bash
# 8-GPU 性能测试（prod 配置，13 experts/rank）
SKIP_CORRECTNESS=1 PERF_VERBOSE=1 TEST_CONFIG=prod \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 --master_port=29530 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose

# 8-GPU 性能测试（12 experts/rank）
SKIP_CORRECTNESS=1 PERF_VERBOSE=1 TEST_CONFIG=prod12 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 --master_port=29530 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose

# 2-GPU 正确性 + 性能
FULL_CORRECTNESS=1 PERF_VERBOSE=1 \
  CUDA_VISIBLE_DEVICES=2,3 \
  torchrun --nproc_per_node=2 --master_port=29530 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose

# 4-GPU 正确性 + 性能
FULL_CORRECTNESS=1 PERF_VERBOSE=1 \
  CUDA_VISIBLE_DEVICES=2,3,4,5 \
  torchrun --nproc_per_node=4 --master_port=29531 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose
```

### 宿主机调用方式

```bash
sudo docker exec 98d3ee763c25 bash -c "cd /DeepGemm_workspace/codebase/DeepGemm-block-copy && \
  SKIP_CORRECTNESS=1 PERF_VERBOSE=1 TEST_CONFIG=prod \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 --master_port=29530 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose"
```

### 注意事项

- 运行前用 `ppu-smi` 确认目标 GPU 空闲（避免抢占导致性能数据不准）
- `--master_port` 需避免端口冲突，多组测试并行时使用不同端口
- JIT 首次编译耗时约 1-2 分钟，后续命中缓存（`~/.deep_gemm/cache/`）
- 性能数据取 median of 20 iterations，建议连跑 3 次确认稳定性
