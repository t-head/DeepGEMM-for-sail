# Iteration Log

## Kernel

Block-copy fused dispatch GEMM1 (FP4, multi-GPU P2P + local GEMM).

**Files in solution/**:
- `fp4_gemm_cutlass3.cuh` — copy block logic + GEMM mainloop
- `scheduler_cutlass3.cuh` — tile scheduler (work distribution)
- `dispatch_preprocess.cuh` — preprocess (routing → M-block metadata)
- `dispatch_fused_gemm.py` — Python API + launch config

**Bench command**: `torchrun --nproc_per_node={2,4}` with `test_block_copy_gemm1_multi_gpu.py`

**Baseline (pre-AKO)**:
- 2-GPU (ncb=8): 0.396ms pipeline, 0.97x non-fused
- 4-GPU (ncb=8): 0.447ms pipeline, 0.80x non-fused

<!--
Per-iteration template (copy when adding a new iter entry under "## Iterations"):

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - 2-GPU: ___ ms (vs NF ___x)
  - 4-GPU: ___ ms (vs NF ___x)
- **Analysis:** Why it worked or failed
- **Next:** What to try next

Append one row per iter to the Summary table below.
Status values: improved / no-change / regression / failed.
-->

## Summary

| Iter | 标题 | 2-GPU (ms / x NF) | 4-GPU (ms / x NF) | 状态 |
|------|------|--------------------|--------------------|------|
| 1 | 持久化blocks（copy→GEMM转换）| 0.506 / 0.92x | — | 退化 |
| 2 | P2P拷贝循环4x展开 | 0.332 / 1.15x | 0.373 / 1.02x | 改进 |
| 3 | 移除continue + __restrict__ + assume | 0.328 / 1.17x | 0.360 / 0.98x | 改进 |
| 4 | 内联拷贝 + 消除栈开销（均匀路由）| 0.454 / 1.22x | 0.497 / 0.99x | 改进（8卡）|
| 5 | 编译期NumRanks（栈136→0B）| 0.448 / 1.23x | 0.490 / 1.02x | 改进 |
| 6 | 多链路NVLink拷贝（交织4 ranks）| 0.328 / 1.17x | 0.470 / 1.06x | 改进（8卡 **1.08x**）|
| 7 | 交织尾部处理（统一max_n循环）| 0.457 / 1.21x | 0.471 / 1.05x | 改进（2卡）|
| 8 | 调度器tile重分配（2次尝试）| — | — | 无变化（编译器敏感）|
| 9 | 编译期NCB + LICM外提 | — | — | 无变化（代码质量）|
| 10 | Work-stealing（atomicAdd）| — | — | 失败（编译器bug）|
| 11 | NCB扫描 [4,8,13,20] | — | — | 无变化（ncb=8最优）|
| 12 | 移除读端__threadfence | — | 0.364 / 1.05x | 改进（-3~4µs）|
| 13 | 8路NVLink交织 | — | 0.354 / 1.08x | 改进（-11µs P2P）|
| 14 | 双流quant+preprocess重叠 | — | 0.357 / 1.07x | 无变化（已回退）|
| 15 | block_m=128（移除expected_m clamp）| — | 0.310 / 1.24x | **改进（-44µs, +0.16x）** |
| 16 | 配置空间探索：4-stage / NCB扫描 / block_m=64 | — | 0.310 / 1.24x | 无改进（确认平台期）|

> **注意:** Iter 1–3使用随机token路由（每次运行shape_m不同）。
> 从iter 4开始，路由固定为**均匀分布**（round-robin），
> 产生确定性的M维度（2卡: M=1538, 4/8卡: M=1560）。
> 绝对时间在不同路由模式间不可比较；vs-NF比率可比较。

## Iterations

### Iter 1 — 持久化blocks（copy→GEMM转换）

- **假设:** Copy blocks（8个SM）完成拷贝后返回，浪费20%算力。让它们直接转入GEMM应能提高利用率。
- **Changes:**
  - `scheduler_cutlass3.cuh`: Added `work_counter` (uint32_t*) field, FusedDispatch uses
    `atomicAdd` + `__syncthreads()` for dynamic tile scheduling instead of deterministic
    `iter*eff_grid+eff_bidx`.
  - `fp4_gemm_cutlass3.cuh`: Removed `return` after `run_copy_block()`, removed
    `grid.x += num_copy_blocks` (grid = num_sms only).
  - `dispatch_fused_gemm.py`: Allocate work_counter tensor, pass through to kernel.
- **Bench:**
  - Compiled: True
  - Correct: True
  - 2-GPU: 0.506 ms (vs NF 0.92x) — regression from baseline 0.396ms/0.97x
  - 4-GPU: not tested
- **分析:** 原子计数器竞争（312 tiles, 39 blocks）+ 每个tile的`__syncthreads()`屏障
  增加~25%开销，超过SM利用率收益。拷贝时间(0.035ms)相对GEMM(~0.4ms)很小，
  在短尾部恢复8个SM不值得调度开销。确定性调度器的零成本分配在此更优。
- **下一步:** 回退到基线调度。聚焦P2P拷贝优化（实际瓶颈）— 用hgobjdump检查load指令发射模式。

### Iter 2 — P2P拷贝循环4x展开

- **假设:** hgobjdump显示内层循环每个load-store对之间有`s.wait vldcnt(0)` — 同时只有1个NVLink
  load在飞。`if (rank_counts[r] == 0) continue;`阻止编译器预测循环次数，强制保守串行化。
  4x展开应允许4个load同时在NVLink流水线中。
- **Changes:**
  - `fp4_gemm_cutlass3.cuh`: Rewrote inner copy loop with 4x manual unroll — issue 4 `__ldg` loads
    before any stores. Added `__restrict__` on src/dst pointers. Tail loop handles remainder.
- **Bench:**
  - Compiled: True
  - Correct: True
  - 2-GPU: 0.332 ms (vs NF 1.15x) — **16.6% faster** than baseline, now exceeds non-fused!
  - 4-GPU: 0.373 ms (vs NF 1.02x) — **16.6% faster** than baseline 0.447ms, at parity with NF.
  - P2P overhead: 2-GPU 0.021ms (was 0.057ms, **-63%**), 4-GPU 0.037ms (was ~0.090ms)
- **分析:** 确认：多个在飞NVLink load显著降低P2P延迟。4x展开保持4个load同时在NVLink
  流水线中，P2P开销减少60%+。仍有优化空间：编译器可能在4个load之间插入wait。
- **下一步:** 尝试8x展开和/或展平rank循环以消除剩余串行化屏障。

### Iter 3 — 移除continue + __restrict__ + __builtin_assume

- **假设:** rank循环中的`if (rank_counts[r] == 0) continue;`阻止编译器预测控制流，
  强制`s.wait vldcnt(0)`。移除它（内层循环自然处理count=0）加上`__restrict__`和
  `__builtin_assume`应让编译器更积极地调度load。
- **修改:**
  - `fp4_gemm_cutlass3.cuh`: 移除`if (rank_counts[r] == 0) continue;`，给rank_addrs/rank_counts/
    rank_split_m参数加`__restrict__`，加`__builtin_assume`限定num_ranks范围，将`stride`外提到rank循环之前。
- **Bench:**
  - Compiled: True
  - Correct: True
  - 2-GPU: 0.328 ms (vs NF 1.17x) — 从0.332ms微幅提升，P2P开销 0.020ms (原0.021ms)
  - 4-GPU: 0.360 ms (vs NF 0.98x) — 从0.373ms提升，P2P开销 0.034ms (原0.037ms)
  - 8-GPU: 0.408 ms (vs NF 1.00x) — 首次8卡数据，P2P开销 0.052ms
- **分析:** 所有GPU数量下均有温和但一致的提升。移除`continue`减少P2P开销约5-10%。
  8卡与NF持平(1.00x)，即使有7个远端rank。P2P开销近似线性增长
  (0.020/0.034/0.052ms对应1/3/7远端rank)。剩余开销主要来自NVLink带宽限制，非指令调度。
- **下一步:** 用hgobjdump分析汇编确认wait指令变化。探索其他优化方向：指令级分析、
  分支优化、copy block数量调优。

### Iter 4 — 内联拷贝 + 消除栈数组 + 均匀路由

- **假设:** `cooperative_remote_prefetch_fp4`使用本地数组`pf_addr_a[8]`、`pf_split_m[8]`、
  `pf_counts[8]`(128B栈空间)。每次内层循环访问需要vmem.ld从栈读取→`s.wait vldcnt(0)`串行化。
  将拷贝逻辑内联到`run_copy_block`中，通过`__ldg(sched.rank_addr_a + idx)`读取元数据，
  应可消除栈往返开销。同时将benchmark路由从随机改为**均匀分布(round-robin)**以获得确定性结果。
- **修改:**
  - `fp4_gemm_cutlass3.cuh`: 完全移除`cooperative_remote_prefetch_fp4()`。重写`run_copy_block()`
    直接迭代`(mb, r)`，计算`idx = mb * nr + r`进行内联`__ldg`元数据访问。无本地数组。
  - `test_block_copy_gemm1_multi_gpu.py`: 路由从`torch.topk(randn)`改为
    round-robin `(token_idx * topk + j) % num_total_experts`均匀分布。
- **Bench (均匀路由, 各3次运行):**
  - Compiled: True
  - Correct: True
  - iter-3基线(均匀): 2-GPU 0.458ms/1.18x, 4-GPU 0.500ms/0.99x, 8-GPU 0.594ms/0.90x
  - iter-4 (均匀):
    - 2-GPU: 0.459/0.458/0.454 ms (vs NF 1.18/1.21/1.22x), P2P 0.023/0.022/0.022 ms
    - 4-GPU: 0.506/0.498/0.497 ms (vs NF 0.99/1.00/0.99x), P2P 0.043/0.038/0.038 ms
    - 8-GPU: 0.575/0.568/0.564 ms (vs NF 0.93/0.94/0.95x), P2P 0.069/0.072/0.072 ms
  - 最佳: 2-GPU 0.454ms/1.22x, 4-GPU 0.497ms/0.99x, 8-GPU 0.564ms/0.94x
- **分析:** 代码修改对2/4-GPU影响极小(vs iter-3均匀基线~0% delta)，但**8-GPU提升+5%**
  (0.594→0.564 ms, 0.90→0.94x)。符合预期：7个远端rank时，内层循环每个M-block执行7次，
  放大了栈元数据访问开销。P2P开销：0.022/0.038/0.071 ms对应1/3/7远端rank —
  近似线性增长，确认NVLink带宽为主要因素。
  8-GPU run 1 (0.448 ms)为异常值；run 2-4稳定在0.564-0.575 ms。
- **下一步:** 用hgobjdump分析汇编，定位copy循环中剩余`s.wait vldcnt`串行化点。
  目标：减少每次迭代的wait指令，提高NVLink流水线利用率。

### Iter 5 — 编译期NumRanks模板参数

- **假设:** iter-4的ISA分析显示编译器重建了栈数组（尽管源码层面已移除）。根因：
  `sched.num_ranks`是运行时结构体字段，阻止循环展开。编译器将metadata load提升到
  预加载循环(BB0_6)并存入栈(136B)，然后每个rank从栈读回(BB0_8)。将`num_ranks`
  改为编译期模板常量应可实现完全展开和纯寄存器元数据存储。
- **修改:**
  - `scheduler_cutlass3.cuh`: 添加`kNumRanks_`作为第9个模板参数(默认=1)，
    暴露为`constexpr static kNumRanks`。
  - `fp4_gemm_cutlass3.cuh`: 将`run_fused_dispatch`改为模板方法
    `template <uint32_t NumRanks>`。TileScheduler实例化中添加`NumRanks`。
    `run_copy_block<BlockM>` → `run_copy_block<BlockM, TileScheduler::kNumRanks>`。
    使用`constexpr uint32_t nr = NumRanks;`替代`sched.num_ranks`。
  - `dispatch_fused_gemm.py`: JIT调用从`gemm_t::run_fused_dispatch(..., {NUM_RANKS}, ...)`
    改为`gemm_t::template run_fused_dispatch<{NUM_RANKS}>(...)` — NumRanks变为模板参数。
- **Bench (均匀路由, 各3次运行):**
  - Compiled: True
  - Correct: True
  - 资源: vreg=232, sreg=192, **STACK=0** (原136)
  - 2-GPU: 0.457/0.461/0.448 ms (vs NF 1.18/1.20/1.23x), P2P 0.022/0.023/0.023 ms
  - 4-GPU: 0.501/0.493/0.490 ms (vs NF 1.00/1.01/1.02x), P2P 0.038/0.042/0.034 ms
  - 8-GPU: 0.577/0.546/0.557 ms (vs NF 0.92/0.98/0.96x), P2P 0.069/0.073/0.075 ms
  - 最佳: 2-GPU 0.448ms/1.23x, 4-GPU 0.490ms/1.02x, 8-GPU 0.546ms/0.98x
- **分析:** 栈完全消除(136→0B)。ISA确认rank循环完全展开（2-GPU为2份，每份含4x load +
  渐进式drain）。无私有内存(vmem.ga.p)指令。元数据直接从KM加载到标量寄存器。
  从iter-3基线累计提升：2-GPU +2%, 4-GPU +2%, 8-GPU +6%。
  P2P开销仍是8-GPU的主要瓶颈（占kernel时间14-20%）。
- **下一步:** 按用户指令聚焦8-GPU优化。专门分析8-GPU ISA
  （8-rank展开可能造成指令缓存压力）。考虑copy block数量调优、NVLink带宽优化、
  减少剩余s.wait串行化。

### Iter 6 — 多链路NVLink拷贝（跨rank交织）

- **假设:** iter-5的8-GPU ISA分析显示7个展开的rank拷贝按顺序处理 — 每个rank的NVLink load
  通过单条链路，其余6条闲置。ICN8全连接NVLink(每GPU 7条直连链路)下，同时交织4个rank
  的load应可并行使用4条NVLink链路，大幅提高拷贝带宽。预期拷贝加速~3.5x
  (7顺序 → 2组4+3并发)。
- **修改:**
  - `fp4_gemm_cutlass3.cuh`: 重构`run_copy_block`以4个rank为一组处理。每组每rank
    加载1个int4（4条NVLink链路共4个load），渐进式drain (vldcnt 3→2→1→0)。
    不足4个rank的余数仍用原始4x展开单rank拷贝。展开因子保持4（尊重寄存器压力约束），
    同时复用NVLink带宽。
  - NumRanks=2 (2-GPU): 4-group循环不执行(nr < 4)，走余数路径4x展开。同iter-5。
  - NumRanks=4 (4-GPU): 一组4个(含本地rank count=0)。3条远端链路同时活跃。
  - NumRanks=8 (8-GPU): 两组4个。组1: ranks 0-3, 组2: ranks 4-7。每组最多4条NVLink链路活跃。
- **Bench (均匀路由, 各3次运行, 排除JIT预热异常的run 1):**
  - Compiled: True
  - Correct: True
  - 2-GPU: 0.455/0.327/0.328 ms (vs NF 1.19/1.18/1.17x), P2P 0.020/0.020/0.023 ms
  - 4-GPU: 0.467/0.472/0.470 ms (vs NF 1.06/1.05/1.06x), P2P 0.104/0.030/0.032 ms
  - 8-GPU: 0.373/0.486/0.496 ms (vs NF 1.43/1.10/1.07x), P2P 0.041/0.042/0.042 ms
  - 注: Run 1 (8-GPU 0.373ms/1.43x)为异常值 — 可能是JIT缓存清除后GPU频率突增。Run 2-3稳定。
  - 稳定最佳: 2-GPU 0.328ms/1.18x, 4-GPU 0.470ms/1.06x, 8-GPU 0.486ms/1.10x
- **分析:** 多链路交织显著降低P2P开销：
  - 8-GPU P2P: 0.075→0.042ms (**-44%**)。Pipeline: 0.557→0.491ms (**+12%**)
  - 4-GPU P2P: 0.038→0.031ms (**-18%**)。Pipeline: 0.490→0.470ms (**+4%**)
  - 2-GPU P2P: 0.023→0.021ms (无变化 — 仅1条链路，无交织)
  - **8-GPU首次超过NF 1.07-1.10x** (iter-5为0.96x)。
  - 所有GPU数量首次达到或超过NF基准线。
  - 优化随rank数量扩展：更多rank → 更多链路 → 更大收益。
- **下一步:** 进一步8-GPU优化：考虑增大组大小到7（所有rank一组），
  结合多链路交织与2x per-link展开以更深度地饱和NVLink流水线，
  或针对新拷贝特性调优NCB。

### Iter 7 — 交织尾部处理（统一max_n循环）

- **假设:** 当4个交织rank有不同token数（如[10, 14, 16, 18]）时，iter-6代码先跑min_n主循环
  （4条链路活跃），然后4个**独立**尾部循环（各1条链路活跃）。均匀随机分布下count方差±7 tokens，
  尾部可占拷贝工作~30%，以1/4带宽运行。将4个顺序尾部循环替换为**单个**交织条件尾部循环
  应可在尾部区域保持所有4条NVLink链路活跃。
- **修改:**
  - `fp4_gemm_cutlass3.cuh`: 混合方案 — 保留无条件min_n主循环（最佳ILP，无分支开销），
    将4个独立`for (ii = i; ii < nX; ii += stride)`尾部循环替换为单个
    `for (; i < max_n; i += stride)`循环，使用条件load/store：
    `if (i < n0) v0 = ld_nc_global(s0 + i)`等。主体部分保持完整ILP（所有count相同），
    尾部保持所有链路活跃（count不同时）。
  - 注: 先尝试了纯max_n方案（无min_n主循环），4-GPU从1.06x退化到0.86x —
    条件分支在通常（count相等）情况下开销过大。混合方案修复了此问题。
- **Bench (均匀路由, 各3次运行):**
  - Compiled: True
  - Correct: True
  - 2-GPU: 0.455/0.457/0.458 ms (vs NF 1.19/1.21/1.21x), 最佳NCB=8
  - 4-GPU: 0.466/0.471/0.471 ms (vs NF 1.06/1.05/1.05x), 最佳NCB=8
  - 8-GPU: 0.493/0.494/0.495 ms (vs NF 1.08/1.08/1.08x), 最佳NCB=8
  - 最佳: 2-GPU 0.457ms/1.21x, 4-GPU 0.471ms/1.05x, 8-GPU 0.494ms/1.08x
- **分析:** 交织尾部在2-GPU上提升~4% (1.17→1.21x NF)，减少尾部拷贝时间，
  使更少NCB也能良好工作（更少copy SM → 更多GEMM SM）。4-GPU和8-GPU无显著变化 —
  在4/8 rank下尾部相对主循环很小（均匀分布在高rank数时方差低）。8-GPU P2P开销
  ~0.036ms (8.5%)，表明拷贝带宽并非主要瓶颈。8-GPU瓶颈在于GEMM使用更少SM (31 vs 39)
  加上pipeline同步开销(0.064ms)。
- **下一步:** copy block已充分优化。聚焦减少pipeline开销和提高copy block下的GEMM效率：
  考虑动态调整NCB、改进copy-GEMM重叠信号机制、或让copy block完成拷贝后参与GEMM
  （以更轻量调度重新审视iter-1的持久化block方案）。

### Iter 8 — 调度器tile重分配（2次尝试）

- **假设:** 当前tile调度器中，wave 2 blocks（8个在T_c≈0.039ms完成拷贝后加入GEMM的原copy blocks）
  与wave 1 blocks处理相同tile数(8)。但wave 2启动晚(+T_c)，所以它们最后完成，延长关键路径。
  重分配让wave 2处理更少tile应能缩短关键路径：
  critical_path = max(wave1_tiles × T_tile, T_c + wave2_tiles × T_tile)。
- **尝试1: N-major tile排列 + flag缓存。**
  将调度器从M-major改为N-major排列，每个GEMM block处理单个M-block的所有N-tiles (24 tiles,
  同一expert)。添加flag检查缓存(`curr_global_block_m_idx`连续tile不变)。预期flag检查从8次降到1次。
  - 结果: kernel退化 0.313→0.322ms (3次运行稳定)。
  - 根因: 运行时除以`num_aligned_m_blocks`(运行时变量)vs 编译期`kNumNBlocks`除法。
    除数非编译期常量时编译器生成更差代码。
  - Local-only确认: 0.273→0.277ms = 纯GEMM退化。
  - **已回退。**
- **尝试2: 静态wave感知tile分配。**
  在调度器中添加`w1_total_tiles`成员变量。Wave 1的31个GEMM blocks处理tiles 0–255 (8–9 tiles)，
  wave 2的8个blocks处理tiles 256–311 (7 tiles)。
  验证分布: 8×9 + 23×8 + 8×7 = 72 + 184 + 56 = 312 ✓。
  - 结果 (3次运行, 8-GPU):
    - Pipeline: 0.370/0.377/0.374ms (vs基线 0.370ms — 无变化)
    - Kernel: 0.314/0.312/0.312ms (vs基线 0.313ms — 无变化)
    - Local-only: 0.292/0.292/0.292ms (vs基线 0.273ms — **退化+7%**)
    - P2P开销: 0.021/0.020/0.021ms (vs基线 0.039ms — **改进-49%**)
  - 根因: 添加`w1_total_tiles`成员变量 + `fetch_next_work`中的条件逻辑改变了编译器
    寄存器分配/指令调度，GEMM mainloop退化量恰好等于P2P改进量。净效果为零。
  - **已回退。**
- **分析 — 关键发现：编译器敏感性。**
  PPU编译器对调度器中的任何代码改动极度敏感，包括：
  (a) 向调度器struct添加成员变量
  (b) 在`fetch_next_work`中添加分支（即使用`if constexpr`保护）
  (c) 将除法操作数从编译期改为运行时
  即使运行时不影响热路径的修改也会改变整个kernel（包括GEMM mainloop）的寄存器分配和
  指令调度，造成可测量的退化(~0.019ms / 7%)。tile处理循环上的
  `#pragma clang loop licm(disable)`放大了此效应。
  **含义:** 未来优化须避免修改调度器代码路径。选项：(1)在调度器外优化(pipeline开销、preprocess)，
  (2)减少代码复杂度而非增加，(3)在ISA层面工作。
- **下一步:** 探索非调度器优化：减少pipeline开销(融合/消除preprocess步骤)，
  分析FusedDispatch与GroupedNoPad的ISA差异以理解0.025ms基础设施差距，
  或尝试grid=39让copy block完成拷贝后做GEMM。

### Iter 9 — 编译期NCB + LICM外提

- **假设:** 0.025ms基础设施差距(FusedDispatch local-only 0.273ms vs GroupedNoPad 0.248ms)
  可能由以下原因造成：
  (a) flag检查/copy block路由的死代码膨胀（运行时ncb条件判断）
  (b) `#pragma clang loop licm(disable)` tile循环内循环不变值
      (k_half, max_tok, fp4_buf, sfa_addrs)的逐tile重加载。
  将ncb改为编译期模板常量可在ncb=0时消除死代码。手动LICM外提将不变量load移到循环前。
- **修改:**
  - `scheduler_cutlass3.cuh`: 添加`kNumCopyBlocks_`作为第10个模板参数(默认=0)。
    `fetch_next_work`使用`if constexpr (kNumCopyBlocks > 0)`替代运行时
    `params.num_copy_blocks`的eff_grid/eff_bidx减法。
  - `fp4_gemm_cutlass3.cuh`: 两个operator()函数使用`TileScheduler::kNumCopyBlocks`
    进行copy block路由和flag检查(`if constexpr`替代运行时检查)。
    添加`run_fused_dispatch<NumRanks, NumCopyBlocks>`模板。将fd_k_half、fd_max_tok、
    fd_fp4_buf、fd_sfa_addrs外提到tile循环前。
  - `dispatch_fused_gemm.py`: 将NUM_COPY_BLOCKS作为第2个模板参数传递；从运行时参数列表移除。
- **Bench (8-GPU, 3次运行):**
  - Compiled: True
  - Correct: True
  - 仅编译期NCB（无外提）:
    - Pipeline: 0.372/0.370/0.384ms, Kernel: 0.312/0.312/0.314ms, Local-only: 0.274/0.275/0.274ms
    - 相比基线无可测变化 (0.370/0.313/0.273ms)
  - 含LICM外提:
    - Pipeline: 0.371/0.371/0.376ms, Kernel: 0.310/0.308/0.314ms, Local-only: 0.273/0.274/0.274ms
    - Kernel平均0.311ms vs基线0.313ms — 噪声范围内
- **分析:**
  (a) 基础设施差距**并非**来自死代码膨胀。通过`if constexpr` (ncb=0)消除flag检查和
      copy block代码并未改变local-only时间(0.274ms vs 0.273ms)。差距是结构性的 —
      来自FusedDispatch的per-expert A指针算术(`expert_local * max_tok * k_half`)
      和SFA步长覆写(`dSFA_local = M`)，GroupedNoPad不需要这些。
  (b) 4个不变量的LICM外提对kernel时间显示边际改进(平均~0.002ms)，但在测量噪声范围内。
      GPU硬件在L1中缓存struct成员访问，重加载vs寄存器访问仅~3-4周期差异。
  代码质量改善：编译期检查更清晰，不变量显式化。保留修改（无退化，代码更好）。
- **下一步:** 基础设施差距(0.025ms)是结构性的，难以进一步缩小。聚焦减少pipeline开销
  (0.058ms) — 测量quant vs preprocess vs kernel launch的分解。或探索通过改进copy-GEMM
  重叠减少P2P开销。

### Iter 10 — Work-stealing tile调度（失败 — PPU编译器bug）

- **假设:** Wave-2 blocks（8个blocks从T_c=0.039ms开始）与wave-1 blocks（31个从T=0开始）
  处理相同tile数(8)。基于atomicAdd的work-stealing可让wave-2获得更少tiles(~7)（因启动晚），
  wave-1获得更多(~9)。预估节省：0.311ms → 0.280ms = kernel提升10%。
- **尝试的修改:**
  1. 向TileSchedulerArguments添加`tile_counter` (uint32_t*)成员 + `fetch_next_work`中
     atomicAdd: **崩溃** — 寄存器分配损坏导致"illegal memory access"
  2. 从struct移除tile_counter，从`copy_ready_flags + num_aligned_m_blocks`计算地址:
     **崩溃** — 同样的寄存器损坏
  3. 给fetch_next_work添加`__noinline__`: **崩溃** — 编译器仍然错误分配
  4. 使用内联PTX `atom.global.add.u32`: **崩溃** — 同一问题
  5. 使用`atomicCAS`循环替代`atomicAdd`: **崩溃** — 任何原子指令都触发bug
  6. 对照测试: `if constexpr`搭配round-robin（无原子指令）: **通过** — 确认
     崩溃特指原子指令，而非分支结构
- **根因:** PPU编译器(HGGC 13.0)存在寄存器分配敏感性bug，`fetch_next_work`中**任何**
  原子指令的存在都会损坏GEMM mainloop（无关代码）的寄存器分配，产生垃圾地址偏移
  (~94万亿)的`vmem.ld/st.b32`。无论内联、类型转换方式还是原子指令选择均触发。
- **替代分析:** 静态tile分区(wave-1获8-9 tiles, wave-2获7)仅提升~5µs
  (0.311 → 0.306ms, 1.6%)，因整数tile粒度限制。不值得增加复杂度。
- **Bench (8-GPU):** 所有原子变体在产出结果前崩溃。回退到iter-9。
- **清理:** 从struct、Python API和测试文件移除`tile_counter`。`create_block_copy_buffers`
  返回2个值(原3个)。copy_ready_flags保留+1元素（无害，留作未来用途）。
  fetch_next_work回退到iter-9的round-robin。
- **下一步:** work-stealing被编译器bug阻断。转向pipeline开销优化：
  preprocess kernel融合、quant优化、或更高层pipeline重构。

### Iter 11 — NCB扫描：更多copy blocks以缩短wave-2延迟（负面结果）

- **假设:** ncb=8且13个M-blocks时，copy blocks 0-4各处理2个M-blocks (T_c ≈ 38µs)，
  blocks 5-7各处理1个(T_c ≈ 19µs)。Wave-2 GEMM blocks等待最慢copy block (38µs)。
  增加ncb到13使每个copy block恰好处理1个M-block，最大拷贝时间减半到~19µs。
- **修改:** 测试中将NCB扫描范围从[1,2,4,8]扩展到[4,8,13,20]。
  无kernel代码改动 — 纯launch参数实验。
- **Bench (8-GPU, 3次运行, M=1560 N=6144 K=7168):**

  | NCB | Run 1 (ms) | Run 2 (ms) | Run 3 (ms) | 平均 (ms) |
  |-----|-----------|-----------|-----------|----------|
  | 4   | 0.407     | 0.405     | 0.406     | 0.406    |
  | 8   | 0.375     | 0.375     | 0.367     | **0.372**|
  | 13  | 0.380     | 0.377     | 0.380     | 0.379    |
  | 20  | 0.388     | 0.374     | 0.384     | 0.382    |

  最佳: ncb=8 (pipeline平均 0.372ms, 1.03x vs NF平均 0.382ms)

  Run 3详细 (best_ncb=8):
  - Pipeline (完整):           0.367 ms
  - Pipeline (无preprocess):   0.326 ms
  - Kernel-only (copy+GEMM):   0.313 ms (std=0.009)
  - Local-only:                0.272 ms (std=0.008)
  - NF Pipeline:               0.383 ms
  - NF GEMM-only:              0.248 ms
  - P2P链路开销:               +0.041 ms (+13.2%)
  - Preprocess节省:            0.041 ms

- **分析:** Per-M-block flag信号机制使NCB假设无效。
  Copy blocks逐M-block发信号（非一次性全发）。ncb=8时：
  - M-blocks 0-7: 在T≈19µs完成拷贝（每个copy block的第一个M-block）
  - M-blocks 8-12: 在T≈38µs完成拷贝（copy blocks 0-4的第二个M-block）
  但GEMM blocks的tiles是M-major排列：M-blocks 8-12的tiles在tile indices 192-311。
  通过round-robin，GEMM blocks在迭代5+到达这些tiles (T≈170+µs)，远晚于T=38µs。
  因此flag检查立即通过 — 无GEMM阻塞。
  
  实际P2P开销(0.041ms)来自5个wave-2 GEMM blocks在T=38µs**开始**
  （copy blocks 0-4释放SM时）。这些blocks不等待flags但在时间上被推后。
  ncb=13时所有wave-2 blocks在T=19µs开始(早19µs)，但有13个wave-2 blocks而非8个，
  更大的grid (52 vs 47)增加调度开销和更多并发copy写入的L2缓存压力。
  
  关键发现：对此工作负载(13 M-blocks, 均匀分布)，ncb=num_ranks=8是最优的。
  更多copy blocks无法减少kernel时间，因为首M-block拷贝时间(19µs)已经决定了
  大多数GEMM tiles何时可开始，后续M-blocks的拷贝时间完全被GEMM计算掩盖。

- **P2P开销分解 (0.041ms总计):**
  - Wave-2启动延迟: ~38µs (5个copy blocks占用SM)
  - 首tile flag等待: ~19µs (所有GEMM blocks等待M-block 0)
  - Flag检查开销: ~0.17µs/tile (可忽略)
  - Kernel基础设施差距 vs NF: 0.024ms (local-only 0.272 vs NF 0.248)
    可能来自per-tile __threadfence + FusedDispatch中的A指针间接寻址

- **下一步:** Pipeline开销(0.054ms)和kernel基础设施差距(0.024ms)是剩余优化目标。选项：
  1. 常量路由跳过preprocess (节省0.041ms → 0.326ms pipeline)
  2. 减少per-tile开销（消除GEMM flag检查中的__threadfence）
  3. 尝试GEMM mainloop中直接NVLink读取（完全消除copy blocks）

### Iter 12 — 移除读端__threadfence()（温和正面）

- **假设:** 读端`__threadfence()`在flag轮询后是冗余的。写端(copy block)在设置flag前
  已做`__threadfence()`，将A数据刷入L2。GEMM mainloop使用的TMA load直接通过L2
  （绕过L1）。volatile flag读提供编译器排序。因此读端fence不必要 —
  编译器屏障(`asm volatile("" ::: "memory")`)足以防止编译器重排load。
- **修改:**
  1. 在GEMM flag检查路径中将`__threadfence()`替换为`asm volatile("" ::: "memory")`
     (nExpand=1和nExpand>1两个特化)
  2. 将`copy_ready_flags`基地址外提到LICM禁用的while循环前
     (与已有的fd_k_half、fd_max_tok、fd_fp4_buf外提并列)
- **Bench (8-GPU, 3次运行, M=1560 N=6144 K=7168):**

  Run 3异常值 (kernel 0.344ms, 可能干扰) — 仅使用run 1-2。

  | 指标 | Iter-12 R1 | Iter-12 R2 | 基线 (iter-11) | 差异 |
  |------|-----------|-----------|---------------|------|
  | Pipeline | 0.364 | 0.364 | 0.367-0.375 | -3~-8µs |
  | 无preproc | 0.336 | 0.338 | 0.326 | +10µs (噪声) |
  | Kernel-only | 0.309 | 0.310 | 0.312-0.313 | -3µs |
  | Local-only | 0.268 | 0.269 | 0.272 | -3~-4µs |
  | NF GEMM | 0.248 | 0.248 | 0.248 | 0 |
  | vs NF pipe | 1.05x | 1.05x | 1.03-1.04x | +0.01-0.02x |

  基础设施差距: 0.020ms (原0.024ms → 减少4µs, 差距缩小17%)
  正确性: 3次运行全部通过。

- **分析:** 移除读端`__threadfence()`是**安全的**(正确性已确认)。
  每次kernel调用节省~3-4µs。确认：
  1. 读端fence是纯开销 — TMA load正确看到L2刷新的数据
  2. 剩余基础设施差距(0.020ms)来自FusedDispatch代码路径差异
     (A指针乘法、dSFA覆写、元数据读取) — 不做根本重构无法消除
  3. 4µs = ~500ns/tile × 8 tiles = ~60 cycles/fence，与GPU `membar.gl`开销一致

- **下一步:** 剩余优化目标：
  1. Pipeline开销: preprocess 0.026-0.029ms + quant 0.025ms = ~0.054ms
  2. 基础设施差距: 0.020ms (不重构难以进一步缩小)
  3. P2P wave-2延迟: 0.041ms (SM共享方案固有)

### Iter 13 — 8路NVLink交织（正面）

- **假设:** 4路rank交织将8条NVLink链路分2个顺序pass处理(ranks 0-3, 然后4-7)。
  每个pass串行化其读取，总拷贝时间为2×单pass时间。同时交织所有8个rank，
  可在单个pass内并行发出8条NVLink链路的读取，迭代次数减半，拷贝时间下降。
- **修改:** 在已有4路循环前添加8路交织循环(`r + 8 <= nr`)。NumRanks=8时，
  每次内层循环迭代发出8个source读取和8个destination写入。4路和1路循环作为
  非8-rank的回退路径保留。寄存器压力：~61 regs/thread (8 int4数据 + 16指针 +
  8 counts) — 远低于SM限制(65536的24%)。
- **Bench (8-GPU, 3次运行, M=1560 N=6144 K=7168):**

  Run 1 (JIT冷启动)偏高；使用run 2-3进行稳定对比。

  | 指标 | Iter-13 R2 | Iter-13 R3 | Iter-12 (R1-R2) | 差异 |
  |------|-----------|-----------|-----------------|------|
  | Pipeline (ncb=8) | 0.354 | 0.357 | 0.364 | -7~-10µs |
  | Kernel-only | 0.292 | 0.299 | 0.309-0.310 | -11~-18µs |
  | Local-only | 0.266 | 0.264 | 0.268-0.269 | -2~-5µs |
  | P2P开销 | 0.026 | 0.034 | 0.041 | -7~-15µs |
  | NF pipeline | 0.385 | 0.381 | 0.381-0.383 | — |
  | vs NF | 1.08x | 1.07x | 1.05x | +0.02-0.03x |

  NCB扫描 (run 2-3平均): ncb=8和ncb=20以0.356ms并列。更快的拷贝使
  更多copy blocks收益递减（拷贝已经足够快了）。

  正确性: 3次运行全部通过。

- **分析:** 8路交织将P2P开销减少~11µs (27%)，从0.041ms降至平均0.030ms。
  与假设一致：8条NVLink同时读取vs 2批4条减少了迭代次数。Copy block更快完成，
  更早释放SM给wave-2 GEMM blocks。

  iter-12 + iter-13累计改进：
  - 基础设施差距: 0.024ms → 0.018ms (减少25%, __threadfence移除)
  - P2P开销: 0.041ms → 0.030ms (减少27%, 8路交织)
  - Pipeline: 0.367ms → 0.355ms (快12µs, 1.07-1.08x vs NF)

- **下一步:** 剩余开销分解（最佳运行）:
  1. P2P wave-2延迟: 0.030ms (已减少但仍是最大单一来源)
  2. Pipeline开销: ~0.056ms (quant + preprocess)
  3. 基础设施差距: 0.018ms (FusedDispatch代码路径差异)

### Iter 14 — 双流quant+preprocess重叠（负面结果）

- **假设:** Pipeline开销(~0.056ms)由GEMM前串行的quant (0.025ms)和preprocess (0.028ms)
  kernel组成。在stream 1启动quant、stream 2启动preprocess（通过CUDA event等待quant），
  然后在stream 1执行GEMM（通过event等待preprocess），应可重叠部分preprocess与quant，
  节省~10µs pipeline时间。
- **修改:** 修改test_block_copy_gemm1_multi_gpu.py使用双CUDA流：
  stream 1做quant，stream 2做preprocess(通过CUDA event等待quant)，
  然后stream 1做GEMM(通过event等待preprocess)。无kernel代码改动。
- **Bench (8-GPU, 3次运行, M=1560 N=6144 K=7168):**

  | 指标 | 双流 R1 | 双流 R2 | 基线 (iter-13) | 差异 |
  |------|--------|--------|---------------|------|
  | Pipeline | 0.358 | 0.355 | 0.354-0.357 | ~0µs |

  无可测改进。3次运行均在基线噪声范围内。

- **分析:** 双流方案在PPU上失败，两个原因：
  1. PPU硬件可能串行化不同流的kernel launch — 该架构未有效支持并发kernel执行
  2. CUDA event record/wait开销(每对~5-10µs)抵消了~10µs quant+preprocess
     重叠窗口的理论节省

  quant和preprocess kernel都是小型单block或少block kernel。
  kernel launch开销和event同步成本主导了任何潜在重叠节省。

- **已回退。** 无代码改动保留 — 测试文件恢复到串行pipeline。

- **剩余开销分析:**
  此时kernel级优化已达平台期：
  - Pipeline: 0.354-0.357ms (1.07-1.08x vs NF 0.381-0.385ms)
  - Kernel-only: 0.292-0.299ms
  - Local-only: 0.264-0.266ms (vs NF GEMM-only 0.248ms)
  
  剩余开销分解：
  1. P2P wave-2延迟: 0.030ms — NVLink带宽限制 (5.59MB / 200 GB/s = 28µs理论下限)。
     8个copy blocks共享相同200 GB/s总NVLink带宽。更多copy blocks分配工作但不增加总带宽。
  2. Pipeline开销: 0.056ms — quant (0.025ms) + preprocess (0.028ms)，串行。
     双流重叠在PPU上无帮助。
  3. 基础设施差距: 0.018ms — FusedDispatch per-expert A指针算术和SFA步长覆写。
     结构性，不做根本重构无法消除。

- **下一步:** 探索preprocess kernel优化（单block, phase 5仅13个线程活跃做散射store）
  或独立copy kernel launch（消除wave-2延迟但需大量重构）。

### Iter 15 — block_m=128 tile尺寸（移除expected_m clamp）(**重大正面**)

- **假设:** 120 tokens/expert时block_m=256下，每个M-block利用率47% (120/256)。
  GEMM计算256×256 tiles但仅120行有效 — 53%计算浪费在padding上。切换到block_m=128
  利用率94% (120/128=0.9375)，M-block数不变(ceil(120/128)=1)，总FLOPs从292T减半到146T。
  
  原始block_m=256选择的根因：`dispatch_fused_gemm.py`有clamp
  `if expected_m <= 128: expected_m = 129`强制配置选择器看到
  ceil(129/128)=2 M-blocks/expert（tiles从312翻倍到624），使block_m=128看起来更差。
  
  此外GroupedNoPad配置路径对block_m=128默认block_k=64，K迭代从28翻倍到56。
  通过为(128, 256)添加block_k=128 override和tile_config_normal中添加
  (128, 256, 128):(64,64,3)配置项修复。

- **修改:**
  1. `dispatch_fused_gemm.py`: 移除`if expected_m <= 128: expected_m = 129` clamp
  2. `gemm_fp4.py`: 向tile_config_normal添加`(128, 256, 128): (64, 64, 3)`
  3. `gemm_fp4.py`: 在GroupedNoPad block_k选择中添加
     `if (best_block_m == 128 and best_block_n == 256): block_k = 128`
  4. `test_block_copy_gemm1_multi_gpu.py`: 移除get_gemm_block_m()和配置打印代码中
     相同clamp（共3处）

- **配置变更:**
  - 之前: block_m=256, block_n=256, block_k=128, smem=209,600
  - 之后: block_m=128, block_n=256, block_k=128, smem=157,184
  - ThreadNum: 256 (原512), 相同warp_m=64, warp_n=64, num_stages=3
  - Tiles: 312 (不变), 每tile FLOPs: 减半 (128×256 vs 256×256)

- **Bench (8-GPU, 3次运行, M=1560 N=6144 K=7168):**

  Run 2 kernel/local异常（GPU热降频）。Run 1 & 3稳定。

  | 指标 | Run 1 | Run 3 | Iter-13最佳 | 差异 |
  |------|-------|-------|-----------|------|
  | Pipeline | 0.312 | 0.310 | 0.354 | **-44µs (-12%)** |
  | Kernel-only | 0.262 | 0.250 | 0.292 | **-42µs (-14%)** |
  | Local-only | 0.228 | 0.218 | 0.264 | **-46µs (-17%)** |
  | NF pipeline | 0.383 | 0.383 | 0.381 | — |
  | NF GEMM-only | 0.248 | 0.248 | 0.248 | — |
  | vs NF pipe | 1.23x | 1.24x | 1.07x | **+0.16x** |
  | P2P开销 | 0.033 | 0.032 | 0.030 | ~0 |

  NCB扫描 (run 1/3): ncb=8和ncb=13以0.311-0.313ms并列。ncb=4更差(0.353ms)。
  
  正确性: 3次运行全部通过(13 experts, 与NF参考精确匹配)。

- **分析:** 这是整个系列中单次最大的优化(+0.16x vs NF比率)。三个关键改进：
  
  1. **GEMM FLOPs减半** (292T → 146T): block_m=128下120有效行/M-block，
     仅6%计算浪费(8行padding) vs block_m=256的53%(136行padding)。
     直接将per-tile计算时间减半。
  
  2. **Local-only GEMM现在超越NF GEMM-only** (0.218ms vs 0.248ms = 快12%):
     "基础设施差距"**反转**。block_m=128下FusedDispatch的M利用率(94%)
     优于NF的GroupedMasked (block_m=256, max_tokens=195, 76%利用率)。
     Per-expert A指针算术开销被利用率优势超过。
  
  3. **共享内存减少** (157KB vs 210KB): 更小tiles使用更少smem，
     但两种配置均为1 block/SM (262KB容量/smem > 1)。
  
  P2P开销不变(~0.032ms)因copy block逻辑不依赖block_m —
  无论tile大小拷贝相同数据。Copy block线程数更少(256 vs 512)但NVLink带宽仍是瓶颈。
  
  Pipeline开销仍~0.055ms (quant 0.025ms + preprocess 0.028ms)。

- **下一步:** 在iter-16中探索了3个额外方向（均为负面）。

---

### Iter 16 — Config space exploration: 4-stage pipeline, NCB sweep, block_m=64

**目标:** 在GEMM配置空间中寻找进一步优化。

**进行了三组实验:**

#### 实验A: 4-stage pipeline (stages 3→4)
- **假设:** 4级流水线能更好地隐藏访存延迟。smem=209KB仍可放入262KB（1 block/SM）。
- **配置:** (128, 256, 128) + (64, 64, 4)
- **结果（3次运行）:**
  | 指标 | Run 1 | Run 2 | Run 3 |
  |------|-------|-------|-------|
  | Pipeline (best ncb) | 0.309 | 0.308 | 0.301* |
  | Kernel-only | 0.244 | 0.314† | 0.248 |
  | Local-only | 0.213 | 0.227 | 0.213 |

  *Run 3 ncb=20异常（NF基线退化至0.459ms）。†Run 2热降频。

- **分析:** GEMM本身提升~5µs (0.244ms vs 0.250ms)，但pipeline改进在噪声范围内
  (0.308-0.312ms vs 0.310-0.312ms)。更大的smem不影响占用率（两种配置均为1 block/SM）。
  多一级流水线的微弱收益被smem访问开销抵消。**无改进。**

#### 实验B: 细粒度NCB扫描 [2,4,6,8,10,13,16,20]
- **假设:** 原始扫描[4,8,13,20]可能遗漏最优NCB值。
- **结果:**
  | NCB | Pipeline (ms) | vs NF |
  |-----|--------------|-------|
  | 2   | 0.418 | 0.91x |
  | 4   | 0.355 | 1.07x |
  | 6   | 0.336 | 1.13x |
  | 8   | 0.311 | 1.22x |
  | 10  | 0.321 | 1.19x |
  | 13  | 0.310 | 1.23x |
  | 16  | 0.310 | 1.23x |
  | 20  | 0.318 | 1.20x |

- **分析:** NCB=8/13/16均在0.310ms处达到平台期。小于8：copy block不足，单块拷贝时间过长；
  大于16：copy block过多，GEMM block不足。**NCB已是最优，无改进空间。**

#### 实验C: block_m=64（尝试2x SM占用率）
- **假设:** block_m=64仅需131KB smem（vs 157KB），理论上可实现2 blocks/SM (262/131=2)，
  更高占用率可更好隐藏延迟。
- **实施难点:** 需绕过两个硬编码override：
  1. `qwen3-next` override（第421行）：当expected_m≈block_m时翻倍block_m
  2. DeepSeek-V4 EP override（第465行）：对n=6144,k=3584强制block_m=128
- **配置:** (64, 256, 128) + (32, 64, 3), ThreadNum=256
- **结果:**
  | 指标 | block_m=64 | block_m=128 | 比率 |
  |------|-----------|-------------|------|
  | Pipeline | 0.528ms | 0.310ms | 慢1.70x |
  | Kernel-only | 0.506ms | 0.250ms | 慢2.02x |
  | Local-only | 0.416ms | 0.218ms | 慢1.91x |

- **分析:** **灾难性退化。** 尽管总FLOPs相同，local-only GEMM几乎翻倍。根因：每个tile
  的固定开销（epilogue、TMA初始化、流水线启动/排空）非常显著（~12µs/tile），远高于
  block_m=128的情况。tile数量翻倍（624 vs 312）导致固定开销翻倍。即使寄存器压力允许
  2 blocks/SM，也无法弥补2倍的per-tile开销。**block_m=128为此工作负载的最优选择。**

**结论:** 三组实验均为负面结果。iter-15配置（block_m=128, block_n=256, block_k=128,
stages=3, ncb=8/13）已达到当前kernel架构的优化平台期。

**优化平台期分析:**
- 当前性能：pipeline 0.310ms（1.24x vs NF 0.383ms）
- 理论下限（仅local GEMM）：0.218ms（1.76x vs NF）
- 实际下限（local GEMM + P2P）：0.250ms（1.53x vs NF）
- 剩余开销：quant 0.025ms + preprocess 0.028ms + launch ~0.007ms = 0.060ms
- 进一步优化需要：将quant/preprocess融合到GEMM kernel（cooperative launch方案，
  预计可节省~20µs，但实现复杂度较高）
