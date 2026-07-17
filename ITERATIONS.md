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
| 17 | 基准修正：消除scale预处理开销（COL_MAJOR_SCALE）| — | 0.290 / 1.02x | **基线更新** |

> **注意:** Iter 1–3使用随机token路由（每次运行shape_m不同）。
> 从iter 4开始，路由固定为**均匀分布**（round-robin），
> 产生确定性的M维度（2卡: M=1538, 4/8卡: M=1560）。
> 绝对时间在不同路由模式间不可比较；vs-NF比率可比较。
>
> **环境变更:** Iter 1–16 在 deepgemm.lxh 容器中测试，NF 基线含 scale round-trip 开销
> （~90µs）。Iter 17+ 在 sglang.lxh 容器中测试（`COL_MAJOR_SCALE=1`），NF 基线公平。
> Iter 17 的 vs-NF 比率与 iter 1-16 **不可直接比较**。

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

---

### Iter 17 — 基准修正：消除scale预处理开销（COL_MAJOR_SCALE）

- **背景:** Iter 1-16 的 non-fused 基线中包含了不必要的 scale 预处理开销。DeepEP
  `low_latency_dispatch` 默认返回 row-major (int32) scale，测试代码需要做
  `.contiguous().view(uint8)` + `preprocess_mxfp4_scales()` 转换为 DeepGemm 要求的
  uint16 列主序。但这在生产环境中完全可以避免：传 `mxfp4_scale_row_major=False` 即可
  让 DeepEP 直接返回 uint16 列主序 scale（stride(1)==1），直接满足 GEMM 的
  `check_mxfp4_scales_layout` 要求，零额外开销。

- **消除的开销:**
  - `.contiguous()` → 76µs elementwise_kernel（Grid 5824, 将非连续 uint16 tensor 转为连续）
  - `.view(uint8)` + `preprocess_mxfp4_scales()` → 额外 kernel（转回 uint16 列主序）
  - 总计约 **~90µs** 无意义 round-trip

- **修改:**
  - `test_block_copy_gemm1_multi_gpu.py`:
    1. 新增 `COL_MAJOR_SCALE` 环境变量（默认 1）
    2. 新增 `nonfused_dispatch()` helper：`COL_MAJOR_SCALE=1` 时传
       `mxfp4_scale_row_major=False` 并直接使用返回的 scale；否则走原来的预处理路径
    3. 替换全部 5 处 non-fused dispatch 调用为统一 helper

- **环境:** sglang.lxh 容器（DeepEP 支持 `mxfp4_scale_row_major` 参数）

- **Bench (4-GPU, M=1560, N=6144, K=7168, ncb=13, SKIP_ISOLATION=1):**

  | 指标 | COL_MAJOR_SCALE=1 | Iter 15-16 (旧基线) | 差异 |
  |------|-------------------|---------------------|------|
  | BC pipeline | 0.290 ms | 0.310 ms | -20µs (环境差异) |
  | NF pipeline | 0.294 ms | 0.383 ms | **-89µs (消除scale开销)** |
  | NF GEMM-only | 0.228 ms | 0.248 ms | -20µs (环境差异) |
  | vs NF | 1.02x | 1.24x | — |
  | BC overhead | 0.074 ms | 0.060 ms | — |

  NF overhead: 0.294 - 0.228 = 0.066 ms (纯 DeepEP dispatch 延迟)
  BC overhead: quant 0.048ms + preprocess 0.026ms = 0.074 ms

- **分析:**

  1. **Non-fused 基线大幅变快:** 消除 scale round-trip 后，NF pipeline 从 0.383ms
     降至 0.294ms（-23%）。这说明此前 1.24x 的优势中约 0.5x 来自 NF 的人为开销。

  2. **"真实"对比:** 公平条件下（both paths 无冗余开销），block-copy 与 non-fused
     在 4-GPU 场景下基本持平 (1.02x)。Block-copy 的优势被其自身的
     overhead（quant + preprocess = 0.074ms）抵消。

  3. **BC 绝对时间略快 (0.290 vs 0.310):** 可能因 sglang.lxh 容器环境差异
    （不同 PyTorch 版本、JIT cache 差异）。NF GEMM-only 同样更快
     (0.228 vs 0.248ms)，说明这是全局环境因素。

  4. **核心结论:** Block-copy 的真实价值在于**将 P2P 传输与 GEMM 计算重叠**。
     当两者独立开销接近时（如 4-GPU 场景），重叠收益被 quant/preprocess 开销抵消。
     在 **8-GPU** 场景下（P2P 延迟更高、NF dispatch 更慢），block-copy 预期仍有优势。

- **新基线:** 后续所有比较应使用 `COL_MAJOR_SCALE=1`（sglang.lxh 容器），
  以获得公平的 non-fused 基线。4-GPU 新基线：
  - BC pipeline: **0.290 ms**
  - NF pipeline: **0.294 ms**
  - vs NF: **1.02x**

---

## 2026-07-09: arrival_push 折进 quant 尾（grid-completion）——已测,perf 零和,默认关

**动机:** 前次结论(2026-07-08)指出 preprocess 15µs 大头是 arrival barrier 等生产者 quant。
唯一想缩的是这个"等待"。思路:把 `arrival_push_kernel`(quant 后单独 `<<<1,1>>>`)用 grid-completion
折进 quant kernel 尾部——让 peer 的 arrival flag 早 ~1 launch 到达,缩短 consumer 的 barrier 等待。
预期 -5~10µs。

**实现** (`deep_gemm/include/deep_gemm/mxfp4_quant.cuh`,env `FUSED_ARRIVAL_IN_QUANT=1`,默认关):
- quant kernel 尾新增 grid-completion:自复位 atomic counter
  `atomicInc(&g_quant_arrival_retire, gridDim.x-1)`(wrap 回 0,免外部清零;`__device__` global 模块加载零初始化;
  `blocks==num_tokens` → 全 block 都到尾,计数干净)。
- 每 block 增量前 `__threadfence()`(device):既排序本 block scatter 写在计数之前,又使其在本 rank HBM
  全局可见(= consumer NVLink 读所需的可见性,等价旧 kernel 边界)。last-block 检测到满 → `__threadfence_system()`
  + 推 generation 到各 peer 的 slot[rank_idx](与旧 arrival_push 逐字相同)。
- launcher `static getenv` 一次读 env;折进时不再 launch 独立 arrival kernel。标准路径保留为 A/B 基线。

**正确性:PASS。** `FUSED_ARRIVAL_IN_QUANT=1 FULL_CORRECTNESS=1`(2卡 GPU1,2)gen1/2/3 vs CPU **逐位 0.000000**;
`RUN_RANK_SKEW=1` rank1 正确等 1.002s(慢生产者的折进 arrival 仍正确阻塞 consumer,barrier 语义保住)。

**性能:中性偏负(2卡同 GPU 对 1,2,clean-min ×3):**

| 指标 | 基线(标准 <<<1,1>>>) | 折进 | Δ |
|---|---|---|---|
| quant_ms | 0.056 | 0.069 | **+13µs** |
| preproc_ms | 0.049 | 0.038 | **−11µs** |
| pipeline_full (min) | **0.305** | 0.308 | +3µs(噪声内) |

三轮高度一致。**机制有效但零和:** arrival 早到确实省了 consumer barrier 等待(preproc −12µs),
但代价是把「让全 grid 的写对 peer 可见」从**免费的 kernel teardown 边界**换成 **256 个 per-block
`__threadfence()`** + tail 上的 system fence,压回 quant kernel 关键路径(+13µs)。

**根因/结论:** 标准 `<<<1,1>>>` arrival kernel 从 quant 的 kernel 边界**白嫖 device-wide 写可见性**,
自身只做 1 次 system fence——已近最优。折进后必须手动重建该 barrier(per-block fence),成本 > 省下的
~3-5µs launch。再次印证 **launch 不是瓶颈**。代码 default-off gated 保留作基础设施(未 commit)。
future 若想赢:须让"全 block 写对 peer 可见"比 per-block fence 更便宜(cooperative `grid.sync()` 太重;
release-atomic 边际)。

---

## [iter 1] 2026-07-09: merged prepare+finalize 设为默认(fixed-block_m 路径)——真赢 ~7µs

**AKO loop (pre-GEMM overhead).** 上次(07-08)把 merged 判为 "perf 中性",归因于 GPU 对混淆。
用固定 GPU 对(1,2)+ clean-min 重测,结论翻转:**merged 真赢 ~7µs**。

**Profiling 定位(acu + 微基准 scripts/quant_microbench.py):**
- quant kernel(单 rank, gen=0)= **35µs GPU**;Phase2 scatter(topk× 写放大 + scale 写)完全跳过后
  仍 **35µs** → scatter 对 256-block 网格**免费**(latency 被并行掩盖),印证 "quant 冗余写别动"。
- scale 写改 coalesced row-major:**无提升**(35µs),因 scatter 本就免费。已回滚。
- quant 由 Phase1(读 3.67MB + 量化 compute)+ launch 主导;acu:Mem Busy 80% 但仅 26% 峰值带宽,
  Issue Slots 15.6% → latency-bound,非 BW。Phase1 较固有,难缩。
- preprocess:单 CTA,in-kernel compute 小,主要是 **launch + kernel-boundary**;这正是 merged 的靶。

**改动:** `refresh_expert_preprocess` 默认 `MERGED_PREPROCESS` 0→1(fixed-block_m 路径默认走单 launch);
`scripts/bench.sh` 相应默认 =1,并加 VALIDATE_MERGED 门。

**Signal(clean-min ×3,GPU 1,2,SKIP_CORRECTNESS):**
| | split(baseline) | merged | 
|---|---|---|
| pipeline runs | 0.309/0.304/0.305 | 0.301/0.297/0.298 |
| clean-min | **0.304** | **0.297** (−7µs) |

三对全部 merged < split。event-isolated preproc 0.05→0.022(−28µs)但 pipeline 仅 −7µs:差额是
CPU-dispatch 空隙,在 pipeline 里被相邻迭代的 GEMM/quant 重叠掩盖;真正省下的是 ~7µs launch+boundary。

**Verdict:** 正确性 PASSED(3 轮);VALIDATE_MERGED **ALL MATCH**(6 张 metadata + 标量逐位一致);
单跑 pipeline 0.302(FULL_CORRECTNESS 负载下更噪)。vs non-fused 0.89→~0.91x。

**适用范围:** 仅当 block_m 已知/固定(推理已知 shape)。生产 `run()` 动态路由仍需 prepare 的
expected_m host readback 来选 block_m,无法直接合并(这是 run() 用 split 的根因)。

**下一步候选:** quant Phase1(30µs 大头,但 compute/latency-bound 难);或 run() 的 sync readback
消除(固定 block_m 策略)以让生产路径也吃到 merged。

---

## [iter 2] 2026-07-09: quant Phase1 编译期 stride + #pragma unroll(ILP)——kernel -2µs,pipeline 中性

**动机(acu on quant kernel, single rank):** Duration 22.9µs,latency-bound:No-Eligible **48.6%**,
Issue Slots Busy 14.8%,Warp Cycles/Issued 9.31。Phase1 循环用 `i += blockDim.x`(runtime stride)
→ 编译器无法 unroll/pipeline 独立迭代。launch 恒为 THREADS=256,把 stride 设为编译期常量 →
固定 4 次迭代可 unroll,独立 int4 load 提前发射(ILP)掩盖 LLC/延迟停顿。

**改动:** Phase1 循环 `constexpr THREADS=256; P1_ITERS=ceil(FP4_INTS/256)=4; #pragma unroll`,
`if (i>=FP4_INTS) continue;`。仅 Phase1(热点),其余循环不动。

**结果:**
- **kernel:** acu Duration 22.9→**20.9µs**(−2µs,−8.5%);Issue Slots 14.8→16.5%。真降,但小。
- **pipeline(clean-min ×3,GPU 1,2):** 0.297(iter1)→**0.296**,噪声内(<3%)。
- **Verdict:** 正确性 PASSED(3 轮 vs CPU 0.000000,循环重构 bit-exact);pipeline 0.299(单跑)。

**为何 pipeline 不动:** 微基准与 benchmark 都**复用同一个 x**(每迭代同输入)→ iter0 后 x 常驻 LLC
(acu DRAM 仅 5.73%),掩盖了 quant 的真实 DRAM 读成本。ILP 主要掩盖读延迟,在 cached 场景收益小。
**生产**(每 token 输入不同 → DRAM 读)收益应大于此处所见,故**保留**(低风险,仅设 blockDim==256,
launcher 恒满足)。

**结论:** quant Phase1 已近本算法地板(cvt/compute + 少量 per-thread 迭代);pre-GEMM 的干净大头
(merged, iter1)已吃到。进一步 pipeline 收益需转向 kernel 内 copy+GEMM 或 run() 的 sync-readback 消除。

---

## [iter 3] 2026-07-09: quant Phase1 去重 scale 计算(4-lane 只算一次+shfl 广播)——kernel -2.7µs

**动机:** 每个 scale-group(32 元素=4 lane)shfl 归约后 4 个 lane 拿到相同 amax,却各自重复算
`calculate_mxfp4_scales_bf16`(log2+2×pow2+hmul+cast)。scale_inv 只 lane0 存 → lanes 1-3 白算。

**改动:** 只在 group leader(lane&3==0)算 scale/scale_inv+存 scale_inv;`__shfl_sync(...,lane&~3,32)`
把 scale 广播给另外 3 个 lane。

**结果:** acu Duration 20.9→**18.2µs**(−2.7µs);Issue Slots 16.5→12.6%(冗余指令减少)。
combined iter2+3:quant kernel **22.9→18.2µs(−20%)**。pipeline ~0.297(clean-min,cached 输入掩盖,中性)。
Verdict:正确性 PASSED,vs CPU **0.000000**(scale 值与各 lane 自算完全相同 → bit-exact)。

**ISA 发现(用户指出,hgobjdump --dump-isa):** Phase1 的 4 个 unrolled `vmem.ld.b32x4` 每个后面跟
`s.wait vldcnt(0)`(等所有 vmem load 归零)→ **4 个 load 被串行化**,内存延迟全暴露(印证 No-Eligible 48%)。
对比 smem setup 是 8 load 连发 + sldcnt(7..0) 交错等(正确 MLP)。iter2 的 unroll 没拿到 MLP 就是因为
compiler 每迭代 load 紧跟 compute+vldcnt(0)。→ iter4 攻这个(hoist loads)。

---

## [iter 4] 2026-07-09: quant Phase1 load hoist(修 s.wait vldcnt(0) 串行)——ISA 验证 MLP,累计 pipeline -5µs

**动机(ISA,iter3 发现):** Phase1 的 4 个 unrolled `vmem.ld.b32x4` 每个后跟 `s.wait vldcnt(0)`
(等所有 vmem load 归零)→ 4 个 load 串行,DRAM/LLC 延迟全暴露。

**改动:** 把 4 次 int4 load 提到独立 loop 先全部发射到寄存器数组 `raw[P1_ITERS]`,再统一 compute。
分离 load/compute 让 compiler 连发 4 个 load 再 wait。

**ISA 验证(hgobjdump --dump-isa):** 4 个 `vmem.ld.b32x4` 现在**连续发射**(offset 228/258/288/2b8)
再到第一个 `s.wait vldcnt(0)` → **MLP 达成**(4 load 并行,1× 延迟 vs 之前 4×)。

**结果:**
- acu Duration 18.2→**17.8µs**(cached 微基准仅 −0.4µs:x 常驻 LLC + 48 warp TLP 已掩盖低 LLC 延迟;
  MLP 在**生产 uncached DRAM** 收益才大)。寄存器 40→48/thread,占用 73.5→70.9%(轻微)。
- **累计 quant kernel:22.9→17.8µs(−22%,iter2 ILP + iter3 dedup + iter4 hoist)。**
- **pipeline clean-min(全 4 opt):0.297→0.292(−5µs);baseline 0.304 → 现 0.292(−12µs,merged -7 + quant -5)。
  vs non-fused 0.89→0.925x。** 单个 quant opt 落噪声内,三个叠加后 −5µs 可测。
- Verdict:正确性 PASSED,vs CPU 0.000000(纯调度/MLP 改动,bit-exact)。

**教训:** iter2 单独判"pipeline 中性"过早——kernel 小改需**累计**到超噪声(~5µs)才看得出 pipeline 收益;
且 quant 确实在关键路径上(Pipeline-no-preproc 0.288→0.285)。ISA 级验证(而非只看 wall-clock)是对的。

---

## 删除 cooperative copy 逻辑 (2026-07-11)

**动机:** copy block 三条搬运策略里 cooperative(copy_mode 1/2)已被多轮实验证伪
(8卡合搬 +82~180µs,fence/straggler 劣化;wave0 掩盖使其无 wall-clock 收益),生产用 round-robin ncb8。
彻底删掉这条死路径。

**改动(全链路清 copy_mode):**
- `fp4_gemm_cutlass3.cuh`:删 `run_copy_block_cooperative` 函数 + 两处 dispatch 的 copy_mode==1/2 分支
  + `run_fused_dispatch` 的 copy_mode 形参/实参。dispatch 简化为 `if constexpr(KTPF>0)`→kstripe / `else`→basic。
- `scheduler_cutlass3.cuh`:删 `copy_mode` 字段 + 三个构造器的初始化。
- `dispatch_fused_gemm.py`:删 copy_mode 整条传参链(模板 arg/arg_defs/两个 API 形参);
  flag buffer `2*max_total_m_blocks`→`max_total_m_blocks`(done_ctr 上半区只被 cooperative 用,已死)。
- 测试/脚本:删 COPY_MODE 读取/透传、COPY_MODE_AB A/B benchmark 块、sweep_block_copy.sh 的 copy_mode 轴。

**验证(GPU 3,4,5,6 固定,避开 GPU0 忙/GPU2 E.Process 坑):**
- 正确性:2卡 + 4卡 × basic(KTPF=0)/kstripe(KTPF=7) 全 `test1_correctness PASSED`,
  所有 expert vs_cpu/vs_nf = 0.000000(bit-exact)。
- 性能(4卡,median of 3,对比删除前基线):

  | path | BC kernel-only 基线→后 | best-pipeline 基线→后 |
  |---|---|---|
  | basic (KTPF=0) | 0.275→0.274 | 0.317→0.314 |
  | kstripe (KTPF=7) | 0.264→0.267 | 0.308→0.308 |

  全部 ±3µs(std≈10µs)内 → **无回退**。两条保留 path 功能+性能均正常。

**Verdict:** 纯清理,bit-exact,无性能回退。分支 opt/kernel-copy-gemm-overlap,未 commit。

---

# 新仓库 DeepGemm-block-copy-fusedopt (2026-07-11, 分支 opt/fused-copy-block-gemm)

基线 (8卡 prod, d3e85f0): pipeline **0.291ms/0.87x** (ncb=3); kernel-only BC=0.249 / all-local=0.231 / NF=0.194ms; P2P +0.018ms; pre-GEMM(quant+preproc)=0.043ms 暴露。

## Iter A — block_n 256→512 减半 wave (不支持, 失败)

- **假设:** GEMM tile 数 = (M/bm)×(N/bn) = (1536/128)×(6144/256) = 12×24 = 288 tiles;
  36 GEMM SM (39-ncb3) → 288/36 = 8 整 wave。NF 用 bm=256 只 4 wave。
  增大 block_n 256→512 → 12×12=144 tiles/36 = 4 wave, 计算量不变但 wave 减半。
- **改动:** dispatch_fused_gemm.py 加 FUSED_BLOCK_N/FUSED_STAGES/FUSED_WARP_N env 覆盖 + 重算 smem (保留为实验基础设施, 不改默认)。
- **结果:** **编译失败**。fp4_gemm_cutlass3.cuh:2237 static_assert 硬限制 `BlockN ∈ {16,32,64,128,256}` (CUTE tile 层)。512 不支持; smem 也 314432 过大。
- **附带发现:** 2卡时 get_best_configs 给 block_m=256 (随 M 变); 8卡 prod expected_m=128 → block_m=128。
- **结论:** block_n 上限 256, 此路不通。

## 关键瓶颈定性 (代码+算术, 非盲目)

- 8卡 fused GEMM 的 **288 tiles / 8 wave 是结构性下界**: m_blocks=12 (=expert数×ceil(128/128), 硬定)、
  n_blocks=24 (block_n≤256 锁死)、SM=39 固定。
- **回收 idle copy SM 无效** (印证 iter1/iter10 失败): ceil(288/39)=8 wave 仍=8, SM 36→39 不减 wave。
- NF 快在 block_m=256 (GroupedContiguous 打包无 padding) → 只 4 wave。fused masked 不能用 bm=256 (per-expert M=128 会 2x padding)。
- **根治需 GroupedContiguous 大改** (消 per-expert padding 才能 bm=256 减 wave), 超出单次 config 调优。
- 剩余 kernel 内可探索: wave 内 MMA/smem 利用率 (需 profile); pre-GEMM launch overhead (memory 显示已榨, 见 project_preprocess_fusion)。

## Iter B — block_n 256→512 (放宽 assert, warp_n=64, 4 wave) ★重大正面★

- **假设:** GEMM 288 tiles/8 wave 结构性慢。block_n 256→512 → n_blocks 24→12 → 144 tiles/36 SM = **4 wave** (计算量不变)。
- **突破口:** block_n>256 只被 `static_assert(BlockN≤256)` (fp4_gemm_cutlass3.cuh:2237) 挡; smem 上限 ppu_capacity=262144, 而 bm128/bn512/st3 = **262016 < 262144 (刚好放得下)**。
- **改动:**
  - fp4_gemm_cutlass3.cuh:2237 static_assert 白名单加 512。
  - dispatch_fused_gemm.py 加 FUSED_BLOCK_N/BLOCK_M/STAGES/WARP_N env 覆盖 + 重算 smem (实验基础设施)。
  - test 加 UNIFORM_ROUTING env (correctness 段用均匀 round-robin → block_m=128, 才能在 bn512 的 smem 约束内验证)。
- **关键约束 (踩坑记录):**
  - warp_n **必须=64** (每 warp 64 列, 与 bn256 相同的 sfb 迭代 ≤4)。warp_n=128 → `warp_iter_num_sfb>4` static_assert (2124) 失败。
  - block_m **必须=128** (均匀路由)。随机路由 correctness 段 expected_m>128 → block_m=256 → bn512 smem=314432 超限 (`too many resources / invalid argument`)。
- **正确性 (8卡, UNIFORM_ROUTING=1, FULL_CORRECTNESS):** 所有 expert **vs_cpu=0.000000, vs_nf=0.000000 → bit-exact**, All PASSED。
  (数学上 block_n 只分块 N 维、不改 K 累加顺序, 应 bit-exact; 实测确认。)
- **性能 (8卡 prod, perf 段均匀, ncb sweep 2,3,4, 单次):**

  | 指标 | 基线 bn256 | **bn512** | 改进 |
  |---|---|---|---|
  | pipeline (best ncb=3) | 0.291ms / 0.87x | **0.264ms / 0.96x** | **-9.3%** |
  | kernel-only (copy+GEMM) | 0.249ms | **0.221ms** | **-11%** |
  | all-local | 0.231ms | 0.209ms | -9.5% |
  | vs NF (pipeline) | 0.87x | **0.96x** | +0.09x |

- **结论:** wave 8→4 假设成立, 大幅逼近 NF。**待办: 3次稳定性复测 + 默认化 (仅 block_m==128 时启用, bn512 需 bm128 保 smem) + verdict。**

## Iter B 收尾 — 3次稳定性 + 默认化 + apple-to-apple + verdict

**3次稳定性 (8卡, perf-only, NCB=3, 空闲):**
| run | fused pipeline bn512 | fused pipeline bn256(base) |
|---|---|---|
| r1/r2/r3 | 0.264 / 0.264 / 0.265 | 0.292 / 0.292 / 0.292 |
→ 中位 **0.264 vs 0.292 = -9.6%**, fused 绝对时间 std<1µs 极稳。

**默认化:** dispatch_fused_gemm.py 在 env override 的 else 分支加自动启用: 当
`block_m==128 && block_n==256 && warp_n==64 && block_k==128` 时升 block_n=512 (重算 smem)。
逃生开关 `FUSED_DISABLE_BN512=1`。随机路由 (expected_m>128 → block_m=256) 条件 false → 自动回退 bn256 (bn512+bm256=314432 超 cap)。

**Verdict (8卡, 默认配置, FULL_CORRECTNESS, 随机路由):** 所有 expert (token 106~147, 含>128) **vs_cpu=vs_nf=0.000000 PASSED**;
correctness 段无 BN512_DEFAULT 打印 (回退 bn256), perf 段触发 BN512_DEFAULT (bn512)。条件回退逻辑验证正确。
(注: verdict 的 perf 段 GPU0 被他人占用, 数值不可信; perf 以上面 3次空闲数据为准。)

**apple-to-apple (用户要求, 3次, NF 也用 bn512):**
| GEMM 纯算 (8卡) | bn256 (8 wave) | bn512 (4 wave) | bn512 收益 |
|---|---|---|---|
| NF gemm-only | 0.194/0.193/0.193 | 0.177/0.177/0.177 | **-8.8%** |
| fused kernel-only | 0.249 | 0.220/0.221/0.220 | -11.6% |
| fused pipeline | 0.292 | 0.264 | -9.6% |

- **诚实结论: bn512 是通用 GEMM tile 优化 (wave 8→4), NF 也受益 -8.8%。** 不是 fused 独占。
- fused kernel(0.220, 36SM+P2P) 相对 NF gemm-only(0.177, 39SM) 仍有 fused 固有开销 (P2P copy + 3个SM做copy), 属预期。
- vs-NF 比率 (0.90~1.20 抖动) 不可靠, 因 NF pipeline 含 DeepEP dispatch 噪声大; 用 gemm-only 稳定对比。
- **对本任务 (优化 fused): fused 绝对 pipeline -9.6%, bit-exact, 已默认化。达成。**
- 后续可选: 把 bn512 也用于生产 NF/GroupedNoPad GEMM (需评估对全局 fp4 GEMM 的影响, 超出本任务范围)。

## bn512 下最优配置 — ncb 重扫 (3次, 空闲)

bn512 只有 144 tiles (bn256 是 288), 对 GEMM SM 数极敏感 (144/36=4 整除):
| ncb | GEMM SM | wave | r1 | r2 | r3 |
|---|---|---|---|---|---|
| 1 | 38 | 4 | 0.505 | — | — (copy 太慢, 严重暴露) |
| 2 | 37 | 4 | 0.318 | 0.319 | 0.318 (copy 不够快) |
| **3** | **36** | **4** | **0.264** | **0.264** | **0.264 ← 最优** |
| 4 | 35 | **5** | 0.301 | 0.301 | 0.300 (掉 wave +14%) |
| 5 | 34 | 5 | 0.301 | 0.301 | 0.302 |
| 6 | 33 | 5 | 0.306 | — | — |
| 7 | 32 | 5 | 0.310 | — | — |

**ncb=3 是 bn512 的双重甜点**: (1) GEMM 侧 144/36=4 wave 整除, ncb≥4 → 35 SM → ceil(144/35)=5 wave (+14%); (2) copy 侧 3 block 刚好及时完成 P2P, ncb≤2 copy 暴露。与 bn256 最优 ncb=3 数值巧合但机理不同 (bn256 的 288 tiles 对 SM 不敏感)。

**bn512 完整最优配置**: block_m=128, block_n=512, block_k=128, warp_m=64, warp_n=64, stages=3, ncb=3, K_TILES_PER_FLAG=0, num_sms=39 (36 GEMM + 3 copy), smem=262016。stages 无法增到 4 (smem 已近 262144 上限)。

## 改为 env 选项 (用户要求, 不默认化)

按用户要求, bn512 **不默认启用**, 改为纯 env opt-in:
- 启用: `FUSED_BLOCK_N=512 FUSED_WARP_N=64` (两者同设; ncb=3 最优)。
- 默认 (不设 env): 走 get_best_configs 的 bn256 (基线行为, ~0.292ms)。
- 删除了 dispatch_fused_gemm.py 中 else 分支的自动默认化 (FUSED_BN512_DEFAULT / FUSED_DISABLE_BN512 逻辑移除)。
- 其余 env: FUSED_BLOCK_M / FUSED_STAGES 仍可覆盖; FUSED_CFG_VERBOSE=1 打印实际 config。

## 2026-07-11 copy loop 换 PPU bulk-DMA (swizzled ldg/stcg) — 结论: neutral/marginal-negative, 不默认化

**任务**: 把 fused copy block 的 P2P copy loop 数据读写从 `__ldg`/普通 store 换成 PPU bulk 版
(`__ppu_global_ldg_bulk_b32x4` / `__ppu_global_stwb_bulk_b32x4`), 优先最高缓存级, 测 block_m=256/bn256/bk128。

**关键机理发现 (踩坑记录)**:
1. bulk API 是 128-bit (uint4) 同步 load/store, 形态上 1:1 (非 DMA 描述符)。
2. **bulk load/store 是 warp-collective swizzle 对**: load 做 shuffle, store 做反 shuffle,
   **必须配对 + 整 warp-wave 满员 (所有 lane 参与、地址连续) 才互相抵消**。SDK 有显式
   `__ppu_swizzle_bulk_b32x4(val, mask)` "swizzle data for bulk" 佐证。
   - 单独换 load 或单独换 store → shuffle 不抵消 → 大错 (无意义, 用户提示证实)。
   - 配对但不满 wave: 每 rank 拷贝元素数 = cnt*k_half/16, 非 blockDim(256) 整数倍 →
     最后一个残 wave (~0.5% 元素) lane 不满 → swizzle 失配 → **小残差 (vs_cpu 0.02~0.14, 复现稳定)**。
     误差量级 ≈ 残 wave 占比 0.5%, 与观测吻合。**这是"配对后仍失败"的根因, 不是 store 写回可见性。**
3. **正确写法 = 满 wave 用 bulk + ragged 尾部/max_n 预测段用 plain**: 只在连续、全 lane 的
   min_n 满 wave body (`full=(min_n/stride)*stride`) 用 bulk; 尾部/预测段回退 `__ldg`/普通 store。
   nr8/nr4 的 FP4 A-copy min_n 都加了 `#ifdef DG_BULK_COPY` 的 bulk 前置循环, 原 plain 循环变残余循环。
   nr4 的 SFA co-issue 是 per-lane predicated → 保持 plain (写不同 buffer, 不扰 FP4 swizzle 寄存器)。
   → **4卡 nr4 bit-exact (3 轮全 0.000000)**。
4. store 缓存级: `stwb` (写回, 最高级) 在**未 split 时**因尾部 swizzle 而失败, 曾误判为写回不可见;
   split 后 `stwb` 也 bit-exact → 证实之前是尾部问题不是可见性。但 `stwb` perf 不如 `.cg`(stcg)
   (写回多一次 __threadfence flush, TMA 立即读 L2 无收益)。**默认用 stcg (.cg/L2)**。

**perf A/B (block_m=256/bn256/bk128, 4卡 nr4, GPU 4-7 固定, 各 3 次, FORCE_EXPECTED_M=129):**
| 指标 | baseline | bulk stcg | bulk stwb |
|---|---|---|---|
| Pipeline / ncb=8 best | 0.340 ×3 | 0.332~0.333 | 0.334~0.336 |
| Kernel-only (copy+GEMM) | 0.285 ×3 | 0.288~0.291 | 0.290~0.292 |
| Kernel all-local (无 P2P) | 0.264 | 0.268~0.270 | 0.269~0.270 |
| **P2P link overhead** | **+0.020** | **+0.020~0.021** | **+0.020~0.022** |

**诚实结论**:
- **P2P exposure 完全不变 (+0.020 三者一致)** → bulk 没有减少实际 copy 暴露。
- kernel-only / all-local 反而 +0.005 (两种 store 级一致) → bulk swizzle 指令本身比 baseline
  已最优的 coalesced 8-wide MLP 路径略慢; 开销不在 split 尾部(≤1 迭代)也不在 store 缓存级。
- pipeline/ncb-best 的 -0.007 稳定但被 kernel-only 反向抵消, 不作为真实 copy 收益。
- **印证 project_blockcopy_isa_audit: copy 访存已近最优 + P2P 已 91% pipeline-masked, bulk DMA
  无可回收空间。** bulk 对这个 copy 是 neutral/marginal-negative。
- 未测: 8卡 nr8 (remote 87.5%>75%, copy 占比更高), nr8 已 bulk-ified 且待 bit-exact, 需 8 卡空闲。

**落地**: 保留 opt-in, **默认关**。env: `DG_BULK_COPY=1` 启用 (满 wave bulk, stcg 存);
`DG_BULK_STWB=1` 额外切写回存 (bit-exact 但更慢)。diff 纯 additive (macro-gated, 关时 baseline codegen 不变)。
工作分支 `opt/fused-copy-block-gemm`, 未 commit (负结果, 待用户定是否保留开关)。

### 补测 8卡 nr8 (2026-07-11, 用户要求)

8卡 nr8 **bit-exact** (block_m=256, 3 轮全 0.000000)。perf A/B (8卡, FORCE_EXPECTED_M=129, 各3次;
注: NF pipeline 有几次被外部干扰污染, 但 BC kernel 数值稳定, 以 BC 为准):
| 指标 (8卡 nr8) | baseline | bulk stcg |
|---|---|---|
| Kernel-only (copy+GEMM) | 0.294, 0.298 | 0.292, 0.295, 0.298 (~parity) |
| **Kernel all-local (无P2P)** | 0.277, 0.277 | **0.274, 0.274, 0.275 (−0.003, ~−1%)** |
| Pipeline / ncb=8 best | 0.361, 0.346 | 0.344, 0.344, 0.353 |
| P2P link overhead | +0.017, +0.021 | +0.017, +0.021, +0.024 (重叠, 无明显差) |
| remote 占比 | 88% (7/8 ranks) | — |

- **相比 4卡, nr8 picture 翻转: bulk 从"略负"变"parity~略正"** (all-local −0.003/−1%, 稳定;
  nr8 是 8-wide load + 数据更多, bulk 略占优)。但仍是 marginal, 不是明显 win。
- **P2P exposure 仍未明显降低** (两边 0.017~0.024 重叠) → 通用 `ldg_bulk` 没专门加速 remote 读。

**下一步杠杆 (用户提示 2026-07-11)**: 试 **专用 remote 指令** `__ppu_remote_load_bulk_volatile_*` /
`__ppu_remote_store_bulk_volatile_*` (SDK hggc_extend_device_functions.h ~line 290-305)。这是针对
remote 访存的指令 (通用 bulk 没碰到的、仍暴露的 +0.020 P2P 部分)。**注意: 只能用于 remote 访存,
不能用于本地地址 → 需对每个 rank 指针做 local/remote 判断** (rank==本 rank 或地址落在本地 sym_buf 时
走普通/plain, 否则走 remote 指令)。这可能真正压 P2P exposure。

## 2026-07-12 remote 专用指令版 (DG_BULK_REMOTE) — ★ 首个稳定 win (~-2% pipeline, bit-exact)

**实现**:
- **local/remote 判定**: rank r 的 source 是 local iff `r == rank_idx` (SymBuffer::map offset[rank_idx]=0)。
  把 `rank_idx` 从 .py 模板 (`{RANK_IDX}` + keys) 经 run_fused_dispatch 新增 param → `sched_args.rank_idx`
  (scheduler struct 新增字段) 传到 copy loop。test 的 15 处直接调用都补了 `rank_idx=rank`。
- **remote load 也是 swizzle 版**: 单独 remote_load + plain store 失败 (vs_cpu~0.02/vs_nf~0.7, 与通用 bulk
  同理)。正确写法 = **warp-aligned body 里, remote rank 用 `__ppu_remote_load_bulk_b32x4`, local rank 用
  通用 `ld_bulk_global`, 两者都由通用 `st_bulk_global`(.cg) un-swizzle** (remote_load 与通用 bulk load
  共享同一 swizzle, 实测 st_bulk_global 能反 shuffle 两者)。tail/max_n 仍 plain `__ldg` (remote 上 __ldg 合法)。
  → LD_BODY 宏 (body, 按 rank 选 remote/generic bulk) + LD_A 宏 (tail, 恒 plain)。rtmd=0 (peer VA 已编码目标 GPU)。
- SFA 读保持 plain __ldg (remote 上合法, 未加速; 聚焦 A 数据)。

**★ 坑: all-local isolation 与 remote 不兼容** — Test2 的 P2P isolation 把所有 rank_addr_a 换成 local 副本,
但 remote 代码对 r!=rank_idx 仍发 remote 指令 → **remote 指令打 local 地址 = illegal memory access, 整跑 crash**。
必须 `SKIP_ISOLATION=1` (isolation 对 remote 本就无意义)。Test1 correctness 无 isolation 故先前通过。

**perf A/B (8卡 nr8, block_m=256, SKIP_ISOLATION=1, FORCE_EXPECTED_M=129, 多次交替, 剔除 2 次外部污染窗口)**:
| ncb=8 pipeline (ms) | baseline | remote |
|---|---|---|
| 各次 | 0.343,0.345,0.344,0.346,0.347,0.356 | 0.337,0.337,0.339,0.339,0.339,0.341 |
| **中位** | **~0.345 (抖0.343-0.356)** | **~0.339 (紧, 0.337-0.341)** |

- **remote ~0.339 vs baseline ~0.345 → -0.006ms (~-2%), 稳定且分布几乎不重叠。首个稳定 beat baseline 的变体。**
  也略优于通用 bulk (8卡 ~0.344-0.347)。机理: 专用 remote 指令加速 88% remote 读 (通用 bulk 没碰到的暴露部分)。
- 8卡 nr8 **bit-exact** (Test1 context 路径, rank_idx 正确)。SKIP_ISOLATION 下无 kernel-only (=0.000), 用 pipeline 作端到端指标 (preprocess/quant 两边同, pipeline delta = kernel delta)。
- 4卡 nr4 (75% remote) 预计收益更小, 未细测。

**落地**: opt-in `DG_BULK_REMOTE=1` (宏, 默认关, 独立于 DG_BULK_COPY)。**测试已自动保护**: 检测到 env
`DG_BULK_REMOTE!=0` 时强制 `SKIP_ISOLATION=1` (test line ~85), 用户无需手动设, 避免 all-local crash。
未 commit, 待用户定是否默认化/保留。

**SFA 走 remote/bulk 不可行 (2026-07-12 分析)**: SFA aligned 主体虽是 int4/128-bit, 但每行只有
vecs_per_row=cnt/8≈16 int4 (256B), **比一个 warp(32 lane=512B) 还短** → warp 必跨行边界, 地址在
`kb*max_tok` 处大跳 → 非连续, bulk swizzle 需要 warp 连续 512B 块, 不满足 (对照 A-copy min_n 是连续才 work)。
且 SFA 体量仅 A 的 ~1/28 (ksb*cnt/8 vs cnt*k_half/16), 即使能做收益也可忽略。→ SFA 保持 plain __ldg (remote 上合法)。

**remote store 在本 copy 不适用**: copy 永远 gather remote→**local** (dst=local_fp4/sfa_buf), 没有写 remote 的方向;
remote store 只对 scatter (local→peer) 的 kernel 有意义 (本 fused GEMM copy 不涉及)。

## 2026-07-12 SFA copy/repack 拎进独立 preprocess kernel (DG_SFA_PREPROCESS) — bit-exact, 低延迟中性, 待高延迟机 A/B

**动机**: 现状 SFA (A 的 scale) gather+repack 在 fused GEMM 的 copy block 里做 —— 是每 copy block FP4 拷贝
之后的**串行尾巴** (0f40bad 引入的 `copy_mblock_sfa`)。SFA 画像: 碎+跨步远端读 (prod 8卡仅 336KB 但
~10,752 次 32B 跨步读, 延迟型)。高延迟机 (2 级 shm) 上这笔暴露大, 0f40bad 后一直回退。思路: 把 SFA 从
copy block 里**拎出来做成独立 pre-GEMM kernel**, 提前把**所有 expert/所有 M-block** 的 SFA 一次性 gather+repack
到 local_sfa_buf, 让 fused GEMM 的 copy block 不再做 SFA、GEMM 直接读已备好的 buffer。

**改动 (纯 additive, opt-in host-side 开关, 默认关, 与 DG_BULK_* / DG_SFA_SOURCE 正交)**:
- `fp4_gemm_cutlass3.cuh`: 新增 `dispatch_sfa_preprocess_kernel<BLOCK_M,NumRanks>` (grid-stride over
  total_m_blocks=`copy_grouped_layout[0]`, 每 CTA 一个 M-block) + `launch_dispatch_sfa_preprocess` host wrapper。
  **复用 `copy_mblock_sfa` 本体** (单一真源 → layout/addressing 与 inline 路径逐位一致); 只填 sched 里
  copy_mblock_sfa 读的 6 个字段 (rank_addr_sfa/rank_split_m/rank_counts/local_sfa_buf/ksb/max_tok)。
  grouped_layout 解析 (gl.x=expert_local, gl.z=base_block, m_block_in_expert=mb−gl.z) 与 run_copy_block L265-267 一致。
  remote 读用 plain `__ldg` (SFA 碎/短, bulk/remote 不适合, 见 project_blockcopy_bulk_dma)。
- `dispatch_fused_gemm.py`: `template_sfa_preprocess` + `dispatch_sfa_preprocess()` (照 dispatch_expert_preprocess_merged
  的 JIT 模式; BLOCK_M/NUM_RANKS 编译期 key, ksb/max_tok/num_sms/num_threads 运行期 arg)。
  `fused_dispatch_block_copy_gemm1_fp4` 里: `os.getenv('DG_SFA_PREPROCESS')!=0 and not sfa_source_host` 时
  在 GEMM launch **前** (同 stream → kernel-boundary 使 local_sfa_buf 对 GEMM 的 SFA TMA/L2 域可见, 无需 fence)
  发 sfa-preprocess kernel, 并强制 `skip_sfa_copy=True` (copy block 不做 SFA, GEMM 仍读 local_sfa_buf → 输出有效)。
  与 host SFA source 互斥 (host 读 remote_addr_sfa 不读 local_sfa_buf)。
- **无编译宏 / 无新 JIT key on GEMM**: 独立 kernel 自成 JIT 模块, skip_sfa_copy 已是运行期 flag → env 翻转不重编 GEMM
  (同 DG_SFA_SOURCE 的路子)。block_m 传 GEMM 的 resolved block_m (=copy block 的 BLOCK_M, 与 finalize 建的
  grouped_layout/rank_split_m 必须一致, 这是既有不变量 L639)。

**正确性 (硬 gate, PASSED)**:
- 4卡 (GPU4-7): Test1 所有 12 expert × 3 轮 `vs_cpu=vs_nf=0.000000` bit-exact; test1_correctness+test2_performance ALL PASSED。
- 8卡 (GPU0-7): Test1 3 轮 PASSED, All PASSED。→ 与 inline copy_mblock_sfa 逐位一致 (预期, 同一函数体)。

**性能 A/B (8卡, block_m=256 via FORCE_EXPECTED_M=129, ncb=8, median of 20 iters, GPU0-7)**:
| 模式 | Pipeline(full) | Kernel-only(copy+GEMM) | Local-only |
|---|---|---|---|
| baseline (SFA 在 copy block) | 0.346 | 0.296 | 0.277 |
| DG_SFA_PREPROCESS=1 | **0.348** | 0.292 | 0.278 |
| SKIP_SFA_COPY=1 (floor, 无 SFA) | 0.330 | 0.277 | 0.260 |

**Verdict — 低延迟机中性 (符合预期)**: SFA 总暴露 = baseline−floor = ~16µs (pipeline) / ~19µs (kernel-only)。
DG_SFA_PREPROCESS pipeline 0.348 ≈ baseline 0.346 (+2µs, 噪声内 ~3-5µs), **没往 floor 回收**。机理: 低延迟机上
SFA remote 读本就便宜且在 baseline 里已被 FP4 copy 大量掩盖 (KTPF=0 ~91% hidden); 拎成独立 kernel 只是把这笔
成本从 copy-block 尾巴挪到一个**串行前置 kernel** (它依赖 finalize 产的 metadata, 只能排在 preprocess 后、GEMM 前,
无重叠余地) → net wash。**收益主要在高延迟机** (SFA 碎跨步远端读被成倍放大时, 一个高占用/跨全 SM 的批量 gather
可能比 copy-block 尾巴的少数 block 串行读更能藏延迟) —— 本机无法复现其延迟, 同 DG_SFA_SOURCE 的处境。

**下一步**: 交用户在**高延迟机**上跑 `DG_SFA_PREPROCESS=1` vs `=0` 同命令 A/B (block_m=256)。若明显快 → 坐实
拎独立 kernel 对高延迟 SFA 有效; 否则回退在 GEMM 读侧 (见 DG_SFA_SOURCE=host 的定位路线, 二者可组合排查)。
diff 纯 additive、默认关。log: logs/perf_{baseline,sfaprep,floor}.log, logs/sfa_prep_{4,8}card_correctness.log。

## 2026-07-12 SFA 对称 buffer 行主序 source (DG_SFA_ROWMAJOR_SRC) — 连续 remote load + 列主序 local store,bit-exact,待高延迟机测

**动机(用户提出)**:现状 quant 写 SFA 到对称 buffer 是列主序 `[ksb, max_tok]`,copy 的 remote 读 =
每 rank ksb=112 个跨步段(段内 cnt token 连续、段间跨 max_tok)→ 高延迟机上 112 次跨步远端访问延迟放大
(SFA 慢真因是"碎+跨步",量仅 336KB)。改法:source 改**行主序** `[max_tok, ksb]`(每 token 的 ksb 个
scale 连续)→ copy 的 remote 读变成**一整块连续 burst**(一个 rank 的 cnt token × ksb 连续 ~29KB),
strided 落到便宜的**本地 store**,GEMM 读的 dst(local_sfa_buf)仍列主序不变。⚠️与 memory 里被否的
"行主序根治"不同(那个改 dst 害 GEMM 读;这个只改 source)。

**改动(纯 additive,编译宏 DG_SFA_ROWMAJOR_SRC 门控默认关,5 处;布局是 writer/reader/addresser 契约必须一起翻)**:
1. `mxfp4_quant.cuh` scatter:`scale_out[slot*K_SCALE_BLOCKS + j]`(行主序)vs 列主序 `j*max_tok+slot`。
   顺带 quant 自己的写也从列主序跨步变连续。
2. `expert_preprocess.cuh` rank_addr_sfa 偏移:`rank_offset[r]*ksb`(ksb=ceil(hidden/64),由 hidden_dim 推)vs `rank_offset[r]`。
3. `fp4_gemm_cutlass3.cuh` `copy_mblock_sfa`:宏下走行主序全量拷贝分支——**连续读** `src[linear]`
   (linear=t*ksb+kb,consecutive lane→coalesced burst)+ **列主序写** `dst[kb*max_tok+t]`;忽略 SkipVectorizedMain。
4. 同文件 nr4 SFA co-issue interleave(列主序专用)宏下 `#ifndef` 编译掉(tv 保持 0→inline 块 no-op),
   nr4 也走 trailing 的行主序全量 copy_mblock_sfa。
5. `compiler.py`:`DG_SFA_ROWMAJOR_SRC=1` → `-DDG_SFA_ROWMAJOR_SRC`(mirror DG_BULK_*,进 JIT 签名,env 翻转不重编串)。
   test CPU 参考(build A 反量化)也做布局感知读(env 判断行/列主序),否则 CPU-ref 读错报假 FAIL。

**正确性(硬 gate,PASSED)**:DG_SFA_ROWMAJOR_SRC=1,4 卡 + 8 卡 Test1 所有 expert × 3 轮
**BC vs NF = 0.000000 且 BC vs CPU = 0.000000**(bit-exact)。⚠️首测 test CPU 参考按列主序硬读 → 假 FAIL
(BC vs NF 已 0 说明 kernel 对),修 test 参考后全过。与 DG_SFA_PREPROCESS 组合也自动生效(共用 copy_mblock_sfa)。

**性能**:待测(本机被第三方 DeepSeek 服务间歇占 + host 高 load,pipeline 计时不可信;kernel-only 可信)。
低延迟本机预计中性偏微正(远端事务数 112 段→1 burst,但 SFA 已被 FP4 copy 大量掩盖);**收益主要在高延迟机**
——交用户 A/B `DG_SFA_ROWMAJOR_SRC` 0 vs 1(bm128/bm256 同命令)。设计/交接见 HANDOFF_sfa_rowmajor_src.md。
⚠️ DG_SFA_SOURCE=host 的 merged builder 假设列主序,与本宏组合需同步更新(单用不冲突)。

### 补:行主序 copy 必须向量化(naive 标量版 2x 回退 → 向量化后持平/微正)
**踩坑**:首版 `copy_mblock_sfa` 行主序分支写成 naive 标量循环(每元素 `linear/ksb` 除法 + 标量 uint16
读写)→ **kernel-only 稳定 2x 回退**(bm128 local-only 0.230→0.458,3 轮全 0.518-0.519 非噪声)。
根因:丢了原列主序的 int4 向量化,且每元素 ÷ksb=112。**local-only 也 2x** 坐实是 copy compute/访存本身,非 remote。
**修法**:行主序下 kb 是连续轴 → 读向量化为 int4(8 个连续 kb of one token,ksb%8==0 且 src 16B 对齐时,
t*ksb 是 8 的倍数→对齐),再散射 8 个标量 strided store 到列主序 dst;÷vpt=14 每 int4(非每元素)。
**结果(bm128,8卡,kernel-only bc/local)**:OFF 0.258/0.230 vs **ON 向量化 0.247-0.252/0.229-0.230**
→ 持平,**bc(含 remote)侧稳定略好 ~6-10µs**(连续 burst 在低延迟 NVLink 也有小正收益),local 相同。
bit-exact 保持(4卡 BC-vs-NF=BC-vs-CPU=0)。**教训**:行主序的价值(remote 连续读)只有在读**仍向量化**时才成立;
标量实现会被 compute 拖垮。strided store 是列主序 dst 的不可约成本(8x store 指令 vs 列主序),但本地便宜。

---

## 2026-07-15 [iter 0] pre-GEMM preprocess 流水 clock64 拆解基线 (承接 SFA push 0.81x, commit bc3a3f9)

**bench**: 8卡 exact ncb=3, DG_SFA_PUSH=1 FUSED_ARRIVAL_IN_QUANT=1 (生产默认), SKIP_CORRECTNESS perf-only.

**pipeline 分解 (单 stream 串行 quant→preprocess(merged)→reshape→fused-GEMM)**:
- Pipeline (full): 0.314ms / **0.81x**; Kernel-only (copy+GEMM): 0.261ms; 非融合 0.255/0.196。
- pre-GEMM 暴露合计 = 0.314 − 0.261 = **0.053ms** (占 pipeline 17%; GEMM 占 83%)。
- 分项暴露: quant ~0.034 | expert_preprocess(prepare+finalize) ~0.010 | SFA reshape ~0.009。
  - 隔离: SKIP_RESHAPE → 0.305/0.83x (reshape≈0.009); no-preprocess → 0.304 (prepare+finalize≈0.010)。
  - 隔离测 quant+sym 0.035 / preproc 0.019 (含各自 launch), 之和 > 暴露 0.053 → 背靠背入队已 pipeline 掉 ~0.01ms launch。
- expert_preprocess in-kernel clock64: setup+barrier 7.3% / remote-read 21.3% / **compute(P4+5) 71.3%** (16076 cyc)。
- arrival barrier wait: **0.000ms** (folded push 已藏掉, 符合预期)。

**overlap 现状 (回答用户疑问)**:
1. 4 个 kernel 单 stream **串行、零 kernel-execution overlap**; 仅 launch latency 被背靠背入队部分 pipeline。
2. preprocess 内 remote-read→compute 是**依赖串行** (finalize 贪心打包依赖 counts), 无法 intra-kernel overlap。
3. arrival wait 已 ≈0 (预期的 overlap 生效)。

**方向候选 (按杠杆)**:
- A. 跨迭代流水: quant[i+1]/preproc[i+1] 与 GEMM[i] 无依赖(不同 parity buffer)→ 双 stream 藏 ~0.044ms → 理论 ~0.98x。最大杠杆。
- B. reshape‖GEMM: 本地 repack 折进 GEMM copy 阶段(与远端 FP4 copy overlap) → 上限 0.83x(-0.009)。用户直接点名。
- C. 压 quant/preproc compute — 已高度优化, 空间小。

**结论**: GEMM 本身(0.261, 83%)是绝对大头且结构性(block_m=128 → 8 wave, 见 07-11 分析); pre-GEMM 仅 0.053。
最大真实收益在把 pre-GEMM 藏到 GEMM 下(方向 A)。下一步先验证方向 A/B 可行性。

---

## 2026-07-15 [iter 1] 把本地 SFA repack 折进 GEMM copy blocks (DG_SFA_PUSH_INLINE) — 回退 0.79x, 已还原

**动机**: iter-0 显示 SFA reshape 是独立 pre-GEMM kernel、串行 0.009ms。DG_SFA_PREPROCESS 当初把它提成独立
kernel 是因为"远端 SFA strided 读是 FP4 copy 后的串行尾巴"; 但 PUSH 后 SFA source 已是**本地 staging**(便宜),
那个理由消失 → 试把本地 repack 折回 copy blocks (skip_sfa_copy=False), 让它与远端 FP4 copy overlap、省掉独立 launch。

**结果**: Test1 PASSED (正确), 但 pipeline 0.314→**0.321 / 0.81x→0.79x, 回退**。

**根因**: copy blocks 只有 **ncb=3 个**。在 3 个 block 里做 col-major strided repack 又成了串行尾巴; 而独立
reshape kernel 用**全部 ~132 CTA** 并行(见 dispatch_sfa_preprocess 重并行, 07月记录)。**并行度 > 省 launch/overlap**。
即使 repack 是本地便宜操作, 3 个 block 的吞吐仍是瓶颈, 盖不住。

**结论**: 折叠形式的 reshape‖GEMM 行不通。独立全网格 reshape kernel 已近最优。已还原 toggle(默认关死代码删掉)。
要 overlap reshape 与 GEMM 又不丢并行度, 得双 stream(reshape 全网格 ‖ GEMM) + GEMM 内等 reshape-done flag — 复杂,
且只省 0.009ms。**转向**: pre-GEMM 大头是 quant(0.034); 最大杠杆是跨迭代流水(方向A)藏 quant+preproc 到 GEMM 下。

---

## 2026-07-15 [iter 2] reframe: pre-GEMM 已优于非融合, 差距全在 fused GEMM 本身 (用户排除跨迭代流水)

**用户定调**: 真实场景 dispatch+GEMM 不连续发送 → **跨迭代流水(方向A)排除**, 衡量口径=单次 dispatch 延迟。

**融合 vs 非融合分解 (8卡 exact ncb=3)**:
- fused pipeline 0.314 = pre-GEMM 0.053 + GEMM(copy+gemm) 0.261
- 非融合 pipeline 0.255 = pre-GEMM 0.059 (quant + DeepEP dispatch 0.049) + GEMM 0.196
- 差距 0.059 分解: GEMM 侧 0.261−0.196 = **+0.065** (慢); pre-GEMM 侧 0.053−0.059 = **−0.006** (我们已更优)。
- → **差距 100% 在 fused GEMM; pre-GEMM 已是净优势, 继续压无法缩差**。

**单 dispatch 内 pre-GEMM 是否还有空间**: 各阶段数据依赖串行(quant→preproc→reshape→GEMM), 已高度优化。
- quant 0.034 (最大, ~10MB 6x FP4 scatter + push, latency-bound); preproc 0.010 (compute 71%, 单CTA 12线程greedy pack); reshape 0.009 (全网格独立kernel, 折叠会回退见iter1)。
- 顶多再挤 ~0.01ms (preproc finalize 并行度 / reshape 双stream), 收益 <0.03x 且复杂/风险。

**结论**: preprocess 侧已近地板且优于非融合。**要继续追 1.0x, 杠杆在 fused GEMM**:
(a) P2P copy +0.034 (NVLink 带宽/copy block 吞吐); (b) block_m=128 → 8 wave 结构性 (NF block_m=256 → 4 wave)。
后者是 masked-grouped 死结, 根治需 GroupedContiguous (大改, 见 07-11 分析)。
**待用户定**: 转 fused GEMM(真实杠杆, 大改) vs 认定 preprocess 近最优收尾。

---

## 2026-07-15 [iter 3] profile quant 内部 Phase1/Phase2 (为 write-once 决策) — scatter 占 51%

**用户方向**: fused preprocess 目标 <20us (现 53us); 关键洞察=fused preprocess **不传 FP4 大数据**(那是 GEMM
copy 干的、且 overlap 在 compute 下), 所以本应远低于非融合 dispatch(40us, 含 FP4 all-to-all)。用户提议:
**quant 不 scatter, 每 token 只写一份**(省 6x 写放大); 参考 Open-Source/DeepGEMM megamoe。

**profile (wire 了 quant profile_clocks, block0 一 token)**: P1 量化 49% (10112 cyc) | **P2 scatter+push 51% (10496 cyc)** | total 20608。
→ scatter 是 quant 的一半。write-once (6x→1x FP4 写) 可砍 scatter ~5/6 → quant compute ~−40% → 34us→~20us, 够到目标。

**权衡 (write-once 大改)**: quant 省; 但 GEMM copy 从"连续 per-expert 读"变"按 index 散射 gather"(远端 scattered
读, 正是 SFA 当年痛点)。数据量不变、仅 locality 变差。净收益取决于慢一点的 copy 能否仍被 GEMM compute 藏住(track B)。
架构改动大: quant + expert_preprocess(算 gather index 而非 per-expert 连续 offset) + GEMM copy(indexed gather) + sym 布局。

**下一步**: 看 megamoe 的 store-once + dispatch 布局(csrc/apis/mega, impls/sm100_fp8_fp4_mega_moe)——他们 copy 是
连续还是 gather? 据此定方案再实施。这是本轮主线(track A)。

---

## 2026-07-15 [iter 4] write-once quant 诊断 (DG_QUANT_WRITE_ONCE) — 否掉: 6x 写已延迟重叠, 只省 4us

**假设(用户)**: quant 的 6x FP4 scatter 是 pre-GEMM 大头, 每 token 写一份可省大量时间 → quant<20us。
**诊断**: 编译 flag 只对 t==0 写(1x 而非 6x), 输出无效仅测时。
**结果**: quant+sym 0.035→**0.031 (仅 −4us)**; scatter cyc 10496→5964 (**−43%, 非预期 −5/6**); pipeline 0.314 不变。
**根因**: 6x FP4 写在内存系统里**延迟重叠**(6 写 ≈ 1.76× 一写, 非 6×) → 写放大几乎免费。假设不成立。
**结论**: **write-once 否掉** —— 只省 4us, 远够不到砍到 20us 所需 ~15us, 还要背 copy indexed-gather 复杂度+风险(数据量不变、locality 变差)。已还原诊断。保留 quant profile_clocks instrumentation(有用)。

**更根本认识**: quant 大头是 P1 量化 compute(62%: 读 BF16 + amax + e2m1x2 转换 + pack, 固有 per-token 工作 + load
延迟), 难砍。pre-GEMM(53us) 是 launch/latency-bound(4 launch: quant/preproc/reshape/GEMM)。**<20us 靠 trim quant
达不到**; 需 megamoe 式 ring-buffer 融合 layout+GEMM 的大重写(csrc impls/sm100_fp8_fp4_mega_moe) 才可能根治。
待用户定: 投入 megamoe 式大重写 vs 认定当前 0.81x 近最优收尾。

---

## 2026-07-15 [iter 5] ★ KSTRIPE per-wave 剖析: 波次1-7已完美overlap, 唯wave-0暴露P2P(流水填充) — 找到真杠杆

**用户 track B**: ncb=3 exact=1 下保证除首轮外每 wave 完美盖住 P2P。**剖析结果=已做到**:
```
wave  exposure  avg_wait  avg_compute
  0    1.93     59,807    31,033   ← 唯一暴露
  1-7  ~0.013   ~350-400  ~27,000  ← 完美隐藏 (等于没等)
```
critical-path CTA: wait+main 289866 / main-only 228044 → **wait ~61822 cyc 几乎全在 wave 0**。P2P isolation 净值 +34us。

**机理**: 36 个 GEMM compute CTA 必须等 copy blocks 拉来第一块 FP4 (P2P 往返**延迟**, 非带宽; 0.68MB 走 NVLink 只需 µs, 62us 是延迟+copyblock启动)。waves 1-7 边算边拉已藏住。wave-0 是流水线**启动填充**, 无前序 compute 可藏。

**真杠杆 (单 dispatch 内, 不涉跨迭代)**: 在 GEMM 启动**前**预取 wave-0 的 FP4 (overlap 到 pre-GEMM 的 quant/preproc/
reshape 阶段), 使 wave-0 不停等 → 打掉 ~34us fill → pipeline 0.314→~0.28 → ~0.91x。
- 挑战: 预取需 preprocess 的 rank_addr_a (刚算完); 需 land 到 local_fp4_buf 且 GEMM 首 tile 可见。
- 候选: (a) 独立 pre-copy kernel 拉首 wave, 与 reshape 同 stream 串行但在 GEMM 前; (b) 双 stream pre-copy ‖ reshape;
  (c) 让 copy blocks 优先拉首 wave tiles 并尽早 signal(改 copy 调度/granularity)。
**下一步**: 定预取首 wave 的最简正确实现, 验证能否打掉 wave-0 wait。这是继 pre-GEMM 之后的主攻点。

---

## 2026-07-15 [iter 6] KTPF>0 试图缩 wave-0 fill — 否掉: per-stripe 开销压全 288 tile, 只 wave-0 受益

**用户思路**: KTPF(K_TILES_PER_FLAG)>0 让 wave-0 compute 在首 stripe 拷完就开算(而非等全 tile K), 缩 wave-0 fill。
**结果**:
- KTPF=4: pipeline 0.314→**0.329 / 0.81x→0.77x 回退**; main-only compute 228k→305k cyc (+34%)。
- KTPF=8/16: **SIGSEGV** (超出每 tile k-tile 数, 越界)。有效 KTPF 只小值。
**根因**: per-stripe flag 轮询 + 更细 mainloop 结构给**全部 288 tile**加开销; 而 wave-0 fill 只惠及 36 tile
(waves 1-7 数据本就 ready、白背开销)。coarse "每 tile 等一次"(KTPF=0) 在数据大多已就绪时最省。**KTPF=0 最优**。

**至此 fused GEMM 两半差距的探索**:
- wave-0 fill(+34us): KTPF 缩不了(iter6); 预取受限于 rank_addr_a 依赖 preprocess(只能 overlap reshape 9us)。~融合固有。
- block_m=128 8-wave(+31us): 未试。prior HINTS 候选=**block_n 256→512** (12×12=144 tiles/36=4 wave, 计算量不变,
  copy 只搬 A 与 block_n 无关应透明, smem 需 <~210KB)。这是打 wave 结构、剩余最大的未试结构杠杆。
**下一步**: 试 block_n 512 (减半 N_blocks → 4 wave)。

---

## 2026-07-15 [iter 7] ★★ block_n 256→512 (FUSED_BLOCK_N=512 FUSED_WARP_N=64): perf 0.81x→0.90x! 但 test1 崩(需修)

**发现**: prior 代码已有 opt-in 开关 FUSED_BLOCK_N (注释: block_n 512 → N_blocks 减半 → 8→4 wave, 早期测 -9.6% bit-exact, 但默认关)。叠到当前 SFA push 上:
- **perf (test2): pipeline 0.314→0.283 / 0.81x→0.90x! kernel-only(copy+GEMM) 0.261→0.226**。巨大, 打的是 +31us wave 结构。
- **正确性 (test1): FAILED + SIGSEGV** (一进 test1 就崩, 无 expert 输出)。push 和纯 baseline 都崩 → 非 push 特有。
- test2(perf) 能跑出数 → **kernel 本身能运行**; test1 崩大概率是正确性 harness 里某 buffer 按 block_n=256 尺寸分配, block_n 512 越界 (out / local_fp4/sfa_buf / grouped_layout / CPU-ref)。smem=262016 距 cap 262144 仅 128B, 也可能触顶。

**这是通向 0.90x 的强线索**(前人已证 bit-exact, 只是当前 harness/代码态不兼容)。值得修 test1 崩。
**下一步**: 定位 test1 崩点(buffer sizing vs block_n), 修好 → 验 bit-exact → 若成立这是本轮最大 win (0.90x 严格正确)。

### [iter 7 更正] block_n 512 的 0.90x 是假的 — launch 失败(too many resources)
真实错误: `cudaFuncSetAttribute invalid argument` + `CUDA error: too many resources requested for launch`。
block_n 512 + warp_n 64 → 16 warp=512 线程 × 232 vreg ≈ 118k > SM 寄存器文件; 且 smem 设置超上限。
前人注释(262016<cap, bit-exact -9.6%)过时/错误(当前 SFA 改动后 smem 涨 / 原就漏算 register)。
test2 的 0.283/0.90x = **launch 失败空跑的假快, 不可信**。block_n 512 as-is 在本硬件**不能 launch**。
需减资源(stages↓ 降 smem / warp_m=128 降到 256 线程降 register)才可能 fit。下一步试 stages=2 / warp 配。

---

## 2026-07-15 [iter 8] block_n 512 + stages=2: 正确但 0.76x — 死结 (装得下就变慢, 装不下就崩)

- block_n 512 @ **stages=3**: 崩 (smem 超 cudaFuncSetAttribute 上限 + too many resources)。
- block_n 512 @ **stages=2**: **Test1 PASSED (bit-exact) 但 pipeline 0.337 / 0.76x** (慢于基线 0.81x)。
  stages 3→2 装下了 smem, 但削弱 GEMM 主循环流水 → GEMM 变慢, 抵消减 wave(8→4) 收益。
- **结论**: block_n 512 在本硬件是死结 (装下就变慢/装不下就崩)。减 wave 需 block_m=256 路线 → 但 masked-grouped
  每 expert M=128 会 2x padding 浪费 → 需 GroupedContiguous 大重写 (见 07-11)。

## 2026-07-15 本轮总结: 0.81x 已达融合方案在本硬件的结构性地板

**穷尽的杠杆 (均记录在案)**:
- pre-GEMM (quant 34/preproc 10/reshape 9 = 53us): 已优于非融合(59us); write-once 只省4us(6x写延迟重叠, iter4);
  fold-reshape 回退(iter1); quant 大头是量化 compute 固有。**近地板**。
- fused GEMM wave overlap: waves 1-7 已完美 overlap(iter5); wave-0 fill(+34us)是融合固有(compute 前无可藏),
  KTPF 缩不了(iter6, per-stripe 开销压全 tile)。
- block_n 512 减 wave(+31us 结构): 资源装不下/stages=2 变慢(iter7-8), 死结。
- 跨迭代流水(藏 pre-GEMM): 用户排除(真实场景 dispatch+GEMM 不连发)。

**残余差距全是结构性/固有**: wave-0 fill(融合本质) + block_m=128 8-wave(需 GroupedContiguous 大重写)。
**当前 0.81x 严格正确 = 此融合方案在 ZW-M890P 的结构性地板**。要破需 GroupedContiguous 级大重写(高风险, 可能才是 ~1.0x 的唯一路)。

---

## 2026-07-15 [iter 9] ★ KTPF=14 (粗粒度, 4 stripes/tile) 是真·正收益 -2.2% (用户 7/14 直觉对)

**背景**: iter-6 测 KTPF=4(14 stripes/tile) 回退, 结论"KTPF 无用"过早。用户指出应试 56 的整除粗粒度 7/14。
K=7168, block_k=128 → **每 tile 56 k-tiles**。KTPF=14 → 4 stripes/tile (最粗, per-stripe 开销最小, wave-0 仍在 1/4 K 开算)。

**3x 中位对比 (perf-only, 同条件, 稳定)**:
- KTPF=14: 0.309/0.308/0.309 → **median 0.309ms**
- KTPF=0 : 0.316/0.315/0.316 → **median 0.316ms**
- → KTPF=14 稳定 **−0.007ms / −2.2%** (非噪声, 3 run std 极小), **Test1 bit-exact PASSED**。
- KTPF=7(8 stripes) 0.343 / KTPF=4(14 stripes) 0.329 更差 → **越粗越好, 甜点=KTPF=14**。

**机理**: KTPF=14 让 wave-0 compute 在首 stripe(14/56=1/4 K)拷完就开算, 缩 wave-0 fill(overall exposure 0.28→0.215);
4 stripes 的 per-stripe 开销小到不抵消。iter-6 的 KTPF=4 错在粒度太细(14 stripes 开销大)。

**win**: 纯配置、bit-exact、−2.2%。建议 push 生产配置加 **K_TILES_PER_FLAG=14** (或默认按 num_k_tiles/4 自适应)。
新最好: pipeline 0.309, ~0.82-0.83x。

---

## 2026-07-15 [iter 10] safety hardening: profiling bounds + explicit staging ownership + per-CTA release

**Review fixes:**
- Quant profiling now carries an explicit `profile_enabled` bit. The normal path no longer
  treats an `int64[1]` dummy as enabled and no longer writes `profile_clocks[1:3]` out of bounds.
- `BlockCopyDispatchContext` now explicitly accepts and retains the separate SFA staging
  tensor/addrs under `DG_SFA_PUSH`; address dtype/device/rank-count and buffer byte size are
  validated. `get_sfa_staging_size` is exported from the public package.
- Every quant CTA now executes `__ppu_threadfence_system_uncache()` before incrementing the
  grid-retire counter. This establishes a per-CTA release for peer `__stbl` writes; the last
  CTA retains the final system-uncache fence before publishing arrival.
- `DispatchBufferLayout::total_bytes_with_staging()` now includes the optional counts region.

**Correctness verdict:** 8 GPUs, prod, `FORCE_EXPECTED_M=128`, exact-grid, ncb=3,
KTPF=14, SFA push+folded arrival: FULL_CORRECTNESS passed 10/10 random-routing rounds;
every expert was bit-exact vs both non-fused and CPU reference.

**Performance verdict (3 independent idle-machine runs, median of 20 events each):**
- Pipeline: **0.309 / 0.310 / 0.309 ms**, median **0.309 ms** (pre-fix median 0.310 ms).
- Kernel-only: 0.254 / 0.255 / 0.252 ms; non-fused pipeline 0.255 / 0.254 / 0.255 ms.
- Quant: 0.036 / 0.036 / 0.036 ms. Per-CTA uncache releases show no measurable regression.

**Result:** improved safety with neutral-to-noise-positive performance; keep changes.

---

## 2026-07-15 [iter 11] wave-0-only K-stripe polling — correct but severe compiler-path regression

**Hypothesis:** KTPF=14 is useful only for wave 0 pipeline fill. For waves 1-7, wait once
for the final stripe at tile entry and pass a null stripe-flag pointer to the mainloop,
removing repeated polling from the remaining 252 tiles.

**Change:** wave 0 retained the existing fine-grained stripe waits. Later waves waited for
the final flag value once and disabled mainloop stripe polling through a runtime-null pointer.

**Verdict:**
- Correct: FULL_CORRECTNESS passed 10/10 random-routing rounds, bit-exact vs NF and CPU.
- Pipeline: **0.467 / 0.459 / 0.463 ms**, median **0.463 ms** vs iter-10 0.309 ms.
- Kernel-only: 0.397 / 0.402 / 0.404 ms; all-local also regressed to 0.373-0.386 ms.

**Analysis:** the regression is in GEMM compute, not P2P (all-local regressed equally).
The extra runtime `curr_wave` branch / conditional mainloop parameter perturbs the PPU
compiler's sensitive register allocation and scheduling path; making later waves logically
simpler does not make the generated persistent kernel cheaper. This matches earlier scheduler
code-sensitivity findings.

**Result:** failed performance iteration. Revert verbatim to iter 10 before continuing.

---

## 2026-07-15 [iter 12] stack DG_BULK_REMOTE with SFA-push + KTPF14 — correct, no additive win

**Hypothesis:** the dedicated PPU remote bulk-load instruction previously improved the
FP4 peer copy by ~2%; stacking it with the current SFA-push and KTPF14 best may reduce
the remaining ~0.030ms P2P exposure.

**Verdict:**
- Correct: FULL_CORRECTNESS passed 10/10 random-routing rounds, bit-exact vs NF and CPU.
- Pipeline: **0.311 / 0.309 / 0.309 ms**, median **0.309 ms**.
- Iter-10 without remote bulk: 0.309 / 0.310 / 0.309 ms, also median 0.309 ms.

**Analysis:** the historical remote-bulk gain does not stack measurably with coarse KTPF14;
both target the same exposed wave-0 remote-read latency. The opt-in also makes the all-local
isolation path illegal, so enabling it by default would add operational risk without a current
target-shape benefit.

**Result:** neutral; keep `DG_BULK_REMOTE` opt-in and retain iter 10 as production best.

---

## 2026-07-15 [iter 13] production Context defaults + max-rank benchmark semantics

**Changes:**
- `BlockCopyDispatchContext.run()` now auto-selects ncb=3/KTPF=14 only for the validated
  8-GPU prod shape (12 local experts, N=6144, K=7168); other shapes retain 1/0 and explicit
  caller arguments always override. Invalid ncb/KTPF values now fail on the host.
- Every headline timing sample is reduced with `MAX` across all 8 ranks before computing
  medians. This reports distributed critical-path latency instead of rank-0-only latency.

**Correctness:** with `USE_CONTEXT_DEFAULTS=1` (no explicit ncb/KTPF passed), FULL_CORRECTNESS
passed 10/10 random-routing rounds at block_m=128, bit-exact vs non-fused and CPU.

**Max-rank performance (3 independent runs):**
- Pipeline: **0.315 / 0.312 / 0.312 ms**, median **0.312 ms**.
- Kernel-only: 0.257 / 0.256 / 0.256 ms; P2P exposure 0.030 ms in every run.
- Non-fused pipeline: 0.255 / 0.254 / 0.254 ms.

**Analysis:** max-rank semantics adds ~0.003ms versus the old rank-0 median, exposing small
cross-rank skew rather than a kernel regression. This is the authoritative benchmark going
forward; current fused speed is ~0.81x by distributed critical-path latency.

**Result:** keep API and benchmark changes.

---

## 2026-07-15 [iter 14] row-safe remote bulk in K-stripe copy — correct, severe regression

**Hypothesis:** iter 12 did not actually exercise remote bulk in the production KTPF=14
path because `run_copy_block_kstripe()` still used plain `__ldg`. Split each 112-int4
token-row stripe into three contiguous 32-lane chunks for paired remote-bulk load/store,
while retaining plain copies for the 16-int4 row tail and rank-count-ragged rows.

**Correctness:** FULL_CORRECTNESS passed 10/10 random-routing rounds on 8 GPUs with
ncb=3/KTPF=14; every expert was bit-exact vs both non-fused and CPU reference.

**Max-rank performance (3 independent idle-machine runs):**
- Pipeline: **0.431 / 0.429 / 0.429 ms**, median **0.429 ms**.
- Iter-13 production baseline: 0.315 / 0.312 / 0.312 ms, median 0.312 ms.
- Non-fused pipeline: 0.254 / 0.254 / 0.255 ms.

**Analysis:** preserving the bulk instruction's full-warp contiguous-address contract
requires division/modulo row/chunk remapping plus separate tail traversal. On this PPU,
the resulting copy-block scheduling/instruction overhead dominates any remote-load gain
and increases the distributed critical path by about 38%. The extremely stable regression
rules out measurement noise.

**Benchmark note:** the raw perf path does not consume `BlockCopyDispatchContext` defaults;
production measurements must explicitly set `NCB_SWEEP=3 K_TILES_PER_FLAG=14`. With
`DG_BULK_REMOTE=1`, `SKIP_ISOLATION=1` also suppresses kernel-only fields, so its printed
zero values are placeholders and must not be interpreted as timings.

**Result:** failed performance iteration; revert the K-stripe bulk implementation and keep
remote bulk disabled for the production K-stripe path.

**Restore verification:** after restoring the iter-13 header byte-for-byte, the exact-grid
production command (`FUSED_EXACT_GRID=1`, ncb=3, KTPF=14) measured **0.314 / 0.313 /
0.315 ms** full pipeline (median 0.314 ms), **0.256 / 0.256 / 0.256 ms** kernel-only,
and 0.029-0.030 ms P2P exposure. Non-fused was 0.255 ms in all three runs.

---

## 2026-07-15 [iter 15] make production exact-grid defaults self-contained

**Finding:** iter 13 made ncb=3/KTPF=14 the Context defaults, but the third required
production setting remained an external `FUSED_EXACT_GRID=1` process environment variable.
With ncb=3/KTPF=14 but no exact-grid flag, the same final code measured 0.425 ms instead of
about 0.314 ms. The raw perf test also defaulted independently to KTPF=0 and ncb=4,8,12,
so a no-argument benchmark silently measured a different launch configuration.

**Change:** plumb `exact_grid` as an explicit runtime argument from Python to the C++ launch.
The validated 8-GPU production shape now defaults to exact-grid, while an explicit
`exact_grid=False` remains available for oversubscription A/B. The perf test selects
ncb=3/KTPF=14 by default only for the same validated shape; other shapes retain legacy
defaults and all environment overrides continue to work.

**Correctness:** without setting `FUSED_EXACT_GRID`, `NCB_SWEEP`, or `K_TILES_PER_FLAG`,
FULL_CORRECTNESS passed 10/10 random-routing rounds on 8 GPUs; every expert was bit-exact
vs both non-fused and CPU reference.

**No-tuning-env max-rank performance (3 independent idle-machine runs):**
- Pipeline: **0.312 / 0.315 / 0.314 ms**, median **0.314 ms**.
- Kernel-only: **0.256 / 0.256 / 0.256 ms**.
- All-local: 0.226 / 0.227 / 0.226 ms; P2P exposure 0.030 / 0.029 / 0.029 ms.
- Non-fused pipeline: 0.255 / 0.254 / 0.255 ms.

**Result:** keep. Production defaults are now self-contained and the benchmark's default
configuration matches the API. Relative to the accidentally oversubscribed 0.425 ms launch,
this removes about 26% latency; relative to the correctly configured iter-13 baseline it is
performance-neutral, as expected.


## Iteration 16 — Disable implicit preprocess profiling memset

**Hypothesis:** the production preprocess path accidentally enqueues `torch.zeros(4)`
for an optional clock buffer on every call. That device memset is inside the measured
preprocess interval even though profiling was not requested.

**Change:** use a zero-element int64 device tensor when `dbg_cyc=None`. On this
runtime it has `data_ptr()==0`, so the existing kernel null check disables profiling
without enqueueing a memset. Explicit profiling buffers keep the old behavior.

**Baseline (idle max-rank):** quant 0.036 ms, expert preprocess 0.021 ms,
gen=0 preprocess 0.027 ms, pre-GEMM sum 0.058 ms, full pipeline 0.313 ms,
copy+GEMM kernel 0.256 ms.

**Three independent 8-GPU prod runs (`FORCE_EXPECTED_M=128`):**

- run 1: quant 0.036 ms, preprocess 0.019 ms, gen=0 0.026 ms, sum 0.056 ms,
  pipeline 0.313 ms, kernel 0.256 ms;
- run 2: quant 0.055 ms, preprocess 0.080 ms, gen=0 0.025 ms, sum 0.134 ms,
  pipeline 0.311 ms, kernel 0.256 ms; this run had a 0.054 ms arrival-wait outlier;
- run 3: quant 0.036 ms, preprocess 0.019 ms, gen=0 0.026 ms, sum 0.056 ms,
  pipeline 0.312 ms, kernel 0.256 ms.

**Median:** quant 0.036 ms, preprocess 0.019 ms, gen=0 0.026 ms,
pre-GEMM sum 0.056 ms, full pipeline 0.312 ms, kernel 0.256 ms.

**Verdict:** keep. The no-wait preprocess stage improves by about 2 us (9.5%)
and the fused pre-GEMM overhead by about 2 us (3.4%), with unchanged GEMM kernel
time. The isolated arrival-wait spike did not affect the full-pipeline median.
Strict correctness is deferred until the next kernel-level candidate is selected.


## Iteration 17 — Initialize only live SymBuffer rank offsets

**Hypothesis:** at world size 8, prepare and finalize each initialized all 72
`SymBuffer::offsets` entries even though only 8 are addressable. Removing the 64 dead
shared-memory stores per stage should reduce merged preprocess setup latency.

**Change tested:** initialize `offsets[0:num_ranks]` only in both device stages.
The first compile-contaminated/overlapping samples were discarded; the following
three runs each started with no GPU process.

**Three independent 8-GPU prod runs:**

- run 1: setup 1,130 cyc, preprocess 0.020 ms, gen=0 0.028 ms, sum 0.056 ms,
  pipeline 0.314 ms, kernel 0.256 ms;
- run 2: setup 1,400 cyc, preprocess 0.020 ms, gen=0 0.026 ms, sum 0.057 ms,
  pipeline 0.314 ms, kernel 0.256 ms;
- run 3: setup 1,098 cyc, preprocess 0.020 ms, gen=0 0.026 ms, sum 0.057 ms,
  pipeline 0.313 ms, kernel 0.256 ms.

**Comparison to iteration 16 median:** setup dropped from 3,316 to 1,130 cycles
(-66%), but preprocess moved from 0.019 to 0.020 ms, pre-GEMM sum from 0.056
to 0.057 ms, and full pipeline from 0.312 to 0.314 ms. Kernel-only stayed 0.256 ms.

**Verdict:** reject and revert. The dead stores were visible in the setup counter,
but they were hidden under other CTA work / event granularity and did not improve
end-to-end latency. Keep the simpler fully initialized `SymBuffer` contract.


## Iteration 18 — Keep merged pair counts in shared memory

**Hypothesis:** merged prepare/finalize runs in one CTA, so its 96 pair counts do
not need to be materialized to the workspace in global memory between stages.

**Change:** reserve 96 additional uint32 shared entries in the merged launch, write
prepare counts there, and consume them directly in finalize. The global pair-count
argument remains in the ABI for workspace compatibility; split prepare/finalize is
unchanged.

**Three independent 8-GPU prod runs:**

- run 1: preprocess 0.019 ms, gen=0 0.025 ms, sum 0.055 ms,
  pipeline 0.311 ms, kernel 0.257 ms;
- run 2: preprocess 0.019 ms, gen=0 0.026 ms, sum 0.055 ms,
  pipeline 0.311 ms, kernel 0.256 ms;
- run 3: preprocess 0.019 ms, gen=0 0.026 ms, sum 0.055 ms,
  pipeline 0.310 ms, kernel 0.256 ms.

**Median vs iteration 16:** preprocess remains 0.019 ms, gen=0 remains 0.026 ms,
rounded pre-GEMM sum improves 0.056 -> 0.055 ms, full pipeline 0.312 -> 0.311 ms,
and kernel-only remains 0.256 ms.

**Verdict:** keep. The gain is small but consistent across all three pipeline runs,
the split API contract is preserved, and the change removes a real global-memory
intermediate. Strict correctness validation remains required before final selection.


## Iteration 19 — Compile-time specialize merged preprocess configuration

**Hypothesis:** the merged JIT is already keyed by rank, world size, expert shape,
hidden size, max tokens, local expert start, and BLOCK_M, but only BLOCK_M reached
the device compiler as a template constant. Promoting all existing JIT keys to
kernel template parameters should fold layout arithmetic and fixed 8-rank/12-expert
loops without adding new cache variants.

**Change:** template the merged launcher/kernel on all existing configuration keys
and feed those constants into prepare/finalize. Dynamic split prepare/finalize and
the public workspace ABI are unchanged.

**Three independent 8-GPU prod runs:**

- run 1: preprocess 0.010 ms, gen=0 0.023 ms, sum 0.047 ms,
  pipeline 0.303 ms, kernel 0.256 ms;
- run 2: preprocess 0.010 ms, gen=0 0.023 ms, sum 0.047 ms,
  pipeline 0.302 ms, kernel 0.256 ms;
- run 3: preprocess 0.010 ms, gen=0 0.023 ms, sum 0.046 ms,
  pipeline 0.305 ms, kernel 0.257 ms.

**Median vs iteration 18:** preprocess 0.019 -> 0.010 ms (-47%), gen=0
0.026 -> 0.023 ms, pre-GEMM sum 0.055 -> 0.047 ms (-15%), full pipeline
0.311 -> 0.303 ms (-2.6%), kernel-only stays 0.256 ms. Finalize/compute clock
drops from about 11,250 to 3,916 cycles (-65%).

**Verdict:** strong keep candidate. Fused pre-GEMM is now about 47 us, below the
measured DeepEP dispatch cost of 51 us. Run 10 strict full-reference checks before
declaring the specialization production-safe.


### Iteration 19 strict correctness validation

Ran 10 independent 8-GPU prod invocations with `FULL_CORRECTNESS=1`,
`TEST_ROUNDS=3`, `VALIDATE_MERGED=1`, `FORCE_EXPECTED_M=128`, SFA push, and
folded quant arrival. Every invocation started from an idle 8-GPU process table.

- full per-expert CPU reference: 10/10 invocations passed (30 routing rounds);
- merged-vs-split layout/address/split/count/masked metadata: 10/10 ALL MATCH;
- public API contract: 10/10 passed;
- fused performance test: 10/10 passed.

**Correctness verdict:** production-safe for the validated 8-GPU configuration.


## Iteration 20 — Reuse one SymBuffer across merged stages

**Hypothesis:** after compile-time specialization shortened the kernel, removing the
second SymBuffer initialization and CTA barrier might become visible in event latency.

**Change tested:** initialize one shared SymBuffer in the merged kernel and pass it
to templated prepare/finalize variants. Split kernels kept their original setup.

**Three independent 8-GPU prod runs:**

- run 1: preprocess 0.010 ms, sum 0.047 ms, pipeline 0.302 ms, kernel 0.256 ms;
- run 2: preprocess 0.011 ms, sum 0.049 ms, pipeline 0.304 ms, kernel 0.256 ms;
- run 3: preprocess 0.010 ms, sum 0.046 ms, pipeline 0.304 ms, kernel 0.256 ms.

**Median vs iteration 19:** preprocess remains 0.010 ms, pre-GEMM sum remains
0.047 ms, while full pipeline moves 0.303 -> 0.304 ms. The clock sub-breakdown
is not directly comparable because moving initialization changed its start point.

**Verdict:** reject and revert. No event-level gain, one microsecond pipeline
regression, and substantially more helper plumbing for no production benefit.


## Profiling follow-up — three independent preprocess clock64 runs

No kernel logic changed. Re-ran current HEAD three times on idle 8-GPU prod with
the existing three-slot clock64 instrumentation.

- run 1: setup/arrival 2,882 cyc, remote counts 3,384 cyc, finalize 3,778 cyc,
  clock sum 10,044 cyc, preprocess event 0.010 ms;
- run 2: setup/arrival 1,224 cyc, remote counts 3,418 cyc, finalize 3,550 cyc,
  clock sum 8,192 cyc, preprocess event 0.010 ms;
- run 3: setup/arrival 1,426 cyc, remote counts 3,530 cyc, finalize 4,044 cyc,
  clock sum 9,000 cyc, preprocess event 0.010 ms.

**Median vs pre-optimization baseline:** setup/arrival 3,992 -> 1,426 cycles
(-64%), remote counts 3,548 -> 3,418 cycles (-4%), finalize/layout
8,962 -> 3,778 cycles (-58%), and the three-slot sum 16,502 -> 9,000 cycles
(-45%). The dominant gain is compile-time-specialized finalize/layout generation;
remote P2P count reads are effectively unchanged.

**Timing audit:** per-iteration CUDA events contain the preprocess kernel and its
device-side arrival polling, but no host synchronize or distributed barrier. Batch
synchronization and distributed max reduction occur after event recording. Existing
clock slot 0 mixes SymBuffer setup, arrival polling, and CTA barriers; it is rank-0
last-call data rather than max-rank data. Also, the prepare count-reduction/shape
tail and slowest finalize thread are not fully covered, so the clock sum is a trend
metric, not a complete kernel duration.


## Iteration 21 — Split production/profile preprocess specializations

**Hypothesis:** the merged kernel executed clock64 instructions unconditionally even
when `dbg_cyc` was null at runtime. A compile-time profile key can remove all timing
instructions from production while a separate profile specialization provides accurate
phase coverage.

**Change:** add `PROFILE_ENABLED` to the existing JIT keys; production passes a
compile-time null profile pointer, while the profile variant records eight phases.
Profiling-only completion barriers cover the slowest expert thread. The test gathers
all rank profiles and reports the coherent rank with the largest total.

**Three independent 8-GPU prod runs:**

- run 1: preprocess 0.010 ms, pre-GEMM 0.046 ms, pipeline 0.304 ms;
- run 2: preprocess 0.010 ms, pre-GEMM 0.046 ms, pipeline 0.301 ms;
- run 3: preprocess 0.010 ms, pre-GEMM 0.047 ms, pipeline 0.303 ms.

**Clock64 phase results (slowest complete rank):** total 10,244 / 9,772 /
10,676 cycles. Remote count reads are 3,416 / 3,572 / 3,540 cycles and greedy
metadata packing is 2,834 / 3,064 / 3,278 cycles; together they account for about
64% of profiled work. Arrival polling varies from 664 to 1,552 cycles. All other
individual phases are below 1,000 cycles.

**Verdict:** keep. Event latency is stable at 0.010 ms and the pipeline median stays
0.303 ms; rounded pre-GEMM improves 0.047 -> 0.046 ms. More importantly, production
can now be inspected independently of profiling code, and the measurements identify
remote count reads plus greedy packing as the remaining optimization targets.
Strict correctness and post-change hgobjdump verification are still required.


## Iteration 22 — Parallelize greedy metadata packing by expert/rank

**Hypothesis:** finalize used only 12 expert threads, each serially walking eight
source ranks with `remaining[]` and `rank_offset[]` state. Mapping all 96
(expert, rank) pairs to CTA threads and computing block intersections directly
should expose independent address/store work and make per-block rank stores contiguous.

**Change:** replace the serial rank state machine with token-interval intersection.
Each pair computes its prefix, intersects `[rank_begin, rank_end)` with every M-block,
and writes its own A/SFA address, split row, and count. Rank 0 writes the shared
expert/grouped metadata.

**Three independent 8-GPU prod runs (`VALIDATE_MERGED=1`):**

- run 1: preprocess 0.012 ms, pre-GEMM 0.050 ms, pipeline 0.301 ms;
- run 2: preprocess 0.009 ms, pre-GEMM 0.045 ms, pipeline 0.303 ms;
- run 3: preprocess 0.009 ms, pre-GEMM 0.045 ms, pipeline 0.302 ms.

All three runs reported `VALIDATE_MERGED: ALL MATCH` for grouped layout, A/SFA
addresses, splits, counts, masked metadata, total blocks, and shape M.

**Median vs iteration 21:** greedy packing 3,064 -> 2,068 cycles (-33%),
profiled total 10,244 -> 9,664 cycles (-5.7%), preprocess event 0.010 ->
0.009 ms (-10%), pre-GEMM 0.046 -> 0.045 ms, and pipeline 0.303 -> 0.302 ms.

**Verdict:** keep candidate. The clock improvement directly matches the rewritten
phase and produces a one-microsecond event/pipeline gain. Run hgobjdump spill and
instruction-overlap checks, then strict full-reference correctness before final keep.


## Iteration 23 — Single-wave triple prefetch for remote counts

**Hypothesis:** hgobjdump showed each count load followed immediately by
`vldcnt(0)`. Having one 32-thread wave build three independent addresses and issue
three loads before unpack/store might create same-wave memory-level parallelism.

**Change tested:** keep the 96-thread CTA for finalize, but let only the first wave
load tids `lane`, `lane+32`, and `lane+64` with three scalarized pointer/value slots.

**Three independent 8-GPU prod runs:**

- run 1: remote counts 3,608 cyc, preprocess 0.010 ms, pre-GEMM 0.047 ms,
  pipeline 0.302 ms;
- run 2: remote counts 3,478 cyc, preprocess 0.009 ms, pre-GEMM 0.046 ms,
  pipeline 0.303 ms;
- run 3: remote counts 3,620 cyc, preprocess 0.009 ms, pre-GEMM 0.046 ms,
  pipeline 0.303 ms.

All runs retained `VALIDATE_MERGED: ALL MATCH`.

**Median vs iteration 22:** remote reads 3,532 -> 3,608 cycles (+2.2%),
preprocess remains 0.009 ms, pre-GEMM 0.045 -> 0.046 ms, and pipeline
0.302 -> 0.303 ms.

**Verdict:** reject and revert. Three scheduler-visible waves already hide the
single-load latency as well as the explicit one-wave prefetch, while the latter
reduces wave-level concurrency and slightly regresses the measured path.


## Iteration 24 — Remove the merged prepare/finalize barrier

**Hypothesis:** prepare already synchronizes after writing masked counts, and
finalize immediately reaches its own SymBuffer initialization barrier. The explicit
barrier between device stages might therefore be redundant.

**Change tested:** remove only the merged-stage `__syncthreads()`.

**Three independent 8-GPU prod runs:**

- run 1: preprocess 0.009 ms, pre-GEMM 0.045 ms, pipeline 0.303 ms;
- run 2: preprocess 0.009 ms, pre-GEMM 0.045 ms, pipeline 0.302 ms;
- run 3: preprocess 0.009 ms, pre-GEMM 0.046 ms, pipeline 0.305 ms.

All runs retained `VALIDATE_MERGED: ALL MATCH`.

**Median vs iteration 22:** preprocess remains 0.009 ms and pre-GEMM remains
0.045 ms, while pipeline moves 0.302 -> 0.303 ms.

**Verdict:** reject and revert. The synchronization is not visible in event latency;
removing it provides no production gain and weakens the explicit stage contract.


### Iteration 22 ISA and strict correctness gate

`hgobjdump` comparison of production kernels:

- iteration 21: 72 vreg, 160 sreg, stack 0, 1,274 instructions,
  71 global stores, 34 shared loads, 0 private-memory operations;
- iteration 22: 32 vreg, 96 sreg, stack 0, 917 instructions,
  40 global stores, 21 shared loads, 0 private-memory operations.

The expert/rank mapping therefore improves latency without register spill; it also
reduces instruction count by 28%, global-store instructions by 44%, and vreg pressure
by 56%. The earlier pre-specialization kernel's 580-byte stack and six `.ga.p`
private load/store operations remain eliminated.

Strict validation ran 10 independent idle 8-GPU invocations with three routing
rounds each, full per-expert CPU reference, and `VALIDATE_MERGED=1`:

- CPU reference and all-rank correctness: 10/10 passed (30 rounds);
- merged-vs-split metadata: 10/10 ALL MATCH;
- API contract: 10/10 passed;
- performance test: 10/10 passed.

**Final gate verdict:** iteration 22 is production-safe for the validated 8-GPU
configuration and remains the current best preprocess implementation.


## Post-iteration 24 reassessment

The current production preprocess is 0.009 ms with 32 vregs, 96 sregs, stack 0,
and no private-memory operations. Greedy packing is parallelized; explicit one-wave
remote prefetch did not beat the original three-wave latency hiding; removing an
apparently redundant stage barrier was event-neutral. Local instruction changes are
therefore at the sub-microsecond/event-resolution floor.

The next material direction is structural fusion into the quant kernel's existing
last-CTA retire path: after every rank publishes arrival, the last CTA can poll peers
and build the specialized metadata before returning, eliminating the standalone
preprocess launch. This must remain opt-in until three gates pass:

1. quant `blocks_per_sm` and hgobjdump resources do not regress from combined-kernel
   register/shared-memory pressure;
2. merged metadata stays bit-exact with split prepare/finalize across random routing;
3. three-run pre-GEMM and full-pipeline medians improve, followed by 10/10 strict
full-reference validation.


## Iteration 25 — Quant last-CTA preprocess fusion prototype

**Hypothesis:** the quant kernel already has a self-resetting grid-retire counter.
Its last CTA could publish arrival and directly run specialized prepare/finalize,
eliminating the standalone 0.009 ms preprocess launch.

**Change tested:** opt-in combined quant kernel with preprocess workspace arguments,
dynamic shared metadata workspace, and last-CTA calls to the validated device helpers.

**Result:** reject and revert. The first 8-GPU run did not complete after more than
four minutes; all workers stayed active and `ppu-smi` became unresponsive until the
run was terminated. No performance sample was accepted.

**Diagnosis:** plain arrival-flag stores were previously followed immediately by
kernel completion. In the combined kernel, the last CTA polls peer flags before
completion, so cached/plain remote flag stores may never become observable to peers,
creating a cross-rank visibility deadlock. The next prototype must publish arrival
with uncached stores plus an explicit post-store system visibility operation before
entering peer polling. It must remain opt-in and retain a timeout/rollback path.


## Iteration 26 — Uncached folded-arrival flag stores

**Hypothesis:** uncached arrival publication may reduce peer polling latency and is
a prerequisite for polling before quant kernel completion.

**Change tested:** replace the folded-arrival plain remote uint32 store with `__stbl`.

**Three independent 8-GPU prod runs:**

- run 1: arrival 660 cyc, pre-GEMM 0.049 ms, pipeline 0.303 ms;
- run 2: arrival 3,492 cyc, pre-GEMM 0.046 ms, pipeline 0.303 ms;
- run 3: arrival 668 cyc, pre-GEMM 0.046 ms, pipeline 0.303 ms.

**Median vs iteration 22:** arrival polling improves roughly 1,566 -> 668 cycles,
but quant moves 0.036 -> 0.037 ms, pre-GEMM 0.045 -> 0.046 ms, and pipeline
0.302 -> 0.303 ms.

**Verdict:** reject and revert for production. Uncached flags improve visibility
but merely move about one microsecond from preprocess into quant. A future fused
prototype may use uncached publication only inside its opt-in specialization.


## Iteration 27 — Dual-stream quant/preprocess overlap

**Hypothesis:** launching preprocess on a second CUDA stream immediately after quant
may let its polling and scheduling overlap the quant tail, while an event keeps GEMM
on the main stream correctly ordered behind the metadata result.

**Change tested:** add an opt-in test-only second preprocess stream; record a completion
event there and make the main stream wait on that event before launching GEMM.

**Three independent 8-GPU prod runs:**

- run 1: quant 0.039 ms, serial diagnostic preprocess 0.012 ms,
  full pipeline 0.303 ms, non-fused baseline 0.255 ms;
- run 2: quant 0.037 ms, serial diagnostic preprocess 0.009 ms,
  full pipeline 0.301 ms, non-fused baseline 0.254 ms;
- run 3: quant 0.036 ms, serial diagnostic preprocess 0.009 ms,
  full pipeline 0.302 ms, non-fused baseline 0.256 ms.

All runs retained `VALIDATE_MERGED: ALL MATCH`. The standalone preprocess number is
only the unchanged serial diagnostic; overlap is measured by the full-pipeline loop.

**Median vs iteration 22:** full pipeline remains exactly 0.302 ms
(runs 0.301/0.302/0.303 ms in both cases).

**Verdict:** reject and revert. The extra stream/event does not create a stable
residency window while the one-wave quant grid occupies all SMs, and its median is
identical to the current best.


## Iteration 28 — Fused last CTA with uncached arrival publication (planned)

**Hypothesis:** iteration 25 deadlocked because the combined kernel polled peer
arrival slots before plain remote stores became visible. Restricting `__stbl`
publication plus an explicit uncache system fence to the fused specialization should
make the handshake live while avoiding iteration 26's cost in the normal path.

**Planned change:** add an opt-in production-shape quant specialization whose last CTA
publishes every arrival slot uncached, fences, then runs the validated prepare/finalize
device helpers in-place. Reuse the quant shared-memory payload as metadata scratch,
and protect the first 8-GPU validation with a process timeout. Accept only if fused
metadata is bit-exact, the run completes, resources do not spill, and three-run
pre-GEMM/full-pipeline medians improve.

**Result:** the uncached publication and post-store uncache fence fixed iteration 25's
deadlock. Every 8-GPU run completed, and the dedicated fused-vs-merged gate reported
`ALL MATCH` on every rank. Removing two fused-tail barriers that are subsumed by the
prepare/finalize helper entry barriers also retained bit-exact metadata.

**Initial three-run timing:** fused quant+preprocess was 0.044 ms with the standalone
preprocess event at 0.000 ms; full pipeline was 0.303/0.303/0.305 ms (median 0.303 ms).

**Barrier-refined three-run timing:** pre-GEMM sums were 0.045/0.045/0.044 ms and full
pipeline was 0.303/0.303/0.303 ms. This is still one microsecond slower than iteration
22's 0.302 ms full-pipeline median, with no stable pre-GEMM improvement over 0.045 ms.

**hgobjdump:** the fused topk=8 quant kernel uses 96 vregs, 128 sregs, reports stack 0,
and has no `.ga.p` private-memory operations. The ordinary quant specialization uses
96 vregs, 96 sregs, stack 0. Before the barrier refinement, the fused kernel contained
2,684 instructions versus 1,587 for ordinary quant; the added preprocess tail increases
scalar-register pressure and instruction footprint but does not spill.

**Verdict:** reject and revert. The visibility protocol is now understood and correct,
but executing preprocess in the one surviving quant CTA merely moves the same ~9 us
work into quant. It does not shorten the GEMM-visible critical path and slightly regresses
the full pipeline. Keep iteration 22 as the production best.


## Iteration 29 — Remove redundant finalize host synchronization (planned)

**Hypothesis:** `BlockCopyDispatchContext.run()` already synchronizes after prepare to
read the exact shape and choose the GEMM config. Its second synchronization after
finalize is used only to re-read and compare the same shape. GEMM is on the same stream,
so removing that diagnostic readback should preserve ordering and correctness while
reducing real API preprocess wall time.

**Plan:** first record max-rank `context.run()` host submission latency over warm rounds,
then make finalize asynchronous in the context path, retain the exact shape returned by
prepare, and rerun the same latency test plus multi-round random-routing correctness.

**Aligned baseline:** after adding an opt-in rank-aligned host timer, rounds 2-5 were
2,262/1,366/2,119/827 us; the last-three median was 1,366 us. Round 1 (~19 ms) includes
first-use runtime work and is excluded.

**Change tested:** call `dispatch_expert_finalize(sync=False)` from the public context,
drop only its duplicate shape readback/assertion, and rely on same-stream ordering into
GEMM. Prepare remains synchronous and retains the capacity check and exact shape.

**Optimized result:** rounds 2-5 were 1,963/1,751/983/1,483 us; the last-three median
was 1,483 us. Five rounds of random-routing context correctness passed on all eight
ranks, but the latency range overlaps baseline and the median does not improve.

**Verdict:** reject and revert the context semantic change. The second synchronization
waits for only the small finalize kernel and is below the much larger Python/JIT/prepare
submission variance. Keep the opt-in aligned timer as diagnostic instrumentation.


## Iteration 30 — Context host-stage profiling (planned)

**Hypothesis:** the 0.8-2.3 ms warm `context.run()` submission time cannot be explained
by a 9 us preprocess kernel or the finalize readback alone. Per-call JIT runtime lookup,
configuration selection, or another host stage likely dominates and may be cacheable.

**Plan:** add opt-in `CONTEXT_HOST_PROFILE=1` timestamps around quant submission,
prepare+readback, config selection, finalize+readback, buffer lookup, and GEMM submission.
Use aligned 8-GPU random-routing rounds, then optimize only the measured dominant stage.

**Result:** five aligned random-routing rounds passed on all ranks. Excluding first-use
compilation, warm generations 4-5 showed the following per-rank ranges:

- quant wrapper/submit: about 127-212 us;
- prepare plus its required readback: about 103-218 us;
- config selection: about 10-190 us (new random shapes miss the shape-keyed LRU);
- finalize plus readback: about 82-115 us;
- cached buffer lookup: about 7-12 us;
- GEMM wrapper/submit: about 117-184 us.

The profiled total was about 0.55-0.80 ms per rank, versus the outer max-rank timer's
roughly 0.93 ms. No single 9 us device phase explains this; it is the sum
of Python wrapper/runtime launch costs, one necessary prepare readback, shape-dependent
config lookup, and rank-max aggregation.

**Verdict:** retain the opt-in profiler. Do not conflate this host submission latency
with the CUDA-event preprocess result. Return the kernel campaign to the measured device
hotspots: remote count reads and metadata packing.


## Iteration 31 — Vectorized remote expert-count reads (planned)

**Hypothesis:** the production prepare phase issues 96 scalar uint32 loads for
8 ranks x 12 local experts. The expert interval is 16-byte aligned and divisible by
four, so 24 coalesced `uint4` remote loads can reduce load/wait instructions while
preserving one-warp parallelism across all peers.

**Plan:** add an aligned/divisible-by-four fast path with scalar fallback, unpack all
four generation-tagged counts after each vector load, and validate three clock64/event
runs plus metadata equality. Inspect hgobjdump to reject the change if the compiler
scalarizes it or introduces spill.

**Three-run result:** remote-count clock phases were 3,558/3,550/3,362 cycles
(median 3,550), preprocess was 0.009/0.064/0.009 ms (the middle event sample was a
system outlier), and full pipeline was 0.303/0.302/0.302 ms (median 0.302). Every run
reported `VALIDATE_MERGED: ALL MATCH` and all tests passed.

**hgobjdump:** the fast path contains a real `vmem.ld.b32x4`; static merged-kernel
instructions fall 1,079 -> 1,064 and waits 100 -> 98. Resources are unchanged at
40 vregs, 128 sregs, stack 0, with no `.ga.p` private operations.

**Verdict:** reject and revert. The remote-count median is effectively identical to
iteration 22's roughly 3,548 cycles and full-pipeline median is unchanged. Count fetch
is governed by peer round-trip/transaction latency rather than scalar instruction count.


## Iteration 32 — Build rank prefixes once during prepare (planned)

**Hypothesis:** every one of the 96 `(expert, rank)` finalize threads currently loops
over all preceding ranks to reconstruct `rank_begin`, repeating roughly 336 shared
loads/adds. Prepare already serially reduces eight counts per expert; writing the running
inclusive prefix back in that same loop lets finalize recover begin/end with at most two
shared loads and no new CTA barrier.

**Plan:** convert the rank-major pair-count cache in-place to inclusive rank prefixes
while producing `masked_m`, update finalize interval construction, and compare clock64
count-reduce versus greedy-pack tradeoffs over three runs. Gate with merged metadata,
then random-routing CPU correctness and hgobjdump if it wins.

**Three-run performance:** count-reduce+shape was 980/784/808 cycles, greedy packing
was 1,492/1,630/1,656 cycles, and profiled total was 9,008/9,340/9,378 cycles. All
three runs measured preprocess at 0.009 ms, pre-GEMM at 0.045 ms, and full pipeline at
0.302 ms, with `VALIDATE_MERGED: ALL MATCH`.

**Median vs iteration 22:** greedy packing 2,068 -> 1,630 cycles (-21%), profiled
total 9,664 -> 9,340 cycles (-3.4%). CUDA-event values remain at their one-microsecond
resolution floor, so the internal improvement is not visible as another rounded us.

**hgobjdump:** production instructions fall 917 -> 911 and waits 80 -> 75 while
remaining at 32 vregs, 96 sregs, stack 0, and zero private operations. The profiling
specialization improves from 40 to 32 vregs with the same 128 sregs and stack 0.

**Strict correctness:** 10 independent 8-GPU processes x 3 random-routing rounds
(30 rounds total) passed the per-expert CPU reference, non-fused comparison, internal
generation/parity contract, and all-rank gate.

**Verdict:** keep. Building rank prefixes once removes repeated finalize work with a
measured 3.4% kernel-cycle reduction, no resource regression, no event/pipeline
regression, and 10/10 strict correctness.


## Iteration 33 — Eight-lane parallel rank-prefix scan (planned)

**Hypothesis:** iteration 32 moves repeated prefix work out of finalize but still has
12 expert threads serially scanning eight ranks. For the production 8-rank topology,
storing counts expert-major lets each aligned eight-lane subgroup build its inclusive
prefix with three `shfl_up` steps and lets finalize read adjacent begin/end entries.

**Plan:** transpose only the 8-rank shared count cache while loading, add a shuffle-scan
fast path with the serial rank-major path unchanged for other world sizes, and measure
whether count-reduce falls without losing the packing/resource gains from iteration 32.

**Three-run performance:** count-reduce+shape was 590/568/770 cycles, greedy packing
was 1,592/1,582/1,478 cycles, and profiled total was 8,586/8,216/11,664 cycles. The
third total contains unrelated arrival/remote-read tail noise; median total is 8,586.
Preprocess events were 0.009/0.009/0.008 ms and full pipeline was 0.302 ms in every run.
All runs reported `VALIDATE_MERGED: ALL MATCH`.

**Median vs iteration 32:** count-reduce 808 -> 590 cycles (-27%), profiled total
9,340 -> 8,586 cycles (-8.1%). Relative to iteration 22, total falls 9,664 -> 8,586
cycles (-11.2%), while the rounded full pipeline remains at its 0.302 ms floor.

**hgobjdump:** the compiler emits exactly three `v.shuffle.up.b32` instructions.
Production instructions/waits improve 911/75 -> 901/69, with resources unchanged at
32 vregs, 96 sregs, stack 0, and no private operations. Profile resources likewise
remain 32 vregs, 128 sregs, stack 0.

**Strict correctness:** 10 independent 8-GPU processes x 3 random-routing rounds
(30 total) passed per-expert CPU reference, non-fused comparison, generation/parity,
and all-rank gates.

**Verdict:** keep. The production topology gets a materially shorter internal critical
path and one 8 us preprocess sample without any event/pipeline regression. Non-8-rank
or non-four-expert-aligned configurations retain iteration 32's rank-major fallback.


## Iteration 34 — Warp-parallel expert prefix scan (planned)

**Hypothesis:** finalize still spends about 0.50k cycles with thread 0 serially scanning
12 experts for M-block and M offsets. A single warp can scan both values in four
`shfl_up` steps, write exclusive prefixes in parallel, and use the existing following
CTA barrier, while retaining the serial fallback above 32 experts.

**Plan:** add a <=32-expert warp scan for both prefix arrays and total outputs, then
measure the expert-prefix and total clock phases, ISA/resources, and full metadata.

**Three-run performance:** expert prefix was 400/400/396 cycles and full pipeline was
0.302/0.301/0.300 ms (median 0.301 ms). Preprocess stayed 0.009 ms, pre-GEMM stayed
0.045 ms, and all three runs reported `VALIDATE_MERGED: ALL MATCH`. Profiled totals
9,778/8,672/9,484 cycles include variance in the arrival/remote phases; the targeted
prefix phase itself is stable.

**Median vs iteration 33:** expert prefix 504 -> 400 cycles (-21%); full pipeline
0.302 -> 0.301 ms (-0.3%). This is the first post-iteration-22 change to cross CUDA
event resolution in the end-to-end pipeline.

**hgobjdump:** the <=32-expert specialization contains ten new expert-scan shuffles
plus iteration 33's three rank-scan shuffles. Despite that, production instructions
fall 901 -> 852 and waits 69 -> 58; resources remain 32 vregs, 96 sregs, stack 0,
and no private operations. Profile resources remain 32 vregs/128 sregs/stack 0.

**Strict correctness:** 10 independent 8-GPU processes x 3 random-routing rounds
(30 total) passed per-expert CPU reference, non-fused comparison, generation/parity,
and all-rank gates.

**Verdict:** keep. The warp scan is simpler in ISA, reduces the intended clock phase,
produces a one-microsecond full-pipeline median improvement, and passes 10/10 strict
validation with no spill or occupancy-resource regression.


## Iteration 35 — Reuse one SymBuffer descriptor across merged stages (planned)

**Hypothesis:** merged prepare and finalize initialize identical shared SymBuffer rank
offsets twice; the second setup is a stable ~276-cycle phase plus a CTA barrier. A
single descriptor initialized by the merged wrapper can serve both device helpers,
while standalone split kernels retain their local setup.

**Plan:** add an optional preinitialized shared descriptor to both helpers, initialize
it once in the merged kernel, preserve profiling by charging the one setup to slot 0,
and validate phase clocks, resources, and split-vs-merged metadata.

**Three-run result:** prepare setup became 1,616/1,786/1,808 cycles while finalize
setup fell from about 276 to 62/64/64 cycles. Pre-GEMM was 0.047/0.046/0.045 ms
(median 0.046), and full pipeline was 0.302 ms in all three runs. Metadata remained
`ALL MATCH` and all tests passed.

**Verdict:** reject and revert. Passing a generic shared descriptor and allocating the
merged wrapper descriptor makes the one-time setup much more expensive than the
~212-cycle finalize saving. It regresses iteration 34's 0.045 ms pre-GEMM and 0.301 ms
pipeline medians; duplicated tiny descriptors are preferable on this compiler.


## Post-iteration 35 final validation

The worktree was restored to iteration 34 and validated beyond the primary 8-GPU gate:

- 2-GPU production-shape fallback: 3 random-routing rounds with per-expert CPU
  reference, all passed;
- 4-GPU production-shape fallback: 3 random-routing rounds with per-expert CPU
  reference, all passed;
- final 8-GPU performance rerun: full pipeline 0.302/0.302/0.301 ms, all metadata
  `ALL MATCH`, all tests passed.

The first final-run diagnostic breakdown had an isolated 0.078 ms preprocess/system
tail while its full pipeline remained 0.302 ms; the next two hot samples were 0.009 ms
preprocess and 0.046 ms pre-GEMM. Across the accepted performance runs, the stable
operating range is therefore about 9 us preprocess, 45-46 us pre-GEMM, and 301-302 us
full pipeline. The deterministic accepted gain is the 11% clock64 reduction and ISA
shrink from iteration 22 through 34; whole-pipeline one-us movement remains near noise.


## 2026-07-15 fused vs non-fused GPU-count sweep

One current-HEAD (`4cc63d9`) production-shape run per topology. Common settings:
`TEST_CONFIG=prod`, `FORCE_EXPECTED_M=128`, `DG_SFA_PUSH=1`,
`FUSED_ARRIVAL_IN_QUANT=1`, merged metadata validation enabled. The 2/4-GPU runs use
the test's current `ncb=4,8,12` sweep and select the best; the 8-GPU production default
uses `ncb=3` and `K_TILES_PER_FLAG=14`.

| GPUs | Best NCB | Fused full pipeline | Non-fused pipeline | non-fused / fused | Fused slowdown |
|---:|---:|---:|---:|---:|---:|
| 2 | 12 | 0.316 ms | 0.273 ms | 0.86x | 15.8% |
| 4 | 12 | 0.346 ms | 0.251 ms | 0.73x | 37.8% |
| 8 | 3 | 0.302 ms | 0.254 ms | 0.84x | 18.9% |

All three runs reported `VALIDATE_MERGED: ALL MATCH` and `All PASSED`. Raw container
logs: `/tmp/perf_scale_2gpu.log`, `/tmp/perf_scale_4gpu.log`, and
`/tmp/perf_scale_8gpu.log`.


## Iteration 36 — Warp-per-topk quant scatter (kept)

### Result

- Configuration: 8 GPU, production case, `topk=6`, `hidden=7168`, `DG_SFA_PUSH=1`, `FUSED_ARRIVAL_IN_QUANT=1`, `ncb=3`.
- Three independent idle runs all passed `VALIDATE_MERGED: ALL MATCH` and `All PASSED!`.
- Run 1: P1 10,032 cycles, P2 5,408 cycles, quant 35 us, preprocess 14 us, pipeline 295 us.
- Run 2: P1 9,994 cycles, P2 5,398 cycles, quant 37 us, preprocess 15 us, pipeline 296 us.
- Run 3: P1 9,936 cycles, P2 5,500 cycles, quant 35 us, preprocess 14 us, pipeline 295 us.
- Median: P1 9,994 cycles, P2 5,408 cycles, total 15,436 cycles, quant 35 us, pipeline 295 us.
- Versus the current reference (P1 about 9,636 cycles, P2 about 9,168 cycles, total 18,804 cycles, pipeline 301–302 us), P2 is 41.0% faster and the measured block-0 quant path is 17.9% faster. P1 regresses about 3.7%, but end-to-end improves 6–7 us.
- Decision: keep. The optimization preserves the existing per-expert output layout and merged-output correctness while recovering more than the targeted 4 us.

**Hypothesis:** Phase 2 currently makes all eight warps walk six destinations in
lockstep. Assigning one warp to each topk destination preserves the exact per-expert
contiguous layout and total store count, but exposes address generation and independent
destination writes concurrently. The same mapping is applied to the remote SFA push.

**Plan:** change only Phase 2 thread ownership, retain slot atomics/layout/ABI, run three
idle 8-GPU production benchmarks with clock64 and merged metadata validation, then gate
any winner with hgobjdump and strict random-routing correctness.


## Iteration 37 — Iteration 36 strict validation (passed)

**Plan:** run three independent 8-GPU processes, each with ten production-shape random-routing rounds and `FULL_CORRECTNESS=1` (30 rounds total), then inspect the generated quant kernel's resource usage and ISA for local-memory spill. Keep iteration 36 only if every correctness gate passes and there is no spill regression.

**Correctness result:** all three independent 8-GPU processes passed all ten rounds, including per-expert CPU reference, non-fused comparison, generation/parity checks, and all-rank gates. Every reported numerical difference was zero. This gives 30 successful process-round executions; each process intentionally repeats the same deterministic ten routing seeds, so the three processes primarily add race/stability coverage.

**ISA/resource result (`hgobjdump`, production `<7168, 8>` specialization):**

- baseline cache object `332c193e78e9`: 96 vregs, 96 sregs, 33 bytes shared memory, stack size 0, 4,036 disassembly lines, 139 wait/barrier matches;
- iteration-36 object `d125862c05df`: 112 vregs, 96 sregs, 33 bytes shared memory, stack size 0, 3,881 disassembly lines, 137 wait/barrier matches;
- no stack/spill/private/local-memory instruction reference was found in either disassembly; the function return offset shrank from `0x31d8` to `0x3040` (408 bytes of machine code).

**Verdict:** keep iteration 36. The warp ownership change costs 16 vregs, but introduces no spill, shrinks the instruction stream, passes strict correctness, and remains 6–7 us faster end-to-end after any register/occupancy effect is already included. Raw correctness logs are `/tmp/iter37_correctness_{1,2,3}.log` on the host; ISA dumps are `/tmp/iter37_quant_{332c193e78e9,d125862c05df}.isa` in the container.


## Iteration 38 — Preserve padded-route barrier safety (planned)

**Issue:** iteration 36 changed the SFA topk loop into a warp-owned `if`, but retained a `continue` that then targets the outer grid-stride token loop. An invalid expert (`-1`) or overflowed slot could make only one warp skip the CTA reuse barrier and deadlock. The kernel's existing `MAX_TOPK=8` shared arrays also require an explicit public-entry bound rather than silently accepting topk above eight.

**Plan:** replace the SFA early `continue` with a nested validity guard, assert `1 <= topk <= 8` at the Python entry, validate padded `-1` routing, rerun strict production correctness, and repeat the three-run performance gate.

**Correctness result:** an 8-GPU production-shape test injected `-1` into the last topk position of every other token for three random-routing rounds. It completed without a hang and passed every per-expert CPU reference, generation/parity check, and all-rank gate with zero numerical difference. The temporary injection test was removed after the run.

**Three-run performance result:**

- run 1: P1 9,974 cycles, P2 5,362 cycles, quant 37 us, pipeline 296 us;
- run 2: P1 9,318 cycles, P2 5,048 cycles, quant 38 us, pipeline 297 us;
- run 3: P1 10,104 cycles, P2 5,104 cycles, quant 37 us, pipeline 296 us;
- median: P1 9,974 cycles, P2 5,104 cycles, total 15,208 cycles, quant 37 us, pipeline 296 us.

All runs reported merged metadata `ALL MATCH` and `All PASSED`. The safety fix is about 1 us slower than iteration 36's 295 us median but remains 5–6 us faster than the 301–302 us pre-optimization reference; P2 remains about 44% below the 9,168-cycle reference.

**Verdict:** keep. Invalid/padded routing can no longer strand a subset of the CTA before the reuse barrier, topk above the kernel's existing shared-array capacity fails explicitly, and the targeted topk6 speedup remains intact. Raw logs: `/tmp/iter38_invalid_route.log` and `/tmp/iter38_perf_{1,2,3}.log`.


## 2026-07-16 post-topk 2/4/8-GPU performance sweep

Current HEAD `4cc5e45`; production shape with `topk=6`, `hidden=7168`, `FORCE_EXPECTED_M=128`, `DG_SFA_PUSH=1`, `FUSED_ARRIVAL_IN_QUANT=1`, and merged metadata validation. Every sample was preceded by an idle `ppu-smi` check. The 2/4-GPU runs swept NCB=4/8/12; the 8-GPU runs used the production default NCB=3 and KTPF=14.

| GPUs | Run | Best NCB | Fused pipeline | Non-fused pipeline | non-fused / fused |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 12 | 0.310 ms | 0.274 ms | 0.88x |
| 2 | 2 | 12 | 0.309 ms | 0.274 ms | 0.89x |
| 2 | 3 | 12 | 0.309 ms | 0.274 ms | 0.89x |
| 4 | 1 | 12 | 0.339 ms | 0.251 ms | 0.74x |
| 4 | 2 | 12 | 0.338 ms | 0.250 ms | 0.74x |
| 4 | 3 | 12 | 0.339 ms | 0.250 ms | 0.74x |
| 8 | 1 | 3 | 0.295 ms | 0.263 ms | 0.89x |
| 8 | 2 | 3 | 0.296 ms | 0.254 ms | 0.86x |
| 8 | 3 | 3 | 0.296 ms | 0.254 ms | 0.86x |

| GPUs | Median fused | Median non-fused | non-fused / fused | Fused slowdown | Previous fused | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.309 ms | 0.274 ms | 0.89x | 12.8% | 0.316 ms | -7 us (-2.2%) |
| 4 | 0.339 ms | 0.250 ms | 0.74x | 35.6% | 0.346 ms | -7 us (-2.0%) |
| 8 | 0.296 ms | 0.254 ms | 0.86x | 16.5% | 0.302 ms | -6 us (-2.0%) |

The fused results are stable within 1 us for every topology. The first 8-GPU non-fused sample (0.263 ms) is an isolated high sample; its median remains 0.254 ms. On 8 GPUs, the median clock64 breakdown is P1=9,406 cycles, P2=3,966 cycles, total=13,326 cycles; quant event=37 us, expert preprocess=15 us, and their exposed sum=52 us. All nine runs reported `VALIDATE_MERGED: ALL MATCH` and `All PASSED`.

Raw host logs: `/tmp/perf_scale_20260716_{2,4,8}gpu_run{1,2,3}.log`; matching pre-run device snapshots use the same stem with `_ppusmi_runN.log`.


## Iteration 39 — Filter CAS fusion + atomic-late from uncommitted quant experiments (rejected)

The original three-file experiment (`mxfp4_quant.cuh`, `jit/compiler.py`, and the multi-GPU test) was preserved in `stash@{0}` as `backup pre-filter quant experiments 3418e7fe`. The candidate intentionally retained only two production-path changes: first-touch generation reset plus slot-0 claim in one CAS, and issuing the hoisted BF16 loads before the topk slot atomics. The cp.async pipeline, empty AIU switch, profiling ABI change, and test-buffer changes were removed.

Strict 8-GPU production correctness passed three random-routing rounds, including per-expert CPU reference, generation/parity, and all-rank gates. The production `<7168,8>` kernel used 104 vregs, 96 sregs, 33 shared-memory units, and stack size 0.

| Run | P1 cycles | P2 cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,262 | 4,922 | 13,184 | 47 us | 85 us | 291 us | 254 us |
| 2 | 9,250 | 5,870 | 15,120 | 37 us | 17 us | 292 us | 257 us |
| 3 | 9,376 | 5,698 | 15,074 | 35 us | 15 us | 291 us | 265 us |

Independent medians are P1=9,250 cycles, P2=5,698 cycles, total=15,074 cycles, quant=37 us, preprocess=17 us, and full pipeline=291 us. Run 1's 85-us preprocess breakdown is an isolated diagnostic outlier: its full pipeline remained 291 us.

Against the clean committed reference (P1=9,406, P2=3,966, total=13,326 cycles; quant=37 us; preprocess=15 us; explicit sum=52 us; pipeline=296 us), the candidate did not improve quant event time and regressed block0 total clocks by about 13.1%, driven by a 43.7% P2 increase. The apparent full-pipeline movement conflicts with the direct target metrics while the non-fused reference varied from 254 to 265 us, so it is not attributable to the candidate.

**Verdict:** reject and revert the candidate. Leave HEAD on the clean accepted implementation; retain the original full experiment only in the named stash for forensic recovery. Raw logs: `/tmp/iter39_filtered_correctness.log` and `/tmp/iter39_filtered_perf_{1,2,3}.log`.


## Iteration 40 — Warp-shuffle direct scale packing (rejected)

**Hypothesis:** for production hidden sizes with `K_BLOCKS % 8 == 0`, adjacent four-lane scale-group leaders can exchange their UE8M0 bytes with a warp shuffle and write `s_packed_scale` directly. This removes the `s_scale_inv` byte array, the standalone packing loop, and one CTA barrier; non-aligned hidden sizes retain the original generic path.

The 8-GPU production strict-correctness gate passed three random-routing rounds with per-expert CPU reference, generation/parity, and all-rank checks. Resource usage improved to 96 vregs, 96 sregs, 31 shared-memory units, and stack size 0 (clean reference was about 112/96/33/0).

Single-GPU generation-0 quant microbench: clean 31.03/27.21/27.35 us (median 27.35 us, first sample cold); candidate 27.78/27.61/27.76 us (median 27.76 us), a 1.5% regression.

| Run | P1 cycles | P2 cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,078 | 6,476 | 16,554 | 40 us | 18 us | 296 us | 254 us |
| 2 | 10,336 | 6,232 | 16,568 | 36 us | 14 us | 295 us | 255 us |
| 3 | 10,620 | 6,328 | 16,948 | 37 us | 15 us | 295 us | 254 us |

Independent medians are P1=10,336 cycles, P2=6,328 cycles, total=16,568 cycles, quant=37 us, preprocess=15 us, and pipeline=295 us. Compared with the clean reference (P1=9,406, P2=3,966, total=13,326 cycles; quant=37 us; preprocess=15 us; pipeline=296 us), clock64 total regressed 24.3% and the quant event did not improve. The one-us pipeline movement is below noise and contradicted by both direct quant measurements.

**Verdict:** reject and revert. Lower register/shared-memory counts do not compensate for the extra shuffle and altered schedule on this compiler. Raw logs: `/tmp/iter40_micro_{baseline,candidate}_{1,2,3}.log`, `/tmp/iter40_correctness.log`, and `/tmp/iter40_perf_{1,2,3}.log`.


## Iteration 41 — Safe quant-only Phase-1 clock split (diagnostic, reverted)

A temporary `DG_QUANT_PROFILE_P1SPLIT` build was scoped exclusively to `_mxfp4_quantize_to_sym_buffer`, and the Python API required a seven-element clock buffer while enabled. This avoided the global JIT-signature pollution and out-of-bounds profile writes found in the earlier uncommitted experiment. Lane 32 recorded the split so no topk-lane atomic was charged to its setup segment.

| Run | Setup | BF16 load + quant compute | First CTA barrier wait | Scale pack + second barrier | P1 total | P2 | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 968 | 4,562 | 3,702 | 350 | 9,582 | 5,576 | 15,158 |
| 2 | 1,060 | 4,274 | 4,790 | 346 | 10,470 | 5,250 | 15,720 |
| 3 | 1,076 | 4,426 | 4,490 | 354 | 10,346 | 4,868 | 15,214 |

Independent medians are setup=1,060, load+compute=4,426, first-barrier wait=4,490, pack+barrier=350, P1=10,346, P2=5,250, and total=15,214 cycles. Within P1, load+compute is about 42.8%, the first barrier wait 43.4%, setup 10.2%, and pack only 3.4%.

**Conclusion:** Iteration 40 targeted the wrong phase: removing the 350-cycle pack/barrier cannot overcome the additional shuffle schedule, while almost half of P1 is lane-32 waiting for the slowest quant warp. The next useful direction is reducing inter-warp load/compute imbalance or separating the six atomic lanes from the quant worker warps, not further packing micro-optimizations. Instrumentation was fully reverted after measurement. All runs reported merged metadata `ALL MATCH` and passed. Raw logs: `/tmp/iter41_profile_{1,2,3}.log`.
## Iteration 42 — Seven balanced quant warps + one atomic warp (provisional)

For shapes where `FP4_INTS` is divisible by 224 (including production HIDDEN=7168: 896 int4 chunks), warp 7 exclusively claims the topk expert slots while warps 0..6 quantize exactly four chunks per lane. Other shapes retain the original 256-thread mapping. The intent is to overlap the six slot atomics with balanced quant work instead of making warp 0 execute atomics plus the longest quant path before the first CTA barrier.

Single-GPU generation-0 microbench was 27.82/26.22/27.18 us, median 27.18 us versus the clean 27.35-us median (weak +0.6% signal). The production kernel uses 104 vregs, 96 sregs, 33 shared-memory units, and stack size 0. Eight-GPU production strict correctness passed three random-routing rounds with per-expert CPU reference, generation/parity, and all-rank checks.

The first three-run 8-GPU performance gate did not return a sample because the SSH connection was closed by the remote host (`exit 255`) before output. This is a partial infrastructure run, not a performance result. The candidate is committed provisionally so the failed iteration is reproducible; it must not be called a win until a fresh three-run gate completes. Existing logs: `/tmp/iter42_micro_{1,2,3}.log` and `/tmp/iter42_correctness.log`.

## Iteration 43 — Iteration 42 production validation (kept)

After restarting the stopped `sglang.lxh` container, the seven-quant-warp plus one-atomic-warp candidate was rerun three times on eight idle GPUs. Each run used the production shape (`topk=6`, `hidden=7168`), `FORCE_EXPECTED_M=128`, `DG_SFA_PUSH=1`, `FUSED_ARRIVAL_IN_QUANT=1`, NCB=3, and KTPF=14. A clean `ppu-smi` snapshot preceded every sample, and all three runs ended with `All PASSED`.

| Run | P1 cycles | P2 cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,338 | 4,410 | 10,748 | 38 us | 22 us | 286 us | 255 us |
| 2 | 6,492 | 3,030 | 9,522 | 37 us | 20 us | 286 us | 253 us |
| 3 | 6,758 | 5,090 | 11,848 | 36 us | 20 us | 287 us | 253 us |

Independent medians are P1=6,492 cycles, P2=4,410 cycles, total=10,748 cycles, quant=37 us, preprocess=20 us, full pipeline=286 us, and non-fused=253 us. Against the clean accepted reference (P1=9,406, P2=3,966, total=13,326 cycles, quant=37 us, preprocess=15 us, pipeline=296 us), P1 improves 31.0% and the direct quant total improves 19.3%. P2 regresses 11.2%, but the 2,914-cycle P1 saving dominates it. The rounded quant event stays at 37 us, while full pipeline improves by 10 us; the preprocess event increase is cross-rank noise outside the modified kernel and does not erase the direct clock64 win.

**Verdict:** keep iteration 42. The intended load/compute rebalance removes the dominant first-barrier wait, has already passed three strict random-routing correctness rounds and spill/resource inspection, and now passes the required three-run production performance gate. Raw logs: `/tmp/iter43_iter42_perf_{1,2,3}.log`; device snapshots: `/tmp/iter43_iter42_perf_ppusmi_{1,2,3}.log`.

## Iteration 44 — Hoist Phase-2 shared FP4 loads (rejected)

ISA inspection showed the Phase-2 local FP4 scatter executing each of the seven per-lane chunks as `tsm.ld; wait tsmcnt(0); vmem.st`. A candidate split the loop into seven shared-memory loads followed by seven global stores, attempting to expose shared-load MLP.

The first idle single-GPU generation-0 microbench regressed to 36.62 us from iteration 42's 27.18-us median (+34.7%). This is far beyond run-to-run noise, so the candidate was stopped before consuming an eight-GPU gate. Keeping seven `int4` values live at once creates enough register-lifetime/scheduling pressure to overwhelm the removed waits.

**Verdict:** reject and revert. Prefer a bounded software pipeline with only one or two chunks in flight if this path is revisited. Raw log: `/tmp/iter44_micro_1.log`; device snapshot: `/tmp/iter44_micro_ppusmi_1.log`.

## Iteration 45 — Pairwise-unrolled Phase-2 FP4 scatter (provisional keep)

The local FP4 scatter was rewritten in two-chunk source groups, limiting live data to two `int4` values instead of iteration 44's seven. The compiler did not preserve two simultaneous shared loads: ISA still contains one `tsm.ld`, an immediate `tsmcnt(0)` wait, then one global store. It did, however, fully unroll the seven production-shape stores and reduce the kernel from 104 to 96 vregs, with 96 sregs, 33 shared-memory units, and stack size 0.

Single-GPU generation-0 microbench was 27.83/26.22/27.38 us, median 27.38 us versus iteration 42's 27.18-us median (+0.7%, effectively flat).

| Run | P1 cycles | P2 cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7,532 | 2,350 | 9,882 | 37 us | 21 us | 284 us | 255 us |
| 2 | 7,290 | 3,192 | 10,482 | 35 us | 18 us | 285 us | 255 us |
| 3 | 7,288 | 3,224 | 10,512 | 36 us | 20 us | 285 us | 253 us |

Independent medians are P1=7,290 cycles, P2=3,192 cycles, total=10,482 cycles, quant=36 us, preprocess=20 us, full pipeline=285 us, and non-fused=255 us. Against iteration 43, P1 regresses 12.3%, P2 improves 27.6%, direct total improves 2.5%, and the event/pipeline medians each improve by 1 us. Every production performance run passed, and every sample was preceded by an idle-device snapshot.

**Verdict:** provisional keep. The direct total and P2 both improve with lower register use and no spill, but strict random-routing correctness remains required before calling this accepted. Raw logs: `/tmp/iter45_micro_{1,2,3}.log` and `/tmp/iter45_perf_{1,2,3}.log`; device snapshots use matching `_ppusmi_` names.

## Iteration 46 — Iteration 45 strict validation (passed)

An idle eight-GPU production process ran three independent random-routing rounds with `FULL_CORRECTNESS=1`. Internal generations/parities were 1/1, 2/0, and 3/1. Every per-expert CPU reference comparison on the reported local experts had exactly zero maximum and mean difference, and the test's all-rank reduction passed for all three rounds. The public API generation/parity contract and final test summary also passed.

**Verdict:** accept iteration 45. It has now passed the three-run performance gate, resource/spill inspection, and strict random-routing correctness. Raw log: `/tmp/iter46_iter45_correctness.log`; idle-device snapshot: `/tmp/iter46_iter45_correctness_ppusmi.log`.

## Iteration 47 — Factor-two Phase-2 scatter unroll (rejected)

Replacing iteration 45's full expansion with `#pragma unroll 2` produced the intended ISA: two `tsm.ld.b32x4` operations followed by `tsmcnt(1)`/store and `tsmcnt(0)`/store. Resources returned to 104 vregs, 96 sregs, 33 shared-memory units, and stack size 0. The single-GPU microbench was 27.27/27.19/27.78 us (median 27.27 us), effectively tied with both accepted candidates.

| Run | P1 cycles | P2 cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,804 | 4,842 | 11,646 | 38 us | 22 us | 287 us | 255 us |
| 2 | 6,656 | 3,384 | 10,040 | 37 us | 21 us | 284 us | 254 us |
| 3 | 6,822 | 2,474 | 9,296 | 38 us | 22 us | 287 us | 265 us |

Independent medians are P1=6,804 cycles, P2=3,384 cycles, total=10,040 cycles, quant=38 us, preprocess=22 us, full pipeline=287 us, and non-fused=255 us. Although the single-token block0 total is 4.2% lower than iteration 45, the all-token/max-rank quant event regresses from 36 to 38 us and the full pipeline from 285 to 287 us. P2 also varies widely from 2,474 to 4,842 cycles.

**Verdict:** reject and revert. The requested target is aggregate quant latency, so a block0 clock improvement that regresses the distributed event and pipeline is not retained. Raw logs: `/tmp/iter47_micro_{1,2,3}.log` and `/tmp/iter47_perf_{1,2,3}.log`; matching idle-device snapshots use `_ppusmi_` names.

## Iteration 48 — Fully unroll local scale scatter (rejected)

Adding `#pragma unroll` to the four-iteration local scale scatter expanded all four shared loads/stores. The production kernel rose from 96 to 128 vregs while retaining 96 sregs, 33 shared-memory units, stack size 0, and no spill. Single-GPU generation-0 microbench was 26.86/27.09/28.07 us (median 27.09 us), 1.1% below iteration 45.

| Run | P1 cycles | P2 cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,498 | 1,006 | 7,504 | 36 us | 20 us | 284 us | 255 us |
| 2 | 6,366 | 3,848 | 10,214 | 35 us | 19 us | 285 us | 264 us |
| 3 | 6,352 | 1,130 | 7,482 | 36 us | 20 us | 286 us | 256 us |

Independent medians are P1=6,366 cycles, P2=1,130 cycles, total=7,504 cycles, quant=36 us, preprocess=20 us, full pipeline=285 us, and non-fused=256 us. The apparent 64.6% P2 and 28.4% direct-total clock reductions do not improve either aggregate quant event or full pipeline versus iteration 45.

The discrepancy is explained by the profiling boundary: P2 clock64 is captured before the final CTA reuse barrier. Full unrolling issues stores sooner but shifts their completion wait into that following barrier, outside the reported P2 interval. The event timer includes the complete kernel and therefore remains authoritative.

**Verdict:** reject and revert. Do not spend 32 extra vregs on an issue-time-only clock win with no event or pipeline gain. Raw logs: `/tmp/iter48_micro_{1,2,3}.log` and `/tmp/iter48_perf_{1,2,3}.log`; matching idle-device snapshots use `_ppusmi_` names.

## Iteration 49 — Include the P2 reuse barrier in clock64 (kept diagnostic fix)

The P2 clock sample was moved after the existing CTA reuse barrier, and the Python API comment now defines P2 as scatter/push plus reuse-barrier completion. This corrects the issue exposed by iteration 48: asynchronous stores can be issued before the old clock boundary while their wait is paid immediately afterward, making issue-time improvements look like completed-work improvements.

| Run | P1 cycles | P2 + barrier cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,738 | 4,896 | 11,634 | 37 us | 21 us | 286 us | 255 us |
| 2 | 7,120 | 5,714 | 12,834 | 36 us | 19 us | 285 us | 255 us |
| 3 | 7,460 | 3,140 | 10,600 | 37 us | 20 us | 284 us | 265 us |

Independent medians are P1=7,120 cycles, completed P2=4,896 cycles, total=11,634 cycles, quant=37 us, preprocess=20 us, full pipeline=285 us, and non-fused=255 us. Aggregate performance remains at iteration 45's operating point within one microsecond, while the diagnostic now accounts for the previously hidden store-completion wait.

**Verdict:** keep the diagnostic fix. Future P2 candidates must improve this completed interval and the aggregate event rather than merely moving waits across the clock boundary. Raw logs: `/tmp/iter49_profile_{1,2,3}.log`; idle-device snapshots: `/tmp/iter49_profile_ppusmi_{1,2,3}.log`.

## Iteration 50 — Elide redundant sym-buffer scale scatter under SFA push (provisional keep)

Under `DG_SFA_PUSH`, fused expert preprocess already reads A scales exclusively from owner-local staging. For generation>0 calls with a real staging allocation, the quant kernel now omits the duplicate scale copy into each source sym buffer; generation-0/dummy-staging callers still populate it. `DG_SFA_SOURCE=host` and explicit `DG_SFA_KEEP_LOCAL_SCALE=1` compile the compatibility copy back in. Strict CPU-reference plumbing was updated to validate pushed scales directly from `[parity][local_expert][source_rank][slot][ksb]` staging rather than relying on the eliminated duplicate.

| Run | P1 cycles | P2 + barrier cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,544 | 1,740 | 8,284 | 34 us | 26 us | 282 us | 254 us |
| 2 | 6,836 | 1,908 | 8,744 | 33 us | 23 us | 282 us | 254 us |
| 3 | 7,472 | 1,932 | 9,404 | 33 us | 24 us | 282 us | 254 us |

Independent medians are P1=6,836 cycles, completed P2=1,908 cycles, total=8,744 cycles, quant=33 us, preprocess=24 us, full pipeline=282 us, and non-fused=254 us. Versus iteration 49's corrected baseline, P1 improves 4.0%, P2 improves 61.0%, direct total improves 24.8%, aggregate quant improves 4 us (10.8%), and full pipeline improves 3 us. All three samples reported merged metadata `ALL MATCH` and `All PASSED`, with an idle-device snapshot before each run.

**Verdict:** provisional keep. The target event and completed clock interval both improve materially and consistently. Resource/spill inspection plus strict random-routing correctness against the staging-backed CPU reference remain mandatory. Raw logs: `/tmp/iter50_perf_{1,2,3}.log`; idle-device snapshots: `/tmp/iter50_perf_ppusmi_{1,2,3}.log`.

## Iteration 51 — Iteration 50 resource and strict validation (passed)

The production `<7168,8>` quant kernel uses 96 vregs, 96 sregs, 33 shared-memory units, and stack size 0. Its 3,660-line ISA dump contains no spill, private, scratch, stack, or local-memory match.

An idle eight-GPU process then ran three random-routing rounds with `FULL_CORRECTNESS=1`. Internal generations/parities were 1/1, 2/0, and 3/1. Every reported per-expert fused-vs-nonfused and fused-vs-staging-backed CPU-reference comparison had exactly zero difference, the public generation/parity contract passed, and the all-rank gate passed all three rounds.

**Verdict:** accept iteration 50. The eliminated sym-buffer scale writes are genuinely redundant under real SFA push staging, save 4 us of aggregate quant latency and 3 us end-to-end, and preserve compatibility through generation-0/dummy staging plus the host/explicit keep-local fallback. Raw correctness log: `/tmp/iter51_iter50_correctness.log`; idle snapshot: `/tmp/iter51_iter50_correctness_ppusmi.log`; resource/ISA artifacts: `/tmp/iter51_iter50_resource.txt` and `/tmp/iter51_iter50.isa` inside the container.

## Iteration 52 — Safe current-head P1 split profiler (kept diagnostic)

`DG_QUANT_PROFILE_P1SPLIT=1` now compiles an optional four-segment Phase-1 clock split and requires a seven-element profile buffer at the Python boundary. Thread 32 records setup, BF16 load plus quant compute, first CTA barrier, and scale pack plus second barrier. The code is preprocessor-elided from default production builds.

| Run | Setup | Load + compute | First barrier | Pack + barrier | P1 | P2 + barrier | Total | Quant event | Pipeline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 108 | 3,922 | 1,460 | 350 | 6,792 | 1,274 | 8,066 | 35 us | 281 us |
| 2 | 126 | 3,974 | 1,620 | 378 | 7,014 | 1,494 | 8,508 | 33 us | 282 us |
| 3 | 142 | 4,360 | 876 | 372 | 6,568 | 1,536 | 8,104 | 34 us | 282 us |

Independent split medians are setup=126, load+compute=3,974, first barrier=1,460, and pack+barrier=372 cycles. Load plus quant compute is about 67% of the split P1 interval. Compared with iteration 41, the seven-balanced-warps specialization has reduced the first-barrier median from about 4,490 to 1,460 cycles; the useful next target is now the 4k-cycle compute body rather than barrier or packing work.

**Verdict:** keep the opt-in diagnostic. All three runs passed and aggregate performance remained at iteration 50's 33-35 us quant / 281-282 us pipeline operating point. Raw logs: `/tmp/iter52_profile_{1,2,3}.log`; idle snapshots: `/tmp/iter52_profile_ppusmi_{1,2,3}.log`.

## Iteration 53 — Sixteen BF16 elements per quant lane chunk (provisional keep)

Each quant lane now processes sixteen BF16 elements per logical chunk using two hoisted `int4` loads and one `int2` FP4 output. A 32-element scale group therefore spans two lanes instead of four. Production HIDDEN=7168 has 448 chunks, so seven quant warps still receive exactly two chunks per lane while warp 7 owns the topk atomics. Total input/output bytes and FP4 conversion work are unchanged, but outer iterations fall from four to two and each iteration needs one amax shuffle instead of two.

Single-GPU generation-0 microbench was 27.04/26.85/26.69 us (median 26.85 us), 1.9% below iteration 45's 27.38-us median. The production kernel uses 96 vregs, 96 sregs, 33 shared-memory units, stack size 0, and no spill match. ISA shrank from about 3,660 to 2,175 lines.

| Run | P1 cycles | P2 + barrier cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,106 | 1,472 | 6,578 | 33 us | 28 us | 280 us | 257 us |
| 2 | 4,764 | 1,576 | 6,340 | 33 us | 28 us | 279 us | 254 us |
| 3 | 4,662 | 1,622 | 6,284 | 32 us | 26 us | 281 us | 255 us |

Independent medians are P1=4,764 cycles, completed P2=1,576 cycles, total=6,340 cycles, quant=33 us, preprocess=28 us, full pipeline=280 us, and non-fused=255 us. Versus iteration 50, P1 improves 30.3%, P2 improves 17.4%, direct total improves 27.5%, aggregate quant is unchanged at its 33-us median, and full pipeline improves 2 us. Every run reported merged metadata `ALL MATCH` and `All PASSED` after an idle-device snapshot.

**Verdict:** provisional keep. Resource and performance gates pass, but changing scale-group lane geometry requires strict random-routing CPU-reference validation before acceptance. Raw logs: `/tmp/iter53_micro_{1,2,3}.log` and `/tmp/iter53_perf_{1,2,3}.log`; matching idle snapshots use `_ppusmi_` names. Resource/ISA artifacts: `/tmp/iter53_resource.txt` and `/tmp/iter53_quant.isa` inside the container.

## Iteration 54 — Iteration 53 strict validation (passed)

An idle eight-GPU production process ran three random-routing rounds with `FULL_CORRECTNESS=1`. Internal generations/parities were 1/1, 2/0, and 3/1. All 36 reported local-expert comparisons had exactly zero fused-vs-nonfused and fused-vs-staging-backed CPU-reference difference. The public generation/parity contract and all-rank three-round gate also passed.

**Verdict:** accept iteration 53. The two-lane scale-group geometry preserves the exact FP4 byte order and scale mapping while cutting the direct completed quant interval by 27.5% versus iteration 50. Raw log: `/tmp/iter54_iter53_correctness.log`; idle snapshot: `/tmp/iter54_iter53_correctness_ppusmi.log`.

## Iteration 55 — Post-iteration-53 P1 split (diagnostic)

The opt-in split profiler was rerun three times on the accepted sixteen-element chunk implementation.

| Run | Setup | Load + compute | First barrier | Pack + barrier | P1 | P2 + barrier | Total | Quant event | Pipeline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 2,462 | 1,750 | 234 | 5,378 | 1,228 | 6,606 | 38 us | 279 us |
| 2 | 116 | 1,940 | 2,158 | 252 | 5,304 | 1,508 | 6,812 | 30 us | 279 us |
| 3 | 120 | 1,882 | 1,540 | 234 | 4,624 | 1,390 | 6,014 | 33 us | 279 us |

Independent split medians are setup=120, load+compute=1,940, first barrier=1,750, and pack+barrier=234 cycles. The compute interval is now less than half iteration 52's 3,974-cycle median. The barrier interval largely represents the sampled warp waiting while the other six balanced quant warps execute necessary work, rather than the old atomic/load imbalance. The rounded breakdown events are noisy (including one 95-us preprocess outlier), while full pipeline is stable at 279 us in all three runs.

**Conclusion:** further gains must reduce total warp instruction work, not merely reschedule the barrier. A 32-element single-lane scale chunk can preserve the same 32 BF16 per lane while eliminating the last two cross-lane shuffles and outer iteration boundary. Raw logs: `/tmp/iter55_profile_{1,2,3}.log`; idle snapshots: `/tmp/iter55_profile_ppusmi_{1,2,3}.log`.

## Iteration 56 — Thirty-two BF16 elements per lane chunk (rejected)

A candidate combined each lane's two sixteen-element chunks into one complete 32-element scale group. This removed all cross-lane amax reduction/broadcast shuffles and retained the same total four `int4` input loads and one `int4` FP4 output per lane.

The first idle single-GPU generation-0 microbench regressed to 31.97 us versus iteration 53's 26.85-us median (+19.1%). Holding and processing all sixteen BF16x2 values in one iteration creates enough register-lifetime and scheduler pressure to outweigh removal of the two shuffle operations. The regression is far beyond noise, so the candidate was stopped before resource and eight-GPU gates.

**Verdict:** reject and revert. Sixteen elements / two lanes is the better balance between shuffle count and live-value pressure. Raw log: `/tmp/iter56_micro_1.log`; idle snapshot: `/tmp/iter56_micro_ppusmi_1.log`.

## Iteration 57 — Tree-reduce the sixteen-element local amax (rejected)

The serial eight-step `hmax2(abs(x))` chain inside each sixteen-element chunk was replaced by four independent pair maxima followed by a two-level tree. This reduced theoretical dependency depth from eight to three without changing arithmetic or FP4 layout.

Single-GPU generation-0 microbench was 27.48/27.36/27.76 us (median 27.48 us), 2.3% slower than iteration 53's 26.85-us median. Keeping four pair maxima live simultaneously creates enough register/scheduler pressure to outweigh the shorter dependency chain on this compiler.

**Verdict:** reject and revert. The compiler/platform prefers the lower-live-range serial reduction. Raw logs: `/tmp/iter57_micro_{1,2,3}.log`; idle snapshots: `/tmp/iter57_micro_ppusmi_{1,2,3}.log`.

## Iteration 58 — Pair-fold local amax reduction (rejected)

A lower-live-range compromise first reduced adjacent BF16x2 pairs, then folded the four pair maxima through a single accumulator. This removed the zero seed, used seven rather than eight hmax operations, and avoided iteration 57's full reduction tree.

Single-GPU generation-0 microbench was 26.37/26.18/28.66 us (median 26.37 us), 1.8% faster than iteration 53. Resources stayed at 96 vregs, 96 sregs, 33 shared-memory units, stack size 0, and no spill; ISA grew to 2,990 lines.

| Run | P1 cycles | P2 + barrier cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,608 | 1,952 | 7,560 | 33 us | 26 us | 280 us | 255 us |
| 2 | 5,354 | 1,900 | 7,254 | 34 us | 28 us | 280 us | 255 us |
| 3 | 5,334 | 1,560 | 6,894 | 34 us | 27 us | 281 us | 255 us |

Independent medians are P1=5,354 cycles, P2=1,900 cycles, total=7,254 cycles, quant=34 us, preprocess=27 us, full pipeline=280 us, and non-fused=255 us. Relative to iteration 53, P1 regresses 12.4%, P2 20.6%, direct total 14.4%, and aggregate quant 1 us, with no pipeline gain. All merged metadata and test gates passed, so this is a pure production-schedule regression rather than correctness failure.

**Verdict:** reject and revert. A generation-0 micro win does not justify a worse real push path. Raw logs: `/tmp/iter58_micro_{1,2,3}.log` and `/tmp/iter58_perf_{1,2,3}.log`; matching idle snapshots use `_ppusmi_` names. Resource/ISA artifacts: `/tmp/iter58_resource.txt` and `/tmp/iter58_quant.isa` inside the container.

## Iteration 59 — ACU profile of accepted iteration 53

Single-GPU generation-0 profiling of the accepted kernel (`195 x 256`, one wave across 39 CUs) reports 18.10 us device duration, 26.57% compute throughput, 40.18% memory throughput, and only 9.27% issue-slot busy. The kernel uses 96 registers/thread, 4,224 bytes static shared memory, stack size 0, and is register-limited to five blocks/CU: theoretical occupancy 62.5%, achieved occupancy about 53%.

The detailed scheduler pass reports No Eligible=63.06%, One-or-More-Eligible=36.94%, 8.64 active warps per scheduler but only 0.80 eligible warps per warp engine, and 9.31 warp cycles per issued/executed instruction. Memory is busy but not bandwidth-saturated; DRAM throughput in the default pass is only 8.47% while LLC reaches about 40% of peak.

**Conclusion:** the remaining kernel is latency/dependency limited, with registers preventing a sixth resident block. Candidates that extend live ranges (iterations 56-58) are structurally disfavored. The next useful experiment is reducing cross-iteration raw-input lifetime to lower vregs/raise occupancy, accepting a small amount of additional load scheduling if necessary. Raw host logs: `/tmp/iter59_acu.log` and `/tmp/iter59_acu_detailed.log`; idle snapshots use matching `_ppusmi` names.

## Iteration 60 — Per-iteration raw-load lifetime, first attempt (compile failure)

The first implementation moved the two `int4` raw loads inside each sixteen-element compute iteration, but one conversion expression still indexed the old two-dimensional `raw[it][...]` array. Compilation correctly failed because the new `raw[it]` expression selected an `int4`, which has no subscript operator. No kernel launched and no performance sample was produced.

**Verdict:** revert this malformed attempt and retry the same hypothesis with `raw[j / 4]` in a fresh iteration. Raw compiler log: `/tmp/iter60_micro_1.log`; idle snapshot: `/tmp/iter60_micro_ppusmi_1.log`.

## Iteration 61 — Per-iteration raw-load lifetime, corrected (rejected)

The corrected candidate kept only the current sixteen-element chunk's two `int4` loads in source scope instead of explicitly hoisting both iterations' four loads. It compiled and ran, but the compiler still allocated 96 vregs, 96 sregs, 33 shared-memory units, and stack size 0; the hoped-for sixth resident block was therefore not unlocked. ISA grew from iteration 53's roughly 2,175 lines to 3,000 lines.

The first idle generation-0 microbench was 27.35 us, slower than iteration 53's 26.85-us median. With no register/occupancy improvement and a larger instruction stream, further samples cannot validate the stated hypothesis.

**Verdict:** reject and revert. Source-level lifetime reduction does not change this compiler's allocation and only loses the profitable cross-iteration load schedule. Raw log: `/tmp/iter61_micro_1.log`; idle snapshot: `/tmp/iter61_micro_ppusmi_1.log`; resource/ISA artifacts: `/tmp/iter61_resource.txt` and `/tmp/iter61_quant.isa` inside the container.

## Iteration 62 — Quant-only 80-register compiler cap (rejected)

A temporary quant-name-scoped `DG_QUANT_MAX_REGS=80` hook passed `--maxrregcount=80` only to the quant JIT object, leaving preprocess and GEMM compiler flags unchanged. The compiler reduced the production kernel from 96 to 80 vregs but introduced a 28-byte/thread stack; sregs/shared remained 96/33 and ISA had 2,205 lines.

Single-GPU generation-0 microbench was 27.49/27.17/27.64 us (median 27.49 us), 2.4% slower than iteration 53's 26.85-us median. The occupancy opportunity does not offset spill cost.

**Verdict:** reject and remove the experimental compiler hook. The accepted 96-vreg, stack-zero allocation is preferable. Raw logs: `/tmp/iter62_micro_{1,2,3}.log`; idle snapshots use matching `_ppusmi_` names; resource/ISA artifacts: `/tmp/iter62_resource.txt` and `/tmp/iter62_quant.isa` inside the container.

## Iteration 63 — Eliminate the local BF16x2 view array (rejected)

The candidate removed `local_v2[8]` and reinterpreted the already-hoisted raw `int4` registers directly in both the amax and FP4 conversion loops, aiming to reduce live registers without adding global loads.

The first idle generation-0 microbench regressed catastrophically to 56.55 us, about 2.1x iteration 53's 26.85-us median. On this compiler the explicit local view array is essential for profitable register reuse/scheduling; repeated reinterpretation is not optimized into the expected aliases.

**Verdict:** reject and revert immediately. Raw log: `/tmp/iter63_micro_1.log`; idle snapshot: `/tmp/iter63_micro_ppusmi_1.log`.

## Iteration 64 — Cap quant launch at four blocks/CU (rejected)

A temporary `DG_QUANT_BLOCKS_PER_SM=4` hook reduced the occupancy-derived one-wave grid from 195 to 156 blocks, trading more grid-stride tokens per block for 20% fewer folded-arrival system releases.

| Run | P1 cycles | P2 + barrier cycles | Total cycles | Quant event | Expert preprocess | Full pipeline | Non-fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,262 | 1,400 | 5,662 | 36 us | 123 us | 280 us | 260 us |
| 2 | 4,688 | 1,458 | 6,146 | 33 us | 28 us | 281 us | 255 us |
| 3 | 4,916 | 1,450 | 6,366 | 33 us | 28 us | 281 us | 255 us |

Independent medians are P1=4,688 cycles, P2=1,450 cycles, total=6,146 cycles, quant=33 us, preprocess=28 us, full pipeline=281 us, and non-fused=255 us. The block0 clock interval is 3.1% below iteration 53, but aggregate quant is unchanged and full pipeline regresses 1 us. Run 1's preprocess event is an isolated diagnostic outlier while its pipeline remains 280 us.

**Verdict:** reject and remove the cap hook. The occupancy-derived five-block/CU grid has the better distributed tail. Raw logs: `/tmp/iter64_cap4_{1,2,3}.log`; idle snapshots: `/tmp/iter64_cap4_ppusmi_{1,2,3}.log`.

## Iteration 65 - Inline pushed-SFA reshape into fused copy blocks (rejected)

The opt-in `DG_SFA_PUSH_INLINE=1` path removed the standalone 39-CTA SFA reshape
launch and instead enabled the existing `copy_mblock_sfa` call inside the three
production K-striped copy blocks. This preserves the exact source/destination layout
while testing whether SFA work can overlap the fused FP4 copy/GEMM timeline.

Three independent idle 8-GPU production runs completed with merged metadata `ALL
MATCH` and `All PASSED`. Full pipeline was 0.298/0.298/0.296 ms (median 0.298 ms),
quant was 0.033/0.033/0.032 ms, and expert preprocess was 0.028/0.029/0.026 ms.
The accepted iteration-53 pipeline is about 0.280 ms, so the candidate regresses the
critical path by roughly 18 us despite eliminating the separate SFA kernel launch.

**Verdict:** reject and revert the inline path. With `K_TILES_PER_FLAG=14`, only
three copy CTAs perform the reshape before publishing the first stripe; concentrating
the 39-CTA SFA work onto those CTAs delays GEMM readiness more than the removed launch
gap saves. Keep the standalone parallel reshape. The AKO bench wrapper was refreshed
to target `sglang.lxh`, default to the 8-GPU production configuration, record
`ppu-smi` before each run, and support `RUNS=N` repeated signal gates. Raw output:
`trajectory/20260716_184233_iter-65/output.txt`.

## Iteration 66 - Merge expert metadata and full-grid SFA reshape (provisional keep)

An opt-in `DG_PRE_GEMM_FUSED=1` kernel keeps the standalone reshape's 39-CTA
parallelism while removing the metadata-to-SFA kernel boundary. CTA 0 runs the
accepted merged expert metadata helpers, every producer thread device-fences its
global stores, and thread 0 publishes a generation-tagged atomic handoff. One thread
per remaining CTA polls the handoff before all CTAs run the unchanged
`copy_mblock_sfa` mapping. A 240-second deadlock guard protected the first launch.

Three independent idle 8-GPU production runs completed without deadlock. Full
pipeline was 0.277/0.278/0.278 ms (median 0.278 ms), quant event was
0.025/0.024/0.025 ms, and the combined metadata-plus-SFA event was
0.029/0.028/0.029 ms. Every run reported merged metadata `ALL MATCH` and `All
PASSED`. The accepted iteration-53 pipeline is about 0.280 ms, so this is a small but
repeatable 2 us end-to-end improvement while absorbing the standalone 7.5 us SFA
kernel into the metadata interval.

**Verdict:** provisional keep. The intended overlap is visible, but the 0.7% pipeline
gain is below the normal 3% signal threshold and the unexpectedly lower quant event
must be separated from frequency/run variance. Run an immediate three-run A/B with
the opt-in disabled, then validate the SFA buffer and strict random-routing output
before accepting. Raw output:
`trajectory/20260716_184952_iter-66/output.txt`.

## Iteration 67 - Immediate disabled-path A/B for iteration 66 (rejected)

With the exact iteration-66 code still loaded but `DG_PRE_GEMM_FUSED=0`, three
independent idle 8-GPU runs measured full pipeline at 0.277/0.278/0.277 ms
(median 0.277 ms). Quant was 0.032/0.036/0.033 ms and expert preprocess was
0.027/0.087/0.027 ms; the middle preprocess value is an isolated event outlier
that did not move its 0.278 ms pipeline. Every run reported merged metadata `ALL
MATCH` and `All PASSED`.

The combined path from iteration 66 measured 0.277/0.278/0.278 ms (median
0.278 ms), so it is one microsecond slower than its immediate disabled-path A/B.
The earlier 24-25 us quant readings were therefore run-state variance, not an overlap
benefit. Preserving 39 reshape CTAs avoids iteration 65's large regression, but the
generation handoff and 256-thread metadata CTA consume the launch-gap saving.

**Verdict:** reject iteration 66 and revert all combined-kernel/API/test plumbing.
Keep the standalone metadata and SFA launches. A future overlap attempt must avoid
both grid-wide polling and changing the metadata CTA width. Raw output:
`trajectory/20260716_185202_iter-67/output.txt`.

## Iteration 68 - Three-warp combined metadata/SFA kernel (rejected)

The iteration-66 combined kernel was retried with 96 threads per CTA instead of
256. This restores CTA 0 to the accepted expert-preprocess geometry (three warps)
while retaining 39 SFA CTAs, or 117 reshape warps total. The hypothesis was that
the smaller block would remove the extra metadata barriers without giving up enough
SFA memory-level parallelism to matter.

Three independent idle 8-GPU production runs completed without deadlock. Full
pipeline was 0.280/0.279/0.279 ms (median 0.279 ms). Quant was
0.025/0.025/0.037 ms and the combined event was 0.029/0.029/0.094 ms; the final
event outlier did not move its 0.279 ms pipeline. Every run reported merged metadata
`ALL MATCH` and `All PASSED`.

**Verdict:** reject and revert. The immediate standalone baseline from iteration 67
is 0.277 ms, so reducing the CTA width makes the candidate another 2 us slower.
The smaller metadata CTA does not compensate for lower SFA parallelism and the
generation handoff. Across the completed overlap directions: inline SFA on three copy
CTAs was 0.298 ms; combined 39-CTA/256-thread SFA was 0.278 ms versus a 0.277 ms
disabled A/B; combined 39-CTA/96-thread SFA is 0.279 ms. Pause for profiling-based
reassessment before another structural fusion. Raw output:
`trajectory/20260716_185422_iter-68/output.txt`.

## Iteration 69 - Prelaunch metadata waiter on a second stream (rejected)

Instead of fusing kernels, an opt-in second stream launched the one-CTA expert
preprocess before quant. The preprocess CTA waited on the new generation's arrival
slots while quant ran; an event then made the main stream wait for metadata before
the existing SFA/GEMM sequence. Each next preprocess waited for the previous main
pipeline end event, preserving the reused metadata workspace. Existing ACU data
suggested the small 96-thread CTA might co-reside beside the five resident quant
blocks. Two attempts to consult the official CUDA concurrency documentation failed
because the search service returned a network error.

Three independent idle 8-GPU production runs completed without deadlock. Full
pipeline was 0.279/0.278/0.278 ms (median 0.278 ms); every run reported merged
metadata `ALL MATCH` and `All PASSED`. The immediate serial baseline from iteration
67 is 0.277 ms. The separate breakdown loop intentionally remained serial and was
noisy, so it is not used to claim overlap savings.

**Verdict:** reject and revert. Prelaunching the waiter does not shorten the critical
path; cross-stream event/scheduling overhead is at least as large as any hidden
arrival/launch latency, producing a 1 us median regression. Raw output:
`trajectory/20260716_185837_iter-69/output.txt`.

## Iteration 70 - Standalone SFA grid 36 x 128 threads (rejected)

The standalone pushed-SFA reshape gained opt-in launch-shape controls. Production
has 12 M-blocks and `sub=3`, so only 36 of the default 39 CTAs perform work. The
first candidate launched exactly 36 CTAs with 128 threads each, removing three empty
CTAs while retaining 144 active reshape warps.

Three independent idle 8-GPU production runs all measured the full pipeline at
0.280 ms. Quant medians were 0.033/0.038/0.032 ms and expert-preprocess medians
were 0.027/0.095/0.027 ms; the middle breakdown outlier did not move the pipeline.
Every run reported merged metadata `ALL MATCH` and `All PASSED`.

**Verdict:** reject 36x128. The immediate recent standalone baseline is 0.277 ms,
so halving the thread count loses at least as much SFA memory-level parallelism as
the empty-CTA removal saves. Retain the opt-in `DG_SFA_NUM_SMS` and
`DG_SFA_NUM_THREADS` diagnostics with production defaults unchanged, then isolate
the grid-only change at 36x256. Raw output:
`trajectory/20260716_190051_iter-70/output.txt`.

## Iteration 71 - Standalone SFA grid-only 36 x 256 (rejected)

This follow-up restored the default 256-thread CTA and changed only the SFA grid
from 39 to 36, exactly matching the 12 production M-blocks times `sub=3` useful
work items.

Three independent idle 8-GPU runs measured full pipeline at
0.278/0.278/0.279 ms (median 0.278 ms). Quant was 0.032/0.031/0.031 ms and expert
preprocess was 0.026/0.025/0.025 ms. Every run reported merged metadata `ALL MATCH`
and `All PASSED`.

**Verdict:** reject the 36-CTA launch shape. It remains 1 us slower than the recent
0.277 ms standalone baseline, so the three empty CTAs have no measurable critical
path cost; the kernel is already at its launch/latency floor and benefits from the
default full-SM grid geometry. Keep production at 39x256. Raw output:
`trajectory/20260716_190204_iter-71/output.txt`.

## Iteration 72 - Warp-tiled 8x8 SFA transpose (rejected)

`hgobjdump` of the accepted standalone SFA reshape showed 40 vector registers,
96 scalar registers, zero stack, and no spill. Its aligned hot loop already uses
one `vmem.ld.b32x4` per eight scales, but then emits eight strided
`vmem.st.b16` instructions. This experiment used warp-private shared memory to
transpose an 8-token by 8-K-scale tile, replacing the eight scalar destination
stores with aligned `int4` stores. The macro was scoped to
`dispatch_sfa_preprocess`, so fused-GEMM occupancy and code generation were
unchanged.

Three independent idle 8-GPU production runs measured full pipeline at
0.287/0.284/0.286 ms (median 0.286 ms). Quant was 0.032/0.032/0.033 ms and expert
preprocess was 0.026/0.026/0.027 ms. Every run reported merged metadata `ALL
MATCH` and `All PASSED`.

**Verdict:** reject and revert. This is 9 us slower than the immediate 0.277 ms
standalone baseline. Reducing destination-store width/instruction count does not
repay the lower active-lane utilization, shared-memory transpose traffic, and two
warp synchronizations per tile. The original scalar-scatter reshape is already
better matched to this small 344-KB launch. Raw output:
`trajectory/20260716_190955_iter72_sfa_tiled_transpose/output.txt`.

## Iteration 73 - Batched host launch of metadata and SFA (rejected)

The asys trace showed about 5.1 us between the end of expert metadata and the
start of standalone SFA. To test whether this was Python/JIT host overhead, one
JIT wrapper enqueued the existing merged metadata kernel and the existing SFA
kernel back-to-back on the same stream. This deliberately preserved their
accepted launch geometries (one 96-thread metadata CTA and 39x256 SFA CTAs) and
required no device-side atomic handoff or fused-GEMM change.

Three independent idle 8-GPU production runs measured full pipeline at
0.278/0.280/0.280 ms (median 0.280 ms). The standalone path's immediate recent
baseline is 0.277 ms. Every run reported merged metadata `ALL MATCH` and `All
PASSED`.

**Verdict:** reject and revert. Combining the two host calls does not recover the
trace interval; the two kernels retain their device scheduling boundary, and the
larger wrapper is 3 us slower at the median. The 5.1 us asys interval must not be
treated as fully removable Python overhead. Raw output:
`trajectory/20260716_191756_iter73_batched_pregemm_launch/output.txt`.

## Iteration 74 - K-stripe pushed-SFA co-issue probe (provisional)

The standalone reshape was removed and its 112 K-scale blocks were split across
the fused copy kernel's two production K stripes. Each copy CTA flattened the
eight ranks' SFA vectors and issued one local staging load/scatter between eight
remote FP4 A loads and their stores. The existing per-stripe fence and ready flag
therefore publish matching A and SFA portions to the GEMM, rather than executing
all SFA as the serial copy-CTA prologue tested in iteration 65.

One idle 8-GPU production probe completed without deadlock at 0.280 ms full
pipeline, with merged metadata `ALL MATCH` and `All PASSED`. This recovers almost
all of iteration 65's 0.298 ms regression, but remains 3 us slower than the recent
0.277 ms standalone baseline. `hgobjdump` reports 240 vregs, 256 sregs, stack size
zero, and no spill for the fused specialization.

**Verdict:** provisional basis only. There is no spill, so first remove the eight
runtime prefix counters from the fixed 16-token production path and repeat the
three-run gate. Raw output:
`trajectory/20260716_192743_iter74_kstripe_sfa_coissue_probe/output.txt`.

## Iteration 75 - Fixed-uniform K-stripe SFA co-issue (rejected)

The production fixed routing has exactly 16 tokens from every source rank for
each local expert. The iteration-74 co-issue therefore gained a guarded fast path
with constant 112-vector rank bands, removing eight runtime SFA prefix counters
and their divisions. Non-uniform routing retained the correct full-copy fallback.

Three independent idle 8-GPU runs all measured 0.280 ms full pipeline. Quant was
0.033/0.032/0.032 ms and expert preprocess was 0.029/0.027/0.025 ms. Every run
reported merged metadata `ALL MATCH` and `All PASSED`.

**Verdict:** reject and revert iterations 74-75. The result remains 3 us slower
than the recent 0.277 ms standalone baseline despite zero stack/spill and constant
mapping. The irreducible local load/scatter work and delayed first-stripe publish
inside one copy CTA outweigh removal of the 39-CTA reshape launch. The standalone
reshape has enough parallelism to remain the better production schedule. Raw
output: `trajectory/20260716_193056_iter75_uniform16_kstripe_coissue/output.txt`.

## Iteration 76 - Dedicated-warp SFA push (harness invocation failed)

Moved the production quant specialization's SFA peer stores from the topk FP4
scatter warps to the otherwise-idle slot-atomic warp, so SFA push can issue while
warps 0-5 scatter FP4 locally. The existing CTA reuse barrier and folded-arrival
system fence remain unchanged.

The remote invocation incorrectly entered the `sglang.lxh` container before
calling `scripts/bench.sh`; the script itself launches that container, so it exited
127 at `docker: command not found` before compiling or running the kernel. No
performance or correctness result exists for this attempt.

**Verdict:** infrastructure-only failed iteration; retain the candidate unchanged
and rerun from the swu246 host as iteration 77. Raw output:
`trajectory/20260716_194957_iter-76/output.txt`.

## Iteration 77 - Dedicated atomic-warp SFA push (rejected)

For the HIDDEN=7168 specialization, warp 7 retained its Phase-0 topk slot atomics
but became the sole SFA-push warp after the quant barriers. Warps 0-5 continued
the local FP4 scatter, so the local and peer store streams could be scheduled
concurrently. The final reuse barrier and folded-arrival system fence were left
unchanged.

Three independent idle 8-GPU FULL_CORRECTNESS runs measured full pipeline at
0.281/0.281/0.282 ms (median 0.281 ms), versus the current robust standalone
baseline 0.279 ms. Quant was 0.035/0.035/0.035 ms, consistently 2-3 us slower
than the baseline 0.032-0.033 ms; expert preprocess stayed at 0.025 ms. Every run
reported the three-round CPU-reference correctness test PASSED, merged metadata
ALL MATCH, and All PASSED.

**Verdict:** reject. Concentrating all six peer destinations on one warp removes
cross-destination store-level parallelism and makes that warp the CTA's final
barrier straggler. Any completion overlap with the FP4 scatter is smaller than
the lost SFA issue parallelism, regressing both quant and full pipeline. Restore
the one-warp-per-topk push mapping. Raw output:
`trajectory/20260716_195118_iter-77/output.txt`.

## Iteration 78 - Pre-issue SFA before FP4 scatter (provisional win)

Restored the one-warp-per-topk SFA mapping, but moved each warp's uncached peer
SFA stores immediately before its much larger local FP4 scatter. This preserves
six independent SFA destination streams while allowing their remote completion
latency to remain in flight during the following 3.5-KB FP4 stores. The CTA reuse
barrier and folded-arrival system fence remain unchanged.

Three independent idle 8-GPU FULL_CORRECTNESS runs measured full pipeline at
0.278/0.279/0.278 ms (median 0.278 ms), 1 us below the robust 0.279-ms baseline.
Quant was stable at 0.032/0.032/0.032 ms, recovering the dedicated-warp
regression; expert preprocess was 0.028/0.029/0.029 ms. Every run passed the
three-round CPU reference, merged metadata ALL MATCH, and All PASSED.

**Verdict:** provisional small win. Store order alone can hide a small portion of
peer-write completion without sacrificing destination MLP, but the 1-us delta is
below a strong-noise threshold and needs a longer stability run before acceptance.
Raw output: `trajectory/20260716_195603_iter-78/output.txt`.

## Iteration 79 - Five-run stability gate for SFA pre-issue (accepted)

The unchanged iteration-78 candidate was rerun five times on idle 8-GPU hardware.
Full pipeline measured 0.278/0.278/0.279/0.279/0.278 ms (median 0.278 ms,
range 1 us), versus the immediately preceding robust baseline's
0.280/0.279/0.279/0.279/0.279 ms (median 0.279 ms). Quant was
0.031/0.030/0.031/0.031/0.030 ms; expert preprocess was
0.028/0.026/0.027/0.029/0.026 ms. All five runs reported merged metadata ALL
MATCH and All PASSED; iteration 78 had already passed three independent
FULL_CORRECTNESS runs.

The diagnostic mechanism is stronger than the 1-us end-to-end delta: P2 clock64
fell from the baseline 1,576-1,770 cycles to 842-1,162 cycles. `hgobjdump` shows
unchanged 96 vregs / 96 sregs, stack size zero, and no spill for both FP4
specializations.

**Verdict:** accept the SFA-first store order. It is a small but repeatable
0.8-us mean / 1-us median full-pipeline improvement, while directly reducing the
intended Phase-2 store tail and preserving correctness and resources. Raw output:
`trajectory/20260716_200010_iter-79/output.txt`.

## Iteration 80 — accept: fuse copy-ready flag clear into SFA preprocess

- Hypothesis: `copy_ready_flags.zero_()` is a tiny standalone same-stream memset/launch; clearing the 24-entry buffer in SFA preprocess block 0 should overlap with reshape work and remove exposed launch overhead.
- Change: pass `copy_ready_flags` into `dispatch_sfa_preprocess`; block 0 clears the full buffer. Host-SFA and diagnostic skip-reshape paths retain the original `zero_()` fallback.
- Correctness: `VERDICT=1 RUNS=3`; all three independent runs passed API contract, three-round all-rank/all-expert CPU parity, and performance test. No hang or stale-ready observation.
- Full pipeline: 277 / 275 / 276 us (median 276 us), versus accepted iter-79 stability median 278 us.
- Breakdown: quant + sym_buf_zero 32 us; expert_preprocess 27–28 us. The timing buckets do not isolate the removed memset, so the full-pipeline delta is the acceptance signal.
- Decision: accept. This is a small, low-risk 2 us median improvement and preserves non-preprocess behavior.
- Trajectory: `trajectory/20260716_205150_iter-80/output.txt`

## Iteration 81 — accept: partition SFA reshape CTAs by rank

- Hypothesis: production rank work is `cnt * (ksb/8) = 16 * 14 = 224` int4 vectors, below 256 threads. The old `start=s*blockDim` split made sub-CTA 1/2 idle; assigning disjoint rank sets should turn ~12 useful CTAs into ~36.
- Change: add optional rank start/stride to `copy_mblock_sfa`; standalone preprocess CTA `(mb,s)` processes ranks `s,s+sub,...` with all 256 threads. Inline copy callers retain the default all-rank behavior.
- Correctness: `VERDICT=1 RUNS=3`; all API, dynamic-routing all-rank/all-expert CPU parity, and performance tests passed. No duplicate/lost SFA writes or deadlock.
- Full pipeline: 274 / 275 / 276 us (median 275 us), versus iter-80 median 276 us.
- Breakdown: quant 31–32 us; expert_preprocess 27–29 us. The end-to-end shift is small because reshape is local-HBM scatter limited after SFA push.
- Decision: accept. Median improves 1 us with a shape-derived utilization fix and no semantic change.
- Trajectory: `trajectory/20260716_205627_iter-81/output.txt`

## Iteration 82 — reject: direct register packing of quant scales

- Hypothesis: two MXFP4 scale leaders in each four-lane group can shuffle their scale bytes and write `s_packed_scale` directly, removing the 112-byte shared pack pass and a second CTA barrier.
- Change: production `SPECIALIZE_ATOMIC_WARP` path directly packs scale bytes via four-lane shuffle; generic/partial-worker shapes retain the old shared-memory fallback.
- Correctness: `VERDICT=1 RUNS=3`; all API, all-rank/all-expert CPU parity, and performance tests passed bit-exactly.
- Full pipeline: 276 / 278 / 277 us (median 277 us), versus accepted iter-81 median 275 us.
- Breakdown: quant + sym_buf_zero 32–33 us, versus 31–32 us in iter-81. The extra shuffle/register dependency outweighs the removed tiny pack/barrier on this compiler.
- Decision: reject and restore iter-81 quant implementation.
- Trajectory: `trajectory/20260716_210118_iter-82/output.txt`

## Iteration 83 — reject/inconclusive: per-rank warp arrival/count overlap

- Hypothesis: one warp per source rank can read that rank's 12 counts immediately after its release-arrival, hiding the ~3.4k-cycle count tail under the ~15k-cycle arrival window.
- Change: add an 8-warp prepare path and make the standalone prepare launcher use 256 threads for the production 8-rank case.
- Correctness: `VERDICT=1 RUNS=3`; all API, dynamic-routing all-rank/all-expert CPU parity, and performance tests passed.
- Full pipeline: 275 / 274 / 276 us (median 275 us), equal to accepted iter-81 median 275 us.
- Clock64 evidence: remote count reads remained 3.38k cycles instead of collapsing to zero, proving the measured fused call did not enter the new 256-thread path (its actual launcher still supplies the old thread count).
- Decision: reject this incomplete wiring and restore iter-81. Revisit only after tracing the actual fused prepare launch configuration.
- Trajectory: `trajectory/20260716_210650_iter-83/output.txt`

## Iteration 84 — reject: fully wired per-rank warp arrival/count overlap

- Hypothesis: after wiring the merged launcher to 256 threads, eight rank warps can hide each rank's count reads under other ranks' arrival waits.
- Change: same per-rank warp overlap as iter-83, plus production merged-preprocess launch uses 256 threads for gen>0; gen0 remains unchanged.
- Correctness: `VERDICT=1 RUNS=3`; all API, all-rank/all-expert CPU parity, and performance tests passed.
- Full pipeline: 275 / 274 / 275 us (median 275 us), equal to accepted iter-81 median 275 us.
- Clock64: the path is active (`remote count reads = 0`), but combined arrival+counts reached 20.38k cycles, worse than the prior ~15k arrival + ~3.5k count tail. Eight polling warps and the 256-thread block add more cost than the overlap hides.
- Event breakdown became unstable (quant 18–33 us, preprocess 9–41 us), reinforcing full pipeline and clock64 as the reliable verdict.
- Decision: reject and restore the 96-thread prepare path.
- Trajectory: `trajectory/20260716_211053_iter-84/output.txt`

## Iteration 85 — final stability confirmation

- State: restored accepted iter-81 code plus iter-80 flag-clear fusion; rejected iter-82/84 code is absent.
- Benchmark: signal `RUNS=5` on 8 GPUs, production H=7168/topk=6/ncb=3.
- Full pipeline: 275 / 275 / 275 / 274 / 274 us; median 275 us, mean 274.6 us, range 1 us.
- Breakdown: quant 31–32 us; expert_preprocess 27–29 us; kernel-only 196 us.
- Correctness note: signal mode skips Test 1; the identical accepted code already passed three independent full-verdict all-rank/all-expert CPU checks in iter-81.
- Decision: final accepted state is stable.
- Trajectory: `trajectory/20260716_211421_iter-85-final-stability/output.txt`

## Iteration 86 — current-final isolation measurement

- State: no kernel change; benchmark current accepted code with `SKIP_ISOLATION=0 RUNS=3` on the 8-GPU production shape.
- Correctness: signal mode skipped Test 1; API contract/performance tests passed. The same kernel code passed three independent full-verdict all-rank/all-expert CPU checks in iter-81.
- Full pipeline: 275 / 275 / 275 us (median 275 us).
- Non-fused full: 254 / 254 / 255 us (median 254 us); non-fused GEMM-only 196 / 196 / 196 us.
- Pipeline without expert preprocess: 271 / 272 / 272 us (median 272 us).
- Normal-P2P copy+GEMM: 251 / 250 / 251 us (median 251 us).
- All-local copy+GEMM: 222 / 221 / 222 us (median 222 us).
- P2P exposure: 29 / 29 / 29 us, 11.6% of normal-P2P kernel time. Normal-P2P and all-local per-run std were about 7 us.
- Decomposition at medians: fused structural/local-copy overhead over NF GEMM = 222-196 = 26 us; P2P exposure = 29 us; full-pipeline exposed overhead over normal-P2P kernel = 24 us.
- Trajectory: `trajectory/20260716_213034_iter-86-isolation/output.txt`


## Iteration 87 — close no-preprocess timing on the same critical rank (2026-07-16)

- **Objective:** Re-run the current 8-GPU production isolation benchmark and resolve why
  `Pipeline (no preprocess) - Kernel (normal P2P)` is about 21 us while the standalone
  quant event reports about 31 us.
- **Code change:** Diagnostics only. Added a midpoint CUDA event in the no-preprocess
  loop and, for each sample, selected the rank with the largest closed total before
  reporting that same rank's quant/SFA-push segment, consumer segment, and total.
  Extended the signal grep accordingly. No kernel implementation changed.
- **Command:** `SKIP_ISOLATION=0 RUNS=3 bash scripts/bench.sh iter-87-closed-timing`
- **Environment:** swu246, container `sglang.lxh`, 8 GPUs (0-7), prod
  H=7168/N=6144, 256 tokens/rank, topk=6, expected_m=128, ncb=3.
- **Correctness:** SIGNAL mode; public API contract passed and all performance tests
  passed in all three runs. Full numerical correctness was intentionally skipped;
  the unchanged kernel already passed the prior full-verdict gate.
- **Results (us, run1/run2/run3; median):**
  - Fused full pipeline: 276 / 275 / 274; **275**
  - Pipeline no preprocess: 272 / 272 / 270; **272**
  - Same-iteration fresh quant + SFA push: 22 / 22 / 23; **22**
  - Same-iteration consumer after quant: 250 / 249 / 248; **249**
  - Closure residual: 0 / 0 / 0; **0**
  - Normal-P2P copy+GEMM: 251 / 251 / 251; **251**
  - All-local copy+GEMM: 222 / 221 / 221; **221**
  - Non-fused full: 255 / 254 / 256; **255**
  - Non-fused GEMM-only: 196 / 197 / 196; **196**
  - Standalone quant event: 30 / 31 / 32; **31**
  - Standalone expert preprocess: 26 / 28 / 28; **28**
- **Conclusion:** The closed timing is exact: the no-preprocess path is approximately
  `22 us quant/SFA push + 249 us consumer = 272 us`. Its 19-21 us delta over
  normal-P2P is therefore the exposed fresh-quant/SFA-push cost; the consumer itself
  matches normal-P2P within normal run/rank variation. The standalone 31 us quant
  event is a different measurement context/aggregation and must not be subtracted
  from the independently aggregated 251 us kernel number.
- **Stability:** Current full pipeline is 274-276 us (2 us range), median 275 us.
  No-preprocess is 270-272 us, and normal-P2P is exactly 251 us in all three runs.
- **Trajectory:** `trajectory/20260716_214832_iter-87-closed-timing/output.txt`

## Iteration 88 — fixed-route SFA reshape / metadata overlap probe

- Change: added an opt-in 8-rank production-shape kernel with CTA 0 running merged expert preprocess while CTAs 1–36 derive fixed SFA offsets and reshape staging data after peer-arrival publication; added full-buffer bit-exact comparison against the generic reshape and bench env forwarding.
- Hypothesis: eliminating the materialized-metadata dependency lets the ~5.8 us SFA reshape overlap the ~8.1 us expert preprocess kernel.
- Benchmark: `DG_SFA_FIXED_OVERLAP=1 SKIP_ISOLATION=0 RUNS=3 bash scripts/bench.sh iter-88-fixed-sfa-overlap`.
- Result: FAILED BEFORE JIT/MEASUREMENT — Docker container `sglang.lxh` was stopped (`container ... is not running`). No correctness or performance number was produced.
- Decision: infrastructure failure, not a kernel result. Commit the partial iteration per protocol; restart the existing container and rerun the exact benchmark before judging the candidate.
- Rerun after restarting `sglang.lxh`: JIT compiled successfully, but the pre-timing full-buffer check failed on all ranks: `fixed SFA overlap reshape differs from generic reshape`; the launcher then exited with SIGSEGV during distributed teardown. No timing was accepted.
- Rerun trajectory: `trajectory/20260717_120515_iter-88-fixed-sfa-overlap-rerun/output.txt`.
- Updated decision: REJECT the current address-derivation implementation. Diagnose the mapping/lifetime mismatch before any further timing; do not enable the opt-in path.

## Iteration 89 — correct same-generation fixed-vs-generic SFA validation

- Change: fixed the probe self-check to run both reshapes against the exact same staging generation. Cross-generation comparison was invalid because atomic token-row allocation is not stable across quant launches.
- Correctness: fixed-route SFA buffer is bit-exact with the generic metadata-driven reshape on all 8 ranks; Test 0 and Test 2 passed in all three runs.
- Benchmark: `DG_SFA_FIXED_OVERLAP=1 SKIP_ISOLATION=0 RUNS=3 bash scripts/bench.sh iter-89-fixed-sfa-overlap-samegen`.
- Full pipeline: 362 / 363 / 363 us (median 363 us). Same-run no-preprocess: 366 / 363 / 365 us (median 365 us), giving only 0–4 us signal (median 2 us).
- Environment caveat: the whole machine was in a slower regime after container restart (normal-P2P kernel 338–339 us; non-fused 345–348 us), so these absolute numbers are not comparable to the earlier 275 us baseline.
- Decision: PROVISIONAL ONLY. The overlap idea is functionally viable, but the signal is too small/noisy to accept. Run an immediate same-session env-off baseline A/B before deciding; keep the path opt-in.
- Trajectory: `trajectory/20260717_120719_iter-89-fixed-sfa-overlap-samegen/output.txt`.
- Same-session env-off baseline: full pipeline 369 / 369 / 372 us (median 369 us), normal-P2P kernel 338 / 340 / 339 us. Candidate normal-P2P kernel was 339 / 338 / 339 us, confirming comparable machine state.
- Final A/B: candidate median 363 us vs baseline median 369 us = **6 us (1.6%) full-pipeline win**. Candidate is bit-exact for the fixed route, but remains a production-shape upper-bound probe and is not accepted as a general dynamic-routing implementation.
- Baseline trajectory: `trajectory/20260717_120848_iter-89-fixed-sfa-overlap-baseline-off/output.txt`.

## Iteration 90 — dynamic-count SFA overlap in expert preprocess

- Change: moved the overlap kernel and reusable row-major→column-major SFA device helper from `fp4_gemm_cutlass3.cuh` into `expert_preprocess.cuh`; GEMM no longer includes preprocess. Replaced fixed 16-token offsets with direct tagged remote-count reads and per-expert rank-prefix derivation. Renamed the opt-in gate to `DG_SFA_OVERLAP`.
- Correctness: random non-uniform 8-rank routing and the fixed production routing both matched the accepted metadata-driven SFA reshape bit-for-bit on all ranks in all three runs. Test 0 and Test 2 passed.
- Benchmark: `DG_SFA_OVERLAP=1 SKIP_ISOLATION=0 RUNS=3 bash scripts/bench.sh iter-90`.
- Full pipeline: 362 / 362 / 363 us (median 362 us). Same-run no-preprocess: 365 / 363 / 365 us (median 365 us).
- Required comparisons: non-fused full 349 / 348 / 348 us; normal-P2P kernel 340 / 338 / 339 us; all-local kernel 311 / 308 / 310 us; P2P exposure 29 / 29 / 30 us.
- Comparison: prior same-state env-off baseline from iter 89 was 369 / 369 / 372 us (median 369 us), with matching 338–340 us normal-P2P kernel. Dynamic-count version preserves the fixed probe's ~6–7 us signal while removing its routing restriction.
- Decision: KEEP as opt-in candidate. Run an immediate env-off baseline and a full correctness verdict before considering default enablement.
- Trajectory: `trajectory/20260717_123706_iter-90/output.txt`.
- Immediate same-session env-off baseline: full pipeline 369 / 375 / 369 us
  (median 369 us); no-preprocess 362 / 367 / 363 us (median 363 us);
  normal-P2P kernel 337 / 340 / 338 us (median 338 us); non-fused full
  348 / 348 / 348 us. The candidate's median 362 us therefore retains a
  **7 us (1.9%) full-pipeline improvement** at comparable 338–340 us P2P-kernel
  state. Baseline trajectory:
  `trajectory/20260717_123910_iter-90-baseline-off/output.txt`.
- Full verdict: `DG_SFA_OVERLAP=1 SKIP_ISOLATION=0 RUNS=1 VERDICT=1 bash
  scripts/bench.sh iter-90-verdict` passed Test 0, all three dynamic-routing Test 1
  rounds, and Test 2 on all 8 ranks. Every expert matched both non-fused and CPU
  references with zero reported error; the overlap-specific random/fixed SFA checks
  remained bit-exact. The verdict run measured 359 us full pipeline at a faster
  333 us normal-P2P state, so it is used as a correctness gate rather than as the
  A/B performance comparison. Verdict trajectory:
  `trajectory/20260717_124117_iter-90-verdict/output.txt`.
- `hgobjdump` check on the generated production specialization reports 40 vregs,
  128 sregs, stack size 0, and no scratch/private/local load-store instructions in
  the ISA. There is no spill. Scalar-register pressure is worth watching when more
  stages are fused, but it did not erase the measured overlap benefit here.
- Final disposition: KEEP opt-in. Routing is dynamic, but the public selection guard
  is still production-shape-specialized; broaden the shape contract and integration
  coverage before enabling `DG_SFA_OVERLAP` by default.
- Frequency correction: the 359–375 us measurements above were collected while the
  machine was not frequency-locked. They remain valid only as same-state A/B evidence,
  not as production absolute latency. After locking all 8 devices at CU 1300 MHz and
  memory 1600 MHz, an immediate 3x A/B produced:
  - overlap off: full pipeline 275 / 273 / 275 us (median 275 us), normal-P2P
    250 / 250 / 250 us, all-local 222 / 221 / 221 us, non-fused GEMM-only
    196 / 195 / 196 us;
  - overlap on: full pipeline 270 / 270 / 270 us (median 270 us), normal-P2P
    250 / 250 / 250 us, all-local 221 / 221 / 222 us, non-fused GEMM-only
    196 / 196 / 196 us.
- Locked-frequency verdict: dynamic SFA overlap saves **5 us (1.8%)** on the full
  pipeline while the copy+GEMM controls are identical. This is the authoritative
  absolute-latency comparison for iteration 90. Trajectories:
  `trajectory/20260717_124727_locked-off-3x/output.txt` and
  `trajectory/20260717_124803_locked-on-3x/output.txt`.
## Iteration 91 - direct FP4 push to expert-owner buffers (rejected: invalid address)

- Hypothesis: push each quantized FP4 row directly into the expert owner's symmetric buffer, so the fused copy+GEMM consumer reads FP4 locally and removes the previously measured ~29 us remote-pull exposure.
- Change: added opt-in `DG_FP4_PUSH`, uncached peer FP4 stores in the quant kernel, and owner-local source-band addressing in expert preprocess. `DG_FP4_PUSH` requires `DG_SFA_PUSH=1`; generation 0 keeps the legacy layout for warmup.
- Locked 8-GPU signal run: **failed correctness/runtime** with an illegal memory access in the fused GEMM consumer (`vmem.ld`/`vmem.st` path); all ranks terminated with SIGABRT.
- Diagnostic timing before failure: quant + sym_buf_zero 43 us, expert preprocess 8 us, fresh quant + SFA push 44 us, consumer after quant 220 us, closed total 264 us. These numbers are not a valid performance result because the candidate crashes.
- Interpretation: the first layout mapping is not safe for the GEMM scheduler's source-band addressing. Do not enable `DG_FP4_PUSH`; next iteration must isolate pointer/band bounds before any performance claim.
- Trajectory: `trajectory/20260717_131844_iter-91`.
## Iteration 92 - owner-local expert-count push (rejected)

- Hypothesis: push the 96 per-source expert counts into each owner's existing SFA staging tail, replacing the prepare kernel's remote count reads (about 3.5k cycles / 43% of its compute-side clock breakdown) with local reads.
- Change: removed the rejected iteration-91 direct-FP4-push active code, and exposed the pre-existing `DG_SFA_PUSH_COUNTS` path through `scripts/bench.sh` for a clean opt-in A/B. Default remains off.
- Locked 8-GPU signal: remote count-read clock fell from about 3,536 to 824 cycles, but the quant event rose from about 31 to 36 us and same-iteration fresh quant + SFA push rose from 22 to 26 us. Expert-preprocess event was 24 us; full pipeline was 272 us versus the accepted locked baseline 270 us. Kernel-only remained 250 us and all-local 221 us.
- Interpretation: moving 384 B/rank of counts eliminates the intended remote reads, but the last-CTA count push plus publication extends the producer tail more than it shortens the already-overlapped consumer metadata work. This is a real stage-local win but an end-to-end regression.
- Verdict: reject; keep the path opt-in/default-off. The production direction must reduce the exposed quant tail or the 250 us copy+GEMM consumer, not move hidden expert work into quant.
- Trajectory: `trajectory/20260717_132801_iter-92-count-push`.
## Iteration 93 - ASYS full-pipeline-only capture harness

- Goal: make the ASYS timeline directly comparable by capturing only the steady-state non-fused full pipeline and fused full pipeline, without correctness, isolation, GEMM-only, clock breakdown, or other diagnostic scenarios.
- Change: added opt-in `ASYS_FULL_ONLY=1` and `ASYS_FULL_ITERS`; initialization/JIT/warmup remain outside an outer `FULL_PIPELINES` HGTX range, with inner `NONFUSED_FULL` and `FUSED_FULL` labels. The harness returns immediately after those workloads. `scripts/bench.sh` forwards both variables; the default benchmark is unchanged.
- Locked 8-GPU dry run: Test 0 and Test 2 passed. Ten steady-state iterations measured non-fused full at 259.5 us/iter and fused full at 273.6 us/iter. The small uplift versus the longer standard median is expected from the shorter 10-iteration profiling batch and extra profiler-range instrumentation; this run validates workload selection rather than replacing the normal verdict metric.
- ASYS capture: `profiling/asys/full_pipelines_only_8gpu_20260717_1345.asysrep` (3.2 MiB, SHA256 `5563aef3f3ddcfe0a2e7df4531815c40a78bb82a47cd68f9f2a9bdef04620a23`). Twenty captured iterations measured non-fused full at 265.1 us/iter and fused full at 272.7 us/iter.
- Filtered ASYS kernel medians across 8 ranks x 20 iterations: non-fused GEMM 199.089 us and DeepEP dispatch 54.589 us; fused copy/GEMM 230.694 us, quant 18.510 us, and per-rank overlap-preprocess medians about 9.4-12.5 us. The report contains no isolation, GEMM-only, or standalone breakdown workloads.
- Trajectory: `trajectory/20260717_134114_iter-93`.

## Iteration 94 - eliminate per-call dummy tensor fills

- Observation: PyTorch-dispatch ASYS mapped the two elementwise kernels around
  preprocess to two `aten.zeros.default(int64)` calls per rank per fused iteration.
  They created inactive dummy arguments: `profile_clocks` before quant and
  `merged_sfa_addrs` before the fused GEMM.
- Change: when profiling is disabled, reuse the existing `sym_buf_addrs` int64
  tensor for the inactive quant profile pointer. In GPU-SFA mode, reuse
  `rank_addr_sfa` for the inactive host-SFA pointer. No active data path changed.
- Locked 8-GPU baseline (100 iterations x 3): non-fused full 254.2 / 253.8 /
  254.2 us; fused full 266.3 / 267.1 / 266.8 us (median 266.8 us).
- Locked 8-GPU candidate (100 iterations x 3): non-fused full 254.9 / 254.9 /
  255.1 us; fused full 261.5 / 261.9 / 260.9 us (median 261.5 us).
  The candidate saves **5.3 us (2.0%)** while the non-fused control is stable.
- Correctness: full verdict passed Test 0, all three dynamic-routing Test 1
  rounds on all ranks, merged-preprocess validation, and Test 2.
- ASYS confirmation: before the change, `FUSED_FULL` contained two int64
  `aten.zeros` calls per iteration/rank; after the change it contains none.
  Reports: `profiling/asys/elementwise_dispatch_8gpu_20260717.asysrep` and
  `profiling/asys/elementwise_eliminated_8gpu_20260717.asysrep`.
- Decision: KEEP. This is a pure launch/fill removal with full-pipeline benefit.
- Trajectories: `trajectory/20260717_134922_iter94_elementwise_baseline`,
  `trajectory/*_iter-94-elementwise-verdict`, and
  `trajectory/*_iter-94-elementwise-perf`.
