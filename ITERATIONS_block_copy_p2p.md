# Block-Copy Fused Dispatch —— P2P 与 GEMM 掩盖分析 & 优化

记录 block-copy 融合 GEMM1 里 "P2P 搬运 vs GEMM 计算" 的掩盖程度分析,以及后续优化的数据对照。

环境:PPU ZW-M890P(39 SM),容器 `sglang.lxh`,路径 `/DeepGemm_workspace/codebase/DeepGemm-block-copy`,2 卡(`CUDA_VISIBLE_DEVICES=2,3`)。
测试:`tests/test_block_copy_gemm1_multi_gpu.py`(prod 配置:M≈1536, N=6144, K=7168)。

---

## 1. 加的计时工具

1. **per-wave / per-tile 打点**(`KSTRIPE_PROFILE=1`)
   - 每个 GEMM tile 记 4×int64:`[wait, mainloop_cycles, wave, gemm_cta]`,按 `tile_idx` 索引(每 tile 独占槽,无 atomic)。
   - `wait` = 该 tile 等 P2P 的 cycle(KTPF=0 是入口整块等待,KTPF>0 是 mainloop 内 stripe 自旋)。
   - host 按 `wave` 分组 → 每个 wave 的暴露率 `wait/mainloop`。
   - 代码:scheduler 里 `curr_tile_idx/curr_wave/curr_cta_id`(fetch_next_work 设置);kernel operator() 写 `[2][3]` + 入口等待写 `[0]`;mainloop flush 写 `[0]+=stripe_wait, [1]=mainloop_cycles`。

2. **copy-only 带宽扫描**(`COPY_BW_SWEEP=1`,可选 `COPY_NCB_LIST`)
   - `copy_only=True` → `grid.x = ncb` 只跑 copy block(全部 blockIdx<ncb 走 copy 路径 return,不跑 GEMM),host CUDA event 计时。
   - 测 copy 带宽随 ncb 的饱和曲线。

改动文件:`scheduler_cutlass3.cuh`、`fp4_gemm_cutlass3.cuh`、`jit_kernels/dispatch_fused_gemm.py`、`tests/test_block_copy_gemm1_multi_gpu.py`。
说明:不传 profiling buffer / copy_only=False 时零开销,正确性路径不变(已验 test1 PASSED)。

---

## 2. 关键概念

- **wave(波次)**:持久化 kernel,gemm CTA 数固定 = 39(SM 数,1 block/SM)。tile 数 > 39,每个 CTA 循环领 tile,第 N 轮 = wave N。`next_block_idx=(current_iter++)*39+eff_bidx`。wave 数 = ⌈总tile/39⌉。
- **tile = (m_block, n_block)**,M-major:`m=tile/num_n_blocks, n=tile%num_n_blocks`。先算完 m0 的所有 n_block,再 m1。
- **copy block**:blockIdx `0..ncb-1`(grid 前 ncb 个);gemm 是后 39 个。copy 与 gemm 抢同一 39 SM(同 kernel,1 block/SM)——每多一个 copy block 少一个 gemm SM。
- 当前 copy 划分:m-block 静态 round-robin(`mb=blockIdx; mb+=ncb`),每 m-block 单 block 搬,搬完置 `copy_ready_flags[mb]`。

---

## 3. 已有发现(优化前)

### 3.1 KTPF sweep(暴露率与 kernel 时间同序 → 打点可信)
| K_TILES_PER_FLAG | 暴露率 | best kernel | vs非fused |
|---|---|---|---|
| 0(整块入口等一次) | 9.6% | 0.299ms | 0.89x |
| 1(每 tile 握手) | 38.5% | 0.423ms | 0.64x |
| 2 | 26.6% | 0.360ms | 0.74x |
| 4 | 11.8% | 0.309ms | 0.87x |

**KTPF=0 最优,P2P 已掩盖 ~91%;per-stripe overlap 越细越差**(细粒度握手让 GEMM 反复追上 copy producer 停等)。

### 3.2 per-wave 分解(KTPF=0,26 m-block 场景 288 tiles=8 wave)
| wave | KTPF=0 暴露 | KTPF=2 暴露 |
|---|---|---|
| 0(第一批) | 63.9%(等21801/算12309) | 58.9% |
| 1–7(稳态) | ~1% | ~15% |
| overall | 9.6% | 26.3% |

**结论:P2P 暴露几乎全在 wave 0 启动**(第一批 m-block 还没 copy 完 GEMM 就开跑);wave 1+ 完全掩盖。KTPF=0 只有第一批付费,KTPF>0 每批都交 ~15% 税。

### 3.3 copy-only 带宽扫描(纯 copy,2 卡,5.5MB/call)
| ncb | time_us | GB/s | vs_prev |
|---|---|---|---|
| 1 | 196.5 | 28.0 | — |
| 2 | 101.4 | 54.3 | 1.94x |
| 4 | 56.9 | 96.7 | 1.78x |
| 8 | 53.7 | 102.5 | 1.06x ← 拐点 |
| 12 | 51.5 | 106.8 | 1.04x |
| 16 | 53.8 | 102.3 | 0.96x |
| 24 | 54.5 | 100.9 | 0.99x |

**单 block 远打不满(28 GB/s),~4 接近、~8 饱和(~105 GB/s 平台)。最优 ncb≈8**:刚饱和 copy 带宽 + 留 31 SM 给 GEMM;ncb=12 偷 SM 反而慢(与全 kernel NCB sweep 一致)。

---

## 4. Baseline(优化前稳定性,3 轮)

配置:KTPF=0,COPY_BW_SWEEP=1,per-wave on。命令见文末。3 轮(2026-07-04):

| 指标 | round1 | round2 | round3 | 均值/波动 |
|---|---|---|---|---|
| overall 暴露率 | 9.91% | 9.32% | 10.18% | ~9.5% ±0.5% |
| wave0 暴露率 | 65.9% | 61.2% | 67.8% | ~65% ±3% |
| wave0 avg_wait (cyc) | 22,541 | 21,009 | 23,140 | ~22K ±1K |
| kernel-only copy+GEMM (ms) | 0.258 | 0.258 | 0.270 | **~0.26** |
| local-only 无P2P (ms) | 0.248 | 0.249 | 0.249 | **0.249**(地板) |
| non-fused (ms) | 0.193 | 0.192 | 0.192 | 0.192 |
| pipeline @ncb8 (ms) | 0.296 | 0.315 | 0.304 | ~0.30 ±0.01 |
| best ncb | 8 | 8 | 4 | 8(4/8 接近) |
| copy BW @ncb8 (GB/s) | 98.9 | 99.2 | 101.3 | ~100 |

**关键:优化 headroom = kernel-only(0.258) − local-only(0.249) ≈ 9µs(~3.5%)。**
wave0 暴露率看着大(65%),但它只占 kernel 1/8 且多数被后续 wave 掩盖,折到 wall-clock 只有 ~9µs 的 P2P 净开销 —— 这就是 "m0 优先合搬" 能抢回的上限。稳定性好(轮间 kernel-only ±0.01ms),优化若能省 5µs+ 是可信的。

---

## 5. 计划优化:动态领取 + m0 优先合搬

**动机**:暴露全在 wave 0;GEMM M-major 最先要 m0。当前静态 round-robin 把带宽平摊到 m0..m7,m0 只有单 block 搬(28 GB/s)。

**修正**:方案B(动态领整个 m-block,每 m-block 单 block)**压不动 wave0**(m0 还是单 block 28GB/s)。真正杠杆是**方案A 合搬**:所有 ncb block 按序合搬同一 m-block,m0 用满带宽 ~100GB/s → ~3.5× 更早就绪。

**实现(方案A,`run_copy_block_cooperative`,behind 运行时 `copy_mode=1`)**:
- 所有 ncb block 同序遍历 m0→m25,每个对当前 m-block 做 1/ncb 切片(int4 index 按 `blockIdx*blockDim+tid`,stride=`ncb*blockDim`)。
- 每 m-block done-counter(`copy_ready_flags[total_m_blocks+mb]`,buffer 已扩 2×),最后一个完成的 block `__threadfence` 后置 `copy_ready_flags[mb]=1`。
- `copy_mode` 运行时参数(不是编译宏,便于同 binary A/B),默认 0=原 round-robin。KTPF=0 才用。

**开关**:运行时参数 `copy_mode`(0/1)。正确性测试用 `COPY_MODE=` env;性能 A/B 用 `COPY_MODE_AB=1`(一次跑同时测两条 path 的 kernel-only + per-wave)。

**测试矩阵**:baseline(copy_mode=0)和 A/B 都要跑 **2/4/8 卡**。

**预期**:wave0 暴露从 ~65% 下降;kernel-only 从 0.26 往 local-only 0.249 靠(headroom ~9µs)。

### 结果(2026-07-04)

正确性:copy_mode 0/1 均 PASSED(2 卡)。合搬数值正确。

**A/B(同 binary,ncb=8,KTPF=0,COPY_MODE_AB=1):**

| GPU | base kernel | coop kernel | Δ | base overall_exp | coop overall_exp | base wave0 | coop wave0 |
|---|---|---|---|---|---|---|---|
| 2 | 0.242 | 0.266 | +24.8µs (+10%) | 3.9% | 4.6% | 0.21 | 0.27 |
| 4 | 0.323 | 0.367 | +44µs (+14%) | 25.0% | 10.4%↓ | 1.76 | 0.69↓ |
| 8 | 0.275 | 0.476 | +200µs (+73%) | 10.2% | 34.1%↑ | 0.66 | 1.06↑ |

**结论:全量合搬是净负优化,随卡数急剧恶化(8 卡 +73%)。**
- 4 卡:合搬确实降暴露(25%→10%),但 kernel 仍 +44µs → copy 侧开销吃掉收益。
- 8 卡:暴露率反升(wave0 0.66→1.06)→ flag 置得更晚。
- 根因:①每 block 遍历所有 m-block,8×26=208 次 device `__threadfence`(baseline 26),多卡重流量下极贵;②straggler——m0 flag 要等全部 8 block 搬完,取决于最慢者(baseline 只等 1 个)。

**多轮复核(3 轮 × 2/4/8 卡,关键)**:单轮结论被推翻——单轮 4 卡 coop +44µs,**多轮真实 +155µs(+53%)且极稳**(base 0.29±0.006 / coop 0.445±0.004)。单轮抽到了最好样本。9 轮无一例外 coop 大幅更慢。**全量合搬确定性死亡。**

| GPU | base (3轮) | coop (3轮) | delta |
|---|---|---|---|
| 2 | 0.251/0.273/0.267 | 0.286/0.448/0.330 | +35~175µs(noisy) |
| 4 | 0.291/0.283/0.295 | 0.440/0.447/0.448 | +155µs(+53%,稳) |
| 8 | 0.270/0.441/0.441 | 0.690/0.619/0.621 | +180~420µs |

**教训**:必须多轮取稳定值,单轮会误判(把 +155µs 看成 +44µs)。
**A/B 测量偏噪**(base 8卡 0.27 vs 0.44 跳变):A/B 块跑在测试靠后,受前序 sweep/热/争用影响。以后信小 delta 需隔离测量(独立进程/更多 warmup/放最前)。

**留存**:copy_mode=1 代码保留备用(默认 0,baseline 不受影响)。方法论有效:A/B + 多轮挡住错误方向。

**观察(有价值)**:4 卡 baseline 暴露最重(overall 25%,wave0 1.76),是最该优化的场景;2 卡几乎无暴露(3.9%)。多卡暴露非单调,取决于每 rank 数据量/路由分布。

### ncb × mode 扫描(补做,回答"cooperative 换 ncb 会不会赢")
kernel-only ms,2 轮,避开被占用的 GPU4(2卡=2,3;4卡=0,1,2,3;8卡=0-7),A/B 块加 prewarm。

- **round-robin**:ncb 6–12 最优且平坦(2卡~0.242、4卡~0.29、8卡~0.27-0.32)。
- **cooperative**:单调随 ncb 变好,**最优永远 ncb=16**(需堆 block 补单-m-block 带宽),但**从不超过 round-robin**:
  - 2 卡 @16:coop 0.249–0.273 vs rr 0.242 → +7~17µs(最接近,基本追平)
  - 4 卡 @16:coop 0.306–0.340 vs rr 0.29–0.30 → +3~47µs
  - 8 卡 @16:coop 0.401–0.470 vs rr 0.27–0.32 → +82~180µs(仍惨)

**最终结论:换 ncb 也救不回 cooperative;gap 随 ncb 缩小但永不闭合。copy 内部重切分(合搬)彻底证伪。**
**采用:baseline round-robin,ncb=8(6–12 均可,平坦稳健)。cooperative 代码留档(copy_mode=1)不启用。**

### wave0 GEMM 等待 clock 直接对比(2 卡,3 轮,同 per-wave 测量路径,COPY_MODE env)
问:合搬有没有让第一轮 GEMM 等待变短?**答:没有,反而变长 ~23%。**

| copy_mode | wave0 wait (r1/r2/r3, cyc) | 均值 | wave0 exp | overall |
|---|---|---|---|---|
| 0 round-robin | 19,931 / 24,195 / 18,276 | ~20,800 | ~0.61 | ~9.3% |
| 1 cooperative | 25,657 / 25,516 / 25,711 | ~25,600 | ~0.75 | ~11.2% |

wave1+ 两模式均 ~390 cyc(完全掩盖,一致)。cooperative wave0 wait 极稳(±100)→ 非噪声。

**坐实 straggler:** 合搬 m0 的 flag 要等全部 ncb 个 block 搬完 m0 切片(最慢者 + 8× fence);round-robin m0 由单个专属 block 搞定就置位 → 合搬 flag 反而置得更晚,消费端 GEMM wave0 等更久。**两端都更差**,比"总 kernel 慢"更彻底地否定合搬。

### 汇编确认:`__threadfence()` 在 PPU 上很贵(缓存 wbinv,非单条 membar)
反汇编 JIT 产物(`hgobjdump --dump-isa` 抽 `.hggc_fatbin`)。PPU 缓存软件管理、非硬件一致,`__threadfence()` 编成 4 条一组:
```
vmem.wbinv.kp0        ; 向量内存缓存 write-back+invalidate
smem.wbinv.kp0        ; 标量内存缓存 write-back+invalidate
s.wait vmem_wbinv     ; 停等完成
s.wait smem_wbinv
```
对比:`__syncthreads()`→`s.blksyn.defer`;cp.async 提交/等待→`vmem.acp.commit.grp`/`s.wait commit_group`/`pipe_flush`(非 fence)。
**含义**:每次 fence = 全缓存回写+作废+停等,比 NV `MEMBAR.GL`(只排序)贵一个量级。round-robin 全程 ~26 次 wbinv;cooperative 26×8=~208 次 → 从指令层坐实合搬爆炸的硬件根因。

**三 scope 隔离验证**(编 `__threadfence{_block,,_system}` 单独 kernel):PPU **有** fence 指令但 device scope 用不了——
| 源码 | 编出 |
|---|---|
| `__threadfence_block()` | `vmem.fence.blk`+`smem.fence.blk`(真 fence,便宜) |
| `__threadfence()` (device) | `s.wait(所有mem计数)`+`vmem.wbinv.kp0`+`smem.wbinv.kp0`+`s.wait wbinv`(无 fence,用 wbinv) |
| `__threadfence_system()` | `smem.fence.sys`+`vmem/smem.wbinv`(fence+wbinv 都要) |

原因:PPU 每 SM 缓存非一致。block scope 只需 CTA 内可见→排序(fence)够;**device scope 要跨 SM 可见,数据还在本 SM 缓存里,fence 只排序搬不动数据→必须物理 write-back+invalidate**。deep_gemm 编译带 `-ppu-patch-fence-ppu=false` 控制此降级。
反汇编法:`llvm-objcopy -O binary --only-section=.hggc_fatbin kernel.so fatbin.bin; hgobjdump --dump-isa fatbin.bin`(工具在 /usr/local/PPU_SDK/bin)。

### 单线程 fence 尝试(copy_mode=2)——收益微小
猜想:wbinv 是 address-less full-SM-cache 操作,`__syncthreads()` 后一个线程做 wbinv 即可刷全 block 数据,省掉冗余 per-warp issue。实测(2 卡,coop-1thF vs coop):ncb=16 仅省 ~5-8µs(0.299→0.291),离 round-robin(0.261)差距 ~40µs 仍在。正确性 PASSED。
**原因**:硬件已把 block 内并发 wbinv **合并**(一个 block 一次 fence 本就只刷一次缓存),减线程省不下。真正开销 = fence 的**次数**(208 次 full-cache wbinv,每次排空+回写+作废+停等),由合搬"每 block 碰每个 m-block"的结构决定,单线程改不了。**单线程 fence 救不了合搬。copy_mode=2 留档。**

### 若还要压 wave0,唯一剩下的杠杆
让 copy 更早启动(GEMM launch 前 / 独立 stream / 减 copy 起步延迟),而非改 copy 内部划分——后者(合搬)已反复证明只加 fence/straggler 开销,连消费端 wave0 等待都更长。
**更新:研究了 grid 精配(见 §6)——澄清了小 ncb 超发慢的机制(启动 bubble),但 wall-clock 地板 ~0.27 现默认已到,精配非加速。**

---

## 6. Grid 精配:超发 bubble 才是小 ncb 的真凶(2026-07-04)

### 起因:exposure 与 wall-clock 背离
per-wave exposure(GEMM 等 copy flag 的占比)在 ncb=2 就已把 steady-state 完全掩盖(wave1+ ~370 cyc),但 **wall-clock 却比 ncb=8 慢 63µs**(0.335 vs 0.272)。exposure 是 GEMM 视角指标,看不到 copy 占 SM / 启动调度对总时间的贡献 → **判性能必须看 kernel-only wall,exposure 只说明"P2P 藏没藏住"**。

### 累计时间校准(新增打点)
给 profiled 那次调用套 CUDA event,并按 CTA 求和 `wait+mainloop` 取 max = 关键路径 cyc,`implied clock = 关键路径cyc / profiled_wall`。真实 PPU clock ≈ 1.0 GHz;implied clock 越低 → 越多 wall 落在 mainloop 之外(= 启动 bubble)。

### 当前 grid 结构
`grid.x = get_grid_shape()(=sm_count 39) + ncb`([fp4_gemm_cutlass3.cuh:2566](deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh#L2566)),scheduler `eff_grid = gridDim.x - ncb = 39`([scheduler_cutlass3.cuh:249](deep_gemm/include/deep_gemm/scheduler_cutlass3.cuh#L249))。即 **39 个 GEMM CTA + ncb 个 copy CTA 超发到 39 个 SM**,每 wave 39 tile。copy CTA 搬完 return,释放的 SM 回补被堵的 GEMM CTA。

### A/B 结果(prod 2 卡,3 轮均值,GPU 独占)
| 配置 | GEMM CTA | kernel-only | local-only | 累计 main-only | implied clk | 说明 |
|---|---|---|---|---|---|---|
| exact=0 ncb=2(超发) | 39 | 0.334 | 0.290 | ~288K | 0.65–0.83 | **~60µs bubble** |
| exact=0 ncb=8(超发,原默认) | 39 | 0.272 | 0.260 | ~280K | 0.80–0.98 | ~5µs bubble |
| exact=1 ncb=2(精配 37) | 37 | 0.269 | 0.260 | ~294K | 0.98–0.99 | 无 bubble |
| exact=1 ncb=8(精配 31) | 31 | 0.311 | 0.305 | ~344K | ~1.0 | 无 bubble 但永久少 21% SM |

### 机制(完整闭环)
1. **超发在大 ncb 能用**:ncb=8 的 copy ~55µs 搬完、SM 早释放 → bubble 仅 ~5µs。
2. **超发在小 ncb 崩**:ncb=2 的 2 个 copy block @54GB/s 占 2 个 SM 长达 ~100µs,把 2 个 GEMM CTA 堵在启动线 → **~60µs bubble;local-only 也有(0.290),证明是纯调度、与 P2P 无关**。
3. **精配救小 ncb**:grid=39 精配,37 个 GEMM CTA 立即开跑、零 bubble;只少 2 个 SM(5%),mainloop 几乎没涨(288K→294K)→ 净赚。P2P 净开销仅 0.269−0.260 = **~9µs**,正好等于 §4 的 wave0 headroom 上限 → 已压到理论下限。
4. **精配害大 ncb**:31 个 CTA 永久少 8 个 SM(21%),每 CTA 多干活,mainloop 暴涨(280K→344K)→ 亏。

### 波数拐点:精配无害的真正边界是"波数不跳增"(2026-07-04 实测)
288 tile,GEMM wall ≈ 波数 × 单tile时间(每 SM 每波并行 1 tile)。波数 = ⌈288 / (39−ncb)⌉,**只要波数还是 8,精配对 GEMM 零影响**:

| exact ncb | GEMM CTA | 波数 | kernel-only | 累计 main | implied clk |
|---|---|---|---|---|---|
| 2 | 37 | 8 | 0.269 | ~282K | 0.97–1.00 |
| 3 | 36 | **8**(288 整除,末波满载) | 0.270 | ~288K | 0.96–1.00 |
| 4 | 35 | **9** ← 跳档 | 0.289(+20µs) | ~312K | 0.99–1.01 |

- **ncb≤3 精配 = 8 波 → GEMM 不变**(0.269≈0.270)。ncb=3 是最干净落点:36×8=288 整除、末波零空闲、copy block 更多。
- **ncb=4 跨到 9 波才变慢**(累计 main ×9/8=1.125,对上 288K→312K)。所以"精配害大 ncb"精确说是**"精配把 ncb 加到波数跳增才害"**,边界在 ⌈288/(39−ncb)⌉ 从 8 跳到 9,即 ncb=4。

### Clock 记录:真实 PPU clock ≈ 1.0 GHz
从所有无 bubble(精配)配置 `implied_clock = 关键路径cyc / profiled_wall` 反推,稳定聚在 **0.96–1.01 GHz** → 这就是真实 core clock。校准判读:**implied ≈ 1.0 = 无 bubble;明显 < 0.95 = 有 bubble**。
- 精配 ncb 2/3/4/8:implied 全在 0.96–1.02(无 bubble)。
- 超发 ncb=2:implied 稳定 **0.65–0.83**(被 ~60µs bubble 拉低)——唯一低于 0.95 的,坐实 bubble。
- 超发 ncb=8:0.80–0.98(bubble 小 ~5µs,单发 profiled-wall 偏噪)。

### 逐波暴露(exact ncb=2/3,KTPF=0,2 轮均值,clock≈1GHz 故 cyc≈ns)
| wave | ncb=2 tiles | ncb=2 avg_wait | ncb=3 tiles | ncb=3 avg_wait |
|---|---|---|---|---|
| 0 | 37 | **~25,900** (exp 0.71) | 36 | **~25,700** (exp 0.72) |
| 1–6 | 37 | ~370 (exp ~0.011) | 36 | ~375 (exp ~0.011) |
| 7 | **29**(37×7 余,8 SM 闲) | ~345 | **36**(288 整除满载) | ~375 |

overall exposure ≈ **10.2–10.3%**。要点:
- **暴露完全集中在 wave0(~72%,~25.8µs),wave1–7 全部 ~1%(~370 cyc,只是查 flag,完全掩盖)**——P2P 只付 wave0 一次启动税。
- ncb=2≈ncb=3(wave0 差 ~200 cyc,噪声内);ncb=3 末波满载更均衡。
- **exact 的 overall exposure(10.3%)反而略高于超发 ncb=8(9.6%)**:精配无 bubble → 所有 GEMM CTA 立即就绪、老实等 m0/m1(纯 P2P wait 显式入账);超发把部分 P2P wait 藏在调度 bubble 后(exposure 看不到但 wall 更差)。**又一例 exposure vs wall 背离:精配把"隐形 bubble"换成"显式 wave0 等待"。**

### Stripe(KTPF)能压 wave0 暴露但换不来 wall(exact ncb=3,2 轮均值)
K=7168 / block_k=128 → 56 k-tile/m-block;KTPF=k_tiles_per_flag,越小=握手越细。

| KTPF | wave0 exp | wave0 wait | wave1 exp | overall exp | kernel-only wall |
|---|---|---|---|---|---|
| 0(整块入口等) | 0.71 | ~25.9µs | 0.010 | ~10% | **0.271** |
| 1 | 0.67 | ~78µs* | 0.27 | ~39% | 0.406 ✗ |
| 2 | 0.65 | ~69µs* | 0.16 | ~36% | 0.366 ✗ |
| 4 | **0.19** | **~8.9µs** | 0.084 | ~9.6% | 0.266 |

\* KTPF>0 的 wait 是 mainloop 内 per-stripe 累计,被放大,不与 KTPF=0 可比;看 wall。

- **KTPF=4 确实砍掉 wave0 暴露**(0.71→0.19,wave0 wait 25.9→8.9µs):粗粒度握手让 wave0 GEMM 不等整块 m0 搬完就开 mainloop,把后段 K-stripe 拷贝叠进前段 stripe 计算。
- **但 wall 没赚**(0.271→0.266,噪声内):省下的 wave0 ~17µs 被稳态 per-stripe 税吃掉(wave1–7 从 ~0.4µs/波 涨到 ~3.2µs/波,×7≈+20µs)。**stripe 只把等待从 wave0 挪到稳态,没消除。**
- **KTPF=1/2 灾难**(0.41/0.37):细握手让每 stripe 都追上 producer 停等(和 §3.1 老结论一致)。
- 正面点:**KTPF=4 在 exact grid 下不再变慢(≈KTPF=0),而老配置里 KTPF=4 是 0.309 更差** → 去 bubble 后 stripe 稳态税变轻。
- **又一例 exposure≠wall**:wave0 暴露能做到 0.19 好看,总时间(~0.27)纹丝不动。要省 wall 仍需让 copy 更早启动,而非改 tile 内握手粒度。

### 采用建议(诚实结论)
**exact=1 ncb=2(0.269)与 exact=0 ncb=8(0.272)只差 ~3µs,在噪声内 —— 不是有意义的加速。** wall-clock 地板就是 ~0.27,现默认(ncb=8 超发)已到地板。本实验的真正价值不是"更快",而是:
1. **机制搞清了**:小 ncb 超发慢是启动 bubble(不是 copy 带宽),精配可消。
2. **换了条到达地板的路**:用 2 个 copy block + 不超发,即可追平 8 个 copy block 超发 —— 若那几个 SM/资源另有用处(如给别的 kernel),精配 ncb=2 是等速但更省的选择。
3. **别把小 ncb 超发当选项**(0.334,坑)。

即:**保持现默认 ncb=8 超发即可;想省 copy 资源时改用精配 ncb=2/3,速度等价。** 不值得为这 3µs 专门改默认逻辑。

### 开启方式
运行时 env `FUSED_EXACT_GRID=1`:grid 不 `+= ncb`,GEMM 用 `39−ncb` 个 CTA(精配,无超发)。默认(不设或=0)= 原超发行为。门控在 [fp4_gemm_cutlass3.cuh:2565](deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh#L2565)。
```bash
# 精配 A/B(注意 NCB_SWEEP 固定 ncb → 决定 best_ncb → 决定 profiled/精配的 ncb)
SKIP_CORRECTNESS=1 KSTRIPE_PROFILE=1 K_TILES_PER_FLAG=0 NCB_SWEEP=2 FUSED_EXACT_GRID=1 \
  CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29888 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose
```
**测量方法论**:必须 GPU 独占 + 3 轮取稳(profiled-wall event 单发偏噪,但 kernel-only median 稳;implied clock 是判 bubble 的干净指标)。

---

## 7. 测量脚本 & 复现(2/4/8 卡全 sweep)

`sweep_blockcopy.sh`(仓库根):一条命令跑完某卡数的 **ncb / exact-grid / stripe(KTPF)** 全 sweep,3 轮,只打印制表字段 + 时间戳。

### 用法(容器内,仓库根目录)
```bash
bash sweep_blockcopy.sh 2 0,1              29500   # 2 卡
bash sweep_blockcopy.sh 4 0,1,2,3          29600   # 4 卡
bash sweep_blockcopy.sh 8 0,1,2,3,4,5,6,7  29700   # 8 卡
```
每卡数跑 3 轮 × 9 配置:
- `exact=0 ncb∈{2,4,8}`:超发 grid(默认),ncb sweep + 暴露小 ncb bubble。
- `exact=1 ncb∈{2,3,4}`:精配 grid(`FUSED_EXACT_GRID=1`),验证波数拐点。
- `exact=1 ncb=3 KTPF∈{1,2,4}`:stripe sweep(KTPF=0 已含于上组)。

抓取字段:`gemm_ctas / waves / overall exposure / 每 wave 行(wave0–7)/ cumulative critical-path / implied clock / kernel-only wall / local-only`。

### 前置 & 方法论
- **GPU 必须独占**;脚本 `preflight` 检查目标卡显存 <3GB 才开跑(依赖 ppu-smi)。
- **每配置 3 轮取稳**:wave0 有噪声,稳态(wave1+)极稳;小 delta 看 kernel-only median,别信单轮。

### ⚠ 避坑(踩过)
1. **杀进程**:脚本 + torchrun 跑在 docker 容器内,**host 的 pkill 杀不到**(PID namespace 不同),且外层 `&&` 链会复活下一个 torchrun。中止须在容器内按 PID 杀:
   ```bash
   docker exec <ctr> bash -lc 'kill -9 $(ps -eo pid,cmd | \
     grep -E "sweep_blockcopy|test_block_copy_gemm1" | grep -v grep | awk "{print \$1}")'
   ```
2. **端口**:master_port 必须 <65536,别用 `printf %02d` 拼出 6 位端口(会全崩)。
3. **抢占监控**(可选,host 端):别人临时起任务会污染计时(实测撞过一次 DeepSeek benchmark 占 4-7)。跑长 sweep 时并行一个采样器,任一卡显存 >7GB(我们最多 ~4GB)即标 FOREIGN,配合脚本 TS 时间戳可定位并只重跑被污染的配置:
   ```bash
   while :; do mx=$(ppu-smi|grep -oE "[0-9]+MiB / 98304"|grep -oE "^[0-9]+"|sort -n|tail -1); \
     echo "$(date +%T) max=${mx}MiB $([ ${mx:-0} -gt 7000 ] && echo FOREIGN)"; sleep 15; done
   ```

### 需要用到的代码开关(均已在仓库)
- `FUSED_EXACT_GRID=1`:grid 不 `+= ncb`,GEMM 用 `39−ncb` CTA(精配)。[fp4_gemm_cutlass3.cuh:2565](deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh#L2565)。
- `KSTRIPE_PROFILE=1` + `NCB_SWEEP` + `K_TILES_PER_FLAG`:per-wave 打点 + 累计/implied-clock 打印(test 里,诊断工具,默认零开销)。

---

## 8. 2/4/8 卡实测结果(单-switch 机,2026-07-05)

`sweep_blockcopy.sh` 全 sweep。**方法:min-of-rounds**——GPU 抢占只会抬高计时不会降低,故取多轮最小值=真实无污染值。2/4 卡轮间稳(干净);8 卡被间歇外部任务撞(用全卡躲不掉),取 6 轮(两次 run)最小值。单位 ms。

### 8.1 grid/ncb × kernel-only & 端到端 pipeline(KTPF=0)
| 卡 | 配置 | kernel-only | pipeline 融合 | pipeline 非融合 | 融合/非融合 |
|---|---|---|---|---|---|
| 2 | exact0 ncb2(超发) | 0.335 | 0.366 | 0.268 | 0.73x |
| 2 | exact0 ncb8 | 0.271 | 0.307 | 0.268 | 0.87x |
| 2 | **exact1 ncb2(精配)** | **0.269** | 0.304 | 0.268 | 0.88x |
| 4 | exact0 ncb2 | 0.384 | 0.419 | 0.246 | 0.59x |
| 4 | exact0 ncb8 | 0.296 | 0.323 | 0.247 | 0.76x |
| 4 | **exact1 ncb2** | **0.278** | 0.315 | 0.246 | 0.78x |
| 8 | exact0 ncb2 | 0.313 | 0.357 | 0.251 | 0.70x |
| 8 | exact0 ncb8 | 0.275 | 0.323 | 0.251 | 0.78x |
| 8 | **exact1 ncb2** | **0.265** | 0.309 | 0.251 | 0.81x |

**要点:**
- **grid bubble 跨卡一致**:超发小 ncb(exact0 ncb2)最差(4 卡 0.384),精配 exact1 ncb2 每卡救回,是各卡最优 kernel(8 卡 0.265)。
- **⚠️ 端到端 pipeline:融合始终慢于非融合(~0.7–0.88x)**。kernel 内 copy 给 GEMM 加 ~0.07ms;再叠 quant+preprocess,全流程没赢过 DeepEP 非融合。**block-copy 融合在 kernel 级掩盖了 P2P,但端到端被 preprocess/quant 开销吃掉。**
- 非融合 pipeline 稳定 ~0.25(2 卡 0.268 / 4 卡 0.246 / 8 卡 0.251)。

### 8.2 KTPF × ncb kernel-only(exact1,ncb=3 代表,min)
| 卡 | k0 | k1 | k2 | k4 | k7 | k14 |
|---|---|---|---|---|---|---|
| 2 | 0.270 | 0.406 | 0.366 | 0.269 | 0.279 | **0.262** |
| 4 | 0.281 | 0.379 | 0.307 | 0.283 | **0.268** | 0.271 |
| 8 | 0.267 | 0.382 | 0.303 | 0.269 | 0.260 | **0.259** |

### 8.3 KTPF overall exposure(exact1 ncb=3,min)
| 卡 | k0 | k2 | k4 | k7 | k14 |
|---|---|---|---|---|---|
| 2 | 0.103 | 0.351 | 0.097 | 0.132 | **0.079** |
| 4 | 0.146 | 0.206 | 0.142 | 0.116 | **0.099** |
| 8 | 0.086 | 0.190 | 0.095 | 0.069 | **0.063** |

**KTPF 结论(跨卡一致):**
- **ktpf=1/2 灾难**(kernel +40~130%,exposure 翻倍)——细握手让 GEMM 逐 stripe 追上 producer(见 §"拍频")。
- **ktpf=4 ≈ ktpf0**(中性)。
- **ktpf=7/14 exposure 最低,kernel 略优 ktpf0(~5–8µs)**——coarse stripe 蹭到一点 wave0 overlap 又不付稳态税。但幅度接近噪声,不是大杠杆。
- **4 卡 baseline 暴露最重**(k0 exposure 0.146 > 2 卡 0.103 > 8 卡 0.086),和历史观察一致(多卡暴露非单调,取决于每 rank 数据量/路由)。
- **implied clock:KTPF=0 各卡 ~0.99GHz(真实 core clock);KTPF>0 显示 1.1–1.5GHz 是账面假象**(per-stripe wait 被折进 mainloop_cycles,把关键路径撑大),不代表真频率。

### 8.5 地板差根因:NoPad grouping vs masked(2/4/8 卡,min,极稳)
拆"融合 GEMM 地板(0.25)vs 非融合 GEMM(0.19)"这 ~0.06 差。三方 GEMM-only 对比(ms):

| 卡 | masked(无copy) | 融合自定义 NoPad local(有copy) | 生产 NoPad(无copy) |
|---|---|---|---|
| 2 | **0.192** | 0.253 | **0.289** |
| 4 | 0.194 | 0.263 | 0.291 |
| 8 | 0.194 | 0.256 | 0.294 |

**跨卡完全一致**(GEMM problem 恒定 ~1536 行,GEMM-only 每 rank 独立,与卡数/网络无关)。masked ≪ 融合自定义 NoPad < 生产 NoPad。

**逐一排除(全部否定后锁定 grouping):**
- ❌ **SM 数**:36–39 都 8 波(波数分析已定论)。
- ❌ **block_m**:两边都 128(非融合 `NF_BLOCK_M` 默认 128;融合 heuristic 对此 shape 固定选 128,`FORCE_EXPECTED_M` 拉不动)。
- ❌ **copy 竞争**:ncb2 vs ncb3 的 local 只差 ~3µs;且**生产 NoPad 无 copy 仍 0.289**。
- ❌ **自定义融合 kernel 的 overhead**:自定义 NoPad(0.253)反而**比生产 NoPad(0.289)还快**,不是它的锅。
- ✅ **NoPad grouping / 调优**:生产 NoPad(0.289)比 masked(0.192)慢 ~0.1,纯 grouping 差异。masked 是调优过的生产 fp4 路径;**NoPad fp4 未调**(`m_grouped_gemm_fp4.py` 明写 `# TODO: enable fp4 get_best_configs`)。

**结论 & 杠杆:GroupedMasked 明显优于 GroupedNoPad(0.19 vs 0.25–0.29)。** 要闭合融合 vs 非融合的地板差,方向是:①把 block-copy 改建在 masked GEMM 上(copy 目标改 masked 布局,设计改动),或 ②调优 NoPad fp4 的 get_best_configs。trade-off:masked 会 padding 到 max_tokens(分布不均时浪费 compute),但此 shape 下因调优好仍更快。
诊断开关:`NOPAD_GEMM_BENCH=1`(test 内合成输入跑生产 NoPad GEMM,仅计时)。

### 8.4 数据出处
原始 log:`sweep_logs/sweep_m1_0705_1122.log`(2/4/8 卡首轮)+ `sweep_logs/sweep_m1_8card_rerun_*.log`(8 卡补跑)。抢占核对靠轮间一致性(比 GPU 监控可靠——监控纯显存阈值会误报自身峰值)。

---

### 8.6 GroupedMasked 移植结果（2026-07-05）

- 新增 `FUSED_GEMM_GROUPING=nopad|masked`，默认 `nopad` 保持兼容；两条路径编译为独立 JIT 实例。
- masked 使用 `[expert, max_tokens, K/2]` staging / `[expert, max_tokens, N]` 输出和 `masked_m` 调度；copy 仍复用 block-granular metadata，并通过 per-expert block offset 映射 ready flag。
- 2 卡 prod `FULL_CORRECTNESS=1`，最终 `block_m=128,ncb=3`：12/12 experts 的 BC vs NF、BC vs CPU element-wise diff 均为 **0**。
- 同机 exact-grid A/B（2 卡）：NoPad kernel/pipeline `0.259/0.302 ms`；Masked `0.239/0.267 ms`，分别快 `0.020/0.035 ms`。
- Masked min：2 卡 `0.239/0.267 ms`（ncb3），4 卡 `0.277/0.308 ms`（ncb4，链路波动明显），8 卡 `0.236/0.273 ms`（ncb3）。

### 8.7 复核 + 公平 MNK 对照：masked 更快的真因是 block_m，不是 grouping（2026-07-06）

在 test 里加了 **MNK partition check** log（融合/非融合各自实际解析出的
`(block_m,block_n,block_k,warp_m,warp_n,stages,num_sms)` 与 `expected_m` 并排打印 +
`TILE MATCH`）。关键背景：`get_best_configs` 除 Dense 外**不看 gemm_type**，tile 只由
`expected_m/n/k/num_groups/num_sms` 决定。

**实测 tile（prod，per-expert token 完全均匀=128）：**
- masked 融合 & non-fused：`expected_m=128 → block_m=128`，`TILE MATCH=True`。
- nopad 融合：nopad 路径的 `expected_m<=128 → 129` clamp（iter15 遗留）使
  `get_best_configs` 选 **block_m=256**，`TILE MATCH=False`。每 expert 只有 128 行却用
  256 高 tile → **M 维 50% padding 浪费**。

**跨卡 min（3 轮取 min，ms；pipe / kern(copy+gemm) / kern_local(纯 GEMM 无 P2P)）：**

| 卡 | nopad(bm256) | masked(bm128) | **nopad+FORCE bm128** | non-fused(pipe/gemm-only) |
|---|---|---|---|---|
| 2 | 0.302 / 0.248 / 0.247 | 0.267 / 0.240 / 0.223 | **0.267 / 0.240 / 0.223** | 0.268 / 0.195 |
| 4 | 0.313 / 0.279 / 0.260 | 0.307 / 0.280 / 0.243 | —（2/8 已定论） | 0.246 / 0.194 |
| 8 | 0.309 / 0.265 / 0.259 | 0.271 / 0.236 / 0.216 | **0.273 / 0.235 / 0.216** | 0.251 / 0.194 |

**结论（修正 §8.5）：`nopad + FORCE_EXPECTED_M=128`（强制 block_m=128）与 masked 逐位一致**
（2 卡 0.267/0.240/0.223 完全相同；8 卡差 ≤0.002=噪声）。所以 §8.5“NoPad grouping 本身慢”
是**误判**——真因是 nopad 路径的 expected_m clamp 选了次优 block_m=256。masked 移植之所以更快，
本质是它顺带绕过了这个 clamp（不 clamp → block_m=128）。

**杠杆修正：** 拿到这份加速有两条等价路径——① 保留 masked 移植；② 更简单：删掉 nopad 融合路径的
`expected_m<=128→129` clamp（或对此 shape 直接 block_m=128）。两者性能相同。masked 的额外代价是
输出/staging padding 到 `max_tokens`（token 分布不均时浪费 compute），nopad 紧凑；本 shape（均匀
128）下等价，生产若分布不均则 nopad+bm128 反而更省。

**融合 vs 非融合（pipeline，masked）：** 2 卡追平（0.267 vs 0.268）；8 卡融合慢 ~0.020；4 卡链路
波动大（0.307 vs 0.246）。注意 non-fused pipeline 口径含 EP-buffer dispatch，与融合 pipeline
（quant+preprocess+copy+gemm）机制不同，非严格同口径；纯 GEMM（gemm-only 0.194–0.196）稳定。

诊断开关：`FUSED_GEMM_GROUPING=nopad|masked`；`FORCE_EXPECTED_M=128` 强制 block_m；MNK check 每次
打印在 `--- MNK partition check ---` 段。多卡对照脚本 `tests/sweep_masked_vs_nopad.sh`。

### 8.8 masked 配置扫描（8 卡，3 轮 min，2026-07-06）

功能验证：ktpf∈{0,2,4,7,14,28} + copy_mode∈{0,1,2}(仅 ktpf=0) 全 **FULL_CORRECTNESS PASSED**。
性能：轴 ktpf × exact{0,1} × copy_mode(ncb∈{2,3,4} test 内扫)。copy_mode 仅 ktpf=0 生效
（ktpf>0 一律走 kstripe copy 路径，忽略 copy_mode）。全程 clean（min-of-3 兜底）。

pipeline ms（exact=1, cm=0, best=ncb3）；kern=copy+GEMM，local=all-local(无 P2P)：

| ktpf | ex=1 best_pipe | kern | local | ex=0 best(ncb4) |
|---|---|---|---|---|
| 0  | 0.271 | 0.235 | 0.215 | 0.310 |
| 2  | 0.288 | 0.244 | 0.228 | 0.379 |
| 4  | 0.270 | 0.229 | 0.214 | 0.347 |
| 7  | 0.268 | 0.230 | 0.210 | 0.341 |
| **14** | **0.263** | 0.226 | 0.211 | 0.319 |
| 28 | 0.271 | 0.236 | 0.215 | 0.311 |

**结论：**
- **exact=1（FUSED_EXACT_GRID=1）全面优于 exact=0**（0.26–0.29 vs 0.31–0.38）。exact=0 超发
  39+ncb，pipeline 里 quant/preprocess 与 GEMM 争 SM 放大 bubble。
- **cooperative copy（copy_mode=1/2）pipeline 灾难性**（0.79–1.7ms，min-of-3 仍高=真实非污染），
  复现证伪；采用 round-robin（cm=0）。
- **ncb=3 稳定最优**（exact=1 下 ncb2/4 都更差）。
- **best_ncb=3 时 ktpf 影响在噪声级**（0.263–0.271，跨 ktpf 仅 ~8µs）。ktpf=14>7>4 略优但接近噪声；
  ktpf=0（单次握手）与 ktpf=28 同为 0.271。**k-stripe overlap 在 exact 网格下有微弱正收益**，与
  §3.1（无 exact 时 KTPF=0 最优）不矛盾（网格变了）。
- **top-3（做 per-wave）：ktpf=14 / 7 / 4，均 exact=1 / cm=0 / ncb3。**

### 8.9 top-3 per-wave 掩盖 + Q1 定量（8 卡，exact1/cm0/ncb3，2026-07-06）

KSTRIPE_PROFILE per-wave（288 tiles, gemm_ctas=36, 8 waves）。ktpf=7 的 profiling 被**插桩本身**
扭曲（4-stripe 调度 + per-stripe clock 记录 → 交替 wave 假停，kernel 虚高 0.68ms，复现两次；干净
sweep 无插桩值 0.230），故 per-wave 用 ktpf=0/4/14 代表：

| ktpf | overall_exp | wave0_exp | wave1-7_exp | kern | local |
|---|---|---|---|---|---|
| 0  | 0.200 | 1.40 | ~0.015 | 0.234 | 0.215 |
| 4  | 0.183 | 0.484 | ~0.10 | 0.232 | 0.216 |
| 14 | 0.137 | 0.497 | ~0.03 | 0.226 | 0.210 |

- **P2P 暴露几乎全在 wave0**，wave1-7 完全掩盖（1-10%），复现 §3.2。
- ktpf=0：wave0 单次大等待（exp 1.40，~42k cyc）；wave1-7 ~1.5%。
- k-stripe(ktpf>0)把 wave0 等待拆小（GEMM 用部分 K 就起跑，wave0 exp 1.40→~0.50），但给
  wave1-7 加每-stripe 小税。ktpf=14(2 stripe) 净收益最好；ktpf=4(7 stripe) 每-stripe 税更大。
- 但 wall-clock 差异都在噪声级（0.226-0.234），**exposure≠wall-clock**。

**Q1 解答（更正版）：gap = wave0 的 copy 填充停等，不是 SM 数。**
288 tile 分给 36 或 39 CTA，关键路径都是 ceil(288/CTA)=**8 波**（36→8.0，39→ceil7.38=8），SM 数不造成
差异；`0.194×39/36=0.210` 是**巧合**。真机制：

| | GEMM CTA | 波 | 实测 | 机制 |
|---|---|---|---|---|
| non-fused | 39 | 8 | 0.194 | buffer 已 dispatch，wave0 立即开算 |
| fused local | 36(+3 copy) | 8 | 0.210 | wave0 等本地 copy 填首批 M-block（+0.016）|
| fused normal | 36(+3 copy) | 8 | 0.226 | wave0 等 P2P copy（更慢，+0.032）|

- 决定性证据：normal vs local **同为 36 GEMM CTA**（SM 数相同）却差 0.016 → SM 模型无法解释，只有
  “wave0 等 copy、P2P 比本地慢”能同时解释两段 gap。
- wave1-7 完全掩盖（后续 M-block copy 与前面 GEMM 重叠），per-wave 暴露 1-10% 印证。三者纯算都 ≈0.194，
  差异全在 wave0 启动等 staging buffer 填好；copy 越慢等越久。
- **含义：要缩小 gap 须缩短 wave0 首批 M-block 的 copy 就绪时间（如首批优先/更细粒度 flag 让 GEMM 更早
  起跑），而非动 SM 预算。**

### 8.10 Q2：quant+preprocess vs DeepEP dispatch 基线（8 卡，2026-07-06）

test 新增 DeepEP dispatch 单独计时（median/40，与 quant/preprocess 同口径）：

| 项 | ms |
|---|---|
| quant + sym_buf_zero | 0.025 |
| expert_preprocess | 0.043 |
| **sum** | **0.068** |
| DeepEP dispatch (nonfused) | **0.048** |
| Q2 目标 (≤½ DeepEP) | **0.024** → **NOT MET**（当前 0.068 > 0.048，比 DeepEP 还贵）|

**瓶颈定位：`expert_preprocess`(0.043) 是大头，且 kernel 以 `<<<1, min_threads>>>` 单 block 启动**
（prod: 96 线程 / 1 block），所有跨-rank metadata 准备挤在 1 个 SM 串行跑。quant 另一独立 kernel(0.025)。
与 [[project_blockcopy_crossrank_overhead]] 的“跨rank准备 66µs”同源。

**优化方向（需 kernel 工程）：** ① preprocess 从单 block → 多 block 并行（expert×rank 对可并行，cumsum
用两趟或原子）；② 融合 quant+setflag+preprocess 减少 launch + 重复读 sym_buf_addrs。目标把 sum 从
0.068 压到 ≤0.024（~2.8×）。

### 8.11 Q2 preprocess 细分（clock64 + gen0 对照，8 卡，2026-07-06）

gen=0 跳过 arrival barrier（Phase 2 gated on generation>0），故 preproc(gen>0)−preproc(gen=0)=barrier。
in-kernel clock64 三段（一次 gen>0 调用，thread0 记录）：

| 阶段 | cycles | 占比 |
|---|---|---|
| setup+barrier | 61,590 | 80.5% |
| remote-count-reads | 2,098 | 2.7% |
| compute(Phase4+5) | 12,830 | 16.8% |

- **远程 count 读没被串行化**（2.7%），不是瓶颈（无需 hgobjdump 查汇编）。
- **大头是 arrival barrier 跨-rank 等待**（等其他 rank quant 完成+push flag），随 rank skew 变化：ranks
  对齐时 barrier ~5-13µs、preproc 地板 ~0.032ms；skew 大时 barrier 可达 ~0.050ms。**这是固有的**（消费者
  必须等生产者量化完才能读 count），非 kernel 并行能消除。
- **可寻址的只有 compute(13µs)+launch**。preproc 地板 0.032 = launch + compute + reads + setup。

**含义：** Q2 目标（quant+preprocess ≤0.024）受限于固有跨-rank 等待；kernel 侧能动的是 compute+launch
（~15-20µs）。真正大杠杆是降 barrier 暴露（atomic-arrival，[[project_atomic_arrival_barrier]] 记录过死锁，慎）。

诊断开关：test 里 `dbg_cyc` 传入 `dispatch_expert_preprocess` 打印三段 cycle；gen=0 对照自动打印。

### 8.12 Q2 优化尝试记录

（下面逐条记录每次尝试的改动+性能，含无效/负优化，回退保留基线 masked=exact1/cm0/ncb3，
preproc 基线 gen0≈0.032ms / compute≈12,830cyc / sum≈0.067ms。）

**opt#1 [KEEP] 删 Phase 5 死代码（first pass）** — dispatch_preprocess.cuh。
first pass 算的 `tokens_in_block`/`temp_remaining`/`tokens_placed_total` 从未被使用（entry.y 用
total_count，per-block count 不需要）。删除后：
- 正确性 **PASSED**（8 卡 FULL_CORRECTNESS，diff 阈值内）。
- compute(P4+5) **12,830 → 7,930 cyc（−38%，~-4µs）**；remote-reads/barrier 不变。
- 单-block preproc 少 num_ranks 次/block 迭代。**保留**（安全正收益，虽绝对时间被 barrier 掩盖）。
- 回归：opt#1 后最优 masked pipeline **0.265**（3 轮 min，vs 基线 0.263，噪声内）→ 对 pipeline 中性
  （compute 省的被 barrier 掩盖），但代码更干净、正确性保持。

**quant 侧分析（用户点名）：**
- `sym_buf.zero_()` 在 perf 路径只清最后 128B（flag 区，line 619），**不清整 buffer** → 归零非开销。
- quant grid = **256 blocks**（每 token 一块，已 quantize-once 存 smem），topk=6 是把量化结果 scatter 到
  6 个 expert 槽（fp4 写 6×≈5.5MB vs 写一次 0.9MB）。**这是 expert-contiguous 布局的代价**：换成
  “写一次+记地址让 remote gather”会把当前 wave0 暴露的 P2P copy 从连续块变 per-token gather（丢
  128B 向量化），在最贵/最暴露的路径上很可能净负。故不改。另有 `arrival_push_kernel<<<1,1>>>` 一个小 launch。

**剩余 Q2 杠杆（未在 overnight 做，风险高）：**
1. 融合 quant+preprocess+arrival 三 kernel → 省 2 个 launch（~10µs），但 preprocess 需等他 rank quant，
   融合后 kernel 会占 SM 自旋等 barrier；[[project_atomic_arrival_barrier]] 记录过跨-rank 可见性死锁，
   今晚也踩过一次 barrier 死锁。需专门 session 谨慎做。
2. 降 barrier 暴露（atomic-arrival）——同上风险。
结论：quant+preprocess 的可寻址 work 已小（compute 8µs + ~2 launch），**到 ≤0.024 的主要障碍是固有跨-rank
等待**，非 kernel micro-opt 能达成。

### 8.13 gen>0 (atomic-arrival) 正确性验证（2026-07-06）

背景:Test1 默认用 gen=0(跳过 Phase 2 arrival barrier + 恒 parity 0),从不验证 gen>0 的真实
perf 路径(atomic-arrival 屏障 + 双缓冲)。memory `project_atomic_arrival_barrier` 记录过 iter20
死锁/元数据坏的担忧。

**做法(第一次尝试失败=测法错):** 先写了个"复用 buffer 重跑 gen=1/2 与 gen0 golden 比对"的循环 →
diff≈0.8 FAILED。**加 g=0 控制组(完全同 gen0 路径)也 FAILED(diff 0.8)→ 判定是"重跑"本身不可信(harness 假象)**。
注:flags 不是原因——fused_dispatch 内部 `copy_ready_flags.zero_()`(dispatch_fused_gemm.py:547)每次
自动重置，复用 bc_fp4/flags 安全。真凶更可能是:重跑循环缺 golden 的 quant↔copy 同步(golden 在 quant
后有 `cuda.sync + dist.barrier`)→ gen=0 无 arrival 屏障时 copy 读到 peer 半写的远程 sym_buf；+ merged_sfa
只按 parity-0 建一次却给 parity-1 用;+ ep_buffer.destroy 后状态。未逐一坐实。回退该循环。

**正确做法:** 把整个 Test1(单次 + 完整 CPU/NF 参考)参数化 `BC_GEN`(默认 0);用 **BC_GEN=2**
(偶→parity 0，与 merged_sfa 的 parity-0 读一致)干净验证 arrival-barrier 路径。
**结果:8 卡 FULL_CORRECTNESS `BC_GEN=2` → 12/12 experts vs NF 和 CPU diff=0.000000，ALL PASSED。**

**结论:gen>0 atomic-arrival 屏障路径数值正确**(iter20 的死锁/坏元数据在本分支 c6fae89 已不复现，
`__threadfence_system` release + 本地 slot 轮询是 known-good)。**残留小盲区:parity=1(奇数 gen)** 未直接
验证——需让 `build_merged_sfa` 读 parity-1(parity 只是 `gen&1` 对称偏移，addressing 由 preprocess
统一计算，风险低)。诊断开关:`BC_GEN=<偶数>` 跑 Test1。

## 命令速查

```bash
# 容器内
cd /DeepGemm_workspace/codebase/DeepGemm-block-copy

# masked 配置扫描（功能验证 + 纯性能）
NC=8 PHASE=func bash tests/sweep_block_copy.sh   # 各配置 FULL_CORRECTNESS
NC=8 PHASE=perf REPEATS=3 bash tests/sweep_block_copy.sh  # 3轮 min

# 融合 GEMM grouping A/B（默认 nopad，masked 使用 per-expert padded 输出）
export FUSED_GEMM_GROUPING=masked  # 或 nopad

# per-wave 暴露(KTPF 可调) + copy 带宽扫描
SKIP_CORRECTNESS=1 KSTRIPE_PROFILE=1 K_TILES_PER_FLAG=0 COPY_BW_SWEEP=1 PERF_VERBOSE=1 \
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29581 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose

# 正确性
FULL_CORRECTNESS=1 CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29566 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose
```
