# SFA (A-scale) copy 优化：两套方案 + 实验开关说明

目标配置：8 卡 prod，`FUSED_EXACT_GRID=1`，`ncb=3`（M=1536, N=6144, K=7168，num_tokens=256/rank，topk=6）。
基线（列主序 SFA、in-fused pull）：pipeline 0.379ms，**0.67x** vs 非融合。上限（skip-SFA，输出无效）：kernel 0.248ms，**0.86x**。

---

## 背景：SFA copy 为什么慢

- SFA = A-scale，每 token `ksb=112` 个 uint16（E8M0，已最优打包，无可压缩量），约占 NVLink 传输量的 ~6%，却吃掉 ~26% kernel 时间。
- 8 卡（`nr==8`）路径下，原代码的 `nr==4` co-issue 是**死代码**：SFA 是 FP4 拷贝之后一段**完全串行、只有 ncb=3 个 CTA 在做**的尾巴（~0.088ms）。
- 瓶颈是**并行度 / 远端读往返延迟**，不是带宽、不是访存连续性（行主序单独只回收 ~0.01ms）、不是寄存器（vreg 恒 232、STACK 0）。

---

## 方案一：pull-preprocess + rowmajor（已验证，推荐，0.81x）

把 SFA gather 从「融合 kernel 内 3-CTA 串行尾巴」提到**独立全网格 pre-GEMM kernel**（用满所有 SM），GEMM 走 `skip_sfa_copy=True`。再叠行主序（远端读变连续 burst）。

- 机制：`dispatch_sfa_preprocess_kernel`（在 `fp4_gemm_cutlass3.cuh`）grid-stride 扫所有 M-block，复用 `copy_mblock_sfa` 把远端 SFA repack 进本地 `local_sfa_buf`；kernel 边界天然对 GEMM 的 SFA TMA 可见。
- 我额外把这个 preprocess kernel **从 ~12 个活跃 CTA 重并行到 ~132 个**（每个 M-block 拆 `sub=gridDim/total_m_blocks` 个 CTA，复用 `copy_mblock_sfa` 的 `(start,stride)`，不重不漏、无需 sync）。
- 结果：**kernel 0.276 / pipeline 0.315 / 0.81x，Test1 逐 expert 0 误差**。回收了 SFA 可用 headroom 的 ~74%（(0.81−0.67)/(0.86−0.67)）。
- 开启：`DG_SFA_PREPROCESS=1`（host env，运行时）+ `DG_SFA_ROWMAJOR_SRC=1`（编译期，三方布局契约）。**目前唯一严格正确且稳的生产可用配置。**

---

## 方案二：push（`DG_SFA_PUSH`，实验中，未定稿）

仿 DeepEP direct：quant 阶段把每 token 的 ksb 个 scale **push（P2P 写）**到 owner rank 的**独立** symm staging buffer（固定 per-source-rank band，slot = 本地 atomicAdd，无需前缀和、无 per-token 远端 atomic）；expert_preprocess 把 `rank_addr_sfa` 指向**本地** staging 切片；GEMM 前用**本地** reshape（复用 preprocess kernel + rowmajor 读）repack 进 `local_sfa_buf`。

数据流：`quant(push→对端 staging) → expert_preprocess(rank_addr_sfa=本地 staging) → 本地 reshape → 融合 GEMM(skip_sfa_copy)`。

**踩过并解决的两个硬坑：**
1. **staging 必须独立 symm buffer**。把 staging 追加进主 sym buffer（涨 ~11MB）会让该 buffer 的 remote FP4 P2P 读**慢 ~3x**（平台 per-buffer 大小阈值；实测再开一个独立 16MB buffer 不触发）。→ `get_sfa_staging_size` 单独分配，`set_sfa_staging_addrs` 注册（模块级，避免穿参所有 quant 调用点）。这也解释了原设计为何只往对端写极小 arrival flag、大数据全走 pull。
2. **64 位设备指针不能走 JIT `int`**（会截断成 32 位 → 非法地址）。staging 本地基址用 int64 tensor 的 `staging_addrs[rank_idx]` 传。

**性能/正确性权衡（演进，8卡 exact ncb=3）：**

| 变体 | quant+sym | pipeline | vs 非融合 | 正确性 |
|---|---|---|---|---|
| 原 folded（256 block，每 block 一次 system fence）| 0.089 | 0.366 | 0.69x | ✅ 严格 |
| **grid-stride**（block→~132 常驻单 wave，每 block system fence）| 0.066 | 0.351 | 0.73x | ✅ 严格（commit ed8bba7）|
| **stbl + 单 uncache**（见下 #2）| 0.054 | 0.330 | 0.77x | ✅ 严格（构造上）（commit 798fde3）|
| **+ atomic 前置(Phase0)+topk 并行**（**当前默认**，见下 #4）| **0.035** | **0.315** | **0.81x** | ✅ **严格**（当前默认路径）|
| coop（cudaLaunchCooperativeKernel + grid.sync + 单 uncache）| — | 0.359 | 0.71x | ✅ 严格但更慢，**已删除**（见 #3）|
| 非 folded（arrival_push 单 fence，历史）| 0.039 | 0.313 | 0.82x | ⚠️ 不严格，仅作 headroom 参考 |

> **当前 push 默认路径 = grid-stride + stbl 单 uncache fence + atomic 前置 = 0.81x 严格正确**（quant+sym 0.035 已低于历史非 folded 0.039；追平方案一 0.81x，逼近非 folded 上限 0.82x）。

**已完成（严格正确 0.69x → 0.77x，回收现实 headroom 大部分；非 folded 0.82x 是不严格上限）：**
1. **grid-stride**（原待做#2）：quant 从 1 block/token（256，~2 wave）改 grid-stride 到 occupancy 常驻上限（~132，单 wave）；每 block 处理完所有 token 后再 fence 一次。fence 数 256→~132、消除第二 wave 的往返暴露。→ 0.73x。
2. **stbl push + 折叠 arrival 单次 uncache fence**（原待做#3/#4 + 单 fence 合并，**默认路径**）：push 改 `__stbl`（**uncached / bypass-L1**）→ push **不进 per-SM 写回缓存** → **去掉 per-block fence**；retire 计数门保证所有 block 已发完 push → 末块一次 `__ppu_threadfence_system_uncache()`。**关键**：uncache fence 是**针对 uncached 访问的轻量 system fence——不做 cache flush**（配套的 stbl 本就没进 cache，无可刷），只保证 system-scope 顺序/完成；比普通 `__threadfence_system()`（要刷写回缓存）**更轻**。**构造上严格正确**（唯一 system fence 在末块），~15 轮 Test1 逐 expert 0 误差。→ 0.77x，quant+sym 0.054。`DG_QUANT_LASTFENCE=0` 退回旧 per-block `__threadfence_system` 做 A/B。
3. **coop（原待做#1，仿 DeepEP）已试并验证但更慢——已删除代码**：`cudaLaunchCooperativeKernel` + `cg::this_grid().sync()` + 单 uncache。严格正确，但 grid.sync 全局屏障 + coop launch 开销 > 省下的 per-block fence → 0.71x，慢于 last_fence。因无收益已从代码删除（不再有 `DG_QUANT_COOP`）。**保留的结论（重要）**：这条路验证了「grid 内所有 uncached stbl push 发完后，单个线程一次 uncache system fence 即可保证 system-scope 可见」——单 fence 覆盖全网格的正确性成立（配合 stbl uncached store，无需 cache flush）。这也是 last_fence（retire 计数门 + 单 uncache）默认路径正确的根据。
4. **atomic 前置 + topk 并行（收益最大的一步，0.77x→0.81x）**：slot 认领的 `atomicAdd` 原在量化（Phase 1）之后由 thread 0 **串行**发 topk 个，延迟压在关键路径。slot 与量化互不依赖（量化只用输入，slot 到 scatter/push 才用），故提到循环体最前（Phase 0），且一个 topk lane 一个线程并行发（token 的 topk experts 互不相同 → token 内无冲突）；atomic 往返与其他线程的 Phase 1 计算 overlap。slot 由 Phase 1 结尾的 `__syncthreads` 一并可见，删掉原冗余 sync。→ quant+sym 0.054→**0.035**（隐藏了 ~0.019ms 的 atomic 延迟）。
5. **dist.barrier 实验（原以为能省同步）无效——已删除**：`DG_PIPE_BARRIER=1` 每迭代 barrier → 0.70x（回退，暴露 launch 开销）；`=2` 循环前一次 → 0.74x（噪声内持平）。因 folded-push 的 arrival 机制（generation 打标 + 本地 slot 轮询）已把跨 rank 漂移吸收到 ~0.006ms，无可省。无收益（且 =1 回退），已删除测试代码。
6. **staging 读改 `__ldbl`（bypass-L1 load）**：push 写用 `__stbl`（绕 L1），reshape 读 push 进来的本地 staging 改用配套的 `__ldbl`（绕 L1），形成完整「绕 L1 写 + 绕 L1 读」——从 L2/mem 直读，不依赖 kernel 边界失效 L1 read-only cache 来避免读到隔 2 代同 parity 的陈旧值。性能与 `__ldg` 持平（reshape 读占比小、无复用），是零成本的正确性鲁棒性改进。仅 `DG_SFA_PUSH` 本地 staging 路径用；`DG_SFA_ROWMAJOR_SRC` 的远端 src 仍 `__ldg`。

**关键量化发现**：
- per-block fence（device 还是 system 差别在噪声内）固定 ~0.013ms；去掉它靠 `__stbl`（绕 L1）保证正确（stbl+单 uncache）。
- **atomic 延迟 ~0.019ms** 曾被埋在关键路径（Phase 1 之后串行发）；前置 + 并行后隐藏，是收益最大的一步。
- 当前 quant+sym 0.035ms 已低于历史非 folded 0.039；末块单次 uncache 的远端 drain 是正确性必需，是剩余可压空间的下限。

**已试但净零：metadata（counts）push（`DG_SFA_PUSH_COUNTS`，默认关）**。quant 末块把每 (src_rank,expert) 的最终 count `__stbl` push 到 owner 的 staging 尾部小区（与 SFA push 共用那次 uncache fence），expert_preprocess 改本地读。实测：preprocess remote-read 占比 22.7%→7.3%、expert_preprocess 0.021→0.018，**但** quant+sym 0.035→0.038，**pipeline 仍 0.315=0.81x → 净零**。原因：远端往返只是从「preprocess 远端读」搬到「quant 末块远端写」（串行尾巴、无法 overlap），未消除；本平台远端读≈远端写。metadata 仅 384B，省的是往返延迟不是带宽。故默认关，仅作 opt-in。

**仍可探索（收益递减）：**
- push 的 store 换 `__ppu_global_stbl_bulk_*`（stbl+bulk）：quant+sym 已低于非 folded，store 发射不是瓶颈，预计收益很小。
- 若要真省远端往返：需让 quant 的 counts push 与后续计算 overlap（现在卡在末块 arrival 尾巴）——但 counts 依赖全部 atomic 完成，难提前。

**更多待探索（明天一起看）：**
- **push 连 metadata 一起 push**：现在 counts（`expert_token_counts`）写在源自己 buffer，dest 在 expert_preprocess prepare **远端读**（那 ~0.024ms remote-read+compute + arrival 等待）。若源在 quant 里把 counts 也 push 到 dest，dest 就能**本地读** counts、去掉 prepare 的远端读延迟，且与 SFA push 天然 overlap。价值较高。
- **pull-preprocess 与 prepare 远端读 overlap**：受限——SFA gather 依赖 finalize 算出的 `rank_addr_sfa`（依赖 prepare 的 counts），不能直接并行；要 overlap 得重构（counts 一到即预取）。收益有限。

---

## 所有实验开关（含义 + 类型）

| 开关 | 类型 | 含义 |
|---|---|---|
| `DG_SFA_PREPROCESS=1` | host env（运行时）| 方案一：SFA gather 提到独立全网格 pre-GEMM kernel，GEMM `skip_sfa_copy=True`。无需重编译 GEMM。|
| `DG_SFA_ROWMAJOR_SRC=1` | 编译期（`-D`）| SFA 存布局改行主序 `[max_tok, ksb]`（每 token scale 连续）；quant/expert_preprocess/copy 三方契约，必须一起。|
| `DG_SFA_PUSH=1` | 编译期 + host | 方案二总开关：quant push、expert_preprocess 指本地 staging、host 分配 staging + 走 reshape + `skip_sfa_copy`。**跑正确性需配 `FUSED_ARRIVAL_IN_QUANT=1`（folded，严格正确）。** copy_mblock_sfa 在此宏下走 rowmajor 读路径。|
| `FUSED_ARRIVAL_IN_QUANT=1` | host env（既有）| arrival push 折进 quant 最后一个 block。push 的 folded 严格正确路径依赖它。不开则走非 folded（不严格）。|
| `DG_QUANT_LASTFENCE` | host env（**默认 1**）| push+folded 下用「retire 计数门 + 末块单次 `__ppu_threadfence_system_uncache()`」替代 per-block system fence（配合 `__stbl` 绕 L1 构造上正确）。`=0` 退回旧 per-block fence 做 A/B。|
| `DG_SFA_PUSH_COUNTS=1` | 编译期 | metadata（counts）push：quant 末块 push counts 到 owner staging，expert_preprocess 本地读。严格正确但本配置净零（远端读→远端写，未消除），默认关。需配 DG_SFA_PUSH。|
| `DG_PIPE_BARRIER` | host env（测试，默认关）| Test2 pipeline 循环加 `dist.barrier`：`=1` 每迭代（暴露 launch 开销，变慢），`=2` 循环前一次（噪声内持平）。诊断跨 rank 同步影响用。|
| `SKIP_SFA_COPY=1` | host（运行时）| 融合 kernel 里跳过 SFA copy（计时探针，**输出无效**），测上限。|
| `DG_SFA_NO_COISSUE=1` | 编译期 | A/B 探针：强制关掉 `nr==4` 的 in-FP4-loop co-issue（tv 变编译期 0、死代码消除）。用于证明 co-issue 净影响。|
| `DG_SFA_PUSH_NOWRITE=1` | 编译期 | 诊断：push 保留寻址/fence，但**跳过 store**。隔离「远端写 vs fence」的开销。|
| `DG_SFA_PUSH_SKIP_RESHAPE=1` | host | 诊断：push 模式下**跳过 reshape kernel**（保留 `skip_sfa_copy`，输出无效）。隔离 reshape 的开销。|

（既有诊断开关：`DG_SFA_SOURCE=host`、`KSTRIPE_PROFILE=1`、`DG_BULK_COPY/DG_BULK_REMOTE`，见 `SFA_COPY_OPTIMIZATION_TASK.md`。）

---

## 运行命令（8 卡目标配置）

```bash
# 方案一（推荐，验证正确 + 测性能）
docker exec -w /mnt/ssd/home/lixianghan.lxh/DeepGemm sglang.lxh bash -c '
FULL_CORRECTNESS=1 PERF_VERBOSE=1 FUSED_EXACT_GRID=1 NCB_SWEEP=3 \
DG_SFA_PREPROCESS=1 DG_SFA_ROWMAJOR_SRC=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29551 \
  tests/test_block_copy_gemm1_multi_gpu.py --verbose' 2>&1 | grep -v "WARN: use_mxfp4" | tail -60

# 方案二（push，严格正确的 folded 路径）
#   把上面的 DG_SFA_PREPROCESS/ROWMAJOR 换成： DG_SFA_PUSH=1 FUSED_ARRIVAL_IN_QUANT=1
```

改 header（.cuh）后需清 JIT cache（header 不进 cache hash）：
`docker exec sglang.lxh bash -c 'rm -rf ~/.deep_gemm/cache/kernel.*'`

---

## 改动文件

- `deep_gemm/jit/compiler.py`：`DG_SFA_PUSH` / `DG_SFA_PUSH_NOWRITE` / `DG_SFA_NO_COISSUE` 编译 flag。
- `deep_gemm/include/deep_gemm/dispatch_layout.cuh`：`staging_offset/staging_parity_bytes`（供 staging 索引；现独立 buffer，offset 从 0 起）。
- `deep_gemm/include/deep_gemm/mxfp4_quant.cuh`：quant 内 SFA push（int4 向量化写对端 staging）+ push 下 per-block fence 升 system 级。
- `deep_gemm/include/deep_gemm/expert_preprocess.cuh`：finalize 在 `DG_SFA_PUSH` 下把 `rank_addr_sfa` 指向本地 staging；`staging_addrs` 参数穿过 finalize/merged kernel + launcher。
- `deep_gemm/jit_kernels/dispatch_fused_gemm.py`：`get_sfa_staging_size` / `set_sfa_staging_addrs`（模块级 staging addrs）；quant/finalize/merged 的 args 增 `staging_addrs`；`DG_SFA_PUSH` 触发 reshape + `skip_sfa_copy`；`DG_SFA_PUSH_SKIP_RESHAPE` 诊断；preprocess kernel 重并行（在 .cuh）。
- `tests/test_block_copy_gemm1_multi_gpu.py`：两处 sym_buf 分配点在 `DG_SFA_PUSH` 下额外分配独立 staging symm buffer 并 `set_sfa_staging_addrs`。
