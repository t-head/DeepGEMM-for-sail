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

**当前性能/正确性权衡：**

| 变体 | quant+sym | pipeline | vs 非融合 | 正确性 |
|---|---|---|---|---|
| folded（每 block 一次 system fence）| 0.089 | 0.366 | 0.69x | ✅ 严格正确（Test1 PASSED，用 `FUSED_ARRIVAL_IN_QUANT=1`）|
| 非 folded（arrival_push 单 fence）| 0.039 | 0.313 | 0.82x | ⚠️ **不严格**：异 kernel/异线程的 fence 盖不住 quant 的 push，靠平台 NVLink 保序+kernel drain 碰运气 |

**开销拆解（quant+sym_buf_zero）**：baseline 0.032 → NOWRITE（留 fence、去写）0.043（fence 本身只 +0.011）→ 全 push 0.089（**远端写 +0.046 为主因**）。folded 慢是因为 **256 个 block 各 fence 一次、等本 block 远端写落地**（分 ~2 wave，暴露 ~2× 往返延迟）。

**明天待做（把 push 做成严格正确且快）：**
1. **grid-once fence（首选，仿 DeepEP）**：quant 改 cooperative kernel → 所有 push 发完 `cg::this_grid().sync()` → **一次** `__ppukernel_threadfence_system_uncache_void()`（SDK 有）→ 置 flag。fence 只 1 次、覆盖全网格、严格正确。
2. **简单版：quant grid-stride** 到 ~num_sms 个 block（现在是严格 1 block/token = 256 个），fence 数 256→~132（约减半），改动最小。
3. **push 的 store 指令**：目前是普通 `st.global` 到 peer int4（`d4[v]=s4[v]`）；是否换 bulk（remote bulk 只能用于 remote 地址，peer VA 算不算待定）——**由你明天定**。
4. fence 加 `if(tid==0)` 守卫（现在 256 线程都在调，冗余）。

> ⚠️ **非 folded 的 0.82x 不能算数**（正确性无保证）。要 push 既快又对，必须做第 1/2 项。

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
| `FUSED_ARRIVAL_IN_QUANT=1` | host env（既有）| arrival push 折进 quant 最后一个 block。push 的 folded 严格正确路径依赖它（per-block system fence）。不开则走非 folded（不严格）。|
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
