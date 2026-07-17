# Handoff: 把 SFA copy/repack 拎进独立 preprocess kernel (宏门控的单独 path)

## ✅ 状态 (2026-07-12 已实现, 见 ITERATIONS.md 末条)
- **已交付**: `DG_SFA_PREPROCESS=1` host-side opt-in (默认关, 无编译宏, 不重编 GEMM)。独立 kernel
  `dispatch_sfa_preprocess_kernel` (fp4_gemm_cutlass3.cuh, 复用 copy_mblock_sfa 单一真源) + Python
  `dispatch_sfa_preprocess()` (dispatch_fused_gemm.py) + fused 函数里 env 触发 (GEMM 前发 kernel + skip_sfa_copy=True)。
- **正确性**: 4卡+8卡 FULL_CORRECTNESS bit-exact (vs_cpu=vs_nf=0)。
- **性能 (低延迟本机, 8卡 bm256 ncb8)**: pipeline 0.348 ≈ baseline 0.346 (中性, 符合预期; floor SKIP_SFA_COPY=1=0.330)。
  SFA 暴露 ~16µs 在本机已被 FP4 copy 掩盖, 拎出只是挪成串行前置 → net wash。
- **下一步 (待用户)**: 高延迟机 A/B `DG_SFA_PREPROCESS=1` vs `=0` (block_m=256, 同命令) —— 收益在那边。

## 任务
在本仓库 (`/mnt/ssd/lixianghan.lxh/codebase/DeepGemm-block-copy-fusedopt`, 容器 `sglang0512.lxh`,
挂载 `/DeepGemm_workspace/codebase/DeepGemm-block-copy-fusedopt`) 里, 把 fused GEMM1 里的 **SFA (A 的
scale) copy/repack** 从 fused GEMM 的 copy block 里**拎出来, 做成一个独立的 preprocess kernel**:
在 fused GEMM 之前把**所有 expert / 所有 M-block 的 SFA 一次性 gather+repack 到 `local_sfa_buf`**,
后面 fused GEMM 的 copy block **不再做 SFA**, GEMM 直接读已备好的 `local_sfa_buf`。
**用一个新宏 (建议 `DG_SFA_PREPROCESS`) 做成一条单独的 path, 默认关, 与现有路径正交。**

**动机**: 高延迟机器上 SFA 的影响挺大 (SFA 的 remote 读 + copy 暴露在 fused 关键路径上)。拎成
独立 pre-kernel 后, SFA 可提前完成 / 与其它 preprocess 重叠, 从 fused GEMM 关键路径上移除。

## 先读这些 memory (背景, 关键)
- `project_sfa_overlap_gemm1`: 当前 SFA 处理方式 —— nr4 把 SFA co-issue 进 FP4 nr4 copy loop;
  以及 `SKIP_SFA_COPY` 探针。**这是现状的直接前情。**
- `project_blockcopy_bulk_dma`: 我对 SFA 的分析 —— SFA 每行 `vecs_per_row=cnt/8≈16 int4`(256B) 比一个
  warp(32 lane=512B) 短 → warp 跨行非连续 → bulk/remote 不适合 SFA; SFA 体量仅 A 的 ~1/28;
  copy CTA=512 线程=16 warp。**做 pre-kernel 时这些 layout/线程事实直接可用。**
- `project_preprocess_fusion`: preprocess (prepare+finalize merged)、quant 各 phase、launch 开销机理。
  新 pre-kernel 怎么排布 launch / 与谁重叠, 参考这里。
- `feedback_measure_small_kernel_changes` / `feedback_verify_correctness` / `feedback_check_gpu_before_run`
  / `reference_docker_toolchain` / `project_ppu_hardware`。

## 现状 SFA 机制 (改之前务必对着当前代码核实, sfa 分支在演进)
- **源**: 每 rank 对称 buffer 里的 scale, 列主序 `[k_scale_blocks, max_tokens]`, K-stride = `max_tokens`;
  peer VA 由 preprocess 写进 `rank_addr_sfa[mb*num_ranks + r]` (已指到 kb=0、该 rank 的 token 偏移)。
  rank r 是 local iff `r == rank_idx` (见 SymBuffer::map, offset[rank_idx]=0)。
- **目标**: `local_sfa_buf` 每 expert `[k_scale_blocks, max_tokens]` 列主序, K-stride = `max_tokens`;
  行落在 `m_block_in_expert*BLOCK_M + rank_split_m`。这正是 GEMM 读 SFA 的 layout。
- **当前拷贝**: `deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh` 的 `copy_mblock_sfa(...)`
  (约 line 100+, 我加 bulk 后行号有移动, grep 定位) 按 M-block、所有 rank 做 gather+repack:
  - aligned 主体: `cnt%16==0 && aligned` 时按 int4(8×uint16) 向量搬; 否则标量 uint16 + 标量尾。
  - nr4: `copy_mblock_sfa<...,SkipVectorizedMain=true>` 只做残余, 主体 co-issue 进 run_copy_block 的 nr4 FP4 loop;
    其它 nr: `copy_mblock_sfa<...>` 做整份。
  - 由 `run_copy_block` / `run_copy_block_kstripe` 在 fused kernel 的 copy block 内调用。
- **元数据来源**: `expert_preprocess.cuh` 的 finalize/merged 已产出 `rank_addr_sfa / rank_split_m /
  rank_counts / grouped_layout (含 copy_grouped_layout: expert_local、base_block、m_block_in_expert 等)`。
  pre-kernel 复用这些即可, 无需重算路由。

## 已有可复用的探针/开关 (来自 sfa 分支最新 commit b16825e, 已在本分支)
- `SKIP_SFA_COPY=1` (env → `sched.skip_sfa_copy`): copy block 跳过 SFA (timing 探针, 输出无效)。
  **第一步先用它量出 SFA 在目标机(高延迟)上的暴露成本, 确认收益再动手。**
- `DG_SFA_SOURCE=host` (`sched.sfa_source_host`): GEMM 改从 host 建的 merged_sfa 读 SFA
  (`remote_addr_sfa[expert]`, K-stride=M), 而非 `local_sfa_buf`(K-stride=max_tokens)。这是 pre-0f40bad
  读路径。说明**读侧 layout 有两套**, 做 pre-kernel 时保持写 `local_sfa_buf`(GPU 侧 layout) 最省事。
- test 里 `build_merged_sfa` / `merged_sfa_addrs` (host 建 merged SFA, 用于 host-vs-gpu 自检)。

## 建议实现方向 (供参考, 自行判断)
1. 新增一个 device kernel `dispatch_sfa_preprocess`(放 `expert_preprocess.cuh` 或新文件), 输入同 copy 用的
   `rank_addr_sfa / rank_split_m / rank_counts / copy_grouped_layout / local_sfa_buf / rank_idx / ksb /
   max_tokens / BLOCK_M`, grid 覆盖**所有 total_m_blocks**, 每个 block 干一个 M-block 的全 rank SFA
   gather+repack —— 逻辑基本 = 现在的 `copy_mblock_sfa` 整份版, 只是独立成 kernel、跨所有 M-block。
   - 复用 `copy_mblock_sfa` 本体最省 (它已是"单一真源": 给它 (mb, expert_local, m_block_in_expert, start, stride))。
   - remote 读用普通 `__ldg` (peer 地址合法); 不要用 bulk/remote(见上, SFA 非连续不适合)。
2. Python 侧 (`deep_gemm/jit_kernels/dispatch_fused_gemm.py`): 宏 `DG_SFA_PREPROCESS` 开时,
   在 fused GEMM launch **之前**多发一个 sfa-preprocess kernel(同 stream 或独立 stream 预热), 并让
   fused GEMM 走 `skip_sfa_copy=true` (copy block 不做 SFA), GEMM 读 `local_sfa_buf` (默认读路径不变)。
   - 宏门控编译期: `compiler.py` 里按 `os.getenv('DG_SFA_PREPROCESS')` 加 `-DDG_SFA_PREPROCESS`
     (照抄我加 `DG_BULK_COPY`/`DG_BULK_REMOTE` 的写法, build() 里, 约 line 143 前)。
   - 或者做成 host 侧开关(不重编 kernel), 视你要不要 kernel 内也分叉而定。
3. 关键: **preprocess kernel 写完 local_sfa_buf 必须在 GEMM 读之前对 GEMM 的 TMA(L2 域)可见**
   —— 同 stream 顺序即可 (kernel 间有隐式同步); 若走独立 stream 需 event 同步。注意 L2/TMA 可见性
   (见 `project_copy_flag_l3_race`: 数据须在 L2 域, TMA 读 L2)。

## 环境 / 流程 / 纪律
- **测量纪律**: 每次跑前 `sudo docker exec sglang0512.lxh ppu-smi` 看空闲 (这机器共享, 常有别人
  8卡/多卡任务占用; 进程行判空闲: `ppu-smi | grep -cE '^\| +[0-9]+ +N/A +N/A +[0-9]+ +C '` == 0);
  每配置**至少空闲跑 3 次**看方差。GPU 忙时挑空闲卡跑 (如只 4,5,6,7 空则 4 卡 nr4)。
- **正确性 gate (硬性)**: `FULL_CORRECTNESS=1` 所有 expert `vs_cpu=vs_nf=0.000000` (bit-exact) 才算数;
  随机路由 correctness 段正好 block_m=256。命令模板:
  ```
  sudo docker exec -i -e CUDA_VISIBLE_DEVICES=4,5,6,7 -e DG_SFA_PREPROCESS=1 \
    -e FULL_CORRECTNESS=1 -e SKIP_CORRECTNESS=0 \
    -w /DeepGemm_workspace/codebase/DeepGemm-block-copy-fusedopt sglang0512.lxh \
    torchrun --nproc_per_node=4 --master_port=29xxx tests/test_block_copy_gemm1_multi_gpu.py
  ```
- **性能**: perf 段默认 block_m=128; 目标场景 block_m=256 用 `FORCE_EXPECTED_M=129`; `SKIP_CORRECTNESS=1
  PERF_VERBOSE=1`; 看 NCB sweep 的 ncb=8 pipeline + Pipeline(full)。若同时开 remote 需 `SKIP_ISOLATION=1`
  (all-local isolation 与 remote 指令不兼容; 见 ITERATIONS)。理想在高延迟机上测 (SFA 收益主要在那)。
- **编译**: 本地无 nvcc, 走容器 JIT (改 .cuh/.py 后跑测试自动按 MD5 重编)。nvcc 输出勿 pipe 到 head。
- **记录**: `ITERATIONS.md` + `HINTS.md` 按 AKO 流程 bench→log→commit。

## 当前 git 状态 (2026-07-12)
- 工作分支 `opt/fused-copy-block-gemm` (已 push origin), 基于 sfa 分支最新 `b16825e` (含 DG_SFA_SOURCE)。
- 已有正交的 opt-in 宏 (默认关, 别误动): `DG_BULK_COPY`(通用 bulk, neutral)、`DG_BULK_REMOTE`
  (remote 专用 load, 8卡/4卡 nr ~-2% bit-exact)、`DG_BULK_STWB`。`rank_idx` 已 plumb 到
  `sched.rank_idx` (copy loop 判 local/remote 用, 你的 SFA pre-kernel 判 local/remote 也能直接用)。
- 新任务的 `DG_SFA_PREPROCESS` 与上面这些正交, 独立一条 path。
- 存在快照分支 `opt/fused-copy-block-gemm-prerebase` (rebase 前, 可忽略/删)。
```
