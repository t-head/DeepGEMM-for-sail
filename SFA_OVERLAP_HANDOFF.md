# SFA-overlap 优化 handoff(block-copy fused GEMM1)

任务:评估并优化 block-copy fused GEMM1 里 **SFA copy 的隐藏** —— 把 SFA 的拉取融进 FP4(token)
copy,让访存 load 连续发射(MLP),把 SFA 的延迟藏进 FP4 的带宽型传输里。先验证暴露程度,再实现,
重点看多卡收益。

## 背景 / 大目标
在攻 block-copy 的 **kernel 内 copy+GEMM** gap(2 卡 BC kernel-only 0.255 vs NF 纯 GEMM 0.194 ≈ 61µs;
其中 P2P link 2 卡 ~21µs、4 卡 ~40µs)。已实测:Local-only(无 NVLink)2→4 卡几乎不变(0.234→0.237),
**多卡掉的部分全在 P2P copy**;vs non-fused 2 卡 0.925x、4 卡 0.79x。SFA copy 随卡数退化比 FP4 快
(见下"为什么"),是这条线的一个具体优化点。

## 环境
- 容器 sglang.lxh,WS=/DeepGemm_workspace/codebase/DeepGemm-block-copy。
- 命令走 `sudo docker exec -w $WS sglang.lxh ...`;跑容器内 python/torchrun **必须加 `-i`**(否则 stdin
  不传、静默空跑)。若已 `docker exec ... bash` 进容器内则不用 `-i`。
- 主机侧同目录挂在 /mnt/ssd/lixianghan.lxh/codebase/DeepGemm-block-copy(可直接编辑 .cuh,JIT 按 MD5
  自动重编,无需清 cache;想强制干净重编:`rm -rf /root/.deep_gemm/cache`)。
- PPU ZW-M890P:仅 39 个 SM,ICN8 全连接 NVLink。编译 `-gencode=arch=compute_89,code=sm_89`。

## 分支
- 从 **feature/block-copy-gemm1 @ 79d886b** 起一个新分支(该 commit 已含上一轮 pre-GEMM 优化,已推远端
  gitlab PPU-Libraries/DeepGemm)。别动 quant/preprocess(pre-GEMM 已榨干)。

## 卡状态(重要)
- 跑前 `ppu-smi`。**GPU 0 目前处于 HW error 状态**(显示 100% util 但无进程,报 "HW under unreliable
  state, a RESET operation is needed");用它会直接挂。GPU 1-7 健康。
- **non-fused 对照(DeepEP)只支持 world_size ∈ {2,4,8,16}**。所以:2 卡(如 1,2)、4 卡(如 1,2,3,4)
  可跑;**8 卡需要先 reset GPU 0**(`ppu-smi -r -i 0`,共享机器,重置前确认无人用、且可能需宿主机权限);
  7 卡跑不了(DeepEP 断言)。

## 当前 copy 结构(已核实)
文件 deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh:
- 默认路径 `run_copy_block`(~L200-285):先 FP4 rank 循环(4-way unroll `ld_nc_global` 搬 int4)→
  **然后**单独调 `copy_mblock_sfa`(L276)→ 之后才 `__syncthreads()+__threadfence()+copy_ready_flags[mb]=1`。
  **FP4 和 SFA 两段之间没有 barrier**,只有最后一个 fence。→ SFA 是跟在 FP4 后的一条尾巴。
- kstripe 路径 `run_copy_block_kstripe`(L287+):SFA 在 stripe 前 up-front 单独搬一段,同样是独立段。
- `copy_mblock_sfa`(L61):SFA 列主序 [k_scale_blocks=112, max_tokens=256] per expert;按 `for r<NumRanks`
  gather+compact;每 rank 搬 ksb×cnt 个 uint16;cnt≥8 且对齐时 int4 向量化(8 uint16/次),否则标量 2B。

## 关键量化(为什么 SFA 值得藏 / 随卡数放大)
- 每 rank 每 M-block:FP4 = `cnt×224` 个 int4;SFA = `112×(cnt/8) = 14×cnt` 个 int4 → **SFA 只有 FP4 的 1/16**。
  把 SFA load 铺进 FP4 循环的前 ~1/16 迭代就能被 FP4 传输罩住。
- `cnt ≈ BLOCK_M/W`(uniform):2 卡 cnt≈64(满向量化,尾巴小);多卡 cnt 变小(8 卡≈16,更多卡 <8 退标量
  2B 跨步远端读)→ SFA 尾巴变大;而 FP4 也更大(更多远端)→ 更好藏。**收益主要在多卡。**

## 步骤(先验证,再实现,别盲改)
1. **先量 SFA 暴露多少**(否则可能已被 TLP 掩盖、白改):
   - 探针法:给 `copy_mblock_sfa` 加个 env 门跳过它,用 `copy_only=True`(见 dispatch_fused_gemm.py 的
     COPY_BW_SWEEP 路径)测 有/无 SFA 的 copy-only 时间差 = SFA 净暴露。仅计时,测完回滚。
   - ISA 法:`hgobjdump --dump-isa <kernel.so>`(JIT 产物在 /root/.deep_gemm/cache/kernel.*/kernel.so),
     看 SFA 的 `vmem.ld` 后是不是紧跟 `s.wait vldcnt(0)` 把它和 FP4 隔开(上一轮 quant 就是靠这个发现串行)。
   - 在 2 卡和 4 卡分别量(看随卡数变化)。
2. **实现**(若确认暴露):把 SFA 的 int4 load **融进 FP4 的 unrolled 循环**(发 FP4 v0..v3 的同时发 SFA
   load、一起 store),让 load 连续发射拿 MLP。⚠️ **只把 copy_mblock_sfa 挪到 FP4 前面(reorder)没用**
   —— 那还是两段串行、发射窗口没合并;必须真正 interleave。两条路径(run_copy_block + kstripe)都要改。
3. **验 ISA**:改完再 dump-isa,确认 SFA load 和 FP4 load 连发、SFA 后不再有独立 vldcnt(0)。
4. **正确性**:SFA/FP4 仍都在 fence+flag 之前完成 → 逻辑不变,但必须过 bit-exact
   (FULL_CORRECTNESS=1,vs CPU 0.000000)。

## 测量纪律(踩过的坑)
- 用 tests/test_block_copy_gemm1_multi_gpu.py(torchrun --nproc_per_node=2/4)。
- perf:`SKIP_CORRECTNESS=1 PERF_VERBOSE=1 NCB_SWEEP=8,12,16 FUSED_EXACT_GRID=0`;
  看 Kernel-only(copy+GEMM)、Local-only、P2P link overhead、NCB sweep;KSTRIPE_PROFILE=1 看 per-wave exposure。
- 正确性:`FULL_CORRECTNESS=1`(vs CPU + vs non-fused)。
- A/B 用**同一固定 GPU 对/组**;小改动多轮 clean-min,**累计超噪声(~3-5µs)才算数**(单改常落噪声内)。
- 仓库有 scripts/bench.sh(VERDICT=1 切正确性门;注意它硬编码 CONTAINER=sglang.lxh、GPUS 默认 1,2)。

## 工具(上一轮证明极有效)
- `acu`(/usr/local/PPU_SDK/asight/bin/acu,PPU 版 ncu):看 Duration/No-Eligible/Issue Slots/Mem Busy/
  Max Bandwidth/DRAM。分布式 replay 可能死锁,必要时写单 GPU 微基准隔离。
- `hgobjdump --dump-isa <kernel.so>`(/usr/local/PPU_SDK/bin/hgobjdump):dump 设备 ISA。ISA 级审计
  比只看 wall-clock 靠谱,查 `s.wait vldcnt/vstcnt` 串行、threadfence 的 wbinv 位置。

## 先读这些 memory(坑已踩过)
- project_blockcopy_isa_audit(copy 访存已近最优、threadfence wbinv 贵、跨节点须 RDMA)
- project_blockcopy_ktpf_exposure(KTPF=0 已掩盖 91%、暴露在 wave0、round-robin ncb8、多轮取稳)
- project_p2p_overhead_analysis(block-copy P2P 是显著瓶颈)
- project_blockcopy_gpu_side_sfa(GPU 侧 SFA copy/repack 的由来,commit 32959eb,无开关默认生效)
- project_preprocess_fusion(上一轮 pre-GEMM,含 ISA 审计方法论:hgobjdump 发现 vldcnt(0) 串行)
- HINTS.md(原始 top direction: persistent copy blocks —— copy 完转 GEMM,回收 ~20% 闲置 SM)

## 相关文件
- deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh(copy_mblock_sfa L61;run_copy_block ~L200-285;
  run_copy_block_kstripe L287+;都在改动范围)
- deep_gemm/include/deep_gemm/scheduler_cutlass3.cuh(TileSchedulerArguments)
- deep_gemm/jit_kernels/dispatch_fused_gemm.py(旋钮 num_copy_blocks/k_tiles_per_flag/copy_mode/copy_only)
