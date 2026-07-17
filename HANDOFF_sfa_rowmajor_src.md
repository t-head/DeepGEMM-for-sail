# Handoff: SFA 对称 buffer 改行主序 source(连续 remote load + 列主序 local store)

## 状态:仅设计,未动代码(2026-07-12)
这是把 SFA 从"碎+跨步远端读"变"连续 burst"的实质优化,与已交付的 `DG_SFA_PREPROCESS`
(commit ba808ec,把 SFA 拎进独立 pre-kernel)**正交且互补**——行主序改的是"读的形状",
pre-kernel 改的是"在哪读";两者都通过复用 `copy_mblock_sfa` 受益。

## 动机(核心)
现状 quant 写 SFA 到对称 buffer 是**列主序** `[ksb, max_tok]`(K-stride=max_tok),
所以 copy 的 remote 读 = 每 rank **ksb=112 个跨步段**(每段 cnt≈130 token 连续,段间跨 max_tok)。
高延迟机上 112 次跨步远端访问的延迟被成倍放大(SFA 慢的真因是"碎+跨步"非量大,仅 336KB)。

**改法**:把 source 改**行主序** `[max_tok, ksb]`(每 token 的 ksb 个 scale 连续)。则:
- copy 的 **remote 读**:一个 rank 的 cnt 个连续 slot × ksb = **一整块 ~29KB 连续**(cnt130×ksb112×2B)
  → 从 112 个延迟型跨步段变成 **1 次带宽型 burst**。
- **store 仍列主序**写进 `local_sfa_buf`(dst[kb*max_tok+t])——本地 HBM 写,跨步便宜、可 coalesce。
- **GEMM 读 local_sfa_buf 完全不变**(dst 仍列主序)——不碰 GEMM。
- **额外好处**:quant 现在的写本身就列主序跨步(每 token 的 ksb 个 scale 各写不同 row),
  改行主序后 **quant 写也变连续**,quant 可能顺带小提速。

⚠️ 与 memory 里被否掉的"行主序根治"**不同**:那个是把 **dst(local_sfa_buf)**改行主序,害了 GEMM 读;
这里 **dst 保持列主序**,只改 **source(对称 buffer)**。

## 关键常量
`ksb = K_SCALE_BLOCKS = ceil(hidden/64)`(prod hidden=7168 → 112);`max_tok = max_tokens_per_expert`(prod 256)。
source 是每 rank 对称 buffer 里 `scale_ptr(expert)` 指向的区域,每 expert 大小 `ksb*max_tok` uint16。

## 3 个改动点(布局是 writer/reader/addresser 三方契约,必须一起翻)
1. **quant 写** — `deep_gemm/include/deep_gemm/mxfp4_quant.cuh:~234`
   ```cpp
   // 现状(列主序): scale_out[j * max_tokens_per_expert + slot] = s_packed_scale[j];
   // 改为(行主序): scale_out[(int64_t)slot * K_SCALE_BLOCKS + j] = s_packed_scale[j];
   ```
   (K_SCALE_BLOCKS 是该 kernel 里的 constexpr,line ~108。注意 scatter 循环 `for j<K_SCALE_BLOCKS`。)
2. **preprocess 写 rank_addr_sfa** — `deep_gemm/include/deep_gemm/expert_preprocess.cuh:~285-286`
   ```cpp
   // 现状: rank_addr_sfa[idx] = (uint64_t)(remote_scale + rank_offset[r]);          // token 偏移(列主序 kb=0 行内)
   // 改为: rank_addr_sfa[idx] = (uint64_t)(remote_scale + (uint64_t)rank_offset[r] * ksb);  // 行主序 token 偏移
   ```
   ksb 由 `hidden_dim` 推:`uint32_t ksb = ((hidden_dim/2) + 31) / 32;`(= ceil(hidden/64))。
   `dispatch_expert_finalize_device` 签名里有 `hidden_dim` 和 `max_tokens_per_expert`,现成可用。
3. **copy 读** — `deep_gemm/include/deep_gemm/fp4_gemm_cutlass3.cuh` 的 `copy_mblock_sfa`(~line 119-185)
   现状:src 列主序,内层按 kb-row 连续读 cnt token、行间跳 max_tok;dst 同样列主序连续写。
   改为:**src 行主序**——一个 rank 的数据是 `[cnt, ksb]` 连续块(base=rank_addr_sfa[idx],
   token t 的 ksb 个 scale 在 `base + t*ksb`);**读连续** `cnt*ksb` 个 uint16(可 int4 向量化,
   cnt*ksb 通常 16B 对齐),**写列主序** dst[kb*max_tok + (mrow_base+split+t)]。
   即 load index = t*ksb+kb 连续,store index = kb*max_tok+row 跨步。tail/非对齐照旧 plain。
   ⚠️ `copy_mblock_sfa` 是**单一真源**,被 inline copy block + 我的 `dispatch_sfa_preprocess_kernel` 共用
   → 改它两条路径都自动跟着变(好事,但测试要两条都过)。

## 门控与兼容
- **必须编译宏门控**(如 `DG_SFA_ROWMAJOR_SRC`,默认关),3 处同一宏一起翻——布局是契约,不能半翻。
  照抄 `DG_BULK_COPY`/`DG_BULK_REMOTE` 在 `compiler.py` build() 里加 `-D` 的写法(约 line 143 前)。
  注意 quant 是**独立 JIT 模块**(`_mxfp4_quantize_to_sym_buffer`),它的 build 也要吃到这个宏
  (确认 compiler.py 对所有 kernel 统一加 -D,或 quant 单独处理)。
- **DG_SFA_SOURCE=host 的 merged builder**(test `build_merged_sfa`)假设列主序 source,
  开 DG_SFA_ROWMAJOR_SRC 时要同步改其重建逻辑,或在该 flag 下禁用 host self-check。
- **self-rank(local)**:rank==rank_idx 读自己的对称 buffer,同布局,自动一致。

## 正确性 gate(硬)
`FULL_CORRECTNESS=1` 4 卡 + 8 卡所有 expert `vs_cpu=vs_nf=0.000000`(纯数据搬运顺序变化,scale 值不变 → 必 bit-exact)。
命令见 HANDOFF_sfa_preprocess.md;correctness 段随机路由=block_m 256。**DG_SFA_PREPROCESS 开/关都要各过一遍**
(因为两条路径共用 copy_mblock_sfa)。

## 性能测量(⚠️ 本机现状)
- 这台机(sglang0512.lxh)被别人的 DeepSeek 服务**间歇占用 + host 高 load**,**pipeline 计时不可信**
  (跨 rank arrival barrier 空等被吹大:干净 0.000ms → 污染 0.037ms);**kernel-only 是纯本地 GEMM,始终可信**。
  测时用 load 门控(busy=0 且 load1<3~4),多轮取稳;或直接看 kernel-only。
- 收益主要在**高延迟机**(碎跨步→连续 burst 的价值在那);低延迟本机预计中性偏微正(远端事务数骤降)。
  理想让用户在高延迟机 A/B `DG_SFA_ROWMAJOR_SRC` 0 vs 1。

## 当前 git / 协调
- 分支 `opt/fused-copy-block-gemm` @ **ba808ec**(DG_SFA_PREPROCESS)+ 未 commit:test 的 FORCE_EXPECTED_M
  对称公平性修复 + 若干 sweep/wait 脚本。
- ⚠️ **与本 session 冲突点**:`copy_mblock_sfa` 三处改动点之一被本 session 的 SFA-preprocess 复用;
  若另开 session 做,**强烈建议单独 branch/worktree**(避免同文件竞争),且**别与本 session 同时跑 benchmark**
  (机器已被第三方占,再叠加两个 8 卡 perf loop 会互相污染)。
