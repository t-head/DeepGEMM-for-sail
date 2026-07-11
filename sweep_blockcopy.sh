#!/bin/bash
# ============================================================================
# block-copy fused GEMM1 —— 2/4/8 卡 SFA 归因 sweep
# ----------------------------------------------------------------------------
# 用法(在容器内、仓库根目录执行):
#   bash sweep_blockcopy.sh <nproc> <devices_csv> [portbase]
#   例:
#     bash sweep_blockcopy.sh 2 0,1               29500
#     bash sweep_blockcopy.sh 4 0,1,2,3           29600
#     bash sweep_blockcopy.sh 8 0,1,2,3,4,5,6,7   29700
#
# ★ SFA 归因(三个 mode,可用 MODES 覆盖):
#     base    = DG_SFA_SOURCE=gpu  SKIP_SFA_COPY=0   (当前默认)
#     skipsfa = DG_SFA_SOURCE=gpu  SKIP_SFA_COPY=1   (copy 侧 SFA 搬运 A/B,输出无效只计时)
#     host    = DG_SFA_SOURCE=host SKIP_SFA_COPY=0   (GEMM 读侧 SFA A/B,读 host merged_sfa,K-stride=M)
#   → base vs host 的 mainloop cycle 差 = 读侧代价;base vs skipsfa 的 wait cycle 差 = 搬运暴露。
#
# 采集的 cycle(PASS2 KSTRIPE_PROFILE=1):
#   - GEMM per-tile:wait / mainloop(→ Overall exposure、每 wave avg_wait/avg_compute、
#     Cumulative critical-path、Implied clock)
#   - preprocess in-kernel clock64:setup+barrier / remote-reads / compute(P4+5)(跨 rank barrier 读数)
#   - Pipeline Overhead Breakdown:quant / expert_preprocess / arrival barrier wait / DeepEP dispatch
#   - host mode 额外打印 host-vs-gpu bit-exact self-check
#   PASS1(KSTRIPE_PROFILE=0)是 wall-clock ms(Pipeline full/no-preproc/kernel-only/vs 非融合/Local-only)。
#
# 前置:GPU 必须独占(否则计时被污染)。preflight 检查目标卡是否空闲(任一卡显存 >3GB 视为占用 → 等待)。
#
# 测量方法论:每配置 ROUNDS 轮取稳;wave0 有噪声、稳态(wave1+)极稳;cycle 是 SM cycle 非时间,
#       换算 µs 用 "Implied clock" 校准;host/gpu 与 skip 的 cycle 差/比受频率影响小。
#
# 快速验证采集(不跑全矩阵):
#   MODES="base skipsfa host" ROUNDS=1 NCB_E0="" NCB_E1="3" DO_KTPF=0 \
#     bash sweep_blockcopy.sh 8 0,1,2,3,4,5,6,7 29720
#
# ⚠ 杀进程注意:脚本+torchrun 跑在 docker 容器内,host 的 kill/pkill 杀不到(PID namespace 不同,
#       外层 && 链会复活下一个 torchrun)。要中止须在容器内按 PID 杀:
#       docker exec <ctr> bash -lc 'kill -9 $(ps -eo pid,cmd | \
#         grep -E "sweep_blockcopy|test_block_copy_gemm1" | grep -v grep | awk "{print \$1}")'
# ============================================================================
set -u
NPROC=${1:?need nproc};  DEVS=${2:?need devices csv};  PORT=${3:-29700}
cd "$(dirname "$0")"

# --- 可用 env 覆盖的矩阵(默认 = 全量;快速验证时收窄)---
MODES=${MODES:-"base skipsfa host"}    # base|skipsfa|host 的子集
ROUNDS=${ROUNDS:-3}
NCB_E0=${NCB_E0-""}              # exact-grid=0 的 ncb 列表(设为空串 NCB_E0='' 可跳过)
NCB_E1=${NCB_E1-"2 3 4"}             # exact-grid=1 的 ncb 列表(设为空串可跳过)
KTPF_LIST=${KTPF_LIST-"2 4 7 14"}    # k-stripe 粒度
DO_KTPF=${DO_KTPF:-1}                  # 是否扫 KTPF
DO_PASS1=${DO_PASS1:-1}               # wall-clock pass
DO_PASS2=${DO_PASS2:-1}               # KSTRIPE cycle pass

# 两个 pass 都要抓的公共行(SFA 归因 + preprocess/barrier cycle + 前处理 breakdown)
COMMON_GREP='self-check|DG_SFA_SOURCE=host:|quant \+ sym_buf|expert_preprocess|arrival barrier|DeepEP dispatch|setup\+barrier|remote-reads|compute\(P4|total in-kernel'

preflight() {   # 等目标机器所有卡空闲
  command -v ppu-smi >/dev/null 2>&1 || { echo "[preflight] 无 ppu-smi,跳过空闲检查"; return 0; }
  while :; do
    mx=$(ppu-smi 2>/dev/null | grep -oE "[0-9]+MiB / 98304" | grep -oE "^[0-9]+" | sort -n | tail -1)
    [ "${mx:-0}" -lt 3000 ] && break
    echo "[preflight] 有卡占用(max ${mx}MiB > 3GB),30s 后重试…"; sleep 30
  done
  echo "[preflight] 目标卡空闲,开跑"
}

run() {   # args: mode exact ncb ktpf profile(0|1)
  local mode=$1 exact=$2 ncb=$3 ktpf=$4 prof=$5
  local dg skip
  case "$mode" in
    base)    dg=gpu;  skip=0 ;;
    skipsfa) dg=gpu;  skip=1 ;;
    host)    dg=host; skip=0 ;;
    *) echo "[run] 未知 mode=$mode(用 base|skipsfa|host)"; return 1 ;;
  esac
  local ks_grep
  if [ "$prof" = "1" ]; then
    ks_grep="gemm_ctas|Overall exposure|^ +[0-9]+ +[0-9]+ +[0-9]+\.|Cumulative|Implied clock|Local-only timing|$COMMON_GREP"
  else
    ks_grep="block_m=.*best ncb|Pipeline \(full\)|Pipeline \(no preprocess\)|Kernel-only \(copy|Pipeline overhead|vs non-fused|Local-only timing|$COMMON_GREP"
  fi
  echo "@@@ nproc=$NPROC mode=$mode(dg=$dg skip=$skip) exact=$exact ncb=$ncb ktpf=$ktpf profile=$prof port=$PORT TS=$(date '+%H:%M:%S')"
  SKIP_CORRECTNESS=1 DG_SFA_SOURCE=$dg SKIP_SFA_COPY=$skip KSTRIPE_PROFILE=$prof \
    K_TILES_PER_FLAG=$ktpf NCB_SWEEP=$ncb FUSED_EXACT_GRID=$exact PERF_VERBOSE=1 \
    CUDA_VISIBLE_DEVICES=$DEVS \
    torchrun --nproc_per_node=$NPROC --master_port=$PORT \
    tests/test_block_copy_gemm1_multi_gpu.py --verbose 2>&1 | \
    grep -iE "$ks_grep"
  PORT=$((PORT+1))
}

run_matrix() {   # args: profile(0|1)
  local prof=$1
  for mode in $MODES; do
    for ncb in $NCB_E0; do run $mode 0 $ncb 0 $prof; done
    for ncb in $NCB_E1; do run $mode 1 $ncb 0 $prof; done
    if [ "$DO_KTPF" = "1" ]; then
      for ktpf in $KTPF_LIST; do for ncb in $NCB_E1; do run $mode 1 $ncb $ktpf $prof; done; done
    fi
  done
}

preflight
if [ "$DO_PASS1" = "1" ]; then
  echo "========== PASS 1: clean perf (wall-clock, KSTRIPE_PROFILE=0) =========="
  for r in $(seq 1 $ROUNDS); do
    echo "==================== ROUND $r (nproc=$NPROC) ===================="
    run_matrix 0
  done
fi
if [ "$DO_PASS2" = "1" ]; then
  echo "========== PASS 2: cycle breakdown (KSTRIPE_PROFILE=1) =========="
  for r in $(seq 1 $ROUNDS); do
    echo "==================== ROUND $r (nproc=$NPROC) ===================="
    run_matrix 1
  done
fi
echo "==================== DONE nproc=$NPROC ===================="
