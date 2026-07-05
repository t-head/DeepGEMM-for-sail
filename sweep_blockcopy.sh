#!/bin/bash
# ============================================================================
# block-copy fused GEMM1 —— 2/4/8 卡 ncb / exact-grid / stripe(KTPF) sweep
# ----------------------------------------------------------------------------
# 用法(在容器内、仓库根目录执行):
#   bash sweep_blockcopy.sh <nproc> <devices_csv> [portbase]
#   例:
#     bash sweep_blockcopy.sh 2 0,1               29500
#     bash sweep_blockcopy.sh 4 0,1,2,3           29600
#     bash sweep_blockcopy.sh 8 0,1,2,3,4,5,6,7   29700
#
# 前置:GPU 必须独占(否则计时被污染)。脚本 preflight 会检查目标卡是否空闲
#       (任一卡显存 >3GB 视为有人占用 → 等待);依赖 ppu-smi,没有则跳过检查。
#
# 输出:每个配置只打印制表所需字段(gemm_ctas / waves / overall exposure /
#       每 wave 行 / cumulative critical-path / implied clock / kernel-only wall /
#       local-only),并带 TS=时间戳,便于跟 GPU 监控日志按时间对齐。
#
# 测量方法论:每配置 3 轮取稳;wave0 有噪声、稳态(wave1+)极稳;
#       小 delta 看 kernel-only median,别信单轮。
#
# ⚠ 杀进程注意:本脚本+torchrun 跑在 docker 容器内,host 的 kill/pkill 杀不到
#       (PID namespace 不同,且外层 && 链会复活下一个 torchrun)。要中止须在
#       容器内按 PID 杀:
#       docker exec <ctr> bash -lc 'kill -9 $(ps -eo pid,cmd | \
#         grep -E "sweep_blockcopy|test_block_copy_gemm1" | grep -v grep | awk "{print \$1}")'
# ============================================================================
set -u
NPROC=${1:?need nproc};  DEVS=${2:?need devices csv};  PORT=${3:-29700}
cd "$(dirname "$0")"

preflight() {   # 等目标机器所有卡空闲
  command -v ppu-smi >/dev/null 2>&1 || { echo "[preflight] 无 ppu-smi,跳过空闲检查"; return 0; }
  while :; do
    mx=$(ppu-smi 2>/dev/null | grep -oE "[0-9]+MiB / 98304" | grep -oE "^[0-9]+" | sort -n | tail -1)
    [ "${mx:-0}" -lt 3000 ] && break
    echo "[preflight] 有卡占用(max ${mx}MiB > 3GB),30s 后重试…"; sleep 30
  done
  echo "[preflight] 目标卡空闲,开跑"
}

run() {   # args: exact ncb ktpf profile(0|1)
  local exact=$1 ncb=$2 ktpf=$3 prof=$4
  local ks_grep
  if [ "$prof" = "1" ]; then
    ks_grep="gemm_ctas|Overall exposure|^ +[0-9]+ +[0-9]+ +[0-9]+\.|Cumulative|Implied clock|Local-only timing"
  else
    ks_grep="block_m=.*best ncb|Pipeline \(full\)|Pipeline \(no preprocess\)|Kernel-only \(copy|Pipeline overhead|vs non-fused|Local-only timing"
  fi
  echo "@@@ nproc=$NPROC exact=$exact ncb=$ncb ktpf=$ktpf profile=$prof port=$PORT TS=$(date '+%H:%M:%S')"
  SKIP_CORRECTNESS=1 KSTRIPE_PROFILE=$prof K_TILES_PER_FLAG=$ktpf NCB_SWEEP=$ncb \
    FUSED_EXACT_GRID=$exact PERF_VERBOSE=1 CUDA_VISIBLE_DEVICES=$DEVS \
    torchrun --nproc_per_node=$NPROC --master_port=$PORT \
    tests/test_block_copy_gemm1_multi_gpu.py --verbose 2>&1 | \
    grep -iE "$ks_grep"
  PORT=$((PORT+1))
}

preflight
echo "========== PASS 1: clean perf (KSTRIPE_PROFILE=0) =========="
for r in 1 2 3; do
  echo "==================== ROUND $r (nproc=$NPROC) ===================="
  for ncb in 2 4 8;  do run 0 $ncb 0 0;  done
  for ncb in 2 3 4;  do run 1 $ncb 0 0;  done
  for ktpf in 1 2 4 7 14; do for ncb in 2 3 4; do run 1 $ncb $ktpf 0; done; done
done
echo "========== PASS 2: breakdown (KSTRIPE_PROFILE=1) =========="
for r in 1 2 3; do
  echo "==================== ROUND $r (nproc=$NPROC) ===================="
  for ncb in 2 4 8;  do run 0 $ncb 0 1;  done
  for ncb in 2 3 4;  do run 1 $ncb 0 1;  done
  for ktpf in 2 4 7 14; do for ncb in 2 3 4; do run 1 $ncb $ktpf 1; done; done
done
echo "==================== DONE nproc=$NPROC ===================="
