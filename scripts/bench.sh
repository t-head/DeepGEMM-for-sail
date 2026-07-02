#!/bin/bash
# AKO4ALL Bench Script — Block-copy fused dispatch GEMM1
# Usage: bash scripts/bench.sh [label]
#
# Kernel sources live directly under deep_gemm/ (single source of truth).
# Runs the multi-GPU benchmark inside docker container deepgemm.lxh and
# archives the kernel sources + output to trajectory/ for AKO iteration history.
set -eo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- GPU selection ---
# Default: GPU 0,1 (2-GPU). Override with GPUS env var.
# For 4-GPU: GPUS=0,1,3,6 bash scripts/bench.sh
GPUS="${GPUS:-0,1}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}
NCB="${NCB:-8}"

echo "=== AKO Bench: ${NPROC}-GPU (${GPUS}), ncb=${NCB}, label=${LABEL:-baseline} ==="

# --- Kernel source files (edited in place; deep_gemm/ is the single source) ---
INCLUDE_DIR="deep_gemm/include/deep_gemm"
JIT_KERNELS_DIR="deep_gemm/jit_kernels"
KERNEL_SRCS=(
    "${INCLUDE_DIR}/fp4_gemm_cutlass3.cuh"
    "${INCLUDE_DIR}/scheduler_cutlass3.cuh"
    "${INCLUDE_DIR}/dispatch_preprocess.cuh"
    "${JIT_KERNELS_DIR}/dispatch_fused_gemm.py"
)

# --- Container paths ---
CONTAINER="deepgemm.lxh"
CONTAINER_WS="/DeepGemm_workspace/codebase/DeepGemm-block-copy"

# --- Run benchmark inside docker ---
set +e
sudo docker exec \
    -e CUDA_VISIBLE_DEVICES="${GPUS}" \
    -e NCB="${NCB}" \
    -e FULL_CORRECTNESS=1 \
    -e SKIP_CORRECTNESS=0 \
    -w "${CONTAINER_WS}" \
    "${CONTAINER}" \
    torchrun --nproc_per_node="${NPROC}" \
        tests/test_block_copy_gemm1_multi_gpu.py \
    2>&1 | tee _bench_output.txt
BENCH_EXIT=${PIPESTATUS[0]}
set -e

# --- Trajectory ---
if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
for f in "${KERNEL_SRCS[@]}"; do
    [ -f "$f" ] && cp "$f" "$TRAJ_DIR/"
done
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
