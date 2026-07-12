#!/bin/bash
# AKO4ALL Bench Script — Block-copy fused dispatch GEMM1 (pre-GEMM overhead focus)
# Usage:
#   bash scripts/bench.sh [label]           # perf SIGNAL (fast: SKIP_CORRECTNESS=1)
#   VERDICT=1 bash scripts/bench.sh [label] # full VERDICT (FULL_CORRECTNESS=1 + perf)
#
# Kernel sources live directly under deep_gemm/ (single source of truth); the JIT
# compiler auto-detects edits via MD5. Container mount is the same host dir.
# Measurement discipline (see memory feedback_measure_small_kernel_changes):
#   - fixed GPU pair (default 1,2) so A/B never crosses GPU-pair variance
#   - docker exec -i (stdin), unique master_port to avoid collisions
#   - the test already reports median-of-N; signal = Pipeline(full)/preproc/quant
set -eo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- GPU selection: fixed pair for stable A/B. Override with GPUS=... ---
GPUS="${GPUS:-1,2}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}
NCB="${NCB:-12}"
PORT="${PORT:-$((29500 + RANDOM % 400))}"

# --- Correctness gate: full only in VERDICT mode; signal runs perf-only ---
if [ "${VERDICT:-0}" = "1" ]; then
    SKIP_CORR=0; FULL_CORR=1; MODE="VERDICT (correctness+perf)"
else
    SKIP_CORR=1; FULL_CORR=0; MODE="SIGNAL (perf-only)"
fi

echo "=== AKO Bench [$MODE]: ${NPROC}-GPU (${GPUS}), ncb=${NCB}, port=${PORT}, label=${LABEL:-baseline} ==="

INCLUDE_DIR="deep_gemm/include/deep_gemm"
JIT_KERNELS_DIR="deep_gemm/jit_kernels"
KERNEL_SRCS=(
    "${INCLUDE_DIR}/mxfp4_quant.cuh"
    "${INCLUDE_DIR}/expert_preprocess.cuh"
    "${INCLUDE_DIR}/dispatch_layout.cuh"
    "${JIT_KERNELS_DIR}/dispatch_fused_gemm.py"
)

CONTAINER="sglang0512.lxh"
CONTAINER_WS="/DeepGemm_workspace/codebase/DeepGemm-block-copy-fusedopt"

set +e
sudo docker exec -i \
    -e CUDA_VISIBLE_DEVICES="${GPUS}" \
    -e NCB="${NCB}" \
    -e NCB_SWEEP="${NCB_SWEEP:-${NCB}}" \
    -e FUSED_EXACT_GRID="${FUSED_EXACT_GRID:-0}" \
    -e FUSED_ARRIVAL_IN_QUANT="${FUSED_ARRIVAL_IN_QUANT:-0}" \
    -e MERGED_PREPROCESS="${MERGED_PREPROCESS:-1}" \
    -e VALIDATE_MERGED="${VALIDATE_MERGED:-0}" \
    -e PERF_VERBOSE=1 \
    -e FULL_CORRECTNESS="${FULL_CORR}" \
    -e SKIP_CORRECTNESS="${SKIP_CORR}" \
    -w "${CONTAINER_WS}" \
    "${CONTAINER}" \
    torchrun --nproc_per_node="${NPROC}" --master_port="${PORT}" \
        tests/test_block_copy_gemm1_multi_gpu.py \
    2>&1 | tee _bench_output.txt
BENCH_EXIT=${PIPESTATUS[0]}
set -e

# --- Signal summary (grep the key lines) ---
echo ""
echo "--- signal ---"
grep -iE "quant \+ sym|expert_preprocess:|arrival barrier wait|Pipeline \(full|Pipeline \(no preproc|Kernel-only|Preprocess savings|vs non-fused:|Test 1:" _bench_output.txt || true

# --- Trajectory archive ---
if [ -n "$LABEL" ]; then TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"; else TRAJ_DIR="trajectory/${TIMESTAMP}"; fi
mkdir -p "$TRAJ_DIR"
for f in "${KERNEL_SRCS[@]}"; do [ -f "$f" ] && cp "$f" "$TRAJ_DIR/"; done
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
