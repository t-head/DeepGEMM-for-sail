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

# --- GPU selection: production topology by default. Override with GPUS=... ---
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}
NCB="${NCB:-3}"
PORT="${PORT:-$((29500 + RANDOM % 400))}"
RUNS="${RUNS:-3}"

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

CONTAINER="${CONTAINER:-sglang.lxh}"
CONTAINER_WS="${CONTAINER_WS:-$(pwd)}"

: > _bench_output.txt
BENCH_EXIT=0
for RUN_INDEX in $(seq 1 "$RUNS"); do
    echo "=== independent run ${RUN_INDEX}/${RUNS} ===" | tee -a _bench_output.txt
    SMI_OUTPUT="$(ppu-smi)"
    printf '%s\n' "$SMI_OUTPUT" | tee -a _bench_output.txt
    if ! grep -Fq "No running processes found" <<< "$SMI_OUTPUT"; then
        echo "ABORT: PPU machine is not idle before run ${RUN_INDEX}." | tee -a _bench_output.txt
        BENCH_EXIT=75
        break
    fi
    ppu-smi -q -d CLOCK | tee -a _bench_output.txt
    set +e
    docker exec -i \
    -e CUDA_VISIBLE_DEVICES="${GPUS}" \
    -e TEST_CONFIG="${TEST_CONFIG:-prod}" \
    -e FORCE_EXPECTED_M="${FORCE_EXPECTED_M:-128}" \
    -e DG_SFA_PUSH="${DG_SFA_PUSH:-1}" \
    -e DG_PPU_LEGACY_FLAGS="${DG_PPU_LEGACY_FLAGS:-0}" \
    -e DG_NVCC_OVERRIDE_CPP_STANDARD="${DG_NVCC_OVERRIDE_CPP_STANDARD:-20}" \
    -e DG_SFA_PUSH_COUNTS="${DG_SFA_PUSH_COUNTS:-0}" \
    -e DG_SFA_PUSH_INLINE="${DG_SFA_PUSH_INLINE:-0}" \
    -e DG_SFA_OVERLAP="${DG_SFA_OVERLAP:-1}" \
    -e ASYS_FULL_ONLY="${ASYS_FULL_ONLY:-0}" \
    -e ASYS_FULL_ITERS="${ASYS_FULL_ITERS:-10}" \
    -e DG_SFA_NUM_SMS="${DG_SFA_NUM_SMS:-39}" \
    -e DG_SFA_NUM_THREADS="${DG_SFA_NUM_THREADS:-256}" \
    -e NCB="${NCB}" \
    -e NCB_SWEEP="${NCB_SWEEP:-${NCB}}" \
    -e DG_BULK_COPY="${DG_BULK_COPY:-0}" \
    -e DG_BULK_PIPE2="${DG_BULK_PIPE2:-0}" \
    -e DG_BULK_ROLL2="${DG_BULK_ROLL2:-0}" \
    -e DG_COPY_PIPE2="${DG_COPY_PIPE2:-1}" \
    -e K_TILES_PER_FLAG="${K_TILES_PER_FLAG:-14}" \
    -e FUSED_EXACT_GRID="${FUSED_EXACT_GRID:-0}" \
    -e FUSED_ARRIVAL_IN_QUANT="${FUSED_ARRIVAL_IN_QUANT:-1}" \
    -e MERGED_PREPROCESS="${MERGED_PREPROCESS:-1}" \
    -e VALIDATE_MERGED="${VALIDATE_MERGED:-1}" \
    -e PERF_VERBOSE=1 \
    -e SKIP_ISOLATION="${SKIP_ISOLATION:-0}" \
    -e FULL_CORRECTNESS="${FULL_CORR}" \
    -e SKIP_CORRECTNESS="${SKIP_CORR}" \
    -w "${CONTAINER_WS}" \
    "${CONTAINER}" \
    torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${PORT}" \
        tests/test_block_copy_gemm1_multi_gpu.py \
    2>&1 | tee -a _bench_output.txt
    RUN_EXIT=${PIPESTATUS[0]}
    set -e
    if [ "$RUN_EXIT" -ne 0 ]; then
        BENCH_EXIT="$RUN_EXIT"
        break
    fi
done

# --- Signal summary (grep the key lines) ---
echo ""
echo "--- signal ---"
grep -iE "quant \+ sym|expert_preprocess:|arrival barrier wait|Same-iteration|fresh quant|consumer after quant|closed total|closure residual|Pipeline \(full|Pipeline \(no preproc|Kernel-only|Preprocess savings|vs non-fused:|Test 1:" _bench_output.txt || true

# --- Trajectory archive ---
if [ -n "$LABEL" ]; then TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"; else TRAJ_DIR="trajectory/${TIMESTAMP}"; fi
mkdir -p "$TRAJ_DIR"
for f in "${KERNEL_SRCS[@]}"; do [ -f "$f" ] && cp "$f" "$TRAJ_DIR/"; done
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
