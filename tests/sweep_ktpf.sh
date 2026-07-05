#!/bin/bash
# K-stripe configuration sweep: test different K_TILES_PER_FLAG values
# Each config runs 3 times, reports median kernel timing and profiling data

cd /DeepGemm_workspace/codebase/DeepGemm-block-copy

GPUS="2,3"
PORT_BASE=29540
NCB=4
CONFIGS="0 2 4 7 14"
REPEATS=3
TEST_CFG="prod"

echo "============================================================"
echo "K_TILES_PER_FLAG Sweep (NCB=$NCB, GPU=$GPUS, prod config)"
echo "  Each config repeated $REPEATS times"
echo "============================================================"
echo ""

for KTPF in $CONFIGS; do
    echo "--- K_TILES_PER_FLAG=$KTPF ---"
    for i in $(seq 1 $REPEATS); do
        PORT=$((PORT_BASE + KTPF * 10 + i))
        LOG="/tmp/sweep_ktpf${KTPF}_run${i}.log"
        KSTRIPE_PROFILE=1 SKIP_CORRECTNESS=1 \
            K_TILES_PER_FLAG=$KTPF NCB=$NCB \
            CUDA_VISIBLE_DEVICES=$GPUS \
            torchrun --nproc_per_node=2 --master_port=$PORT \
            tests/test_block_copy_gemm1_multi_gpu.py > "$LOG" 2>&1
        
        # Extract key metrics
        BC_BEST=$(grep "<-- best" "$LOG" | awk '{print $2}')
        KERNEL=$(grep "Kernel (normal P2P):" "$LOG" | awk '{print $3}')
        LOCAL=$(grep "Kernel (all-local):" "$LOG" | awk '{print $3}')
        EXPOSURE=$(grep "P2P exposure fraction:" "$LOG" | awk '{print $4}')
        WAIT_CTA=$(grep "Avg P2P wait/CTA:" "$LOG" | awk '{print $4}')
        COMPUTE_CTA=$(grep "Avg GEMM compute/CTA:" "$LOG" | awk '{print $4}')
        STATUS=$(grep -c "All PASSED" "$LOG")
        
        if [ "$STATUS" -gt 0 ]; then
            echo "  run$i: bc_best=${BC_BEST}  kernel=${KERNEL}ms  local=${LOCAL}ms  exposure=${EXPOSURE}  wait=${WAIT_CTA}  compute=${COMPUTE_CTA}"
        else
            echo "  run$i: FAILED"
            grep -E 'Error|error|SIGKILL' "$LOG" | tail -2
        fi
    done
    echo ""
done

echo "============================================================"
echo "Summary (all runs)"
echo "============================================================"
for KTPF in $CONFIGS; do
    echo ""
    echo "K_TILES_PER_FLAG=$KTPF:"
    for i in $(seq 1 $REPEATS); do
        LOG="/tmp/sweep_ktpf${KTPF}_run${i}.log"
        if grep -q "PASSED" "$LOG" 2>/dev/null; then
            grep -E "Pipeline \(full\)|Kernel-only|P2P exposure|NCB Sweep.*best" "$LOG" | head -4
        fi
    done
done
