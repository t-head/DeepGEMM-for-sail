#!/bin/bash
# 对比 fused-nopad vs fused-masked vs non-fused，跨 2/4/8 卡，多轮取 min。
# 用法：在容器内 bash tests/sweep_masked_vs_nopad.sh
cd /DeepGemm_workspace/codebase/DeepGemm-block-copy

REPEATS=${REPEATS:-3}
NCB_SWEEP=${NCB_SWEEP:-2,3,4}
CARDS=${CARDS:-"2 4 8"}
PORT_BASE=29600
LOGDIR=/tmp/mnk_sweep
mkdir -p $LOGDIR

extract() {  # $1=logfile  -> "pipe_bc pipe_nf kern_bc kern_local kern_nf best_ncb tilematch"
    local f=$1
    local pipe_bc=$(grep "Pipeline (full):" "$f" | awk '{print $3}')
    local pipe_nf=$(grep "Pipeline (full):" "$f" | awk '{print $NF}')
    local kern_bc=$(grep "Kernel-only (copy+GEMM):" "$f" | awk '{print $3}')
    local kern_local=$(grep "Kernel-only (copy+GEMM):" "$f" | awk '{print $4}')
    local kern_nf=$(grep "Kernel-only (copy+GEMM):" "$f" | awk '{print $NF}')
    local best_ncb=$(grep "best ncb" "$f" | head -1 | sed 's/.*best ncb=//' | awk '{print $1}')
    local tm=$(grep "TILE MATCH" "$f" | head -1 | sed 's/.*TILE MATCH: //' | awk '{print $1}')
    echo "$pipe_bc $pipe_nf $kern_bc $kern_local $kern_nf $best_ncb $tm"
}

# min over repeats of a whitespace field index
minfield() { awk -v c=$1 'BEGIN{m=1e9} {if($c+0>0 && $c+0<m) m=$c} END{printf "%.3f", m}'; }

for nc in $CARDS; do
    GPUS=$(seq -s, 0 $((nc-1)))
    for grp in nopad masked; do
        goff=0; [ "$grp" = "masked" ] && goff=50
        rows=""
        for i in $(seq 1 $REPEATS); do
            PORT=$((PORT_BASE + nc*100 + i + goff))
            LOG="$LOGDIR/c${nc}_${grp}_r${i}.log"
            SKIP_CORRECTNESS=1 NCB_SWEEP=$NCB_SWEEP FUSED_EXACT_GRID=1 \
                FUSED_GEMM_GROUPING=$grp CUDA_VISIBLE_DEVICES=$GPUS \
                torchrun --nproc_per_node=$nc --master_port=$PORT \
                tests/test_block_copy_gemm1_multi_gpu.py > "$LOG" 2>&1
            if grep -q "All PASSED" "$LOG"; then
                r=$(extract "$LOG")
                rows="$rows$r"$'\n'
                echo "  [${nc}c ${grp} r${i}] $r"
            else
                echo "  [${nc}c ${grp} r${i}] FAILED"
                grep -iE 'error|assert|traceback' "$LOG" | tail -3
            fi
        done
        pbc=$(printf "%s" "$rows" | minfield 1)
        pnf=$(printf "%s" "$rows" | minfield 2)
        kbc=$(printf "%s" "$rows" | minfield 3)
        kloc=$(printf "%s" "$rows" | minfield 4)
        knf=$(printf "%s" "$rows" | minfield 5)
        echo ">> ${nc}c ${grp} MIN: pipe_bc=$pbc pipe_nf=$pnf kern_bc=$kbc kern_local=$kloc kern_nf=$knf"
        echo ""
    done
done
echo "DONE"
