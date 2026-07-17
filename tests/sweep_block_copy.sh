#!/bin/bash
# Block-copy masked 配置扫描：功能验证 + 纯性能。
# 轴：ktpf∈{0,2,4,7,14,28}, exact∈{0,1}；ncb∈{2,3,4} 由 test 内部扫。
# 用法： NC=8 PHASE=func|perf REPEATS=2 bash tests/sweep_block_copy.sh
cd /DeepGemm_workspace/codebase/DeepGemm-block-copy

NC=${NC:-8}
PHASE=${PHASE:-perf}
REPEATS=${REPEATS:-2}
KTPFS=${KTPFS:-"0 2 4 7 14 28"}
GPUS=$(seq -s, 0 $((NC-1)))
PORT_BASE=$((30000 + NC*111))
LOGDIR=/tmp/bcsweep_${NC}c
mkdir -p $LOGDIR
export FUSED_GEMM_GROUPING=masked

pidx=0
if [ "$PHASE" = "func" ]; then
    echo "=== FUNCTIONAL (${NC}c, FULL_CORRECTNESS) ==="
    for ktpf in $KTPFS; do
        pidx=$((pidx+1)); PORT=$((PORT_BASE+pidx))
        LOG="$LOGDIR/func_ktpf${ktpf}.log"
        FULL_CORRECTNESS=1 K_TILES_PER_FLAG=$ktpf NCB=4 \
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NC \
            --master_port=$PORT tests/test_block_copy_gemm1_multi_gpu.py > "$LOG" 2>&1
        if grep -q "test1_correctness: PASSED" "$LOG"; then
            diag=$(grep -E "max diff|element-wise" "$LOG" | head -1)
            echo "  [ktpf=$ktpf] PASSED  $diag"
        else
            echo "  [ktpf=$ktpf] FAILED"
            grep -iE 'error|assert|mismatch|FAILED' "$LOG" | grep -v PASSED | tail -3
        fi
    done
    echo "FUNC DONE"
    exit 0
fi

# PERF phase
echo "=== PERF (${NC}c, masked, min over $REPEATS) ==="
echo "cfg(ktpf,exact) | ncb2 ncb3 ncb4 (pipeline ms) | best_ncb best_pipe | kern local | tile_match"
for ktpf in $KTPFS; do
    for exact in 0 1; do
        # min accumulators
        declare -A minp
        best_pipe=1e9; best_line=""
        for i in $(seq 1 $REPEATS); do
            pidx=$((pidx+1)); PORT=$((PORT_BASE+pidx))
            LOG="$LOGDIR/perf_ktpf${ktpf}_ex${exact}_r${i}.log"
            SKIP_CORRECTNESS=1 NCB_SWEEP=2,3,4 FUSED_EXACT_GRID=$exact \
                K_TILES_PER_FLAG=$ktpf \
                CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NC \
                --master_port=$PORT tests/test_block_copy_gemm1_multi_gpu.py > "$LOG" 2>&1
            grep -q "All PASSED" "$LOG" || { echo "  [ktpf=$ktpf ex=$exact r$i] FAILED"; continue; }
        done
        # aggregate min across repeats per ncb from all repeat logs of this cfg
        py=$(python3 - "$LOGDIR" "$ktpf" "$exact" <<'PYEOF'
import sys,glob,re
d,ktpf,ex=sys.argv[1:4]
files=glob.glob(f"{d}/perf_ktpf{ktpf}_ex{ex}_r*.log")
ncb={}; kern=[]; local=[]; tm="?"
for f in files:
    t=open(f,errors='ignore').read()
    for m in re.finditer(r"ncb=(\d+): ([\d.]+) ms",t):
        n=int(m.group(1)); v=float(m.group(2)); ncb.setdefault(n,[]).append(v)
    mk=re.search(r"Kernel-only \(copy\+GEMM\):\s+([\d.]+)\s+([\d.]+)",t)
    if mk: kern.append(float(mk.group(1))); local.append(float(mk.group(2)))
    mt=re.search(r"TILE MATCH: (\w+)",t)
    if mt: tm=mt.group(1)
def mn(x): return min(x) if x else float('nan')
row={n:mn(v) for n,v in ncb.items()}
p2=row.get(2,float('nan')); p3=row.get(3,float('nan')); p4=row.get(4,float('nan'))
valid={n:v for n,v in row.items() if v==v}
if valid:
    bn=min(valid,key=valid.get); bp=valid[bn]
else:
    bn,bp=0,float('nan')
print(f"{p2:.3f} {p3:.3f} {p4:.3f} {bn} {bp:.3f} {mn(kern):.3f} {mn(local):.3f} {tm}")
PYEOF
)
        echo "  ktpf=$ktpf ex=$exact | $py"
    done
done
echo "PERF DONE"
