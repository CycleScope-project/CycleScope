#!/usr/bin/env bash
# tma_store_loaded protocol-v5 ncu cells (b300 only): the four
# red {0,1} x fresh {0,1} cells at N = all SMs, stages 2, box 16384,
# counted with dram / lts sector counters.
#   BIN=<store_dest binary> OUT=<run dir> [NCU=<ncu>] bash run_dest_ncu.sh
set -uo pipefail
BIN=${BIN:?}; OUT=${OUT:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; NCU=${NCU:-$CUDA_HOME/bin/ncu}
M="dram__bytes_read.sum,dram__bytes_write.sum,lts__d_sectors_fill_device.sum,lts__t_sectors_op_red.sum,lts__t_sectors_op_write.sum,sm__cycles_active.max"
fails=0
for red in 0 1; do for fresh in 0 1; do
  d="$OUT/ncu/red$red.fresh$fresh"; mkdir -p "$d"
  "$NCU" --cache-control none --clock-control none --metrics "$M" --csv \
    "$BIN" --red $red --fresh $fresh --stages 2 --ctas 148 \
    --box-bytes 16384 --samples 1 --out "$d/probe.jsonl" \
    > "$d/ncu.csv" 2> "$d/ncu.err" || { echo "CELL_FAIL ncu red$red.fresh$fresh"; fails=$((fails + 1)); }
done; done
echo "run_dest_ncu.sh: done fails=$fails cells=$(ls -d "$OUT"/ncu/* | wc -l)"
