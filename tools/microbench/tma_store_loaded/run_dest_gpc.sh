#!/usr/bin/env bash
# tma_store_loaded protocol-v5 addendum "placement": does a reduce store's destination fetch depend on which
# GPCs the storing SMs are in?  TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash run_dest_gpc.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
M="$TREE/tools/microbench/tma_store_loaded"
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$M/build_dest.sh" > "$OUT/build.log" 2>&1
B="$OUT/build/bin/store_dest"
case "$SM_ARCH" in sm_90a) NMAX=132; DEV=h100;; sm_103a) NMAX=148; DEV=b300;; esac
python3 "$M/gpc_sets.py" "$TREE/configs/hardware/$DEV/sm_topology.json" "$OUT/sets" > "$OUT/sets.txt"
fails=0
cell() { "$B" "$@" --stages 2 --box-bytes 16384 --ctas $NMAX --samples 8 --out "$OUT/run-$run.jsonl" >> "$OUT/run-$run.stdout" 2>&1 || { echo "CELL_FAIL $run $*"; fails=$((fails + 1)); }; }
for run in a b; do
  for s in one spread halfgpc halfeach all; do cell --red 1 --fresh 1 --active-file "$OUT/sets/$s.txt" --tag red.fresh.$s; done
  for s in one spread; do cell --red 1 --fresh 0 --active-file "$OUT/sets/$s.txt" --tag red.ring.$s; done
  for s in halfgpc halfeach; do cell --red 0 --fresh 1 --active-file "$OUT/sets/$s.txt" --tag plain.fresh.$s; done
done
echo "run_dest_gpc.sh: done fails=$fails rows_a=$(wc -l < "$OUT/run-a.jsonl") rows_b=$(wc -l < "$OUT/run-b.jsonl")"
[ "$fails" = 0 ]
