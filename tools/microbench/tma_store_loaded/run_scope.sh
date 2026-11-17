#!/usr/bin/env bash
# protocol-v5 addendum "gpc_scope": the store service and the L2 read service per GPC and per card.
# TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash run_scope.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
M="$TREE/tools/microbench/tma_store_loaded"
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$M/build_dest.sh" > "$OUT/build.log" 2>&1
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$TREE/tools/microbench/ldgsts_loaded/build.sh" >> "$OUT/build.log" 2>&1
B="$OUT/build/bin/store_dest"; LB="$OUT/build/bin/ldgsts_loaded"
case "$SM_ARCH" in sm_90a) NMAX=132; DEV=h100;; sm_103a) NMAX=148; DEV=b300;; esac
python3 "$M/scope_sets.py" "$TREE/configs/hardware/$DEV/sm_topology.json" "$OUT/sets" > "$OUT/sets.txt"
SETS="g1k2 g1k4 g1k8 g1k12 g1kall ek1 ek2 ek4 ek8 ekall"
fails=0
st() { "$B" "$@" --stages 2 --box-bytes 16384 --ctas $NMAX --samples 8 --out "$OUT/store-$run.jsonl" >> "$OUT/store-$run.stdout" 2>&1 || { echo "CELL_FAIL $run $*"; fails=$((fails + 1)); }; }
rd() { "$LB" --ctas $NMAX --warps "$2" --rounds 4096 --samples 4 --table "$OUT/sets/$1.tbl" --out "$OUT/read-$run.jsonl" >> "$OUT/read-$run.stdout" 2>&1 || { echo "CELL_FAIL $run read $*"; fails=$((fails + 1)); }; }
for run in a b; do
  for s in $SETS; do
    st --red 0 --fresh 0 --active-file "$OUT/sets/$s.txt" --tag plain.ring.$s
    st --red 1 --fresh 0 --active-file "$OUT/sets/$s.txt" --tag red.ring.$s
    rd $s 8
  done
  for s in ek4 ek8 ekall g1kall; do st --red 0 --fresh 1 --active-file "$OUT/sets/$s.txt" --tag plain.fresh.$s; done
  for s in g1k8 g1kall ek8 ekall; do rd $s 4; done
done
echo "run_scope.sh: done fails=$fails store_rows=$(cat "$OUT"/store-*.jsonl | wc -l) read_rows=$(cat "$OUT"/read-*.jsonl | wc -l)"
[ "$fails" = 0 ]
