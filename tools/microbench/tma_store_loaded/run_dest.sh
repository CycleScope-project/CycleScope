#!/usr/bin/env bash
# tma_store_loaded protocol-v5 cells (probe/store_dest.cu), two runs (a, b) in one job.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash run_dest.sh
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
case "$SM_ARCH" in sm_90a) NMAX=132;; sm_103a) NMAX=148;; esac
fails=0
cell() { "$B" "$@" --samples 8 --out "$OUT/run-$run.jsonl" >> "$OUT/run-$run.stdout" 2>&1 || { echo "CELL_FAIL $run $*"; fails=$((fails + 1)); }; }
for run in a b; do
  for red in 0 1; do for fresh in 0 1; do
    for N in 1 $NMAX; do
      for st in 2 4; do cell --red $red --fresh $fresh --stages $st --ctas $N --box-bytes 16384; done
      cell --red $red --fresh $fresh --stages 2 --ctas $N --box-bytes 8192
    done
  done; done
  for red in 0 1; do cell --red $red --fresh 1 --stages 2 --ctas 1 --box-bytes 16384 --evict-mib 0; done
done
echo "run_dest.sh: done fails=$fails rows_a=$(wc -l < "$OUT/run-a.jsonl") rows_b=$(wc -l < "$OUT/run-b.jsonl")"
[ "$fails" = 0 ]
