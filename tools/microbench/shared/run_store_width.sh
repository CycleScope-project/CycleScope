#!/usr/bin/env bash
# shared protocol-v5 addendum "width": shared stores by a part of the warp's lanes, two runs in one job.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash run_store_width.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
M="$TREE/tools/microbench/shared"
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$M/build_store_step.sh" > "$OUT/build.log" 2>&1
B="$OUT/build/bin/store_step"
fails=0
cell() { "$B" "$@" --tail 0 --wshift 0 --samples 8 --out "$OUT/run-$run.jsonl" >> "$OUT/run-$run.stdout" 2>&1 || { echo "CELL_FAIL $run $*"; fails=$((fails + 1)); }; }
W128="--wide 1 --stride 16 --kstep 512"; W64="--wide 2 --stride 8 --kstep 256"
for run in a b; do
  for k in 4 8; do for m in 1 15; do
    cell $W128 --lanes 16 --k $k --mask $m                              # (1)
    cell $W64 --lanes 16 --k $k --mask $m                               # (2)
    cell $W128 --lanes 8 --k $k --mask $m                               # (3)
    cell --wide 1 --stride 24 --kstep 768 --lanes 32 --k $k --mask $m   # (4)
  done; done
  for k in 1 2 4 8; do cell $W128 --lanes 32 --k $k --mask 1; done      # (5)
  for m in 1 15; do cell $W128 --lanes 32 --k 8 --mask $m; cell $W64 --lanes 32 --k 8 --mask $m; done  # (6)
  for m in 3 7; do cell $W128 --lanes 16 --k 8 --mask $m; done          # (7)
done
echo "run_store_width.sh: done fails=$fails rows_a=$(wc -l < "$OUT/run-a.jsonl") rows_b=$(wc -l < "$OUT/run-b.jsonl")"
[ "$fails" = 0 ]
