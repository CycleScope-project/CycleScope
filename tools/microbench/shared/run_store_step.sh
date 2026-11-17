#!/usr/bin/env bash
# shared protocol-v3 cells (probe/store_step.cu), two runs (a, b) in one job.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash run_store_step.sh
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
cell() { "$B" "$@" --samples 8 --out "$OUT/run-$run.jsonl" >> "$OUT/run-$run.stdout" 2>&1 || { echo "CELL_FAIL $run $*"; fails=$((fails + 1)); }; }
G16="--wide 1 --stride 64 --kstep 16 --wshift 0"
for run in a b; do
  # A: free-running stores (rate, concurrency)
  for m in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do cell $G16 --k 4 --tail 0 --mask $m; done
  for k in 1 2 8; do for m in 1 3 15; do cell $G16 --k $k --tail 0 --mask $m; done; done
  # B: stores then a serial fill chain (who is held)
  for n in 8 24 64; do for k in 1 2 4 8; do for m in 0 1 3 15; do
    cell $G16 --k $k --tail 3 --nfill $n --mask $m
  done; done; done
  # C: stores then barriers
  for m in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do cell $G16 --k 4 --tail 2 --mask $m; done
  for k in 1 2 8; do for m in 0 1 3 15; do cell $G16 --k $k --tail 2 --mask $m; done; done
  for m in 0 1 3 7 15; do cell $G16 --k 4 --tail 1 --mask $m; done
  for k in 1 2 8; do for m in 0 1 15; do cell $G16 --k $k --tail 1 --mask $m; done; done
  # D: fill / barrier order
  for t in 4 5; do for n in 24 64; do for m in 0 1 3 15; do
    cell $G16 --k 4 --tail $t --nfill $n --mask $m
  done; done; done
  # E: other geometries
  for g in "--wide 1 --k 4 --stride 64 --kstep 512 --wshift 0" \
           "--wide 1 --k 4 --stride 64 --kstep 16 --wshift 16" \
           "--wide 1 --k 4 --stride 16 --kstep 512 --wshift 0" \
           "--wide 0 --k 8 --stride 4 --kstep 128 --wshift 0" \
           "--wide 0 --k 4 --stride 64 --kstep 4 --wshift 0" \
           "--wide 0 --k 4 --stride 128 --kstep 4 --wshift 0"; do
    for m in 1 3 15; do cell $g --tail 0 --mask $m; done
    for m in 0 1 3 15; do cell $g --tail 2 --mask $m; done
  done
  # F: fence after the stores
  for k in 1 2 4 8; do for m in 0 1 3 15; do cell $G16 --k $k --tail 6 --mask $m; done; done
  # G: 4 wavefronts per store (the swizzled epilogue pattern): fence, barriers, hold
  G4="--wide 1 --stride 16 --kstep 512 --wshift 0"
  for k in 1 2 4 8; do for m in 0 1 3 15; do cell $G4 --k $k --tail 6 --mask $m; done; done
  for k in 1 2 8; do for m in 0 1 3 15; do cell $G4 --k $k --tail 2 --mask $m; done; done
  for k in 4 8; do for m in 0 1 15; do cell $G4 --k $k --tail 3 --nfill 24 --mask $m; done; done
done
echo "run_store_step.sh: done fails=$fails rows_a=$(wc -l < "$OUT/run-a.jsonl") rows_b=$(wc -l < "$OUT/run-b.jsonl")"
[ "$fails" = 0 ]
