#!/usr/bin/env bash
# shared protocol-v5 cells: free-running shared stores (probe/store_step.cu, tail 0) over wavefront counts
# 1..32 at two wavefront widths, two runs (a, b) in one job.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash run_store_wave.sh
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
for run in a b; do
  # plain 32-bit stores: lane stride 4 / 8 / 16 / 32 / 64 / 128 B = 1 / 2 / 4 / 8 / 16 / 32 wavefronts of 32 / 16 / 8 / 4 / 2 / 1 words
  # 128-bit stores: lane stride 16 / 32 / 64 / 128 B = 4 / 8 / 16 / 32 wavefronts of 32 / 16 / 8 / 4 words
  for g in "--wide 0 --stride 4 --kstep 128" "--wide 0 --stride 8 --kstep 4" "--wide 0 --stride 16 --kstep 4" \
           "--wide 0 --stride 32 --kstep 4" "--wide 0 --stride 64 --kstep 4" "--wide 0 --stride 128 --kstep 4" \
           "--wide 1 --stride 16 --kstep 512" "--wide 1 --stride 32 --kstep 16" "--wide 1 --stride 64 --kstep 16" \
           "--wide 1 --stride 128 --kstep 16"; do
    for k in 4 8; do for m in 1 3 7 15; do cell $g --k $k --mask $m; done; done
  done
done
echo "run_store_wave.sh: done fails=$fails rows_a=$(wc -l < "$OUT/run-a.jsonl") rows_b=$(wc -l < "$OUT/run-b.jsonl")"
[ "$fails" = 0 ]
