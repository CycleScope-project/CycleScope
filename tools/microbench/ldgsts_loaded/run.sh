#!/usr/bin/env bash
# ldgsts_loaded cells: N in {1,16,64,128,SMs} x W in {1,2,4,8}, two runs.
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$TREE/tools/microbench/ldgsts_loaded/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/ldgsts_loaded"
grep -c 'LDGSTS.E.BYPASS.128' "$OUT/build/sass/ldgsts_loaded.sass" > "$OUT/sass-ldgsts-count.txt" || true
case "$SM_ARCH" in sm_90a) NMAX=132;; sm_103a) NMAX=148;; esac
for run in a b; do
  for N in 1 16 64 128 $NMAX; do
    for W in 1 2 4 8; do
      "$B" --ctas $N --warps $W --rounds 4096 --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run N$N W$W"
    done
  done
done
echo "run.sh: ldgsts_loaded done"
