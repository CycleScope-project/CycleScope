#!/usr/bin/env bash
# memlat_loaded cells (protocol.json): two levels x {N at W=16} + {N=SMs-1 at W in 2,4,8},
# two runs (a, b) in the same job.  TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/memlat_loaded/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/memlat_loaded"

case "$SM_ARCH" in sm_90a) NMAX=131;; sm_103a) NMAX=147;; esac
for run in a b; do
  for level in dram l2; do
    for N in 0 16 32 64 96 128 $NMAX; do
      "$B" --level $level --loaders $N --warps 16 --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run $level N$N W16"
    done
    for W in 2 4 8; do
      "$B" --level $level --loaders $NMAX --warps $W --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run $level N$NMAX W$W"
    done
  done
done
echo "run.sh: memlat_loaded done"
