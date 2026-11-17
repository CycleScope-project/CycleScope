#!/usr/bin/env bash
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/control_yield/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/yield"
[ -x "$B" ] || { echo "build failed"; exit 1; }
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 3600 "$B" --out "$OUT/$run" --launches 16 --reps 65 > "$OUT/$run.stdout" 2>&1 \
      || echo "PROBE_FAIL $run"
done
