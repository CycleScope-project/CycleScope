#!/usr/bin/env bash
# dsmem: clock64 probe, cluster of 2 CTAs — plain execution.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/dsmem/build.sh" > "$OUT/build.log" 2>&1
fi
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 1800 "$OUT/build/bin/dsmem" --out "$OUT/$run" \
      --launches 8 --reps 33 > "$OUT/$run.dsmem.stdout" 2>&1 \
      || echo "PROBE_FAIL dsmem $run rc=$?"
done
