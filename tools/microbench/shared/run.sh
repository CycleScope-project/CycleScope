#!/usr/bin/env bash
# smem probes are clock64 probes: plain execution, no ncu.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/shared/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/smem"
[ -x "$B" ] || { echo "SKIP smem (build failed)"; exit 1; }
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 3600 "$B" --out "$OUT/$run" --launches 8 --reps 33 > "$OUT/$run.smem.stdout" 2>&1 \
      || echo "PROBE_FAIL $run rc=$?"
done
