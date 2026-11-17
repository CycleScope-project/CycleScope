#!/usr/bin/env bash
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/crossings/l1_residency/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/l1cap"
[ -x "$B" ] || { echo "SKIP l1cap (build failed)"; exit 1; }
( cd "$TREE/tools/microbench/crossings/l1_residency" && \
  python3 sass_check.py --sass "$OUT/build/sass" --out "$OUT/checks.json" ) \
  || echo "SASS_CHECK_FAIL rc=$?"
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 3600 "$B" --out "$OUT/$run" --launches 8 --reps 33 > "$OUT/$run.stdout" 2>&1 \
      || echo "PROBE_FAIL $run rc=$?"
done
