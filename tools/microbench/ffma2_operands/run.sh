#!/usr/bin/env bash
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:-sm_103a}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" bash "$TREE/tools/microbench/ffma2_operands/build.sh" > "$OUT/build.log" 2>&1
fi
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 1800 "$OUT/build/bin/ffma2" --out "$OUT/$run" --launches 8 --reps 33 > "$OUT/$run.ffma2.stdout" 2>&1 \
      || echo "PROBE_FAIL ffma2 $run rc=$?"
done
