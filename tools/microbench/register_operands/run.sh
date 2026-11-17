#!/usr/bin/env bash
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" bash "$TREE/tools/microbench/register_operands/build.sh" > "$OUT/build.log" 2>&1
fi
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 2400 "$OUT/build/bin/regops" --out "$OUT/$run" --launches 8 --reps 33 > "$OUT/$run.regops.stdout" 2>&1 \
      || echo "PROBE_FAIL regops $run rc=$?"
done
