#!/usr/bin/env bash
# store_dram_port cell: runs mem --cells smport of the l1_bandwidth build
# (the same verbatim mem.cu copy; no separate probe directory per the
# protocol).  Output layout matches l1_bandwidth/run.sh's smport subdir.
#   TREE=<clone> OUT=<dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
# same binary as the l1_bandwidth cell: reuse its build when the crossings
# runner produced it next to this cell, else build here
if [ -x "$OUT/../l1_bandwidth/build/bin/mem" ]; then
  BUILD="$OUT/../l1_bandwidth/build"
else
  BUILD="$OUT/build"
  [ "${SKIP_BUILD:-0}" = 1 ] || OUT="$BUILD" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/crossings/l1_bandwidth/build.sh" > "$OUT/build.log" 2>&1
fi
M="$BUILD/bin/mem"
[ -x "$M" ] || { echo "SKIP mem (build failed)"; exit 1; }
( cd "$TREE/tools/microbench/crossings/l1_bandwidth" && \
  python3 sass_check.py --sass "$BUILD/sass" --out "$OUT/checks.json" ) \
  || echo "SASS_CHECK_FAIL rc=$?"
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 3600 "$M" --cells smport --out "$OUT/$run" --launches 8 --reps 33 > "$OUT/$run.stdout" 2>&1 \
      || echo "PROBE_FAIL $run rc=$?"
done
