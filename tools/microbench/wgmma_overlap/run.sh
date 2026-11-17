#!/usr/bin/env bash
# wgmma_overlap probes are clock64 probes: plain execution, no ncu.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/wgmma_overlap/build.sh" > "$OUT/build.log" 2>&1
fi
BIN=overlap
if [ "${V4:-0}" = 1 ]; then
  BIN=overlap_v4
elif [ "${V3:-0}" = 1 ]; then
  BIN=overlap_v3
  cp "$OUT/build/sass/checks_v3.json" "$OUT/build/sass/checks.json"
elif [ "${V2:-0}" = 1 ]; then
  BIN=overlap_v2
  # model_v2.py reads build/sass/checks.json: put the v2 gate there
  cp "$OUT/build/sass/checks_v2.json" "$OUT/build/sass/checks.json"
fi
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 3600 "$OUT/build/bin/$BIN" --out "$OUT/$run" \
      --launches 8 --reps 33 > "$OUT/$run.$BIN.stdout" 2>&1 \
      || echo "PROBE_FAIL $run rc=$?"
done
