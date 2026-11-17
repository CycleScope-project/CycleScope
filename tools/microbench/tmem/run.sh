#!/usr/bin/env bash
# tmem probes are clock64 probes: plain execution, no ncu.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_103a bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/tmem/build.sh" > "$OUT/build.log" 2>&1
fi
PROBE=${PROBE:-ldtm_warps}
if [ "$PROBE" = x64 ]; then LA=16; RP=65; else LA=8; RP=33; fi
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 1800 "$OUT/build/bin/$PROBE" --out "$OUT/$run" \
      --launches "$LA" --reps "$RP" > "$OUT/$run.$PROBE.stdout" 2>&1 \
      || echo "PROBE_FAIL $run rc=$?"
done
