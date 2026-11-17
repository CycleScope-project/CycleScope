#!/usr/bin/env bash
# l1_bandwidth cell: mem (default cells: L1-hit sector bytes/cycle) + memq
# (L1-hit 1-sector request rate); mem --cells smport is the store_dram_port
# cell (store_dram_port/run.sh reuses this build).
#   TREE=<clone> OUT=<dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/crossings/l1_bandwidth/build.sh" > "$OUT/build.log" 2>&1
fi
M="$OUT/build/bin/mem"; Q="$OUT/build/bin/memq"
[ -x "$M" ] && [ -x "$Q" ] || { echo "SKIP mem/memq (build failed)"; exit 1; }
( cd "$TREE/tools/microbench/crossings/l1_bandwidth" && \
  python3 sass_check.py --sass "$OUT/build/sass" --out "$OUT/checks.json" ) \
  || echo "SASS_CHECK_FAIL rc=$?"
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  timeout 3600 "$M" --out "$OUT/$run" --launches 8 --reps 33 > "$OUT/$run.mem.stdout" 2>&1 \
      || echo "PROBE_FAIL $run mem rc=$?"
  timeout 3600 "$Q" --out "$OUT/$run" --launches 8 --reps 33 > "$OUT/$run.memq.stdout" 2>&1 \
      || echo "PROBE_FAIL $run memq rc=$?"
  mkdir -p "$OUT/memory_port/$run"
  timeout 3600 "$M" --cells smport --out "$OUT/memory_port/$run" --launches 8 --reps 33 > "$OUT/$run.smport.stdout" 2>&1 \
      || echo "PROBE_FAIL $run mem.smport rc=$?"
done
