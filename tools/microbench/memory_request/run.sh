#!/usr/bin/env bash
# memory_request protocol r1: identity + sass gate + two runs
# (run-a/run-b) at --launches 8 --reps 33.
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm --format=csv > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/memory_request/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/memq"
[ -x "$B" ] || { echo "SKIP memq (build failed)"; exit 1; }
( cd "$TREE/tools/microbench/memory_request" && \
  python3 ${SASS_CHECK:-sass_check.py} --sass "$OUT/build/sass/memq.sass" \
      --out "$OUT/sass/checks.json" ) || echo "SASS_CHECK_FAIL rc=$?"
for run in run-a run-b; do
  mkdir -p "$OUT/$run"
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader \
      > "$OUT/$run/clocks_before.txt"
  if [ "${V2:-0}" = 1 ]; then
    # v2: dram uses 7 launches x 17 reps; the other cells 8 x 33.
    # MbWriter truncates its jsonl, so each --only batch writes to a
    # part dir; merged afterwards.
    for spec in "ld.l1. 8 33" "ld.l2. 8 33" "ld.dram. 7 17" "st. 8 33" "ack. 8 33"; do
      set -- $spec
      mkdir -p "$OUT/$run/parts-$1"
      timeout 3600 "$B" --out "$OUT/$run/parts-$1" --launches "$2" \
          --reps "$3" --only "$1" >> "$OUT/$run.stdout" 2>&1 \
          || echo "PROBE_FAIL $run $1 rc=$?"
    done
    cat "$OUT/$run"/parts-*/memq.jsonl > "$OUT/$run/memq.jsonl"
  else
    timeout 3600 "$B" --out "$OUT/$run" --launches 8 --reps 33 \
        > "$OUT/$run.stdout" 2>&1 || echo "PROBE_FAIL $run rc=$?"
  fi
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader \
      > "$OUT/$run/clocks_after.txt"
done
echo "run.sh: memq done" > "$OUT/done.txt"
