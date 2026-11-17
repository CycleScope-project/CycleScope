#!/usr/bin/env bash
# trywait_wake cells: form x completion x delay D, two runs.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/trywait_wake/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/trywait_wake"
grep -c 'SYNCS.PHASECHK.TRANS64.TRYWAIT' "$OUT/build/sass/trywait_wake.sass" > "$OUT/sass-trywait-count.txt" || true
grep -c 'NANOSLEEP.SYNCS' "$OUT/build/sass/trywait_wake.sass" > "$OUT/sass-nanosleep-count.txt" || true
for run in a b; do
  for form in trywait trywait_hint testwait; do
    for comp in arrive tx; do
      for D in 0 50 100 200 400 800 1600 3200 6400; do
        "$B" --form $form --completion $comp --delay $D --reps 256 --launches 4 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run $form $comp D$D"
      done
    done
  done
done
echo "run.sh: trywait_wake done"
