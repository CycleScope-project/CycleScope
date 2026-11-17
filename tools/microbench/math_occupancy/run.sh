#!/usr/bin/env bash
# TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [V=2] bash run.sh
set -uo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
V=${V:-1}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv" 2>/dev/null || true
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
    bash "$TREE/tools/microbench/math_occupancy/build.sh" > "$OUT/build.log" 2>&1 \
  || { echo BUILD_FAIL; exit 1; }
BIN="$OUT/build/bin/math_occupancy"
SASS="$OUT/build/sass/math_occupancy.sass"
python3 "$TREE/tools/microbench/math_occupancy/check.py" "$SASS" \
    > "$OUT/build/sass-counts.txt" 2>&1
rc=$?
cat "$OUT/build/sass-counts.txt"
[ $rc -ne 0 ] && { echo SASS_FAIL; exit $rc; }
if [ "$V" = 2 ]; then
  ARGS="--streams 0,4,5,6,7,8,9,10,11,12,13,14,15 --ws 4,8 --placements 1 --ctas 1,2 --launches 8"
else
  ARGS="--launches 8"
fi
for run in a b; do
  "$BIN" --out "$OUT/run-$run.jsonl" --reps 33 $ARGS \
      >>"$OUT/cells.log" 2>&1 || echo "CELL_FAIL run=$run"
done
echo CELLS_DONE
python3 "$TREE/tools/microbench/math_occupancy/table.py" "$OUT" \
    > "$OUT/table.txt" 2>&1 || true
echo "MATH_OCC_RC=$?"
