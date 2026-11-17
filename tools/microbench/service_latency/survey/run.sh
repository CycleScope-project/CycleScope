#!/usr/bin/env bash
# idiom survey: compile idioms.cu, dump SASS, decompose clock windows.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
SRC="$TREE/tools/microbench/service_latency/probe/idioms.cu"
BIN="$OUT/idioms"
if ! "$CUDA_HOME/bin/nvcc" -arch="$SM_ARCH" -O3 -std=c++17 -o "$BIN" "$SRC" \
    > "$OUT/build.log" 2>&1; then
  # keep generated PTX so error line numbers can be mapped back to idioms
  "$CUDA_HOME/bin/nvcc" -arch="$SM_ARCH" -O3 -std=c++17 -ptx \
      -o "$OUT/idioms.ptx" "$SRC" > "$OUT/ptx.log" 2>&1 || true
  tail -40 "$OUT/build.log"; exit 1
fi
"$CUDA_HOME/bin/cuobjdump" -sass "$BIN" > "$OUT/idioms.sass"
python3 "$TREE/tools/microbench/service_latency/survey_idioms.py" \
    "$OUT/idioms.sass" "$OUT/survey.json" > "$OUT/survey.stdout" 2>&1 || \
    { cat "$OUT/survey.stdout"; exit 1; }
# execute once so obvious runtime faults surface in the log
"$BIN" --out "$OUT" > "$OUT/run.stdout" 2>&1 || echo "PROBE_RUN rc=$?"
echo "survey done: $OUT"
