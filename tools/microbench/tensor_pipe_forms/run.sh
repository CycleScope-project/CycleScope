#!/usr/bin/env bash
# tensor_pipe_forms probe run: build -> sass gate -> run-a/run-b per
# kind x cta_group binary.  A single kind's failure does not stop the rest.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_103a bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/tensor_pipe_forms/build.sh" > "$OUT/build.log" 2>&1
fi
( cd "$TREE/tools/microbench/tensor_pipe_forms" && \
  python3 sass_check.py --sass "$OUT/build/sass" --gen "$OUT/build/gen" \
      --out "$OUT/checks.json" ) \
  || echo "SASS_CHECK_FAIL rc=$?"
for B in "$OUT"/build/bin/umma_factors_*; do
  name=$(basename "$B")
  [ -x "$B" ] || { echo "SKIP $name (build failed)"; continue; }
  for run in run-a run-b; do
    d="$OUT/$run/$name"; mkdir -p "$d"
    timeout 5400 "$B" --out "$d" --launches 8 --reps 129 > "$d.stdout" 2>&1 \
        || echo "PROBE_FAIL $name $run rc=$?"
  done
done
echo "run.sh done"
