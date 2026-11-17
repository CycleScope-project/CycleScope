#!/usr/bin/env bash
# load_after_stores cells (protocol.json), runs a and b.  TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT/build"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" --expt-relaxed-constexpr \
    "$TREE/tools/microbench/load_after_stores/probe/load_after_stores.cu" -o "$OUT/build/load_after_stores" > "$OUT/build.log" 2>&1
"$CUDA_HOME/bin/cuobjdump" -sass "$OUT/build/load_after_stores" > "$OUT/build/load_after_stores.sass" 2>/dev/null || true
# SASS gate: both kernels hold STG .U16 stores and LDG loads
grep -c 'STG.*U16' "$OUT/build/load_after_stores.sass" > "$OUT/build/sass_check.txt" || { echo 'SASS GATE FAILED'; exit 3; }
for run in a b; do "$OUT/build/load_after_stores" --out "$OUT/run-$run.jsonl"; done
echo "run.sh: load_after_stores done"
