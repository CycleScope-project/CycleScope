#!/usr/bin/env bash
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_};; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then NVCC=$CUDA_HOME/bin/nvcc; CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
else NVCC=nvcc; CUOBJDUMP=${CUOBJDUMP:-cuobjdump}; fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr \
    "$MECH_DIR/probe/store_step.cu" -o "$OUT/bin/store_step" > "$OUT/sass/store_step.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/store_step" > "$OUT/sass/store_step.sass" 2>/dev/null || true
python3 "$MECH_DIR/sass_check_store_step.py" "$OUT/sass/store_step.sass" > "$OUT/sass/sass_check.txt" 2>&1 \
    && echo "build_store_step.sh: sass gate ok" || { echo "build_store_step.sh: SASS GATE FAILED (see $OUT/sass/sass_check.txt)"; exit 3; }
echo "build_store_step.sh: store_step done ($SM_ARCH)"
