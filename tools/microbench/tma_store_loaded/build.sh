#!/usr/bin/env bash
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_};; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then NVCC=$CUDA_HOME/bin/nvcc; CUOBJDUMP=$CUDA_HOME/bin/cuobjdump
else NVCC=nvcc; CUOBJDUMP=cuobjdump; fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr \
    "$MECH_DIR/probe/tma_store_loaded.cu" -o "$OUT/bin/tma_store_loaded" -lcuda -ldl > "$OUT/sass/tma_store_loaded.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/tma_store_loaded" > "$OUT/sass/tma_store_loaded.sass" 2>/dev/null || true
python3 "$MECH_DIR/sass_check.py" "$OUT/sass/tma_store_loaded.sass" > "$OUT/sass/sass_check.txt" 2>&1 \
    && echo "build.sh: sass gate ok" || { echo "build.sh: SASS GATE FAILED (see $OUT/sass/sass_check.txt)"; exit 3; }
echo "build.sh: tma_store_loaded done ($SM_ARCH)"
