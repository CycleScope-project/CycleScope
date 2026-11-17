#!/usr/bin/env bash
set -euo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_};; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then NVCC=$CUDA_HOME/bin/nvcc; CUOBJDUMP=$CUDA_HOME/bin/cuobjdump
else NVCC=nvcc; CUOBJDUMP=cuobjdump; fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -Xptxas -fmad=false -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo \
    "$D/probe/math_occupancy.cu" -o "$OUT/bin/math_occupancy" \
    > "$OUT/sass/math_occupancy.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/math_occupancy" > "$OUT/sass/math_occupancy.sass" \
    2>"$OUT/sass/cuobjdump.log" || true
if [ "$(wc -l < "$OUT/sass/math_occupancy.sass")" -lt 20 ]; then
  cuobjdump -sass \
      "$OUT/bin/math_occupancy" > "$OUT/sass/math_occupancy.sass" \
      2>>"$OUT/sass/cuobjdump.log" || true
fi
echo "build.sh: math_occupancy done ($SM_ARCH)"
