#!/usr/bin/env bash
set -euo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_};; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then NVCC=$CUDA_HOME/bin/nvcc; CUOBJDUMP=$CUDA_HOME/bin/cuobjdump
else NVCC=nvcc; CUOBJDUMP=cuobjdump; fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo \
    "$D/probe/gather_hit.cu" -o "$OUT/bin/gather_hit" -lcuda -ldl \
    > "$OUT/sass/gather_hit.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/gather_hit" > "$OUT/sass/gather_hit.sass" 2>"$OUT/sass/cuobjdump.log" || true
if [ "$(wc -l < "$OUT/sass/gather_hit.sass")" -lt 20 ]; then
  cuobjdump -sass "$OUT/bin/gather_hit" \
      > "$OUT/sass/gather_hit.sass" 2>>"$OUT/sass/cuobjdump.log" || true
fi
echo "build.sh: gather_hit done ($SM_ARCH)"
