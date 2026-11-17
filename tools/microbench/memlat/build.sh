#!/usr/bin/env bash
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_};; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then NVCC=$CUDA_HOME/bin/nvcc; CUOBJDUMP=$CUDA_HOME/bin/cuobjdump
else NVCC=nvcc; CUOBJDUMP=cuobjdump; fi
mkdir -p "$OUT/bin" "$OUT/sass"
for p in stage27_memlat_memory stage27_memlat_ops; do
  "$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr \
      -I"$MECH_DIR/probe" "$MECH_DIR/probe/$p.cu" -o "$OUT/bin/$p" > "$OUT/sass/$p.ptxas.txt" 2>&1
  "$CUOBJDUMP" -sass "$OUT/bin/$p" > "$OUT/sass/$p.sass" 2>/dev/null || true
done
echo "build.sh: memlat done ($SM_ARCH)"
