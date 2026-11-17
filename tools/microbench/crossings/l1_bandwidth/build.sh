#!/usr/bin/env bash
# Build the verbatim blackwell-prototype mem.cu / memq.cu probes for one SM arch.
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash build.sh
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_};; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=$CUDA_HOME/bin/nvcc; CUOBJDUMP=$CUDA_HOME/bin/cuobjdump
  export PATH="$CUDA_HOME/bin:$PATH"
else NVCC=nvcc; CUOBJDUMP=cuobjdump; fi
mkdir -p "$OUT/bin" "$OUT/sass"
for p in mem memq; do
  "$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr \
      "$MECH_DIR/probe/$p.cu" -o "$OUT/bin/$p" > "$OUT/sass/$p.ptxas.txt" 2>&1
  "$CUOBJDUMP" -sass "$OUT/bin/$p" > "$OUT/sass/$p.sass" 2>/dev/null || true
  echo "build.sh: $p done ($SM_ARCH)"
done
