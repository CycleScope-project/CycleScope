#!/usr/bin/env bash
# memory_request: OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME] bash build.sh
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in
  sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_} ;;
  *) echo "build.sh: unsupported SM_ARCH=$SM_ARCH" >&2; exit 2 ;;
esac
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}; CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
  export PATH="$CUDA_HOME/bin:$PATH"
else
  NVCC=${NVCC:-nvcc}; CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O2 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -Xptxas -v \
    "$MECH_DIR/probe/memq.cu" -o "$OUT/bin/memq" \
    > "$OUT/sass/memq.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/memq" > "$OUT/sass/memq.sass"
echo "build.sh: memq done (SM_ARCH=$SM_ARCH)"
