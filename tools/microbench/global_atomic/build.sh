#!/usr/bin/env bash
# global_atomic: OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME] bash build.sh
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in
  sm_90a|sm_103a) ;;
  *) echo "build.sh: unsupported SM_ARCH=$SM_ARCH" >&2; exit 2 ;;
esac
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}; CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
  export PATH="$CUDA_HOME/bin:$PATH"
else
  NVCC=${NVCC:-nvcc}; CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O2 -arch="$SM_ARCH" -Xptxas -v \
    "$MECH_DIR/probe/global_atomic.cu" -o "$OUT/bin/global_atomic" \
    > "$OUT/sass/global_atomic.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/global_atomic" > "$OUT/sass/global_atomic.sass"
echo "build.sh: global_atomic done (SM_ARCH=$SM_ARCH)"
