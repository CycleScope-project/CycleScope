#!/usr/bin/env bash
# dsmem probe build + SASS dump
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash build.sh
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in
  sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_} ;;
  *) echo "unsupported SM_ARCH=$SM_ARCH" >&2; exit 2 ;;
esac
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}
  CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
else
  NVCC=${NVCC:-nvcc}; CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -I"$MECH_DIR" \
    -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo \
    "$MECH_DIR/probe/dsmem.cu" -o "$OUT/bin/dsmem" -lcuda \
    > "$OUT/sass/dsmem.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/dsmem" > "$OUT/sass/dsmem.sass"
echo "build.sh: dsmem done (SM_ARCH=$SM_ARCH)"
