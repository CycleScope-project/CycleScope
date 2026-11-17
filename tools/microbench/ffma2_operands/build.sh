#!/usr/bin/env bash
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:-sm_103a}
case "$SM_ARCH" in sm_103a) COMPUTE=compute_103 ;; *) echo "unsupported $SM_ARCH">&2;exit 2;; esac
NVCC=${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}
CUOBJDUMP=${CUOBJDUMP:-${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump}
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -I"$MECH_DIR" -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo \
    "$MECH_DIR/probe/ffma2.cu" -o "$OUT/bin/ffma2" > "$OUT/sass/ffma2.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/ffma2" > "$OUT/sass/ffma2.sass"
echo "build.sh: ffma2 done (SM_ARCH=$SM_ARCH)"
