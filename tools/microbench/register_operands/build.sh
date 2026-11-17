#!/usr/bin/env bash
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_103a) COMPUTE=compute_103;; sm_90a) COMPUTE=compute_90;; *) exit 2;; esac
NVCC=${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}
CUOBJDUMP=${CUOBJDUMP:-${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump}
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -I"$MECH_DIR" -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo \
    "$MECH_DIR/probe/regops.cu" -o "$OUT/bin/regops" > "$OUT/sass/regops.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/regops" > "$OUT/sass/regops.sass"
echo "build.sh: regops done (SM_ARCH=$SM_ARCH)"
