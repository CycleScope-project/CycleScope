#!/usr/bin/env bash
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash build.sh
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_} ;; *) exit 2 ;; esac
NVCC=${NVCC:-${CUDA_HOME:+$CUDA_HOME/bin/nvcc}}; NVCC=${NVCC:-nvcc}
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    -lineinfo --expt-relaxed-constexpr -I"$MECH_DIR/.." \
    "$MECH_DIR/probe/branch.cu" -o "$OUT/bin/branch" > "$OUT/sass/branch.ptxas.txt" 2>&1
echo "build.sh: control_branch done ($SM_ARCH)"
