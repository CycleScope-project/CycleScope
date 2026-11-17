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
    "$MECH_DIR/probe/contention.cu" -o "$OUT/bin/contention" \
    > "$OUT/sass/contention.ptxas.txt" 2>&1
cuobjdump -sass "$OUT/bin/contention" > "$OUT/sass/contention.sass"
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    -lineinfo --expt-relaxed-constexpr -I"$MECH_DIR/.." \
    "$MECH_DIR/probe/contention_v2.cu" -o "$OUT/bin/contention_v2" \
    > "$OUT/sass/contention_v2.ptxas.txt" 2>&1
cuobjdump -sass "$OUT/bin/contention_v2" > "$OUT/sass/contention_v2.sass"
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    -lineinfo --expt-relaxed-constexpr -I"$MECH_DIR/.." \
    "$MECH_DIR/probe/contention_v3.cu" -o "$OUT/bin/contention_v3" \
    > "$OUT/sass/contention_v3.ptxas.txt" 2>&1
cuobjdump -sass "$OUT/bin/contention_v3" > "$OUT/sass/contention_v3.sass"
echo "build.sh: dispatch_contention done ($SM_ARCH)"
