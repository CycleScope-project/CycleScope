#!/usr/bin/env bash
# Build the barrier_release probes for one SM arch and dump SASS.
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash build.sh
set -uo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_};; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}; CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
else NVCC=${NVCC:-nvcc}; CUOBJDUMP=${CUOBJDUMP:-cuobjdump}; fi
mkdir -p "$OUT/bin" "$OUT/sass"
rc=0
for p in sync_mb sync_bar sync_xw async; do
  if "$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" \
      -lineinfo --expt-relaxed-constexpr -I"$MECH_DIR" \
      "$MECH_DIR/probe/$p.cu" -o "$OUT/bin/$p" > "$OUT/sass/$p.ptxas.txt" 2>&1; then
    "$CUOBJDUMP" -sass "$OUT/bin/$p" > "$OUT/sass/$p.sass" 2>/dev/null || true
    echo "$p	0"
  else
    echo "$p	1"
    rc=1
  fi
done | tee "$OUT/build.tsv"
