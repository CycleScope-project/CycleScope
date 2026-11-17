#!/usr/bin/env bash
# Build the shared (smem) probe for one SM architecture and dump SASS.
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash build.sh
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?OUT is required}
SM_ARCH=${SM_ARCH:?SM_ARCH is required (sm_90a or sm_103a)}
case "$SM_ARCH" in
  sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_} ;;
  *) echo "build.sh: unsupported SM_ARCH=$SM_ARCH" >&2; exit 2 ;;
esac
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}
  CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
  export PATH="$CUDA_HOME/bin:$PATH"
else
  NVCC=${NVCC:-nvcc}
  CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
fi
mkdir -p "$OUT/bin" "$OUT/sass"
GUARD=""
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    -lineinfo --expt-relaxed-constexpr $GUARD \
    "$MECH_DIR/probe/smem.cu" -o "$OUT/bin/smem" > "$OUT/sass/smem.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/smem" > "$OUT/sass/smem.sass" 2>/dev/null || true
echo "build.sh: shared done (SM_ARCH=$SM_ARCH)"
