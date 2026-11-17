#!/usr/bin/env bash
# Build the mma_sync probe for one SM architecture, dump SASS, gate it.
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
"$NVCC" -O3 -std=c++17 \
    -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    --ptxas-options=-v \
    "$MECH_DIR/probe/mma.cu" -o "$OUT/bin/mma" \
    > "$OUT/sass/mma.ptxas.txt" 2>&1
# h100: the fatbin wrapper prints headers only -> extract cubins then -sass
SASS="$OUT/sass/mma.sass"
: > "$SASS"
if ! "$CUOBJDUMP" -sass "$OUT/bin/mma" > "$SASS" 2>/dev/null || \
   ! grep -q "Function :" "$SASS"; then
  mkdir -p "$OUT/sass/cubins"
  ( cd "$OUT/sass/cubins" && "$CUOBJDUMP" -xelf all "$OUT/bin/mma" >/dev/null 2>&1 )
  : > "$SASS"
  for c in "$OUT"/sass/cubins/*.cubin; do
    [ -e "$c" ] || continue
    "$CUOBJDUMP" -sass "$c" >> "$SASS" 2>/dev/null || true
  done
fi
python3 "$MECH_DIR/sass_check.py" --sass "$SASS" --out "$OUT/sass/checks.json"
