#!/usr/bin/env bash
# Build the wgmma_handoff probe for sm_90a, dump SASS, gate it.
#   OUT=<dir> SM_ARCH=sm_90a [CUDA_HOME=<dir>] bash build.sh
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?OUT is required}
SM_ARCH=${SM_ARCH:?SM_ARCH is required (sm_90a)}
case "$SM_ARCH" in
  sm_90a) COMPUTE=compute_90a ;;
  *) echo "build.sh: unsupported SM_ARCH=$SM_ARCH (wgmma is sm_90a only)" >&2; exit 2 ;;
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
python3 "$MECH_DIR/../tensor_pipe/probe/gen_wgmma.py" \
    "$MECH_DIR/probe/wgmma_impl.cuh" "$OUT/wgmma_cells.json"
"$NVCC" -O3 -std=c++17 \
    -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    --ptxas-options=-v --expt-relaxed-constexpr -I"$MECH_DIR" \
    "$MECH_DIR/probe/handoff.cu" -o "$OUT/bin/handoff" \
    > "$OUT/sass/handoff.ptxas.txt" 2>&1
SASS="$OUT/sass/handoff.sass"
_cub="$OUT/sass/.cub_handoff"; rm -rf "$_cub"; mkdir -p "$_cub"
( cd "$_cub" && "$CUOBJDUMP" -xelf all "$OUT/bin/handoff" >/dev/null 2>&1 )
: > "$SASS"
for c in "$_cub"/*.cubin; do
  [ -e "$c" ] || continue
  "$CUOBJDUMP" -sass "$c" >> "$SASS" 2>/dev/null || true
done
rm -rf "$_cub"
python3 "$MECH_DIR/sass_check.py" --sass "$SASS" --out "$OUT/sass/checks.json"
