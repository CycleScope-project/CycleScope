#!/usr/bin/env bash
# Build the wgmma_overlap probe for sm_90a, dump SASS, gate it.
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
    "$MECH_DIR/probe/overlap.cu" -o "$OUT/bin/overlap" \
    > "$OUT/sass/overlap.ptxas.txt" 2>&1
"$NVCC" -O3 -std=c++17 \
    -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    --ptxas-options=-v --expt-relaxed-constexpr -I"$MECH_DIR" \
    "$MECH_DIR/probe/overlap_v2.cu" -o "$OUT/bin/overlap_v2" \
    > "$OUT/sass/overlap_v2.ptxas.txt" 2>&1
"$NVCC" -O3 -std=c++17 \
    -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    --ptxas-options=-v --expt-relaxed-constexpr -I"$MECH_DIR" \
    "$MECH_DIR/probe/overlap_v3.cu" -o "$OUT/bin/overlap_v3" \
    > "$OUT/sass/overlap_v3.ptxas.txt" 2>&1
"$NVCC" -O3 -std=c++17 \
    -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    --ptxas-options=-v --expt-relaxed-constexpr -I"$MECH_DIR" \
    "$MECH_DIR/probe/overlap_v4.cu" -o "$OUT/bin/overlap_v4" \
    > "$OUT/sass/overlap_v4.ptxas.txt" 2>&1
for bin in overlap overlap_v2 overlap_v3 overlap_v4; do
SASS="$OUT/sass/$bin.sass"
_cub="$OUT/sass/.cub_$bin"; rm -rf "$_cub"; mkdir -p "$_cub"
( cd "$_cub" && "$CUOBJDUMP" -xelf all "$OUT/bin/$bin" >/dev/null 2>&1 )
: > "$SASS"
for c in "$_cub"/*.cubin; do
  [ -e "$c" ] || continue
  "$CUOBJDUMP" -sass "$c" >> "$SASS" 2>/dev/null || true
done
rm -rf "$_cub"
done
# gate failures are recorded in checks.json and make their cells unusable in
# model.py; they do not stop the run
python3 "$MECH_DIR/sass_check.py" \
    --sass "$OUT/sass/overlap.sass" --out "$OUT/sass/checks.json" \
    || echo "SASS_GATE_RC=$?"
python3 "$MECH_DIR/sass_check_v2.py" \
    --sass "$OUT/sass/overlap_v2.sass" --out "$OUT/sass/checks_v2.json" \
    || echo "SASS_GATE_V2_RC=$?"
python3 "$MECH_DIR/sass_check_v3.py" \
    --sass "$OUT/sass/overlap_v3.sass" --out "$OUT/sass/checks_v3.json" \
    || echo "SASS_GATE_V3_RC=$?"
python3 "$MECH_DIR/sass_check_grpq.py" \
    "$OUT/sass/overlap_v4.sass" "$OUT/sass" \
    || echo "SASS_GATE_GRPQ_RC=$?"
