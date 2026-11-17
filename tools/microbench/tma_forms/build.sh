#!/usr/bin/env bash
set -euo pipefail
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC="$CUDA_HOME/bin/nvcc"; CUOBJDUMP="$CUDA_HOME/bin/cuobjdump"
MECH_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -gencode "arch=compute_${SM_ARCH#sm_},code=$SM_ARCH" \
    -lineinfo --expt-relaxed-constexpr -I"$MECH_DIR/.." \
    "$MECH_DIR/probe/tma_forms.cu" -o "$OUT/bin/tma_forms" \
    > "$OUT/sass/tma_forms.ptxas.txt" 2>&1
# also build tma_cold's tma.cu (run.sh drives its cell sets)
"$NVCC" -O3 -std=c++17 -gencode "arch=compute_${SM_ARCH#sm_},code=$SM_ARCH" \
    -lineinfo --expt-relaxed-constexpr -I"$MECH_DIR/.." \
    "$MECH_DIR/../tma_cold/probe/tma.cu" -o "$OUT/bin/tma" \
    > "$OUT/sass/tma.ptxas.txt" 2>&1
for p in tma tma_forms; do
  _cub="$OUT/sass/.cub_$p"; rm -rf "$_cub"; mkdir -p "$_cub"
  (cd "$_cub" && "$CUOBJDUMP" -xelf all "$OUT/bin/$p" >/dev/null 2>&1)
  : > "$OUT/sass/$p.sass"
  for c in "$_cub"/*.cubin; do
    [ -e "$c" ] || continue
    "$CUOBJDUMP" -sass "$c" >> "$OUT/sass/$p.sass" 2>/dev/null || true
  done
  rm -rf "$_cub"
done
echo "build.sh: tma_forms done (SM_ARCH=$SM_ARCH)"
