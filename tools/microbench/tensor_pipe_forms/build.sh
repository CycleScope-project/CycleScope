#!/usr/bin/env bash
# Build the tensor_pipe_forms umma_factors probes: one binary per
# kind x cta_group (UMMA_FACTOR_KIND_FILTER x UMMA_FACTOR_CG_FILTER),
# ws and sparse included.  sm_103a only.
#   OUT=<dir> SM_ARCH=sm_103a [CUDA_HOME=<dir>] [JOBS=12] bash build.sh
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?OUT is required}
SM_ARCH=${SM_ARCH:?SM_ARCH is required}
[ "$SM_ARCH" = sm_103a ] || { echo "build.sh: tensor_pipe_forms is sm_103a only" >&2; exit 2; }
COMPUTE=compute_103a
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}
  CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
  export PATH="$CUDA_HOME/bin:$PATH"
else
  NVCC=${NVCC:-nvcc}; CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
fi
mkdir -p "$OUT/bin" "$OUT/sass" "$OUT/gen"
KINDS="${KINDS:-f16 tf32 f8f6f4 mxf8f6f4 mxf4 mxf4nvf4}"
declare -A KIND_ID=( [f16]=0 [tf32]=1 [f8f6f4]=2 [mxf8f6f4]=3 [mxf4]=4 [mxf4nvf4]=5 )
for k in $KINDS; do
  mkdir -p "$OUT/gen/$k"
  ( cd "$MECH_DIR/gen" && python3 gen_umma_full_forms.py --batch forms --kind "$k" \
      --out "$OUT/gen/$k.json" --header "$OUT/gen/$k/umma_factor_forms.hpp" ) \
      > "$OUT/gen/$k.gen.log" 2>&1 || echo "gen $k failed" >> "$OUT/gen/$k.gen.log"
done
JOBS=${JOBS:-12}
i=0
for k in $KINDS; do
  [ -f "$OUT/gen/$k/umma_factor_forms.hpp" ] || continue
  for cg in 1 2; do
    name="umma_factors_${k}_cg${cg}"
    (
      "$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" \
          -lineinfo --expt-relaxed-constexpr -I"$MECH_DIR" -I"$OUT/gen/$k" \
          -DUMMA_FACTOR_KIND_FILTER=${KIND_ID[$k]} -DUMMA_FACTOR_CG_FILTER=$cg \
          -DUMMA_FACTOR_INCLUDE_WS=1 -DUMMA_FACTOR_INCLUDE_SPARSE=1 \
          "$MECH_DIR/probe/umma_factors.cu" -o "$OUT/bin/$name" \
          > "$OUT/sass/$name.ptxas.txt" 2>&1
      _cub="$OUT/sass/.cub_$name"; rm -rf "$_cub"; mkdir -p "$_cub"
      (cd "$_cub" && "$CUOBJDUMP" -xelf all "$OUT/bin/$name" >/dev/null 2>&1)
      : > "$OUT/sass/$name.sass"
      for c in "$_cub"/*.cubin; do
        [ -e "$c" ] || continue
        "$CUOBJDUMP" -sass "$c" >> "$OUT/sass/$name.sass" 2>/dev/null || true
      done
      rm -rf "$_cub"
    ) &
    i=$((i+1)); [ $((i % JOBS)) -eq 0 ] && wait
  done
done
wait
for k in $KINDS; do
  for cg in 1 2; do
    name="umma_factors_${k}_cg${cg}"
    [ -f "$OUT/gen/$k/umma_factor_forms.hpp" ] || continue
    [ -x "$OUT/bin/$name" ] || { echo "build.sh: $name failed (see sass/$name.ptxas.txt)" >&2; continue; }
    _c=$(grep -c 'C7511' "$OUT/sass/$name.ptxas.txt" || true)
    [ "$_c" -gt 0 ] && echo "build.sh: WARN $name has $_c C7511-serialized kernels" >&2
    grep -q 'Function :' "$OUT/sass/$name.sass" || \
        echo "build.sh: WARN $name.sass has no functions" >&2
  done
done
echo "build.sh: tensor_pipe_forms done ($SM_ARCH kinds: $KINDS)"
