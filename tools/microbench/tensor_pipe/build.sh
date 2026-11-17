#!/usr/bin/env bash
# Build the tensor_pipe probes for one SM architecture and dump SASS.
#   sm_103a: probe/umma.cu (verbatim blackwell-prototype, umma_impl.cuh + mb.cuh)
#   sm_90a:  probe/wgmma.cu (new; see protocol.json)
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] [JOBS=8] bash build.sh
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
if [ "$SM_ARCH" = sm_103a ]; then PROBES="${UMMA_PROBES:-umma umma_cadence}";
elif [ "${WGMMA_SP:-0}" = 1 ]; then PROBES="${WGMMA_PROBES:-wgmma_spf16 wgmma_spbf16 wgmma_spe4m3e4m3}";
else PROBES="${WGMMA_PROBES:-wgmma_f16 wgmma_f16a wgmma_bf16 wgmma_tf32 wgmma_e4m3e4m3 wgmma_e4m3e5m2 wgmma_e5m2e5m2 wgmma_s8s8}"; fi
if [ "$SM_ARCH" = sm_90a ]; then
  _gen_args=""
  [ -n "${WGMMA_SRC:-}" ] && _gen_args="--only-src=$WGMMA_SRC"
  python3 "$MECH_DIR/probe/gen_wgmma.py" "$MECH_DIR/probe/wgmma_impl.cuh" \
      "$OUT/wgmma_cells.json" $_gen_args
fi
# probe TUs compile in parallel (wgmma is split per kind)
for p in $PROBES; do
  "$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" \
      -lineinfo --expt-relaxed-constexpr -I"$MECH_DIR" \
      "$MECH_DIR/probe/$p.cu" -o "$OUT/bin/$p" > "$OUT/sass/$p.ptxas.txt" 2>&1 &
done
wait
for p in $PROBES; do
  [ -x "$OUT/bin/$p" ] || { echo "build.sh: $p failed (see sass/$p.ptxas.txt)" >&2; }
  _c=$(grep -c 'C7511' "$OUT/sass/$p.ptxas.txt" || true)
  [ "$_c" -gt 0 ] && echo "build.sh: WARN $p has $_c C7511-serialized kernels" >&2
  # cuobjdump -sass on the fatbin wrapper yields headers only;
  # extract cubins first, then disassemble each
  _cub="$OUT/sass/.cub_$p"; rm -rf "$_cub"; mkdir -p "$_cub"
  (cd "$_cub" && "$CUOBJDUMP" -xelf all "$OUT/bin/$p" >/dev/null 2>&1)
  : > "$OUT/sass/$p.sass"
  for c in "$_cub"/*.cubin; do
    [ -e "$c" ] || continue
    "$CUOBJDUMP" -sass "$c" >> "$OUT/sass/$p.sass" 2>/dev/null || true
  done
  rm -rf "$_cub"
  grep -q 'Function :' "$OUT/sass/$p.sass" || \
      echo "build.sh: WARN $p.sass has no functions" >&2
done
echo "build.sh: tensor_pipe done (SM_ARCH=$SM_ARCH probes=$PROBES)"
