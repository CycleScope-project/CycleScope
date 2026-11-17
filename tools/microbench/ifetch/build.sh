#!/usr/bin/env bash
# Build the ifetch probe for one SM architecture and run the SASS gate.
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] [IFETCH_N=<N>] bash build.sh
# nvcc/cuobjdump come from $CUDA_HOME/bin when set, else PATH.
# IFETCH_N=<N>: per-footprint build (one k_line + its k_mix widths, no jump
# kernels) -- the H100 image's ptxas cannot build the full binary in time.
# The generated bodies/jump headers are regenerated here (gitignored, like
# the predecessor's tools/microbench/build.sh did).
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?OUT is required}
SM_ARCH=${SM_ARCH:?SM_ARCH is required (sm_90a or sm_103a)}
PROBE_DIR=$MECH_DIR/probe
case "$SM_ARCH" in
  sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_} ;;
  *) echo "build.sh: unsupported SM_ARCH=$SM_ARCH" >&2; exit 2 ;;
esac
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}
  CUOBJDUMP=${CUOBJDUMP:-$CUDA_HOME/bin/cuobjdump}
else
  NVCC=${NVCC:-nvcc}
  CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
fi
mkdir -p "$OUT/bin" "$OUT/sass"
GEN_DIR=$OUT/gen; mkdir -p "$GEN_DIR"
if [ -n "${IFETCH_N:-}" ]; then
  python3 "$PROBE_DIR/gen_ifetch_bodies.py" --out "$GEN_DIR/ifetch_bodies.hpp" --only-n "$IFETCH_N"
  python3 "$PROBE_DIR/gen_ifetch_jump.py" --out "$GEN_DIR/ifetch_jump.hpp" --none
  DEFS="-DIFETCH_N=$IFETCH_N"
else
  python3 "$PROBE_DIR/gen_ifetch_bodies.py" --out "$GEN_DIR/ifetch_bodies.hpp"
  python3 "$PROBE_DIR/gen_ifetch_jump.py" --out "$GEN_DIR/ifetch_jump.hpp"
  DEFS=""
fi
# sm_XXXa arch-specific features need the explicit 'a' gencode.
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" \
    -lineinfo --expt-relaxed-constexpr -Xptxas -v ${PTXAS_OPT:+-Xptxas "$PTXAS_OPT"} \
    -I"$GEN_DIR" $DEFS "$PROBE_DIR/ifetch.cu" -o "$OUT/bin/ifetch" > "$OUT/sass/ifetch.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/ifetch" > "$OUT/sass/ifetch.sass" 2>/dev/null
python3 "$MECH_DIR/sass_check_ifetch.py" --sass "$OUT/sass/ifetch.sass" \
    --ptxas "$OUT/sass/ifetch.ptxas.txt" --out "$OUT/sass/ifetch.check.json"
echo "build.sh: ifetch compile+gate done (SM_ARCH=$SM_ARCH)"
