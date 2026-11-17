#!/usr/bin/env bash
set -euo pipefail
MECH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in sm_90a|sm_103a) COMPUTE=${SM_ARCH/sm_/compute_};; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=$CUDA_HOME/bin/nvcc; CUOBJDUMP=$CUDA_HOME/bin/cuobjdump
  export PATH="$CUDA_HOME/bin:$PATH"
else NVCC=nvcc; CUOBJDUMP=cuobjdump; fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr \
    "$MECH_DIR/probe/ctrl2.cu" -o "$OUT/bin/ctrl2" > "$OUT/sass/ctrl2.ptxas.txt" 2>&1
# cuobjdump -sass on the fatbin wrapper yields headers only; extract cubins
# first, then disassemble each
_cub="$OUT/sass/.cub_ctrl2"; rm -rf "$_cub"; mkdir -p "$_cub"
(cd "$_cub" && "$CUOBJDUMP" -xelf all "$OUT/bin/ctrl2" >/dev/null 2>&1)
: > "$OUT/sass/ctrl2.sass"
for c in "$_cub"/*.cubin; do
  [ -e "$c" ] || continue
  "$CUOBJDUMP" -sass "$c" >> "$OUT/sass/ctrl2.sass" 2>/dev/null || true
done
rm -rf "$_cub"
grep -q 'Function :' "$OUT/sass/ctrl2.sass" || \
    echo "build.sh: WARN ctrl2.sass has no functions" >&2
echo "build.sh: ctrl2 done ($SM_ARCH)"
