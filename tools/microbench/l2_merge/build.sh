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
# -lcuda: cuTensorMapEncodeTiled is a driver API
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr \
    "$MECH_DIR/probe/tma_scan.cu" -o "$OUT/bin/tma_scan" -lcuda > "$OUT/sass/tma_scan.ptxas.txt" 2>&1
"$CUOBJDUMP" -sass "$OUT/bin/tma_scan" > "$OUT/sass/tma_scan.sass" 2>/dev/null || true
# ldg_scan (protocol v2, LDG.E.128 streamer): no driver API -> no -lcuda
"$NVCC" -O3 -std=c++17 -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr \
    "$MECH_DIR/probe/ldg_scan.cu" -o "$OUT/bin/ldg_scan" > "$OUT/sass/ldg_scan.ptxas.txt" 2>&1
# cuobjdump -sass on the fatbin wrapper yields headers only; extract cubins
# first, then disassemble each (tensor_pipe pattern)
_cub="$OUT/sass/.cub_ldg_scan"; rm -rf "$_cub"; mkdir -p "$_cub"
(cd "$_cub" && "$CUOBJDUMP" -xelf all "$OUT/bin/ldg_scan" >/dev/null 2>&1)
: > "$OUT/sass/ldg_scan.sass"
for c in "$_cub"/*.cubin; do
  [ -e "$c" ] || continue
  "$CUOBJDUMP" -sass "$c" >> "$OUT/sass/ldg_scan.sass" 2>/dev/null || true
done
rm -rf "$_cub"
grep -q 'Function :' "$OUT/sass/ldg_scan.sass" || \
    echo "build.sh: WARN ldg_scan.sass has no functions" >&2
# protocol-v2 sass gate: scan_kernel<false> must carry exactly UNROLL LDG.E.128
python3 "$MECH_DIR/sass_check.py" "$OUT/sass/ldg_scan.sass" > "$OUT/sass_check.txt" || true
head -1 "$OUT/sass_check.txt"
echo "build.sh: tma_scan + ldg_scan done ($SM_ARCH)"
