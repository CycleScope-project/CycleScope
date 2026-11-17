#!/usr/bin/env bash
# Build the row_gather probe, dump SASS, run the SASS gate.
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash build.sh
# Put the dump tools on PATH for the SASS dump.
set -euo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?OUT is required}
SM_ARCH=${SM_ARCH:?SM_ARCH is required}
case "$SM_ARCH" in
  sm_90a) COMPUTE=compute_90a ;;
  sm_103a) COMPUTE=compute_103a ;;
  *) echo "build.sh: unsupported SM_ARCH=$SM_ARCH" >&2; exit 2 ;;
esac
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}
  export PATH="$CUDA_HOME/bin:$PATH"
else
  NVCC=${NVCC:-nvcc}
fi
mkdir -p "$OUT/bin" "$OUT/sass"
"$NVCC" -O3 -std=c++17 \
    -gencode "arch=$COMPUTE,code=$SM_ARCH" -lineinfo \
    "$D/probe/row_gather.cu" -o "$OUT/bin/row_gather" \
    > "$OUT/sass/row_gather.ptxas.txt" 2>&1
cuobjdump --dump-sass "$OUT/bin/row_gather" > "$OUT/sass/row_gather.sass"
for k in k_single_plain k_single_hint k_block_plain k_block_hint k_pair; do
  awk -v k="$k" 'index($0,"Function : ")&&index($0,k){p=1} p{print} p&&/^\s*$/{p=0}' \
      "$OUT/sass/row_gather.sass" > "$OUT/sass/$k.sass"
  wc -l "$OUT/sass/$k.sass" >&2
done
python3 "$D/probe/sass_gate.py" "$OUT/sass/row_gather.sass" "$OUT/sass/gate.txt"
