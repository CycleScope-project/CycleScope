#!/usr/bin/env bash
# Build the load_port_lines probe and dump k_port SASS.
#   OUT=<dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash build.sh
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
    "$D/probe/port_lines.cu" -o "$OUT/bin/port_lines" \
    > "$OUT/sass/port_lines.ptxas.txt" 2>&1
cuobjdump --dump-sass "$OUT/bin/port_lines" \
    | awk '/Function : .*k_port/{p=1} p{print} p&&/^\s*$/{p=0}' \
    > "$OUT/sass/port_lines.sass"
wc -l "$OUT/sass/port_lines.sass" >&2
