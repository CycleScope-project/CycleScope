#!/usr/bin/env bash
# copy CUDA 13.2 toolkit from the B300 image's /usr/local/cuda-13.2.
# Runs INSIDE the Blackwell toolchain image (the toolkit's own
# origin). evidence: $B/tools/cuda-13.2-image/exit.txt recorded image+src.
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
SRC="${CUDA_SRC:-/usr/local/cuda-13.2}"
[ -d "$SRC" ] || SRC=/usr/local/cuda
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/cuda-toolkit-13.2
rm -rf "$T"
mkdir -p "$T"
cp -a "$SRC"/. "$T"/
test -x "$T/bin/nvdisasm"
