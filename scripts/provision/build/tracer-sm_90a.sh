#!/usr/bin/env bash
# tracer-sm_90a: NVBit tracer .so for sm_90a. delegates to tools/tracer/build.sh which
# writes $CYCLESCOPE_ARTIFACT_ROOT/build/tracer/<ARCH>/tracer.so (driver default).
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
rm -rf "$CYCLESCOPE_ARTIFACT_ROOT/build/tracer/sm_90a"
export CYCLESCOPE_NVBIT_DIR="${CYCLESCOPE_NVBIT_DIR:-$CYCLESCOPE_ARTIFACT_ROOT/tools/nvbit}"
ARCH=sm_90a bash "$CYCLESCOPE_ROOT/tools/tracer/build.sh"
test -f "$CYCLESCOPE_ARTIFACT_ROOT/build/tracer/sm_90a/tracer.so"
