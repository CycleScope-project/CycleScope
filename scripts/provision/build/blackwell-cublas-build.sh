#!/usr/bin/env bash
# blackwell-cublas-build: cublaslt_case for B300 sm_103a.
# delegates to in-repo suites/blackwell/runners/cublas/build.sh
# (OUT=$CYCLESCOPE_ARTIFACT_ROOT/tools/blackwell-cublas-build).
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
OUT=$CYCLESCOPE_ARTIFACT_ROOT/tools/blackwell-cublas-build \
  bash "$CYCLESCOPE_ROOT/suites/blackwell/runners/cublas/build.sh"
