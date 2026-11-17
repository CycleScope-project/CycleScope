#!/usr/bin/env bash
# hopper-flashinfer-delivery: FLASHINFER_DELIVERY_ARTIFACTS_ROOT content
# (build/cubins + build/cuda-link per build.sh:19,32; environment_far.sh:30-32).
# delegates to harness suites/flashinfer/scripts/build.sh h100 with
# SOURCE_ROOT=suite dir (upstream symlink -> hopper-flashinfer-source).
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-flashinfer-delivery
H=$CYCLESCOPE_ROOT/suites/hopper/runners/harness
rm -rf "$T"
mkdir -p "$T"
ln -sfn "$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-flashinfer-source" \
  "$H/suites/flashinfer/upstream"
export FLASHINFER_DELIVERY_ARTIFACTS_ROOT="$T"
# image ships flashinfer-jit-cache 0.6.6 which breaks 0.6.17 checks
# (evidence: run_cluster_sm.sh:31-32)
python3 -m pip uninstall -y flashinfer-jit-cache || true
build_log="$(mktemp)"
bash "$H/suites/flashinfer/scripts/build.sh" h100 2>&1 | tee "$build_log"
mkdir -p "$T/logs" && mv "$build_log" "$T/logs/build.log"
test -d "$T/build/cuda-link" -o -d "$T/build/cubins"
