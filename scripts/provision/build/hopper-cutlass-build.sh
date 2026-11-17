#!/usr/bin/env bash
# hopper-cutlass-build: CUTLASS sm_90a example/delivery binaries.
# evidence: scripts/rounds/c296_build_pod.sh L17-28 — env
#   CUTLASS_DELIVERY_ARTIFACTS_ROOT=$OUT, CUTLASS_DELIVERY_COMPILE_ONLY=1,
#   CUTLASS_DELIVERY_BUILD_MANIFEST=.../cutlass_cluster_sm100.json;
#   4 shards: build.sh h100 {0..3} > group<g>.log, each execs
#   <suite>/harness/build_algorithm_shard.sh <g> h100 (build.sh L10-12).
# inputs: hopper-cutlass-source (linked as suites/cutlass/upstream).
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-cutlass-build
H=$CYCLESCOPE_ROOT/suites/hopper/runners/harness
rm -rf "$T"
mkdir -p "$T" "$T/logs"
ln -sfn "$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-cutlass-source" \
  "$H/suites/cutlass/upstream"
# nvdisasm/nvcc location varies (/usr/local/cuda[-X.Y]/bin); add all
# nvcc stays the image toolkit (12.9 — evidence: old logs/build/nvcc_version_*);
# append cuda-toolkit bin only for nvdisasm (cuobjdump looks on PATH)
export PATH="$PATH:$CYCLESCOPE_ARTIFACT_ROOT/tools/cuda-toolkit-13.2/bin"
export CUTLASS_DELIVERY_ARTIFACTS_ROOT="$T"
export CUTLASS_DELIVERY_COMPILE_ONLY=1
# default manifest = suites/cutlass/manifests/cutlass_100.json
# (build_algorithm_shard.sh:23; old group logs built the 100-case set:
#  delivery53/dense_algorithms/example49-69 — evidence:
#  $A/tools/hopper-cutlass-build/group0.log)
for g in 0 1 2 3; do
  bash "$H/suites/cutlass/scripts/build.sh" h100 "$g" \
    > "$T/logs/group$g.log" 2>&1 &
done
wait
find "$T/build" -maxdepth 1 -type f -executable | grep -q .
