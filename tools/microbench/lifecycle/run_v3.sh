#!/usr/bin/env bash
# Protocol v3 GPU driver: ncu hardware runs (run-a/run-b) then one traced
# capture per engine cell.  Launched per device from the clone:
#   TREE=<clone> OUT=$A/microbench/lifecycle/<device>-v3r1 SM_ARCH=...
#   [NCU=<ncu>] bash tools/microbench/lifecycle/run_v3.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
A=${CYCLESCOPE_ARTIFACT_ROOT:?}
export PYTHONPATH="$TREE/python${PYTHONPATH:+:$PYTHONPATH}"

TREE="$TREE" OUT="$OUT" SM_ARCH="$SM_ARCH" NCU="${NCU:-}" \
    bash "$TREE/tools/microbench/lifecycle/run_ncu_v3.sh"

python3 "$TREE/tools/microbench/lifecycle/capture_probe.py" \
    --build "$OUT/build" --out "$OUT/capture" --sm-arch "$SM_ARCH" \
    --tracer "$A/build/tracer/$SM_ARCH/tracer-072fa0a.so"
