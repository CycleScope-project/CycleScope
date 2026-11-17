#!/usr/bin/env bash
# constant_launch GPU driver: ncu hardware runs (run-a/run-b, 34 launches
# per variant each) + plain passes.  Launched per device from the clone:
#   TREE=<clone> OUT=$A/microbench/constant_launch/<device>-r1 SM_ARCH=...
#   [NCU=<ncu>] bash tools/microbench/constant_launch/run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
TREE="$TREE" OUT="$OUT" SM_ARCH="$SM_ARCH" NCU="${NCU:-}" \
    bash "$TREE/tools/microbench/constant_launch/run_ncu.sh"
