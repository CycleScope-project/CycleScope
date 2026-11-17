#!/usr/bin/env bash
# constant_launch protocol v2 GPU driver.
#   TREE=<clone> OUT=$A/microbench/constant_launch/<device>-r2 SM_ARCH=...
#   [NCU=<ncu>] bash tools/microbench/constant_launch/run_v2.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
TREE="$TREE" OUT="$OUT" SM_ARCH="$SM_ARCH" NCU="${NCU:-}" \
    bash "$TREE/tools/microbench/constant_launch/run_ncu_v2.sh"
