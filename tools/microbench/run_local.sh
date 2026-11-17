#!/usr/bin/env bash
# Local microbench runner (replaces the cluster launcher): runs the
# probe driver on the machine it is started on.
#
#   PROTOCOL=<name> DEVICE=h100|b300 RUN=<r> \
#       [CELL=<subdir under tools/microbench/>] [OUT_ROOT=<dir>] \
#       [NCU=<ncu path>] [TIMEOUT=3600] \
#       bash tools/microbench/run_local.sh
#
# Runs the driver in <repo>/tools/microbench/<dir> (dir = CELL under its
# protocol dir, or PROTOCOL itself): run.sh, else run_probes.sh, else
# run_ncu.sh (ncu probes get NCU=...). Environment:
# TREE=<repo> SM_ARCH=sm_90a|sm_103a
# OUT=${OUT_ROOT:-<artifact root>/microbench}/<dir>/<device>-<run>,
# under `timeout $TIMEOUT` (default 3600 s).
set -euo pipefail
PROTOCOL=${PROTOCOL:?protocol name, e.g. service_latency}
DEVICE=${DEVICE:?h100 or b300}
RUN=${RUN:?run tag, e.g. r1}
CELL=${CELL:-}
OUT_ROOT=${OUT_ROOT:-}
TIMEOUT=${TIMEOUT:-3600}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SITE=(python3 -m cyclescope.site)
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
A=$("${SITE[@]}" --get artifact_root)
A=${A%/}

case "$DEVICE" in
  h100) SM_ARCH=sm_90a ;;
  b300) SM_ARCH=sm_103a ;;
  *) echo "run_local.sh: unsupported DEVICE=$DEVICE" >&2; exit 2 ;;
esac

DIR=${CELL:-$PROTOCOL}
OUT="${OUT_ROOT:-$A/microbench}/$DIR/$DEVICE-$RUN"
mkdir -p "$OUT"

SCRIPT=""
for s in run.sh run_probes.sh run_ncu.sh; do
  if [ -f "$REPO/tools/microbench/$DIR/$s" ]; then SCRIPT=$s; break; fi
done
[ -n "$SCRIPT" ] || { echo "run_local.sh: no run.sh/run_probes.sh/run_ncu.sh under $REPO/tools/microbench/$DIR" >&2; exit 2; }

EXTRA_ENV=()
for v in $(env | grep -oE '^(WGMMA_|UMMA_|V2|V3|SASS_CHECK|RUNS|CELLS|SKIP_BUILD|KINDS|PROBE)[A-Z_]*' | sort -u); do
  EXTRA_ENV+=("$v=${!v}")
done
if [ "$SCRIPT" = run_ncu.sh ]; then
  NCU=${NCU:-$("${SITE[@]}" --get ncu)}
  EXTRA_ENV+=("NCU=$NCU")
fi
[ -n "${SET:-}" ] && EXTRA_ENV+=("SET=$SET")

export PATH="$A/tools/cuda-toolkit-13.2/bin:$A/tools/bin:$PATH"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
env TREE="$REPO" SM_ARCH="$SM_ARCH" OUT="$OUT" "${EXTRA_ENV[@]}" \
    timeout "$TIMEOUT" bash "$REPO/tools/microbench/$DIR/$SCRIPT"
echo "run_local.sh: $DIR $DEVICE-$RUN -> $OUT"
