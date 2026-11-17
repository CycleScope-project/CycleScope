#!/usr/bin/env bash
# row_gather run-3 (side map) per protocol.json addendum run_3.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a bash run3.sh
set -uo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT/cells"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/row_gather/build.sh" > "$OUT/build.log" 2>&1 \
    || { echo "BUILD_FAIL"; tail -20 "$OUT/build.log"; exit 1; }
fi
BIN="$OUT/build/bin/row_gather"
timeout 3600 "$BIN" --mode sweep --warps 1 --rows 1 --chunk 1 --spread far \
    --grid 0 --iters 32 --launches 1 --table-mib 1024 \
    --evict-mib 512 --spoil-mib 4 \
    > "$OUT/cells/sweep.txt" 2> "$OUT/cells/sweep.err" \
  || echo "CELL_FAIL sweep rc=$?"
echo RUN3_DONE
