#!/usr/bin/env bash
# row_gather cells per protocol.json "cells".
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a bash run.sh
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
cell() {  # cell <name> <args...>
  local name=$1; shift
  timeout 1800 "$BIN" "$@" --iters 32 --launches 5 --table-mib 1024 \
      --evict-mib 512 --spoil-mib 4 \
      > "$OUT/cells/$name.txt" 2> "$OUT/cells/$name.err" \
    || echo "CELL_FAIL $name rc=$?"
}
# single: one warp, grid 1
for L in 1 2 4 8 16 32; do
  for H in plain kernel; do
    cell "single.far.l$L.c1.$H" --mode single --hint "$H" --warps 1 \
        --rows "$L" --chunk 1 --spread far --grid 1
  done
done
for H in plain kernel; do
  cell "single.far.l8.c4.$H" --mode single --hint "$H" --warps 1 \
      --rows 8 --chunk 4 --spread far --grid 1
  cell "single.adjacent.l32.c1.$H" --mode single --hint "$H" --warps 1 \
      --rows 32 --chunk 1 --spread adjacent --grid 1
done
# block: kernel hints on grid 1 and all SMs, plain on grid 1
for W in 1 4; do
  cell "block.w$W.g1.kernel" --mode block --hint kernel --warps "$W" --grid 1
  cell "block.w$W.gall.kernel" --mode block --hint kernel --warps "$W"
done
cell "block.w1.g1.plain" --mode block --hint plain --warps 1 --grid 1
echo CELLS_DONE
