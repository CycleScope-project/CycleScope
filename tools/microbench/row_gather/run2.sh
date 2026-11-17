#!/usr/bin/env bash
# row_gather run-2 cells per protocol.json addendum run_2.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a bash run2.sh
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
# pair: one warp, grid 1
for D in 128 4096 65536 2097152 67108864; do
  cell "pair.d$D" --mode pair --delta "$D" --warps 1 --grid 1
done
# matrix: one warp per CTA, grid = all SMs, only active_cta works, shared slots
for C in 0 11 22 33 44 55 66 77 88 99 110 121; do
  cell "matrix.c$C" --mode single --hint plain --warps 1 --rows 1 --chunk 1 \
      --spread far --active-cta "$C" --shared-slots 1
done
echo CELLS_DONE
