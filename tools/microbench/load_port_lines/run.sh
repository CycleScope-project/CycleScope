#!/usr/bin/env bash
# load_port_lines cells: clock64 probe, plain execution.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a bash run.sh
set -uo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT/cells"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/load_port_lines/build.sh" > "$OUT/build.log" 2>&1
fi
BIN="$OUT/build/bin/port_lines"
for LV in l1 l2 dram; do
  case "$LV" in
    l1) WS="1 4" ;;
    l2) WS="1 4 8" ;;
    dram) WS="1 4 16" ;;
  esac
  for W in $WS; do
    for S in 4 32 128 512; do
      name="$LV.w$W.s$S"
      timeout 900 "$BIN" --level "$LV" --warps "$W" --spread "$S" --launches 6 \
          > "$OUT/cells/$name.txt" 2> "$OUT/cells/$name.err" \
        || echo "CELL_FAIL $name rc=$?"
    done
  done
done
echo CELLS_DONE
