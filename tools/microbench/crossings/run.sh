#!/usr/bin/env bash
# crossings protocol-v2: run all six cells, each under $OUT/<cell>, one
# cell's failure does not stop the rest (CELL_FAIL lines, summary at end).
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a bash run.sh
set -uo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
CELLS="trywait_timeout constant_cache l1_residency descriptor_refetch l1_bandwidth store_dram_port"
FAILS=""
for cell in $CELLS; do
  echo "== cell $cell -> $OUT/$cell"
  OUT="$OUT/$cell" TREE="$TREE" SM_ARCH="$SM_ARCH" \
      timeout 1500 bash "$TREE/tools/microbench/crossings/$cell/run.sh" \
      || { rc=$?; echo "CELL_FAIL $cell rc=$rc"; FAILS="$FAILS $cell"; }
done
echo "crossings run.sh done; failed cells:${FAILS:- none}"
