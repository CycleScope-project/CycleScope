#!/usr/bin/env bash
# gather_latency cells: clock64 probe, plain execution.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a bash run.sh
set -uo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT/cells"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/gather_latency/build.sh" > "$OUT/build.log" 2>&1
fi
BIN="$OUT/build/bin/gather"
for W in 1 4 16; do
  for K in 1 7; do
    for R in 64 65536; do
      for G in all 1; do
        if [ "$G" = 1 ]; then name="w$W.k$K.r$R.g1"; extra="--grid 1"
        else name="w$W.k$K.r$R.gall"; extra=""; fi
        timeout 900 "$BIN" --warps "$W" --k "$K" --rows "$R" \
            --iters 16 --launches 5 --evict-mib 512 $extra \
            > "$OUT/cells/$name.txt" 2> "$OUT/cells/$name.err" \
          || echo "CELL_FAIL $name rc=$?"
      done
    done
  done
done
echo CELLS_DONE
