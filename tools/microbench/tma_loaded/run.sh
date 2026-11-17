#!/usr/bin/env bash
# tma_loaded cells (protocol.json): N in {1,16,64,128,SMs} x S in {1,2,4,8}, two runs.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/tma_loaded/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/tma_loaded"
case "$SM_ARCH" in sm_90a) NMAX=132;; sm_103a) NMAX=148;; esac
for run in a b; do
  for N in 1 16 64 128 $NMAX; do
    for S in 1 2 4 8; do
      "$B" --ctas $N --slots $S --rounds 2048 --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run N$N S$S"
    done
  done
done
# v3 cells: every CTA loads the same cold boxes (followers of one DRAM fill)
for run in a b; do
  for N in 1 16 64 $NMAX; do
    for S in 1 4; do
      "$B" --mode shared_cold --ctas $N --slots $S --rounds 512 --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run shared_cold N$N S$S"
    done
  done
done
# v4 cells: two participating CTAs on SM 0 + SM 1 (same TPC) vs SM 0 + SM 2 (different TPCs)
for run in a b; do
  for pr in same diff; do
    for S in 2 8; do
      "$B" --pair $pr --slots $S --rounds 2048 --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run pair_$pr S$S"
    done
  done
done
# v5 cells: every CTA walks the same hot boxes (hot-line serialisation question)
for run in a b; do
  for N in 1 16 64 $NMAX; do
    for S in 2 8; do
      "$B" --mode shared_hot --ctas $N --slots $S --rounds 2048 --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run shared_hot N$N S$S"
    done
  done
done
echo "run.sh: tma_loaded done"
