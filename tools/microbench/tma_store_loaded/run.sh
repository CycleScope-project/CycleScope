#!/usr/bin/env bash
# tma_store_loaded cells, two runs (a, b) in one job.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... [V2=1] bash run.sh
# V2=1: protocol-v2.json (probe/store_round.cu); V3=1: protocol-v3.json
# (probe/port_pairs.cu); otherwise protocol.json v1.
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
M="$TREE/tools/microbench/tma_store_loaded"
case "$SM_ARCH" in sm_90a) NMAX=132;; sm_103a) NMAX=148;; esac
if [ "${V3:-0}" = 4 ]; then   # protocol-v4.json (confirmation cells)
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$M/build_v3.sh" > "$OUT/build.log" 2>&1
  B="$OUT/build/bin/port_pairs"
  cell() { "$B" "$@" --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run $*"; }
  G="--row-bytes 64 --swizzle 64 --promo 128 --stride-bytes 10496"
  for run in a b; do
    for bb in 2048 8192 16384; do cell --mode store $G --box-bytes $bb --stages 4; done
    cell --mode stg
    cell --mode sts --writers 4
    cell --mode bars
    for L in 1 2 4; do cell --mode store $G --lkind tma --loaders $L --epi 0 --load-rounds 256; done
    cell --mode store $G --lkind ldg --loaders 4 --epi 0 --load-rounds 1024
    cell --mode store $G --lkind ldgsts --loaders 4 --epi 0 --load-rounds 2048
    for L in 1 2 4; do
      for bb in 2048 8192 16384; do cell --mode store $G --box-bytes $bb --stages 4 --lkind tma --loaders $L; done
      cell --mode stg --lkind tma --loaders $L
      cell --mode sts --writers 4 --lkind tma --loaders $L
    done
    for K in tma ldg ldgsts; do cell --mode bars --lkind $K --loaders 4; done
    for K in ldg ldgsts; do
      cell --mode store $G --stages 4 --lkind $K --loaders 4
      cell --mode stg --lkind $K --loaders 4
      cell --mode sts --writers 4 --lkind $K --loaders 4
    done
    # small L2-resident loader region, one SM and all SMs
    for N in 1 $NMAX; do
      cell --mode store $G --stages 4 --ctas $N --load-region-boxes 8
      cell --mode store $G --ctas $N --lkind tma --loaders 4 --epi 0 --load-rounds 256 --load-region-boxes 8
      cell --mode store $G --stages 4 --ctas $N --lkind tma --loaders 4 --load-region-boxes 8
    done
  done
  echo "run.sh: port_pairs (v4) done"
  exit 0
fi
if [ "${V3:-0}" = 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$M/build_v3.sh" > "$OUT/build.log" 2>&1
  B="$OUT/build/bin/port_pairs"
  cell() { "$B" "$@" --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run $*"; }
  G64W="--row-bytes 64 --swizzle 64 --promo 128 --stride-bytes 10496"
  for run in a b; do
    # each side alone
    cell --mode store $G64W --stages 4
    cell --mode stg
    cell --mode sts --writers 4
    cell --mode fence --writers 0
    cell --mode full $G64W --stages 4
    cell --mode store $G64W --lkind tma --loaders 4 --epi 0 --load-rounds 256
    cell --mode store $G64W --lkind ldgsts --loaders 4 --epi 0 --load-rounds 2048
    cell --mode store $G64W --lkind ldg --loaders 4 --epi 0 --load-rounds 1024
    # pairs
    for K in tma ldgsts ldg; do
      cell --mode store $G64W --stages 4 --lkind $K --loaders 4
      cell --mode stg --lkind $K --loaders 4
      cell --mode sts --writers 4 --lkind $K --loaders 4
      cell --mode fence --writers 0 --lkind $K --loaders 4
      cell --mode full $G64W --stages 4 --lkind $K --loaders 4
    done
  done
  echo "run.sh: port_pairs (v3) done"
  exit 0
fi
if [ "${V2:-0}" = 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$M/build_v2.sh" > "$OUT/build.log" 2>&1
  B="$OUT/build/bin/store_round"
  cell() { "$B" "$@" --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run $*"; }
  G64W="--row-bytes 64 --swizzle 64 --promo 128 --stride-bytes 10496"   # the captured GEMM store map
  G64="--row-bytes 64 --swizzle 64 --promo 128"
  G64N="--row-bytes 64 --swizzle 64 --promo none"
  G128="--row-bytes 128 --swizzle 128 --promo 128"                      # the tma_forms / nvjet store map
  G128N="--row-bytes 128 --swizzle 128 --promo none"
  G128P="--row-bytes 128 --swizzle 128 --promo 256"
  G256N="--row-bytes 256 --swizzle none --promo none"                   # the v1 store map
  for run in a b; do
    # controls
    for bb in 2048 8192 16384; do
      cell --mode store $G128 --box-bytes $bb --stages 1 --stamps 1
    done
    for N in 1 $NMAX; do cell --mode store $G128 --ctas $N --loaders 4 --epi 0 --load-rounds 256; done
    for st in 0 1; do cell --mode sts --writers 4 --stamps $st; done
    # store-only: geometry x ring depth, box scan on two geometries
    for st in 0 1; do
      for D in 1 4; do
        cell --mode store $G64W --stages $D --stamps $st
        cell --mode store $G64 --stages $D --stamps $st
        cell --mode store $G64N --stages $D --stamps $st
        cell --mode store $G128 --stages $D --stamps $st
        cell --mode store $G128N --stages $D --stamps $st
        cell --mode store $G128P --stages $D --stamps $st
        cell --mode store $G256N --stages $D --stamps $st
      done
    done
    for bb in 2048 16384; do
      cell --mode store $G64W --box-bytes $bb --stages 1 --stamps 1
      cell --mode store $G64W --box-bytes $bb --stages 4 --stamps 1
      cell --mode store $G128 --box-bytes $bb --stages 4 --stamps 1
    done
    for N in 1 $NMAX; do
      for st in 0 1; do cell --mode store $G64W --ctas $N --loaders 4 --stages 4 --stamps $st; done
    done
    cell --mode store $G64W --ctas $NMAX --stages 4 --stamps 0
    # fence-only: bytes written per round, loads in flight
    for st in 0 1; do
      for W in 0 1 4; do cell --mode fence --writers $W --stamps $st; done
      cell --mode fence --writers 4 --loaders 4 --stamps $st
      cell --mode fence --writers 0 --loaders 4 --stamps $st
    done
    # full round
    for st in 0 1; do
      for N in 1 $NMAX; do
        for L in 0 4; do
          cell --mode full $G64W --ctas $N --loaders $L --stages 4 --stamps $st
          cell --mode full $G128 --ctas $N --loaders $L --stages 4 --stamps $st
        done
      done
      cell --mode full $G256N --stages 4 --stamps $st
      cell --mode full $G64W --stages 1 --stamps $st
      cell --mode full $G64W --writers 1 --stages 4 --stamps $st
    done
  done
  echo "run.sh: store_round (v2) done"
  exit 0
fi
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$M/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/tma_store_loaded"
cell() { "$B" "$@" --samples 8 --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run $*"; }
for run in a b; do
  # controls: one store at a time (wait_group 0), loader-only baselines
  cell --kind tma --ctas 1 --loaders 0 --stages 1 --dst ring --rounds 264 --stamps 1
  cell --kind tma --ctas 1 --loaders 0 --stages 1 --dst ring --rounds 264 --stamps 0
  cell --kind tma --ctas $NMAX --loaders 0 --stages 1 --dst ring --rounds 264 --stamps 0
  for N in 1 16 64 $NMAX; do
    cell --kind tma --ctas $N --loaders 4 --epi 0 --load-rounds 256
  done
  # main: 4-stage ring
  for N in 1 16 64 $NMAX; do
    for L in 0 4; do
      for st in 0 1; do
        cell --kind tma --ctas $N --loaders $L --stages 4 --dst ring --rounds 264 --stamps $st
        cell --kind tma --ctas $N --loaders $L --stages 4 --dst fresh --state warm --rounds 72 --stamps $st
      done
      cell --kind tma --ctas $N --loaders $L --stages 4 --dst fresh --state cold --rounds 72 --stamps 0
      # direct global stores of the same bytes
      cell --kind stg --ctas $N --loaders $L --stages 4 --dst ring --rounds 264 --stamps 0
      cell --kind stg --ctas $N --loaders $L --stages 4 --dst fresh --state warm --rounds 72 --stamps 0
    done
  done
done
echo "run.sh: tma_store_loaded done"
