#!/usr/bin/env bash
# dispatch_refill run 4 (protocol.json addendum 4): what a PREEXIT costs per CTA,
# per warp and per launch (G block), and what the slow SM groups depend on
# (P block).  TREE=<clone> OUT=<run dir> SM_ARCH=... [SKIP_BUILD=1] bash run4.sh

set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
case "$SM_ARCH" in
  sm_103a) SMS=148 ;;
  sm_90a)  SMS=132 ;;
  *) echo "unknown SM_ARCH $SM_ARCH"; exit 2 ;;
esac
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/dispatch_refill/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/refill"

mkdir -p "$OUT/cells"
# <cell name> <probe args>; all cells --launches 5
BASE="--mode spin --occ 0 --jitter 150"
CELLS=(
  # block P
  "p.spin.a|--mode spin --threads 128 --occ 0 --waves 12 --hold 3000 --jitter 300"
  "p.cpasync|--mode cpasync --threads 128 --occ 0 --waves 12 --hold 3000"
  "p.ldg|--mode ldg --threads 128 --occ 0 --waves 12 --hold 3000"
  "p.rms|--mode rms --threads 128 --occ 0 --grid $((SMS*16*12)) --hold 1400"
  "p.cpasync.m250|--mode cpasync --threads 128 --occ 0 --waves 12 --hold 3000 --mem-permille 250"
  "p.cpasync.m500|--mode cpasync --threads 128 --occ 0 --waves 12 --hold 3000 --mem-permille 500"
  "p.cpasync.h6000|--mode cpasync --threads 128 --occ 0 --waves 12 --hold 6000"
  "p.cpasync.h0|--mode cpasync --threads 128 --occ 0 --waves 4 --hold 0"
  "p.spin.b|--mode spin --threads 128 --occ 0 --waves 12 --hold 3000 --jitter 300"
  # block G
  "g.none|$BASE --threads 128 --waves 6 --hold 2200"
  "g.c250|$BASE --threads 128 --waves 6 --hold 2200 --pdl 2 --pdl-cta 250"
  "g.c500|$BASE --threads 128 --waves 6 --hold 2200 --pdl 2 --pdl-cta 500"
  "g.c750|$BASE --threads 128 --waves 6 --hold 2200 --pdl 2 --pdl-cta 750"
  "g.c1000|$BASE --threads 128 --waves 6 --hold 2200 --pdl 2 --pdl-cta 1000"
  "g.w1|$BASE --threads 128 --waves 6 --hold 2200 --pdl 2 --pdl-warps 1"
  "g.w2|$BASE --threads 128 --waves 6 --hold 2200 --pdl 2 --pdl-warps 2"
  "g.early|$BASE --threads 128 --waves 6 --hold 2200 --pdl 2 --pdl-early 1"
  "g.attr3|$BASE --threads 128 --waves 6 --hold 2200 --pdl 3 --pdl-attr 1"
  "g.attr0|$BASE --threads 128 --waves 6 --hold 2200 --pdl 0 --pdl-attr 1"
  "g.h4500|$BASE --threads 128 --waves 6 --hold 4500 --pdl 2"
  "g.h6000|$BASE --threads 128 --waves 6 --hold 6000 --pdl 2"
  "g.t256.none|$BASE --threads 256 --waves 12 --hold 800"
  "g.t256.pdl|$BASE --threads 256 --waves 12 --hold 800 --pdl 2"
  "g.t32.none|$BASE --threads 32 --waves 6 --hold 2200"
  "g.t32.pdl|$BASE --threads 32 --waves 6 --hold 2200 --pdl 2"
  "g.t512.none|$BASE --threads 512 --waves 24 --hold 300"
  "g.t512.pdl|$BASE --threads 512 --waves 24 --hold 300 --pdl 2"
)
for c in "${CELLS[@]}"; do
  name=${c%%|*}; args=${c#*|}
  # shellcheck disable=SC2086
  $B $args --launches 5 > "$OUT/cells/$name.txt" 2> "$OUT/cells/$name.err" \
      || echo "CELL_FAIL $name rc=$?"
done
echo "run4.sh: dispatch_refill done"
