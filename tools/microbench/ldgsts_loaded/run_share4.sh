#!/usr/bin/env bash
# ldgsts_loaded shared-read cells by SM id (protocol addendum
# "shared_read_domain"): all SMs, SMs that differ only in the mask bits of
# their id read the same region in lockstep.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [NCU=<ncu>] bash run_share4.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; NCU=${NCU:-$CUDA_HOME/bin/ncu}
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
"$NCU" --version > "$OUT/ncu-version.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$TREE/tools/microbench/ldgsts_loaded/build.sh" > "$OUT/build.log" 2>&1
B="$OUT/build/bin/ldgsts_loaded"
grep -c 'LDGSTS.E.BYPASS.128' "$OUT/build/sass/ldgsts_loaded.sass" > "$OUT/sass-ldgsts-count.txt"
case "$SM_ARCH" in sm_90a) NMAX=132;; sm_103a) NMAX=148;; esac
M=lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__m_xbar2l1tex_read_sectors.sum,dram__sectors_read.sum,sm__cycles_active.max
fails=0
for run in a b; do
  for W in 1 4; do
    for MASK in 0 1 2 4 8 16 32 64 3 6 7 15 31 63 127 255; do
      d="$OUT/run-$run/w$W.mask$MASK"; mkdir -p "$d"
      "$NCU" --cache-control none --clock-control none --metrics "$M" --csv \
        "$B" --ctas $NMAX --warps $W --smid-mask $MASK --rounds 4096 --samples 2 --out "$d/probe.jsonl" \
        > "$d/ncu.csv" 2> "$d/ncu.err" || { echo "CELL_FAIL $run w$W.mask$MASK"; fails=$((fails + 1)); }
    done
  done
done
echo "run_share4.sh: done fails=$fails cells=$(ls -d "$OUT"/run-*/* | wc -l)"
