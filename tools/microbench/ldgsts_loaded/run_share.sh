#!/usr/bin/env bash
# ldgsts_loaded shared-read cells (protocol addendum "shared_read"): G CTAs
# read the same region in lockstep or rotated; ncu counts L2 read sectors
# against sectors delivered to the SMs.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [NCU=<ncu>] bash run_share.sh
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
  for cc in none all; do
    for cell in "1 0" "2 0" "2 1" "4 0" "4 1"; do
      set -- $cell; G=$1; R=$2
      name="g$G.rot$R.cc$cc"
      d="$OUT/run-$run/$name"; mkdir -p "$d"
      "$NCU" --cache-control "$cc" --clock-control none --metrics "$M" --csv \
        "$B" --ctas $NMAX --warps 4 --group "$G" --rot "$R" --rounds 4096 --samples 2 --out "$d/probe.jsonl" \
        > "$d/ncu.csv" 2> "$d/ncu.err" || { echo "CELL_FAIL $run $name"; fails=$((fails + 1)); }
    done
  done
done
echo "run_share.sh: done fails=$fails cells=$(ls -d "$OUT"/run-*/* | wc -l)"
