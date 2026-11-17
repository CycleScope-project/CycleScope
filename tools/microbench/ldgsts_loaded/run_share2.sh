#!/usr/bin/env bash
# ldgsts_loaded shared-read cells below the aggregate L2 read rate (protocol
# addendum "shared_read_unsaturated"): CTAS SMs, 4 warps, G CTAs per region in
# lockstep, and G = 2 with member 1 starting D cycles later.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [NCU=<ncu>] bash run_share2.sh
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
M=lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__m_xbar2l1tex_read_sectors.sum,dram__sectors_read.sum,sm__cycles_active.max
fails=0
cell() {  # name ctas warps group rot shift
  local d="$OUT/run-$run/$1"; mkdir -p "$d"
  "$NCU" --cache-control none --clock-control none --metrics "$M" --csv \
    "$B" --ctas $2 --warps $3 --group $4 --rot $5 --shift $6 --rounds 4096 --samples 2 --out "$d/probe.jsonl" \
    > "$d/ncu.csv" 2> "$d/ncu.err" || { echo "CELL_FAIL $run $1"; fails=$((fails + 1)); }
}
for run in a b; do
  for N in 16 32; do
    cell n$N.w4.g1.d0 $N 4 1 0 0
    for G in 2 4 8; do cell n$N.w4.g$G.d0 $N 4 $G 0 0; done
    cell n$N.w4.g2.rot $N 4 2 1 0
  done
  for D in 32 64 128 192 256 384 512 1024 2048; do cell n32.w4.g2.d$D 32 4 2 0 $D; done
  for D in 128 512; do cell n32.w4.g4.d$D 32 4 4 0 $D; done
  cell n32.w1.g2.d0 32 1 2 0 0
  cell n32.w1.g4.d0 32 1 4 0 0
done
echo "run_share2.sh: done fails=$fails cells=$(ls -d "$OUT"/run-*/* | wc -l)"
