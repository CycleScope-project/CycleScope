#!/usr/bin/env bash
# Sharing-domain cells of L2 reads for TMA tile loads (tma_scan) and cp.async
# (ldgsts_loaded): protocol-v2.json addendum "read_domain".  Every SM runs
# one CTA; which SMs read the same data is set by SM id (tables from
# domain_tables.py) or by block index (group / far); lockstep start.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [NCU=<ncu>] [CUDA_HOME=<dir>] bash run_domain.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; NCU=${NCU:-$CUDA_HOME/bin/ncu}
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
"$NCU" --version > "$OUT/ncu-version.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
case "$SM_ARCH" in sm_90a) NMAX=132; DEV=h100;; sm_103a) NMAX=148; DEV=b300;; *) exit 2;; esac
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$TREE/tools/microbench/l2_merge/build.sh" > "$OUT/build.log" 2>&1
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" bash "$TREE/tools/microbench/ldgsts_loaded/build.sh" >> "$OUT/build.log" 2>&1
TB="$OUT/build/bin/tma_scan"; LB="$OUT/build/bin/ldgsts_loaded"
grep -c 'UTMALDG' "$OUT/build/sass/tma_scan.sass" > "$OUT/sass-utmaldg-count.txt" || true
grep -c 'LDGSTS.E.BYPASS.128' "$OUT/build/sass/ldgsts_loaded.sass" > "$OUT/sass-ldgsts-count.txt" || true
python3 "$TREE/tools/microbench/l2_merge/domain_tables.py" "$DEV" "$TREE/configs/hardware/$DEV/sm_topology.json" "$OUT/tables" > "$OUT/predictions.txt"
T="$OUT/tables"
M=lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__m_xbar2l1tex_read_sectors.sum,dram__sectors_read.sum,sm__cycles_active.max
fails=0
# tma <cell> <probe args...>: 24 MB footprint, 256 MB per CTA, launch 0 discarded by the probe, launches 1 and 2 read
tma() {
  local d="$OUT/run-$run/tma.$1"; shift; mkdir -p "$d"
  "$NCU" --cache-control none --clock-control none --kernel-name regex:tma_scan_kernel --metrics "$M" --csv \
    "$TB" $NMAX 24 256 2 sync_start=1 "$@" "log=$d/probe.txt" > "$d/ncu.csv" 2> "$d/ncu.err" \
    || { echo "CELL_FAIL $d"; fails=$((fails + 1)); }
}
# cpa <cell> <probe args...>: one warp per CTA
cpa() {
  local d="$OUT/run-$run/cpa.$1"; shift; mkdir -p "$d"
  "$NCU" --cache-control none --clock-control none --metrics "$M" --csv \
    "$LB" --ctas $NMAX --warps 1 --rounds 4096 --samples 2 "$@" --out "$d/probe.jsonl" > "$d/ncu.csv" 2> "$d/ncu.err" \
    || { echo "CELL_FAIL $d"; fails=$((fails + 1)); }
}
for run in a b; do
  for c in m0 m1 m2 m4 m8 m16 m32 m64 m3 m7 m15 m31 m63 m255 halfx gpcx speca blk gpc; do
    tma "$c" "table=$T/$c.tbl"
  done
  tma g2.far0 group=2
  tma g2.far1 group=2 far=1
  tma g4.far0 group=4
  tma g4.far1 group=4 far=1
  tma g2.rot group=2 group_rot=1
  for D in 128 256 512; do
    tma "m1.s$D" "table=$T/m1.tbl" shift_cycles=$D
    tma "halfx.s$D" "table=$T/halfx.tbl" shift_cycles=$D
  done
  tma mc2 multicast=2
  for c in m0 m1 halfx gpcx speca blk gpc; do
    cpa "$c" --table "$T/$c.tbl"
  done
  for D in 256 1024; do
    cpa "m1.s$D" --table "$T/m1.tbl" --shift $D
    cpa "halfx.s$D" --table "$T/halfx.tbl" --shift $D
  done
done
echo "run_domain.sh: done fails=$fails cells=$(ls -d "$OUT"/run-*/* | wc -l)"
