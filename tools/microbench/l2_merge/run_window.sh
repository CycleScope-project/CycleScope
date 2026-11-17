#!/usr/bin/env bash
# cp.async cells of merge_mechanics parts A and D again (protocol-v2.json
# addendum "async_copy_window"): cp.async (ldgsts_loaded) and TMA (tma_scan) cells in
# which only a named set of SMs issues loads (tables from domain_tables.py
# ... mechanics; the other SMs' CTAs stay idle).
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [NCU=<ncu>] [CUDA_HOME=<dir>] bash run_window.sh
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
grep -c 'GLOBALTIMER' "$OUT/build/sass/ldgsts_loaded.sass" > "$OUT/sass-gtimer-count.txt" || true
python3 "$TREE/tools/microbench/l2_merge/domain_tables.py" "$DEV" "$TREE/configs/hardware/$DEV/sm_topology.json" "$OUT/tables" mechanics > "$OUT/tables.txt"
T="$OUT/tables"
M=lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__m_xbar2l1tex_read_sectors.sum,dram__sectors_read.sum,sm__cycles_active.max
fails=0
tma() {
  local d="$OUT/run-$run/tma.$1"; shift; mkdir -p "$d"
  "$NCU" --cache-control none --clock-control none --kernel-name regex:tma_scan_kernel --metrics "$M" --csv \
    "$TB" $NMAX 24 256 2 sync_start=1 "$@" "log=$d/probe.txt" > "$d/ncu.csv" 2> "$d/ncu.err" \
    || { echo "CELL_FAIL $d"; fails=$((fails + 1)); }
}
# cpa <cell> <warps> <probe args...>
cpa() {
  local d="$OUT/run-$run/cpa.$1"; local w=$2; shift 2; mkdir -p "$d"
  "$NCU" --cache-control none --clock-control none --metrics "$M" --csv \
    "$LB" --ctas $NMAX --warps $w --rounds 4096 --samples 2 "$@" --out "$d/probe.jsonl" > "$d/ncu.csv" 2> "$d/ncu.err" \
    || { echo "CELL_FAIL $d"; fails=$((fails + 1)); }
}
for run in a b; do
  for D in 0 8 16 24 32 40 48 64 96; do
    cpa "m1.p1200.s$D" 1 --table "$T/m1.tbl" --pace 1200 --shift $D
    cpa "xp.p1200.s$D" 1 --table "$T/xp.tbl" --pace 1200 --shift $D
  done
done
echo "run_window.sh: done fails=$fails cells=$(ls -d "$OUT"/run-*/* | wc -l)"
