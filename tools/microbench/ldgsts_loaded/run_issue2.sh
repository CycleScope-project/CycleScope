#!/usr/bin/env bash
# ldgsts_loaded protocol addendum "issue_cost_probe_check": why ldgsts_issue streams half of ldgsts_loaded.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash run_issue2.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
test ! -e "$OUT"
mkdir -p "$OUT/build/bin" "$OUT/build/sass"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
case "$SM_ARCH" in sm_90a) NMAX=132; DEV=h100;; sm_103a) NMAX=148; DEV=b300;; *) exit 2;; esac
if [ -n "${CUDA_HOME:-}" ]; then NVCC=$CUDA_HOME/bin/nvcc; CUOBJDUMP=$CUDA_HOME/bin/cuobjdump; else NVCC=nvcc; CUOBJDUMP=cuobjdump; fi
M="$TREE/tools/microbench/ldgsts_loaded"
"$NVCC" -O3 -std=c++17 -gencode "arch=${SM_ARCH/sm_/compute_},code=$SM_ARCH" -lineinfo --expt-relaxed-constexpr \
    "$M/probe/ldgsts_issue.cu" -o "$OUT/build/bin/ldgsts_issue" -lcuda > "$OUT/build/sass/ldgsts_issue.ptxas.txt" 2>&1 \
    || { echo "run_issue2.sh: BUILD FAILED"; tail -30 "$OUT/build/sass/ldgsts_issue.ptxas.txt"; exit 3; }
"$CUOBJDUMP" -sass "$OUT/build/bin/ldgsts_issue" > "$OUT/build/sass/ldgsts_issue.sass" 2>/dev/null || true
grep -c 'LDGSTS.E.BYPASS.128' "$OUT/build/sass/ldgsts_issue.sass" > "$OUT/sass-ldgsts-count.txt" || true
echo "run_issue2.sh: build done, LDGSTS in sass $(cat "$OUT/sass-ldgsts-count.txt")"
python3 "$TREE/tools/microbench/l2_merge/domain_tables.py" "$DEV" "$TREE/configs/hardware/$DEV/sm_topology.json" "$OUT/tables" mechanics > "$OUT/tables.txt"
B="$OUT/build/bin/ldgsts_issue"
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="${CUDA_HOME:-}" bash "$M/build.sh" >> "$OUT/build.log" 2>&1
LB="$OUT/build/bin/ldgsts_loaded"
fails=0
cell() { "$B" --ctas $NMAX --rounds 2048 --samples 4 "$@" --out "$OUT/run-$run.jsonl" >> "$OUT/run-$run.stdout" 2>&1 || { echo "CELL_FAIL $run $*"; fails=$((fails + 1)); }; }
ref() { "$LB" --ctas $NMAX --warps "$2" --rounds 4096 --samples 4 --table "$OUT/tables/$1.tbl" --out "$OUT/ref-$run.jsonl" >> "$OUT/ref-$run.stdout" 2>&1 || { echo "CELL_FAIL $run ref $*"; fails=$((fails + 1)); }; }
for run in a b; do
  for S in t1_m0 all_m0; do
    for W in 4 8; do
      ref $S $W
      for NC in 0 1; do
        for IL in 0 1; do cell --table "$OUT/tables/$S.tbl" --warps $W --copies 4 --keep 3 --noclock $NC --interleave $IL --tag "$S.w$W.c4.k3.nc$NC.il$IL"; done
        cell --table "$OUT/tables/$S.tbl" --warps $W --copies 16 --keep 3 --noclock $NC --tag "$S.w$W.c16.k3.nc$NC.il0"
      done
    done
  done
done
echo "run_issue2.sh: done fails=$fails rows_a=$(wc -l < "$OUT/run-a.jsonl") rows_b=$(wc -l < "$OUT/run-b.jsonl") ref_a=$(wc -l < "$OUT/ref-a.jsonl")"
[ "$fails" = 0 ]
