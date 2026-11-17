#!/usr/bin/env bash
# ldgsts_loaded protocol addendum "issue_cost": what issuing a cp.async copy costs the warp.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a [CUDA_HOME=<dir>] bash run_issue.sh
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
    || { echo "run_issue.sh: BUILD FAILED"; tail -30 "$OUT/build/sass/ldgsts_issue.ptxas.txt"; exit 3; }
"$CUOBJDUMP" -sass "$OUT/build/bin/ldgsts_issue" > "$OUT/build/sass/ldgsts_issue.sass" 2>/dev/null || true
grep -c 'LDGSTS.E.BYPASS.128' "$OUT/build/sass/ldgsts_issue.sass" > "$OUT/sass-ldgsts-count.txt" || true
echo "run_issue.sh: build done, LDGSTS in sass $(cat "$OUT/sass-ldgsts-count.txt")"
python3 "$TREE/tools/microbench/l2_merge/domain_tables.py" "$DEV" "$TREE/configs/hardware/$DEV/sm_topology.json" "$OUT/tables" mechanics > "$OUT/tables.txt"
B="$OUT/build/bin/ldgsts_issue"
fails=0
cell() { "$B" --ctas $NMAX --rounds 2048 --samples 4 "$@" --out "$OUT/run-$run.jsonl" >> "$OUT/run-$run.stdout" 2>&1 || { echo "CELL_FAIL $run $*"; fails=$((fails + 1)); }; }
for run in a b; do
  for S in t1_m0 all_m0; do
    for W in 1 2 4 8; do
      for C in 1 2 4 8 16; do
        for K in 0 3; do cell --table "$OUT/tables/$S.tbl" --warps $W --copies $C --keep $K --tag "$S.w$W.c$C.k$K"; done
      done
      cell --table "$OUT/tables/$S.tbl" --warps $W --copies 4 --keep 1 --tag "$S.w$W.c4.k1"
    done
  done
done
echo "run_issue.sh: done fails=$fails rows_a=$(wc -l < "$OUT/run-a.jsonl") rows_b=$(wc -l < "$OUT/run-b.jsonl")"
[ "$fails" = 0 ]
