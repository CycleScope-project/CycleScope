#!/usr/bin/env bash
# Protocol v3 run of the ifetch probe: only the straight-line variants
#   ifetch.line.n<N>.w1  N in V3_NS
#   ifetch.line.n<N>.w4  N in V3_NS_W4 (produced anyway by the w={1,4} loop;
#     --only selects per N, so w4 rows exist for every built N)
# Same ncu settings as run_ncu.sh (--cache-control all, K = 4 passes);
# per-N builds on both devices.
#   TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a NCU=<ncu path>
#   [CUDA_HOME=<dir>] [NCU_LAUNCHES=16] [PTXAS_OPT=-O1] bash run_v3.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NCU=${NCU:-$CUDA_HOME/bin/ncu}
NCU_LAUNCHES=${NCU_LAUNCHES:-16}
NCU_METRICS=${NCU_METRICS:-sm__cycles_active.max,smsp__warps_issue_stalled_no_instruction.sum,smsp__warps_active.sum,smsp__inst_executed.sum}
NS="1792 1856 1920 1984 2048 2112 2176 2240 2304 2432 2560 2816 3072"
PTXAS_OPT=${PTXAS_OPT:--O1}
export PTXAS_OPT
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
"$NCU" --version > "$OUT/ncu-version.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"

build_one() {
  local b=$1 n=$2
  OUT="$b" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" IFETCH_N="$n" \
      timeout 1500 bash "$TREE/tools/microbench/ifetch/build.sh"
}
run_one() {  # <bin dir> <run dir> <only substring>
  local bin=$1 run=$2 only=$3; mkdir -p "$run"
  local command=("$NCU" --csv --page raw --log-file "$run/counters.csv"
    --metrics "$NCU_METRICS"
    --kernel-name-base mangled --kernel-name 'regex:k_(line|mix|jump)ILi'
    --replay-mode kernel --cache-control all --clock-control none
    --check-exit-code yes
    "$bin/ifetch" --out "$run" --launches "$NCU_LAUNCHES" --reps 1
    --only "$only")
  printf '%q ' "${command[@]}" > "$run/command.txt"; echo >> "$run/command.txt"
  timeout 1500 "${command[@]}" > "$run/launches.log" 2>&1 || echo "RUN_FAIL $run rc=$?"
  test -s "$run/counters.csv" || echo "NO_COUNTERS $run"
}
: > "$OUT/build.log"
for n in $NS; do
  if build_one "$OUT/parts/n$n/build" "$n" >> "$OUT/build.log" 2>&1; then echo "BUILD_OK n$n"; else echo "BUILD_FAIL n$n rc=$?"; fi
done
for run in run-a run-b; do
  for n in $NS; do
    [ -x "$OUT/parts/n$n/build/bin/ifetch" ] || continue
    run_one "$OUT/parts/n$n/build/bin" "$OUT/parts/n$n/$run" "ifetch.line.n$n"
  done
  d="$OUT/$run"; mkdir -p "$d"; rm -f "$d/counters.csv" "$d/launches.log"; first=1
  for n in $NS; do
    f="$OUT/parts/n$n/$run/counters.csv"; [ -s "$f" ] || continue
    if [ $first = 1 ]; then cat "$f" > "$d/counters.csv"; first=0; else awk 'NR>2' "$f" >> "$d/counters.csv"; fi
    cat "$OUT/parts/n$n/$run/launches.log" >> "$d/launches.log"
  done
done
python3 - "$OUT/checks.json" "$OUT"/parts/n*/build/sass/ifetch.check.json <<'PY'
import json, sys
out = {"kernels": {}, "n_ok": 0, "n_fail": 0}
for p in sys.argv[2:]:
    d = json.load(open(p)); out["kernels"].update(d.get("kernels", {}))
out["n_ok"] = sum(1 for r in out["kernels"].values() if r.get("ok")); out["n_fail"] = len(out["kernels"]) - out["n_ok"]
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"sass gate (merged): {out['n_ok']} ok {out['n_fail']} fail")
PY
