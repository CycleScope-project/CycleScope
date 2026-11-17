#!/usr/bin/env bash
# TREE=<clone> OUT=<run dir> SM_ARCH=sm_90a|sm_103a bash run.sh
set -uo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv" 2>/dev/null || true
OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
    bash "$TREE/tools/microbench/gather_hit/build.sh" > "$OUT/build.log" 2>&1 \
  || { echo BUILD_FAIL; exit 1; }
BIN="$OUT/build/bin/gather_hit"
SASS="$OUT/build/sass/gather_hit.sass"
ALL=$(nvidia-smi --query-gpu=count --format=csv,noheader >/dev/null 2>&1; python3 - "$SASS" <<'PY' > "$OUT/sass-counts.txt" || echo SASSFAIL
import re, sys
sass = open(sys.argv[1]).read()
# split into functions, per k_gather_hit<K> instance count LDG.E.128
# strictly between consecutive SR_CLOCKLO reads; flag other mem/branch.
fns = re.split(r"Function : ", sass)
for f in fns:
    m = re.match(r"(_Z\d*k_gather_hitILi(\d+)E[^\s]*)", f)
    if not m: continue
    name, K = m.group(1), int(m.group(2))
    lines = [l.split("*/", 1)[-1] for l in f.splitlines()]
    # windows between consecutive clock reads
    clk_idx = [i for i, l in enumerate(lines) if "SR_CLOCKLO" in l]
    ok = True
    pairs = 0
    for a, b in zip(clk_idx, clk_idx[1:]):
        seg = lines[a + 1:b]
        ldg = sum(1 for l in seg if "LDG.E.128" in l)
        other = [l for l in seg
                 if re.search(r"\b(LDG|STG|LDS|STS|ATOM|RED|UTMA|BRA|LDL|STL)\b", l)
                 and "LDG.E.128" not in l]
        pairs += 1
        if ldg != K or other:
            ok = False
            print(f"{name}: pair {pairs}: ldg128={ldg} expect={K} other={other[:3]}")
    print(f"{name}: pairs={pairs} ok={ok}")
    if not ok or pairs == 0: sys.exit(1)
PY
)
cat "$OUT/sass-counts.txt"
grep -q SASSFAIL "$OUT/sass-counts.txt" && exit 1
if [ "$SM_ARCH" = sm_90a ]; then NALL=132; else NALL=148; fi
run_cell() {  # mode N W K L run
  local mode=$1 N=$2 W=$3 K=$4 L=$5 run=$6
  "$BIN" --mode "$mode" --ctas "$N" --warps "$W" --loads "$K" --lines "$L" \
      --launches 8 --out "$OUT/run-$run.jsonl" >>"$OUT/cells.log" 2>&1 \
    || echo "CELL_FAIL $mode.N$N.W$W.K$K.L$L run=$run"
}
for run in a b; do
  for mode in hit cold; do
    for N in 1 $NALL; do
      for W in 1 2 4; do for K in 1 4 8; do for L in 16 32; do
        run_cell $mode $N $W $K $L $run
      done; done; done
      for W in 1 4; do run_cell $mode $N $W 0 32 $run; done
    done
  done
done
echo CELLS_DONE
python3 "$TREE/tools/microbench/gather_hit/table.py" "$OUT" > "$OUT/table.txt" 2>&1 || true
# counter pass: N=1 W=4 K=8 L=32 hit + cold
for mode in hit cold; do
  ${NCU:-ncu} --cache-control none \
      --metrics dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,l1tex__m_xbar2l1tex_read_sectors.sum,sm__cycles_active.max \
      --csv "$BIN" --mode "$mode" --ctas 1 --warps 4 --loads 8 --lines 32 \
      --launches 2 --out /dev/null > "$OUT/ncu-$mode.csv" 2>>"$OUT/ncu.log" \
    || echo "NCU_FAIL $mode"
done
echo "GATHER_HIT_RC=$?"
