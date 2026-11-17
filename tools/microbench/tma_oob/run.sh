#!/usr/bin/env bash
# tma_oob v2 cells (protocol-v2.json): 13 configs x N in {1, all SMs} x S in {1,4,8},
# runs a and b (78 rows per run file); then a counter pass (ncu) on 4 configs.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; echo "TMA_OOB_RC=$rc"; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/tma_oob/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/tma_oob"
cp "$OUT/build/sass/tma_oob.sass" "$OUT/build/tma_oob.sass" 2>/dev/null || true
python3 - "$OUT/build/sass/tma_oob.sass" "$OUT/build/sass-counts.txt" <<'PY'
import re, sys
sass = open(sys.argv[1]).read()
parts = re.split(r"Function : ", sass)
counts = {}
for p in parts[1:]:
    name = p.split()[0]
    counts[name] = len(re.findall(r"UTMALDG", p))
with open(sys.argv[2], "w") as f:
    for n, c in counts.items():
        f.write(f"{n}\t{c}\n")
if not counts or any(c == 0 for c in counts.values()):
    print("SASS_FAIL", counts)
    sys.exit(3)
print("sass counts:", counts)
PY
[ $? -eq 0 ] || exit 3
case "$SM_ARCH" in sm_90a) NMAX=132;; sm_103a) NMAX=148;; esac
run_cell() {  # run_cell <run> <extra args...>
  local run=$1; shift
  for N in 1 $NMAX; do
    for S in 1 4 8; do
      "$B" --ctas $N --slots $S --rounds 2048 --samples 8 "$@" --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run $* N$N S$S"
    done
  done
}
for run in a b; do
  for x in 0 16 32 48 64; do run_cell $run --box-rows 128 --x $x; done            # (a)
  for R in 128 96 64 32; do run_cell $run --box-rows 128 --x 0 --rows-in $R; done # (b)
  for x in 0 32; do run_cell $run --box-rows 32 --x $x; done                      # (c1)
  for R in 32 16; do run_cell $run --box-rows 32 --x 0 --rows-in $R; done         # (c2)
done
# counter pass: N = all SMs, S = 4, --samples 2, --cache-control none
NCU=${NCU:?ncu path required (run_ncu.sh exports NCU)}
for cfg in "x16 --box-rows 128 --x 16" "x48 --box-rows 128 --x 48" "r64 --box-rows 128 --x 0 --rows-in 64" "b32x32 --box-rows 32 --x 32"; do
  name=${cfg%% *}; args=${cfg#* }
  "$NCU" --cache-control none --csv --metrics \
    dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,l1tex__m_xbar2l1tex_read_sectors.sum,sm__cycles_active.max \
    "$B" --ctas $NMAX --slots 4 --rounds 2048 --samples 2 $args --out /dev/null \
    > "$OUT/ncu-$name.csv" 2> "$OUT/ncu-$name.log" || echo "NCU_FAIL $name"
done
echo "run.sh: tma_oob done"
