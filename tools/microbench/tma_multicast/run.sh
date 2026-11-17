#!/usr/bin/env bash
# tma_multicast cells (protocol.json): N in {2,16,all SMs even} x S in {2,4,8}
# x mode {plain,plain_nohs,mc_both,mc_one}; 2048 rounds, 8 samples, runs a/b.
#   TREE=<clone> OUT=<run dir> SM_ARCH=... bash run.sh
set -euo pipefail
TREE=${TREE:?}; OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"
trap 'rc=$?; printf "end=%s exit=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$OUT/exit.txt" || :; printf "TMA_MCAST_RC=%s\n" "$rc"; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$OUT/identity.csv"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  OUT="$OUT/build" SM_ARCH="$SM_ARCH" CUDA_HOME="$CUDA_HOME" \
      bash "$TREE/tools/microbench/tma_multicast/build.sh" > "$OUT/build.log" 2>&1
fi
B="$OUT/build/bin/tma_multicast"
for run in a b; do
  for N in 2 16 0; do
    for S in 2 4 8; do
      for m in plain plain_nohs mc_both mc_one; do
        "$B" --mode $m --ctas $N --slots $S --rounds 2048 --samples 8 \
            --out "$OUT/run-$run.jsonl" || echo "CELL_FAIL $run $m N$N S$S"
      done
    done
  done
done
# counter pass (ncu, cache-control none): N = all, S = 4, three modes
if command -v ncu >/dev/null 2>&1; then
  for m in plain mc_both mc_one; do
    ncu --cache-control none --csv --target-processes all \
        --metrics lts__t_sectors_srcunit_tex_op_read.sum,l1tex__m_xbar2l1tex_read_sectors.sum,dram__bytes_read.sum,sm__cycles_active.max \
        "$B" --mode $m --ctas 0 --slots 4 --rounds 512 --samples 1 \
        --out "$OUT/ncu-$m.jsonl" > "$OUT/ncu-$m.csv" 2>"$OUT/ncu-$m.err" \
        || echo "NCU_FAIL $m" >> "$OUT/ncu-fail.txt"
  done
else
  echo "NCU_FAIL ncu missing" > "$OUT/ncu-fail.txt"
fi
python3 "$TREE/tools/microbench/tma_multicast/table.py" "$OUT" > "$OUT/table.txt"
echo "run.sh: tma_multicast done"
