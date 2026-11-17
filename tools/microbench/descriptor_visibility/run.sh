set -e
OUT=${OUT:?}; SM_ARCH=${SM_ARCH:?}
TREE=${TREE:-$(cd "$(dirname "$0")/../../.." && pwd)}
if [ ! -x "$OUT/bin/descvis" ]; then
  SM_ARCH=$SM_ARCH OUT=$OUT bash \
    "$TREE/tools/microbench/descriptor_visibility/build.sh"
fi
# correction (c): one process per cell, control first, then D ascending
for run in a b; do
  mkdir -p "$OUT/run-$run"
  for cell in ctl d0 d128 d256 d512 d1024 d2048 d4096 d8192; do
    "$OUT/bin/descvis" --out "$OUT/run-$run" --launches 8 --reps 33 \
      --cell "$cell" >> "$OUT/run-$run.stdout" 2>&1 || exit 1
    [ -f "$OUT/run-$run/descvis.jsonl" ] && \
      mv "$OUT/run-$run/descvis.jsonl" "$OUT/run-$run/descvis_$cell.jsonl"
  done
done
echo done > "$OUT/exit.txt"
