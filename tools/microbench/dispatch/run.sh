# dispatch: SM_ARCH OUT — tput c4/c16 waves 2/8/32 + tail control;
# run-a + run-b.
set -e
OUT=${OUT:?}
SM_ARCH=${SM_ARCH:?}
TREE=${TREE:-$(cd "$(dirname "$0")/../../.." && pwd)}
if [ ! -x "$OUT/bin/dispatch" ]; then
  SM_ARCH=$SM_ARCH OUT=$OUT bash \
    "$TREE/tools/microbench/dispatch/build.sh"
fi
BIN=$OUT/bin/dispatch
for run in a b; do
  mkdir -p "$OUT/run-$run"
  for c in 4 16; do
    for w in 2 8 32; do
      $BIN tput 6 $c $w > "$OUT/run-$run/tput.c${c}.w${w}.txt" || exit 1
    done
  done
  $BIN tail 10 20 > "$OUT/run-$run/tail.txt" || exit 1
done
echo "run.sh: dispatch done" > "$OUT/exit.txt"
