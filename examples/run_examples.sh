#!/usr/bin/env bash
# Run the example captures through the CPU pipeline (convert -> build ->
# simulate) and compare the result with the shipped baseline.  No GPU.
#
#   bash examples/run_examples.sh [all|hopper|blackwell] [case ...]
#
# Without arguments all ten examples run.  The engine is built with CMake
# first when its binaries are not there yet; the Python package is used
# straight from this checkout (no pip install needed).
#
# Environment:
#   OUT                   output directory (default <repo>/examples/out)
#   CYCLESCOPE_BUILD_DIR  engine build directory (default <repo>/build)
#   JOBS                  parallel build / shard jobs (default: nproc)
#
# Exit status: 0 when every case ran and matches the baseline, else 1.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARCH=${1:-all}
[ $# -gt 0 ] && shift
CASES=("$@")
OUT=${OUT:-$REPO/examples/out}
BUILD=${CYCLESCOPE_BUILD_DIR:-$REPO/build}
JOBS=${JOBS:-$(nproc)}

case "$ARCH" in
  all) ARCHS=(hopper blackwell) ;;
  hopper|blackwell) ARCHS=("$ARCH") ;;
  *) echo "usage: bash examples/run_examples.sh [all|hopper|blackwell] [case ...]" >&2; exit 2 ;;
esac

for tool in python3 zstd; do
  command -v "$tool" >/dev/null || { echo "run_examples.sh: '$tool' not found on PATH" >&2; exit 2; }
done
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "run_examples.sh: Python >= 3.10 required" >&2; exit 2; }

missing=0
for b in cyclescope-engine cyclescope-convert cyclescope-capture-facts cyclescope-trace-split; do
  [ -x "$BUILD/$b" ] || missing=1
done
if [ "$missing" = 1 ]; then
  command -v cmake >/dev/null || { echo "run_examples.sh: engine not built in $BUILD and cmake not found" >&2; exit 2; }
  echo "== building the engine in $BUILD"
  mkdir -p "$BUILD"
  cmake -S "$REPO" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release > "$BUILD/cmake.log" 2>&1 \
    || { echo "cmake configure failed, see $BUILD/cmake.log" >&2; exit 1; }
  cmake --build "$BUILD" -j "$JOBS" > "$BUILD/build.log" 2>&1 \
    || { echo "engine build failed, see $BUILD/build.log" >&2; exit 1; }
fi

export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export CYCLESCOPE_BUILD_DIR="$BUILD"

fail=0; ran=0; outs=()
for a in "${ARCHS[@]}"; do
  caps="$REPO/examples/captures/$a"
  list=()
  if [ ${#CASES[@]} -gt 0 ]; then
    for c in "${CASES[@]}"; do [ -d "$caps/$c" ] && list+=("$c"); done
  else
    mapfile -t list < <(ls "$caps" | grep -v '^reference$')
  fi
  [ ${#list[@]} -gt 0 ] || continue
  mkdir -p "$OUT/$a"
  outs+=("$OUT/$a")
  for c in "${list[@]}"; do
    ran=$((ran + 1))
    printf '== %s/full_gpu %s ... ' "$a" "$c"
    if python3 -m cyclescope.cli run --suite "$a/full_gpu" --case "$c" \
         --captures "$caps" --out "$OUT/$a" --jobs "$JOBS" > "$OUT/$a/run-$c.log" 2>&1; then
      echo ok
    else
      echo "FAILED (see $OUT/$a/run-$c.log and $OUT/$a/sim/$c/)"
      fail=1
    fi
  done
done
[ "$ran" -gt 0 ] || { echo "run_examples.sh: no matching example case (see examples/captures/<arch>/)" >&2; exit 2; }

echo
python3 -m cyclescope.cli report "${outs[@]}" || fail=1
exit "$fail"
