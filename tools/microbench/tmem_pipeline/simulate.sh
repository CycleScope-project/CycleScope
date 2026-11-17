#!/usr/bin/env bash
set -euo pipefail
TREE=${TREE:?}
ROOT=${ROOT:?}
BUILD=${BUILD:?}
BEFORE_ENGINE=${BEFORE_ENGINE:?}
CASES=${CASES:?}
REFERENCE=${REFERENCE:?}
A=${CYCLESCOPE_ARTIFACT_ROOT:?}
export PYTHONPATH="$TREE/python" PATH="$BUILD:$A/tools/bin:$PATH"
mkdir -p "$ROOT/tmp"
export TMPDIR="$ROOT/tmp"
trap 'rc=$?; printf "TMEM_REPLAY_RC=%s\n" "$rc"; sync' EXIT
while IFS= read -r cell; do
  for arm in before after; do
    out="$ROOT/$arm/$cell"
    engine="$BUILD/cyclescope-engine"
    tree="$TREE/configs/trees/b300.json"
    if [ "$arm" = before ]; then
      engine="$BEFORE_ENGINE"
      tree="$ROOT/before-tree.json"
    fi
    mkdir -p "$out"
    if [ -n "${FACTS_ROOT:-}" ]; then
      test -s "$FACTS_ROOT/$arm/$cell/facts/launch.json"
      ln -s "$FACTS_ROOT/$arm/$cell/facts" "$out/facts"
    else
      python3 -m cyclescope.cli convert --capture "$ROOT/cap/$cell" \
        --placement "$ROOT/cap/$cell/placement" --arch blackwell --tree "$tree" \
        --out "$out/facts" --no-cache --reference "$REFERENCE" \
        > "$out/convert.log" 2>&1
    fi
    python3 "$TREE/tools/microbench/tmem_pipeline/replay.py" check-launch "$out/facts/launch.json"
    python3 -m cyclescope.cli convert --facts "$out/facts" --builder cpp \
      --arch blackwell --tree "$tree" --out "$out/out-cpp" --no-cache \
      > "$out/segments.log" 2>&1
    python3 "$TREE/tools/microbench/tmem_pipeline/replay.py" check-launch "$out/out-cpp/launch.json"
    python3 -m cyclescope.cli build "$out/out-cpp" --out "$out/out-cpp/job.bin" \
      --keep-segments > "$out/build.log" 2>&1
    "$engine" simulate --job "$out/out-cpp/job.bin" --out "$out/out-cpp/result.bin" \
      --journal "$out/out-cpp/journal.bin" --wait-ledger "$out/out-cpp/wait.bin" \
      > "$out/sim.log" 2>&1
    printf "CELL_REPLAYED=%s:%s\n" "$arm" "$cell"
  done
done < "$CASES"
sync
