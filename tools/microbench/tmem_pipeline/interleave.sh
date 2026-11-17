#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:?}
HERE=$(dirname "${BASH_SOURCE[0]}")
trap 'rc=$?; sync; printf "TMEM_INTERLEAVE_RC=%s\n" "$rc"' EXIT
ls -ld "$(dirname "$ROOT")"
test ! -e "$ROOT/build"
test ! -e "$ROOT/environment"
test ! -e "$ROOT/cases"
mkdir -p "$ROOT/build" "$ROOT/environment"
/usr/local/cuda/bin/nvcc --version > "$ROOT/environment/nvcc.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$ROOT/environment/gpu.csv"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a \
  -lineinfo -Xptxas=-v "$HERE/interleave.cu" -o "$ROOT/build/tmem_interleave" > "$ROOT/build/build.log" 2>&1
/usr/local/cuda/bin/cuobjdump --dump-sass "$ROOT/build/tmem_interleave" > "$ROOT/build/tmem_interleave.sass"
python3 "$HERE/check_interleave.py" "$ROOT/build/tmem_interleave.sass" "$ROOT/build/check.json"
CELLS=$(python3 - "$ROOT/build/check.json" <<'PY'
import json
import sys
for k in json.load(open(sys.argv[1]))["kernels"]:
    print("c%d.r%d.cg%d.as%d.n%d%s" % (k["copies"], k["mma"], k["cta_group"], k["a_source"], k["n"],
                                       ".ind" if k["independent"] else ""))
PY
)
for run in a b; do
  seed=0
  if [ "$run" = b ]; then seed=64; fi
  for cell in $CELLS; do
    out="$ROOT/cases/$cell/run-$run"
    mkdir -p "$out"
    test ! -e "$out/tmem_interleave.jsonl"
    timeout 300 env -u CUDA_INJECTION64_PATH "$ROOT/build/tmem_interleave" \
      --only "=$cell" --out "$out" --launches 16 --reps 129 --seed-base "$seed" \
      > "$out/run.log" 2>&1 || printf '%s run-%s rc=%s\n' "$cell" "$run" "$?" >> "$ROOT/cases/failures.txt"
  done
done
python3 - "$ROOT" $CELLS <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cells = sys.argv[2:]
assert len(cells) == 19, cells
invalid = []
for run, seed in (("a", 0), ("b", 64)):
    for cell in cells:
        path = root / "cases" / cell / f"run-{run}" / "tmem_interleave.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 1 or "error" in rows[0]:
            invalid.append((cell, run, rows[0].get("error") if rows else "no row"))
            continue
        row = rows[0]
        assert row["variant"] == cell + ".w0", row["variant"]
        assert row["n"] == len(row["samples"]) == 2048, (cell, run)
        assert len(row["admission_samples"]) == 2048
        assert row["params"]["seed_base"] == seed
for item in invalid:
    print("INVALID", *item)
print(f"collected: {2 * len(cells) - len(invalid)} of {2 * len(cells)} cell collections valid")
raise SystemExit(1 if invalid else 0)
PY
