#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:?}
HARDWARE=${HARDWARE:?}
HERE=$(dirname "${BASH_SOURCE[0]}")
trap 'rc=$?; sync; printf "TMEM_OPERAND_SOURCE_RC=%s\n" "$rc"' EXIT
ls -ld "$(dirname "$ROOT")" "$HARDWARE/build"
test ! -e "$ROOT/build"
test ! -e "$ROOT/environment"
test ! -e "$ROOT/cases"
mkdir -p "$ROOT/build" "$ROOT/environment"
/usr/local/cuda/bin/nvcc --version > "$ROOT/environment/nvcc.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$ROOT/environment/gpu.csv"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a \
  -lineinfo -Xptxas=-v "$HERE/admission.cu" -o "$ROOT/build/tmem_admission" > "$ROOT/build/build.log" 2>&1
/usr/local/cuda/bin/cuobjdump --dump-sass "$ROOT/build/tmem_admission" > "$ROOT/build/tmem_admission.sass"
python3 "$HERE/check_admission.py" "$ROOT/build/tmem_admission.sass" "$ROOT/build/check.json"
python3 - "$HERE" "$HARDWARE" "$ROOT" <<'PY'
import importlib.util
import json
import sys
from itertools import zip_longest
from pathlib import Path

here, hardware, root = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("operand_source_sass", here.parent / "crossings/trywait_timeout/sass_ctrl.py")
parser = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parser
spec.loader.exec_module(parser)
before = parser.parse((hardware / "build/tmem_admission.sass").read_text())
after = parser.parse((root / "build/tmem_admission.sass").read_text())
assert len(before) == 10 and len(after) == 16
rows = []
for name, kernel in before.items():
    new_name = name.replace("EEv", "ELi0EEv", 1)
    original = [(i.addr, i.low, i.high) for i in kernel.insts]
    current = [(i.addr, i.low, i.high) for i in after[new_name].insts]
    rows.append({"before": name, "after": new_name,
                 "different_instructions": sum(a != b for a, b in zip_longest(original, current))})
with (root / "build/equivalence.json").open("x") as f:
    json.dump(rows, f, indent=2)
    f.write("\n")
assert all(row["different_instructions"] == 0 for row in rows), rows
print("all ten original SMEM-A kernels have identical instruction words")
PY
for run in a b; do
  seed=0
  if [ "$run" = b ]; then seed=64; fi
  for cg in 1 2; do
    for shape in cp0.mma4 cp32.mma0 cp32.mma4; do
      cell="$shape.cg$cg.tmem_a"
      out="$ROOT/cases/$cell/run-$run"
      mkdir -p "$out"
      test ! -e "$out/tmem_admission.jsonl"
      timeout 300 env -u CUDA_INJECTION64_PATH "$ROOT/build/tmem_admission" \
        --only "=$cell" --out "$out" --launches 16 --reps 129 --seed-base "$seed" \
        > "$out/run.log" 2>&1
    done
  done
done
python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for run, seed in (("a", 0), ("b", 64)):
    for group in (1, 2):
        for copies, mma in ((0, 4), (32, 0), (32, 4)):
            cell = f"cp{copies}.mma{mma}.cg{group}.tmem_a"
            path = root / "cases" / cell / f"run-{run}" / "tmem_admission.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 3
            assert {r["variant"] for r in rows} == {cell + "." + who for who in ("w0", "w1", "joint")}
            for row in rows:
                assert "error" not in row and row["n"] == len(row["samples"]) == 2048
                params = row["params"]
                assert (params["copies"], params["mma"], params["cta_group"], params["seed_base"], params["a_source"]) == (copies, mma, group, seed, 1)
                if row["variant"].endswith(".joint"):
                    assert len(row["start_skew_samples"]) == 2048
                else:
                    assert len(row["admission_samples"]) == 2048 and len(row["warp_slot_samples"]) == 16
print("complete: six TMEM-A cells, two seed cohorts, three observers per collection")
PY
