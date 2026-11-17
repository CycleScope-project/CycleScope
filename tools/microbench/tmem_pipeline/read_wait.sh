#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:?}
HARDWARE=${HARDWARE:?}
HERE=$(dirname "${BASH_SOURCE[0]}")
trap 'rc=$?; sync; printf "TMEM_ADMISSION_WAIT_RC=%s\n" "$rc"' EXIT
ls -ld "$(dirname "$ROOT")" "$HARDWARE/build"
test ! -e "$ROOT/build"
test ! -e "$ROOT/environment"
test ! -e "$ROOT/cases"
mkdir -p "$ROOT/build" "$ROOT/environment"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$ROOT/environment/gpu.csv"
/usr/local/cuda/bin/cuobjdump --version > "$ROOT/environment/cuobjdump.txt"
python3 "$HERE/patch_admission_wait.py" patch \
  "$HARDWARE/build/tmem_admission" "$HARDWARE/build/tmem_admission.sass" \
  "$HARDWARE/build/check.json" "$ROOT/build/tmem_admission" "$ROOT/build/patch.json"
/usr/local/cuda/bin/cuobjdump --dump-sass "$ROOT/build/tmem_admission" > "$ROOT/build/tmem_admission.sass"
python3 "$HERE/patch_admission_wait.py" verify "$ROOT/build/patch.json" "$ROOT/build/tmem_admission.sass"
if python3 "$HERE/patch_admission_wait.py" verify "$ROOT/build/patch.json" \
    "$HARDWARE/build/tmem_admission.sass" > "$ROOT/build/negative-verify.log" 2>&1; then
  printf "unpatched SASS was incorrectly accepted\n" >&2
  exit 1
fi
python3 "$HERE/check_admission.py" "$ROOT/build/tmem_admission.sass" "$ROOT/build/check.json"
for run in a b; do
  seed=0
  if [ "$run" = b ]; then seed=64; fi
  for cg in 1 2; do
    for copies in 8 32; do
      cell="cp${copies}.mma0.cg${cg}"
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
        for copies in (8, 32):
            cell = f"cp{copies}.mma0.cg{group}"
            path = root / "cases" / cell / f"run-{run}" / "tmem_admission.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 3
            assert {r["variant"] for r in rows} == {cell + "." + who for who in ("w0", "w1", "joint")}
            for row in rows:
                assert "error" not in row and row["n"] == len(row["samples"]) == 2048
                params = row["params"]
                assert (params["copies"], params["mma"], params["cta_group"], params["seed_base"]) == (copies, 0, group, seed)
                if row["variant"].endswith(".joint"):
                    assert len(row["start_skew_samples"]) == 2048
                else:
                    assert len(row["admission_samples"]) == 2048 and len(row["warp_slot_samples"]) == 16
print("complete: four cells, two seed cohorts, three observers per collection")
PY
