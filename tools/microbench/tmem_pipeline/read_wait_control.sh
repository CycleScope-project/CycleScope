#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:?}
REFERENCE=${REFERENCE:?}
trap 'rc=$?; sync; printf "TMEM_ADMISSION_CONTROL_RC=%s\n" "$rc"' EXIT
ls -ld "$(dirname "$ROOT")" "$REFERENCE/build"
test ! -e "$ROOT/environment"
test ! -e "$ROOT/run-a"
test ! -e "$ROOT/run-b"
mkdir -p "$ROOT/environment" "$ROOT/run-a" "$ROOT/run-b"
BINARY=$(python3 - "$REFERENCE/build/patch.json" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text())
assert len(record["changes"]) == 4
binary = Path(record["binary"])
assert binary.is_file()
print(binary)
PY
)
printf '%s\n' "$BINARY" > "$ROOT/environment/binary.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader > "$ROOT/environment/gpu.csv"
for run in a b; do
  seed=0
  if [ "$run" = b ]; then seed=64; fi
  timeout 300 env -u CUDA_INJECTION64_PATH "$BINARY" --only '=cp32.mma0.cg1' \
    --out "$ROOT/run-$run" --launches 16 --reps 129 --seed-base "$seed" \
    > "$ROOT/run-$run/run.log" 2>&1
done
python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for run, seed in (("a", 0), ("b", 64)):
    rows = [json.loads(line) for line in (root / f"run-{run}" / "tmem_admission.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {r["variant"] for r in rows} == {"cp32.mma0.cg1." + who for who in ("w0", "w1", "joint")}
    for row in rows:
        assert "error" not in row and row["n"] == len(row["samples"]) == 2048
        params = row["params"]
        assert (params["copies"], params["mma"], params["cta_group"], params["seed_base"]) == (32, 0, 1, seed)
        if row["variant"].endswith(".joint"):
            assert len(row["start_skew_samples"]) == 2048
        else:
            assert len(row["admission_samples"]) == 2048 and len(row["warp_slot_samples"]) == 16
print("complete: one native control cell, two seed cohorts, three observers")
PY
