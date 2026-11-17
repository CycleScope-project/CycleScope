import argparse
import csv
import json
import os
from pathlib import Path

from cyclescope.capture.launch_selection import census_row, traced_launch_key
from cyclescope.reference.ncu_runner import TIMING_COLUMNS, _rows, _run_once, _tsv_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("cases", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    names = args.cases.read_text().splitlines()
    assert names and len(set(names)) == len(names)
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop("CUDA_INJECTION64_PATH", None)
    env.pop("LD_PRELOAD", None)
    argv = [str(Path(manifest["hardware"]) / "build/tmem_pipeline"),
            "--only", ",".join("=" + name for name in names),
            "--out", str(args.output), "--launches", "1", "--reps", "17"]
    rc, records = _run_once(argv, env, args.output, args.output / "ncu_r0.csv", 1800)
    assert rc == 0, f"ncu rc={rc}"
    rows = _rows(records, 0)
    references, observations = [], {}
    for name in names:
        cell = manifest["cases"][name]
        key = traced_launch_key(args.root / "cap" / name)
        assert key == (cell["kernel"], 2), (name, key, cell["kernel"])
        pick = census_row(rows, *key)
        assert pick is not None, (name, key, rows)
        group = cell["params"]["cta_group"]
        assert pick["grid"] == pick["cluster"] == [group, 1, 1], pick
        assert pick["block"] == [128, 1, 1], pick
        cycles, hz = pick["sm__cycles_active.max"], pick["sm__cycles_elapsed.avg.per_second"]
        assert cycles and cycles > 0 and hz and hz > 0, pick
        row = _tsv_row(name, None, pick, 1, "success",
                       cycles=[cycles], clocks=[int(hz / 1000)])
        row[-1] = f"tmem_pipeline reference.py; observed ncu --cache-control all; {args.output}/ncu_r0.csv"
        references.append(row)
        observations[name] = pick
    raw = [json.loads(line) for line in (args.output / "tmem_pipeline.jsonl").read_text().splitlines()]
    expected = sum(manifest["cases"][name]["params"]["producers"] + 1 for name in names)
    assert len(raw) == expected
    for row in raw:
        assert "error" not in row and row["n"] == 16 and len(row["samples"]) == 16
    with (args.output / "reference.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(TIMING_COLUMNS)
        writer.writerows(references)
    (args.output / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")
    print(f"observed cold-cache references: {len(references)}, timer rows: {len(raw)}")


if __name__ == "__main__":
    main()
