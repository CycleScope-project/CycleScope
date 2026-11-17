import json
import statistics
import sys
from pathlib import Path

root, output = map(Path, sys.argv[1:3])
check = json.loads((root / "build" / "check.json").read_text())
assert check["passed"]
prefix = {(r["copies"], r["mma"], r["rotate"], r["producers"], r["cta_group"], r["shape"]): r["prefix_stalls"]
          for r in check["kernels"] if not r["split"]}
result = {"runs": {}, "source": str(root)}
for run in ("a", "b"):
    cells = {}
    for line in (root / f"run-{run}" / "tmem_pipeline.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert "error" not in row, row
        samples = row["samples"]
        assert samples and len(samples) == row["n"]
        p = row["params"]
        key = (p["copies"], p["mma"], p["rotate"], p["producers"], p["cta_group"], p["shape"])
        baseline_prefix = prefix[(0, 0, 1, p["producers"], p["cta_group"], p["shape"])]
        correction = prefix[key] - baseline_prefix if not p["split"] else None
        quantiles = statistics.quantiles(samples, n=10, method="inclusive")
        assert row["variant"] not in cells
        cells[row["variant"]] = {
            "params": p, "n": len(samples), "mean": statistics.mean(samples),
            "median": statistics.median(samples), "p10": quantiles[0], "p90": quantiles[-1],
            "prefix_correction": correction,
            "corrected_mean": statistics.mean(samples) - correction if correction is not None else None,
        }
        if "start_skew_samples" in row:
            skews = row["start_skew_samples"]
            assert len(skews) == len(samples)
            cells[row["variant"]]["start_skew_mean"] = statistics.mean(skews)
            cells[row["variant"]]["start_skew_p90"] = statistics.quantiles(
                skews, n=10, method="inclusive")[-1]
            cells[row["variant"]]["start_skew_max"] = max(skews)
    assert len(cells) == 438, len(cells)
    fits = {}
    for cta_group in (1, 2):
        for shape in range(5):
            suffix = (".cg2" if cta_group == 2 else "") + (f".shape{shape}" if shape != 4 else "")
            for producers in ((1, 2) if shape == 4 else (1,)):
                for rotate in ((0, 1) if shape == 4 else (1,)):
                    for observer in [f"w{i}" for i in range(producers)] + ["joint"]:
                        xs = [1, 2, 4, 8, 16, 32] if shape == 4 else [1, 2, 4, 8, 16]
                        ys = [cells[f"cp{u}.mma0.rot{rotate}.p{producers}{suffix}.{observer}"]["corrected_mean"] for u in xs]
                        xm, ym = statistics.mean(xs), statistics.mean(ys)
                        slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
                        intercept = ym - slope * xm
                        fits[f"cg{cta_group}.shape{shape}.rot{rotate}.p{producers}.{observer}"] = {
                            "slope": slope, "intercept": intercept,
                            "points": list(zip(xs, ys)),
                            "residuals": [y - (intercept + slope * x) for x, y in zip(xs, ys)],
                        }
    result["runs"][run] = {"cells": cells, "fits": fits}
    for cta_group in (1, 2):
        print(run, f"cg{cta_group} independent-copy one-producer", fits[f"cg{cta_group}.shape4.rot1.p1.joint"])
        print(run, f"cg{cta_group} independent-copy two-producer", fits[f"cg{cta_group}.shape4.rot1.p2.joint"])
output.write_text(json.dumps(result, indent=2) + "\n")
