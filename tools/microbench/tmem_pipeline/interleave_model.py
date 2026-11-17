"""Interleaved copy/MMA contrasts against two predeclared occupancy models.

serial: every copy and every MMA beat occupy one shared frontier, so C
copies before each of R MMAs add R * C * copy_service to the chain.
operand_path: an MMA holds the path shared with copies only while its
shared-memory operands are read (bytes this SM reads / 128 B per cycle,
capped by its beat); copies issued behind a queued MMA use the rest of
that beat, so only the first group and any excess over the slack add.
"""
import json
import statistics
import sys
from pathlib import Path

COPY_SERVICE = {1: 8.372779309368337, 2: 8.259262074116027}
READ_BYTES_PER_CYCLE = 128.0
OPERAND_BYTES = 4096
GEOMETRIES = {
    # (cta_group, a_source, n): beat cycles of the installed tensor.pipe leaf
    (1, 0, 128): 64.0, (1, 1, 128): 64.0, (1, 0, 256): 128.0,
    (2, 0, 256): 128.0, (2, 0, 128): 64.0,
}


def read_cycles(group, a_source, n):
    a_bytes = 0 if a_source else OPERAND_BYTES
    b_bytes = OPERAND_BYTES * n // 128 // group
    return (a_bytes + b_bytes) / READ_BYTES_PER_CYCLE


def predictions(group, a_source, n, copies, mma):
    service = COPY_SERVICE[group] * copies
    beat = GEOMETRIES[(group, a_source, n)]
    slack = max(0.0, beat - min(beat, read_cycles(group, a_source, n)))
    return {"serial": mma * service,
            "operand_path": service + (mma - 1) * max(0.0, service - slack),
            "slack": slack}


def main():
    root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    result = {"source": str(root), "runs": {}}
    for run in ("a", "b"):
        cells = {}
        for path in sorted(root.glob(f"cases/*/run-{run}/tmem_interleave.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if "error" in row:
                    cells[row["variant"]] = {"error": row["error"], "params": row["params"]}
                    continue
                cells[row["variant"]] = {
                    "params": row["params"], "n": row["n"], "gpu_uuid": row["gpu_uuid"],
                    "completion_mean": statistics.fmean(row["samples"]),
                    "completion_median": statistics.median(row["samples"]),
                    "admission_mean": statistics.fmean(row["admission_samples"]),
                    "source": str(path)}
        contrasts = {}
        for name, cell in cells.items():
            if "error" in cell:
                continue
            p = cell["params"]
            if not p["copies"]:
                continue
            base = "c0.r%d.cg%d.as%d.n%d.w0" % (p["mma"], p["cta_group"], p["a_source"], p["n"])
            if base not in cells or "error" in cells[base]:
                continue
            predicted = predictions(p["cta_group"], p["a_source"], p["n"], p["copies"], p["mma"])
            added = cell["completion_mean"] - cells[base]["completion_mean"]
            contrasts[name] = {
                "added_completion": added,
                "added_admission": cell["admission_mean"] - cells[base]["admission_mean"],
                "added_per_mma": added / p["mma"],
                "predicted": predicted,
                "error_serial": added - predicted["serial"],
                "error_operand_path": added - predicted["operand_path"]}
        result["runs"][run] = {"cells": cells, "contrasts": contrasts}
    with output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for run, data in result["runs"].items():
        for name, c in sorted(data["contrasts"].items()):
            print(run, name, "added %.1f serial %.1f operand_path %.1f" % (
                c["added_completion"], c["predicted"]["serial"], c["predicted"]["operand_path"]))


if __name__ == "__main__":
    main()
