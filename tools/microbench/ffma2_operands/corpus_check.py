#!/usr/bin/env python3
"""corpus_check: FFMA2 source-pair / reuse / bank statistics over the 450
blackwell captures + the 19 slowed cases."""
import glob, json, os, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa

def bswap16(h):
    return int.from_bytes(int(h, 16).to_bytes(8, "little"), "big")

REG = re.compile(r"R(\d+)")

def scan(tsv):
    c = Counter()
    for line in open(tsv):
        p = line.split("\t")
        if len(p) < 4 or not p[3].strip().startswith("FFMA2"):
            continue
        i = sc.Inst(0, p[3].rstrip().rstrip(";").strip(),
                    bswap16(p[2][:16]), bswap16(p[2][16:]))
        toks = [t.strip(",") for t in i.body.split()[1:]]
        regs = [t for t in toks if re.fullmatch(r"R\d+", t)]
        srcs = regs[1:]
        npairs = len(set(srcs))
        c[f"pairs{npairs}"] += 1
        c["total"] += 1
        if (i.ctrl >> 17) & 0xF:
            c["reuse"] += 1
        banks = [int(r[1:]) % 4 for r in srcs]
        if len(banks) - len(set(banks)):
            c["bank_share"] += 1
    return c

def main():
    A = (sys.argv[2] if len(sys.argv) > 2
         else os.environ["CYCLESCOPE_ARTIFACT_ROOT"])
    import csv
    slow = set()
    for r in csv.DictReader(open(f"{A}/merge/mb4p2/compare.tsv"),
                            delimiter="\t"):
        if r["rel"] and float(r["rel"]) > 0.03:
            slow.add(r["case"])
    # bw-only slowed cases
    slowed_bw = {c for c in slow
                 if (Path(f"{A}/merge/r0/bw-sc") / c).exists()
                 or c.startswith("BL3")}
    allc, slc = Counter(), Counter()
    for d in sorted(glob.glob(f"{A}/captures/cold4/bw-sc/*/")):
        case = Path(d).name
        import json as j
        lj = Path(d) / "launches.jsonl"
        fids = {j.loads(l)["function_id"] for l in lj.read_text().splitlines()
                if l.strip() and j.loads(l).get("traced")} if lj.is_file() else set()
        cc = Counter()
        for fid in fids:
            fp = Path(d) / "functions" / f"{fid}.tsv"
            if fp.is_file():
                cc += scan(fp)
        allc += cc
        if case in slowed_bw:
            slc += cc
    out = {"all450": dict(allc), "slowed": dict(slc),
           "slowed_cases": sorted(slowed_bw)}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print("all", dict(allc)); print("slowed", dict(slc))

if __name__ == "__main__":
    main()
