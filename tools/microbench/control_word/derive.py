#!/usr/bin/env python3
"""control_word v1 (protocol.json): for every fixed-latency producer P of
class X that sets no write barrier and writes register R, followed in the
same basic block by a consumer C reading R before any other write of R:

  distance(P) = stall(P) + sum(stall of every inst after P up to C)
  L_ptx(X)  = min distance over all pairs of that device
  n(X)      = number of pairs

Corpus: $A/captures/cold4/{hp-sc,bw-sc}/<case>/functions/<traced id>.tsv
(rows: idx, pc, 128-bit hex, sass, regs) plus the service_latency probe
sass dumps.  Operation classes via the arch rules operation_class().
Control word = bits 105..127 of the instruction = high 64-bit half
(second 16 hex digits of the 32-hex column) >> 41.

usage:
  PYTHONPATH=<repo>/python python3 derive.py \
      --arch hopper --cases '<glob>' [--probe-sass sass ...] \
      --tree <tree.json> --out out.json --table out.tsv
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

# instructions that break the "same basic block" rule of the protocol
BREAK_ROOTS = {"BRA", "BRX", "BRXU", "CALL", "RET", "EXIT", "BSSY",
               "BSYNC", "BREAK", "WARPSYNC", "BAR", "SYNCS", "CSYNC",
               "ARRIVES", "DEPBAR", "MBAR", "NANOSLEEP", "YIELD",
               "PREEXIT", "ENDCOLLECTIVE"}

REG = re.compile(r"[RUP]T?\d+|SR_[A-Z0-9_.]+|P[T0-9]|UR[0-9]+|U?P[T0-9]")


def _bswap16(h):
    return int.from_bytes(int(h, 16).to_bytes(8, "little"), "big")


def inst_from_row(cols):
    addr = int(cols[1], 16)
    text = cols[3].rstrip().rstrip(";").strip()
    hexw = cols[2]
    # each 64-bit half in the tsv is byte-reversed vs cuobjdump
    i = sc.Inst(addr, text, _bswap16(hexw[:16]),
                _bswap16(hexw[16:]))
    return i


def regs_of(i):
    """(dst, srcs) heuristic: first token after opcode = dest."""
    body = i.body
    body = re.sub(r"\[.*?\]", " @ ", body)
    toks = [t.rstrip(",") for t in re.split(r"[\s,]+", body)[1:]
            if t]
    out = [t for t in toks
           if re.fullmatch(r"U?R\d+|U?PT?\d+|P[T0-9]", t)]
    return out


def classes_of(rules, i):
    try:
        cs = rules.operation_classes(i)
    except Exception:
        cs = [i.root]
    return cs[0] if cs else i.root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--cases", required=True,
                    help="glob of case roots (each has launches.jsonl "
                         "and functions/<id>.tsv)")
    ap.add_argument("--probe-sass", nargs="*", default=[])
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", required=True)
    a = ap.parse_args()

    import importlib
    rules = importlib.import_module(f"cyclescope.arch.{a.arch}.rules")
    tree = json.loads(Path(a.tree).read_text())
    installed = {}
    for n in tree["nodes"]:
        f = n.get("parameter", {}).get("fields", {})
        r = f.get("result_latency_cycles", {})
        if r.get("model", {}).get("kind") == "constant":
            cls = n["id"].rsplit("operation_class=", 1)[-1]
            installed[cls] = (r["model"]["value"],
                              r.get("sources", []),
                              f.get("issue_interval_cycles", {})
                              .get("model", {}).get("value"))

    per_cls = {}          # cls -> [min_dist, n]
    seen_pairs = Counter()
    roots = sorted(glob.glob(a.cases))
    files = []
    for root in roots:
        lj = Path(root) / "launches.jsonl"
        if not lj.is_file():
            continue
        fids = {json.loads(l)["function_id"] for l in lj.read_text()
                .splitlines() if l.strip()
                and json.loads(l).get("traced")}
        for fid in fids:
            fp = Path(root) / "functions" / f"{fid}.tsv"
            if fp.is_file():
                files.append(fp)
    for fp in files:
        insts = []
        for line in fp.read_text().splitlines():
            cols = line.split("\t")
            if len(cols) >= 4 and cols[2]:
                try:
                    insts.append(inst_from_row(cols))
                except Exception:
                    pass
        global writers, consumed
        writers, consumed = [], {}
        pending = []       # (writer_idx, reg, cls)
        last_writer = {}   # reg -> index into pending
        for idx, ins in enumerate(insts):
            if ins.root in BREAK_ROOTS:
                pending = []
                last_writer = {}
                continue
            rs = regs_of(ins)
            if rs:
                dst, srcs = rs[0], rs[1:]
                for r in srcs:
                    pi = last_writer.get(r)
                    if pi is not None:
                        widx, wr, wcls = pending[pi]
                        dist = sum(x.stall
                                   for x in insts[widx:idx])
                        seen_pairs[(wcls)] += 1
                        e = per_cls.setdefault(wcls, [10**9, 0])
                        e[0] = min(e[0], dist)
                        e[1] += 1
                        # R consumed: drop writer
                        pending[pi] = None
                        last_writer.pop(r)
                if ins.wbar == 7 and dst not in ("PT", "URZ",
                                               "RZ", "P0R"):
                    # producer: overwrites any previous writer of dst
                    if dst in last_writer:
                        pending[last_writer[dst]] = None
                    pending.append((idx, dst,
                                    classes_of(rules, ins)))
                    last_writer[dst] = len(pending) - 1
                elif dst in last_writer:
                    pending[last_writer[dst]] = None
                    del last_writer[dst]
    # probe sass adds (same derivation)
    for sf in a.probe_sass:
        for name, k in sc.parse(Path(sf).read_text()).items():
            insts = k.insts
            pending = []
            last_writer = {}
            for idx, ins in enumerate(insts):
                if ins.root in BREAK_ROOTS:
                    pending = []
                    last_writer = {}
                    continue
                rs = regs_of(ins)
                if rs:
                    dst, srcs = rs[0], rs[1:]
                    for r in srcs:
                        pi = last_writer.get(r)
                        if pi is not None:
                            widx, wr, wcls = pending[pi]
                            dist = sum(x.stall
                                       for x in insts[widx:idx])
                            e = per_cls.setdefault(wcls,
                                                   [10**9, 0])
                            e[0] = min(e[0], dist)
                            e[1] += 1
                            pending[pi] = None
                            last_writer.pop(r)
                    if ins.wbar == 7 and dst not in ("PT", "URZ",
                                                     "RZ"):
                        if dst in last_writer:
                            pending[last_writer[dst]] = None
                        pending.append((idx, dst,
                                        classes_of(rules, ins)))
                        last_writer[dst] = len(pending) - 1
                    elif dst in last_writer:
                        pending[last_writer[dst]] = None
                        del last_writer[dst]
    table = []
    for cls, (lo, n) in sorted(per_cls.items()):
        inst = installed.get(cls)
        row = {"class": cls, "L_ptx": lo if lo < 10**9 else None,
               "n": n}
        if inst:
            row["installed"] = inst[0]
            row["sources"] = inst[1]
            row["match"] = (lo is not None
                            and abs(lo - inst[0]) <= 1)
        table.append(row)
    out = {"arch": a.arch, "protocol":
           "tools/microbench/control_word/protocol.json",
           "n_cases": len(roots), "n_functions": len(files),
           "classes": table}
    Path(a.out).write_text(json.dumps(out, indent=1))
    with open(a.table, "w") as f:
        f.write("class\tL_ptx\tn\tinstalled\tmatch\n")
        for r in table:
            f.write(f"{r['class']}\t{r['L_ptx']}\t{r['n']}\t"
                    f"{r.get('installed','')}\t{r.get('match','')}\n")
    print(f"{a.arch}: {len(files)} functions, {len(table)} classes")


if __name__ == "__main__":
    main()
