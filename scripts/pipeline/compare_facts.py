#!/usr/bin/env python3
"""Compare two GSIMCAP01 facts files table by table."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cyclescope.convert.facts_format import FactsReader, _SPACE_R, _AK_R

def scan(path):
    r = FactsReader(path)
    warps = list(r.warps())
    roots = Counter(); mrefs = Counter(); ukeys = Counter()
    n_lane = n_ranks = n_after = n_tm = 0
    statics = r.statics()
    for w in warps:
        for d in w.instructions:
            st = statics[d.static_index]
            roots[st.opcode.split(".", 1)[0].split()[0].upper()] += 1
            for m in d.memory:
                mrefs[(_SPACE_R.get(m.space, m.space)
                       if isinstance(m.space, int) else m.space,
                       _AK_R.get(m.access_kind, m.access_kind)
                       if isinstance(m.access_kind, int)
                       else m.access_kind)] += 1
            if d.operands is not None:
                ukeys.update(d.operands.uniform)
                n_lane += len(d.operands.lane)
                n_ranks += len(d.operands.ranks)
                n_after += d.operands.after_ready_mask is not None
                n_tm += d.operands.tensor_map is not None
    return dict(reader=r, warps=warps, statics=statics, roots=roots,
                mrefs=mrefs, ukeys=ukeys, n_lane=n_lane,
                n_ranks=n_ranks, n_after=n_after, n_tm=n_tm)

def tables(r):
    return {tag: r.f.table(tag) for tag in
            ["META","CTAS","WARP","STAT","STAX","DYNI","MREF","MREX",
             "ADDR","OPSZ","OPSX","UKVT","LANE","LVAL","CORD","DESC",
             "GSYN","GSYX","OPTK","REGL","RROL","TMAP","TMDI","MBPI"]}

def main(a_path, b_path):
    a, b = scan(a_path), scan(b_path)
    print("== table row counts (A vs B) ==")
    ta, tb = tables(a["reader"]), tables(b["reader"])
    for tag in sorted(set(ta) | set(tb)):
        print(f"  {tag:5s} {len(ta.get(tag) or b''):>10d}  {len(tb.get(tag) or b''):>10d}")
    print("== DYN opcode-root histogram (top-25 by |diff|) ==")
    for k in sorted(set(a["roots"]) | set(b["roots"]),
                    key=lambda k: -abs(a["roots"][k]-b["roots"][k]))[:25]:
        print(f"  {k:14s} {a['roots'][k]:>7d} {b['roots'][k]:>7d} "
              f"{a['roots'][k]-b['roots'][k]:+8d}")
    print("== MREF (space,access_kind) ==")
    for k in sorted(set(a["mrefs"]) | set(b["mrefs"]),
                    key=lambda k: -abs(a["mrefs"][k]-b["mrefs"][k])):
        print(f"  {k} {a['mrefs'][k]:>7d} {b['mrefs'][k]:>7d}")
    print("== uniform key histogram (top-25 |diff|) ==")
    for k in sorted(set(a["ukeys"]) | set(b["ukeys"]),
                    key=lambda k: -abs(a["ukeys"][k]-b["ukeys"][k]))[:25]:
        print(f"  {k:30s} {a['ukeys'][k]:>7d} {b['ukeys'][k]:>7d}")
    print("== lane/ranks/after/tensor_map ==")
    for k in ("n_lane","n_ranks","n_after","n_tm"):
        print(f"  {k:10s} {a[k]:>8d} {b[k]:>8d}")
    print("== CTAS ==")
    for c in a["reader"].ctas()[:8]:
        print("  A", vars(c) if hasattr(c,'__dict__') else c)
    for c in b["reader"].ctas()[:8]:
        print("  B", vars(c) if hasattr(c,'__dict__') else c)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
