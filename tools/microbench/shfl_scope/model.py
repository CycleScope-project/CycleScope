#!/usr/bin/env python3
"""shfl_scope v1 decision.  model.py --root <run dir> --out model.json
Variants: <kind>.n<N> (stream), chain.<kind>.n1 ; one row per (variant, launch)."""
import argparse, json, statistics
from pathlib import Path
def load(d):
    rows=[]
    for jl in Path(d).glob("*.jsonl"):
        for l in jl.read_text().splitlines():
            try: r=json.loads(l)
            except json.JSONDecodeError: continue
            if "samples" in r: rows.append(r)
    return rows
def T(rows, variant, u):
    per={}
    for r in rows:
        if r["variant"]==variant: per[int(r["params"].get("launch",0))]=r["samples"]
    meds=[statistics.median(s)/u for l,s in sorted(per.items()) if l and s]
    return statistics.median(meds) if meds else None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    out={"protocol":"shfl_scope v1","runs":{}}
    for run in ("run-a","run-b"):
        rows=load(Path(a.root)/run)
        if not rows: continue
        d={}
        for kind in ("bfly","idx"):
            for n in (1,2,4):
                d[f"{kind}.n{n}"]=T(rows,f"{kind}.n{n}",128)
            d[f"chain.{kind}.n1"]=T(rows,f"chain.{kind}.n1",128)
        out["runs"][run]=d
    dec={}
    for kind in ("bfly","idx"):
        r=[out["runs"][x] for x in out["runs"] if out["runs"][x].get(f"{kind}.n1")]
        if not r: continue
        t1=statistics.mean(x[f"{kind}.n1"] for x in r); t2=statistics.mean(x[f"{kind}.n2"] for x in r); t4=statistics.mean(x[f"{kind}.n4"] for x in r)
        agree=all(abs(x[k]-y[k])<=0.03*max(x[k],y[k]) for k in (f"{kind}.n1",f"{kind}.n2",f"{kind}.n4") for x in r for y in r)
        verdict="S-SMSP" if (t2/t1<=1.15 and t4/t1<=1.15) else "S-SM" if t4/t1>=3.4 else "record"
        dec[kind]={"t1":t1,"t2":t2,"t4":t4,"r2":t2/t1,"r4":t4/t1,"verdict":verdict,"evidence":"measured" if agree else "experimental"}
    out["decision"]=dec
    Path(a.out).write_text(json.dumps(out,indent=1))
    print(json.dumps(out,indent=1))
if __name__=="__main__": main()
