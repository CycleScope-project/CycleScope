"""descvis model (protocol-v1): E(D)=median tw-ta at delay D; X(D)=
E(D)-E_c; V=X(0) if X decreases slope -1+-15% on D<V and X(D)<=5%E_c
for D>=V; else no V. Acceptance: V within 10% across runs -> exp."""
import json, statistics, sys
from pathlib import Path

def load(root, run):
    out = {}
    for fp in sorted((Path(root)/run).glob("descvis*.jsonl")):
     for l in fp.read_text().splitlines():
        j = json.loads(l)
        v = j["variant"]
        s = j.get("samples") or []
        if v == "ctl.descvis":
            out["ctl"] = statistics.median(s) if s else None
        elif v.startswith("lat.descvis.d"):
            out[int(v.rsplit("d", 1)[1])] = statistics.median(s) if s else None
    return out

def model(root):
    ea, eb = load(root, "run-a"), load(root, "run-b")
    res = {}
    for tag, e in (("run-a", ea), ("run-b", eb)):
        ec = e.get("ctl")
        xs = {d: (v - ec) for d, v in e.items()
              if d != "ctl" and v is not None and ec is not None}
        ds = sorted(xs)
        # slope test over the decreasing part
        dec = [d for d in ds if xs[d] > 0.05 * (ec or 1)]
        slope_ok, v = False, None
        if len(dec) >= 2:
            x0, xN = xs[dec[0]], xs[dec[-1]]
            slope = (xN - x0) / (dec[-1] - dec[0])
            slope_ok = abs(slope + 1) <= 0.15
            v = xs[0] if slope_ok else None
        res[tag] = {"E_c": ec, "X": xs, "slope_ok": slope_ok, "V": v}
    return res

if __name__ == "__main__":
    print(json.dumps(model(sys.argv[1]), indent=1))
