#!/usr/bin/env python3
"""sass_check_alu5 — composition-exact window gate (protocol-v3 addendum).

Every cell declares its per-step composition {class: count} in
cells-v4.json.  The window must contain exactly U x composition (lat) /
8K x composition (thr, dthr) of the declared classes; only per-window
harness constants (loads/stores, barriers, branch, NOP, clock CS2R and
loop control that does not scale with U/K) may appear besides.

Classes are credited with the converter's operation_classes() from
python/cyclescope/arch/<arch>/rules.py: each SASS instruction is credited to
the first class in its class list that has an instruction.service leaf
on that device.
"""
import argparse
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import sass_ctrl as sc  # noqa: E402

_TREE = Path(__file__).resolve().parents[2].parent

# per-window harness constants: loads/stores, control, clock, loop
# control — classes that never count toward a cell's composition unless
# the cell declares them
HARNESS = {
    "LDS", "STS", "LD", "ST", "LDC", "ULDC", "LDCU", "BAR", "BRA",
    "NOP", "EXIT", "WARPSYNC", "DEPBAR", "DEPBAR.FIFO", "BSSY", "BSYNC",
    "CS2R", "CS2R.CLOCK", "CS2R.GLOBALTIMER", "CS2R.ZERO", "S2R", "S2UR",
    "IADD3", "UIADD3", "VIADD", "IMAD", "MOV", "UMOV", "LOP3",
    "ISETP", "UISETP", "SEL", "USEL", "P2R", "R2P", "ULOP3",
    "RET", "CALL", "PRET", "RPCMOV",
}

_NAME = re.compile(
    r"^_Z\d+(k_lat|k_thr|k_dthr)I(\d+)([A-Z0-9_]+)Li(\d+)E.*$")
_RE_TMPL2 = re.compile(
    r"^_Z\d+k_lat_d64ILi(\d+)E.*$|"
    r"^_Z\d+k_lat_f32ILi(\d+)E.*$|"
    r"^_Z\d+k_dlatI(\d+)([A-Z0-9_]+)Li(\d+)E.*$|"
    r"^_Z\d+k_cs2rILi(\d+)ELi(\d+)E.*$")


def demangle(name):
    m = _NAME.match(name)
    if m:
        return m.group(1), m.group(3), int(m.group(4))
    for pat, kind, cell, u in (
        (r"^_Z\d+k_lat_d64ILi(\d+)E.*$", "k_lat_d64", "DSETP_P", 1),
        (r"^_Z\d+k_lat_f32ILi(\d+)E.*$", "k_lat_f32", "FSETP_P", 1),
        (r"^_Z\d+k_dlatI(\d+)([A-Z0-9_]+)Li(\d+)E.*$", "k_dlat", None, 3),
        (r"^_Z\d+k_cs2rILi(\d+)ELi(\d+)E.*$", "k_cs2r", None, 1)):
        m = re.match(pat, name)
        if m:
            if kind == "k_dlat":
                return kind, m.group(2), int(m.group(3))
            if kind == "k_cs2r":
                return kind, ("CS2R_CLOCK" if m.group(2) == "0"
                              else "CS2R_GT"), int(m.group(1))
            return kind, cell, int(m.group(u))
    return None


def widest_window(k):
    ws = sc.clock_windows(k)
    return max(ws, key=len) if ws else None


def service_classes(tree_path):
    """operation_class values that carry an instruction.service leaf."""
    t = json.loads(Path(tree_path).read_text())
    out = set()
    pref = "instruction.service.root/predicated_on=1/operation_class="
    for n in t["nodes"]:
        i = n["id"]
        if i.startswith(pref):
            out.add(i[len(pref):].split("/")[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["hopper", "blackwell"], required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--cells", default=str(
        Path(__file__).with_name("cells-v4.json")))
    a = ap.parse_args()

    rules = importlib.import_module(
        f"cyclescope.arch.{a.arch}.rules")
    cells = json.loads(Path(a.cells).read_text())["cells"]
    svc = service_classes(a.tree)

    kernels = sc.parse(Path(a.sass).read_text())
    def cs2r_window_ok(window, kk, root):
        # K back-to-back reads; anything else fails the cell
        n = 0
        for op in window:
            if op.split(".")[0].startswith("CS2R"):
                n += 1
            else:
                return False, f"non-CS2R op {op} inside window"
        return (n == kk), f"{n} CS2R reads != K={kk}"
    out = {"schema": "mechtree.sass_check.v3", "probe": "alu5",
           "sass": a.sass, "arch": a.arch,
           "kernels": {}, "unclassified": []}
    for name, k in kernels.items():
        d = demangle(name)
        if d is None:
            out["unclassified"].append(name)
            continue
        kind, cell, u = d
        mode = {"k_lat": "lat", "k_thr": "thr", "k_dthr": "dthr",
                "k_lat_d64": "dlat", "k_lat_f32": "dlat",
                "k_cs2r": "issue", "k_dlat": "dlat"}[kind]
        variant = f"{mode}.{cell}"
        steps = u if kind in ("k_lat", "k_lat_d64", "k_lat_f32",
                              "k_cs2r") else 8 * u
        if kind in ("k_lat_d64", "k_lat_f32", "k_dlat"):
            steps = 2 * u   # x and y ops both fire per iteration
        spec = cells.get(cell) or cells.get(
            {"DSETP_P": "DSETP", "FSETP_P": "FSETP"}.get(cell, cell))
        res = {"key": f"alu5/{variant}/{u}", "op": variant, "mode": mode,
               "u": u, "cell": cell, "kernel": name,
               "credit": None, "ok": False, "why": []}
        if spec is None:
            res["why"].append(f"cell {cell} not in cells-v4.json")
            out["kernels"][res["key"]] = res
            continue
        comp = spec["composition"]
        res["composition"] = comp
        w = widest_window(k)
        if not w:
            res["why"].append("no clock window")
            out["kernels"][res["key"]] = res
            continue
        counts = Counter()
        raw = Counter()
        for i in w:
            raw[i.opcode] += 1
            cred = None
            try:
                classes = rules.operation_classes(i)
            except Exception:
                classes = [i.opcode.split(".")[0]]
            for cls in classes:
                if cls in svc:
                    cred = cls
                    break
            counts[cred or i.opcode.split(".")[0]] += 1
        res["sass_counts"] = dict(counts)
        res["raw_counts"] = dict(raw)
        # exact composition
        for cls, n in comp.items():
            got = counts.get(cls, 0)
            if got != n * steps:
                res["why"].append(
                    f"{cls} count {got} != {n}x{steps}")
        stray = {c: n for c, n in counts.items()
                 if c not in comp and c not in HARNESS}
        if stray:
            res["why"].append(
                f"non-harness instructions in the window: {stray}")
        if not res["why"]:
            res["credit"] = next(iter(comp))
        res["window_pcs"] = [hex(w[0].addr), hex(w[-1].addr)]
        res["ok"] = not res["why"]
        out["kernels"][res["key"]] = res
    # addendum: per-window constants must not scale with U/K — for each
    # (cell, mode), every class outside the composition must have
    # max-min count <= 2 across the U/K kernels, else all fail
    groups = {}
    for r in out["kernels"].values():
        groups.setdefault((r["cell"], r["mode"]), []).append(r)
    for (cell, mode), rs in groups.items():
        comp = rs[0].get("composition") or {}
        extra = {}
        for r in rs:
            for cls, n in (r.get("sass_counts") or {}).items():
                if cls not in comp:
                    extra.setdefault(cls, []).append(n)
        bad = {c: sorted(v) for c, v in extra.items()
               if v and max(v) - min(v) > 2}
        if bad:
            for r in rs:
                if "harness scales with U" not in " ".join(r["why"]):
                    r["why"].append(
                        f"harness scales with U: {bad}")
                    r["ok"] = False
                    r["credit"] = None
    n = len(out["kernels"])
    out["n_ok"] = sum(1 for r in out["kernels"].values() if r["ok"])
    out["n_fail"] = n - out["n_ok"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"sass_check alu5: {out['n_ok']}/{n} gated kernels ok")
    for r in out["kernels"].values():
        if not r["ok"]:
            print("  FAIL", r["key"], r["why"])
    return 0 if n else 2


if __name__ == "__main__":
    sys.exit(main())
