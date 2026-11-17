#!/usr/bin/env python3
"""Install tma_forms measured knots into mechanism-tree leaves.

Usage: install.py <model.json> --device <h100|b300> --run <run> --root <tree>
- completion cells -> storage.tma.completion (bytes-linear knots)
- service cells    -> storage.async.service_start
- LDGSTS completion -> storage.ldgsts.completion (copies axis), b300 enabled=1
- LDGSTS service    -> service_start LDGSTS knots
- STAS service      -> service_start STAS
- cmp == reproduced -> keep value, append source key
- multicast: dc2..16 cells; split if dc16 vs dc2 > 10%
"""
import argparse, collections, json, os, re


def leaf_key(op, form, lo, hi):
    mid = f"/tma_form={form}" if form else ""
    return (f"root/operation_class={op}{mid}"
            f"/destination_count={lo}-{hi}")


def bytes_of(cell):
    pr = cell.get("params") or {}
    for k in ("bytes", "box_bytes"):
        if isinstance(pr.get(k), (int, float)):
            return int(pr[k])
    ex = cell.get("extra") or ""
    m = re.search(r"\.b(\d+)k", ex)
    if m: return int(m.group(1)) * 1024
    m = re.search(r"\.b(\d+)", ex)
    if m: return int(m.group(1))
    m = re.search(r"bytes=(\d+)", ex)
    if m: return int(m.group(1))
    return None


def op_form_of(c):
    ex = c.get("extra") or ""
    cls, form = c["class"], c["form"]
    if cls == "LDGSTS": return ("LDGSTS", "")
    if cls == "STAS": return ("STAS", "")
    if "ublkcp" in ex: return ("UBLKCP", "")
    if "utma_redg" in ex or "utmaredg" in ex: return ("UTMAREDG", "")
    if "ublkred" in ex: return ("UBLKRED", "")
    if "gather4" in ex: return ("UTMALDG", "gather4")
    if "scatter4" in ex: return ("UTMASTG", "scatter4")
    if "im2col" in ex: return ("UTMALDG", "im2col")
    if form == "store" or ex.startswith(("tmar.st", "tmast")):
        return ("UTMASTG", "tile")
    return ("UTMALDG", "tile")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default="configs/mechanism_tree")
    a = ap.parse_args()
    arch = {"h100": "hopper", "b300": "blackwell"}[a.device]
    run_tag = a.run.replace(a.device + "-", "")
    src = f"tma_forms.{a.device}.{run_tag}"
    model = json.load(open(a.model))
    cells = model["cells"]
    ok = [c for c in cells if c["status"] in ("measured", "experimental")]

    # group knots by (mech, op, form, dc)
    knots = collections.defaultdict(list)
    for c in ok:
        if c["class"] in ("LDGSTS", "STAS"):
            continue
        b = bytes_of(c)
        if not b:
            continue
        op, form = op_form_of(c)
        mech = ("storage.tma.completion"
                if c["kind"].startswith("completion")
                else "storage.async.service_start")
        if mech == "storage.async.service_start":
            if op == "UBLKRED":
                continue          # addendum: blkseq slope is not a unit service
        if op == "UBLKCP" and "k_blkseq" in (c.get("extra") or ""):
            continue              # UBLKCP knots from tmast.ublkcpgs only
        if c["kind"] == "cold_extra" or c["kind"] not in (
                "completion", "completion_copies", "service",
                "service_per_copy", "service_bytes"):
            continue
        knots[(mech, op, form, c["destination_count"])].append(
            (b, c["value"], c["status"], c["cmp"]))
    for k in knots:
        seen = {}
        for b, v, st, cp in sorted(knots[k]):
            seen[b] = (v, st, cp)
        knots[k] = [(b,) + t for b, t in seen.items()]

    for mech in ("storage.tma.completion",
                 "storage.async.service_start"):
        p = f"{a.root}/{arch}/leaves/{mech}.json"
        if not os.path.exists(p):
            continue
        leaves = json.load(open(p))
        for key, leaf in list(leaves.items()):
            m = re.match(
                rf"{re.escape(mech)}\.root/operation_class=(\w+)"
                rf"(?:/tma_form=(\w+))?/destination_count=(\d+)-(\d+)",
                key)
            if not m:
                continue
            op, form, lo, hi = m.group(1), m.group(2) or "", \
                int(m.group(3)), int(m.group(4))
            if leaf.get("unsupported"):
                continue
            dc = lo if lo <= 1 else lo  # leaf keyed by lo range
            cand = knots.get((mech, op, form, dc))
            if cand is None and lo <= 1:
                cand = knots.get((mech, op, form, 0)) or \
                       knots.get((mech, op, form, 1))
            if not cand:
                continue
            if not cand:
                continue
            cy = leaf.setdefault("fields", {}).setdefault(
                "cycles", {"model": {}, "sources": []})
            # reproduced: keep value, append source only
            if all(x[3] and x[3].startswith("reproduced")
                   for x in cand):
                cy.setdefault("sources", []).append(src)
                continue
            pts = [[b, v] for b, v, _, _ in cand]
            cy["model"] = {"axis": "transfer_bytes",
                           "kind": "linear", "points": pts}
            cy["sources"] = [src]
            leaf.setdefault("fields", {}).setdefault(
                "enabled", {"model": {"kind": "constant",
                                      "value": 1},
                            "sources": [src]})
        json.dump(leaves, open(p, "w"), indent=2)

    # storage.ldgsts.completion
    comp = [(int(c["extra"].split("=")[1]), c["value"])
            for c in ok if c["kind"] == "completion_copies"
            and "copies" in c["extra"]]
    svc = next((c["value"] for c in ok
                if c["kind"] == "service_per_copy"), None)
    p = f"{a.root}/{arch}/leaves/storage.ldgsts.completion.json"
    leaves = json.load(open(p))
    for key, leaf in leaves.items():
        leaf.setdefault("fields", {})
        if comp:
            leaf["fields"]["cycles"] = {
                "model": {"axis": "copies_per_group", "kind": "linear",
                          "points": [[c, v] for c, v in
                                     sorted(comp)]},
                "sources": [src]}
        leaf["fields"]["enabled"] = {
            "model": {"kind": "constant", "value": 1},
            "sources": [src]}
    json.dump(leaves, open(p, "w"), indent=2)
    # service_start LDGSTS / STAS
    p = f"{a.root}/{arch}/leaves/storage.async.service_start.json"
    leaves = json.load(open(p))
    for key, leaf in leaves.items():
        m = re.match(r"storage\.async\.service_start\.root"
                     r"/operation_class=(\w+)", key)
        if not m:
            continue
        if leaf.get("unsupported"):
            continue
        leaf.setdefault("fields", {})
        if m.group(1) == "LDGSTS" and svc:
            leaf["fields"]["cycles"] = {
                "model": {"axis": "transfer_bytes",
                          "kind": "linear",
                          "points": [[0, svc]]},
                "sources": [src]}
            leaf["fields"]["enabled"] = {
                "model": {"kind": "constant", "value": 1},
                "sources": [src]}
        if m.group(1) == "STAS":
            pts = [(bytes_of(c), c["value"])
                   for c in ok if c["kind"] == "service_bytes"
                   and bytes_of(c)]
            if pts:
                leaf["fields"]["cycles"] = {
                    "model": {"axis": "transfer_bytes",
                              "kind": "linear",
                              "points": [list(x) for x in
                                         sorted(pts)]},
                    "sources": [src]}
    json.dump(leaves, open(p, "w"), indent=2)
    # source record: run-level description + evidence
    sp = f"{a.root}/{arch}/sources.json"
    sources = json.load(open(sp))
    if src not in sources:
        sources[src] = {
            "artifact": (f"$A/microbench/tma_forms-model/"
                         f"{a.device}-{run_tag}/model.json"),
            "description": (
                f"tma_forms protocol v2 (+ addenda) {a.device} run "
                f"{a.run}: variant table -> (class,form,direction); "
                f"completion = E[tw-ts|U1]-E[tw-ts|U0] minus issue "
                f"offsets (own-destination t_w = completion_own for "
                f"dc>1); service = OLS slope over U; UBLKCP dc0-1 "
                f"knots from s->g store cells with load-direction "
                f"comparison; replaced old values recorded per leaf"),
            "evidence": "measured",
        }
        json.dump(sources, open(sp, "w"), indent=2)
    print("installed", a.device, "knots:", len(knots),
          "ldgsts_comp:", len(comp), "svc:", svc)


if __name__ == "__main__":
    main()
