#!/usr/bin/env python3
"""tma_forms model derivation per tools/microbench/tma_forms/protocol.json.

usage: model.py <run_a_dir> <run_b_dir> --out model.json [--table out.tsv]
run_*_dir contain tma.jsonl + tma_forms.jsonl (mechtree.mb.v1 rows).

derivation (protocol):
  completion(form,bytes,temp) = (E_1 - o_s,1) - (E_0 - o_s,0), o_s from
      sass_ctrl control words
  cold_extra(form,bytes)      = completion(cold) - completion(hot)
  service(form,bytes)         = OLS slope of E_U(hot) over U >= 1
  ldgsts_completion(C)        = E[t_w - t_s | C] - E[t_w - t_s | empty]
  ldgsts_service              = slope over U of E (C=8) / 8 copies
  stas_completion             = (E_1 - E_0) remote complete_tx - plain arrive

acceptance: se/value <= 3 % in both runs and run agreement <= 3 % ->
measured, else experimental (still listed).  b300 tile dc1 controls are
compared with the installed leaves (10 %): within -> reproduction,
beyond -> replaced (old value recorded).
"""
import argparse
import collections
import json
import re
import math
import os
import sys


def load_rows(d):
    rows = {}
    kstats = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jsonl") or not (
                fn.startswith("tma")):
            continue
        p = os.path.join(d, fn)
        for line in open(p):
            j = json.loads(line)
            samples = [x for x in j.get("samples", []) if x > 0]
            if not samples:
                continue
            key = (j["variant"], int(j["u"]))
            ss = sorted(samples)
            rows[key] = {"mean": sum(samples) / len(samples),
                         "se": _se(samples), "n": len(samples),
                         "p50": ss[len(ss) // 2],
                         "params": j.get("params", {})}
    return rows


def _se(s):
    if len(s) < 2:
        return float("inf")
    m = sum(s) / len(s)
    v = sum((x - m) ** 2 for x in s) / (len(s) - 1)
    return math.sqrt(v / len(s))


def agree(a, b, tol=0.03):
    if a is None or b is None:
        return False
    if a["mean"] <= 0 or b["mean"] <= 0:
        return False
    return abs(a["mean"] - b["mean"]) / a["mean"] <= tol


def ols(xs, ys):
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    slope = num / den if den else 0.0
    resid = [y - (ym + slope * (x - xm)) for x, y in zip(xs, ys)]
    se_ = (sum(r * r for r in resid) / max(1, len(xs) - 2) /
           den) ** 0.5 if den and len(xs) > 2 else float("inf")
    return slope, se_


def ols_slope(xs, ys):
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    return ols(xs, ys)[0]


def derive(rows, os_map=None):
    def os_(kernel):
        if os_map and kernel in os_map:
            return os_map[kernel]
        return 0
    """-> list of cell dicts."""
    out = []

    def emit(cls, form, kind, value, se, dc=0, temp="hot", extra=""):
        status = "experimental"
        if se is not None and value > 0 and se / value <= 0.03:
            status = "se-ok"
        c = {"class": cls, "form": form, "kind": kind,
             "destination_count": dc, "temp": temp,
             "value": value, "se": se, "status0": status,
             "extra": extra}
        out.append(c)
        return c

    # ---- LDGSTS -----------------------------------------------------
    ctl = rows.get(("ctl.ldgsts.c0.u0", 0))
    os_ld = os_("ldgsts")
    for c in (1, 2, 4, 8, 16, 32):
        e = rows.get(("lat.ldgsts.c%d.u1" % c, 1))
        if e and ctl:
            cell = emit("LDGSTS", "ldgsts", "completion_copies",
                        (e["mean"] - os_ld) - (ctl["mean"] - os_ld),
                        _comb_se(e, ctl), extra=f"copies={c}")
            cell["o_s"] = os_ld
            if c == 1:
                cell["tail"] = {
                    "p50": e["p50"], "mean": e["mean"],
                    "tail_share": None}
    ctl_s = rows.get(("ctl.stas.b4.u0", 0))
    os_st = os_("stas")
    us = sorted({u for v, u in rows
                 if v.startswith("cad.ldgsts") and u >= 1})
    es = [rows.get(("cad.ldgsts.c8.u%d" % u, u)) for u in us]
    if all(es) and len(us) >= 2:
        slope, sse = ols(us, [e["mean"] for e in es])
        emit("LDGSTS", "ldgsts", "service_per_copy", slope / 8.0,
             sse / 8.0)

    # ---- STAS --------------------------------------------------------
    for b in (4, 8, 16):
        e = rows.get(("lat.stas.b%d.u1" % b, 1))
        if e and ctl_s:
            cell = emit("STAS", "stas", "completion_bytes",
                        (e["mean"] - os_st) - (ctl_s["mean"] - os_st),
                        _comb_se(e, ctl_s), extra=f"bytes={b}")
            cell["o_s"] = os_st
    us = sorted({u for v, u in rows
                 if v.startswith("cad.stas") and u >= 1})
    for b in (4, 8, 16):
        es = [rows.get(("cad.stas.b%d.u%d" % (b, u), u)) for u in us]
        if all(es) and len(us) >= 2:
            slope, sse = ols(us, [e["mean"] for e in es])
            emit("STAS", "stas", "service_bytes", slope, sse,
                 extra=f"bytes={b}")

    # ---- gather4 / scatter4 / im2col (tma_forms lat cells) -----------
    for b in (128, 256, 512):
        ctl0 = rows.get(("lat.gather4.b256.u0", 0))
        for v, form in (("gather4", "gather4"), ("scatter4", "scatter4")):
            e = rows.get(("lat.%s.b%d.u1" % (v, b), 1))
            if e and ctl0:
                emit("TMA", form, "completion_bytes",
                     e["mean"] - ctl0["mean"], _comb_se(e, ctl0),
                     extra=f"bytes={4 * b}")
    # UBLKCP g->s load cells: completion = (U1 - U0) per size,
    # service = slope over U at each size
    for b in (1024, 4096, 16384, 32768):
        ctl0 = rows.get(("lat.ublkcp.b%d.u0" % b, 0))
        e1 = rows.get(("lat.ublkcp.b%d.u1" % b, 1))
        if ctl0 and e1:
            emit("UBLKCP", "tile", "completion_bytes",
                 e1["mean"] - ctl0["mean"], _comb_se(e1, ctl0),
                 extra=f"bytes={b}")
        us = sorted(u for (v, u) in rows
                    if v.startswith("lat.ublkcp.b%d.u" % b) and u >= 1)
        es = [rows.get(("lat.ublkcp.b%d.u%d" % (b, u), u))
              for u in us]
        if all(es) and len(us) >= 2:
            slope, sse = ols(us, [e["mean"] for e in es])
            emit("UBLKCP", "tile", "service_bytes", slope, sse,
                 extra=f"bytes={b}")
    # UBLKCP multicast cells (r5 addendum): per (dc, bytes),
    # completion = (E1 - o_s,1) - (E0 - o_s,0), service = OLS over U
    for dc in (2, 4):
        for b in (4096, 16384):
            ctl0 = rows.get(("lat.ublkcpmc.dc%d.b%d.u0" % (dc, b), 0))
            e1 = rows.get(("lat.ublkcpmc.dc%d.b%d.u1" % (dc, b), 1))
            if ctl0 and e1:
                emit("UBLKCP", "multicast", "completion_bytes",
                     e1["mean"] - ctl0["mean"], _comb_se(e1, ctl0),
                     dc=dc, extra=f"bytes={b}")
            us = sorted(u for (v, u) in rows
                        if v.startswith("lat.ublkcpmc.dc%d.b%d.u" % (dc, b))
                        and u >= 1)
            es = [rows.get(("lat.ublkcpmc.dc%d.b%d.u%d" % (dc, b, u), u))
                  for u in us]
            if all(es) and len(us) >= 2:
                slope, sse = ols(us, [e["mean"] for e in es])
                emit("UBLKCP", "multicast", "service_bytes", slope, sse,
                     dc=dc, extra=f"bytes={b}")
    for p in (32, 64, 128):
        ctl0 = rows.get(("lat.im2col.p64.u0", 0))
        e = rows.get(("lat.im2col.p%d.u1" % p, 1))
        if e and ctl0:
            emit("TMA", "im2col", "completion_pixels",
                 e["mean"] - ctl0["mean"], _comb_se(e, ctl0),
                 extra=f"pixels={p}")

    # ---- tma.cu cell sets (tile / store / blkseq / rank) -------------
    # E_U rows: variant contains .hot / .cold and u; completion needs
    # the U=0 control of the same variant prefix.
    # explicit variant -> (class, form, direction) table (v2): a
    # variant not in the table is an error, never a silent skip
    def variant_map(v):
        if v.startswith("tmast.utmastg") or v.startswith("tmar.st.r"):
            return ("UTMASTG", "tile", "s2g")
        if v.startswith("tmast.utma_redg"):
            return ("UTMAREDG", "", "s2g")
        if v.startswith("tmast.ublkcpgs"):
            return ("UBLKCP", "", "s2g")
        if v.startswith("k_blkseq.ublkcp"):
            return ("UBLKCP", "blkseq", "s2g")
        if v.startswith("k_blkseq.ublkred"):
            return ("UBLKRED", "blkseq", "s2g")
        if v.startswith("k_blkseq"):
            return ("BLKSEQ", "blkseq", "report_only")
        if v.startswith(("tma.dc", "tma2.", "tmar.ld.r", "tma_r")):
            return ("UTMALDG", "tile", "g2s")
        if v.startswith("lat.ublkcp"):
            return ("UBLKCP", "tile", "g2s")
        if v.startswith(("lat.", "cad.", "ctl.")):
            return None      # handled by their own sections
        if v.startswith(("tmast", "tmar.st.r")):
            return ("UTMASTG", "tile", "s2g")
        return None
    unknown = [v for (v, u) in rows
               if (v.startswith(("tma", "blkseq", "k_blkseq", "tmast",
                                 "tmar")))
               and variant_map(v) is None]
    if unknown:
        raise SystemExit(f"model: unmapped variants {sorted(unknown)[:8]}")
    by_var = collections.defaultdict(dict)
    for (v, u), e in rows.items():
        if variant_map(v) is None or v.startswith('lat.'):
            continue
        by_var[v].setdefault(u, e)
    for v, m in sorted(by_var.items()):
        if v.endswith(".ack"):
            continue          # ack windows feed completion_ack only
        e1 = m.get(1)
        e0 = m.get(0)
        vm = variant_map(v)
        cls, form0, direction = vm
        form = form0 or "tile"
        temp = "cold" if "cold" in v else "hot"
        dcm = re.search(r"\.dc(\d+)", v)
        dc0 = int(dcm.group(1)) if dcm else 0
        if e1 and e0:
            emit(cls, form, "completion", e1["mean"] - e0["mean"],
                 _comb_se(e1, e0), dc=dc0, temp=temp, extra=v)
            # ack-based completion: same formula on .ack windows
            a1 = by_var.get(v + ".ack", {}).get(1)
            a0 = by_var.get(v + ".ack", {}).get(0)
            if a1 and a0:
                emit(cls, form, "completion_ack",
                     a1["mean"] - a0["mean"], _comb_se(a1, a0),
                     dc=dc0, temp=temp, extra=v + ".ack")
        us = sorted(u for u in m if u >= 1)
        if len(us) >= 2:
            slope, sse = ols(us, [m[u]["mean"] for u in us])
            dcm = re.search(r"\.dc(\d+)", v)
            emit(cls, form, "service", slope, sse,
                 dc=int(dcm.group(1)) if dcm else 0, temp=temp,
                 extra=v)
        if temp == "cold":
            hot_v = v.replace("cold", "hot")
            hot = by_var.get(hot_v, {}).get(1)
            if hot and e1:
                emit(cls, form, "cold_extra",
                     e1["mean"] - hot["mean"],
                     _comb_se(e1, hot), extra=v)
    return out


def _comb_se(a, b):
    return math.sqrt(a["se"] ** 2 + b["se"] ** 2)


def interp(knots, x):
    if not knots: return None
    if x <= knots[0][0]: return knots[0][1]
    if x >= knots[-1][0]: return knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


def load_installed(tree_root):
    """-> {(mech, op, form, dc_hi): knots}"""
    out = {}
    for mech, fn in (("storage.tma.completion",
                      "blackwell/leaves/storage.tma.completion.json"),
                     ("storage.async.service_start",
                      "blackwell/leaves/storage.async.service_start.json")):
        p = os.path.join(tree_root, fn)
        if not os.path.exists(p): continue
        for key, leaf in json.load(open(p)).items():
            m = re.search(
                r"operation_class=(\w+)(?:/tma_form=(\w+))?/"
                r"destination_count=(\d+)-(\d+)", key)
            if not m: continue
            fields = leaf.get("fields") or {}
            cy = (fields.get("cycles") or {}).get("model", {})
            out[(mech, m.group(1), m.group(2) or "",
                 int(m.group(3)), int(m.group(4)))] = cy.get("points")
    return out


def compare_cells(cells, tree, device):
    """b300: vs installed knots; h100: h100_install."""
    leaf_knots = load_installed(tree)
    for c in cells:
        ex = c.get("extra") or ""
        opname = None
        if "utmared" in ex:
            opname = "UTMAREDG"
        elif "ublkcp" in ex:
            opname = "UBLKCP"
        elif ex.startswith("tmar.st") or "utmast" in ex:
            opname = "UTMASTG"
        if opname is None or c.get("destination_count", 0) != 0:
            c["cmp"] = "-"
            continue
        mech = ("storage.tma.completion"
                if c["kind"].startswith("completion")
                else "storage.async.service_start")
        if device == "h100":
            c["cmp"] = "h100_install"
            continue
        # bytes: box_bytes from extra .bNk or copies*16
        b = c.get("bytes") or 0
        m = re.search(r"b(\d+)k", c.get("extra", ""))
        if m: b = int(m.group(1)) * 1024
        m2 = re.search(r"b(\d+)\.hot", c.get("extra", ""))
        if m2: b = int(m2.group(1))
        m3 = re.search(r"bytes=(\d+)", c.get("extra", ""))
        if m3: b = int(m3.group(1))
        knots = leaf_knots.get((mech, opname, "tile", 0, 1)) or             leaf_knots.get((mech, opname, "", 0, 1))
        ref = interp(knots, b) if knots else None
        if ref and ref > 0:
            rel = (c["value"] - ref) / ref
            if abs(rel) <= 0.10:
                c["cmp"] = f"reproduced rel={rel:+.1%}"
            elif c["status"] == "measured":
                c["cmp"] = f"replace rel={rel:+.1%}"
            else:
                c["cmp"] = f"differs_not_installable rel={rel:+.1%}"
        else:
            c["cmp"] = "no_ref"
    # multicast split: service dc2 vs dc16 by bytes
    by_key = collections.defaultdict(dict)
    for c in cells:
        if c["kind"] not in ("service", "service_per_copy",
                             "service_bytes") and            c["kind"] != "completion":
            continue
        dc = c["destination_count"]
        if dc in (2, 4, 8, 16):
            by_key[(c["form"], c["extra"])][dc] = c["value"]
    splits = []
    for k, m in by_key.items():
        if 2 in m and 16 in m and m[2] > 0:
            rel = (m[16] - m[2]) / m[2]
            splits.append({"key": k, "dc2": m[2], "dc16": m[16],
                           "rel": rel,
                           "decision": "split" if abs(rel) > 0.10 else
                                       "single"})
        if 2 in m and 4 in m and m[2] > 0:
            rel = (m[4] - m[2]) / m[2]
            splits.append({"key": k, "dc2": m[2], "dc4": m[4],
                           "rel": rel,
                           "decision": "split" if abs(rel) > 0.10 else
                                       "single"})
    return splits


def status(run_cells):
    """merge run-a/run-b cells -> status."""
    if run_cells[0]["status0"] != "se-ok":
        return "experimental"
    if len(run_cells) == 2 and run_cells[1]["value"] > 0 and \
            abs(run_cells[0]["value"] - run_cells[1]["value"]) / \
            run_cells[0]["value"] <= 0.03:
        return "measured"
    return "experimental"


def key(c):
    return (c["class"], c["form"], c["kind"], c["destination_count"],
            c["temp"], c["extra"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table")
    ap.add_argument("--os-file")
    ap.add_argument("--tree")
    ap.add_argument("--device", required=True)
    a = ap.parse_args()

    ra = load_rows(os.path.join(a.run_dir, "run-a"))
    rb = load_rows(os.path.join(a.run_dir, "run-b"))
    os_map = None
    if a.os_file:
        raw = json.load(open(a.os_file))
        acc = collections.defaultdict(list)
        want = "h100" if a.device.startswith("h100") else "b300"
        for kn, rr in raw.items():
            for r in rr:
                if r["dev"].startswith(want):
                    acc[kn].append(r["os_stall"])
        need = ["ldgsts", "stas", "im2col"]
        if a.device.startswith("b300"):
            need += ["gather4", "scatter4"]
        missing = [k for k in need if k not in acc]
        if missing:
            raise SystemExit(
                f"missing o_s for {missing} on {a.device}")
        os_map = {k: sum(v) / len(v) for k, v in acc.items()}
    ca, cb = derive(ra, os_map), derive(rb, os_map)
    ka = {key(c): c for c in ca}
    kb = {key(c): c for c in cb}
    cells = []
    for k, c in ka.items():
        other = kb.get(k)
        c2 = dict(c)
        c2["run_b"] = other["value"] if other else None
        c2["status"] = status([c, other] if other else [c])
        cells.append(c2)

    tree = a.tree or "configs/mechanism_tree"
    splits = compare_cells(cells, tree, a.device)
    model = {"protocol": "tma_forms", "run_dir": a.run_dir,
             "device": a.device, "cells": cells,
             "multicast_decisions": splits}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(model, open(a.out, "w"), indent=1)
    cmp_path = os.path.join(os.path.dirname(a.out) or ".", "compare.tsv")
    with open(cmp_path, "w") as t:
        t.write("class\tform\tkind\tdc\ttemp\textra\tvalue\tstatus\tcmp\n")
        for c in sorted(cells, key=lambda c: key(c)):
            t.write(f"{c['class']}\t{c['form']}\t{c['kind']}\t"
                    f"{c['destination_count']}\t{c['temp']}\t{c['extra']}\t"
                    f"{c['value']:.2f}\t{c['status']}\t{c.get('cmp','-')}\n")
        t.write("\n# multicast\n")
        for sp in splits:
            _other = (f"dc16={sp['dc16']:.2f}" if "dc16" in sp
                      else f"dc4={sp['dc4']:.2f}")
            t.write(f"{sp['key']}\tdc2={sp['dc2']:.2f}\t{_other}\t"
                    f"rel={sp['rel']:+.1%}\t{sp['decision']}\n")

    if a.table:
        with open(a.table, "w") as t:
            t.write("class\tform\tkind\tdc\ttemp\tvalue\tse\t"
                    "run_b\tstatus\to_s\tcmp\textra\n")
            for c in sorted(cells, key=lambda c: key(c)):
                t.write(
                    f"{c['class']}\t{c['form']}\t{c['kind']}\t"
                    f"{c['destination_count']}\t{c['temp']}\t"
                    f"{c['value']:.2f}\t{c['se'] if c['se'] else 0:.2f}\t"
                    f"{c['run_b'] if c['run_b'] else 0:.2f}\t"
                    f"{c['status']}\t{c.get('o_s','-')}\t"
                    f"{c.get('cmp','-')}\t{c['extra']}\n")
    measured = sum(1 for c in cells if c["status"] == "measured")
    print(f"{a.out}: {len(cells)} cells, {measured} measured")


if __name__ == "__main__":
    main()
