#!/usr/bin/env python3
"""Compare re-captured cases with their previous captures: static facts.

    compare_captures.py <old-root> <new-root> [--old-alt ROOT ...]
        [--ref-tsv reference.tsv] [--old-ref-root ROOT] [--tsv OUT]

<old-root>/<case> (or the first --old-alt root holding the case) is the
previous capture, <new-root>/<case> the new one.  Per case, verdicts:

  launches     every launches.jsonl row, noise fields dropped (ctx /
               addr / stream / module / function_id); DIFF lists the
               differing keys
  traced       the traced launch (trace_check.txt "launch" index) as
               kernel / grid / block / cluster / occurrence
  functions    functions.jsonl rows by mangled name (same noise dropped)
  sass         functions/<id>.tsv byte equality per (name, instructions)
  tensor_maps  tensor_maps.jsonl with the pointer bytes of `image`
               masked ({0:8, 64:72, 88:96}: global addresses)
  trace_check  new side trace_check OK
  ref_cycles   reference cycles_median old vs new and the relative
               difference -- information only, never a verdict (old:
               --ref-tsv row or <old-ref-root>/<case>/<case>/timing.json;
               new: <new-root>/reference/<case>/<case>/timing.json)

Exit status 1 when any case has a DIFF in launches / traced / sass /
tensor_maps or a missing side.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

NOISE = {"ctx", "addr", "stream", "module", "function_id"}
PTR_RANGES = (range(0, 8), range(64, 72), range(88, 96))
VERDICT_KEYS = ("launches", "traced", "sass", "tensor_maps")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in NOISE}


def _diff_keys(a: list[dict], b: list[dict]) -> list[str]:
    keys = set()
    for x, y in zip(a, b):
        keys |= {k for k in set(x) | set(y) if x.get(k) != y.get(k)}
    return sorted(keys)


def _traced(d: Path) -> dict | None:
    """The traced launch: run.json launch_selection '<kernel>@<occ>'
    resolved to its launches.jsonl row (grid / block / cluster)."""
    rj = d / "run.json"
    if not rj.is_file():
        return None
    sel = json.loads(rj.read_text()).get("launch_selection") or ""
    if "@" not in sel:
        return None
    kernel, occ = sel.rsplit("@", 1)
    rows = [r for r in load_jsonl(d / "launches.jsonl")
            if (r.get("mangled") or r.get("function")) == kernel]
    if not occ.isdigit() or not 0 < int(occ) <= len(rows):
        return {"kernel": kernel, "occurrence": occ, "row": None}
    row = rows[int(occ) - 1]
    return {"kernel": kernel, "occurrence": int(occ), "grid": row.get("grid"),
            "block": row.get("block"), "cluster": row.get("cluster")}


def _mask(hexstr: str) -> str:
    b = bytearray(bytes.fromhex(hexstr))
    for r in PTR_RANGES:
        for i in r:
            if i < len(b):
                b[i] = 0
    return bytes(b).hex()


def _ref_cycles_new(root: Path, case: str) -> list | None:
    t = root / "reference" / case / case / "timing.json"
    if not t.is_file():
        return None
    return (json.loads(t.read_text()).get("pick") or {}).get("cycles")


def _ref_cycles_old(case: str, ref_tsv: Path | None,
                    old_ref_root: Path | None) -> float | None:
    if old_ref_root is not None:
        t = old_ref_root / case / case / "timing.json"
        if t.is_file():
            cyc = (json.loads(t.read_text()).get("pick") or {}).get("cycles")
            if cyc:
                return statistics.median(cyc)
    if ref_tsv is not None and ref_tsv.is_file():
        for line in ref_tsv.read_text().splitlines():
            cols = line.split("\t")
            if cols and cols[0] == case and len(cols) > 6:
                try:
                    return float(cols[6])
                except ValueError:
                    return None
    return None


def compare_case(o: Path, n: Path, new_root: Path, ref_tsv: Path | None,
                 old_ref_root: Path | None) -> dict:
    res = {"case": n.name}
    if not (o / "launches.jsonl").is_file():
        res["launches"] = "MISSING_OLD"
        return res
    ol = [_strip(r) for r in load_jsonl(o / "launches.jsonl")]
    nl = [_strip(r) for r in load_jsonl(n / "launches.jsonl")]
    if ol == nl:
        res["launches"] = f"SAME({len(ol)})"
    else:
        res["launches"] = (f"DIFF({len(ol)}/{len(nl)}"
                           f" keys={','.join(_diff_keys(ol, nl))})")
    ot, nt = _traced(o), _traced(n)
    if ot is None or nt is None:
        res["traced"] = f"MISSING({'old' if ot is None else 'new'})"
    elif ot == nt:
        res["traced"] = (f"SAME({ot['kernel'][:40]}@{ot['occurrence']}"
                         f" {ot['grid']}/{ot['block']}/{ot['cluster']})")
    else:
        res["traced"] = f"DIFF(old={ot} new={nt})"
    def fnmap(d: Path) -> dict:
        return {(r.get("mangled") or r.get("name")): _strip(r)
                for r in load_jsonl(d / "functions.jsonl")}
    om, nm = fnmap(o), fnmap(n)
    missing = sorted(set(om) - set(nm))
    extra = sorted(set(nm) - set(om))
    diff = sorted(k for k in om.keys() & nm.keys() if om[k] != nm[k])
    res["functions"] = (f"m={len(om)} same={len(om.keys() & nm.keys()) - len(diff)}"
                        f" diff={len(diff)} miss={len(missing)} extra={len(extra)}")
    if diff:
        res["functions"] += " keys=" + ",".join(
            _diff_keys([om[k] for k in diff], [nm[k] for k in diff]))
    if missing or extra:
        res["functions"] += (f" miss={[k[:50] for k in missing[:3]]}"
                             f" extra={[k[:50] for k in extra[:3]]}")
    def tmap(d: Path) -> dict:
        m = {}
        for r in load_jsonl(d / "functions.jsonl"):
            fid = r.get("function_id")
            sp = d / "functions" / f"{fid}.tsv"
            if fid is not None and sp.exists():
                m[(r.get("mangled") or r.get("name"),
                   r.get("instructions"))] = sp.read_bytes()
        return m
    osa, nsa = tmap(o), tmap(n)
    tdiff = [k for k in osa if k in nsa and osa[k] != nsa[k]]
    smiss = len(set(osa) - set(nsa))
    same = len(osa) - len(tdiff) - smiss
    if tdiff or smiss or not osa:
        res["sass"] = (f"DIFF({same}/{len(osa)} same diff={len(tdiff)}"
                       f" miss={smiss}"
                       + (f" first={tdiff[0][0][:40]}" if tdiff else "") + ")")
    else:
        res["sass"] = f"SAME({same}/{len(osa)})"
    try:
        oms = load_jsonl(o / "tensor_maps.jsonl")
        nms = load_jsonl(n / "tensor_maps.jsonl")
        offs = set()
        for a, b in zip(oms, nms):
            ia, ib = bytes.fromhex(_mask(a["image"])), bytes.fromhex(_mask(b["image"]))
            offs |= {i for i, (x, y) in enumerate(zip(ia, ib)) if x != y}
            offs |= {k for k in set(a) | set(b)
                     if k != "image" and a.get(k) != b.get(k)}
        if len(oms) == len(nms) and not offs:
            res["tensor_maps"] = f"SAME({len(oms)})"
        else:
            # byte offsets < 64 are decoded descriptor fields (convert/
            # tensor_map.py: control, strides, dims, box); bytes 64..127
            # are not decoded.  Before calling a field diff a descriptor
            # change, compare with the offsets that differ between two
            # runs of the previous capture (run-to-run shapes).
            fields = sorted(i for i in offs if isinstance(i, int) and i < 64)
            tail = sorted(i for i in offs if isinstance(i, int) and i >= 64)
            keys = sorted(k for k in offs if isinstance(k, str))
            res["tensor_maps"] = (f"DIFF({len(oms)}/{len(nms)} fields={fields}"
                                  f" tail={len(tail)}B keys={keys})")
    except (OSError, KeyError, ValueError):
        res["tensor_maps"] = "SAME(none)" if not (o / "tensor_maps.jsonl").exists() \
            and not (n / "tensor_maps.jsonl").exists() else "DIFF(unreadable)"
    tc = n / "trace_check.txt"
    res["trace_check"] = ("OK" if tc.exists() and "OK" in tc.read_text()
                          else "FAIL")
    rj = n / "run.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        res["source_sha"] = (r.get("source_sha") or "")[:9] or "none"
    newc = _ref_cycles_new(new_root, n.name)
    oldc = _ref_cycles_old(n.name, ref_tsv, old_ref_root)
    if newc and oldc:
        med = statistics.median(newc)
        res["ref_cycles"] = f"old={oldc:.0f} new={med:.0f} rel={(med - oldc) / oldc:+.3%}"
    else:
        res["ref_cycles"] = f"old={'?' if oldc is None else int(oldc)} new={'?' if not newc else int(statistics.median(newc))}"
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_root", type=Path)
    ap.add_argument("new_root", type=Path)
    ap.add_argument("--old-alt", type=Path, action="append", default=[])
    ap.add_argument("--ref-tsv", type=Path, default=None)
    ap.add_argument("--old-ref-root", type=Path, default=None)
    ap.add_argument("--tsv", type=Path, default=None)
    args = ap.parse_args()
    cols = ["case", "launches", "traced", "functions", "sass", "tensor_maps",
            "trace_check", "source_sha", "ref_cycles"]
    rows, bad = [], 0
    for d in sorted(args.new_root.iterdir()):
        if not d.is_dir() or d.name == "reference" or \
                not (d / "run.json").exists():
            continue
        old = next((r / d.name for r in [args.old_root, *args.old_alt]
                    if (r / d.name / "launches.jsonl").is_file()),
                   args.old_root / d.name)
        r = compare_case(old, d, args.new_root, args.ref_tsv,
                         args.old_ref_root)
        rows.append(r)
        if any(not str(r.get(k, "")).startswith("SAME") for k in VERDICT_KEYS):
            bad += 1
        print(r["case"], "|", " ".join(f"{k}={r.get(k, '')}" for k in cols[1:]))
    if args.tsv:
        with args.tsv.open("w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(k, "")) for k in cols) + "\n")
    print(f"# {len(rows)} cases, {bad} with a DIFF/MISSING verdict",
          file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
