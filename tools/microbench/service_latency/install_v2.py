#!/usr/bin/env python3
"""Install a service_latency v2 model.json into a device's instruction.service leaves.

    install_v2.py --device h100|b300 --model <model.json> --run <run tag>
        --source-root configs/mechanism_tree [--dry-run]

Rules (protocol-v2.json `acceptance` / `install`):
  bucket classes       -> result 1 / completion 1 / issue_interval 0,
                          source structural.service_owner (leaves marked
                          unsupported on this device are left alone)
  measured classes     -> result_latency and completion (= result) from
                          the cell; issue_interval when derived.  h100:
                          every passing cell replaces the leaf.  b300: an
                          installed value within 10 % is kept and the run
                          is cited as reproduction; beyond 10 % the cell
                          value replaces it.  Evidence per cell verdict
                          (measured / experimental), written as the source
                          key service_latency.<device>.<run>[.experimental].
Rounded to 3 decimals as the version-1 install did.
"""
import argparse
import json
from pathlib import Path

CONTROL = 0.10
FIELDS = {"result_latency": "result_latency_cycles",
          "completion_latency": "completion_latency_cycles",
          "issue_interval": "issue_interval_cycles"}


def leaf_key(cls):
    return f"instruction.service.root/predicated_on=1/operation_class={cls}"


def set_field(leaf, field, value, source):
    f = leaf.setdefault("fields", {})
    f[field] = {"model": {"kind": "constant", "value": value}, "sources": [source]}


def cite(leaf, field, source):
    s = leaf["fields"][field]["sources"]
    if source not in s:
        s.append(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["h100", "b300"])
    ap.add_argument("--model", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--source-root", default="configs/mechanism_tree")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="comma list: install only these classes (no buckets)")
    a = ap.parse_args()
    arch = "hopper" if a.device == "h100" else "blackwell"
    root = Path(a.source_root) / arch
    leaves_p = root / "leaves" / "instruction.service.json"
    sources_p = root / "sources.json"
    leaves = json.loads(leaves_p.read_text())
    sources = json.loads(sources_p.read_text())
    model = json.loads(Path(a.model).read_text())
    key_m = f"service_latency.{a.device}.{a.run}"
    key_e = key_m + ".experimental"
    base = (f"{a.device.upper()} service_latency protocol v2 run {a.run} "
            f"(tools/microbench/service_latency/protocol-v2.json, cells-v2.json; probes alu / alu2 "
            f"(blackwell-prototype verbatim) + misc.cu + svc.cu k_fence + alu3.cu, two runs x 8 launches x 33 "
            f"reps; SASS gates blackwell-prototype sass_check.py (alu / alu2 / misc / svc) and sass_check_alu3.py). "
            f"result_latency = OLS slope over U of the dependent-chain medians (alu2 pairs minus the partner), "
            f"completion = result; issue_interval = thr slope / (8 chains x wps 1) per SMSP; misc single shots: "
            f"mean(t1 - t0) - window stall sum - t0 stall + the instruction's own stall; k_fence: result = slope "
            f"over K of the one-warp medians (issue -> next dependent issue), issue_interval = slope(4 warps on "
            f"one SMSP) / 4 when that exceeds one warp's slope by > 5 %, else 0 (the /4 value is the unresolved "
            f"upper bound). Model {a.model}.")
    sources[key_m] = {"artifact": a.model, "evidence": "measured",
                      "description": base + " Cells with fit residual <= 5 % and runs within 5 %."}
    sources[key_e] = {"artifact": a.model, "evidence": "experimental",
                      "description": base + " Cells beyond the residual / agreement bounds, alu3 thr (wps 4) "
                      "and the k_fence issue_interval rule (fixed after the data)."}
    log = []
    only = {c for c in a.only.split(",") if c}
    for cls, b in ({} if only else model["buckets"]).items():
        leaf = leaves.get(leaf_key(cls))
        if leaf is None or leaf.get("unsupported") or "children" in leaf:
            log.append(f"bucket {cls}: leaf {'absent' if leaf is None else 'unsupported' if leaf.get('unsupported') else 'partitioned (per-shape leaves kept)'} -> skipped")
            continue
        for fld, val in b["install"].items():
            set_field(leaf, FIELDS[fld], val, "structural.service_owner")
        log.append(f"bucket {cls} ({b['bucket']}): 1 / 1 / 0")
    for cls, c in model["classes"].items():
        if only and cls not in only:
            continue
        leaf = leaves.get(leaf_key(cls))
        if leaf is None or leaf.get("unsupported") or "children" in leaf:
            log.append(f"class {cls}: leaf {'absent' if leaf is None else 'unsupported' if leaf.get('unsupported') else 'partitioned'} -> skipped")
            continue
        inst = c.get("installed") or {}
        for fld in ("result_latency", "issue_interval"):
            if fld not in c or c[fld] is None:
                continue
            val = round(float(c[fld]), 3)
            src = key_m if c["verdict"].get(fld) == "measured" else key_e
            ref = inst.get(fld)
            keep = (a.device == "b300" and ref is not None and
                    ((ref and abs(val - ref) / abs(ref) <= CONTROL) or (not ref and val == 0)))
            targets = [FIELDS[fld]] + (["completion_latency_cycles"] if fld == "result_latency" else [])
            for t in targets:
                if keep and t in leaf.get("fields", {}):
                    cite(leaf, t, src)
                else:
                    set_field(leaf, t, val, src)
            log.append(f"class {cls} {fld}: {'kept ' + str(ref) + ' (cite)' if keep else f'{ref} -> {val}'}"
                       f" [{c['verdict'].get(fld)}]")
    for line in log:
        print(line)
    if a.dry_run:
        return
    leaves_p.write_text(json.dumps(leaves, indent=1, ensure_ascii=False) + "\n")
    # tree build rejects declared sources that no leaf cites any more:
    # drop the predecessor records the replaced fields used to cite
    used = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "sources" and isinstance(v, list):
                    used.update(x for x in v if isinstance(x, str))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for f in list((root / "leaves").glob("*.json")) + list((root / "mechanisms").glob("*.json")) \
            + list((root.parent / "common").rglob("*.json")):
        walk(json.loads(f.read_text()))
    dropped = [k for k in sources if k not in used]
    for k in dropped:
        del sources[k]
    print(f"dropped {len(dropped)} unreferenced sources")
    sources_p.write_text(json.dumps(sources, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
