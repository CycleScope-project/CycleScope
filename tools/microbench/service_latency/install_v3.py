#!/usr/bin/env python3
"""Install service_latency protocol v3 credited values.

    install_v3.py --model h100.json --model b300.json
        --source-root configs/mechanism_tree
        --run-h100 h100-v3r1 --run-b300 b300-v3r1

Per credited class with status measured/experimental: replace
placeholder / predecessor-hash / table_collapse sources on
result/completion/issue_interval with service_latency.<dev>.<tag>
[.experimental]; write the source record into <arch>/sources.json;
drop sources unreferenced anywhere in the mechanism tree.
"""
import argparse
import glob
import json
from pathlib import Path

DEVICES = {"h100": "hopper", "b300": "blackwell"}
FIELDS = ("result_latency_cycles", "completion_latency_cycles",
          "issue_interval_cycles")
DROP_PAT = ("placeholder", "hopper.", "service_latency.h100.r1.table_collapse")


def droppable(s, evidence):
    return s == "placeholder.p0" or s.startswith("hopper.placeholder.") or \
        s == "service_latency.h100.r1.table_collapse"


def leaf_key(cls):
    return ("instruction.service.root/predicated_on=1/"
            f"operation_class={cls}")


def install_device(source_root, arch, device, tag, model,
                   revert_tree=None):
    leaves_path = (Path(source_root) / arch / "leaves" /
                   "instruction.service.json")
    sources_path = Path(source_root) / arch / "sources.json"
    leaves = json.loads(leaves_path.read_text())
    sources = json.loads(sources_path.read_text())
    done = {}
    lines = []
    for cls, e in model["classes"].items():
        status = e.get("status")
        if status not in ("measured", "experimental"):
            done[cls] = f"skipped ({status})"
            continue
        node = leaves.get(leaf_key(cls))
        if node is None:
            done[cls] = "skipped (no leaf)"
            continue
        suffix = "" if status == "measured" else ".experimental"
        src = f"service_latency.{device}.{tag}{suffix}"
        e.setdefault("_run_desc", {})
        vals = {"result_latency_cycles": e.get("result_latency"),
                "completion_latency_cycles": e.get("completion_latency"),
                "issue_interval_cycles": e.get("issue_interval")}
        # b300 control rule: keep the installed value when the new one
        # is within 10 %, citing the new source as reproduction
        control = device == "b300" and cls in ("DFMA", "DADD", "DMUL",
            "UIADD3", "UIMAD", "ULOP3", "USHF", "UPRMT", "UPOPC",
            "UFLO", "USEL", "UISETP")
        replaced = {}
        for f in FIELDS:
            fld = node["fields"].get(f)
            if fld is None or vals[f] is None:
                continue
            old = fld["model"].get("value")
            if (control and isinstance(old, (int, float))
                    and old not in (0, 0.0)
                    and abs(vals[f] - old) <= 0.10 * abs(old)):
                fld["sources"] = [s for s in fld["sources"]
                                  if not droppable(s, None)]
                if src not in fld["sources"]:
                    fld["sources"].append(src)
                continue
            fld["model"]["value"] = vals[f]
            if old != vals[f]:
                replaced[f] = old
            # sources list must be exactly the keys supporting the new
            # value: the new source plus any older MEASURED source
            # reproducing it within 5 %
            keep = []
            for skey in fld["sources"]:
                sev = (sources.get(skey) or {}).get("evidence")
                if sev != "measured" or not isinstance(
                        old, (int, float)):
                    continue
                if abs(old - vals[f]) <= 0.05 * abs(vals[f]):
                    keep.append(skey)
            fld["sources"] = keep + ([src] if src not in keep else [])
        if replaced:
            e["_replaced_old_values"] = replaced
        lines.append(
            f"{cls}: cell={e.get('cell')} comp={e.get('composition')} "
            f"result={e.get('result_latency')} "
            f"ii={e.get('issue_interval')} "
            f"resid={e.get('lat_resid')},{e.get('thr_resid')} "
            f"agree={e.get('lat_agree')},{e.get('thr_agree')} "
            f"partners={e.get('partner_result_latency')}"
            f"{e.get('partner_issue_interval')} "
            f"replaced={e.get('_replaced_old_values')}")
        done[cls] = status
    # run-level source descriptions: one per evidence level actually
    # cited on a field
    cited = set()
    for n in leaves.values():
        for v in n.get("fields", {}).values():
            cited.update(v.get("sources", []))
    for suf, ev in (("", "measured"), (".experimental", "experimental")):
        src = f"service_latency.{device}.{tag}{suf}"
        if src in cited and src not in sources:
            sources[src] = {
                "artifact": (f"$A/microbench/service_latency-model/"
                             f"{device}-{tag}/model.json"),
                "description": (
                    f"service_latency protocol v3 (+ addenda 1,2) "
                    f"{device} run {tag}; run dirs "
                    f"$A/microbench/service_latency/{device}-{tag}/"
                    f"run-a,run-b; classes installed from this run: "
                    + " | ".join(lines)),
                "evidence": ev,
            }
    # revert: classes whose fields cite service_latency.* but whose
    # model status is not installed get their pre-v3 value+sources back
    # from the control tree when supplied via --revert-tree
    if revert_tree is not None:
        rt = json.loads(Path(revert_tree).read_text())
        rm = {n["id"]: n for n in rt["nodes"]}
        for cls, e in model["classes"].items():
            if e.get("status") in ("measured", "experimental"):
                continue
            node = leaves.get(leaf_key(cls))
            o = rm.get(leaf_key(cls))
            if node is None or o is None:
                continue
            for f in FIELDS:
                fld = node.get("fields", {}).get(f)
                if fld and any(k.startswith("service_latency.")
                               for k in fld.get("sources", [])):
                    of = o["parameter"]["fields"].get(f)
                    if of:
                        fld["model"]["value"] = of["model"]["value"]
                        fld["sources"] = list(of["sources"])
                        done[cls] = f"reverted ({e.get('status')})"
    if arch == "hopper":
        for cls in ("DFMA", "UIADD3", "UIMAD", "ULOP3"):
            if done.get(cls) in ("measured", "experimental"):
                continue
            node = leaves.get(leaf_key(cls))
            if node is None:
                continue
            fld = node["fields"].get("issue_interval_cycles")
            if fld and "structural.service_owner" in fld["sources"]:
                fld["sources"] = [
                    "placeholder.v3.mislabelled_service_owner"
                    if s == "structural.service_owner" else s
                    for s in fld["sources"]]
                sources["placeholder.v3.mislabelled_service_owner"] = {
                    "artifact": "none",
                    "description": (
                        "v3 alu2/f64 cell did not pass; predecessor "
                        "interval kept. The structural.service_owner "
                        "source was a mislabelled measurement claim, "
                        "relabeled as placeholder"),
                    "evidence": "placeholder",
                }
                done[cls] = "relabeled (cell not accepted)"
    blob = "".join(
        Path(lf).read_text()
        if not lf.endswith("instruction.service.json")
        else json.dumps(leaves)
        for lf in glob.glob(str(Path(source_root) / arch /
                                "leaves" / "*.json")))
    for lf in glob.glob(str(Path(source_root) / arch / "mechanisms" /
                             "*.json")) + glob.glob(str(
                                 Path(source_root) / arch / "*.json")):
        if not lf.endswith("sources.json"):
            blob += Path(lf).read_text()
    gone = [k for k in list(sources) if f'"{k}"' not in blob]
    for k in gone:
        del sources[k]
    if gone:
        done["sources_drop"] = gone
    leaves_path.write_text(json.dumps(leaves, indent=1) + "\n")
    sources_path.write_text(json.dumps(sources, indent=1) + "\n")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--run-h100", default="h100-v3r1")
    ap.add_argument("--run-b300", default="b300-v3r1")
    ap.add_argument("--revert-h100", default=None)
    ap.add_argument("--revert-b300", default=None)
    a = ap.parse_args()
    for mp in a.model:
        model = json.loads(Path(mp).read_text())
        device = model["device"]
        tag = {"h100": a.run_h100, "b300": a.run_b300}[device]
        arch = DEVICES[device]
        done = install_device(a.source_root, arch, device, tag,
                              model,
                              revert_tree=getattr(a, "revert_" + device))
        print(f"{device} ({arch}, run {tag}):")
        for cls, st in done.items():
            print(f"  {cls:14s} {st}")


if __name__ == "__main__":
    main()
