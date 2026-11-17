#!/usr/bin/env python3
"""Install mma_sync protocol v2 values.

    install_v2.py --model h100.json --model b300.json
        --source-root configs/mechanism_tree
        --run-h100 h100-r8 --run-b300 b300-r8

DMMA: issue_interval_cycles only (result/completion keep their v1
sources), replacing placeholder.dmma.issue_interval_unmeasured.
BMMA: result / completion / issue_interval replacing the placeholder
sources on those fields.  BMMA.SPARSE untouched (ISA-absent).  Source
key mma_sync.<device>.<tag>[.experimental] with the quotient table and
wps4/wps1 ratio in the description.  Unreferenced sources dropped.
"""
import argparse
import glob
import json
from pathlib import Path

DEVICES = {"h100": "hopper", "b300": "blackwell"}
TARGET = {
    # class -> fields installed from the model entry
    "DMMA": ("issue_interval_cycles",),
    "BMMA": ("result_latency_cycles", "completion_latency_cycles",
             "issue_interval_cycles"),
}
PLACEHOLDERS = {"placeholder.p0", "placeholder.dmma.issue_interval_unmeasured"}


def leaf_key(cls):
    return ("instruction.service.root/predicated_on=1/"
            f"operation_class={cls}")


def install_device(source_root, arch, device, tag, model):
    leaves_path = (Path(source_root) / arch / "leaves" /
                   "instruction.service.json")
    sources_path = Path(source_root) / arch / "sources.json"
    leaves = json.loads(leaves_path.read_text())
    sources = json.loads(sources_path.read_text())
    done = {}
    for cls, fields in TARGET.items():
        e = model["classes"].get(cls) or {}
        status = e.get("status")
        if status not in ("measured", "experimental"):
            done[cls] = f"skipped ({status})"
            continue
        node = leaves.get(leaf_key(cls))
        if node is None:
            done[cls] = "skipped (no leaf)"
            continue
        suffix = "" if status == "measured" else ".experimental"
        tag = tag[len(device) + 1:] if tag.startswith(device + "-") else tag
        src = f"mma_sync.{device}.{tag}{suffix}"
        value_of = {
            "result_latency_cycles": e.get("result_latency"),
            "completion_latency_cycles": e.get("completion_latency"),
            "issue_interval_cycles": e.get("issue_interval"),
        }
        for f in fields:
            fld = node["fields"][f]
            fld["model"]["value"] = value_of[f]
            fld["sources"] = [s for s in fld["sources"]
                              if s not in PLACEHOLDERS]
            if src not in fld["sources"]:
                fld["sources"].append(src)
        if src not in sources:
            sources[src] = {
                "artifact": (f"$A/microbench/mma_sync-model/"
                             f"{device}-{tag}/model.json"),
                "description": (
                    f"mma_sync protocol v2 {device} run {tag}: cell "
                    f"{e['cell']} -> {cls} ({status}); "
                    f"result_latency={e.get('result_latency')} "
                    f"issue_interval={e.get('issue_interval')} "
                    f"quotients={e.get('quotients')} "
                    f"thr_slopes={e.get('thr_slopes')} "
                    f"wps4/wps1={e.get('wps4_wps1_ratio')} "
                    f"per_warp_blocking={e.get('per_warp_blocking')} "
                    f"resource={e.get('resource')} "
                    f"scope={e.get('scope')} "
                    f"scope_divisor={e.get('scope_divisor')} "
                    f"gate={'PASS' if e.get('gate') else 'FAIL'}"),
                "evidence": ("measured" if status == "measured"
                             else "experimental"),
            }
        done[cls] = status
    # drop sources no longer referenced by any leaf (substring scan
    # over all leaf files, with the updated instruction.service dict)
    blob = "".join(Path(lf).read_text()
                   if not lf.endswith("instruction.service.json")
                   else json.dumps(leaves)
                   for lf in glob.glob(str(Path(source_root) / arch /
                                           "leaves" / "*.json"))
                   + glob.glob(str(Path(source_root) / arch /
                                   "mechanisms" / "*.json"))
                   + [str(Path(source_root) / arch / "resources.json")])
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
    ap.add_argument("--run-h100", default="h100-r8")
    ap.add_argument("--run-b300", default="b300-r8")
    a = ap.parse_args()
    for mp in a.model:
        model = json.loads(Path(mp).read_text())
        device = model["device"]
        tag = {"h100": a.run_h100, "b300": a.run_b300}[device]
        arch = DEVICES[device]
        done = install_device(a.source_root, arch, device, tag, model)
        print(f"{device} ({arch}, run {tag}):")
        for cls, st in done.items():
            print(f"  {cls:14s} {st}")


if __name__ == "__main__":
    main()
