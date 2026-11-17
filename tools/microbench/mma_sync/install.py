#!/usr/bin/env python3
"""Install mma_sync protocol v1 values into the mechanism tree.

    install.py --model h100.json --model b300.json
        --source-root configs/mechanism_tree
        --run-h100 h100-r1 --run-b300 b300-r1

For each device and class whose status is measured/experimental set
result_latency_cycles = completion_latency_cycles = lat slope and
issue_interval_cycles = thr value, cite
mma_sync.<device>.<run>[.experimental], and drop the cited placeholder
keys on those three fields.  Adds the source record to
<arch>/sources.json; drops placeholder.dmma.unmeasured when nothing
cites it any more.
"""
import argparse
import json
from pathlib import Path

DEVICES = {"h100": "hopper", "b300": "blackwell"}
FIELDS = ("result_latency_cycles", "completion_latency_cycles",
          "issue_interval_cycles")
# class -> fields' current placeholder sources (per device)
PLACEHOLDERS = {
    "blackwell": ["placeholder.p0", "placeholder.dmma.unmeasured",
                  "placeholder.dmma.issue_interval_unmeasured"],
    "hopper": ["placeholder.dmma.unmeasured",
               "placeholder.dmma.issue_interval_unmeasured",
               "service_latency.h100.r1.table_collapse"],
}


def leaf_key(cls):
    return ("instruction.service.root/predicated_on=1/"
            f"operation_class={cls}")


def install_device(source_root, arch, device, run, model):
    leaves_path = Path(source_root) / arch / "leaves" / \
        "instruction.service.json"
    sources_path = Path(source_root) / arch / "sources.json"
    leaves = json.loads(leaves_path.read_text())
    sources = json.loads(sources_path.read_text())
    done = {}
    for cls, e in model["classes"].items():
        status = e["status"]
        if status not in ("measured", "experimental"):
            done[cls] = f"skipped ({status})"
            continue
        key = leaf_key(cls)
        node = leaves.get(key)
        if node is None:
            done[cls] = "skipped (no leaf)"
            continue
        suffix = "" if status == "measured" else ".experimental"
        tag = run[len(device) + 1:] if run.startswith(device + "-") else run
        src = f"mma_sync.{device}.{tag}{suffix}"
        for f in FIELDS:
            fld = node["fields"][f]
            fld["model"]["value"] = (
                e["result_latency"] if f != "issue_interval_cycles"
                else e["issue_interval"])
            fld["sources"] = [s for s in fld["sources"]
                              if s not in PLACEHOLDERS[arch]
                              and not s.startswith("mma_sync.")]
            if src not in fld["sources"]:
                fld["sources"].append(src)
        if src not in sources:
            sources[src] = {
                "artifact": (f"$A/microbench/mma_sync-model/"
                             f"{device}-{run}/model.json"),
                "description": (
                    f"mma_sync protocol v1 {device} run {run}: cell "
                    f"{e['cell']} -> {cls} ({status}); "
                    f"result_latency={e['result_latency']} "
                    f"issue_interval={e['issue_interval']} "
                    f"lat_runs={e.get('lat_runs')} "
                    f"thr_runs={e.get('thr_runs')} "
                    f"control={e.get('control_result_latency')} "
                    f"sparse_call_subroutine="
                    f"{e.get('sparse_call_subroutine')} "
                    f"gate={'PASS' if e['gate'] else 'FAIL'}"),
                "evidence": ("measured" if status == "measured"
                             else "experimental"),
            }
        done[cls] = status
    # drop placeholder.dmma.unmeasured when no leaf cites it any more
    key = "placeholder.dmma.unmeasured"
    cited = any(key in fld.get("sources", [])
                for node in leaves.values()
                for fld in node.get("fields", {}).values())
    if not cited:
        cited = any(
            key in fld.get("sources", [])
            for lf in (Path(source_root) / arch / "leaves").glob("*.json")
            if lf.name != "instruction.service.json"
            for node in json.loads(lf.read_text()).values()
            for fld in node.get("fields", {}).values())
    if not cited:
        sources.pop(key, None)
        done["sources_drop"] = key
    leaves_path.write_text(json.dumps(leaves, indent=1) + "\n")
    sources_path.write_text(json.dumps(sources, indent=1) + "\n")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--run-h100", default="h100-r1")
    ap.add_argument("--run-b300", default="b300-r1")
    a = ap.parse_args()
    for mp in a.model:
        model = json.loads(Path(mp).read_text())
        device = model["device"]
        run = {"h100": a.run_h100, "b300": a.run_b300}[device]
        arch = DEVICES[device]
        done = install_device(a.source_root, arch, device, run, model)
        print(f"{device} ({arch}, run {run}):")
        for cls, st in done.items():
            print(f"  {cls:14s} {st}")


if __name__ == "__main__":
    main()
