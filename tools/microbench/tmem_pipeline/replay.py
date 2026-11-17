import argparse
import importlib.util
import json
import statistics
import struct
import sys
from pathlib import Path

from cyclescope.io.file_header import FileReader
from cyclescope.io.fixed_tables import TableReader
from cyclescope.job import container, segment_format as sf
from cyclescope.job.result import read_result


def name_of(p):
    return (f"cp{p['copies']}.mma{p['mma']}.rot1.p{p['producers']}"
            + (".split" if p["split"] else "")
            + (".cg2" if p["cta_group"] == 2 else "")
            + (f".shape{p['shape']}" if p["shape"] != 4 else ""))


def prepare(hardware, output):
    source = json.loads((hardware / "measurement.json").read_text())
    check = json.loads((hardware / "build/check.json").read_text())
    assert check["passed"] and len(check["kernels"]) == 180
    path = Path(__file__).parents[1] / "crossings/trywait_timeout/sass_ctrl.py"
    spec = importlib.util.spec_from_file_location("tmem_replay_sass", path)
    parser = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parser
    spec.loader.exec_module(parser)
    kernels = parser.parse((hardware / "build/tmem_pipeline.sass").read_text())
    cases = {}
    for k in check["kernels"]:
        u, m, p, shape = k["copies"], k["mma"], k["producers"], k["shape"]
        selected = k["rotate"] == 1 and (
            (shape == 4 and not k["split"] and u in (0, 8, 32) and m in (0, 4))
            or (shape == 4 and k["split"] and u in (0, 32) and m in (0, 4))
            or (shape != 4 and not k["split"] and p == 1 and u in (0, 16) and m == 0))
        if not selected:
            continue
        params = {key: k[key] for key in ("copies", "mma", "producers", "split", "cta_group", "shape")}
        name = name_of(params)
        insts = kernels[k["name"]].insts
        targets = [i for i, x in enumerate(insts) if x.root in ("UTCCP", "UTCOMMA")]
        commits = [i for i, x in enumerate(insts) if x.root == "UTCBAR"]
        assert len(commits) == 1
        first = min(targets) if targets else commits[0]
        clocks = [i for i, x in enumerate(insts) if "SR_CLOCKLO" in x.text]
        start = max(i for i in clocks if i < first)
        ends = [i for i in clocks if i > commits[0]]
        assert len(ends) == 1, (name, ends)
        observers = [f"w{i}" for i in range(p)] + ["joint"]
        cases[name] = {
            "params": params, "kernel": k["name"],
            "start_pc": insts[start].addr, "end_pc": insts[ends[0]].addr,
            "hardware": {run: {who: source["runs"][run]["cells"][f"{name}.{who}"]
                               for who in observers} for run in ("a", "b")},
        }
    assert len(cases) == 48
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"hardware": str(hardware), "repetitions": 17,
                "reference_sm_clock_khz": 2032000, "cases": cases}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "cases.txt").write_text("\n".join(sorted(cases)) + "\n")
    pilot = ["cp8.mma4.rot1.p2", "cp8.mma4.rot1.p2.cg2"]
    (output / "pilot.txt").write_text("\n".join(pilot) + "\n")
    (output / "remaining.txt").write_text("\n".join(sorted(set(cases) - set(pilot))) + "\n")
    print(f"prepared {len(cases)} cells, {len(pilot)} pilot cells")


def check_launch(path):
    launch = json.loads(path.read_text())
    for field in ("missing_facts", "unsupported"):
        assert not launch.get(field), (str(path), field, launch[field])


def measure_cell(path, cell, repetitions):
    check_launch(path / "launch.json")
    result = read_result((path / "result.bin").read_bytes())
    assert result["issued"] == result["completed"]
    assert result.get("hard_constraint_failures", 0) == 0
    images = list(container.segment_images(container.read_container((path / "job.bin").read_bytes())))
    assert len(images) == 1, len(images)
    segment = sf.read_segment(images[0])
    cta_table = TableReader(sf.CTAS, segment.require_table("CTAS"))
    ctas = [cta_table.unpack(i) for i in range(len(cta_table))]
    warps = TableReader(sf.WARP, segment.require_table("WARP"))
    instances = TableReader(sf.INST, segment.require_table("INST"))
    semantics = TableReader(sf.SEMA, segment.require_table("SEMA"))
    templates = TableReader(sf.TMPL, segment.require_table("TMPL"))
    strings = bytes(segment.require_table("STRS"))
    params = cell["params"]
    assert len(ctas) == params["cta_group"]
    assert all(c["cluster_shape_x"] == params["cta_group"] and
               c["cluster_shape_y"] == c["cluster_shape_z"] == 1 for c in ctas)
    leaders = [c["cta_index"] for c in ctas if c["cluster_rank"] == 0]
    assert len(leaders) == 1
    roles = {warps.unpack(i)["warp_in_cta"]: i for i in range(len(warps))
             if warps.unpack(i)["cta_index"] == leaders[0]}
    addresses = {i: address for i, _, address in struct.iter_unpack("<IIQ", segment.require_table("CODE"))}
    base = min(a for a in addresses.values() if a)
    classes, kinds = {}, {}
    for i in range(len(instances)):
        instance = instances.unpack(i)
        offset = templates.unpack(instance["template"])["operation_class"]
        classes[i] = strings[offset:strings.index(b"\0", offset)].decode().split(".", 1)[0]
        kinds[i] = (semantics.unpack(instance["semantics"])["kind"]
                    if instance["semantics"] != 0xFFFFFFFF else None)
    reader = FileReader((path / "journal.bin").read_bytes(), expect="journal")
    events = [row for row in struct.iter_unpack("<IIIQIIIQQ", reader.table("JRNL"))]
    samples, starts, ends, counts = {}, [], [], {}
    for p in range(params["producers"]):
        producer, detector = roles[2 * p], roles[2 * p + 1]
        begin = sorted(row[3] for row in events
                       if row[0] == producer and addresses[row[2]] - base == cell["start_pc"])
        end = sorted(row[3] for row in events
                     if row[0] == detector and addresses[row[2]] - base == cell["end_pc"])
        assert len(begin) == len(end) == repetitions, (p, len(begin), len(end), repetitions)
        assert all(b >= a for a, b in zip(begin, end)), (p, begin, end)
        copies = params["copies"] if not params["split"] or p == 0 else 0
        mma = params["mma"] if not params["split"] or p == 1 else 0
        counts[f"w{p}"] = {}
        for operation, kind, expected in (("UTCCP", 30, copies), ("UTCOMMA", 23, mma)):
            issued = [row for row in events if row[0] == producer and classes[row[2]] == operation]
            executed = sum(kinds[row[2]] == kind for row in issued)
            assert executed == expected * repetitions, (
                str(path), p, operation, executed, expected * repetitions)
            assert all(kinds[row[2]] in (kind, None) for row in issued)
            counts[f"w{p}"][operation] = {
                "executed": executed, "predicated_off": len(issued) - executed}
        samples[f"w{p}"] = [b - a for a, b in zip(begin[1:], end[1:])]
        starts.append(begin)
        ends.append(end)
    samples["joint"] = [max(x[r] for x in ends) - min(x[r] for x in starts)
                        for r in range(1, repetitions)]
    skew = [max(x[r] for x in starts) - min(x[r] for x in starts)
            for r in range(1, repetitions)]
    output = {"observers": {}, "start_skew": skew, "counts": counts,
              "kernel_cycles": result["cycles"]}
    for who, values in samples.items():
        hw = [cell["hardware"][run][who] for run in ("a", "b")]
        correction = hw[0]["prefix_correction"]
        assert correction == hw[1]["prefix_correction"]
        mean = statistics.mean(values)
        output["observers"][who] = {
            "samples": values, "mean": mean,
            "corrected_mean": mean - correction if correction is not None else None,
            "hardware_means": [r["mean"] for r in hw],
            "hardware_corrected_means": [r["corrected_mean"] for r in hw],
            "error_cycles": [mean - r["mean"] for r in hw],
        }
    return output


def measure(root, case_file, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "manifest.json").read_text())
    names = case_file.read_text().splitlines()
    pilot_path = root / "pilot-measurement.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.is_file() else {"arms": {}}
    if pilot["arms"]:
        assert pilot["manifest"] == str(root / "manifest.json")
    out = {"manifest": str(root / "manifest.json"), "arms": {}}
    for arm in ("before", "after"):
        cells = {}
        for name in names:
            if name in pilot["arms"].get(arm, {}):
                cells[name] = pilot["arms"][arm][name]
                cells[name]["frozen_source"] = str(pilot_path)
            else:
                cells[name] = measure_cell(root / arm / name / "out-cpp",
                                           manifest["cases"][name], manifest["repetitions"])
        for name, cell in cells.items():
            params = dict(manifest["cases"][name]["params"])
            params["copies"] = 0
            base_name = name_of(params)
            if base_name not in cells:
                continue
            for who, values in cell["observers"].items():
                reference = cells[base_name]["observers"][who]
                corrected = values["corrected_mean"] is not None
                key = "corrected_mean" if corrected else "mean"
                hardware_key = "hardware_corrected_means" if corrected else "hardware_means"
                values["copy_increment"] = values[key] - reference[key]
                values["hardware_copy_increment"] = [
                    a - b for a, b in zip(values[hardware_key], reference[hardware_key])]
        out["arms"][arm] = cells
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(output)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("hardware", type=Path)
    prep.add_argument("root", type=Path)
    report = sub.add_parser("measure")
    report.add_argument("root", type=Path)
    report.add_argument("cases", type=Path)
    report.add_argument("output", type=Path)
    check = sub.add_parser("check-launch")
    check.add_argument("launch", type=Path)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.hardware, args.root)
    elif args.command == "measure":
        measure(args.root, args.cases, args.output)
    else:
        check_launch(args.launch)


if __name__ == "__main__":
    main()
