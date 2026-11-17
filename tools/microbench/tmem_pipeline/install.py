import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
calibration = json.loads(Path(sys.argv[1]).read_text())
assert calibration["schema"] == "cyclescope.tmem-copy-calibration.v1"
base = root / "configs/mechanism_tree/blackwell"
sources = json.loads((base / "sources.json").read_text())
mechanism_path = base / "mechanisms/blackwell.tmem.copy.json"
mechanism = json.loads(mechanism_path.read_text())
forms = ["copy_default", "4dp256bit", "128dp128bit",
         "2x64dp128bit_lw02_lw13", "4x32dp128bit"]
for child in mechanism["root"]["children"]:
    group = str(child["guard"]["value"])
    values = calibration["values"][group]
    assert list(values) == forms and all(v > 0 for v in values.values())
    source = f"tmem_pipeline.b300.r5.cg{group}"
    sources.pop(f"mechtree.w1.tmem.copy.cta_group.{group}", None)
    sources[source] = {
        "artifact": calibration["artifact"] + "/calibration.json",
        "description": (
            f"B300 elected tcgen05.cp cta_group::{group}, all five native forms; "
            "copy service is the predeclared OLS slope over independent-copy counts, "
            "not the predecessor's elected-loop-contaminated latency. Two runs, "
            "16 launches x 128 retained repetitions, exact value and SASS checks passed. "
            "Shape 32x128b.warpx4 two-producer and disjoint copy/MMA cells establish "
            "SM-shared service. Single-copy minus service is within five cycles "
            "of zero for every form; no additional completion tail is installed. "
            "The engine reserves this full copy service on the existing SM tensor "
            "pipe and tracks its completion in tcgen05.commit. "
            f"Protocol {calibration['protocol']}; hardware {calibration['identity']}."
        ),
        "evidence": "measured",
    }
    child["parameter"] = {
        "fields": {
            "cycles": {"model": {"kind": "discrete", "axes": ["tmem_shape"],
                                 "fail_closed": True,
                                 "points": [[name, values[name]] for name in forms]},
                       "sources": [source]},
            "enabled": {"model": {"kind": "constant", "value": 1},
                        "sources": [source]},
        },
        "scope": "sm",
    }
mechanism["doc"] = (
    "Complete per-SM tensor-pipe service of tcgen05.cp, not an extra completion "
    "tail. The Blackwell module reserves this duration on the pipe shared with "
    "MMA and includes copy completion in UTCBAR's tracked operations. Distinct "
    "native SASS forms are facts of tmem_shape; the bare UTCCP.T.S encoding is "
    "copy_default (compiler mapping: PTX 128x256b), never a missing-shape fallback. "
    "Evidence and the failed predecessor control are in tmem_pipeline/protocol.json."
)
axes_path = base / "axes.json"
axes = json.loads(axes_path.read_text())
for form in forms:
    if form not in axes["tmem_shape"]["values"]:
        axes["tmem_shape"]["values"].append(form)
for path, data in ((base / "sources.json", sources), (mechanism_path, mechanism),
                   (axes_path, axes)):
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
