import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "tools" / "microbench" / "mma_sync"))
import sass_check  # noqa: E402
import importlib.util  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] /
        "tools" / "microbench" / "mma_sync" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


model = _load("model")


def _sass(kernels):
    """Synthetic cuobjdump text: {mangled_name: [inst texts]}."""
    parts = []
    for name, insts in kernels.items():
        parts.append(f"\t\tFunction : {name}")
        for i, text in enumerate(insts):
            parts.append(
                f"\t/*{i*16:04x}*/ {text} ;\t/* 0x{i:016x} */")
        parts.append("")
    return "\n".join(parts)


CLOCK = "CS2R R0, SR_CLOCKLO ;"
MMA = "HMMA.16816.F32 R0, R1, R2, R0 ;"


def _lat_kernel(n_mma, u=4):
    body = [CLOCK] + [MMA] * n_mma + [CLOCK] + ["EXIT ;"]
    return {"_Z5k_latI8HMMA_F16Li%dEEvPmmj" % u: body}


def test_sass_check_pass(tmp_path):
    sass = tmp_path / "m.sass"
    sass.write_text(_sass(_lat_kernel(4)))
    out = tmp_path / "c.json"
    rc = sass_check.main() if False else None
    argv = ["--sass", str(sass), "--out", str(out)]
    import argparse
    sys.argv = ["sass_check.py"] + argv
    assert sass_check.main() == 0
    r = json.loads(out.read_text())
    k = r["kernels"]["mma/lat.HMMA_F16/4"]
    assert k["ok"] and k["n_hits"] == 4


def test_sass_check_wrong_count(tmp_path):
    sass = tmp_path / "m.sass"
    sass.write_text(_sass(_lat_kernel(3)))
    out = tmp_path / "c.json"
    sys.argv = ["sass_check.py", "--sass", str(sass), "--out", str(out)]
    assert sass_check.main() == 0
    r = json.loads(out.read_text())
    assert not r["kernels"]["mma/lat.HMMA_F16/4"]["ok"]


def test_sass_check_sparse_shape(tmp_path):
    # no .SP printed: gate on the sparse-only 16832 HMMA shape
    sp = "HMMA.16832.F32 R0, R1, R2, R0 ;"
    kern = {"_Z5k_latI7HMMA_SPLi4EEvPmmj": [CLOCK] + [sp] * 4 + [CLOCK]}
    sass = tmp_path / "m.sass"
    sass.write_text(_sass(kern))
    out = tmp_path / "c.json"
    sys.argv = ["sass_check.py", "--sass", str(sass), "--out", str(out)]
    assert sass_check.main() == 0
    r = json.loads(out.read_text())["kernels"]["mma/lat.HMMA_SP/4"]
    assert r["ok"] and "shape 16832" in r["sparse_gate"]


def _jsonl_cell(cell, u_vals):
    return [{"params": {"mode": m, "op": cell, "wps": w},
             "u": u, "samples": [float(v)]}
            for m, us, w in (("lat", u_vals, 0), ("thr", (4, 8, 16), 1))
            for u, v in us]


def _write_cell_run(d, cell, lat_base, lat_slope, thr_slope):
    d.mkdir(parents=True, exist_ok=True)
    cells = cell if isinstance(cell, list) else [cell]
    lines = []
    for c in cells:
        for u in (4, 8, 16, 32):
            lines.append(json.dumps(
                {"params": {"mode": "lat", "op": c, "wps": 0},
                 "u": u, "samples": [lat_base + lat_slope * u] * 4}))
        for k in (4, 8, 16):
            lines.append(json.dumps(
                {"params": {"mode": "thr", "op": c, "wps": 1},
                 "u": k, "samples": [thr_slope * k] * 4}))
    (d / "mma.jsonl").write_text("\n".join(lines))


def _gate(root, ok=True):
    ks = {}
    for cell in model.CELLS:
        ks[f"mma/lat.{cell}/4"] = {"ok": ok, "cell": cell}
    (root / "build" / "sass").mkdir(parents=True, exist_ok=True)
    (root / "build" / "sass" / "checks.json").write_text(
        json.dumps({"kernels": ks}))


def test_model_slope_and_verdict(tmp_path, monkeypatch):
    root = tmp_path / "run"
    for run, slope in (("run-a", 30.0), ("run-b", 30.1)):
        _write_cell_run(root / run, model.CELLS, 100.0, slope, 8.0 * slope)
    _gate(root)
    out = tmp_path / "model.json"
    monkeypatch.setattr(
        "sys.argv",
        ["model.py", "--device", "h100", "--root", str(root),
         "--out", str(out)])
    model.main()
    classes = json.loads(out.read_text())["classes"]
    h = classes["HMMA"]
    assert h["status"] == "measured"
    assert abs(h["result_latency"] - 30.05) < 0.01
    assert abs(h["issue_interval"] - 30.05) < 0.01


def test_model_gate_fail_unrealised(tmp_path, monkeypatch):
    root = tmp_path / "run"
    for run in ("run-a", "run-b"):
        _write_cell_run(root / run, model.CELLS, 100.0, 30.0, 240.0)
    _gate(root, ok=False)
    out = tmp_path / "model.json"
    monkeypatch.setattr(
        "sys.argv",
        ["model.py", "--device", "h100", "--root", str(root),
         "--out", str(out)])
    model.main()
    classes = json.loads(out.read_text())["classes"]
    assert classes["DMMA"]["status"] == "unrealised"


def test_install(tmp_path):
    inst = _load("install")
    src_root = Path(__file__).resolve().parents[2] / \
        "configs" / "mechanism_tree"
    work = tmp_path / "mt"
    import shutil
    shutil.copytree(src_root, work)
    m = {"device": "h100",
         "classes": {
             "DMMA": {"status": "measured", "cell": "dmma",
                      "result_latency": 12.0,
                      "issue_interval": 0.5,
                      "lat_runs": [12.0, 12.0],
                      "thr_runs": [4.0, 4.0],
                      "gate": True},
             "HMMA": {"status": "experimental", "cell": "hmma_f16",
                      "result_latency": 6.0,
                      "issue_interval": 1.0,
                      "lat_runs": [6.0, 6.5],
                      "thr_runs": [8.0, 8.0],
                      "gate": True}}}
    inst.install_device(work, "hopper", "h100", "h100-r1", m)
    leaves = json.loads((work / "hopper" / "leaves" /
                         "instruction.service.json").read_text())
    n = leaves["instruction.service.root/predicated_on=1/"
               "operation_class=DMMA"]
    assert n["fields"]["result_latency_cycles"]["model"]["value"] == 12.0
    assert n["fields"]["issue_interval_cycles"]["sources"] == \
        ["mma_sync.h100.r1"]
    hm = leaves["instruction.service.root/predicated_on=1/"
                "operation_class=HMMA"]
    assert "mma_sync.h100.r1.experimental" in \
        hm["fields"]["result_latency_cycles"]["sources"]
    src = json.loads((work / "hopper" / "sources.json").read_text())
    assert "mma_sync.h100.r1" in src
    assert "placeholder.dmma.unmeasured" not in src
