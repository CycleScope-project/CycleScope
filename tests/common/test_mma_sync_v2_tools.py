import json
import shutil
import sys
from pathlib import Path

import importlib.util


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] /
        "tools" / "microbench" / "mma_sync" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "tools" / "microbench" / "mma_sync"))
model_v2 = _load("model_v2")
install_v2 = _load("install_v2")

LAT_U = (4, 8, 16, 32)
THR_K = (4, 8, 16)


def _write_run(d, cell_slopes):
    """cell_slopes: {cell: (lat_slope, {wps: thr_slope})}"""
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for cell, (lat, thr) in cell_slopes.items():
        for u in LAT_U:
            lines.append(json.dumps(
                {"params": {"mode": "lat", "op": cell, "wps": 0},
                 "u": u, "samples": [100.0 + lat * u] * 4}))
        for w, slope in thr.items():
            for k in THR_K:
                lines.append(json.dumps(
                    {"params": {"mode": "thr", "op": cell, "wps": w},
                     "u": k, "samples": [slope * k] * 4}))
    (d / "mma.jsonl").write_text("\n".join(lines))


def _gate(root, ok=True):
    ks = {}
    for cell in model_v2.CELLS:
        ks[f"mma/lat.{cell}/4"] = {"ok": ok, "cell": cell}
    (root / "build" / "sass").mkdir(parents=True, exist_ok=True)
    (root / "build" / "sass" / "checks.json").write_text(
        json.dumps({"kernels": ks}))


def _model(root, out, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["model_v2.py", "--device", "h100", "--root", str(root),
         "--out", str(out)])
    model_v2.main()
    return json.loads(out.read_text())["classes"]


def _all_cells(lat, thr_by_w):
    return {c: (lat, thr_by_w) for c in model_v2.CELLS}


def test_queuing_measured(tmp_path, monkeypatch):
    # shared-unit queuing: per-warp slope grows with w -> quotient T
    root = tmp_path / "run"
    T = 2.5
    for run in ("run-a", "run-b"):
        _write_run(root / run,
                   _all_cells(30.0, {w: 8 * w * T for w in (1, 2, 4)}))
    _gate(root)
    classes = _model(root, tmp_path / "m.json", monkeypatch)
    d = classes["DMMA"]
    assert d["status"] == "measured"
    assert not d["per_warp_blocking"]
    # v3 scope rule: DMMA -> fp64_sm is sm-scoped -> quotient / 4
    assert d["resource"] == "fp64_sm"
    assert d["scope"] == "sm"
    assert d["scope_divisor"] == 4
    assert abs(d["issue_interval"] - T / 4) < 1e-9
    assert abs(d["result_latency"] - 30.0) < 1e-9
    for w in ("1", "2", "4"):
        assert abs(d["quotients"][w] - T) < 1e-9


def test_blocking_experimental(tmp_path, monkeypatch):
    # per-warp blocking: window does not grow with w -> ratio ~1 ->
    # experimental, issue_interval = quotient at w = 4 (L/4)
    root = tmp_path / "run"
    L = 40.0
    for run in ("run-a", "run-b"):
        _write_run(root / run,
                   _all_cells(L, {w: 8 * L for w in (1, 2, 4)}))
    _gate(root)
    classes = _model(root, tmp_path / "m.json", monkeypatch)
    d = classes["DMMA"]
    assert d["status"] == "experimental"
    assert d["per_warp_blocking"]
    # quotient(w=4) = L/4; DMMA is sm-scoped -> /4 -> L/16
    assert abs(d["issue_interval"] - L / 16) < 1e-9
    assert abs(d["wps4_wps1_ratio"] - 1.0) < 1e-9


def test_bmma_sparse_isa_absent(tmp_path, monkeypatch):
    root = tmp_path / "run"
    for run in ("run-a", "run-b"):
        _write_run(root / run,
                   _all_cells(30.0, {w: 20.0 * w for w in (1, 2, 4)}))
    _gate(root)
    classes = _model(root, tmp_path / "m.json", monkeypatch)
    assert classes["BMMA.SPARSE"]["status"] == "unrealised:isa_absent"
    assert classes["BMMA"]["status"] == "measured"


def test_bmma_emulated(tmp_path, monkeypatch):
    # b300-style gate failure: no BMMA opcode at all, CALL emulation
    root = tmp_path / "run"
    for run in ("run-a", "run-b"):
        _write_run(root / run,
                   _all_cells(30.0, {w: 20.0 * w for w in (1, 2, 4)}))
    ks = {}
    for cell in model_v2.CELLS:
        ok = cell != "bmma"
        rec = {"ok": ok, "cell": cell}
        if not ok:
            rec["n_hits"] = 0
            rec["sass_counts"] = {"CALL.REL.NOINC": 8, "MOV": 40}
        ks[f"mma/lat.{cell}/4"] = rec
    (root / "build" / "sass").mkdir(parents=True, exist_ok=True)
    (root / "build" / "sass" / "checks.json").write_text(
        json.dumps({"kernels": ks}))
    classes = _model(root, tmp_path / "m.json", monkeypatch)
    assert classes["BMMA"]["status"] == "unrealised:emulated"


def test_install_v2(tmp_path):
    src_root = (Path(__file__).resolve().parents[2] /
                "configs" / "mechanism_tree")
    work = tmp_path / "mt"
    shutil.copytree(src_root, work)
    m = {"device": "h100",
         "classes": {
             "DMMA": {"status": "experimental", "cell": "dmma",
                      "result_latency": 512.0,
                      "completion_latency": 512.0,
                      "issue_interval": 131.0,
                      "per_warp_blocking": True,
                      "wps4_wps1_ratio": 1.0,
                      "quotients": {"1": 512.0, "2": 256.0, "4": 128.0},
                      "thr_slopes": {"1": 4096.0, "2": 4096.0,
                                     "4": 4096.0},
                      "gate": True},
             "BMMA": {"status": "measured", "cell": "bmma",
                      "result_latency": 12.0,
                      "completion_latency": 12.0,
                      "issue_interval": 0.75,
                      "per_warp_blocking": False,
                      "wps4_wps1_ratio": 4.0,
                      "quotients": {"1": 1.0, "2": 0.8, "4": 0.75},
                      "thr_slopes": {"1": 8.0, "2": 12.8, "4": 24.0},
                      "gate": True}}}
    done = install_v2.install_device(work, "hopper", "h100", "h100-r8", m)
    leaves = json.loads((work / "hopper" / "leaves" /
                         "instruction.service.json").read_text())
    dm = leaves["instruction.service.root/predicated_on=1/"
                "operation_class=DMMA"]
    assert dm["fields"]["issue_interval_cycles"]["model"]["value"] == 131.0
    assert "mma_sync.h100.r8.experimental" in \
        dm["fields"]["issue_interval_cycles"]["sources"]
    # result_latency untouched (v1 source stays)
    assert "mma_sync.h100.r8" not in \
        dm["fields"]["result_latency_cycles"]["sources"]
    bm = leaves["instruction.service.root/predicated_on=1/"
                "operation_class=BMMA"]
    assert bm["fields"]["result_latency_cycles"]["model"]["value"] == 12.0
    assert bm["fields"]["issue_interval_cycles"]["model"]["value"] == 0.75
    srcs = json.loads((work / "hopper" / "sources.json").read_text())
    rec = srcs["mma_sync.h100.r8.experimental"]
    assert "per_warp_blocking=True" in rec["description"]
    # BMMA on hopper cited hopper.* hash sources (not in PLACEHOLDERS):
    # spec replaces only placeholder sources -> hash sources stay
    assert "mma_sync.h100.r8" in \
        bm["fields"]["result_latency_cycles"]["sources"]
