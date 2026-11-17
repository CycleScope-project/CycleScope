import json
import shutil
import sys
from pathlib import Path

import importlib.util


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] /
        "tools" / "microbench" / "service_latency" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sass_check = _load("sass_check_alu4")
model_v3 = _load("model_v3")
install_v3 = _load("install_v3")

_REPO = Path(__file__).resolve().parents[2]
_H100_TREE = str(_REPO / "configs" / "trees" / "h100.json")


def _sass(kernels):
    parts = []
    for name, insts in kernels.items():
        parts.append(f"\t\tFunction : {name}")
        for i, text in enumerate(insts):
            parts.append(f"\t/*{i*16:04x}*/ {text} ;\t/* 0x{i:016x} */")
        parts.append("")
    return "\n".join(parts)


CLK = "CS2R R0, SR_CLOCKLO ;"
GATE = ["--arch", "hopper", "--tree", _H100_TREE]


def test_gate_composition_exact(tmp_path):
    # VIADDMNMX declares {VIADDMNMX:1}: 16 in window, U=16 -> ok
    kern = {"_Z5k_latI9VIADDMNMXLi16EEvPmPKmPji":
            [CLK] + ["VIADDMNMX R0, R1, R2, R3 ;"] * 16 + [CLK]}
    sass = tmp_path / "a.sass"
    sass.write_text(_sass(kern))
    out = tmp_path / "c.json"
    sys.argv = ["sass_check_alu4.py", "--sass", str(sass), "--out",
                str(out)] + GATE
    assert sass_check.main() == 0
    r = json.loads(out.read_text())["kernels"]["alu4/lat.VIADDMNMX/16"]
    assert r["ok"] and r["credit"] == "VIADDMNMX"


def test_gate_unrealised_wrong_count(tmp_path):
    kern = {"_Z5k_latI9VIADDMNMXLi16EEvPmPKmPji":
            [CLK] + ["VIADDMNMX R0, R1 ;"] * 3 + [CLK]}
    sass = tmp_path / "a.sass"
    sass.write_text(_sass(kern))
    out = tmp_path / "c.json"
    sys.argv = ["sass_check_alu4.py", "--sass", str(sass), "--out",
                str(out)] + GATE
    sass_check.main()
    r = json.loads(out.read_text())["kernels"]["alu4/lat.VIADDMNMX/16"]
    assert not r["ok"] and r["credit"] is None


def test_gate_stray_fails(tmp_path):
    # a stray non-harness class in the window fails the cell
    kern = {"_Z5k_latI4IABSLi16EEvPmPKmPji":
            [CLK] + ["IABS R0, R1 ;"] * 16 +
            ["LOP3.LUT R0, R1, R2 ;"] * 16 +
            ["DMUL R0, R1, R2 ;"] + [CLK]}
    sass = tmp_path / "a.sass"
    sass.write_text(_sass(kern))
    out = tmp_path / "c.json"
    sys.argv = ["sass_check_alu4.py", "--sass", str(sass), "--out",
                str(out)] + GATE
    sass_check.main()
    r = json.loads(out.read_text())["kernels"]["alu4/lat.IABS/16"]
    assert not r["ok"]


def _run(d, cell, lat, thr, mode="thr"):
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"params": {"mode": "lat", "op": f"lat.{cell}"},
                         "u": u, "variant": f"lat.{cell}",
                         "samples": [lat * u] * 4}) for u in (16, 32, 64)]
    lines += [json.dumps({"params": {"mode": mode, "op": f"{mode}.{cell}"},
                          "u": u, "variant": f"{mode}.{cell}",
                          "samples": [thr * u] * 4}) for u in (16, 32, 64)]
    (d / "alu4.jsonl").write_text("\n".join(lines))


def _checks(root, cell, modes=("lat", "thr")):
    kerns = {}
    for m in modes:
        for u in (16, 32, 64):
            kerns[f"alu4/{m}.{cell}/{u}"] = {
                "ok": True, "credit": cell, "cell": cell,
                "mode": m, "u": u}
    (root / "build" / "sass").mkdir(parents=True, exist_ok=True)
    (root / "build" / "sass" / "checks_alu4.json").write_text(
        json.dumps({"kernels": kerns}))


def test_model_single(tmp_path, monkeypatch):
    root = tmp_path / "run"
    for run in ("run-a", "run-b"):
        _run(root / run, "VIADDMNMX", 5.0, 8.0)
    _checks(root, "VIADDMNMX")
    out = tmp_path / "m.json"
    monkeypatch.setattr(
        "sys.argv",
        ["model_v3.py", "--device", "h100", "--root", str(root),
         "--out", str(out), "--tree", _H100_TREE])
    model_v3.main()
    m = json.loads(out.read_text())
    e = m["classes"]["VIADDMNMX"]
    assert e["status"] == "measured"
    assert abs(e["result_latency"] - 5.0) < 1e-6
    assert abs(e["issue_interval"] - 0.25) < 1e-9  # thr slope 8 / 32, smsp
    return m


def test_model_pair_subtraction(tmp_path, monkeypatch):
    # IABS = IABS + LOP3 on the critical path; h100 LOP3 is measured at
    # 4.0 result_latency -> result = slope - 4.0, status experimental
    root = tmp_path / "run"
    for run in ("run-a", "run-b"):
        _run(root / run, "IABS", 9.0, 16.0)
    _checks(root, "IABS")
    out = tmp_path / "m.json"
    monkeypatch.setattr(
        "sys.argv",
        ["model_v3.py", "--device", "h100", "--root", str(root),
         "--out", str(out), "--tree", _H100_TREE])
    model_v3.main()
    m = json.loads(out.read_text())
    e = m["classes"]["IABS"]
    assert e["status"] == "experimental"          # pair cap
    assert e["partner_result_latency"] == {"LOP3": 4.0}
    assert abs(e["result_latency"] - 5.0) < 1e-6  # 9 - 4


def test_model_and_install(tmp_path, monkeypatch):
    m = test_model_single(tmp_path, monkeypatch)
    src = _REPO / "configs" / "mechanism_tree"
    work = tmp_path / "mt"
    shutil.copytree(src, work)
    done = install_v3.install_device(work, "hopper", "h100", "h100-v3r2", m)
    leaves = json.loads((work / "hopper" / "leaves" /
                         "instruction.service.json").read_text())
    n = leaves["instruction.service.root/predicated_on=1/"
               "operation_class=VIADDMNMX"]
    assert abs(n["fields"]["result_latency_cycles"]["model"]["value"]
               - 5.0) < 1e-6
    srcs = json.loads((work / "hopper" / "sources.json").read_text())
    assert "service_latency.h100.h100-v3r2" in srcs


def test_gate_scaling_check(tmp_path):
    # a harness class scaling with U fails every kernel of the (cell, mode)
    kerns = {}
    for u in (16, 32, 64):
        kerns[f"_Z5k_latI4IABSLi{u}EEvPmPKmPji"] = (
            [CLK] + ["IABS R0, R1 ;"] * u +
            ["LOP3.LUT R0, R1, R2 ;"] * u +
            ["IMAD R0, R1, R2 ;"] * (u // 2) + [CLK])
    sass = tmp_path / "a.sass"
    sass.write_text(_sass(kerns))
    out = tmp_path / "c.json"
    sys.argv = ["sass_check_alu4.py", "--sass", str(sass), "--out",
                str(out)] + GATE
    sass_check.main()
    ks = json.loads(out.read_text())["kernels"]
    assert all(not v["ok"] for v in ks.values())
    assert any("harness scales" in w for v in ks.values()
               for w in v["why"])
