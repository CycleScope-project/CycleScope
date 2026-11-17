"""model_parallelism.py (ifetch protocol-v2 addendum, fetch_parallelism
from the k_mix variants): synthetic launches.log with a known L2 cadence
and known q(W) -> value 4 * round(4 / q(16)); mode disagreement -> unrealised."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "microbench" / "ifetch" / "model_parallelism.py"

CPI_L2 = 5.5      # k_line warm cpi on the L2 plateau
A0 = 20           # per-pass fixed overhead
LINE_NS = [256, 512, 1024, 1280, 1536, 1792, 3072, 4096, 5120, 6144,
           10240, 12288, 16384, 32768]


def _cpi_line(n):
    if n <= 1792:
        return 1.0
    if n <= 6144:
        return 2.2
    return CPI_L2


def _launch_lines(variant, w, cycles_per_pass, launches=4):
    out = []
    for launch in range(launches):
        for ps, cyc in enumerate(cycles_per_pass):
            vals = ",".join([str(int(cyc))] * w)
            out.append(f"LAUNCH ifetch variant={variant} launch={launch} "
                       f"smid=7 pass={ps} cycles={vals}")
    return out


def _write_mode(run, mode, q_of_w, mix_points):
    lines = []
    for n in LINE_NS:
        t = A0 + _cpi_line(n) * n
        lines += _launch_lines(f"ifetch.line.n{n}.w1", 1, [3 * t, t, t, t])
    for w, n in mix_points:
        cpi = 1.0 + q_of_w[w] * (CPI_L2 - 1.0)
        t = A0 + cpi * n
        lines += _launch_lines(f"ifetch.mix.n{n}.w{w}", w, [3 * t, t, t, t])
    d = run / mode
    d.mkdir(parents=True)
    (d / "launches.log").write_text("\n".join(lines) + "\n")
    (d / "counters.csv").write_text("")


def _run(run, out):
    r = subprocess.run([sys.executable, str(TOOL), "--run", str(run),
                        "--capacity-l1", "114688", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(out.read_text())


MIX = [(4, 4096), (4, 8192), (8, 2048), (8, 4096), (16, 2048), (16, 4096),
       (4, 1024)]  # the last one (64 KiB total) must be skipped


def test_parallelism_eight_channels(tmp_path):
    run = tmp_path / "run"
    q = {4: 1.1, 8: 1.4, 16: 1.9}
    _write_mode(run, "all", q, MIX)
    _write_mode(run, "none", {4: 1.12, 8: 1.42, 16: 1.93}, MIX)
    d = _run(run, tmp_path / "model.json")
    f = d["fields"]["fetch_parallelism"]
    assert f["verdict"] == "experimental"
    assert f["streams_per_smsp"] == 2 and f["value"] == 8
    assert abs(f["q16"] - 1.915) < 0.01
    assert abs(f["p_w4"] - 4 / 1.11) < 0.02
    assert "w4.n1024" in d["modes"]["all"]["mix_skipped"]
    assert d["modes"]["all"]["l2_plateau_ns"] == [10240, 12288, 16384, 32768]
    assert abs(d["modes"]["all"]["cpi_line_l2"] - CPI_L2) < 1e-6


def test_parallelism_per_smsp_and_per_warp(tmp_path):
    # q(16) = 4 -> one stream per SMSP -> 4 channels; q(16) ~ 1 -> 16
    for q16, want in ((4.0, 4), (1.0, 16)):
        run = tmp_path / f"run{want}"
        q = {4: 1.0, 8: q16 / 2 if q16 > 1 else 1.0, 16: q16}
        _write_mode(run, "all", q, MIX)
        _write_mode(run, "none", q, MIX)
        d = _run(run, tmp_path / f"m{want}.json")
        assert d["fields"]["fetch_parallelism"]["value"] == want


def test_parallelism_mode_disagreement_is_unrealised(tmp_path):
    run = tmp_path / "run"
    _write_mode(run, "all", {4: 1.1, 8: 1.4, 16: 1.9}, MIX)
    _write_mode(run, "none", {4: 1.1, 8: 1.4, 16: 2.3}, MIX)
    d = _run(run, tmp_path / "model.json")
    f = d["fields"]["fetch_parallelism"]
    assert f["verdict"] == "unrealised" and f["value"] == 1
    assert any("q(16)" in r for r in f["reasons"])


def test_hopper_leaf_cites_mix_source():
    leaf = json.loads((ROOT / "configs" / "mechanism_tree" / "hopper" /
                       "leaves" / "instruction.fetch.json").read_text())
    f = leaf["instruction.fetch.root"]["fields"]["fetch_parallelism"]
    assert f["model"]["value"] == 8
    assert f["sources"] == ["ifetch.h100.r1.mix"]
    src = json.loads((ROOT / "configs" / "mechanism_tree" / "hopper" /
                      "sources.json").read_text())
    assert src["ifetch.h100.r1.mix"]["evidence"] == "experimental"
    assert "ifetch.h100.r1.unrealised" not in src
