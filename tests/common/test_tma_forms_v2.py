"""tma_forms v2: variant table + install mapping."""
import sys, json, importlib.util, pytest
from pathlib import Path
TOOLS = Path(__file__).resolve().parents[2] / "tools/microbench/tma_forms"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"tma_forms_{name}", TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load("model")
I = _load("install")


def test_unknown_variant_errors():
    rows = {("tma.weird.u0", 0): {"mean": 1.0, "se": 0.1, "n": 3,
                                 "p50": 1.0, "params": {}}}
    with pytest.raises(SystemExit):
        M.derive(rows)


def test_op_form_mapping():
    assert I.op_form_of({"class": "TMA", "form": "store",
                         "extra": "tmast.utma_redg.b4k.u1"})[0] == "UTMAREDG"
    assert I.op_form_of({"class": "TMA", "form": "blkseq",
                         "extra": "k_blkseq.ublkred.u1"})[0] == "UBLKRED"
    assert I.op_form_of({"class": "TMA", "form": "tile",
                         "extra": "lat.ublkcp.b1024.u1"})[0] == "UBLKCP"
