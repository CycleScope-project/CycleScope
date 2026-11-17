"""Blackwell main-last launch selection hands the tracer name@occurrence."""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cyclescope.capture import driver

GEMM = ("_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute"
        "5tupleIJiiiiEEEvNT_6ParamsE")
INIT = "_ZN7cutlass9reference6device6kernel12BlockForEachIfEEvT_"
REF = "_ZN7cutlass9reference6device6kernel4GemmIfEEvT_"


class _Regex:
    def __init__(self, pattern):
        self.pattern = pattern


def _census(tmp_path, rows):
    out = tmp_path / "case"
    out.mkdir()
    with (out / "ncu_launches.jsonl").open("w") as fh:
        for i, (name, cycles) in enumerate(rows):
            fh.write(json.dumps({"ordinal": i, "function": name,
                                 "cycles_active_max": cycles,
                                 "grid": [1, 1, 1],
                                 "block": [128, 1, 1]}) + "\n")
    return out


def test_main_last_selects_by_name_and_occurrence(tmp_path):
    # census: three init kernels (heavy), the GEMM (light), a reference
    # GEMM, a compare kernel -- the main_kernel regex picks the GEMM and
    # the tracer gets "<mangled>@1" (its launch ids are 1-based; the
    # census ordinal 3 would have traced the init kernel before it)
    out = _census(tmp_path, [(INIT, 350000), (INIT, 344000), (INIT, 356000),
                             (GEMM, 13551), (REF, 612609),
                             ("_ZN7cutlass9reference6device6kernel"
                              "17BlockCompareEqualIfEEvP", 2132)])
    res = driver.Resolved(argv=[], env={}, cwd=None, arch="blackwell",
                          calls=1, main_regex=_Regex(".*GemmUniversal.*"),
                          family="cutlass")
    sel = driver._select_launch(res, out)
    assert sel == re.escape(GEMM) + "@1"
    assert re.fullmatch(sel.rsplit("@", 1)[0], GEMM)


def test_main_last_occurrence_counts_same_name(tmp_path):
    # calls=2: the GEMM runs twice per call -> 4 launches, the pick is
    # the first launch of the last call = 3rd GEMM launch
    out = _census(tmp_path, [(INIT, 1), (GEMM, 10), (GEMM, 10), (INIT, 1),
                             (GEMM, 10), (GEMM, 10)])
    res = driver.Resolved(argv=[], env={}, cwd=None, arch="blackwell",
                          calls=2, main_regex=_Regex(".*GemmUniversal.*"),
                          family="cutlass")
    assert driver._select_launch(res, out) == re.escape(GEMM) + "@3"
