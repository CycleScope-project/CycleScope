"""_ncu_occupancy resolves per-case symlink farms to find reference ncu csvs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from cyclescope.convert.capture_pipeline import _ncu_occupancy

NCU_CSV = """some preamble line
"ID","Kernel Name","launch__block_size","launch__occupancy_limit_registers","launch__occupancy_limit_shared_mem","launch__occupancy_limit_blocks","launch__occupancy_limit_warps"
"0","matmul_kernel","128","3","4","32","5"
"""

ROW = {"mangled": "matmul_kernel", "function_prior_launches": 0,
       "block": [128, 1, 1]}


def _make(tmp_path: Path) -> Path:
    case = tmp_path / "real" / "CASE-01"
    ref = tmp_path / "real" / "reference" / "CASE-01" / "CASE-01"
    ref.mkdir(parents=True)
    case.mkdir(parents=True)
    (case / "launches.jsonl").write_text("")
    (ref / "ncu_r0.csv").write_text(NCU_CSV)
    farm = tmp_path / "farm"
    farm.mkdir()
    (farm / "CASE-01").symlink_to(case)
    return farm / "CASE-01"


def test_ncu_occupancy_through_symlink_farm(tmp_path):
    link = _make(tmp_path)
    assert _ncu_occupancy(link, ROW) == 3


def test_ncu_occupancy_missing_reference_returns_none(tmp_path):
    case = tmp_path / "farm" / "CASE-01"
    case.mkdir(parents=True)
    (case / "launches.jsonl").write_text("")
    assert _ncu_occupancy(case, ROW) is None
