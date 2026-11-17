"""python vs cpp segment-builder parity on a minimal GSIMCAP01 dir."""
import json
import shutil
from pathlib import Path

import pytest

from cyclescope.convert.facts_format import FactsWriter
from cyclescope.convert import model
from cyclescope.convert.pipeline import convert_facts
from cyclescope.job.segment_format import read_segment, REQUIRED_TABLES

from .test_facts_format import _unit

REPO = Path(__file__).resolve().parents[2]
TREE = REPO / "configs" / "trees" / "h100.json"


def _facts_dir(tmp_path):
    d = tmp_path / "facts"
    d.mkdir()
    FactsWriter().write_unit(_unit(), d / "000000.gsc")
    (d / "launch.json").write_text(json.dumps({
        "device": "h100", "sm_arch": "sm_90a",
        "grid": [1, 1, 1], "block": [32, 1, 1], "cluster": [1, 1, 1],
        "capture": "test_capture"}))
    return d


def test_cpp_builder_parity(tmp_path):
    bin_ = shutil.which("cyclescope-convert") or str(REPO / "build" /
                                               "cyclescope-convert")
    if not Path(bin_).exists():
        pytest.skip("cyclescope-convert not built")
    facts = _facts_dir(tmp_path)
    out_py = tmp_path / "out_py"
    out_cpp = tmp_path / "out_cpp"
    convert_facts(facts, "hopper", TREE, out_py, cache_dir=None,
                  builder="python")
    convert_facts(facts, "hopper", TREE, out_cpp, cache_dir=None,
                  builder="cpp")
    assert json.loads((out_py / "launch.json").read_text()) == \
        json.loads((out_cpp / "launch.json").read_text())
    segs_py = sorted((out_py / "segments").glob("*.seg"))
    segs_cpp = sorted((out_cpp / "segments").glob("*.seg"))
    assert [p.name for p in segs_py] == [p.name for p in segs_cpp]
    for pa, pb in zip(segs_py, segs_cpp):
        from cyclescope.io.file_header import FileReader
        prog = FileReader((pa.parent / "program.tables").read_bytes())
        a = read_segment(pa.read_bytes(), program=prog)
        b = read_segment(pb.read_bytes(), program=prog)
        for tag in REQUIRED_TABLES:
            if tag == "META":
                continue
            assert bytes(a.table(tag)) == bytes(b.table(tag)), \
                f"{pa.name} table {tag}"


def _preexit_unit(pred_mask=1):
    """_unit() + one PREEXIT instance (executed unless pred_mask is 0)."""
    u = _unit()
    u.statics.append(model.StaticInstruction(
        pc=0x18, code_address=None, opcode="PREEXIT",
        operation_class="PREEXIT"))
    u.warps[0].instructions.append(model.DynamicInstruction(
        static_index=len(u.statics) - 1, active_mask=0xFFFFFFFF,
        predicate_mask=pred_mask))
    return u


def _convert_and_sema_kinds(tmp_path, name, unit, bin_):
    facts = tmp_path / f"facts_{name}"
    facts.mkdir()
    FactsWriter().write_unit(unit, facts / "000000.gsc")
    (facts / "launch.json").write_text(json.dumps({
        "device": "h100", "sm_arch": "sm_90a",
        "grid": [1, 1, 1], "block": [32, 1, 1], "cluster": [1, 1, 1],
        "capture": "test_capture"}))
    out_py, out_cpp = tmp_path / f"{name}_py", tmp_path / f"{name}_cpp"
    convert_facts(facts, "hopper", TREE, out_py, cache_dir=None,
                  builder="python")
    convert_facts(facts, "hopper", TREE, out_cpp, cache_dir=None,
                  builder="cpp")
    from cyclescope.io.file_header import FileReader
    from cyclescope.job import segment_format as sf
    kinds = {}
    for out in (out_py, out_cpp):
        seg = sorted((out / "segments").glob("*.seg"))[0]
        prog = FileReader((seg.parent / "program.tables").read_bytes())
        a = read_segment(seg.read_bytes(), program=prog)
        tbl = a.table("SEMA")
        kinds[out.name] = [sf.SEMA.unpack(bytes(tbl), i)["kind"]
                           for i in range(len(tbl) // sf.SEMA.size)]
    assert kinds[f"{name}_py"] == kinds[f"{name}_cpp"]
    return kinds[f"{name}_py"]


def test_preexit_dependents_launch_row(tmp_path):
    # PREEXIT executed -> one SEMA row of kind 40 (dependents_launch);
    # predicated off -> none.  py and cpp converters agree byte-wise.
    bin_ = shutil.which("cyclescope-convert") or str(REPO / "build" /
                                               "cyclescope-convert")
    if not Path(bin_).exists():
        pytest.skip("cyclescope-convert not built")
    kinds = _convert_and_sema_kinds(tmp_path, "px", _preexit_unit(), bin_)
    assert kinds.count(40) == 1
    kinds = _convert_and_sema_kinds(tmp_path, "pxoff",
                                    _preexit_unit(pred_mask=0), bin_)
    assert 40 not in kinds


def _blackwell_unit():
    """Minimal blackwell facts unit carrying TMAP + MBPI."""
    unit = _unit()
    unit.tensor_maps = {0: model.TensorMap(
        rank=2, element_bytes=2, global_dim=(128, 64),
        stride_bytes=(256,), box_dim=(4, 2), global_address=0x2000)}
    unit.preinitialized_mbarriers = [(0, 0x48, 1)]
    for c in unit.ctas:
        c.tmem_columns = 512
    return unit


def test_cpp_builder_parity_blackwell_tmap_mbpi(tmp_path):
    bin_ = shutil.which("cyclescope-convert") or str(REPO / "build" /
                                               "cyclescope-convert")
    if not Path(bin_).exists():
        pytest.skip("cyclescope-convert not built")
    facts = tmp_path / "facts_bw"
    facts.mkdir()
    FactsWriter().write_unit(_blackwell_unit(), facts / "000000.gsc")
    (facts / "launch.json").write_text(json.dumps({
        "device": "b300", "sm_arch": "sm_103a",
        "grid": [1, 1, 1], "block": [32, 1, 1], "cluster": [1, 1, 1],
        "capture": "test_capture_bw"}))
    out_py = tmp_path / "bw_py"
    out_cpp = tmp_path / "bw_cpp"
    convert_facts(facts, "blackwell", REPO / "configs" / "trees" /
                  "b300.json", out_py, cache_dir=None, builder="python")
    convert_facts(facts, "blackwell", REPO / "configs" / "trees" /
                  "b300.json", out_cpp, cache_dir=None, builder="cpp")
    segs_py = sorted((out_py / "segments").glob("*.seg"))
    segs_cpp = sorted((out_cpp / "segments").glob("*.seg"))
    assert [p.name for p in segs_py] == [p.name for p in segs_cpp]
    for pa, pb in zip(segs_py, segs_cpp):
        from cyclescope.io.file_header import FileReader
        prog = FileReader((pa.parent / "program.tables").read_bytes())
        a = read_segment(pa.read_bytes(), program=prog)
        b = read_segment(pb.read_bytes(), program=prog)
        for tag in REQUIRED_TABLES:
            if tag == "META":
                continue
            assert bytes(a.table(tag)) == bytes(b.table(tag)), \
                f"{pa.name} table {tag}"


def test_gsimcap02_parity(tmp_path):
    """Same unit as GSIMCAP01 vs GSIMCAP02+statics.gsc: segments from
    either facts format must be byte-identical under both builders."""
    import json as _json, shutil
    from cyclescope.convert.facts_format import (
        StreamingFactsWriter, write_statics_file)
    u = _unit()
    d2 = tmp_path / "f2"
    d2.mkdir()
    w = StreamingFactsWriter(split_statics=True)
    w.begin_warp(u.warps[0].cta_linear, u.warps[0].cta,
                 u.warps[0].sm_id, u.warps[0].warp_slot,
                 u.warps[0].warp_in_cta)
    for d in u.warps[0].instructions:
        w.add_warp_instruction(d)
    w.end_warp()
    w.finish(u.cluster_id, u.ctas, u.statics, {}, [], d2 / "000007.gsc")
    write_statics_file(u.statics, d2 / "statics.gsc")
    (d2 / "launch.json").write_text(_json.dumps({
        "device": "h100", "sm_arch": "sm_90a",
        "grid": [1, 1, 1], "block": [32, 1, 1], "cluster": [1, 1, 1],
        "capture": "test_capture"}))
    old = _facts_dir(tmp_path)
    bin_ = shutil.which("cyclescope-convert") or str(REPO / "build" /
                                               "cyclescope-convert")
    builders = ["python"] + (["cpp"] if Path(bin_).exists() else [])
    for builder in builders:
        o1, o2 = tmp_path / f"o1-{builder}", tmp_path / f"o2-{builder}"
        convert_facts(old, "hopper", TREE, o1, cache_dir=None,
                      builder=builder)
        convert_facts(d2, "hopper", TREE, o2, cache_dir=None,
                      builder=builder)
        s1 = sorted((o1 / "segments").glob("*.seg"))
        s2 = sorted((o2 / "segments").glob("*.seg"))
        assert [f.name for f in s1] == [f.name for f in s2]
        for f, g in zip(s1, s2):
            assert f.read_bytes() == g.read_bytes(), f"{builder}:{f.name}"


def test_gsimcap02_dynamic_strings_cpp(tmp_path):
    """GSIMCAP02 per-file STRS must serve dynamic UKV/LANE strings for the
    cpp builder (mbarrier wait reads uniform 'parity')."""
    import dataclasses, json as _json, shutil
    from cyclescope.convert.facts_format import (
        StreamingFactsWriter, write_statics_file)
    bin_ = shutil.which("cyclescope-convert") or str(REPO / "build" /
                                               "cyclescope-convert")
    if not Path(bin_).exists():
        pytest.skip("cyclescope-convert not built")
    u = _unit()
    st = dataclasses.replace(u.statics[0], opcode="SYNCS.PHASECHK",
                             operation_class="sync", memory_space=None,
                             branch_target=None)
    ops = dataclasses.replace(
        u.warps[0].instructions[0].operands,
        uniform={"parity": 1, "mbarrier_address": 0x40},
        lane={}, ranks={}, tensor_map=None, tensor_coordinates=None,
        descriptor_words=None)
    d = dataclasses.replace(u.warps[0].instructions[0], memory=(),
                            operands=ops, global_sync=None)
    u.statics = [st]
    u.warps = [dataclasses.replace(u.warps[0], instructions=[d])]
    d2 = tmp_path / "f2"
    d2.mkdir()
    w = StreamingFactsWriter(split_statics=True)
    w.begin_warp(u.warps[0].cta_linear, u.warps[0].cta,
                 u.warps[0].sm_id, u.warps[0].warp_slot,
                 u.warps[0].warp_in_cta)
    for ins in u.warps[0].instructions:
        w.add_warp_instruction(ins)
    w.end_warp()
    w.finish(u.cluster_id, u.ctas, u.statics, {}, [], d2 / "000007.gsc")
    write_statics_file(u.statics, d2 / "statics.gsc")
    (d2 / "launch.json").write_text(_json.dumps({
        "device": "h100", "sm_arch": "sm_90a",
        "grid": [1, 1, 1], "block": [32, 1, 1], "cluster": [1, 1, 1],
        "capture": "test_capture"}))
    outs = {}
    for builder in ("python", "cpp"):
        o = tmp_path / f"o-{builder}"
        convert_facts(d2, "hopper", TREE, o, cache_dir=None,
                      builder=builder)
        outs[builder] = sorted((o / "segments").glob("*.seg"))
    assert [f.name for f in outs["python"]] == \
        [f.name for f in outs["cpp"]]
    for f, g in zip(outs["python"], outs["cpp"]):
        assert f.read_bytes() == g.read_bytes(), f.name


def test_split_program_parity(tmp_path):
    """v2 segments (no program tables) + program.tables must rebuild the
    same container tables as v1 segments; py and cpp v2 bytes equal."""
    import json as _json, shutil
    from cyclescope.job import segment_format as sf
    from cyclescope.job.container import merge_segments
    facts = _facts_dir(tmp_path)
    out1, out2 = tmp_path / "v1", tmp_path / "v2"
    convert_facts(facts, "hopper", TREE, out1, cache_dir=None,
                  builder="python", split_program=False)
    convert_facts(facts, "hopper", TREE, out2, cache_dir=None,
                  builder="python", split_program=True)
    assert (out2 / "segments" / "program.tables").is_file()
    s1 = sorted((out1 / "segments").glob("*.seg"))
    s2 = sorted((out2 / "segments").glob("*.seg"))
    m1 = merge_segments([f.read_bytes() for f in s1])
    prog = (out2 / "segments" / "program.tables").read_bytes()
    m2 = merge_segments([f.read_bytes() for f in s2], program=prog)
    assert set(m1) == set(m2)
    for k in m1:
        v1, v2 = m1[k], m2[k]
        if isinstance(v1, (bytes, bytearray)):
            assert bytes(v1) == bytes(v2), k
        else:
            assert v1 == v2, k
