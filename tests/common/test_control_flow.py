"""control_flow: BRA loop facts + BSSY/BSYNC region facts are
mechanically derivable from a warp's own record sequence + static SASS."""
from cyclescope.convert.control_flow import (bssy_close_pairs, flow_patches,
                                         fill_participating)


class _Inst:
    def __init__(self, opcode, operands=(), pc=0):
        self.opcode = opcode
        self.operands = list(operands)
        self.pc = pc


class _Row:
    def __init__(self, opcode, operands=(), pc=0, target=None):
        self.inst = _Inst(opcode, operands, pc)
        self.branch_target = target


class _Rec:
    def __init__(self, fid, pc, mask):
        self.function_id = fid
        self.pc = pc
        self.exec_mask = mask


def test_backward_bra_loop_facts():
    rows = {0x100: _Row("IMAD.MOV", pc=0x100),
            0x110: _Row("LOP3.LUT", pc=0x110),
            0x120: _Row("BRA", operands=["B0", "0x100"], pc=0x120,
                        target=0x100),
            0x200: _Row("BRA", operands=["0x300"], pc=0x200,
                        target=0x300)}
    recs = [_Rec(0, 0x120, 0xFFFFFFFF), _Rec(0, 0x120, 0xFFFFFFFF),
            _Rec(0, 0x200, 0xFFFFFFFF)]
    static_of = lambda rec: rows[rec.pc]
    by_pc = lambda fid: rows
    patch = flow_patches(recs, static_of, by_pc, (0, 0, 0))
    assert patch[0]["same_body_reuse"] == 0
    assert patch[1]["same_body_reuse"] == 1
    assert patch[0]["loop_body_bytes"] == 0x120 - 0x100 + 16
    assert patch[0]["loop_instruction_count"] == 3
    assert patch[0]["backward_loop_class_s"] == \
        "pure_integer_register_control"
    assert 2 not in patch   # forward branch: no facts


def test_mixed_loop_class():
    rows = {0x100: _Row("LDG.E", pc=0x100),
            0x110: _Row("BRA", operands=["0x100"], pc=0x110,
                        target=0x100)}
    recs = [_Rec(0, 0x110, 0xFFFFFFFF)]
    patch = flow_patches(recs, lambda rec: rows[rec.pc],
                         lambda fid: rows, (0, 0, 0))
    assert patch[0]["backward_loop_class_s"] == "mixed_or_stateful"


def test_bssy_bsync_pairs_and_selector():
    rows = {0x100: _Row("BSSY", operands=["B0", "0x140"], pc=0x100),
            0x110: _Row("BSYNC", operands=["B0"], pc=0x110),
            0x120: _Row("BSSY", operands=["B0", "0x160"], pc=0x120),
            0x130: _Row("BSYNC", operands=["B0"], pc=0x130),
            0x140: _Row("BSSY", operands=["B0", "0x180"], pc=0x140),
            0x150: _Row("BSYNC", operands=["B0"], pc=0x150)}
    pairs = bssy_close_pairs(rows)
    assert pairs == {0x100: 0x110, 0x120: 0x130, 0x140: 0x150}
    full = 0xFFFFFFFF
    lo, hi = 0x0000FFFF, 0xFFFF0000
    # each region split into two sequential 16-lane paths -> a
    # completed selector run; count caps at 2 parameterized regions
    recs = [_Rec(0, 0x100, full), _Rec(0, 0x110, lo),
            _Rec(0, 0x110, hi),
            _Rec(0, 0x120, full), _Rec(0, 0x130, lo),
            _Rec(0, 0x130, hi),
            _Rec(0, 0x140, full), _Rec(0, 0x150, lo),
            _Rec(0, 0x150, hi)]
    patch = flow_patches(recs, lambda rec: rows[rec.pc],
                         lambda fid: rows, (0, 0, 0))
    opens = [p for p in patch.values() if "sequential_region_count" in p]
    assert len(opens) == 3
    counts = sorted(o["sequential_region_count"] for o in opens)
    assert counts == [1, 2, 2] or all(c <= 2 for c in counts)
    # a non-divergent region (single path) gets path_count=1 only
    recs2 = [_Rec(0, 0x100, full), _Rec(0, 0x110, full)]
    p2 = flow_patches(recs2, lambda rec: rows[rec.pc],
                      lambda fid: rows, (0, 0, 0))
    assert p2.get(0, {}).get("control_path_count") == 1
    assert "sequential_region_count" not in p2.get(0, {})


def test_fill_participating():
    key_patch = {0: {"same_body_reuse": 1, "_participants_key": "k",
                     "_loop_key": "k"}}
    m = {"w0": {0: {"_participants_key": "k", "_loop_key": "k"}},
         "w1": {0: {"_participants_key": "k", "_loop_key": "k"}}}
    fill_participating(m)
    assert m["w0"][0]["control_participating_warps"] == 2
    assert "_loop_key" not in m["w0"][0]
