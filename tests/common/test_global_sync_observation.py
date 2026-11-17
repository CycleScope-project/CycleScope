"""GSYNC records -> observations: tracer layout (phase 0 = source operands
+ pre_value tail, phase 1 = destination words) and atomic post values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from types import SimpleNamespace

from cyclescope.convert import global_sync_observation as gso


def _row(opcode):
    return SimpleNamespace(inst=SimpleNamespace(opcode=opcode))


def _rec(gsync, addr=0x7f00):
    return SimpleNamespace(gsync=gsync, mrefs=[(addr,)])


def test_red_add_pre_value_gives_before_and_post():
    # REDG.ADD: phase-0 words per lane = [addend, pre_value]
    rec = _rec([(0, 2, 1, [(1, 0)], 2767)])
    (obs,), lane = gso.observations(rec, _row("REDG.E.ADD.S32.STRONG.GPU"))
    assert obs.kind == 1 and obs.before == 0 and obs.value == 1
    assert obs.post_known == 3 and obs.sequence == 2767
    assert lane["gsync_before"] == (0,) and lane["gsync_after"] == (1,)


def test_ld_strong_destination_is_after():
    rec = _rec([(1, 1, 0, [(0,)], 2318)])
    (obs,), _ = gso.observations(rec, _row("LDG.E.STRONG.GPU"))
    assert obs.kind == 2 and obs.value == 0 and obs.post_known == 1


def test_atom_return_is_old_value():
    # ATOMG.ADD: phase 0 = [addend], phase 1 = returned old value 5
    rec = _rec([(0, 1, 0, [(2,)], 10), (1, 1, 0, [(5,)], 11)])
    (obs,), _ = gso.observations(rec, _row("ATOMG.E.ADD.STRONG.GPU"))
    assert obs.kind == 3 and obs.before == 5 and obs.value == 7
    assert obs.post_known == 3 and obs.sequence == 10


def test_store_operand_is_after_and_64bit_words():
    rec = _rec([(0, 2, 0, [(0x12345678, 0x9)], 3)])
    (obs,), _ = gso.observations(rec, _row("STG.E.64.STRONG.GPU"))
    assert obs.value == (0x9 << 32) | 0x12345678 and obs.post_known == 1
