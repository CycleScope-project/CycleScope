from cyclescope.convert.sass_decode import (
    decode_mbarrier_init_expected_arrivals as dec)
from cyclescope.convert.operand_facts import _init_expected_arrivals


def test_small_operand_not_init():
    assert dec(32) is None           # UR data value, not an init word
    assert dec((1 << 20) - 32 * 2) is None or dec((1 << 20) - 32 * 2) <= 1 << 16


def test_hopper_init_pair():
    count = 32
    low = ((1 << 20) - count) << 1
    high = (0x80000000 - (count << 11)) & 0xFFFFFFFF
    assert dec(low | (high << 32)) == count


def test_bracket_ur_not_init():
    # SYNCS.EXCH.64 URZ,[UR6+0x150],UR4: bracket base UR6=32 must not
    # decode as an init; the data operand UR4 is the only candidate
    count, value = _init_expected_arrivals(
        ["URZ", "UR6", "UR4"], {"UR6": 32, "UR4": 32}, {})
    assert count is None
