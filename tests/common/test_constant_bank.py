"""constant reference addresses carry the c[0xB] bank in bits 32+:
c[0x0][a] stays unchanged, c[0x3][a] no longer aliases bank 0."""
import collections
import types

from cyclescope.convert.capture_pipeline import _constant_refs


def _refs(operand, uniform=None, lane=None):
    rec = types.SimpleNamespace(exec_mask=0xFFFFFFFF)
    srow = types.SimpleNamespace(
        inst=types.SimpleNamespace(
            operands=("R0", operand), pc=0x0),
        opcode_text="LDC R0, " + operand + " ;")
    ops = types.SimpleNamespace(uniform=uniform or {}, lane=lane or {})
    return _constant_refs(rec, srow, ops, collections.Counter())


def test_bank_in_address():
    a0 = _refs("c[0x0][0x380]")
    a3 = _refs("c[0x3][0x380]")
    assert a0[0].lane_addresses == (0x380,) * 32          # unchanged
    assert a3[0].lane_addresses == ((0x3 << 32) | 0x380,) * 32
    assert a0[0].lane_addresses != a3[0].lane_addresses


def test_bank_in_register_indexed_address():
    a0 = _refs("c[0x0][R6+0x10]", lane={"R6": (0x200,) * 32})
    a3 = _refs("c[0x3][R6+0x10]", lane={"R6": (0x200,) * 32})
    assert a0[0].lane_addresses == (0x210,) * 32
    assert a3[0].lane_addresses == ((0x3 << 32) | 0x210,) * 32


def test_bank_in_uniform_register_address():
    a0 = _refs("c[0x0][UR4+0x8]", uniform={"UR4": 0x100})
    a3 = _refs("c[0x3][UR4+0x8]", uniform={"UR4": 0x100})
    assert a0[0].lane_addresses == (0x108,) * 32
    assert a3[0].lane_addresses == ((0x3 << 32) | 0x108,) * 32


def test_bank_in_immediate_wraps_at_32_bits():
    # offset part wraps at 32 bits as before; the bank is unaffected
    a3 = _refs("c[0x3][UR4+0x4]", uniform={"UR4": 0xFFFFFFFE})
    assert a3[0].lane_addresses == ((0x3 << 32) | 0x2,) * 32
