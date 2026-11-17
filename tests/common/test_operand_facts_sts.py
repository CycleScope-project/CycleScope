import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cyclescope.convert import model
from cyclescope.convert.operand_facts import operand_facts
from cyclescope.convert.static_code import StaticRow
from cyclescope.convert.trace_reader import DynRecord
from cyclescope.arch.hopper import capture_rules as hopper_capture
from cyclescope.arch.blackwell import capture_rules as blackwell_capture


def _row(opcode, operands, layout):
    inst = model.StaticInstruction(pc=0, code_address=0, opcode=opcode,
                                   operation_class="storage",
                                   operands=tuple(t.strip() for t in
                                                  operands.split(",")))
    return StaticRow(idx=0, sass=f"{opcode} {operands}", ops_layout=layout,
                     mref_count=1, opcode_text=opcode,
                     operands_text=operands, branch_target=None,
                     static_index=0, inst=inst)


def _rec(ops):
    return DynRecord(launch=0, cta=(0, 0, 0), warp=0, sm=0, warp_slot=0,
                     function_id=0, idx=0, active_mask=1, exec_mask=1,
                     mrefs=[(0x40,)], ops=ops)


# hopper init word for count 2: (2**20 - 2) << 1 in the low word, high
# word mirrors 0x80000000 - (2 << 11) with bit 31 = initial phase
LOW2 = ((1 << 20) - 2) << 1
HIGH2 = (0x80000000 - (2 << 11)) & 0x7FFFFFFF


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_sts64_pair_decodes_init(rules):
    row = _row("STS.64", "[R2], R4", ("R2", "R3", "R4", "R5"))
    ops = operand_facts(_rec({"R4": [LOW2], "R5": [HIGH2]}), row, rules)
    assert ops.uniform["expected_arrivals"] == 2
    assert ops.uniform["initial_phase"] == 0


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_sts64_phase_one(rules):
    row = _row("STS.64", "[R2], R4", ("R2", "R3", "R4", "R5"))
    ops = operand_facts(_rec({"R4": [LOW2], "R5": [HIGH2 | 0x80000000]}),
                        row, rules)
    assert ops.uniform["expected_arrivals"] == 2
    assert ops.uniform["initial_phase"] == 1


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_sts32_single_register_decodes_init(rules):
    row = _row("STS", "[UR21+0x180], R9", ("UR21", "UR22", "R9"))
    ops = operand_facts(_rec({"UR21": 0x400, "UR22": 0, "R9": [LOW2]}),
                        row, rules)
    assert ops.uniform["expected_arrivals"] == 2


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_sts32_without_data_register_has_no_count(rules):
    row = _row("STS", "[UR21+0x180], R9", ("UR21", "UR22"))
    ops = operand_facts(_rec({"UR21": 0x400, "UR22": 0}), row, rules)
    assert "expected_arrivals" not in ops.uniform


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_sts32_non_init_value_has_no_count(rules):
    row = _row("STS", "[R2], R9", ("R9",))
    ops = operand_facts(_rec({"R9": [0x12345678]}), row, rules)
    assert "expected_arrivals" not in ops.uniform


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_arrives_bracket_displacement_added_to_barrier_word(rules):
    row = _row("ARRIVES.LDGSTSBAR.64.TRANSCNT", "[UR7+0x38000]",
               ("UR7", "UR8", "RANK0"))
    rec = _rec({"UR7": 1024, "UR8": 0, "URANK7": [0]})
    rec.mrefs = [(0x38400,)]
    ops = operand_facts(rec, row, rules)
    assert ops.uniform["mbarrier_address"] == 0x38400


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_syncs_bracket_word_from_mref0(rules):
    row = _row("SYNCS.ARRIVE.TRANS64.A1T0", "RZ, [UR4], RZ", ("UR4", "UR5"))
    rec = _rec({"UR4": 0x38408, "UR5": 0})
    rec.mrefs = [(0x38408,)]
    ops = operand_facts(rec, row, rules)
    assert ops.uniform["mbarrier_address"] == 0x38408


def test_stas_barrier_word_is_pair_high_register():
    # STAS [R4.64], R11: R4 = remote data word (rank 1), R5 = remote
    # mbarrier word 0x1000048 -> barrier at rank 1's 0x48, 4 B per lane
    row = _row("STAS", "[R4.64], R11", ("R4", "R5", "R11", "RANK0"))
    rec = _rec({"R4": [0x1000580], "R5": [0x1000048], "R11": [0],
                "RANK0": [1]})
    rec.mrefs = [(0x1000580,)]
    ops = operand_facts(rec, row, blackwell_capture)
    assert ops.uniform["mbarrier_address"] == 0x1000048
    assert ops.uniform["owner_rank"] == 1
    assert ops.uniform["bytes"] == 4


def test_stas_pair_with_displacement():
    # hopper nvjet: STAS [R8.64+0x10], R3 -> data at R8+0x10, barrier R9
    row = _row("STAS", "[R8.64+0x10], R3", ("R8", "R9", "R3", "RANK0"))
    rec = _rec({"R8": [0x1034480], "R9": [0x1034480], "R3": [0x84],
                "RANK0": [1]})
    rec.mrefs = [(0x103448001034490,)]
    ops = operand_facts(rec, row, hopper_capture)
    assert ops.uniform["mbarrier_address"] == 0x1034480
    assert ops.uniform["owner_rank"] == 1 and ops.uniform["bytes"] == 4


def test_atoms_names_no_barrier_word():
    row = _row("ATOMS.OR", "RZ, [UR5+0x14], R8", ("UR5", "UR6", "R8"))
    rec = _rec({"UR5": 0x400, "UR6": 0, "R8": [1]})
    rec.mrefs = [(0x414,)]
    ops = operand_facts(rec, row, blackwell_capture)
    assert "mbarrier_address" not in ops.uniform


def test_tmem_alloc_columns_granules_and_dealloc_mask():
    # 2CTA alloc: UR4 = 16 granules of 32 columns -> 512
    row = _row("UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN", "UP0, UR4, UR4",
               ("UR4",))
    rec = _rec({"UR4": 16})
    rec.mrefs = []
    ops = operand_facts(rec, row, blackwell_capture)
    assert ops.uniform["tmem_columns"] == 512
    # dealloc: the operand is a column mask, not a count
    row = _row("UTCATOMSWS.AND", "URZ, UR5", ("UR5",))
    rec = _rec({"UR5": 0xFFFF0000})
    rec.mrefs = []
    ops = operand_facts(rec, row, blackwell_capture)
    assert "tmem_columns" not in ops.uniform
    assert ops.uniform["tmem_dealloc_mask"] == 0xFFFF0000


def test_clc_query_barrier_is_last_bracket():
    row = _row("UGETNEXTWORKID.BROADCAST", "[UR8], [UR9]",
               ("UR8", "UR9", "UR10"))
    rec = _rec({"UR8": 1376, "UR9": 1232, "UR10": 1})
    rec.mrefs = []
    ops = operand_facts(rec, row, blackwell_capture)
    assert ops.uniform["mbarrier_address"] == 1232


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_syncs_bracket_word_from_registers_not_mref(rules):
    # sm_103 NVBit reports the SYNCS mref at +0x7FFF; the word is R13+0x8
    row = _row("SYNCS.PHASECHK.TRANS64.TRYWAIT", "P1, [R13+URZ+0x8], R2",
               ("R13", "R2"))
    rec = _rec({"R13": [0x1000040], "R2": [0x80000000]})
    rec.mrefs = [(0x1008047,)]
    ops = operand_facts(rec, row, rules)
    assert ops.uniform["mbarrier_address"] == 0x1000048
    # register not captured -> mref fallback
    rec = _rec({"R2": [0]})
    rec.mrefs = [(0x8047,)]
    ops = operand_facts(rec, row, rules)
    assert ops.uniform["mbarrier_address"] == 0x8047


def test_predicated_off_instance_names_no_barrier():
    row = _row("SYNCS.EXCH.64", "URZ, [UR6+0x150], UR4",
               ("UR6", "UR7", "UR4", "UR5"))
    rec = _rec({"UR6": 16384, "UR7": 1, "UR4": 32, "UR5": 0})
    rec.exec_mask = 0
    rec.mrefs = []
    ops = operand_facts(rec, row, blackwell_capture)
    assert "mbarrier_address" not in ops.uniform
    assert "expected_arrivals" not in ops.uniform


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_tma_store_names_no_barrier(rules):
    row = _row("UTMASTG.3D", "[UR4], [UR8]", ("UR4", "UR5", "UR8", "UR9"))
    rec = _rec({"UR4": 0x1600, "UR5": 0x580, "UR8": 1, "UR9": 2})
    rec.mrefs = [(0x1600,)]
    ops = operand_facts(rec, row, rules)
    assert "mbarrier_address" not in ops.uniform


def test_bulk_store_g_s_names_no_barrier():
    row = _row("UBLKCP.G.S", "[UR18], [UR21], UR10", ("UR18", "UR19", "UR21", "UR22", "UR10"))
    rec = _rec({"UR18": 0x7f00, "UR19": 0x7fb0, "UR21": 1, "UR22": 2, "UR10": 256})
    rec.mrefs = [(0x7f00,)]
    ops = operand_facts(rec, row, blackwell_capture)
    assert "mbarrier_address" not in ops.uniform


def test_red_arrive_untagged_word_is_rank_zero():
    row = _row("SYNCS.ARRIVE.TRANS64.RED.A1T0", "RZ, [URZ+0x34488], RZ", ())
    rec = _rec({})
    rec.mrefs = [(0x34488,)]
    ops = operand_facts(rec, row, hopper_capture)
    assert ops.uniform["mbarrier_address"] == 0x34488
    assert ops.uniform["owner_rank"] == 0
    # the local (non-RED) arrive keeps the issuer's own object
    row = _row("SYNCS.ARRIVE.TRANS64.A1T0", "RZ, [URZ+0x34488], RZ", ())
    ops = operand_facts(rec, row, hopper_capture)
    assert "owner_rank" not in ops.uniform


def test_red_arrive_per_lane_words_from_mrefs_when_registers_missing():
    # nvjet 2CTA: SYNCS.ARRIVE.TRANS64.RED RZ, [R6+UR9], R7 with lane 0 ->
    # rank 0 and lane 1 -> rank 1 (R6 = 0 / 1 << 24); UR9 not captured, so
    # the per-lane words come from the per-lane mref addresses (rank tag
    # in bits 24+) instead of collapsing onto one rank-0 object
    row = _row("SYNCS.ARRIVE.TRANS64.RED", "RZ, [R6+UR9], R7",
               ("R6", "UR9", "R7"))
    rec = _rec({"R6": [0, 1 << 24], "R7": [16, 16]})
    rec.active_mask = rec.exec_mask = 0x3
    rec.mrefs = [(0x344c0, 0x10344c0)]
    ops = operand_facts(rec, row, blackwell_capture)
    assert ops.uniform["mbarrier_address"] == 0x344c0
    assert ops.lane["mbarrier_words"] == (0x344c0, 0x10344c0)


def _art_row(operands):
    inst = model.StaticInstruction(
        pc=0, code_address=0,
        opcode="SYNCS.ARRIVE.TRANS64.RED.ART0",
        operation_class="sync",
        operands=tuple(t.strip() for t in operands.split(",")))
    return StaticRow(
        idx=0, sass=f"SYNCS.ARRIVE.TRANS64.RED.ART0 {operands}",
        ops_layout=tuple(t.strip() for t in operands.split(",")),
        mref_count=1, opcode_text="SYNCS.ARRIVE.TRANS64.RED.ART0",
        operands_text=operands, branch_target=None, static_index=0,
        inst=inst)


def _art_rec(ops, exec_mask=0xFFFFFFFF):
    return DynRecord(launch=0, cta=(0, 0, 0), warp=0, sm=0, warp_slot=0,
                     function_id=0, idx=0, active_mask=exec_mask,
                     exec_mask=exec_mask, mrefs=[(0x23610,)], ops=ops)


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_art_register_arrival_count(rules):
    row = _art_row("RZ, [UR12+0x23610], R0")
    ops = operand_facts(_art_rec({"R0": [2] * 32}), row, rules)
    assert ops.uniform["arrival_count"] == 64
    assert ops.uniform["transaction_bytes"] == 0


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_art_mixed_lane_values(rules):
    row = _art_row("RZ, [UR12+0x23610], R0")
    ops = operand_facts(_art_rec({"R0": [1, 2, 0, 3] + [0] * 28}),
                        row, rules)
    assert ops.uniform["arrival_count"] == 6


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_art_rz_arrival_count(rules):
    row = _art_row("RZ, [UR12+0x23610], RZ")
    ops = operand_facts(_art_rec({}), row, rules)
    assert ops.uniform["arrival_count"] == 0


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_art_uncaptured_register_missing(rules):
    from cyclescope.convert.sass_decode import MissingFact
    row = _art_row("RZ, [UR12+0x23610], R7")
    with pytest.raises(MissingFact):
        operand_facts(_art_rec({"R0": [2] * 32}), row, rules)


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_art_predicated_off_instance_needs_no_lane_values(rules):
    # @!P2 false on every lane: no lane values are recorded (exec_mask 0)
    # and the instance arrives nowhere -> no arrival fact, no MissingFact
    row = _art_row("RZ, [UR12+0x23610], R3")
    rec = _art_rec({"UR12": 1024}, exec_mask=0)
    rec.active_mask = 0xFFFFFFFF
    rec.mrefs = [()]
    ops = operand_facts(rec, row, rules)
    assert "arrival_count" not in ops.uniform


def test_utmacctl_iv_descriptor_address_pair():
    inst = model.StaticInstruction(
        pc=0, code_address=0, opcode="UTMACCTL.IV",
        operation_class="sync", operands=("[UR28]",))
    row = StaticRow(idx=0, sass="UTMACCTL.IV [UR28]",
                    ops_layout=("[UR28]",), mref_count=0,
                    opcode_text="UTMACCTL.IV", operands_text="[UR28]",
                    branch_target=None, static_index=0, inst=inst)
    ops = operand_facts(_art_rec({"UR28": 3580032512,
                                  "UR29": 32700}), row, hopper_capture)
    assert ops.uniform["descriptor_address"] == \
        (32700 << 32) | 3580032512
    # either UR missing -> key stays absent (decoder reports missing)
    ops = operand_facts(_art_rec({"UR28": 3580032512}), row,
                        hopper_capture)
    assert "descriptor_address" not in ops.uniform


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_a0tx_is_complete_tx_not_expect(rules):
    # nvjet 2CTA: .A0TX posts the 16-byte CLC response as complete_tx
    # (0 arrivals) when CLC is not queried; .A0TR keeps expect_tx
    row = _row("SYNCS.ARRIVE.TRANS64.RED.A0TX", "RZ, [R6+UR9], R7",
               ("R6", "UR9", "R7"))
    rec = _rec({"R6": [0, 1 << 24], "R7": [16, 16]})
    rec.active_mask = rec.exec_mask = 0x3
    rec.mrefs = [(0x344c0, 0x10344c0)]
    ops = operand_facts(rec, row, rules)
    assert ops.uniform["arrival_count"] == 0
    assert ops.uniform["transaction_bytes"] == 16
    assert ops.uniform["transaction_complete"] == 1
    assert ops.lane["mbarrier_words"] == (0x344c0, 0x10344c0)


@pytest.mark.parametrize("rules", [hopper_capture, blackwell_capture])
def test_a0tr_stays_expect_tx(rules):
    row = _row("SYNCS.ARRIVE.TRANS64.RED.A0TR", "RZ, [R6+UR9], R7",
               ("R6", "UR9", "R7"))
    rec = _rec({"R6": [0, 1 << 24], "R7": [16, 16]})
    rec.active_mask = rec.exec_mask = 0x3
    rec.mrefs = [(0x344c0, 0x10344c0)]
    ops = operand_facts(rec, row, rules)
    assert ops.uniform["transaction_bytes"] == 16
    assert "transaction_complete" not in ops.uniform
