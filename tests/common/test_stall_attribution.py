"""cyclescope.stall: hardware scaling, loop regions, class mapping, the
per-case comparison and the suite summary on hand-computed inputs."""
import json
from pathlib import Path

import pytest

from cyclescope.stall.batch import run_compare, run_summarize
from cyclescope.stall.class_mapping import ClassMapping, load_mapping
from cyclescope.stall.hardware import load_hardware
from cyclescope.stall.ledger import load_ledger
from cyclescope.stall.regions import find_loops, mnemonic_root, region_of
from cyclescope.stall.summarize import family_of, percentile_90

COUNTER = "smsp__average_warp_latency_issue_stalled_{}.ratio"
SAMPLE_COLUMNS = ["barrier", "drain", "lg", "long_sb", "selected", "wait"]

# offset, sass, executed, samples
PCS = [
    (0x00, "MOV R1, R2", 4, {"selected": 2}),
    (0x10, "LDG.E R4, [R2]", 40, {"selected": 6}),
    (0x20, "FADD R5, R4, R5", 40, {"long_sb": 120, "wait": 30}),
    (0x30, "@P0 BRA 0x10", 40, {"selected": 8, "wait": 10}),
    (0x40, "STG.E [R6], R5", 4, {"lg": 20, "selected": 4}),
    (0x50, "EXIT", 4, {"drain": 20}),
]
BASE = 0x7F0000001000


def write_reference(directory, ordinal=1):
    directory.mkdir(parents=True)
    (directory / "timing.json").write_text(json.dumps(
        {"status": "success", "pick": {"ordinal": ordinal}}))
    names = ["selected", "long_scoreboard", "wait", "lg_throttle", "gmma"]
    values = {"selected": "10", "long_scoreboard": "60", "wait": "20",
              "lg_throttle": "10", "gmma": ""}
    header = ["ID", "Kernel Name", "sm__cycles_active.max",
              "smsp__inst_executed.sum"] + [COUNTER.format(n) for n in names]
    lines = ["==PROF== Connected to process 1",
             ",".join(f'"{h}"' for h in header),
             ",".join(['""', '""', '"cycle"', '"inst"'] +
                      ['"inst/warp"'] * len(names)),
             ",".join(['"0"', '"other"', '"9"', '"9"'] +
                      ['"1"'] * len(names)),
             ",".join([f'"{ordinal}"', '"kern"', '"400"', '"40"'] +
                      [f'"{values[n]}"' for n in names])]
    (directory / "ncu_metrics.csv").write_text("\n".join(lines) + "\n")
    header = ["Address", "Source", "# Samples", "Instructions Executed"] + \
        [f"stall_{c}" for c in SAMPLE_COLUMNS] + \
        [f"stall_{c} (Not Issued)" for c in SAMPLE_COLUMNS]
    lines = ["==PROF== Disconnected", '"Kernel Name","kern",',
             ",".join(f'"{h}"' for h in header)]
    for offset, sass, executed, samples in PCS:
        cells = [hex(0x55AA0000 + offset), sass, str(sum(samples.values())),
                 str(executed)]
        cells += [str(samples.get(c, 0)) for c in SAMPLE_COLUMNS]
        cells += ["999"] * len(SAMPLE_COLUMNS)   # not-issued: never read
        lines.append(",".join(f'"{c}"' for c in cells))
    (directory / "ncu_source.csv").write_text("\n".join(lines) + "\n")


def write_capture(directory):
    (directory / "functions").mkdir(parents=True)
    (directory / "functions.jsonl").write_text(
        json.dumps({"function_id": 1, "name": "other", "addr": "0x100",
                    "instructions": 4}) + "\n" +
        json.dumps({"function_id": 2, "name": "kern", "addr": hex(BASE),
                    "instructions": len(PCS)}) + "\n")
    (directory / "functions" / "2.tsv").write_text("".join(
        f"{i}\t{offset:x}\t00\t{sass} ;\t\t0\n"
        for i, (offset, sass, _, _) in enumerate(PCS)))


def write_ledger(path):
    rows = [
        ("residency", BASE + 0x00, "MOV", "dispatch", 400, 4),
        ("residency", BASE + 0x00, "MOV", "issued", 4, 4),
        ("residency", BASE + 0x10, "LDG", "issued", 40, 40),
        ("residency", BASE + 0x20, "FADD", "issued", 40, 40),
        ("residency", BASE + 0x20, "FADD", "reg:LDG:l2", 160, 40),
        ("residency", BASE + 0x30, "BRA", "issued", 40, 40),
        ("residency", BASE + 0x30, "BRA", "reg:FADD", 40, 40),
        ("residency", BASE + 0x40, "STG", "issued", 4, 4),
        ("residency", BASE + 0x40, "STG", "store_buffer", 80, 4),
        ("residency", BASE + 0x50, "EXIT", "issued", 4, 4),
        ("residency", 0, "?", "reg:IADD3", 8, 2),
        ("transition", BASE + 0x00, BASE + 0x10, 4),
        ("transition", BASE + 0x10, BASE + 0x20, 40),
        ("transition", BASE + 0x20, BASE + 0x30, 40),
        ("transition", BASE + 0x30, BASE + 0x10, 36),
        ("transition", BASE + 0x30, BASE + 0x40, 4),
        ("transition", BASE + 0x40, BASE + 0x50, 4),
    ]
    path.parent.mkdir(parents=True)
    with path.open("w") as fh:
        fh.write("#schema\tcyclescope.stall-ledger.v1\n#warps\t4\n#cycles\t500\n")
        for row in rows:
            if row[0] == "residency":
                fh.write(f"residency\t{row[1]:x}\t{row[2]}\t{row[3]}"
                         f"\t{row[4]}\t{row[5]}\n")
            else:
                fh.write(f"transition\t{row[1]:x}\t{row[2]:x}\t{row[3]}\n")


def test_hardware_scale(tmp_path):
    write_reference(tmp_path / "ref" / "c")
    hw = load_hardware(tmp_path / "ref" / "c")
    assert hw.kernel == "kern" and hw.cycles == 400
    assert hw.residency_per_warp == 100 and hw.warps == 4
    assert hw.sample_total == 220
    # 100 cycles per warp over the 200 non-drain samples
    assert hw.cycles_per_sample == pytest.approx(0.5)
    assert hw.reference_consistency() == pytest.approx(0.0)
    assert hw.selected_scale_ratio() == pytest.approx(1.0)
    assert [p.offset for p in hw.pcs] == [o for o, *_ in PCS]
    assert hw.pcs[2].samples == {"long_scoreboard": 120, "wait": 30}


def test_hardware_fail_closed(tmp_path):
    write_reference(tmp_path / "c")
    text = (tmp_path / "c" / "ncu_source.csv").read_text()
    (tmp_path / "c" / "ncu_source.csv").write_text(
        text.replace('"stall_wait"', '"stall_novel"'))
    with pytest.raises(ValueError, match="unsupported:stall_sample_class"):
        load_hardware(tmp_path / "c")
    (tmp_path / "c" / "timing.json").write_text(json.dumps({"pick": {}}))
    with pytest.raises(ValueError, match="missing_fact:reference_launch"):
        load_hardware(tmp_path / "c")


def test_regions():
    assert mnemonic_root("@!P0  BRA.U !UP2, 0x10 ;") == "BRA"
    assert mnemonic_root("HGMMA.64x128x16.F32 R4, R8") == "HGMMA"
    sass = {0: "MOV R1, R2", 16: "LDG R1, [R2]", 32: "BRA 0x10",
            48: "SYNCS.TRYWAIT P0, [UR4]", 64: "@!P0 BRA 0x30",
            80: "UTMALDG.2D [UR4], [UR6]", 96: "SYNCS.TRYWAIT P0, [UR4]",
            112: "@!P0 BRA 0x50", 128: "EXIT"}
    transitions = {(0, 16): 1, (16, 32): 5, (32, 16): 4, (32, 48): 1,
                   (48, 64): 9, (64, 48): 8, (64, 80): 1, (80, 96): 3,
                   (96, 112): 3, (112, 80): 2, (112, 128): 1,
                   (128, 0): 0}                 # zero count: not an edge
    loops = find_loops(transitions, sass)
    assert [(l.kind, sorted(l.pcs)) for l in loops] == [
        ("loop", [16, 32]),                     # memory work
        ("wait_loop", [48, 64]),                # only polls
        ("loop", [80, 96, 112]),                # polls inside a work loop
    ]
    regions = region_of(loops)
    assert {pc for pc, r in regions.items() if r == "loop"} == \
        {16, 32, 80, 96, 112}
    # a pc without text counts as work; a self loop is a loop
    loops = find_loops({(0, 0): 3, (16, 32): 1, (32, 16): 1},
                       {16: "SYNCS.TRYWAIT P0, [UR4]"})
    assert [(l.kind, sorted(l.pcs)) for l in loops] == [
        ("loop", [0]), ("loop", [16, 32])]


def test_basic_blocks():
    from cyclescope.stall.regions import basic_blocks
    transitions = {(0, 16): 1, (16, 32): 5, (32, 16): 4, (32, 48): 1,
                   (48, 96): 1}
    blocks = basic_blocks([0, 16, 32, 48, 96, 112], transitions)
    # 16 is a branch target; 48 follows the branch source 32; 96 is a
    # target after a gap; 112 runs on from 96
    assert blocks == {0: 0, 16: 16, 32: 16, 48: 48, 96: 96, 112: 96}


def test_sites():
    from cyclescope.stall.regions import site_kind, sites
    assert site_kind("SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR6]") == \
        "mbarrier_poll"
    assert site_kind("@!P0 NANOSLEEP.SYNCS 0xc350") == "mbarrier_poll"
    assert site_kind("@!P1 SYNCS.PHASECHK.TRANS64 P1, [R5+URZ], R4") == \
        "mbarrier_poll"
    assert site_kind("SYNCS.ARRIVE.TRANS64 RZ, [UR6]") == "mbarrier_other"
    assert site_kind("LDTM.x4 R8, tmem[UR5+0x40]") == "tensor_memory"
    assert site_kind("WARPGROUP.DEPBAR.LE gsb0, 0x0") == "tensor"
    assert site_kind("HGMMA.64x128x16.F32.BF16 R24, gdesc[UR16]") == "tensor"
    assert site_kind("UCGABAR_WAIT UR4, UR5") == "tensor"
    assert site_kind("BAR.SYNC 0x0, 0x80") == "cta_barrier"
    assert site_kind("DEPBAR.LE SB0, 0x0") == "async_wait"
    assert site_kind("UTMALDG.2D [UR4], [UR6]") == "tma"
    assert site_kind("ULDC UR4, c[0x0][0x10]") is None
    assert site_kind("FADD R1, R2, R3") is None
    sass = {0: "MOV R1, R2", 16: "BAR.SYNC 0x0, 0x80", 32: "BRA 0x60",
            48: "IADD3 R1, R1, 0x1, RZ", 64: "LDG.E R4, [R2]",
            96: "FADD R5, R4, R5", 112: "FADD R5, R4, R5",
            128: "STG.E [R6], R5", 144: "EXIT"}
    transitions = {(0, 16): 1, (16, 32): 1, (32, 96): 1, (48, 64): 1,
                   (96, 112): 3, (112, 96): 2, (112, 128): 1,
                   (128, 144): 1}
    assert sites(sass, transitions, sass) == {
        0: (0, "compute"), 16: (16, "cta_barrier"), 32: (16, "cta_barrier"),
        # 48 follows a branch source without a fall-through transition
        48: (48, "compute"), 64: (64, "global_load"),
        # 96: after a gap, branch target only
        96: (96, "compute"), 112: (96, "compute"),
        # 128 follows the branch source 112 by an observed fall-through
        128: (128, "global_store"), 144: (144, "exit")}
    # a run without an anchor continues the site it falls through from
    sass = {0: "LDG.E R4, [R2]", 16: "@P0 BRA 0x40", 32: "FADD R5, R4, R5"}
    assert sites(sass, {(0, 16): 2, (16, 32): 1, (16, 64): 1}, sass) == {
        0: (0, "global_load"), 16: (0, "global_load"),
        32: (0, "global_load")}


def test_sites_after_barrier():
    from cyclescope.stall.regions import sites
    # the instruction right after a BAR stays in the BAR's site; the
    # anchor after that one starts its own
    sass = {0: "BAR.SYNC 0x0, 0x80", 16: "LDS R1, [R2]", 32: "LDS R3, [R2]",
            48: "BAR.SYNC.DEFER_BLOCKING 0x8, 0x80", 64: "R2UR UR5, R3",
            80: "LDTM.x4 R8, tmem[UR5]"}
    assert sites(sass, {}, sass) == {
        0: (0, "cta_barrier"), 16: (0, "cta_barrier"),
        32: (32, "shared_local"), 48: (48, "cta_barrier"),
        64: (48, "cta_barrier"), 80: (80, "tensor_memory")}
    sass = {0: "DEPBAR.LE SB0, 0x0", 16: "BAR.SYNC 0x0, 0x80",
            32: "LDGSTS.E.BYPASS.128 [R4], [R2]", 48: "LDS R1, [R2]"}
    assert sites(sass, {}, sass) == {
        0: (0, "async_wait"), 16: (0, "async_wait"),
        32: (32, "async_wait"), 48: (48, "shared_local")}


def test_sites_branch_on_try_wait():
    from cyclescope.stall.regions import sites
    # the branch on a try-wait's predicate belongs to the try-wait's
    # site across an anchor (0x20) and across a run boundary (0x50 is a
    # branch target); a branch on another predicate does not move
    sass = {0x00: "SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [R3+UR29], R4",
            0x10: "UIADD3 UR29, UPT, UPT, UR29, UR24, URZ",
            0x20: "LDS R1, [R2]", 0x30: "@!P1 BRA 0x700",
            0x40: "ISETP.GE.U32.AND P0, PT, R7, 0x80, PT",
            0x50: "BRA !P0, 0x900",
            0x60: "@!P2 SYNCS.PHASECHK.TRANS64 P2, [R5+URZ], R4",
            0x70: "LDG.E R4, [R2]", 0x80: "FADD R5, R4, R5",
            0x90: "BRA !P2, 0x300"}
    transitions = {(0xA0, 0x80): 3}
    assert sites(sass, transitions, sass) == {
        0x00: (0x00, "mbarrier_poll"), 0x10: (0x00, "mbarrier_poll"),
        0x20: (0x20, "shared_local"), 0x30: (0x00, "mbarrier_poll"),
        0x40: (0x20, "shared_local"), 0x50: (0x20, "shared_local"),
        0x60: (0x60, "mbarrier_poll"), 0x70: (0x70, "global_load"),
        0x80: (0x80, "compute"), 0x90: (0x60, "mbarrier_poll")}


def test_place_unissued():
    from cyclescope.stall.regions import branch_text_shift, place_unissued
    # listing targets are printed 0x100 above the function offsets.
    # 0x00..0x20 loop body; 0x30 SYNCS / 0x40 BRA: inline try-wait the
    # replay dropped; 0x50 loop tail; 0x200.. out-of-line wait path that
    # returns to 0x50; 0x300 a never-issued path that jumps nowhere known
    sass = {0x00: "MOV R1, R2", 0x10: "LDG.E R4, [R2]", 0x20: "IADD3 R1, R1",
            0x30: "SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [R5]",
            0x40: "@!P1 BRA 0x300", 0x50: "@P0 BRA 0x110", 0x60: "EXIT",
            0x200: "SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [R5]",
            0x210: "@!P1 NANOSLEEP.SYNCS 0x989680", 0x220: "@!P1 BRA 0x300",
            0x230: "BRA 0x150",
            0x300: "NOP", 0x310: "BRA 0x900"}
    transitions = {(0x00, 0x10): 1, (0x10, 0x20): 5, (0x20, 0x50): 5,
                   (0x50, 0x10): 4, (0x50, 0x60): 1}
    assert branch_text_shift(transitions, sass) == 0x100
    assert branch_text_shift({(0x00, 0x10): 1}, sass) is None
    regions = region_of(find_loops(transitions, sass))
    assert regions == {0x10: "loop", 0x20: "loop", 0x50: "loop"}
    issued = [0x00, 0x10, 0x20, 0x50, 0x60]
    placed, far = place_unissued(regions, issued, sass, sass, transitions)
    # near rule: 0x30 / 0x40 follow the loop pc 0x20
    assert placed[0x30] == placed[0x40] == "loop"
    # jump rule: the wait path's `BRA 0x150` returns to 0x50 (loop)
    assert placed[0x200] == placed[0x210] == placed[0x220] == \
        placed[0x230] == "loop"
    # neither: stays outside and is reported
    assert far == {0x300, 0x310} and 0x300 not in placed
    assert 0x60 not in placed and 0x00 not in placed


def test_poll_blocks():
    from cyclescope.stall.regions import basic_blocks, poll_blocks
    sass = {0: "MOV R1, R2", 16: "SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR6]",
            32: "@!P0 NANOSLEEP.SYNCS 0xc350", 48: "BRA !P0, 0x10",
            64: "SYNCS.ARRIVE.TRANS64 RZ, [UR6]", 80: "EXIT"}
    blocks = basic_blocks(sass, {(48, 16): 9, (0, 16): 1, (48, 64): 1})
    assert blocks == {0: 0, 16: 16, 32: 16, 48: 16, 64: 64, 80: 64}
    # the arrive is not a poll: only the run with the try-wait counts
    assert poll_blocks(blocks, sass) == {16}


def test_class_mapping():
    mapping = load_mapping()
    expect = {
        "issued": "selected", "dispatch": None,
        "reg:LDG:dram": "long_scoreboard", "reg:LD": "long_scoreboard",
        "reg:LDS": "short_scoreboard", "reg:LDS:x": "short_scoreboard",
        "reg:LDGSTS": "long_scoreboard", "reg:ATOMS": "short_scoreboard",
        "reg:ATOMG": "long_scoreboard", "reg:ULDC:miss": "imc_miss",
        "reg:ULDC.64:miss": "imc_miss", "reg:ULDC:hit": "short_scoreboard",
        "reg:IMAD": "wait", "unit:alu_smsp": "math_pipe_throttle",
        "unit:lsu_smsp": "lg_throttle", "unit_outbid:lsu_smsp": "not_selected",
        "issue_slot": "not_selected", "control_floor": "wait",
        "control:branch": "branch_resolving",
        "control:yield": "branch_resolving",
        "control:bsync": "branch_resolving",
        "suspended:sleep": "sleeping",
        "semantic:mbarrier_wait": "long_scoreboard",
        "semantic:cta_barrier_wait": "barrier",
        "cta_barrier": "barrier", "cluster_barrier": "barrier",
        "arch:warpgroup_wait": "barrier", "arch:wgmma_rendezvous": "gmma",
        "reg:LDL:l2": "long_scoreboard", "scoreboard_slot": "long_scoreboard",
        "other:anything": "misc",
        # placement v1 (rules T, M and the unit mapping addenda)
        "control:branch_fallthrough": "wait",
        "control:branch_refetch": "no_instruction",
        "control:membar:MEMBAR.SC.GPU": "membar",
        "control:membar:MEMBAR.ALL.CTA": "long_scoreboard",
        "control:membar:MEMBAR.SC.CTA": "wait",
        "control:membar:MEMBAR.SC.VC": "membar",
        "control:fence:FENCE.VIEW.ASYNC.S": "long_scoreboard",
        "control:fence:FENCE.G.XYZ": "membar",
        "unit:fence_view_async_sm": "long_scoreboard",
        "unit:membar_cta_smsp": "membar",
    }
    for reason, cls in expect.items():
        assert mapping.classify(reason) == cls, reason
    assert mapping.issue_pc_shift == 1
    strict = ClassMapping({"schema_version": "cyclescope.stall-class-mapping.v1",
                           "rules": [{"reason": "^issued$",
                                      "class": "selected",
                                      "evidence": "calibrated"}]})
    with pytest.raises(ValueError, match="unsupported:stall_reason_unmapped"):
        strict.classify("reg:LDG")
    with pytest.raises(ValueError, match="unsupported:stall_class"):
        ClassMapping({"schema_version": "cyclescope.stall-class-mapping.v1",
                      "rules": [{"reason": "x", "class": "novel",
                                 "evidence": "hypothesis"}]})
    with pytest.raises(ValueError, match="missing_fact:stall_rule_evidence"):
        ClassMapping({"schema_version": "cyclescope.stall-class-mapping.v1",
                      "rules": [{"reason": "x", "class": "wait"}]})


def test_ledger_fail_closed(tmp_path):
    write_ledger(tmp_path / "a" / "stall.tsv")
    ledger = load_ledger(tmp_path / "a" / "stall.tsv")
    assert ledger.warps == 4 and ledger.cycles == 500
    assert ledger.residency[(BASE + 0x20, "reg:LDG:l2")] == [160, 40]
    assert ledger.transitions[(BASE + 0x30, BASE + 0x10)] == 36
    text = (tmp_path / "a" / "stall.tsv").read_text()
    bad = tmp_path / "bad.tsv"
    bad.write_text(text.replace("stall-ledger.v1", "stall-ledger.v0"))
    with pytest.raises(ValueError, match="unsupported:stall_ledger_schema"):
        load_ledger(bad)
    bad.write_text(text + "residency\t10\tMOV\n")
    with pytest.raises(ValueError, match="unsupported:stall_ledger_row"):
        load_ledger(bad)


def test_class_mapping_operation():
    mapping = load_mapping()
    assert mapping.classify("unit:xu_smsp", "MUFU.EX2") == "mio_throttle"
    assert mapping.classify("unit:xu_smsp", "MUFU") == "mio_throttle"
    assert mapping.classify("unit:xu_smsp", "F2I.FTZ.U32.F32.TRUNC") == \
        "math_pipe_throttle"
    assert mapping.classify("unit:xu_smsp") == "math_pipe_throttle"
    assert mapping.classify("unit:alu_smsp", "MUFU.EX2") == \
        "math_pipe_throttle"
    assert mapping.rule_index("unit:xu_smsp", "MUFU.EX2") != \
        mapping.rule_index("unit:xu_smsp", "F2I.FTZ.U32.F32.TRUNC")
    with pytest.raises(ValueError,
                       match=r"unsupported:stall_rule_operation:x \(src\)"):
        ClassMapping({"schema_version": "cyclescope.stall-class-mapping.v1",
                      "rules": [{"reason": "x", "operation": "",
                                 "class": "wait",
                                 "evidence": "hypothesis"}]}, "src")
    with pytest.raises(ValueError,
                       match="unsupported:stall_rule_operation"):
        ClassMapping({"schema_version": "cyclescope.stall-class-mapping.v1",
                      "rules": [{"reason": "x", "operation": 3,
                                 "class": "wait",
                                 "evidence": "hypothesis"}]})


def test_class_mapping_warp_queue():
    mapping = load_mapping()
    assert mapping.classify("warp_queue:lg") == "lg_throttle"
    assert mapping.classify("warp_queue:mio") == "mio_throttle"
    assert mapping.classify("warp_queue:mio", "STS") == "mio_throttle"


def test_compare_operation_class(tmp_path):
    write_reference(tmp_path / "ref" / "fam_case_00")
    write_capture(tmp_path / "cap" / "fam_case_00")
    ledger = tmp_path / "sim" / "fam_case_00" / "out-cpp" / "stall.tsv"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        "#schema\tcyclescope.stall-ledger.v1\n#warps\t4\n#cycles\t500\n"
        f"residency\t{BASE + 0x10:x}\tLDG\tissued\t40\t40\n"
        f"residency\t{BASE + 0x10:x}\tLDG\tunit:xu_smsp\t20\t2\n"
        f"residency\t{BASE + 0x20:x}\tMUFU.EX2\tunit:xu_smsp\t40\t4\n")
    assert run_compare(["fam_case_00"], tmp_path / "sim", tmp_path / "ref",
                tmp_path / "cap", tmp_path / "out",
                _no_shift_mapping(tmp_path)) == {"ok": 1}
    result = json.loads(
        (tmp_path / "out" / "cases" / "fam_case_00.json").read_text())
    got = {(r["reason"], r["class"]): r["cycles"]
           for r in result["reasons"]}
    assert got == {("issued", "selected"): 10.0,
                   ("unit:xu_smsp", "mio_throttle"): 10.0,
                   ("unit:xu_smsp", "math_pipe_throttle"): 5.0}


def _no_shift_mapping(tmp_path):
    """The shipped mapping without issue_pc_shift: the hand-computed
    tables of the compare tests are those of an issue cycle booked on
    the instruction's own pc."""
    from cyclescope.stall.class_mapping import DEFAULT_PATH
    document = json.loads(Path(DEFAULT_PATH).read_text())
    document.pop("issue_pc_shift", None)
    path = tmp_path / "mapping_no_shift.json"
    path.write_text(json.dumps(document))
    return path


def test_compare_and_summarize(tmp_path):
    write_reference(tmp_path / "ref" / "fam_case_00")
    write_capture(tmp_path / "cap" / "fam_case_00")
    write_ledger(tmp_path / "sim" / "fam_case_00" / "out-cpp" / "stall.tsv")
    statuses = run_compare(["fam_case_00", "fam_case_01"], tmp_path / "sim",
                           tmp_path / "ref", tmp_path / "cap",
                           tmp_path / "out",
                           _no_shift_mapping(tmp_path))
    assert statuses == {"ok": 1, "missing_fact:stall_ledger": 1}
    result = json.loads(
        (tmp_path / "out" / "cases" / "fam_case_00.json").read_text())
    assert result["function"] == "kern"
    # hardware: 0.5 cycles per warp per sample
    assert result["table"]["loop"] == {
        "long_scoreboard": [60.0, 40.0], "selected": [7.0, 30.0],
        "wait": [20.0, 10.0]}
    assert result["table"]["outside"] == {
        "drain": [10.0, 0.0], "lg_throttle": [10.0, 20.0],
        "selected": [3.0, 3.0]}
    assert result["table"]["unplaced"] == {"wait": [0.0, 2.0]}
    m = result["metrics"]
    assert m["attribution_error"] == pytest.approx(75 / 110)
    assert m["attribution_error_loop"] == pytest.approx(53 / 110)
    assert m["attribution_error_outside"] == pytest.approx(22 / 110)
    assert m["residency_ratio"] == pytest.approx(105 / 110)
    assert m["residency_ratio_loop"] == pytest.approx(80 / 87)
    assert m["residency_ratio_outside"] == pytest.approx(1.0)
    assert m["hardware_loop_share"] == pytest.approx(87 / 110)
    assert m["class_error"] == pytest.approx(71 / 110)
    assert m["kind_class_error"] == pytest.approx(87 / 110)
    assert m["pc_class_error"] == pytest.approx(87 / 110)
    assert m["block_class_error"] == pytest.approx(75 / 110)
    assert m["site_class_error"] == pytest.approx(77 / 110)
    assert m["kind_residency_error"] == pytest.approx(63 / 110)
    assert result["scope"] == "kernel" and result["floor"] is None
    assert m["cycle_ratio"] == pytest.approx(1.25)
    # loop shares: hw 7/87 60/87 20/87, sim 30/80 40/80 10/80
    assert m["shape_distance_loop"] == pytest.approx(
        0.5 * (abs(7 / 87 - 30 / 80) + abs(60 / 87 - 40 / 80)
               + abs(20 / 87 - 10 / 80)))
    # per pc |sim - hw|: 0, 7, 25, 11, 9, 9 and 2 unplaced
    assert m["pc_residency_error"] == pytest.approx(63 / 110)
    # blocks: {0x00}, {0x10..0x30}, {0x40, 0x50}: |1-1| + |87-80| + |22-22|
    # and 2 unplaced
    assert m["block_residency_error"] == pytest.approx(9 / 110)
    # sites: compute {0x00}, global_load {0x10..0x30}, global_store
    # {0x40}, exit {0x50}: |1-1| + |87-80| + |12-21| + |10-1|, 2 unplaced
    assert m["site_residency_error"] == pytest.approx(27 / 110)
    assert {(s["region"], s["kind"]): (s["hw_issue"], s["hw_wait"],
                                       s["sim_issue"], s["sim_wait"])
            for s in result["sites"]} == {
        ("outside", "compute"): (1.0, 0.0, 1.0, 0.0),
        ("loop", "global_load"): (7.0, 80.0, 30.0, 50.0),
        ("outside", "global_store"): (2.0, 10.0, 1.0, 20.0),
        ("outside", "exit"): (0.0, 10.0, 1.0, 0.0)}
    by_site = {(s["region"], s["kind"]): s for s in result["sites"]}
    assert by_site[("loop", "global_load")]["hw_classes"] == {
        "long_scoreboard": 60.0, "wait": 20.0}
    assert by_site[("loop", "global_load")]["sim_reasons"] == {
        "reg:LDG:l2": 40.0, "reg:FADD": 10.0}
    assert by_site[("outside", "exit")]["hw_classes"] == {"drain": 10.0}
    assert result["site_rows"][0] == {
        "first": 0x40, "kind": "global_store", "region": "outside",
        "sass": "STG.E [R6], R5", "hw": 12.0, "sim": 21.0}
    # loop block waits: hw long_scoreboard 60 + wait 20, engine
    # reg:LDG:l2 40 + reg:FADD 10 -> overlap 50 spread by both shares;
    # tail block: hw lg 10 + drain 10, engine store_buffer 20
    pairs = {(c["reason"], c["class"]): c["cycles"]
             for c in result["cooccurrence"]}
    assert pairs == {
        ("reg:LDG:l2", "long_scoreboard"): pytest.approx(30.0),
        ("reg:LDG:l2", "wait"): pytest.approx(10.0),
        ("reg:FADD", "long_scoreboard"): pytest.approx(7.5),
        ("reg:FADD", "wait"): pytest.approx(2.5),
        ("store_buffer", "lg_throttle"): pytest.approx(10.0),
        ("store_buffer", "drain"): pytest.approx(10.0),
    }
    assert result["engine"]["excluded_per_warp"] == pytest.approx(100.0)
    assert result["engine"]["unplaced_per_warp"] == pytest.approx(2.0)
    assert result["alignment"] == {
        "function_instructions": 6, "mnemonic_mismatches": 0,
        "execution_count_error": 0.0, "execution_count_error_all": 0.0,
        "warps_ratio": 1.0, "hardware_unissued_share": 0.0,
        "hardware_unissued_unplaced_share": 0.0}
    assert result["loops"] == [{"index": 0, "kind": "loop", "first": 0x10,
                                "last": 0x30, "pcs": 3, "hw": 87.0,
                                "sim": 80.0}]
    summary = run_summarize(tmp_path / "out", "unit")
    assert summary["cases"] == 2 and summary["statuses"]["ok"] == 1
    assert summary["overall"]["attribution_error"]["mean"] == \
        pytest.approx(75 / 110)
    gaps = {(g["region"], g["class"]): g for g in summary["class_gaps"]}
    assert gaps[("loop", "selected")]["mean_signed"] == \
        pytest.approx(23 / 110)
    assert gaps[("loop", "long_scoreboard")]["cases_below"] == 1
    sites = {(g["region"], g["kind"]): g for g in summary["site_gaps"]}
    assert sites[("loop", "global_load")]["mean_signed"] == \
        pytest.approx(-7 / 110)
    assert sites[("loop", "global_load")]["signed_issue"] == \
        pytest.approx(23 / 110)
    assert sites[("loop", "global_load")]["signed_wait"] == \
        pytest.approx(-30 / 110)
    assert summary["overall"]["site_residency_error"]["mean"] == \
        pytest.approx(27 / 110)
    parts = (tmp_path / "out" / "site_parts.tsv").read_text()
    assert "loop\tglobal_load\thw\tlong_scoreboard\t0.54545\n" in parts
    assert "loop\tglobal_load\tsim\treg:LDG:l2\t0.36364\n" in parts
    assert "loop\tglobal_load\t1\t" in \
        (tmp_path / "out" / "site_gaps.tsv").read_text()
    reasons = (tmp_path / "out" / "reason_classes.tsv").read_text()
    assert "reg:LDG:l2\t1\t" in reasons
    assert "long_scoreboard=0.750 wait=0.250" in reasons
    kinds = [l.split("\t") for l in (
        tmp_path / "out" / "kind_gaps.tsv").read_text().splitlines()]
    fadd = next(r for r in kinds if r[:2] == ["FADD", "long_scoreboard"])
    assert fadd[2] == "0.54545" and fadd[3] == "0.36364"
    assert "loop\tglobal_load\t" in \
        (tmp_path / "out" / "site_pc.tsv").read_text()
    assert summary["floor"] is None
    lines = (tmp_path / "out" / "cases.tsv").read_text().splitlines()
    assert lines[0].split("\t")[:3] == ["case", "family", "status"]
    assert len(lines) == 3


def test_summary_helpers():
    assert family_of("cublas_bf16_bigK_00") == "cublas"
    assert family_of("CUF-C84A-SPARSE-R01") == "CUF"
    assert family_of("FIF-BS_X-R01") == "FIF"
    assert percentile_90([float(i) for i in range(1, 11)]) == 9.0
    assert percentile_90([5.0]) == 5.0


def _probe_row(case, cooccurrence, hw_table, **over):
    """A compare result with only the fields the calibration reads."""
    result = {
        "case": case, "status": "ok",
        "alignment": {"mnemonic_mismatches": 0,
                      "execution_count_error": 0.01, "warps_ratio": 1.0},
        "hardware": {"reference_consistency": 0.03, "samples": 9000},
        "metrics": {"cycle_ratio": 1.02},
        "table": {"loop": {c: [v, 0.0] for c, v in hw_table.items()}},
        "cooccurrence": [{"reason": r, "class": c, "cycles": v}
                         for (r, c), v in cooccurrence.items()],
    }
    for key, value in over.items():
        section, name = key.split("__")
        result[section][name] = value
    return result


def test_calibration_table():
    from cyclescope.stall.calibrate import calibration_table, row_status
    # a mapping of its own: the shipped file changes as rules calibrate
    mapping = ClassMapping({
        "schema_version": "cyclescope.stall-class-mapping.v1",
        "rules": [{"reason": r, "class": c, "evidence": "hypothesis"}
                  for r, c in (("^fetch", "no_instruction"),
                               ("^arch", "gmma"),
                               ("^control_floor", "wait"), ("", "misc"))]})
    table = {"barrier": 80.0, "wait": 10.0, "selected": 10.0}
    pairs = {("arch:warpgroup_wait", "barrier"): 60.0,
             ("arch:warpgroup_wait", "wait"): 5.0,
             ("control_floor", "wait"): 3.0,
             ("control_floor", "barrier"): 3.0}
    rows = [_probe_row("a", pairs, table), _probe_row("b", pairs, table),
            _probe_row("slow", pairs, table, metrics__cycle_ratio=1.3),
            _probe_row("thin", pairs, table, hardware__samples=500),
            _probe_row("other", pairs, table,
                       alignment__execution_count_error=0.2),
            _probe_row("shaky", pairs, table)]
    repeat = [_probe_row(r["case"], pairs, table) for r in rows[:5]]
    repeat.append(_probe_row("shaky", pairs,
                             {"barrier": 70.0, "wait": 20.0,
                              "selected": 10.0}))
    out = calibration_table(rows, repeat, mapping)
    assert out["rows"] == {"a": "usable", "b": "usable", "slow": "timing",
                           "thin": "thin", "other": "same_launch",
                           "shaky": "unstable"}
    assert row_status(rows[0], None) == "no_repeat"
    assert out["usable"] == 2
    rules = {r["rule"]: r for r in out["rules"]}
    arch = rules["^arch"]
    # 65 % of the residency overlaps, 60 / 65 of it on barrier
    assert arch["decision"] == "A" and arch["calibrated_class"] == "barrier"
    assert arch["class"] == "gmma" and arch["rows"] == 2
    assert arch["mean_overlap"] == pytest.approx(0.65)
    assert arch["shares"]["barrier"] == pytest.approx(60 / 65)
    floor = rules["^control_floor"]
    assert floor["decision"] == "B" and floor["calibrated_class"] is None
    assert floor["shares"] == {"wait": pytest.approx(0.5),
                               "barrier": pytest.approx(0.5)}
    assert rules["^fetch"]["decision"] == "C"


def test_hardware_scale_per_class(tmp_path):
    """A class is scaled by its own counter; classes only the samples
    report take the pooled scale; a counter class without samples is
    listed as unsampled."""
    write_reference(tmp_path / "c")
    path = tmp_path / "c" / "ncu_metrics.csv"
    lines = path.read_text().splitlines()
    # selected 10, long_scoreboard 90 (was 60), wait 20, lg_throttle 10,
    # gmma 5 (never sampled)
    lines[-1] = lines[-1].replace('"60"', '"90"').replace(
        '"10",""', '"10","5"')
    path.write_text("\n".join(lines) + "\n")
    hw = load_hardware(tmp_path / "c")
    scales = hw.class_scales()
    assert scales["long_scoreboard"] == pytest.approx(90 / 120)
    assert scales["selected"] == pytest.approx(10 / 20)
    assert scales["wait"] == pytest.approx(20 / 40)
    # pooled over the classes with a counter and samples: 130 / 200
    assert hw.cycles_per_sample == pytest.approx(0.65)
    assert scales["drain"] == pytest.approx(0.65)
    assert hw.unsampled() == {"gmma": 5.0}
    ratios = hw.class_scale_ratios()
    assert ratios["long_scoreboard"] == pytest.approx(0.75 / 0.65)
    assert "drain" not in ratios
    # every counter class with samples reproduces its counter
    total = {}
    for pc in hw.pcs:
        for cls, n in pc.samples.items():
            total[cls] = total.get(cls, 0.0) + n * scales[cls]
    assert total["long_scoreboard"] == pytest.approx(90)
    assert total["selected"] == pytest.approx(10)


def _case_tree(tmp_path, name="fam_case_00"):
    write_reference(tmp_path / "ref" / name)
    write_capture(tmp_path / "cap" / name)
    write_ledger(tmp_path / "sim" / name / "out-cpp" / "stall.tsv")


def _run(tmp_path, out="out", **kwargs):
    if "mapping_path" not in kwargs:
        kwargs["mapping_path"] = _no_shift_mapping(tmp_path)
    status = run_compare(["fam_case_00"], tmp_path / "sim", tmp_path / "ref",
                tmp_path / "cap", tmp_path / out, **kwargs)
    return json.loads((tmp_path / out / "cases" /
                       "fam_case_00.json").read_text())


def _shifted_mapping(tmp_path, shift):
    from cyclescope.stall.class_mapping import DEFAULT_PATH
    document = json.loads(Path(DEFAULT_PATH).read_text())
    document["issue_pc_shift"] = shift
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(document))
    return path


def test_issue_pc_shift(tmp_path):
    _case_tree(tmp_path)
    path = _shifted_mapping(tmp_path, {
        "instructions": 1, "evidence": "calibrated", "source": "test"})
    result = _run(tmp_path, mapping_path=path)
    m = result["metrics"]
    assert m["pc_residency_error"] == pytest.approx(67 / 110)
    assert m["attribution_error"] == pytest.approx(75 / 110)
    assert result["alignment"]["execution_count_error"] == 0.0
    path = _shifted_mapping(tmp_path, {"instructions": 1})
    with pytest.raises(ValueError, match="missing_fact:stall_rule_evidence"):
        load_mapping(path)
    path = _shifted_mapping(tmp_path, {
        "instructions": -1, "evidence": "calibrated", "source": "test"})
    with pytest.raises(ValueError,
                       match="unsupported:stall_issue_pc_shift"):
        load_mapping(path)


def test_reference_floor(tmp_path):
    _case_tree(tmp_path)
    write_reference(tmp_path / "rep" / "fam_case_00")
    floor = _run(tmp_path, "out_same", repeat_root=tmp_path / "rep")["floor"]
    assert set(floor) == {"class", "region_class", "kind_class",
                          "site_class", "block_class", "pc_class", "kind",
                          "site", "block", "pc"}
    assert all(v == pytest.approx(0.0) for v in floor.values())
    # a repeat whose FADD row samples differently
    write_reference(tmp_path / "rep2" / "fam_case_00")
    source = tmp_path / "rep2" / "fam_case_00" / "ncu_source.csv"
    source.write_text("\n".join(
        l.replace('"120"', '"60"').replace('"30"', '"90"')
        if "FADD" in l else l for l in source.read_text().splitlines())
        + "\n")
    floor = _run(tmp_path, "out_diff", repeat_root=tmp_path / "rep2")["floor"]
    assert floor["pc_class"] > 0 and floor["pc"] > 0
    # another kernel name, or missing files: no floor, the case is ok
    write_reference(tmp_path / "rep3" / "fam_case_00")
    metrics = tmp_path / "rep3" / "fam_case_00" / "ncu_metrics.csv"
    metrics.write_text(metrics.read_text().replace('"kern"', '"other2"'))
    (tmp_path / "rep4" / "fam_case_00").mkdir(parents=True)
    (tmp_path / "rep4" / "fam_case_00" / "timing.json").write_text("{}")
    for name in ("rep3", "rep4"):
        result = _run(tmp_path, f"out_{name}", repeat_root=tmp_path / name)
        assert result["status"] == "ok" and result["floor"] is None


def test_loops_scope(tmp_path):
    _case_tree(tmp_path)
    result = _run(tmp_path, "out_loops", scope="loops")
    assert result["scope"] == "loops"
    # pcs 0x10-0x30 and the pc behind the loop (0x40): 87 + 12 = 99
    assert result["metrics"]["pc_residency_error"] == pytest.approx(52 / 99)
    status = run_compare(["fam_case_00"], tmp_path / "sim", tmp_path / "ref",
                         tmp_path / "cap", tmp_path / "out_bad",
                         _no_shift_mapping(tmp_path), scope="nowhere")
    assert status == {"unsupported:stall_compare_scope:nowhere": 1}


def test_instruction_kind():
    from cyclescope.stall.regions import instruction_kind
    assert instruction_kind(
        "@!P0 SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R9+URZ], R2") == \
        "SYNCS.PHASECHK"
    assert instruction_kind("BAR.SYNC.DEFER_BLOCKING 0x0") == "BAR.SYNC"
    assert instruction_kind("FFMA R1, R2, R3, R4") == "FFMA"
    assert instruction_kind("") == "?"
