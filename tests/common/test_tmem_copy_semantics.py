import json
import shutil
from pathlib import Path

import pytest

from cyclescope.convert import model
from cyclescope.convert.facts_format import FactsWriter
from cyclescope.convert.pipeline import convert_facts
from cyclescope.io.file_header import FileReader
from cyclescope.io.fixed_tables import TableReader
from cyclescope.job import segment_format as sf
from cyclescope.job.semantics import SemanticsKind
from cyclescope.arch.blackwell import semantics as bw_semantics
from cyclescope.convert.semantics_common import SemanticOperation

REPO = Path(__file__).resolve().parents[2]
FORMS = [("", 0), (".4dp256bit", (4 << 16) | 256),
         (".128dp128bit", (128 << 16) | 128),
         (".2x64dp128bit_lw02_lw13", (2 << 16) | 128),
         (".4x32dp128bit", (4 << 16) | 128)]

@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("builder", ["python", "cpp"])
def test_native_tmem_copy_semantic_rows(tmp_path, group, builder):
    if builder == "cpp" and not shutil.which("cyclescope-convert"):
        pytest.skip("cyclescope-convert not built")
    unit = model.ClusterUnit(cluster_id=0)
    for i, (suffix, _) in enumerate(FORMS):
        opcode = "UTCCP.T.S" + (".2CTA" if group == 2 else "") + suffix
        unit.statics.append(model.StaticInstruction(
            pc=16*i, code_address=0x100+16*i, opcode=opcode,
            operation_class="UTCCP", predicate="P0",
            operands=("tmem[UR0]", "gdesc[UR2]"),
            source_registers=("UR0", "UR2"),
            register_roles={"UR0": "address", "UR2": "address"},
            control=model.ControlWord(stall=1)))
    unit.statics.append(model.StaticInstruction(
        pc=80, code_address=0x150, opcode="EXIT", operation_class="EXIT"))
    for rank in range(group):
        unit.ctas.append(model.CtaFacts(
            cta_linear=rank, cta=(rank, 0, 0), cluster_rank=rank,
            sm_id=rank, peer_sm_id=1-rank if group == 2 else 0xFFFFFFFF,
            multicast_mask=(1 << group)-1, cta_group=group,
            is_pair_leader=int(rank == 0), tmem_columns=512))
        instructions = [model.DynamicInstruction(
            static_index=i, active_mask=1, predicate_mask=mask,
            operands=model.SemanticOperands(uniform={"cta_group": group}))
            for i in range(5) for mask in (1, 0)] if rank == 0 else []
        instructions.append(model.DynamicInstruction(static_index=5,
                            active_mask=0xFFFFFFFF, predicate_mask=0xFFFFFFFF))
        unit.warps.append(model.WarpStream(cta_linear=rank, cta=(rank, 0, 0),
                          warp_in_cta=0, sm_id=rank, warp_slot=0,
                          instructions=instructions))
    facts = tmp_path / "facts"
    facts.mkdir()
    FactsWriter().write_unit(unit, facts / "000000.gsc")
    (facts / "launch.json").write_text(json.dumps({
        "device": "b300", "sm_arch": "sm_103a", "grid": [group, 1, 1],
        "cluster": [group, 1, 1], "block": [32, 1, 1], "capture": "tmem_semantics"}))
    out = tmp_path / "out"
    convert_facts(facts, "blackwell", REPO / "configs/trees/b300.json", out,
                  cache_dir=None, builder=builder)
    launch = json.loads((out / "launch.json").read_text())
    assert not launch.get("missing_facts")
    assert not launch.get("unsupported")
    program = out / "segments/program.tables"
    segment = sf.read_segment(next((out / "segments").glob("*.seg")).read_bytes(),
                program=FileReader(program.read_bytes()) if program.is_file() else None)
    inst = TableReader(sf.INST, segment.require_table("INST"))
    semantics = TableReader(sf.SEMA, segment.require_table("SEMA"))
    for i, (_, shape) in enumerate(FORMS):
        active = inst.unpack(2*i)
        inactive = inst.unpack(2*i+1)
        assert active["semantics"] != 0xFFFFFFFF
        row = semantics.unpack(active["semantics"])
        assert row["kind"] == int(SemanticsKind.tmem_copy)
        assert (row["f0"], row["f1"]) == (shape, 1)
        assert inactive["semantics"] == 0xFFFFFFFF


@pytest.mark.parametrize("opcode,operation", [
    ("UTCOMMA.BLOCK16", SemanticOperation.umma_issue),
    ("UTCBAR", SemanticOperation.umma_commit),
])
def test_predicated_off_tensor_semantics_need_no_operand_facts(opcode, operation):
    static = model.StaticInstruction(pc=0, code_address=None, opcode=opcode,
                                     operation_class=opcode.split(".", 1)[0])
    dyn = model.DynamicInstruction(static_index=0, active_mask=0xFFFFFFFF,
                                   predicate_mask=0)
    assert bw_semantics.decode(static, dyn, None, operation, 0, 0) is None
