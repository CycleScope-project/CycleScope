from cyclescope.arch.hopper.rules import operation_class
from cyclescope.convert import model


def _s(op):
    return model.StaticInstruction(pc=0, code_address=None, opcode=op,
        operation_class=op.split(".", 1)[0], operands=(),
        source_registers=(), destination_registers=(), register_roles={})


def test_membar_fence_classes():
    assert operation_class(_s("MEMBAR.ALL.GPU")) == "MEMBAR.ALL.GPU"
    assert operation_class(_s("MEMBAR.SC.CTA")) == "MEMBAR.SC.CTA"
    assert operation_class(_s("FENCE.VIEW.ASYNC.S")) == "FENCE.VIEW.ASYNC.S"
    assert operation_class(_s("MEMBAR")) == "MEMBAR"
