"""poll_folding.fold_warp: maximal clean suffix + U* gap whitelist."""
from types import SimpleNamespace

from cyclescope.convert.poll_folding import fold_warp


def _rec(opcode, after_ready=None, mrefs=((0x1000, 32),), ops=None):
    return SimpleNamespace(
        after_ready=after_ready, exec_mask=0xFFFFFFFF,
        mrefs=list(mrefs), ops=ops or {})


def _static(opcode):
    inst = SimpleNamespace(opcode=opcode)
    return SimpleNamespace(inst=inst, opcode_text=opcode,
                           ops_layout=[])


def _run(opcodes):
    records = [_rec(op) for op in opcodes]
    return records, lambda r: _static(
        r._op if hasattr(r, "_op") else r._opcode)


def make(opcodes, after):
    """records: opcode list; after: per-record after_ready or None."""
    recs = []
    for op, a in zip(opcodes, after):
        r = _rec(op, after_ready=a)
        r._opcode = op
        recs.append(r)
    st = {id(r): _static(r._opcode) for r in recs}
    return recs, lambda r: st[id(r)]


def test_dirty_prefix_only_folds_suffix():
    # [PHASECHK fail, IABS dirty, PHASECHK fail, ULOP3, PHASECHK fail,
    #  PHASECHK ready] -> only the clean suffix folds
    ops = ["SYNCS.PHASECHK", "IABS", "SYNCS.PHASECHK", "ULOP3",
           "SYNCS.PHASECHK", "SYNCS.PHASECHK"]
    after = [0, None, 0, None, 0, 0xFFFFFFFF]
    recs, st = make(ops, after)
    folded, dropped = fold_warp(recs, st)
    assert 0 not in folded          # dirty prefix kept
    assert 2 in folded and 4 in folded   # clean suffix folded
    assert dropped == 2


def test_u_alu_gap_is_clean():
    ops = ["SYNCS.PHASECHK", "UIADD3", "SYNCS.PHASECHK",
           "SYNCS.PHASECHK"]
    after = [0, None, 0, 0xFFFFFFFF]
    recs, st = make(ops, after)
    folded, dropped = fold_warp(recs, st)
    assert dropped == 2
    assert 0 in folded and 2 in folded and 1 in folded


def test_fully_clean_run_folds_all():
    ops = ["SYNCS.PHASECHK", "NANOSLEEP", "SYNCS.PHASECHK",
           "SYNCS.PHASECHK"]
    after = [0, None, 0, 0xFFFFFFFF]
    recs, st = make(ops, after)
    folded, dropped = fold_warp(recs, st)
    assert dropped == 2
