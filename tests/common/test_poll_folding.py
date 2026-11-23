"""poll_folding.fold_warp: maximal clean suffix + U* gap whitelist."""
from types import SimpleNamespace

from cyclescope.convert.poll_folding import fold_warp


def _rec(opcode, after_ready=None, mrefs=((0x1000, 32),), ops=None):
    return SimpleNamespace(
        after_ready=after_ready, exec_mask=0xFFFFFFFF,
        mrefs=list(mrefs), ops=ops or {})


def _static(opcode, pc=0, wbar=7, wait=0):
    control = SimpleNamespace(write_barrier=wbar, wait_mask=wait)
    inst = SimpleNamespace(opcode=opcode, pc=pc, code_address=0x7000 + pc,
                           control=control)
    return SimpleNamespace(inst=inst, opcode_text=opcode,
                           operands_text="", ops_layout=[])


def _run(opcodes):
    records = [_rec(op) for op in opcodes]
    return records, lambda r: _static(
        r._op if hasattr(r, "_op") else r._opcode)


def make(opcodes, after, pcs=None, wbars=None, waits=None):
    """records: opcode list; after: per-record after_ready or None;
    pcs / wbars / waits: optional per-record static pc, write barrier,
    wait mask."""
    recs = []
    for op, a in zip(opcodes, after):
        r = _rec(op, after_ready=a)
        r._opcode = op
        recs.append(r)
    st = {id(r): _static(r._opcode,
                         pcs[i] if pcs else 16 * i,
                         wbars[i] if wbars else 7,
                         waits[i] if waits else 0)
          for i, r in enumerate(recs)}
    return recs, lambda r: st[id(r)]


def test_dirty_prefix_only_folds_suffix():
    # [PHASECHK fail, IABS dirty, PHASECHK fail, ULOP3, PHASECHK fail,
    #  PHASECHK ready] -> only the clean suffix folds
    ops = ["SYNCS.PHASECHK", "IABS", "SYNCS.PHASECHK", "ULOP3",
           "SYNCS.PHASECHK", "SYNCS.PHASECHK"]
    after = [0, None, 0, None, 0, 0xFFFFFFFF]
    recs, st = make(ops, after)
    folded, dropped, _origins = fold_warp(recs, st)
    assert 0 not in folded          # dirty prefix kept
    assert 2 in folded and 4 in folded   # clean suffix folded
    assert dropped == 2


def test_u_alu_gap_is_clean():
    ops = ["SYNCS.PHASECHK", "UIADD3", "SYNCS.PHASECHK",
           "SYNCS.PHASECHK"]
    after = [0, None, 0, 0xFFFFFFFF]
    recs, st = make(ops, after)
    folded, dropped, _origins = fold_warp(recs, st)
    assert dropped == 2
    assert 0 in folded and 2 in folded and 1 in folded


def test_fully_clean_run_folds_all():
    ops = ["SYNCS.PHASECHK", "NANOSLEEP", "SYNCS.PHASECHK",
           "SYNCS.PHASECHK"]
    after = [0, None, 0, 0xFFFFFFFF]
    recs, st = make(ops, after)
    folded, dropped, _origins = fold_warp(recs, st)
    assert dropped == 2


POLL = "SYNCS.PHASECHK.TRANS64.TRYWAIT"
READY = 0xFFFFFFFF


def mk(specs):
    """specs: dicts with op, pc, ready (None / 0 / 1), wbar, wait,
    text (operands_text), ops (captured registers), mref."""
    recs, rows = [], {}
    for sp in specs:
        ready = sp.get("ready")
        r = SimpleNamespace(
            after_ready=None if ready is None else (READY if ready else 0),
            exec_mask=0xFFFFFFFF, ops=dict(sp.get("ops", {})),
            mrefs=[[sp.get("mref", 0x1000), 32]])
        c = SimpleNamespace(write_barrier=sp.get("wbar", 7),
                            wait_mask=sp.get("wait", 0))
        rows[id(r)] = SimpleNamespace(
            inst=SimpleNamespace(opcode=sp["op"], pc=sp["pc"], control=c,
                                 code_address=0x7000 + sp["pc"]),
            opcode_text=sp["op"], operands_text=sp.get("text", ""),
            ops_layout=["R2"])
        recs.append(r)
    return recs, lambda r: rows[id(r)]


def P(pc, ready, wbar=0, **kw):
    return dict(op=POLL, pc=pc, ready=ready, wbar=wbar, **kw)


def X(op, pc, **kw):
    return dict(op=op, pc=pc, **kw)


def test_origin_same_address_none():
    # (i) same-address retry: folded as today, no origin
    recs, st = mk([P(0x100, 0), X("BRA", 0x110, wait=1), P(0x100, 0),
                   X("BRA", 0x110, wait=1), P(0x100, 1)])
    folded, dropped, origins = fold_warp(recs, st)
    assert folded == {0, 1, 2, 3} and dropped == 2 and origins == {}


def test_origin_pollcute_layout():
    # (ii) inline TRYWAIT (barrier 0), ISETP, @P0 BRA (wait 0x1), loop
    recs, st = mk([P(0x100, 0), X("ISETP", 0x110), X("BRA", 0x120, wait=1),
                   P(0x200, 0), X("NANOSLEEP", 0x210), P(0x200, 1)])
    folded, dropped, origins = fold_warp(recs, st)
    assert folded == {0, 1, 2, 3, 4} and dropped == 2
    assert origins == {5: 0x7000 + 0x120}


def test_origin_reach_16():
    # (iii) reader 10 and 14 behind the inline poll; (iv) 17 behind
    for behind, want in ((10, True), (14, True), (16, True), (17, False)):
        specs = [P(0x100, 0)]
        specs += [X("SHF", 0x110 + 16 * n) for n in range(behind - 1)]
        specs.append(X("BRA", 0x110 + 16 * (behind - 1), wait=1))
        specs += [P(0x400, 0), P(0x400, 1)]
        recs, st = mk(specs)
        _f, _d, origins = fold_warp(recs, st)
        bra = 0x7000 + 0x110 + 16 * (behind - 1)
        assert origins == ({len(specs) - 1: bra} if want else {})


def test_origin_inline_poll_not_in_fold_run():
    # (v) u.pollfar.lds: an LDS between the inline poll and its BRA is
    # not gap-free, the fold's run starts at the loop poll; the chain
    # still starts at the inline poll
    recs, st = mk([P(0x100, 0, text="P0, [R8+URZ], R3", ops={"R8": 0x40}),
                   X("LDS", 0x110, wbar=1), X("IMAD", 0x120, wait=2),
                   X("BRA", 0x130, wait=1),
                   P(0x200, 0, text="P0, [R8+URZ], R3", ops={"R8": 0x40}),
                   P(0x200, 1, text="P0, [R8+URZ], R3", ops={"R8": 0x40})])
    folded, dropped, origins = fold_warp(recs, st)
    assert 0 not in folded and 4 in folded
    assert origins == {5: 0x7000 + 0x130}


def test_origin_barrier_word_from_registers_not_mref():
    # (vi) inline [UR4+0x28040] with mref A, loop [R3+URZ+0x28040] with
    # R3 == UR4 and mref A + 32649: the same word; other word / parity:
    # not in the chain
    A = 0x5000
    inl = dict(text="P0, [UR4+0x28040], R2", ops={"UR4": 0x100, "R2": 0},
               mref=A)
    def loop(ready, r3=0x100, r2=0):
        return P(0x200, ready, text="P0, [R3+URZ+0x28040], R2",
                 ops={"R3": r3, "R2": r2}, mref=A + 32649)
    recs, st = mk([P(0x100, 0, **inl), X("BRA", 0x110, wait=1),
                   loop(0), loop(1)])
    assert fold_warp(recs, st)[2] == {3: 0x7000 + 0x110}
    for other in (loop(1, r3=0x140), loop(1, r2=1 << 31)):
        recs, st = mk([P(0x100, 0, **inl), X("BRA", 0x110, wait=1),
                       loop(0), other])
        assert fold_warp(recs, st)[2] == {}
        # chain {loop(0)} alone: its first poll is the loop poll itself
        # with the same code address -> nothing


def test_origin_barrier_rewritten_before_waiter():
    # (vii) a record writing b between P1 and its would-be waiter
    recs, st = mk([P(0x100, 0), X("LDS", 0x110, wbar=0), X("BRA", 0x120,
                   wait=1), P(0x200, 0), P(0x200, 1)])
    assert fold_warp(recs, st)[2] == {}
    # a record that both waits for and writes b is W
    recs, st = mk([P(0x100, 0), X("LDS", 0x110, wbar=0, wait=1),
                   P(0x200, 0), P(0x200, 1)])
    assert fold_warp(recs, st)[2] == {3: 0x7000 + 0x110}


def test_origin_ready_unknown_ends_chain():
    # (viii) a poll with unknown result in front of S ends the chain
    recs, st = mk([P(0x100, 0), X("BRA", 0x110, wait=1), P(0x300, None),
                   P(0x200, 0), P(0x200, 1)])
    # chain of S = {P(0x200, 0)}: first poll has S's address -> none
    assert fold_warp(recs, st)[2] == {}


def test_origin_cublas_layout():
    # (ix) inline TRYWAIT writes barrier 2; first waiter is the
    # out-of-line TRYWAIT with wait mask 0x4
    recs, st = mk([P(0x100, 0, wbar=2), X("ISETP", 0x110),
                   P(0x200, 0, wait=4), X("NANOSLEEP", 0x210),
                   P(0x200, 1, wait=4)])
    folded, dropped, origins = fold_warp(recs, st)
    assert dropped == 2 and origins == {4: 0x7000 + 0x200}


def test_origin_none_without_barrier():
    recs, st = mk([P(0x100, 0, wbar=7), X("BRA", 0x110, wait=1),
                   P(0x200, 0), P(0x200, 1)])
    assert fold_warp(recs, st)[2] == {}


def test_wait_row_carries_origin(monkeypatch):
    # converter level: flag 0x200 and the address in earliest_issue
    from cyclescope.convert import model, semantics_common as sc
    monkeypatch.setattr(sc, "_mbarrier_object", lambda *a: 3)
    monkeypatch.setattr(sc, "_object_address", lambda dyn: 0)
    static = SimpleNamespace(opcode=POLL, operands=(), pc=0x100)
    def row(uniform):
        dyn = SimpleNamespace(operands=model.SemanticOperands(
            uniform=dict(uniform), after_ready_mask=0xFFFFFFFF))
        return sc._d_mbarrier_wait(static, dyn, None, 0, 0)
    got = row({"parity": 1, "wait_origin": 0x7000120})
    assert got.flags & 0x200 and got.earliest_issue == 0x7000120
    assert got.flags & sc.FLAG_AFTER_READY
    plain = row({"parity": 1})
    assert not plain.flags & 0x200 and plain.earliest_issue == 0
