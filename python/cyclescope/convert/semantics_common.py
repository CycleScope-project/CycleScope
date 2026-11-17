"""Common dynamic-instruction semantic decoder (docs/en/design/convert.md).

Pure function: ``decode(static, dynamic, context) -> SemanticRow | None``.
Facts come only from ``StaticInstruction`` (opcode/operands/predicate) and
``DynamicInstruction.operands`` (uniform/lane values, ranks,
after_ready_mask, tensor map facts).  No cross-warp inference, no
generation resolution — unresolved generations carry the
``GENERATION_UNRESOLVED`` sentinel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum

from ..job.semantics import SemanticsKind, GENERATION_UNRESOLVED
from .model import DynamicInstruction, StaticInstruction

from .sass_decode import decode_mbarrier_init_expected_arrivals
from ..job.semantics import (SEMA_FLAG_CTA_PAIR, SEMA_FLAG_IMPLICIT_COMMIT,
                             SEMA_FLAG_SYNCS, SEMA_FLAG_COMPLETE_TX)

# SEMA.flags bits
FLAG_AFTER_READY = 0x1          # mbarrier_wait: after_ready_mask operand present
FLAG_ALSO_ARRIVES = 0x2         # cta_barrier_wait: BAR.SYNC / BAR.RED

# descriptor_visibility operation values
DV_PUBLISH = 1
DV_STORE = 2
DV_ERROR_CHECK = 3
DV_ACQUIRE = 4
DV_PROXY_FENCE = 5

# scope values for mbarrier kinds
SCOPE_CTA = 0
SCOPE_CLUSTER = 1


class SemanticOperation(IntEnum):
    """Arch-neutral semantic operation; rules map SASS -> this."""
    none = 0
    mbarrier_init = 1
    mbarrier_arrive = 2
    mbarrier_expect_tx = 3
    mbarrier_wait = 4
    mbarrier_inval = 5
    cta_barrier_arrive = 6
    cta_barrier_wait = 7
    cluster_barrier_arrive = 8
    cluster_barrier_wait = 9
    tma_load = 10
    bulk_group_issue = 11
    bulk_group_commit = 12
    bulk_group_wait = 13
    async_copy_issue = 14
    async_copy_commit = 15
    async_copy_wait = 16
    bulk_copy = 17
    bulk_reduce = 18
    shared_store_async = 19
    descriptor_visibility_publish = 20
    descriptor_visibility_store = 21
    descriptor_visibility_error_check = 22
    descriptor_visibility_acquire = 23
    descriptor_visibility_proxy_fence = 24
    nanosleep = 25
    depbar_wait = 26
    dependents_launch = 27
    # --- arch-owned operations (2b-3; decoded by arch/<arch>/semantics) ---
    warpgroup_arrive = 32
    warpgroup_wait = 33
    wgmma = 34
    umma_issue = 35
    umma_commit = 36
    tmem_alloc = 37
    tmem_dealloc = 38
    tmem_relinquish = 39
    tmem_load = 40
    tmem_store = 41
    tmem_copy = 42
    tmem_wait_store = 43
    clc_query = 44


# Operations decoded by arch/<arch>/semantics.py, not the common decoder.
ARCH_OPERATIONS = frozenset({
    SemanticOperation.warpgroup_arrive,
    SemanticOperation.warpgroup_wait,
    SemanticOperation.wgmma,
    SemanticOperation.umma_issue,
    SemanticOperation.umma_commit,
    SemanticOperation.tmem_alloc,
    SemanticOperation.tmem_dealloc,
    SemanticOperation.tmem_relinquish,
    SemanticOperation.tmem_load,
    SemanticOperation.tmem_store,
    SemanticOperation.tmem_copy,
    SemanticOperation.tmem_wait_store,
    SemanticOperation.clc_query,
})


class ConversionError(Exception):
    """missing_fact:<name> / unsupported:<reason> (<where>)."""


def missing(fact: str, where: str) -> ConversionError:
    return ConversionError(f"missing_fact:{fact} ({where})")


def unsupported(reason: str, where: str) -> ConversionError:
    return ConversionError(f"unsupported:{reason} ({where})")


@dataclass(slots=True)
class SemanticRow:
    kind: SemanticsKind = SemanticsKind.none
    flags: int = 0
    fields: dict = field(default_factory=dict)      # named slots -> u64
    object_list: tuple[int, ...] | None = None      # -> MBOL rows
    split: tuple[tuple[int, int, int], ...] | None = None  # (rank, credit, pad) -> SPLT


@dataclass(slots=True)
class SemanticsContext:
    """Per-unit state: mbarrier objects and per-warp async-group counters."""
    unit: object                                    # ClusterUnit
    mbarrier_objects: dict = field(default_factory=dict)  # (cta, addr) -> obj index
    ldgsts_group: dict = field(default_factory=dict)      # warp_slot -> next group
    bulk_group: dict = field(default_factory=dict)        # warp_slot -> next group
    saw_lgdepbar: set = field(default_factory=set)        # warp slots
    saw_utmaflush: set = field(default_factory=set)
    bulk_group_waited: dict = field(default_factory=dict)  # wslot -> [waited]
    desc_stores: dict = field(default_factory=dict)        # wslot -> [{base,chunk}]
    desc_pending: dict = field(default_factory=dict)       # wslot -> pending
    bulk_open_issues: dict = field(default_factory=dict)   # wslot -> count
    extra: dict = field(default_factory=dict)              # arch state
    local_shared: object = None        # arch local_shared_address hook

    def mbarrier(self, cta_index: int, address: int | None,
                 where: str) -> int:
        if address is None:
            raise missing("mbarrier_address", where)
        key = (cta_index, address)
        index = self.mbarrier_objects.get(key)
        if index is None:
            index = len(self.mbarrier_objects)
            self.mbarrier_objects[key] = index
        return index

    def next_group(self, table: dict, warp_slot: int) -> int:
        group = table.get(warp_slot, 0)
        table[warp_slot] = group + 1
        return group

    def current_group(self, table: dict, warp_slot: int) -> int:
        return table.get(warp_slot, 0)


def _where(static: StaticInstruction) -> str:
    return f"{static.opcode} @{static.pc:#x}"


def _mbarrier_object(dyn, ctx: SemanticsContext, cta: int, address,
                     where: str) -> int:
    """Resolve the mbarrier object for (cta, address); when the fact
    carries a remote ``owner_rank`` (a DSMEM rank other than the issuing
    CTA's cluster rank), the object belongs to that peer CTA and the
    address is the peer's local shared offset."""
    # mbarrier object keys are always local shared offsets: the DSMEM
    # address embeds its owner's cluster rank in bits 24+ even for the
    # issuing CTA's own objects — strip it in every case (bug 4-5.1)
    if ctx.local_shared is not None:
        address = ctx.local_shared(address)
    rank = _uniform_value(dyn, "owner_rank")
    if rank is not None:
        rank = int(rank)
        issuing = 0
        for c in ctx.unit.ctas:
            if c.cta_linear == cta:
                issuing = c.cluster_rank
                break
        if rank != issuing:
            for c in ctx.unit.ctas:
                if c.cluster_rank == rank:
                    if ctx.local_shared is not None:
                        address = ctx.local_shared(address)
                    return ctx.mbarrier(c.cta_linear, address, where)
            raise missing("owner_rank_cta", where)
    return ctx.mbarrier(cta, address, where)


def _root(opcode: str) -> str:
    return opcode.split(".", 1)[0].split(" ", 1)[0].upper()


def _uniform_value(dyn: DynamicInstruction, *names: str):
    if dyn.operands is None:
        return None
    for name in names:
        value = dyn.operands.uniform.get(name)
        if value is not None:
            return value
    return None


def depbar_operands(static: StaticInstruction,
                    dyn: DynamicInstruction):
    """(keep, sb0) for DEPBAR, from operand tokens (hopper
    barrier_operands / static.operands)."""
    import re
    where = _where(static)
    tokens = list(static.operands or ())
    keep = None
    sb0 = False
    if static.opcode.upper().startswith("DEPBAR.LE"):
        for tok in tokens:
            m = re.fullmatch(r"(?:SB|GSB)(\d+)", str(tok), re.I)
            if m:
                sb0 = int(m.group(1)) == 0
                continue
            if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", str(tok)):
                keep = int(str(tok), 0)
        if keep is None:
            keep = _first_op(dyn, "pending_allowed", "N")
        if keep is None:
            raise missing("depbar_keep", where)
    else:
        for tok in tokens:
            bits = re.fullmatch(r"\{([^}]*)\}", str(tok))
            if bits and re.search(r"\b0\b", bits.group(1)):
                keep, sb0 = 0, True
    return keep, sb0


def _first_op(dyn, *names, default=None):
    value = _uniform_value(dyn, *names)
    if value is None:
        value = _last_op(dyn)
    return default if value is None else value


def _last_op(dyn: DynamicInstruction):
    """Last populated opN operand fact (count/bytes immediates)."""
    if dyn.operands is None:
        return None
    for index in range(7, -1, -1):
        value = dyn.operands.uniform.get(f"op{index}")
        if value is not None:
            return value
    return None


def _lane_address(dyn: DynamicInstruction):
    """First (base) memory reference's shared/local address, if scalar."""
    if not dyn.memory:
        return None
    ref = dyn.memory[0]
    lanes = ref.lane_addresses or ()
    if not lanes:
        return None
    return lanes[0]


def _object_address(dyn: DynamicInstruction, ref_index: int = 0):
    """mbarrier target: explicit operand fact first, then memory dst."""
    value = _uniform_value(dyn, "mbarrier_address", "mbarrier",
                           "object", "mbar")
    if value is not None:
        return value
    return _dst_address(dyn, ref_index)


def _dst_address(dyn: DynamicInstruction, ref_index: int = 0):
    """Shared-destination address of a memory reference."""
    if len(dyn.memory) <= ref_index:
        return None
    lanes = dyn.memory[ref_index].lane_addresses or ()
    return lanes[0] if lanes else None


def _multicast_mask(dyn: DynamicInstruction) -> int:
    """captured multicast_mask first, else per-lane getctarank union."""
    value = _uniform_value(dyn, "multicast_mask")
    if value is not None:
        return int(value)
    return _ranks_mask(dyn)


def _ranks_mask(dyn: DynamicInstruction, ref_index: int = 0) -> int:
    """destination_mask: set bits of getctarank facts over active lanes."""
    if dyn.operands is None:
        return 0
    ranks = dyn.operands.ranks.get(ref_index)
    if not ranks:
        return 0
    mask = 0
    for rank in ranks:
        mask |= 1 << rank
    return mask


# --- decoders (dispatch table at the bottom) ---------------------------

def _d_mbarrier_init(static, dyn, ctx, cta, wslot):
    where = _where(static)
    # distinct barrier objects over the active lanes' mref addresses
    # (one predicated store initialising several stage barriers); the
    # captured barrier-word fact, when present, names the object
    objects = None if _uniform_value(dyn, "mbarrier_address") is not None \
        else _distinct_objects(ctx, dyn, cta, where,
                               dyn.active_mask & dyn.predicate_mask)
    object_list = None
    if objects and len(objects) > 1:
        object_field, object_list = objects[0], tuple(objects)
    elif objects:
        object_field = objects[0]
    else:
        object_field = _mbarrier_object(dyn, ctx, cta,
                                        _object_address(dyn), where)
    # a promoted store's per-lane init words must decode to the same
    # (expected_arrivals, initial_phase); divergent lanes are not a
    # legal multi-barrier init
    vals = dyn.operands.lane.get("sts64_value") if dyn.operands else None
    if vals:
        mask = dyn.active_mask & dyn.predicate_mask
        seen: set = set()
        for value in _executing_entries(vals, mask):
            seen.add((decode_mbarrier_init_expected_arrivals(value),
                      (int(value) >> 63) & 1))
        if len(seen) != 1 or next(iter(seen))[0] is None:
            raise unsupported("mbarrier_init:divergent_lane_values", where)
    arrivals = _first_op(dyn, "expected_arrivals", "count", default=0)
    scope = SCOPE_CLUSTER if _ranks_mask(dyn) else SCOPE_CTA
    return SemanticRow(SemanticsKind.mbarrier_init,
                       fields={"object": object_field, "expected_arrivals":
                               arrivals, "scope": scope,
                               "initial_phase": _uniform_value(
                                   dyn, "initial_phase") or 0},
                       object_list=object_list)


def _executing_entries(values, mask: int) -> list:
    """Per-lane array entries for the executing lanes.  Facts compact
    these arrays to exec_mask.bit_count() entries in lane order; a
    32-entry form is full-width and indexed by lane id.  Never silently
    drops entries."""
    entries = list(values or ())
    if len(entries) == mask.bit_count():
        return entries
    if len(entries) == 32:
        return [v for i, v in enumerate(entries) if (mask >> i) & 1]
    return entries


def _distinct_objects(ctx: SemanticsContext, dyn: DynamicInstruction,
                      cta_index: int, where: str, mask: int | None = None):
    """(cta,addr) objects for every active lane's mref address (fallback
    when no per-lane barrier words were captured)."""
    if not dyn.memory:
        return None
    ref = dyn.memory[0]
    lanes = ref.lane_addresses or ()
    if mask is None:
        mask = dyn.active_mask
    # lane_mask is the record's own executing mask; absent -> caller's
    ref_mask = ref.lane_mask or mask
    objects: list[int] = []
    seen: set[int] = set()
    for addr in _executing_entries(lanes, ref_mask):
        obj = _mbarrier_object(dyn, ctx, cta_index, addr, where)
        if obj not in seen:
            seen.add(obj)
            objects.append(obj)
    return objects or None


def _issuer_rank(ctx: SemanticsContext, cta: int) -> int:
    for c in ctx.unit.ctas:
        if c.cta_linear == cta:
            return int(c.cluster_rank)
    return 0


def _per_lane_destinations(dyn, ctx: SemanticsContext, cta: int,
                           where: str):
    """Per-lane barrier words (operand fact ``mbarrier_words``) that name
    different objects -> (object, destination_mask, split, object_list).
    Same CTA-local offset on several ranks (CLC response arrive: lane i
    -> CTA i): the issuer's object + destination_mask + SPLT credit
    (lanes per rank); different offsets: MBOL list.  None when the lanes
    agree."""
    words = dyn.operands.lane.get("mbarrier_words") if dyn.operands else None
    if not words or len(set(words)) < 2:
        return None
    strip = ctx.local_shared or (lambda a: a)
    tagged = any(int(w) >> 24 for w in words)
    issuer = _issuer_rank(ctx, cta)
    def rank_of(w):
        return (int(w) >> 24) if tagged else issuer
    offsets = {strip(int(w)) for w in words}
    if len(offsets) == 1:
        offset = offsets.pop()
        per_rank: dict = {}
        for w in words:
            per_rank[rank_of(w)] = per_rank.get(rank_of(w), 0) + 1
        mask = 0
        for rank in per_rank:
            mask |= 1 << rank
        split = tuple((rank, n, 0) for rank, n in sorted(per_rank.items()))
        return ctx.mbarrier(cta, offset, where), mask, split, None
    objects: list = []
    for w in words:
        rank = rank_of(w)
        target = cta
        for c in ctx.unit.ctas:
            if c.cluster_rank == rank:
                target = c.cta_linear
                break
        obj = ctx.mbarrier(target, strip(int(w)), where)
        if obj not in objects:
            objects.append(obj)
    return objects[0], 0, None, tuple(objects)


def _d_mbarrier_arrive(static, dyn, ctx, cta, wslot):
    where = _where(static)
    per_lane = _per_lane_destinations(dyn, ctx, cta, where)
    split = None
    if per_lane is not None:
        object_field, mask, split, object_list = per_lane
    else:
        # a captured barrier word names the object; lane mref addresses
        # are only the fallback for facts without one (sm_103 NVBit
        # reports SYNCS mrefs at +0x7FFF)
        objects = None if _uniform_value(dyn, "mbarrier_address") is not None \
            else _distinct_objects(ctx, dyn, cta, where)
        if objects and len(objects) > 1:
            object_field, object_list = objects[0], tuple(objects)
        elif objects:
            object_field, object_list = objects[0], None
        else:
            object_field = _mbarrier_object(dyn, ctx, cta,
                                            _object_address(dyn), where)
            object_list = None
        mask = _ranks_mask(dyn)
    tx = _first_op(dyn, "transaction_bytes", "expect_tx_bytes",
                   default=0)
    count = _first_op(dyn, "arrival_count", "count", default=1)
    if split is not None:
        # per-lane arrival factor: the uniform count is lanes x factor
        lanes = sum(n for _r, n, _p in split)
        factor = max(1, int(count) // max(1, lanes))
        split = tuple((r, n * factor, p) for r, n, p in split)
    # .A<n>TX: complete the captured transaction bytes instead of
    # arriving / expecting (nvjet CLC-response arrive when CLC is unused)
    flags = (SEMA_FLAG_COMPLETE_TX
             if _first_op(dyn, "transaction_complete", default=0) else 0)
    return SemanticRow(SemanticsKind.mbarrier_arrive,
                       flags=flags,
                       fields={"object": object_field,
                               "generation": GENERATION_UNRESOLVED,
                               "transaction_bytes": tx,
                               "arrival_count": count,
                               "destination_mask": mask},
                       object_list=object_list, split=split)


def _d_expect_tx(static, dyn, ctx, cta, wslot):
    object = _mbarrier_object(dyn, ctx, cta, _object_address(dyn), _where(static))
    tx = _first_op(dyn, "transaction_bytes", default=0)
    return SemanticRow(SemanticsKind.mbarrier_expect_tx,
                       fields={"object": object,
                               "generation": GENERATION_UNRESOLVED,
                               "transaction_bytes": tx,
                               "scope": (SCOPE_CLUSTER if _ranks_mask(dyn)
                                         else SCOPE_CTA)})


def _d_mbarrier_wait(static, dyn, ctx, cta, wslot):
    where = _where(static)
    parity = _uniform_value(dyn, "parity", "phase", "UR0", "R0")
    if parity is None:
        raise missing("parity", where)
    fields = {"object": _mbarrier_object(dyn, ctx, cta, _object_address(dyn), where),
              "generation": GENERATION_UNRESOLVED,
              "parity": parity & 1}
    flags = 0
    if dyn.operands is not None and dyn.operands.after_ready_mask is not None:
        fields["after_ready_mask"] = dyn.operands.after_ready_mask
        flags |= FLAG_AFTER_READY
    block = _uniform_value(dyn, "block_offset", "offset")
    if block:
        fields["block_offset"] = block
    timeout = _uniform_value(dyn, "timeout_hint", "suspend_time_hint")
    if timeout:
        fields["timeout_hint"] = timeout
    return SemanticRow(SemanticsKind.mbarrier_wait, flags=flags,
                       fields=fields)


def _d_mbarrier_inval(static, dyn, ctx, cta, wslot):
    object = _mbarrier_object(dyn, ctx, cta, _object_address(dyn), _where(static))
    return SemanticRow(SemanticsKind.mbarrier_inval,
                       fields={"object": object})


def _bar_operand(static, dyn, index: int, where: str, shift):
    """BAR operand `index`: immediate as-is; a register from its captured
    first-lane value.  ptxas packs `(count << 16) | id` into one register
    for the register form (`BAR.SYNC R0, R0`), so with a register id the
    id is the low 16 bits and the count the high 16 (predecessor rule);
    a register count next to an immediate id is the raw count."""
    tokens = [t for t in static.operands if not re.fullmatch(r"!?P[T0-6]", t)]
    if index >= len(tokens):
        return None
    tok = tokens[index].strip()
    if re.fullmatch(r"0[xX][0-9a-fA-F]+|[0-9]+", tok):
        return int(tok, 0)
    if re.fullmatch(r"R\d+", tok):
        vals = dyn.operands.lane.get(tok) if dyn.operands else None
        if not vals:
            raise missing(f"cta_barrier_operand {tok}", where)
        value = int(vals[0])
        if shift is None:
            return value
        return (value & 0xFFFF) if shift == 0 else (value >> 16)
    raise missing(f"cta_barrier_operand {tok}", where)


def _d_cta_barrier(static, dyn, ctx, cta, wslot, arrives: bool):
    where = _where(static)
    # immediates arrive as op0/op1 facts; register forms need the lane
    # values of the static operand tokens
    barrier = _uniform_value(dyn, "barrier", "barrier_id", "op0")
    threads = _uniform_value(dyn, "expected_threads", "count", "op1")
    if barrier is None:
        barrier = _bar_operand(static, dyn, 0, where, shift=0)
        if barrier is None:
            raise missing("cta_barrier_id", where)
    if threads is None:
        id_is_register = bool(static.operands) and re.fullmatch(
            r"R\d+", static.operands[0].strip()) is not None
        threads = _bar_operand(static, dyn, 1, where,
                               shift=16 if id_is_register else None)
        if threads is None:
            # PTX bar.sync without a count: every non-exited CTA warp
            # takes part; 0 lets the engine deduct exited warps
            threads = 0
    if not 0 <= int(barrier) <= 15:
        raise unsupported(f"cta_barrier_id {barrier}", where)
    fields = {"barrier": int(barrier), "expected_threads": int(threads),
              "generation": GENERATION_UNRESOLVED}
    if arrives:
        return SemanticRow(SemanticsKind.cta_barrier_arrive,
                           fields=fields)
    flags = FLAG_ALSO_ARRIVES
    return SemanticRow(SemanticsKind.cta_barrier_wait, flags=flags,
                       fields=fields)


def _d_cluster_barrier(static, dyn, ctx, cta, wslot, arrives: bool):
    kind = (SemanticsKind.cluster_barrier_arrive if arrives
            else SemanticsKind.cluster_barrier_wait)
    fields = {"generation": GENERATION_UNRESOLVED}
    mask = _multicast_mask(dyn)
    if mask:
        fields["destination_mask"] = mask
    return SemanticRow(kind, fields=fields)


def _d_tma_load(static, dyn, ctx, cta, wslot):
    where = _where(static)
    object = _mbarrier_object(dyn, ctx, cta, _object_address(dyn), where)
    fields = {"object": object, "generation": GENERATION_UNRESOLVED,
              "multicast_mask": _multicast_mask(dyn)}
    if dyn.operands is not None and dyn.operands.tensor_map is not None:
        fields["tensor_map"] = dyn.operands.tensor_map
    tm = None
    if dyn.operands is not None and dyn.operands.tensor_map is not None:
        tm = ctx.unit.tensor_maps.get(dyn.operands.tensor_map)
    if tm is not None:
        box = 1
        for d in (tm.box_dim or tm.global_dim):
            box *= d
        # tile::gather4 / scatter4: four row coordinates, each a box row
        if "GATHER4" in static.opcode.upper() or \
                "SCATTER4" in static.opcode.upper():
            box *= 4
        fields["shared_bytes"] = box * (tm.element_bytes or 1)
        fields["global_bytes"] = box * (tm.element_bytes or 1)
    # cta_group::2: every destination's bytes post on its pair leader
    flags = SEMA_FLAG_CTA_PAIR if "2CTA" in static.opcode.upper().split(
        ".")[1:] else 0
    return SemanticRow(SemanticsKind.tma_load, flags=flags, fields=fields)


def _d_bulk_group_issue(static, dyn, ctx, cta, wslot):
    ctx.bulk_open_issues[wslot] = ctx.bulk_open_issues.get(wslot, 0) + 1
    group = ctx.current_group(ctx.bulk_group, wslot)
    size = _first_op(dyn, "bytes", "size", default=0)
    return SemanticRow(SemanticsKind.bulk_group_issue,
                       fields={"group": group, "pending_allowed": 0,
                               "bytes": size})


def _d_bulk_group_commit(static, dyn, ctx, cta, wslot):
    ctx.saw_utmaflush.add(wslot)
    ctx.bulk_open_issues[wslot] = 0
    group = ctx.next_group(ctx.bulk_group, wslot)
    return SemanticRow(SemanticsKind.bulk_group_commit,
                       fields={"group": group, "pending_allowed": 0})


def _d_bulk_group_wait(static, dyn, ctx, cta, wslot):
    group = ctx.current_group(ctx.bulk_group, wslot)
    pending = _first_op(dyn, "pending_allowed", "N", default=0)
    return SemanticRow(SemanticsKind.bulk_group_wait,
                       fields={"group": group,
                               "pending_allowed": pending})


def _d_async_copy_issue(static, dyn, ctx, cta, wslot):
    group = ctx.current_group(ctx.ldgsts_group, wslot)
    return SemanticRow(SemanticsKind.async_copy_issue,
                       fields={"group": group, "pending_allowed": 0})


def _d_async_copy_commit(static, dyn, ctx, cta, wslot):
    if static.opcode.upper().startswith("ARRIVES"):
        # cp.async.mbarrier.arrive (ARRIVES.LDGSTSBAR): binds pending
        # cp.async copies to an mbarrier; consumes pending_ldgsts in the
        # predecessor and is NOT a commit-group advance.
        # no pending cp.async in the thread -> the arrive triggers at
        # once (PTX cp.async.mbarrier.arrive); predicated-off LDGSTS
        # records still mark issue, so no guard is needed
        object_addr = _object_address(dyn)
        if object_addr is None:
            raise missing("mbarrier_address", _where(static))
        group = ctx.current_group(ctx.ldgsts_group, wslot)
        lanes = (dyn.active_mask & dyn.predicate_mask).bit_count()
        # .TRANSCNT (cp.async.mbarrier.arrive without .noinc): the
        # paired SYNCS.ARRIVE.RED.A0T1 already raised the tx count by one
        # per lane; the deferred completion lowers it again — no
        # arrival.  Without .TRANSCNT (.noinc) each lane arrives once.
        transcnt = "TRANSCNT" in static.opcode.upper().split(".")
        return SemanticRow(
            SemanticsKind.async_copy_commit,
            fields={"group": group, "pending_allowed": 0,
                    "object": _mbarrier_object(dyn, ctx, cta, object_addr,
                        _where(static)),
                    "arrival_count": 0 if transcnt else lanes,
                    "transaction_count": lanes if transcnt else 0})
    # LDGDEPBAR: cp.async.commit_group closes the current group and is
    # SB0-FIFO evidence for a later DEPBAR.
    ctx.saw_lgdepbar.add(wslot)
    group = ctx.next_group(ctx.ldgsts_group, wslot)
    return SemanticRow(SemanticsKind.async_copy_commit,
                       fields={"group": group, "pending_allowed": 0})


def _d_async_copy_wait(static, dyn, ctx, cta, wslot):
    group = ctx.current_group(ctx.ldgsts_group, wslot)
    pending = _first_op(dyn, "pending_allowed", "N", default=0)
    return SemanticRow(SemanticsKind.async_copy_wait,
                       fields={"group": group,
                               "pending_allowed": pending})


def _d_depbar(static, dyn, ctx, cta, wslot, keep, sb0):
    """DEPBAR wait on the warp's shared SB0 commit-group FIFO (blackwell
    semantics.py:386-443)."""
    if not sb0 or keep is None:
        return None
    saw_cpcopy = wslot in ctx.saw_lgdepbar
    saw_bulk = wslot in ctx.saw_utmaflush
    group_wait = (0 <= keep <= 7
                  or (keep <= 63 and (saw_cpcopy or saw_bulk)))
    if not group_wait:
        return None
    if saw_cpcopy and saw_bulk:
        raise unsupported("mixed_async_groups", _where(static))
    if saw_cpcopy:
        group = ctx.current_group(ctx.ldgsts_group, wslot)
        return SemanticRow(SemanticsKind.async_copy_wait,
                           fields={"group": group,
                                   "pending_allowed": keep})
    # predecessor: DEPBAR implies a commit when the warp has open
    # bulk issues (semantics.py: pending_store_issues -> _store_commit)
    flags = 0
    group = ctx.current_group(ctx.bulk_group, wslot)
    if ctx.bulk_open_issues.get(wslot, 0) > 0:
        group = ctx.next_group(ctx.bulk_group, wslot)
        ctx.bulk_open_issues[wslot] = 0
        flags = SEMA_FLAG_IMPLICIT_COMMIT
    return SemanticRow(SemanticsKind.bulk_group_wait, flags=flags,
                       fields={"group": group, "pending_allowed": keep})


def _d_bulk_copy(static, dyn, ctx, cta, wslot, kind):
    object = _mbarrier_object(dyn, ctx, cta, _object_address(dyn), _where(static))
    size = _first_op(dyn, "bytes", "size", default=0)
    fields = {"object": object, "generation": GENERATION_UNRESOLVED,
              "bytes": size}
    mask = _multicast_mask(dyn)
    if mask:
        fields["multicast_mask"] = mask
    return SemanticRow(kind, fields=fields)


def _d_shared_store_async(static, dyn, ctx, cta, wslot):
    where = _where(static)
    total = _first_op(dyn, "bytes", "size", default=0)
    fields = {"bytes": total}
    object_addr = _object_address(dyn)
    if object_addr is not None:
        fields["object"] = _mbarrier_object(dyn, ctx, cta, object_addr, where)
    ranks = dyn.operands.ranks.get(0) if dyn.operands else None
    if ranks:
        fields["destination_rank"] = ranks[0]
    words = dyn.operands.lane.get("mbarrier_words") \
        if dyn.operands else None
    if words and len(set(words)) > 1:
        # lanes store into different ranks' barriers: each lane
        # completes its bytes on its own rank's object (duplicates
        # kept: two lanes on the same CTA complete twice)
        strip = ctx.local_shared or (lambda a: a)
        objects = []
        for w in words:
            rank = int(w) >> 24
            target = cta
            for c in ctx.unit.ctas:
                if c.cluster_rank == rank:
                    target = c.cta_linear
                    break
            objects.append(ctx.mbarrier(target, strip(int(w)), where))
        if total % len(objects):
            raise missing("stas_lane_bytes", where)
        fields["bytes"] = total // len(objects)
        fields["object"] = objects[0]
        return SemanticRow(SemanticsKind.shared_store_async,
                           fields=fields, object_list=tuple(objects))
    return SemanticRow(SemanticsKind.shared_store_async, fields=fields)


def _d_descriptor_visibility(static, dyn, ctx, cta, wslot, op: int):
    fields = {"operation": op}
    addr = _first_op(dyn, "descriptor_address", "address")
    if addr is not None:
        fields["descriptor_address"] = addr
    return SemanticRow(SemanticsKind.descriptor_visibility, fields=fields)


def _d_nanosleep(static, dyn, ctx, cta, wslot):
    ns = _first_op(dyn, "requested_ns", "ns", default=0)
    # NANOSLEEP.SYNCS is the suspended try_wait: it also wakes when the
    # last polled mbarrier's awaited phase completes
    flags = SEMA_FLAG_SYNCS if "SYNCS" in static.opcode.upper().split(
        ".")[1:] else 0
    return SemanticRow(SemanticsKind.nanosleep, flags=flags,
                       fields={"requested_ns": ns})


# opcode dispatch: list of (test, decoder).  Test sees upper-cased opcode.
def _has(op, *words):
    return any(w in op for w in words)


def promote_store_to_mbarrier_init(static, dyn, ctx: SemanticsContext,
                                   cta_index: int = 0, warp_slot: int = 0
                                   ) -> SemanticRow | None:
    """STS / STS.64 writing a barrier word named by explicit barrier
    instructions in the unit (pre-scanned into ``ctx.extra``) is an
    mbarrier init: with a decoded init value it decodes as
    ``mbarrier_init``; without one it is a capture gap
    (``missing_fact:mbarrier_init_value``).  Any other store returns
    None (not promoted).  Only the 32-bit STS and the STS.64 pair can
    carry an init state (STS.128 / sub-word forms / STSM are data
    stores)."""
    upper = static.opcode.upper()
    if upper not in ("STS", "STS.64") or dyn is None or ctx is None:
        return None
    named = (ctx.extra or {}).get("mbarrier_addresses")
    if not named or not dyn.memory:
        return None
    strip = ctx.local_shared
    if not any((strip(int(a)) if strip else int(a)) in named
               for a in dyn.memory[0].lane_addresses):
        return None
    ops = dyn.operands
    if ops is not None and ops.uniform.get("expected_arrivals") is not None:
        return decode(static, dyn, ctx, SemanticOperation.mbarrier_init,
                      cta_index, warp_slot)
    if ops is not None and ops.uniform.get("store_value_captured"):
        # the store's value was captured and is not a legal init state
        # (e.g. a data word written before the SYNCS.EXCH init): a data
        # store into memory that later becomes a barrier, not an init
        return None
    raise missing("mbarrier_init_value", _where(static))


# GSYN class codes by opcode root (tracer gsync_class order)
_GLOBAL_SYNC_CLASS_ROOTS = ("ATOM", "RED", "LD", "ST")


def global_sync_class(opcode: str) -> int:
    """GSYN class of a global-sync observed instruction: 1 atom / 2 red /
    3 ld_strong / 4 st_strong, 0 unknown."""
    root = _root(opcode)
    for code, prefix in enumerate(_GLOBAL_SYNC_CLASS_ROOTS, 1):
        if root.startswith(prefix):
            return code
    return 0


def global_sync_row(static: StaticInstruction, dyn: DynamicInstruction
                    ) -> SemanticRow | None:
    """kind-34 row for an instruction carrying a GSYN observation.  The
    engine keys its acquire wait and atomic-predecessor gate (engine.md
    §13.10) on this row; readers (ld_strong) and updaters (atom / red)
    carry it, a st_strong is a producer only and carries none.  Pairing
    stays in the engine: the row holds only the observation kind and the
    class."""
    g = dyn.global_sync
    if g is None or (dyn.active_mask & dyn.predicate_mask) == 0:
        return None
    cls = global_sync_class(static.opcode)
    if cls not in (1, 2, 3):
        return None
    return SemanticRow(kind=SemanticsKind.global_sync,
                       fields={"observation_kind": g.kind, "class": cls})


# operation -> decoder
def decode(static: StaticInstruction, dyn: DynamicInstruction,
           ctx: SemanticsContext, operation: SemanticOperation,
           cta_index: int = 0, warp_slot: int = 0) -> SemanticRow | None:
    """Decode one dynamic instruction; operation comes from arch rules."""
    if (dyn.active_mask & dyn.predicate_mask) == 0:
        return None                                  # predicated off
    if operation == SemanticOperation.mbarrier_init:
        return _d_mbarrier_init(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.mbarrier_arrive:
        return _d_mbarrier_arrive(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.mbarrier_expect_tx:
        return _d_expect_tx(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.mbarrier_wait:
        return _d_mbarrier_wait(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.mbarrier_inval:
        return _d_mbarrier_inval(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.cta_barrier_arrive:
        return _d_cta_barrier(static, dyn, ctx, cta_index, warp_slot,
                              arrives=True)
    if operation == SemanticOperation.cta_barrier_wait:
        return _d_cta_barrier(static, dyn, ctx, cta_index, warp_slot,
                              arrives=False)
    if operation == SemanticOperation.cluster_barrier_arrive:
        return _d_cluster_barrier(static, dyn, ctx, cta_index, warp_slot,
                                  arrives=True)
    if operation == SemanticOperation.cluster_barrier_wait:
        return _d_cluster_barrier(static, dyn, ctx, cta_index, warp_slot,
                                  arrives=False)
    if operation == SemanticOperation.tma_load:
        return _d_tma_load(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.bulk_group_issue:
        return _d_bulk_group_issue(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.bulk_group_commit:
        return _d_bulk_group_commit(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.bulk_group_wait:
        return _d_bulk_group_wait(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.async_copy_issue:
        return _d_async_copy_issue(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.async_copy_commit:
        return _d_async_copy_commit(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.async_copy_wait:
        return _d_async_copy_wait(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.bulk_copy:
        return _d_bulk_copy(static, dyn, ctx, cta_index, warp_slot,
                            SemanticsKind.bulk_copy)
    if operation == SemanticOperation.bulk_reduce:
        return _d_bulk_copy(static, dyn, ctx, cta_index, warp_slot,
                            SemanticsKind.bulk_reduce)
    if operation == SemanticOperation.shared_store_async:
        return _d_shared_store_async(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.descriptor_visibility_publish:
        if not static.opcode.upper().startswith("MEMBAR.ALL.GPU"):
            return None
        stores = ctx.desc_stores.pop(warp_slot, [])
        chunks = {}
        for entry in stores:
            chunks.setdefault(entry["base"], set()).add(entry["chunk"])
        complete = sorted(b for b, cs in chunks.items() if len(cs) == 8)
        if not complete:
            return None
        if len(complete) > 4:
            raise unsupported("descriptor_count", _where(static))
        pend = ctx.desc_pending.get(warp_slot)
        if pend and pend["acquired"] < len(pend["bases"]):
            raise unsupported("descriptor_republished", _where(static))
        fields = {"operation": DV_PUBLISH,
                  "descriptor_count": len(complete)}
        for i, base in enumerate(complete):
            fields[f"descriptor_address_{i}"] = base
        ctx.desc_pending[warp_slot] = {
            "bases": tuple(complete), "checks": set(), "acquired": 0}
        return SemanticRow(SemanticsKind.descriptor_visibility,
                           fields=fields)
    if operation == SemanticOperation.descriptor_visibility_store:
        # one-lane 16B STG on 16B-aligned address only (bw _desc_stg_candidate)
        if dyn.active_mask & dyn.predicate_mask == 0:
            return None
        if not dyn.memory or (dyn.active_mask & dyn.predicate_mask).bit_count() != 1:
            return None
        ref = dyn.memory[0]
        if ref.width_bytes != 16 or not ref.lane_addresses                 or ref.lane_addresses[0] % 16:
            return None
        addr = ref.lane_addresses[0]
        ctx.desc_stores.setdefault(warp_slot, []).append(
            {"base": addr & ~0x7F, "chunk": (addr & 0x7F) // 16})
        return None
    if operation == SemanticOperation.descriptor_visibility_error_check:
        pending = ctx.desc_pending.get(warp_slot)
        if pending is None:
            return None
        scope = ("cluster" if static.opcode.upper()
                 .startswith("CGAERRBAR") else "sm")
        expected = "sm" if not pending["checks"] else "cluster"
        if scope != expected or scope in pending["checks"]:
            raise unsupported("descriptor_check_order", _where(static))
        pending["checks"].add(scope)
        return _d_descriptor_visibility(static, dyn, ctx, cta_index,
                                        warp_slot, DV_ERROR_CHECK)
    if operation == SemanticOperation.descriptor_visibility_acquire:
        base = _uniform_value(dyn, "descriptor_address", "address",
                              "op1", "op0")
        if base is None:
            raise missing("descriptor_address", _where(static))
        # acquire is defined across warps (engine resolves any warp's
        # publish); same-warp pending bookkeeping only when it exists
        pending = ctx.desc_pending.get(warp_slot)
        if pending is not None and base in pending["bases"]:
            pending["acquired"] += 1
        return SemanticRow(SemanticsKind.descriptor_visibility,
                           fields={"operation": DV_ACQUIRE,
                                   "descriptor_count": 1,
                                   "descriptor_address_0": base})
    if operation == SemanticOperation.descriptor_visibility_proxy_fence:
        return _d_descriptor_visibility(static, dyn, ctx, cta_index,
                                        warp_slot, DV_PROXY_FENCE)
    if operation == SemanticOperation.depbar_wait:
        keep, sb0 = depbar_operands(static, dyn)
        return _d_depbar(static, dyn, ctx, cta_index, warp_slot,
                         keep, sb0)
    if operation == SemanticOperation.nanosleep:
        return _d_nanosleep(static, dyn, ctx, cta_index, warp_slot)
    if operation == SemanticOperation.dependents_launch:
        # PREEXIT executed (griddepcontrol.launch_dependents): marker
        # only; the engine charges the dispatch serializer at CTA
        # completion
        return SemanticRow(SemanticsKind.dependents_launch)
    return None
