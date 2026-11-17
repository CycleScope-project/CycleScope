"""Segment builder: ClusterUnit -> GSIMSEG image (docs/en/design/convert.md).

Step 2b-1 scope: templates (dedup), instances, register refs, scoreboard
fields from the control word, memory accesses + 32 B sectors (dedup,
store byte masks, exact atomic lane addresses, TMA-box sectors, bulk
ranges), code addresses. Semantics kind = none; no TIME rows; USES rows
are the initialization rows that came with the compiled program tables.

One pass over dynamic instructions appending into table bytearrays; only
the template dedup dict is retained.
"""

from __future__ import annotations

import re

from ..convert import model
from ..convert import semantics_common
from ..convert.semantics_common import ConversionError, SemanticsContext
from . import semantics as sem_mod
from ..convert import axes as axes_mod
from ..convert import mechanism_uses as muses
from . import segment_format as sf

_TMPL_TB_OFF = sf.field_offset(sf.TMPL, "timing_begin")
_TMPL_TC_OFF = sf.field_offset(sf.TMPL, "timing_count")

SECTOR_BYTES = 32
FETCH_LINE_BYTES = 128
NONE32 = 0xFFFFFFFF

MACC_REMOTE = 0x20
MACC_UNIFORM = 0x10
MACC_L1_ALLOCATE = 0x200
_ACCESS_KIND_CODES = {"load": 0x0, "store": 0x40, "atomic": 0x80,
                      "async_copy": 0xC0, "async_reduce": 0x100,
                      "prefetch": 0x140}
_SPACE_CODES = {"global": 0x0, "shared": 0x1, "local": 0x2,
                "constant": 0x3, "generic": 0x4,
                # resolved pointer spaces (facts _SPACE): a generic
                # pointer that resolved to cluster shared memory must
                # time on the shared path; global_system is global
                "shared_cluster": 0x1, "global_system": 0x0}
_TMA_ROOTS = {"UTMALDG", "UTMASTG", "UTMAREDG", "UTMAPF"}


class BuildError(Exception):
    pass


def _smsp_of(warp: model.WarpStream) -> int:
    return warp.warp_slot % 4


def _register_id(name: str) -> int | None:
    """REGS-row register namespace: R<n> -> n, UR<n> -> 256+n,
    P<n> -> 512+n, UP<n> -> 768+n, B<n> -> 1024+n.  Only the prefix
    and the digits right after it matter (R12.64, R10.H0_H0, R2.reuse,
    UR4.64).  RZ/URZ/PT/UPT/SR_* and any other form -> None: hard-wired
    or special registers carry no dependency, so they emit no REGS row
    and do not enter the template key."""
    s = str(name)
    for prefix, base in (("UR", 256), ("UP", 768), ("R", 0),
                         ("P", 512), ("B", 1024)):
        if not s.startswith(prefix):
            continue
        i = len(prefix)
        n = 0
        while i < len(s) and s[i].isdigit():
            n = n * 10 + (ord(s[i]) - 48)
            i += 1
        return base + n if i > len(prefix) else None
    return None


# opcodes whose register operands are always 64-bit pairs (REGS flag
# bit1); ".64" / ".F32x2" operand suffixes set the same flag, and for
# the f32x2 classes (FFMA2/FADD2/FMUL2) every operand is a pair except
# a plain ".F32"-suffixed broadcast scalar (protocol v4 OP2p)
_PAIR_ROOTS = frozenset(
    ("DADD", "DMUL", "DFMA", "DSETP", "DMNMX"))
_F32X2_ROOTS = frozenset(("FFMA2", "FADD2", "FMUL2"))
_PLAIN_PRED = re.compile(r"U?P(T|[0-9]+)$")


def _operand_base(tok) -> str:
    """Operand token -> base name: leading - ~ ! and every | (abs bars,
    also around the name as in |R9|.reuse) stripped, then the name up
    to the first '.' (R12.reuse -> R12, -R3.64 -> R3, R10.H0_H0 -> R10)."""
    t = str(tok).lstrip("-~!|").replace("|", "")
    return t.split(".", 1)[0]


def _source_slots(static) -> list:
    """Operand tokens of the real source slots in control-word reuse
    bit order: everything after the register destination(s) and the
    secondary predicate destinations SASS prints right after them
    (plain PT / UPT / P<n> / UP<n>; a !-prefixed predicate is a
    source and stops the skip)."""
    rest = list(static.operands[len(static.destination_registers):])
    i = 0
    while i < len(rest) and _PLAIN_PRED.fullmatch(rest[i]):
        i += 1
    return rest[i:]


class _Builder:
    """Streaming table emitters; only `templates` is a live dict."""

    def __init__(self, unit, program_tables, launch_meta, runtime=None):
        self.runtime = runtime
        self._tmpl_uses: dict = {}       # template index -> True (rows emitted)
        self.unit = unit
        self.meta = launch_meta or {}
        self.tables = {tag: bytearray() for tag in sf.RECORDS}
        self.strings = bytearray(program_tables["STRS"])
        self.string_offsets = dict(program_tables.get("STRS_OFFSETS", {}))
        for tag in ("PROG", "NODE", "GRD ", "LEAF", "PARM", "PTS ",
                    "FACT", "USES", "AXIS"):
            self.tables[tag] += program_tables.get(tag, b"")
        self.templates: dict[tuple, int] = {}
        self.sema_count = 0
        self._sema_noop = None
        self._readers_cache = {}
        self._emit_memo = {}
        self._emit_seen = {}
        self._memo_key = None
        self._cap = None          # [(kind, *vals)] sector ops being recorded
        self._cap_before = None
        self._next_static_pc = None
        self.msec_count = 0
        self.macc_count = 0
        self.inst_count = 0
        self.sem_ctx = SemanticsContext(unit=unit)
        # blackwell STS.64-as-mbarrier-init: mbarrier words referenced by
        # explicit barrier instructions anywhere in the unit (pre-scan).
        self.mbol_count = 0
        self.gsyn_rows = []
        self.missing_facts: dict = {}
        self.unsupported: dict = {}
        self._class_domain = frozenset(
            (runtime.axes.get("operation_class") or {})
            .get("values") or ()) if runtime is not None else None
        self._fact_count = len(program_tables["FACT"]) // sf.FACT.size
        self._time_count = 0
        self._flow_by_warp: dict = {}
        self._launch_obj = getattr(unit, "launch", None)
        self._schema_targets = dict((launch_meta or {}).get(
            "schema_targets") or {})
        self._uses_cache: dict = {}
        # protocol v4 operand_read: pc -> static (reuse-served pre-pass)
        self._static_by_pc = {st.pc: st for st in unit.statics}
        self._src_flags_cache: dict = {}
        self.regs_flag_fallbacks = 0

    def _prescan_mbarriers(self, unit):
        mbar_addrs = set()
        strip = getattr(self.rules, "local_shared_address",
                        None) if getattr(self, "rules", None) else None
        # only captured barrier-word facts of executing instances name
        # barriers (mbarrier_address / per-lane mbarrier_words); mref
        # addresses are not used (sm_103 NVBit reports SYNCS mrefs at
        # +0x7FFF, and a data word so "named" turns an ordinary STS into
        # a rejected init)
        for _w in unit.warps:
            for _d in _w.instructions:
                _ops = _d.operands
                if _ops is None or not (_d.active_mask & _d.predicate_mask):
                    continue
                if _ops.uniform.get("mbarrier_address") is not None:
                    _a = int(_ops.uniform["mbarrier_address"])
                    mbar_addrs.add(strip(_a) if strip else _a)
                for _a2 in _ops.lane.get("mbarrier_words") or ():
                    mbar_addrs.add(strip(int(_a2)) if strip else int(_a2))
        self.sem_ctx.extra["mbarrier_addresses"] = mbar_addrs


    def string_offset(self, text: str) -> int:
        off = self.string_offsets.get(text)
        if off is None:
            off = len(self.strings)
            self.strings += text.encode() + b"\0"
            self.string_offsets[text] = off
        return off

    def emit_warp(self, warp: model.WarpStream) -> None:
        warp_row = sf.WARP.pack_row((
            warp.cta_linear, warp.warp_in_cta, warp.warp_slot,
            _smsp_of(warp), self.inst_count, len(warp.instructions)))
        flow = self._flow_by_warp.setdefault(
            warp.warp_slot, axes_mod.FlowContext())
        sig = None
        insts = warp.instructions
        if insts and getattr(insts[0], "raw_key", b""):
            import hashlib as _hl
            h = _hl.blake2b(digest_size=16)
            for d in insts:
                h.update(d.raw_key)
            sig = h.digest()
        if sig is not None and sig in self._readers_cache:
            flow.readers = self._readers_cache[sig]
        else:
            flow.readers = self._scan_first_readers(warp)
            if sig is not None:
                self._readers_cache[sig] = flow.readers
        insts = warp.instructions
        statics = self.unit.statics
        for index, dyn in enumerate(insts):
            flow.reader = flow.readers.get(index)
            self._next_static_pc = (statics[insts[index + 1].static_index].pc
                                    if index + 1 < len(insts) else None)
            self.emit_instance(dyn, warp, flow)
        self._next_static_pc = None
        flow.reader = None
        self.tables["WARP"] += warp_row

    def _scan_first_readers(self, warp: model.WarpStream) -> dict:
        """Per-warp pre-scan (spec §3.4): instance index ->
        (reader_static, role) of the first later *executing* reader of
        one of its destination registers, before that register is
        overwritten by another executing instruction."""
        readers: dict = {}
        pending: dict = {}      # reg -> (inst_index, static, role)
        insts = warp.instructions
        statics = self.unit.statics
        for i in range(len(insts) - 1, -1, -1):
            dyn = insts[i]
            if (dyn.active_mask & dyn.predicate_mask) == 0:
                continue
            static = statics[dyn.static_index]
            dests = [str(r) for r in static.destination_registers]
            srcs = {str(r) for r in static.source_registers}
            best = None
            for reg in dests:
                cand = pending.get(reg)
                if cand is not None and (best is None
                                         or cand[0] < best[0]):
                    best = cand
            if best is not None:
                readers[i] = (best[1], best[2])
            for reg in dests:
                if reg in srcs:
                    pending[reg] = (i, static,
                                    static.register_roles.get(reg))
                else:
                    pending.pop(reg, None)
            for reg in srcs - set(dests):
                pending[reg] = (i, static,
                                static.register_roles.get(reg))
        return readers

    def emit_instance(self, dyn: model.DynamicInstruction,
                      warp=None, flow=None) -> None:
        static = self.unit.statics[dyn.static_index]
        executing = (dyn.active_mask & dyn.predicate_mask) != 0
        cls = self.rules.operation_class(static, dyn)
        use_rows = self.compute_uses(static, dyn, warp, flow, cls)
        if executing and self._class_domain is not None and                 getattr(self.rules, "operation_classes", None):
            classes = self.rules.operation_classes(static, dyn)
            if not self._class_domain & frozenset(classes):
                k = (f"unsupported:engine:mechanism_tree:"
                     f"operation_class={cls} ({static.opcode})")
                self.unsupported[k] = self.unsupported.get(k, 0) + 1
            if use_rows is not None and not any(
                    self.schema_target(r["prog"].get("parameter_schema"))
                    == "InstructionTemplate"
                    for r in use_rows if "missing" not in r):
                blocker = next(
                    (r["missing"].split(":", 1)[-1]
                     for r in use_rows if "missing" in r), "none")
                k = (f"missing_fact:program:instruction.service "
                     f"({cls}; {blocker})")
                self.missing_facts[k] =                     self.missing_facts.get(k, 0) + 1
        template, created = self.template_index(static, cls, use_rows)
        if created:
            self._tmpl_uses[template] = True
            begin, count = self.emit_uses_rows(static, use_rows,
                                               template)
            self._patch_tmpl_timing(template, begin, count)
        else:
            self.emit_uses_rows(static, use_rows, template,
                                instance_timing_only=True)
        if dyn.global_sync is not None:
            g = dyn.global_sync
            cls_code = semantics_common.global_sync_class(static.opcode)
            self.gsyn_rows.append(
                (self.inst_count, g.lane, g.kind, cls_code,
                 g.post_known if g.post_known
                 else 1 | (2 if g.before else 0),
                 g.address, g.value, g.before, g.sequence))
        semantics_index = self.emit_semantics(static, dyn, warp)
        memory_index = NONE32
        if dyn.memory:
            memory_index = self.macc_count
            for ref in dyn.memory:
                self.emit_access(static, dyn, ref)
        code_line = (static.pc // FETCH_LINE_BYTES
                     if static.pc is not None else 0)
        self.tables["INST"] += sf.INST.pack_row(
            (template, semantics_index, memory_index, code_line))
        if static.code_address is not None:
            self.tables["CODE"] += sf.CODE.pack_row(
                (self.inst_count, 0, static.code_address))
        self.inst_count += 1
        if flow is not None and executing:
            op = static.opcode.upper()
            if op.startswith("LDGSTS"):
                flow.cpasync_count += 1
            if op.startswith("ARRIVES"):
                flow.cpasync_count = 0
            if op.startswith("LDGDEPBAR"):
                flow.cpasync_count = 0
        if self._memo_key is not None:
            key = self._memo_key
            self._memo_key = None
            if not created:
                _memo_store(self, key, template)

    def _src_flags(self, static: model.StaticInstruction) -> tuple:
        """Per-REGS-source-row flags (protocol v4 register_file.
        operand_read, v5 addendum 1): bit0 reuse_served -- the static
        predecessor at pc - 16 carries the control-word reuse flag on
        this operand slot with the same register; bit1 pair -- 64-bit
        source (.64-suffixed name or a pair-operand class); bit2
        reuse_carrier -- this instruction's own control-word
        reuse_mask bit for the operand slot.  Falls back to all-zero
        flags when the operand tokens and the recorded source
        registers do not align."""
        cached = self._src_flags_cache.get(static.pc)
        if cached is not None:
            return cached
        srcs = _source_slots(static)
        reg_toks = [t for t in srcs
                    if _register_id(_operand_base(t)) is not None]
        rows = [r for r in static.source_registers
                if _register_id(r) is not None]
        if (len(reg_toks) != len(rows)
                or any(_operand_base(t) != _operand_base(r)
                       for t, r in zip(reg_toks, rows))):
            self.regs_flag_fallbacks += 1
            out = tuple(0 for _ in rows)
            self._src_flags_cache[static.pc] = out
            return out
        pred = self._static_by_pc.get(static.pc - 16)
        pred_srcs = _source_slots(pred) if pred is not None else ()
        root = static.opcode.split(".", 1)[0]
        flags = []
        for j, tok in enumerate(srcs):
            if _register_id(_operand_base(tok)) is None:
                continue
            f = 0
            if (pred is not None and j < len(pred_srcs)
                    and (pred.control.reuse_mask >> j) & 1
                    and _operand_base(pred_srcs[j])
                    == _operand_base(tok)):
                f |= 1
            if (".64" in tok or ".F32x2" in tok
                    or root in _PAIR_ROOTS
                    or (root in _F32X2_ROOTS
                        and ".F32" not in tok)):
                f |= 2
            if (static.control.reuse_mask >> j) & 1:
                f |= 4
            flags.append(f)
        out = tuple(flags)
        self._src_flags_cache[static.pc] = out
        return out

    def template_index(self, static: model.StaticInstruction,
                       cls: str, use_rows=()):
        """-> (template index, created).  Key = 2b-1 key + effective
        operation_class + per-program (name, facts) (spec §3.3) + the
        register operands and the control word: the TMPL row carries
        the register refs and the scoreboard / stall bits the engine
        tracks dependencies with, so two static instructions may only
        share a template when those are identical (before 2026-11-05
        the key stopped at the operand counts and every instruction of
        a class inherited the first one's registers and barriers)."""
        cw = static.control
        src_ids = tuple(i for r in static.source_registers
                        if (i := _register_id(r)) is not None)
        src_flags = self._src_flags(static)
        dst_ids = tuple(i for r in static.destination_registers
                        if (i := _register_id(r)) is not None)
        key = (cls, src_ids, src_flags, dst_ids,
               static.predicate is not None,
               static.memory_space,
               (cw.read_barrier, cw.write_barrier, cw.wait_mask & 0xFF,
                cw.stall & 0xF, cw.yield_hint & 1),
               tuple((r["program"],
                      tuple(sorted(r["facts"].items())))
                     for r in use_rows))
        index = self.templates.get(key)
        if index is not None:
            return index, False
        index = len(self.templates)
        self.templates[key] = index
        regs = self.tables["REGS"]
        source_begin = len(regs) // sf.REGS.size
        for rid, fl in zip(src_ids, src_flags):
            regs += sf.REGS.pack_row((rid | (fl << 24),))
        destination_begin = len(regs) // sf.REGS.size
        for rid in dst_ids:
            regs += sf.REGS.pack_row((rid,))
        cw = static.control
        self.tables["TMPL"] += sf.TMPL.pack_row((
            self.string_offset(cls),
            self.rules.instruction_flags(static),
            NONE32, NONE32,
            source_begin, len(src_ids),
            destination_begin, len(dst_ids),
            cw.read_barrier, cw.write_barrier, cw.wait_mask,
            (cw.stall & 0xF) | ((cw.yield_hint & 1) << 4),
            0, 0, 0, 0))
        return index, True

    def _patch_tmpl_timing(self, template: int, begin: int,
                           count: int) -> None:
        base = template * sf.TMPL.size
        self.tables["TMPL"][base + _TMPL_TB_OFF:base + _TMPL_TB_OFF + 8] = \
            begin.to_bytes(8, "little")
        self.tables["TMPL"][base + _TMPL_TC_OFF:base + _TMPL_TC_OFF + 4] = \
            count.to_bytes(4, "little")

    @staticmethod
    def _uses_key(dyn, flow, cls, next_pc):
        """Memoization key covering every input the axis functions read
        (spec §3.5); no object identity."""
        ops = dyn.operands
        ops_key = None
        if ops is not None:
            ops_key = (
                tuple(sorted(ops.uniform.items())),
                tuple(sorted((k, tuple(v))
                             for k, v in ops.lane.items())),
                tuple(sorted((k, tuple(v))
                             for k, v in ops.ranks.items())),
                ops.after_ready_mask,
                ops.tensor_map,
                tuple(ops.tensor_coordinates or ()),
                tuple(ops.descriptor_words or ()))
        reader = flow.reader if flow is not None else None
        return (dyn.static_index, dyn.active_mask, dyn.predicate_mask,
                tuple((r.space, r.access_kind, r.width_bytes,
                       tuple(r.lane_addresses), r.lane_mask,
                       r.remote_rank, r.destination_count,
                       r.transfer_bytes, r.cut_rows) for r in dyn.memory),
                ops_key,
                flow.cpasync_count if flow is not None else 0,
                (reader[0].opcode, reader[1]) if reader else None,
                # the branch_taken axis reads the next instance's pc
                next_pc,
                cls)

    def compute_uses(self, static, dyn, warp, flow=None, cls=None):
        """Traverse mechanism programs; return per-program rows."""
        if self.runtime is None or dyn is None or warp is None:
            return ()
        raw = getattr(dyn, "raw_key", None) or None
        reader = flow.reader if flow is not None else None
        if raw is not None:
            key = (raw, flow.cpasync_count if flow is not None else 0,
                   (reader[0].opcode, reader[1]) if reader else None,
                   # the branch_taken axis reads the next instance's pc
                   self._next_static_pc,
                   cls)
        else:
            key = self._uses_key(dyn, flow, cls, self._next_static_pc)
        cached = self._uses_cache.get(key)
        if cached is not None:
            return cached
        ctx = self.axis_context(warp, flow)
        where = f"{static.opcode} @{hex(static.pc)}"
        try:
            rows = muses.program_facts(self.runtime, static, dyn,
                                       ctx, where, cls)
        except ConversionError as e:    # missing_fact / unsupported
            self.missing_facts[str(e)] = \
                self.missing_facts.get(str(e), 0) + 1
            return ()
        out = []
        for row in rows:
            if "missing" in row:
                k = (f"missing_fact:{row['missing']} ({static.opcode}; "
                     f"{row.get('program')})")
                self.missing_facts[k] = self.missing_facts.get(k, 0) + 1
            else:
                out.append(row)
        self._uses_cache[key] = out
        return out

    def emit_uses_rows(self, static, use_rows, template,
                       instance_timing_only=False):
        """Append FACT + TIME + USES rows for the traversals.
        -> (time_begin, time_count) of this call's TimingRow rows."""
        time_begin = self._time_count
        for row in use_rows:
            prog = row["prog"]
            schema_target = self.schema_target(
                prog.get("parameter_schema"))
            if instance_timing_only and schema_target != "InstanceTiming":
                continue
            facts = row["facts"]
            fact_begin = self._fact_count
            bad_axis = None
            for name in sorted(facts):
                ax = self.runtime.axes[name]
                raw = facts[name]
                if ax["kind"] == "enum" and raw not in ax["values"]:
                    bad_axis = name
                    break
                val = (ax["values"].index(raw)
                       if ax["kind"] == "enum" else int(raw))
                self.tables["FACT"] += sf.FACT.pack_row(
                    (self.string_offset(name), val))
                self._fact_count += 1
            if bad_axis is not None:
                key = (f"missing_fact:axis:{bad_axis} "
                       f"({static.opcode} domain)")
                self.missing_facts[key] = \
                    self.missing_facts.get(key, 0) + 1
                del self.tables["FACT"][fact_begin * sf.FACT.size:]
                self._fact_count = fact_begin
                continue
            fact_count = self._fact_count - fact_begin
            node_row = self.runtime.row_of[row["node"]]
            if schema_target == "TimingRow":
                kind = self._timing_kind(prog)
                time_row = self._time_count
                self.tables["TIME"] += sf.TIME.pack_row(
                    (kind, 0, 0, 0, 0, 0.0, 1, b"" * 7))
                self._time_count += 1
                target, target_index = 1, time_row
            elif schema_target == "InstructionTemplate":
                target, target_index = 0, template
            elif schema_target == "InstanceTiming":
                target, target_index = 2, self.inst_count
            else:
                from .program_tables import TARGET_BY_NAME
                target = TARGET_BY_NAME.get(schema_target, 15)
                target_index = prog.get("target_index", 0)
            self.tables["USES"] += sf.USES.pack_row(
                (node_row, target, b"" * 3, target_index,
                 prog["schema_id"], prog["field_mask"],
                 fact_begin, fact_count))
        return time_begin, self._time_count - time_begin

    def schema_target(self, schema_name):
        return self._schema_targets.get(schema_name, "none")

    def _timing_kind(self, prog):
        from .timing_rows import TimingKind
        name = prog.get("timing_kind")
        return TimingKind[name].value if name else 0

    def axis_context(self, warp, flow=None):
        if flow is None:
            flow = self._flow_by_warp.setdefault(
                warp.warp_slot, axes_mod.FlowContext())
        return axes_mod.AxisContext(
            unit=self.unit, launch=self._launch_obj, warp=warp,
            flow=flow, rules=self.rules,
            next_static_pc=self._next_static_pc)

    def emit_semantics(self, static, dyn=None, warp=None) -> int:
        """2b-2: SEMA row only when the decoder produces semantics."""
        if dyn is None or warp is None:
            return NONE32
        try:
            row = self.rules.semantics(static, dyn, self.sem_ctx,
                                       warp.cta_linear, warp.warp_slot)
        except ConversionError as exc:
            # Facts absent from the bundle: emit no semantics for this
            # row and record the missing fact (nothing invented).
            self.missing_facts[str(exc)] = self.missing_facts.get(str(exc), 0) + 1
            return NONE32
        gsync_row = semantics_common.global_sync_row(static, dyn)
        if gsync_row is not None:
            if row is None:
                row = gsync_row
            else:
                key = str(semantics_common.unsupported(
                    "global_sync_semantics_conflict",
                    semantics_common._where(static)))
                self.unsupported[key] = self.unsupported.get(key, 0) + 1
        if row is None:
            return NONE32
        fields = dict(row.fields)
        if row.object_list:
            fields["object_list_begin"] = self.mbol_count
            fields["object_list_count"] = len(row.object_list)
            for obj in row.object_list:
                self.tables["MBOL"] += sf.MBOL.pack_row((obj,))
                self.mbol_count += 1
        index = self.sema_count
        if row.split:
            for rank, credit, pad in row.split:
                self.tables["SPLT"] += sf.SPLT.pack_row(
                    (index, rank, credit, pad))
        slots = sem_mod.pack_fields(row.kind, **fields)
        self.tables["SEMA"] += sf.SEMA.pack_row(
            (row.kind, row.flags, 0, *slots))
        self.sema_count += 1
        return index

    def emit_access(self, static, dyn, ref: model.MemoryReference) -> None:
        flags = _SPACE_CODES.get(ref.space or "generic", 0x4) | \
            _ACCESS_KIND_CODES.get(ref.access_kind, 0)
        if static.opcode.upper().split(".", 1)[0] in ("ULDC", "LDCU"):
            flags |= MACC_UNIFORM
        if ref.remote_rank is not None:
            flags |= MACC_REMOTE
        if ref.access_kind == "async_copy" and \
                static.opcode.upper().startswith("LDGSTS") and \
                "BYPASS" not in static.opcode.upper():
            flags |= MACC_L1_ALLOCATE
        begin = self.msec_count
        if self._is_tma(static):
            # TMA refs carry captured global 32 B sector bases
            # (lane_mask 0); bundles without a sector list emit none.
            for address in ref.lane_addresses:
                self._sector(address // SECTOR_BYTES, 0xFFFFFFFF, 0)
        elif ref.access_kind == "store":
            self._emit_store(ref)
        elif ref.access_kind in ("async_copy", "async_reduce") \
                and not ref.lane_addresses:
            self._emit_bulk_range(ref)
        else:
            # local-space plain loads carry precise byte masks (the
            # frame interleave needs the real bytes); other spaces
            # keep the full-sector mask
            self._emit_plain(ref.lane_addresses,
                             ref.width_bytes if ref.space == "local"
                             else 0)
        self.tables["MACC"] += sf.MACC.pack_row(
            (begin, self.msec_count - begin, flags, 0, 0,
             NONE32, 0, ref.cache_hint))
        self.macc_count += 1
        if self._cap is not None:
            self._cap_macc.append(list(self._cap))
            self._cap.clear()

    def _is_tma(self, static) -> bool:
        return static.opcode.upper().split(".", 1)[0] in _TMA_ROOTS

    def _sector(self, key: int, byte_mask: int, lane_mask: int) -> None:
        if self._cap is not None:
            self._cap.append(("row", key, byte_mask, lane_mask))
        self.tables["MSEC"] += sf.MSEC.pack_row(
            (key * SECTOR_BYTES, byte_mask, lane_mask))
        self.msec_count += 1

    def _emit_plain(self, lane_addresses, width: int = 0) -> None:
        # lane->sector merge is per access (one MACC): an earlier
        # instruction's MSEC rows are not this MACC's rows, so sector
        # reuse across instructions must still emit a row here
        # width > 0 (local-space loads): precise byte mask like
        # _emit_store -- a 4-byte LDL covers 4 bytes, not 32
        width_mask = (1 << width) - 1 if width else 0xFFFFFFFF
        sector_index: dict[int, int] = {}
        for lane, address in enumerate(lane_addresses):
            key = address // SECTOR_BYTES
            bm = (width_mask << (address % SECTOR_BYTES)) & 0xFFFFFFFF
            if self._cap is not None:
                self._cap.append(("lane", key, lane))
            prev = sector_index.get(key)
            if prev is None:
                sector_index[key] = self.msec_count
                self._sector(key, bm, 1 << lane)
            else:
                # OR the byte and lane bits into the existing row
                base = prev * sf.MSEC.size
                if width:
                    off = base + 8
                    oldb = int.from_bytes(
                        self.tables["MSEC"][off:off + 4], "little")
                    self.tables["MSEC"][off:off + 4] = (
                        oldb | bm).to_bytes(4, "little")
                off = base + 12
                old = int.from_bytes(
                    self.tables["MSEC"][off:off + 4], "little")
                self.tables["MSEC"][off:off + 4] = (
                    old | (1 << lane)).to_bytes(4, "little")

    def _emit_store(self, ref: model.MemoryReference) -> None:
        lanes = ref.lane_addresses
        mask = ref.lane_mask or ((1 << len(lanes)) - 1)
        width_mask = (1 << ref.width_bytes) - 1
        per_sector: dict[int, list] = {}
        for lane, address in enumerate(lanes):
            if not (mask >> lane) & 1:
                continue
            key = address // SECTOR_BYTES
            slot = per_sector.setdefault(key, [0, 0])
            slot[0] |= width_mask << (address % SECTOR_BYTES)
            slot[1] |= 1 << lane
        for key in sorted(per_sector):
            byte_mask, lane_mask = per_sector[key]
            self._sector(key, byte_mask & 0xFFFFFFFF, lane_mask)

    def _emit_bulk_range(self, ref: model.MemoryReference) -> None:
        address = ref.lane_addresses[0] if ref.lane_addresses else 0
        size = max(ref.width_bytes, 1)
        for key in range(address // SECTOR_BYTES,
                         (address + size - 1) // SECTOR_BYTES + 1):
            self._sector(key, 0xFFFFFFFF, 0)

def build_segment(unit: model.ClusterUnit, rules, program_tables: dict,
                  launch_meta=None, report=None, runtime=None,
                  split_program: bool = True) -> bytes:
    """One ClusterUnit -> GSIMSEG bytes.

    program_tables: payloads dict from program_tables.compile() for the
    device tree (shared by every segment of the launch).
    """
    b = _Builder(unit, program_tables, launch_meta, runtime)
    b.rules = rules
    b._prescan_mbarriers(unit)
    b.sem_ctx.local_shared = getattr(rules, "local_shared_address",
                                     None)
    if report is not None:
        if b.regs_flag_fallbacks:
            print(f"note: regs flag fallback (flags 0) on "
                  f"{b.regs_flag_fallbacks} static instruction(s)")
        report["missing_facts_builder"] = b.missing_facts
        report["unsupported_builder"] = b.unsupported
    meta = launch_meta or {}
    shape = meta.get("cluster_shape", (1, 1, 1))
    warps_by_cta: dict[int, list] = {}
    for warp in unit.warps:
        warps_by_cta.setdefault(warp.cta_linear, []).append(warp)
    first_warp = 0
    for cta in unit.ctas:
        cta_warps = warps_by_cta.get(cta.cta_linear, [])
        b.tables["CTAS"] += sf.CTAS.pack_row((
            cta.cta_linear, cta.cta[0], cta.cta[1], cta.cta[2],
            cta.cluster_rank, shape[0], shape[1], shape[2],
            cta.sm_id, cta.peer_sm_id, cta.multicast_mask,
            cta.cta_group, cta.is_pair_leader, cta.tmem_columns,
            first_warp, len(cta_warps)))
        for warp in cta_warps:
            b.emit_warp(warp)
            first_warp += 1
    for cta_linear, address, arrivals in unit.preinitialized_mbarriers:
        obj = b.sem_ctx.mbarrier(cta_linear, address,
                                 "preinitialized_mbarrier")
        b.tables["MBPI"] += sf.MBPI.pack_row(
            (cta_linear, obj, address, arrivals, 0))
    if b.sem_ctx.mbarrier_objects:
        mbob = sf.OPTIONAL_RECORDS["MBOB"]
        for (cta_idx, addr), _ in sorted(
                b.sem_ctx.mbarrier_objects.items(),
                key=lambda kv: kv[1]):
            b.tables.setdefault("MBOB", bytearray())
            b.tables["MBOB"] += mbob.pack_row((cta_idx, addr))
    if b.gsyn_rows:
        gsyn = sf.OPTIONAL_RECORDS["GSYN"]
        b.tables.setdefault("GSYN", bytearray())
        for row in b.gsyn_rows:
            b.tables["GSYN"] += gsyn.pack_row(row)
    b.tables["META"] += sf.META.pack_row((
        meta.get("tree_digest", b"\0" * 32),
        meta.get("program_digest", b"\0" * 32),
        meta.get("converter_digest", b"\0" * 32),
        unit.cluster_id, len(unit.ctas), len(unit.warps),
        b.inst_count, meta.get("flags", 0)))
    b.tables["STRS"] += b.strings
    tables = {tag: bytes(buf) for tag, buf in b.tables.items()}
    return sf.write_segment(tables, split_program=split_program)


_USES_FB_OFF = 26          # USES.fact_begin byte offset (Q)
_USES_TI_OFF = 8           # USES.target_index byte offset (Q)
_USES_TG_OFF = 4           # USES.target byte offset (B)
_MACC_SB_OFF = 0           # MACC.sector_begin (Q)


def _memo_store(self, key, template):
    sizes, f0, t0, m0, s0, b0, miss0, unsup0 = self._cap_before
    cap = self._cap
    self._cap = None
    # forbid memo if MBOL grew (SEMA embeds object_list_begin in packed
    # fields that cannot be patched generically)
    if len(self.tables["MBOL"]) != sizes["MBOL"]:
        return
    entry = {"template": template, "missing": {}, "unsup": {},
             "deltas": {}, "uses": [], "sema": [], "splt": [],
             "macc_ops": list(cap or []), "fact": None, "time": None}
    entry["fact"] = bytes(self.tables["FACT"][sizes["FACT"]:])
    entry["time"] = bytes(self.tables["TIME"][sizes["TIME"]:])
    entry["deltas"]["_fact_count"] = self._fact_count - f0
    entry["deltas"]["_time_count"] = self._time_count - t0
    uses_b = self.tables["USES"][sizes["USES"]:]
    fb_base, ti_base_t, ti_base_i = f0, t0, 0
    for off in range(0, len(uses_b), sf.USES.size):
        row = bytearray(uses_b[off:off + sf.USES.size])
        tg = row[_USES_TG_OFF]
        patches = {}
        fb = int.from_bytes(row[_USES_FB_OFF:_USES_FB_OFF + 8], "little")
        patches[_USES_FB_OFF] = (fb - fb_base, "_fact_count")
        if tg == 1:
            ti = int.from_bytes(
                row[_USES_TI_OFF:_USES_TI_OFF + 8], "little")
            patches[_USES_TI_OFF] = (ti - ti_base_t, "_time_count")
        elif tg == 2:
            ti = int.from_bytes(
                row[_USES_TI_OFF:_USES_TI_OFF + 8], "little")
            patches[_USES_TI_OFF] = (ti - ti_base_i, "inst_count")
        entry["uses"].append((bytes(row), patches))
    s_b = self.tables["SEMA"][sizes["SEMA"]:]
    for off in range(0, len(s_b), sf.SEMA.size):
        entry["sema"].append(bytes(s_b[off:off + sf.SEMA.size]))
    sp_b = self.tables["SPLT"][sizes["SPLT"]:]
    for off in range(0, len(sp_b), sf.SPLT.size):
        row = bytearray(sp_b[off:off + sf.SPLT.size])
        v = int.from_bytes(row[0:4], "little")
        entry["splt"].append((bytes(row), v - s0))
    m_b = self.tables["MACC"][sizes["MACC"]:]
    entry["macc"] = [bytes(m_b[o:o + sf.MACC.size])
                     for o in range(0, len(m_b), sf.MACC.size)]
    entry["macc_ops"] = list(self._cap_macc)
    self._cap_macc = []
    for k, v in self.missing_facts.items():
        d = v - miss0.get(k, 0)
        if d:
            entry["missing"][k] = d
    for k, v in self.unsupported.items():
        d = v - unsup0.get(k, 0)
        if d:
            entry["unsup"][k] = d
    self._emit_memo[key] = entry


def _memo_replay(self, memo, dyn, warp, flow):
    static = self.unit.statics[dyn.static_index]
    executing = (dyn.active_mask & dyn.predicate_mask) != 0
    for k, d in memo["missing"].items():
        self.missing_facts[k] = self.missing_facts.get(k, 0) + d
    for k, d in memo["unsup"].items():
        self.unsupported[k] = self.unsupported.get(k, 0) + d
    if memo["fact"]:
        self.tables["FACT"] += memo["fact"]
    if memo["time"]:
        self.tables["TIME"] += memo["time"]
    self._fact_count += memo["deltas"]["_fact_count"]
    self._time_count += memo["deltas"]["_time_count"]
    for row, patches in memo["uses"]:
        for off, (base, attr) in patches.items():
            v = base + getattr(self, attr)
            row = row[:off] + v.to_bytes(8, "little") + row[off + 8:]
        self.tables["USES"] += row
    semantics_index = NONE32
    if memo["sema"]:
        semantics_index = self.sema_count
        for row in memo["sema"]:
            self.tables["SEMA"] += row
            self.sema_count += 1
        for row, base in memo["splt"]:
            v = base + self.sema_count
            self.tables["SPLT"] += v.to_bytes(4, "little") + row[4:]
    memory_index = NONE32
    if memo["macc"]:
        memory_index = self.macc_count
        cap = self._cap
        self._cap = None          # replay ops must not re-record
        for i, row in enumerate(memo["macc"]):
            begin = self.msec_count
            sector_index: dict[int, int] = {}   # per-MACC, as _emit_plain
            for op in memo["macc_ops"][i]:
                if op[0] == "lane":
                    _t, key, lane = op
                    prev = sector_index.get(key)
                    if prev is None:
                        sector_index[key] = self.msec_count
                        self._sector(key, 0xFFFFFFFF, 1 << lane)
                    else:
                        base = prev * sf.MSEC.size + 12
                        old = int.from_bytes(
                            self.tables["MSEC"][base:base + 4],
                            "little")
                        self.tables["MSEC"][base:base + 4] = (
                            old | (1 << lane)).to_bytes(4, "little")
                else:
                    _t, key, bm, lm = op
                    self._sector(key, bm, lm)
            fields = sf.MACC.unpack(row)
            self.tables["MACC"] += sf.MACC.pack_row(
                (begin, self.msec_count - begin, fields["flags"],
                 fields["sequence"], fields["shape"],
                 fields["timing_begin"], fields["timing_count"],
                 fields["cache_hint"]))
            self.macc_count += 1
        self._cap = cap
    code_line = (static.pc // FETCH_LINE_BYTES
                 if static.pc is not None else 0)
    self.tables["INST"] += sf.INST.pack_row(
        (memo["template"], semantics_index, memory_index, code_line))
    if static.code_address is not None:
        self.tables["CODE"] += sf.CODE.pack_row(
            (self.inst_count, 0, static.code_address))
    self.inst_count += 1
    if executing:
        op = static.opcode.upper()
        if op.startswith("LDGSTS"):
            flow.cpasync_count += 1
        if op.startswith("ARRIVES"):
            flow.cpasync_count = 0
        if op.startswith("LDGDEPBAR"):
            flow.cpasync_count = 0

