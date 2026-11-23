# Shared synthetic-job helpers for the CTest fixtures and the
# perf-job generator (extracted from make_sync_fixtures.py).

from cyclescope.job import container as cf, segment_format as sf


from cyclescope.job import container as cf, segment_format as sf

NONE = 0xFFFFFFFF
AXES = ["ready_warps_smsp", "ready_warps_sm", "expected_tx_bytes",
        "actual_participant_warps", "actual_participant_ctas",
        "expected_arrivals", "completed_tx_bytes", "same_parity",
        "launch_warps_per_cta", "input_dtype", "transfer_bytes",
        "source_mode", "tensor_k", "cut_rows"]
AXID = {n: i for i, n in enumerate(AXES)}


class Seg:
    def __init__(self, cta_count: int, warp_count: int,
                 cluster_shape_x: int = 1):
        self.strings = bytearray()
        self._so: dict[str, int] = {}
        self.ctas: list[tuple] = []
        self.warps: list[tuple] = []
        self.tmpl: list[list] = []
        self.inst: list[tuple] = []
        self.sema: list[tuple] = []
        self.time: list[tuple] = []
        self.prog: list[tuple] = []
        self.node: list[list] = []
        self.grd: list[tuple] = []
        self.leaf: list[tuple] = []
        self.parm: list[tuple] = []
        self.uses: list[tuple] = []
        self.fact: list[tuple] = []
        self.mbpi: list[tuple] = []
        self.mbol: list[tuple] = []
        self.mbob: list[tuple] = []
        self.splt: list[tuple] = []
        self.pts: list[tuple] = []
        self.code: list[tuple] = []
        self.regs: list[int] = []
        self.macc: list[tuple] = []
        self.msec: list[tuple] = []
        self._cta_count = cta_count
        self._warp_count = warp_count
        self._shape_x = cluster_shape_x
        self.svc_node = self.leaf_node(1, {1: 1.0, 2: 1.0, 7: 1.0})

    def so(self, text: str) -> int:
        if text not in self._so:
            self._so[text] = len(self.strings)
            self.strings.extend(text.encode() + b"\0")
        return self._so[text]

    # -- mechanism tree bits --
    def leaf_node(self, schema_id: int, fields: dict[int, float],
                  linear: list = ()) -> int:
        """linear: (field_id, axis_index, [(axis_value, value), ...])
        -> a piecewise-linear PARM row (model_kind 2)."""
        pb = len(self.parm)
        for fid, v in sorted(fields.items()):
            self.parm.append((schema_id, fid, 0, b"\0" * 3, NONE,
                              float(v), 0.0, 0.0, 0, 0))
        for fid, ax, pts in linear:
            ptb = len(self.pts)
            for x, y in pts:
                self.pts.append((float(x), float(y)))
            self.parm.append((schema_id, fid, 2, b"\0" * 3, ax,
                              0.0, 0.0, 0.0, ptb, len(pts)))
        li = len(self.leaf)
        self.leaf.append((pb, 0, b"\0" * 3))
        ni = len(self.node)
        self.node.append([NONE, 0, 0, 0, 0, li])
        return ni

    def const_program(self, name: str, event: int, schema_id: int,
                      timing_kind: int, cycles: float,
                      enabled: float = 1.0,
                      extra: dict | None = None) -> None:
        fields = {0: cycles, 1: enabled}
        if extra:
            fields.update(extra)
        root = self.leaf_node(schema_id, fields)
        self.prog.append((self.so(name), event, schema_id, timing_kind,
                          root, 0x3, 0, 0))

    def guarded_program(self, name: str, event: int, schema_id: int,
                        timing_kind: int,
                        branches: list[tuple[int, int, int, int, float]]
                        ) -> None:
        """branches: (axis, op, low, high, cycles) -> leaf nodes."""
        root = len(self.node)
        self.node.append([NONE, root + 1, len(branches), 0, 0, NONE])
        for ax, op, lo, hi, cyc in branches:
            gi = len(self.grd)
            self.grd.append((ax, op, b"\0" * 3, lo, hi))
            ni = self.leaf_node(schema_id, {0: cyc, 1: 1.0})
            self.node[ni][3] = gi
            self.node[ni][4] = 1
        self.prog.append((self.so(name), event, schema_id, timing_kind,
                          root, 0x3, 0, 0))

    # -- per-instance rows --
    def regs_rows(self, ids: list) -> tuple[int, int]:
        """ids: register ids or (register, flags) tuples; flags live
        in bits 24-31 of the REGS u32."""
        begin = len(self.regs)
        self.regs.extend(r[0] | (r[1] << 24)
                         if isinstance(r, tuple) else r
                         for r in ids)
        return begin, len(ids)

    def template(self, op_class: str, timing: tuple = (),
                 control: int = 0, flags: int = 0,
                 svc: dict | None = None,
                 src: list[int] | None = None,
                 dst: list[int] | None = None,
                 scoreboard_read: int = 0xFF,
                 scoreboard_write: int = 0xFF,
                 scoreboard_wait_mask: int = 0) -> int:
        """timing: sequence of (kind, leaf_fields|None) or
        (kind, leaf_fields|None, facts) where facts is a list of
        (axis_name, value) recorded as the row's capture FACT rows.
        svc: optional instruction.service leaf fields overriding the
        shared latency-1 service leaf."""
        tb = len(self.time)
        for row in timing:
            kind, fields = row[0], row[1]
            facts = row[2] if len(row) > 2 else None
            linear = row[3] if len(row) > 3 else ()
            self.time.append((kind, 0, 0, 0, 0, 0.0, 1, b"\0" * 7))
            if fields is not None or facts or linear:
                node = self.leaf_node(4, fields or {}, linear)
                fact_begin = len(self.fact)
                for name, v in facts or []:
                    self.fact.append((self.so(name), v))
                self.uses.append((node, 1, b"\0" * 3,
                                  len(self.time) - 1, 4, 0x3,
                                  fact_begin,
                                  len(self.fact) - fact_begin))
        svc_node = self.svc_node if svc is None else self.leaf_node(1, svc)
        sb, sc = self.regs_rows(src or [])
        db, dc = self.regs_rows(dst or [])
        idx = len(self.tmpl)
        self.tmpl.append([self.so(op_class), flags, NONE, NONE,
                          sb, sc, db, dc, scoreboard_read,
                          scoreboard_write, scoreboard_wait_mask,
                          control, tb, len(timing), 0, 0])
        self.uses.append((svc_node, 0, b"\0" * 3, idx, 1, 0x86,
                          0, 0))
        return idx

    def operand_read(self, enabled: float = 1.0,
                     reads_per_cycle: float = 2.0,
                     collector: float = 2.0) -> None:
        """schema-33 initialization USES row (OperandRead target)."""
        node = self.leaf_node(33, {0: enabled, 1: reads_per_cycle,
                                   2: collector})
        self.uses.append((node, 0, b"\0" * 3, 0, 33, 0x7, 0, 0))

    def semantic(self, kind: int, flags: int = 0,
                 fields: tuple = ()) -> int:
        slots = tuple(fields) + (0,) * (6 - len(fields))
        self.sema.append((kind, flags, 0) + slots)
        return len(self.sema) - 1

    def instance(self, tmpl: int, sema: int | None = None,
                 macc: int | None = None) -> None:
        self.inst.append(
            (tmpl, NONE if sema is None else sema,
             NONE if macc is None else macc, len(self.inst)))

    def warp(self, cta: int, warp_in_cta: int, slot: int,
             insts: list[tuple[int, int | None]]) -> None:
        first = len(self.inst)
        for t, s in insts:
            self.instance(t, s)
        self.warps.append((cta, warp_in_cta, slot, slot % 4,
                           first, len(insts)))

    def cta(self, cta_index: int, cluster_rank: int, sm: int,
            warp_count: int, first_warp: int, cta_group: int = 0,
            is_pair_leader: int = 0) -> None:
        self.ctas.append((cta_index, cta_index, 0, 0, cluster_rank,
                          self._shape_x, 1, 1, sm, NONE, 0, cta_group,
                          is_pair_leader, 0, first_warp, warp_count))

    def build(self, cluster_id: int = 0) -> bytes:
        tables = {t: w.payload()
                  for t, w in sf.SegmentWriter().writers.items()}
        tables["META"] = sf.META.pack(
            tree_digest=b"\1" * 32, program_digest=b"\2" * 32,
            converter_digest=b"\3" * 32, cluster_id=cluster_id,
            cta_count=self._cta_count, warp_count=self._warp_count,
            instruction_count=len(self.inst), flags=0)
        tables["AXIS"] = b"".join(sf.AXIS.pack_row((self.so(a),))
                                  for a in AXES)
        tables["STRS"] = bytes(self.strings)
        tables["CTAS"] = b"".join(sf.CTAS.pack_row(r) for r in self.ctas)
        tables["WARP"] = b"".join(sf.WARP.pack_row(r) for r in self.warps)
        tables["TMPL"] = b"".join(sf.TMPL.pack_row(r) for r in self.tmpl)
        tables["INST"] = b"".join(sf.INST.pack_row(r) for r in self.inst)
        tables["SEMA"] = b"".join(sf.SEMA.pack_row(r) for r in self.sema)
        tables["TIME"] = b"".join(sf.TIME.pack_row(r) for r in self.time)
        tables["PROG"] = b"".join(sf.PROG.pack_row(r) for r in self.prog)
        tables["NODE"] = b"".join(sf.NODE.pack_row(r) for r in self.node)
        tables["GRD "] = b"".join(sf.GRD_.pack_row(r) for r in self.grd)
        tables["LEAF"] = b"".join(sf.LEAF.pack_row(r) for r in self.leaf)
        tables["PARM"] = b"".join(sf.PARM.pack_row(r) for r in self.parm)
        tables["USES"] = b"".join(sf.USES.pack_row(r) for r in self.uses)
        tables["FACT"] = b"".join(sf.FACT.pack_row(r) for r in self.fact)
        tables["SPLT"] = b"".join(sf.SPLT.pack_row(r) for r in self.splt)
        tables["MBPI"] = b"".join(sf.MBPI.pack_row(r) for r in self.mbpi)
        tables["MBOL"] = b"".join(sf.MBOL.pack_row(r) for r in self.mbol)
        tables["PTS "] = b"".join(sf.PTS_.pack_row(r) for r in self.pts)
        tables["CODE"] = b"".join(sf.CODE.pack_row(r) for r in self.code)
        tables["REGS"] = b"".join(sf.REGS.pack_row((r,))
                                  for r in self.regs)
        tables["MACC"] = b"".join(sf.MACC.pack_row(r) for r in self.macc)
        tables["MSEC"] = b"".join(sf.MSEC.pack_row(r) for r in self.msec)
        if self.mbob:
            tables["MBOB"] = b"".join(
                sf.OPTIONAL_RECORDS["MBOB"].pack_row(r)
                for r in self.mbob)
        return sf.write_segment(tables)


def container(seg: bytes, cluster_x: int = 1,
              global_sync: bytes = b"", placements: bytes = b"",
              topo: bytes = b"", **launch_kw) -> bytes:
    launch = {"device": "h100", "sm_count": 132,
              "cluster_x": cluster_x,
              "reference_sm_clock_khz": 1000000,
              "reference_cache_control": 2}
    launch.update(launch_kw)
    return cf.write_container(launch, [seg], placements=placements,
                              global_sync=global_sync, topo=topo)


def container_multi(segs: list[bytes],
                    placements: bytes = b"", topo: bytes = b"",
                    **launch_kw) -> bytes:
    launch = {"device": "h100", "sm_count": 132, "cluster_x": 1,
              "reference_sm_clock_khz": 1000000,
              "reference_cache_control": 2}
    launch.update(launch_kw)
    return cf.write_container(launch, segs, placements=placements,
                              topo=topo)


def plce(cta: int, segment: int, rank: int, placed: int, sm: int,
         slot: int) -> tuple:
    return (cta, segment, rank, placed, b"\0" * 3, sm, slot, 0,
            b"\0" * 3)


def topo(sm: int, domain: int = 0, tpc: int = 0) -> tuple:
    return (sm, domain, 0, tpc, 0, 0)


def init_use(node: int, schema: int) -> tuple:
    return (node, 0, b"\0" * 3, 0, schema, 0, 0, 0)


GS_ATOM, GS_RED, GS_LD, GS_ST = 1, 2, 3, 4


def gsyn(segment: int, instance: int, class_: int, *,
         address: int = 0, after: int = 0, before: int = 0,
         sequence: int = 0, lane: int = 0, kind: int = 0,
         flags: int | None = None) -> tuple:
    if flags is None:
        flags = (1 if after else 0) | (2 if before else 0)
    return (segment, instance, lane, kind, class_, flags,
            address, after, before, sequence)


def fills(t: int, n: int) -> list[tuple[int, None]]:
    return [(t, None)] * n


# kind ids (SemanticsKind / TimingKind)
MBAR_INIT, MBAR_ARRIVE, MBAR_EXPECT_TX, MBAR_WAIT, MBAR_INVAL = 1, 2, 3, 4, 5
CTA_ARRIVE, CTA_WAIT = 6, 7
CL_ARRIVE, CL_WAIT = 8, 9
NANOSLEEP = 37
DEPENDENTS_LAUNCH = 40
TK_VIS_GLOBAL, TK_NANOSLEEP, TK_CTA_REL, TK_CL_REL, TK_TRYWAIT = \
    4, 16, 17, 42, 61
TK_ASYNC_ARRIVE = 18
AFTER_READY = 0x1
SYNCS = 0x40          # NANOSLEEP.SYNCS (SEMA_FLAG_SYNCS)
ALSO_ARRIVES = 0x2


