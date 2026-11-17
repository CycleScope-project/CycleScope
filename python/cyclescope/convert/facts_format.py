"""GSIMCAP01 — persisted converter model (capture -> convert boundary).

One file = one ClusterUnit; launch facts ride in the case-level
launch.json sidecar (already the build input). Layout: META header for
header-only cache checks, fixed-size row tables, variable data in aux
tables + STRS.
"""
from __future__ import annotations

import struct
from pathlib import Path

from cyclescope.io.file_header import write_file, FileReader
from cyclescope.convert import model

MAGIC = "GSIMCAP01"
MAGIC2 = "GSIMCAP02"      # cluster file; statics in statics.gsc
MAGIC_STATICS = "GSIMCAPS1"   # per-dir statics bundle
N32 = 0xFFFFFFFF

META = struct.Struct("<6I4Q")      # cluster_id, n_ctas, n_warps, n_stat,
                                   # n_dyn, n_mref, digest(4xQ)
CTAS = struct.Struct("<4i8I")      # cta_linear, x,y,z, cluster_rank, sm_id,
                                   # peer_sm_id, multicast, cta_group,
                                   # pair_leader, tmem_columns, pad
WARP = struct.Struct("<6I2i2I")    # cta_linear, x,y,z, sm_id, warp_slot,
                                   # warp_in_cta, dyn_begin, dyn_count, pad
STAT = struct.Struct("<3Q10I")      # pc, code_addr, opcode, op_class,
                                   # optk_begin, optk_count, src_begin,
                                   # src_count, dst_begin, dst_count,
                                   # rrol_begin, rrol_count, predicate
STAX = struct.Struct("<4i2I")     # memory_space, branch_target,
                                   # wait_mask, reuse_mask, packed ctrl,
                                   # predicate_inverted
DYN  = struct.Struct("<3I4I")      # static_index, active_mask,
                                   # predicate_mask, mref_begin,
                                   # mref_count, ops_idx, gsync_idx
MREF = struct.Struct("<4i2Ii")      # space, access_kind, width_bytes,
                                   # remote_rank, lane_mask, cache_hint,
                                   # destination_count
MREX = struct.Struct("<q2Ii")     # transfer_bytes, addr_begin, addr_count,
                                   # cut_rows (tiled TMA dim-0 cut)
ADDR = struct.Struct("<Q")
OPTK = struct.Struct("<I")
REGL = struct.Struct("<I")
RROL = struct.Struct("<IB")
UKV  = struct.Struct("<IBQ")
LANE = struct.Struct("<3I")
LVAL = struct.Struct("<Q")
CORD = struct.Struct("<q")
DESC = struct.Struct("<Q")
OPS  = struct.Struct("<8I")        # ukv_begin, ukv_count, lane_begin,
                                   # lane_count, ranks_begin, ranks_count,
                                   # after_ready_mask, flags
                                   # (bit0: after_ready observation
                                   #  present; mask 0 is a real
                                   #  failed-poll observation)
OPSX = struct.Struct("<4I")        # tensor_map, cord_begin, cord_count,
                                   # desc_begin, desc_count ... 20B pad
GSYN = struct.Struct("<IIQqqQ")   # kind, lane, address u64, before/after i64, sequence u64
GSYX = struct.Struct("<2i")       # acquire_mode, post_known
TMAP = struct.Struct("<3I4Q")      # rank, element_bytes, pad, global_addr,
                                   # dims_begin(in TMDI), strides_begin, box_begin
TMDI = struct.Struct("<q")
STRS_B = struct.Struct("<I")
MBPI = struct.Struct("<4i4I")      # retained for byte-compat placeholder

_SPACE = {"global": 0, "shared": 1, "local": 2, "constant": 3,
          "generic": 4, "param": 5, "shared_cluster": 6,
          "global_system": 7}
_SPACE_R = {v: k for k, v in _SPACE.items()}
_AK = {"load": 0, "store": 1, "atomic": 2, "async_copy": 3,
       "async_reduce": 4, "prefetch": 5, "reduction": 6,
       "other": 7}
_AK_R = {v: k for k, v in _AK.items()}


class FactsWriter:
    def __init__(self):
        self._str = bytearray(b"\x00")
        self._str_map = {}

    def _s(self, s) -> int:
        if s is None:
            return 0
        off = self._str_map.get(s)
        if off is None:
            off = len(self._str)
            self._str += s.encode() + b"\x00"
            self._str_map[s] = off
        return off

    def _dyn(self, d, out):
        mb = len(out["mref"])
        for m in d.memory:
            ab = len(out["addr"])
            for a in m.lane_addresses:
                out["addr"].append(ADDR.pack(a))
            out["mref"].append(MREF.pack(
                _SPACE[m.space], _AK[m.access_kind], m.width_bytes,
                -1 if m.remote_rank is None else m.remote_rank,
                m.lane_mask, m.cache_hint,
                -1 if m.destination_count is None else m.destination_count))
            out["mrefx"].append(MREX.pack(
                -1 if m.transfer_bytes is None else m.transfer_bytes,
                ab, len(out["addr"]) - ab,
                -1 if m.cut_rows is None else m.cut_rows))
        ops_i = N32
        if d.operands is not None:
            o = d.operands
            kb = len(out["ukv"])
            for k, v in o.uniform.items():
                if isinstance(v, int):
                    out["ukv"].append(UKV.pack(
                        self._s(k), 1, v & 0xFFFFFFFFFFFFFFFF))
                else:
                    out["ukv"].append(UKV.pack(self._s(k), 0,
                                               self._s(v)))
            lb = len(out["lane"])
            def _key(k):
                return (0x80000000 | k) if isinstance(k, int) else self._s(k)
            for k, vs in o.lane.items():
                vb = len(out["lval"])
                for v in vs:
                    out["lval"].append(
                        LVAL.pack(int(v) & 0xFFFFFFFFFFFFFFFF))
                out["lane"].append(LANE.pack(_key(k), vb,
                                             len(out["lval"]) - vb))
            rb = len(out["lane"])
            for k, vs in o.ranks.items():
                vb = len(out["lval"])
                for v in vs:
                    out["lval"].append(
                        LVAL.pack(int(v) & 0xFFFFFFFFFFFFFFFF))
                out["lane"].append(LANE.pack(_key(k), vb,
                                             len(out["lval"]) - vb))
            cb = len(out["cord"])
            for v in (o.tensor_coordinates or ()):
                out["cord"].append(CORD.pack(int(v)))
            db = len(out["desc"])
            for v in (o.descriptor_words or ()):
                out["desc"].append(DESC.pack(int(v)))
            ops_i = len(out["ops"])
            out["ops"].append(OPS.pack(kb, len(out["ukv"]) - kb, lb,
                                       rb - lb, rb, len(out["lane"]) - rb,
                                       o.after_ready_mask or 0,
                                       1 if o.after_ready_mask is not None
                                       else 0))
            out["opsx"].append(OPSX.pack(o.tensor_map or 0, cb,
                                         len(out["cord"]) - cb, db))
            out["opsx"][-1] += struct.pack("<I", len(out["desc"]) - db)
        gi = N32
        if d.global_sync is not None:
            g = d.global_sync
            gi = len(out["gsyn"])
            out["gsyn"].append(GSYN.pack(
                g.kind, g.lane, g.address, g.before, g.value,
                g.sequence))
            out["gsynx"].append(GSYX.pack(g.acquire_mode, g.post_known))
        out["dyn"].append(DYN.pack(d.static_index, d.active_mask,
                                   d.predicate_mask, mb,
                                   len(out["mref"]) - mb, ops_i, gi))

    def write_unit(self, unit, out: Path, digest: bytes = b"") -> None:
        sw = _UnitStream(self)
        for w in unit.warps:
            sw.add_warp(w)
        sw.finish(unit.cluster_id, unit.ctas, unit.statics,
                  unit.tensor_maps, unit.preinitialized_mbarriers,
                  out, digest)


class _UnitStream:
    """Shared row accumulation for write_unit and StreamingFactsWriter:
    identical row order keeps both paths byte-for-byte equal."""

    def __init__(self, writer, split_statics: bool = False):
        self.w = writer
        self.out_d = {"mref": [], "mrefx": [], "addr": [], "ops": [],
                      "opsx": [], "ukv": [], "lane": [], "lval": [],
                      "cord": [], "desc": [], "gsyn": [], "gsynx": [],
                      "dyn": []}
        self.warp_rows = []
        # split_statics: emit GSIMCAP02 (no STAT-family tables, STRS =
        # dynamic strings only); statics.gsc is written once per dir by
        # write_statics_file
        self.split_statics = split_statics

    def add_warp(self, warp) -> None:
        db = len(self.out_d["dyn"])
        for d in warp.instructions:
            self.w._dyn(d, self.out_d)
        self.warp_rows.append(
            WARP.pack(warp.cta_linear, *warp.cta, warp.sm_id,
                      warp.warp_slot, warp.warp_in_cta, db,
                      len(self.out_d["dyn"]) - db, 0))

    def finish(self, cluster_id: int, ctas, statics, tensor_maps,
               preinitialized_mbarriers, out: Path,
               digest: bytes = b"") -> None:
        writer = self.w
        out_d = self.out_d
        cta_rows = [CTAS.pack(c.cta_linear, *c.cta, c.cluster_rank,
                              c.sm_id, c.peer_sm_id, c.multicast_mask,
                              c.cta_group, c.is_pair_leader,
                              c.tmem_columns, 0) for c in ctas]
        warp_rows = self.warp_rows
        stat_rows, statx_rows = [], []
        optk, regl, rrol = [], [], []
        if not self.split_statics:
            for s in statics:
                ob = len(optk)
                for o in s.operands:
                    optk.append(OPTK.pack(writer._s(o)))
                sb = len(regl)
                for r in s.source_registers:
                    regl.append(REGL.pack(writer._s(r)))
                db = len(regl)
                for r in s.destination_registers:
                    regl.append(REGL.pack(writer._s(r)))
                rb = len(rrol)
                for n, role in s.register_roles.items():
                    rrol.append(RROL.pack(writer._s(n),
                                          1 if role == "address" else 0))
                c = s.control
                packed = (c.stall & 0xF) | ((c.yield_hint & 0x1) << 4) | \
                    ((c.read_barrier & 0xF) << 5) | \
                    ((c.write_barrier & 0xF) << 9)
                stat_rows.append(STAT.pack(
                    s.pc, s.code_address or 0, writer._s(s.opcode),
                    writer._s(s.operation_class), ob, len(optk) - ob,
                    sb, len(s.source_registers),
                    db, len(s.destination_registers), rb,
                    len(s.register_roles), writer._s(s.predicate)))
                statx_rows.append(STAX.pack(
                    writer._s(s.memory_space),
                    -1 if s.branch_target is None else s.branch_target,
                    c.wait_mask, c.reuse_mask, packed,
                    1 if s.predicate_inverted else 0))
            stat_bytes = b"".join(stat_rows)
            statx_bytes = b"".join(statx_rows)
            optk_bytes = b"".join(optk)
            regl_bytes = b"".join(regl)
            rrol_bytes = b"".join(rrol)
        else:
            stat_bytes = statx_bytes = optk_bytes = regl_bytes = rrol_bytes = b""
        tmap_rows, tmdi_rows = [], []
        # TMAP row: id, element_bytes, rank, global_address, dims_begin,
        # strides_begin, box_begin (TMDI indices).  Each map's TMDI run is
        # contiguous: global_dim[rank], stride_bytes[rank-1], box_dim[rank].
        for _i in sorted(tensor_maps or {}):
            _tm = tensor_maps[_i]
            _d = len(tmdi_rows)
            tmdi_rows += [TMDI.pack(int(v)) for v in _tm.global_dim]
            _s = len(tmdi_rows)
            tmdi_rows += [TMDI.pack(int(v)) for v in _tm.stride_bytes]
            _b = len(tmdi_rows)
            tmdi_rows += [TMDI.pack(int(v)) for v in _tm.box_dim]
            tmap_rows.append(TMAP.pack(
                _i, _tm.element_bytes, _tm.rank, _tm.global_address,
                _d, _s, _b))
        mbpi_rows = [MBPI.pack(int(r[0]), int(r[1]), int(r[2]), 0,
                               0, 0, 0, 0)
                     for r in (preinitialized_mbarriers or ())]
        meta = META.pack(cluster_id, len(ctas),
                         len(warp_rows), len(statics),
                         len(out_d["dyn"]), len(out_d["mref"]),
                         *struct.unpack("<4Q", digest[:32].ljust(32, b"\0")))
        tables = [
            ("META", META.size, meta),
            ("CTAS", CTAS.size, b"".join(cta_rows)),
            ("WARP", WARP.size, b"".join(warp_rows)),
            *(() if self.split_statics else
              (("STAT", STAT.size, stat_bytes),
               ("STAX", STAX.size, statx_bytes))),
            ("DYNI", DYN.size, b"".join(out_d["dyn"])),
            ("MREF", MREF.size, b"".join(out_d["mref"])),
            ("MREX", MREX.size, b"".join(out_d["mrefx"])),
            ("ADDR", ADDR.size, b"".join(out_d["addr"])),
            ("OPSZ", OPS.size, b"".join(out_d["ops"])),
            ("OPSX", 20, b"".join(out_d["opsx"])),
            ("UKVT", UKV.size, b"".join(out_d["ukv"])),
            ("LANE", LANE.size, b"".join(out_d["lane"])),
            ("LVAL", LVAL.size, b"".join(out_d["lval"])),
            ("CORD", CORD.size, b"".join(out_d["cord"])),
            ("DESC", DESC.size, b"".join(out_d["desc"])),
            ("GSYN", GSYN.size, b"".join(out_d["gsyn"])),
            ("GSYX", GSYX.size, b"".join(out_d["gsynx"])),
            *(() if self.split_statics else
              (("OPTK", OPTK.size, optk_bytes),
               ("REGL", REGL.size, regl_bytes),
               ("RROL", RROL.size, rrol_bytes))),
            ("TMAP", TMAP.size, b"".join(tmap_rows)),
            ("TMDI", TMDI.size, b"".join(tmdi_rows)),
            ("MBPI", MBPI.size, b"".join(mbpi_rows)),
            ("STRS", 0, bytes(writer._str)),
        ]
        Path(out).write_bytes(write_file(
            MAGIC2 if self.split_statics else MAGIC, tables))


def write_statics_file(statics, out) -> None:
    """Per-facts-dir statics bundle (GSIMCAPS1): the STAT-family tables
    plus a string table containing only static strings.  Intern order
    matches the STAT row pack order in _UnitStream.finish."""
    writer = FactsWriter()
    stat_rows, statx_rows = [], []
    optk, regl, rrol = [], [], []
    for s in statics:
        ob = len(optk)
        for o in s.operands:
            optk.append(OPTK.pack(writer._s(o)))
        sb = len(regl)
        for r in s.source_registers:
            regl.append(REGL.pack(writer._s(r)))
        db = len(regl)
        for r in s.destination_registers:
            regl.append(REGL.pack(writer._s(r)))
        rb = len(rrol)
        for n, role in s.register_roles.items():
            rrol.append(RROL.pack(writer._s(n),
                                  1 if role == "address" else 0))
        c = s.control
        packed = (c.stall & 0xF) | ((c.yield_hint & 0x1) << 4) | \
            ((c.read_barrier & 0xF) << 5) | ((c.write_barrier & 0xF) << 9)
        stat_rows.append(STAT.pack(
            s.pc, s.code_address or 0, writer._s(s.opcode),
            writer._s(s.operation_class), ob, len(optk) - ob,
            sb, len(s.source_registers),
            db, len(s.destination_registers), rb,
            len(s.register_roles), writer._s(s.predicate)))
        statx_rows.append(STAX.pack(
            writer._s(s.memory_space),
            -1 if s.branch_target is None else s.branch_target,
            c.wait_mask, c.reuse_mask, packed,
            1 if s.predicate_inverted else 0))
    meta = META.pack(N32, 0, 0, len(statics), 0, 0, 0, 0, 0, 0)
    tables = [("META", META.size, meta),
              ("STAT", STAT.size, b"".join(stat_rows)),
              ("STAX", STAX.size, b"".join(statx_rows)),
              ("OPTK", OPTK.size, b"".join(optk)),
              ("REGL", REGL.size, b"".join(regl)),
              ("RROL", RROL.size, b"".join(rrol)),
              ("STRS", 0, bytes(writer._str))]
    Path(out).write_bytes(write_file(MAGIC_STATICS, tables))


class StreamingFactsWriter:
    """Per-warp append writer: same byte output as
    FactsWriter.write_unit without building the whole unit object
    graph.  add_warp_instruction/dynamic rows flow straight into the
    row buffers; finish() needs the ctas/statics/aux lists."""

    def __init__(self, split_statics: bool = False):
        self._writer = FactsWriter()
        self._stream = _UnitStream(self._writer,
                                   split_statics=split_statics)
        self._warp_instructions: list = []
        self._warp_header = None

    def begin_warp(self, cta_linear: int, cta, sm_id: int, warp_slot: int,
                   warp_in_cta: int) -> None:
        self._warp_header = model.WarpStream(
            cta_linear=cta_linear, cta=tuple(cta), warp_in_cta=warp_in_cta,
            sm_id=sm_id, warp_slot=warp_slot, instructions=[])

    def add_warp_instruction(self, inst) -> None:
        self._warp_header.instructions.append(inst)

    def end_warp(self) -> None:
        self._stream.add_warp(self._warp_header)
        self._warp_header = None

    def finish(self, cluster_id: int, ctas, statics, tensor_maps,
               preinitialized_mbarriers, out, digest: bytes = b"") -> None:
        self._stream.finish(cluster_id, ctas, statics, tensor_maps,
                            preinitialized_mbarriers, Path(out), digest)


def _str_at(strs: bytes, off: int):
    if not off:
        return None
    e = strs.index(b"\x00", off)
    return strs[off:e].decode()


class FactsReader:
    _statics_cache: dict = {}
    """Streaming per-warp reader: yields WarpStream objects one at a time;
    statics shared; aux tables accessed via memoryview slices."""

    def __init__(self, path):
        data = memoryview(Path(path).read_bytes())
        self.f = FileReader(data)
        m = META.unpack(self.f.require_table("META")[:META.size])
        (self.cluster_id, self.n_ctas, self.n_warps, self.n_stat,
         self.n_dyn, self.n_mref) = m[:6]
        self.digest = struct.pack("<4Q", *m[6:])
        self._strs = bytes(self.f.require_table("STRS"))
        self._sstrs = None        # statics.gsc STRS (GSIMCAP02)
        comp = None
        if self.f.kind == MAGIC2:
            from .semantics_common import ConversionError
            sp = Path(path).parent / "statics.gsc"
            if not sp.is_file():
                raise ConversionError(
                    f"missing_fact:facts_statics ({sp})")
            key = str(sp)
            comp = FactsReader._statics_cache.get(key)
            if comp is None:
                cd = memoryview(sp.read_bytes())
                cf = FileReader(cd)
                comp = (bytes(cf.require_table("STRS")),) + tuple(
                    cf.require_table(t) for t in
                    ("STAT", "STAX", "OPTK", "REGL", "RROL"))
                FactsReader._statics_cache[key] = comp
            self._sstrs = comp[0]
        lay = self.f.table_layout("OPSZ")
        if lay is None or lay[0] != OPS.size:
            from .semantics_common import ConversionError
            raise ConversionError(
                "unsupported:facts_format:opsz_record_bytes "
                "(regenerate facts)")
        layx = self.f.table_layout("MREX")
        if layx is not None and layx[0] != MREX.size:
            from .semantics_common import ConversionError
            raise ConversionError(
                "missing_fact:mrex_record_bytes (regenerate facts)")
            from .semantics_common import ConversionError
            raise ConversionError(
                "unsupported:facts_format:opsz_record_bytes "
                "(regenerate facts)")
        self._t = {tag: self.f.require_table(tag) for tag in
                   ("CTAS", "WARP", "DYNI", "MREF",
                    "MREX", "ADDR", "OPSZ", "OPSX", "UKVT", "LANE",
                    "LVAL", "CORD", "DESC", "GSYN", "GSYX")}
        if comp is not None:
            self._t["STAT"], self._t["STAX"] = comp[1], comp[2]
            self._aux = {"OPTK": comp[3], "REGL": comp[4],
                         "RROL": comp[5]}
        else:
            self._t["STAT"] = self.f.require_table("STAT")
            self._t["STAX"] = self.f.require_table("STAX")
            self._aux = {tag: self.f.table(tag) for tag in
                         ("OPTK", "REGL", "RROL")}
        # statics parsed lazily on demand
        self._statics = None

    def _s(self, off):
        return _str_at(self._strs, off)

    def _ss(self, off):
        return _str_at(self._sstrs if self._sstrs is not None
                       else self._strs, off)

    def tensor_maps(self):
        """{id: model.TensorMap} from TMAP/TMDI; empty when absent."""
        tm_t = self.f.table("TMAP")
        di_t = self.f.table("TMDI")
        out = {}
        if not tm_t:
            return out
        rows = len(tm_t) // TMAP.size
        total = len(di_t) // TMDI.size

        def _vals(lo, hi):
            return tuple(TMDI.unpack(di_t[j * TMDI.size:(j + 1) * TMDI.size])[0]
                         for j in range(lo, hi))
        fields = [TMAP.unpack(tm_t[i * TMAP.size:(i + 1) * TMAP.size])
                  for i in range(rows)]
        for i, (tm_id, elem, rank, gaddr, db, sb, bb) in enumerate(fields):
            end = fields[i + 1][4] if i + 1 < rows else total
            out[tm_id] = model.TensorMap(
                rank=rank, element_bytes=elem,
                global_dim=_vals(db, sb), stride_bytes=_vals(sb, bb),
                box_dim=_vals(bb, end), global_address=gaddr)
        return out

    def preinitialized_mbarriers(self):
        t = self.f.table("MBPI")
        if not t:
            return []
        return [MBPI.unpack(t[i:i + MBPI.size])[:3]
                for i in range(0, len(t), MBPI.size)]

    def ctas(self):
        t = self._t["CTAS"]
        return [model.CtaFacts(
            cta_linear=r[0], cta=(r[1], r[2], r[3]), cluster_rank=r[4],
            sm_id=r[5], peer_sm_id=r[6], multicast_mask=r[7],
            cta_group=r[8], is_pair_leader=r[9], tmem_columns=r[10])
            for r in (CTAS.unpack(t[i:i + CTAS.size])
                      for i in range(0, len(t), CTAS.size))]

    def _stat(self, i):
        st = self._t["STAT"]; sx = self._t["STAX"]
        ok = self._aux["OPTK"]; rg = self._aux["REGL"]
        rl = self._aux["RROL"]
        (pc, ca, opc, ocl, ob, oc, sb, sc, db, dc, rb, rc, pr) = \
            STAT.unpack(st[i * STAT.size:(i + 1) * STAT.size])
        (ms, bt, wm, rm, pk, inv) = STAX.unpack(
            sx[i * STAX.size:(i + 1) * STAX.size])
        operands = tuple(self._ss(OPTK.unpack(
            ok[(ob + j) * OPTK.size:(ob + j + 1) * OPTK.size])[0])
            for j in range(oc))
        src = tuple(self._ss(REGL.unpack(
            rg[(sb + j) * REGL.size:(sb + j + 1) * REGL.size])[0])
            for j in range(sc))
        dst = tuple(self._ss(REGL.unpack(
            rg[(db + j) * REGL.size:(db + j + 1) * REGL.size])[0])
            for j in range(dc))
        roles = {self._ss(RROL.unpack(
            rl[(rb + j) * RROL.size:(rb + j + 1) * RROL.size])[0]):
            "address" if RROL.unpack(
                rl[(rb + j) * RROL.size:(rb + j + 1) * RROL.size])[1]
            else "data" for j in range(rc)}
        cw = model.ControlWord(stall=pk & 0xF, yield_hint=(pk >> 4) & 1,
            read_barrier=(pk >> 5) & 0xF, write_barrier=(pk >> 9) & 0xF,
            wait_mask=wm, reuse_mask=rm)
        return model.StaticInstruction(
            pc=pc, code_address=ca or None, opcode=self._ss(opc),
            operation_class=self._ss(ocl), operands=operands,
            predicate=self._ss(pr), predicate_inverted=bool(inv),
            source_registers=src, destination_registers=dst,
            register_roles=roles, control=cw,
            memory_space=self._ss(ms),
            branch_target=None if bt < 0 else bt)

    def statics(self):
        if self._statics is None:
            self._statics = [self._stat(i) for i in range(self.n_stat)]
        return self._statics

    def warps(self):
        wt = self._t["WARP"]; dt = self._t["DYNI"]
        mt = self._t["MREF"]; mx = self._t["MREX"]
        at = self._t["ADDR"]; op = self._t["OPSZ"]
        ox = self._t["OPSX"]; uk = self._t["UKVT"]
        ln = self._t["LANE"]; lv = self._t["LVAL"]
        cd = self._t["CORD"]; dc = self._t["DESC"]
        gs = self._t["GSYN"]; gx = self._t["GSYX"]
        for i in range(0, len(wt), WARP.size):
            (cl, x, y, z, sm, ws, wic, db, dn, _p) = WARP.unpack(
                wt[i:i + WARP.size])
            insts = []
            for j in range(db, db + dn):
                row = bytes(dt[j * DYN.size:(j + 1) * DYN.size])
                (si, am, pm, mb, mc, oi, gi) = DYN.unpack(row)
                raw = bytes(row)
                mem = []
                for k in range(mb, mb + mc):
                    (sp, ak, wb, rr, lm, ch, dcn) = MREF.unpack(
                        mt[k * MREF.size:(k + 1) * MREF.size])
                    (tb, ab, ac, cr) = MREX.unpack(
                        mx[k * MREX.size:(k + 1) * MREX.size])
                    mem.append(model.MemoryReference(
                        space=_SPACE_R[sp], access_kind=_AK_R[ak],
                        width_bytes=wb, lane_addresses=tuple(
                            ADDR.unpack(at[(ab + l) * 8:
                                           (ab + l + 1) * 8])[0]
                            for l in range(ac)),
                        lane_mask=lm,
                        remote_rank=None if rr < 0 else rr,
                        cache_hint=ch,
                        destination_count=None if dcn < 0 else dcn,
                        transfer_bytes=None if tb < 0 else tb,
                        cut_rows=None if cr < 0 else cr))
                ops = None
                if oi != N32:
                    (kb, kc, lb, lc, rb, rc, arm, fl) = OPS.unpack(
                        op[oi * OPS.size:(oi + 1) * OPS.size])
                    (tm, cb, cc, dbb) = struct.unpack(
                        "<4I", ox[oi * 20:oi * 20 + 16])
                    dcc = struct.unpack("<I", ox[oi * 20 + 16:
                                                oi * 20 + 20])[0]
                    uni = {}
                    for l in range(kc):
                        ko, is_i, kv = UKV.unpack(
                            uk[(kb + l) * UKV.size:
                               (kb + l + 1) * UKV.size])
                        if kv >= 0x8000000000000000:
                            kv -= 1 << 64
                        uni[self._s(ko)] = kv if is_i else self._s(kv)
                    lane_v = {}
                    rank_v = {}
                    def _k(raw):
                        return (raw & 0x7FFFFFFF) if raw & 0x80000000 \
                            else self._s(raw)
                    for l in range(lb, lb + lc):
                        ko, vb, vc = LANE.unpack(
                            ln[l * LANE.size:(l + 1) * LANE.size])
                        lane_v[_k(ko)] = tuple(LVAL.unpack(
                            lv[(vb + m) * 8:(vb + m + 1) * 8])[0]
                            for m in range(vc))
                    for l in range(rb, rb + rc):
                        ko, vb, vc = LANE.unpack(
                            ln[l * LANE.size:(l + 1) * LANE.size])
                        rank_v[_k(ko)] = tuple(LVAL.unpack(
                            lv[(vb + m) * 8:(vb + m + 1) * 8])[0]
                            for m in range(vc))
                    ops = model.SemanticOperands(
                        uniform=uni, lane=lane_v, ranks=rank_v,
                        after_ready_mask=arm if (fl & 1) else None,
                        tensor_map=tm or None,
                        tensor_coordinates=tuple(CORD.unpack(
                            cd[(cb + m) * 8:(cb + m + 1) * 8])[0]
                            for m in range(cc)) or None,
                        descriptor_words=tuple(DESC.unpack(
                            dc[(dbb + m) * 8:(dbb + m + 1) * 8])[0]
                            for m in range(dcc)) or None)
                g = None
                if gi != N32:
                    (kk, la, ad, bf, vv, sq) = GSYN.unpack(
                        gs[gi * GSYN.size:(gi + 1) * GSYN.size])
                    (am2, pk) = GSYX.unpack(
                        gx[gi * GSYX.size:(gi + 1) * GSYX.size])
                    g = model.GlobalSyncObservation(
                        kind=kk, address=ad, value=vv, lane=la,
                        acquire_mode=am2, post_known=pk, before=bf,
                        sequence=sq)
                if mc:
                    raw += bytes(mt[mb * MREF.size:(mb + mc) * MREF.size])
                    raw += bytes(mx[mb * MREX.size:(mb + mc) * MREX.size])
                    for k in range(mb, mb + mc):
                        _tb, ab, ac, _cr = MREX.unpack(
                            mx[k * MREX.size:(k + 1) * MREX.size])
                        raw += bytes(at[ab * 8:(ab + ac) * 8])
                if oi != N32:
                    raw += bytes(op[oi * OPS.size:(oi + 1) * OPS.size])
                    raw += bytes(ox[oi * 20:(oi + 1) * 20])
                    (kb_, kc_, lb_, lc_, rb_, rc_, _a, _f) = OPS.unpack(
                        op[oi * OPS.size:(oi + 1) * OPS.size])
                    raw += bytes(uk[kb_ * UKV.size:(kb_ + kc_) * UKV.size])
                    raw += bytes(ln[lb_ * LANE.size:(lb_ + lc_ + rc_) *
                                    LANE.size])
                    (tm_, cb_, cc_, dbb_) = struct.unpack(
                        "<4I", ox[oi * 20:oi * 20 + 16])
                    dcc_ = struct.unpack("<I", ox[oi * 20 + 16:
                                                  oi * 20 + 20])[0]
                    raw += bytes(cd[cb_ * 8:(cb_ + cc_) * 8])
                    raw += bytes(dc[dbb_ * 8:(dbb_ + dcc_) * 8])
                if gi != N32:
                    raw += bytes(gs[gi * GSYN.size:(gi + 1) * GSYN.size])
                insts.append(model.DynamicInstruction(
                    raw_key=raw, static_index=si, active_mask=am,
                    predicate_mask=pm, memory=tuple(mem),
                    operands=ops, global_sync=g))
            yield model.WarpStream(
                cta_linear=cl, cta=(x, y, z), warp_in_cta=wic,
                sm_id=sm, warp_slot=ws, instructions=insts)


class _RefView:
    __slots__ = ("space", "access_kind", "width_bytes", "lane_addresses",
                 "lane_mask", "remote_rank", "cache_hint",
                 "destination_count", "transfer_bytes", "cut_rows")

    def __init__(self, mref_row, mrex_row, addr_t):
        (sp, ak, wb, rr, lm, ch, dcn) = MREF.unpack(mref_row)
        (tb, ab, ac, cr) = MREX.unpack(mrex_row)
        self.space = _SPACE_R[sp]
        self.access_kind = _AK_R[ak]
        self.width_bytes = wb
        self.lane_addresses = tuple(
            ADDR.unpack(addr_t[(ab + l) * 8:(ab + l + 1) * 8])[0]
            for l in range(ac))
        self.lane_mask = lm
        self.remote_rank = None if rr < 0 else rr
        self.cache_hint = ch
        self.destination_count = None if dcn < 0 else dcn
        self.transfer_bytes = None if tb < 0 else tb
        self.cut_rows = None if cr < 0 else cr


class DynView:
    """Lazy per-instance view over raw GSIMCAP01 table slices — same
    attribute surface as model.DynamicInstruction; fields decode on
    first access (axis functions touch only a subset)."""
    __slots__ = ("static_index", "active_mask", "predicate_mask",
                 "raw_key", "_r", "_mb", "_mc", "_oi", "_gi",
                 "_memory", "_operands", "_gsync")

    def __init__(self, r, row):
        (si, am, pm, mb, mc, oi, gi) = DYN.unpack(row)
        self._r = r
        self.static_index = si
        self.active_mask = am
        self.predicate_mask = pm
        self._mb, self._mc, self._oi, self._gi = mb, mc, oi, gi
        self._memory = self._operands = self._gsync = None
        raw = bytes(row)
        r_ = r._t
        if mc:
            raw += bytes(r_["MREF"][mb * MREF.size:(mb + mc) * MREF.size])
            raw += bytes(r_["MREX"][mb * MREX.size:(mb + mc) * MREX.size])
            at = r_["ADDR"]
            for k in range(mb, mb + mc):
                _tb, ab, ac, _cr = MREX.unpack(
                    r_["MREX"][k * MREX.size:(k + 1) * MREX.size])
                raw += bytes(at[ab * 8:(ab + ac) * 8])
        if oi != N32:
            op = r_["OPSZ"]; ox = r_["OPSX"]
            uk = r_["UKVT"]; ln = r_["LANE"]
            cd = r_["CORD"]; dc = r_["DESC"]
            raw += bytes(op[oi * OPS.size:(oi + 1) * OPS.size])
            raw += bytes(ox[oi * 20:(oi + 1) * 20])
            (kb, kc, lb, lc, rb, rc, _a, _f) = OPS.unpack(
                op[oi * OPS.size:(oi + 1) * OPS.size])
            raw += bytes(uk[kb * UKV.size:(kb + kc) * UKV.size])
            raw += bytes(ln[lb * LANE.size:(lb + lc + rc) * LANE.size])
            (tm, cb, cc, dbb) = struct.unpack(
                "<4I", ox[oi * 20:oi * 20 + 16])
            dcc = struct.unpack("<I", ox[oi * 20 + 16:oi * 20 + 20])[0]
            raw += bytes(cd[cb * 8:(cb + cc) * 8])
            raw += bytes(dc[dbb * 8:(dbb + dcc) * 8])
        if gi != N32:
            raw += bytes(r_["GSYN"][gi * GSYN.size:(gi + 1) * GSYN.size])
        self.raw_key = raw

    @property
    def memory(self):
        if self._memory is None:
            r_ = self._r._t
            self._memory = tuple(_RefView(
                r_["MREF"][k * MREF.size:(k + 1) * MREF.size],
                r_["MREX"][k * MREX.size:(k + 1) * MREX.size],
                r_["ADDR"]) for k in range(self._mb, self._mb + self._mc))
        return self._memory

    @property
    def operands(self):
        if self._operands is None:
            if self._oi == N32:
                return None
            r_ = self._r._t
            op, ox = r_["OPSZ"], r_["OPSX"]
            uk, ln = r_["UKVT"], r_["LANE"]
            lv, cd, dc = r_["LVAL"], r_["CORD"], r_["DESC"]
            oi = self._oi
            (kb, kc, lb, lc, rb, rc, arm, fl) = OPS.unpack(
                op[oi * OPS.size:(oi + 1) * OPS.size])
            (tm, cb, cc, dbb) = struct.unpack(
                "<4I", ox[oi * 20:oi * 20 + 16])
            dcc = struct.unpack("<I", ox[oi * 20 + 16:oi * 20 + 20])[0]
            uni = {}
            for l in range(kc):
                ko, is_i, kv = UKV.unpack(
                    uk[(kb + l) * UKV.size:(kb + l + 1) * UKV.size])
                if kv >= 0x8000000000000000:
                    kv -= 1 << 64
                uni[self._r._s(ko)] = kv if is_i else self._r._s(kv)
            def _kv(raw):
                return (raw & 0x7FFFFFFF) if raw & 0x80000000 \
                    else self._r._s(raw)
            lane_v = {}
            for l in range(lb, lb + lc):
                ko, vb, vc = LANE.unpack(
                    ln[l * LANE.size:(l + 1) * LANE.size])
                lane_v[_kv(ko)] = tuple(LVAL.unpack(
                    lv[(vb + m) * 8:(vb + m + 1) * 8])[0]
                    for m in range(vc))
            rank_v = {}
            for l in range(rb, rb + rc):
                ko, vb, vc = LANE.unpack(
                    ln[l * LANE.size:(l + 1) * LANE.size])
                rank_v[_kv(ko)] = tuple(LVAL.unpack(
                    lv[(vb + m) * 8:(vb + m + 1) * 8])[0]
                    for m in range(vc))
            self._operands = model.SemanticOperands(
                uniform=uni, lane=lane_v, ranks=rank_v,
                after_ready_mask=arm if (fl & 1) else None,
                tensor_map=tm or None,
                tensor_coordinates=tuple(CORD.unpack(
                    cd[(cb + m) * 8:(cb + m + 1) * 8])[0]
                    for m in range(cc)) or None,
                descriptor_words=tuple(DESC.unpack(
                    dc[(dbb + m) * 8:(dbb + m + 1) * 8])[0]
                    for m in range(dcc)) or None)
        return self._operands

    @property
    def global_sync(self):
        if self._gsync is None and self._gi != N32:
            r_ = self._r._t
            gs, gx = r_["GSYN"], r_["GSYX"]
            (kk, la, ad, bf, vv, sq) = GSYN.unpack(
                gs[self._gi * GSYN.size:(self._gi + 1) * GSYN.size])
            (am2, pk) = GSYX.unpack(
                gx[self._gi * GSYX.size:(self._gi + 1) * GSYX.size])
            self._gsync = model.GlobalSyncObservation(
                kind=kk, address=ad, value=vv, lane=la,
                acquire_mode=am2, post_known=pk, before=bf,
                sequence=sq)
        return self._gsync


class WarpView:
    __slots__ = ("cta_linear", "cta", "warp_in_cta", "sm_id",
                 "warp_slot", "instructions")

    def __init__(self, cl, cta, wic, sm, ws, insts):
        self.cta_linear = cl
        self.cta = cta
        self.warp_in_cta = wic
        self.sm_id = sm
        self.warp_slot = ws
        self.instructions = insts


def reader_warp_views(self):
    """Yields WarpView per warp: DynView instructions + a shared
    StaticsList. No DynamicInstruction materialization."""
    r_ = self._t
    wt, dt = r_["WARP"], r_["DYNI"]
    statics = self.statics()
    for i in range(0, len(wt), WARP.size):
        (cl, x, y, z, sm, ws, wic, db, dn, _p) = WARP.unpack(
            wt[i:i + WARP.size])
        insts = [DynView(self, dt[j * DYN.size:(j + 1) * DYN.size])
                 for j in range(db, db + dn)]
        yield WarpView(cl, (x, y, z), wic, sm, ws, insts)


FactsReader.warp_views = reader_warp_views
