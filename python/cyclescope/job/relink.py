"""Relink a segment (or container) onto a new mechanism tree (§4).

Recompiles the program tables for the new tree, replays each USES row's
node by re-walking the new tree with the stored FACT axis values
(axis names come via STRS text, not indices), and rewrites
PROG/NODE/GRD /LEAF/PARM/PTS, USES.schema_id/field_mask, TIME.kind and
META.tree_digest/program_digest.  Programs present in the new tree but
absent in the old have no stored facts -> unsupported:relink_new_program;
programs gone from the new tree drop their USES/TIME rows (TIME rows are
renumbered).  The segment STRS is rebuilt as the new program STRS plus
the old builder-appended suffix re-interned in order, so a relinked
segment is byte-identical to a direct convert under the new tree.

Fast path: when the new tree compiles to the same NODE+GRD bytes, the
re-walk must land on the same nodes; USES/FACT are kept verbatim
(schema_id/field_mask/TIME.kind still rewritten).
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

from ..convert.semantics_common import ConversionError
from . import program_tables as pt_mod
from . import segment_format as sf
from .timing_rows import TimingKind


def _rows(reader, tag, record):
    raw = reader.table(tag)
    if raw is None:
        return []
    size = record.size
    return [record.struct.unpack(raw[i:i + size])
            for i in range(0, len(raw), size)]


def _strings(reader) -> bytes:
    return bytes(reader.table("STRS") or b"")


def _str_at(buf: bytes, off: int) -> str:
    end = buf.index(b"\0", off)
    return buf[off:end].decode()


def _prog_strs_end(strings: bytes, prog_rows) -> int:
    """End offset of the program-table STRS region (max program-name
    end); builder-appended strings live after it."""
    end = 0
    for row in prog_rows:
        name_end = row[0] + strings.index(b"\0", row[0]) - row[0] + 1
        end = max(end, name_end)
    return end


def _unsupported(reason: str) -> ConversionError:
    return ConversionError(f"unsupported:{reason}")


def _rewalk(runtime, program, facts):
    """Walk the new tree with stored axis-name->value facts; engine
    guards stop at their parent like traverse()."""
    nid = program["root"]
    while runtime.nodes[nid]["children"]:
        nxt = None
        for kid in runtime.nodes[nid]["children"]:
            g = runtime.nodes[kid]["guard"]
            if g is None:
                nxt = kid
                continue
            ax = runtime.axes[g["axis"]]
            if ax["owner"] == "engine":
                return nid
            name = g["axis"]
            if name not in facts:
                return None
            val = facts[name]
            if ax["kind"] == "enum":
                val = ax["values"][val] if isinstance(val, int) else val
                ok = (val == g["value"]) if g["operator"] == "eq" else \
                    (ax["values"].index(g["value"][0])
                     <= ax["values"].index(val)
                     <= ax["values"].index(g["value"][1]))
            else:
                ok = (int(val) == int(g["value"])) \
                    if g["operator"] == "eq" else \
                    int(g["value"][0]) <= int(val) <= int(g["value"][1])
            if ok:
                nxt = kid
                break
        if nxt is None:
            return None
        nid = nxt
    return nid


def _node_programs(strings, prog_rows, node_count):
    """node row -> program name via old PROG subtree ranges."""
    node_prog = {}
    sorted_rows = sorted(prog_rows, key=lambda r: r[4])
    for i, prow in enumerate(sorted_rows):
        root_row = prow[4]
        end = sorted_rows[i + 1][4] if i + 1 < len(sorted_rows) \
            else node_count
        name = _str_at(strings, prow[0])
        for r in range(root_row, end):
            node_prog[r] = name
    return node_prog


def relink_segment(data: bytes, new_tree: dict,
                   runtime=None) -> bytes:
    from ..convert.mechanism_uses import TreeRuntime
    fr = sf.read_segment(data)
    if getattr(fr, "version", 1) >= 2:
        from ..convert.semantics_common import ConversionError
        raise ConversionError(
            "unsupported:relink_split_program (rebuild)")
    reader = fr
    strings = _strings(reader)
    new_pt = pt_mod.compile_program_tables(new_tree)
    if runtime is None:
        runtime = TreeRuntime(new_tree)
    new_progs = new_tree["programs"]

    old_uses = _rows(reader, "USES", sf.USES)
    old_facts = _rows(reader, "FACT", sf.FACT)
    old_time = _rows(reader, "TIME", sf.TIME)
    old_prog_rows = _rows(reader, "PROG", sf.PROG)
    old_nodes = _rows(reader, "NODE", sf.NODE)
    old_meta = sf.META.struct.unpack(reader.require_table("META"))

    old_names = {_str_at(strings, r[0]) for r in old_prog_rows}
    for pname in new_progs:
        if pname not in old_names:
            raise _unsupported(f"relink_new_program ({pname})")

    # --- rebuild STRS: new program STRS + old suffix in order --------
    boundary = _prog_strs_end(strings, old_prog_rows)
    suffix = strings[boundary:]
    if suffix and not suffix.endswith(b"\0"):
        raise _unsupported("relink_string_table (unterminated STRS)")
    out_strings = bytearray(new_pt["STRS"])
    str_off = dict(new_pt.get("STRS_OFFSETS", {}))
    offset_map: dict[int, int] = {}

    def soff(text):
        if text not in str_off:
            str_off[text] = len(out_strings)
            out_strings.extend(text.encode() + b"\0")
        return str_off[text]

    pos = boundary
    while pos < len(strings):
        end = strings.index(b"\0", pos)
        offset_map[pos] = soff(strings[pos:end].decode())
        pos = end + 1

    def map_off(off):
        if off >= boundary:
            return offset_map[off]
        return soff(_str_at(strings, off))

    # old node row -> program name
    node_prog = _node_programs(strings, old_prog_rows, len(old_nodes))

    def time_kind(name):
        kind = new_progs[name].get("timing_kind")
        return TimingKind[kind].value if kind else 0

    same_shape = (new_pt["NODE"] == bytes(reader.require_table("NODE"))
                  and new_pt["GRD "] == bytes(
                      reader.require_table("GRD ")))

    new_uses = bytearray(new_pt["USES"])
    new_facts = bytearray(new_pt["FACT"])
    new_time = bytearray()
    time_remap: dict[int, int] = {}

    # TIME rows are re-emitted in old-table order (direct conversion
    # emits them at template creation).  A row survives if its owning
    # program still exists; ownership comes from any USES row that
    # references the old index.
    time_progs: dict[int, set] = {}
    for u in old_uses:
        node_row, target, _pad, target_index, _sid, _fm, _fb, _fc = u
        if target == 1:
            time_progs.setdefault(target_index, set()).add(
                node_prog.get(node_row))
    for old_i, old_t in enumerate(old_time):
        owners = time_progs.get(old_i, set()) & new_progs.keys()
        if not owners:
            continue            # referenced only by dropped programs
        # kind comes from the owning program's timing_kind; when
        # several programs share an index they must agree — any kept
        # name yields the same row bytes, so prefer a deterministic one
        name = sorted(owners)[0]
        time_remap[old_i] = len(new_time) // sf.TIME.size
        new_time += sf.TIME.pack_row((time_kind(name),) + old_t[1:])

    for u in old_uses:
        node_row, target, pad, target_index, schema_id, field_mask, \
            fb, fc = u
        if target not in (0, 1, 2, 3, 8):
            continue            # init-program row: new_pt carries them
        name = node_prog.get(node_row)
        if name is None or name not in new_progs:
            continue            # program dropped in the new tree
        prog = new_progs[name]
        facts = {_str_at(strings, f[0]): f[1]
                 for f in old_facts[fb:fb + fc]}
        if same_shape:
            new_node = node_row
        else:
            nid = _rewalk(runtime, prog, facts)
            if nid is None:
                continue        # program no longer applicable
            new_node = runtime.row_of[nid]
        if target == 1:
            target_index = time_remap[target_index]
        nfb = len(new_facts) // sf.FACT.size
        for f in old_facts[fb:fb + fc]:
            new_facts += sf.FACT.pack_row(
                (map_off(f[0]), f[1]))
        new_uses += sf.USES.pack_row(
            (new_node, target, pad, target_index,
             prog["schema_id"], prog["field_mask"], nfb, fc))

    out_tables = {tag: bytes(reader.table(tag) or b"")
                  for tag in sf.RECORDS}
    for tag in sf.OPTIONAL_RECORDS:
        payload = reader.table(tag)
        if payload is not None:
            out_tables[tag] = bytes(payload)
    out_tables.update({k: new_pt[k] for k in (
        "PROG", "NODE", "GRD ", "LEAF", "PARM", "PTS ", "AXIS")})
    out_tables["USES"] = bytes(new_uses)
    out_tables["FACT"] = bytes(new_facts)
    out_tables["TIME"] = bytes(new_time)
    out_tables["STRS"] = bytes(out_strings)
    out_tables["TMPL"] = _rewrite_tmpl(
        bytes(reader.require_table("TMPL")), map_off, time_remap)
    out_tables["META"] = sf.META.pack_row((
        bytes.fromhex(new_tree.get("content_digest", "00" * 32)),
        bytes.fromhex(new_tree.get(
            "program_digest",
            new_tree.get("content_digest", "00" * 32))),
        bytes.fromhex(_converter_digest()), old_meta[3], old_meta[4],
        old_meta[5], old_meta[6], old_meta[7]))
    return sf.write_segment(out_tables)


def _converter_digest() -> str:
    from .cache import converter_digest
    return converter_digest()


_TB = sf.field_offset(sf.TMPL, "timing_begin")
_TC = sf.field_offset(sf.TMPL, "timing_count")


def _rewrite_tmpl(tmpl: bytes, map_off, time_remap) -> bytes:
    """TMPL.operation_class is a STRS offset; remap into new STRS.
    timing_begin/count track the surviving template TIME rows."""
    kept_ids = sorted(time_remap)
    out = bytearray(tmpl)
    for i in range(0, len(out), sf.TMPL.size):
        old = int.from_bytes(out[i:i + 4], "little")
        out[i:i + 4] = map_off(old).to_bytes(4, "little")
        begin = int.from_bytes(out[i + _TB:i + _TB + 8], "little")
        count = int.from_bytes(out[i + _TC:i + _TC + 4], "little")
        kept = [time_remap[t] for t in range(begin, begin + count)
                if t in time_remap]
        if kept:
            out[i + _TB:i + _TB + 8] = \
                min(kept).to_bytes(8, "little")
            out[i + _TC:i + _TC + 4] = \
                len(kept).to_bytes(4, "little")
        else:
            out[i + _TB:i + _TB + 8] = \
                bisect.bisect_left(kept_ids, begin).to_bytes(8, "little")
            out[i + _TC:i + _TC + 4] = (0).to_bytes(4, "little")
    return bytes(out)


def relink_container(data: bytes, new_tree: dict) -> bytes:
    """Container relink = per-segment relink; container-level tables
    pass through (spec §4)."""
    from . import container
    reader = container.read_container(data)
    images = container.segment_images(reader)
    new_images = [relink_segment(bytes(image), new_tree)
                  for image in images]
    lnch = container.TableReader(
        container.LNCH, reader.require_table("LNCH")).unpack(0)
    launch = {f: lnch[f] for f in container.LNCH.fields[:-1]}
    launch["device"] = lnch["device"]
    return container.write_container(
        launch, new_images,
        placements=reader.table("PLCE") or b"",
        global_sync=reader.table("GSYN") or b"",
        topo=reader.table("TOPO") or b"")


def relink_file(path: Path, tree_path: Path) -> bytes:
    tree = json.loads(Path(tree_path).read_text())
    data = Path(path).read_bytes()
    head = data[:8]
    if head.startswith(b"GSIMJOB"):
        return relink_container(data, tree)
    return relink_segment(data, tree)
