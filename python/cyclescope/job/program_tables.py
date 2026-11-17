"""Compile a generated mechanism tree into program tables (spec §6).

Input is a ``configs/trees/<device>.json`` generated tree (step 1 format).
Output is PROG/NODE/GRD /LEAF/PARM/PTS /FACT/USES payloads plus the STRS
payload carrying mechanism names and operation_class strings.

``USES`` rows are emitted for the job-global programs: ``event ==
initialization`` (Resource, GpuMemory, Scheduler…) and the class-less
``event == memory_issue`` memory-level programs (``memory.*.service``: the
engine reads their level latencies / intervals / bandwidths at load).
Instruction-level uses are appended by the segment builder in step 2b via
``instruction_uses_row``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..io.fixed_tables import TableWriter
from . import segment_format as sf

TARGET_BY_NAME = {
    "InstructionTemplate": 0, "TimingRow": 1, "InstanceTiming": 2,
    "MemoryAccess": 3, "Resource": 4, "MemoryLevel": 5, "GpuMemory": 6,
    "Scheduler": 7, "Instance": 8, "none": 15,
}
MODEL_KINDS = {"constant": 0, "discrete": 1, "linear": 2, "affine": 3}
GUARD_OPS = {"eq": 0, "range": 1}
STATUS_IDS = {"measured": 0, "experimental": 1, "placeholder": 2,
              "unsupported": 3, "structural": 4}
EVENT_IDS = {"initialization": 0, "issue": 1, "memory_issue": 2,
             "barrier_release": 3, "collective_launch": 4,
             "control_transition": 5}
NONE = 0xFFFFFFFF


def _guard_row(axes: Mapping[str, Any], axis_index: Mapping[str, int],
               guard: Mapping[str, Any]) -> tuple:
    ax = axes[guard["axis"]]
    index = axis_index[guard["axis"]]
    if ax["kind"] == "enum":
        if guard["operator"] == "eq":
            v = ax["values"].index(guard["value"])
            return (index, GUARD_OPS["eq"], b"\0" * 3, v, v)
        lo, hi = (ax["values"].index(v) for v in guard["value"])
        return (index, GUARD_OPS["range"], b"\0" * 3, lo, hi)
    if guard["operator"] == "eq":
        v = int(guard["value"])
        return (index, GUARD_OPS["eq"], b"\0" * 3, v, v)
    return (index, GUARD_OPS["range"], b"\0" * 3,
            int(guard["value"][0]), int(guard["value"][1]))


def _fact_value(ax: Mapping[str, Any], guard: Mapping[str, Any]) -> int:
    value = guard["value"]
    if isinstance(value, list):
        value = value[0]
    if ax["kind"] == "enum":
        return ax["values"].index(value)
    return int(value)


def _program_leaves(nodes: Mapping[str, Any], root: str) -> list[str]:
    out, stack = [], [root]
    while stack:
        nid = stack.pop()
        if nodes[nid]["children"]:
            stack.extend(reversed(nodes[nid]["children"]))
        else:
            out.append(nid)
    return out


def compile_program_tables(tree: Mapping[str, Any],
                           registry=None) -> dict[str, bytes]:
    axes = tree["axes"]
    axis_index = {name: i for i, name in enumerate(sorted(axes))}
    node_by_id = {n["id"]: n for n in tree["nodes"]}
    prog_names = sorted(tree["programs"])

    # --- node rows in DFS order so children are contiguous --------------
    node_rows: list[list] = []
    row_of: dict[str, int] = {}
    leaf_ids: list[str] = []
    leaf_row_of: dict[str, int] = {}

    def emit(node_id: str, parent_row: int) -> None:
        row = len(node_rows)
        row_of[node_id] = row
        guard = node_by_id[node_id]["guard"]
        node_rows.append([parent_row, 0, 0, 0, 0, NONE])
        kids = node_by_id[node_id]["children"]
        if kids:
            node_rows[row][1] = len(node_rows)
            node_rows[row][2] = len(kids)
            for kid in kids:
                emit(kid, row)
        else:
            leaf_row_of[node_id] = len(leaf_ids)
            leaf_ids.append(node_id)
            node_rows[row][5] = leaf_row_of[node_id]
        if guard is not None:
            # guard rows are appended lazily below via guard_of
            pass

    for name in prog_names:
        emit(tree["programs"][name]["root"], NONE)
    # every generated node is reachable from a program root
    if len(row_of) != len(tree["nodes"]):
        missing = set(node_by_id) - set(row_of)
        raise ValueError(f"unreachable nodes: {sorted(missing)[:5]}")

    # guard rows (one row per guarded node; guard_count is 0 or 1 per node)
    grd = TableWriter(sf.GRD_)
    for node_id, row in row_of.items():
        guard = node_by_id[node_id]["guard"]
        if guard is None:
            continue
        node_rows[row][3] = grd.count
        node_rows[row][4] = 1
        grd.append(_guard_row(axes, axis_index, guard))

    node_t = TableWriter(sf.NODE)
    for row in node_rows:
        node_t.append(tuple(row))

    # --- PROG + STRS -----------------------------------------------------
    strings = bytearray()
    string_offset: dict[str, int] = {}

    def str_off(text: str) -> int:
        if text not in string_offset:
            string_offset[text] = len(strings)
            strings.extend(text.encode("utf-8") + b"\0")
        return string_offset[text]

    prog = TableWriter(sf.PROG)
    for name in prog_names:
        prog_def = tree["programs"][name]
        classes = prog_def.get("operation_classes") or []
        op_begin = len(strings)
        for cls in classes:
            strings.extend(cls.encode("utf-8") + b"\0")
        _tk = 0
        if prog_def.get("timing_kind"):
            from .timing_rows import TimingKind
            _tk = TimingKind[prog_def["timing_kind"]].value
        prog.append((str_off(name), EVENT_IDS[prog_def["event"]],
                     prog_def["schema_id"], _tk,
                     row_of[prog_def["root"]],
                     prog_def["field_mask"], op_begin, len(classes)))

    # --- LEAF + PARM + PTS + FACT ---------------------------------------
    leaf_t = TableWriter(sf.LEAF)
    parm = TableWriter(sf.PARM)
    pts = TableWriter(sf.PTS_)
    fact = TableWriter(sf.FACT)
    fact_range_of: dict[str, tuple[int, int]] = {}

    for nid in leaf_ids:
        node = node_by_id[nid]
        leaf = node.get("parameter") or {
            "fields": {}, "status": "unsupported"}
        parm_begin = parm.count
        for name, fdef in sorted(leaf["fields"].items()):
            model = fdef["model"]
            kind = model["kind"]
            base = (fdef["schema_id"], fdef["field_id"], MODEL_KINDS[kind],
                    b"\0" * 3)
            if kind == "constant":
                parm.append(base + (NONE, float(model["value"]), 0.0, 0.0,
                                    0, 0))
            elif kind == "discrete":
                begin = pts.count
                width = len(model["axes"])
                # each axis row carries its numeric axis id + 1 in the
                # value column (0 = legacy row without an id): the engine
                # matches every axis of the point, not only the first
                for point in model["points"]:
                    *avs, val = point
                    for ax_name, av in zip(model["axes"], avs):
                        ax = axes[ax_name]
                        pts.append((float(av if ax["kind"] == "int"
                                          else ax["values"].index(av)),
                                    float(axis_index[ax_name] + 1)))
                    pts.append((float(val), 0.0))
                # a multi-axis discrete model without an explicit
                # placeholder_default treats a point miss as a missing
                # fact: emit fail_closed so the engine rejects instead
                # of silently taking 0
                fail_closed = bool(model.get("fail_closed")) or (
                    "placeholder_default" not in model)
                default = model.get("placeholder_default", 0.0)
                parm.append(base + (axis_index[model["axes"][0]],
                                    float(default), float(width),
                                    float(fail_closed),
                                    begin,
                                    len(model["points"]) * (width + 1)))
            elif kind == "linear" and "points" in model:
                begin = pts.count
                for point in model["points"]:
                    pts.append((float(point[0]), float(point[1])))
                parm.append(base + (axis_index[model["axis"]], 0.0, 0.0,
                                    0.0, begin, len(model["points"])))
            else:  # affine / slope-intercept linear
                ax = model["axis"]
                domain = model.get("domain") or [0.0, 0.0]
                parm.append(base + (axis_index[ax], float(domain[0]),
                                    float(model.get("slope", 0.0)),
                                    float(model.get("intercept", 0.0)),
                                    0, 0))
        leaf_t.append((parm_begin, STATUS_IDS[leaf["status"]], b"\0" * 3))

        # facts: axis assignments along the guard path, root-first
        path, cur = [], nid
        while cur is not None:
            g = node_by_id[cur]["guard"]
            if g is not None:
                path.append(g)
            cur = node_by_id[cur]["parent"]
        path.reverse()
        begin = fact.count
        for g in path:
            fact.append((axis_index[g["axis"]],
                         _fact_value(axes[g["axis"]], g)))
        fact_range_of[nid] = (begin, fact.count - begin)

    # --- USES: initialization programs only (non-instruction targets) ---
    uses = TableWriter(sf.USES)
    for name in prog_names:
        prog_def = tree["programs"][name]
        if prog_def["event"] != "initialization" and not (
                prog_def["event"] == "memory_issue"
                and not prog_def["operation_classes"]):
            continue
        target_name = (registry.schema(prog_def["parameter_schema"]).target
                       if registry is not None else "none")
        target = TARGET_BY_NAME.get(target_name, 15)
        target_index = prog_def.get("target_index", 0)
        for nid in _program_leaves(node_by_id, prog_def["root"]):
            fb, fc = fact_range_of[nid]
            uses.append((row_of[nid], target, b"\0" * 3, target_index,
                         prog_def["schema_id"],
                         prog_def["field_mask"], fb, fc))

    axis_t = TableWriter(sf.AXIS)
    for name in sorted(axes):
        off = string_offset.get(name)
        if off is None:
            off = len(strings)
            strings.extend(name.encode("utf-8") + b"\0")
            string_offset[name] = off
        axis_t.append((off,))
    # required capture axes per node row (post-order over subtree
    # leaf models) — C++ cannot reconstruct discrete multi-axis lists
    # from PARM rows, so emit them (mirrors TreeRuntime.required_axes).
    _req = {}
    _order = []
    _seen = set()
    def _post(nid):
        if nid in _seen:
            return
        _seen.add(nid)
        for k in node_by_id[nid]["children"]:
            _post(k)
        _order.append(nid)
    for name in prog_names:
        _post(tree["programs"][name]["root"])
    _engine_axes_of: dict = {}
    for _pname, _prog in tree["programs"].items():
        ea = frozenset(_prog.get("engine_axes") or ())
        stack = [_prog["root"]]
        while stack:
            nid = stack.pop()
            _engine_axes_of[nid] = ea
            stack.extend(node_by_id[nid]["children"])
    for nid in _order:
        node = node_by_id[nid]
        need = set()
        for fdef in (node.get("parameter") or {}).get(
                "fields", {}).values():
            m = fdef.get("model") or {}
            ax = m.get("axis")
            if ax and axes.get(ax, {}).get("owner") == "capture":
                need.add(ax)
            for ax2 in m.get("axes", ()):
                if axes.get(ax2, {}).get("owner") == "capture":
                    need.add(ax2)
        for k in node["children"]:
            need |= set(_req[k])
        # axes the engine supplies for this program are not required
        # capture facts (program.engine_axes)
        need -= _engine_axes_of[nid]
        _req[nid] = sorted(need)
    rax_t = TableWriter(sf.RAX_)
    raxv_t = TableWriter(sf.RAXV)
    for _nid, _row in sorted(row_of.items(), key=lambda kv: kv[1]):
        rax_t.append((raxv_t.count, len(_req[_nid])))
        for a in _req[_nid]:
            raxv_t.append((axis_index[a],))
    axv_t = TableWriter(sf.AXV_)
    axvn = bytearray()
    for name in sorted(axes):
        ax = axes[name]
        vals = ax.get("values") or ()
        vb = len(axvn)
        for v in vals:
            axvn.extend(str(v).encode("utf-8") + b"\0")
        axv_t.append((1 if ax["kind"] == "enum" else 0,
                      1 if ax["owner"] == "engine" else 0,
                      0, b"\0" * 2, vb, len(vals)))

    # --- USES: execution-unit (resource) definitions ----------------
    # tree["resources"] carries the unit table the service leaves bind
    # against by id; emit one leaf + USES row per entry so the engine
    # can resolve scope/copies/max_inflight/interval.  RAX_ stays
    # node-index aligned (empty required-axes row per appended node).
    _SCOPE_IDS = {"smsp": 0, "sm": 1, "cluster": 2, "gpu": 3}
    for i, res in enumerate(tree.get("resources") or []):
        leaf_row = leaf_t.count
        parm_begin = parm.count
        for fid, val in enumerate((
                _SCOPE_IDS.get(res["scope"], 0),
                res.get("copy_count", 0),
                res.get("max_inflight_per_copy", 0.0),
                res.get("initiation_interval_cycles", 0.0))):
            parm.append((2, fid, MODEL_KINDS["constant"], b"\0" * 3,
                         NONE, float(val), 0.0, 0.0, 0, 0))
        leaf_t.append((parm_begin, STATUS_IDS["structural"], b"\0" * 3))
        node_row = node_t.count
        node_t.append((NONE, 0, 0, 0, 0, leaf_row))
        rax_t.append((raxv_t.count, 0))
        prog.append((str_off(f"execution_units.{res['name']}"),
                     EVENT_IDS["initialization"], 2, 0, node_row,
                     0xF, len(strings), 0))
        uses.append((node_row, TARGET_BY_NAME.get("Resource", 4),
                     b"\0" * 3, i, 2, 0xF, 0, 0))

    return {
        "PROG": prog.payload(), "NODE": node_t.payload(),
        "GRD ": grd.payload(), "LEAF": leaf_t.payload(),
        "PARM": parm.payload(), "PTS ": pts.payload(),
        "FACT": fact.payload(), "USES": uses.payload(),
        "AXIS": axis_t.payload(),
        "AXV ": axv_t.payload(),
        "RAX ": rax_t.payload(), "RAXV": raxv_t.payload(),
        "AXVN": bytes(axvn),
        "STRS": bytes(strings),
        "STRS_OFFSETS": dict(string_offset),
        "_meta": {"row_of": row_of, "leaf_row_of": leaf_row_of,
                  "axis_index": axis_index},
    }


PROGRAM_FILE_TABLES = ("PROG", "NODE", "GRD ", "LEAF", "PARM", "PTS ",
                       "FACT", "USES", "AXIS", "STRS", "AXV ", "RAX ",
                       "RAXV", "AXVN")


def program_file_image(tree: Mapping[str, Any], registry) -> bytes:
    """GSIMPRG01 image (docs/en/formats.md §3): compiled tables + SCHM
    (schema_id -> USES target) + META (tree/program digests)."""
    import struct
    from ..io.file_header import write_file
    tabs = compile_program_tables(tree)
    schm = b"".join(
        struct.pack("<HBB", s.schema_id, TARGET_BY_NAME.get(s.target, 15), 0)
        for s in sorted(registry.schemas.values(), key=lambda x: x.schema_id))
    digest = bytes.fromhex(tree.get("content_digest", "00" * 32))
    pdig = bytes.fromhex(tree.get("program_digest",
                                  tree.get("content_digest", "00" * 32)))
    tables = [(k, 0, tabs[k]) for k in PROGRAM_FILE_TABLES]
    tables.append(("SCHM", 4, schm))
    tables.append(("META", 0, digest + pdig))
    return write_file("GSIMPRG01", tables)


def instruction_uses_row(leaf_index: int, target: int, target_index: int,
                         schema_id: int, field_mask: int,
                         fact_begin: int, fact_count: int) -> tuple:
    """Row the segment builder (step 2b) appends per instruction leaf."""
    return (leaf_index, target, b"\0" * 3, target_index, schema_id,
            field_mask, fact_begin, fact_count)
