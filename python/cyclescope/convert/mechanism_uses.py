"""Mechanism-tree traversal and per-instruction row emission (2b-4).

Walks each candidate program's guard tree for a dynamic instruction,
collects the capture-axis facts, and yields TIME/USES/FACT rows.
Engine-owned axes stop the walk (the engine resumes at that node).
A capture guard whose value is missing -> missing_fact recorded per
program; a guard with no matching child -> program not applicable, no
rows.  Non-executing instances bind only programs whose guard tree
contains the ``predicated_on`` axis.
"""

from __future__ import annotations

from .axes import axis_value, NOT_APPLICABLE


class TreeRuntime:
    """Precompiled tree: node rows, guard rows, programs, leaf axes."""

    def __init__(self, tree):
        self.axes = tree["axes"]
        nodes = {n["id"]: n for n in tree["nodes"]}
        self.nodes = nodes
        self.programs = tree["programs"]
        # DFS node-row order identical to program_tables emit
        self.row_of = {}
        self.leaf_of = {}
        leaf_count = 0

        def emit(nid):
            row = len(self.row_of)
            self.row_of[nid] = row
            node = nodes[nid]
            for k in node["children"]:
                emit(k)
            if not node["children"]:
                self.leaf_of[nid] = len(self.leaf_of)

        for name in sorted(self.programs):
            emit(self.programs[name]["root"])
        # children split into: unguarded kids, engine-guarded kids,
        # capture-eq index (when every guarded kid shares one capture
        # axis with eq), else a linear capture-guard list
        self.eq_index = {}      # nid -> (axis, {value: kid}, fallback kids)
        self.plan = {}          # nid -> [(kid, guard|None)]
        self.engine_child = {}  # nid -> True
        for nid, node in nodes.items():
            kids = node["children"]
            if not kids:
                continue
            plan = [(k, nodes[k]["guard"]) for k in kids]
            self.plan[nid] = plan
            guarded = [(k, g) for k, g in plan if g is not None]
            if any(self.axes[g["axis"]]["owner"] == "engine"
                   for _k, g in guarded):
                self.engine_child[nid] = True
            cap = [(k, g) for k, g in guarded
                   if self.axes[g["axis"]]["owner"] != "engine"]
            if (cap and not self.engine_child.get(nid)
                    and all(g["operator"] == "eq"
                            and g["axis"] == cap[0][1]["axis"]
                            for _k, g in cap)):
                axis = cap[0][1]["axis"]
                self.eq_index[nid] = (
                    axis, {g["value"]: k for k, g in cap},
                    [k for k, g in plan if g is None])
        # required capture axes per node (subtree leaf models) — one
        # post-order pass
        self.required_axes = {}
        order = []
        seen = set()

        def post(nid):
            if nid in seen:
                return
            seen.add(nid)
            for k in nodes[nid]["children"]:
                post(k)
            order.append(nid)

        # capture axes the engine supplies for a program (relation
        # programs bound on the consumer side): not required, and a
        # guard on one stops the walk like an engine-owner guard
        self.engine_axes_of = {}
        for name in self.programs:
            post(self.programs[name]["root"])
            ea = frozenset(self.programs[name].get("engine_axes") or ())
            stack = [self.programs[name]["root"]]
            while stack:
                nid = stack.pop()
                self.engine_axes_of[nid] = ea
                stack.extend(nodes[nid]["children"])
        for nid in order:
            node = nodes[nid]
            need = set()
            leaf = node.get("parameter") or {}
            for fdef in leaf.get("fields", {}).values():
                ax = (fdef.get("model") or {}).get("axis")
                if ax and self.axes.get(ax, {}).get("owner") == "capture":
                    need.add(ax)
                for ax2 in (fdef.get("model") or {}).get("axes", ()):
                    if self.axes.get(ax2, {}).get("owner") == "capture":
                        need.add(ax2)
            for k in node["children"]:
                need |= set(self.required_axes[k])
            need -= self.engine_axes_of[nid]
            self.required_axes[nid] = tuple(sorted(need))
        # programs whose guard tree contains the predicated_on axis
        # (the only ones a non-executing instance may bind)
        self.predicated_programs = set()
        for name, prog in self.programs.items():
            stack = [prog["root"]]
            while stack:
                n = nodes[stack.pop()]
                g = n["guard"]
                if g and g["axis"] == "predicated_on":
                    self.predicated_programs.add(name)
                    break
                stack.extend(n["children"])
        # programs indexed by operation_class
        self.by_class = {}
        for name, prog in self.programs.items():
            if prog.get("event") in (None, "initialization") \
                    or prog["event"] == "initialization":
                continue
            for cls in prog.get("operation_classes", ()):
                self.by_class.setdefault(cls, []).append(name)


def _guard_match(axis_def, value, guard):
    if axis_def["kind"] == "enum":
        gval = guard["value"]
        if guard["operator"] == "eq":
            return value == gval
        lo, hi = gval
        vals = axis_def["values"]
        return vals.index(lo) <= vals.index(value) <= vals.index(hi)
    if guard["operator"] == "eq":
        return int(value) == int(guard["value"])
    return int(guard["value"][0]) <= int(value) <= int(guard["value"][1])


def _guard_set_match(axis_def, values, guard):
    """True when any member of the class set satisfies the guard
    (operation_class only; enum or int range)."""
    vals = axis_def["values"]
    if guard["operator"] == "eq":
        return guard["value"] in values
    lo, hi = guard["value"]
    lo_i, hi_i = vals.index(lo), vals.index(hi)
    return any(lo_i <= vals.index(v) <= hi_i for v in values
               if v in vals)


def _axis(runtime, name, static, dyn, context, cache):
    if name in cache:
        return cache[name]
    val = axis_value(name, static, dyn, context)
    cache[name] = val
    return val


def traverse(runtime: TreeRuntime, static, dyn, context, where: str,
             cls=None):
    """-> list of {program, node_id, stop, facts} plus {"missing": key}
    rows for programs dropped by a missing capture fact."""
    if cls is None:
        cls = context.rules.operation_class(static, dyn) \
            if context and context.rules else \
            static.opcode.upper().split(".", 1)[0]
    classes = [cls]
    if context and getattr(context.rules, "operation_classes", None):
        classes = context.rules.operation_classes(static, dyn)
    elif cls not in classes:
        classes.append(cls)
    executing = dyn is None or \
        (dyn.active_mask & dyn.predicate_mask) != 0
    candidates = []
    seen = set()
    for member in classes:
        for name in runtime.by_class.get(member, ()):
            if name not in seen:
                seen.add(name)
                candidates.append(name)
    if not executing:
        candidates = [n for n in candidates
                      if n in runtime.predicated_programs]
    out = []
    cache: dict = {}
    for name in candidates:
        prog = runtime.programs[name]
        # operation_class axis inside this program: recorded FACT =
        # effective class when listed (or the program is classless),
        # else the class-set member that named this program; guards
        # evaluate the axis against the whole class set (spec 3.1).
        prog_classes = set(prog.get("operation_classes") or ())
        if cls in prog_classes or not prog_classes:
            member = cls
        else:
            member = next((c for c in classes if c in prog_classes),
                          cls)
        facts = {}
        nid = prog["root"]
        stop_node = None
        applicable = True
        lost = None
        while True:
            node = runtime.nodes[nid]
            if not node["children"]:
                stop_node = nid
                break
            nxt = None
            eq = runtime.eq_index.get(nid)
            if eq is not None:
                ax_name, by_value, unguarded = eq
                if ax_name == "operation_class":
                    val = member
                    nxt = None
                    for cand in classes:
                        if cand in by_value:
                            nxt = by_value[cand]
                            break
                    if nxt is None and unguarded:
                        nxt = unguarded[-1]
                else:
                    val = _axis(runtime, ax_name, static, dyn,
                                context, cache)
                    if val is NOT_APPLICABLE:
                        applicable = False
                        break
                    if val is None:
                        lost = f"axis:{ax_name}"
                        break
                    nxt = by_value.get(val)
                    if nxt is None and unguarded:
                        nxt = unguarded[-1]
                facts[ax_name] = val
            else:
                for kid, g in runtime.plan[nid]:
                    if g is None:
                        nxt = kid
                        continue
                    ax_name = g["axis"]
                    ax = runtime.axes[ax_name]
                    if ax["owner"] == "engine" or \
                            ax_name in runtime.engine_axes_of[nid]:
                        stop_node = nid      # engine evaluates all
                        break                # sibling guards
                    if ax_name == "operation_class":
                        val = member
                        if not _guard_set_match(ax, classes, g):
                            continue
                    else:
                        val = _axis(runtime, ax_name, static, dyn,
                                    context, cache)
                        if val is NOT_APPLICABLE:
                            applicable = False
                            break
                        if val is None:
                            lost = f"axis:{ax_name}"
                            break
                        if not _guard_match(ax, val, g):
                            continue
                    facts[ax_name] = val
                    nxt = kid
                    break
            if lost is not None or not applicable \
                    or stop_node is not None:
                break
            if nxt is None:
                applicable = False
                break
            nid = nxt
        if lost is not None:
            out.append({"missing": lost, "program": name})
            continue
        if not applicable:
            continue
        out.append({"program": name, "prog": prog, "node": stop_node,
                    "facts": facts, "cache": cache})
    return out


def program_facts(runtime: TreeRuntime, static, dyn, context, where,
                  cls=None):
    """Traverse + collect required leaf-model axes into FACT tuples."""
    rows = traverse(runtime, static, dyn, context, where, cls)
    out = []
    for row in rows:
        if "missing" in row:
            out.append(row)
            continue
        facts = dict(row["facts"])
        cache = row["cache"]
        lost = None
        for ax_name in runtime.required_axes[row["node"]]:
            if ax_name in facts:
                continue
            val = _axis(runtime, ax_name, static, dyn, context, cache)
            if val is NOT_APPLICABLE:
                row["skip"] = True
                break
            if val is None:
                lost = f"axis:{ax_name}"
                break
            facts[ax_name] = val
        if lost is not None:
            out.append({"missing": lost, "program": row["program"]})
            continue
        if row.get("skip"):
            continue
        row["facts"] = facts
        out.append(row)
    return out
