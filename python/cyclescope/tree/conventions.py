"""Leaf-convention checks across both device trees
.

V1 field model: every device leaf field of a common mechanism uses the
   kind and axis/axes declared in the mechanism's ``field_models``.
V2 discrete axes: a ``discrete`` model may only key on enum axes
   (ISA-encoded quantities); continuous quantities are affine / linear.
V3 units: ``ns`` fields only in the GPU-fabric-domain mechanisms.
V4 evidence crossing: a common mechanism measured on one device and
   placeholder on the other.
V5 class resource: an instruction.service leaf's `resource` equals the
   binding in common/class_resource.json for its operation_class.

``check_conventions`` returns the violation lists; the CLI prints them as
warnings and ``--strict`` turns them into an error (plan P5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .build import _derive_status
from .source import expand_fragment, load_source_tree

ARCHES = ("hopper", "blackwell")
NS_ALLOWED = {"memory.shared_service", "memory.global_atomic", "launch.dispatch"}


def _axes_of(model: Mapping[str, Any]) -> tuple[str, ...]:
    if "axes" in model:
        return tuple(model["axes"])
    if "axis" in model:
        return (model["axis"],)
    return ()


def _declared(entry: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    return entry["kind"], tuple(entry.get("axes") or
                                ([entry["axis"]] if "axis" in entry else []))


def _crossing_status(field_sources, sources) -> str:
    """Leaf status for the evidence-crossing check: a leaf whose every
    source is structural (e.g. an unreachable-on-this-device record) is not
    an unmeasured placeholder, so it does not cross a measured leaf."""
    evidence = {sources[k]["evidence"] for keys in field_sources for k in keys}
    if evidence == {"structural"}:
        return "structural"
    return _derive_status(field_sources, sources)


def check_conventions(source_root: str | Path) -> dict[str, list[str]]:
    trees = {a: load_source_tree(source_root, a) for a in ARCHES}
    out: dict[str, list[str]] = {"field_model": [], "discrete_axis": [],
                                 "unit": [], "evidence_crossing": [],
                                 "class_resource": []}
    import json
    cr_path = Path(source_root) / "common" / "class_resource.json"
    class_res = json.loads(cr_path.read_text())["classes"] if cr_path.exists() else {}
    status: dict[str, dict[str, set[str]]] = {}
    common = {m for m in trees[ARCHES[0]].fragments
              if m in trees[ARCHES[1]].fragments}
    for arch, st in trees.items():
        for mech, frag in st.fragments.items():
            decl = frag.get("field_models") or {}
            for node in expand_fragment(mech, frag):
                p = node.get("parameter")
                if not p or not isinstance(p.get("fields"), Mapping):
                    continue
                status.setdefault(mech, {}).setdefault(arch, set()).add(
                    _crossing_status([list(x.get("sources", [])) for x in p["fields"].values()],
                                     st.sources))
                if mech == "instruction.service" and class_res:
                    cls = [seg.split("=", 1)[1] for seg in node["id"].split("/")
                           if seg.startswith("operation_class=")]
                    r = p["fields"].get("resource", {}).get("model", {})
                    if cls and cls[-1] in class_res and r.get("kind") == "constant" \
                            and r.get("value") != class_res[cls[-1]]:
                        out["class_resource"].append(
                            f"{arch} {node['id']}: resource {r.get('value')} != {class_res[cls[-1]]}")
                for f, spec in p["fields"].items():
                    m = spec["model"]
                    kind, axes = m["kind"], _axes_of(m)
                    if mech in common:
                        if f not in decl:
                            out["field_model"].append(
                                f"{arch} {node['id']}.{f}: no field_models declaration")
                        elif "axes_subset_of" in decl[f]:
                            allowed = set(decl[f]["axes_subset_of"])
                            if kind != decl[f]["kind"] or not axes or not set(axes) <= allowed:
                                out["field_model"].append(
                                    f"{arch} {node['id']}.{f}: {kind}{list(axes)} not a "
                                    f"{decl[f]['kind']} over a subset of {sorted(allowed)}")
                        else:
                            dk, da = _declared(decl[f])
                            if (kind, axes) != (dk, da):
                                out["field_model"].append(
                                    f"{arch} {node['id']}.{f}: {kind}{list(axes)} "
                                    f"!= declared {dk}{list(da)}")
                    if kind == "discrete":
                        for ax in axes:
                            a = st.axes.get(ax)
                            if a is None or (a.get("kind") != "enum" and not a.get("discrete_ok")):
                                out["discrete_axis"].append(
                                    f"{arch} {node['id']}.{f}: discrete on non-enum axis {ax}")
                    unit = spec.get("unit") or ""
                    if (unit == "ns" or f.endswith("_ns") or "_per_ns" in f) \
                            and mech not in NS_ALLOWED:
                        out["unit"].append(f"{arch} {node['id']}.{f}: ns outside the fabric domain")
    for mech in sorted(common):
        s = status.get(mech, {})
        a, b = s.get(ARCHES[0], set()), s.get(ARCHES[1], set())
        if ("measured" in a and b == {"placeholder"}) or ("measured" in b and a == {"placeholder"}):
            out["evidence_crossing"].append(f"{mech}: {ARCHES[0]} {sorted(a)} / {ARCHES[1]} {sorted(b)}")
    return out


def summary(viol: Mapping[str, list[str]]) -> str:
    return ", ".join(f"{k} {len(v)}" for k, v in viol.items())
