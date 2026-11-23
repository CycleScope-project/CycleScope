"""Loop regions of a traced launch from its observed pc transitions.

Nodes are pcs, edges are the transitions the launch took (stall ledger
`transition` rows: a replay of captured facts).  A non-trivial strongly
connected component is one top-level loop (nested loops share their
outer loop's component).  A loop that only polls a synchronisation
object is a `wait_loop` and counts as outside.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_GUARD = re.compile(r"^@!?U?P\w*\s+")
# instructions that make a loop do work: data memory, tensor, async
# transfers, barriers, calls.  Constant-bank loads are not work.
_WORK = re.compile(
    r"^(LD|LDG|LDL|LDS|LDSM|LDTM|ST|STG|STL|STS|STSM|STTM|LDGSTS|LDGDEPBAR"
    r"|DEPBAR|CCTL|MEMBAR|FENCE|BAR|ERRBAR|WARPSYNC|CALL|SHFL|ATOM\w*|RED\w*"
    r"|TEX\w*|TLD\w*|TXQ\w*|UTMA\w*|UBLK\w*|UTC\w*|UCGABAR\w*|WARPGROUP\w*"
    r"|\w*MMA\w*)$")
_POLL = re.compile(r"^(SYNCS|NANOSLEEP)$")
# site anchors: instructions that can hold a warp by something other
# than plain register flow.  (kind, mnemonic root); the first match
# wins.  Constant-bank loads and arithmetic are not anchors.
_SITE_ANCHORS = (
    ("tensor", re.compile(r"^(WARPGROUP\w*|UCGABAR\w*|UTC\w*|\w*MMA\w*)$")),
    ("tensor_memory", re.compile(r"^(LDTM|STTM)$")),
    ("cta_barrier", re.compile(r"^BAR$")),
    ("async_wait", re.compile(r"^(DEPBAR|LDGDEPBAR|LDGSTS)$")),
    ("fence", re.compile(r"^(MEMBAR|FENCE|CCTL)$")),
    ("tma", re.compile(r"^(UTMA\w*|UBLK\w*)$")),
    ("global_load", re.compile(r"^(LD|LDG)$")),
    ("global_store", re.compile(r"^(ST|STG|ATOM\w*|RED\w*)$")),
    ("shared_local", re.compile(r"^(LDS|LDSM|LDL|STS|STSM|STL)$")),
    ("warpsync", re.compile(r"^WARPSYNC$")),
    ("exit", re.compile(r"^(EXIT|RET)$")),
)
SITE_KINDS = ("mbarrier_poll", "mbarrier_other") + \
    tuple(kind for kind, _ in _SITE_ANCHORS) + ("compute",)


def mnemonic_root(sass: str) -> str:
    """`@!P0  BRA.U !UP2, 0x10 ;` -> `BRA`."""
    text = _GUARD.sub("", sass.strip())
    token = text.split()[0] if text.split() else ""
    return token.split(".")[0].rstrip(";").upper()


# instruction kinds: the mnemonic root, plus its first modifier where
# the modifier names a different operation (SYNCS.ARRIVE against
# SYNCS.PHASECHK, BAR.SYNC against BAR.ARV, ...)
_KIND_MODIFIER = re.compile(
    r"^(SYNCS|WARPGROUP|BAR|FENCE|MEMBAR|CCTL|NANOSLEEP|SHFL|CS2R"
    r"|UTC\w*|UCGABAR\w*)$")


def instruction_kind(sass: str) -> str:
    """`@!P0 SYNCS.PHASECHK.TRANS64.TRYWAIT P0, ...` -> `SYNCS.PHASECHK`;
    `FFMA R1, ...` -> `FFMA`; empty text -> `?`."""
    text = _GUARD.sub("", sass.strip())
    token = text.split()[0] if text.split() else ""
    parts = token.rstrip(";").upper().split(".")
    if not parts[0]:
        return "?"
    if _KIND_MODIFIER.match(parts[0]) and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0]


@dataclass
class Loop:
    index: int                # ordered by lowest pc
    kind: str                 # "loop" | "wait_loop"
    pcs: frozenset[int]

    @property
    def first(self) -> int:
        return min(self.pcs)

    @property
    def last(self) -> int:
        return max(self.pcs)


def strongly_connected_components(nodes, successors) -> list[list[int]]:
    """Tarjan, iterative."""
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[int] = []
    components: list[list[int]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(successors.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(successors.get(nxt, ()))))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    return components


def find_loops(transitions: dict[tuple[int, int], int],
               sass_of: dict[int, str]) -> list[Loop]:
    """Top-level loops of the transition graph.  `sass_of` maps a pc to
    its SASS text (for the wait-loop rule); a pc without text counts as
    work."""
    successors: dict[int, list[int]] = {}
    nodes: set[int] = set()
    self_loops: set[int] = set()
    for (src, dst), count in transitions.items():
        if count <= 0:
            continue
        nodes.update((src, dst))
        successors.setdefault(src, []).append(dst)
        if src == dst:
            self_loops.add(src)
    loops = []
    for component in strongly_connected_components(sorted(nodes),
                                                   successors):
        if len(component) == 1 and component[0] not in self_loops:
            continue
        roots = [mnemonic_root(sass_of[pc]) if pc in sass_of else None
                 for pc in component]
        polls = any(r is not None and _POLL.match(r) for r in roots)
        works = any(r is None or _WORK.match(r) for r in roots)
        loops.append((min(component),
                      "wait_loop" if polls and not works else "loop",
                      frozenset(component)))
    loops.sort()
    return [Loop(index=i, kind=kind, pcs=pcs)
            for i, (_, kind, pcs) in enumerate(loops)]


def region_of(loops: list[Loop]) -> dict[int, str]:
    """pc -> "loop" for pcs of work loops; every other pc is outside."""
    out: dict[int, str] = {}
    for loop in loops:
        if loop.kind == "loop":
            for pc in loop.pcs:
                out[pc] = "loop"
    return out


def basic_blocks(pcs, transitions: dict[tuple[int, int], int],
                 step: int = 16) -> dict[int, int]:
    """pc -> leader pc of its straight-line run.  A run ends after a pc
    with an observed non-sequential transition, before a pc that is the
    target of one, and at a gap in `pcs`."""
    sources = {src for (src, dst), n in transitions.items()
               if n > 0 and dst != src + step}
    targets = {dst for (src, dst), n in transitions.items()
               if n > 0 and dst != src + step}
    out: dict[int, int] = {}
    leader = previous = None
    for pc in sorted(set(pcs)):
        if previous is None or pc != previous + step or pc in targets \
                or previous in sources:
            leader = pc
        out[pc] = leader
        previous = pc
    return out


def poll_blocks(blocks: dict[int, int], sass_of: dict[int, str]) -> set[int]:
    """Leaders of the straight-line runs that hold a try-wait or a
    nanosleep.  How often such a run executes depends on timing, so its
    execution counts differ between two runs of the same launch."""
    out = set()
    for pc, leader in blocks.items():
        text = sass_of.get(pc, "")
        root = mnemonic_root(text)
        if root == "NANOSLEEP" or (root == "SYNCS" and "TRYWAIT" in text):
            out.add(leader)
    return out


def site_kind(sass: str) -> str | None:
    """Anchor kind of an instruction, None when it is not an anchor."""
    root = mnemonic_root(sass)
    if root == "NANOSLEEP" or (root == "SYNCS" and (
            "TRYWAIT" in sass or "PHASECHK" in sass)):
        return "mbarrier_poll"
    if root == "SYNCS":
        return "mbarrier_other"
    for kind, pattern in _SITE_ANCHORS:
        if pattern.match(root):
            return kind
    return None


_PREDICATE = re.compile(r"!?(U?P[0-9]+)\b")
_HOLDS_NEXT = re.compile(r"^(BAR|DEPBAR)$")
PREDICATE_LOOKBACK = 16


def _predicate_definition(pc: int, sass_of: dict[int, str],
                          step: int) -> int | None:
    """pc of the closest instruction above `pc` in the listing that
    writes a predicate the branch at `pc` reads (first operand)."""
    used = set(_PREDICATE.findall(sass_of.get(pc, "")))
    if not used:
        return None
    for k in range(1, PREDICATE_LOOKBACK + 1):
        text = sass_of.get(pc - k * step)
        if text is None:
            return None
        operands = _GUARD.sub("", text.strip()).split(None, 1)
        if len(operands) == 2 and \
                operands[1].split(",")[0].strip() in used:
            return pc - k * step
    return None


def sites(pcs, transitions: dict[tuple[int, int], int],
          sass_of: dict[int, str], step: int = 16
          ) -> dict[int, tuple[int, str]]:
    """pc -> (first pc, kind) of its site.  A site starts at an anchor
    and runs to the next anchor of the same straight-line run: the two
    sides park a warp held by an anchor at the anchor or at one of the
    instructions after it, never before it.  Where hardware parks a
    held warp on an instruction that would start another site, that
    instruction stays with the holder: the instruction right after a
    BAR or a DEPBAR, and a branch on the predicate a try-wait / phase
    check wrote (wherever the branch is).  The pcs of a run before its
    first anchor continue the site of the run they fall through from
    (an observed sequential transition), else they are a `compute`
    site."""
    blocks = basic_blocks(pcs, transitions, step)
    out: dict[int, tuple[int, str]] = {}
    current = None
    for pc in sorted(blocks):
        kind = site_kind(sass_of.get(pc, ""))
        held = (blocks[pc] != pc and current is not None
                and current[0] == pc - step
                and _HOLDS_NEXT.match(mnemonic_root(
                    sass_of.get(pc - step, ""))))
        if kind is not None and not held:
            current = (pc, kind)
        elif blocks[pc] == pc:
            previous = out.get(pc - step)
            if previous is None or transitions.get((pc - step, pc), 0) <= 0:
                current = (pc, "compute")
            else:
                current = previous
        out[pc] = current
    for pc in out:
        if mnemonic_root(sass_of.get(pc, "")) != "BRA":
            continue
        definition = _predicate_definition(pc, sass_of, step)
        if definition is not None and \
                site_kind(sass_of[definition]) == "mbarrier_poll":
            out[pc] = out.get(definition, (definition, "mbarrier_poll"))
    return out


_BRANCH_LITERAL = re.compile(r"^BRA(\.\w+)*\s+(?:.*,\s*)?0x([0-9a-fA-F]+)\s*;?$")
_JUMP = re.compile(r"^BRA(\.\w+)*\s+0x([0-9a-fA-F]+)\s*;?$")
NEAR_INSTRUCTIONS = 8


def branch_text_shift(transitions: dict[tuple[int, int], int],
                      sass_of: dict[int, str]) -> int | None:
    """Listing branch targets are printed against another base than the
    function offsets; the difference is read off the taken branches of
    the launch (the most frequent `printed target - observed target`)."""
    votes: dict[int, int] = {}
    for (src, dst), count in transitions.items():
        if count <= 0 or dst == src + 16 or src not in sass_of:
            continue
        match = _BRANCH_LITERAL.match(_GUARD.sub("", sass_of[src].strip()))
        if match:
            shift = int(match.group(2), 16) - dst
            votes[shift] = votes.get(shift, 0) + 1
    return max(votes, key=lambda k: (votes[k], -abs(k))) if votes else None


def place_unissued(regions: dict[int, str], issued, pcs,
                   sass_of: dict[int, str],
                   transitions: dict[tuple[int, int], int],
                   step: int = 16) -> tuple[dict[int, str], set[int]]:
    """Regions of the pcs in `pcs` the launch never issued (the replay
    keeps one instance of a polled wait, so the inline try-wait and its
    branch can be missing from it while hardware waits there).  Such a
    pc takes the region of the closest issued pc below it when that is
    within NEAR_INSTRUCTIONS; else of the pc the next unconditional
    branch of its listing run jumps to (an out-of-line wait path returns
    into the code it serves).  Returns the completed pc -> region map
    and the pcs neither rule could place (they stay outside)."""
    import bisect
    ordered = sorted(issued)
    known = set(ordered)
    out = dict(regions)
    unplaced: set[int] = set()
    shift = None
    shift_known = False

    def issued_at_or_after(pc: int) -> int | None:
        i = bisect.bisect_left(ordered, pc)
        if i < len(ordered) and ordered[i] - pc <= NEAR_INSTRUCTIONS * step:
            return ordered[i]
        return None

    for pc in sorted(set(pcs) - known):
        i = bisect.bisect_left(ordered, pc)
        if i and pc - ordered[i - 1] <= NEAR_INSTRUCTIONS * step:
            if ordered[i - 1] in regions:
                out[pc] = regions[ordered[i - 1]]
            continue
        if not shift_known:
            shift, shift_known = branch_text_shift(transitions, sass_of), True
        target = None
        if shift is not None:
            for k in range(2 * NEAR_INSTRUCTIONS):
                text = sass_of.get(pc + k * step)
                if text is None:
                    break
                match = _JUMP.match(text.strip())
                if match:
                    target = issued_at_or_after(
                        int(match.group(2), 16) - shift)
                    break
        if target is None:
            unplaced.add(pc)
        elif target in regions:
            out[pc] = regions[target]
    return out, unplaced
