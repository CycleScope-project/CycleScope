"""Symbol-boundary matching shared by capture selection and marker audit.

Deliberately NOT a C++ demangler: it walks only the productions the CUDA
C++ kernel symbols of this pipeline actually use, and unparsed or unmatched
symbols fail closed -- a lenient hit would pair the wrong kernel identity,
a refusal downstream stays loud.

The production set was inventoried against the real mangled corpus of this
project's captures (torch/at::native, flashinfer, cutlass/cute, deep_gemm,
cub/thrust, cublasLt): source names, substitutions (``S_``/``S<seq>_`` and
the ``St``/``Sa``/``Sb``/``Ss``/``Si``/``So``/``Sd`` abbreviations; no
vendor substitution letters -- the ``Sc``/``Sm``/``Sp`` strings in the
corpus live inside source names such as ``PersistentTileScheduler`` or
``Sm90``), builtin/qualified types, template args (``I ... E``), argument
packs (``J ... E``), literals (``L <type> <value> E``, including the enum
forms ``LN...E1E``/``LS3_0E``), local names (``Z <encoding> E``) and
closure types (``U l <sig> E [seq] _``) -- the lambda-bearing torch kernels
need those.  Anything beyond this set refuses with ``None``: vendor
substitutions, ``fp`` pack expansions, ``GV`` guards, constructor or
destructor tokens, float/negative literals, unnamed types, and operator
names other than two-letter forms that consume as builtin-type pairs.
"""

from __future__ import annotations

_IDENTIFIER = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
# Single-letter built-in type tokens and cv/ref/pointer qualifiers of the
# Itanium grammar (consumed inside skipped template-argument blocks).
_BUILTIN_TYPES = frozenset("vwbcahstijlmxynofdegzu")
_QUALIFIERS = frozenset("PKROVr")


def kernel_symbol_contains(symbol: str, stem: str) -> bool:
    """Match a resolved stem at a real symbol component boundary.

    - a canonical mangled stem (``_Z...``) is a FULL resolved name and
      matches only exact equality;
    - a mangled symbol matches a demangled stem only against the source
      names its own grammar encodes: components are length-prefixed, so
      ``_Z7abc3foo`` has the single component ``abc3foo`` (the inner
      ``3foo`` is not a boundary) and ``_Z37...KernelInfoEv`` never
      matches the shorter stem ``...Kernel``;
    - a demangled symbol matches at identifier boundaries, with ``<``
      allowed right after the stem (the same component's own template
      arguments), so demangled-template positives keep matching.
    """
    if not symbol or not stem:
        return False
    if stem.startswith("_Z"):
        return symbol == stem
    if symbol.startswith("_Z"):
        names = _mangled_source_names(symbol)
        return names is not None and stem in names
    return _identifier_boundary_match(symbol, stem)


def _identifier_boundary_match(symbol: str, stem: str) -> bool:
    start = symbol.find(stem)
    while start != -1:
        end = start + len(stem)
        before = symbol[start - 1] if start else ""
        after = symbol[end] if end < len(symbol) else ""
        if (not before or before not in _IDENTIFIER) and (
            not after or after not in _IDENTIFIER or after == "<"
        ):
            return True
        start = symbol.find(stem, start + 1)
    return False


def _mangled_source_names(symbol: str) -> list[str] | None:
    """Source-name components of the NAME part of a ``_Z`` mangled symbol.

    Walks the top-level (optionally nested ``N ... E``) name structure:
    substitutions, length-prefixed source names and their template-argument
    blocks.  The parameter encoding after the name is never entered (its
    argument types carry source names that are not the kernel's), and any
    token outside the subset returns None -- the caller refuses instead of
    guessing a boundary.
    """
    body = symbol[2:]
    index = 0
    nested = body.startswith("N")
    if nested:
        index = 1
        while index < len(body) and body[index] in "KVrO":
            index += 1
    names: list[str] = []
    while index < len(body):
        char = body[index]
        if nested and char == "E":
            return names
        if char.isdigit():
            index, name = _read_source_name(body, index)
            if index is None:
                return None
            names.append(name)
        elif char == "S":
            index = _skip_substitution(body, index)
            if index is None:
                return None
            continue
        elif not nested:
            return names  # the parameter encoding follows; never walked
        else:
            return None
        if index < len(body) and body[index] == "I":
            index = _skip_block(body, index)
            if index is None:
                return None
        if not nested:
            return names
    return names


def _read_source_name(body: str, index: int) -> tuple[int | None, str]:
    """Consume ``<length><name>`` at body[index]; (None, "") when malformed."""
    end = index
    while end < len(body) and body[end].isdigit():
        end += 1
    name_end = end + int(body[index:end])
    if name_end > len(body):
        return None, ""
    return name_end, body[end:name_end]


def _skip_substitution(body: str, index: int) -> int | None:
    """Skip ``S_`` / ``S<base36 seq>_`` / the abbreviated std ones (``St``...)."""
    index += 1
    if index >= len(body):
        return None
    if body[index] == "_" or body[index] in "tabsiod":
        return index + 1
    end = index
    while end < len(body) and (body[end].isdigit() or body[end].isupper()):
        end += 1
    if end == index or end >= len(body) or body[end] != "_":
        return None
    return end + 1


def _skip_template_param(body: str, index: int) -> int | None:
    """Skip ``T_`` / ``T<seq>_`` (function/template parameter)."""
    index += 1
    end = index
    while end < len(body) and body[end].isdigit():
        end += 1
    if end > index:
        return end + 1 if end < len(body) and body[end] == "_" else None
    return index + 1 if index < len(body) and body[index] == "_" else None


def _skip_block(body: str, index: int) -> int | None:
    """Skip one bracketed production -- template args (``I ... E``), an
    argument pack (``J ... E``) or a nested name (``N ... E``) -- returning
    the index just past its closing ``E``.
    """
    return _skip_until_close(body, index + 1)


def _skip_until_close(body: str, index: int) -> int | None:
    """Walk productions until the ``E`` that closes the current group."""
    while index < len(body):
        if body[index] == "E":
            return index + 1
        next_index = _skip_production(body, index)
        if next_index is None:
            return None
        index = next_index
    return None


def _skip_production(body: str, index: int) -> int | None:
    """Skip ONE grammar production of the corpus subset; else ``None``."""
    char = body[index]
    if char in "INJ":
        return _skip_block(body, index)
    if char == "L":
        return _skip_literal(body, index)
    if char == "S":
        return _skip_substitution(body, index)
    if char == "T":
        return _skip_template_param(body, index)
    if char == "U":
        return _skip_closure(body, index)
    if char == "Z":
        return _skip_local_name(body, index)
    if char.isdigit():
        next_index, _ = _read_source_name(body, index)
        if next_index is None:
            return None
        if next_index < len(body) and body[next_index] == "I":
            return _skip_block(body, next_index)
        return next_index
    if char in _BUILTIN_TYPES or char in _QUALIFIERS:
        return index + 1
    return None


def _skip_literal(body: str, index: int) -> int | None:
    """Skip ``L <type> <value> E``: the corpus's bool/int/enum literals.

    The type is walked with the shared grammar (builtin, qualified
    ``N ... E`` name or substitution), so an enum literal such as
    ``LNS_15PosEncodingModeE1E`` or ``LS3_0E`` closes on its OWN ``E`` --
    a naive scan to the first ``E`` closes on the nested name's ``E`` and
    mis-spans the rest of the template arguments.
    """
    next_index = _skip_production(body, index + 1)
    if next_index is None:
        return None
    index = next_index
    end = index
    while end < len(body) and body[end].isdigit():
        end += 1
    if end == index or end >= len(body) or body[end] != "E":
        return None
    return end + 1


def _skip_closure(body: str, index: int) -> int | None:
    """Skip a closure type ``U l <signature types> E [seq] _``.

    The optional digit run is the lambda discriminator (``UlvE_``,
    ``UlvE0_``, ``UlvE10_``); a member reached through a nested name (the
    call operator ``cl``) is a production of the ENCLOSING name and stays
    in the caller's stream.
    """
    if index + 1 >= len(body) or body[index + 1] != "l":
        return None
    next_index = _skip_until_close(body, index + 2)
    if next_index is None:
        return None
    index = next_index
    end = index
    while end < len(body) and body[end].isdigit():
        end += 1
    if end >= len(body) or body[end] != "_":
        return None
    return end + 1


def _skip_local_name(body: str, index: int) -> int | None:
    """Skip ``Z <function encoding> E`` of a local (function-enclosed) name.

    The encoding walks with the shared grammar (nested names, parameter
    types, closures, further ``Z`` locals); the entity name after the ``E``
    belongs to the enclosing stream and stays with the caller -- chains
    like ``ZZZ ... E NKUlvE_clEv E ... E UlbbE_`` (a lambda inside a
    lambda's call operator) therefore nest instead of terminating early.
    """
    return _skip_until_close(body, index + 1)
