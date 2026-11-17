#!/usr/bin/env python3
"""Static SASS gate for the FlashInfer delivery marker channel (T0005 rev3-②/rev5).

Gate (mirrors the DG/FlashMLA static gates and the CUTLASS capture-side marker-PC selection —
opcode CS2R with an SR_CLOCKLO *operand*, not the `CS2R R, SRZ` zeroing mnemonic; >=2 PCs per
instrumented kernel): every kernel in the requested class table must carry >=2 SR_CLOCKLO and
>=2 SR_GLOBALTIMERLO reads (entry + exit of the outermost clock-marker pair).

CPU-pod AOT usage (no GPU needed):

    nvcc -cubin (module flags) <probe/real TU> -o probe.cubin
    cuobjdump --dump-sass probe.cubin > probe.sass
    python3 marker_sass_audit.py probe.sass --kernels KernelA[,KernelB] [--summary out.json]

Exit 0 iff every requested kernel passes; kernels absent from the dump fail.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
from hoppersim.kernel_symbols import kernel_symbol_contains  # noqa: E402


def kernel_sections(sass_text: str) -> dict[str, str]:
    """Split a cuobjdump --dump-sass dump into per-kernel text blocks (by `Function :`).

    cuobjdump may print demangled template names or fully mangled names; both are kept as
    keys, template arguments stripped from the demangled base.
    """
    parts = re.split(r"^\s*Function : ", sass_text, flags=re.M)[1:]
    kernels: dict[str, str] = {}
    for part in parts:
        name = part.split("\n", 1)[0].strip()
        base = name.split("<", 1)[0]
        kernels[base] = kernels.get(base, "") + "\n" + part
    return kernels


def find_kernel_section(kernels: dict[str, str], stem: str) -> tuple[str | None, str | None]:
    """Exact or symbol-boundary (mangled-name) lookup; returns (section_key, body).

    The substring fallback is boundary-aware: a stem must not match inside a
    longer identifier.  "AppendPagedKVCacheKernel" is a substring of the
    UNinstrumented NVFP4QuantizeAppendPagedKVCacheKernel variant, and a
    naive substring hit audited that variant's (empty) marker SASS while the
    instrumented kernel carried the CS2R pairs — a false "stale" that failed
    the freshness gate (far-marker pilot r4, 2026-10-09).  Mangled names
    length-prefix each component ("37AppendPagedKVCacheKernelIL..."), so a
    digit prefix or a template-arg opener after the stem is part of the
    encoding, not a longer identifier.
    """
    if stem in kernels:
        return stem, kernels[stem]
    for key, body in kernels.items():
        if kernel_symbol_contains(key, stem):
            return key, body
    return None, None


def clocklo_read_count(body: str) -> int:
    """CS2R instructions whose operand list carries SR_CLOCKLO (operand-level match)."""
    count = 0
    for line in body.splitlines():
        if "CS2R" not in line:
            continue
        if re.search(r"\bCS2R\b[^;]*\bSR_CLOCKLO\b", line):
            count += 1
    return count


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sass", help="cuobjdump --dump-sass output file")
    parser.add_argument("--kernels", required=True,
                        help="comma-separated kernel symbol stems that must pass")
    parser.add_argument("--summary", help="write the per-kernel audit JSON here")
    parser.add_argument("--module", default="", help="label recorded in the summary")
    args = parser.parse_args(argv)

    stems = [s.strip() for s in args.kernels.split(",") if s.strip()]
    if not stems:
        parser.error("--kernels is empty")
    text = open(args.sass, encoding="utf-8").read()
    kernels = kernel_sections(text)

    summary: dict[str, dict] = {}
    failures: list[str] = []
    for stem in stems:
        matched, body = find_kernel_section(kernels, stem)
        clocklo = clocklo_read_count(body) if body else 0
        globaltimerlo = body.count("SR_GLOBALTIMERLO") if body else 0
        entry = {
            "module": args.module,
            "present": body is not None,
            "matched_section": matched,
            "sr_clocklo_cs2r_reads": clocklo,
            "sr_globaltimerlo_reads": globaltimerlo,
            "minimum": 2,
        }
        entry["pass"] = bool(
            entry["present"] and clocklo >= 2 and globaltimerlo >= 2
        )
        summary[stem] = entry
        if not entry["pass"]:
            failures.append(stem)

    payload = {"sass": args.sass, "gate": "per kernel: CS2R SR_CLOCKLO >= 2 and "
                                           "SR_GLOBALTIMERLO >= 2", "kernels": summary}
    formatted = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            handle.write(formatted)
    print(formatted)
    if failures:
        print(f"flashinfer marker SASS audit failed for: {', '.join(failures)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
