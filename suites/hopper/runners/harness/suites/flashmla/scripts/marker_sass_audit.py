#!/usr/bin/env python3
"""Static SASS gate for the FlashMLA delivery marker channel (T0005 rev3-②/rev5).

Called by scripts/build_cluster_sm.sh on the pip-built flash_mla/cuda.so and reusable
standalone (CPU pods, no GPU needed):

    cuobjdump --dump-sass <flash_mla>/cuda*.so > dump.sass
    python3 marker_sass_audit.py dump.sass marker_sass.json

Gate (mirrors the DG static gate
tools/microbench/deepgemm_sm90_runner.py and the CUTLASS
capture-side marker-PC selection — opcode CS2R with an SR_CLOCKLO operand, >=2 PCs): every
instrumented dense-decode kernel must carry >=2 SR_CLOCKLO and >=2 SR_GLOBALTIMERLO reads
(entry + exit of the outermost clock-marker pair).
"""

from __future__ import annotations

import json
import re
import sys


EXPECTED = {
    # kernel base name: (role, minimum SR_CLOCKLO / SR_GLOBALTIMERLO count)
    "flash_fwd_splitkv_mla_kernel": ("formal_denominator", 2),
    "flash_fwd_mla_combine_kernel": ("secondary_evidence", 2),
    "get_mla_metadata_kernel": ("secondary_evidence", 2),
}


def kernel_sections(sass_text: str) -> dict[str, str]:
    """Split a cuobjdump --dump-sass dump into per-kernel text blocks (by `Function :`).

    cuobjdump may print demangled template names (`kernel<...>`) or fully mangled names
    (`_ZN...kernel...`), depending on version; both are kept verbatim as keys."""
    parts = re.split(r"^\s*Function : ", sass_text, flags=re.M)[1:]
    kernels: dict[str, str] = {}
    for part in parts:
        name = part.split("\n", 1)[0].strip()
        base = name.split("<", 1)[0]
        kernels[base] = kernels.get(base, "") + "\n" + part
    return kernels


def find_kernel_section(kernels: dict[str, str], base: str) -> tuple[str | None, str | None]:
    """Exact or substring (mangled-name) lookup; returns (section_key, body)."""
    if base in kernels:
        return base, kernels[base]
    for key, body in kernels.items():
        if base in key:
            return key, body
    return None, None


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: marker_sass_audit.py <dump.sass> <summary.json>", file=sys.stderr)
        return 2
    sass_path, summary_path = argv[1], argv[2]
    text = open(sass_path, encoding="utf-8").read()
    kernels = kernel_sections(text)
    summary: dict[str, dict] = {}
    failures: list[str] = []
    for base, (role, minimum) in EXPECTED.items():
        matched, body = find_kernel_section(kernels, base)
        entry = {
            "role": role,
            "present": body is not None,
            "matched_section": matched,
            "sr_clocklo_count": body.count("SR_CLOCKLO") if body else 0,
            "sr_globaltimerlo_count": body.count("SR_GLOBALTIMERLO") if body else 0,
        }
        entry["pass"] = bool(
            entry["present"]
            and entry["sr_clocklo_count"] >= minimum
            and entry["sr_globaltimerlo_count"] >= minimum
        )
        summary[base] = entry
        if not entry["pass"]:
            failures.append(base)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "sass": sass_path,
                "gate": "per instrumented kernel: SR_CLOCKLO >= 2 and SR_GLOBALTIMERLO >= 2",
                "kernels": summary,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        print(f"flashmla marker SASS audit failed for: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
