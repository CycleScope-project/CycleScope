#!/usr/bin/env python3
"""l2_merge protocol v2 sass gate (tools/microbench/l2_merge/protocol-v2.json):
the ld streamer kernel scan_kernel<false> (mangled _Z11scan_kernelILb0E...)
in the cuobjdump -sass dump of ldg_scan must contain exactly UNROLL = 8
LDG.E.128 instructions.

usage: sass_check.py <ldg_scan.sass>
First output line is `PASS ...` or `FAIL ...`; exit 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EXPECTED_LDG128 = 8  # UNROLL
FUNC_RE = re.compile(r"^\s*Function : (\S+)")
INST_RE = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(\S+)")


def main() -> int:
    if len(sys.argv) != 2:
        print("FAIL usage: sass_check.py <ldg_scan.sass>")
        return 1
    text = Path(sys.argv[1]).read_text(errors="replace")
    kernels: dict[str, int] = {}
    name = None
    for line in text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            name = m.group(1)
            continue
        if name is None:
            continue
        ins = INST_RE.match(line)
        if ins and ins.group(1).startswith("LDG.E.128"):
            kernels[name] = kernels.get(name, 0) + 1
    if not kernels and "Function :" not in text:
        print("FAIL no functions listed in sass dump")
        return 1
    match = {n: c for n, c in kernels.items() if "scan_kernel" in n and "Lb0" in n}
    # also count matching kernels that had zero LDG.E.128 (listed but no entry)
    zero = [n for n in re.findall(r"Function : (\S*scan_kernel\S*Lb0\S*)", text)
            if n not in kernels]
    match.update({n: 0 for n in zero})
    lines = []
    for n, c in sorted(match.items()):
        lines.append(f"{n}: LDG.E.128 x{c}")
    for n, c in sorted(kernels.items()):
        if n not in match:
            lines.append(f"(other) {n}: LDG.E.128 x{c}")
    if len(match) != 1:
        print(f"FAIL expected exactly one scan_kernel<false> kernel, found {len(match)}")
        print("\n".join(lines))
        return 1
    (kname, count), = match.items()
    if count != EXPECTED_LDG128:
        print(f"FAIL {kname}: LDG.E.128 x{count} (expected x{EXPECTED_LDG128})")
        print("\n".join(lines))
        return 1
    print(f"PASS {kname}: LDG.E.128 x{count}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
