#!/usr/bin/env python3
"""Compare a rebuilt tool dir with its previous artifact.

Rules (H3/A2):
  - file sets: report missing/extra files
  - binaries/.so containing GPU code: extract per-function SASS with
    ``cuobjdump -sass`` (fallback ``-elf``), normalize away
    address/offset-only noise, then compare function by function
  - other regular files: sha256
  - verdict per capture-used kernel (mangled names taken from cold4
    functions.jsonl via --used-kernels) must be identical; all other
    differences are reported only.

Usage:
  compare_tool_outputs.py <name> <old-dir> <new-dir>
      [--used-kernels functions.jsonl ...] [--cuobjdump PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


IGNORE = {".build.log", "build.log", "pip.log", "nvcc.log",
          "EFFSM_PATCH.json"}  # path-bearing manifest, checked separately

def _files(root: Path) -> dict[str, Path]:
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            rel = str(p.relative_to(root))
            if p.name in IGNORE or p.name.endswith(".pyc") \
                    or "/logs/" in "/" + rel or "__pycache__" in rel:
                continue
            out[rel] = p
    return out


BIN_EXTS = (".so", ".a", "",)


def _looks_binary(p: Path) -> bool:
    try:
        with p.open("rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


_NORM_RES = [
    (re.compile(r"identifier\s*=.*"), ""),          # embedded src paths
    (re.compile(r"/\*[0-9a-fA-F]+\*/"), ""),          # /*0040*/ addrs
    (re.compile(r"0x[0-9a-fA-F]+\s*;"), "0xADDR;"),
    (re.compile(r"// .*"), ""),                        # trailing comments
    (re.compile(r"\s+"), " "),
]


def _sass_functions(binary: Path, cuobjdump: str) -> dict[str, str]:
    """Per-function normalized SASS text."""
    try:
        out = subprocess.run(
            [cuobjdump, "-sass", str(binary)],
            capture_output=True, text=True, timeout=600)
        text = out.stdout if out.returncode == 0 else ""
    except Exception:
        text = ""
    if not text:
        out = subprocess.run([cuobjdump, "-elf", str(binary)],
                             capture_output=True, text=True, timeout=600)
        text = out.stdout
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        m = re.search(r"Function\s*:\s*(\S+)", line) or \
            re.search(r"\.text\.(\S+)", line)
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, [])
            continue
        if cur is None:
            continue
        s = line
        for rx, rep in _NORM_RES:
            s = rx.sub(rep, s)
        s = s.strip()
        if s:
            funcs[cur].append(s)
    return {k: "\n".join(v) for k, v in funcs.items()}


def _host_functions(p: Path) -> dict[str, list[str]]:
    """Layout-insensitive host compare: split .text into functions and
    normalize each instruction (mnemonic + operands kept; branch/call
    targets -> symbol names, rip-displacements, absolute addrs and
    `# <addr>` comments -> placeholders). PLT/GOT shifts then no longer
    cascade into false diffs (libcuda compat-link path evidence:
    old cuda-12.8/compat vs new /usr/local/nvidia/lib64)."""
    import re as _re
    try:
        out = subprocess.run(
            ["objdump", "-d", "--no-show-raw-insn", str(p)],
            capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return {}
    def _norm_name(n: str) -> str:
        n = _re.sub(r"GLOBAL__N__[0-9a-f]+", "G", n)
        n = _re.sub(r"_cu_[0-9a-f]+_[0-9]+", "_cu_H_", n)
        n = _re.sub(r"_cA[0-9]+_c", "_cA_C", n)
        n = _re.sub(r"__cudaRegisterLinkedBinary_[0-9a-f]+", "CRB", n)
        n = _re.sub(r"libcudart_static_[0-9a-f]+", "L", n)
        n = _re.sub(r"_[0-9a-f]{8,}(?=_|\b)", "H", n)
        n = _re.sub(r"^\.plt.*|^\.cold", ".plt", n)
        return n
    def _norm_insn(line: str) -> str:
        line = _re.sub(r"#\s*[0-9a-f]+\s*<([^>]+)>", r"<\1>", line)
        line = _re.sub(r"#\s*[0-9a-f]+", "# X", line)
        line = _re.sub(r"\+0x[0-9a-f]+>", ">", line)
        line = _re.sub(r"\b[0-9a-f]+ <([^>]+)>", r"<\1>", line)
        line = _re.sub(r"0x[0-9a-f]+", "0xX", line)
        line = _re.sub(r"GLOBAL__N__[0-9a-f]+", "G", line)
        # cudafe linkage hash in local symbol names: _cu_<hash>_<idx>ignoreE
        line = _re.sub(r"_cu_[0-9a-f]+_[0-9]+", "_cu_H_", line)
        # torch check-message template instantiations embed the length of
        # the char[N] __FILE__ argument: cA<len>_c varies with the build
        # path (blackwell flashmla: old/new source prefixes differ by 1)
        line = _re.sub(r"_cA[0-9]+_c", "_cA_C", line)
        line = _re.sub(r"__cudaRegisterLinkedBinary_[0-9a-f]+", "CRB", line)
        return line.strip()
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in out.splitlines():
        m = _re.match(r"^[0-9a-f]+ <([^>]+)>:", line)
        if m:
            cur = _norm_name(m.group(1))
            funcs.setdefault(cur, [])
            continue
        m = _re.match(r"^\s*[0-9a-f]+:\t(.+)$", line)
        if m and cur is not None:
            funcs[cur].append(_norm_insn(m.group(1)))
    return {k: v for k, v in funcs.items() if v}


def _host_differs(a: Path, b: Path) -> dict:
    fa, fb = _host_functions(a), _host_functions(b)
    same = sum(1 for k in fa if k in fb and fa[k] == fb[k])
    diff_all = [k for k in fa if k in fb and fa[k] != fb[k]]
    reg = re.compile(r"__cudaRegister|_GLOBAL__sub_I|__sti__|CRB|@plt")
    regdiff = [k for k in diff_all if reg.search(k)]
    diff = [k for k in diff_all if k not in regdiff]
    return {
        "identical": same,
        "diff": diff,
        "register_diff": len(regdiff),
        "only_old": sorted(set(fa) - set(fb)),
        "only_new": sorted(set(fb) - set(fa)),
        "samples": {
            k: next(
                ((x, y) for x, y in zip(fa[k], fb[k]) if x != y),
                (fa[k][0] if fa[k] else "", fb[k][0] if fb[k] else ""))
            for k in diff[:3]
        },
    }


def _host_text(p: Path) -> str:
    """.text disassembly with addresses stripped, plus .rodata strings.
    Host-code compare: catches changes that -sass misses entirely
    (e.g. host-side gate patches like src/sc CC-10.3 edits)."""
    import re as _re
    try:
        out = subprocess.run(
            ["objdump", "-d", "--no-show-raw-insn", str(p)],
            capture_output=True, text=True, timeout=300).stdout
        text = _re.sub(r"^\s*[0-9a-f]+:", "A:", out, flags=_re.M)
        text = _re.sub(r"0x[0-9a-f]+", "0xX", text)
        # build-hash noise: anonymous-namespace / cudart symbol hashes,
        # objdump address comments
        text = _re.sub(r"GLOBAL__N__[0-9a-f]+", "G", text)
        text = _re.sub(r"libcudart_static_[0-9a-f]+", "L", text)
        text = _re.sub(r"#\s*[0-9a-f]+", "# X", text)
    except Exception:
        text = ""
    try:
        ro = subprocess.run(
            ["objdump", "-s", "-j", ".rodata", str(p)],
            capture_output=True, text=True, timeout=120).stdout
        words = set(_re.findall(r"[ -~]{4,}", ro))
        rodata = "\n".join(sorted(
            w for w in words if "/" not in w))
    except Exception:
        rodata = ""
    return text + "\n--rodata--\n" + rodata


def _sass_differs(a: Path, b: Path, cuo: str) -> list[str]:
    fa, fb = _sass_functions(a, cuo), _sass_functions(b, cuo)
    diff = [k for k in sorted(set(fa) | set(fb))
            if fa.get(k) != fb.get(k)]
    return diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--used-kernels", nargs="*", default=[])
    ap.add_argument("--cuobjdump", default="cuobjdump")
    args = ap.parse_args()
    old, new = Path(args.old), Path(args.new)
    of, nf = _files(old), _files(new)
    only_old = sorted(set(of) - set(nf))
    only_new = sorted(set(nf) - set(of))
    common = sorted(set(of) & set(nf))
    used = set()
    for fj in args.used_kernels:
        for line in Path(fj).read_text().splitlines():
            try:
                used.add(json.loads(line).get("mangled")
                         or json.loads(line).get("name"))
            except Exception:
                pass
    content_diff, sass_same, sass_diff = [], 0, []
    host_same, host_diff = 0, []
    for rel in common:
        a, b = of[rel], nf[rel]
        if _looks_binary(a) and _looks_binary(b):
            dd = _sass_differs(a, b, args.cuobjdump)
            if dd:
                sass_diff.append((rel, dd))
            else:
                sass_same += 1
            hd = _host_differs(a, b)
            if not hd["diff"] and not hd["only_old"] and not hd["only_new"]:
                host_same += 1
            else:
                host_diff.append({rel: hd})
        else:
            if _sha256(a) != _sha256(b):
                content_diff.append(rel)
    used_missing = sorted(u for u in used
                          for rel, dd in sass_diff if u in str(dd))
    old_elves = [rel for rel in common
                 if _looks_binary(of[rel])]
    empty_sass = bool(old_elves) and (sass_same + len(sass_diff) == 0)
    verdict = ("FAIL" if empty_sass else
               "PASS" if not only_old and not content_diff
               and not used_missing else "DIFF")
    report = {
        "name": args.name, "verdict": verdict,
        "only_old": only_old, "only_new": only_new,
        "content_diff": content_diff,
        "host_identical_bins": host_same,
        "host_diff": host_diff,
        "sass_identical_bins": sass_same,
        "sass_diff": {r: d for r, d in sass_diff},
        "used_kernel_diffs": used_missing,
    }
    print(json.dumps(report, indent=1))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
