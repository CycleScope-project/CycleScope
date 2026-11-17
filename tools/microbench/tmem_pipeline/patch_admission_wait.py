import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout" / "sass_ctrl.py"
spec = importlib.util.spec_from_file_location("admission_wait_sass", source)
sass = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sass
spec.loader.exec_module(sass)
WAIT_MASK = 0x3F << 52


def cuda_texts(data):
    result = {}
    offset = 0
    while True:
        offset = data.find(b"\x7fELF", offset)
        if offset < 0:
            return result
        base, offset = offset, offset + 4
        if data[base + 4:base + 6] != b"\x02\x01":
            continue
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", data, base)
        if header[2] != 190:
            continue
        assert header[8] == 64 and header[11] == 64
        assert 0 < header[13] < header[12]
        sections = [struct.unpack_from("<IIQQQQIIQQ", data, base + header[6] + i * 64)
                    for i in range(header[12])]
        names = sections[header[13]]
        strings = data[base + names[4]:base + names[4] + names[5]]
        assert len(strings) == names[5]
        for section in sections:
            name = strings[section[0]:].split(b"\0", 1)[0].decode()
            if not name.startswith(".text."):
                continue
            name = name.removeprefix(".text.")
            begin, size = base + section[4], section[5]
            assert name not in result and begin + size <= len(data)
            result[name] = (begin, size)


def patch(binary, native_sass, check_file, output, report):
    assert binary != output and not output.exists() and not report.exists()
    data = binary.read_bytes()
    texts = cuda_texts(data)
    kernels = sass.parse(native_sass.read_text())
    check = json.loads(check_file.read_text())
    assert check["passed"] and len(check["kernels"]) == len(kernels) == 10
    assert set(kernels) <= set(texts)
    for name, kernel in kernels.items():
        begin, size = texts[name]
        for inst in kernel.insts:
            assert inst.addr % 16 == 0 and inst.addr + 16 <= size
            assert struct.unpack_from("<QQ", data, begin + inst.addr) == (inst.low, inst.high)
    selected = [x for x in check["kernels"] if x["copies"] in (8, 32) and x["mma"] == 0]
    assert {(x["copies"], x["cta_group"]) for x in selected} == {(u, cg) for u in (8, 32) for cg in (1, 2)}
    assert len(selected) == 4
    patched = bytearray(data)
    changes = []
    for cell in selected:
        insts = kernels[cell["name"]].insts
        index = next(i for i, x in enumerate(insts) if x.addr == cell["admitted_pc"])
        clock, follower = insts[index:index + 2]
        assert clock.root == "CS2R" and "SR_CLOCKLO" in clock.text and clock.wait == 0
        assert follower.root == "R2UR" and follower.wait == 0x3F
        offset = texts[cell["name"]][0] + clock.addr
        struct.pack_into("<Q", patched, offset + 8, clock.high | WAIT_MASK)
        changes.append({"name": cell["name"], "pc": clock.addr, "file_offset": offset,
                        "low": clock.low, "before_high": clock.high,
                        "after_high": clock.high | WAIT_MASK})
    changed = {i for i, (a, b) in enumerate(zip(data, patched)) if a != b}
    assert changed == {c["file_offset"] + byte for c in changes for byte in (14, 15)}
    with output.open("xb") as f:
        f.write(patched)
    output.chmod(binary.stat().st_mode & 0o777)
    with report.open("x") as f:
        json.dump({"binary": str(binary), "native_sass": str(native_sass),
                   "check": str(check_file), "output": str(output), "changes": changes}, f, indent=2)
        f.write("\n")
    print(f"patched {len(changes)} admission clocks; changed bytes={len(changed)}")


def verify(report_file, patched_sass):
    report = json.loads(report_file.read_text())
    before = sass.parse(Path(report["native_sass"]).read_text())
    after = sass.parse(patched_sass.read_text())
    assert set(before) == set(after)
    changes = {(x["name"], x["pc"]): x for x in report["changes"]}
    seen = set()
    for name, kernel in before.items():
        other = after[name].insts
        assert len(kernel.insts) == len(other)
        for a, b in zip(kernel.insts, other):
            assert (a.addr, a.low) == (b.addr, b.low)
            key = (name, a.addr)
            if key not in changes:
                assert a.high == b.high
                continue
            change = changes[key]
            assert (a.high, b.high) == (change["before_high"], change["after_high"])
            assert b.high == a.high | WAIT_MASK and a.wait == 0 and b.wait == 0x3F
            seen.add(key)
    assert seen == set(changes) and len(seen) == 4
    print("verified all ten kernels; only four clock wait masks differ")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("patch")
    for name in ("binary", "native_sass", "check", "output", "report"):
        p.add_argument(name, type=Path)
    p = sub.add_parser("verify")
    p.add_argument("report", type=Path)
    p.add_argument("patched_sass", type=Path)
    args = parser.parse_args()
    if args.command == "patch":
        patch(args.binary, args.native_sass, args.check, args.output, args.report)
    else:
        verify(args.report, args.patched_sass)


if __name__ == "__main__":
    main()
