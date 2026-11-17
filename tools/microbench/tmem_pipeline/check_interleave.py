import importlib.util
import json
import re
import sys
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout" / "sass_ctrl.py"
spec = importlib.util.spec_from_file_location("interleave_sass", source)
parser = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parser
spec.loader.exec_module(parser)
rows = []
for name, kernel in parser.parse(Path(sys.argv[1]).read_text()).items():
    match = re.search(r"k_interleaveILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EE", name)
    if not match:
        continue
    copies, mma, group, a_source, n, independent = map(int, match.groups())
    instructions = kernel.insts
    cp = [i for i, x in enumerate(instructions) if x.root == "UTCCP"]
    mm = [i for i, x in enumerate(instructions) if x.root == "UTCOMMA"]
    commits = [i for i, x in enumerate(instructions) if x.root == "UTCBAR"]
    clocks = [i for i, x in enumerate(instructions) if "SR_CLOCKLO" in x.text]
    fail = []
    if len(cp) != copies * mma or len(mm) != mma or len(commits) != 1:
        fail.append("target count")
    expected = "UTCCP.T.S." + ("2CTA." if group == 2 else "") + "4x32dp128bit"
    if any(instructions[i].opcode != expected for i in cp):
        fail.append("copy form")
    if any("BLOCK16" not in instructions[i].opcode or
           ("2CTA" in instructions[i].opcode) != (group == 2) or
           instructions[i].text.lower().count("gdesc[") != 2 - a_source for i in mm):
        fail.append("mma form")
    order = "".join("c" if instructions[i].root == "UTCCP" else "m" for i in sorted(cp + mm))
    if order != ("c" * copies + "m") * mma:
        fail.append("interleave order")
    before = [i for i in clocks if commits and i < commits[0]]
    after = [i for i in clocks if commits and i > commits[0]]
    start = admitted = None
    if len(before) < 2 or not after:
        fail.append("missing clock boundary")
    else:
        start, admitted = before[-2:]
        if any(not start < i < admitted for i in cp + mm):
            fail.append("target outside admission window")
    forbidden = {"BRA", "ELECT", "WARPSYNC", "LDC", "LDCU", "ULDC", "LDG", "STG", "LDL", "STL", "CS2R"}
    targets = sorted(cp + mm)
    if targets and any(x.root in forbidden for x in instructions[targets[0]:targets[-1] + 1]):
        fail.append("wrapper inside target span")
    rows.append({"name": name, "copies": copies, "mma": mma, "cta_group": group,
                 "a_source": a_source, "n": n, "independent": independent,
                 "start_pc": instructions[start].addr if start is not None else None,
                 "admitted_pc": instructions[admitted].addr if admitted is not None else None,
                 "detector_pcs": [instructions[i].addr for i in after], "fail": fail,
                 "prefix_stall": sum(x.stall for x in instructions[start:targets[0]]) if start is not None and targets else None,
                 "span_stall": sum(x.stall for x in instructions[targets[0]:targets[-1] + 1]) if targets else None,
                 "targets": [{"pc": instructions[i].addr, "sass": instructions[i].text,
                              "stall": instructions[i].stall} for i in targets]})
expected_cells = set()
for group, a_source, n in ((1, 0, 128), (1, 1, 128), (1, 0, 256), (2, 0, 256), (2, 0, 128)):
    need = 3 if n == 256 else 2
    expected_cells |= {(c, 8, group, a_source, n, 0) for c in (0, need, 8)}
for group, a_source, n in ((1, 1, 128), (2, 0, 256)):
    need = 3 if n == 256 else 2
    expected_cells |= {(c, 8, group, a_source, n, 1) for c in (need, 8)}
seen = {(r["copies"], r["mma"], r["cta_group"], r["a_source"], r["n"], r["independent"]) for r in rows}
passed = len(rows) == len(expected_cells) and seen == expected_cells and all(not r["fail"] for r in rows)
output = Path(sys.argv[2])
with output.open("x") as f:
    json.dump({"passed": passed, "kernels": rows}, f, indent=2)
    f.write("\n")
print(f"kernels={len(rows)} passed={passed}")
for r in rows:
    if r["fail"]:
        print("FAIL", r["name"], r["fail"])
raise SystemExit(0 if passed else 1)
