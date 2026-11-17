import importlib.util
import json
import re
import sys
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout" / "sass_ctrl.py"
spec = importlib.util.spec_from_file_location("admission_sass", source)
parser = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parser
spec.loader.exec_module(parser)
rows = []
for name, kernel in parser.parse(Path(sys.argv[1]).read_text()).items():
    match = re.search(r"k_admissionILi(\d+)ELi(\d+)ELi(\d+)E(?:Li(\d+)E)?E", name)
    if not match:
        continue
    copies, mma, group = map(int, match.groups()[:3])
    legacy = match.group(4) is None
    a_source = 0 if legacy else int(match.group(4))
    instructions = kernel.insts
    cp = [i for i, x in enumerate(instructions) if x.root == "UTCCP"]
    mm = [i for i, x in enumerate(instructions) if x.root == "UTCOMMA"]
    commits = [i for i, x in enumerate(instructions) if x.root == "UTCBAR"]
    clocks = [i for i, x in enumerate(instructions) if "SR_CLOCKLO" in x.text]
    fail = []
    if len(cp) != copies or len(mm) != mma or len(commits) != 1:
        fail.append("target count")
    expected = "UTCCP.T.S." + ("2CTA." if group == 2 else "") + "4x32dp128bit"
    if any(instructions[i].opcode != expected for i in cp):
        fail.append("copy form")
    if any("BLOCK16" not in instructions[i].opcode or
           ("2CTA" in instructions[i].opcode) != (group == 2) or
           instructions[i].text.lower().count("gdesc[") != 2 - a_source for i in mm):
        fail.append("mma form")
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
    for span in (cp, mm):
        if span and any(x.root in forbidden for x in instructions[span[0]:span[-1] + 1]):
            fail.append("wrapper inside target span")
    rows.append({"name": name, "copies": copies, "mma": mma, "cta_group": group,
                 "a_source": a_source, "legacy": legacy,
                 "start_pc": instructions[start].addr if start is not None else None,
                 "admitted_pc": instructions[admitted].addr if admitted is not None else None,
                 "detector_pcs": [instructions[i].addr for i in after], "fail": fail,
                 "targets": [{"pc": instructions[i].addr, "sass": instructions[i].text,
                              "stall": instructions[i].stall} for i in cp + mm]})
expected_cells = {(u, m, cg, 0) for u in (0, 32) for m in (0, 4) for cg in (1, 2)}
expected_cells |= {(8, 0, cg, 0) for cg in (1, 2)}
if rows and not any(r["legacy"] for r in rows):
    expected_cells |= {(u, m, cg, 1) for u, m in ((0, 4), (32, 0), (32, 4)) for cg in (1, 2)}
passed = len(rows) == len(expected_cells) and {(r["copies"], r["mma"], r["cta_group"], r["a_source"]) for r in rows} == expected_cells
passed = passed and (all(r["legacy"] for r in rows) or all(not r["legacy"] for r in rows))
passed = passed and all(not r["fail"] for r in rows)
output = Path(sys.argv[2])
with output.open("x") as f:
    json.dump({"passed": passed, "kernels": rows}, f, indent=2)
    f.write("\n")
print(f"kernels={len(rows)} passed={passed}")
raise SystemExit(0 if passed else 1)
