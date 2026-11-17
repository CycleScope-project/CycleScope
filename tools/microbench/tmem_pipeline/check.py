import importlib.util
import json
import re
import sys
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "crossings" / "trywait_timeout" / "sass_ctrl.py"
spec = importlib.util.spec_from_file_location("pipeline_sass", source)
parser = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parser
spec.loader.exec_module(parser)
rows = []
for name, kernel in parser.parse(Path(sys.argv[1]).read_text()).items():
    match = re.search(r"k_pipelineILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EE", name)
    if not match:
        continue
    instructions = kernel.insts
    copies, mma, rotate, producers, split, cta_group, shape = map(int, match.groups())
    cp = [i for i, x in enumerate(instructions) if x.root == "UTCCP"]
    mm = [i for i, x in enumerate(instructions) if x.root == "UTCOMMA"]
    commits = [i for i, x in enumerate(instructions) if x.root == "UTCBAR"]
    fail = []
    if len(cp) != copies or len(mm) != mma or len(commits) != 1:
        fail.append("target count")
    forms = ["", "4dp256bit", "128dp128bit", "2x64dp128bit_lw02_lw13", "4x32dp128bit"]
    expected = ".".join(["UTCCP", "T", "S"] + (["2CTA"] if cta_group == 2 else []) +
                        ([forms[shape]] if forms[shape] else []))
    if any(instructions[i].opcode != expected for i in cp):
        fail.append("copy form")
    if any("BLOCK16" not in instructions[i].opcode or
           ("2CTA" in instructions[i].opcode) != (cta_group == 2) for i in mm):
        fail.append("mma form")
    targets = sorted(cp + mm)
    first = targets[0] if targets else (commits[0] if commits else len(instructions))
    clocks = [i for i, x in enumerate(instructions[:first]) if "SR_CLOCKLO" in x.text]
    if not clocks:
        fail.append("no start clock")
    if targets and commits and commits[0] <= targets[-1]:
        fail.append("commit precedes target")
    if targets:
        forbidden = {"BRA", "ELECT", "WARPSYNC", "LDC", "LDCU", "ULDC", "LDG", "STG"}
        spans = [cp, mm] if split else [targets]
        unexpected = [x.text for span in spans if span
                      for x in instructions[span[0]:span[-1] + 1] if x.root in forbidden]
        if unexpected:
            fail.append(unexpected)
    prefix = sum(x.stall for x in instructions[clocks[-1]:first]) if clocks and not split else None
    rows.append({"name": name, "copies": copies, "mma": mma, "rotate": rotate,
                 "producers": producers, "split": split, "cta_group": cta_group,
                 "shape": shape, "prefix_stalls": prefix, "fail": fail,
                 "targets": [{"pc": instructions[i].addr, "sass": instructions[i].text,
                              "stall": instructions[i].stall} for i in targets]})
passed = len(rows) == 180 and all(not r["fail"] for r in rows)
Path(sys.argv[2]).write_text(json.dumps({"passed": passed, "kernels": rows}, indent=2) + "\n")
print(f"kernels={len(rows)} passed={passed}")
raise SystemExit(0 if passed else 1)
