import importlib.util
import json
import re
import sys
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "lsu_accept" / "sass_check_store_queue.py"
spec = importlib.util.spec_from_file_location("store_queue_check", source)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.WIDTH[4] = "STG.E.U16"
kernels = checker.parse(Path(sys.argv[1]).read_text())
results = []
for name, kernel in sorted(kernels.items()):
    match = re.search(r"k_cellILi(\d+)E", name)
    if match:
        results.append(checker.check_kernel(kernel, int(match.group(1))))
passed = len(results) == 5 and all(r["ok"] for r in results)
Path(sys.argv[2]).write_text(json.dumps({"passed": passed, "kernels": results}, indent=2) + "\n")
print(f"kernels={len(results)} passed={passed}")
raise SystemExit(0 if passed else 1)
