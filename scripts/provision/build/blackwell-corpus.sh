#!/usr/bin/env bash
# blackwell-corpus: rebuild the 24 cutlass example binaries actually
# used by the 1000 bw cases via the runner's own compile_one.
# evidence: ~/tmp/capture-repro/h3/corpus_used.tsv (23 bin/ + 1
#   bin-c450v2/, all third_party/cutlass/examples, no defs); old
#   corpus_sc/cutlass/*/compile.txt = suites/blackwell/runners/cutlass/
#   run.py compile_one nvcc lines (-std=c++17 -O3
#   --expt-relaxed-constexpr -gencode 103a -lcuda + 4 -I).
# input: blackwell-cutlass-source (runners/third_party/cutlass symlink).
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/blackwell-corpus
R=$CYCLESCOPE_ROOT/suites/blackwell/runners
rm -rf "$T"
rm -rf "$T"
mkdir -p "$T/cutlass" "$T/logs"
# point the runner's third_party link at the provisioned source
mkdir -p "$R/third_party" "$T/work"
# runtime note (verified): all 1000 cpp cases run mode=full ->
# case_cpp prebuilt branch reads BW_CUTLASS_BIN_ROOT (env.py:36
# pins it to tools/blackwell-corpus/cutlass); compile_one fallback
# is never exercised, so the source entry stays unpatched.
# patch the 4 host-side example files (untracked local edits in
# blackwell-prototype/third_party/cutlass @ f614dc4, mtime 2026-09-17 —
# see suites/blackwell/runners/patches/cutlass/*.patch and
# capture_tools.json note); build in a copy, source stays read-only
cp -a "$CYCLESCOPE_ARTIFACT_ROOT/tools/blackwell-cutlass-source" "$T/work/cutlass"
for p in "$CYCLESCOPE_ROOT"/suites/blackwell/runners/patches/cutlass/*.patch; do
  patch -s -p1 -d "$T/work/cutlass" -i "$p"
done
ln -sfn "$T/work/cutlass" "$R/third_party/cutlass"
export CYCLESCOPE_RUNNER_ROOT="$R"
python3 - "$T" <<'PYEOF'
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
T = Path(sys.argv[1])
R = Path(os.environ["CYCLESCOPE_RUNNER_ROOT"])
sys.path.insert(0, str(R / "cutlass"))
import run as runner  # noqa: E402

tasks = {}
for suite in ("full_gpu",):
    base = R.parents[0] / suite
    used = {l.split("\t")[0] for l in
            (base / "launches.tsv").read_text().splitlines()[1:]}
    for c in json.loads((base / "cases_cutlass.json")
                        .read_text())["cases"]:
        if c["id"] not in used or not c.get("src"):
            continue
        key = (c.get("bin_dir") or "bin", c["bin"], c["src"],
               tuple(c.get("defs") or ()))
        tasks[key] = c

def build(item):
    (bin_dir, name, src, defs), c = item
    # _sc bins use the gate-patched sources (src/sc/<basename>, host-side
    # "SC: allow CC 10.3" patch; SASS identical, host differs — see
    # blackwell-prototype/delivery_tests/cutlass/src/sc, 6 files were git-untracked)
    if name.endswith("_sc"):
        sc = R / "cutlass" / "src" / "sc" / Path(src).name
        if sc.exists():
            src = sc
    # 77_blackwell_mla.cu requires -DFP8 or -DFP16; old bin contains
    # half_t SASS symbols -> FP16 (evidence: strings on old corpus bin)
    if not defs and name == "77_blackwell_mla":
        defs = ("FP16",)
    return runner.compile_one(
        name, R / src, reuse=False, defs=defs,
        bin_root=T / "cutlass" / bin_dir)

fails = []
with ThreadPoolExecutor(max_workers=8) as ex:
    for bin_dir, name, (rc, out, dst) in ex.map(
            lambda i: (i[0][0], i[0][1], build(i)), tasks.items()):
        (T / "logs" / f"{bin_dir}-{name}.log").write_text(out)
        if rc:
            fails.append((bin_dir, name, rc))
print("built", len(tasks), "fails", fails)
sys.exit(1 if fails else 0)
PYEOF
# the runner link must outlive the build copy: CuTe-DSL cases resolve
# third_party/cutlass/examples/python/... at capture time (a dangling
# link makes them report "missing <script>" with rc 0 and no trace)
ln -sfn "$CYCLESCOPE_ARTIFACT_ROOT/tools/blackwell-cutlass-source" "$R/third_party/cutlass"
rm -rf "$T/work"
