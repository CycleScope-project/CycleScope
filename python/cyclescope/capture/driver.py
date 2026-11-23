"""``cyclescope capture`` — resolve a suite case to a concrete traced run.

One public entry point :func:`capture_case`; the CLI wires
``cyclescope capture --suite <arch>/<set> --case <id> --out <dir>``.  The
driver does: case resolve (argv/env/cwd), an optional ncu launch census,
launch selection, the traced run, the placement run, ``run.json`` and
``trace_check``.

Hopper cases come from ``suites/hopper/<set>/cases.tsv`` (argv_json /
env_json / kernel_regex / occurrence columns).  Blackwell cases resolve
through ``suites/blackwell/runners`` adapters (``CYCLESCOPE_RUNNER_ROOT`` +
``runners/env.py`` defaults).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .ncu_tools import (CACHE_CONTROL, CLOCK_CONTROL, TIMEOUT_RC,
                        ncu_binary, run_group)

REPO = Path(__file__).resolve().parents[3]
from .. import site  # noqa: E402

NCU_LAUNCH_METRICS = (
    "launch__shared_mem_config_size", "launch__grid_size",
    "launch__block_size", "launch__registers_per_thread",
    "launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z",
    "launch__block_dim_x", "launch__block_dim_y", "launch__block_dim_z",
    "launch__cluster_dim_x", "launch__cluster_dim_y",
    "launch__cluster_dim_z", "sm__cycles_active.max",
    "sm__cycles_active.sum", "sm__ctas_launched.sum",
    "sm__cycles_elapsed.avg.per_second",
)


@dataclass
class Resolved:
    argv: list
    env: dict
    cwd: Path | None
    arch: str                      # hopper | blackwell
    kernel_regex: str | None = None
    occurrence: int | None = None
    calls: int = 0
    main_regex: re.Pattern | None = None
    env_profile: str = ""
    family: str = ""
    # resolved from an argv row in suites/<arch>/<sset>/cases.tsv
    # (kernel_regex@occurrence launch selection); False for the
    # blackwell adapter path (main-last selection)
    argv_row: bool = False
    spec: dict = field(default_factory=dict)
    # fixed launch table row (suites/<arch>/<set>/launches.tsv): the
    # exact (mangled, occurrence) pair to trace, plus the raw row for
    # the post-trace geometry check
    launch_key: tuple | None = None
    launch_row: dict | None = None


def _expand(value: str) -> str:
    """Expand $CYCLESCOPE_ROOT (this checkout) and $CYCLESCOPE_ARTIFACT_ROOT
    inside an argv token / cwd / env value.  ':'-joined values
    (PYTHONPATH) are expanded segment by segment."""
    if not value:
        return value
    aroot = str(site.artifact_root())

    def one(v: str) -> str:
        return (v.replace("$CYCLESCOPE_ROOT", str(REPO))
                 .replace("$CYCLESCOPE_ARTIFACT_ROOT", aroot))

    if ":" in value:
        return ":".join(one(v) for v in value.split(":"))
    return one(value)


_HOPPER_UPSTREAM_LINKS = {
    # harness/suites/<fam>/upstream -> $CYCLESCOPE_ARTIFACT_ROOT/tools/<name>
    # (third-party source snapshots; see configs/capture_tools.json)
    "cutlass": "hopper-cutlass-source",
    "deepgemm": "hopper-deepgemm-source",
    "flashattention3": "hopper-flashattention-source",
    "flashinfer": "hopper-flashinfer-source",
    "flashmla": "hopper-flashmla-source",
}


def _harness_root() -> Path:
    return REPO / "suites" / "hopper" / "runners" / "harness"


def _provision_harness_upstreams() -> None:
    """Link suites/<fam>/upstream inside the in-repo hopper harness
    mirror to the third-party source snapshot under
    $CYCLESCOPE_ARTIFACT_ROOT/tools/ (same pattern as the blackwell
    runners' .deps/third_party links)."""
    aroot = site.artifact_root()
    suites = _harness_root() / "suites"
    for fam, name in _HOPPER_UPSTREAM_LINKS.items():
        link = suites / fam / "upstream"
        target = aroot / "tools" / name
        if link.exists() or not target.is_dir():
            continue
        try:
            link.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            pass  # a concurrent shard created it first


def _argv_row(arch: str, sset: str, case_id: str) -> dict:
    tsv = REPO / "suites" / arch / sset / "cases.tsv"
    rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
    for r in rows:
        if r["case_id"] == case_id:
            return r
    raise KeyError(f"{case_id} not in {tsv}")


def _launch_table_row(arch: str, sset: str, case_id: str):
    """(launch_key, row) from the suite's fixed launch table; (None,
    None) when the table or the case's row is absent."""
    from .launch_selection import load_launch_table
    row = load_launch_table(
        REPO / "suites" / arch / sset / "launches.tsv").get(case_id)
    if row is None:
        return None, None
    return (row["kernel"], int(row["occurrence"])), row



def _ensure_link(link: Path, target: Path) -> None:
    """Point a runner-tree symlink at ``target`` (a dangling or stale link
    — e.g. left behind by a build script's deleted work copy — is
    replaced; a real directory is left alone)."""
    if not target.is_dir():
        return
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        return
    try:
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    except FileExistsError:
        pass  # a concurrent shard created it first

def resolve_case(suite: str, case_id: str, workdir: Path) -> Resolved:
    site.export_environment()
    arch, sset = suite.split("/", 1)
    argv_tsv = REPO / "suites" / arch / sset / "cases.tsv"
    if arch == "hopper" or argv_tsv.is_file():
        row = _argv_row(arch, sset, case_id)
        argv = json.loads(row["argv_json"].replace('""', '"'))
        argv = [_expand(a) for a in argv]
        env = json.loads(row["env_json"].replace('""', '"')) \
            if row["env_json"] not in ("{}", "-") else {}
        env = {k: _expand(str(v)) for k, v in env.items()}
        # hopper cases resolve their interpreter modules from the
        # in-repo harness mirror (suites/hopper/runners/harness):
        # legacy `tools.*` shims live at the mirror root, while the
        # harness cwd contains a shadow `tools/` package that wins for
        # `python3 -m` (cwd precedes PYTHONPATH).  Rule: mirror root
        # goes FIRST on PYTHONPATH, and `python -m tools.*` argv is
        # wrapped in a -c shim that inserts the mirror at sys.path[0]
        # and aliases modules that moved under tools/microbench/deepgemm/.
        hop_clone = _harness_root() if arch == "hopper" else None
        if hop_clone is not None and hop_clone.is_dir():
            _provision_harness_upstreams()
            env["PYTHONPATH"] = (str(hop_clone) + ":"
                                 + env.get("PYTHONPATH", "")).rstrip(":")
            fa3_site = (site.artifact_root()
                        / "tools" / "hopper-fa3-build" / "site-packages")
            if fa3_site.is_dir():
                env["PYTHONPATH"] = (str(fa3_site) + ":"
                                     + env["PYTHONPATH"])
            if (len(argv) > 2 and argv[1] == "-m"
                    and argv[2].startswith("tools.")):
                mod, rest = argv[2], argv[3:]
                shim = (
                    "import sys, runpy, importlib\n"
                    f"sys.path.insert(0, {str(hop_clone)!r})\n"
                    "sys.argv = sys.argv[1:]\n"
                    "try:\n"
                    "    sys.modules.setdefault("
                    " 'tools.microbench.deepgemm_sm90_suite',"
                    " importlib.import_module("
                    "  'tools.microbench.deepgemm.deepgemm_sm90_suite'))\n"
                    "except ModuleNotFoundError:\n"
                    "    pass\n"
                    f"runpy.run_module({mod!r}, run_name='__main__')")
                argv = [argv[0], "-c", shim, "-", *rest]
            elif (len(argv) > 1 and argv[1].endswith(".py")
                    and Path(argv[1]).is_absolute()):
                # script argv: the script's own dir wins sys.path[0] —
                # harness dirs contain a shadow `tools` package that
                # hides the clone's `tools.microbench.*`; insert the
                # clone ahead of it via a -c run_path wrapper
                script, rest = argv[1], argv[2:]
                shim = (
                    "import sys, runpy\n"
                    # run_path does not add the script's directory to
                    # sys.path the way `python3 script.py` does; keep
                    # it right behind the clone so sibling imports
                    # (flashmla marker_timing) still resolve
                    f"sys.path.insert(0, {str(Path(script).parent)!r})\n"
                    f"sys.path.insert(0, {str(hop_clone)!r})\n"
                    # modules the harnesses import as
                    # tools.microbench.<name> live one level deeper in
                    # the clone (tools/microbench/deepgemm/<name>)
                    "try:\n"
                    "    import tools.microbench as _tm\n"
                    "    _tm.__path__.append("
                    f"{str(hop_clone / 'tools' / 'microbench' / 'deepgemm')!r})\n"
                    "except ImportError:\n"
                    "    pass\n"
                    f"sys.argv = [{script!r}] + sys.argv[2:]\n"
                    f"runpy.run_path({script!r}, run_name='__main__')")
                argv = [argv[0], "-c", shim, "-", *rest]
            # the mirror root is also appended (the old asset_paths
            # pythonpath_append rule): it supplies `tools.*` modules such
            # as tools.microbench.deepgemm_sm90_runner behind case dirs
            env["PYTHONPATH"] = (env.get("PYTHONPATH", "")
                                 + ":" + str(hop_clone)).lstrip(":")
        cwd = None if row["cwd"] == "-" else Path(_expand(row["cwd"]))
        occ = int(row["occurrence"]) if row["occurrence"].isdigit() else None
        lkey, lrow = _launch_table_row(arch, sset, case_id)
        return Resolved(argv=argv, env=env, cwd=cwd, arch=arch,
                        kernel_regex=row["kernel_regex"] or None,
                        occurrence=occ,
                        env_profile=row.get("env_profile") or "",
                        argv_row=True,
                        spec=row, launch_key=lkey, launch_row=lrow)
    # blackwell: adapters under suites/blackwell/runners
    rroot = REPO / "suites" / "blackwell" / "runners"
    os.environ.setdefault("CYCLESCOPE_RUNNER_ROOT", str(rroot))
    sys.path.insert(0, str(rroot / "common"))
    sys.path.insert(0, str(rroot))
    import env as runner_env  # noqa: E402  (runners/env.py)
    for k, v in runner_env.defaults().items():
        os.environ.setdefault(k, v)
    # the case spec (calls / main_kernel / args) lives in the suite set's
    # own cases_<family>.json; the runner default points at single_cluster
    os.environ["BW_CASES_DIR"] = str(rroot.parent / sset)
    # .deps symlink: runners reference BW_FA4_PREFIX=$ROOT/.deps/fa4
    # (etc.); the old gpu_capture.sh linked it to the deps mirror, do
    # the same so runners don't re-clone them.
    _ensure_link(rroot / ".deps", Path(os.environ["BW_DEPS_DIR"]))
    # CuTe-DSL scripts resolve as $CYCLESCOPE_RUNNER_ROOT/third_party/cutlass
    # (runners/cutlass/run.py); link the artifact-root cutlass checkout
    # the same way the old gpu_capture.sh did.
    _ensure_link(rroot / "third_party" / "cutlass",
                 Path(os.environ["BW_CUTLASS_DIR"]))
    import capture_adapters as ca  # noqa: E402
    family = None
    cases_txt = rroot.parent / sset / "cases.txt"
    for line in cases_txt.read_text().splitlines():
        if line.strip() and line.split()[-1] == case_id:
            family = line.split()[0]
            break
    if family is None:
        raise KeyError(f"{case_id} not in {cases_txt}")
    case = ca.load_case(case_id)
    res = ca.get_adapter(family).resolve(case, Path(workdir))
    calls = int((case.spec or {}).get("calls") or 0)
    lkey, lrow = _launch_table_row(arch, sset, case_id)
    return Resolved(argv=list(res.argv), env=dict(res.env or {}),
                    cwd=Path(res.workdir) if getattr(res, "workdir", None)
                    else None,
                    arch=arch, calls=calls,
                    main_regex=getattr(res, "main_regex", None),
                    family=str(getattr(case, "suite", "") or ""),
                    spec=dict(case.spec or {}),
                    launch_key=lkey, launch_row=lrow)


def case_command(resolved: Resolved, target: Path,
                 env: dict) -> tuple[list, dict]:
    """argv/env to run the case: ``<CAPTURE>`` -> target; env_profile
    wrapping (far: bash -c 'source environment_far.sh && exec "$@"',
    CYCLESCOPE_RUNNER_ROOT)."""
    # <CAPTURE> is the historical placeholder the case argv uses for
    # per-run scratch/outputs (old drivers did the same replacement).
    argv = [a.replace("<CAPTURE>", str(target)) for a in resolved.argv]
    run_env = {k: (v.replace("<CAPTURE>", str(target))
                   if isinstance(v, str) else v)
               for k, v in env.items()}
    run_env["CYCLESCOPE_RUNNER_ROOT"] = str(
        REPO / "suites" / resolved.arch / "runners")
    if resolved.env_profile == "far":
        # the far profile lives in a sourced shell env (pip pin, wheel
        # paths); wrap the argv under bash so it applies.
        far = (REPO / "suites" / resolved.arch / "runners" /
               "environment_far.sh")
        argv = ["bash", "-c",
                f'source "{far}" && exec "$@"', "_", *argv]
    return argv, run_env


def prepare_case(resolved: Resolved, workdir: Path, log_path: Path,
                 autotune_from: Path | None = None,
                 timeout_s: int = 1200) -> dict:
    """Unprofiled per-case prepass, run once before ncu/tracer runs.

    Only triton cases whose spec mode is "full" (the official/run.py
    runner) need it: that runner refuses to run without
    ``<BW_TRITON_WORKDIR>/triton_autotune.json``, written by an untraced
    ``--prepass`` run that warms the autotuner and pins the winning
    config.  With ``autotune_from`` the pinned file is copied instead so
    reference and traced runs share one config.  The triton_kernels
    import root is linked under ``workdir`` and injected into
    ``resolved.env`` so every later run of this case (census, trace,
    placement) sees the same PYTHONPATH.  Returns a status dict;
    ``rc != 0`` means the case must not run.
    """
    if resolved.family != "triton" or resolved.spec.get("mode") != "full":
        return {"status": "not_needed", "rc": 0}
    rroot = REPO / "suites" / "blackwell" / "runners"
    source = Path(str(resolved.env.get("TRITON_SOURCE_ROOT")
                      or os.environ.get("TRITON_SOURCE_ROOT", "")))
    kernels_src = source / "python" / "triton_kernels" / "triton_kernels"
    if not kernels_src.is_dir():
        return {"status": "missing_source", "rc": 1,
                "source": str(kernels_src)}
    import_root = workdir / "triton-kernels-pythonpath"
    import_root.mkdir(parents=True, exist_ok=True)
    link = import_root / "triton_kernels"
    if not link.exists():
        try:
            link.symlink_to(kernels_src, target_is_directory=True)
        except FileExistsError:
            pass
    resolved.env["TRITON_KERNELS_IMPORT_ROOT"] = str(import_root)
    resolved.env["PYTHONPATH"] = (str(import_root) + ":"
                                  + resolved.env.get("PYTHONPATH", "")
                                  ).rstrip(":")
    autotune = workdir / "triton_autotune.json"
    if autotune_from is not None:
        autotune.write_text(Path(autotune_from).read_text())
        return {"status": "ok", "rc": 0, "autotune": "copied"}
    if autotune.is_file():
        return {"status": "ok", "rc": 0, "autotune": "existing"}
    env = dict(os.environ)
    env.update({k: str(v) for k, v in resolved.env.items()})
    env["BW_TRITON_WORKDIR"] = str(workdir)
    env["TRITON_CACHE_DIR"] = str(workdir / "triton_cache")
    env["TRITON_KERNELS_IMPORT_ROOT"] = str(import_root)
    env["PYTHONPATH"] = (str(import_root) + ":"
                         + env.get("PYTHONPATH", "")).rstrip(":")
    argv = [a.replace("<CAPTURE>", str(workdir)) for a in resolved.argv]
    with log_path.open("w") as log:
        log.write("command " + json.dumps(argv + ["--prepass"]) + "\n")
        rc = run_group(argv + ["--prepass"], env=env, stdout=log,
                       stderr=subprocess.STDOUT, cwd=resolved.cwd,
                       timeout_s=timeout_s)
    if rc == TIMEOUT_RC:
        return {"status": "timeout", "rc": 1}
    return {"status": "ok" if rc == 0 else "prepass_failed", "rc": rc}


def _ncu_launch_pass(out: Path, resolved: Resolved, env: dict,
                     timeout_s: int) -> dict:
    """No-tracer ncu census -> ncu_launches.jsonl (returns status dict)."""
    try:
        ncu = ncu_binary()
    except FileNotFoundError:
        return {"status": "ncu_missing"}
    argv, run_env = case_command(resolved, out, env)
    csv_path, log_path = out / "ncu.csv", out / "ncu.log"
    cmd = [ncu, "--csv", "--page", "raw", "--print-units", "base",
           "--print-kernel-base", "mangled", "--clock-control",
           CLOCK_CONTROL, "--cache-control", CACHE_CONTROL,
           "--target-processes", "all",
           "--force-overwrite", "--metrics",
           ",".join(NCU_LAUNCH_METRICS), "--log-file", str(csv_path),
           "--", *argv]
    with log_path.open("w") as log:
        log.write("command " + json.dumps(cmd) + "\n")
        rc = run_group(cmd, env=run_env, stdout=log,
                       stderr=subprocess.STDOUT, cwd=resolved.cwd,
                       timeout_s=min(timeout_s, 900))
    if rc == TIMEOUT_RC:
        return {"status": "ncu_timeout"}
    if not csv_path.is_file():
        return {"status": "ncu_failed", "rc": rc}
    text = csv_path.read_text(errors="replace")
    start = text.find('"ID"')
    if start < 0:
        start = text.find("ID,")
    rows = []
    if start >= 0:
        parsed = list(csv.reader(text[start:].splitlines()))
        if parsed:
            hdr = parsed[0]
            for row in parsed[1:]:
                if len(row) == len(hdr) and str(row[0]).isdigit():
                    rows.append(dict(zip(hdr, row)))
    def _int(r, name):
        v = (r.get(name) or "").strip().replace(",", "")
        try:
            return int(v)
        except ValueError:
            try:
                return int(float(v))
            except ValueError:
                return None
    entries = []
    for i, row in enumerate(sorted(rows, key=lambda r: int(r["ID"]))):
        hz = _int(row, "sm__cycles_elapsed.avg.per_second")
        entries.append({
            "ordinal": i, "kernel_id": int(row["ID"]),
            "function": (row.get("Kernel Name") or "").strip(),
            "grid": [_int(row, "launch__grid_dim_x"),
                     _int(row, "launch__grid_dim_y"),
                     _int(row, "launch__grid_dim_z")],
            "block": [_int(row, "launch__block_dim_x"),
                      _int(row, "launch__block_dim_y"),
                      _int(row, "launch__block_dim_z")],
            "grid_size": _int(row, "launch__grid_size"),
            "block_size": _int(row, "launch__block_size"),
            "registers_per_thread":
                _int(row, "launch__registers_per_thread"),
            "cluster": [_int(row, "launch__cluster_dim_x"),
                        _int(row, "launch__cluster_dim_y"),
                        _int(row, "launch__cluster_dim_z")],
            "cycles_active_max": _int(row, "sm__cycles_active.max"),
            "ctas_launched": _int(row, "sm__ctas_launched.sum"),
            "sm_clock_khz": int(hz / 1000) if hz else None,
        })
    with (out / "ncu_launches.jsonl").open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return {"status": "ok", "rc": rc, "launches": len(entries)}


def _select_launch(resolved: Resolved, out: Path) -> str:
    """Return the CYCLESCOPE_CAPTURE_LAUNCHES value."""
    if resolved.argv_row:
        # verify the occurrence exists in the ncu census
        rows = [json.loads(l) for l in
                (out / "ncu_launches.jsonl").read_text().splitlines()
                if l.strip()] if (out / "ncu_launches.jsonl").is_file() else []
        if resolved.occurrence is not None and rows:
            matches = sum(
                1 for r in rows
                if resolved.kernel_regex
                and re.search(resolved.kernel_regex,
                              r.get("function") or ""))
            if matches < resolved.occurrence:
                raise KeyError(
                    f"occurrence {resolved.occurrence} of "
                    f"{resolved.kernel_regex} beyond ncu census "
                    f"({matches} matches)")
        sel = (f"{resolved.kernel_regex}@{resolved.occurrence}"
               if resolved.occurrence else resolved.kernel_regex)
        return sel or "all"
    # blackwell: main-last via launch_selection.pick_repeat
    from .launch_selection import (CYCLES_METRIC, demangle_many,
                                   main_kernel_matches, pick_repeat,
                                   selection_for)
    if not resolved.calls:
        raise KeyError("blackwell launch selection needs spec 'calls'")
    rows = [json.loads(l) for l in
            (out / "ncu_launches.jsonl").read_text().splitlines()
            if l.strip()] if (out / "ncu_launches.jsonl").is_file() else []
    if not rows:
        raise KeyError("main-last pick needs ncu_launches.jsonl")
    pick_rows = [{"mangled": r["function"], "ordinal": r["ordinal"],
                  CYCLES_METRIC: r.get("cycles_active_max") or 0,
                  "grid": r.get("grid"), "block": r.get("block")}
                 for r in rows]
    if resolved.main_regex is not None:
        dem = demangle_many(sorted({p["mangled"] for p in pick_rows}))
        pick_rows = [p for p in pick_rows if main_kernel_matches(
            resolved.main_regex.pattern, p["mangled"],
            dem.get(p["mangled"]))]
        if not pick_rows:
            raise ValueError("main-last pick: no ncu launch matches "
                             "main_kernel regex")
    pick = pick_repeat(pick_rows, resolved.calls, resolved.family,
                       aux_regexes=[])
    if pick.get("error"):
        raise ValueError(f"main-last pick: {pick['error']}")
    # Hand the tracer the picked kernel by name + occurrence (the Nth
    # launch of that mangled name, 1-based), not by census ordinal: the
    # ncu census is 0-based while tracer launch ids are 1-based, and the
    # two runs need not even see the same launch list.  Every round-1/2
    # blackwell capture traced the launch before the GEMM because of it.
    same = [p for p in pick_rows if p["mangled"] == pick["name"]]
    occurrence = next(i for i, p in enumerate(same)
                      if p["ordinal"] == pick["ordinal"]) + 1
    return selection_for(pick["name"], occurrence)


def tracer_env(base_env: dict, tracer: Path, target: Path,
               selection: str) -> dict:
    """Environment for a traced run (trace or placement mode)."""
    run_env = dict(base_env)
    run_env.update({
        "CUDA_INJECTION64_PATH": str(tracer),
        "CYCLESCOPE_TRACE_DIR": str(target),
        # diagnostics may override the selection from the caller's
        # environment (bisecting an app that hangs under the tracer);
        # the default is the capture contract
        "CYCLESCOPE_CAPTURE_LAUNCHES": os.environ.get(
            "CYCLESCOPE_CAPTURE_LAUNCHES_OVERRIDE", selection),
        "CYCLESCOPE_CAPTURE_CLUSTERS": "all",
        # L2 footprint capture is retired (cold reference); the tracer
        # defaults it on when the variable is unset, so pin it off
        "CYCLESCOPE_L2_FOOTPRINT": "0",
        "CYCLESCOPE_CUDA_STACK_SIZE": "65536",
        "CYCLESCOPE_TRACER_VERBOSE": "1",
        # a writable HOME: CUDA honors injection only when the user
        # config/cache dir is writable (pods may run as rootless users
        # whose $HOME is read-only) — smoke scripts used /tmp.
        "HOME": "/tmp",
        # prepend the driver dirs so the app binds the real libcuda
        # (injection-capable), not a stub or vendored copy.
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/nvidia/lib64"
        + (":" + base_env["LD_LIBRARY_PATH"]
           if base_env.get("LD_LIBRARY_PATH") else ""),
        "NVDISASM": site.nvdisasm(),
        "PATH": str(site.tools_dir() / "bin") + ":"
        + base_env.get("PATH", os.defpath),
    })
    return run_env


def _ensure_scratch_dirs(argv: list, run_env: dict, target: Path) -> None:
    """Per-case scratch roots referenced by argv/env are empty writable
    dirs by contract (JIT caches, workspaces); file outputs only need
    their parent.  Create every ``<target>/scratch/<name>`` dir a token
    points at; names containing '.' are file outputs and skipped."""
    for tok in list(argv) + [v for v in run_env.values()
                             if isinstance(v, str)]:
        m = re.search(re.escape(str(target)) + r"/scratch/([^/]+)/?$",
                      tok)
        if m and "." not in m.group(1):
            (target / "scratch" / m.group(1)).mkdir(
                parents=True, exist_ok=True)


def _traced_run(out: Path, resolved: Resolved, env: dict, tracer: Path,
                selection: str, mode: str, timeout_s: int) -> dict:
    target = out if mode == "trace" else out / "placement"
    target.mkdir(parents=True, exist_ok=True)
    run_env = tracer_env(env, tracer, target, selection)
    if mode == "placement":
        run_env["CYCLESCOPE_CAPTURE_MODE"] = "placement"
    argv, run_env = case_command(resolved, target, run_env)
    _ensure_scratch_dirs(argv, run_env, target)
    log = target / "app.log"
    (target / "run_env.json").write_text(
        json.dumps({k: ("<redacted>" if _SECRET_KEY.search(k) else str(v))
                    for k, v in sorted(run_env.items())
                    if _record_env(k, v)}))
    t0 = time.time()
    with log.open("w") as fh:
        rc = run_group(argv, env=run_env, stdout=fh,
                       stderr=subprocess.STDOUT, cwd=resolved.cwd,
                       timeout_s=timeout_s)
    if rc == TIMEOUT_RC:
        rc = -9
    return {"rc": rc, "wall_s": round(time.time() - t0, 2),
            "dir": str(target), "mode": mode}


_SECRET_KEY = re.compile(
    r"(?i)(SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|ACCESS_KEY"
    r"|API_KEY|_KEY$)")

_ENV_PREFIXES = ("CUDA", "NVIDIA", "CYCLESCOPE_", "LD_", "BW_", "FLASHINFER",
                 "TRITON")
_ENV_NAMES = {"PATH", "PYTHONPATH", "HOME"}


def _record_env(key: str, value) -> bool:
    """run_env.json records a variable when the run sets/changes it
    relative to the parent environment, or when it belongs to the
    toolchain/GPU surface."""
    return (os.environ.get(key) != str(value)
            or key.startswith(_ENV_PREFIXES) or key in _ENV_NAMES)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_revision() -> tuple[str | None, bool | None]:
    """(HEAD sha, tracked-file dirty flag) of the checkout this driver
    runs from; (None, None) when git or the checkout is unavailable."""
    try:
        head = subprocess.run(
            ["git", "-c", f"safe.directory={REPO}",
             "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30)
        sha = head.stdout.strip() if head.returncode == 0 else None
        dirty = None
        if sha:
            st = subprocess.run(
                ["git", "-c", f"safe.directory={REPO}",
                 "-C", str(REPO), "status", "--porcelain",
                 "--untracked-files=no"],
                capture_output=True, text=True, timeout=60)
            if st.returncode == 0:
                dirty = bool(st.stdout.strip())
        return sha or None, dirty
    except Exception:
        return None, None


def _gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=15).stdout.strip()
    except Exception:
        out = ""
    return {"gpu": out}


class _CensusFailed(Exception):
    """ncu census failed where selection depends on it (Blackwell)."""


def _capture_selection(resolved: Resolved, out: Path,
                       reference_case_dir: Path | None,
                       census_fn) -> tuple[str, str]:
    """(selection, source) for the traced run.  Priority: --reference
    key > fixed launch table > ncu census.  Only the last leg runs the
    census (``census_fn`` wraps ``_ncu_launch_pass``); a reference or a
    table row skips the prepass entirely."""
    if reference_case_dir is not None:
        return _reference_selection(reference_case_dir, out), "reference"
    if resolved.launch_key is not None:
        from .launch_selection import selection_for
        return selection_for(*resolved.launch_key), "table"
    ncu = census_fn()
    if ncu.get("status") != "ok":
        # a partial census must never feed launch selection
        if (out / "ncu_launches.jsonl").is_file() and                 ncu.get("status") != "ncu_missing":
            (out / "ncu_launches.jsonl").unlink()
        if not resolved.argv_row:
            # main-last selection needs the census: fail closed
            raise _CensusFailed(ncu.get("status"))
        # argv-row cases select by kernel_regex@occurrence; the census is
        # only a cross-check and its absence is recorded, not fatal
    return _select_launch(resolved, out), "census"


def _check_traced_against_table(out: Path, row: dict) -> str | None:
    """Post-trace check for launch_source=table: the traced launch in
    launches.jsonl must have the row's kernel/grid/block/cluster.  None
    when it matches or no launch was traced (trace_check owns that)."""
    launches = out / "launches.jsonl"
    if not launches.is_file():
        return None
    traced = [json.loads(l) for l in launches.read_text().splitlines()
              if l.strip() and json.loads(l).get("traced")]
    if not traced:
        return None
    t = traced[0]
    dims = lambda v: "x".join(str(x) for x in (v or []))
    diffs = []
    if t.get("mangled") != row.get("kernel"):
        diffs.append(f"kernel {t.get('mangled')} != {row.get('kernel')}")
    for k in ("grid", "block", "cluster"):
        if dims(t.get(k)) != row.get(k):
            diffs.append(f"{k} {dims(t.get(k))} != {row.get(k)}")
    return "; ".join(diffs) or None


def _toolchain_record(env: dict) -> dict:
    """Provenance of the binaries a capture ran under.  Facts only;
    never converter input, never a capture failure reason."""
    try:
        path = env.get("PATH", os.environ.get("PATH", ""))
        tools = ["nvcc", "ptxas", "ptxas-blackwell", "nvdisasm",
                 "cuobjdump", "ncu"]
        which = {}
        for t in tools:
            found = shutil.which(t, path=path)
            if found:
                which[t] = found
        for k in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH"):
            if env.get(k):
                which[k] = env[k]
        versions = {}
        for t, exe in which.items():
            try:
                rc = subprocess.run([exe, "--version"],
                                    capture_output=True, text=True,
                                    timeout=10)
                # the version number is not on the first line of the
                # NVIDIA tools' banners: keep the whole (short) output
                text = " | ".join(l.strip() for l in
                                  (rc.stdout or rc.stderr).splitlines()
                                  if l.strip())
                versions[t] = text[:400] or None
            except Exception:
                versions[t] = None
        packages = {}
        import importlib.metadata
        for p in ("torch", "triton", "numpy", "flashinfer-python",
                  "nvidia-cutlass-dsl", "cuda-python"):
            try:
                packages[p] = importlib.metadata.version(p)
            except Exception:
                pass
        return {"which": which, "versions": versions,
                "python": sys.version, "packages": packages}
    except Exception as e:
        return {"error": str(e)}


def _reference_selection(case_dir: Path, out: Path) -> str:
    """Tracer selection for the launch a `cyclescope reference` case directory
    timed (kernel + occurrence), checked against this run's census."""
    from ..reference.ncu_runner import reference_launch_key
    from .launch_selection import census_row, selection_for
    key = reference_launch_key(case_dir)
    if key is None:
        raise KeyError(f"no timed launch in {case_dir}")
    census = out / "ncu_launches.jsonl"
    if census.is_file():
        rows = [json.loads(l) for l in census.read_text().splitlines()
                if l.strip()]
        if census_row(rows, *key, name_key="function") is None:
            raise KeyError(f"reference launch {key[0]}@{key[1]} not in "
                           f"the ncu census")
    return selection_for(*key)


def capture_case(suite: str, case_id: str, out: Path, *,
                 mode: str = "both", ncu_repeats: int = 0,
                 tracer: Path | None = None, timeout_s: int = 1800,
                 dry_run: bool = False,
                 reference: Path | None = None) -> dict:
    """``reference`` (a `cyclescope reference` output root,
    <root>/<case>/<case>/timing.json): trace the launch that reference
    timed instead of re-picking with the suite rule."""
    site.export_environment()
    arch = suite.split("/", 1)[0]
    if tracer is None:
        sm = "sm_90a" if arch == "hopper" else "sm_103a"
        tracer = site.tracer(sm)
    workdir = out / ".work"
    workdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    resolved = resolve_case(suite, case_id, workdir)
    base_env = dict(os.environ)
    base_env.update(resolved.env)
    rec = {"suite": suite, "case": case_id, "argv": resolved.argv,
           "env": resolved.env, "cwd": str(resolved.cwd or ""),
           "tracer": str(tracer), "mode": mode,
           "capture_format": "cyclescope.capture.v1"}
    if dry_run:
        rec["launch_selection"] = (
            f"{resolved.kernel_regex}@{resolved.occurrence}"
            if resolved.argv_row else "main-last")
        print(json.dumps(rec, indent=2, default=str))
        return rec

    t_start = time.time()
    rec["utc_start"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["gpu"] = _gpu_info()
    rec["toolchain"] = _toolchain_record(base_env)
    rec["tracer_sha256"] = _sha256(tracer)

    prep = prepare_case(resolved, workdir, out / "triton_prepass.log",
                        timeout_s=min(timeout_s, 1200))
    rec["prepass"] = prep
    if prep.get("rc", 0) != 0:
        rec["status"] = "app_failed:triton_prepass"
        rec["rc"] = 1
        (out / "run.json").write_text(
            json.dumps(rec, indent=2, default=str))
        return rec
    # prepare_case may inject env (triton import root / PYTHONPATH)
    base_env.update(resolved.env)

    if mode == "both":
        modes = ["trace", "placement"]
    else:
        modes = [mode]
    ref_dir = reference / case_id / case_id if reference is not None else None
    if ref_dir is not None:
        rec["launch_from"] = str(ref_dir)

    def _census():
        ncu = _ncu_launch_pass(out, resolved, base_env, timeout_s)
        rec["ncu"] = ncu
        return ncu
    try:
        selection, source = _capture_selection(resolved, out, ref_dir,
                                               _census)
    except _CensusFailed as e:
        rec["status"] = f"runtime_env:{e.args[0]}"
        rec["rc"] = 1
        (out / "run.json").write_text(
            json.dumps(rec, indent=2, default=str))
        return rec
    except Exception as e:
        # fail closed: never trace an arbitrary launch
        rec["status"] = f"selection_failed:{e}"
        rec["rc"] = 1
        (out / "run.json").write_text(json.dumps(rec, indent=2, default=str))
        return rec
    rec["launch_source"] = source
    rec["launch_selection"] = selection

    for m in modes:
        rec[f"run_{m}"] = _traced_run(out, resolved, base_env, tracer,
                                      selection, m, timeout_s)

    if "placement" in modes and rec["run_placement"]["rc"] != 0:
        # a crashed/hung placement run must not silently pass: the
        # verdict is FAIL even when the trace itself is complete
        rec["status"] = f"placement_failed:rc{rec['run_placement']['rc']}"
        rec["rc"] = 1
        (out / "run.json").write_text(
            json.dumps(rec, indent=2, default=str))
        return rec

    # trace_check as completion criterion
    sys.path.insert(0, str(REPO / "python"))
    try:
        from cyclescope.capture import trace_check  # noqa
        rc = subprocess.run(
            [sys.executable, "-m", "cyclescope.capture.trace_check", str(out),
             "--quick"],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONPATH": str(REPO / "python")})
        rec["trace_check_rc"] = rc.returncode
        rec["trace_check_mode"] = "quick"
        (out / "trace_check.txt").write_text(rc.stdout + rc.stderr)
    except Exception as e:
        rec["trace_check_rc"] = -1
        rec["trace_check_error"] = str(e)

    if rec.get("launch_source") == "table" and resolved.launch_row:
        diff = _check_traced_against_table(out, resolved.launch_row)
        if diff:
            rec["status"] = f"launch_table_mismatch:{diff}"
            rec["rc"] = 1

    rec["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["wall_s"] = round(time.time() - t_start, 2)
    rec["source_sha"], rec["source_dirty"] = _source_revision()
    (out / "run.json").write_text(json.dumps(rec, indent=2, default=str))
    for root, dirs, files in os.walk(out):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)
    return rec
