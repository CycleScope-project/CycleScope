#!/usr/bin/env python3
"""Resolve one qualified DeepGEMM cluster-SM case's formal invocation.

The formal cluster-SM qualification (cluster_sm_400_20260920) ran every case
with the pinned ``tools.microbench.deepgemm_sm90_runner`` /
``deepgemm_attention_runner.py`` and persisted the exact ``ENV``/``CMD``
header in each formal attempt's run.log.  This module is the deterministic
resolver of that formal invocation:

- :func:`load_case` / :func:`formal_attempt` / :func:`parse_formal_invocation`
  recover the qualified case row, its final formal attempt and the pinned
  environment/command pair;
- :func:`plan_trace_invocation` redirects the formal command's mutable paths
  (JIT cache, runner result) into one work directory and appends ``--trace``,
  the runner protocol that keeps the correctness launch and the collection
  warmup loop outside exactly one ``cudaProfilerStart/Stop``-bracketed target
  launch (the warmup the library needs for correctness stays inside the
  runner).

The capture is not orchestrated here anymore (stage-6 cutover): the only
real capture entry is
``hoppersim capture via suites/common/capture_case.py``,
driven by ``suites/common/capture_case.py`` with the invocation that
``scripts/gen_invocations.py`` derives through this resolver.
``harness/run_once_case.py`` is the pure run-once program that invocation
points at (it replays the formal command once with the delivery environment
and no capture semantics); the marker-timing measurement body stays in the
runners themselves (``tools/microbench/deepgemm_sm90_runner.py``,
``deepgemm_attention_runner.py``).
"""

from __future__ import annotations

import json
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = SUITE_ROOT / "harness"
RUNNER_MODULE = "tools.microbench.deepgemm_sm90_runner"
RUNTIME_MANIFEST = SUITE_ROOT / "manifests/deepgemm_core_runtime_manifest.json"
ATTENTION_RUNNER = HARNESS_ROOT / "deepgemm_attention_runner.py"
DEFAULT_QUALIFICATION_ROOT = Path(
    "/path/to/hopper-prototype/artifacts/cluster_sm_400_20260920"
)
RUNNER_RESULT_NAME = "deepgemm_runner_result.json"


def load_case(qualification_path: Path, case_id: str) -> dict:
    payload = json.loads(qualification_path.read_text(encoding="utf-8"))
    matches = [
        dict(row)
        for row in payload["cases"]
        if str(row.get("source_case_id") or "") == case_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one qualified DeepGEMM case {case_id!r} in {qualification_path}, "
            f"found {len(matches)}"
        )
    case = matches[0]
    if case.get("status") != "qualified":
        raise RuntimeError(
            f"{case_id}: qualification status is {case.get('status')!r}, not 'qualified'"
        )
    if int(case.get("target_ctas") or 0) not in (1, 2):
        raise RuntimeError(f"{case_id}: unsupported target_ctas {case.get('target_ctas')!r}")
    return case


def formal_attempt(case: dict) -> dict:
    """The final attempt is the formal one (retunes rerun ``formal`` last)."""

    attempts = [dict(row) for row in case.get("attempts") or []]
    if not attempts:
        raise RuntimeError(f"{case.get('source_case_id')}: qualification has no attempts")
    attempt = attempts[-1]
    if attempt.get("tag") != "formal":
        raise RuntimeError(
            f"{case.get('source_case_id')}: final attempt tag is "
            f"{attempt.get('tag')!r}, not 'formal'"
        )
    return attempt


def parse_formal_invocation(run_log: Path) -> tuple[dict[str, str], list[str]]:
    """Read the pinned ``ENV``/``CMD`` header the formal launchers wrote.

    The launchers persist the command as ``" ".join(cmd)`` without shell
    quoting, so the faithful inverse is a plain space split: no formal
    argument (path or ``--shape-json`` blob) contains a space, and any
    drift shows up as an argparse failure instead of silent mangling.
    """

    env: dict[str, str] = {}
    command: list[str] | None = None
    for line in run_log.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENV "):
            key, sep, value = line[4:].partition("=")
            if sep and key:
                env[key] = value
        elif line.startswith("CMD "):
            command = line[4:].split(" ")
    if command is None:
        raise RuntimeError(f"{run_log}: formal run.log has no CMD line")
    return env, command


def plan_trace_invocation(
    case: dict,
    command: list[str],
    delivery_env: dict[str, str],
    *,
    output_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    """Redirect the formal command into one work directory, plus --trace.

    Two formal runner forms exist and both replay verbatim: the core GEMM
    module (``-m tools.microbench.deepgemm_sm90_runner``) and the public
    attention script (``deepgemm_attention_runner.py`` with its per-case
    generated manifest).  Manifest/package/case-id/class-id/shape-json or
    attention-manifest/mode/num-sms/round are copied from the formal CMD;
    only the cache and output paths move under ``output_dir``, a logged
    runner/manifest path that no longer exists falls back to this
    checkout's pinned copy, and ``--trace`` is appended.
    """

    case_id = str(case.get("source_case_id"))
    argv = list(command)
    for flag in ("--manifest", "--package", "--cache-root", "--output", "--mode"):
        if flag not in argv:
            raise RuntimeError(f"{case_id}: formal command lacks {flag}: {argv}")
    if argv[argv.index("--mode") + 1] != "cluster_sm":
        raise RuntimeError(
            f"{case_id}: formal command mode is not cluster_sm: "
            f"{argv[argv.index('--mode') + 1]!r}"
        )
    if "--trace" in argv or "--pilot" in argv:
        raise RuntimeError(f"{case_id}: formal command must be a timing run: {argv}")

    manifest = Path(argv[argv.index("--manifest") + 1])
    package = Path(argv[argv.index("--package") + 1])
    if argv[1:3] == ["-m", RUNNER_MODULE]:
        if not manifest.is_file():
            manifest = RUNTIME_MANIFEST
    elif argv[1].endswith("deepgemm_attention_runner.py"):
        if not Path(argv[1]).is_file():
            argv[1] = str(ATTENTION_RUNNER)
        # The attention manifest is generated per formal attempt and stays
        # inside the qualification tree; it is identity, not rebuildable.
        if not manifest.is_file():
            raise RuntimeError(
                f"{case_id}: formal attention manifest is missing on disk: {manifest}"
            )
    else:
        raise RuntimeError(
            f"{case_id}: formal command does not invoke a formal DeepGEMM "
            f"runner: {command[:3]}"
        )
    if not package.is_dir():
        raise RuntimeError(
            f"{case_id}: formal marker package is missing on disk: {package}"
        )

    argv[argv.index("--manifest") + 1] = str(manifest)
    argv[argv.index("--package") + 1] = str(package)
    argv[argv.index("--cache-root") + 1] = str(output_dir / "deepgemm_jit_cache")
    argv[argv.index("--output") + 1] = str(output_dir / RUNNER_RESULT_NAME)
    argv.append("--trace")
    return argv, dict(delivery_env)
