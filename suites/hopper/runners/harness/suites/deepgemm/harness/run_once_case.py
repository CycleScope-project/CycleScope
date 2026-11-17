#!/usr/bin/env python3
"""Run one qualified DeepGEMM case's formal invocation exactly once.

Stage-6 run-once program: resolves the case's formal ``ENV``/``CMD`` from
the qualification record and replays that exact runner invocation once in a
child process — the runner's ``--trace`` protocol keeps the correctness
launch and the collection warmup loop (the warmup the library needs for
correctness) outside exactly one ``cudaProfilerStart/Stop``-bracketed
target launch.  This program owns no capture semantics: it never sets any
NVBit injection variable, never post-processes traces and never captures in
process — under the unified entry
(``hoppersim capture via suites/common/capture_case.py``
via ``suites/common/capture_case.py``) the tracer environment and
the post chain belong to the capture entry, and standalone it is just the
real program.

The work directory (JIT cache + runner result) defaults to ``$TRACES_FOLDER``
— the capture directory the unified entry exports to the traced program;
``--work-dir`` overrides it for standalone runs and ``--plan`` prints the
resolved replay command without running anything (CPU-safe validation:
missing manifests/packages fail closed before any GPU is touched).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HARNESS_ROOT = Path(__file__).resolve().parent
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from trace_cluster_sm_case import (  # noqa: E402
    DEFAULT_QUALIFICATION_ROOT,
    RUNNER_RESULT_NAME,
    formal_attempt,
    load_case,
    parse_formal_invocation,
    plan_trace_invocation,
)

TRACE_WORK_DIR_PLACEHOLDER = "<TRACES_FOLDER>"


def replay_environment(delivery_env: dict[str, str]) -> dict[str, str]:
    """Delivery identity for the replay: formal ENV delta plus the suite's
    harness PYTHONPATH.  No NVBit injection variable is ever set here."""

    environment = os.environ.copy()
    environment.update(delivery_env)
    environment.setdefault("GEMM_DELIVERY_GPU", "h100")
    harness = str(HARNESS_ROOT)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        harness + os.pathsep + existing if existing else harness
    )
    return environment


def resolve_work_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    """(work_dir, placeholder) — the capture dir under the unified entry."""

    if args.work_dir is not None:
        return args.work_dir.expanduser().resolve(), False
    traces_folder = os.environ.get("TRACES_FOLDER", "")
    if traces_folder:
        return Path(traces_folder).resolve(), False
    if args.plan:
        # CPU plan print: show the placeholder the capture entry substitutes.
        return Path(TRACE_WORK_DIR_PLACEHOLDER), True
    raise SystemExit(
        "run_once_case: no work directory (pass --work-dir; under the unified "
        "capture entry $TRACES_FOLDER provides it)"
    )


def plan(args: argparse.Namespace) -> int:
    qualification_path = (
        args.qualification_root.resolve() / "deepgemm" / "qualification.json"
    )
    work_dir, placeholder = resolve_work_dir(args)
    case = load_case(qualification_path, args.case_id)
    attempt = formal_attempt(case)
    formal_env, command = parse_formal_invocation(Path(attempt["log"]))
    argv, delivery_env = plan_trace_invocation(
        case, command, formal_env, output_dir=work_dir
    )
    environment = dict(delivery_env)
    environment.setdefault("GEMM_DELIVERY_GPU", "h100")
    print(
        json.dumps(
            {
                "status": "plan",
                "case_id": args.case_id,
                "argv": argv,
                "env_delta": environment,
                "cwd": str(HARNESS_ROOT),
                "work_dir": str(work_dir),
                "work_dir_placeholder": placeholder,
                "formal_attempt": str(attempt.get("result") or ""),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_once(args: argparse.Namespace) -> int:
    qualification_path = (
        args.qualification_root.resolve() / "deepgemm" / "qualification.json"
    )
    work_dir, _ = resolve_work_dir(args)
    case = load_case(qualification_path, args.case_id)
    attempt = formal_attempt(case)
    formal_env, command = parse_formal_invocation(Path(attempt["log"]))
    argv, delivery_env = plan_trace_invocation(
        case, command, formal_env, output_dir=work_dir
    )
    environment = replay_environment(delivery_env)
    completed = subprocess.run(argv, cwd=str(HARNESS_ROOT), env=environment)
    summary = {
        "status": (
            "run_once_complete" if completed.returncode == 0 else "run_once_failed"
        ),
        "case_id": args.case_id,
        "argv": argv,
        "returncode": completed.returncode,
        "runner_result": str(work_dir / RUNNER_RESULT_NAME),
        "runner_result_status": None,
    }
    result_path = work_dir / RUNNER_RESULT_NAME
    if completed.returncode == 0:
        if not result_path.is_file():
            print(
                f"run_once_case: {args.case_id}: runner exited 0 but wrote no "
                f"result file: {result_path}",
                file=sys.stderr,
            )
            return 4
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"run_once_case: {args.case_id}: unreadable runner result "
                f"{result_path}: {exc}",
                file=sys.stderr,
            )
            return 4
        summary["runner_result_status"] = result.get("status")
        if result.get("status") != "trace_pass":
            print(
                f"run_once_case: {args.case_id}: runner status "
                f"{result.get('status')!r}, expected 'trace_pass'",
                file=sys.stderr,
            )
            summary["status"] = "run_once_failed"
            print(json.dumps(summary, sort_keys=True))
            return 5
    print(json.dumps(summary, sort_keys=True))
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_once_case",
        description=(
            "Replay one qualified DeepGEMM case's formal invocation exactly "
            "once (no capture semantics; --trace window protocol included)"
        ),
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--qualification-root",
        type=Path,
        default=DEFAULT_QUALIFICATION_ROOT,
        help="cluster-SM qualification tree holding deepgemm/qualification.json",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="work directory for the JIT cache and runner result "
        "(default: $TRACES_FOLDER under the unified capture entry)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="resolve and print the replay command without running anything",
    )
    args = parser.parse_args(argv)
    if args.plan:
        return plan(args)
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
