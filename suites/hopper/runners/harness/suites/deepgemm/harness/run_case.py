#!/usr/bin/env python3
"""Replay a qualified DG formal invocation as a bare, non-traced measurement.

Only host contacts/package and mutable output paths are redirected. Original
trial count, aggregation, shape, grid/warp/cluster choices and device headers
are retained. No activity profiler or cache-name identity inference is used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from trace_cluster_sm_case import (DEFAULT_QUALIFICATION_ROOT, HARNESS_ROOT,
                                   RUNNER_MODULE, RUNTIME_MANIFEST, ATTENTION_RUNNER,
                                   load_case, formal_attempt, parse_formal_invocation)
from tools.microbench.actual_binding import sha256


def prepare_replay(args: argparse.Namespace) -> tuple[list[str], dict[str, str], dict]:
    case = load_case(args.qualification_root / "deepgemm/qualification.json", args.case_id)
    attempt = formal_attempt(case)
    env_delta, argv = parse_formal_invocation(Path(attempt["log"]))
    original_result = json.loads(Path(attempt["result"]).read_text())
    aggregation = original_result["timing"]["sample_aggregation"]
    raw_per_sample = int(aggregation["raw_spans_per_recorded_sample"])
    if raw_per_sample not in (1, 3, 4):
        raise RuntimeError("original raw aggregation protocol is unsupported")
    env_delta["DEEPGEMM_DELIVERY_RAW_PER_SAMPLE"] = str(raw_per_sample)
    for flag in ("--package", "--manifest", "--cache-root", "--output", "--mode"):
        if flag not in argv:
            raise RuntimeError(f"original formal command missing {flag}")
    if "--trace" in argv or "--pilot" in argv:
        raise RuntimeError("bare replay requires original non-trace formal command")
    if argv[1:3] == ["-m", RUNNER_MODULE]:
        if not Path(argv[argv.index("--manifest") + 1]).is_file():
            argv[argv.index("--manifest") + 1] = str(RUNTIME_MANIFEST)
    elif argv[1].endswith("deepgemm_attention_runner.py"):
        argv[1] = str(ATTENTION_RUNNER)
    else:
        raise RuntimeError("unsupported original formal runner")
    original_cache = Path(argv[argv.index("--cache-root") + 1])
    original_package = Path(argv[argv.index("--package") + 1])
    if not original_cache.is_dir() or not args.package.is_dir():
        raise RuntimeError("original cache or host-observer package missing")
    work = args.work_dir.resolve()
    replacements = {"--package": args.package.resolve(), "--cache-root": work / "cache", "--output": work / "result.json"}
    for flag, value in replacements.items():
        argv[argv.index(flag) + 1] = str(value)
    argv[0] = sys.executable
    env = os.environ.copy()
    env.update(env_delta)
    if any(env.get(key) for key in ("CUDA_INJECTION64_PATH", "NVBIT_TOOL", "NVBIT_TOOL_NAME", "KINETO_LOG_LEVEL")):
        raise RuntimeError("bare measurement must not inherit tracer/profiler injection")
    if env.get("LD_PRELOAD"):
        raise RuntimeError("bare measurement requires an explicit observer-only preload")
    if env.get("FLASHINFER_CLUSTER_SM_TARGET_SMS"):
        raise RuntimeError("observer must not override original device multiprocessor count")
    env.update({"PYTHONPATH": str(HARNESS_ROOT), "GEMM_DELIVERY_GPU": "h100",
                "DEEPGEMM_ACTUAL_BINDING_REQUIRED": "1", "LD_PRELOAD": str(args.observer.resolve()),
                "DEEPGEMM_ORIGINAL_DEVICE_LIBRARY_ROOT": str(original_package / "deep_gemm"),
                "FLASHINFER_CLUSTER_LAUNCH_LOG": str(work / "launches.jsonl"),
                "FLASHINFER_LOADED_CODE_LOG": str(work / "loaded_code.jsonl"),
                "DEEPGEMM_ACTUAL_SOURCE_MANIFEST": str(work / "source_manifest.json")})
    if args.unit_json:
        env["DEEPGEMM_ACTUAL_UNIT_JSON"] = str(args.unit_json.resolve())
    return argv, env, {"case_id": args.case_id, "formal_attempt": attempt,
                       "original_cache": str(original_cache), "original_package": str(original_package),
                       "machine": socket.gethostname(), "argv": argv, "env_delta": {key: value for key, value in env.items() if os.environ.get(key) != value}}


def run(args: argparse.Namespace) -> int:
    argv, env, receipt = prepare_replay(args)
    if args.plan:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    work = args.work_dir.resolve()
    if work.exists():
        raise RuntimeError("bare work-dir must be new; historical results are never overwritten")
    work.parent.mkdir(parents=True, exist_ok=True)
    source = Path(receipt["original_cache"])
    files = [path for path in source.rglob("*") if path.is_file()]
    required = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(work.parent).free - required < 90 * 1024**3:
        raise RuntimeError("90 GiB free-space guard prevents immutable cache staging")
    work.mkdir(parents=True)
    shutil.copytree(source, work / "cache")
    inventory = {}
    for old in files:
        new = work / "cache" / old.relative_to(source)
        if old.suffix not in (".cubin", ".cu", ".ptx", ".fatbin"):
            continue
        digest = sha256(old)
        if sha256(new) != digest:
            raise RuntimeError("original-source cache snapshot changed during copy")
        new.chmod(0o444)
        stat = new.stat()
        inventory[str(new)] = {"source": str(old), "sha256": digest, "size": stat.st_size, "dev": stat.st_dev, "ino": stat.st_ino}
    (work / "source_manifest.json").write_text(json.dumps({"schema": "deepgemm.actual-source-inventory.v1", "files": inventory}, indent=2, sort_keys=True) + "\n")
    receipt["observer_sha256"] = sha256(args.observer)
    (work / "invocation.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with (work / "run.log").open("w") as log:
        completed = subprocess.run(argv, env=env, cwd=HARNESS_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    receipt["returncode"] = completed.returncode
    receipt["source_post_unchanged"] = all(sha256(Path(path)) == row["sha256"] for path, row in inventory.items())
    result_path = work / "result.json"
    receipt["runner_status"] = json.loads(result_path.read_text()).get("status") if result_path.is_file() else None
    passed = completed.returncode == 0 and receipt["source_post_unchanged"] and receipt["runner_status"] == "pass"
    receipt["status"] = "bare_complete" if passed else "bare_failed"
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return completed.returncode if completed.returncode else (0 if passed else 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--qualification-root", type=Path, default=DEFAULT_QUALIFICATION_ROOT)
    parser.add_argument("--package", type=Path, required=True, help="host-only observer package built from original full7 source")
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--unit-json", type=Path)
    parser.add_argument("--plan", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
