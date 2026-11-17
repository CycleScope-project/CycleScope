#!/usr/bin/env python3
"""B300S CUTLASS 1SM/2SM cases. One file. No hopper-prototype pipeline."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ["CYCLESCOPE_RUNNER_ROOT"])
sys.path.insert(0, str(ROOT / "suites"))
from common import write_result, write_text  # noqa: E402

CUTLASS = ROOT / "third_party" / "cutlass"
# Honor BW_CORPUS_ROOT (SC campaign root) exactly like suites/common.py;
# unset -> byte-identical legacy paths.
ART = Path(os.environ.get("BW_CORPUS_ROOT", str(ROOT / "artifacts" / "corpus")))
BIN = ART / "cutlass" / "bin"
DSL = CUTLASS / "examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py"
SPEC_SC = ROOT / "suites" / "c450_full" / "cases_cutlass.json"
NVCC = ["nvcc", "-std=c++17", "-O3", "--expt-relaxed-constexpr",
        "-gencode=arch=compute_103a,code=compute_103a",
        "-gencode=arch=compute_103a,code=sm_103a",
        "-lcuda"]
INCS = [
    f"-I{CUTLASS / 'include'}",
    f"-I{CUTLASS / 'tools' / 'util' / 'include'}",
    f"-I{CUTLASS / 'examples' / 'common'}",
    f"-I{CUTLASS / 'examples' / 'cute' / 'tutorial' / 'blackwell'}",
]

def load_sc_cases() -> list[dict]:
    cases_dir = os.environ.get("BW_CASES_DIR")
    path = Path(cases_dir) / "cases_cutlass.json" if cases_dir else SPEC_SC
    with open(path) as f:
        return json.load(f)["cases"]


def run(cmd, cwd=None, env=None, timeout=1800):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout or ""


def sass_hits(path: Path) -> str:
    if not path.exists():
        return ""
    blobs = []
    for cmd in (["cuobjdump", "-ptx", str(path)], ["cuobjdump", "--dump-sass", str(path)]):
        _, out = run(cmd, timeout=180)
        blobs.append(out)
    text = "\n".join(blobs)
    hits = [ln.strip() for ln in text.splitlines() if "cta_group" in ln.lower() or "tcgen05.mma" in ln.lower()]
    return "\n".join(hits[:80])


def proof_ok_2c(proof: str) -> bool:
    p = proof.lower()
    return "cta_group::2" in p or "knummulticast=2" in p or "use_2cta_instrs=true" in p


def compile_one(name: str, src: Path, reuse: bool = True, defs=(),
                bin_dir: str | None = None,
                bin_root: Path | None = None) -> tuple[int, str, Path]:
    bin_root = Path(bin_root) if bin_root is not None else \
        ART / "cutlass" / (bin_dir or "bin")
    bin_root.mkdir(parents=True, exist_ok=True)
    dst = bin_root / name
    # Examples with subdirectory layouts (77 fmha/mla/bwd: reference/ -> collective/,
    # device/ -> kernel/) need the example dir itself on the include path.
    cmd = NVCC + INCS + [f"-I{src.parent}"] + [f"-D{d}" for d in defs] + [str(src), "-o", str(dst)]
    if name.startswith("89_"):
        reuse = False
        try:
            dst.unlink()
        except OSError:
            pass
    if reuse and dst.exists():
        return 0, "reuse " + " ".join(cmd), dst
    rc, out = run(cmd, timeout=2400)
    return rc, " ".join(cmd) + "\n" + out, dst


def payload_with_extra(cid: str, kw: dict) -> dict:
    # Key order matches the legacy payload exactly; spec-only fields
    # (family/est_cycles/est_basis) are appended after via `extra`.
    payload = dict(
        id=cid,
        suite="cutlass",
        ok=False,
        cta_group=kw.get("cta_group", 1),
        cluster=kw.get("cluster", ""),
        sms=kw.get("sms", 1),
        dtype=kw.get("dtype", ""),
        shape=kw.get("shape", ""),
        kernel=kw.get("kernel", ""),
        proof=kw.get("proof", ""),
        error="",
        log=kw.get("log", ""),
    )
    extra = kw.get("extra")
    if extra:
        payload.update(extra)
    return payload


def fail(cid, **kw):
    payload = payload_with_extra(cid, kw)
    payload["ok"] = False
    payload["error"] = kw.get("error", "")
    write_result("cutlass", cid, payload)


def ok(cid, **kw):
    payload = payload_with_extra(cid, kw)
    payload["ok"] = True
    payload["error"] = None
    if payload["cta_group"] == 2 and not proof_ok_2c(payload["proof"]):
        payload["ok"] = False
        payload["error"] = "2CTA without cta_group::2 / kNumMulticast=2 / use_2cta_instrs=True"
    write_result("cutlass", cid, payload)


def case_cpp(cid, bin_name, src, args, *, cta_group, cluster, sms, dtype, shape, kernel, extra=None, defs=(),
             bin_dir=None):
    full_mode = extra is not None and extra.get("mode") == "full"
    shared_bin = os.environ.get("BW_CUTLASS_BIN_ROOT") if full_mode else None
    if shared_bin:
        dst = Path(shared_bin) / (bin_dir or "bin") / bin_name
        if not dst.is_file():
            clog = f"prebuilt {dst} missing"
            write_text("cutlass", cid, "compile.txt", clog)
            fail(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
                 shape=shape, kernel=kernel, proof=clog,
                 error=f"missing prebuilt binary {dst}", log=clog, extra=extra)
            return
        clog = (f"prebuilt {dst} sha256={hashlib.sha256(dst.read_bytes()).hexdigest()} "
                f"mtime={datetime.fromtimestamp(dst.stat().st_mtime, timezone.utc).isoformat()}")
        rc = 0
    else:
        rc, clog, dst = compile_one(
            bin_name, src, defs=defs, bin_dir=bin_dir)
    write_text("cutlass", cid, "compile.txt", clog)
    if rc != 0 or not dst.exists():
        fail(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
             shape=shape, kernel=kernel, proof=clog[-4000:], error="compile failed", log=clog[-8000:],
             extra=extra)
        return
    timeout_s = (int(os.environ.get(
        "BW_CUTLASS_RUN_TIMEOUT", os.environ.get("TIMEOUT", "1200")))
                 if full_mode else 600)
    try:
        rrc, rout = run([str(dst), *args], timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or error.output or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        write_text("cutlass", cid, "run.txt", output)
        fail(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
             shape=shape, kernel=kernel, proof=clog,
             error=f"timeout after {timeout_s} s", log=output, extra=extra)
        return
    if full_mode and "This example requires" in rout:
        write_text("cutlass", cid, "run.txt", rout)
        fail(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
             shape=shape, kernel=kernel, proof=clog,
             error="example device gate rejected CC 10.3", log=rout,
             extra=extra)
        return
    sass = sass_hits(dst)
    proof = f"compile={' '.join(NVCC+[str(src)])}\nCluster launched / args={args}\n{sass}\n{rout[-3000:]}"
    if cta_group == 2 and "cta_group::2" not in sass.lower():
        fail(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
             shape=shape, kernel=kernel, proof=proof,
             error="no cta_group::2 in SASS/PTX", log=rout, extra=extra)
        return
    if rrc != 0:
        fail(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
             shape=shape, kernel=kernel,
             proof=proof + "\nCLC fail-path: 1-tile/pinned cluster; PipelineTransactionAsync Stages_=0",
             error=f"run rc={rrc}", log=rout, extra=extra)
        return
    ok(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
       shape=shape, kernel=kernel, proof=proof, log=rout, extra=extra)


FA4_PREFIX = ROOT / ".deps" / "fa4"


def fa4_pythonpath() -> list[str]:
    """FA4-prefix packages whose cutlass has cutlass.testing (needed by some
    spec DSL scripts); mirrors suites/flashattention/run.py site_packages()."""
    pypaths: list[str] = []
    for pat in (
        str(FA4_PREFIX / "lib" / "python*" / "site-packages"),
        str(FA4_PREFIX / "lib" / "python*" / "dist-packages"),
        str(FA4_PREFIX / "local" / "lib" / "python*" / "site-packages"),
        str(FA4_PREFIX / "local" / "lib" / "python*" / "dist-packages"),
    ):
        pypaths.extend(sorted(glob.glob(pat)))
    subs: list[str] = []
    for p in pypaths:
        for s in ("nvidia_cutlass_dsl/dsl_packages", "nvidia_cutlass_dsl/python_packages"):
            subs.extend(sorted(glob.glob(os.path.join(p, s))))
    return subs + pypaths


def case_dsl(cid, *, script, args, cta_group, cluster, sms, dtype, shape, kernel, extra=None):
    script = Path(script)
    if not script.exists():
        fail(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
             shape=shape, kernel=kernel, error=f"missing {script}", extra=extra)
        return
    dump = ART / "cutlass" / cid / "dsl_dump"
    dump.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), *args]
    env = os.environ.copy()
    # Honor campaign/dump-only env: overwriting DUMP_DIR sent cubin to
    # corpus/cutlass/<id>/dsl_dump, and KEEP_SASS ICEs
    # ("Cannot dump SASS: CUBIN file does not exist").
    if not env.get("CUTE_DSL_DUMP_DIR"):
        env["CUTE_DSL_DUMP_DIR"] = str(dump)
    if not env.get("CUTE_DSL_KEEP"):
        env["CUTE_DSL_KEEP"] = "cubin,ptx"
    if extra is not None:
        # Spec cases: some scripts import cutlass.testing, only present in the
        # FA4-prefix DSL; legacy cases keep their env byte-identical.
        fa4 = fa4_pythonpath()
        if fa4:
            env["PYTHONPATH"] = os.pathsep.join(fa4 + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        if extra.get("mode") == "full":
            if not env.get("CUTE_DSL_CACHE_DIR"):
                env["CUTE_DSL_CACHE_DIR"] = str(dump.parent / "cute_cache")
            if not env.get("CUTE_DSL_NO_CACHE"):
                env["CUTE_DSL_NO_CACHE"] = "1"
            if extra.get("single_cluster"):
                hook = str(Path(__file__).resolve().parent
                           / "single_cluster_hook")
                env["BW_SINGLE_CLUSTER_MAX_CLUSTERS"] = "1"
                env["PYTHONPATH"] = os.pathsep.join(
                    [hook, env.get("PYTHONPATH", "")])
    effective_dump = Path(env.get("CUTE_DSL_DUMP_DIR") or dump)
    effective_dump.mkdir(parents=True, exist_ok=True)
    rrc, rout = run(cmd, env=env, timeout=900)
    if extra and extra.get("mode") == "full":
        flag = (f"explicit full-mode variant={extra['cluster_variant']}"
                if extra.get("cluster_variant") else
                "library default cluster heuristic")
    else:
        flag = f"use_2cta_instrs={'True' if cta_group == 2 else 'False'}"
    sass = ""
    sass_excerpt = ""
    for p in effective_dump.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in {".ptx", ".sass"}:
            text = p.read_text(errors="replace")
            sass += text
            sass_excerpt += text[:8000]
        elif suffix in {".cubin", ".so"} or "sass" in p.name.lower():
            text = sass_hits(p)
            sass += text
            sass_excerpt += text[:8000]
    proof_tokens = [token for token in (
        "cta_group::2", "knummulticast=2", "use_2cta_instrs=true",
    ) if token in sass.lower()]
    proof = (f"{flag} script={script}\ndump_dir={effective_dump} "
             f"keep={env.get('CUTE_DSL_KEEP')}\ncmd={' '.join(cmd)}\n"
             f"compiled_proof_tokens={','.join(proof_tokens)}\n"
             f"{sass_excerpt}\n{rout[-4000:]}")
    write_text("cutlass", cid, "run.txt", rout)
    if rrc != 0:
        fail(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
             shape=shape, kernel=kernel, proof=proof,
             error=f"dsl rc={rrc}", log=rout, extra=extra)
        return
    if cta_group == 2 and not proof_ok_2c(sass):
        fail(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
             shape=shape, kernel=kernel, proof=proof,
             error="2CTA without cta_group::2 / kNumMulticast=2 / use_2cta_instrs=True",
             log=rout, extra=extra)
        return
    ok(cid, cta_group=cta_group, cluster=cluster, sms=sms, dtype=dtype,
       shape=shape, kernel=kernel, proof=proof, log=rout, extra=extra)


def _set_cli_option(args: list[str], option: str, value: str | None) -> list[str]:
    out = []
    found = False
    index = 0
    while index < len(args):
        arg = str(args[index])
        if arg == option:
            found = True
            if value is not None:
                out.extend((option, value))
            index += 2
        elif arg.startswith(option + "="):
            found = True
            if value is not None:
                out.append(f"{option}={value}")
            index += 1
        else:
            out.append(arg)
            index += 1
    if value is not None and not found:
        out.extend((option, value))
    return out


def _cpp_equal_options(args: list[str]) -> list[str]:
    out = []
    index = 0
    while index < len(args):
        arg = str(args[index])
        if (arg.startswith("--") and "=" not in arg
                and index + 1 < len(args)
                and not str(args[index + 1]).startswith("--")):
            out.append(f"{arg}={args[index + 1]}")
            index += 2
        else:
            out.append(arg)
            index += 1
    return out


def _full_mode_args(case: dict) -> list[str]:
    args = [str(arg) for arg in case.get("args") or []]
    calls = int(case.get("calls", 4 if case.get("runner") == "cpp" else 5))
    if calls < 1:
        raise ValueError("full-mode calls must be positive")
    if not case.get("single_cluster"):
        for option in ("--max_active_clusters", "--max_sm_count"):
            args = _set_cli_option(args, option, None)
    args = [arg for arg in args
            if arg not in {"--skip_ref_check", "--without_verify"}]
    if case.get("runner") == "cpp":
        iterations = int(case.get("full_iterations", max(calls - 1, 0)))
        return _cpp_equal_options(
            _set_cli_option(args, "--iterations", str(iterations)))
    args = _set_cli_option(args, "--warmup_iterations", str(max(calls - 2, 0)))
    return _set_cli_option(args, "--iterations", "1")


def _full_cluster_meta(args: list[str]) -> tuple[int | None, list[int] | None, str | None]:
    def value(option):
        for index, arg in enumerate(args):
            if arg == option and index + 1 < len(args):
                return args[index + 1]
            if arg.startswith(option + "="):
                return arg.split("=", 1)[1]
        return None

    use_2cta = "--use_2cta_instrs" in args
    mnk = value("--cluster_shape_mnk")
    mn = value("--cluster_shape_mn")
    if mnk is not None:
        dims = [int(item) for item in mnk.strip("()[]").split(",")]
        cluster = dims[:3]
    elif mn is not None:
        dims = [int(item) for item in mn.strip("()[]").split(",")]
        cluster = [*dims[:2], 1]
    else:
        cluster_m, cluster_n = value("--cluster_m"), value("--cluster_n")
        cluster = ([int(cluster_m or 1), int(cluster_n or 1), 1]
                   if cluster_m is not None or cluster_n is not None else None)
    public_preference = any(arg.startswith((
        "--preferred_cluster_", "--fallback_cluster_")) for arg in args)
    variant = None
    if use_2cta:
        variant = f"use_2cta_instrs cluster={cluster}"
    elif cluster is not None:
        variant = f"cluster_shape={cluster}"
    elif public_preference:
        variant = "preferred/fallback cluster heuristic"
    return (2 if use_2cta else (1 if cluster is not None else None),
            cluster, variant)


def run_spec_case(c: dict):
    full_mode = c.get("mode") == "full"
    extra = {"family": c.get("family"), "est_cycles": c.get("est_cycles"),
             "est_basis": c.get("est_basis")}
    if full_mode:
        default_calls = 4 if c.get("runner") == "cpp" else 5
        extra.update({"mode": "full",
                      "calls": int(c.get("calls", default_calls)),
                      "single_cluster": bool(c.get("single_cluster"))})
    meta = dict(
        cta_group=(None if full_mode else c.get("cta_group", 1)),
        cluster=(None if full_mode else c.get("cluster")),
        sms=(148 if full_mode else c.get("sms")),
        dtype=c.get("dtype"), shape=c.get("shape"), kernel=c.get("kernel"), extra=extra,
    )
    try:
        args = _full_mode_args(c) if full_mode else list(c.get("args") or [])
        if full_mode:
            cta_group, cluster, cluster_variant = _full_cluster_meta(args)
            meta.update({"cta_group": cta_group, "cluster": cluster})
            if cluster_variant:
                extra["cluster_variant"] = cluster_variant
        if c.get("runner") == "cpp":
            case_cpp(c["id"], c["bin"], ROOT / c["src"], args, defs=tuple(c.get("defs", ())),
                     bin_dir=c.get("bin_dir"), **meta)
        else:
            case_dsl(c["id"], script=ROOT / c["script"], args=args, **meta)
    except Exception as e:  # noqa: BLE001
        import traceback
        fail(c["id"], proof=traceback.format_exc()[-2000:],
             error=f"{type(e).__name__}: {e}", **meta)


def main():
    argv = sys.argv[1:]
    sc_by_id: dict[str, dict] = {c["id"]: c for c in load_sc_cases()}

    if "--list" in argv:
        for cid in sc_by_id:
            print(cid)
        return 0

    if "--delivery300sc" in argv:
        want = list(sc_by_id)
    else:
        want = argv

    for cid in want:
        print(f"=== {cid} ===", flush=True)
        if cid in sc_by_id:
            run_spec_case(sc_by_id[cid])
        else:
            print(f"unknown case id: {cid}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
