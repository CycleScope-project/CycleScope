"""cyclescope command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "configs" / "mechanism_tree"
DEFAULT_OUT = REPO_ROOT / "configs" / "trees"
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "schema_registry.json"
DEFAULT_CPP_HEADER = REPO_ROOT / "cpp" / "core" / "generated" / "parameter_schema.hpp"
DEFAULT_PY_SCHEMA = REPO_ROOT / "python" / "cyclescope" / "tree" / "generated_schema.py"

DEVICE_ARCH = {"h100": "hopper", "b300": "blackwell"}


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="mechanism tree source root")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                        help="schema registry path")


def default_configs_dir() -> str:
    env = os.environ.get("CYCLESCOPE_CONFIGS")
    if env:
        return env
    return str(REPO_ROOT / "configs")


def _devices(value: str) -> list[str]:
    if value == "all":
        return list(DEVICE_ARCH)
    return [value]


def _cmd_tree_build(args: argparse.Namespace) -> int:
    from cyclescope.tree.build import (
        build_tree, emit_cpp_header, emit_python_schema, write_tree)
    from cyclescope.tree.registry import load_registry
    from cyclescope.tree.validate import validate_generated_tree

    for device in _devices(args.device):
        arch = DEVICE_ARCH[device]
        tree = build_tree(args.source, arch, args.registry)
        path = write_tree(tree, args.out)
        validate_generated_tree(tree)
        counts = tree["status_counts"]
        print(f"{device}: {tree['leaf_count']} leaves "
              f"({', '.join(f'{k}={v}' for k, v in counts.items())}) "
              f"-> {path}")
    if args.emit_cpp:
        registry = load_registry(args.registry)
        header = emit_cpp_header(registry, DEFAULT_CPP_HEADER)
        print(f"cpp header -> {header}")
        schema = emit_python_schema(registry, DEFAULT_PY_SCHEMA)
        print(f"python schema -> {schema}")
    return 0


def _cmd_tree_validate(args: argparse.Namespace) -> int:
    from cyclescope.tree.build import build_tree
    from cyclescope.tree.validate import validate_generated_tree

    from cyclescope.tree.source import load_source_tree

    for device in _devices(args.device):
        arch = DEVICE_ARCH[device]
        tree = build_tree(args.source, arch, args.registry)
        validate_generated_tree(tree)
        print(f"{device}: ok ({tree['leaf_count']} leaves, "
              f"digest {tree['content_digest'][:16]})")
        # common definitions: device subtrees are transitional shape
        # differences (mechanism-merge-plan §3); report them
        stats = load_source_tree(args.source, arch).device_stats
        subtrees = {m: s["device_subtrees"] for m, s in stats.items()
                    if s.get("device_subtrees")}
        print(f"  common mechanisms {len(stats)}, device subtrees "
              f"{sum(subtrees.values())}"
              + (f": {subtrees}" if subtrees else ""))
    # a device subtree is only allowed under a class that does not exist
    # on every other device (ISA-only instruction, e.g. LDTM / UTC*MMA)
    import json
    from pathlib import Path
    leaves = {}
    for arch in set(DEVICE_ARCH.values()):
        leaves[arch] = {}
        for f in (Path(args.source) / arch / "leaves").glob("*.json"):
            leaves[arch].update(json.loads(f.read_text()))
    bad = []
    for arch, entries in leaves.items():
        for node, entry in entries.items():
            if "children" not in entry:
                continue
            for other, oe in leaves.items():
                if other != arch and not oe.get(node, {}).get("unsupported"):
                    bad.append((arch, node, other))
    if bad:
        for arch, node, other in bad:
            print(f"error: device subtree {node} on {arch} but the node is "
                  f"supported on {other}", file=sys.stderr)
        return 1
    # leaf conventions (mechanism-leaf-alignment-plan.md §2): warnings
    # until the alignment packages land, errors with --strict
    from cyclescope.tree.conventions import check_conventions, summary
    viol = check_conventions(args.source)
    total = sum(len(v) for v in viol.values())
    print(f"  leaf conventions: {summary(viol)}")
    if args.verbose:
        for kind, items in viol.items():
            for line in items:
                print(f"    {kind}: {line}")
    if total and args.strict:
        print(f"error: {total} leaf convention violations", file=sys.stderr)
        return 1
    return 0


def _cmd_tree_diff(args: argparse.Namespace) -> int:
    from cyclescope.tree.diff import diff_files, format_diff

    diff = diff_files(args.tree_a, args.tree_b,
                      leaf_values=args.leaf_values)
    print(format_diff(diff))
    return 0


def _cmd_tree_show(args: argparse.Namespace) -> int:
    tree = json.loads(args.tree.read_text())
    nodes = {n["id"]: n for n in tree["nodes"]}
    root_id = f"{args.mechanism}.root"
    if root_id not in nodes:
        print(f"mechanism {args.mechanism!r} not in {args.tree}",
              file=sys.stderr)
        return 1

    def visit(node_id: str, depth: int) -> None:
        node = nodes[node_id]
        indent = "  " * depth
        if "parameter" in node or node.get("unsupported"):
            if node.get("unsupported"):
                print(f"{indent}{node['guard']}: UNSUPPORTED")
            else:
                fields = node["parameter"]["fields"]
                print(f"{indent}{node['guard']}: "
                      f"status={node['parameter']['status']}")
                for name, fdef in fields.items():
                    print(f"{indent}  {name} = "
                          f"{json.dumps(fdef['model'], sort_keys=True)}")
            return
        print(f"{indent}{node.get('guard') or 'root'}: "
              f"partition {node['partition_axis']}")
        for child in node["children"]:
            visit(child, depth + 1)

    visit(root_id, 0)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyclescope", description="cyclescope toolchain")
    sub = parser.add_subparsers(dest="group", required=True)

    tree = sub.add_parser("tree", help="mechanism tree commands")
    tree_sub = tree.add_subparsers(dest="command", required=True)

    p_build = tree_sub.add_parser("build", help="generate trees")
    p_build.add_argument("--device", default="all",
                         choices=["h100", "b300", "all"])
    p_build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_build.add_argument("--emit-cpp", action="store_true",
                         help="also write parameter_schema.hpp and "
                              "generated_schema.py")
    _add_source_args(p_build)
    p_build.set_defaults(func=_cmd_tree_build)

    p_val = tree_sub.add_parser("validate", help="build + check trees")
    p_val.add_argument("--strict", action="store_true",
                       help="leaf convention violations are errors")
    p_val.add_argument("--verbose", action="store_true",
                       help="list every leaf convention violation")
    p_val.add_argument("--device", default="all",
                       choices=["h100", "b300", "all"])
    _add_source_args(p_val)
    p_val.set_defaults(func=_cmd_tree_validate)

    p_cmp = tree_sub.add_parser("compile",
        help="compile tree -> GSIMPRG01 program tables")
    p_cmp.add_argument("--tree", type=Path, required=True)
    p_cmp.add_argument("--out", type=Path, required=True)
    p_cmp.set_defaults(func=_cmd_tree_compile)
    p_diff = tree_sub.add_parser("diff", help="diff two generated trees")
    p_diff.add_argument("tree_a", type=Path)
    p_diff.add_argument("tree_b", type=Path)
    p_diff.add_argument("--leaf-values", action="store_true", default=True)
    p_diff.set_defaults(func=_cmd_tree_diff)

    p_show = tree_sub.add_parser("show", help="print one mechanism")
    p_show.add_argument("tree", type=Path)
    p_show.add_argument("mechanism")
    p_show.set_defaults(func=_cmd_tree_show)

    p_conv = sub.add_parser(
        "convert", help="capture -> facts, facts -> segments")
    p_conv.add_argument("--facts", type=Path, default=None)
    p_conv.add_argument("--capture", type=Path, default=None,
                        help="capture case_dir -> GSIMCAP01 facts")
    p_conv.add_argument("--launch", type=int, default=None)
    p_conv.add_argument("--clusters", default="all")
    p_conv.add_argument("--shard", default=None,
                        help="k/N shard of cluster linear ids")
    p_conv.add_argument("--split-dir", type=Path, default=None,
                        help="dir with cyclescope-trace-split output; "
                             "with --shard reads only this shard's "
                             "pre-split records")
    p_conv.add_argument("--split-shards", type=int, default=None,
                        help="run cyclescope-trace-split on the traced "
                             "launch into --out, then exit")
    p_conv.add_argument("--reference", type=Path, default=None)
    p_conv.add_argument("--placement", type=Path, default=None)
    p_conv.add_argument("--builder", choices=["cpp", "python"],
                        default=None,
                        help="facts->segments builder (default cpp); "
                             "with --capture, capture->facts builder "
                             "(default cpp; --builder python keeps the "
                             "reference path)")
    p_conv.add_argument("--arch", choices=["hopper", "blackwell"],
                        required=True)
    p_conv.add_argument("--tree", type=Path, default=None)
    p_conv.add_argument("--out", type=Path, required=True)
    p_conv.set_defaults(func=_cmd_convert)

    p_job = sub.add_parser(
        "build", help="segments + launch.json -> GSIMJOB container")
    p_job.add_argument("dir", type=Path)
    p_job.add_argument("--out", type=Path, required=True)
    p_job.set_defaults(func=_cmd_build)

    p_re = sub.add_parser("relink", help="relink seg onto a new tree")
    p_re.add_argument("seg", type=Path)
    p_re.add_argument("--tree", type=Path, required=True)
    p_re.set_defaults(func=_cmd_relink)

    p_conv.add_argument("--cache", nargs="?", const="", default=None,
                        help="segment cache dir (default "
                             "$CYCLESCOPE_ARTIFACT_ROOT/cache/segments; "
                             "on unless --no-cache)")
    p_conv.add_argument("--no-cache", action="store_true",
                        help="disable the segment cache")
    p_conv.add_argument("--reference-sm-clock-khz", type=int,
                        default=None,
                        help="reference SM clock fact (source: "
                             "command_line)")
    p_conv.add_argument("--reference-cache-control",
                        choices=["all"], default=None,
                        help="reference cache-warmth control")
    p_job.add_argument("--reference-sm-clock-khz", type=int,
                       default=None,
                       help="override launch.json reference clock")
    p_job.add_argument("--reference-cache-control",
                       choices=["all"], default=None,
                       help="override launch.json cache control")
    p_job.add_argument("--keep-segments", action="store_true",
                       help="keep the segments dir after build")

    p_sp = sub.add_parser("split", help="job.bin -> segments dir")
    p_sp.add_argument("job", type=Path)
    p_sp.add_argument("--out", type=Path, required=True)
    p_sp.set_defaults(func=_cmd_split)
    p_cap = sub.add_parser("capture",
        help="run a suite case under the tracer")
    p_cap.add_argument("--suite", required=True,
        help="hopper/full_gpu|blackwell/full_gpu")
    p_cap.add_argument("--case", required=True)
    p_cap.add_argument("--out", type=Path, required=True)
    p_cap.add_argument("--mode", choices=["trace", "placement", "both"],
                       default="both")
    p_cap.add_argument("--ncu-repeats", type=int, default=0)
    p_cap.add_argument("--tracer", type=Path, default=None)
    p_cap.add_argument("--timeout", type=int, default=1800)
    p_cap.add_argument("--dry-run", action="store_true")
    p_cap.add_argument("--reference", type=Path, default=None,
        help="cyclescope reference output root (<root>/<case>/<case>/"
             "timing.json): trace the launch that reference timed "
             "instead of re-picking")
    p_cap.set_defaults(func=_cmd_capture)
    p_ref = sub.add_parser("reference",
        help="ncu timing repeats -> timing.json + reference.tsv")
    p_ref.add_argument("--suite", required=True)
    p_ref.add_argument("--cases", required=True,
        help="case id, comma list, 'all', or a cases-file path")
    p_ref.add_argument("--repeats", type=int, default=3)
    p_ref.add_argument("--out", type=Path, required=True)
    p_ref.add_argument("--timeout", type=int, default=1800)
    p_ref.add_argument("--triton-autotune", type=Path, default=None,
        help="pinned triton_autotune.json copied into each case workdir "
             "instead of running the autotune prepass (keeps reference "
             "and capture on the same pinned config)")
    p_ref.add_argument("--captures", type=Path, default=None,
        help="capture root (<root>/<case>): time the launch each capture "
             "traced instead of re-picking")
    p_ref.add_argument("--metrics", default=None,
        help="extra ncu metrics, comma-separated; the launch/timing "
             "metrics are unchanged — the raw ncu csv of repeat 0 is "
             "also saved as <out>/<case>/ncu_metrics.csv")
    p_ref.add_argument("--source-stalls", action="store_true",
        help="extra ncu SourceCounters pass on the picked launch; raw "
             "SASS-level stall sampling saved as <out>/<case>/"
             "ncu_source.csv")
    p_ref.set_defaults(func=_cmd_reference)

    p_run = sub.add_parser(
        "run", help="one case: capture -> convert -> build -> simulate")
    p_run.add_argument("--suite", required=True)
    p_run.add_argument("--case", required=True)
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--captures", type=Path, default=None,
        help="existing capture root; GPU stages are skipped")
    p_run.add_argument("--reference-tsv", type=Path, default=None)
    p_run.add_argument("--skip-reference", action="store_true")
    p_run.add_argument("--tree", type=Path, default=None)
    p_run.add_argument("--engine", type=str, default=None)
    p_run.add_argument("--builder", choices=["cpp", "python"],
                       default="cpp")
    p_run.add_argument("--jobs", type=int, default=None)
    p_run.add_argument("--shard-records", type=int, default=1_500_000)
    p_run.add_argument("--shard-max", type=int, default=16)
    p_run.add_argument("--no-placement", action="store_true")
    p_run.add_argument("--keep", action="store_true")
    p_run.add_argument("--ledgers", action="store_true")
    p_run.add_argument("--repeats", type=int, default=3)
    p_run.add_argument("--timeout", type=int, default=5400)
    p_run.add_argument("--tracer", type=Path, default=None)
    p_run.add_argument("--scratch", type=Path, default=None)
    p_run.set_defaults(func=_cmd_run)

    p_rep = sub.add_parser("report",
        help="sim/<case>/status.json rows vs the expected tables")
    p_rep.add_argument("run_out", type=Path, nargs="*")
    p_rep.add_argument("--expected", type=Path, action="append",
                       default=None)
    p_rep.add_argument("--tolerance", type=float, default=3.0)
    p_rep.add_argument("--out", type=Path, default=None)
    p_rep.add_argument("--baseline", type=Path, default=None,
        help="print the accuracy summary of a shipped expected table")
    p_rep.set_defaults(func=_cmd_report)

    return parser


def _artifact_root() -> str:
    from cyclescope import site
    return str(site.artifact_root())


def _cmd_convert(args: argparse.Namespace) -> int:
    cache = None
    if not args.no_cache:
        cache = args.cache or f"{_artifact_root()}/cache/segments"
    if getattr(args, "capture", None):
        if getattr(args, "split_shards", None):
            from cyclescope.convert.capture_pipeline import split_capture
            split_capture(args.capture, args.split_shards, args.out)
            return 0
        shard = None
        if args.shard:
            k, n = args.shard.split("/")
            shard = (int(k), int(n))
        facts_out = args.out
        if (getattr(args, "builder", None) == "cpp"
                or args.builder is None):
            import shutil, subprocess
            binary = (shutil.which("cyclescope-capture-facts")
                      or os.environ.get("CYCLESCOPE_CAPTURE_FACTS"))
            if binary is None:
                print("error: cyclescope-capture-facts binary not on PATH "
                      "(set CYCLESCOPE_CAPTURE_FACTS)", file=sys.stderr)
                return 2
            cmd = [binary, "--case", str(args.capture),
                   "--arch", args.arch,
                   "--out", str(facts_out)]
            if args.launch is not None:
                cmd += ["--launch", str(args.launch)]
            if args.clusters is not None:
                cmd += ["--clusters", str(args.clusters)]
            if shard is not None:
                cmd += ["--shard", f"{shard[0]}/{shard[1]}"]
            if getattr(args, "split_dir", None):
                cmd += ["--split-dir", str(args.split_dir)]
            if args.reference is not None:
                cmd += ["--reference", str(args.reference)]
            if args.placement is not None:
                cmd += ["--placement", str(args.placement)]
            configs = default_configs_dir()
            cmd += ["--configs", configs]
            print("+", " ".join(cmd))
            return subprocess.call(cmd)
        from cyclescope.convert.capture_pipeline import convert_capture
        convert_capture(args.capture, args.arch, facts_out,
                        launch=args.launch, clusters=args.clusters,
                        shard=shard, reference=args.reference,
                        placement=args.placement,
                        cache=not args.no_cache,
                        split_dir=args.split_dir)
        import json as _json
        print(_json.dumps({"facts_dir": str(facts_out)}, indent=1))
        return 0
    if not getattr(args, "facts", None):
        print("error: one of --capture / --facts is required",
              file=sys.stderr)
        return 2
    if args.tree is None:
        print("error: --tree is required for --facts",
              file=sys.stderr)
        return 2
    from cyclescope.convert.pipeline import convert_facts
    result = convert_facts(args.facts, args.arch, args.tree,
                           args.out, cache_dir=cache,
                           builder=args.builder or "cpp")
    if (args.reference_sm_clock_khz is not None or
            args.reference_cache_control is not None):
        import json
        lp = args.out / "launch.json"
        doc = json.loads(lp.read_text())
        if args.reference_sm_clock_khz is not None:
            doc["reference_sm_clock_khz"] = (
                args.reference_sm_clock_khz)
        if args.reference_cache_control is not None:
            doc["reference_cache_control"] = {
                "all": 2}[args.reference_cache_control]
        doc["reference_source"] = "command_line"
        lp.write_text(json.dumps(doc))
    print(f"{len(result['segments'])} segments -> {args.out}")
    return 0


def _cmd_relink(args: argparse.Namespace) -> int:
    from cyclescope.job.relink import relink_file
    out = relink_file(args.seg, args.tree)
    args.seg.write_bytes(out)
    print(f"relinked -> {args.seg}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from cyclescope.job.container import build_container_from_dir

    if (args.reference_sm_clock_khz is not None or
            args.reference_cache_control is not None):
        import json
        lp = args.dir / "launch.json"
        doc = json.loads(lp.read_text())
        if args.reference_sm_clock_khz is not None:
            doc["reference_sm_clock_khz"] = (
                args.reference_sm_clock_khz)
        if args.reference_cache_control is not None:
            doc["reference_cache_control"] = {
                "all": 2}[args.reference_cache_control]
        doc["reference_source"] = "command_line"
        lp.write_text(json.dumps(doc))
    path = build_container_from_dir(args.dir, args.out)
    if not args.keep_segments:
        import shutil
        shutil.rmtree(args.dir / "segments", ignore_errors=True)
    print(f"container -> {path}")
    return 0


def _cmd_tree_compile(args) -> int:
    """GSIMPRG01: compiled program tables + SCHM table per spec."""
    import json
    from cyclescope.job import program_tables
    from cyclescope.tree.registry import load_registry
    tree = json.loads(args.tree.read_text())
    reg = load_registry(Path(__file__).resolve().parents[2] / "configs" /
                        "schema_registry.json")
    args.out.write_bytes(program_tables.program_file_image(tree, reg))
    print(f"program tables -> {args.out}")
    return 0


def _cmd_capture(args) -> int:
    from .capture.driver import capture_case
    try:
        rec = capture_case(args.suite, args.case, args.out,
                           mode=args.mode, ncu_repeats=args.ncu_repeats,
                           tracer=args.tracer, timeout_s=args.timeout,
                           dry_run=args.dry_run, reference=args.reference)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not args.dry_run and rec.get("trace_check_rc") not in (0, None):
        return 1
    if not args.dry_run and rec.get("rc", 0) != 0:
        return 1
    return 0


def _cmd_reference(args) -> int:
    from .reference.ncu_runner import run_reference
    spec = args.cases
    if spec == "all":
        arch, sset = args.suite.split("/", 1)
        if arch == "hopper":
            import csv as _csv
            tsv = Path(__file__).resolve().parents[2] / "suites" / arch                 / sset / "cases.tsv"
            case_ids = [r["case_id"] for r in
                        _csv.DictReader(tsv.open(), delimiter="	")]
        else:
            txt = Path(__file__).resolve().parents[2] / "suites" / arch                 / sset / "cases.txt"
            case_ids = [l.split()[-1] for l in
                        txt.read_text().splitlines() if l.strip()]
    elif Path(spec).is_file():
        case_ids = [l.split()[-1] for l in
                    Path(spec).read_text().splitlines() if l.strip()]
    else:
        case_ids = [c for c in spec.split(",") if c]
    try:
        rec = run_reference(args.suite, case_ids, args.out,
                            repeats=args.repeats, timeout_s=args.timeout,
                            triton_autotune=args.triton_autotune,
                            captures=args.captures,
                            metrics=args.metrics,
                            source_stalls=args.source_stalls)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(rec, default=str))
    return 0 if all(s == "success" for s in rec["statuses"]) else 1


def _cmd_run(args) -> int:
    from .pipeline.run_case import run_case
    try:
        status = run_case(args.suite, args.case, args.out,
                          captures=args.captures,
                          reference_tsv=args.reference_tsv,
                          skip_reference=args.skip_reference,
                          tree=args.tree, engine=args.engine,
                          builder=args.builder, jobs=args.jobs,
                          shard_records=args.shard_records,
                          shard_max=args.shard_max,
                          placement=not args.no_placement,
                          keep=args.keep, ledgers=args.ledgers,
                          repeats=args.repeats, timeout_s=args.timeout,
                          tracer=args.tracer, scratch=args.scratch)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(status, indent=1, default=str))
    return 0 if (status["stage"] == "done"
                 and status["class"] == "ok") else 1


def _cmd_report(args) -> int:
    from .pipeline import report as _report
    if args.baseline is not None:
        print(_report.baseline_summary(args.baseline), end="")
        return 0
    if not args.run_out:
        print("error: report needs at least one RUN_OUT (or --baseline)",
              file=sys.stderr)
        return 2
    return _report.report(args.run_out, expected=args.expected,
                          tolerance=args.tolerance, out=args.out)


def _cmd_split(args) -> int:
    from cyclescope.io.file_header import FileReader
    from cyclescope.job import container as cf
    data = args.job.read_bytes()
    segs = cf.segment_images(FileReader(memoryview(data)))
    args.out.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(segs):
        (args.out / f"{i:06d}.seg").write_bytes(s)
    reader = FileReader(memoryview(data))
    if getattr(reader, "version", 1) >= 2:
        from cyclescope.job import segment_format as _sf
        (args.out / "program.tables").write_bytes(
            _sf.program_tables_payload(
                {t: bytes(reader.require_table(t))
                 for t in _sf.PROGRAM_TAGS}))
    print(f"{len(segs)} segments -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not DEFAULT_REGISTRY.is_file():
        print("error: cyclescope runs from a source checkout (configs/, "
              "suites/ and scripts/ sit next to the package): git clone "
              "the repository and `pip install -e .` there; "
              f"{DEFAULT_REGISTRY} not found", file=sys.stderr)
        return 2
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI surface
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
