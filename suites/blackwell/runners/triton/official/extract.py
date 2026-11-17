#!/usr/bin/env python3
"""Extract import-safe, source-identical definitions from Triton tutorials."""
from __future__ import annotations

import ast
import difflib
import os
from pathlib import Path

SOURCE = Path(os.environ.get(
    "TRITON_SOURCE_ROOT",
    str(Path(os.environ["CYCLESCOPE_ARTIFACT_ROOT"]) / "tools" / "triton-source"),
))
DEST = Path(__file__).resolve().parent
TAG = "v3.7.0"
COMMIT = "5f3f125e8f63c24613f1f73b937442864f263f94"


def _keep(node: ast.stmt) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name.startswith(("benchmark", "bench_", "test_"))):
        return False
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                         ast.AsyncFunctionDef, ast.ClassDef)):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return isinstance(node.value.value, str)
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = node.value
        if value is None:
            return False
        try:
            ast.literal_eval(value)
            return True
        except (ValueError, TypeError):
            return False
    return False


def _assigned_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        targets = []
        if isinstance(child, ast.Assign):
            targets = child.targets
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [child.target]
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            targets = [child.target]
        for target in targets:
            names.update(
                item.id for item in ast.walk(target)
                if isinstance(item, ast.Name))
    return names


def _loaded_names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}


def extract_module(source: Path) -> tuple[str, list[str]]:
    text = source.read_text()
    tree = ast.parse(text, filename=str(source))
    lines = text.splitlines(keepends=True)
    snippets = []
    stripped = []
    keep_nodes = [_keep(node) for node in tree.body]
    needed_names = set()
    for node, keep in zip(tree.body, keep_nodes):
        if keep and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
            for decorator in getattr(node, "decorator_list", []):
                needed_names.update(_loaded_names(decorator))
    if source.is_relative_to(SOURCE) and source.relative_to(SOURCE).as_posix() == \
            "python/examples/gluon/01-attention-forward.py":
        needed_names.update({"SharedMemoryChannel", "TensorMemoryChannel"})
    changed = True
    while changed:
        changed = False
        for index, node in enumerate(tree.body):
            if keep_nodes[index] or not (_assigned_names(node) & needed_names):
                continue
            keep_nodes[index] = True
            needed_names.update(_loaded_names(node))
            changed = True
    for node, keep in zip(tree.body, keep_nodes):
        if keep:
            start = min([node.lineno] + [
                decorator.lineno for decorator in
                getattr(node, "decorator_list", [])
            ])
            end = node.end_lineno or node.lineno
            snippets.append("".join(lines[start - 1:end]).rstrip())
        else:
            end = node.end_lineno or node.lineno
            excerpt = lines[node.lineno - 1].strip()[:100]
            stripped.append(f"{node.lineno}-{end}: {excerpt}")
    extracted = "\n\n".join(snippets) + "\n"
    if source.name == "10-block-scaled-matmul.py":
        marker = "    # Flatten scales for cuBLAS\n"
        if marker not in extracted:
            raise ValueError(f"expected cuBLAS scale block in {source}")
        extracted = extracted.replace(
            marker,
            marker + "    a_scale_cublas = None\n"
            "    b_scale_cublas = None\n",
            1,
        )
    return extracted, stripped


def module_name(source: Path) -> str:
    number, _, rest = source.stem.partition("-")
    rest = rest.replace("-", "_")
    return f"tutorial_{number}_{rest}.py"


def main() -> None:
    tutorial_dir = SOURCE / "python" / "tutorials"
    tutorials = list(tutorial_dir.glob("*.py"))
    tutorials.extend((tutorial_dir / "gluon" / name for name in (
        "07-persistence.py",
        "08-warp-specialization.py",
        "09-tma-gather-scatter.py",
        "10-tcgen05-copy.py",
        "11-tcgen05-mma-scaled.py",
    )))
    tutorials.append(SOURCE / "python" / "examples" / "gluon" /
                     "01-attention-forward.py")
    tutorials = sorted(tutorials)
    if not tutorials:
        raise FileNotFoundError(f"no tutorials found in {tutorial_dir}")
    diff_parts = []
    source_lines = [
        f"Triton tag: {TAG}",
        f"Triton commit: {COMMIT}",
        f"Upstream source: {tutorial_dir}",
        "Extraction: retain imports, kernels/helpers/wrappers, module docstrings, and literal constants;",
        "remove top-level device probes, benchmark/test functions, and __main__ code.",
        "",
    ]
    for source in tutorials:
        name = module_name(source)
        extracted, stripped = extract_module(source)
        destination = DEST / name
        destination.write_text(extracted)
        source_lines.append(f"{source.relative_to(SOURCE)} -> {name}")
        if source.name == "10-block-scaled-matmul.py":
            source_lines.append(
                "  adaptation: initialize optional cuBLAS scales for the MXFP4-only path")
        if stripped:
            source_lines.extend(f"  stripped {item}" for item in stripped)
        diff_parts.extend(difflib.unified_diff(
            source.read_text().splitlines(),
            extracted.splitlines(),
            fromfile=str(source.relative_to(SOURCE)),
            tofile=name,
            lineterm="",
        ))
    (DEST / "SOURCE.txt").write_text("\n".join(source_lines) + "\n")
    (DEST / "upstream-extraction.diff").write_text(
        "\n".join(diff_parts) + "\n")


if __name__ == "__main__":
    main()
