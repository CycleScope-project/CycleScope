#!/usr/bin/env python3
"""FamilyAdapter abstraction for the capture v2 pipeline (DESIGN §2).

One adapter per capture family (deepgemm / cutlass / flashattention).  All
case differences flow through adapter code driven by the registry
(``suites/common/capture_registry.json``) and the spec
(``suites/c450_full/cases_<suite>.json``); case-id special casing is
prohibited.  Kernel-name regexes follow the C-line merged registry: the
corpus_sc ``result.json["kernel"]`` token is tried first, registry rows are
the fallback (DESIGN §2).

Read-only dependencies: scripts/trace_campaign.py (semantics reference) and
python/blackwellsim/trace_spool.py (imported only by the campaign driver).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ["CYCLESCOPE_RUNNER_ROOT"])
REGISTRY_PATH = ROOT / "suites" / "common" / "capture_registry.json"
DELIVERY_FLAG = "--delivery300sc"


def _prepare_cute_dsl_dirs(workdir: Path) -> tuple[Path, Path]:
    dump_dir = Path(workdir) / "cute_dump"
    cache_dir = Path(workdir) / "cute_cache"
    dump_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return dump_dir, cache_dir


# Case-id prefix -> suite (family routing, mirrors delivery_tests layout).
SUITE_BY_PREFIX = (("DG", "deepgemm"), ("CU", "cutlass"), ("FA", "flashattention"),
                   ("ML", "flashmla"), ("FI", "flashinfer"), ("TR", "triton"), ("BL", "cublas"))
def _corpus_root(root: Path) -> Path:
    """corpus_sc root: BW_CORPUS_ROOT, the tree-local
    artifacts/corpus_sc link, then BW_CUTLASS_BIN_ROOT's parent."""
    value = os.environ.get("BW_CORPUS_ROOT")
    if value:
        return Path(value)
    local = Path(root) / "artifacts" / "corpus_sc"
    if (local / "cutlass" / "bin").is_dir():
        return local
    bin_root = os.environ.get("BW_CUTLASS_BIN_ROOT")
    if bin_root:
        return Path(bin_root).parent
    return local


# corpus_sc result.json["kernel"] token -> plausible runtime symbol.
# DSL scripts carry "<fn>.py"; the DSL-generated kernel symbol embeds <fn>.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{5,}")
_MAX_TOKENS = 4


def guess_suite(case_id: str) -> str:
    for prefix, suite in SUITE_BY_PREFIX:
        if case_id.startswith(prefix):
            return suite
    return ""


def _torch_cuda_lib() -> Path | None:
    """libtorch_cuda.so of THIS interpreter's torch, located without importing
    torch (importlib.util.find_spec never executes the package).  None when
    torch is absent or the library file does not exist -- absence is recorded
    by the caller, never substituted."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    candidate = Path(list(spec.submodule_search_locations)[0]) / "lib" / "libtorch_cuda.so"
    return candidate if candidate.is_file() else None


def _fa4_prefix_pythonpath(prefix: Path) -> list[str]:
    """Site-packages under an FA4 .deps prefix whose cutlass carries
    cutlass.testing (mirrors suites/cutlass/run.py fa4_pythonpath)."""
    pats = (
        str(prefix / "lib" / "python*" / "site-packages"),
        str(prefix / "lib" / "python*" / "dist-packages"),
        str(prefix / "local" / "lib" / "python*" / "site-packages"),
        str(prefix / "local" / "lib" / "python*" / "dist-packages"),
    )
    out: list[str] = []
    for pat in pats:
        out.extend(sorted(glob.glob(pat)))
        for sub in ("nvidia_cutlass_dsl/dsl_packages", "nvidia_cutlass_dsl/python_packages"):
            out.extend(sorted(glob.glob(os.path.join(pat, sub))))
    return out


def _flashinfer_source() -> Path:
    """Vendored flashinfer source prefix (BW_FI_SRC explicit override,
    else <BW_DEPS_DIR>/flashinfer)."""
    src = os.environ.get("BW_FI_SRC")
    if src:
        return Path(src)
    deps = os.environ.get("BW_DEPS_DIR", "")
    return Path(deps) / "flashinfer" if deps else         Path("/nonexistent")


def _fa4_pythonpath(root: Path) -> list[str]:
    """FA4 site-packages under <root>/.deps/fa4 (harness mirror)."""
    return _fa4_prefix_pythonpath(Path(root) / ".deps" / "fa4")


@dataclass(frozen=True)
class CaptureCase:
    """Merged spec + corpus view of one delivery300sc case."""

    case_id: str
    suite: str
    spec: dict = field(default_factory=dict)    # cases_<suite>.json entry
    corpus: dict = field(default_factory=dict)  # result.json (corpus_sc, legacy fallback)

    @property
    def kind(self) -> str:
        return str(self.spec.get("kind") or self.corpus.get("kind") or "")

    @property
    def runner(self) -> str:
        return str(self.spec.get("runner") or "")

    @property
    def want_sms(self) -> int:
        # audit_single_cluster.want_sms_of semantics: spec sms > cta_group > cta.
        for key in ("sms", "cta_group", "cta"):
            value = self.spec.get(key)
            if value:
                return int(value)
        return 0

    @property
    def multi_cluster_launch(self) -> bool:
        # G2 split exception whitelist lives in the spec (precedent FA4-FWD-SPLIT).
        return bool(self.spec.get("multi_cluster_launch"))


@dataclass(frozen=True)
class RegexChoice:
    pattern: str
    source: str


@dataclass(frozen=True)
class SassSource:
    path: Path
    kind: str          # nvcc_bin | dg_jit_cubin | dsl_cubin | none
    exists: bool
    note: str


@dataclass(frozen=True)
class Resolve:
    """S0 product: everything S1-S3 need for one case."""

    case: CaptureCase
    argv: list[str]
    env: dict
    regex_nvbit: RegexChoice
    regex_ncu: RegexChoice
    regex_ncu_registry: RegexChoice | None
    regex_nvbit_fallbacks: list[RegexChoice]
    main_regex: RegexChoice
    sass: SassSource
    budgets: dict
    want_sms: int
    workdir: Path
    # The adapter instance that produced this Resolve: the prepared-asset
    # facts (probe binary, driver-JIT image, JIT dump dirs) live HERE, so
    # every later stage re-reads assets from the same configured input instead
    # of re-deriving an adapter from the suite registry (which loses the
    # entry's binary/driver-image wiring).  No case-name branch, no global.
    adapter: "FamilyAdapter | None" = None


class Registry:
    """Loaded view of suites/common/capture_registry.json."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or REGISTRY_PATH)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    # -- row access ---------------------------------------------------------
    def suite_row(self, suite: str) -> dict:
        row = self.data.get("suites", {}).get(suite)
        if row is None:
            raise KeyError(f"capture registry has no suite row for {suite!r}")
        return row

    @property
    def aux_regex(self) -> str:
        return str(self.data.get("aux_regex") or "")

    def budgets(self, suite: str) -> dict:
        merged = dict(self.data.get("budgets_default") or {})
        merged.update(self.suite_row(suite).get("budgets") or {})
        return merged

    # -- selector matching (family-level only; no case ids) -----------------
    @staticmethod
    def _row_matches(entry, case: CaptureCase) -> bool:
        if isinstance(entry, str):
            return True
        kind = entry.get("when_kind")
        if kind is not None and kind not in case.kind:
            return False
        prefix = entry.get("when_id_prefix")
        if prefix is not None and not case.case_id.startswith(prefix):
            return False
        runner = entry.get("when_runner")
        if runner is not None and case.runner != runner:
            return False
        return True

    def _column(self, suite: str, column: str, case: CaptureCase) -> list[RegexChoice]:
        out: list[RegexChoice] = []
        for entry in self.suite_row(suite).get(column) or []:
            if not self._row_matches(entry, case):
                continue
            pattern = entry if isinstance(entry, str) else str(entry.get("re") or "")
            if not pattern:
                continue
            source = f"registry:{suite}.{column}[{len(out)}]"
            out.append(RegexChoice(pattern, source))
        return out


def derive_corpus_regex(case: CaptureCase) -> RegexChoice | None:
    """corpus_sc result.json["kernel"] -> regex, corpus-first per DESIGN §2.

    Splits the multi-value form ("<Symbol>|kernels=<profiler dump>") on "|"
    and keeps only identifier-shaped segments (the profiler-dump segments
    carry spaces and '=' and are dropped wholesale, so auxiliary kernels such
    as preprocess never enter the trace window).  "<fn>.py" strips to <fn>,
    matching how CuTe-DSL embeds the python function name in the symbol.
    """
    raw = str((case.corpus or {}).get("kernel") or "").strip()
    if not raw:
        return None
    tokens: list[str] = []
    for segment in raw.split("|"):
        segment = segment.strip()
        if not segment or " " in segment or "=" in segment:
            continue
        if segment.endswith(".py"):
            segment = segment[: -len(".py")]
        if _TOKEN_RE.fullmatch(segment) and segment not in tokens:
            tokens.append(segment)
        if len(tokens) >= _MAX_TOKENS:
            break
    if not tokens:
        return None
    pattern = ".*(?:" + "|".join(re.escape(token) for token in tokens) + ").*"
    return RegexChoice(pattern, f"corpus_sc:kernel:{tokens[0]}" + ("+…" if len(tokens) > 1 else ""))


class FamilyAdapter:
    """Base class: shared harness/regex/budget mechanics, family hooks below."""

    suite = ""

    def __init__(self, registry: Registry, root: Path = ROOT):
        self.registry = registry
        self.root = Path(root)

    # -- family hooks -------------------------------------------------------
    def jit_env(self, case: CaptureCase, workdir: Path) -> dict:
        """Per-case JIT cache / dump env so the dumped cubin is the one that ran
        (process-wide cache hits skip compile and therefore skip dump)."""
        raise NotImplementedError

    def sass_source(self, case: CaptureCase, workdir: Path) -> SassSource:
        """Where S3's cuobjdump reads from (DESIGN §2: nvcc bin / DG dg_cache
        cubin / DSL cute_dump cubin).  Must not require GPU-side state to answer."""
        raise NotImplementedError

    def sass_sources(self, case: CaptureCase, workdir: Path) -> list["SassSource"]:
        """W3: every code asset this adapter can deliver for the case.  S3
        consumes these as the DELIVERED-ASSET fallback: kernels whose real
        loaded image is bound by the tracer's loaded_function_images.csv
        index never trigger this asset walk (no whole-library member
        traversal for functions with exact loaded evidence); every kernel
        without such a binding keeps this strict content-selection
        contract.  An adapter only prepares inputs/calls/JIT assets and
        never decides which kernel is 'the' kernel."""
        main = self.sass_source(case, workdir)
        return [main] if main.exists else []

    def jit_assets(self, case: CaptureCase, workdir: Path) -> list[dict]:
        """W3 declaration of the JIT assets the adapter delivered for this
        case (paths + kinds).  Evidence for the launch integrity ledger, not
        a semantic input."""
        return [
            {"kind": source.kind, "path": str(source.path), "exists": source.exists,
             "note": source.note}
            for source in self.sass_sources(case, workdir)
        ]

    # -- shared mechanics ---------------------------------------------------
    def _pick_main_cubin(self, case: CaptureCase, cubins: list[Path]) -> Path | None:
        """Choose the main-kernel cubin from a JIT dump dir.  JIT runs also
        dump auxiliary kernels (DG quant helpers like transpose/pack ue8m0,
        DSL preprocess kernels; FA4 bwd additionally recompiles the fwd
        kernel for LSE), and lexicographic or size order picks those first.
        Family-level rule: the registry main_ncu rows are ordered
        most-specific-first ("按序优先"), so rank a cubin by its EARLIEST
        matching row, then by file size; no match at all -> None (fail
        closed)."""
        if not cubins:
            return None
        rows = self.registry.suite_row(self.suite).get("main_ncu") or []

        def best_row_index(path_str: str) -> int | None:
            for i, entry in enumerate(rows):
                text = entry if isinstance(entry, str) else str(entry.get("re") or "")
                if text and re.search(text, path_str, re.IGNORECASE):
                    return i
            return None

        def score(p: Path):
            idx = best_row_index(str(p))
            return (0 if idx is None else 1, -(idx or 0), p.stat().st_size)

        best = max(cubins, key=score)
        return best if score(best)[0] else None

    def harness_argv(self, case: CaptureCase) -> list[str]:
        harness = f"suites/{self.suite}/run.py"
        return ["python3", str(self.root / harness), case.case_id, DELIVERY_FLAG]

    def harness_env(self, case: CaptureCase, workdir: Path) -> dict:
        env = dict(self.jit_env(case, workdir))
        # Tracer keys are appended by the campaign (S1 inject / S2 strip);
        # adapters never touch them.
        env.pop("CUDA_INJECTION64_PATH", None)
        return env

    def kernel_regex_candidates_nvbit(self, case: CaptureCase) -> list[RegexChoice]:
        """Ordered: corpus_sc kernel first, then registry rows (DESIGN §2)."""
        out: list[RegexChoice] = []
        corpus = derive_corpus_regex(case)
        if corpus is not None:
            out.append(corpus)
        out.extend(self.registry._column(self.suite, "main_nvbit", case))
        return out

    def kernel_regex_nvbit(self, case: CaptureCase) -> RegexChoice:
        candidates = self.kernel_regex_candidates_nvbit(case)
        if not candidates:
            raise KeyError(f"{case.case_id}: no nvbit regex candidate for suite {self.suite}")
        return candidates[0]

    def kernel_regex_ncu(self, case: CaptureCase) -> RegexChoice:
        out: list[RegexChoice] = []
        corpus = derive_corpus_regex(case)
        if corpus is not None:
            out.append(corpus)
        rows = self.registry._column(self.suite, "main_ncu", case)
        if rows:
            out.append(rows[0])
        if not out:
            raise KeyError(f"{case.case_id}: no ncu regex candidate for suite {self.suite}")
        return out[0]

    def kernel_regex_ncu_registry(self, case: CaptureCase) -> RegexChoice | None:
        """First registry main_ncu row (corpus-free).  Used when S1's nvbit
        side fell back to the registry, signalling the corpus kernel token is
        an API name that also cannot match NCU's mangled-side kernel names."""
        rows = self.registry._column(self.suite, "main_ncu", case)
        return rows[0] if rows else None

    def budgets(self, case: CaptureCase) -> dict:
        return self.registry.budgets(self.suite)

    def main_kernel_regex(self, case: CaptureCase) -> RegexChoice:
        """The case's main-kernel identity: spec field ``main_kernel``
        (regex, full-match on the mangled name) when declared, else the
        first nvbit candidate.  The main launch is the FIRST launch whose
        mangled name full-matches this regex."""
        pattern = case.spec.get("main_kernel")
        if pattern:
            return RegexChoice(str(pattern), "spec.main_kernel")
        return self.kernel_regex_nvbit(case)

    def resolve(self, case: CaptureCase, workdir: Path) -> Resolve:
        candidates = self.kernel_regex_candidates_nvbit(case)
        if not candidates:
            raise KeyError(f"{case.case_id}: no nvbit regex candidate")
        return Resolve(
            case=case,
            argv=self.harness_argv(case),
            env=self.harness_env(case, workdir),
            regex_nvbit=candidates[0],
            regex_ncu=self.kernel_regex_ncu(case),
            regex_ncu_registry=self.kernel_regex_ncu_registry(case),
            regex_nvbit_fallbacks=candidates[1:],
            main_regex=self.main_kernel_regex(case),
            sass=self.sass_source(case, workdir),
            budgets=self.budgets(case),
            want_sms=case.want_sms,
            workdir=Path(workdir),
            adapter=self,
        )


class DeepGemmAdapter(FamilyAdapter):
    """deepgemm: DG JIT dumps SASS cubins into DG_JIT_CACHE_DIR; numerics skip
    under the tracer is existing harness semantics (nothing to set here)."""

    suite = "deepgemm"

    def jit_env(self, case: CaptureCase, workdir: Path) -> dict:
        return {
            "DG_JIT_DUMP_SASS": "1",
            "DG_JIT_CACHE_DIR": str(Path(workdir) / "dg_cache"),
        }

    def sass_source(self, case: CaptureCase, workdir: Path) -> SassSource:
        cache = Path(workdir) / "dg_cache"
        cubins = sorted(cache.rglob("*.cubin")) if cache.is_dir() else []
        best = self._pick_main_cubin(case, cubins)
        if best is not None:
            return SassSource(best, "dg_jit_cubin", True,
                              f"{len(cubins)} cubin(s) in dg_cache; main-kernel pick {best.parent.name}/{best.name}")
        return SassSource(cache / "kernel.cubin", "dg_jit_cubin", False,
                          "dg_cache/<kernel>.cubin appears after the S1 run "
                          "(DG_JIT_DUMP_SASS=1 + per-case DG_JIT_CACHE_DIR)")

    def sass_sources(self, case: CaptureCase, workdir: Path) -> list[SassSource]:
        """Every dumped JIT cubin of this run plus the torch shared object
        (W3): DG aux kernels (transpose/pack ue8m0) and any libtorch kernels
        are real workload launches and get their own per-kernel SASS
        selection.  The torch library is located via importlib.util.find_spec
        (never imported) in the interpreter that also runs S1."""
        sources: list[SassSource] = []
        cache = Path(workdir) / "dg_cache"
        cubins = sorted(cache.rglob("*.cubin")) if cache.is_dir() else []
        sources.extend(SassSource(cubin, "dg_jit_cubin", True, "dg_cache JIT dump")
                       for cubin in cubins)
        torch_so = _torch_cuda_lib()
        if torch_so is not None:
            sources.append(SassSource(torch_so, "torch_so_bin", True,
                                      f"libtorch_cuda.so of this interpreter ({torch_so}); "
                                      "cuobjdump reads its embedded cubins"))
        return sources


class CutlassAdapter(FamilyAdapter):
    """cutlass: spec.runner routes cpp (nvcc binary) vs dsl (CuTe-DSL dump).

    S1/S2 drive the case directly (binary/script + spec args), NOT through
    suites/cutlass/run.py: with --delivery300sc that harness ignores
    the positional case id and sweeps the whole SC catalog through the
    delivery300 (non-SC) spec entries, so SC shapes never reach the GPU.
    v2 needs only "launch this exact kernel"; result.json plumbing is the
    corpus phase's job, already done.
    """

    suite = "cutlass"

    def harness_argv(self, case: CaptureCase) -> list[str]:
        if case.spec.get("mode") == "full":
            return [sys.executable,
                    str(self.root / "suites" / "cutlass" / "run.py"),
                    case.case_id]
        args = [str(a) for a in (case.spec.get("args") or [])]
        if case.runner == "cpp":
            bin_path = self._bin_path(case)
            if bin_path is None:
                raise KeyError(f"{case.case_id}: spec entry has no bin field")
            return [str(bin_path), *args]
        if case.runner == "dsl":
            script = str(case.spec.get("script") or "")
            if not script:
                raise KeyError(f"{case.case_id}: spec entry has no script field")
            if os.environ.get("CUTE_INLINE_TMA_LIB"):
                return [sys.executable, str(self.root / "scripts" / "cute_inline_tma.py"), str(self.root / script), *args]
            return [sys.executable, str(self.root / script), *args]
        return super().harness_argv(case)

    def harness_env(self, case: CaptureCase, workdir: Path) -> dict:
        env = super().harness_env(case, workdir)
        if case.spec.get("mode") == "full" and case.runner == "cpp":
            env["BW_CUTLASS_BIN_ROOT"] = os.environ.get(
                "BW_CUTLASS_BIN_ROOT",
                str(Path(os.environ["CYCLESCOPE_ARTIFACT_ROOT"])
                    / "tools" / "blackwell-corpus" / "cutlass"))
        if case.runner == "dsl":
            # Mirror run.py case_dsl: some SC DSL scripts import cutlass.testing,
            # only present under the FA4 prefix.
            fa4 = _fa4_pythonpath(self.root)
            if fa4:
                env["PYTHONPATH"] = os.pathsep.join(
                    fa4 + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        return env

    def jit_env(self, case: CaptureCase, workdir: Path) -> dict:
        if case.runner == "dsl":
            dump_dir, cache_dir = _prepare_cute_dsl_dirs(workdir)
            return {
                "CUTE_DSL_KEEP": "cubin,ptx",
                "CUTE_DSL_DUMP_DIR": str(dump_dir),
                "CUTE_DSL_CACHE_DIR": str(cache_dir),
                "CUTE_DSL_NO_CACHE": "1",
            }
        # runner == "cpp": nvcc binary compiled once by the harness into the
        # corpus bin dir; no JIT env.
        return {}

    def _bin_path(self, case: CaptureCase) -> Path | None:
        bin_name = str(case.spec.get("bin") or "")
        if not bin_name:
            return None
        corpus_root = _corpus_root(self.root)
        # Explicit spec field only; cases without it resolve to the shared
        # dev-set bin/ (unchanged behavior).
        sub = str(case.spec.get("bin_dir") or "bin")
        return corpus_root / "cutlass" / sub / bin_name

    def sass_source(self, case: CaptureCase, workdir: Path) -> SassSource:
        workdir = Path(workdir)
        if case.runner == "dsl":
            for sub in ("cute_dump", "cute_cache"):
                base = workdir / sub
                cubins = sorted(base.rglob("*.cubin")) if base.is_dir() else []
                best = self._pick_main_cubin(case, cubins)
                if best is not None:
                    return SassSource(best, "dsl_cubin", True,
                                      f"main-kernel cubin from {sub} ({len(cubins)} dumped): "
                                      f"{best.name[:80]}")
            return SassSource(workdir / "cute_dump" / "kernel.cubin", "dsl_cubin", False,
                              "cute_dump/<kernel>.cubin appears after the S1 run "
                              "(CUTE_DSL_KEEP=cubin,ptx + per-case CUTE_DSL_DUMP_DIR)")
        if case.runner == "cpp":
            bin_path = self._bin_path(case)
            if bin_path is None:
                return SassSource(workdir / "unknown_bin", "nvcc_bin", False,
                                  "spec entry has no bin field")
            return SassSource(bin_path, "nvcc_bin", bin_path.is_file(),
                              "spec bin under BW_CORPUS_ROOT/cutlass/bin (cuobjdump reads it directly)")
        return SassSource(workdir / "unknown_bin", "none", False,
                          "spec.runner unknown; pass --sass explicitly (reparse path)")

    def sass_sources(self, case: CaptureCase, workdir: Path) -> list[SassSource]:
        workdir = Path(workdir)
        if case.runner == "cpp":
            source = self.sass_source(case, workdir)
            return [source] if source.exists else []
        out: list[SassSource] = []
        for sub in ("cute_dump", "cute_cache"):
            base = workdir / sub
            for cubin in sorted(base.rglob("*.cubin")) if base.is_dir() else []:
                out.append(SassSource(cubin, "dsl_cubin", True, f"{sub} JIT dump"))
        return out


class FlashAttentionAdapter(FamilyAdapter):
    """flashattention (FA4): CuTe-DSL under the .deps/fa4 prefix (the harness
    manages the prefix); dumps land in the per-case CUTE_DSL_DUMP_DIR."""

    suite = "flashattention"

    def jit_env(self, case: CaptureCase, workdir: Path) -> dict:
        dump_dir, cache_dir = _prepare_cute_dsl_dirs(workdir)
        return {
            "CUTE_DSL_KEEP": "cubin,ptx",
            "CUTE_DSL_DUMP_DIR": str(dump_dir),
            "CUTE_DSL_CACHE_DIR": str(cache_dir),
            "CUTE_DSL_NO_CACHE": "1",
        }

    def sass_source(self, case: CaptureCase, workdir: Path) -> SassSource:
        workdir = Path(workdir)
        for sub in ("cute_dump", "cute_cache"):
            base = workdir / sub
            cubins = sorted(base.rglob("*.cubin")) if base.is_dir() else []
            best = self._pick_main_cubin(case, cubins)
            if best is not None:
                return SassSource(best, "dsl_cubin", True,
                                  f"main-kernel cubin from {sub} ({len(cubins)} dumped): "
                                  f"{best.name[:80]}")
        return SassSource(workdir / "cute_dump" / "kernel.cubin", "dsl_cubin", False,
                          "cute_dump/<kernel>.cubin appears after the S1 run "
                          "(FA4 reuses the .deps/fa4 prefix; KEEP=cubin,ptx)")

    def sass_sources(self, case: CaptureCase, workdir: Path) -> list[SassSource]:
        out: list[SassSource] = []
        for sub in ("cute_dump", "cute_cache"):
            base = Path(workdir) / sub
            for cubin in sorted(base.rglob("*.cubin")) if base.is_dir() else []:
                out.append(SassSource(cubin, "dsl_cubin", True, f"{sub} JIT dump"))
        return out


_ADAPTERS = {
    DeepGemmAdapter.suite: DeepGemmAdapter,
    CutlassAdapter.suite: CutlassAdapter,
    FlashAttentionAdapter.suite: FlashAttentionAdapter,
}




class FlashInferAdapter(FamilyAdapter):
    """flashinfer: one real-call runner entry per case
    (suites/flashinfer/run.py), selection by the spec only.
    The runner imports the vendored source from BW_FI_SRC; CUTE_DSL_ARCH
    stays pinned to sm_103a and the per-case CuTe dump/cache dirs land
    under the case workdir so capture and timing reuse the same cubins."""

    suite = "flashinfer"

    def harness_argv(self, case: CaptureCase) -> list[str]:
        return [sys.executable,
                str(self.root / "suites" / "flashinfer" / "run.py"),
                case.case_id]

    def harness_env(self, case: CaptureCase, workdir: Path) -> dict:
        env = super().harness_env(case, workdir)
        env["BW_FI_SRC"] = str(_flashinfer_source())
        if os.environ.get("BW_FA4_PREFIX"):
            env["BW_FA4_PREFIX"] = os.environ["BW_FA4_PREFIX"]
            site = _fa4_prefix_pythonpath(Path(env["BW_FA4_PREFIX"]))
            if site:
                env["PYTHONPATH"] = os.pathsep.join(
                    site + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        return env

    def jit_env(self, case: CaptureCase, workdir: Path) -> dict:
        dump_dir, cache_dir = _prepare_cute_dsl_dirs(workdir)
        return {"CUTE_DSL_ARCH": "sm_103a",
                "CUTE_DSL_KEEP": "cubin,ptx",
                "CUTE_DSL_DUMP_DIR": str(dump_dir),
                "CUTE_DSL_CACHE_DIR": str(cache_dir),
                "CUTE_DSL_NO_CACHE": "1"}

    def sass_source(self, case: CaptureCase, workdir: Path) -> SassSource:
        workdir = Path(workdir)
        for sub in ("cute_dump", "cute_cache"):
            base = workdir / sub
            cubins = sorted(base.rglob("*.cubin")) if base.is_dir() else []
            if cubins:
                return SassSource(cubins[0], "dsl_cubin", True,
                                  f"{len(cubins)} cubin(s) under {sub}")
        return SassSource(workdir / "cute_dump" / "kernel.cubin", "dsl_cubin",
                          False, "cute_dump/<kernel>.cubin after the run")

    def sass_sources(self, case: CaptureCase, workdir: Path) -> list[SassSource]:
        out: list[SassSource] = []
        for sub in ("cute_dump", "cute_cache"):
            base = Path(workdir) / sub
            for cubin in sorted(base.rglob("*.cubin")) if base.is_dir() else []:
                out.append(SassSource(cubin, "dsl_cubin", True, f"{sub} JIT dump"))
        return out

    def budgets(self, case: CaptureCase) -> dict:
        return {"timeout_s": 900, "dyn_lines_cap": 200_000_000,
                "sidecar_cap_bytes": 2 * 2 ** 30, "trace_bytes_cap": 8 * 2 ** 30,
                "spool_bytes_cap": 8 * 2 ** 30, "regex_fallback_retry": 0}

    def kernel_regex_candidates_nvbit(self, case: CaptureCase) -> list[RegexChoice]:
        pattern = case.spec.get("main_kernel")
        if not pattern:
            raise KeyError(f"{case.case_id}: spec needs main_kernel")
        return [RegexChoice(str(pattern), "spec.main_kernel")]

    def kernel_regex_ncu(self, case: CaptureCase) -> RegexChoice:
        return RegexChoice(".*", "full_workload")

    def kernel_regex_ncu_registry(self, case: CaptureCase) -> RegexChoice | None:
        return None


class TritonAdapter(FamilyAdapter):
    """triton: one runner entry per case (suites/triton/run.py).
    Per-case TRITON_CACHE_DIR keeps the runtime-compiled cubin inside the
    case workdir so capture and timing load identical binaries."""

    suite = "triton"

    def harness_argv(self, case: CaptureCase) -> list[str]:
        runner = ("official/run.py" if case.spec.get("mode") == "full"
                  else "run.py")
        return [sys.executable,
                str(self.root / "suites" / "triton" / runner),
                case.case_id]

    def harness_env(self, case: CaptureCase, workdir: Path) -> dict:
        env = super().harness_env(case, workdir)
        if case.spec.get("mode") == "full":
            source = Path(os.environ.get(
                "TRITON_SOURCE_ROOT",
                str(Path(os.environ["CYCLESCOPE_ARTIFACT_ROOT"])
                    / "tools" / "triton-source")))
            env["TRITON_SOURCE_ROOT"] = str(source)
            env["BW_TRITON_WORKDIR"] = str(workdir)
            kernels_import = os.environ.get("TRITON_KERNELS_IMPORT_ROOT")
            current = os.environ.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                ([kernels_import] if kernels_import else []) +
                [str(self.root / "suites"),
                 *([current] if current else [])])
        return env

    def jit_env(self, case: CaptureCase, workdir: Path) -> dict:
        return {"TRITON_CACHE_DIR": str(Path(workdir) / "triton_cache")}

    def sass_source(self, case: CaptureCase, workdir: Path) -> SassSource:
        cache = Path(workdir) / "triton_cache"
        cubins = sorted(cache.rglob("*.cubin")) if cache.is_dir() else []
        if cubins:
            return SassSource(cubins[0], "triton_cubin", True,
                              f"{len(cubins)} cubin(s) in TRITON_CACHE_DIR")
        return SassSource(cache / "kernel.cubin", "triton_cubin", False,
                          "triton_cache/<...>/*.cubin appears after the run")

    def sass_sources(self, case: CaptureCase, workdir: Path) -> list[SassSource]:
        cache = Path(workdir) / "triton_cache"
        return [SassSource(c, "triton_cubin", True, "triton cache dump")
                for c in sorted(cache.rglob("*.cubin"))]             if cache.is_dir() else []

    def kernel_regex_candidates_nvbit(self, case: CaptureCase) -> list[RegexChoice]:
        pattern = case.spec.get("main_kernel")
        if not pattern:
            raise KeyError(f"{case.case_id}: spec needs main_kernel")
        return [RegexChoice(str(pattern), "spec.main_kernel")]

    def kernel_regex_ncu(self, case: CaptureCase) -> RegexChoice:
        return RegexChoice(".*", "full_workload")

    def kernel_regex_ncu_registry(self, case: CaptureCase) -> RegexChoice | None:
        return None


class FlashMlaAdapter(FamilyAdapter):
    """flashmla: one runner entry per case (suites/flashmla/run.py);
    the built extension lives under BW_FLASHMLA_PREFIX (deps/flashmla)."""

    suite = "flashmla"

    def harness_argv(self, case: CaptureCase) -> list[str]:
        return [sys.executable,
                str(self.root / "suites" / "flashmla" / "run.py"),
                case.case_id]

    def jit_env(self, case: CaptureCase, workdir: Path) -> dict:
        return {"CUTE_DSL_KEEP": "cubin,ptx",
                "CUTE_DSL_DUMP_DIR": str(Path(workdir) / "cute_dump")}

    def sass_source(self, case: CaptureCase, workdir: Path) -> SassSource:
        workdir = Path(workdir)
        cubins = sorted((workdir / "cute_dump").rglob("*.cubin"))             if (workdir / "cute_dump").is_dir() else []
        if cubins:
            return SassSource(cubins[0], "dsl_cubin", True,
                              f"{len(cubins)} cubin(s) under cute_dump")
        prefix = os.environ.get("BW_FLASHMLA_PREFIX")
        so = sorted(Path(prefix).rglob("*.so")) if prefix else []
        if so:
            return SassSource(so[0], "ext_so", True, "flashmla extension .so")
        return SassSource(workdir / "cute_dump" / "kernel.cubin", "dsl_cubin",
                          False, "cute_dump/<kernel>.cubin after the run")

    def kernel_regex_candidates_nvbit(self, case: CaptureCase) -> list[RegexChoice]:
        pattern = case.spec.get("main_kernel")
        if not pattern:
            raise KeyError(f"{case.case_id}: spec needs main_kernel")
        return [RegexChoice(str(pattern), "spec.main_kernel")]

    def kernel_regex_ncu(self, case: CaptureCase) -> RegexChoice:
        return RegexChoice(".*", "full_workload")

    def kernel_regex_ncu_registry(self, case: CaptureCase) -> RegexChoice | None:
        return None




class CublasAdapter(FamilyAdapter):
    """cublas: prebuilt cublaslt_case binary (blackwell-cublas-build);
    spec carries the full arg vector.  cpp runner, no JIT env."""

    suite = "cublas"

    def _bin(self) -> Path:
        return Path(os.environ.get(
            "CYCLESCOPE_CUBLAS_BIN",
            str(Path(os.environ["CYCLESCOPE_ARTIFACT_ROOT"]) /
                "tools" / "blackwell-cublas-build" / "cublaslt_case")))

    def harness_argv(self, case: CaptureCase) -> list[str]:
        args = [str(a) for a in (case.spec.get("args") or [])]
        return [str(self._bin()), *args]

    def jit_env(self, case: CaptureCase, workdir: Path) -> dict:
        return {}

    def sass_source(self, case: CaptureCase, workdir: Path) -> SassSource:
        return SassSource(self._bin(), "nvcc_bin", True,
                          "cublaslt_case binary (cuBLAS host code)")

    def sass_sources(self, case: CaptureCase, workdir: Path) -> list[SassSource]:
        return [self.sass_source(case, workdir)]
_ADAPTERS.update({
    CublasAdapter.suite: CublasAdapter,
    FlashInferAdapter.suite: FlashInferAdapter,
    TritonAdapter.suite: TritonAdapter,
    FlashMlaAdapter.suite: FlashMlaAdapter,
})


def get_adapter(suite: str, registry: Registry | None = None, root: Path = ROOT) -> FamilyAdapter:
    cls = _ADAPTERS.get(suite)
    if cls is None:
        raise KeyError(f"no FamilyAdapter for suite {suite!r}; known: {sorted(_ADAPTERS)}")
    return cls(registry or Registry(), root)


def load_case(case_id: str, root: Path = ROOT, registry: Registry | None = None) -> CaptureCase:
    """Locate spec + corpus entries for a delivery300sc case id.

    Spec: BW_CASES_DIR/cases_<suite>.json when overridden, otherwise
    suites/c450_full/cases_<suite>.json.  Corpus: BW_CORPUS_ROOT
    (default artifacts/corpus_sc) result.json, falling back to the legacy
    artifacts/corpus tree so historical ids stay reparseable.  Both sides are
    optional; gates degrade to not_evaluable instead of crashing.
    """
    root = Path(root)
    suite = guess_suite(case_id)
    if not suite:
        raise KeyError(f"cannot route {case_id!r} to a capture family")
    spec: dict = {}
    cases_dir = os.environ.get("BW_CASES_DIR")
    spec_path = (Path(cases_dir) if cases_dir else
                 root / "suites" / "c450_full") / f"cases_{suite}.json"
    if spec_path.is_file():
        for entry in json.loads(spec_path.read_text(encoding="utf-8")).get("cases") or []:
            if entry.get("id") == case_id:
                spec = entry
                break
    corpus: dict = {}
    corpus_root = _corpus_root(root)
    for base in (corpus_root, root / "artifacts" / "corpus"):
        candidate = base / suite / case_id / "result.json"
        if candidate.is_file():
            try:
                corpus = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                corpus = {}
            break
    return CaptureCase(case_id=case_id, suite=suite, spec=spec, corpus=corpus)
