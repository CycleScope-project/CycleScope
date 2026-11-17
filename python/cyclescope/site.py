"""Site configuration: artifact root, build dir, tracer, ncu, nvdisasm.

Resolution order for every setting: environment variable > site file >
built-in default.  Site file = ``$CYCLESCOPE_SITE_CONFIG`` when set, else
``<repo>/configs/site.json`` when it exists, else none (see
``configs/site.example.json`` for the schema).

    python3 -m cyclescope.site --show            # resolved settings as JSON
    python3 -m cyclescope.site --get <name>      # one value, for shell use
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_KEYS = {"artifact_root", "build_dir", "tracer", "ncu", "nvdisasm"}
_cache: dict[str, dict] = {}


def _config_path() -> Path | None:
    env = os.environ.get("CYCLESCOPE_SITE_CONFIG")
    if env:
        return Path(env)
    default = REPO / "configs" / "site.json"
    return default if default.is_file() else None


def config() -> dict:
    """Site file contents ({} without a file), cached per file path."""
    path = _config_path()
    key = str(path)
    if key not in _cache:
        if path is None:
            _cache[key] = {}
        else:
            data = json.loads(path.read_text())
            bad = set(data) - _KEYS - {k for k in data
                                       if k.startswith("_")}
            if bad:
                raise ValueError(
                    f"{path}: unknown keys {sorted(bad)} "
                    f"(allowed: {sorted(_KEYS)}, '_' prefix = note)")
            _cache[key] = data
    return _cache[key]


def artifact_root() -> Path:
    """$CYCLESCOPE_ARTIFACT_ROOT > site artifact_root > <repo>/artifacts."""
    value = (os.environ.get("CYCLESCOPE_ARTIFACT_ROOT")
             or config().get("artifact_root"))
    if value:
        p = Path(value)
        return p if p.is_absolute() else (REPO / p).resolve()
    return REPO / "artifacts"


def build_dir() -> Path:
    """$CYCLESCOPE_BUILD_DIR > site build_dir > <repo>/build."""
    value = os.environ.get("CYCLESCOPE_BUILD_DIR") or config().get("build_dir")
    if value:
        p = Path(value)
        return p if p.is_absolute() else (REPO / p).resolve()
    return REPO / "build"


def tools_dir() -> Path:
    return artifact_root() / "tools"


def tracer(sm_arch: str) -> Path:
    """$CYCLESCOPE_TRACER > site tracer[sm_arch] > <root>/build/tracer/..."""
    env = os.environ.get("CYCLESCOPE_TRACER")
    if env:
        return Path(env)
    cfg = config().get("tracer") or {}
    if sm_arch in cfg:
        return Path(cfg[sm_arch])
    return artifact_root() / "build" / "tracer" / sm_arch / "tracer.so"


def ncu() -> str | None:
    """$CYCLESCOPE_NCU > site ncu > None."""
    return os.environ.get("CYCLESCOPE_NCU") or config().get("ncu")


def nvdisasm() -> str:
    """$NVDISASM > site nvdisasm > <root>/tools/cuda-toolkit-13.2/..."""
    return (os.environ.get("NVDISASM") or config().get("nvdisasm")
            or str(tools_dir() / "cuda-toolkit-13.2" / "bin" / "nvdisasm"))


def export_environment(env=None) -> None:
    """Export the resolved artifact root for child processes (runner
    scripts, the far profile, the apps read the variable)."""
    if env is None:
        env = os.environ
    env.setdefault("CYCLESCOPE_ARTIFACT_ROOT", str(artifact_root()))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="cyclescope.site")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--get",
                    choices=["artifact_root", "build_dir", "tools_dir",
                             "ncu", "nvdisasm"])
    args = ap.parse_args(argv)
    if args.get:
        v = {"artifact_root": artifact_root, "build_dir": build_dir,
             "tools_dir": tools_dir, "ncu": ncu,
             "nvdisasm": nvdisasm}[args.get]()
        print(v or "")
        return 0
    print(json.dumps({
        "site_file": str(_config_path() or ""),
        "artifact_root": str(artifact_root()),
        "build_dir": str(build_dir()),
        "tools_dir": str(tools_dir()),
        "tracer_sm_90a": str(tracer("sm_90a")),
        "tracer_sm_103a": str(tracer("sm_103a")),
        "ncu": ncu(),
        "nvdisasm": nvdisasm(),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
