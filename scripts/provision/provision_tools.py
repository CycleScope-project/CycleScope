#!/usr/bin/env python3
"""Provision capture tools into $CYCLESCOPE_ARTIFACT_ROOT/tools/<name>.

Subcommands:
  fetch <name>...|--all  download git/pypi_wheels/download entries
  build <name>           run the entry's registered build script
  verify [--all]         check fetched trees and build outputs
  bundle --out DIR       export pristine artifacts for offline machines
  plan                   print the reproduction steps in dependency order

Usage example (build): run inside the toolchain environment:
  cd <clone> && scripts/provision/provision_tools.py build hopper-fa3-build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_sha256 import tree_sha256  # noqa: E402

sys.path.insert(0, str(REPO / "python"))
from cyclescope import site  # noqa: E402

ARTIFACTS = site.artifact_root()
TOOLS = ARTIFACTS / "tools"
REGISTRY = REPO / "configs" / "capture_tools.json"
LOCKS = REPO / "configs" / "capture_locks"


def _tmp_root() -> Path:
    """Temp root for provisioning: $CYCLESCOPE_TMP if set, else
    <artifact_root>/tmp (created on demand)."""
    root = (Path(os.environ["CYCLESCOPE_TMP"]) if os.environ.get("CYCLESCOPE_TMP")
            else ARTIFACTS / "tmp")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _child_env() -> dict:
    env = dict(os.environ)
    env["TMPDIR"] = str(_tmp_root())
    return env


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, cache: Path | None) -> Path:
    if cache is not None:
        cand = cache / dest.name
        if cand.is_file():
            shutil.copy2(cand, dest)
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        subprocess.run(["curl", "-sfL", "--retry", "3", "-o",
                        str(dest), url], check=True, env=_child_env())
    return dest


def _unpack(archive: Path, dest: Path, how: str,
            strip: int = 0) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if how in ("tar", "tar.gz", "tgz", "tar.bz2", "tar.xz"):
        mode = {"tar": "r:", "tar.gz": "r:gz", "tgz": "r:gz",
                "tar.bz2": "r:bz2", "tar.xz": "r:xz"}[how]
        with tarfile.open(archive, mode) as tf:
            if strip:
                for m in tf.getmembers():
                    parts = Path(m.name).parts[strip:]
                    if not parts:
                        continue
                    m.name = str(Path(*parts))
                    tf.extract(m, dest)
            else:
                tf.extractall(dest)
    elif how == "zip" or zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
            # extractall drops the unix mode: wheels ship executables
            # (ptxas, nvdisasm, ...) that must stay runnable
            for info in zf.infolist():
                mode = (info.external_attr >> 16) & 0o777
                if mode and not info.is_dir():
                    (dest / info.filename).chmod(mode)
    elif how == "none":
        shutil.copy2(archive, dest / archive.name)
    else:
        raise ValueError(f"unknown unpack rule {how}")


def _gen_console_scripts(dist_dir: Path, bin_dir: Path) -> None:
    """Generate console_scripts wrappers the way pip --target does."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for ep in dist_dir.glob("*.dist-info/entry_points.txt"):
        in_cs = False
        for line in ep.read_text().splitlines():
            line = line.strip()
            if line.startswith("["):
                in_cs = line == "[console_scripts]"
                continue
            if not in_cs:
                continue
            m = re.match(r"([\w.-]+)\s*=\s*([\w.]+):(\w+)", line)
            if not m:
                continue
            name, mod, fn = m.groups()
            script = bin_dir / name
            script.write_text(
                "#!/usr/bin/python3\nimport sys\n"
                f"from {mod} import {fn}\n"
                "if __name__ == '__main__':\n"
                "    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
                f"    sys.exit({fn}())\n")
            script.chmod(0o755)


def _filtered_tree_sha(root, ignore) -> str | None:
    """tree_sha256 with extra ignore list (relative paths or prefixes)."""
    import tempfile, shutil as sh
    if not ignore:
        return tree_sha256(root)
    tmp = Path(tempfile.mkdtemp(prefix="tsh-", dir=_tmp_root()))
    try:
        shutil.copytree(root, tmp / "t", symlinks=True,
                        ignore=sh.ignore_patterns("__pycache__"))
        for pat in ignore:
            tgt = tmp / "t" / pat
            if tgt.is_dir() and not tgt.is_symlink():
                sh.rmtree(tgt, ignore_errors=True)
            else:
                tgt.unlink(missing_ok=True)
        return tree_sha256(tmp / "t")
    finally:
        sh.rmtree(tmp, ignore_errors=True)


def _lock_path(entry: dict) -> Path:
    return REPO / entry["requirements"]


def _lock_lines(path: Path) -> list[tuple[str, str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z0-9_.-]+)==(\S+)\s+--hash=sha256:(\w+)",
                     line)
        if not m:
            raise ValueError(f"bad lock line: {line}")
        out.append(m.groups())
    return out


def _fetch_wheel(spec: str, version: str, sha: str, index: str,
                 dest_dir: Path, cache: Path | None,
                 pyver: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    before = set(dest_dir.iterdir())
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps",
           "--no-cache-dir", "--index-url", index, "-d", str(dest_dir)]
    if pyver:
        cmd += ["--python-version", pyver, "--only-binary=:all:"]
    cmd.append(f"{spec}=={version}")
    subprocess.run(cmd, check=True, env=_child_env())
    got = [p for p in dest_dir.iterdir() if p not in before]
    if len(got) != 1:
        raise RuntimeError(f"pip download {spec}: {len(got)} files")
    whl = got[0]
    if _sha256(whl) != sha:
        raise RuntimeError(f"{whl.name}: sha256 mismatch")
    return whl


def _fetch_git(e: dict, tmp: Path, cache: Path | None) -> Path:
    """Shallow-fetch rev into tmp (bare-ish clone), return worktree."""
    rev = e["rev"]
    work = tmp / "wt"
    if cache is not None:
        for cand in (cache / f"{e.get('url','').rsplit('/',1)[-1]}.bundle",
                     cache / "bundle.git"):
            if cand.is_file():
                subprocess.run(["git", "clone", "-q", str(cand),
                                str(tmp / "clone")], check=True, env=_child_env())
                return tmp / "clone"
        for cand in cache.glob("*.tar*"):
            subprocess.run(["tar", "xf", str(cand), "-C", str(tmp)],
                           check=True, env=_child_env())
            sub = [d for d in tmp.iterdir() if d.is_dir()]
            if len(sub) == 1:
                return sub[0]
    clone = tmp / "clone"
    try:
        subprocess.run(["git", "init", "-q", str(clone)], check=True, env=_child_env())
        subprocess.run(["git", "remote", "add", "origin", e["url"]],
                       cwd=clone, check=True, env=_child_env())
        subprocess.run(["git", "fetch", "--depth", "1", "origin", rev],
                       cwd=clone, check=True, timeout=600, env=_child_env())
        subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"],
                       cwd=clone, check=True, env=_child_env())
    except Exception:
        # git protocol flaky on this network -> codeload tarball at rev
        import re as _re
        m = _re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
                      e["url"])
        if not m:
            raise
        org, repo = m.groups()
        tar = tmp / "src.tar.gz"
        _download(f"https://codeload.github.com/{org}/{repo}/tar.gz/{rev}",
                  tar, None)
        clone.mkdir(exist_ok=True)
        with tarfile.open(tar, "r:gz") as tf:
            for mm in tf.getmembers():
                parts = Path(mm.name).parts[1:]
                if not parts:
                    continue
                mm.name = str(Path(*parts))
                tf.extract(mm, clone)
    # recurse into recorded gitlinks if any
    if (tmp / "clone" / ".gitmodules").is_file() and             e.get("submodules"):
        for sub in e["submodules"]:
            path, srev = sub["path"], sub["rev"]
            stmp = tmp / ("sub_" + path.replace("/", "_"))
            try:
                subprocess.run(["git", "init", "-q", str(stmp)], check=True, env=_child_env())
                subprocess.run(["git", "remote", "add", "origin",
                                sub["url"]], cwd=stmp, check=True, env=_child_env())
                subprocess.run(["git", "fetch", "--depth", "1", "origin",
                                srev], cwd=stmp, check=True, timeout=600, env=_child_env())
                subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"],
                               cwd=stmp, check=True, env=_child_env())
            except Exception:
                import re as _re
                m2 = _re.match(
                    r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
                    sub["url"])
                tar2 = tmp / "sub.tar.gz"
                _download(
                    f"https://codeload.github.com/{m2.group(1)}"
                    f"/{m2.group(2)}/tar.gz/{srev}", tar2, None)
                stmp.mkdir(exist_ok=True)
                with tarfile.open(tar2, "r:gz") as tf:
                    for mm in tf.getmembers():
                        parts = Path(mm.name).parts[1:]
                        if not parts:
                            continue
                        mm.name = str(Path(*parts))
                        tf.extract(mm, stmp)
            shutil.copytree(stmp, tmp / "clone" / path,
                            dirs_exist_ok=True, symlinks=True)
    return tmp / "clone"


def _apply_overlay(e: dict, target: Path) -> None:
    """Copy registry `overlay` files (repo-relative `src`) into the
    exported tree; fails when a recorded sha256 does not match."""
    for rel, ov in (e.get("overlay") or {}).items():
        src = REPO / ov["src"]
        if not src.is_file():
            raise RuntimeError(f"overlay source missing: {ov['src']}")
        want = ov.get("sha256")
        got = _sha256(src)
        if want and got != want:
            raise RuntimeError(
                f"overlay {ov['src']}: sha256 {got} != {want}")
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _prune_empty_dirs(root: Path) -> None:
    """Remove empty directories bottom-up. A git checkout leaves an
    empty dir where a non-requested submodule gitlink sits;
    tree_sha256 ignores empty dirs, so registry values are unaffected."""
    for d in sorted((p for p in root.rglob("*")
                     if p.is_dir() and not p.is_symlink()),
                    key=lambda p: -len(p.parts)):
        try:
            d.rmdir()
        except OSError:
            pass


def _export_git(e: dict, src: Path, target: Path) -> None:
    """Checkout files to target without .git, honoring paths filter."""
    paths = e.get("paths")
    excl = e.get("exclude", [])
    target.mkdir(parents=True, exist_ok=True)
    if not paths:
        shutil.copytree(src, target, dirs_exist_ok=True, symlinks=True,
                        ignore=shutil.ignore_patterns(".git"))
        for rel in excl:
            t = target / rel
            if t.is_dir() and not t.is_symlink():
                shutil.rmtree(t, ignore_errors=True)
            else:
                t.unlink(missing_ok=True)
        return
    for item in paths:
        if isinstance(item, dict):
            rel, dst = item["src"], item["dest"]
        else:
            rel = dst = item
        s = src / rel
        d = target / dst
        if s.is_dir() and not s.is_symlink():
            shutil.copytree(s, d, dirs_exist_ok=True, symlinks=True)
        elif s.exists() or s.is_symlink():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d, follow_symlinks=False)


def fetch_entry(name: str, e: dict, cache: Path | None) -> str:
    kind = e.get("kind")
    target = ARTIFACTS / e.get("target", f"tools/{name}")
    if kind == "pypi_wheels":
        index = os.environ.get("CYCLESCOPE_PIP_INDEX_URL") or e["index_url"]
        lines = _lock_lines(_lock_path(e))
        tmp = Path(tempfile.mkdtemp(prefix="prov-", dir=_tmp_root()))
        try:
            wheels = [_fetch_wheel(s, v, h, index, tmp, cache,
                                   e.get("python_version"))
                      for s, v, h in lines]
            inst = e["install"]
            if inst == "unpack":
                target.mkdir(parents=True, exist_ok=True)
                for w in wheels:
                    _unpack(w, target / "site-packages"
                            if e.get("site_packages") else target, "zip")
            elif inst.startswith("unpack-ep:"):
                # unpack wheels into <libdir>, generate console scripts
                # into <bindir> (pip --target equivalent without a
                # matching local interpreter)
                lib_sub, bin_sub = inst.split(":", 1)[1].split(",")
                lib = target / lib_sub
                lib.mkdir(parents=True, exist_ok=True)
                for w in wheels:
                    _unpack(w, lib, "zip")
                _gen_console_scripts(lib, target / bin_sub)
                for patch in e.get("patches", []):
                    subprocess.run(
                        ["patch", "-p1", "-d",
                         str(target / e["patch_apply"]),
                         "-i", str(REPO / patch)], check=True, env=_child_env())
            elif inst.startswith("pip-target:"):
                sub = inst.split(":", 1)[1]
                dest = target / sub
                dest.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-deps",
                     "--no-cache-dir", "--target", str(dest)] +
                    [str(w) for w in wheels], check=True, env=_child_env())
            else:
                raise ValueError(inst)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    elif kind == "download":
        if e.get("manual"):
            return "skipped(manual)"
        tmp = Path(tempfile.mkdtemp(prefix="prov-", dir=_tmp_root()))
        try:
            if "files" in e:
                fl = json.loads((REPO / e["files"]).read_text())
                target.mkdir(parents=True, exist_ok=True)
                for rel, sha in fl["files"].items():
                    dest = target / rel
                    _download(e["url"].rstrip("/") + "/" + rel, dest, cache)
                    if _sha256(dest) != sha:
                        raise RuntimeError(f"{rel}: sha256 mismatch")
            else:
                arc = _download(e["url"], tmp / Path(e["url"]).name, cache)
                if _sha256(arc) != e["sha256"]:
                    raise RuntimeError(f"{name}: archive sha256 mismatch")
                _unpack(arc, target, e.get("unpack", "none"),
                        e.get("strip_components", 0))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    elif kind == "git":
        tmp = Path(tempfile.mkdtemp(prefix="prov-git-", dir=_tmp_root()))
        try:
            src = _fetch_git(e, tmp, cache)
            _export_git(e, src, target)
            for patch in e.get("patches", []):
                subprocess.run(["patch", "-p1", "-d", str(target),
                                "-i", str(REPO / patch)], check=True, env=_child_env())
            _apply_overlay(e, target)
            _prune_empty_dirs(target)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    elif kind == "composite":
        return compose_entry(name, e)
    elif kind == "build":
        return "skipped(build)"
    else:
        return f"skipped(kind={kind})"
    got = _filtered_tree_sha(target, e.get("ignored_extras", []))
    want = e.get("tree_sha256")
    if want and got != want:
        return f"FAIL tree_sha256 {got} != {want}"
    # sources are immutable once verified: any later write fails loudly
    if kind == "git":
        subprocess.run(["chmod", "-R", "a-w", str(target)], check=False)
    return "ok" + ("" if want else " (no tree_sha256 recorded)")


def _composite_links(name: str, e: dict) -> list[tuple[Path, Path, Path]]:
    """(link, link text, absolute target) per composite component:
    ``composed_of`` maps <relpath under the entry> -> <entry
    name>[/<subpath>]; the link text is relative so the artifact root
    stays relocatable."""
    root = ARTIFACTS / e.get("target", f"tools/{name}")
    out = []
    for rel, comp in e["composed_of"].items():
        link = root / rel
        dest = TOOLS / comp
        out.append((link, Path(os.path.relpath(dest, link.parent)), dest))
    return out


def compose_entry(name: str, e: dict) -> str:
    """Materialize a composite entry as a directory of symlinks (the
    runners' deps layout: blackwell-deps/{fa4,flashinfer,...},
    blackwell-flashinfer-deps/src/flashinfer)."""
    for link, rel, dest in _composite_links(name, e):
        if not dest.exists():
            return f"FAIL component missing: {dest}"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            if link.is_symlink() and os.readlink(link) == str(rel):
                continue
            return f"FAIL {link} exists and is not the expected link"
        link.symlink_to(rel)
    return "ok"


def verify_entry(name: str, e: dict) -> str:
    kind = e.get("kind")
    target = ARTIFACTS / e.get("target", f"tools/{name}")
    if e.get("manual"):
        return "manual"
    if not target.exists():
        return f"FAIL missing dir {target}"
    if kind in ("git", "pypi_wheels", "download"):
        want = e.get("tree_sha256")
        got = _filtered_tree_sha(target, e.get("ignored_extras", []))
        if not want:
            return "FAIL no tree_sha256 recorded"
        return "ok" if got == want else f"FAIL {got} != {want}"
    if kind == "build":
        if not e.get("outputs"):
            return "FAIL no outputs recorded"
        missing = [o["path"] if isinstance(o, dict) else o
                   for o in e.get("outputs", [])
                   if not (target / (o["path"] if isinstance(o, dict) else o)).exists()]
        return "ok" if not missing else f"missing {missing}"
    if kind == "composite":
        bad = [str(link) for link, rel, dest in _composite_links(name, e)
               if not link.is_symlink() or os.readlink(link) != str(rel)
               or not dest.exists()]
        return "ok" if not bad else f"FAIL links {bad}"
    return "skipped"


def bundle(entries: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, e in entries.items():
        kind = e.get("kind")
        if kind == "pypi_wheels":
            for spec, ver, sha in _lock_lines(_lock_path(e)):
                whl = out / f"{spec}-{ver}.whl"
                if whl.is_file():
                    continue
                subprocess.run(
                    [sys.executable, "-m", "pip", "download", "--no-deps",
                     "--no-cache-dir",
                     "--index-url",
                     os.environ.get("CYCLESCOPE_PIP_INDEX_URL")
                     or e["index_url"], "-d", str(out),
                     f"{spec}=={ver}"], check=True)
        elif kind == "download" and not e.get("manual"):
            if "files" in e:
                d = out / name
                fl = json.loads((REPO / e["files"]).read_text())
                for rel in fl["files"]:
                    _download(e["url"].rstrip("/") + "/" + rel,
                              d / rel, None)
            else:
                _download(e["url"], out / Path(e["url"]).name, None)


def plan(entries: dict) -> list[str]:
    steps = []
    done = set()
    def emit(name):
        if name in done or name not in entries:
            return
        e = entries[name]
        for i in e.get("inputs", []):
            emit(i)
        kind = e.get("kind")
        if kind in ("git", "pypi_wheels", "download"):
            steps.append(f"fetch {name} [{kind}]"
                         + (" MANUAL" if e.get("manual") else ""))
        elif kind == "build":
            steps.append(
                f"build {name} [toolchain={e.get('toolchain')} "
                f"device={e.get('device')}]")
        elif kind == "composite":
            steps.append(f"compose {name} {e.get('composed_of')}")
        done.add(name)
    for n in entries:
        emit(n)
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--source-cache", type=Path, default=None)
    p = sub.add_parser("verify")
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("build")
    p.add_argument("name")
    p = sub.add_parser("bundle")
    p.add_argument("--out", type=Path, required=True)
    sub.add_parser("plan")
    args = ap.parse_args()
    reg = load_registry()["tools"]

    if args.cmd == "plan":
        for s in plan(reg):
            print(s)
        return 0
    if args.cmd == "bundle":
        bundle(reg, args.out)
        return 0
    if args.cmd == "build":
        e = reg[args.name]
        if e.get("kind") != "build":
            raise SystemExit(f"{args.name}: kind={e.get('kind')}")
        env = dict(os.environ)
        env["CYCLESCOPE_ARTIFACT_ROOT"] = str(ARTIFACTS)
        env["CYCLESCOPE_ROOT"] = str(REPO)
        wrap = ('trap \'chown -R "$CYCLESCOPE_OWNER" '
                '"$CYCLESCOPE_ARTIFACT_ROOT/tools/$NAME" 2>/dev/null\' EXIT; '
                'bash "$0" "$@"')
        env["NAME"] = args.name
        return subprocess.run(
            ["bash", "-c", wrap, str(REPO / e["script"]), args.name],
            env=env).returncode

    names = list(reg) if getattr(args, "all", False) else args.names
    # composites link into other entries (possibly other composites):
    # materialize them after everything they reference
    ordered: list[str] = []
    def _order(n):
        if n in ordered or n not in reg:
            return
        if reg[n].get("kind") == "composite":
            for comp in reg[n]["composed_of"].values():
                _order(comp.split("/", 1)[0])
        ordered.append(n)
    for n in names:
        _order(n)
    names = [n for n in ordered if n in names]
    rc = 0
    for n in names:
        e = reg[n]
        if args.cmd == "fetch":
            if e.get("kind") not in ("git", "pypi_wheels", "download",
                                     "composite"):
                continue
            try:
                r = fetch_entry(n, e, args.source_cache)
            except Exception as ex:
                r = f"FAIL {ex}"
        else:
            r = verify_entry(n, e)
        print(f"{n}\t{r}")
        if str(r).startswith("FAIL"):
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
