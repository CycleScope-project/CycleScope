import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "provision"))
from tree_sha256 import tree_sha256  # noqa: E402

REG = json.loads((REPO / "configs" / "capture_tools.json").read_text())
TOOLS = REG["tools"]


def test_schema_version():
    assert REG["schema_version"] == "cyclescope.capture_tools.v1"


def test_every_entry_has_kind_and_location():
    for name, e in TOOLS.items():
        assert e["kind"] in ("git", "pypi_wheels", "download", "build",
                             "composite"), name


def test_fields_by_kind():
    for name, e in TOOLS.items():
        k = e["kind"]
        if k == "git":
            if e.get("pending_fields"):
                continue
            assert re.fullmatch(r"[0-9a-f]{40}", e["rev"]), name
        elif k == "pypi_wheels":
            assert (REPO / e["requirements"]).is_file(), name
            assert e["install"] in ("unpack",) or e["install"].startswith("unpack-ep:") or \
                e["install"].startswith("pip-target:"), name
        elif k == "download":
            assert e.get("manual") or e.get("url"), name
        elif k == "build":
            if e.get("pending_fields"):
                continue
            assert e.get("toolchain") in ("h100", "b300", "host"), name
            assert e.get("device"), name
            assert (REPO / e["script"]).is_file(), name
            assert e.get("outputs"), f"{name}: build entry needs outputs"
        for p in e.get("patches", []):
            assert (REPO / p).is_file(), f"{name}:{p}"
        assert isinstance(e.get("ignored_extras", []), list)
        for x in e.get("ignored_extras", []):
            assert x and ".." not in x.split("/"), f"{name}:{x}"


def test_inputs_acyclic():
    deps = {n: e.get("inputs", []) for n, e in TOOLS.items()}
    state = {}

    def visit(n):
        if state.get(n) == "visiting":
            raise AssertionError(f"cycle at {n}")
        if state.get(n) == "done":
            return
        state[n] = "visiting"
        for d in deps.get(n, []):
            assert d in TOOLS, f"{n}: unknown input {d}"
            visit(d)
        state[n] = "done"

    for n in deps:
        visit(n)


def test_tree_sha256(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("hi")
    (tmp_path / "b.txt").write_text("yo")
    (tmp_path / "link").symlink_to(tmp_path / "a")
    (tmp_path / "a" / "__pycache__").mkdir()
    (tmp_path / "a" / "__pycache__" / "x.pyc").write_text("junk")
    h1 = tree_sha256(tmp_path)
    h2 = tree_sha256(tmp_path)
    assert h1 == h2 and len(h1) == 64
    (tmp_path / "b.txt").write_text("yo!")
    assert tree_sha256(tmp_path) != h1


def test_lock_lines_parse():
    import re
    for name, e in TOOLS.items():
        if e["kind"] != "pypi_wheels":
            continue
        for line in (REPO / e["requirements"]).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            assert re.match(r"[\w.-]+==\S+\s+--hash=sha256:[0-9a-f]{64}$",
                            line), f"{name}: {line}"


import re  # noqa: E402


def test_overlay_sources_exist_and_match_sha():
    import hashlib
    for name, e in TOOLS.items():
        for rel, ov in (e.get("overlay") or {}).items():
            src = REPO / ov["src"]
            assert src.is_file(), f"{name}: overlay {rel} src missing {ov['src']}"
            got = hashlib.sha256(src.read_bytes()).hexdigest()
            assert got == ov["sha256"], f"{name}: overlay {rel} sha mismatch"


def test_source_revision_git(tmp_path, monkeypatch):
    """_source_revision returns (sha, dirty) from a real git repo."""
    import subprocess
    from cyclescope.capture import driver
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"],
                   check=True)
    (repo / "f").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "i"], check=True)
    monkeypatch.setattr(driver, "REPO", repo)
    sha, dirty = driver._source_revision()
    assert sha and len(sha) >= 7
    assert dirty is False
    (repo / "f").write_text("y")
    sha2, dirty2 = driver._source_revision()
    assert sha2 == sha and dirty2 is True


def test_verify_entry_fails_closed(tmp_path, monkeypatch):
    """Absent entry dir and build entries without outputs are FAIL, not ok."""
    import provision_tools as pt
    monkeypatch.setattr(pt, "ARTIFACTS", tmp_path)
    build = {"kind": "build", "outputs": [{"path": "x", "sha256": ""}]}
    assert pt.verify_entry("t", build).startswith("FAIL missing dir")
    (tmp_path / "tools" / "t").mkdir(parents=True)
    assert pt.verify_entry("t", {"kind": "build", "outputs": []}) == \
        "FAIL no outputs recorded"
    assert pt.verify_entry("t", build) == "missing ['x']"
    (tmp_path / "tools" / "t" / "x").write_text("")
    assert pt.verify_entry("t", build) == "ok"
    assert pt.verify_entry("m", {"kind": "download", "manual": True}) == \
        "manual"
    assert pt.verify_entry("g", {"kind": "git"}).startswith("FAIL missing dir")


def test_ensure_link_replaces_dangling(tmp_path):
    """A dangling/stale runner link is re-pointed; a real dir is kept."""
    from cyclescope.capture.driver import _ensure_link
    target = tmp_path / "src"
    target.mkdir()
    link = tmp_path / "tp" / "cutlass"
    link.parent.mkdir()
    link.symlink_to(tmp_path / "gone")
    _ensure_link(link, target)
    assert link.resolve() == target.resolve()
    other = tmp_path / "other"
    other.mkdir()
    _ensure_link(link, other)
    assert link.resolve() == other.resolve()
    real = tmp_path / "real"
    real.mkdir()
    _ensure_link(real, target)
    assert not real.is_symlink()


def test_zip_unpack_keeps_executable_bits(tmp_path):
    import importlib.util
    import zipfile
    spec = importlib.util.spec_from_file_location(
        "provision_tools", REPO / "scripts/provision/provision_tools.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    arc = tmp_path / "w.whl"
    with zipfile.ZipFile(arc, "w") as zf:
        exe = zipfile.ZipInfo("pkg/bin/tool")
        exe.external_attr = 0o755 << 16
        zf.writestr(exe, "#!/bin/sh\n")
        plain = zipfile.ZipInfo("pkg/data.txt")
        plain.external_attr = 0o644 << 16
        zf.writestr(plain, "x")
    mod._unpack(arc, tmp_path / "out", "zip")
    assert (tmp_path / "out/pkg/bin/tool").stat().st_mode & 0o111
    assert not (tmp_path / "out/pkg/data.txt").stat().st_mode & 0o111
