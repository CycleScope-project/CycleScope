from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "suites" / "hopper" / "runners" / "harness" / "suites"
DEVICE_PROFILES = [HARNESS / suite / "scripts" / "device_profile.sh" for suite in ("cutlass", "deepgemm", "flashinfer", "flashmla")]


def _run_check(script: Path, tmp_path: Path, gpu_name: str, profile: str) -> subprocess.CompletedProcess[str]:
    stub = tmp_path / "nvidia-smi"
    stub.write_text(f'#!/bin/sh\necho "{gpu_name}"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return subprocess.run(
        ["bash", "-c", 'source "$1"; delivery_check_gpu_profile "$2"', "_", str(script), profile],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize("script", DEVICE_PROFILES, ids=lambda p: p.parts[-3])
def test_model_mismatch_warns_and_continues(script: Path, tmp_path: Path) -> None:
    result = _run_check(script, tmp_path, "NVIDIA H200", "h100")
    assert result.returncode == 0
    assert result.stdout == "NVIDIA H200\n"
    assert "WARNING: expected h100, got NVIDIA H200" in result.stderr


@pytest.mark.parametrize("script", DEVICE_PROFILES, ids=lambda p: p.parts[-3])
def test_matching_model_is_silent(script: Path, tmp_path: Path) -> None:
    name = "NVIDIA H100 80GB HBM3"
    result = _run_check(script, tmp_path, name, "h100")
    assert result.returncode == 0
    assert result.stdout == name + "\n"
    assert "WARNING" not in result.stderr


@pytest.mark.parametrize("script", DEVICE_PROFILES, ids=lambda p: p.parts[-3])
def test_unknown_profile_still_fails(script: Path, tmp_path: Path) -> None:
    assert _run_check(script, tmp_path, "NVIDIA H200", "bogus").returncode == 2


def test_device_profile_scripts_identical() -> None:
    assert len({p.read_bytes() for p in DEVICE_PROFILES}) == 1


def _load_family():
    path = HARNESS / "common" / "hopper_gpu_family.py"
    spec = importlib.util.spec_from_file_location("hopper_gpu_family_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_h100_gpu_name_warns_on_mismatch(capsys: pytest.CaptureFixture[str]) -> None:
    assert _load_family().require_h100_gpu_name("NVIDIA H200") == "NVIDIA H200"
    assert "WARNING: expected NVIDIA H100, got NVIDIA H200" in capsys.readouterr().err


def test_require_h100_gpu_name_silent_on_match(capsys: pytest.CaptureFixture[str]) -> None:
    assert _load_family().require_h100_gpu_name("NVIDIA H100 80GB HBM3") == "NVIDIA H100 80GB HBM3"
    assert capsys.readouterr().err == ""
