"""Unit tests for tools/microbench/l2_merge/sass_check.py."""
import subprocess
import sys
from pathlib import Path

SASS_CHECK = Path(__file__).resolve().parents[2] / "tools/microbench/l2_merge/sass_check.py"
KERNEL = "_Z11scan_kernelILb0EEv10StreamArgs"


def sass(kernel_lines: str, extra: str = "") -> str:
    return ("\t\tFunction : " + KERNEL + "\n"
            "\t\t.headerflags    @\"EF_CUDA_SM90 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM90)\"\n"
            + kernel_lines + extra)


def inst(op: str, addr: int = 0) -> str:
    return f"        /*{addr:04x}*/                   {op} R0, desc[UR6][R0.64] ;\n"


def run(tmp_path: Path, text: str):
    f = tmp_path / "x.sass"
    f.write_text(text)
    return subprocess.run([sys.executable, str(SASS_CHECK), str(f)],
                          capture_output=True, text=True)


def test_pass_eight_ldg128(tmp_path):
    body = inst("BAR.SYNC") + "".join(
        inst(f"LDG.E.128.STRONG.GPU", i) for i in range(8)) + inst("LDG.E")
    p = run(tmp_path, sass(body))
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.startswith("PASS")
    assert "x8" in p.stdout.splitlines()[0]


def test_fail_wrong_count(tmp_path):
    body = "".join(inst("LDG.E.128.STRONG.GPU", i) for i in range(4))
    p = run(tmp_path, sass(body))
    assert p.returncode == 1
    assert p.stdout.startswith("FAIL")
    assert "x4" in p.stdout.splitlines()[0]


def test_fail_no_matching_kernel(tmp_path):
    text = "\t\tFunction : _Z12flush_kernelPK5uint4yPj\n" + inst("LDG.E.128.STRONG.GPU")
    p = run(tmp_path, text)
    assert p.returncode == 1
    assert p.stdout.startswith("FAIL")


def test_fail_empty_dump(tmp_path):
    p = run(tmp_path, "no functions here\n")
    assert p.returncode == 1
    assert p.stdout.startswith("FAIL")


def test_fail_two_ld_instantiations(tmp_path):
    body = "".join(inst("LDG.E.128.STRONG.GPU", i) for i in range(8))
    text = sass(body) + "\t\tFunction : _Z11scan_kernelILb0EEv9OtherArgs\n" + body
    p = run(tmp_path, text)
    assert p.returncode == 1
    assert p.stdout.startswith("FAIL")
