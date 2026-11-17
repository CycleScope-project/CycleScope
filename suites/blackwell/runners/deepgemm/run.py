#!/usr/bin/env python3
"""B300S DeepGEMM 1SM/2SM cases. Image deep_gemm 2.6.1; no hopper-prototype capture."""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DG_PRINT_CONFIGS", "1")
os.environ.setdefault("DG_JIT_DUMP_SASS", "1")
_JC = Path(os.environ.get(
    "DG_JIT_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / "artifacts/corpus/deepgemm/jit_cache")))
_JC.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DG_JIT_CACHE_DIR", str(_JC))

# Heavy deps (torch/deep_gemm/common) load lazily via _load_heavy() so `--list`
# can enumerate cases without torch, CUDA, or a deep_gemm install.
torch = None  # type: ignore[assignment]
deep_gemm = None  # type: ignore[assignment]
per_block_cast_to_fp8 = None  # type: ignore[assignment]
per_channel_cast_to_fp8 = None  # type: ignore[assignment]
per_token_cast_to_fp4 = None  # type: ignore[assignment]
per_token_cast_to_fp8 = None  # type: ignore[assignment]
cast_back_from_fp4 = None  # type: ignore[assignment]
write_result = None  # type: ignore[assignment]
write_text = None  # type: ignore[assignment]


def _deepgemm_root(spec):
    rel = (spec or {}).get("deepgemm_root")
    if not rel:
        return None
    base = Path(os.environ["CYCLESCOPE_ARTIFACT_ROOT"])
    return (base / rel).resolve()


_DG_IMPORT_ROOT = ...


def _load_heavy(dg_root=None):
    global torch, deep_gemm, _DG_IMPORT_ROOT
    global per_block_cast_to_fp8, per_channel_cast_to_fp8, per_token_cast_to_fp4, per_token_cast_to_fp8
    global cast_back_from_fp4
    global write_result, write_text
    if deep_gemm is not None:
        if dg_root != _DG_IMPORT_ROOT:
            raise RuntimeError(
                f"mixed deepgemm_root: already imported from {_DG_IMPORT_ROOT}, "
                f"case wants {dg_root}")
        return
    if dg_root is not None:
        sys.path.insert(0, str(dg_root))
    import torch
    import deep_gemm
    if dg_root is not None:
        imported = Path(deep_gemm.__file__).resolve()
        if not str(imported).startswith(str(dg_root) + os.sep):
            raise RuntimeError(
                f"deep_gemm imported from {imported}, expected under {dg_root}")
    _DG_IMPORT_ROOT = dg_root
    from deep_gemm.utils import (
        per_block_cast_to_fp8,
        per_channel_cast_to_fp8,
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )
    from deep_gemm.utils.math import cast_back_from_fp4

    from common import write_result, write_text

SUITE = "deepgemm"
DENSE = (128, 4096, 7168)
DENSE_1C = (128, 2112, 7168)  # odd N-tiles @128 → cluster_n=2 illegal
GROUPED = (4, 8192, 4096, 4096)
GROUPED_1C = (4, 8192, 2112, 4096)
MASKED = (6, 1024, 4096, 4096)
MASKED_1C = (6, 1024, 2112, 4096)
KGROUP = (4, 4096, 7168, 8192)
KGROUP_1C = (4, 2112, 2112, 8192)  # odd M and N tiles
NUMERICS_MAX_DIFF = 0.05
_LAST = {}
# Per-run shape overrides for delivery300 spec cases; always empty for built-ins.
_OVERRIDE = {}



def _fd_capture(fn):
    sys.stdout.flush()
    sys.stderr.flush()
    r, w = os.pipe()
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(w, 1)
    os.dup2(w, 2)
    os.close(w)
    err = None
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as e:
        err = e
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(saved_out, 1)
    os.dup2(saved_err, 2)
    os.close(saved_out)
    os.close(saved_err)
    with os.fdopen(r, "r", errors="replace") as f:
        text = f.read()
    return text, err


def _latest_sass() -> str:
    files = list(Path(os.environ["DG_JIT_CACHE_DIR"]).rglob("kernel.sass"))
    if not files:
        return ""
    try:
        p = max(files, key=lambda x: x.stat().st_mtime)
    except (ValueError, FileNotFoundError):
        return ""
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def _parse(log: str, sass: str) -> dict:
    cms = [int(x) for x in re.findall(r"cluster_m=(\d+)", log)]
    cns = [int(x) for x in re.findall(r"cluster_n=(\d+)", log)]
    nspc = [int(x) for x in re.findall(r"num_sms_per_cluster=(\d+)", log)]
    cluster = 1
    if cms and cns:
        cluster = cms[-1] * cns[-1]
    elif nspc:
        cluster = nspc[-1]
    has_g2 = "cta_group::2" in sass
    has_g1 = "cta_group::1" in sass
    ucga = "UCGABAR" in sass
    mangled_2 = bool(re.search(r"Lj2ELb[01]ELj2E", sass))
    use_2 = has_g2 or ucga or mangled_2
    bits = [
        f"kNumMulticast={cluster}",
        f"cluster_m={cms[-1] if cms else '?'}",
        f"cluster_n={cns[-1] if cns else '?'}",
        f"num_sms_per_cluster={nspc[-1] if nspc else '?'}",
    ]
    if has_g2:
        bits.append("cta_group::2")
    if has_g1:
        bits.append("cta_group::1")
    if ucga:
        bits.append("UCGABAR")
    if mangled_2:
        bits.append("sass_kNumMulticast=2")
    bits.append(f"use_2cta_instrs={use_2}")
    cfg = [ln for ln in log.splitlines() if "GemmConfig" in ln or "cluster_m=" in ln]
    if cfg:
        bits.append("PRINT_CONFIGS:" + cfg[-1][:400])
    ok_2c = cluster == 2 and ("kNumMulticast=2" in " ".join(bits) or use_2)
    return {"cluster": cluster, "cta_group": cluster, "proof": " ".join(bits), "ok_2c": ok_2c}



def _fp8_a(x, gran=128):
    return per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=gran)


def _fp8_b(x, gran=128):
    return per_block_cast_to_fp8(x, use_ue8m0=True, gran_k=gran)


def _fp4(x, gran=32):
    return per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=gran)


def _stack(xs, fn):
    xs2, ss = zip(*[fn(t) for t in xs])
    return torch.stack(xs2), torch.stack(ss)


def _tuple_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    parts = tuple(int(x) for x in raw.split(","))
    if len(parts) != len(default):
        raise RuntimeError(f"{name} wants {len(default)} ints, got {raw!r}")
    return parts


def _case_id():
    return os.environ.get("DG_CASE_ID", "")


def _invoke(operator, *args, reset_output=None, **kwargs):
    result = None
    for _ in range(int(_OVERRIDE.get("calls", 1))):
        if reset_output is not None:
            reset_output()
        result = operator(*args, **kwargs)
    return result


def _is_hold(cid=""):
    return bool(_OVERRIDE.get("heldout"))


def _dense_shape(cta):
    if "dense" in _OVERRIDE:
        return tuple(_OVERRIDE["dense"])
    return DENSE_1C if cta == 1 else DENSE


def _masked_shape(cta):
    if "masked" in _OVERRIDE:
        return tuple(_OVERRIDE["masked"])
    return MASKED_1C if cta == 1 else MASKED


def _mg_shape(cta):
    if "grouped" in _OVERRIDE:
        return tuple(_OVERRIDE["grouped"])
    if cta == 1:
        return _tuple_env("DG_GROUPED_1C", GROUPED_1C)
    return _tuple_env("DG_GROUPED", GROUPED)


def _kg_shape(cta):
    if "kgroup" in _OVERRIDE:
        return tuple(_OVERRIDE["kgroup"])
    if cta == 1:
        return _tuple_env("DG_KGROUP_1C", KGROUP_1C)
    return _tuple_env("DG_KGROUP", KGROUP)


def _should_check():
    # Tracer injection would also record the reference GEMM; numerics run in a no-tracer pass.
    if os.environ.get("CUDA_INJECTION64_PATH"):
        return False
    if _OVERRIDE.get("full_mode"):
        return True
    if os.environ.get("DG_CHECK_NUMERICS", "") == "1":
        return True
    return _is_hold()


def _dump_dir(cid=""):
    root = os.environ.get("DG_DUMP_DIR", "").strip()
    if not root:
        return None
    p = Path(root) / (cid or _case_id() or "case")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dequant(x, scale):
    if x.dtype == torch.int8:
        shape = tuple(x.shape)
        restored = cast_back_from_fp4(
            x.reshape(-1, x.shape[-1]),
            scale.reshape(-1, scale.shape[-1]),
            gran_k=32,
        )
        return restored.reshape(*shape[:-1], shape[-1] * 2)
    xf = x.to(torch.float32)
    s = scale.to(torch.float32)
    while s.ndim < xf.ndim:
        s = s.unsqueeze(0)
    if s.shape == xf.shape:
        return xf * s
    for dim in range(xf.ndim):
        if s.shape[dim] == xf.shape[dim]:
            continue
        if s.shape[dim] == 1:
            expand = list(s.shape)
            expand[dim] = xf.shape[dim]
            s = s.expand(*expand)
            continue
        if xf.shape[dim] % s.shape[dim] == 0:
            s = s.repeat_interleave(xf.shape[dim] // s.shape[dim], dim=dim)
        else:
            raise RuntimeError(f"cannot broadcast scale {tuple(scale.shape)} onto {tuple(x.shape)}")
    if tuple(s.shape) != tuple(xf.shape):
        s = s.expand_as(xf)
    return xf * s


def _calc_diff(x, y):
    x = x.detach().to(torch.float64).reshape(-1)
    y = y.detach().to(torch.float64).reshape(-1)
    denom = (x * x + y * y).sum().clamp_min(1e-30)
    return float(1 - 2 * (x * y).sum() / denom)


def _verify_full_output(kind, output, reference, tensors):
    diff = _calc_diff(output.float(), reference.float())
    payload = {"calc_diff": diff, "threshold": NUMERICS_MAX_DIFF,
               "pass": bool(diff < NUMERICS_MAX_DIFF)}
    print(f"NUMERICS {kind} calc_diff={diff:.6g} pass={payload['pass']}",
          flush=True)
    _record_numerics(kind, payload, {
        **tensors, "output": output, "reference": reference,
    })
    if not payload["pass"]:
        raise RuntimeError(f"numerics failed calc_diff={diff}")


def _verify_grouped_output(kind, a, b, output, layout, groups,
                           masked_m=None):
    a_f = _dequant(*a) if isinstance(a, tuple) else a.float()
    b_f = _dequant(*b) if isinstance(b, tuple) else b.float()
    reference = torch.zeros_like(output, dtype=torch.float32)
    if masked_m is None:
        for group in range(groups):
            rows = layout == group
            reference[rows] = a_f[rows] @ b_f[group].T
    else:
        for group in range(groups):
            valid = int(masked_m[group].item())
            if valid:
                reference[group, :valid] = a_f[group, :valid] @ b_f[group].T
    _verify_full_output(kind, output, reference, {
        "a": a_f, "b": b_f, "layout": layout,
    })


def _device_identity():
    if not torch.cuda.is_available():
        return {"cuda": False}
    p = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "major": int(p.major),
        "minor": int(p.minor),
        "multi_processor_count": int(p.multi_processor_count),
        "uuid": str(getattr(p, "uuid", "")),
        "total_memory": int(p.total_memory),
    }


def _record_numerics(kind, payload, tensors):
    _LAST["numerics"] = payload
    _LAST["kind"] = kind
    dump = _dump_dir()
    if dump is None:
        return
    cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in tensors.items()}
    torch.save(cpu, dump / "inputs.pt")
    (dump / "numerics.json").write_text(json.dumps(payload, indent=2) + "\n")


def run_fp8(cta):
    m, n, k = _dense_shape(cta)
    meta = ("fp8", f"{m}x{n}x{k}", "fp8_gemm_nt")
    a, b = _fp8_a(torch.randn(m, k, device="cuda", dtype=torch.bfloat16)), _fp8_b(
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    )
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    _invoke(deep_gemm.fp8_gemm_nt, a, b, d)
    return meta




def run_bf16(cta):
    m, n, k = _dense_shape(cta)
    meta = ("bf16", f"{m}x{n}x{k}", "bf16_gemm_nt")
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    _invoke(deep_gemm.bf16_gemm_nt, a, b, d)
    return meta




def run_fp8fp4(cta):
    m, n, k = _dense_shape(cta)
    meta = ("fp8xfp4", f"{m}x{n}x{k}", "fp8_fp4_gemm_nt")
    a = _fp8_a(torch.randn(m, k, device="cuda", dtype=torch.bfloat16), 128)
    b = _fp4(torch.randn(n, k, device="cuda", dtype=torch.bfloat16), 32)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    _invoke(deep_gemm.fp8_fp4_gemm_nt, a, b, d,
            recipe_a=(1, 128), recipe_b=(1, 32))
    return meta




def run_mg_c(cta):
    g, mg, n, k = _mg_shape(cta)
    m = g * mg
    meta = ("fp8_grouped", f"g{g}_{m}x{n}x{k}", "m_grouped_fp8_fp4_gemm_nt_contiguous")
    a = _fp8_a(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    if _OVERRIDE.get("dtype") == "fp8fp4":
        b = _stack(torch.randn(g, n, k, device="cuda",
                               dtype=torch.bfloat16),
                   lambda value: _fp4(value, 32))
        call_options = {"recipe_a": (1, 128), "recipe_b": (1, 32)}
    else:
        b = _stack(torch.randn(g, n, k, device="cuda",
                               dtype=torch.bfloat16), _fp8_b)
        call_options = {}
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    layout = torch.empty(m, device="cuda", dtype=torch.int32)
    for i in range(g):
        layout[i * mg : (i + 1) * mg] = i
    _invoke(deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous, a, b, d,
            layout, **call_options)
    _LAST["num_groups"] = g
    _LAST["group_spec"] = {"g": g, "mg": mg, "n": n, "k": k, "m": m}
    if _should_check():
        a_f = _dequant(*a)
        b_f = _dequant(*b)
        d_ref = torch.empty(m, n, device=d.device, dtype=torch.float32)
        for i in range(g):
            rows = layout == i
            d_ref[rows] = a_f[rows] @ b_f[i].T
        diff = _calc_diff(d.float(), d_ref)
        payload = {
            "kind": "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "calc_diff": diff,
            "threshold": NUMERICS_MAX_DIFF,
            "pass": bool(diff < NUMERICS_MAX_DIFF),
            "num_groups": g,
            "unique_layout": int(torch.unique(layout).numel()),
            "a_scale": list(a[1].shape),
            "b_scale": list(b[1].shape),
        }
        print(f"NUMERICS mg_c calc_diff={diff:.6g} pass={payload['pass']} g={g}", flush=True)
        _record_numerics("mg_c", payload, {
            "a": a[0], "a_scale": a[1], "b": b[0], "b_scale": b[1],
            "layout": layout, "d": d, "d_ref": d_ref,
        })
        if not payload["pass"]:
            raise RuntimeError(f"numerics failed calc_diff={diff}")
    return meta




def run_mg_m(cta):
    g, expected_m, n, k = _masked_shape(cta)
    meta = ("fp8_masked", f"g{g}_em{expected_m}_{expected_m}x{n}x{k}", "m_grouped_fp8_fp4_gemm_nt_masked")
    a = _stack(torch.randn(g, expected_m, k, device="cuda", dtype=torch.bfloat16), _fp8_a)
    b = _stack(torch.randn(g, n, k, device="cuda", dtype=torch.bfloat16), _fp8_b)
    d = torch.empty(g, expected_m, n, device="cuda", dtype=torch.bfloat16)
    if _OVERRIDE.get("full_mode"):
        d.zero_()
    masked_m = torch.tensor([expected_m] * (g - 1) + [0], device="cuda", dtype=torch.int32)
    _invoke(deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked, a, b, d, masked_m,
            expected_m)
    if _OVERRIDE.get("full_mode"):
        _verify_grouped_output("mg_m", a, b, d, None, g, masked_m)
    return meta




def run_kg(cta):
    g, m, n, k_per = _kg_shape(cta)
    if g < 2:
        raise RuntimeError(f"k-grouped requires >=2 groups, got g={g}")
    ks = [k_per] * g
    sum_k = sum(ks)
    meta = ("fp8_kgrouped", f"g{g}_{m}x{n}xk{k_per}", "k_grouped_fp8_gemm_tn_contiguous")
    a = per_channel_cast_to_fp8(torch.randn(sum_k, m, device="cuda", dtype=torch.bfloat16), use_ue8m0=True, gran_k=128)
    b = per_channel_cast_to_fp8(torch.randn(sum_k, n, device="cuda", dtype=torch.bfloat16), use_ue8m0=True, gran_k=128)
    d = torch.zeros(g, m, n, device="cuda", dtype=torch.float)
    layout = torch.tensor(ks, device="cuda", dtype=torch.int32)
    _invoke(deep_gemm.k_grouped_fp8_gemm_tn_contiguous, a, b, d, ks, layout,
            d, recipe=(1, 1, 128), reset_output=lambda: d.zero_())
    _LAST["num_groups"] = g
    _LAST["group_spec"] = {"g": g, "m": m, "n": n, "k_per": k_per, "ks": ks, "sum_k": sum_k}
    if _should_check():
        a_f = _dequant(*a)
        b_f = _dequant(*b)
        d_ref = torch.zeros(g, m, n, device=d.device, dtype=torch.float32)
        off = 0
        for i, kk in enumerate(ks):
            d_ref[i] = a_f[off:off + kk].T @ b_f[off:off + kk]
            off += kk
        diff = _calc_diff(d.float(), d_ref)
        payload = {
            "kind": "k_grouped_fp8_gemm_tn_contiguous",
            "calc_diff": diff,
            "threshold": NUMERICS_MAX_DIFF,
            "pass": bool(diff < NUMERICS_MAX_DIFF),
            "num_groups": g,
            "ks": ks,
            "a_scale": list(a[1].shape),
            "b_scale": list(b[1].shape),
        }
        print(f"NUMERICS kg calc_diff={diff:.6g} pass={payload['pass']} g={g} ks={ks}", flush=True)
        _record_numerics("kg", payload, {
            "a": a[0], "a_scale": a[1], "b": b[0], "b_scale": b[1],
            "layout": layout, "d": d, "d_ref": d_ref,
        })
        if not payload["pass"]:
            raise RuntimeError(f"numerics failed calc_diff={diff}")
    return meta




def _mn_major_bf16(x):
    # (m,k) K-major -> MN-major operand, same logical shape, MN-major storage
    # (upstream tests/generators.py:318 a.T.contiguous().T).
    return x.T.contiguous().T


def _mn_major_fp8(pair):
    # upstream generators.py:301/318: quantize K-major then repack data to
    # MN-major storage keeping (m,k) shape; SF unchanged.
    return (pair[0].mT.contiguous().mT, pair[1])


def _mn_major_fp4(pair):
    from deep_gemm.utils import transpose_packed_fp4
    return (transpose_packed_fp4(pair[0]).T, pair[1])


def _mn_major_grouped_fp4(pair):
    from deep_gemm.utils import transpose_packed_fp4
    transposed = torch.stack(
        [transpose_packed_fp4(group) for group in pair[0]])
    return (transposed.mT, pair[1])


def _run_dense(dtype, layout, cta, accc=False):
    # layout letters: A: 'n'=K-major, 't'=MN-major; B: 't'=K-major, 'n'=MN-major
    # (upstream csrc/apis/gemm.hpp:126-163, tests/test_bf16.py:26-37).
    m, n, k = _dense_shape(cta)
    base_fn = "fp8_fp4" if dtype == "fp8fp4" else dtype
    name = f"{base_fn}_gemm_{layout}"
    meta = (dtype, f"{m}x{n}x{k}@{layout}", name + ("+c" if accc else ""))
    c_acc = None
    if dtype == "bf16":
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        a_ref, b_ref = a.float(), b.float()
        # Alias calls (nn/tn/tt): build the MN-major operand
        # (x.T.contiguous().T), then pass its .T view (generators.py:318,
        # test_bf16.py:33-37).
        if layout[0] == "t":
            a = _mn_major_bf16(a).T
        if layout[1] == "n":
            b = _mn_major_bf16(b).T
        d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        if accc:
            c_acc = torch.randn(m, n, device="cuda", dtype=d.dtype)
            _invoke(getattr(deep_gemm, name), a, b, d, c=c_acc)
        else:
            _invoke(getattr(deep_gemm, name), a, b, d)
    else:  # fp8 or fp8fp4
        a = _fp8_a(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
        if dtype == "fp8fp4":
            b_raw = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            b = _fp4(b_raw, 32)
            kw = dict(recipe_a=(1, 128), recipe_b=(1, 32))
        else:
            b = _fp8_b(torch.randn(n, k, device="cuda", dtype=torch.bfloat16))
            kw = {}
        a_ref, b_ref = _dequant(*a), _dequant(*b)
        if layout[0] == "t":
            a = _mn_major_fp8(a)
            a = (a[0].T, a[1].T)
        if layout[1] == "n":
            b = _mn_major_fp4(b) if dtype == "fp8fp4" else _mn_major_fp8(b)
            b = (b[0].T, b[1].T)

        d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        if accc:
            c_acc = torch.randn(m, n, device="cuda", dtype=d.dtype)
            _invoke(getattr(deep_gemm, name), a, b, d, c=c_acc, **kw)
        else:
            _invoke(getattr(deep_gemm, name), a, b, d, **kw)
    if _OVERRIDE.get("full_mode"):
        reference = a_ref @ b_ref.T
        if c_acc is not None:
            reference = reference + c_acc.float()
        _verify_full_output(f"dense_{dtype}_{layout}", d, reference, {
            "a": a_ref, "b": b_ref, **({"c": c_acc} if c_acc is not None else {}),
        })
    return meta


def _dense_runner(dtype, layout, accc=False):
    def f(cta):
        return _run_dense(dtype, layout, cta, accc)
    return f


_RUNNERS = {
    "fp8": run_fp8,
    "bf16": run_bf16,
    "fp8fp4": run_fp8fp4,
    "mg_c": run_mg_c,
    "mg_m": run_mg_m,
    "kg": run_kg,
}
for _dt in ("fp8", "bf16", "fp8fp4"):
    for _lo in ("nt", "nn", "tn", "tt"):
        _RUNNERS.setdefault(f"{_dt}_{_lo}", _dense_runner(_dt, _lo))
_RUNNERS["fp8_accc"] = _dense_runner("fp8", "nt", accc=True)
_RUNNERS["bf16_accc"] = _dense_runner("bf16", "nt", accc=True)


def run_mg_bf16(cta):
    g, mg, n, k = _mg_shape(cta)
    m = g * mg
    meta = ("bf16_grouped", f"g{g}_{m}x{n}x{k}", "m_grouped_bf16_gemm_nt_contiguous")
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(g, n, k, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    layout = torch.empty(m, device="cuda", dtype=torch.int32)
    for i in range(g):
        layout[i * mg:(i + 1) * mg] = i
    _invoke(deep_gemm.m_grouped_bf16_gemm_nt_contiguous, a, b, d, layout)
    if _OVERRIDE.get("full_mode"):
        _verify_grouped_output("mg_bf16", a, b, d, layout, g)
    return meta


def run_mg_bf16_nn(cta):  # nn: B MN-major -> pass .T view (test_alias 约定)
    g, mg, n, k = _mg_shape(cta)
    m = g * mg
    meta = ("bf16_grouped_nn", f"g{g}_{m}x{n}x{k}", "m_grouped_bf16_gemm_nn_contiguous")
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(g, n, k, device="cuda", dtype=torch.bfloat16).mT.contiguous().mT
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    layout = torch.empty(m, device="cuda", dtype=torch.int32)
    for i in range(g):
        layout[i * mg:(i + 1) * mg] = i
    _invoke(deep_gemm.m_grouped_bf16_gemm_nn_contiguous, a, b.mT, d, layout)
    if _OVERRIDE.get("full_mode"):
        _verify_grouped_output("mg_bf16_nn", a, b, d, layout, g)
    return meta


def run_mg_fp8_nn(cta):
    g, mg, n, k = _mg_shape(cta)
    m = g * mg
    meta = ("fp8_grouped_nn", f"g{g}_{m}x{n}x{k}",
            "m_grouped_fp8_fp4_gemm_nn_contiguous")
    a = _fp8_a(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    b = _stack(torch.randn(g, n, k, device="cuda", dtype=torch.bfloat16), _fp8_b)
    b_reference = _dequant(*b)
    b = _mn_major_fp8(b)
    b = (b[0].mT, b[1].mT)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    layout = torch.empty(m, device="cuda", dtype=torch.int32)
    for i in range(g):
        layout[i * mg:(i + 1) * mg] = i
    _invoke(deep_gemm.m_grouped_fp8_fp4_gemm_nn_contiguous, a, b, d, layout)
    if _OVERRIDE.get("full_mode"):
        _verify_grouped_output("mg_fp8_nn", a, b_reference, d, layout, g)
    _LAST["num_groups"] = g
    _LAST["group_spec"] = {"g": g, "mg": mg, "n": n, "k": k, "m": m}
    return meta


def run_mg_fp8fp4_nn(cta):
    g, mg, n, k = _mg_shape(cta)
    m = g * mg
    meta = ("fp8fp4_grouped_nn", f"g{g}_{m}x{n}x{k}",
            "m_grouped_fp8_fp4_gemm_nn_contiguous")
    a = _fp8_a(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    b = _stack(torch.randn(g, n, k, device="cuda", dtype=torch.bfloat16),
               lambda value: _fp4(value, 32))
    b_reference = _dequant(*b)
    b = _mn_major_grouped_fp4(b)
    b = (b[0].mT, b[1].mT)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    layout = torch.empty(m, device="cuda", dtype=torch.int32)
    for i in range(g):
        layout[i * mg:(i + 1) * mg] = i
    _invoke(deep_gemm.m_grouped_fp8_fp4_gemm_nn_contiguous, a, b, d,
            layout, recipe_a=(1, 128), recipe_b=(1, 32))
    if _OVERRIDE.get("full_mode"):
        _verify_grouped_output("mg_fp8fp4_nn", a, b_reference, d, layout, g)
    _LAST["num_groups"] = g
    _LAST["group_spec"] = {"g": g, "mg": mg, "n": n, "k": k, "m": m}
    return meta


def run_mgm_bf16(cta):
    g, em, n, k = _masked_shape(cta)
    meta = ("bf16_masked", f"g{g}_em{em}", "m_grouped_bf16_gemm_nt_masked")
    a = torch.randn(g, em, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(g, n, k, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(g, em, n, device="cuda", dtype=torch.bfloat16)
    if _OVERRIDE.get("full_mode"):
        d.zero_()
    masked_m = torch.tensor([em] * (g - 1) + [0], device="cuda", dtype=torch.int32)
    _invoke(deep_gemm.m_grouped_bf16_gemm_nt_masked, a, b, d, masked_m, em)
    if _OVERRIDE.get("full_mode"):
        _verify_grouped_output("mgm_bf16", a, b, d, None, g, masked_m)
    return meta


def run_kg_bf16(cta):
    g, m, n, k_per = _kg_shape(cta)
    ks = [k_per] * g
    meta = ("bf16_kgrouped", f"g{g}_{m}x{n}xk{k_per}", "k_grouped_bf16_gemm_tn_contiguous")
    a = torch.randn(sum(ks), m, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(sum(ks), n, device="cuda", dtype=torch.bfloat16)
    d = torch.zeros(g, m, n, device="cuda", dtype=torch.float)
    layout = torch.tensor(ks, device="cuda", dtype=torch.int32)
    c = d
    _invoke(deep_gemm.k_grouped_bf16_gemm_tn_contiguous, a, b, d, ks, layout,
            c, reset_output=lambda: d.zero_())
    if _OVERRIDE.get("full_mode"):
        ref = torch.zeros(g, m, n, device="cuda", dtype=torch.float32)
        offset = 0
        for group, length in enumerate(ks):
            ref[group] = a[offset:offset + length].float().T @ \
                b[offset:offset + length].float()
            offset += length
        _verify_full_output("kg_bf16", d, ref, {"a": a, "b": b})
    return meta


def run_skiphead(cta):
    m, n, k = _dense_shape(cta)
    meta = ("fp8_skiphead", f"{m}x{n}x{k}", "fp8_gemm_nt_skip_head_mid")
    a = _fp8_a(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    b = _fp8_b(torch.randn(n, k, device="cuda", dtype=torch.bfloat16))
    # upstream test_attention.py:34-56: fixed splits (128,64,128), d gets
    # zero mid padding; n must satisfy n%(left+right)==0
    head_splits = (128, 64, 128)
    lh = head_splits[0] + head_splits[2]
    d_raw = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    nh = max(n // lh, 1)
    dl = d_raw[:, :nh * head_splits[0]].view(m, nh, head_splits[0])
    dr = d_raw[:, -nh * head_splits[2]:].view(m, nh, head_splits[2])
    dm = torch.zeros((m, nh, head_splits[1]), dtype=d_raw.dtype, device=d_raw.device)
    d = torch.cat([dl, dm, dr], dim=2).view(m, -1)
    _invoke(deep_gemm.fp8_gemm_nt_skip_head_mid, a, b, d, head_splits)
    if _OVERRIDE.get("full_mode"):
        a_f, b_f = _dequant(*a), _dequant(*b)
        ref = a_f @ b_f.T
        left, mid, right = head_splits
        num_heads = n // (left + right)
        ref = ref.view(m, num_heads, -1)
        ref = torch.cat((
            ref[:, :, :left],
            torch.zeros((m, num_heads, mid), device=ref.device, dtype=ref.dtype),
            ref[:, :, -right:],
        ), dim=2).reshape(m, -1)
        _verify_full_output("skiphead", d, ref, {"a": a_f, "b": b_f})
    return meta


def run_einsum(cta):
    expr = _OVERRIDE.get("einsum", "bhr,hdr->bhd")
    dtype = _OVERRIDE.get("edtype", "fp8")
    dims = _OVERRIDE.get("einsum_dims", [4, 64, 128, 128])
    b, h, r, d_ = dims
    meta = (f"{dtype}_einsum", expr, f"{'fp8_' if dtype=='fp8' else ''}einsum[{expr}]")
    shapes = {"bhr,hdr->bhd": ((b, h, r), (h, d_, r), (b, h, d_)),
              "bhd,hdr->bhr": ((b, h, d_), (h, d_, r), (b, h, r)),
              "bhd,bhr->hdr": ((b, h, d_), (b, h, r), (h, d_, r)),
              "bmk,bnk->mn": ((b, h, d_), (b, r, d_), (h, r))}
    sa, sb, sz = shapes[expr]
    if dtype == "fp8":
        a_raw = torch.randn(*sa, device="cuda", dtype=torch.bfloat16)
        b_raw = torch.randn(*sb, device="cuda", dtype=torch.bfloat16)
        if expr == "bhd,bhr->hdr":
            xa = per_channel_cast_to_fp8(
                a_raw.view(b, -1), use_ue8m0=True)
            xb = per_channel_cast_to_fp8(
                b_raw.view(b, -1), use_ue8m0=True)
            xa = (xa[0].view(*sa), xa[1].view((b + 127) // 128, h, d_))
            xb = (xb[0].view(*sb), xb[1].view((b + 127) // 128, h, r))
            z0 = torch.randn(*sz, device="cuda", dtype=torch.float32) * 10
            z = z0.clone()
            _invoke(
                deep_gemm.fp8_einsum, expr, xa, xb, z, z,
                recipe=(1, 1, 128), reset_output=lambda: z.copy_(z0))
        else:
            def cast_tokens(x):
                values, scales = per_token_cast_to_fp8(
                    x.reshape(-1, x.shape[-1]), use_ue8m0=True)
                return (
                    values.reshape(*x.shape),
                    scales.reshape(*x.shape[:-1], scales.shape[-1]),
                )

            def cast_blocks_by_head(x):
                pairs = [per_block_cast_to_fp8(
                    x[index], use_ue8m0=True) for index in range(x.shape[0])]
                return (torch.stack([pair[0] for pair in pairs]),
                        torch.stack([pair[1] for pair in pairs]))

            xa = cast_tokens(a_raw)
            xb = cast_blocks_by_head(b_raw)
            z = torch.empty(*sz, device="cuda", dtype=torch.bfloat16)
            _invoke(deep_gemm.fp8_einsum, expr, xa, xb, z)
        result = z
        a_ref, b_ref = a_raw.float(), b_raw.float()
    else:
        xa = torch.randn(*sa, device="cuda", dtype=torch.bfloat16)
        xb = torch.randn(*sb, device="cuda", dtype=torch.bfloat16)
        z = torch.empty(*sz, device="cuda", dtype=torch.bfloat16)
        _invoke(deep_gemm.einsum, expr, xa, xb, z)
        result = z
        a_ref, b_ref = xa.float(), xb.float()
    if _OVERRIDE.get("full_mode"):
        if dtype == "fp8" and expr == "bhd,bhr->hdr":
            reference = z0 + torch.einsum(expr, a_ref, b_ref)
        else:
            reference = torch.einsum(expr, a_ref, b_ref)
        _verify_full_output("einsum", result, reference, {
            "a": a_ref, "b": b_ref,
        })
    return meta


def run_mqa(cta):
    seq_len, seq_len_kv = int(_OVERRIDE.get("mqa", [64, 4096])[0]), int(_OVERRIDE.get("mqa", [64, 4096])[1])
    num_heads, head_dim = 64, 128
    meta = ("fp8_mqa", f"s{seq_len}xk{seq_len_kv}", "fp8_mqa_logits")
    from deep_gemm.utils.math import per_custom_dims_cast_to_fp8
    q_raw = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_raw = torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    # upstream test_attention.py:186-198: fp8 fmt -> q plain cast, kv custom-dim
    q = (q_raw.to(torch.float8_e4m3fn), None)
    kv = per_custom_dims_cast_to_fp8(kv_raw, (0, ), False)
    w = torch.randn(seq_len, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    ke = torch.full((seq_len,), seq_len_kv, dtype=torch.int, device="cuda")
    logits = _invoke(deep_gemm.fp8_fp4_mqa_logits, q=q, kv=kv, weights=w,
                     cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                     clean_logits=True, max_seqlen_k=0)
    if _OVERRIDE.get("full_mode"):
        scores = torch.einsum("mhd,nd->hmn", q_raw.float(), kv_raw.float())
        reference = torch.einsum("hmn,hm->mn", scores.relu(),
                                 w.transpose(0, 1).contiguous())
        _verify_full_output("mqa_logits", logits, reference, {
            "q": q_raw, "kv": kv_raw, "weights": w,
        })
    return meta


def _ref_paged_mqa_logits(q, kv_cache, weights, context_lens,
                          block_table, max_model_len):
    batch_size, next_n, num_heads, head_dim = q.shape
    if next_n != 1:
        raise ValueError(f"paged MQA reference expects next_n=1, got {next_n}")
    pages = kv_cache[block_table]
    kx = pages.permute(0, 3, 4, 1, 2).reshape(
        batch_size, 1, head_dim, -1)
    qx = q.permute(0, 2, 1, 3)
    scores = torch.matmul(qx, kx).float()
    key_offsets = torch.arange(max_model_len, device=q.device)
    active = key_offsets[None, :] < context_lens[:, None]
    scores = torch.where(
        active[:, None, None, :], scores, float("-inf"))
    scores = torch.relu(scores) * weights[:, :, None, None]
    scores = scores.sum(dim=1).squeeze(1)
    return torch.where(
        key_offsets[None, :] <= context_lens[:, None], scores,
        float("-inf"))


def _cast_paged_mqa_kv_to_fp8(kv_raw):
    num_blocks, block_kv, num_heads, head_dim = kv_raw.shape
    if num_heads != 1:
        raise ValueError(f"paged MQA KV cache must have one head, got {num_heads}")
    scale = kv_raw.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
    kv_fp8 = (kv_raw * (1.0 / scale)).to(torch.float8_e4m3fn)
    kv_simulated = (kv_fp8.float() * scale).to(kv_raw.dtype)
    fused = torch.empty(
        num_blocks, block_kv * (head_dim + 4),
        device=kv_raw.device, dtype=torch.uint8)
    fused[:, :block_kv * head_dim] = kv_fp8.reshape(
        num_blocks, block_kv * head_dim).view(torch.uint8)
    fused[:, block_kv * head_dim:] = scale.reshape(
        num_blocks, block_kv).view(torch.uint8)
    return fused.view(num_blocks, block_kv, 1, head_dim + 4), kv_simulated


def run_paged_mqa(cta):
    seq_len = int(_OVERRIDE.get("pmqa", [64, 4096, 64])[0])
    seq_len_kv = int(_OVERRIDE.get("pmqa", [64, 4096, 64])[1])
    block_kv = int(_OVERRIDE.get("pmqa", [64, 4096, 64])[2])
    num_heads, head_dim = 64, 128
    api = _OVERRIDE.get("pmqa_api", "fp8_fp4")
    meta = ("paged_mqa", f"s{seq_len}xk{seq_len_kv}xb{block_kv}",
            f"{api}_paged_mqa_logits")
    context_lens_ref = torch.full(
        (seq_len,), seq_len_kv, dtype=torch.int32, device="cuda")
    context_lens = context_lens_ref[:, None].contiguous()
    num_sms = deep_gemm.get_num_sms()
    meta_t = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_kv, num_sms)
    q_raw = torch.randn(
        seq_len, 1, num_heads, head_dim, device="cuda",
        dtype=torch.bfloat16)
    q = q_raw.to(torch.float8_e4m3fn)
    num_blocks = seq_len * ((seq_len_kv + block_kv - 1) // block_kv)
    kv_raw = torch.randn(
        num_blocks, block_kv, 1, head_dim, device="cuda",
        dtype=torch.bfloat16)
    kv, kv_simulated = _cast_paged_mqa_kv_to_fp8(kv_raw)
    w = torch.randn(seq_len, num_heads, device="cuda", dtype=torch.float32)
    kv_indices = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(seq_len, -1)
    if api == "fp8":
        logits = _invoke(
            deep_gemm.fp8_paged_mqa_logits,
            q, kv, w, context_lens, kv_indices, meta_t, seq_len_kv,
            clean_logits=False)
    else:
        logits = _invoke(
            deep_gemm.fp8_fp4_paged_mqa_logits,
            (q, None), kv, w, context_lens, kv_indices, meta_t,
            seq_len_kv, clean_logits=False, logits_dtype=torch.float32)
    if _OVERRIDE.get("full_mode"):
        ref = _ref_paged_mqa_logits(
            q_raw, kv_raw, w, context_lens_ref, kv_indices,
            seq_len_kv)
        simulated = _ref_paged_mqa_logits(
            q.to(torch.bfloat16), kv_simulated, w, context_lens_ref,
            kv_indices, seq_len_kv)
        logits = logits.reshape_as(ref).float()
        diff = _calc_diff(logits, ref)
        simulated_diff = _calc_diff(logits, simulated)
        print(
            f"NUMERICS paged_mqa_logits diff={diff:.6g} "
            f"simulated_diff={simulated_diff:.6g}", flush=True)
        if diff >= 1e-3 or simulated_diff >= 5e-6:
            raise RuntimeError(
                f"paged MQA numerics failed: diff={diff} "
                f"simulated_diff={simulated_diff}")
        _record_numerics("paged_mqa_logits", {
            "calc_diff": diff, "simulated_diff": simulated_diff,
            "threshold": 1e-3, "pass": True,
        }, {
            "q": q_raw, "kv": kv_raw, "weights": w,
            "reference": ref, "simulated_reference": simulated,
            "output": logits,
        })
    return meta


def _mqa_weights_padded(weights):
    # upstream test_attention.py to_mqa_weights: rows padded to 16B for TMA
    from deep_gemm.utils import ceil_div
    element_size = weights.element_size()
    stride = ceil_div(weights.size(1) * element_size, 16) * 16 // element_size
    storage = torch.empty((weights.size(0), stride), device=weights.device,
                          dtype=weights.dtype)
    result = storage[:, :weights.size(1)]
    result.copy_(weights)
    return result


def _mqa_ks_ke(seq_len, seq_len_kv, disable_cp):
    # upstream generate_ks_ke_tests: CP picks an arbitrary rank's two chunks
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    if disable_cp:
        ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (
            seq_len_kv - seq_len)
        return ks, ke
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    offsets = torch.arange(chunk_size, dtype=torch.int, device="cuda")
    ke = torch.cat([cp_id * chunk_size + offsets,
                    (cp_size * 2 - 1 - cp_id) * chunk_size + offsets])
    return ks, ke


def _quantize_mqa(x, fmt):
    # upstream quantize_mqa_q: returns kernel input + bf16 simulated copy
    is_mxfp4 = fmt == "mxfp4"
    head_dim = x.size(-1)
    x_fp, x_sf = (per_token_cast_to_fp4 if is_mxfp4 else per_token_cast_to_fp8)(
        x.view(-1, head_dim), use_ue8m0=True, gran_k=32,
        use_packed_ue8m0=True)
    if is_mxfp4:
        x_sim = cast_back_from_fp4(x_fp, x_sf, gran_k=32,
                                 use_packed_ue8m0=True).to(torch.bfloat16)
    else:
        from deep_gemm.utils import cast_back_from_fp8
        x_sim = cast_back_from_fp8(x_fp, x_sf, gran_k=32,
                                 use_packed_ue8m0=True).to(torch.bfloat16)
    return (x_fp.view(*x.shape[:-1], head_dim // 2 if is_mxfp4 else head_dim),
            x_sf.view(*x.shape[:-1])), x_sim.view(x.shape)


def _ref_mqa_logits(q, kv, weights, ks, ke):
    # upstream ref_mqa_logits: chunk along KV so the score tensor stays bounded
    seq_len_kv = kv.shape[0]
    seq_len = q.shape[0]
    q = q.float()
    k = kv.float()
    w = weights.transpose(0, 1).contiguous()
    kv_chunk = max(1, (256 * 1024 * 1024) // max(1, seq_len * q.shape[1] * 4))
    positions = torch.arange(0, seq_len_kv, device="cuda")
    logits = torch.empty((seq_len, seq_len_kv), dtype=torch.float, device="cuda")
    cost = torch.zeros((), dtype=torch.long, device="cuda")
    for n0 in range(0, seq_len_kv, kv_chunk):
        n1 = min(n0 + kv_chunk, seq_len_kv)
        score = torch.einsum("mhd,nd->hmn", q, k[n0:n1])
        chunk = torch.einsum("hmn,hm->mn", score.relu(), w)
        cols = positions[n0:n1]
        mask = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
        logits[:, n0:n1] = chunk.masked_fill(~mask, float("-inf"))
        cost += mask.sum()
    return logits, cost


def run_mqa_block_scaled(cta):
    seq_len, seq_len_kv = (int(x) for x in _OVERRIDE.get("mqa", [64, 4096]))
    fmt = _OVERRIDE.get("format", "mxfp4")
    if fmt not in ("mxfp4", "mxfp8"):
        raise RuntimeError(f"unsupported mqa format {fmt!r}")
    num_heads = int(_OVERRIDE.get("num_heads", 64))
    head_dim = int(_OVERRIDE.get("head_dim", 128))
    context_parallel = bool(_OVERRIDE.get("context_parallel", True))
    meta = (f"mqa_{fmt}",
            f"s{seq_len}xk{seq_len_kv}xh{num_heads}xd{head_dim}"
            f"xcp{int(context_parallel)}",
            "fp8_fp4_mqa_logits")
    # upstream test_attention.py test_mqa_logits (2.8.2): bf16 q/kv, padded
    # bf16 weights, ks/ke per CP flag, no schedule_meta
    q_raw = torch.randn(seq_len, num_heads, head_dim, device="cuda",
                        dtype=torch.bfloat16)
    kv_raw = torch.randn(seq_len_kv, head_dim, device="cuda",
                         dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device="cuda",
                          dtype=torch.bfloat16)
    kernel_weights = _mqa_weights_padded(weights)
    ks, ke = _mqa_ks_ke(seq_len, seq_len_kv, not context_parallel)
    q_in, q_simulated = _quantize_mqa(q_raw, fmt)
    kv_in, kv_simulated = _quantize_mqa(kv_raw, fmt)
    max_seqlen_k = int((ke - ks).max().item())
    logits = _invoke(deep_gemm.fp8_fp4_mqa_logits,
                     q=q_in, kv=kv_in, weights=kernel_weights,
                     cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                     max_seqlen_k=max_seqlen_k)
    if _OVERRIDE.get("full_mode"):
        assert logits.dtype == torch.bfloat16
        assert logits.size() == (seq_len, max_seqlen_k)
        ref_logits, _ = _ref_mqa_logits(
            q_raw, kv_raw, kernel_weights.float(), ks, ke)
        sim_logits, _ = _ref_mqa_logits(
            q_simulated, kv_simulated, kernel_weights.float(), ks, ke)
        dense = torch.full((seq_len, seq_len_kv), float("-inf"),
                           device="cuda")
        cols = torch.arange(seq_len_kv, device="cuda")[None, :]
        window = (cols >= ks[:, None]) & (cols < ke[:, None])
        valid = torch.arange(max_seqlen_k, device="cuda")[None, :] < (
            ke - ks)[:, None]
        dense[window] = logits[valid].float()
        ref_neg = ref_logits == float("-inf")
        if not torch.equal(dense == float("-inf"), ref_neg):
            raise RuntimeError("mqa logits -inf mask mismatch with reference")
        ref = ref_logits.masked_fill(ref_neg, 0)
        sim = sim_logits.masked_fill(ref_neg, 0)
        got = dense.masked_fill(ref_neg, 0)
        diff = _calc_diff(got, ref)
        sim_diff = _calc_diff(got, sim)
        ok = diff < 0.02 and sim_diff < 3e-5
        print(f"NUMERICS mqa_logits_{fmt} diff={diff:.6g} "
              f"simulated_diff={sim_diff:.6g} pass={ok}", flush=True)
        _record_numerics("mqa_logits_block_scaled", {
            "calc_diff": diff, "simulated_diff": sim_diff,
            "threshold": 0.02, "simulated_threshold": 3e-5, "pass": ok,
        }, {
            "q": q_raw, "kv": kv_raw, "weights": weights,
            "reference": ref, "simulated_reference": sim, "output": got,
        })
        if not ok:
            raise RuntimeError(
                f"mqa logits numerics failed diff={diff} "
                f"simulated_diff={sim_diff}")
    return meta


def run_hc(cta):
    m, n, k = _dense_shape(cta)
    num_splits = _OVERRIDE.get("num_splits")
    meta = ("hc_prenorm", f"{m}x{n}x{k}", "tf32_hc_prenorm_gemm")
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float32)
    d_shape = (m, n) if num_splits is None else (num_splits, m, n)
    s_shape = (m,) if num_splits is None else (num_splits, m)
    d = torch.empty(d_shape, device="cuda", dtype=torch.float32)
    s = torch.empty(s_shape, device="cuda", dtype=torch.float32)
    _invoke(deep_gemm.tf32_hc_prenorm_gemm, a, b, d, s,
            num_splits=num_splits,
            reset_output=lambda: (d.zero_(), s.zero_()))
    if _OVERRIDE.get("full_mode"):
        final_d = d.sum(dim=0) if num_splits is not None else d
        final_s = s.sum(dim=0) if num_splits is not None else s
        _verify_full_output(
            "hc_prenorm_output", final_d, a.float() @ b.T,
            {"a": a, "b": b})
        _verify_full_output(
            "hc_prenorm_norm", final_s, a.float().square().sum(dim=-1),
            {"a": a})
    return meta


def run_mega(cta):
    meta = ("mega_moe", "minimal", "mega_moe")
    try:
        import torch.distributed as dist
        from deep_gemm.utils.dist import init_dist
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        if not dist.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                    init_method="env://", rank=0, world_size=1)
    except Exception as e:
        raise RuntimeError(f"mega_moe needs single-process ProcessGroup: {e}")
    raise RuntimeError("mega_moe full setup not implemented; see upstream test_mega_moe.py")


_RUNNERS.update({
    "mg_bf16": run_mg_bf16, "mg_bf16_nn": run_mg_bf16_nn, "mgm_bf16": run_mgm_bf16,
    "kg_bf16": run_kg_bf16, "skiphead": run_skiphead, "einsum": run_einsum,
    "mqa": run_mqa, "pmqa": run_paged_mqa, "hc": run_hc, "mega": run_mega,
    "mqa_block_scaled": run_mqa_block_scaled,
    "mg_c_nn": run_mg_fp8_nn,
    "mg_c_fp8fp4_nn": run_mg_fp8fp4_nn,
})
_SHAPE_KEY = {
    "fp8": "dense",
    "bf16": "dense",
    "fp8fp4": "dense",
    "mg_c": "grouped",
    "mg_m": "masked",
    "kg": "kgroup",
}
for _dt in ("fp8", "bf16", "fp8fp4"):
    for _lo in ("nt", "nn", "tn", "tt"):
        _SHAPE_KEY.setdefault(f"{_dt}_{_lo}", "dense")
_SHAPE_KEY["fp8_accc"] = "dense"
_SHAPE_KEY["bf16_accc"] = "dense"
_SHAPE_KEY.update({"mg_bf16": "grouped", "mg_bf16_nn": "grouped", "mgm_bf16": "masked",
                   "kg_bf16": "kgroup", "skiphead": "dense", "einsum": "einsum_dims",
                   "mqa": "mqa", "pmqa": "pmqa", "hc": "dense", "mega": "dense",
                   "mqa_block_scaled": "mqa"})
_SHAPE_KEY["mg_c_nn"] = "grouped"
_SHAPE_KEY["mg_c_fp8fp4_nn"] = "grouped"
_SPEC_SC_PATH = Path(__file__).resolve().parents[2] / "suites" / "c450_full" / "cases_deepgemm.json"


def _load_spec_cases(path: Path | None = None) -> tuple[dict, dict]:
    """delivery300sc spec → (id→(cta, runner), id→raw entry); stdlib-only, import safe."""
    cases_dir = os.environ.get("BW_CASES_DIR")
    spec_path = (path or (Path(cases_dir) / "cases_deepgemm.json"
                          if cases_dir else _SPEC_SC_PATH))
    table: dict = {}
    entries: dict = {}
    if not spec_path.exists():
        return table, entries
    raw = json.loads(spec_path.read_text())
    for c in raw.get("cases", []):
        cid = c["id"]
        fname = c["fn"]
        if fname not in _RUNNERS:
            raise RuntimeError(f"{spec_path.name}: case {cid} has unknown fn {fname!r}")
        key = _SHAPE_KEY[fname]
        if key not in c:
            raise RuntimeError(f"{spec_path.name}: case {cid} missing shape field {key!r}")
        table[cid] = (int(c.get("cta", 2)), _RUNNERS[fname])
        entries[cid] = c
    return table, entries


SPEC_CASES, _SPEC_BY_ID = _load_spec_cases()
RUN_ALL = dict(SPEC_CASES)


def run_one(cid: str) -> dict:
    _LAST.clear()
    _OVERRIDE.clear()
    os.environ["DG_CASE_ID"] = cid
    want, fn = RUN_ALL[cid]
    spec = _SPEC_BY_ID.get(cid)
    _load_heavy(_deepgemm_root(spec))
    full_mode = bool(spec and spec.get("mode") == "full")
    calls = int((spec or {}).get("calls", 1))
    if calls < 1:
        raise ValueError(f"{cid}: calls must be positive")
    if spec is not None:
        _OVERRIDE[_SHAPE_KEY[spec["fn"]]] = spec[_SHAPE_KEY[spec["fn"]]]
        _OVERRIDE["heldout"] = bool(spec.get("heldout", False))
        for k in ("einsum", "edtype"):
            if k in spec:
                _OVERRIDE[k] = spec[k]
        if "num_splits" in spec:
            _OVERRIDE["num_splits"] = spec["num_splits"]
        if "pmqa_api" in spec:
            _OVERRIDE["pmqa_api"] = spec["pmqa_api"]
        for k in ("format", "num_heads", "head_dim", "context_parallel"):
            if k in spec:
                _OVERRIDE[k] = spec[k]
    if full_mode:
        _OVERRIDE["calls"] = calls
        _OVERRIDE["full_mode"] = True
        if spec and "dtype" in spec:
            _OVERRIDE["dtype"] = spec["dtype"]
    deep_gemm.set_pdl(False)
    if not full_mode or "num_sms" in (spec or {}):
        deep_gemm.set_num_sms(int((spec or {}).get("num_sms", 2)))
    num_sms = int(deep_gemm.get_num_sms())
    meta = {"dtype": "", "shape": "", "kernel": ""}

    def _go():
        cta_request = "library-default" if full_mode else str(want)
        print(f"CASE {cid} want_cta={cta_request} "
              f"num_sms={deep_gemm.get_num_sms()} pdl={deep_gemm.get_pdl()}",
              flush=True)
        dt, shape, kernel = fn(want)
        meta["dtype"], meta["shape"], meta["kernel"] = dt, shape, kernel

    log, err = _fd_capture(_go)
    sass = _latest_sass() if err is None else ""
    parsed = _parse(log, sass)
    cluster = parsed["cluster"]
    ok = err is None
    topo = f"num_sms={num_sms} cluster_size={cluster} => {num_sms // max(cluster, 1)} cluster(s) x {cluster} CTA"
    parsed["proof"] = f"{parsed['proof']} topology={topo}"
    if full_mode:
        pass
    elif want == 2:
        ok = ok and parsed["ok_2c"] and cluster == 2
        if err is None and not ok:
            err = RuntimeError(f"wanted kNumMulticast=2, got cluster={cluster}")
    else:
        if err is None and cluster != 1:
            ok = False
            err = RuntimeError(f"wanted cluster=1, heuristics picked kNumMulticast={cluster}")
    payload = {
        "id": cid,
        "suite": SUITE,
        "ok": bool(ok),
        "cta_group": int(parsed["cta_group"]),
        "cluster": int(cluster),
        "sms": num_sms if full_mode else 2,
        "dtype": meta["dtype"],
        "shape": meta["shape"],
        "kernel": meta["kernel"],
        "proof": parsed["proof"],
        "error": "" if ok else (f"{type(err).__name__}: {err}" if err else parsed["proof"]),
        "log": log[-8000:],
        "heldout": _is_hold(cid),
        "num_groups": _LAST.get("num_groups"),
        "group_spec": _LAST.get("group_spec"),
        "numerics": _LAST.get("numerics"),
        "device": _device_identity(),
    }
    if full_mode:
        payload["calls"] = calls
        payload["mode"] = "full"
    if spec is not None:
        payload["family"] = spec.get("family", spec["fn"])
        if spec.get("deepgemm_root"):
            payload["deepgemm_version"] = getattr(deep_gemm, "__version__", "?")
            payload["deepgemm_path"] = str(
                Path(deep_gemm.__file__).resolve())
        payload["est_cycles"] = spec.get("est_cycles")
        payload["est_basis"] = spec.get("est_basis")
        payload["heldout"] = bool(spec.get("heldout", False))
        if "sc_note" in spec:
            payload["sc_note"] = spec["sc_note"]
    dump = _dump_dir(cid)
    if dump is not None:
        ident = {
            "id": cid,
            "heldout": payload["heldout"],
            "cta_group": payload["cta_group"],
            "cluster": payload["cluster"],
            "sms": payload["sms"],
            "shape": payload["shape"],
            "kernel": payload["kernel"],
            "num_groups": payload["num_groups"],
            "group_spec": payload["group_spec"],
            "proof": payload["proof"],
            "device": payload["device"],
            "ok": payload["ok"],
        }
        (dump / "identity.json").write_text(json.dumps(ident, indent=2) + "\n")
    write_result(SUITE, cid, payload)
    if sass:
        write_text(SUITE, cid, "kernel.sass.txt", sass[:20000])
        if dump is not None:
            (dump / "kernel.sass.txt").write_text(sass)
    print(cid, "ok=" + str(payload["ok"]), payload["proof"][:240], flush=True)
    return payload


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        # Dry-run: print every case id without importing torch/deep_gemm or touching CUDA.
        for cid in RUN_ALL:
            print(cid)
        return
    if "--resolve" in argv:
        # Harness-stage dry-run: resolve every id exactly like run_one's
        # lookup head, print one RESOLVE line per case, then exit before
        # _load_heavy() — no torch/deep_gemm import, no GPU.
        resolve_ids = [a for a in argv if not a.startswith("--")]
        if not resolve_ids:
            resolve_ids = list(SPEC_CASES)
        for rcid in resolve_ids:
            rwant, rfn = RUN_ALL[rcid]  # KeyError = unresolvable id: dry-run must fail loudly
            rspec = _SPEC_BY_ID.get(rcid)
            rshape = rspec.get(_SHAPE_KEY[rspec["fn"]], "") if rspec else ""
            rheldout = bool(rspec.get("heldout", False)) if rspec else False
            print(
                f"RESOLVE {rcid} src=sc fn={getattr(rfn, '__name__', rfn)}"
                f" cta={rwant} shape={rshape} heldout={rheldout}",
                flush=True,
            )
        return
    ids = [a for a in argv if not a.startswith("--")]
    if not ids:
        ids = list(SPEC_CASES)
    roots = {_deepgemm_root(_SPEC_BY_ID.get(cid)) for cid in ids}
    if len(roots) > 1:
        raise RuntimeError(
            f"mixed deepgemm_root across ids: {sorted(str(r) for r in roots)}")
    _load_heavy(roots.pop() if roots else None)
    torch.manual_seed(0)
    print("deep_gemm", getattr(deep_gemm, "__version__", "?"), deep_gemm.__path__, flush=True)
    deep_gemm.set_pdl(False)
    for cid in ids:
        run_one(cid)


if __name__ == "__main__":
    main()
