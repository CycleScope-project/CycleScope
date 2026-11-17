#!/usr/bin/env python3
"""Full-grid runner for the extracted official Triton 3.7.0 tutorials."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["CYCLESCOPE_RUNNER_ROOT"])
sys.path.insert(0, str(ROOT / "suites"))
SPEC = ROOT / "suites" / "c450_full" / "cases_triton.json"


def _append_triton_source():
    source = Path(os.environ.get(
        "TRITON_SOURCE_ROOT",
        str(Path(os.environ["CYCLESCOPE_ARTIFACT_ROOT"]) / "tools" / "triton-source")))
    source_python = source / "python"
    if source_python.is_dir():
        resolved = str(source_python.resolve())
        sys.path[:] = [item for item in sys.path
                       if str(Path(item or ".").resolve()) != resolved]
        sys.path.append(resolved)


def _spec_path() -> Path:
    cases_dir = os.environ.get("BW_CASES_DIR")
    return Path(cases_dir) / "cases_triton.json" if cases_dir else SPEC


def load_case(case_id: str) -> dict:
    for case in json.loads(_spec_path().read_text())["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def _module_name(kind: str) -> str | None:
    return {
        "tutorial01_vecadd": "tutorial_01_vector_add",
        "tutorial02_softmax": "tutorial_02_fused_softmax",
        "tutorial03_matmul": "tutorial_03_matrix_multiplication",
        "tutorial05_layernorm_fwd": "tutorial_05_layer_norm",
        "tutorial05_layernorm_bwd": "tutorial_05_layer_norm",
        "tutorial06_attention": "tutorial_06_fused_attention",
        "tutorial08_grouped_gemm": "tutorial_08_grouped_gemm",
        "tutorial09_matmul": "tutorial_09_persistent_matmul",
        "tutorial10_block_scaled": "tutorial_10_block_scaled_matmul",
        "tutorial04_seeded_dropout": "tutorial_04_low_memory_dropout",
        "tutorial07_asin": "tutorial_07_extern_functions",
        "tutorial11_pdl_add": "tutorial_11_programmatic_dependent_launch",
        "triton_kernels_matmul_dense": None,
        "triton_kernels_matmul_ragged": None,
        "triton_kernels_topk": None,
        "triton_kernels_swiglu": None,
        "triton_kernels_reduce": None,
        "triton_kernels_compaction": None,
        "gluon07_persistent": "tutorial_07_persistence",
        "gluon08_warp_specialized": "tutorial_08_warp_specialization",
        "gluon09_gather_scatter": "tutorial_09_tma_gather_scatter",
        "gluon10_matmul_accumulate": "tutorial_10_tcgen05_copy",
        "gluon11_mma_scaled": "tutorial_11_tcgen05_mma_scaled",
        "gluon_attention_forward": "tutorial_01_attention_forward",
    }[kind]


def _num_sms_override(case: dict) -> int | None:
    """single_cluster flag: cap the persistent-grid SM source at
    params.num_sms (patched into every device-properties accessor)."""
    if not (case.get("mode") == "full" and case.get("single_cluster")):
        return None
    value = case.get("params", {}).get("num_sms")
    return int(value) if value is not None else None


class _limit_sms:
    def __init__(self, num_sms: int | None):
        self.num_sms = num_sms
        self.patched = []
        self.previous_override = None

    def __enter__(self):
        if self.num_sms is None:
            return self
        self.previous_override = os.environ.get("BW_TRITON_NUM_SMS_OVERRIDE")
        os.environ["BW_TRITON_NUM_SMS_OVERRIDE"] = str(self.num_sms)
        import torch
        import triton

        num_sms = self.num_sms

        orig_torch = torch.cuda.get_device_properties

        def torch_props(device):
            import types

            props = orig_torch(device)
            clone = types.SimpleNamespace(
                **{name: getattr(props, name)
                   for name in dir(props) if not name.startswith("_")})
            clone.multi_processor_count = num_sms
            return clone

        torch.cuda.get_device_properties = torch_props
        self.patched.append((torch.cuda, "get_device_properties", orig_torch))
        try:
            utils = triton.runtime.driver.active.utils
            orig_triton = utils.get_device_properties

            def triton_props(device):
                props = dict(orig_triton(device))
                props["multiprocessor_count"] = num_sms
                return props

            utils.get_device_properties = triton_props
            self.patched.append((utils, "get_device_properties", orig_triton))
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        for obj, name, orig in self.patched:
            setattr(obj, name, orig)
        if self.previous_override is None:
            os.environ.pop("BW_TRITON_NUM_SMS_OVERRIDE", None)
        else:
            os.environ["BW_TRITON_NUM_SMS_OVERRIDE"] = self.previous_override
        return False


class _SingleProgramGrid:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, _grid):
        return self.kernel[(1,)]


def _module(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import extracted Triton tutorial {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _autotuners(module, triton):
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        Autotuner = ()
    return {
        name: value for name, value in vars(module).items()
        if Autotuner and isinstance(value, Autotuner)
    }


def _serialize_config(config) -> dict:
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
        "maxnreg": config.maxnreg,
        "pre_hook": getattr(config.pre_hook, "__name__", None),
    }


def _pin_autotuners(module, configs: dict, triton) -> None:
    if configs and not isinstance(configs, dict):
        raise ValueError("triton_autotune.json configs must be an object")
    objects = _autotuners(module, triton)
    for name, data in configs.items():
        if name not in objects:
            raise KeyError(f"autotuner {name!r} absent from extracted module")
        config_data = dict(data)
        hook_name = config_data.pop("pre_hook", None)
        if hook_name:
            config_data["pre_hook"] = getattr(module, hook_name)
        objects[name].configs = [triton.Config(**config_data)]
        objects[name].cache.clear()


def _vectors(case, module, calls: int, prepass: bool) -> dict:
    import torch
    import triton

    n = int(case["shape"][0])
    block = int(case.get("params", {}).get("block_size", 1024))
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    for _ in range(calls):
        module.add_kernel[(triton.cdiv(n, block),)](
            x, y, out, n, BLOCK_SIZE=block)
    if not prepass:
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    return {"kernel": ".*add_kernel.*", "shape": [n],
            "proof": f"tutorial 01 official add_kernel BLOCK_SIZE={block}"}


def _matmul(case, module, calls: int, prepass: bool) -> dict:
    import torch

    m, n, k = (int(value) for value in case["shape"])
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp8": torch.float8_e5m2}[case["dtype"]]
    if case["dtype"] == "fp8":
        a = (torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1).to(dtype)
        b = (torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1).to(dtype)
    else:
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((k, n), device="cuda", dtype=dtype)
    out = None
    for _ in range(calls):
        out = module.matmul(a, b)
    if not prepass:
        torch.cuda.synchronize()
        ref = torch.matmul(a.float(), b.float()).to(out.dtype)
        tolerance = 0.125 if case["dtype"] == "fp8" else 1e-2
        torch.testing.assert_close(out, ref, rtol=tolerance, atol=tolerance)
    return {"kernel": ".*matmul_kernel.*", "shape": [m, n, k],
            "proof": "tutorial 03 official autotuned matmul_kernel"}


def _persistent_matmul(case, module, calls: int, prepass: bool) -> dict:
    import torch

    m, n, k = (int(value) for value in case["shape"])
    dtype = torch.float16
    variant = str(case.get("params", {}).get("variant", "persistent"))
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    if variant in ("tma", "tma_ws", "tma_persistent", "tma_ws_persistent"):
        b = torch.randn((n, k), device="cuda", dtype=dtype)
        op = {
            "tma": lambda: module.matmul_tma(
                a, b, warp_specialize=False),
            "tma_ws": lambda: module.matmul_tma(
                a, b, warp_specialize=True),
            "tma_persistent": lambda: module.matmul_tma_persistent(
                a, b, warp_specialize=False),
            "tma_ws_persistent": lambda: module.matmul_tma_persistent(
                a, b, warp_specialize=True),
        }[variant]
        ref = lambda out: torch.matmul(a.float(), b.float().T).to(out.dtype)
        kernel = ("matmul_kernel_tma_persistent" if "persistent" in variant
                  else "matmul_kernel_tma")
    elif variant in ("descriptor_persistent", "descriptor_ws_persistent"):
        b = torch.randn((n, k), device="cuda", dtype=dtype)
        warp_specialize = variant == "descriptor_ws_persistent"
        op = lambda: module.matmul_descriptor_persistent(
            a, b, warp_specialize=warp_specialize)
        ref = lambda out: torch.matmul(a.float(), b.float().T).to(out.dtype)
        kernel = "matmul_kernel_descriptor_persistent"
    else:
        b = torch.randn((k, n), device="cuda", dtype=dtype)
        if variant == "persistent":
            op = lambda: module.matmul_persistent(a, b)
        else:
            op = lambda: module.matmul(a, b)
        ref = lambda out: torch.matmul(a.float(), b.float()).to(out.dtype)
        kernel = ("matmul_kernel_persistent" if variant == "persistent"
                  else "matmul_kernel")
    out = None
    for _ in range(calls):
        out = op()
    if not prepass:
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref(out), rtol=2e-2, atol=2e-2)
    return {"kernel": f".*{kernel}.*", "shape": [m, n, k],
            "proof": f"tutorial 09 official {variant} grid"}


def _softmax(case, module, calls: int, prepass: bool) -> dict:
    import torch
    import triton

    device = torch.device("cuda")
    props = triton.runtime.driver.active.utils.get_device_properties(
        device.index or 0)
    module.DEVICE = device
    module.properties = props
    module.NUM_SM = props["multiprocessor_count"]
    module.NUM_REGS = props["max_num_regs"]
    module.SIZE_SMEM = props["max_shared_mem"]
    module.WARP_SIZE = props["warpSize"]
    module.target = triton.runtime.driver.active.get_current_target()
    rows, cols = (int(value) for value in case["shape"])
    x = torch.randn((rows, cols), device=device, dtype=torch.float16)
    out = None
    for _ in range(calls):
        out = module.softmax(x)
    if not prepass:
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.softmax(x, dim=1),
                                   rtol=2e-2, atol=2e-2)
    return {"kernel": ".*softmax_kernel.*", "shape": [rows, cols],
            "proof": "tutorial 02 official persistent-row softmax"}


def _layernorm(case, module, calls: int, prepass: bool) -> dict:
    import torch
    import torch.nn.functional as F

    rows, cols = (int(value) for value in case["shape"])
    eps = 1e-5
    kind = case["kind"]
    if kind == "tutorial05_layernorm_fwd":
        x = torch.randn((rows, cols), device="cuda", dtype=torch.float16)
        weight = torch.randn((cols,), device="cuda", dtype=torch.float16)
        bias = torch.randn((cols,), device="cuda", dtype=torch.float16)
        out = None
        for _ in range(calls):
            out = module.LayerNorm.apply(x, (cols,), weight, bias, eps)
        if not prepass:
            torch.cuda.synchronize()
            ref = F.layer_norm(x, (cols,), weight, bias, eps)
            torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
        kernel = "_layer_norm_fwd_fused"
    else:
        x = -2.3 + 0.5 * torch.randn(
            (rows, cols), device="cuda", dtype=torch.float16)
        weight = torch.rand((cols,), device="cuda", dtype=torch.float16)
        bias = torch.rand((cols,), device="cuda", dtype=torch.float16)
        x.requires_grad_()
        weight.requires_grad_()
        bias.requires_grad_()
        grad = 0.1 * torch.randn_like(x)
        for _ in range(calls):
            x.grad = weight.grad = bias.grad = None
            out = module.LayerNorm.apply(x, (cols,), weight, bias, eps)
            out.backward(grad)
        if not prepass:
            dx_tri = x.grad.clone()
            dw_tri = weight.grad.clone()
            db_tri = bias.grad.clone()
            x.grad = weight.grad = bias.grad = None
            ref = F.layer_norm(x, (cols,), weight, bias, eps).to(x.dtype)
            torch.testing.assert_close(out, ref, rtol=0, atol=1e-2)
            ref.backward(grad)
            torch.cuda.synchronize()
            torch.testing.assert_close(dx_tri, x.grad, rtol=0, atol=1e-2)
            torch.testing.assert_close(dw_tri, weight.grad, rtol=0, atol=1e-2)
            torch.testing.assert_close(db_tri, bias.grad, rtol=0, atol=1e-2)
        kernel = "_layer_norm_bwd_dx_fused"
    return {"kernel": f".*{kernel}.*", "shape": [rows, cols],
            "proof": f"tutorial 05 official {kind}"}


def _fused_attention(case, module, calls: int, prepass: bool) -> dict:
    import torch

    batch, heads, seqlen, head_dim = (int(value) for value in case["shape"])
    params = case.get("params", {})
    mode = params.get("mode", "fwd")
    causal = bool(params.get("causal", False))
    warp_specialize = bool(params.get("warp_specialize", False))
    use_fp8 = case.get("dtype") == "fp8"
    shape = (batch, heads, seqlen, head_dim)
    q = torch.empty(shape, dtype=torch.float16, device="cuda").normal_(
        mean=0.0, std=0.5)
    k = torch.empty(shape, dtype=torch.float16, device="cuda").normal_(
        mean=0.0, std=0.5)
    v = torch.empty(shape, dtype=torch.float16, device="cuda").normal_(
        mean=0.0, std=0.5)
    sm_scale = 0.5
    reference_dtype = torch.float32 if use_fp8 else torch.float16
    q_ref = q.to(reference_dtype)
    k_ref = k.to(reference_dtype)
    v_ref = v.to(reference_dtype)
    if mode == "bwd":
        q_ref = q_ref.detach().requires_grad_()
        k_ref = k_ref.detach().requires_grad_()
        v_ref = v_ref.detach().requires_grad_()
    p = torch.matmul(q_ref, k_ref.transpose(2, 3)) * sm_scale
    if causal:
        causal_mask = torch.tril(torch.ones(
            (seqlen, seqlen), device=q.device))
        p[:, :, causal_mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(reference_dtype)
    ref_out = torch.matmul(p, v_ref).half()
    if mode == "bwd":
        dout = torch.randn_like(q)
        if not prepass:
            ref_out.backward(dout)
            ref_dq = q_ref.grad.clone()
            ref_dk = k_ref.grad.clone()
            ref_dv = v_ref.grad.clone()
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
    if use_fp8:
        kernel_q = q.to(torch.float8_e5m2)
        kernel_k = k.to(torch.float8_e5m2)
        kernel_v = v.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
        kernel_v = kernel_v.to(torch.float8_e5m2)
    else:
        kernel_q, kernel_k, kernel_v = q, k, v

    out = None
    for _ in range(calls):
        if mode == "bwd":
            q.grad = k.grad = v.grad = None
        out = module._attention.apply(
            kernel_q, kernel_k, kernel_v, causal, sm_scale, warp_specialize)
        if mode == "bwd":
            out.backward(dout)
    if not prepass:
        torch.cuda.synchronize()
        atol = 3 if use_fp8 else 1e-2
        torch.testing.assert_close(out.half(), ref_out, atol=atol, rtol=0)
        if mode == "bwd":
            tri_dq = q.grad.clone()
            tri_dk = k.grad.clone()
            tri_dv = v.grad.clone()
            torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=0)
            torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=0)
            torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=0)

    kernel = ".*_attn_bwd.*" if mode == "bwd" else ".*_attn_fwd.*"
    return {"kernel": kernel, "shape": list(shape),
            "proof": f"tutorial 06 official {mode} causal={causal} ws={warp_specialize}"}


def _grouped_gemm(case, module, calls: int, prepass: bool) -> dict:
    import torch

    module.DEVICE = torch.device("cuda")
    group_sizes = [int(value) for value in case["shape"]]
    group_count = int(case.get("params", {}).get("group_count", len(group_sizes)))
    if len(group_sizes) != group_count:
        raise ValueError("grouped GEMM shape length must match group_count")
    group_a = [
        torch.rand((size, size), device="cuda", dtype=torch.float16)
        for size in group_sizes
    ]
    group_b = [
        torch.rand((size, size), device="cuda", dtype=torch.float16)
        for size in group_sizes
    ]
    group_b_t = [tensor.T.contiguous() for tensor in group_b]
    refs = [torch.matmul(a, b) for a, b in zip(group_a, group_b)]
    variant = case.get("params", {}).get("variant", "ptr")
    output = None
    for _ in range(calls):
        if variant == "tma":
            output = module.group_gemm_tma_fn(group_a, group_b_t)
        else:
            output = module.group_gemm_fn(group_a, group_b)
    if not prepass:
        torch.cuda.synchronize()
        for ref, result in zip(refs, output):
            torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)
    kernel = ".*grouped_matmul_tma_kernel.*" if variant == "tma" \
        else ".*grouped_matmul_kernel.*"
    return {"kernel": kernel, "shape": group_sizes,
            "proof": f"tutorial 08 official group_gemm_{variant}"}


def _block_scaled(case, module, calls: int, prepass: bool) -> dict:
    import torch

    m, n, k = (int(value) for value in case["shape"])
    scale_type = case.get("params", {}).get("block_scale_type", "nvfp4")
    (a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k,
     configs, reference, _, _, _, _) = module.initialize_block_scaled(
         m, n, k, scale_type, compute_reference=True)
    out = None
    for _ in range(calls):
        out = module.block_scaled_matmul(
            a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16,
            m, n, k, rep_m, rep_n, rep_k, configs)
    if not prepass:
        torch.cuda.synchronize()
        torch.testing.assert_close(
            reference, out.to(torch.float32), atol=1e-3, rtol=1e-3)
    return {"kernel": ".*block_scaled_matmul_kernel.*", "shape": [m, n, k],
            "proof": f"tutorial 10 official initialize_block_scaled {scale_type}"}


def _elementwise_misc(case, module, calls: int, prepass: bool) -> dict:
    import os
    from pathlib import Path

    import torch
    import triton

    n = int(case["shape"][0])
    params = case.get("params", {})
    kind = case["kind"]
    if kind == "tutorial04_seeded_dropout":
        x = torch.randn((n,), device="cuda", dtype=torch.float32)
        p = float(params.get("p", 0.5))
        seed = int(params.get("seed", 123))
        reference = None
        for _ in range(calls):
            out = module.seeded_dropout(x, p, seed)
            if reference is None:
                reference = out
            elif not prepass:
                torch.testing.assert_close(out, reference, rtol=0, atol=0)
        kernel = ".*_seeded_dropout.*"
    elif kind == "tutorial07_asin":
        block = int(params.get("block_size", 1024))
        x = torch.rand((n,), device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        source_root = Path(os.environ["TRITON_SOURCE_ROOT"])
        libdevice = source_root / "third_party" / "nvidia" / "backend" / "lib" / "libdevice.10.bc"
        extern_libs = {"libdevice": str(libdevice)}
        for _ in range(calls):
            module.asin_kernel[grid](
                x, out, n, BLOCK_SIZE=block, extern_libs=extern_libs)
        if not prepass:
            torch.cuda.synchronize()
            torch.testing.assert_close(
                out, torch.asin(x), rtol=1e-6, atol=1e-6)
        kernel = ".*asin_kernel.*"
    else:
        block = int(params.get("block_size", 1024))
        x = torch.rand((n,), device="cuda", dtype=torch.float32)
        y = torch.rand_like(x)
        launch_pdl = bool(params.get("launch_pdl", True))
        if not launch_pdl:
            raise ValueError("tutorial 11 full-mode family requires launch_pdl=True")
        out = None
        for _ in range(calls):
            out = module.add(x, y, launch_pdl=launch_pdl)
        if not prepass:
            torch.cuda.synchronize()
            torch.testing.assert_close(out, x + y, rtol=0, atol=0)
        kernel = ".*add_kernel.*"
    return {"kernel": kernel, "shape": [n],
            "proof": f"tutorial {case['kind']} official launch"}


def _triton_kernel_test(case, module, calls: int, prepass: bool) -> dict:
    import torch

    source_root = Path(os.environ["TRITON_SOURCE_ROOT"])
    test_path = (source_root / "python" / "triton_kernels" / "tests" /
                 case["params"]["test_file"])
    module_name = f"c450full_{test_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, test_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official triton_kernels test {test_path}")
    test_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = test_module
    spec.loader.exec_module(test_module)
    params = case["params"]
    torch.manual_seed(int(case.get("seed", 20261024)))
    if prepass:
        if case["kind"] in ("triton_kernels_matmul_dense",
                            "triton_kernels_matmul_ragged"):
            test_module._test_op(
                *case["shape"], params["split_k"], params["do_gather"],
                params["do_scatter"], params["inner_expt_opt"],
                params["do_gamma"], params["is_persistent"],
                params["num_warps"], params["n_slices"],
                params["mode"], params["act_dtype_str"],
                params["weight_dtype_str"], params["block_m"],
                params["b_hbm_swizzling"], params["a_hbm_swizzling"],
                params["colmajor_mxfp_weight"], params["epilogue_subtile"],
                params["a_transpose"], params["b_transpose"],
                params["c_transpose"], params["swiglu_opts"], "cuda", None)
            test_module.opt_flags.reset_opt_flags_constraints()
        elif case["kind"] == "triton_kernels_topk":
            test_module.test_topk(
                *case["shape"], params["k"],
                params["apply_softmax"], params["dtype"])
        elif case["kind"] == "triton_kernels_swiglu":
            test_module.test_op(
                *case["shape"], params["limit"], "cuda",
                alpha=params["alpha"])
        elif case["kind"] == "triton_kernels_reduce":
            test_module.test_op(
                *case["shape"], params["dtype"],
                params["dim"], params["mask_mode"], None)
        else:
            test_module.test_compaction(
                *case["shape"], params["k"],
                params["p"], "cuda")
    elif case["kind"] in ("triton_kernels_matmul_dense",
                          "triton_kernels_matmul_ragged"):
        m, n, k = (int(value) for value in case["shape"])
        mode = params["mode"]
        inner_expt_opt = params["inner_expt_opt"]
        act_dtype = test_module.DType(params["act_dtype_str"])
        weight_dtype = test_module.DType(params["weight_dtype_str"])
        c_dtype = test_module.DType(params["act_dtype_str"])
        constraints = test_module.make_constraints(
            params["block_m"], params["split_k"], params["is_persistent"],
            params["epilogue_subtile"], params["b_hbm_swizzling"],
            params["weight_dtype_str"], params["num_warps"])
        test_module.opt_flags.update_opt_flags_constraints(constraints)
        expt_is_inner = inner_expt_opt is not None
        do_gather = params["do_gather"] and mode != "batched"
        do_scatter = params["do_scatter"] and mode != "batched"
        a, a_scales, a_ragged_metadata = test_module.make_random_tensor(
            shape=(m, k), n_slices=params["n_slices"], dtype=act_dtype,
            device="cuda",
            ragged_dim=None if mode != "ragged" else 1 if expt_is_inner else 0,
            mxfp_dim=-1 if act_dtype.has_mx_scale else None,
            transpose=params["a_transpose"],
            ragged_padding=expt_is_inner and "pad_a" in inner_expt_opt,
            squeeze_batch_dim=mode == "plain",
            scale_hbm_swizzling=(
                test_module.layout.make_default_matmul_mxfp8_act_scale_layout
                if params["a_hbm_swizzling"] else None))
        b, b_scale, b_ragged_metadata = test_module.make_random_tensor(
            shape=(k, n), n_slices=params["n_slices"], dtype=weight_dtype,
            device="cuda",
            ragged_dim=None if mode != "ragged" or not expt_is_inner else 0,
            mxfp_dim=-2 if weight_dtype.has_mx_scale else None,
            transpose=params["b_transpose"],
            ragged_padding=expt_is_inner and "pad_b" in inner_expt_opt,
            squeeze_batch_dim=mode == "plain",
            is_mx_rowmajor=not params["colmajor_mxfp_weight"],
            value_hbm_swizzling=(
                test_module.layout.make_default_matmul_mxfp4_w_layout(
                    mx_axis=-2)
                if params["b_hbm_swizzling"] and
                params["colmajor_mxfp_weight"] and weight_dtype.is_mxfloat4
                else None),
            scale_hbm_swizzling=(
                test_module.layout.make_default_matmul_mxfp4_w_scale_layout(
                    mx_axis=-2, num_warps=params["num_warps"])
                if params["b_hbm_swizzling"] and
                params["colmajor_mxfp_weight"] and weight_dtype.is_mxfloat4
                else None))
        gather_indx = (None if not do_gather else torch.randint(
            0, max(m, 1), (m,), dtype=torch.int32, device="cuda"))
        scatter_indx = (None if not do_scatter else torch.randperm(
            m, dtype=torch.int32, device="cuda"))
        bias = (None if expt_is_inner else torch.randn(
            b.shape[:-2] + b.shape[-1:], dtype=torch.float32, device="cuda"))
        gammas = (None if not params["do_gamma"] else 2 ** torch.randint(
            -5, 0, (m,), dtype=torch.float32, device="cuda"))
        fused_activation = None
        if params["swiglu_opts"] is not None:
            fused_activation = test_module.FusedActivation(
                test_module.FnSpecs(
                    "swiglu", test_module.swiglu_fn, ("alpha", "limit"),
                    reduction_n=2),
                params["swiglu_opts"])
        c_shape = ((params["n_slices"],)
                   if mode == "batched" or expt_is_inner else tuple())
        c_shape += (scatter_indx.shape[0] if do_scatter else a.shape[-2],)
        c_shape += (b.shape[-1] //
                    (1 if fused_activation is None else
                     fused_activation.specs.reduction_n),)
        c = torch.empty(c_shape, dtype=c_dtype.torch_dtype, device="cuda")
        if params["c_transpose"]:
            c = c.mT.contiguous().mT
        wrap_list = lambda values: torch.tensor(
            values, dtype=torch.float32, device="cuda")
        flex_a = (test_module.InFlexData(
            c_dtype.torch_dtype, wrap_list([1.25]))
            if c_dtype.has_global_scale else test_module.InFlexData())
        flex_b = (test_module.InFlexData(
            weight_dtype.torch_dtype, wrap_list([1.25]))
            if weight_dtype.has_global_scale else test_module.InFlexData())
        flex_c = (test_module.OutFlexData(
            c_dtype.torch_dtype, wrap_list([4.00]), wrap_list([0]), None)
            if c_dtype.has_global_scale else test_module.OutFlexData())
        precision_opt = test_module.PrecisionConfig(
            flex_ctx=test_module.FlexCtx(flex_a, flex_b, flex_c),
            acc_scale=(2.0 if c_dtype.has_global_scale or
                       weight_dtype.has_global_scale else 1.0),
            out_dtype=c_dtype.torch_dtype,
            a_mx_scale=a_scales,
            b_mx_scale=b_scale)
        epilogue = None
        if c_dtype.has_mx_scale:
            c_scale_shape = c_shape[:-1] + (
                test_module.triton.cdiv(c_shape[-1],
                                        test_module.MXFP_BLOCK_SIZE),)
            precision_opt.c_mx_scale = torch.empty(
                c_scale_shape, dtype=torch.uint8, device=a.device)
            epilogue_spec = test_module.FnSpecs(
                test_module.FnName.QUANTIZE_MXFP8.name,
                test_module.quantize_mxfp8_fn, (), ())
            epilogue = test_module.Epilogue(
                epilogue_spec, tuple(), tuple(), effective_itemsize=6.0)
        print("TRITON_KERNELS_MATMUL_INPUTS_READY", flush=True)
        try:
            for call_index in range(calls):
                print(
                    f"TRITON_KERNELS_MATMUL_API_START {call_index}",
                    flush=True)
                test_module.matmul(
                    a, b, bias, a_ragged_metadata, b_ragged_metadata,
                    gather_indx, scatter_indx, precision_opt, gammas=gammas,
                    epilogue=epilogue, c=c, fused_activation=fused_activation)
                torch.cuda.synchronize()
                print(
                    f"TRITON_KERNELS_MATMUL_API_DONE {call_index}",
                    flush=True)
        finally:
            test_module.opt_flags.reset_opt_flags_constraints()
    elif case["kind"] == "triton_kernels_topk":
        n_rows, n_cols = (int(value) for value in case["shape"])
        x = torch.randn(
            (n_rows, n_cols), dtype=torch.float32, device="cuda")
        for _ in range(calls):
            test_module.topk(
                x, params["k"], apply_softmax=params["apply_softmax"])
    elif case["kind"] == "triton_kernels_swiglu":
        x = test_module.alloc_rand(
            list(case["shape"]), device="cuda", dtype=torch.bfloat16)
        precision_config = test_module.PrecisionConfig(
            limit=params["limit"])
        if case.get("single_cluster"):
            swiglu_module = importlib.import_module("triton_kernels.swiglu")
            original_kernel = swiglu_module._swiglu
            swiglu_module._swiglu = _SingleProgramGrid(original_kernel)
            try:
                for _ in range(calls):
                    test_module.swiglu(x, params["alpha"], precision_config)
            finally:
                swiglu_module._swiglu = original_kernel
        else:
            for _ in range(calls):
                test_module.swiglu(x, params["alpha"], precision_config)
    elif case["kind"] == "triton_kernels_reduce":
        x = torch.randn(
            tuple(case["shape"]), device="cuda", dtype=torch.float32,
            requires_grad=True)
        mask = test_module.init_mask(
            params["mask_mode"], *case["shape"], "cuda")
        for _ in range(calls):
            test_module.reduce(x, dim=params["dim"], mask=mask)
    else:
        n_tokens, n_cols = (int(value) for value in case["shape"])
        k, p = params["k"], params["p"]
        yi = torch.rand((n_tokens, n_cols), device="cuda").argsort(
            dim=-1)[:, :k].to(torch.int32)
        yv = torch.randn(
            (n_tokens, k), dtype=torch.bfloat16, device="cuda")
        mask = torch.zeros(
            (n_tokens, n_cols), dtype=torch.int32, device="cuda")
        keep = torch.rand(yi.shape, device="cuda") < p
        if keep.any():
            rows = torch.arange(
                yi.size(0), device="cuda").unsqueeze(1).expand_as(yi)
            mask[rows[keep], yi[keep]] = 1
        chunks = mask.view(*mask.shape[:-1], -1, 32)
        weights = 1 << torch.arange(32, dtype=torch.int32, device="cuda")
        bitmask = (chunks.int() * weights).sum(dim=-1)
        for _ in range(calls):
            test_module.compaction(yv, yi, bitmask)
    return {"kernel": case["main_kernel"], "shape": case["shape"],
            "proof": f"official triton_kernels/{params['test_file']} test inputs"}


def _gluon_matmul(case, module, calls: int, prepass: bool) -> dict:
    import torch

    m, n, k = (int(value) for value in case["shape"])
    kind = case["kind"]
    dtype = torch.float16
    if kind in ("gluon09_gather_scatter", "gluon10_matmul_accumulate",
                "gluon11_mma_scaled"):
        module.t7 = _module("tutorial_07_persistence")
    if kind in ("gluon10_matmul_accumulate", "gluon11_mma_scaled"):
        module.t8 = _module("tutorial_08_warp_specialization")

    if kind in ("gluon07_persistent", "gluon08_warp_specialized"):
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((k, n), device="cuda", dtype=dtype)
        out = torch.empty((m, n), device="cuda", dtype=dtype)
        if kind == "gluon07_persistent":
            scheduler = module.PersistentTileScheduler
            for _ in range(calls):
                module.persistent_matmul(a, b, out, 128, 256, 64, 3, 4,
                                         scheduler)
            kernel = ".*persistent_matmul_kernel.*"
        else:
            scheduler = _module("tutorial_07_persistence").PersistentTileScheduler
            for _ in range(calls):
                module.matmul_warp_specialized(
                    a, b, out, 128, 256, 64, 3, 4, 4, scheduler)
            kernel = ".*matmul_warp_specialized_kernel.*"
        reference = a @ b
        tolerance = (1e-1, 1e-3)
    elif kind == "gluon09_gather_scatter":
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        gather = torch.randperm(m, device="cuda", dtype=torch.int32)
        scatter = torch.randperm(m, device="cuda", dtype=torch.int32)
        out = None
        for _ in range(calls):
            out = module.matmul_fused_gather_scatter(
                x, gather, w, scatter, 128, 128, 64, num_buffers=3)
        reference = torch.empty_like(out)
        reference[scatter, :] = x[gather, :] @ w
        tolerance = (1e-3, 1e-3)
        kernel = ".*matmul_fused_gather_scatter_kernel.*"
    elif kind == "gluon10_matmul_accumulate":
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((k, n), device="cuda", dtype=dtype)
        c = torch.randn((m, n), device="cuda", dtype=torch.float32)
        out = None
        for _ in range(calls):
            out = module.matmul_accumulate(a, b, c, 128, 128, 64,
                                          num_buffers=3)
        reference = a @ b + c
        tolerance = (5e-3, 1e-2)
        kernel = ".*matmul_accumulate_kernel.*"
    else:
        a_format = case["params"].get("a_format", "nvfp4")
        b_format = case["params"].get("b_format", "nvfp4")
        a, a_scale, a_ref = module.random_quantized_tensor(m, k, a_format)
        b, b_scale, b_ref = module.random_quantized_tensor(n, k, b_format)
        vec_size = 16 if a_format == "nvfp4" else 32
        a_scale = module.swizzle_scales_packed_block(a_scale, vec_size)
        b_scale = module.swizzle_scales_packed_block(b_scale, vec_size)
        reference = a_ref @ b_ref.T
        out = None
        # The corpus names the tutorial-11 implementation (the pipelined
        # variant commits MMA N to its single acc barrier before waiting on
        # MMA N-1 and loses the phase under instruction-level tracing;
        # the warp-specialized variant hands the accumulator over with
        # empty/ready barrier pairs).
        impl = case["params"].get("impl", "mma_scaled_pipelined_kernel")
        impl_kernel = getattr(module, impl)
        for _ in range(calls):
            out = module.mma_scaled(
                a, b, a_scale, b_scale, vec_size, impl_kernel)
        tolerance = (1e-3, 1e-3)
        kernel = f".*{impl}.*"
    if not prepass:
        torch.cuda.synchronize()
        if kind == "gluon11_mma_scaled":
            out = out.to(torch.float32)
        torch.testing.assert_close(
            out, reference, atol=tolerance[0], rtol=tolerance[1])
    return {"kernel": kernel, "shape": list(case["shape"]),
            "proof": f"Gluon official {kind} reference"}


def _gluon_attention(case, module, calls: int, prepass: bool) -> dict:
    import torch
    import torch.nn.functional as F

    batch, heads, seqlen, head_dim = (int(value) for value in case["shape"])
    causal = bool(case["params"].get("causal", False))
    q = torch.empty((batch, heads, seqlen, head_dim), device="cuda",
                    dtype=torch.float16).normal_(mean=0.0, std=0.5)
    k = torch.empty_like(q).normal_(mean=0.0, std=0.5)
    v = torch.empty_like(q).normal_(mean=0.0, std=0.5)
    scale = 0.5
    out = None
    for _ in range(calls):
        out, _ = module.attention_forward(q, k, v, causal, scale, False)
    if not prepass:
        reference = F.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=causal)
        torch.cuda.synchronize()
        torch.testing.assert_close(reference, out, atol=1e-2, rtol=0)
    return {"kernel": ".*gluon_attention.*", "shape": list(case["shape"]),
            "proof": "Gluon official Blackwell attention forward"}


def run(case: dict, *, prepass: bool = False, module=None) -> dict:
    import torch
    import triton

    name = case["kind"]
    modules = {
        "tutorial01_vecadd": ("tutorial_01_vector_add", _vectors),
        "tutorial03_matmul": ("tutorial_03_matrix_multiplication", _matmul),
        "tutorial09_matmul": ("tutorial_09_persistent_matmul", _persistent_matmul),
        "tutorial02_softmax": ("tutorial_02_fused_softmax", _softmax),
        "tutorial05_layernorm_fwd": ("tutorial_05_layer_norm", _layernorm),
        "tutorial05_layernorm_bwd": ("tutorial_05_layer_norm", _layernorm),
        "tutorial06_attention": ("tutorial_06_fused_attention", _fused_attention),
        "tutorial08_grouped_gemm": ("tutorial_08_grouped_gemm", _grouped_gemm),
        "tutorial10_block_scaled": ("tutorial_10_block_scaled_matmul",
                                    _block_scaled),
        "tutorial04_seeded_dropout": ("tutorial_04_low_memory_dropout",
                                      _elementwise_misc),
        "tutorial07_asin": ("tutorial_07_extern_functions", _elementwise_misc),
        "tutorial11_pdl_add": ("tutorial_11_programmatic_dependent_launch",
                               _elementwise_misc),
        "triton_kernels_matmul_dense": (None, _triton_kernel_test),
        "triton_kernels_matmul_ragged": (None, _triton_kernel_test),
        "triton_kernels_topk": (None, _triton_kernel_test),
        "triton_kernels_swiglu": (None, _triton_kernel_test),
        "triton_kernels_reduce": (None, _triton_kernel_test),
        "triton_kernels_compaction": (None, _triton_kernel_test),
        "gluon07_persistent": ("tutorial_07_persistence", _gluon_matmul),
        "gluon08_warp_specialized": ("tutorial_08_warp_specialization",
                                     _gluon_matmul),
        "gluon09_gather_scatter": ("tutorial_09_tma_gather_scatter",
                                   _gluon_matmul),
        "gluon10_matmul_accumulate": ("tutorial_10_tcgen05_copy",
                                      _gluon_matmul),
        "gluon11_mma_scaled": ("tutorial_11_tcgen05_mma_scaled",
                               _gluon_matmul),
        "gluon_attention_forward": ("tutorial_01_attention_forward",
                                    _gluon_attention),
    }
    if name not in modules:
        raise KeyError(f"unsupported official Triton kind {name!r}")
    module_name, call = modules[name]
    calls = 1 if prepass else int(case.get("calls", 4))
    with _limit_sms(_num_sms_override(case)):
        if module is None and module_name is not None:
            module = _module(module_name)
        result = call(case, module, calls, prepass)
    return {**result, "calls": int(case.get("calls", 4)),
            "autotuners": _autotuners(module, triton) if module is not None else {}}


def prepass(case: dict) -> None:
    import torch
    import triton

    workdir = Path(os.environ.get("BW_TRITON_WORKDIR", "/tmp/triton"))
    workdir.mkdir(parents=True, exist_ok=True)
    output = workdir / "triton_autotune.json"
    if case.get("autotune_config"):
        output.write_text(json.dumps({
            "suite": "triton",
            "case": case["id"],
            "kind": case["kind"],
            "shape": case["shape"],
            "configs": case["autotune_config"],
            "triton_kernels_import": "spec_pinned",
        }, indent=2) + "\n")
        print(f"AUTOTUNE_PREPASS {case['id']} {output} spec_pinned", flush=True)
        return
    _append_triton_source()
    torch.manual_seed(int(case.get("seed", 20261024)))
    result = run(case, prepass=True)
    configs = {
        name: _serialize_config(autotuner.best_config)
        for name, autotuner in result["autotuners"].items()
        if getattr(autotuner, "best_config", None) is not None
    }
    import_result = "not_attempted"
    try:
        import triton_kernels
        location = getattr(triton_kernels, "__file__", None)
        if location is None:
            location = ",".join(str(path) for path in
                                getattr(triton_kernels, "__path__", []))
        import_result = f"imported:{location or 'namespace-package'}"
    except Exception as error:  # source-tree package availability is reported, not required for tutorial cases
        import_result = f"unavailable:{type(error).__name__}:{error}"
    output.write_text(json.dumps({
        "suite": "triton",
        "case": case["id"],
        "kind": case["kind"],
        "shape": case["shape"],
        "configs": configs,
        "triton_kernels_import": import_result,
    }, indent=2) + "\n")
    print(f"AUTOTUNE_PREPASS {case['id']} {output} {import_result}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    parser.add_argument("--prepass", action="store_true")
    args = parser.parse_args()
    case = load_case(args.case_id)
    import torch
    import triton

    _append_triton_source()
    torch.manual_seed(int(case.get("seed", 20261024)))
    if args.prepass:
        prepass(case)
        return 0
    workdir = Path(os.environ.get("BW_TRITON_WORKDIR", "/tmp/triton"))
    autotune_file = workdir / "triton_autotune.json"
    if not autotune_file.is_file():
        raise FileNotFoundError(f"missing unprofiled autotune prepass output: {autotune_file}")
    pin_data = json.loads(autotune_file.read_text())["configs"]
    with _limit_sms(_num_sms_override(case)):
        module_name = _module_name(case["kind"])
        module = _module(module_name) if module_name is not None else None
        if module is not None:
            _pin_autotuners(module, pin_data, triton)
        result = run(case, module=module)
    from common import write_result
    write_result("triton", case["id"], {
        "id": case["id"], "suite": "triton", "ok": True,
        "cta_group": case.get("cta_group"), "cluster": case.get("cluster"),
        "sms": case.get("sms"), "dtype": case.get("dtype"),
        "shape": case.get("shape_tag"), "kernel": result["kernel"],
        "proof": result["proof"], "calls": result["calls"], "error": "",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
