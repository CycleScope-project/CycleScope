from __future__ import annotations

import os
from pathlib import Path


def _manager_for_compile(cute, args, active_manager):
    if active_manager is not None:
        return active_manager
    if args:
        dsl_object = getattr(args[0], "_dsl_object", None)
        manager = getattr(dsl_object, "envar", None)
        if manager is not None:
            return manager
    dsl_class = getattr(getattr(cute, "_dsl", None), "CuTeDSL", None)
    if dsl_class is not None:
        try:
            return dsl_class._get_dsl().envar
        except Exception:
            return None
    return None


def install_compile_diagnostic() -> None:
    if os.environ.get("CUTE_DSL_DUMP_DIAG") != "1":
        return

    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.base_dsl.common import get_current_env_manager
    except Exception as error:
        print(f"CUTE_DSL_DIAG import_error={type(error).__name__}:{error}",
              flush=True)
        return

    compile_callable = cute.compile
    if getattr(compile_callable, "_c450full_diagnostic", False):
        return

    class DiagnosticCompile:
        def __init__(self, original):
            self.original = original

        def __call__(self, *args, **kwargs):
            manager = _manager_for_compile(
                cute, args, get_current_env_manager())
            print(
                "CUTE_DSL_DIAG compile"
                f" cutlass_file={cutlass.__file__}"
                f" manager={type(manager).__name__ if manager else None}"
                f" keep_ptx={getattr(manager, 'keep_ptx', None)}"
                f" keep_cubin={getattr(manager, 'keep_cubin', None)}"
                f" dump_dir={getattr(manager, 'dump_dir', None)}"
                f" cache_dir={getattr(manager, 'cache_dir', None)}"
                f" no_cache={getattr(manager, 'no_cache', None)}"
                f" enable_tvm_ffi={getattr(manager, 'enable_tvm_ffi', None)}"
                f" options={kwargs.get('options', None)!r}",
                flush=True,
            )
            return self.original(*args, **kwargs)

        def __getitem__(self, options):
            return DiagnosticCompile(self.original[options])

        def __getattr__(self, name):
            return getattr(self.original, name)

    wrapper = DiagnosticCompile(compile_callable)
    wrapper._c450full_diagnostic = True
    cute.compile = wrapper
    print(f"CUTE_DSL_DIAG installed cutlass_file={cutlass.__file__}", flush=True)


def report_dump_directories() -> None:
    if os.environ.get("CUTE_DSL_DUMP_DIAG") != "1":
        return
    for variable in ("CUTE_DSL_DUMP_DIR", "CUTE_DSL_CACHE_DIR"):
        raw = os.environ.get(variable)
        if not raw:
            print(f"CUTE_DSL_DIAG {variable}=unset", flush=True)
            continue
        directory = Path(raw)
        files = (sorted(str(path.relative_to(directory))
                        for path in directory.rglob("*") if path.is_file())
                 if directory.is_dir() else [])
        print(
            f"CUTE_DSL_DIAG {variable}={directory} exists={directory.is_dir()} "
            f"files={files[:100]}",
            flush=True,
        )
