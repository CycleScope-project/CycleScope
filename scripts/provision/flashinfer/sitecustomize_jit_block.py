import importlib.util
import sys
import types


class _StubLoader:
    """Blocking loader for flashinfer_jit_cache (T0005 far-holdout).

    The delivery pins flashinfer-python==0.6.17 at git snapshot a0a6b019 and
    instruments its csrc; prebuilt AOT modules from ANY flashinfer-jit-cache
    build shadow those sources with a different ABI, and the image's global
    0.6.6 module predates the get_jit_cache_dir API.  The stub reports a
    matching version (passes the gate) and an AOT dir that does not exist, so
    every module JIT-compiles from the instrumented tree.
    """

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__version__ = "0.6.17"
        module.__path__ = []
        # flashinfer/jit/env.py::FLASHINFER_AOT_DIR resolves the module's AOT
        # dir at import time; point it at a nonexistent path so every JIT
        # module compiles from the instrumented wheel tree instead.
        module.get_jit_cache_dir = (
            lambda: __import__("pathlib").Path("/nonexistent_fi_jit_cache_aot")
        )
        return module

    def exec_module(self, module):
        pass


class _BlockFlashInferJitCache:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flashinfer_jit_cache":
            return importlib.util.spec_from_loader(fullname, _StubLoader(), origin="t0005-jitcache-block")
        return None


sys.meta_path.insert(0, _BlockFlashInferJitCache())
