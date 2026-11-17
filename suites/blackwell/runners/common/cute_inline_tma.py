"""Observe actual host-JIT CuTe TMA descriptor construction before launch.

The vendor's ``convert-cute-to-core`` pass inlines the descriptor constructor as
an aligned 16-i64 alloca and copies that object into the CUDA launch argument.
Instrument the exact 128-byte object immediately before the copy; no kernel,
case, PC, input shape, or tensor-map bitfield is used for object selection.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

_CALLBACK = "blackwellsim_cute_inline_tma_fact"
_DESCRIPTOR_TYPE = "!llvm.struct<(array<16 x i64>)>"


def _operation(value):
    return value.owner.operation


def _constructed_descriptor(load):
    if load.operation.name != "llvm.load" or str(load.results[0].type) != _DESCRIPTOR_TYPE:
        return None
    ptr = load.operands[0]
    op = _operation(ptr)
    # The native lowering preserves the alloca through an identity
    # ptrtoint/inttoptr pair before its 16-i64 by-value copy.
    if op.name == "llvm.inttoptr":
        ptr = op.operands[0]
        op = _operation(ptr)
        if op.name != "llvm.ptrtoint":
            return None
        ptr = op.operands[0]
        op = _operation(ptr)
    if op.name != "llvm.alloca" or str(op.attributes.get("elem_type")) != "i64":
        return None
    if int(ir.IntegerAttr(op.attributes["alignment"]).value) != 64:
        return None
    size = _operation(op.operands[0])
    if size.name not in ("llvm.mlir.constant", "arith.constant"):
        return None
    if int(ir.IntegerAttr(size.attributes["value"]).value) != 16:
        return None
    return ptr


def instrument(module: ir.Module) -> int:
    """Insert one call per actual CuTe-constructed 128B object; return count."""
    loads = []
    for function in module.body.operations:
        if function.operation.name not in ("llvm.func", "func.func") or not function.regions[0].blocks:
            continue
        for block in function.regions[0].blocks:
            for op in block.operations:
                ptr = _constructed_descriptor(op) if op.operation.name == "llvm.load" else None
                if ptr is not None:
                    loads.append((op, ptr))
    if not loads:
        return 0
    if any(op.operation.name == "llvm.func" and
           str(op.attributes.get("sym_name", "")).strip('"') == _CALLBACK
           for op in module.body.operations):
        raise RuntimeError("CuTe inline TMA callback already installed")
    with ir.InsertionPoint(module.body):
        llvm.LLVMFuncOp(
            _CALLBACK,
            ir.TypeAttr.get(llvm.FunctionType.get(
                ir.Type.parse("!llvm.void"), [ir.Type.parse("!llvm.ptr")]
            )),
            loc=ir.Location.unknown(),
        )
    for load, ptr in loads:
        with ir.InsertionPoint(load):
            llvm.CallOp(None, [ptr], [], [], callee=_CALLBACK,
                        loc=ir.Location.unknown())
    return len(loads)


def install(callback_library: str) -> None:
    """Use the installed CuTe backend/pipeline; add only the post-lowering hook."""
    from cutlass.base_dsl.compiler import Compiler
    from cutlass.base_dsl.dsl import BaseDSL

    original_compile = Compiler.compile
    original_libs = BaseDSL.get_shared_libs

    def compile_with_inline_fact(self, module, pipeline, *args, **kwargs):
        lowered = original_compile(self, module, pipeline, *args, **kwargs)
        instrument(lowered)
        return lowered

    def libs_with_inline_fact(self):
        libs = original_libs(self)
        if self.name in ("CUTE_DSL", "CUTE_EXPERIMENTAL_DSL"):
            libs.append(callback_library)
        return libs

    Compiler.compile = compile_with_inline_fact
    BaseDSL.get_shared_libs = libs_with_inline_fact


def main() -> None:
    script, *args = sys.argv[1:]
    callback = os.environ.get("CUTE_INLINE_TMA_LIB")
    if callback and os.environ.get("CUDA_INJECTION64_PATH"):
        install(callback)
    sys.argv = [script, *args]
    sys.path[0] = str(Path(script).resolve().parent)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
