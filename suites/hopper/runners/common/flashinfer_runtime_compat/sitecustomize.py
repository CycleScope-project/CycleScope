"""Normalize the measured FlashInfer quantization module layout at startup.

The exact241 capture pins flashinfer-python 0.6.6, where
``get_quantization_module`` lives in the flat ``flashinfer.quantization``
module.  The timing harness was written against the later
``flashinfer.quantization.packbits`` package path.  Formal timing activates
this adapter explicitly, and only when the frozen runtime exposes the same
callable loader.  It changes no distribution, binary, cache, launch, or
observation scope.
"""

from __future__ import annotations

import importlib
import os
import sys


if os.environ.get("HOPPER_FLASHINFER_API_COMPAT") == "1":
    quantization = importlib.import_module("flashinfer.quantization")
    if not hasattr(quantization, "__path__"):
        getter = getattr(quantization, "get_quantization_module", None)
        if callable(getter):
            sys.modules.setdefault(
                "flashinfer.quantization.packbits", quantization
            )
