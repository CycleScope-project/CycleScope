"""Python-side half of the T0011.d host-only SM cap (rev1 item 2).

The FA3 prefill persistent scheduler plans its work bins from the SM count; every C++ site
that reads it is clamped by the wheel patch (scripts/patch_wheel_marker.py, --sm-cap) under
-DFI_DELIVERY_SM_CAP + $FI_DELIVERY_SM_CAP.  Any python-side read of the same attribute must
see the same clamped value or the plan's work_indptr (sized num_ctas+1) would not cover the
bins the C++ plan fills.  This shim clamps torch.cuda.get_device_properties() for the case
process while activated; it is measurement-only and upstream-identical when $FI_DELIVERY_SM_CAP
is unset/0.

Usage (harness/run_case.py):

    with sm_cap.sm_cap_properties(torch):
        build/run the op
"""

from __future__ import annotations

import os
from typing import Any

SM_CAP_ENV = "FI_DELIVERY_SM_CAP"


def requested_cap(env: dict | None = None) -> int:
    environ = os.environ if env is None else env
    try:
        return max(int(environ.get(SM_CAP_ENV, "0") or "0"), 0)
    except ValueError:
        return 0


class _CappedDeviceProperties:
    """Delegating proxy with a clamped multi_processor_count."""

    def __init__(self, inner: Any, cap: int):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_cap", int(cap))

    def __getattr__(self, name: str):
        if name == "multi_processor_count":
            inner_count = int(getattr(self._inner, name))
            return min(inner_count, self._cap)
        return getattr(self._inner, name)


def sm_cap_properties(torch: Any):
    """Context manager clamping torch.cuda.get_device_properties().multi_processor_count."""

    class _Ctx:
        def __enter__(self):
            self._original = torch.cuda.get_device_properties
            cap = requested_cap()
            if cap <= 0:
                return self

            def capped_get_device_properties(*args, **kwargs):
                return _CappedDeviceProperties(self._original(*args, **kwargs), cap)

            torch.cuda.get_device_properties = capped_get_device_properties
            return self

        def __exit__(self, exc_type, exc, tb):
            if getattr(self, "_original", None) is not None:
                torch.cuda.get_device_properties = self._original
            return False

    return _Ctx()
