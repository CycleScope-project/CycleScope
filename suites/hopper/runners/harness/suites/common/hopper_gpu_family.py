"""Shared identity rules for the single-card H100 Hopper family."""

from __future__ import annotations

import sys


GPU_FAMILY = "h100"


def is_h100_gpu_name(name: object) -> bool:
    return isinstance(name, str) and name.startswith("NVIDIA H100")


def require_h100_gpu_name(name: object) -> str:
    if not is_h100_gpu_name(name):
        print(f"WARNING: expected NVIDIA H100, got {name}", file=sys.stderr)
    return str(name)
