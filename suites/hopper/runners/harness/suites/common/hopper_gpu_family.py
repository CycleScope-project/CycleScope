"""Shared identity rules for the single-card H100 Hopper family."""

from __future__ import annotations


GPU_FAMILY = "h100"


def is_h100_gpu_name(name: object) -> bool:
    return isinstance(name, str) and name.startswith("NVIDIA H100")


def require_h100_gpu_name(name: object) -> str:
    if not is_h100_gpu_name(name):
        raise ValueError(f"expected NVIDIA H100, got {name}")
    return str(name)
