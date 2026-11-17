"""
Libdevice (`tl.extra.libdevice`) function
==============================
Triton can invoke a custom function from an external library.
In this example, we will use the `libdevice` library to apply `asin` on a tensor.

Please refer to `CUDA libdevice-users-guide <https://docs.nvidia.com/cuda/libdevice-users-guide/index.html>`_ and/or `HIP device-lib source code <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs/ocml/src>`_ regarding the semantics of all available libdevice functions.

In `libdevice.py`, we try to aggregate functions with the same computation but different data types together.
For example, both `__nv_asin` and `__nv_asinf` calculate the principal value of the arc sine of the input, but `__nv_asin` operates on `double` and `__nv_asinf` operates on `float`.
Triton automatically selects the correct underlying device function to invoke based on input and output types.
"""

import torch

import triton

import triton.language as tl

import inspect

import os

from triton.language.extra import libdevice

from pathlib import Path

@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)

size = 98432

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"
