"""
Native TMA Gather and Scatter
=============================

This tutorial explains how to use the native async TMA gather and scatter
operations available on Blackwell GPUs. Native gather and scatter operations on
Blackwell GPUs are implemented in the `gl.nvidia.blackwell.tma.async_gather` and
`gl.nvidia.blackwell.tma.async_scatter` functions respectively.

TMA gather and scatter operations only support 2D tensor descriptors, where the
first dimension of the block shape must be 1. Gather accepts a 2D tensor
descriptor, a 1D tensor of row offsets, and a scalar column offset. If the block
shape of the 2D tensor descriptor is `[1, BLOCK_Y]`, gather performs the
following operation returning a 2D tensor:

```python
out = tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y]
```

Where `out.shape` is `(x_offsets.shape[0], BLOCK_Y)`. In other words, gather
loads `x_offsets.shape[0]` separately-indexed rows of size `BLOCK_Y` from the
tensor descriptor, starting at `y_offset`.

Scatter accepts a 2D tensor descriptor, a 1D tensor of row offsets, a scalar
column offset, and a 2D source tensor. If the block shape of the 2D tensor
descriptor is `[1, BLOCK_Y]`, scatter performs the following operation:

```python
tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y] = src
```

Where `src.shape` must be `(x_offsets.shape[0], BLOCK_Y)`. In other words,
scatter writes `src` to the tensor descriptor starting at `y_offset` but to
separately-indexed rows of size `BLOCK_Y`.

Like `async_copy_global_to_shared` and `async_copy_shared_to_global`,
`async_gather` and `async_scatter` access shared memory through the async
proxy, so fences need to be inserted as appropriate.
"""

import sys

import pytest

import torch

import triton

import importlib

import triton.experimental.gluon as gluon

import triton.experimental.gluon.language as gl

from triton._C.libtriton import ir, gluon_ir

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from triton.experimental.gluon.language.nvidia.blackwell import (tma, mbarrier, fence_async_shared)

def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10

def to_linear_layout(layout, shape):
    context = ir.context()
    ir.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context)
    return builder.to_linear_layout(layout._to_ir(builder), shape)

@gluon.jit
def async_gather_kernel(out_ptr, out_stride_x, out_stride_y, tensor_desc, x_offsets_ptr, y_offset,
                        BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    # Load the offsets using a coalesced layout for efficient load vectorization.
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))

    # Convert the offsets layout to a slice layout that satisfies the constraints for `async_gather`.
    offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    # `async_gather` loads the rows from a tensor descriptor and writes them into shared memory.
    # The layout of the shared memory descriptor must match the shared memory layout of the tensor descriptor.
    smem_dest = gl.allocate_shared_memory(tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout)

    # `async_gather` is an asynchronous operation that uses an mbarrier to track its completion.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    # Invoke `mbarrier.expect` on the mbarrier with the number of bytes to be loaded.
    mbarrier.expect(bar, BLOCK_X * tensor_desc.block_type.nbytes)

    # Issue the async gather and wait.
    tma.async_gather(tensor_desc, x_offsets, y_offset, barrier=bar, result=smem_dest)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Write the result using a coalesced layout.
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    out = smem_dest.load(coalesced_2d_layout)

    indices_x = gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None] * out_stride_x
    indices_y = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :] * out_stride_y
    gl.store(out_ptr + indices_x + indices_y, out)

def async_gather(input, x_offsets, y_offset, BLOCK_X, BLOCK_Y):
    gl_dtype = getattr(gl, str(input.dtype).split('.')[1])
    # When picking the shared memory layout, we use the dimensions of the shared
    # memory descriptor, which will be [BLOCK_X, BLOCK_Y]. But the block shape of the
    # tensor descriptor must still be [1, BLOCK_Y] to be used with async gather.
    layout = gl.NVMMASharedLayout.get_default_for([BLOCK_X, BLOCK_Y], gl_dtype)
    tensor_desc = TensorDescriptor.from_tensor(input, [1, BLOCK_Y], layout)
    out = torch.empty((BLOCK_X, BLOCK_Y), dtype=input.dtype, device="cuda")
    async_gather_kernel[(1, )](out, *out.stride(), tensor_desc, x_offsets, y_offset, BLOCK_X)
    return out

@gluon.jit
def async_scatter_kernel(tensor_desc, x_offsets_ptr, y_offset, src_ptr, src_stride_x, src_stride_y,
                         BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    # Load the source using a coalesced layout for efficient load vectorization.
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    indices_x = gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None] * src_stride_x
    indices_y = gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :] * src_stride_y
    src = gl.load(src_ptr + indices_x + indices_y)

    # Load the offsets using a coalesced layout for efficient load vectorization.
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))

    # Convert the offsets layout to a slice layout that satisfies the constraints for `async_scatter`.
    offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    # `async_scatter` stores the rows to a tensor descriptor from shared memory.
    smem_src = gl.allocate_shared_memory(tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout)
    smem_src.store(src)
    # An async fence is required between the store to shared memory and the async scatter.
    # Recall from `04-tma` that a fence is needed when using different proxies to access shared
    # memory (generic proxy for the store, and async proxy for the `async_scatter`).
    fence_async_shared()
    tma.async_scatter(tensor_desc, x_offsets, y_offset, smem_src)
    # Wait for the completion of the async scatter using `store_wait`.
    tma.store_wait(0)

def async_scatter(input, x_offsets, y_offset, src, BLOCK_X, BLOCK_Y):
    gl_dtype = getattr(gl, str(input.dtype).split('.')[1])
    # When picking the shared memory layout, we use the dimensions of the shared
    # memory descriptor, which will be [BLOCK_X, BLOCK_Y]. But the block shape of the
    # tensor descriptor must still be [1, BLOCK_Y] to be used with async scatter.
    layout = gl.NVMMASharedLayout.get_default_for([BLOCK_X, BLOCK_Y], gl_dtype)
    tensor_desc = TensorDescriptor.from_tensor(input, [1, BLOCK_Y], layout)
    async_scatter_kernel[(1, )](tensor_desc, x_offsets, y_offset, src, *src.stride(), BLOCK_X)

@gluon.jit
def issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs,
                BLOCK_M: gl.constexpr, num_buffers: gl.constexpr, pred=True):
    # Load the M dimension offsets for the X tensor tile. We expect the load to be small
    # enough (no more than 128 elements) that we don't need to use a coalesced layout. Load directly into the layout
    # required by `async_gather` to avoid the layout conversion.
    gather_indx_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
    offs_x_m = gl.load(X_gather_indx_ptr + off_m + gl.arange(0, BLOCK_M, gather_indx_layout))

    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)

    # The W tensor tile is loaded using a regular `async_copy_global_to_shared`.
    mbarrier.expect(bar, W_desc.block_type.nbytes + BLOCK_M * X_desc.block_type.nbytes)
    tma.async_gather(X_desc, offs_x_m, k, bar, x_bufs.index(index), pred)
    tma.async_copy_global_to_shared(W_desc, [k, off_n], bar, w_bufs.index(index), pred)
    return producer

@gluon.jit
def issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    b_index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(x_bufs.index(index), w_bufs.index(b_index))
    return consumer, mma

@gluon.jit
def matmul_fused_gather_scatter_kernel(X_desc, W_desc, out_desc, X_gather_indx_ptr, out_scatter_indx_ptr,
                                       BLOCK_M: gl.constexpr, SchedulerImpl: gl.constexpr, num_buffers: gl.constexpr):
    BLOCK_N: gl.constexpr = W_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = W_desc.block_type.shape[0]
    dtype: gl.constexpr = X_desc.dtype
    M = X_desc.shape[0]
    N = W_desc.shape[1]
    K = X_desc.shape[1]

    # Allocate shared memory for the input tiles.
    x_bufs = gl.allocate_shared_memory(dtype, [num_buffers, BLOCK_M, BLOCK_K], X_desc.layout)
    w_bufs = gl.allocate_shared_memory(dtype, [num_buffers, BLOCK_K, BLOCK_N], W_desc.layout)

    # Allocate shared memory for the output tile.
    out_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], out_desc.layout)

    # Initialize barriers for multibuffering the loads.
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = t7.MMAv5.initialize(dtype, BLOCK_M, BLOCK_N, gl.num_warps())
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # Peeled inner loop prologue.
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, ki, bars, x_bufs, w_bufs,
                               BLOCK_M, num_buffers)
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs, BLOCK_M,
                           num_buffers)

    for _ in range(num_tiles):
        consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs,
                                   BLOCK_M, num_buffers)
            consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        # Load the M dimension offsets for the output tile. We expect the load to be small
        # enough (no more than 128 elements) that we don't need to use a coalesced layout.
        # Load directly into the layout required by `async_scatter` to avoid the layout conversion.
        scatter_indx_layout: gl.constexpr = gl.SliceLayout(
            0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))
        out_offs_m = gl.load(out_scatter_indx_ptr + epilogue_off_m + gl.arange(0, BLOCK_M, scatter_indx_layout))

        # Peel the next prologue and fuse it with the pipeline drain loop.
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        # Predicate the peeled prologue instead of using a conditional.
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, ki, bars, x_bufs, w_bufs,
                                   BLOCK_M, num_buffers, pred)
            consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k, bars, x_bufs, w_bufs,
                               BLOCK_M, num_buffers)

        mma = mma.wait_num_outstanding(0)
        out, mma = mma.take_result()
        out = out.to(dtype)
        # Pipeline the async scatter by waiting for the previous store to complete.
        tma.store_wait(pendings=0)
        out_smem.store(out)
        fence_async_shared()
        tma.async_scatter(out_desc, out_offs_m, epilogue_off_n, out_smem)
    # Wait for the last async scatter to complete.
    tma.store_wait(pendings=0)

def matmul_fused_gather_scatter(X, X_gather_indx, W, out_scatter_indx, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
                                GROUP_SIZE_M=8, num_buffers=3):
    M = X.shape[0]
    N = W.shape[1]
    out = torch.empty((M, N), dtype=X.dtype, device="cuda")

    # Convert torch dtype to gluon dtype.
    dtype = getattr(gl, str(X.dtype).split('.')[1])
    # Setup descriptors for inputs and outputs.
    X_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], dtype)
    W_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], dtype)
    out_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], dtype)

    X_desc = TensorDescriptor.from_tensor(X, [1, BLOCK_K], X_desc_layout)
    W_desc = TensorDescriptor.from_tensor(W, [BLOCK_K, BLOCK_N], W_desc_layout)
    out_desc = TensorDescriptor.from_tensor(out, [1, BLOCK_N], out_desc_layout)

    # Persistent kernel grid.
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    SchedulerImpl = t7.GroupedPersistentTileScheduler(GROUP_SIZE_M)
    matmul_fused_gather_scatter_kernel[grid](X_desc, W_desc, out_desc, X_gather_indx, out_scatter_indx, BLOCK_M,
                                             SchedulerImpl, num_buffers)
    return out
